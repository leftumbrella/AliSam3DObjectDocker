# SAM 3 + SAM 3D Objects on Alibaba Cloud Function Compute

本项目把 SAM 3 点选分割和 SAM 3D Objects 重建放进同一个阿里云函数计算（FC）GPU 函数。浏览器只配置一个 HTTP 地址：鼠标点选调用 `POST /segment` 得到 Mask，再把原图和 Mask 交给同一地址的 `POST /generate`，返回 PLY 或 GLB。

从全新的香港 ECS 部署时，请直接阅读 [阿里云 FC GPU 从零部署手册](DEPLOYMENT.md)。

## 当前架构

一个组合镜像内保留两个相互隔离的 Python 运行环境，因为两套上游依赖并不兼容：

- SAM 3：Python 3.12、PyTorch 2.7.1 + cu126，内部进程监听 `127.0.0.1:9001`。
- SAM 3D Objects：Python 3.11、PyTorch 2.5.1 + cu121，公网网关监听 FC 的 `0.0.0.0:9000`。
- `app.supervisor` 同时管理两个进程；任何一个进程退出，容器都会退出并由 FC 重建。
- FC Initializer 调用公网进程的 `POST /initialize`，并行初始化两套模型。只有两者均成功，实例才算初始化完成。
- `/segment` 和 `/generate` 由同一个公网入口提供；内部 SAM 3 端口不会暴露给 FC 触发器。

```text
一个 FC GPU 实例 / 一张组合镜像

HTTP 9000 -> SAM3D 网关
               |-- /segment -> 127.0.0.1:9001 -> SAM3
               `-- /generate ----------------> SAM3D

GPU 推理互斥：asyncio 单实例锁 + /tmp/sam3d-gpu.lock 跨进程文件锁
平台上限：instanceConcurrency=1 + reservedConcurrency=1
```

因此，默认运行语义正是“两个模型都预热、同一时刻只允许其中一个推理、在一个 GPU 实例上交替工作”。但这不等于两个模型空闲时不占显存。

## 显存前提

初始化完成后，GPU 计算利用率通常会回落，但模型权重、CUDA context 和 PyTorch 缓存仍然常驻显存。组合实例必须满足：

```text
常驻显存 = SAM3 常驻 + SAM3D 常驻 + 两个 CUDA context/缓存

所需显存 = max(
  常驻显存 + SAM3 单次推理额外峰值,
  常驻显存 + SAM3D 单次推理额外峰值
)
```

串行锁只能消除“两次推理峰值相加”，不能消除另一模型的常驻权重。如果任一模型单独推理就几乎占满 48 GB，那么“两个模型同时常驻并交替推理”无法在 48 GB Ada 实例上可靠运行。此时应改用更大显存的 FC GPU 规格；另一种方案是每次切换时卸载并重新加载模型，但那会破坏本项目的双模型预热语义，也显著增加延迟，因此没有作为默认实现。

默认部署参数仍是 `fc.gpu.ada.1` / 49152 MB，便于先做真实数据验收。上线条件是组合实例初始化成功，并且分别完成一次 `/segment` 与 `/generate` 后仍有安全显存余量。若 48 GB 不足，一键脚本支持 `--gpu-type fc.gpu.hopper.1 --gpu-memory-size 98304`；前提是该规格在目标地域和账号中可用。

## 弹性与费用

默认配置为：

- 最小实例数 `provisioned_instances=0`：空闲时不固定保留 GPU 实例。
- 函数总并发上限 `reserved_concurrency=1`：整个函数最多获得一个并发配额。
- 单实例并发 `instanceConcurrency=1`：同一实例一次只接收一个业务请求。
- 应用层再使用共享异步锁和跨进程文件锁，防止未来配置漂移或内部调用重叠。

最小实例数为 0 时，弹性创建的新实例仍会自动执行 FC Initializer，同时加载两个模型；冷启动耗时和 300 秒 Initializer 上限需要实测。若设为 1，部署脚本会等待组合实例初始化完成，但该 GPU 会持续保留并产生费用。

## 项目结构

```text
.
├── app/
│   ├── main.py              # 单一公网 API：状态、分割代理、3D 生成
│   ├── model.py             # SAM3D 初始化、推理和导出
│   ├── segmenter_client.py  # 仅允许 loopback 的 SAM3 HTTP 客户端
│   ├── supervisor.py        # 启动并监管两个隔离运行时
│   └── settings.py
├── segmenter/               # SAM3 内部服务与点选推理
├── shared/gpu_lock.py       # 两个进程共用的 GPU 文件锁
├── scripts/
│   ├── deploy_from_hk.sh    # 离线资源、OSS、镜像和单函数部署入口
│   ├── configure_fc.py      # 幂等配置一个 FC 函数
│   ├── fc_initializer.sh    # FC Initializer 回调
│   └── prepare_offline_assets.py
├── Dockerfile               # 一张镜像、两个 Python 环境
├── test-client.html         # 单地址浏览器测试台
└── DEPLOYMENT.md
```

原来的 `Dockerfile.segmenter` 已删除；部署路径只构建和推送一张镜像。

## 快速部署

一键脚本在香港 Ubuntu ECS 上完成离线资源准备、上传深圳 OSS、构建并推送组合镜像，以及幂等配置一个深圳 FC GPU 函数：

```bash
./scripts/deploy_from_hk.sh --dry-run \
  --oss-bucket 'your-shenzhen-bucket' \
  --acr-image 'crpi-example.cn-shenzhen.personal.cr.aliyuncs.com/namespace/repository' \
  --acr-username 'your-acr-user'

./scripts/deploy_from_hk.sh
```

以下账号级资源必须提前存在：深圳 OSS Bucket、深圳 ACR 仓库、FC 执行角色，以及下载受限 checkpoint 所需的 Hugging Face 权限。脚本不会越权创建这些资源，也不会自动删除旧的双函数部署。

离线资源的底层入口是 `scripts/prepare_offline_assets.py`；可先用其 `--verify-only` 模式独立核验已下载资源。

## 手动构建

组合镜像固定使用两个上游 commit。FC 自定义容器只接受兼容的 `linux/amd64` 镜像清单，构建时必须关闭 provenance 和 SBOM attestation：

```bash
docker buildx build \
  --load \
  --progress=plain \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --build-arg SAM3D_REF=f91db411c50efee93d8db7aeb323885650f6f722 \
  --build-arg SAM3_REF=8f0b7f4d4e7eda2ed606ebde6702c93359ad01da \
  --build-arg 'TORCH_CUDA_ARCH_LIST=8.9;9.0' \
  --build-arg MAX_JOBS=2 \
  --build-arg NVCC_THREADS=2 \
  -t sam3-sam3d-fc:local .
```

FC 对自定义容器的未压缩镜像大小有 15 GB 限制。一键脚本会在推送前计算 `docker image inspect` 的层大小并硬性拒绝超限镜像。

推送后必须拒绝带 `unknown/unknown` attestation 的清单：

```bash
docker buildx imagetools inspect "$REMOTE_IMAGE"
docker manifest inspect --verbose "$REMOTE_IMAGE" | grep -q 'unknown/unknown' && {
  echo '远程镜像包含 unknown/unknown manifest'
  exit 1
}
```

## 离线模型挂载

模型权重不进入镜像。深圳 OSS 的一个只读版本前缀统一挂载到 `/mnt/nas/sam3d`，至少包含：

```text
/mnt/nas/sam3d/sam3/sam3.pt
/mnt/nas/sam3d/hf/pipeline.yaml
/mnt/nas/sam3d/hf/moge/model.pt
/mnt/nas/sam3d/cache/torch/hub/facebookresearch_dinov2_main/
/mnt/nas/sam3d/cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
```

容器设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1`。扩容时不会从公网下载模型；缺少任一必要文件会使 Initializer 失败，而不是让首个业务请求临时下载或加载。

## HTTP 接口

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/healthz` | 仅检查公网进程是否存活 |
| GET | `/readyz` | 聚合 SAM3 与 SAM3D 的加载状态 |
| GET | `/gpu` | 聚合两个运行时的 CUDA 版本与显存指标 |
| POST | `/initialize` | 仅供 FC Initializer 调用，同时加载两个模型 |
| POST | `/segment` | 原图 + 点选 JSON，返回 PNG Mask |
| POST | `/generate` | 原图 + Mask，返回 PLY 或 GLB |
| POST | `/invoke` | 提示使用同一 HTTP 触发器的业务路由 |

分割请求示例：

```bash
curl -fS -X POST "$FC_URL/segment" \
  -F 'image=@input.png' \
  -F 'points=[{"x":320,"y":240,"label":1}]' \
  -o mask.png
```

3D 请求示例：

```bash
curl -fS -X POST "$FC_URL/generate" \
  -F 'image=@input.png' \
  -F 'mask=@mask.png' \
  -F 'seed=42' \
  -F 'output_format=glb' \
  -o result.glb
```

网页交互可直接打开 `test-client.html`，只需填写一个组合函数 URL。网页点击图片后先调用 `/segment` 合成 Mask，再调用 `/generate`；浏览器本身不运行 SAM 模型。

## 关键环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SAM3D_CONFIG_PATH` | `/mnt/nas/sam3d/hf/pipeline.yaml` | SAM3D 配置 |
| `SAM3_CHECKPOINT_PATH` | `/mnt/nas/sam3d/sam3/sam3.pt` | SAM3 权重 |
| `SAM3_INTERNAL_URL` | `http://127.0.0.1:9001` | 内部服务，只接受 loopback origin |
| `SAM3_INTERNAL_PORT` | `9001` | 内部 SAM3 端口 |
| `SAM3_PYTHON` | `/opt/venv/bin/python` | SAM3 独立运行时 |
| `GPU_LOCK_PATH` | `/tmp/sam3d-gpu.lock` | 两个进程共用的推理锁 |
| `PORT` | `9000` | FC 自定义容器公网端口 |
| `FC_INITIALIZER_HTTP_TIMEOUT` | `295` | 初始化回调超时，低于 FC 300 秒上限 |

镜像内 CUDA 扩展同时编译 `sm_89`（Ada）与 `sm_90`（Hopper），因此切换上述两种规格不需要重建另一份代码镜像；但仍要重新创建实例并跑完整显存验收。

完整默认值见 `.env.example`。

## 验收顺序

1. `GET /healthz` 返回 200，只能证明网关存活。
2. `GET /readyz` 必须显示顶层 `ready=true`，并且 `models.sam3`、`models.sam3d` 都已加载。
3. `GET /gpu` 记录初始化后的空闲显存。
4. 用最大预期图片完成一次 `/segment`，记录峰值和结束后的显存。
5. 用最大预期图片与 Mask 完成一次 `/generate`，再次记录峰值和结束后的显存。
6. 连续交替调用两条路径，确认没有 OOM、内部超时或并发重叠。

只有这组真实 GPU 验收通过后，才能确认 48 GB 规格适合当前模型版本和输入上限。CPU 单元测试和静态镜像检查不能替代显存验收。

## 已知边界

- FC Initializer 最长 300 秒；两个模型都会在这个生命周期阶段初始化。
- 默认弹性配置只允许一个实例。如果将 reserved concurrency 调高，跨进程文件锁只保护单个容器，不能跨 FC 实例互斥。
- 两个 CUDA/PyTorch 运行时会增加组合镜像和常驻显存；脚本会检查镜像大小，但显存只能在目标 GPU 上确认。
- `CORS_ALLOW_ORIGINS=*` 适合匿名测试触发器；生产环境应限制 Origin，并选择符合业务要求的触发器认证方式。
- 项目不自动删除旧函数或旧镜像。组合函数验证成功后，再按部署手册的迁移步骤人工清理。

## 查证资料

- [阿里云 FC GPU 实例类型与规格](https://help.aliyun.com/zh/functioncompute/instance-types-and-specifications)
- [阿里云 FC 自定义容器及 15 GB GPU 镜像上限](https://help.aliyun.com/zh/functioncompute/custom-container/)
- [阿里云 FC PutConcurrencyConfig](https://help.aliyun.com/zh/functioncompute/fc/developer-reference/api-fc-2023-03-30-putconcurrencyconfig)
- [PyTorch CUDA 内存管理](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [SAM 3D Objects 官方安装与显存前提](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
