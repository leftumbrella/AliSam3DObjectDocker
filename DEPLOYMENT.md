# 阿里云 FC GPU 从零部署手册

本手册把 SAM 3 与 SAM 3D Objects 部署为一个深圳 FC GPU 函数、一张组合镜像、一个 HTTP 触发器。香港 ECS 只负责下载受限资源、上传深圳 OSS、构建镜像和调用 FC API，不承载线上推理。

## 部署结构

```text
香港 ECS
  |-- 准备 SAM3 + SAM3D 离线资源
  |-- 上传一个带内容摘要的 OSS 版本前缀
  `-- 构建并推送一张 linux/amd64 组合镜像

深圳 FC GPU 函数 sam3d-object
  |-- 一个只读 OSS 挂载：/mnt/nas/sam3d
  |-- 一个公网端口：9000
  |-- 一个内部端口：127.0.0.1:9001
  |-- 一个 HTTP 触发器 URL
  |-- POST /segment -> SAM3
  `-- POST /generate -> SAM3D
```

FC 配置把 `instanceConcurrency` 设为 1，并把函数级 `reservedConcurrency` 默认设为 1。应用内部还有异步锁与 `/tmp/sam3d-gpu.lock` 跨进程锁。这三层共同保证默认情况下整个函数同一时刻只执行一条 GPU 推理路径。

## 开始前准备资源

### 香港 ECS

建议使用全新的 Ubuntu 22.04 x86_64 ECS，并准备：

- 足够容纳 Docker 构建缓存和一套 Python 3.12 / PyTorch 2.7.1 / CUDA 12.6 用户态环境的系统盘。
- 可访问 GitHub、Hugging Face、PyPI、PyTorch wheel 和深圳公网 OSS/ACR 的网络。
- Docker Buildx。构建过程不要求本机承担最终推理，但 CUDA 扩展构建较重。
- 一个不会在 SSH 断开后继续运行的普通前台终端；脚本可重跑并复用下载、OSS 断点和镜像层。

### 深圳 OSS

提前创建真实存在的深圳 Bucket。模型资源上传阶段从香港访问，必须使用公网 Endpoint：

```text
https://oss-cn-shenzhen.aliyuncs.com
```

FC 与 Bucket 同在深圳，运行时挂载应使用深圳内网 Endpoint：

```text
https://oss-cn-shenzhen-internal.aliyuncs.com
```

香港 ECS 必须使用公网地址；不要把深圳内网 Endpoint 用于跨地域上传。

### 深圳 ACR

提前创建 ACR 仓库。香港 ECS 登录和推送使用完整公网仓库地址，例如：

```text
crpi-xxxx.cn-shenzhen.personal.cr.aliyuncs.com/namespace/sam3dobject
```

不要加 `https://`，不要提前加 tag，也不要使用包含 `-vpc` 或 `-internal` 的地址。FC 拉取镜像时可以使用对应 VPC 地址；一键脚本会从公网地址推导，也可用 `FC_ACR_IMAGE` 显式指定。

### RAM 与 FC 前置条件

提前准备：

- 一个 FC 执行角色 ARN，允许函数只读访问指定 OSS Bucket/前缀。
- 执行部署脚本的身份，至少可创建或更新目标函数、HTTP 触发器、预留配置和函数并发配置。
- 并发配置需要 `fc:GetConcurrencyConfig` 与 `fc:PutConcurrencyConfig`；其余权限按 `configure_fc.py` 实际调用收敛。
- 若使用企业版 ACR，准备对应实例 ID 和镜像拉取权限。
- FC GPU 配额和目标规格在深圳地域可用。

不要把 AccessKey、Hugging Face Token 或 ACR 密码写进 Git、Dockerfile、构建参数或命令行历史。脚本只从隐藏输入、环境变量或阿里云默认凭证链读取敏感值。

### Hugging Face 权限

先在 Meta 对应模型页面接受许可证并获得 SAM 3D Objects 与 SAM 3 checkpoint 的读取权限。Token 只在香港 ECS 下载阶段使用，完成后立即 `unset HF_TOKEN`。

## 初始化香港 ECS

使用具备 sudo 权限的普通用户或 root 登录，并更新基础组件：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git python3 python3-venv
git clone <你的仓库地址> AliSam3DObjectDocker
cd AliSam3DObjectDocker
git status --short
```

一键脚本会幂等安装或检查 Docker、Buildx、ossutil、Hugging Face CLI 和固定版本的阿里云 FC SDK。执行前可以先看帮助与只读计划。

## 推荐：一键完成香港 ECS 动作

先用非敏感示例做 dry-run；它不会安装、下载、上传、登录、构建或修改云端：

```bash
./scripts/deploy_from_hk.sh --dry-run \
  --non-interactive \
  --yes \
  --oss-bucket 'your-real-shenzhen-bucket' \
  --acr-image 'crpi-xxxx.cn-shenzhen.personal.cr.aliyuncs.com/namespace/sam3dobject' \
  --acr-username 'your-acr-user'
```

正式执行最简单的方式是交互运行：

```bash
./scripts/deploy_from_hk.sh
```

脚本会出现类似以下隐藏或普通输入；ACR 地址必须是深圳公网完整仓库地址：

```bash
read -rp '深圳 ACR 完整公网仓库地址（不含协议和 tag）: ' ACR_IMAGE
```

也可以预先通过环境变量提供非敏感参数：

```bash
export OSS_BUCKET='your-real-shenzhen-bucket'
export ACR_IMAGE='crpi-xxxx.cn-shenzhen.personal.cr.aliyuncs.com/namespace/sam3dobject'
export ACR_USERNAME='your-acr-user'
export FC_ROLE_ARN='acs:ram::1234567890123456:role/sam3d-fc-runtime'
export FC_PROVISIONED_INSTANCES=0
export FC_RESERVED_CONCURRENCY=1
./scripts/deploy_from_hk.sh --yes
```

默认最小实例数为 0，不持续预留 GPU；默认函数总并发上限为 1。若要在部署阶段强制拉起一个组合实例并等待双模型 Initializer 完成，可传 `--provisioned-instances 1`。这会持续保留一个 GPU 实例并产生费用，直到再次改回 0。

只准备资源与镜像、不修改 FC 时使用：

```bash
./scripts/deploy_from_hk.sh --skip-configure-fc
```

执行完成后会生成权限为 0600、且只含非敏感值的：

```text
/root/sam3d-transfer/deployment-result.env
/root/sam3d-transfer/fc-deployment-result.json
```

`deployment-result.env` 记录单张不可变镜像、OSS 内容版本、一个函数名和一个触发器 URL。它不记录任何 Token、AccessKey 或密码。

## 准备完整离线模型资源

一键脚本会调用 `scripts/prepare_offline_assets.py`。手动准备时可以执行：

```bash
read -rsp 'Hugging Face Access Token: ' HF_TOKEN && printf '\n'
export HF_TOKEN

python3 scripts/prepare_offline_assets.py \
  --download-sam3d \
  --download-sam3 \
  --transfer-root /root/sam3d-transfer

python3 scripts/prepare_offline_assets.py \
  --verify-only \
  --transfer-root /root/sam3d-transfer

unset HF_TOKEN
```

如果 checkpoint 已经存在，分别用 `--sam3d-source` 和 `--sam3-source` 指向本地路径。脚本验证固定 revision、精确大小和 SHA-256，并生成 `offline-assets.sha256`。最终目录至少为：

```text
/root/sam3d-transfer/storage/
├── offline-assets.sha256
├── sam3/sam3.pt
├── hf/pipeline.yaml
├── hf/moge/model.pt
└── cache/torch/hub/
    ├── facebookresearch_dinov2_main/
    └── checkpoints/dinov2_vitl14_reg4_pretrain.pth
```

离线补丁会让 pipeline 使用本地资源，关键配置应保留：

```yaml
source: local
pretrained: False
```

任何缺失、Git LFS 指针或截断文件都会让校验失败。不要跳过 `--verify-only` 门禁。

## 上传离线资源到深圳 OSS

先确认 ossutil v2 的断点参数存在：

```bash
ossutil cp --help | grep -q -- '--checkpoint-dir'
```

上传到内容寻址版本前缀，不要覆盖正在运行的旧版本：

```bash
export TRANSFER_ROOT=/root/sam3d-transfer
export OSS_BUCKET='your-real-shenzhen-bucket'
export OSS_PREFIX='sam3d/releases/bundle-your-content-digest'

ossutil cp -u -r \
  "$TRANSFER_ROOT/storage/" \
  "oss://${OSS_BUCKET}/${OSS_PREFIX}/" \
  --endpoint 'https://oss-cn-shenzhen.aliyuncs.com' \
  --checkpoint-dir "$TRANSFER_ROOT/oss-upload-checkpoints"
```

FC 把这个前缀只读挂载为 `/mnt/nas/sam3d`。容器内必须能看到：

```text
/mnt/nas/sam3d/sam3/sam3.pt
/mnt/nas/sam3d/hf/pipeline.yaml
/mnt/nas/sam3d/hf/moge/model.pt
/mnt/nas/sam3d/cache/torch/hub/facebookresearch_dinov2_main
/mnt/nas/sam3d/cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
```

FC 运行时显式设置：

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
```

因此冷启动不会临时访问 Hugging Face 或 GitHub。

## 构建 FC 组合镜像

只有一条正式构建命令；SAM 3 与 SAM 3D Objects 共用 Dockerfile 中同一套 Python 3.12 / PyTorch 2.7.1 / CUDA 12.6 环境。以下命令也是 FC 镜像清单契约：

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

`--provenance=false` 与 `--sbom=false` 防止 Buildx 额外生成 `unknown/unknown` attestation manifest。FC 自定义容器未压缩镜像上限为 15 GB；一键脚本会累计各层大小，超限即停止推送。

推送后检查远程镜像清单：

```bash
export REMOTE_IMAGE='crpi-xxxx.cn-shenzhen.personal.cr.aliyuncs.com/namespace/sam3dobject:immutable-tag'

docker buildx imagetools inspect "$REMOTE_IMAGE"
docker manifest inspect --verbose "$REMOTE_IMAGE" | grep -q 'unknown/unknown' && {
  echo '远程镜像包含 unknown/unknown manifest'
  exit 1
}
docker manifest inspect --verbose "$REMOTE_IMAGE" | grep -q 'linux/amd64' || {
  echo '远程镜像不包含 linux/amd64 manifest'
  exit 1
}
```

## 审计路径：一个深圳 FC GPU 函数

`scripts/configure_fc.py` 使用 FC 2023-03-30 SDK 幂等创建或更新一个函数，并在写入后重新读取关键配置。目标状态如下：

| 项目 | 目标值 |
| --- | --- |
| 函数名 | `sam3d-object`，可覆盖 |
| 自定义容器端口 | `9000` |
| 镜像 | 一张组合镜像的不可变 tag |
| OSS | 一个版本前缀，只读挂载到 `/mnt/nas/sam3d` |
| Initializer | `/bin/sh /srv/scripts/fc_initializer.sh`，300 秒 |
| 函数超时 | 1800 秒 |
| 单实例并发 | `1` |
| 函数总并发上限 | `1` |
| 最小实例数 | 默认 `0`，可改为 `1` 做部署期预热 |
| 默认 GPU | `fc.gpu.ada.1`，49152 MB |
| HTTP 路由 | 同一 URL 的 `/segment` 与 `/generate` |

直接调用配置工具时，可显式审查完整非敏感计划：

```bash
python3 scripts/configure_fc.py \
  --dry-run \
  --role-arn 'acs:ram::1234567890123456:role/sam3d-fc-runtime' \
  --oss-bucket 'your-real-shenzhen-bucket' \
  --oss-prefix 'sam3d/releases/bundle-your-content-digest' \
  --image 'registry-vpc.cn-shenzhen.aliyuncs.com/namespace/sam3dobject:immutable-tag' \
  --function-name 'sam3d-object' \
  --provisioned-instances 0 \
  --reserved-concurrency 1
```

如果 48 GB 显存验收失败，请先在 FC 控制台确认深圳地域与账号可用的 `fc.gpu.hopper.1` 96 GB 规格，再对一键脚本同时传 `--gpu-type fc.gpu.hopper.1 --gpu-memory-size 98304`。不要只修改显存数字而保留不匹配的实例类型。组合镜像已经同时编译 Ada `sm_89` 与 Hopper `sm_90` 的 CUDA 扩展。

## 显存与初始化验收

组合函数的 Initializer 会初始化两个模型。两条初始化任务可以同时被触发，但共享 GPU 文件锁会让实际模型加载串行执行，避免两次初始化峰值重叠；代价是 300 秒上限必须容纳两套模型的总加载时间。空闲时 GPU 利用率会下降，但两套权重、两个 CUDA context 和缓存仍占显存。安全条件不是“推理不同时发生”这么简单，而是：

```text
SAM3 常驻 + SAM3D 常驻 + CUDA context/缓存
+ max(SAM3 额外推理峰值, SAM3D 额外推理峰值)
< GPU 总显存 - 安全余量
```

若任一模型单独推理已经几乎吃满 48 GB，两个模型保持预热状态时就不能可靠部署到 48 GB 实例；文件锁无法释放另一模型的常驻权重。应选择更大显存规格，或者另行设计“切换时卸载/重载”的高延迟模式。

默认最小实例数为 0 时，首次请求触发弹性实例，FC 会先执行 Initializer。观察函数日志，确认两个模型都在 300 秒内完成加载。随后获取唯一 HTTP URL：

```bash
source /root/sam3d-transfer/deployment-result.env
export FC_URL="$FC_HTTP_URL"

curl -fS "$FC_URL/healthz"
curl -fS "$FC_URL/readyz"
curl -fS "$FC_URL/gpu"
```

`/readyz` 顶层必须是 `ready=true`，且 `models.sam3` 与 `models.sam3d` 都显示加载完成。`/healthz` 成功只证明网关进程活着，不能证明模型或挂载可用。

然后依次做真实请求：

```bash
curl -fS -X POST "$FC_URL/segment" \
  -F 'image=@input.png' \
  -F 'points=[{"x":320,"y":240,"label":1}]' \
  -D segment.headers \
  -o mask.png

curl -fS -X POST "$FC_URL/generate" \
  -F 'image=@input.png' \
  -F 'mask=@mask.png' \
  -F 'seed=42' \
  -F 'output_format=glb' \
  -o result.glb

curl -fS "$FC_URL/gpu"
```

使用最大预期输入分别测量两条路径，并连续交替请求。确认没有 OOM、Initializer 超时、内部 SAM3 超时或第二实例扩容。网页联调只需把同一个 URL 填入 `test-client.html`。

## 从旧双函数版本迁移

部署脚本不会删除旧的 SAM3 或 SAM3D 函数，也不会删除旧 ACR tag 和 OSS 版本。这是有意的可回滚保护：

1. 先部署新的组合函数，不改旧函数。
2. 用真实输入完成 `/segment`、`/generate`、显存和冷启动验收。
3. 把网页或调用方切到新的单一 URL。
4. 观察一段完整业务周期。
5. 再在 FC 控制台把旧函数最小实例数改为 0，确认无流量后人工删除旧函数与触发器。
6. OSS 旧版本和旧镜像 tag 至少保留到回滚窗口结束。

## 更新与回滚

每次代码提交生成新的不可变镜像 tag，每次模型资源变化生成新的 `bundle-<digest>` OSS 前缀。不要覆盖线上 tag 或资源前缀。

回滚代码时，把同一个组合函数的镜像改回上一 tag；回滚模型时，把 OSS 挂载前缀改回上一 bundle。两种回滚都会创建新实例并重新执行双模型 Initializer。只有 `/readyz` 和两条真实推理都通过后，才结束回滚。

## 常见错误

- `/healthz` 成功而 `/readyz` 失败：优先检查 `/mnt/nas/sam3d/hf/pipeline.yaml`、`sam3/sam3.pt` 与 OSS 挂载前缀。
- Initializer 超过 300 秒：查看两个模型串行加载的各自耗时；使用预留实例重复测量，必要时优化资源或调整架构。
- 初始化 OOM：两套常驻权重已经超过规格，不是并发锁问题。
- `/generate` OOM 而 `/segment` 正常：组合常驻显存加 SAM3D 峰值超过规格。
- 出现多个 GPU 实例：检查 `reservedConcurrency=1` 和 `instanceConcurrency=1` 是否被外部配置覆盖。
- FC 拒绝镜像平台：确认只有 `linux/amd64`，且没有 `unknown/unknown` attestation。
- 香港上传 OSS 超时：确认使用公网 Endpoint；FC 挂载才使用深圳内网 Endpoint。
- 镜像超过 15 GB：检查统一运行时、CUDA 扩展和残留构建产物；不能靠增大 FC 磁盘绕过自定义容器镜像限制。

## 最终检查清单

- [ ] 深圳 OSS Bucket、ACR 仓库、FC 执行角色和 GPU 配额已存在。
- [ ] 离线资源通过 `--verify-only`，并上传到不可变版本前缀。
- [ ] 只构建和推送一张组合镜像。
- [ ] 远程清单只有可用的 `linux/amd64`，不存在 `unknown/unknown`。
- [ ] 未压缩镜像小于 FC 15 GB 上限。
- [ ] FC 只有一个目标函数、一个触发器 URL 和一个只读 OSS 挂载。
- [ ] `instanceConcurrency=1`、`reservedConcurrency=1`。
- [ ] 最小实例数符合费用策略，默认 0。
- [ ] 两个模型都在 Initializer 中成功加载。
- [ ] `/readyz`、`/segment`、`/generate` 和交替调用均通过。
- [ ] 目标 GPU 上的实际峰值显存有安全余量。
- [ ] 新函数稳定前未删除旧函数、旧镜像和旧模型版本。
