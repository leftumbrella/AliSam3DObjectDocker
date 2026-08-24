# SAM 3D Objects on Alibaba Cloud Function Compute

这是一个可直接作为 Docker 构建上下文或 ACR 代码源使用的项目。镜像在构建阶段拉取并安装 Meta 的 SAM 3D Objects，在运行阶段提供适配阿里云函数计算（FC）自定义容器的 HTTP 服务。

服务接收一张图片和一张对象 Mask，返回以下任一结果：

- `PLY`：官方示例使用的 Gaussian Splat。
- `GLB`：SAM 3D Objects 解码得到的网格。

模型权重不进入镜像，而是把 OSS 或 NAS 中的离线资源挂载到 `/mnt/nas/sam3d`，其中 `hf/` 保存 checkpoint，`cache/` 保存上游运行时缓存。这样不会把 Hugging Face Token 写入镜像，也能降低镜像体积。`/mnt/nas/sam3d` 是容器内的本地挂载路径；即使底层使用 OSS，也不需要改这个路径。

从一台全新的香港 ECS 开始部署时，按 [阿里云 FC GPU 从零部署手册](DEPLOYMENT.md) 操作。手册包含服务器初始化、离线模型准备、OSS 上传、ACR 推送、FC 设置、验收和回滚。

完成 `git clone` 后，也可以直接运行 `./scripts/deploy_from_hk.sh`，一次完成香港 ECS 上的依赖安装、资源准备、OSS 上传、镜像构建、ACR 推送和 Manifest 检查。

## 重要前提

- SAM 3D Objects 官方要求 Linux 64 位和至少 32 GB GPU 显存。
- 本项目默认面向 FC `fc.gpu.ada.1` 48 GB 规格，并以 `TORCH_CUDA_ARCH_LIST=8.9` 构建 CUDA 扩展。使用其他 GPU 系列前，必须确认其 Compute Capability 并覆盖此构建参数。
- 上游环境固定 Python 3.11、CUDA Toolkit 12.1.1、PyTorch 2.5.1 + cu121。FC 镜像只编译当前推理路径需要的 PyTorch3D 与 gsplat，并固定使用 PyTorch SDPA；不会安装 FlashAttention 或 xformers。
- Dockerfile 在这些扩展编译完成后裁掉 Conda 中的 GCC/NVCC、Nsight、CUDA 头文件和静态开发库；动态 CUDA 运行库与 PyTorch 需要的 NVRTC 仍保留。FC 清单不再安装上游仅用于构建的 `nvidia-cuda-nvcc-cu12`。
- 上游 checkpoint 是受限资源，必须先在 Hugging Face 申请访问并接受相应条款。
- 上游代码和模型使用 SAM License。部署或分发镜像前，请自行确认用途与许可证要求。

官方资料：

- [SAM 3D Objects 仓库](https://github.com/facebookresearch/sam-3d-objects)
- [SAM 3D Objects 安装说明](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
- [SAM 3D Objects checkpoint](https://huggingface.co/facebook/sam-3d-objects)
- [阿里云 FC 自定义容器说明](https://help.aliyun.com/en/functioncompute/fc/custom-container/)
- [阿里云 FC GPU 实例规格](https://help.aliyun.com/en/functioncompute/fc/product-overview/instance-types-and-specifications)
- [阿里云 FC HTTP 触发器限制](https://help.aliyun.com/en/functioncompute/fc/http-triggers-overview)
- [阿里云 FC `unknown/unknown` 镜像排查](https://help.aliyun.com/zh/functioncompute/fc/custom-image-deployment-fails-with-platform-of-image-is-unknown-unknown)
- [Docker Build attestations](https://docs.docker.com/build/metadata/attestations/)

## 项目结构

```text
.
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 路由、上传校验和文件响应
│   ├── model.py         # 模型惰性加载、串行推理和结果导出
│   ├── serve.py         # 单 worker Uvicorn 入口
│   └── settings.py      # 环境变量配置
├── scripts/
│   ├── check_mamba_removal.py # Conda 裁剪计划安全门禁
│   └── check_runtime_imports.py # 裁剪后的运行时完整性校验
├── patches/
│   └── fc-runtime.patch # 移除上游推理入口的可选依赖导入副作用
├── .dockerignore
├── .env.example
├── .gitattributes
├── .gitignore
├── Dockerfile
├── README.md
├── requirements-fc.txt # FC 最小推理依赖
└── requirements-server.txt
```

## 设计说明

### 快速启动，惰性加载模型

FC 要求自定义容器在启动窗口内监听 `0.0.0.0:CAPort`，默认端口为 9000。本项目启动 HTTP 服务时不会立即加载模型，因此能先通过容器健康检查。以下任一操作会把模型加载到 GPU，并在该容器实例生命周期内复用：

- 调用 `POST /initialize`。
- 首次调用 `POST /generate`。

### 单进程、单次 GPU 推理

Uvicorn 固定使用一个 worker，服务内部也使用异步锁串行执行 GPU 推理，避免同一实例加载多份模型或同时推理导致显存溢出。FC 的单实例并发应设置为 `1`。

### 模型与镜像分离

镜像只包含 Ubuntu、CUDA 用户态环境、Python 依赖、SAM 3D Objects 源码和 HTTP 服务。NVIDIA 内核驱动及 `libcuda.so` 由 FC 平台注入，不应安装或复制到镜像中。

## 1. 在香港服务器准备完整离线资源

深圳 FC 不需要、也不应该在弹性扩容时临时下载模型。先在能访问 Hugging Face、GitHub 和 Meta 下载站点的香港 Linux 服务器上一次性准备完整资源，再上传到深圳 OSS。先在 [SAM 3D Objects 模型页面](https://huggingface.co/facebook/sam-3d-objects)申请权限并登录：

```bash
hf auth login
cd ~/AliSam3DObjectDocker

python3 scripts/prepare_offline_assets.py \
  --download-sam3d \
  --transfer-root /root/sam3d-transfer
```

脚本会完成以下操作：

- 使用当前 `hf` 登录凭证下载固定 revision `2e73555018d2741ccd486e56c24fac41155a1dc6` 的受限主 checkpoint；Token 只保留在香港服务器，不会进入资源包、镜像或 Git。
- 下载固定版本的 DINOv2 源码与 `dinov2_vitl14_reg4_pretrain.pth`，并检查 Git commit、文件大小和 SHA-256。
- 下载固定版本的 MoGe `model.pt`，检查文件大小和 SHA-256。
- 把 `pipeline.yaml` 中的 `Ruicheng/moge-vitl` 替换为容器内路径 `/mnt/nas/sam3d/hf/moge/model.pt`。原配置只备份到 `/root/sam3d-transfer/backups/`，不会上传到 OSS。
- 按固定 revision 的官方文件名和精确字节数检查 `pipeline.yaml` 引用的 6 个配置文件和 6 个 checkpoint，拒绝缺失、Git LFS 指针或截断文件，并生成覆盖主 checkpoint、MoGe、DINOv2 源码和权重的 `offline-assets.sha256` 清单。

如果主 checkpoint 已经下载过，不必重下；直接指定其 `checkpoints` 目录：

```bash
python3 scripts/prepare_offline_assets.py \
  --sam3d-source /实际路径/checkpoints \
  --transfer-root /root/sam3d-transfer
```

准备完成后的目录必须是：

```text
/root/sam3d-transfer/storage/
├── offline-assets.sha256
├── hf/
│   ├── pipeline.yaml
│   ├── moge/model.pt
│   └── pipeline.yaml 引用的主 checkpoint 文件
└── cache/
    ├── huggingface/
    └── torch/hub/
        ├── facebookresearch_dinov2_main/
        └── checkpoints/dinov2_vitl14_reg4_pretrain.pth
```

任何下载中断后都可以重跑同一命令；公共大文件使用 `.partial` 断点续传，已有正式文件只会在完整性检查通过后复用，不会被静默覆盖。上传前还可以执行一次只读门禁：

```bash
python3 scripts/prepare_offline_assets.py \
  --verify-only \
  --transfer-root /root/sam3d-transfer
```

然后上传 `storage/` 的内容到**真实存在的深圳 OSS Bucket** 的 `/sam3d` 前缀。Bucket 名必须是 OSS 控制台中的小写名称，不是 FC 自动生成的组件名或 ID：

```bash
export TRANSFER_ROOT=/root/sam3d-transfer
export OSS_BUCKET='你的真实深圳-bucket-name'
: "${OSS_BUCKET:?请先设置真实的 OSS Bucket 名称}"

ossutil cp -r \
  "$TRANSFER_ROOT/storage/" \
  "oss://${OSS_BUCKET}/sam3d/" \
  --checkpoint-dir /root/oss-upload-checkpoints
```

在 FC 的 OSS 挂载配置中选择该 Bucket，把 Bucket 子目录 `/sam3d` 以读写方式挂载到本地目录 `/mnt/nas/sam3d`。最终容器必须能看到：

```text
/mnt/nas/sam3d/hf/pipeline.yaml
/mnt/nas/sam3d/hf/moge/model.pt
/mnt/nas/sam3d/cache/torch/hub/facebookresearch_dinov2_main/
/mnt/nas/sam3d/cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
```

不要把 Hugging Face Token、OSS AccessKey 或其他密钥放入 `Dockerfile`、构建参数、`.env` 文件或 Git 仓库。FC 应通过函数角色和控制台挂载配置访问 OSS。

## 2. 构建镜像

推荐在可控的 x86_64 Linux 构建机上构建。即使构建机没有 GPU，Dockerfile 也会通过 `FORCE_CUDA=1` 强制构建 PyTorch3D CUDA 扩展；不过上游仍提示，部分环境可能需要在带 GPU 的计算节点完成构建。CUDA 扩展构建通过 `micromamba run` 激活 Conda 环境，使 `targets/x86_64-linux` 下的头文件和库进入编译参数；构建前还会实际预处理 `cuda_runtime_api.h`，避免长时间编译后才暴露环境错误。

`requirements-fc.txt` 不再引用上游的全量 `requirements.txt`、`p3d` 或 `inference` extra。上游包、MoGe、utils3d、PyTorch3D 和 gsplat 均以固定 commit 和 `--no-deps` 安装；构建期完整性检查同时确认以下组件没有被任何传递依赖重新带入镜像：

- `mosaicml-streaming`、`librosa`、SageMaker、Lightning 等训练或云平台依赖；
- Jupyter/IPython/Notebook 组件；
- Gradio、Seaborn、Matplotlib、Open3D、Kaolin 及当前关闭的网格修补/可视化组件；
- pytest、Black、Flake8 等测试和开发工具；
- FlashAttention 与 xformers。FC `fc.gpu.ada.1` 使用 PyTorch SDPA。

普通包只访问 `PYPI_INDEX_URL`。阿里云镜像缺少 Hydra 1.3.2 所需的旧版 ANTLR，因此该单一纯 Python 包在清单中使用带 SHA-256 的 PyPI 官方文件 URL；不会为整次解析开放第二个索引。PyTorch 专用索引只出现在安装 `torch`/`torchvision` 的单独命令中，不再通过全局 `PIP_EXTRA_INDEX_URL` 影响其他依赖解析。当前最小清单没有 NGC-only 包，因此镜像构建完全不配置 NGC 索引；以后如恢复此类包，应只在对应的单条安装命令上指定索引。

默认锁定的上游版本是：

```text
f91db411c50efee93d8db7aeb323885650f6f722
```

构建命令：

```bash
docker buildx build \
  --load \
  --progress=plain \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --build-arg SAM3D_REF=f91db411c50efee93d8db7aeb323885650f6f722 \
  --build-arg TORCH_CUDA_ARCH_LIST=8.9 \
  --build-arg MAX_JOBS=2 \
  --build-arg NVCC_THREADS=2 \
  -t sam3d-fc:cu121 .
```

`--provenance=false` 和 `--sbom=false` 是 FC 兼容性要求，必须保留在每一条 `docker buildx build` 命令中，包括直接使用 `--push` 的命令和 CI/CD 配置。较新的 Buildx 默认会把 provenance 作为额外的 attestation manifest 附加到镜像索引；该 manifest 的平台会显示为 `unknown/unknown`，导致 FC 拒绝镜像或无法完成镜像加速。这两个选项属于 Buildx 输出配置，不能写进 Dockerfile 或作为普通 `--build-arg` 传入。

可用构建参数：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `MICROMAMBA_VERSION` | `2.8.1-0` | 固定 micromamba 版本 |
| `MICROMAMBA_SHA256` | 对应官方发布文件的 SHA-256 | 校验 micromamba 下载内容 |
| `SAM3D_REPOSITORY` | Meta 官方仓库 URL | 指定上游仓库 |
| `SAM3D_REF` | 固定 commit | 固定可复现的上游版本 |
| `PYPI_INDEX_URL` | 阿里云 PyPI 镜像 | 普通 Python 包索引 |
| `PYTORCH_INDEX_URL` | PyTorch cu121 官方索引 | 仅安装 PyTorch/torchvision |
| `TORCH_CUDA_ARCH_LIST` | `8.9` | 目标 GPU Compute Capability |
| `MAX_JOBS` | `2` | CUDA/C++ 并行编译任务数 |
| `NVCC_THREADS` | `2` | NVCC 并行线程数 |

不要把 `SAM3D_REF` 改成浮动的 `main` 用于生产镜像。`patches/fc-runtime.patch` 会在上下文不匹配时主动让构建失败；升级上游版本时，应重新审计并生成该补丁，同时复核其 `environments/default.yml`、依赖版本和推理输出结构。

## 3. 本地 GPU 验证

下面的命令由你在带 NVIDIA Driver 和 NVIDIA Container Toolkit 的 Linux 机器上执行：

```bash
docker run --rm \
  --gpus all \
  -p 9000:9000 \
  -v /mnt/nas/sam3d:/mnt/nas/sam3d \
  sam3d-fc:cu121
```

容器启动后可依次检查：

```bash
curl http://127.0.0.1:9000/healthz
curl http://127.0.0.1:9000/gpu
curl -X POST http://127.0.0.1:9000/initialize
```

生成 PLY：

```bash
curl -X POST http://127.0.0.1:9000/generate \
  -F "image=@image.png" \
  -F "mask=@mask.png" \
  -F "seed=42" \
  -F "output_format=ply" \
  -o sam3d-result.ply
```

生成 GLB：

```bash
curl -X POST http://127.0.0.1:9000/generate \
  -F "image=@image.png" \
  -F "mask=@mask.png" \
  -F "seed=42" \
  -F "output_format=glb" \
  -o sam3d-result.glb
```

Mask 必须与输入图片同宽同高。单通道 Mask 的非零像素表示目标；对于 RGB/RGBA Mask，服务与上游工具保持一致，使用最后一个通道。

FC HTTP 同步调用的请求体总上限为 32 MB。本项目默认把两个文件合计限制为 30 MB，为 multipart 元数据预留空间；更大的输入应先上传 OSS，再扩展服务为受控的 OSS 对象读取流程。

## 4. 推送到阿里云 ACR

在同一阿里云账号下创建与 FC 同地域的 ACR 仓库后，替换下面的地域、命名空间和仓库名。每次发布使用不可变标签，避免 FC 已记录的 Digest 与被覆盖后的标签内容不一致：

```bash
REGISTRY_HOST='registry.cn-hangzhou.aliyuncs.com'
IMAGE_REPOSITORY='YOUR_NAMESPACE/sam3d-fc'
IMAGE_TAG="cu121-$(git rev-parse --short HEAD)"
REMOTE_IMAGE="${REGISTRY_HOST}/${IMAGE_REPOSITORY}:${IMAGE_TAG}"

docker login "$REGISTRY_HOST"
docker tag sam3d-fc:cu121 "$REMOTE_IMAGE"
docker push "$REMOTE_IMAGE"
```

也可以跳过本地 `--load`，直接从 Buildx 推送到 ACR。FC 兼容选项仍然必须显式指定：

```bash
docker buildx build \
  --push \
  --progress=plain \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --build-arg SAM3D_REF=f91db411c50efee93d8db7aeb323885650f6f722 \
  --build-arg TORCH_CUDA_ARCH_LIST=8.9 \
  --build-arg MAX_JOBS=2 \
  --build-arg NVCC_THREADS=2 \
  -t "$REMOTE_IMAGE" .
```

推送后、更新 FC 前检查远程 Manifest：

```bash
MANIFEST_INFO="$(docker buildx imagetools inspect "$REMOTE_IMAGE")"
printf '%s\n' "$MANIFEST_INFO"

if printf '%s\n' "$MANIFEST_INFO" | grep -q 'unknown/unknown'; then
  printf '%s\n' '错误：镜像包含 FC 不支持的 unknown/unknown manifest' >&2
  exit 1
fi

docker manifest inspect --verbose "$REMOTE_IMAGE"
```

只有在检查结果确认为 `linux/amd64` 且不包含 `unknown/unknown` 后，才能把这个新标签配置到 FC。若已经生成了含 attestation 的旧标签，不要继续让 FC 引用它；使用上述兼容选项重新推送一个新标签。构建机与 ACR 不在同一地域或 VPC 时，应使用 ACR 公网地址，而不是带 `-vpc` 的内网地址。

对于此类包含大量 CUDA/C++ 扩展的镜像，优先选择 ACR“本地仓库”，在自己的构建机完成构建后推送。若使用 GitHub、Codeup 或 GitLab 作为 ACR 代码源，构建上下文使用仓库根目录，Dockerfile 路径使用 `/Dockerfile`，并提前确认在线构建的网络、内存、磁盘和超时限制。

若要把当前目录提交为 Git 代码源，由你在本机执行：

```bash
git init
git add .
git commit -m "创建 SAM3D FC 容器服务"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

## 5. 创建 FC GPU 函数

建议配置：

| 配置项 | 建议值 |
| --- | --- |
| 运行时 | 自定义容器 |
| 镜像架构 | `linux/amd64` |
| GPU 规格 | `fc.gpu.ada.1` 48 GB，或其他满足上游要求的规格 |
| CAPort | `9000` |
| Command / Args | 留空，使用镜像 `CMD` |
| 单实例并发 | `1` |
| 函数超时 | 按实测推理时间设置，并为首次加载预留时间 |
| 临时磁盘 | 按最大 GLB/PLY 输出和并发量预留，建议从 10 GB 起验证 |
| OSS 挂载 | Bucket 子目录 `/sam3d` 读写挂载到 `/mnt/nas/sam3d`；`hf/` 存 checkpoint，`cache/` 可写 |
| HTTP 触发器 | 使用需要鉴权的方式，除非明确接受公网匿名访问 |
| 实例策略 | 对延迟敏感时配置最小实例数或常驻实例 |

环境变量配置完成后更新函数并发布新版本，使后续弹性实例统一继承镜像、挂载与变量。部署后先调用 `/gpu` 确认 CUDA 可用，再调用 `/initialize` 主动加载模型。若初始化时间或冷启动不可接受，应使用常驻实例、最小实例数或浅休眠策略。

截至 2026-08-12，FC 官方自定义容器文档规定 GPU 镜像的未压缩大小上限为 15 GB。Dockerfile 使用双阶段构建，并在所有 CUDA 扩展编译完成后保守移除 Conda 构建链、调试工具、头文件和静态库；删除前的 dry-run 门禁会在计划触及关键运行库时让构建失败。即使已切换到 FC 最小 pip 依赖，PyTorch/CUDA 运行库和模型代码仍然很大，推送前必须实际检查镜像的未压缩大小，超限时不能部署。

## HTTP 接口

### `GET /healthz`

仅检查 HTTP 进程是否存活，不触发模型加载。

### `GET /readyz`

返回 checkpoint 配置是否存在、模型是否已经加载，以及最近一次加载错误。

### `GET /gpu`

返回 PyTorch、CUDA 和首张 GPU 的基本信息，不加载 SAM 3D Objects 模型。

### `POST /initialize`

幂等地加载模型。成功后返回：

```json
{
  "initialized": true,
  "config_path": "/mnt/nas/sam3d/hf/pipeline.yaml"
}
```

### `POST /generate`

请求类型为 `multipart/form-data`：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `image` | 文件 | 是 | RGB/RGBA 输入图片 |
| `mask` | 文件 | 是 | 与图片同尺寸的 Mask |
| `seed` | 整数 | 否 | 默认 `42` |
| `output_format` | `ply` 或 `glb` | 否 | 默认 `ply` |

成功时直接返回二进制文件。服务会在响应发送完成或客户端下载中断后删除本次请求的临时目录。

### `POST /invoke`

FC 通过 SDK 的 `InvokeFunction` 调用时会转发到固定路径 `/invoke`。本项目的推理输入包含两个二进制文件，因此推荐使用 HTTP 触发器直接调用 `/generate`；`/invoke` 仅返回这项提示，不执行推理。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LIDRA_SKIP_INIT` | `true` | 跳过上游训练期全局初始化；FC 最小运行时必须保持为 `true` |
| `ATTN_BACKEND` | `sdpa` | 稠密注意力后端；与精简掉的 FlashAttention/xformers 对应 |
| `SPARSE_ATTN_BACKEND` | `sdpa` | 稀疏注意力后端；FC Ada 镜像固定使用 SDPA |
| `SPARSE_BACKEND` | `spconv` | 稀疏卷积后端 |
| `SAM3D_ROOT` | `/opt/sam-3d-objects` | 上游源码目录 |
| `SAM3D_CONFIG_PATH` | `/mnt/nas/sam3d/hf/pipeline.yaml` | 模型配置文件 |
| `TORCH_HOME` | `/mnt/nas/sam3d/cache/torch` | Torch Hub 持久化缓存 |
| `HF_HOME` | `/mnt/nas/sam3d/cache/huggingface` | Hugging Face Hub 持久化缓存 |
| `HF_HUB_OFFLINE` | `1` | 禁止 Hugging Face Hub 发起网络请求 |
| `TRANSFORMERS_OFFLINE` | `1` | 禁止 Transformers 尝试下载模型或配置 |
| `HF_DATASETS_OFFLINE` | `1` | 禁止 Hugging Face Datasets 尝试联网 |
| `HF_HUB_DISABLE_TELEMETRY` | `1` | 关闭 Hugging Face 遥测 |
| `SAM3D_COMPILE` | `false` | 是否启用上游模型编译 |
| `SAM3D_MAX_UPLOAD_MB` | `20` | 单个上传文件上限 |
| `SAM3D_MAX_REQUEST_MB` | `30` | 图片与 Mask 合计上限，低于 FC 的 32 MB 请求体限制 |
| `SAM3D_MAX_IMAGE_PIXELS` | `40000000` | 单张图片或 Mask 的像素上限 |
| `SAM3D_TMP_DIR` | `/tmp/sam3d` | 单次推理输出目录的父目录 |
| `PORT` | `9000` | HTTP 监听端口，必须与 FC CAPort 一致 |
| `KEEP_ALIVE_TIMEOUT` | `900` | Uvicorn Keep-Alive 秒数 |

镜像已经默认设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1`，深圳 FC 的函数配置中也应显式保留这些值。它们不是一次性的登录操作：环境变量和 OSS 挂载属于函数版本配置，每个扩容出来的新实例都会自动继承。

模型初始化还会安装进程级 Torch Hub 门禁：只允许 `/mnt/nas/sam3d/cache/torch/hub/facebookresearch_dinov2_main`，把遗留的 `facebookresearch/dinov2` 请求重定向到这个本地目录，并直接拒绝任何 Torch Hub 下载。FC 运行时因此不会为了模型加载访问 Hugging Face、GitHub 或 DINO 下载站点；任一离线文件缺失时会立即返回明确错误，而不是回退到公网。

SAM3D 的稀疏结构和结构化隐变量生成器各有一个 DINO 条件编码器，因此日志中出现两次 `Loading DINO model` 是两份模型实例，不是同一权重下载两次。新镜像的日志应显示本地目录和 `source: local`；离线模式同时传入 `pretrained=False`，随后由两个官方主 checkpoint 以 `strict=True` 分别加载完整参数，避免重复读取外部 DINO 预训练权重。

## 已知边界与风险

- 本项目没有应用层鉴权、配额和限流，生产环境应使用 FC 触发器鉴权、API 网关或同等控制。
- `/generate` 是同步接口；大文件响应、超长推理或客户端断连时，建议进一步改为 OSS/NAS 结果落盘和异步任务接口。
- 上游当前推理入口会同时生成 Gaussian 与 Mesh；因此即使只请求 PLY，仍可能承担 Mesh/GLB 相关的计算和显存开销。若实测成为瓶颈，需要在固定上游版本上改造解码流程。
- FC 最小运行时固定关闭网格修补、纹理烘焙和布局后处理。若要调用上游 Notebook 演示、平面估计、场景拼接、纹理烘焙或网格修补函数，必须先恢复对应依赖并更新构建期禁止清单；这些能力不属于当前 HTTP API。
- 当前 checkpoint 必须是普通文件。若改用 Lightning 分片 checkpoint，需要恢复 Lightning 依赖；普通官方 checkpoint 的加载路径不需要它。
- 仅上传主 checkpoint 不足以离线初始化。部署前必须运行 `scripts/prepare_offline_assets.py --verify-only`，确认主 checkpoint、MoGe、DINOv2 源码和权重全部就位；FC 开启 `HF_HUB_OFFLINE=1` 后不再依赖预热实例或公网。
- 模型加载与推理尚未在你的目标 FC 实例上执行。GPU 型号、显存、驱动兼容性、CUDA 扩展、峰值显存、推理耗时和输出体积都必须由你实际验证。
- PLY 是上游 README 明确给出的导出路径。GLB 使用上游结果中的 `glb` 网格对象导出，视觉质量和坐标语义应按业务样本验收。
- 不要在镜像内安装 NVIDIA Driver，不要用 `docker commit` 保存一个已注入宿主机驱动的运行容器。
