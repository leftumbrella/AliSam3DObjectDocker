# SAM 3D Objects on Alibaba Cloud Function Compute

这是一个可直接作为 Docker 构建上下文或 ACR 代码源使用的项目。镜像在构建阶段拉取并安装 Meta 的 SAM 3D Objects，在运行阶段提供适配阿里云函数计算（FC）自定义容器的 HTTP 服务。

服务接收一张图片和一张对象 Mask，返回以下任一结果：

- `PLY`：官方示例使用的 Gaussian Splat。
- `GLB`：SAM 3D Objects 解码得到的网格。

模型权重不进入镜像，默认把 NAS 根目录挂载到 `/mnt/nas/sam3d`，其中 `hf/` 保存 checkpoint，`cache/` 保存上游运行时缓存。这样不会把 Hugging Face Token 写入镜像，也能降低镜像体积。该路径位于 FC 允许的 NAS 本地挂载目录 `/mnt` 下。

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

## 1. 准备 checkpoint

先在 [Hugging Face 模型页面](https://huggingface.co/facebook/sam-3d-objects)申请权限，并在一台已安装 Hugging Face CLI 的 Linux 机器上登录：

```bash
hf auth login
```

把模型直接下载到 NAS。例如：

```bash
hf download \
  --repo-type model \
  --local-dir /mnt/nas/sam3d/hf-download \
  --max-workers 1 \
  facebook/sam-3d-objects
```

下载完成后应存在：

```text
/mnt/nas/sam3d/hf-download/checkpoints/pipeline.yaml
```

按上游安装说明整理 checkpoint，并准备运行时缓存目录：

```bash
mv /mnt/nas/sam3d/hf-download/checkpoints /mnt/nas/sam3d/hf
mkdir -p /mnt/nas/sam3d/cache/torch
mkdir -p /mnt/nas/sam3d/cache/huggingface
```

最终 NAS 目录应类似：

```text
/mnt/nas/sam3d/
├── hf/
│   ├── pipeline.yaml
│   └── ...
└── cache/
    ├── torch/
    └── huggingface/
```

在 FC 中把这个 NAS 根目录挂载到 `/mnt/nas/sam3d`。容器最终看到的配置文件应为：

```text
/mnt/nas/sam3d/hf/pipeline.yaml
```

不要把 Hugging Face Token 放入 `Dockerfile`、普通构建参数、环境变量文件或 Git 仓库。

上游模型初始化还会通过 Torch Hub/Hugging Face Hub 获取 DINOv2 等依赖模型。首次预热必须允许函数访问相关站点；缓存会写入 NAS 的 `cache/`，后续实例可以复用。若生产环境禁止出网，请先在单个可出网 GPU 实例上完成 `/initialize`，确认缓存齐全后再关闭出网；首次预热期间不要并发启动多个实例。

## 2. 构建镜像

推荐在可控的 x86_64 Linux 构建机上构建。即使构建机没有 GPU，Dockerfile 也会通过 `FORCE_CUDA=1` 强制构建 PyTorch3D CUDA 扩展；不过上游仍提示，部分环境可能需要在带 GPU 的计算节点完成构建。

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
docker build \
  --platform linux/amd64 \
  --build-arg SAM3D_REF=f91db411c50efee93d8db7aeb323885650f6f722 \
  --build-arg TORCH_CUDA_ARCH_LIST=8.9 \
  -t sam3d-fc:cu121 .
```

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
| `MAX_JOBS` | `4` | CUDA/C++ 并行编译任务数 |
| `NVCC_THREADS` | `4` | NVCC 并行线程数 |

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

在同一阿里云账号下创建与 FC 同地域的 ACR 仓库后，替换下面的地域、命名空间和仓库名：

```bash
docker login registry.cn-hangzhou.aliyuncs.com

docker tag \
  sam3d-fc:cu121 \
  registry.cn-hangzhou.aliyuncs.com/YOUR_NAMESPACE/sam3d-fc:cu121

docker push \
  registry.cn-hangzhou.aliyuncs.com/YOUR_NAMESPACE/sam3d-fc:cu121
```

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
| NAS 挂载 | NAS 根目录映射到 `/mnt/nas/sam3d`；`hf/` 存 checkpoint，`cache/` 可写 |
| HTTP 触发器 | 使用需要鉴权的方式，除非明确接受公网匿名访问 |
| 实例策略 | 对延迟敏感时配置最小实例数或常驻实例 |

部署后先调用 `/gpu` 确认 CUDA 可用，再调用 `/initialize` 主动加载模型。若初始化时间或冷启动不可接受，应使用常驻实例、最小实例数或浅休眠策略。

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
| `SAM3D_COMPILE` | `false` | 是否启用上游模型编译 |
| `SAM3D_MAX_UPLOAD_MB` | `20` | 单个上传文件上限 |
| `SAM3D_MAX_REQUEST_MB` | `30` | 图片与 Mask 合计上限，低于 FC 的 32 MB 请求体限制 |
| `SAM3D_MAX_IMAGE_PIXELS` | `40000000` | 单张图片或 Mask 的像素上限 |
| `SAM3D_TMP_DIR` | `/tmp/sam3d` | 单次推理输出目录的父目录 |
| `PORT` | `9000` | HTTP 监听端口，必须与 FC CAPort 一致 |
| `KEEP_ALIVE_TIMEOUT` | `900` | Uvicorn Keep-Alive 秒数 |

## 已知边界与风险

- 本项目没有应用层鉴权、配额和限流，生产环境应使用 FC 触发器鉴权、API 网关或同等控制。
- `/generate` 是同步接口；大文件响应、超长推理或客户端断连时，建议进一步改为 OSS/NAS 结果落盘和异步任务接口。
- 上游当前推理入口会同时生成 Gaussian 与 Mesh；因此即使只请求 PLY，仍可能承担 Mesh/GLB 相关的计算和显存开销。若实测成为瓶颈，需要在固定上游版本上改造解码流程。
- FC 最小运行时固定关闭网格修补、纹理烘焙和布局后处理。若要调用上游 Notebook 演示、平面估计、场景拼接、纹理烘焙或网格修补函数，必须先恢复对应依赖并更新构建期禁止清单；这些能力不属于当前 HTTP API。
- 当前 checkpoint 必须是普通文件。若改用 Lightning 分片 checkpoint，需要恢复 Lightning 依赖；普通官方 checkpoint 的加载路径不需要它。
- 仅挂载 checkpoint 不足以离线初始化。部署前必须完成 DINOv2 等运行时缓存预热，或保留受控出网能力。
- 模型加载与推理尚未在你的目标 FC 实例上执行。GPU 型号、显存、驱动兼容性、CUDA 扩展、峰值显存、推理耗时和输出体积都必须由你实际验证。
- PLY 是上游 README 明确给出的导出路径。GLB 使用上游结果中的 `glb` 网格对象导出，视觉质量和坐标语义应按业务样本验收。
- 不要在镜像内安装 NVIDIA Driver，不要用 `docker commit` 保存一个已注入宿主机驱动的运行容器。
