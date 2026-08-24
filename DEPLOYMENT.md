# 阿里云 FC GPU 从零部署手册

这份手册覆盖完整部署链路：香港 ECS 下载代码和模型、构建镜像、上传深圳 OSS、推送深圳 ACR、创建深圳函数计算（FC）GPU 函数，以及上线后的验证和回滚。

文中的命令以 Ubuntu 22.04 x86_64 为基准，默认在香港 ECS 的 `root` 用户下执行。使用普通用户也可以，把系统安装命令加上 `sudo`，并把 `/root` 下的工作目录换成该用户有权限的路径。

## 部署结构

| 组件 | 地域 | 用途 |
| --- | --- | --- |
| 香港 ECS | 中国香港 | 访问 GitHub、Hugging Face 和模型下载站；构建镜像并向深圳上传 |
| OSS Bucket | 华南 1（深圳） | 保存主 checkpoint、MoGe、DINOv2 源码和权重 |
| ACR 镜像仓库 | 华南 1（深圳） | 保存 FC 自定义容器镜像 |
| FC GPU 函数 | 华南 1（深圳） | 挂载 OSS，加载模型并提供 HTTP API |

香港 ECS 使用 OSS 和 ACR 的公网地址。带 `-internal` 或 `-vpc` 的地址只适合同地域、同 VPC 的机器，香港 ECS 不能用深圳 VPC 地址。

## 开始前准备资源

### 香港 ECS

推荐配置如下。这不是 FC 的运行规格，只影响下载和镜像构建速度。

| 配置 | 建议值 |
| --- | --- |
| 操作系统 | Ubuntu 22.04 LTS 64 位 |
| 架构 | x86_64 / amd64 |
| vCPU | 8 核或更多 |
| 内存 | 32 GB 或更多 |
| 系统盘或数据盘 | 200 GB 或更多 |
| 公网 | 能访问 GitHub、Hugging Face、Meta 下载站、深圳 OSS 和深圳 ACR |

构建阶段不要求 GPU。若要在 ECS 上运行容器并验证推理，则需要 NVIDIA GPU、宿主机驱动和 NVIDIA Container Toolkit。

安全组只需开放 SSH，并限制来源 IP。不要为了本地容器测试把 9000 端口长期暴露到公网。

### 深圳 OSS

在 OSS 控制台创建 Bucket：

- 地域选择华南 1（深圳）。
- Bucket 名必须全局唯一，只能包含小写字母、数字和连字符。
- 读写权限选择私有。
- 使用标准存储，不要把在线推理所需模型放进归档存储。

记录真实 Bucket 名。FC 自动生成的组件名、资源 ID 和页面标题都不是 Bucket 名。

为上传资源创建专用 RAM 用户或临时凭证，只授予目标 Bucket 的列举、上传和读取权限。不要使用阿里云主账号 AccessKey，也不要把 AccessKey 写进 Git、Dockerfile、镜像构建参数或 FC 环境变量。

使用 RAM 用户时，在 RAM 控制台进入“身份管理 > 用户”，创建仅用于这次上传的用户，并在该用户的“认证管理 > AccessKey”中创建 AccessKey。AccessKey Secret 只会完整显示一次，应临时保存在密码管理器中。

创建下面的自定义权限策略，把两处 `YOUR_BUCKET_NAME` 换成真实 Bucket 名，再把策略授予上传用户：

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "oss:ListObjects"
      ],
      "Resource": [
        "acs:oss:*:*:YOUR_BUCKET_NAME"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "oss:GetObject",
        "oss:PutObject",
        "oss:ListParts",
        "oss:AbortMultipartUpload"
      ],
      "Resource": [
        "acs:oss:*:*:YOUR_BUCKET_NAME/*"
      ]
    }
  ]
}
```

这份策略只能操作目标 Bucket，不需要授予 `AliyunOSSFullAccess`。如果组织要求使用 STS 临时凭证，`ossutil config` 时还要填写对应的 STS Token。

### 深圳 ACR

在容器镜像服务控制台创建深圳实例、命名空间和仓库：

- 使用 ACR 个人版或企业版。ACR 经济版不支持 FC 需要的镜像加速。
- 地域选择华南 1（深圳），与 FC 保持一致。
- ACR、FC 和目标镜像应位于同一阿里云账号。跨账号镜像访问需要额外授权，不属于本手册的默认路径。
- 记录公网登录地址，例如 `crpi-xxxx.cn-shenzhen.personal.cr.aliyuncs.com`。
- 记录仓库路径，例如 `namespace/sam3dobject`。
- 在访问凭证页面设置 Registry 登录密码。

香港 ECS 必须使用公网地址，不要使用包含 `-vpc` 的深圳内网地址。

### Hugging Face 权限

打开 [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects)，登录 Hugging Face，申请模型访问权限并接受许可证。没有通过访问申请时，主 checkpoint 下载会返回 401 或 403。

## 初始化香港 ECS

先确认系统架构、内存和磁盘：

```bash
uname -m
cat /etc/os-release
free -h
df -h /
```

`uname -m` 必须输出 `x86_64`。磁盘空间不足时先扩容，不要等 CUDA 扩展构建到一半再清理 Docker 数据。

安装基础工具：

```bash
apt-get update
apt-get install -y \
  ca-certificates \
  curl \
  git \
  jq \
  python3 \
  python3-pip \
  python3-venv \
  tmux \
  unzip
```

按 Docker 官方 APT 仓库安装 Docker Engine 和 Buildx。这里显式安装 `docker-buildx-plugin`，不要只安装 Ubuntu 自带的旧版 `docker.io`：

```bash
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

cat >/etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

apt-get update
apt-get install -y \
  containerd.io \
  docker-buildx-plugin \
  docker-ce \
  docker-ce-cli \
  docker-compose-plugin

systemctl enable --now docker
docker version
docker buildx version
docker buildx inspect --bootstrap
```

如果 `docker buildx version` 仍然报缺少插件，先检查是否混装了发行版的 `docker.io` 和 Docker 官方软件包。不要用 `DOCKER_BUILDKIT=1 docker build` 绕过这个错误。

长时间下载和构建建议放进 `tmux`，SSH 断线后任务不会随终端退出：

```bash
tmux new -s sam3d-deploy
```

重新连接后使用 `tmux attach -t sam3d-deploy` 回到会话。

## 下载项目代码

```bash
cd /root
git clone https://github.com/leftumbrella/AliSam3DObjectDocker.git
cd /root/AliSam3DObjectDocker

git fetch origin
git pull --ff-only origin main
git status --short --branch
git rev-parse HEAD
```

后续镜像标签使用当前 Git commit，确保 FC 上的镜像可以追溯到源码。

## 准备完整离线模型资源

用独立 Python 虚拟环境安装 Hugging Face CLI：

```bash
python3 -m venv /opt/sam3d-tools
/opt/sam3d-tools/bin/python -m pip install --upgrade pip huggingface_hub
export PATH="/opt/sam3d-tools/bin:$PATH"

hf version
hf auth login
hf auth whoami
```

登录时在交互提示里完成浏览器授权或粘贴只读 Token。不要把 Token 直接写在命令行里，命令行参数可能进入 shell 历史和进程列表。

第一次准备资源：

```bash
cd /root/AliSam3DObjectDocker
export PATH="/opt/sam3d-tools/bin:$PATH"
export TRANSFER_ROOT='/root/sam3d-transfer'

python3 scripts/prepare_offline_assets.py \
  --download-sam3d \
  --transfer-root "$TRANSFER_ROOT"
```

如果主 checkpoint 已经下载过，可以改用现有 `checkpoints` 目录，避免重新下载：

```bash
python3 scripts/prepare_offline_assets.py \
  --sam3d-source /实际路径/checkpoints \
  --transfer-root "$TRANSFER_ROOT"
```

脚本会完成这些工作：

- 下载固定 revision 的 SAM 3D Objects 主 checkpoint。
- 下载并校验 DINOv2 源码和权重。
- 下载并校验 MoGe `model.pt`。
- 把 `pipeline.yaml` 里的 MoGe 仓库名换成 `/mnt/nas/sam3d/hf/moge/model.pt`。
- 检查主 checkpoint 文件名和精确字节数。
- 生成 `offline-assets.sha256`，用于发现上传前的文件损坏。

下载中断后重跑同一命令。DINOv2 和 MoGe 的 `.partial` 文件会断点续传，已通过完整性校验的正式文件会直接复用。

上传前执行只读校验：

```bash
python3 scripts/prepare_offline_assets.py \
  --verify-only \
  --transfer-root "$TRANSFER_ROOT"

du -sh "$TRANSFER_ROOT/storage"
test -f "$TRANSFER_ROOT/storage/offline-assets.sha256"
test -f "$TRANSFER_ROOT/storage/hf/pipeline.yaml"
test -f "$TRANSFER_ROOT/storage/hf/moge/model.pt"
test -f "$TRANSFER_ROOT/storage/cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
test -f "$TRANSFER_ROOT/storage/cache/torch/hub/facebookresearch_dinov2_main/hubconf.py"
```

任何一个 `test` 返回非零都不要继续部署。

## 上传离线资源到深圳 OSS

安装官方 `ossutil 2.0` Linux x86_64 版本。下面固定使用 2.3.0，并在安装前校验阿里云文档公布的 SHA-256：

```bash
cd /tmp
export OSSUTIL_VERSION='2.3.0'
export OSSUTIL_ARCHIVE="ossutil-${OSSUTIL_VERSION}-linux-amd64.zip"

curl -fLO \
  "https://gosspublic.alicdn.com/ossutil/v2/${OSSUTIL_VERSION}/${OSSUTIL_ARCHIVE}"
printf '%s  %s\n' \
  '3ae4d9fc85a7a6e9f5654d1599766f1a3a42a3692870887b5ae9338d582ef65a' \
  "$OSSUTIL_ARCHIVE" | sha256sum -c -

unzip -o "$OSSUTIL_ARCHIVE"
install -m 0755 \
  "ossutil-${OSSUTIL_VERSION}-linux-amd64/ossutil" \
  /usr/local/bin/ossutil

ossutil --version
ossutil help cp | grep -q -- '--checkpoint-dir'
```

如果官方版本已经更新，先从 [ossutil 官方安装文档](https://help.aliyun.com/zh/oss/install-ossutil2) 取得新下载地址和校验值，不要只改版本号而继续使用旧 SHA-256。

运行配置向导：

```bash
ossutil config
```

配置向导中填写：

| 提示 | 填写内容 |
| --- | --- |
| 配置文件 | 使用默认路径 |
| AccessKey ID | 专用 RAM 用户或临时凭证的 AccessKey ID |
| AccessKey Secret | 对应的 AccessKey Secret |
| Region | `cn-shenzhen` |
| Endpoint | `https://oss-cn-shenzhen.aliyuncs.com`，或留空使用 Region 对应的公网地址 |

配置文件包含凭证，限制其读取权限：

```bash
chmod 600 /root/.ossutilconfig 2>/dev/null || true
```

设置本次发布的非敏感变量。`OSS_BUCKET` 必须改成控制台里真实存在的 Bucket 名：

```bash
export OSS_BUCKET=''
export ASSET_RELEASE="bundle-$(git rev-parse --short=12 HEAD)"
export OSS_PREFIX="sam3d/releases/${ASSET_RELEASE}"

: "${OSS_BUCKET:?请填写深圳 OSS Bucket 名称}"
printf 'OSS target: oss://%s/%s/\n' "$OSS_BUCKET" "$OSS_PREFIX"
```

先确认 Bucket 存在且当前凭证有权限：

```bash
ossutil ls "oss://${OSS_BUCKET}/"
```

出现 `NoSuchBucket` 时不要继续上传。回到 OSS 控制台核对 Bucket 的真实名称和地域，不能使用 FC 组件名、资源 ID 或大写名称。

上传 `storage/` 目录的内容。模型约 14 GB，命令中断后可以原样重跑：

```bash
ossutil cp -r \
  "$TRANSFER_ROOT/storage/" \
  "oss://${OSS_BUCKET}/${OSS_PREFIX}/" \
  --checkpoint-dir /root/oss-upload-checkpoints
```

这里使用版本化前缀，不覆盖上一次部署。FC 挂载的是该前缀，容器内路径仍保持 `/mnt/nas/sam3d`。

上传后检查关键对象：

```bash
ossutil ls "oss://${OSS_BUCKET}/${OSS_PREFIX}/offline-assets.sha256"
ossutil ls "oss://${OSS_BUCKET}/${OSS_PREFIX}/hf/pipeline.yaml"
ossutil ls "oss://${OSS_BUCKET}/${OSS_PREFIX}/hf/moge/model.pt"
ossutil ls "oss://${OSS_BUCKET}/${OSS_PREFIX}/cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
ossutil ls "oss://${OSS_BUCKET}/${OSS_PREFIX}/cache/torch/hub/facebookresearch_dinov2_main/hubconf.py"
```

不要删除上一个可用前缀。保留旧资源可以在 FC 更新失败时直接回滚挂载配置。

## 构建 FC 镜像

先设置镜像变量。ACR Host 必须从深圳 ACR 控制台复制公网地址：

```bash
cd /root/AliSam3DObjectDocker

export ACR_HOST=''
export ACR_REPOSITORY=''
export ACR_USERNAME=''
export IMAGE_TAG="cu121-$(git rev-parse --short=12 HEAD)"
export LOCAL_IMAGE="sam3d-fc:${IMAGE_TAG}"

: "${ACR_HOST:?请填写深圳 ACR 公网地址}"
: "${ACR_REPOSITORY:?请填写 namespace/repository}"
: "${ACR_USERNAME:?请填写 ACR 登录用户名}"

export REMOTE_IMAGE="${ACR_HOST}/${ACR_REPOSITORY}:${IMAGE_TAG}"
printf 'Local image:  %s\nRemote image: %s\n' "$LOCAL_IMAGE" "$REMOTE_IMAGE"
```

香港 ECS 的 `ACR_HOST` 示例格式为：

```text
crpi-xxxx.cn-shenzhen.personal.cr.aliyuncs.com
```

不要填成：

```text
crpi-xxxx-vpc.cn-shenzhen.personal.cr.aliyuncs.com
```

使用本地 `--load` 构建。镜像先保留在香港 ECS，后续即使 ACR 推送中断，也只需重跑 `docker push`，不用重新编译 PyTorch3D 和 gsplat：

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
  -t "$LOCAL_IMAGE" \
  .
```

`--provenance=false` 和 `--sbom=false` 必须保留。否则较新的 Buildx 会附加平台为 `unknown/unknown` 的 attestation manifest，FC 可能拒绝镜像或一直等待镜像加速。

构建完成后检查架构和本地尺寸：

```bash
docker image inspect "$LOCAL_IMAGE" \
  --format 'platform={{.Os}}/{{.Architecture}} bytes={{.Size}}'
docker image ls "$LOCAL_IMAGE"
```

平台必须是 `linux/amd64`。FC 当前允许的 GPU 自定义容器镜像未压缩上限为 15 GB，超限时不能继续部署。

## 推送镜像到深圳 ACR

使用交互方式读取密码，避免密码进入 shell 历史：

```bash
read -rsp 'ACR Registry password: ' ACR_PASSWORD
printf '\n'
printf '%s' "$ACR_PASSWORD" | docker login \
  --username "$ACR_USERNAME" \
  --password-stdin \
  "$ACR_HOST"
unset ACR_PASSWORD
```

标记并推送：

```bash
docker tag "$LOCAL_IMAGE" "$REMOTE_IMAGE"
docker push "$REMOTE_IMAGE"
```

网络中断或出现 EOF 时，直接重跑同一条 `docker push "$REMOTE_IMAGE"`。Docker 会复用本地镜像和已上传的层，不要重新构建，也不要换成同标签的不同内容。

推送后检查远程 Manifest：

```bash
MANIFEST_INFO="$(docker buildx imagetools inspect "$REMOTE_IMAGE")"
printf '%s\n' "$MANIFEST_INFO"

if printf '%s\n' "$MANIFEST_INFO" | grep -q 'unknown/unknown'; then
  printf '%s\n' '错误：镜像包含 FC 不支持的 unknown/unknown manifest' >&2
  exit 1
fi

if ! printf '%s\n' "$MANIFEST_INFO" | grep -q 'linux/amd64'; then
  printf '%s\n' '错误：远程镜像不包含 linux/amd64 manifest' >&2
  exit 1
fi

docker manifest inspect --verbose "$REMOTE_IMAGE"
```

只有远程结果包含 `linux/amd64` 且不包含 `unknown/unknown`，才能把该标签配置到 FC。检查失败时用正确参数重新构建，并推送一个新标签，不要覆盖已经被 FC 引用的标签。

## 创建深圳 FC GPU 函数

在函数计算控制台切换到华南 1（深圳），创建 GPU 函数，运行时选择自定义容器。控制台字段可能调整名称，值按下表填写：

| 配置项 | 值 |
| --- | --- |
| 镜像 | 刚推送的完整 `REMOTE_IMAGE`，使用不可变标签 |
| 镜像架构 | `linux/amd64` |
| GPU 规格 | `fc.gpu.ada.1`，48 GB GPU 显存 |
| vCPU / 实例内存 | 稳定性优先使用 8 vCPU / 64 GB；验证峰值后可评估 32 GB |
| CAPort / 监听端口 | `9000` |
| Command | 留空，使用镜像 `CMD` |
| Args | 留空 |
| 单实例并发 | `1` |
| 函数超时 | 首次验证建议 `1800` 秒，稳定后按实测调整 |
| 临时磁盘 | 模型不放临时盘；如控制台提供规格选择，使用 10 GB 即可 |
| 公网访问 | 运行时不需要访问 Hugging Face、GitHub 或 DINO 下载站 |

在 FC 控制台进入“函数管理 > 函数列表”，把地域切换到华南 1（深圳），单击“创建函数”，选择 GPU 函数和弹性实例。函数代码选择“自定义镜像 > 使用 ACR 中的镜像”，再选择刚推送的不可变标签。不要把旧的 `unknown/unknown` 标签重新选回来。

FC `fc.gpu.ada.1` 会给单个容器一张 48 GB 显存的 GPU。实例内存和 GPU 显存不是同一个配置。模型、源码和权重放在 OSS 挂载中，不占用临时磁盘；临时磁盘只保存单次请求产生的 PLY 或 GLB，响应结束后会清理。

不要开启 FC 的实例预热回调。FC Initializer 的最长运行时间是 300 秒，而这个模型的实际初始化可能超过 10 分钟。本项目通过 `/initialize` 或第一次 `/generate` 加载模型。

### 配置 OSS 挂载

创建函数时，在“权限、网络、存储”中开启 OSS 挂载。函数已经存在时，进入“配置 > 高级配置 > 存储”，开启“挂载 OSS”，填写下表后部署：

| 字段 | 值 |
| --- | --- |
| Bucket | `$OSS_BUCKET` 对应的深圳 Bucket |
| Bucket 子目录 | `/$OSS_PREFIX`，例如 `/sam3d/releases/bundle-abc123` |
| 容器本地目录 | `/mnt/nas/sam3d` |
| OSS Endpoint | 使用控制台自动选择的深圳内网 Endpoint |
| 权限 | 只读 |

为函数选择专用执行角色，并授予该角色读取目标前缀的权限。创建下面的自定义策略时，把 `YOUR_BUCKET_NAME` 换成 Bucket 名，把 `YOUR_PREFIX` 换成不带开头和结尾斜杠的真实前缀，例如 `sam3d/releases/bundle-abc123`：

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "oss:ListObjects",
      "Resource": "acs:oss:*:*:YOUR_BUCKET_NAME",
      "Condition": {
        "StringLike": {
          "oss:Prefix": [
            "YOUR_PREFIX",
            "YOUR_PREFIX/*"
          ]
        }
      }
    },
    {
      "Effect": "Allow",
      "Action": "oss:GetObject",
      "Resource": "acs:oss:*:*:YOUR_BUCKET_NAME/YOUR_PREFIX/*"
    }
  ]
}
```

FC 会用执行角色的临时凭证完成挂载。不要把 OSS AccessKey 放进函数环境变量。因为 Bucket 和 FC 都在深圳，应使用自动选择的内网 Endpoint；香港 ECS 上传时才使用公网 Endpoint。

挂载后的容器必须看到这些路径：

```text
/mnt/nas/sam3d/offline-assets.sha256
/mnt/nas/sam3d/hf/pipeline.yaml
/mnt/nas/sam3d/hf/moge/model.pt
/mnt/nas/sam3d/cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
/mnt/nas/sam3d/cache/torch/hub/facebookresearch_dinov2_main/hubconf.py
```

### 配置环境变量

镜像已经包含默认值，FC 控制台仍建议显式配置，方便检查函数版本：

```text
LIDRA_SKIP_INIT=true
ATTN_BACKEND=sdpa
SPARSE_ATTN_BACKEND=sdpa
SPARSE_BACKEND=spconv
SAM3D_ROOT=/opt/sam-3d-objects
SAM3D_CONFIG_PATH=/mnt/nas/sam3d/hf/pipeline.yaml
TORCH_HOME=/mnt/nas/sam3d/cache/torch
HF_HOME=/mnt/nas/sam3d/cache/huggingface
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
PYTHONDONTWRITEBYTECODE=1
SAM3D_COMPILE=false
SAM3D_MAX_UPLOAD_MB=20
SAM3D_MAX_REQUEST_MB=30
SAM3D_MAX_IMAGE_PIXELS=40000000
SAM3D_TMP_DIR=/tmp/sam3d
PORT=9000
KEEP_ALIVE_TIMEOUT=900
```

不要设置 `SAM3D_DINOV2_REPO`，服务会根据 `TORCH_HOME` 自动生成正确的本地目录。

`KEEP_ALIVE_TIMEOUT` 是 Uvicorn 的 HTTP Keep-Alive 设置，不是 FC 函数执行超时。函数超时需要在 FC 资源配置中单独设置。

### 配置 HTTP 触发器

联调阶段可以临时创建匿名 HTTP 触发器。生产环境应启用 FC 签名认证或放到受控 API 网关后面，不能把 AccessKey 写进浏览器 JavaScript。

建议允许的方法：

```text
GET, POST
```

浏览器跨域联调时，在触发器的 CORS 配置中单独设置允许的 Origin、方法和请求头。不要把 `OPTIONS` 加进 `allowMethods`，FC 网关会处理预检请求。生产环境不要使用 `*` Origin 配合凭证。

HTTP 同步请求体上限为 32 MB。项目把图片和 Mask 的合计上限设为 30 MB，给 multipart 元数据留出空间。

### 配置弹性策略

- 单实例并发保持 `1`，避免同一 GPU 同时加载或执行多份推理。
- 联调和低延迟生产环境建议先保留 1 个最小实例或常驻实例。
- 成本优先时可以把最小实例数设为 0，但缩容后下一次请求会经历冷启动和本地模型加载。
- 每个扩容实例都会从同一个 OSS 挂载读取模型，不会重新构建镜像，也不会从 Hugging Face 或 GitHub 下载文件。
- 模型仍需在每个新实例中从 OSS 读入内存和 GPU。这个过程可能需要数分钟，`/initialize` 只会预热实际接到该请求的一个实例，不能一次预热所有未来实例。
- 当前同步接口适合单实例或低并发验证。生产流量若会扩到多个实例，应让调用方容忍首次加载时延，或改造成把输入和结果放入 OSS 的异步任务接口。

保存配置并发布新函数版本。等待镜像加速状态变为可用后再发送初始化请求。

## 上线验证

把触发器公网地址保存到变量中，不要在地址末尾加 `/`：

```bash
export FC_URL='https://your-function.cn-shenzhen.fcapp.run'
: "${FC_URL:?请填写 FC HTTP 触发器地址}"
```

依次检查 HTTP、OSS 挂载和 GPU：

```bash
curl -fsS "$FC_URL/healthz" | jq .
curl -fsS "$FC_URL/readyz" | jq .
curl -fsS "$FC_URL/gpu" | jq .
```

模型初始化前，`/readyz` 应满足：

```json
{
  "ready": false,
  "model_loaded": false,
  "config_present": true,
  "config_path": "/mnt/nas/sam3d/hf/pipeline.yaml",
  "last_load_error": null
}
```

`config_present` 为 `false` 时不要调用初始化，先修正 OSS Bucket 子目录和本地挂载路径。

主动初始化模型：

```bash
curl -fS \
  --max-time 1800 \
  -X POST \
  "$FC_URL/initialize" | jq .
```

模型有两个 DINO 条件编码器，因此日志里会出现两次 `Loading DINO model`。新镜像的两次日志都必须包含本地目录和 `source: local`：

```text
Loading DINO model: dinov2_vitl14_reg from /mnt/nas/sam3d/cache/torch/hub/facebookresearch_dinov2_main (source: local)
DINO backbone kwargs: {'pretrained': False}
```

如果仍然显示 `source: github`，FC 正在运行旧镜像或旧 Digest。核对函数版本中的完整镜像标签，并重新发布版本。

初始化完成后再次检查：

```bash
curl -fsS "$FC_URL/readyz" | jq .
```

此时 `ready` 和 `model_loaded` 应为 `true`。

使用一张图片和同尺寸 Mask 测试 PLY：

```bash
curl -fS \
  --max-time 1800 \
  -X POST \
  "$FC_URL/generate" \
  -F 'image=@image.png' \
  -F 'mask=@mask.png' \
  -F 'seed=42' \
  -F 'output_format=ply' \
  -o sam3d-result.ply

test -s sam3d-result.ply
ls -lh sam3d-result.ply
```

需要 GLB 时把 `output_format` 和输出文件后缀改成 `glb`。

## 常见错误

| 错误 | 原因 | 处理 |
| --- | --- | --- |
| `BuildKit is enabled but the buildx component is missing` | 只安装了旧 Docker 或 Buildx 插件损坏 | 按本手册安装 `docker-buildx-plugin`，再运行 `docker buildx version` |
| `NoSuchBucket` | Bucket 名写错、地域错误，或把组件 ID 当成 Bucket 名 | 在深圳 OSS 控制台复制真实小写 Bucket 名，先执行 `ossutil ls` |
| ACR 推送 EOF | 跨地域公网抖动 | 重跑同一条 `docker push`，本地镜像和已上传层会复用 |
| `platform of image is unknown/unknown` | Buildx 附加了 attestation manifest | 保留 `--provenance=false --sbom=false`，构建并推送新标签 |
| `accelerated image not ready` | FC 镜像加速尚未完成，或镜像 Manifest 不兼容 | 先检查远程 Manifest；兼容时等待加速状态可用，不兼容时换新标签重建 |
| `/readyz` 显示 `config_present=false` | OSS 前缀或本地挂载路径错误 | Bucket 子目录挂到 `/mnt/nas/sam3d`，确认 `hf/pipeline.yaml` 位于前缀根目录下 |
| 缺少 MoGe 或 DINOv2 文件 | 只上传了主 checkpoint，或上传中断 | 在香港 ECS 运行 `--verify-only`，重新上传同一资源前缀 |
| 日志显示 `source: github` | FC 仍引用旧镜像 | 换成新不可变标签，发布新函数版本 |
| `Invocation canceled by client` | 浏览器、控制台或调用方先超时 | 用 `curl --max-time 1800` 验证；生产环境保留最小实例，长任务再改异步接口 |
| OSS 挂载报 `invalid credentials` | 函数执行角色缺少 Bucket 或前缀权限 | 核对执行角色和本手册的只读策略，不要改成函数环境变量中的长期 AccessKey |
| GPU 不可用 | 选错函数规格或镜像 CUDA 扩展架构不匹配 | 使用 `fc.gpu.ada.1`，检查 `/gpu` 和 `TORCH_CUDA_ARCH_LIST=8.9` |

## 更新和回滚

### 更新代码

在香港 ECS 拉取代码，并记录新 commit：

```bash
cd /root/AliSam3DObjectDocker
git fetch origin
git pull --ff-only origin main
git rev-parse HEAD
```

只要模型资源版本没有变化，可以继续挂载原来的 OSS 资源前缀。运行一次 `--verify-only`，确认现有资源仍符合当前代码的离线清单。

每次镜像发布都使用新的 Git commit 标签，不覆盖旧标签。推送后先检查 Manifest，再更新 FC 镜像并发布新函数版本。

### 更新模型资源

脚本中的主 checkpoint、DINOv2 或 MoGe pin 变化时，使用新的 `ASSET_RELEASE` 前缀重新准备和上传。不要覆盖正在被生产函数挂载的前缀。

FC 更新时把镜像标签和 OSS 前缀作为同一批配置一起发布，避免新代码配旧资源或旧代码配新资源。

### 回滚

保留以下两项即可回滚：

- 上一个可用的 ACR 不可变镜像标签。
- 上一个可用的 OSS 版本化资源前缀。

在 FC 中把镜像和 OSS Bucket 子目录同时改回上一个版本，发布新函数版本，然后按 `/healthz`、`/readyz`、`/gpu`、`/initialize` 的顺序重新验证。回滚不需要重新下载模型、构建镜像或登录实例。

新版本稳定前，不要删除旧 ACR 标签和旧 OSS 前缀。

## 凭证收尾

资源上传和镜像推送完成后，可以退出本地登录：

```bash
hf auth logout
docker logout "$ACR_HOST"
```

在阿里云 RAM 控制台禁用或删除仅用于本次上传的临时 AccessKey。不要直接删除仍被其他任务使用的凭证。

FC 运行时不需要 Hugging Face Token、GitHub Token、OSS AccessKey 或 ACR 登录密码。弹性扩容出来的新实例自动继承镜像、OSS 挂载和环境变量。

## 最终检查清单

- 香港 ECS 是 x86_64 Ubuntu，Docker Engine 和 Buildx 均可用。
- `scripts/prepare_offline_assets.py --verify-only` 成功。
- 深圳 OSS 使用真实 Bucket 名，版本化前缀下的五个关键对象都存在。
- 镜像使用 `linux/amd64`、`--provenance=false` 和 `--sbom=false` 构建。
- ACR 远程 Manifest 包含 `linux/amd64`，不包含 `unknown/unknown`。
- FC 使用 `fc.gpu.ada.1`、CAPort 9000、单实例并发 1。
- OSS 前缀挂载到 `/mnt/nas/sam3d`，`/readyz` 显示 `config_present=true`。
- FC 环境变量启用了 Hugging Face 离线模式。
- FC OSS 挂载使用深圳内网 Endpoint 和只读执行角色权限。
- 两次 DINO 日志都显示 `source: local` 和 `pretrained: False`。
- `/initialize` 成功，`/readyz` 显示模型已加载。
- PLY 或 GLB 样例请求成功。
- 旧镜像标签和旧 OSS 前缀仍保留，可随时回滚。

## 官方资料

- [Docker Engine：Ubuntu 安装](https://docs.docker.com/engine/install/ubuntu/)
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
- [ossutil 安装](https://help.aliyun.com/zh/oss/install-ossutil2)
- [ossutil 配置与命令概览](https://help.aliyun.com/zh/oss/developer-reference/ossutil-overview/)
- [FC 自定义容器](https://help.aliyun.com/en/functioncompute/custom-container/)
- [FC 创建 GPU 函数](https://help.aliyun.com/zh/functioncompute/creating-a-gpu-function/)
- [FC 配置 OSS 挂载](https://help.aliyun.com/en/functioncompute/configure-an-oss-file-system-1)
- [FC GPU 实例规格](https://help.aliyun.com/en/functioncompute/fc/product-overview/instance-types-and-specifications)
- [FC HTTP 触发器](https://help.aliyun.com/en/functioncompute/fc/http-triggers-overview)
- [FC 使用限制](https://help.aliyun.com/en/functioncompute/limits-of-usage)
- [FC `unknown/unknown` 镜像排查](https://help.aliyun.com/zh/functioncompute/fc/custom-image-deployment-fails-with-platform-of-image-is-unknown-unknown)
- [Docker Build attestations](https://docs.docker.com/build/metadata/attestations/)
