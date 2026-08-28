# 香港 ECS 上传 OSS 并推送 ACR 手册

本手册使用一个前台脚本连续完成两项工作：优先复用 OSS 中已经完整发布的离线模型资源，必要时准备并校验完整离线模型资源（SAM 3 + SAM 3D Objects）并上传深圳 OSS；构建统一镜像并推送到阿里云容器镜像服务 ACR。

脚本不会创建或修改函数计算，也不会启动 GPU。OSS Bucket、ACR 仓库、模型授权和云端访问身份需要提前准备。

## 开始前准备

### 香港 ECS

构建机需要满足：

- Ubuntu 22.04 或 24.04。
- x86_64/amd64 架构。
- 建议至少 100 GB 可用磁盘和 32 GB 内存。
- 能访问 `v4.gh-proxy.org`、`hf-mirror.com`、ModelScope（魔塔社区）、阿里云 PyPI/PyTorch 镜像、`docker.1ms.run`、Docker CE 软件源、深圳 OSS 公网地址和目标 ACR 公网地址。
- 使用 `root` 运行脚本。

构建过程不需要 GPU。CUDA 扩展会在 CUDA devel 构建镜像内编译。

### 模型来源与深圳 OSS

1. SAM3D 主权重从魔搭社区（ModelScope）的公开模型 [facebook/sam-3d-objects](https://modelscope.cn/models/facebook/sam-3d-objects) 下载。
2. SAM3 从魔搭社区的公开模型 [facebook/sam3](https://modelscope.cn/models/facebook/sam3) 下载。两套主权重都不需要 Hugging Face Token。
3. MoGe 与 DINOv2 权重通过 HF-Mirror 的公开直链下载，不需要登录或 Token。
4. 提前创建深圳地域 OSS Bucket，并准备对目标 Bucket 有读取、列举和上传权限的 ECS RAM Role、ossutil 配置或临时 RAM/STS 凭证。
5. 香港 ECS 跨地域上传固定使用公网 Endpoint：

   ```text
   https://oss-cn-shenzhen.aliyuncs.com
   ```

脚本把完整资源包上传到新的内容寻址前缀 `sam3d/releases/bundle-<资源清单摘要>/`，不会覆盖旧版本资源包。全部对象通过 CRC64 核对后，脚本最后写入 `sam3d/recipes/<资源配方 ID>/complete.json` 完成凭据；下一台全新的构建机可据此确认同一资源配方已经完整发布。

### ACR

提前创建目标 ACR 命名空间和仓库，并准备：

1. 完整公网仓库地址，例如：

   ```text
   crpi-xxxx.cn-shenzhen.personal.cr.aliyuncs.com/namespace/sam3dobject
   ```

2. ACR 登录用户名。
3. ACR Registry 密码。

仓库地址不要包含 `https://`，不要附加 tag，不要使用 `-vpc` 或 `-internal` 地址。

## 获取代码

```bash
sudo -i
apt-get update
apt-get install -y ca-certificates git

cd /root
git clone https://v4.gh-proxy.org/https://github.com/leftumbrella/AliSam3DObjectDocker.git
cd AliSam3DObjectDocker
git status --short
```

`git status --short` 必须没有输出。脚本拒绝从有未提交或未跟踪文件的工作区构建镜像，保证镜像 tag 能对应唯一 Git commit。

如果仓库已经存在，只需更新：

```bash
cd /root/AliSam3DObjectDocker
git fetch origin
git switch main
git pull --ff-only origin main
git status --short
```

## 运行

脚本没有任何可选参数，也不读取部署配置参数。直接运行：

```bash
./scripts/deploy_from_hk.sh
```

脚本首先询问三个非敏感必要信息：

```text
深圳 OSS Bucket 名:
ACR 完整公网仓库地址（不含协议和 tag）:
ACR 登录用户名:
```

SAM3D 或 SAM3 主权重缺失时，脚本会直接从 ModelScope 下载，不会读取 Hugging Face Token。如果现有 ossutil 配置、环境凭证和 ECS RAM Role 都无法访问 Bucket，才会继续读取临时 OSS 凭证。镜像构建完成后再隐藏读取 ACR 密码：

```text
OSS AccessKey ID:
OSS AccessKey Secret:
OSS STS Token（普通 RAM AccessKey 直接回车）:
ACR Registry 密码:
```

AccessKey Secret、STS Token 和 Registry 密码不会出现在命令行参数、Git 或最终结果文件中；脚本退出时会清理进程环境和临时 Docker 凭证目录。

输入完成后，脚本自动连续执行：

1. 检查 Ubuntu、CPU 架构、磁盘、内存和干净的 Git checkout。
2. 安装或复用 Docker CE、Buildx、隔离的 Python 工具环境和校验过的 ossutil 2.3.0。
3. 根据固定 revision、预期文件路径、大小和 SHA-256 计算确定性的资源配方 ID。
4. 访问 `sam3d/recipes/<资源配方 ID>/complete.json`；若完成凭据有效且其中每个远端对象的 CRC64 都一致，直接复用对应 OSS 前缀，跳过全部模型下载和上传。
5. 完成凭据缺失、无效、对象缺失或 CRC64 不一致时，调用 `scripts/prepare_offline_assets.py`，从 ModelScope 下载固定版本的 SAM3 与 SAM3D 主权重，并下载或复用 MoGe 和 DINOv2 文件。
6. 对所有必要文件执行固定版本、大小和 SHA-256 校验；缺少 `sam3/sam3.pt` 等任一资源都会停止。
7. 根据 `offline-assets.sha256` 生成不可变 OSS 版本前缀和精确上传清单；DINOv2 `.git` 等非运行时元数据不会上传。
8. 使用断点目录上传深圳 OSS，错误报告固定写到 `/root/sam3d-transfer/ossutil-output`，不会污染 Git checkout。
9. 对清单中的每个远端对象执行精确 CRC64 核对；缺失或不一致的对象会强制重传并再次校验，全部成功后才写入完成凭据。
10. 构建一张 `linux/amd64` 统一镜像。
11. 构建完成后才登录 ACR，避免 Registry 凭证暴露给模型下载或第三方安装步骤。
12. 根据当前 Git commit 和构建后的镜像摘要自动生成不可变 tag，并用于 ACR 发布。
13. 推送镜像，失败时自动重试并回读远程摘要确认是否已经成功。
14. 验证远程 Manifest 只有 `linux/amd64`，且没有 `unknown/unknown` attestation。
15. 写入不含凭证的部署结果，退出 ACR 登录并清理临时凭证目录。

固定构建设置由脚本维护，不要求操作人员选择：

```text
Python 3.12.11
PyTorch 2.7.1 + cu126
CUDA 12.6
目标平台 linux/amd64
CUDA 架构 sm_89 + sm_90
MAX_JOBS=2
NVCC_THREADS=2
provenance=false
sbom=false
```

完成后终端会输出完整远程镜像，例如：

```text
ACR 镜像：crpi-xxxx.cn-shenzhen.personal.cr.aliyuncs.com/namespace/sam3dobject:sam3-sam3d-40c2973cc934-a1b2c3d4e5f6
OSS Bucket 子目录：/sam3d/releases/bundle-a1b2c3d4e5f6
FC 本地挂载目录：/mnt/nas/sam3d
结果文件：/root/sam3d-transfer/deployment-result.env
```

## 重跑与中断

脚本始终在当前终端前台运行。SSH 中断会停止当前进程；重新运行同一个脚本即可复用已完成的模型下载、OSS 断点、CRC64 已匹配的远端对象、Docker 构建缓存和镜像层。只要 OSS 完成凭据和全部对象仍匹配，即使换成没有 `/root/sam3d-transfer` 的全新 ECS，也会跳过模型下载和上传。OSS 批处理错误报告保存在 `/root/sam3d-transfer/ossutil-output`，不会在仓库内生成 `ossutil_output/`。

旧版本脚本已经上传的资源前缀没有完成凭据。第一次使用新版脚本时，如果当前 ECS 仍保留完整的 `/root/sam3d-transfer`，脚本会复用本地文件、核对远端对象并补写凭据，不会重新下载；如果本地文件也不存在，则需要准备一次资源以建立可信凭据。之后的新 ECS 才能走 OSS 快速复用路径。

镜像 tag 由 Git commit 和本地镜像配置摘要自动产生：

```text
sam3-sam3d-<12位Git提交>-<12位镜像摘要>
```

如果远程 tag 已经是同一镜像，脚本跳过重复推送；如果同一不可变 tag 指向其他镜像，脚本拒绝覆盖。推送后还会重复回读远程配置摘要，避免把网络回包丢失误判为推送失败。

## OSS 资源与镜像边界

镜像包含：

- SAM3 与 SAM3D 源码。
- 一套统一的 Python/PyTorch/CUDA 用户态环境。
- PyTorch3D、gsplat、spconv 等运行依赖。
- 组合 HTTP 服务及 GPU 推理互斥逻辑。

模型 checkpoint 不进入镜像。脚本会把通过完整性校验的资源上传到 OSS，资源包至少包含：

```text
offline-assets.sha256
sam3/sam3.pt
hf/pipeline.yaml
hf/moge/model.pt
cache/torch/hub/facebookresearch_dinov2_main/
cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
```

`sam3d/recipes/<资源配方 ID>/complete.json` 是发布控制信息，不属于模型挂载目录。它记录资源版本前缀、清单 SHA-256、对象数量及每个对象的 CRC64，并且只在所有资源对象上传和复核成功后写入。

把完成输出中的 OSS Bucket 子目录只读挂载到 `/mnt/nas/sam3d`。脚本只输出挂载参数，不会自动修改 FC。

## 常见错误

- “脚本不接受任何参数”：不要添加 `--help`、地址、用户名或其他参数，直接运行脚本并按提示输入。
- “Git checkout 不是干净状态”：检查 `git status --short`，不要在不清楚文件来源时强制删除。
- SAM3D checkpoint 缺失：确认香港 ECS 能访问 [facebook/sam-3d-objects](https://modelscope.cn/models/facebook/sam-3d-objects) 后重新运行；脚本会按固定 revision 断点下载并逐文件校验 SHA-256。
- SAM3 checkpoint 缺失：确认香港 ECS 能访问 `modelscope.cn` 后重新运行脚本；它会复用已有 `hf/` 和 `cache/`，从 ModelScope 只补下载并上传 `sam3/sam3.pt`，然后重建资源清单。
- MoGe 或 DINOv2 权重下载失败：确认香港 ECS 能访问 `hf-mirror.com`；这些公开文件不需要 Token。
- OSS 访问失败：Bucket 必须在深圳；香港 ECS 上传使用公网 Endpoint，并确保 RAM 身份有目标 Bucket 的列举和写入权限。
- OSS 完成凭据缺失或 CRC64 不匹配：这是安全回退，不会直接使用不完整资源；脚本会检查或准备本地资源、修复远端对象，并在最后重写完成凭据。
- OSS 上传中断：直接重跑，`/root/sam3d-transfer/oss-upload-checkpoints` 会继续断点上传；错误明细在 `/root/sam3d-transfer/ossutil-output`。
- ACR 地址格式错误：使用完整公网 `域名/namespace/repository`，不要添加协议或 tag。
- Docker 安装失败：检查系统是否混装 Ubuntu `docker.io` 与 Docker CE。
- Docker 基础镜像元数据超时：基础镜像已固定通过 `docker.1ms.run` 拉取，不再直连 `registry-1.docker.io`；确认 ECS 能访问该域名后直接重跑。
- CUDA 扩展编译失败：保留首次失败日志；修复网络或资源问题后直接重跑，Buildx 会复用已完成层。
- ACR 登录失败：使用容器镜像服务控制台提供的固定密码，不是阿里云控制台登录密码。
- Manifest 校验失败：远程镜像必须且只能包含 `linux/amd64`。
