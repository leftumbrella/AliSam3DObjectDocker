# 香港 ECS 构建并推送 ACR 手册

本手册只完成一件事：在香港 Ubuntu ECS 上构建 SAM 3 + SAM 3D Objects 统一镜像，并推送到阿里云容器镜像服务 ACR。

脚本不会下载模型权重，不会访问 OSS，不会创建或修改函数计算，也不会启动 GPU。

## 开始前准备

### 香港 ECS

构建机需要满足：

- Ubuntu 22.04 或 24.04。
- x86_64/amd64 架构。
- 建议至少 100 GB 可用磁盘和 32 GB 内存。
- 能访问 GitHub、PyPI、PyTorch、Docker Hub 和目标 ACR 公网地址。
- 使用 `root` 运行脚本。

构建过程不需要 GPU。CUDA 扩展会在 CUDA devel 构建镜像内编译。

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
git clone https://github.com/leftumbrella/AliSam3DObjectDocker.git
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

脚本只会询问三个必要信息：

```text
ACR 完整公网仓库地址（不含协议和 tag）:
ACR 登录用户名:
ACR Registry 密码:
```

密码隐藏输入，不会出现在命令行历史、Git 或 Docker 配置残留中。

输入完成后，脚本自动连续执行：

1. 检查 Ubuntu、CPU 架构、磁盘、内存和干净的 Git checkout。
2. 安装或复用 Docker CE、Buildx、Git、curl 和 jq。
3. 登录 ACR，在耗时构建开始前验证 Registry 凭证。
4. 构建一张 `linux/amd64` 统一镜像。
5. 根据当前 Git commit 和构建后的镜像摘要自动生成不可变 tag。
6. 推送镜像，失败时自动重试并回读远程摘要确认是否已经成功。
7. 验证远程 Manifest 只有 `linux/amd64`，且没有 `unknown/unknown` attestation。
8. 退出 ACR 登录并删除临时 Docker 凭证目录。

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
```

## 重跑与中断

脚本始终在当前终端前台运行。SSH 中断会停止当前进程；重新运行同一个脚本即可复用 Docker 构建缓存和已经上传的镜像层。

镜像 tag 由 Git commit 和本地镜像配置摘要自动产生：

```text
sam3-sam3d-<12位Git提交>-<12位镜像摘要>
```

如果远程 tag 已经是同一镜像，脚本跳过重复推送；如果同一不可变 tag 指向其他镜像，脚本拒绝覆盖。推送后还会重复回读远程配置摘要，避免把网络回包丢失误判为推送失败。

## 镜像内容边界

镜像包含：

- SAM3 与 SAM3D 源码。
- 一套统一的 Python/PyTorch/CUDA 用户态环境。
- PyTorch3D、gsplat、spconv 等运行依赖。
- 组合 HTTP 服务及 GPU 推理互斥逻辑。

模型 checkpoint 不进入镜像。未来运行容器时，需要由运行平台另外挂载 SAM3、SAM3D、MoGe 和 DINOv2 模型资源；这不属于本构建推送脚本的职责。

## 常见错误

- “脚本不接受任何参数”：不要添加 `--help`、地址、用户名或其他参数，直接运行脚本并按提示输入。
- “Git checkout 不是干净状态”：检查 `git status --short`，不要在不清楚文件来源时强制删除。
- ACR 地址格式错误：使用完整公网 `域名/namespace/repository`，不要添加协议或 tag。
- Docker 安装失败：检查系统是否混装 Ubuntu `docker.io` 与 Docker CE。
- CUDA 扩展编译失败：保留首次失败日志；修复网络或资源问题后直接重跑，Buildx 会复用已完成层。
- ACR 登录失败：使用容器镜像服务控制台提供的固定密码，不是阿里云控制台登录密码。
- Manifest 校验失败：远程镜像必须且只能包含 `linux/amd64`。
