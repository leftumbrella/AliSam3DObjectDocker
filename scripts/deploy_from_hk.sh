#!/usr/bin/env bash

set -Eeuo pipefail
set +x
IFS=$'\n\t'
umask 077

SCRIPT_NAME="$(basename -- "$0")"
readonly SCRIPT_NAME
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly PROJECT_ROOT
readonly OSS_REGION_ID='cn-shenzhen'
readonly OSS_PUBLIC_ENDPOINT='https://oss-cn-shenzhen.aliyuncs.com'
readonly OSSUTIL_VERSION='2.3.0'
readonly OSSUTIL_SHA256='3ae4d9fc85a7a6e9f5654d1599766f1a3a42a3692870887b5ae9338d582ef65a'
readonly HUGGINGFACE_HUB_VERSION='1.27.0'
readonly SAM3D_REF='f91db411c50efee93d8db7aeb323885650f6f722'
readonly TORCH_CUDA_ARCH_LIST_VALUE='8.9'
readonly FC_IMAGE_LIMIT_BYTES=$((15 * 1024 * 1024 * 1024))

TRANSFER_ROOT="${TRANSFER_ROOT:-/root/sam3d-transfer}"
SAM3D_SOURCE="${SAM3D_SOURCE:-}"
OSS_BUCKET="${OSS_BUCKET:-}"
ACR_IMAGE="${ACR_IMAGE:-}"
ACR_USERNAME="${ACR_USERNAME:-}"
ACR_HOST=''
MAX_JOBS="${MAX_JOBS:-2}"
NVCC_THREADS="${NVCC_THREADS:-2}"

ASSUME_YES=0
NON_INTERACTIVE=0
DRY_RUN=0
CURRENT_STEP='启动'
ACR_LOGIN_ACTIVE=0
OSSUTIL_BIN=''
HF_BIN=''
TOOLS_PYTHON=''
GIT_COMMIT=''
GIT_COMMIT_SHORT=''
IMAGE_TAG=''
LOCAL_IMAGE=''
REMOTE_IMAGE=''
ASSET_RELEASE=''
OSS_PREFIX=''
DEPLOYMENT_RESULT_FILE=''
TEMP_DIR=''

usage() {
  cat <<'EOF'
从香港 Ubuntu ECS 准备 SAM3D 离线资源、上传深圳 OSS、构建镜像并推送深圳 ACR。

用法：
  ./scripts/deploy_from_hk.sh [选项]

必填信息可以通过选项、同名环境变量或交互提示提供：
  --oss-bucket NAME          深圳 OSS Bucket 名
  --acr-image IMAGE          ACR 完整公网仓库地址，不含协议和 tag
  --acr-username USER        ACR 登录用户名

可选：
  --transfer-root PATH       离线资源和断点目录，默认 /root/sam3d-transfer
  --sam3d-source PATH        已有 SAM3D checkpoints 目录，不再下载主 checkpoint
  --max-jobs N               CUDA 扩展编译并发，默认 2
  --nvcc-threads N           NVCC 线程数，默认 2
  --non-interactive          禁止所有提示，缺少信息或凭证时直接失败
  --yes                      跳过执行前确认；--non-interactive 必须同时使用
  --dry-run                  只显示计划，不安装、下载、上传、构建或登录
  -h, --help                 显示帮助

敏感信息不接受命令行参数：
  HF_TOKEN                   Hugging Face Access Token，可不设置并隐藏输入
  OSS_ACCESS_KEY_ID          OSS RAM 用户或 STS AccessKey ID
  OSS_ACCESS_KEY_SECRET      OSS RAM 用户或 STS AccessKey Secret
  OSS_SESSION_TOKEN          使用 STS 时设置，普通 RAM AccessKey 留空
  ACR_PASSWORD              ACR Registry 密码

推荐直接运行脚本并按提示输入。重复运行会复用下载文件、OSS 断点、
Buildx 缓存和已推送镜像层，不会把任何密码写入项目或结果文件。
脚本始终在当前终端前台连续执行；SSH 断开会终止进程，重新运行可续传。
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf '[%s] 警告：%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
  printf '[%s] 错误：%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
  exit 1
}

on_error() {
  local status=$?
  printf '[%s] 失败：步骤“%s”，行号 %s，退出码 %s。修正后可重跑同一命令。\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "$CURRENT_STEP" \
    "${BASH_LINENO[0]:-unknown}" \
    "$status" >&2
  exit "$status"
}

safe_remove_temp_dirs() {
  [[ -n "$TEMP_DIR" ]] || return
  case "$TEMP_DIR" in
    /tmp/sam3d-hk-deploy.*)
      if [[ -d "$TEMP_DIR" && ! -L "$TEMP_DIR" ]]; then
        rm -rf -- "$TEMP_DIR"
      fi
      ;;
    *)
      warn "拒绝清理不符合临时目录规则的路径：$TEMP_DIR"
      ;;
  esac
}

cleanup() {
  local status=$?
  trap - ERR
  set +e

  if [[ "$ACR_LOGIN_ACTIVE" -eq 1 && -n "$ACR_HOST" ]] && command -v docker >/dev/null 2>&1; then
    docker logout "$ACR_HOST" >/dev/null 2>&1
  fi
  unset ACR_PASSWORD HF_TOKEN OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET OSS_SESSION_TOKEN
  unset DOCKER_CONFIG
  safe_remove_temp_dirs
  exit "$status"
}

trap on_error ERR
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_option_value() {
  local option=$1
  local value=${2:-}
  [[ -n "$value" && "$value" != --* ]] || die "$option 缺少参数"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --oss-bucket)
        require_option_value "$1" "${2:-}"
        OSS_BUCKET=$2
        shift 2
        ;;
      --acr-image)
        require_option_value "$1" "${2:-}"
        ACR_IMAGE=$2
        shift 2
        ;;
      --acr-username)
        require_option_value "$1" "${2:-}"
        ACR_USERNAME=$2
        shift 2
        ;;
      --transfer-root)
        require_option_value "$1" "${2:-}"
        TRANSFER_ROOT=$2
        shift 2
        ;;
      --sam3d-source)
        require_option_value "$1" "${2:-}"
        SAM3D_SOURCE=$2
        shift 2
        ;;
      --max-jobs)
        require_option_value "$1" "${2:-}"
        MAX_JOBS=$2
        shift 2
        ;;
      --nvcc-threads)
        require_option_value "$1" "${2:-}"
        NVCC_THREADS=$2
        shift 2
        ;;
      --non-interactive)
        NON_INTERACTIVE=1
        shift
        ;;
      --yes)
        ASSUME_YES=1
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        printf '未知选项：%s\n\n' "$1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
}

prompt_value() {
  local variable_name=$1
  local label=$2
  local value=''
  [[ -t 0 ]] || die "$label 需要交互终端；自动化运行请使用对应选项或环境变量"
  read -r -p "$label: " value
  [[ -n "$value" ]] || die "$label 不能为空"
  printf -v "$variable_name" '%s' "$value"
}

collect_required_inputs() {
  local missing=''
  [[ -n "$OSS_BUCKET" ]] || missing="$missing OSS_BUCKET"
  [[ -n "$ACR_IMAGE" ]] || missing="$missing ACR_IMAGE"
  [[ -n "$ACR_USERNAME" ]] || missing="$missing ACR_USERNAME"

  if [[ -n "$missing" && "$NON_INTERACTIVE" -eq 1 ]]; then
    die "无交互模式缺少：$missing"
  fi

  [[ -n "$OSS_BUCKET" ]] || prompt_value OSS_BUCKET '深圳 OSS Bucket 名'
  [[ -n "$ACR_IMAGE" ]] \
    || prompt_value ACR_IMAGE '深圳 ACR 完整公网仓库地址（域名/namespace/repository，不含 tag）'
  [[ -n "$ACR_USERNAME" ]] || prompt_value ACR_USERNAME 'ACR 登录用户名'
}

validate_optional_number() {
  local name=$1
  local value=$2
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name 必须是正整数"
  (( value <= 32 )) || die "$name 不能大于 32"
}

validate_inputs() {
  local resolved_transfer_root
  local acr_image_pattern='^([A-Za-z0-9.-]+)/([a-z0-9._-]+)/([a-z0-9._-]+)$'
  [[ "$OSS_BUCKET" =~ ^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$ ]] \
    || die 'OSS Bucket 名必须为 3 到 63 位小写字母、数字或连字符'

  if [[ "$ACR_IMAGE" =~ $acr_image_pattern ]]; then
    ACR_HOST=${BASH_REMATCH[1]}
  else
    die 'ACR_IMAGE 必须是域名/namespace/repository，不能包含协议、tag、命令或多余路径'
  fi
  [[ "$ACR_HOST" =~ ^[A-Za-z0-9.-]+$ ]] \
    || die 'ACR_HOST 只能是域名，不能包含协议、端口或路径'
  [[ "$ACR_HOST" == *'.aliyuncs.com' ]] || die 'ACR_HOST 必须使用阿里云 Registry 域名'
  [[ "$ACR_HOST" == *'cn-shenzhen'* ]] || die 'ACR_HOST 必须是深圳地域地址'
  [[ "$ACR_HOST" != *'-vpc'* && "$ACR_HOST" != *'-internal'* ]] \
    || die '香港 ECS 必须使用 ACR 公网地址，不能使用 -vpc 或 -internal'

  [[ ! "$ACR_USERNAME" =~ [[:cntrl:]] ]] || die 'ACR_USERNAME 包含非法控制字符'

  [[ "$TRANSFER_ROOT" == /* && "$TRANSFER_ROOT" != '/' ]] \
    || die 'TRANSFER_ROOT 必须是非根目录的绝对路径'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    resolved_transfer_root="$(realpath -m -- "$TRANSFER_ROOT")"
    case "$resolved_transfer_root" in
      "$PROJECT_ROOT"|"$PROJECT_ROOT"/*)
        die 'TRANSFER_ROOT 不能位于 Git 项目中，否则模型文件会进入 Docker 构建上下文'
        ;;
    esac
  fi
  if [[ -n "$SAM3D_SOURCE" ]]; then
    [[ "$SAM3D_SOURCE" == /* ]] || die 'SAM3D_SOURCE 必须是绝对路径'
  fi

  validate_optional_number MAX_JOBS "$MAX_JOBS"
  validate_optional_number NVCC_THREADS "$NVCC_THREADS"

  if [[ "$NON_INTERACTIVE" -eq 1 && "$ASSUME_YES" -ne 1 ]]; then
    die '--non-interactive 必须同时指定 --yes'
  fi
}

resolve_release_values() {
  GIT_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown')"
  GIT_COMMIT_SHORT="$(git -C "$PROJECT_ROOT" rev-parse --short=12 HEAD 2>/dev/null || printf 'unknown')"
  IMAGE_TAG="cu121-${GIT_COMMIT_SHORT}"
  LOCAL_IMAGE="sam3d-fc:${IMAGE_TAG}"
  if [[ -n "$ACR_IMAGE" ]]; then
    REMOTE_IMAGE="${ACR_IMAGE}:${IMAGE_TAG}"
  else
    REMOTE_IMAGE='<等待填写 ACR 信息>'
  fi
}

print_plan() {
  local bucket=${OSS_BUCKET:-'<交互填写>'}
  local acr_image=${ACR_IMAGE:-'<交互填写>'}
  local username=${ACR_USERNAME:-'<交互填写>'}
  local source=${SAM3D_SOURCE:-'<由 Hugging Face 下载>'}

  cat <<EOF

执行计划
  项目目录：      $PROJECT_ROOT
  Git commit：    $GIT_COMMIT
  离线工作目录：  $TRANSFER_ROOT
  主模型来源：    $source
  OSS Bucket：    $bucket
  OSS Endpoint：  $OSS_PUBLIC_ENDPOINT
  OSS 版本前缀：  sam3d/releases/bundle-<离线清单哈希>
  ACR 完整仓库：  $acr_image
  ACR 用户：      $username
  本地镜像：      $LOCAL_IMAGE
  远程镜像：      $REMOTE_IMAGE
  CUDA 编译并发： MAX_JOBS=${MAX_JOBS}，NVCC_THREADS=${NVCC_THREADS}

脚本将安装基础工具和 Docker/Buildx，准备并校验约 14 GB 离线资源，
上传深圳 OSS，构建 linux/amd64 镜像，登录 ACR 后推送并检查 Manifest。
不会创建或修改 FC 函数，也不会把 Token、AccessKey 或密码写入结果文件。
EOF
}

confirm_execution() {
  local answer=''
  [[ "$ASSUME_YES" -eq 1 ]] && return
  [[ -t 0 ]] || die '当前没有交互终端，请使用 --non-interactive --yes 并通过环境变量提供凭证'
  read -r -p '确认执行以上动作？输入 yes 继续: ' answer
  [[ "$answer" == 'yes' ]] || die '用户取消执行'
}

require_target_host() {
  [[ "$EUID" -eq 0 ]] || die "请先执行 sudo -i 切换到 root，再重新运行 $SCRIPT_NAME"
  [[ "$(uname -m)" == 'x86_64' ]] || die '香港构建机必须是 x86_64/amd64'
  [[ -r /etc/os-release ]] || die '无法读取 /etc/os-release'

  # shellcheck disable=SC1091
  . /etc/os-release
  [[ "${ID:-}" == 'ubuntu' ]] || die '脚本只支持 Ubuntu'
  case "${VERSION_ID:-}" in
    22.04|24.04) ;;
    *) warn "当前 Ubuntu ${VERSION_ID:-unknown} 不是已验证的 22.04/24.04" ;;
  esac
}

require_clean_checkout() {
  [[ -d "$PROJECT_ROOT/.git" ]] || die '项目必须通过 git clone 获取'
  local status
  status="$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=normal)"
  [[ -z "$status" ]] || die "Git checkout 不是干净状态，拒绝构建不可追溯镜像：\n$status"
  [[ "$GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die '无法解析当前 Git commit'
}

check_resources() {
  local free_kb free_gb memory_kb memory_gb storage_probe
  storage_probe=$TRANSFER_ROOT
  while [[ ! -e "$storage_probe" ]]; do
    storage_probe="$(dirname -- "$storage_probe")"
  done
  free_kb="$(df -Pk "$storage_probe" | awk 'NR == 2 {print $4}')"
  free_gb=$((free_kb / 1024 / 1024))
  (( free_gb >= 100 )) || warn "当前文件系统仅剩约 ${free_gb} GB，建议至少准备 100 GB 可用空间"

  memory_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
  memory_gb=$((memory_kb / 1024 / 1024))
  (( memory_gb >= 24 )) || warn "当前内存约 ${memory_gb} GB，建议至少 32 GB"
}

install_base_tools() {
  CURRENT_STEP='安装 Ubuntu 基础工具'
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y \
    ca-certificates \
    curl \
    git \
    jq \
    python3 \
    python3-pip \
    python3-venv \
    unzip
  unset DEBIAN_FRONTEND
}

configure_docker_repository() {
  local codename architecture
  # shellcheck disable=SC1091
  . /etc/os-release
  codename="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
  architecture="$(dpkg --print-architecture)"
  [[ -n "$codename" ]] || die '无法确定 Ubuntu codename'

  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc

  {
    printf 'Types: deb\n'
    printf 'URIs: https://download.docker.com/linux/ubuntu\n'
    printf 'Suites: %s\n' "$codename"
    printf 'Components: stable\n'
    printf 'Architectures: %s\n' "$architecture"
    printf 'Signed-By: /etc/apt/keyrings/docker.asc\n'
  } >/etc/apt/sources.list.d/docker.sources
  apt-get update
}

ensure_docker() {
  CURRENT_STEP='安装并验证 Docker Buildx'
  if command -v docker >/dev/null 2>&1 && docker buildx version >/dev/null 2>&1; then
    log '复用现有 Docker 和 Buildx'
  else
    configure_docker_repository
    export DEBIAN_FRONTEND=noninteractive
    if command -v docker >/dev/null 2>&1; then
      apt-get install -y docker-buildx-plugin \
        || die 'Buildx 安装失败，请检查是否混装了 Ubuntu docker.io 与 Docker CE'
    else
      apt-get install -y \
        containerd.io \
        docker-buildx-plugin \
        docker-ce \
        docker-ce-cli \
        docker-compose-plugin \
        || die 'Docker CE 安装失败，请检查 APT 软件包冲突'
    fi
    unset DEBIAN_FRONTEND
  fi

  systemctl enable --now docker
  docker version >/dev/null
  docker buildx version
  docker buildx inspect --bootstrap >/dev/null
}

ensure_tool_venv() {
  CURRENT_STEP='安装 Hugging Face CLI'
  local tools_root='/opt/sam3d-tools'
  [[ ! -L "$tools_root" ]] || die "$tools_root 不得是符号链接"
  if [[ ! -x "$tools_root/bin/python" ]]; then
    python3 -m venv "$tools_root"
  fi

  TOOLS_PYTHON="$tools_root/bin/python"
  "$TOOLS_PYTHON" -m pip install \
    --disable-pip-version-check \
    "huggingface_hub==${HUGGINGFACE_HUB_VERSION}"
  HF_BIN="$tools_root/bin/hf"
  [[ -x "$HF_BIN" ]] || die '安装 huggingface_hub 后仍未找到 hf 命令'
  "$HF_BIN" version
  export PATH="$tools_root/bin:$PATH"
}

ossutil_is_compatible() {
  local candidate=$1
  local version_output help_output
  [[ -x "$candidate" ]] || return 1
  version_output="$("$candidate" version 2>/dev/null)" || return 1
  [[ "$version_output" == *"$OSSUTIL_VERSION"* ]] || return 1
  help_output="$("$candidate" help cp 2>/dev/null)" || return 1
  grep -q -- '--checkpoint-dir' <<<"$help_output"
}

ensure_temp_dir() {
  if [[ -z "$TEMP_DIR" ]]; then
    TEMP_DIR="$(mktemp -d /tmp/sam3d-hk-deploy.XXXXXX)"
  fi
}

ensure_ossutil() {
  CURRENT_STEP='安装并校验 ossutil'
  local candidate archive temporary

  candidate="$(command -v ossutil 2>/dev/null || true)"
  if [[ -n "$candidate" ]] && ossutil_is_compatible "$candidate"; then
    OSSUTIL_BIN=$candidate
    log "复用现有 ossutil：$OSSUTIL_BIN"
    return
  fi

  candidate='/opt/sam3d-tools/bin/ossutil'
  if ossutil_is_compatible "$candidate"; then
    OSSUTIL_BIN=$candidate
    log "复用已校验的 ossutil：$OSSUTIL_BIN"
    return
  fi

  ensure_temp_dir
  temporary=$TEMP_DIR
  archive="ossutil-${OSSUTIL_VERSION}-linux-amd64.zip"
  curl --fail --location --retry 5 --retry-all-errors \
    --output "$temporary/$archive" \
    "https://gosspublic.alicdn.com/ossutil/v2/${OSSUTIL_VERSION}/${archive}"
  printf '%s  %s\n' "$OSSUTIL_SHA256" "$temporary/$archive" | sha256sum -c -
  unzip -q "$temporary/$archive" -d "$temporary"
  install -m 0755 \
    "$temporary/ossutil-${OSSUTIL_VERSION}-linux-amd64/ossutil" \
    "$candidate"
  ossutil_is_compatible "$candidate" || die 'ossutil 安装后命令契约检查失败'
  OSSUTIL_BIN=$candidate
  "$OSSUTIL_BIN" version
}

ensure_huggingface_access() {
  CURRENT_STEP='验证 Hugging Face 模型权限'
  local token=''
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
  if [[ -z "${HF_TOKEN:-}" ]]; then
    [[ "$NON_INTERACTIVE" -eq 0 ]] \
      || die '无交互模式需要设置有效的 HF_TOKEN，或预先准备完整离线资源'
    [[ -t 0 ]] || die 'Hugging Face Access Token 输入需要交互终端'
    read -r -s -p 'Hugging Face Access Token（输入时不会显示）: ' token
    printf '\n'
    [[ -n "$token" ]] || die 'Hugging Face Access Token 不能为空'
    export HF_TOKEN="$token"
    token=''
  fi
  if [[ "$HF_TOKEN" =~ [[:space:]] ]]; then
    unset HF_TOKEN
    die 'Hugging Face Access Token 不能包含空白字符'
  fi

  if ! "$HF_BIN" auth whoami >/dev/null 2>&1; then
    unset HF_TOKEN
    die 'Hugging Face Access Token 无效，或香港 ECS 无法连接 Hugging Face'
  fi
  if ! "$TOOLS_PYTHON" - <<'PY' >/dev/null 2>&1
import os

from huggingface_hub import HfApi

HfApi().model_info(
    "facebook/sam-3d-objects",
    token=os.environ["HF_TOKEN"],
)
PY
  then
    unset HF_TOKEN
    die '无法访问 facebook/sam-3d-objects；请确认已获批模型权限且香港 ECS 网络正常'
  fi
  log 'Hugging Face Access Token 验证成功；Token 仅保存在当前脚本进程环境中'
}

prepare_offline_assets() {
  CURRENT_STEP='准备并校验完整离线模型资源'
  local prepare_script="$PROJECT_ROOT/scripts/prepare_offline_assets.py"
  local -a args

  mkdir -p -- "$TRANSFER_ROOT"
  [[ ! -L "$TRANSFER_ROOT" ]] || die 'TRANSFER_ROOT 不得是符号链接'
  chmod 700 "$TRANSFER_ROOT"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

  if "$TOOLS_PYTHON" "$prepare_script" \
    --verify-only \
    --transfer-root "$TRANSFER_ROOT" >/dev/null 2>&1; then
    log '现有离线资源完整，跳过下载'
  else
    args=(--transfer-root "$TRANSFER_ROOT")
    if [[ -n "$SAM3D_SOURCE" ]]; then
      args+=(--sam3d-source "$SAM3D_SOURCE")
    elif "$TOOLS_PYTHON" "$prepare_script" \
      --verify-main-only \
      --transfer-root "$TRANSFER_ROOT" >/dev/null 2>&1; then
      log '现有主 checkpoint 完整，补齐其他离线资源'
    else
      ensure_huggingface_access
      args+=(--download-sam3d)
    fi
    "$TOOLS_PYTHON" "$prepare_script" "${args[@]}"
  fi

  "$TOOLS_PYTHON" "$prepare_script" \
    --verify-only \
    --transfer-root "$TRANSFER_ROOT"
  unset HF_TOKEN

  local manifest="$TRANSFER_ROOT/storage/offline-assets.sha256"
  local digest
  digest="$(sha256sum "$manifest" | awk '{print substr($1, 1, 12)}')"
  [[ "$digest" =~ ^[0-9a-f]{12}$ ]] || die '无法计算离线资源清单哈希'
  ASSET_RELEASE="bundle-${digest}"
  OSS_PREFIX="sam3d/releases/${ASSET_RELEASE}"
  DEPLOYMENT_RESULT_FILE="$TRANSFER_ROOT/deployment-result.env"
}

prompt_oss_credentials() {
  local access_key_id='' access_key_secret='' session_token=''
  [[ -t 0 ]] || die 'OSS 凭证输入需要交互终端'
  read -r -p 'OSS AccessKey ID: ' access_key_id
  read -r -s -p 'OSS AccessKey Secret: ' access_key_secret
  printf '\n'
  read -r -s -p 'OSS STS Token（普通 RAM AccessKey 直接回车）: ' session_token
  printf '\n'
  [[ -n "$access_key_id" && -n "$access_key_secret" ]] \
    || die 'OSS AccessKey ID 和 Secret 不能为空'
  export OSS_ACCESS_KEY_ID=$access_key_id
  export OSS_ACCESS_KEY_SECRET=$access_key_secret
  if [[ -n "$session_token" ]]; then
    export OSS_SESSION_TOKEN=$session_token
  else
    unset OSS_SESSION_TOKEN
  fi
  access_key_id=''
  access_key_secret=''
  session_token=''
}

ensure_oss_access() {
  CURRENT_STEP='验证深圳 OSS 凭证和 Bucket'
  export OSS_REGION="$OSS_REGION_ID"
  export OSS_ENDPOINT="$OSS_PUBLIC_ENDPOINT"

  if [[ -n "${OSS_ACCESS_KEY_ID:-}" && -z "${OSS_ACCESS_KEY_SECRET:-}" ]] \
    || [[ -z "${OSS_ACCESS_KEY_ID:-}" && -n "${OSS_ACCESS_KEY_SECRET:-}" ]]; then
    die 'OSS_ACCESS_KEY_ID 和 OSS_ACCESS_KEY_SECRET 必须同时设置'
  fi

  if "$OSSUTIL_BIN" ls "oss://${OSS_BUCKET}/" >/dev/null 2>&1; then
    log '深圳 OSS Bucket 和现有凭证验证成功'
    return
  fi

  [[ "$NON_INTERACTIVE" -eq 0 ]] \
    || die '无法访问 OSS Bucket，请检查环境变量、ECS RAM 角色、Bucket 名和权限'
  warn '现有 ossutil 配置或环境凭证无法访问目标 Bucket，将读取临时 RAM/STS 凭证'
  unset OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET OSS_SESSION_TOKEN
  prompt_oss_credentials
  "$OSSUTIL_BIN" ls "oss://${OSS_BUCKET}/" >/dev/null \
    || die 'OSS 访问失败，请核对 Bucket 名、深圳地域和 RAM 权限'
}

verify_remote_oss_object() {
  local key=$1
  local url="oss://${OSS_BUCKET}/${OSS_PREFIX}/${key}"
  local output
  output="$("$OSSUTIL_BIN" ls "$url")"
  grep -Fq -- "$url" <<<"$output" || die "OSS 缺少关键对象：$url"
}

upload_offline_assets() {
  CURRENT_STEP='断点上传离线资源到深圳 OSS'
  local checkpoint_dir="$TRANSFER_ROOT/oss-upload-checkpoints"
  mkdir -p -- "$checkpoint_dir"
  chmod 700 "$checkpoint_dir"

  "$OSSUTIL_BIN" cp -u -r \
    "$TRANSFER_ROOT/storage/" \
    "oss://${OSS_BUCKET}/${OSS_PREFIX}/" \
    --checkpoint-dir "$checkpoint_dir"

  verify_remote_oss_object 'offline-assets.sha256'
  verify_remote_oss_object 'hf/pipeline.yaml'
  verify_remote_oss_object 'hf/moge/model.pt'
  verify_remote_oss_object 'cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth'
  verify_remote_oss_object 'cache/torch/hub/facebookresearch_dinov2_main/hubconf.py'
  log "OSS 关键对象检查完成：oss://${OSS_BUCKET}/${OSS_PREFIX}/"

  unset OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET OSS_SESSION_TOKEN OSS_REGION OSS_ENDPOINT
}

build_image() {
  CURRENT_STEP='构建 FC linux/amd64 镜像'
  cd "$PROJECT_ROOT"
  docker buildx build \
    --load \
    --progress=plain \
    --platform linux/amd64 \
    --provenance=false \
    --sbom=false \
    --build-arg "SAM3D_REF=${SAM3D_REF}" \
    --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VALUE}" \
    --build-arg "MAX_JOBS=${MAX_JOBS}" \
    --build-arg "NVCC_THREADS=${NVCC_THREADS}" \
    -t "$LOCAL_IMAGE" \
    .

  local platform image_size
  platform="$(docker image inspect "$LOCAL_IMAGE" --format '{{.Os}}/{{.Architecture}}')"
  [[ "$platform" == 'linux/amd64' ]] || die "本地镜像平台错误：$platform"
  image_size="$(docker image inspect "$LOCAL_IMAGE" --format '{{.Size}}')"
  [[ "$image_size" =~ ^[0-9]+$ ]] || die '无法读取镜像大小'
  (( image_size <= FC_IMAGE_LIMIT_BYTES )) \
    || die "镜像未压缩大小 ${image_size} 字节，超过 FC 15 GB 上限"
  log "镜像构建完成：$LOCAL_IMAGE，未压缩大小 ${image_size} 字节"
}

login_acr() {
  CURRENT_STEP='登录深圳 ACR'
  local password="${ACR_PASSWORD:-}"
  ensure_temp_dir
  install -m 0700 -d "$TEMP_DIR/docker-config"
  export DOCKER_CONFIG="$TEMP_DIR/docker-config"
  if [[ -z "$password" ]]; then
    [[ "$NON_INTERACTIVE" -eq 0 ]] || die '无交互模式需要设置 ACR_PASSWORD'
    [[ -t 0 ]] || die 'ACR 密码输入需要交互终端'
    read -r -s -p 'ACR Registry password: ' password
    printf '\n'
  fi
  [[ -n "$password" ]] || die 'ACR Registry password 不能为空'
  printf '%s' "$password" | docker login \
    --username "$ACR_USERNAME" \
    --password-stdin \
    "$ACR_HOST"
  password=''
  unset ACR_PASSWORD
  ACR_LOGIN_ACTIVE=1
}

get_remote_config_digest() {
  local raw lookup_error media_type child_digest config_digest

  if ! raw="$(docker buildx imagetools inspect --raw "$REMOTE_IMAGE" 2>&1)"; then
    lookup_error=${raw,,}
    case "$lookup_error" in
      *'not found'*|*'manifest unknown'*|*'name unknown'*) return 1 ;;
      *)
        warn "无法确定远程标签状态：$raw"
        return 2
        ;;
    esac
  fi

  media_type="$(jq -r '.mediaType // empty' <<<"$raw")"
  case "$media_type" in
    application/vnd.oci.image.index.v1+json|application/vnd.docker.distribution.manifest.list.v2+json)
      child_digest="$(
        jq -er \
          '[.manifests[]? | select(.platform.os == "linux" and .platform.architecture == "amd64") | .digest]
           | if length == 1 then .[0] else empty end' <<<"$raw"
      )" || return 2
      raw="$(docker buildx imagetools inspect --raw "$REMOTE_IMAGE@$child_digest")" \
        || return 2
      ;;
    application/vnd.oci.image.manifest.v1+json|application/vnd.docker.distribution.manifest.v2+json)
      ;;
    *) return 2 ;;
  esac

  config_digest="$(jq -r '.config.digest // empty' <<<"$raw")"
  [[ "$config_digest" =~ ^sha256:[0-9a-f]{64}$ ]] || return 2
  printf '%s' "$config_digest"
}

remote_tag_matches_local_image() {
  local local_digest remote_digest status
  local_digest="$(docker image inspect "$LOCAL_IMAGE" --format '{{.Id}}')"
  [[ "$local_digest" =~ ^sha256:[0-9a-f]{64}$ ]] || die '无法读取本地镜像配置摘要'

  set +e
  remote_digest="$(get_remote_config_digest)"
  status=$?
  set -e

  case "$status" in
    0)
      if [[ "$remote_digest" == "$local_digest" ]]; then
        return 0
      fi
      die "远程不可变标签已指向另一镜像，拒绝覆盖：$REMOTE_IMAGE"
      ;;
    1) return 1 ;;
    *) die "无法安全确认远程标签是否存在，拒绝执行可能覆盖镜像的 push：$REMOTE_IMAGE" ;;
  esac
}

push_image_with_retry() {
  CURRENT_STEP='推送镜像到深圳 ACR'
  local attempt delay
  docker tag "$LOCAL_IMAGE" "$REMOTE_IMAGE"

  if remote_tag_matches_local_image; then
    log '远程不可变标签已经是同一镜像，跳过重复 push'
    return
  fi

  for attempt in 1 2 3; do
    if docker push "$REMOTE_IMAGE"; then
      return
    fi
    if [[ "$attempt" -eq 3 ]]; then
      die 'docker push 连续失败 3 次；本地镜像仍保留，修复网络后重跑脚本即可'
    fi
    delay=$((attempt * 5))
    warn "docker push 第 ${attempt} 次失败，${delay} 秒后复用已上传层重试"
    sleep "$delay"
  done
}

verify_remote_manifest() {
  CURRENT_STEP='检查 ACR 远程 Manifest'
  local manifest_info manifest_json platforms
  manifest_info="$(docker buildx imagetools inspect "$REMOTE_IMAGE")"
  printf '%s\n' "$manifest_info"
  manifest_json="$(docker manifest inspect --verbose "$REMOTE_IMAGE")"
  platforms="$(
    jq -r \
      '.. | objects | select(has("os") and has("architecture"))
       | "\(.os)/\(.architecture)"' <<<"$manifest_json" \
      | sort -u
  )"

  if grep -Fxq 'unknown/unknown' <<<"$platforms"; then
    die '远程镜像包含 FC 不支持的 unknown/unknown attestation manifest'
  fi
  if [[ "$platforms" != 'linux/amd64' ]]; then
    die "远程镜像必须且只能包含 linux/amd64，实际平台：${platforms:-无法识别}"
  fi
  log '远程 Manifest 兼容 FC'
}

write_deployment_result() {
  CURRENT_STEP='写入非敏感部署结果'
  local temporary_result
  [[ ! -L "$DEPLOYMENT_RESULT_FILE" && ! -d "$DEPLOYMENT_RESULT_FILE" ]] \
    || die "结果路径不能是符号链接或目录：$DEPLOYMENT_RESULT_FILE"
  temporary_result="$(mktemp "$TRANSFER_ROOT/.deployment-result.XXXXXX")"
  {
    printf 'GIT_COMMIT=%q\n' "$GIT_COMMIT"
    printf 'REMOTE_IMAGE=%q\n' "$REMOTE_IMAGE"
    printf 'OSS_BUCKET=%q\n' "$OSS_BUCKET"
    printf 'OSS_PREFIX=%q\n' "$OSS_PREFIX"
    printf 'FC_OSS_BUCKET_PATH=%q\n' "/$OSS_PREFIX"
    printf 'FC_OSS_MOUNT_DIR=%q\n' '/mnt/nas/sam3d'
    printf 'FC_PORT=%q\n' '9000'
  } >"$temporary_result"
  chmod 600 "$temporary_result"
  mv -f -- "$temporary_result" "$DEPLOYMENT_RESULT_FILE"
}

print_completion() {
  cat <<EOF

香港 ECS 全部动作已完成

  镜像：             $REMOTE_IMAGE
  OSS Bucket：       $OSS_BUCKET
  OSS Bucket 子目录：/$OSS_PREFIX
  FC 本地挂载目录：  /mnt/nas/sam3d
  FC 监听端口：       9000
  结果文件：          $DEPLOYMENT_RESULT_FILE

下一步只需在深圳 FC 控制台选择上面的镜像，把 OSS 子目录只读挂载到
/mnt/nas/sam3d，并按 DEPLOYMENT.md 配置 GPU、环境变量、触发器和弹性策略。
EOF
}

main() {
  parse_args "$@"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    collect_required_inputs
    validate_inputs
    resolve_release_values
    print_plan
    printf '\ndry-run：未执行任何系统或云端修改。\n'
    return
  fi

  require_target_host
  collect_required_inputs
  validate_inputs
  resolve_release_values
  require_clean_checkout
  print_plan
  confirm_execution
  check_resources
  install_base_tools
  ensure_docker
  ensure_tool_venv
  ensure_ossutil
  prepare_offline_assets
  ensure_oss_access
  upload_offline_assets
  build_image
  login_acr
  push_image_with_retry
  verify_remote_manifest
  write_deployment_result

  docker logout "$ACR_HOST" >/dev/null
  ACR_LOGIN_ACTIVE=0
  print_completion
}

main "$@"
