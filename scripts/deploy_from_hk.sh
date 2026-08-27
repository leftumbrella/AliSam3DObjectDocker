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
readonly TRANSFER_ROOT_DEFAULT='/root/sam3d-transfer'
readonly SAM3D_REF='f91db411c50efee93d8db7aeb323885650f6f722'
readonly SAM3_REF='8f0b7f4d4e7eda2ed606ebde6702c93359ad01da'
readonly TORCH_CUDA_ARCH_LIST_VALUE='8.9;9.0'
readonly MAX_JOBS_VALUE='2'
readonly NVCC_THREADS_VALUE='2'

TRANSFER_ROOT=$TRANSFER_ROOT_DEFAULT
OSS_BUCKET=''
ACR_IMAGE=''
ACR_USERNAME=''
ACR_HOST=''
OSSUTIL_BIN=''
HF_BIN=''
TOOLS_PYTHON=''
OSS_UPLOAD_LIST=''
GIT_COMMIT=''
GIT_COMMIT_SHORT=''
IMAGE_TAG=''
LOCAL_IMAGE=''
REMOTE_IMAGE=''
LOCAL_CONFIG_DIGEST=''
BUILD_METADATA_FILE=''
ASSET_RELEASE=''
OSS_PREFIX=''
DEPLOYMENT_RESULT_FILE=''
CURRENT_STEP='启动'
ACR_LOGIN_ACTIVE=0
TEMP_DIR=''

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
  printf '[%s] 失败：步骤“%s”，行号 %s，退出码 %s。修正后直接重跑脚本即可。\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "$CURRENT_STEP" \
    "${BASH_LINENO[0]:-unknown}" \
    "$status" >&2
  exit "$status"
}

safe_remove_temp_dir() {
  [[ -n "$TEMP_DIR" ]] || return
  case "$TEMP_DIR" in
    /tmp/sam3d-acr-push.*)
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

  if [[ "$ACR_LOGIN_ACTIVE" -eq 1 && -n "$ACR_HOST" ]] \
    && command -v docker >/dev/null 2>&1; then
    docker logout "$ACR_HOST" >/dev/null 2>&1
  fi
  unset ACR_IMAGE ACR_USERNAME ACR_HOST DOCKER_CONFIG
  unset HF_TOKEN OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET OSS_SESSION_TOKEN
  unset OSS_REGION OSS_ENDPOINT
  safe_remove_temp_dir
  exit "$status"
}

trap on_error ERR
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_no_arguments() {
  [[ $# -eq 0 ]] \
    || die "脚本不接受任何参数。请直接运行：./scripts/$SCRIPT_NAME"
}

require_target_host() {
  [[ "$EUID" -eq 0 ]] \
    || die "请先执行 sudo -i 切换到 root，再重新运行 $SCRIPT_NAME"
  [[ "$(uname -m)" == 'x86_64' ]] || die '构建机必须是 x86_64/amd64'
  [[ -r /etc/os-release ]] || die '无法读取 /etc/os-release'

  # shellcheck disable=SC1091
  . /etc/os-release
  [[ "${ID:-}" == 'ubuntu' ]] || die '脚本只支持 Ubuntu'
  case "${VERSION_ID:-}" in
    22.04|24.04) ;;
    *) warn "当前 Ubuntu ${VERSION_ID:-unknown} 不是已验证的 22.04/24.04" ;;
  esac
  [[ -t 0 ]] || die '脚本需要交互终端来读取 OSS、必要时的 Hugging Face 和 ACR 信息'
}

prompt_required_inputs() {
  CURRENT_STEP='读取 OSS 和 ACR 必要信息'
  read -r -p '深圳 OSS Bucket 名: ' OSS_BUCKET
  read -r -p 'ACR 完整公网仓库地址（不含协议和 tag）: ' ACR_IMAGE
  read -r -p 'ACR 登录用户名: ' ACR_USERNAME
  [[ -n "$OSS_BUCKET" ]] || die 'OSS Bucket 名不能为空'
  [[ -n "$ACR_IMAGE" ]] || die 'ACR 仓库地址不能为空'
  [[ -n "$ACR_USERNAME" ]] || die 'ACR 登录用户名不能为空'
}

validate_inputs() {
  local resolved_transfer_root
  local acr_image_pattern='^([A-Za-z0-9.-]+)/([a-z0-9._-]+)/([a-z0-9._-]+)$'
  [[ "$OSS_BUCKET" =~ ^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$ ]] \
    || die 'OSS Bucket 名必须为 3 到 63 位小写字母、数字或连字符'
  if [[ "$ACR_IMAGE" =~ $acr_image_pattern ]]; then
    ACR_HOST=${BASH_REMATCH[1]}
  else
    die 'ACR 仓库地址必须是域名/namespace/repository，不能包含协议、tag 或多余路径'
  fi
  [[ "$ACR_HOST" == *'.aliyuncs.com' ]] \
    || die 'ACR 仓库必须使用阿里云容器镜像服务域名'
  [[ "$ACR_HOST" != *'-vpc'* && "$ACR_HOST" != *'-internal'* ]] \
    || die '构建机推送镜像必须使用 ACR 公网地址，不能使用 -vpc 或 -internal'
  [[ ! "$ACR_USERNAME" =~ [[:cntrl:]] ]] \
    || die 'ACR 登录用户名包含非法控制字符'
  resolved_transfer_root="$(realpath -m -- "$TRANSFER_ROOT")"
  [[ "$resolved_transfer_root" != '/' ]] || die '离线资源目录不能是文件系统根目录'
  case "$resolved_transfer_root" in
    "$PROJECT_ROOT"|"$PROJECT_ROOT"/*)
      die '离线资源目录不能位于 Git 项目中，否则模型权重会进入 Docker 构建上下文'
      ;;
  esac
}

resolve_image_names() {
  GIT_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || true)"
  GIT_COMMIT_SHORT="$(git -C "$PROJECT_ROOT" rev-parse --short=12 HEAD 2>/dev/null || true)"
  [[ "$GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die '无法解析当前 Git commit'
  [[ "$GIT_COMMIT_SHORT" =~ ^[0-9a-f]{12}$ ]] || die '无法生成镜像版本号'
  LOCAL_IMAGE="sam3-sam3d-fc:build-${GIT_COMMIT_SHORT}"
}

require_clean_checkout() {
  [[ -d "$PROJECT_ROOT/.git" ]] || die '项目必须通过 git clone 获取'
  local status
  status="$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=normal)"
  [[ -z "$status" ]] \
    || die "Git checkout 不是干净状态，拒绝构建不可追溯镜像：\n$status"
}

check_resources() {
  local free_kb free_gb memory_kb memory_gb storage_probe
  storage_probe=$TRANSFER_ROOT
  while [[ ! -e "$storage_probe" ]]; do
    storage_probe="$(dirname -- "$storage_probe")"
  done
  free_kb="$(df -Pk "$storage_probe" | awk 'NR == 2 {print $4}')"
  free_gb=$((free_kb / 1024 / 1024))
  (( free_gb >= 100 )) \
    || warn "当前文件系统仅剩约 ${free_gb} GB，建议至少准备 100 GB 可用空间"

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
  if command -v docker >/dev/null 2>&1 \
    && docker buildx version >/dev/null 2>&1; then
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

build_image() {
  CURRENT_STEP='构建 SAM3/SAM3D 统一 linux/amd64 镜像'
  cd "$PROJECT_ROOT"
  BUILD_METADATA_FILE="$TEMP_DIR/build-metadata.json"
  docker buildx build \
    --load \
    --metadata-file "$BUILD_METADATA_FILE" \
    --progress=plain \
    --platform linux/amd64 \
    --provenance=false \
    --sbom=false \
    --build-arg "SAM3D_REF=${SAM3D_REF}" \
    --build-arg "SAM3_REF=${SAM3_REF}" \
    --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VALUE}" \
    --build-arg "MAX_JOBS=${MAX_JOBS_VALUE}" \
    --build-arg "NVCC_THREADS=${NVCC_THREADS_VALUE}" \
    -t "$LOCAL_IMAGE" \
    .

  local platform image_size digest_hex
  platform="$(docker image inspect "$LOCAL_IMAGE" --format '{{.Os}}/{{.Architecture}}')"
  [[ "$platform" == 'linux/amd64' ]] || die "本地镜像平台错误：$platform"
  image_size="$(docker image inspect "$LOCAL_IMAGE" --format '{{.Size}}')"
  [[ "$image_size" =~ ^[0-9]+$ ]] || die '无法读取本地镜像大小'
  LOCAL_CONFIG_DIGEST="$(
    jq -er \
      '."containerimage.config.digest"
       // ."containerimage.descriptor".annotations["config.digest"]
       // empty' \
      "$BUILD_METADATA_FILE"
  )" || die 'Buildx 元数据未包含镜像配置摘要'
  [[ "$LOCAL_CONFIG_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die 'Buildx 返回的镜像配置摘要格式无效'
  digest_hex=${LOCAL_CONFIG_DIGEST#sha256:}
  IMAGE_TAG="sam3-sam3d-${GIT_COMMIT_SHORT}-${digest_hex:0:12}"
  REMOTE_IMAGE="${ACR_IMAGE}:${IMAGE_TAG}"
  log "镜像构建完成：$LOCAL_IMAGE，未压缩大小 ${image_size} 字节"
  log "自动生成内容寻址标签：$REMOTE_IMAGE"
}

ensure_temp_dir() {
  if [[ -z "$TEMP_DIR" ]]; then
    TEMP_DIR="$(mktemp -d /tmp/sam3d-acr-push.XXXXXX)"
  fi
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
  local version_output cp_help_output hash_help_output
  [[ -x "$candidate" ]] || return 1
  version_output="$("$candidate" version 2>/dev/null)" || return 1
  [[ "$version_output" == *"$OSSUTIL_VERSION"* ]] || return 1
  cp_help_output="$("$candidate" cp --help 2>/dev/null)" || return 1
  hash_help_output="$("$candidate" hash --help 2>/dev/null)" || return 1
  grep -q -- '--checkpoint-dir' <<<"$cp_help_output" \
    && grep -q -- '--files-from-raw' <<<"$cp_help_output" \
    && grep -q -- '--output-dir' <<<"$cp_help_output" \
    && grep -q -- 'hash md5|crc64' <<<"$hash_help_output"
}

ensure_ossutil() {
  CURRENT_STEP='安装并校验 ossutil'
  local candidate archive

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
  archive="ossutil-${OSSUTIL_VERSION}-linux-amd64.zip"
  curl --fail --location --retry 5 --retry-all-errors \
    --output "$TEMP_DIR/$archive" \
    "https://gosspublic.alicdn.com/ossutil/v2/${OSSUTIL_VERSION}/${archive}"
  printf '%s  %s\n' "$OSSUTIL_SHA256" "$TEMP_DIR/$archive" | sha256sum -c -
  unzip -q "$TEMP_DIR/$archive" -d "$TEMP_DIR"
  install -m 0755 \
    "$TEMP_DIR/ossutil-${OSSUTIL_VERSION}-linux-amd64/ossutil" \
    "$candidate"
  if ! ossutil_is_compatible "$candidate"; then
    warn 'ossutil 契约检查失败，诊断输出如下（version / cp --help）'
    "$candidate" version || true
    "$candidate" cp --help || true
    die "ossutil 安装后命令契约检查失败：需要 ${OSSUTIL_VERSION} 的断点、文件清单、输出目录和 CRC64 能力"
  fi
  OSSUTIL_BIN=$candidate
  "$OSSUTIL_BIN" version
}

ensure_huggingface_access() {
  CURRENT_STEP='验证 Hugging Face 模型权限'
  local token=''
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
  if [[ -z "${HF_TOKEN:-}" ]]; then
    read -r -s -p 'Hugging Face Access Token（输入时不会显示）: ' token
    printf '\n'
    [[ -n "$token" ]] || die 'Hugging Face Access Token 不能为空'
    export HF_TOKEN=$token
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

api = HfApi()
api.model_info("facebook/sam-3d-objects", token=os.environ["HF_TOKEN"])
PY
  then
    unset HF_TOKEN
    die '无法访问 facebook/sam-3d-objects；请确认该模型已获批且网络正常'
  fi
  log 'Hugging Face Access Token 验证成功；Token 仅保存在当前脚本进程环境中'
}

prepare_offline_assets() {
  CURRENT_STEP='准备并校验完整离线模型资源'
  local prepare_script="$PROJECT_ROOT/scripts/prepare_offline_assets.py"
  local -a args

  mkdir -p -- "$TRANSFER_ROOT"
  [[ ! -L "$TRANSFER_ROOT" ]] || die '离线资源目录不得是符号链接'
  chmod 700 "$TRANSFER_ROOT"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

  if "$TOOLS_PYTHON" "$prepare_script" \
    --verify-only \
    --transfer-root "$TRANSFER_ROOT" >/dev/null 2>&1; then
    log '现有离线资源完整，跳过下载'
  else
    args=(--transfer-root "$TRANSFER_ROOT")
    local needs_huggingface=0
    if "$TOOLS_PYTHON" "$prepare_script" \
      --verify-main-only \
      --transfer-root "$TRANSFER_ROOT" >/dev/null 2>&1; then
      log '现有 SAM3D 主 checkpoint 完整，补齐其他离线资源'
    else
      args+=(--download-sam3d)
      needs_huggingface=1
    fi
    if "$TOOLS_PYTHON" "$prepare_script" \
      --verify-sam3-only \
      --transfer-root "$TRANSFER_ROOT" >/dev/null 2>&1; then
      log '现有 SAM3 checkpoint 完整，复用现有文件'
    else
      args+=(--download-sam3)
      log 'SAM3 checkpoint 缺失，将从 ModelScope 下载公开权重'
    fi
    if [[ "$needs_huggingface" -eq 1 ]]; then
      ensure_huggingface_access
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

  warn '现有 ossutil 配置、环境凭证或 ECS RAM Role 无法访问目标 Bucket，将读取临时 RAM/STS 凭证'
  unset OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET OSS_SESSION_TOKEN
  prompt_oss_credentials
  "$OSSUTIL_BIN" ls "oss://${OSS_BUCKET}/" >/dev/null \
    || die 'OSS 访问失败，请核对 Bucket 名、深圳地域和 RAM 权限'
}

write_oss_upload_list() {
  local manifest="$TRANSFER_ROOT/storage/offline-assets.sha256"
  local line key temporary_list
  OSS_UPLOAD_LIST="$TRANSFER_ROOT/oss-upload-files-${ASSET_RELEASE}.txt"
  [[ ! -L "$OSS_UPLOAD_LIST" && ! -d "$OSS_UPLOAD_LIST" ]] \
    || die "OSS 上传清单路径不能是符号链接或目录：$OSS_UPLOAD_LIST"
  temporary_list="$(mktemp "$TRANSFER_ROOT/.oss-upload-files.XXXXXX")"
  printf '%s\n' 'offline-assets.sha256' >"$temporary_list"
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ ! "$line" =~ ^[0-9a-f]{64}[[:space:]][[:space:]](.+)$ ]]; then
      die "离线资源清单行格式无效：$line"
    fi
    key=${BASH_REMATCH[1]}
    case "$key" in
      /*|../*|*/../*|*/..) die "离线资源清单包含不安全路径：$key" ;;
    esac
    [[ -f "$TRANSFER_ROOT/storage/$key" && ! -L "$TRANSFER_ROOT/storage/$key" ]] \
      || die "离线资源清单对象不是普通文件：$key"
    printf '%s\n' "$key" >>"$temporary_list"
  done <"$manifest"
  chmod 600 "$temporary_list"
  mv -f -- "$temporary_list" "$OSS_UPLOAD_LIST"
}

oss_crc64() {
  local source=$1
  local output line
  output="$("$OSSUTIL_BIN" hash crc64 "$source")" || return 1
  while IFS= read -r line; do
    if [[ "$line" =~ ^([0-9]+)[[:space:]]+(.+)$ ]] \
      && [[ "${BASH_REMATCH[2]}" == "$source" ]]; then
      printf '%s\n' "${BASH_REMATCH[1]}"
      return
    fi
  done <<<"$output"
  return 1
}

verify_or_repair_remote_oss_object() {
  local key=$1
  local checkpoint_dir=$2
  local output_dir=$3
  local source="$TRANSFER_ROOT/storage/$key"
  local url="oss://${OSS_BUCKET}/${OSS_PREFIX}/${key}"
  local local_crc64 remote_crc64=''

  local_crc64="$(oss_crc64 "$source")" \
    || die "无法计算本地资源 CRC64：$source"
  if remote_crc64="$(oss_crc64 "$url")" \
    && [[ "$remote_crc64" == "$local_crc64" ]]; then
    return
  fi

  warn "OSS 对象缺失或 CRC64 不一致，将强制修复：$url"
  "$OSSUTIL_BIN" cp -f \
    "$source" \
    "$url" \
    --checkpoint-dir "$checkpoint_dir" \
    --output-dir "$output_dir"
  remote_crc64="$(oss_crc64 "$url")" \
    || die "修复后仍无法读取 OSS 对象 CRC64：$url"
  [[ "$remote_crc64" == "$local_crc64" ]] \
    || die "修复后 OSS 对象 CRC64 仍与本地不一致：$url"
}

verify_remote_oss_inventory() {
  local checkpoint_dir=$1
  local output_dir=$2
  local manifest="$TRANSFER_ROOT/storage/offline-assets.sha256"
  local line key count=0
  verify_or_repair_remote_oss_object \
    'offline-assets.sha256' "$checkpoint_dir" "$output_dir"
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ ! "$line" =~ ^[0-9a-f]{64}[[:space:]][[:space:]](.+)$ ]]; then
      die "离线资源清单行格式无效：$line"
    fi
    key=${BASH_REMATCH[1]}
    case "$key" in
      /*|../*|*/../*|*/..) die "离线资源清单包含不安全路径：$key" ;;
    esac
    verify_or_repair_remote_oss_object "$key" "$checkpoint_dir" "$output_dir"
    ((count += 1))
  done <"$manifest"
  (( count > 0 )) || die '离线资源清单为空'
  log "OSS 已逐项核对 ${count} 个模型资源对象和校验清单的 CRC64：oss://${OSS_BUCKET}/${OSS_PREFIX}/"
}

upload_offline_assets() {
  CURRENT_STEP='断点上传完整离线资源到深圳 OSS'
  local checkpoint_dir="$TRANSFER_ROOT/oss-upload-checkpoints"
  local output_dir="$TRANSFER_ROOT/ossutil-output"
  mkdir -p -- "$checkpoint_dir"
  mkdir -p -- "$output_dir"
  chmod 700 "$checkpoint_dir"
  chmod 700 "$output_dir"
  write_oss_upload_list

  "$OSSUTIL_BIN" cp -u -r \
    "$TRANSFER_ROOT/storage/" \
    "oss://${OSS_BUCKET}/${OSS_PREFIX}/" \
    --files-from-raw "$OSS_UPLOAD_LIST" \
    --checkpoint-dir "$checkpoint_dir" \
    --output-dir "$output_dir"

  verify_remote_oss_inventory "$checkpoint_dir" "$output_dir"
  unset OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET OSS_SESSION_TOKEN OSS_REGION OSS_ENDPOINT
}

login_acr() {
  CURRENT_STEP='登录 ACR'
  local password=''
  ensure_temp_dir
  install -m 0700 -d "$TEMP_DIR/docker-config"
  export DOCKER_CONFIG="$TEMP_DIR/docker-config"

  read -r -s -p 'ACR Registry 密码: ' password
  printf '\n'
  [[ -n "$password" ]] || die 'ACR Registry 密码不能为空'
  printf '%s' "$password" | docker login \
    --username "$ACR_USERNAME" \
    --password-stdin \
    "$ACR_HOST"
  password=''
  ACR_LOGIN_ACTIVE=1
}

get_remote_config_digest() {
  local remote_image=$1
  local raw lookup_error media_type child_digest config_digest

  if ! raw="$(docker buildx imagetools inspect --raw "$remote_image" 2>&1)"; then
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
      raw="$(docker buildx imagetools inspect --raw "$remote_image@$child_digest")" \
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
  local remote_digest status

  set +e
  remote_digest="$(get_remote_config_digest "$REMOTE_IMAGE")"
  status=$?
  set -e

  case "$status" in
    0)
      if [[ "$remote_digest" == "$LOCAL_CONFIG_DIGEST" ]]; then
        return 0
      fi
      die "远程不可变标签已指向另一镜像，拒绝覆盖：$REMOTE_IMAGE"
      ;;
    1) return 1 ;;
    *) die "无法安全确认远程标签状态，拒绝覆盖：$REMOTE_IMAGE" ;;
  esac
}

remote_digest_matches_local_with_retry() {
  local attempt remote_digest status
  for attempt in 1 2 3 4 5; do
    set +e
    remote_digest="$(get_remote_config_digest "$REMOTE_IMAGE")"
    status=$?
    set -e
    if [[ "$status" -eq 0 && "$remote_digest" == "$LOCAL_CONFIG_DIGEST" ]]; then
      return 0
    fi
    [[ "$attempt" -eq 5 ]] || sleep "$attempt"
  done
  return 1
}

push_image() {
  CURRENT_STEP='推送统一镜像到 ACR'
  local attempt delay
  docker tag "$LOCAL_IMAGE" "$REMOTE_IMAGE"

  if remote_tag_matches_local_image; then
    log '远程不可变标签已经是同一镜像，跳过重复 push'
    return
  fi

  for attempt in 1 2 3; do
    if docker push "$REMOTE_IMAGE"; then
      remote_digest_matches_local_with_retry \
        || die '镜像已推送，但远程配置摘要未与本地镜像一致'
      return
    fi
    if remote_digest_matches_local_with_retry; then
      log 'push 回包失败，但远程配置摘要已确认与本地镜像一致'
      return
    fi
    if [[ "$attempt" -eq 3 ]]; then
      die 'docker push 连续失败 3 次；本地镜像仍保留，修复网络后直接重跑脚本即可'
    fi
    delay=$((attempt * 5))
    warn "docker push 第 ${attempt} 次失败，${delay} 秒后复用已上传层重试"
    sleep "$delay"
  done
}

verify_remote_manifest() {
  CURRENT_STEP='检查 ACR 远程 Manifest'
  local attempt manifest_info manifest_json platforms
  for attempt in 1 2 3; do
    if manifest_info="$(docker buildx imagetools inspect "$REMOTE_IMAGE")" \
      && manifest_json="$(docker manifest inspect --verbose "$REMOTE_IMAGE")"; then
      printf '%s\n' "$manifest_info"
      break
    fi
    if [[ "$attempt" -eq 3 ]]; then
      die 'ACR 远程 Manifest 连续读取失败 3 次'
    fi
    warn "ACR Manifest 第 ${attempt} 次读取失败，稍后自动重试"
    sleep "$attempt"
  done
  platforms="$(
    jq -r \
      '.. | objects | select(has("os") and has("architecture"))
       | "\(.os)/\(.architecture)"' <<<"$manifest_json" \
      | sort -u
  )"

  if grep -Fxq 'unknown/unknown' <<<"$platforms"; then
    die '远程镜像包含 unknown/unknown attestation manifest'
  fi
  [[ "$platforms" == 'linux/amd64' ]] \
    || die "远程镜像必须且只能包含 linux/amd64，实际平台：${platforms:-无法识别}"
  remote_digest_matches_local_with_retry \
    || die 'Manifest 校验后远程配置摘要与本地镜像不一致'
  log 'ACR 远程 Manifest 校验通过'
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
  } >"$temporary_result"
  chmod 600 "$temporary_result"
  mv -f -- "$temporary_result" "$DEPLOYMENT_RESULT_FILE"
}

print_plan() {
  cat <<EOF

即将自动执行
  Git commit：    $GIT_COMMIT
  离线资源目录： $TRANSFER_ROOT
  深圳 OSS：     oss://$OSS_BUCKET/sam3d/releases/bundle-<资源清单摘要>/
  本地镜像：     $LOCAL_IMAGE
  目标仓库：     $ACR_IMAGE

脚本会下载或复用并校验 SAM3、SAM3D、MoGe 和 DINOv2 离线资源，
把完整资源包断点上传到新的内容寻址 OSS 前缀，再构建并推送 linux/amd64 统一镜像。
不会创建或修改 FC，也不会启动 GPU。
EOF
}

print_completion() {
  cat <<EOF

香港 ECS 资源上传和镜像推送完成

  ACR 镜像：        $REMOTE_IMAGE
  OSS Bucket：      $OSS_BUCKET
  OSS Bucket 子目录：/$OSS_PREFIX
  FC 本地挂载目录： /mnt/nas/sam3d
  Git commit：       $GIT_COMMIT
  结果文件：         $DEPLOYMENT_RESULT_FILE
EOF
}

main() {
  require_no_arguments "$@"
  require_target_host
  prompt_required_inputs
  validate_inputs
  resolve_image_names
  require_clean_checkout
  print_plan
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
  push_image
  verify_remote_manifest
  write_deployment_result

  docker logout "$ACR_HOST" >/dev/null
  ACR_LOGIN_ACTIVE=0
  print_completion
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
