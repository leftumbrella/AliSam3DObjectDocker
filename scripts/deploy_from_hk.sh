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
readonly SAM3D_REF='f91db411c50efee93d8db7aeb323885650f6f722'
readonly SAM3_REF='8f0b7f4d4e7eda2ed606ebde6702c93359ad01da'
readonly TORCH_CUDA_ARCH_LIST_VALUE='8.9;9.0'
readonly MAX_JOBS_VALUE='2'
readonly NVCC_THREADS_VALUE='2'

ACR_IMAGE=''
ACR_USERNAME=''
ACR_HOST=''
GIT_COMMIT=''
GIT_COMMIT_SHORT=''
IMAGE_TAG=''
LOCAL_IMAGE=''
REMOTE_IMAGE=''
LOCAL_CONFIG_DIGEST=''
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
  [[ -t 0 ]] || die '脚本需要交互终端来读取 ACR 必要信息'
}

prompt_required_inputs() {
  CURRENT_STEP='读取 ACR 必要信息'
  read -r -p 'ACR 完整公网仓库地址（不含协议和 tag）: ' ACR_IMAGE
  read -r -p 'ACR 登录用户名: ' ACR_USERNAME
  [[ -n "$ACR_IMAGE" ]] || die 'ACR 仓库地址不能为空'
  [[ -n "$ACR_USERNAME" ]] || die 'ACR 登录用户名不能为空'
}

validate_acr_inputs() {
  local acr_image_pattern='^([A-Za-z0-9.-]+)/([a-z0-9._-]+)/([a-z0-9._-]+)$'
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
  local free_kb free_gb memory_kb memory_gb
  free_kb="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
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
    jq
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
  docker buildx build \
    --load \
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
  LOCAL_CONFIG_DIGEST="$(docker image inspect "$LOCAL_IMAGE" --format '{{.Id}}')"
  [[ "$LOCAL_CONFIG_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die '无法读取本地镜像配置摘要'
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

print_plan() {
  cat <<EOF

即将自动执行
  Git commit： $GIT_COMMIT
  本地镜像：   $LOCAL_IMAGE
  目标仓库：   $ACR_IMAGE

脚本只会安装必要构建工具、构建一张 linux/amd64 统一镜像并推送到 ACR。
最终标签由 Git commit 和构建后的镜像摘要自动生成。
不会下载模型权重，不会访问 OSS，不会创建或修改 FC，也不会启动 GPU。
EOF
}

print_completion() {
  cat <<EOF

镜像构建和推送完成

  ACR 镜像：$REMOTE_IMAGE
  Git commit：$GIT_COMMIT
EOF
}

main() {
  require_no_arguments "$@"
  require_target_host
  prompt_required_inputs
  validate_acr_inputs
  resolve_image_names
  require_clean_checkout
  print_plan
  check_resources
  install_base_tools
  ensure_docker
  login_acr
  build_image
  push_image
  verify_remote_manifest

  docker logout "$ACR_HOST" >/dev/null
  ACR_LOGIN_ACTIVE=0
  print_completion
}

main "$@"
