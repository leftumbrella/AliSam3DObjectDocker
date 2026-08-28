FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS unified-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG MICROMAMBA_VERSION=2.8.1-0
ARG MICROMAMBA_SHA256=9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82
ARG SAM3D_REPOSITORY=https://v4.gh-proxy.org/https://github.com/facebookresearch/sam-3d-objects.git
ARG SAM3D_REF=f91db411c50efee93d8db7aeb323885650f6f722
ARG SAM3_REPOSITORY=https://v4.gh-proxy.org/https://github.com/facebookresearch/sam3.git
# The Git source and checkpoint revisions are independently versioned; this
# source revision is separate from the pin in scripts/prepare_offline_assets.py.
ARG SAM3_REF=8f0b7f4d4e7eda2ed606ebde6702c93359ad01da
# PyTorch3D v0.7.9 contains the CUDA 12.6+ header compatibility fix that the
# former upstream SAM3D pin predates.
ARG PYTORCH3D_REF=33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba
ARG PYPI_INDEX_URL=http://mirrors.aliyun.com/pypi/simple/
ARG PYTORCH_WHEEL_URL=https://mirrors.aliyun.com/pytorch-wheels/cu126
ARG TORCH_CUDA_ARCH_LIST=8.9;9.0
ARG MAX_JOBS=2
ARG NVCC_THREADS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bzip2 \
        build-essential \
        ca-certificates \
        curl \
        git \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
        "https://v4.gh-proxy.org/https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64" \
        -o /usr/local/bin/micromamba \
    && echo "${MICROMAMBA_SHA256}  /usr/local/bin/micromamba" | sha256sum -c - \
    && chmod 0755 /usr/local/bin/micromamba

ENV MAMBA_ROOT_PREFIX=/opt/micromamba

# Both applications use one Python 3.12 / PyTorch 2.7 / CUDA 12.6 environment.
# CUDA build tooling comes from the devel stage and is not copied into runtime.
RUN micromamba create -y -p /opt/venv -c conda-forge \
        python=3.12.11 \
        pip=25.1.1 \
        setuptools=80.9.0 \
        wheel=0.45.1 \
    && micromamba clean --all --yes

ENV PATH=/opt/venv/bin:${PATH} \
    CONDA_PREFIX=/opt/venv \
    CUDA_HOME=/usr/local/cuda \
    PIP_INDEX_URL=${PYPI_INDEX_URL} \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORCE_CUDA=1 \
    LIDRA_SKIP_INIT=true \
    ATTN_BACKEND=sdpa \
    SPARSE_ATTN_BACKEND=sdpa \
    SPARSE_BACKEND=spconv \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS} \
    PYTHONPATH=/opt/sam-3d-objects:/opt/sam-3d-objects/notebook:/opt/sam3

WORKDIR /opt/sam-3d-objects

RUN git init \
    && git remote add origin "${SAM3D_REPOSITORY}" \
    && git fetch --depth=1 origin "${SAM3D_REF}" \
    && git checkout --detach FETCH_HEAD

COPY patches/fc-runtime.patch /tmp/fc-runtime.patch

# This patch is coupled to the pinned SAM3D commit and removes optional
# notebook/visualization imports from the FC inference path.
RUN git apply --check /tmp/fc-runtime.patch \
    && git apply /tmp/fc-runtime.patch \
    && rm /tmp/fc-runtime.patch

WORKDIR /opt/sam3

RUN git init \
    && git remote add origin "${SAM3_REPOSITORY}" \
    && git fetch --depth=1 origin "${SAM3_REF}" \
    && git checkout --detach FETCH_HEAD

# Install the single Torch ABI before any native extension is compiled.
RUN python -m pip install \
        --find-links "${PYTORCH_WHEEL_URL}" \
        torch==2.7.1+cu126 \
        torchvision==0.22.1+cu126

COPY requirements-fc.txt /tmp/requirements-fc.txt
COPY requirements-segmenter.txt /tmp/requirements-segmenter.txt
COPY requirements-server.txt /tmp/requirements-server.txt
COPY constraints-unified.txt /tmp/constraints-unified.txt

RUN python -m pip install \
        --constraint /tmp/constraints-unified.txt \
        -r /tmp/requirements-fc.txt \
        -r /tmp/requirements-segmenter.txt \
        -r /tmp/requirements-server.txt \
    && rm \
        /tmp/constraints-unified.txt \
        /tmp/requirements-fc.txt \
        /tmp/requirements-segmenter.txt \
        /tmp/requirements-server.txt

# MoGe and utils3d are installed without demo/OpenGL dependency trees. Both
# source projects are pinned and share the already-installed unified Torch ABI.
RUN python -m pip install \
        --no-deps \
        --no-build-isolation \
        "utils3d @ git+https://v4.gh-proxy.org/https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
        "MoGe @ git+https://v4.gh-proxy.org/https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

WORKDIR /opt/sam-3d-objects

RUN python -m pip install \
        --no-deps \
        -e . \
    && ./patching/hydra

WORKDIR /opt/sam3

RUN python -m pip install \
        --no-deps \
        --no-build-isolation \
        -e . \
    && python -c 'from sam3.model_builder import build_tracker; from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor'

COPY scripts/check_cuda_build_env.py /tmp/check_cuda_build_env.py

# Build every Torch CUDA extension against the same Python 3.12, Torch 2.7.1
# and CUDA 12.6 ABI. The final image receives only their runtime artifacts.
RUN python /tmp/check_cuda_build_env.py \
    && python -m pip install \
        --no-deps \
        --no-build-isolation \
        "pytorch3d @ git+https://v4.gh-proxy.org/https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_REF}"

RUN python -m pip install \
        --no-deps \
        --no-build-isolation \
        "gsplat @ git+https://v4.gh-proxy.org/https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

# Keep the expensive native-extension layers cacheable when only the offline
# source rewrite changes.
COPY scripts/patch_offline_runtime.py /tmp/patch_offline_runtime.py

WORKDIR /opt/sam-3d-objects

RUN python /tmp/patch_offline_runtime.py /opt/sam-3d-objects

COPY scripts/check_runtime_imports.py /tmp/check_runtime_imports.py

RUN rm -rf /opt/sam-3d-objects/.git /opt/sam3/.git \
    && python /tmp/check_runtime_imports.py \
    && rm \
        /tmp/check_cuda_build_env.py \
        /tmp/check_runtime_imports.py \
        /tmp/patch_offline_runtime.py

FROM ubuntu:22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG PYPI_INDEX_URL=http://mirrors.aliyun.com/pypi/simple/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=unified-builder /opt/venv /opt/venv
COPY --from=unified-builder /opt/sam-3d-objects /opt/sam-3d-objects
COPY --from=unified-builder /opt/sam3 /opt/sam3

COPY app /srv/app
COPY segmenter /srv/segmenter
COPY shared /srv/shared
COPY scripts/fc_initializer.sh /srv/scripts/fc_initializer.sh
COPY scripts/check_runtime_imports.py /tmp/check_runtime_imports.py

ENV PATH=/opt/venv/bin:${PATH} \
    CONDA_PREFIX=/opt/venv \
    PIP_INDEX_URL=${PYPI_INDEX_URL} \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    PYTHONPATH=/srv:/opt/sam-3d-objects:/opt/sam-3d-objects/notebook:/opt/sam3 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LIDRA_SKIP_INIT=true \
    ATTN_BACKEND=sdpa \
    SPARSE_ATTN_BACKEND=sdpa \
    SPARSE_BACKEND=spconv \
    SAM3D_ROOT=/opt/sam-3d-objects \
    SAM3D_CONFIG_PATH=/mnt/nas/sam3d/hf/pipeline.yaml \
    TORCH_HOME=/mnt/nas/sam3d/cache/torch \
    HF_HOME=/mnt/nas/sam3d/cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    SAM3D_COMPILE=false \
    SAM3D_MAX_UPLOAD_MB=20 \
    SAM3D_MAX_REQUEST_MB=30 \
    SAM3D_MAX_IMAGE_PIXELS=40000000 \
    SAM3D_TMP_DIR=/tmp/sam3d \
    SAM3_ROOT=/opt/sam3 \
    SAM3_CHECKPOINT_PATH=/mnt/nas/sam3d/sam3/sam3.pt \
    SAM3_MAX_UPLOAD_MB=20 \
    SAM3_MAX_IMAGE_PIXELS=40000000 \
    SAM3_MAX_POINTS=64 \
    SAM3_INTERNAL_PORT=9001 \
    SAM3_INTERNAL_URL=http://127.0.0.1:9001 \
    SAM3_INTERNAL_STARTUP_TIMEOUT=30 \
    SAM3_INTERNAL_REQUEST_TIMEOUT=1800 \
    GPU_LOCK_PATH=/tmp/sam3d-gpu.lock \
    PORT=9000 \
    KEEP_ALIVE_TIMEOUT=900 \
    FC_INITIALIZER_HTTP_TIMEOUT=295

WORKDIR /opt/sam-3d-objects

# Run the import gate again after copying into the final Ubuntu runtime so
# cross-stage ABI or missing-system-library failures stop the image build.
RUN python /tmp/check_runtime_imports.py \
    && rm /tmp/check_runtime_imports.py

EXPOSE 9000

CMD ["python", "-m", "app.supervisor"]
