FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG MICROMAMBA_VERSION=2.8.1-0
ARG MICROMAMBA_SHA256=9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82
ARG SAM3D_REPOSITORY=https://github.com/facebookresearch/sam-3d-objects.git
ARG SAM3D_REF=f91db411c50efee93d8db7aeb323885650f6f722
ARG TORCH_CUDA_ARCH_LIST=8.9
ARG MAX_JOBS=4
ARG NVCC_THREADS=4

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
        "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64" \
        -o /usr/local/bin/micromamba \
    && echo "${MICROMAMBA_SHA256}  /usr/local/bin/micromamba" | sha256sum -c - \
    && chmod 0755 /usr/local/bin/micromamba

ENV MAMBA_ROOT_PREFIX=/opt/micromamba

WORKDIR /opt/sam-3d-objects

RUN git init \
    && git remote add origin "${SAM3D_REPOSITORY}" \
    && git fetch --depth=1 origin "${SAM3D_REF}" \
    && git checkout --detach FETCH_HEAD

RUN micromamba env create -y -f environments/default.yml \
    && micromamba clean --all --yes

ENV PATH=/opt/micromamba/envs/sam3d-objects/bin:${PATH} \
    CONDA_PREFIX=/opt/micromamba/envs/sam3d-objects \
    CUDA_HOME=/opt/micromamba/envs/sam3d-objects \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121" \
    PIP_FIND_LINKS=https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS}

RUN python -m pip install -e . \
    && python -m pip install -e '.[p3d]' \
    && python -m pip install -e '.[inference]' \
    && ./patching/hydra

COPY requirements-server.txt /tmp/requirements-server.txt
COPY scripts/check_mamba_removal.py /tmp/check_mamba_removal.py
COPY scripts/check_runtime_imports.py /tmp/check_runtime_imports.py

RUN python -m pip install -r /tmp/requirements-server.txt \
    && rm /tmp/requirements-server.txt \
    && rm -rf /opt/sam-3d-objects/.git

# CUDA 扩展已完成编译。裁剪 Conda 构建链前先检查实际级联删除计划，
# 确保动态 CUDA 运行库、NVRTC、Python 和 C++ 运行库仍被保留。
RUN set -eu; \
    packages="\
        binutils binutils_impl_linux-64 binutils_linux-64 \
        c-compiler cuda-cccl cuda-cccl-impl cuda-cccl_linux-64 \
        cuda-command-line-tools cuda-compiler \
        cuda-cudart-dev cuda-cudart-dev_linux-64 \
        cuda-cudart-static cuda-cudart-static_linux-64 \
        cuda-cuobjdump cuda-cupti-dev cuda-cuxxfilt \
        cuda-driver-dev cuda-driver-dev_linux-64 cuda-gdb \
        cuda-libraries-dev cuda-nsight \
        cuda-nvcc cuda-nvcc-dev_linux-64 cuda-nvcc-impl \
        cuda-nvcc-tools cuda-nvcc_linux-64 cuda-nvdisasm \
        cuda-nvml-dev cuda-nvprof cuda-nvprune cuda-nvvp \
        cuda-opencl-dev cuda-profiler-api cuda-sanitizer-api \
        cuda-tools cuda-toolkit cuda-visual-tools \
        cxx-compiler gcc gcc_impl_linux-64 gcc_linux-64 gds-tools \
        gxx gxx_impl_linux-64 gxx_linux-64 kernel-headers_linux-64 \
        libcublas-dev libcufft-dev libcufile-dev libcurand-dev \
        libgcc-devel_linux-64 libcusolver-dev libcusparse-dev \
        libnpp-dev libnvjitlink-dev libnvjpeg-dev libsanitizer \
        libstdcxx-devel_linux-64 nsight-compute opencl-headers \
        sysroot_linux-64\
    "; \
    micromamba remove --dry-run --json -n sam3d-objects ${packages} \
        > /tmp/mamba-remove-plan.json; \
    python /tmp/check_mamba_removal.py /tmp/mamba-remove-plan.json ${packages}; \
    micromamba remove -y -n sam3d-objects ${packages}; \
    python /tmp/check_runtime_imports.py; \
    micromamba clean --all --yes; \
    rm \
        /tmp/check_mamba_removal.py \
        /tmp/check_runtime_imports.py \
        /tmp/mamba-remove-plan.json

FROM ubuntu:22.04 AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/micromamba/envs/sam3d-objects /opt/micromamba/envs/sam3d-objects
COPY --from=builder /opt/sam-3d-objects /opt/sam-3d-objects

COPY app /srv/app

ENV PATH=/opt/micromamba/envs/sam3d-objects/bin:${PATH} \
    CONDA_PREFIX=/opt/micromamba/envs/sam3d-objects \
    CUDA_HOME=/opt/micromamba/envs/sam3d-objects \
    PYTHONPATH=/srv:/opt/sam-3d-objects \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SAM3D_ROOT=/opt/sam-3d-objects \
    SAM3D_CONFIG_PATH=/mnt/nas/sam3d/hf/pipeline.yaml \
    TORCH_HOME=/mnt/nas/sam3d/cache/torch \
    HF_HOME=/mnt/nas/sam3d/cache/huggingface \
    SAM3D_COMPILE=false \
    SAM3D_MAX_UPLOAD_MB=20 \
    SAM3D_MAX_REQUEST_MB=30 \
    SAM3D_MAX_IMAGE_PIXELS=40000000 \
    SAM3D_TMP_DIR=/tmp/sam3d \
    PORT=9000 \
    KEEP_ALIVE_TIMEOUT=900

WORKDIR /opt/sam-3d-objects

EXPOSE 9000

CMD ["python", "-m", "app.serve"]
