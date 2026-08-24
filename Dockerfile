FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG MICROMAMBA_VERSION=2.8.1-0
ARG MICROMAMBA_SHA256=9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82
ARG SAM3D_REPOSITORY=https://github.com/facebookresearch/sam-3d-objects.git
ARG SAM3D_REF=f91db411c50efee93d8db7aeb323885650f6f722
ARG PYPI_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_CUDA_ARCH_LIST=8.9
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

COPY patches/fc-runtime.patch /tmp/fc-runtime.patch

# 该补丁只针对上面锁定的 commit。先做完整上下文校验，避免上游升级后静默错改。
RUN git apply --check /tmp/fc-runtime.patch \
    && git apply /tmp/fc-runtime.patch \
    && rm /tmp/fc-runtime.patch

RUN micromamba env create -y -f environments/default.yml \
    && micromamba clean --all --yes

ENV PATH=/opt/micromamba/envs/sam3d-objects/bin:${PATH} \
    CONDA_PREFIX=/opt/micromamba/envs/sam3d-objects \
    CUDA_HOME=/opt/micromamba/envs/sam3d-objects \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORCE_CUDA=1 \
    LIDRA_SKIP_INIT=true \
    ATTN_BACKEND=sdpa \
    SPARSE_ATTN_BACKEND=sdpa \
    SPARSE_BACKEND=spconv \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS}

COPY requirements-fc.txt /tmp/requirements-fc.txt
COPY requirements-server.txt /tmp/requirements-server.txt
COPY scripts/check_mamba_removal.py /tmp/check_mamba_removal.py
COPY scripts/check_runtime_imports.py /tmp/check_runtime_imports.py

# PyTorch 的专用索引只作用于 PyTorch/torchvision，不再参与普通包解析。
RUN python -m pip install \
        --index-url "${PYTORCH_INDEX_URL}" \
        torch==2.5.1+cu121 \
        torchvision==0.20.1+cu121

# 只安装 FC 推理路径的普通 PyPI 依赖。MoGe 和 utils3d 使用 --no-deps，
# 避免 MoGe 声明的 Gradio 演示依赖以及 utils3d 未使用的 OpenGL 依赖。
RUN python -m pip install \
        --index-url "${PYPI_INDEX_URL}" \
        -r /tmp/requirements-fc.txt \
    && python -m pip install \
        --no-deps \
        --no-build-isolation \
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
        "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

# 上游包本身保留可编辑安装，但禁止重新解析其全量 requirements.txt。
RUN python -m pip install \
        --no-deps \
        --index-url "${PYPI_INDEX_URL}" \
        -e . \
    && ./patching/hydra

COPY scripts/check_cuda_build_env.py /tmp/check_cuda_build_env.py

# 仅从源码构建 FC 路径需要的两个 CUDA 扩展；二者均锁定上游 commit，
# 并关闭依赖解析，避免再次从专用索引或 PyPI 搜索整棵依赖树。
# conda-forge 将 CUDA 头文件和库放在 targets/x86_64-linux 下；micromamba
# 激活脚本负责将这些目录加入编译参数，单独设置 PATH/CONDA_PREFIX 不够。
RUN micromamba run -n sam3d-objects \
        python /tmp/check_cuda_build_env.py \
    && micromamba run -n sam3d-objects \
        python -m pip install \
            --no-deps \
            --no-build-isolation \
            "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"

RUN micromamba run -n sam3d-objects \
        python -m pip install \
            --no-deps \
            --no-build-isolation \
            "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

# 运行时源码补丁放在昂贵的 CUDA 扩展构建之后，避免仅修改补丁脚本时
# 让 PyTorch、PyTorch3D 和 gsplat 的缓存层全部失效。
COPY scripts/patch_offline_runtime.py /tmp/patch_offline_runtime.py

RUN python /tmp/patch_offline_runtime.py /opt/sam-3d-objects

RUN python -m pip install \
        --index-url "${PYPI_INDEX_URL}" \
        -r /tmp/requirements-server.txt \
    && rm /tmp/requirements-fc.txt /tmp/requirements-server.txt \
    && rm -rf /opt/sam-3d-objects/.git

# CUDA 扩展已完成编译。裁剪 Conda 构建链前先检查实际级联删除计划，
# 确保动态 CUDA 运行库、NVRTC、Python 和 C++ 运行库仍被保留。
# ocl-icd 在锁文件中依赖 opencl-headers；保留这组很小的运行时依赖，避免删除
# 头文件时级联移除受保护的 cuda-libraries 依赖聚合元包。
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
        libstdcxx-devel_linux-64 nsight-compute \
        sysroot_linux-64\
    "; \
    micromamba remove --dry-run --json -n sam3d-objects ${packages} \
        > /tmp/mamba-remove-plan.json; \
    python /tmp/check_mamba_removal.py /tmp/mamba-remove-plan.json ${packages}; \
    micromamba remove -y -n sam3d-objects ${packages}; \
    python /tmp/check_runtime_imports.py; \
    micromamba clean --all --yes; \
    rm \
        /tmp/check_cuda_build_env.py \
        /tmp/check_mamba_removal.py \
        /tmp/check_runtime_imports.py \
        /tmp/patch_offline_runtime.py \
        /tmp/mamba-remove-plan.json

FROM ubuntu:22.04 AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/micromamba/envs/sam3d-objects /opt/micromamba/envs/sam3d-objects
COPY --from=builder /opt/sam-3d-objects /opt/sam-3d-objects

COPY app /srv/app
COPY scripts/fc_initializer.sh /srv/scripts/fc_initializer.sh

ENV PATH=/opt/micromamba/envs/sam3d-objects/bin:${PATH} \
    CONDA_PREFIX=/opt/micromamba/envs/sam3d-objects \
    CUDA_HOME=/opt/micromamba/envs/sam3d-objects \
    PYTHONPATH=/srv:/opt/sam-3d-objects \
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
    PORT=9000 \
    KEEP_ALIVE_TIMEOUT=900 \
    FC_INITIALIZER_HTTP_TIMEOUT=295

WORKDIR /opt/sam-3d-objects

EXPOSE 9000

CMD ["python", "-m", "app.serve"]
