# CUDA 12.9 base to match vLLM 0.23.0's default prebuilt wheel (cu129). CUDA 13.0
# required NVIDIA driver r580+, which is newer than the drivers on most RunPod hosts,
# so the cu13 stack failed CUDA init at startup (worker exited 1 with no logs). CUDA
# 12.9's driver floor is ~r525+, which the RunPod fleet provides.
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y python3-pip curl git \
    && curl -LsSf https://astral.sh/uv/install.sh  | sh

ENV PATH="/root/.local/bin:$PATH"

RUN ldconfig /usr/local/cuda-12.9/compat/

# vLLM 0.23.0's own wheel is compiled for CUDA 12.9, but `torch` is resolved from PyPI, whose
# default wheel is now CUDA 13.0 (cu130) — which needs NVIDIA driver r580+ (see above) and so
# crashes on RunPod's 12.9-driver hosts at engine init with "NVIDIA driver is too old (found
# version 12090)". UV_TORCH_BACKEND pins torch (and the torchvision/triton family) to the cu129
# index for every uv install below. The builder has no GPU to auto-detect against, so the backend
# is set explicitly rather than =auto.
ENV UV_TORCH_BACKEND=cu129

# Install vLLM with FlashInfer (vLLM 0.23.0's CUDA 12.9 wheel; torch pinned to cu129 above)
RUN uv pip install --system "packaging>=24.2" && \
    uv pip install --system "vllm[flashinfer]==0.23.0" && \
    uv pip install --system git+https://github.com/deepseek-ai/DeepGEMM.git@714dd1a4a980f7937a74343d19a8eba4fe321480 --no-build-isolation

# Install additional Python dependencies (after vLLM to avoid PyTorch version conflicts).
# --excludes drops nixl-cu13 (pulled transitively by lmcache's generic `nixl` meta-package)
# so only the CUDA-12 NIXL backend is installed, matching torch's CUDA 12.9 build. See
# builder/excludes.txt for details.
COPY builder/requirements.txt /requirements.txt
COPY builder/excludes.txt /excludes.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --excludes /excludes.txt -r /requirements.txt

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""
ARG VLLM_NIGHTLY="false"

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    # Suppress Ray metrics agent warnings (not needed in containerized environments)
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    # Prevent rayon thread pool panic in containers where ulimit -u < nproc
    # (tokenizers uses Rust's rayon which tries to spawn threads = CPU cores)
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4 \
    # Disable DeepGEMM MoE kernels by default; override with VLLM_USE_DEEP_GEMM=1 to enable
    VLLM_USE_DEEP_GEMM=0

ENV PYTHONPATH="/:/vllm-workspace"

# Make startup failures visible. PYTHONUNBUFFERED flushes stdout/stderr immediately
# and PYTHONFAULTHANDLER dumps a Python traceback on native fatal signals (SIGSEGV/
# SIGABRT from a CUDA/driver crash). Without these, a native init failure exits the
# worker with code 1 and no logs because block-buffered output is lost on hard exit.
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

RUN if [ "${VLLM_NIGHTLY}" = "true" ]; then \
    uv pip install --system -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly && \
    apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* && \
    uv pip install --system git+https://github.com/huggingface/transformers.git; \
fi

COPY src /src
RUN chmod +x /src/start.sh
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["/bin/bash", "/src/start.sh"]
