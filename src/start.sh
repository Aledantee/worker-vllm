#!/bin/bash
# stdout/stderr are kept unbuffered (python3 -u + PYTHONUNBUFFERED) and faulthandler
# is enabled (PYTHONFAULTHANDLER) so a native CUDA/driver crash at startup is visible
# in the RunPod logs. Previously the worker could exit 1 with no logs at all because
# the failure happened in native init before Python's handler logging, and any
# block-buffered output was lost when the process died.

echo "=== worker boot diagnostics ==="
python3 --version
echo "--- nvidia driver / GPU ---"
nvidia-smi || echo "WARN: nvidia-smi unavailable"
echo "--- torch + CUDA probe (surfaces driver/runtime mismatch before vLLM loads) ---"
python3 -u -c "import torch; print('torch', torch.__version__, '| built for CUDA', torch.version.cuda); print('cuda_available:', torch.cuda.is_available()); torch.zeros(1).cuda(); print('GPU alloc OK')" \
  || echo "ERROR: torch/CUDA probe failed (traceback above) — this is the worker startup failure"
echo "=== end diagnostics ==="

if [ -n "${TRANSFORMERS_VERSION}" ]; then
    echo "Installing transformers==${TRANSFORMERS_VERSION}"
    uv pip install --system "transformers==${TRANSFORMERS_VERSION}" || { echo "ERROR: transformers install failed"; exit 1; }
fi

exec python3 -u /src/handler.py
