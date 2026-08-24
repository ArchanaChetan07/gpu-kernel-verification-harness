#!/usr/bin/env bash
# One-shot bootstrap for a bare rented GPU pod (RunPod, Lambda, Vast, Paperspace,
# or any Ubuntu box with an NVIDIA driver). Docker is the more reproducible
# path; this exists for pods where you get a shell and not a registry.
#
#   curl -fsSL <raw-url>/scripts/bootstrap_pod.sh | bash
# or, having copied the repo up:
#   cd crucible && ./scripts/bootstrap_pod.sh && ./scripts/pipeline.sh --smoke
#
# Idempotent: safe to re-run after a disconnect.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
echo "CRUCIBLE bootstrap in $REPO"

# --- 1. Sanity: is there actually a GPU here? -------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: nvidia-smi not found. This is not a GPU pod, or the driver is not"
  echo "visible to this container. Nothing below is worth doing without it."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

# --- 2. Python ---------------------------------------------------------------
PY=$(command -v python3 || command -v python)
echo "python: $PY ($($PY --version 2>&1))"
$PY -m pip install --quiet --upgrade pip setuptools wheel

# --- 3. torch ----------------------------------------------------------------
# Do not touch a torch that already works. Rented images ship torch built
# against their driver; replacing it with a PyPI default-CUDA wheel is the
# classic way to turn a working pod into a broken one. The tolerance model in
# this project is calibrated per torch build, so a silent swap also invalidates
# every numeric verdict the run produces.
if $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)') - keeping it"
else
  echo "torch missing or CUDA-less; installing the cu124 build"
  $PY -m pip install --quiet torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
fi

# --- 4. Project ---------------------------------------------------------------
$PY -m pip install --quiet -r requirements.txt
$PY -m pip install --quiet -e .

# --- 5. Determinism ------------------------------------------------------------
# Needed for deterministic cuBLAS reductions; without it the same task can pass
# and fail across runs and the tolerance model means nothing.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if ! grep -q CUBLAS_WORKSPACE_CONFIG ~/.bashrc 2>/dev/null; then
  echo 'export CUBLAS_WORKSPACE_CONFIG=:4096:8' >> ~/.bashrc
fi

# --- 6. Triton ------------------------------------------------------------------
# On Linux triton ships inside the torch wheel. If it does not load here, O4's
# lowering checks will SKIP - on a box rented specifically to make them run.
# Surface that now rather than in the verdict JSON three hours later.
if $PY -c "import triton" 2>/dev/null; then
  echo "triton: $($PY -c 'import triton;print(triton.__version__)') OK"
else
  echo "WARNING: triton did not import. O4 lowering and register-spill checks"
  echo "         will SKIP and their tasks will not count as verified."
fi

# --- 7. Profiler ------------------------------------------------------------------
if command -v ncu >/dev/null 2>&1; then
  echo "ncu: $(command -v ncu)"
  echo "     note: inside a container this needs --cap-add=SYS_ADMIN, and on"
  echo "     most cloud pods it also needs the profiling-permission flag set."
else
  echo "ncu: absent. O2 will omit hardware counters rather than report zeros."
fi

chmod +x scripts/*.sh

# --- 8. Verify the install actually works ----------------------------------------
echo
echo "=== capability report ==="
crucible doctor

echo
echo "Bootstrap done. Next, in order:"
echo "  ./scripts/pipeline.sh --smoke     # ~5 min, proves the pod works end to end"
echo "  ./scripts/pipeline.sh             # full run"
echo
echo "Then pull the artifacts down before you kill the pod:"
echo "  tar czf crucible-run.tar.gz runs/"
