# CRUCIBLE — reproducible image for rented-GPU runs (H100 / A100 / L40S).
#
# Why this base: it pins the exact torch/CUDA pair the project was developed
# against (2.6.0 + cu124), so a verdict produced on a rented pod is comparable
# to one produced locally. Bumping torch changes numerics and invalidates the
# tolerance calibration, so the pin is load-bearing, not cosmetic.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Deterministic cuBLAS reductions. O1 compares against a tolerance derived
    # from an error budget; a nondeterministic reduction order would make the
    # same task pass and fail across runs.
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    CRUCIBLE_HOME=/opt/crucible

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
        curl \
        procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/crucible

# Dependency layer first so source edits do not invalidate the pip cache.
COPY pyproject.toml ./
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY crucible ./crucible
COPY tests ./tests
COPY docs ./docs
COPY scripts ./scripts
COPY README.md ./
RUN pip install --no-cache-dir -e "." && chmod +x scripts/*.sh

# Fail at build time, not three hours into a billed run.
#
# Triton is the component that is broken on the author's Windows box
# (libtriton DLL load failure). On Linux it ships with torch and must load, or
# O4's lowering checks would silently SKIP on hardware that was rented
# specifically to make them run. That would be the exact failure this project
# exists to prevent, so it is a hard build gate.
RUN python - <<'PY'
import torch, triton
k = triton.compiler  # noqa: F401  - import path must resolve, not just the package
print("torch", torch.__version__, "triton", triton.__version__)
PY

# Nsight Compute supplies O2's hardware counters (achieved occupancy, DRAM
# throughput, spill loads/stores). It is optional: when absent, O2 omits the
# counter block rather than reporting zeros. Most rented-GPU images already
# carry it; uncomment to install explicitly.
#
# RUN apt-get update && apt-get install -y --no-install-recommends cuda-nsight-compute-12-4 \
#     && rm -rf /var/lib/apt/lists/*
#
# NOTE: profiling inside a container additionally needs
#   docker run --cap-add=SYS_ADMIN
# without it ncu fails with ERR_NVGPUCTRPERM and O2 records that reason.

ENTRYPOINT ["crucible"]
CMD ["doctor"]
