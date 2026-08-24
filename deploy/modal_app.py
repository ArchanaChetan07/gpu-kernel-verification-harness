"""Run CRUCIBLE on Modal's serverless H100s.

Modal is the cheapest way to run this honestly: you pay per second of actual
execution, so the preflight-then-abort design in scripts/pipeline.sh costs
cents when a pod is misconfigured instead of an hour of idle rental.

    pip install modal && modal setup
    modal run deploy/modal_app.py::doctor      # ~30s, proves the image is sane
    modal run deploy/modal_app.py::smoke       # ~5 min, end to end on 6 tasks
    modal run deploy/modal_app.py::full        # the real run
    modal volume get crucible-runs / ./runs    # pull artifacts down

The GPU choice is a real decision, not a default. See GPU below.
"""

from __future__ import annotations

import pathlib
import subprocess

import modal

REPO = pathlib.Path(__file__).parent.parent
APP_DIR = "/opt/crucible"
RUNS_DIR = "/runs"

# H100 for the headline run. The reason is specific: this project's highest
# value task tier is silent numerics, and a bf16 accumulation bug only produces
# a witness on hardware where bf16 accumulation is real. On a pre-Ampere card
# those mutations are discarded as witness-free and the tasks are never made.
# A100-80GB is the cost-effective substitute; L40S works and is cheaper still.
GPU = "H100"

# torch/CUDA pinned to the development target so verdicts are comparable across
# machines. Changing this changes numerics and invalidates the tolerance
# calibration, so treat it as part of the experiment, not as packaging.
image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel")
    .apt_install("git", "build-essential", "procps")
    .env({
        # Deterministic cuBLAS reductions. Without this the same task can pass
        # and fail across runs and the derived tolerance means nothing.
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONUNBUFFERED": "1",
    })
    .pip_install_from_requirements(REPO / "requirements.txt")
    # copy=True bakes the source into the image so the pip install below can
    # run against it. (Older Modal releases spell this .copy_local_dir(); if
    # your client rejects the kwarg, that is the substitution to make.)
    .add_local_dir(
        REPO,
        remote_path=APP_DIR,
        copy=True,
        ignore=["**/__pycache__", "**/.git", "**/runs", "**/bank", "**/*.egg-info"],
    )
    .run_commands(
        f"cd {APP_DIR} && pip install --no-cache-dir -e .",
        # Fail the image build rather than a billed run: on Linux triton ships
        # with torch and must load, or O4 silently SKIPs on hardware rented
        # precisely to make it execute.
        "python -c \"import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)\"",
    )
)

app = modal.App("crucible")

# Artifacts outlive the container. A run whose verdicts died with the pod is a
# run that did not happen.
runs = modal.Volume.from_name("crucible-runs", create_if_missing=True)


def _sh(cmd: str, cwd: str = APP_DIR) -> int:
    """Stream a shell command's output back to the local terminal."""
    print(f"\n$ {cmd}", flush=True)
    proc = subprocess.run(["bash", "-lc", cmd], cwd=cwd)
    return proc.returncode


@app.function(image=image, gpu=GPU, timeout=600, volumes={RUNS_DIR: runs})
def _doctor() -> str:
    """Capability probe on the rented card. Cheap, and it settles what is real."""
    _sh("nvidia-smi")
    _sh("crucible doctor")
    return subprocess.run(
        ["bash", "-lc", "crucible doctor --json"],
        cwd=APP_DIR, capture_output=True, text=True,
    ).stdout


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes={RUNS_DIR: runs})
def _smoke() -> int:
    """Six tasks end to end. Always do this before committing to a full run."""
    rc = _sh(f"OUT={RUNS_DIR}/smoke bash scripts/pipeline.sh --smoke")
    runs.commit()
    return rc


@app.function(image=image, gpu=GPU, timeout=6 * 60 * 60, volumes={RUNS_DIR: runs})
def _full(limit: int = 200, model: str = "stub", k: int = 8) -> int:
    """The real foundry run.

    model defaults to the offline stub so no network call is made and no API
    key is required. Difficulty routing is only meaningful against the actual
    target model, so pass a real one when you have access - and know the
    thresholds must be re-fit against a proxy.
    """
    rc = _sh(
        f"OUT={RUNS_DIR}/full LIMIT={limit} MODEL={model} K={k} "
        f"RATE=2.50 bash scripts/pipeline.sh"
    )
    runs.commit()
    return rc


@app.local_entrypoint()
def doctor() -> None:
    print(_doctor.remote())


@app.local_entrypoint()
def smoke() -> None:
    rc = _smoke.remote()
    print(f"\nsmoke exit={rc}")
    print("artifacts: modal volume get crucible-runs /smoke ./runs")


@app.local_entrypoint()
def full(limit: int = 200, model: str = "stub", k: int = 8) -> None:
    rc = _full.remote(limit=limit, model=model, k=k)
    print(f"\nfull exit={rc}")
    print("artifacts: modal volume get crucible-runs /full ./runs")
