"""Shared fixtures.

Nothing here probes the real machine and nothing here imports a real seed
module: a test that depends on this machine having a working GPU is a test that
reports something other than the code under test. The synthetic seed is a
complete, executable ``SeedSpec`` so any module's tests can drive the sandbox,
the witness search or an oracle without waiting for the real seeds to land.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from crucible.capabilities import Capabilities
from crucible.config import Config
from crucible.schema import (
    CalibrationRecord,
    MutationSpec,
    RubricCriterion,
    RubricSpec,
    SeedRef,
    ShapeSpec,
    Task,
    Witness,
    sha256_text,
)
from crucible.seeds.registry import CompareResult, SeedSpec

# A blocked row-sum. Structurally it carries the mutation sites the seeds are
# supposed to carry: an explicit block loop, an explicit accumulator dtype, an
# explicit empty-block guard and an explicit final store.
TINY_SOURCE = '''import torch


def rowsum(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Sum the last dimension in blocks, accumulating in fp32."""
    n = x.shape[-1]
    acc = torch.zeros(x.shape[:-1], dtype=torch.float32, device=x.device)
    start = 0
    while start < n:
        stop = min(start + block, n)
        if stop > start:                      # empty-block guard
            chunk = x[..., start:stop].to(torch.float32)
            acc = acc + chunk.sum(dim=-1)
        start += block
    out = acc.to(x.dtype)                     # final store
    return out
'''

TINY_SHAPES = [
    ShapeSpec(name="r4_c1", kwargs={"rows": 4, "cols": 1, "dtype": "float32"}),
    ShapeSpec(name="r2_c31", kwargs={"rows": 2, "cols": 31, "dtype": "float32"}),
    ShapeSpec(name="r3_c32", kwargs={"rows": 3, "cols": 32, "dtype": "float32"}),
    ShapeSpec(name="r3_c33", kwargs={"rows": 3, "cols": 33, "dtype": "float32"}),
    ShapeSpec(name="r2_c127", kwargs={"rows": 2, "cols": 127, "dtype": "float32"}),
]


def _tiny_make_inputs(shape: ShapeSpec, device: Any = "cpu", generator: Any = None) -> dict[str, Any]:
    import torch

    dtype = getattr(torch, str(shape.kwargs.get("dtype", "float32")))
    rows = int(shape.kwargs["rows"])
    cols = int(shape.kwargs["cols"])
    x = torch.randn(rows, cols, generator=generator, device=device, dtype=torch.float32)
    return {"x": x.to(dtype)}


def _tiny_reference(x: Any) -> Any:
    # Deliberately a different code path from the blocked baseline: one fused
    # reduction, no blocking, so a shared bug cannot cancel out.
    import torch

    return torch.sum(x.to(torch.float32), dim=-1).to(x.dtype)


def _tiny_compare(got: Any, want: Any) -> CompareResult:
    import torch

    g = torch.as_tensor(got).to(torch.float32)
    w = torch.as_tensor(want).to(torch.float32)
    if g.shape != w.shape:
        return CompareResult(ok=False, detail=f"shape {tuple(g.shape)} != {tuple(w.shape)}", kind="shape")
    abs_err = (g - w).abs()
    rel_err = abs_err / w.abs().clamp_min(1e-12)
    max_abs = float(abs_err.max()) if abs_err.numel() else 0.0
    max_rel = float(rel_err.max()) if rel_err.numel() else 0.0
    return CompareResult(ok=max_rel <= 1e-5, max_abs_err=max_abs, max_rel_err=max_rel)


@pytest.fixture
def tiny_seed() -> SeedSpec:
    """A small, real, CPU-executable seed usable by any test module."""
    return SeedSpec(
        id="synthetic.rowsum",
        domain="pytorch",
        tiers=("T2", "T5"),
        description="Blocked row-sum with an fp32 accumulator (test fixture).",
        entry="rowsum",
        source=TINY_SOURCE,
        make_inputs=_tiny_make_inputs,
        reference=_tiny_reference,
        shape_sweep=list(TINY_SHAPES),
        accum_depth=lambda s: int(s.kwargs["cols"]),
        bytes_moved=lambda s: int(s.kwargs["rows"]) * int(s.kwargs["cols"]) * 4,
        flops=lambda s: int(s.kwargs["rows"]) * int(s.kwargs["cols"]),
        denylist=("torch.sum", "numpy"),
        compare=_tiny_compare,
        supports_cpu=True,
        module="tests.conftest",
    )


@pytest.fixture
def caps() -> Capabilities:
    """A CPU-only machine, with the honest reasons a real probe would record."""
    return Capabilities(
        cuda=False,
        cuda_device_count=0,
        device_name=None,
        cuda_version=None,
        triton=False,
        triton_error="ImportError: DLL load failed while importing libtriton",
        ncu=None,
        can_lock_clocks=False,
        gloo=True,
        nccl=False,
        cxx_compiler=None,
        inductor_cpu=False,
        inductor_cuda=False,
        torch_version="2.6.0+cu124",
        platform="Windows-11-test",
        details={
            "cuda": "torch.cuda.is_available() returned False",
            "triton": "ImportError: DLL load failed while importing libtriton",
            "ncu": "ncu not on PATH",
            "nccl": "torch.distributed.is_nccl_available() returned False",
            "can_lock_clocks": "nvidia-smi not on PATH",
        },
    )


@pytest.fixture
def caps_full() -> Capabilities:
    """A machine where everything works; used to test the non-SKIP path."""
    return Capabilities(
        cuda=True,
        cuda_device_count=1,
        device_name="NVIDIA T1000 8GB",
        cuda_version="12.4",
        triton=True,
        triton_error=None,
        ncu=r"C:\ncu.bat",
        can_lock_clocks=True,
        gloo=True,
        nccl=True,
        cxx_compiler=r"C:\cl.exe",
        inductor_cpu=True,
        inductor_cuda=True,
        torch_version="2.6.0+cu124",
        platform="Windows-11-test",
        details={},
    )


@pytest.fixture
def cfg() -> Config:
    """Fast defaults so a test never waits on a production-sized budget."""
    return Config(
        oracle_timeouts_s={"O1": 30.0, "O2": 30.0, "O3": 30.0, "O4": 30.0, "O5": 60.0},
        default_oracle_timeout_s=30.0,
        sandbox_timeout_s=60.0,
        perf_reps=3,
        perf_warmup=1,
        bootstrap_resamples=50,
        witness_max_shapes=5,
    )


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    d = tmp_path / "work"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def make_task(tiny_seed: SeedSpec) -> Callable[..., Task]:
    """Factory for a fully populated, schema-valid Task."""

    def _make(**overrides: Any) -> Task:
        base: dict[str, Any] = dict(
            task_id="mut-rowsum-boundary_mask-a1b2",
            created_utc=datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(timespec="seconds"),
            seed_source=SeedRef(
                seed_id=tiny_seed.id,
                module=tiny_seed.module or "tests.conftest",
                entry=tiny_seed.entry,
                content_sha256=sha256_text(tiny_seed.source),
            ),
            domain=tiny_seed.domain,
            failure_tier="T5",
            mutation=MutationSpec(
                cls="boundary_mask",
                site="conftest.py:12",
                params={"op": "lt_to_le"},
                description="off-by-one on the tail block guard",
            ),
            baseline_code=TINY_SOURCE,
            mutant_code=TINY_SOURCE.replace("if stop > start:", "if stop >= start:"),
            ground_truth_diff="--- mutant\n+++ baseline\n@@\n-    if stop >= start:\n+    if stop > start:\n",
            witness=Witness(
                shape=TINY_SHAPES[3],
                max_abs_err=0.031,
                max_rel_err=0.0042,
                tolerance=1e-05,
                baseline_checksum="a" * 64,
                mutant_checksum="b" * 64,
                kind="numeric",
                detail="tail block of 1 element double-counted",
            ),
            detect_shapes=[TINY_SHAPES[3], TINY_SHAPES[4]],
            decoy_shapes=[TINY_SHAPES[2]],
            prompt="Fix rowsum. It is verified on r3_c32.",
            oracles=["O1", "O3"],
            rubric=RubricSpec(
                criteria=[
                    RubricCriterion(
                        id="numerics",
                        weight=3.0,
                        anchors={0: "wrong", 3: "within tolerance", 5: "within tolerance, derived"},
                        auto_probe="oracle.O1.max_rel_err",
                        probe_thresholds=[(1e-6, 5), (1e-4, 3), (1.0, 0)],
                        machine_probed=True,
                    ),
                    RubricCriterion(
                        id="explanation",
                        weight=1.0,
                        anchors={0: "absent", 3: "names the site", 5: "derives the failure"},
                    ),
                ]
            ),
            calibration=CalibrationRecord(
                model="stub",
                k=8,
                n_samples=16,
                n_correct=4,
                pass_at_1=0.25,
                pass_at_k=0.75,
                route="gold",
                rationale="pass@1 in the gold band",
                trace_stats={"mean_tokens": 812.5},
            ),
            provenance={"generator": "crucible 0.1.0", "host": "test"},
        )
        base.update(overrides)
        return Task(**base)

    return _make


@pytest.fixture
def sample_task(make_task: Callable[..., Task]) -> Task:
    return make_task()
