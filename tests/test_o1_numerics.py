"""Tests for the tolerance model and the O1 differential-numerics oracle.

The headline behaviour under test is the last group: a candidate whose only
defect is an off-by-one on the tail of a 128-wide block loop is bit-identical to
the reference for every sequence length up to and including 128, and wrong from
129 onwards. O1 must FAIL it and must name ``n129`` as ``first_break_shape`` -
not ``n1023``, which also breaks, and not the last shape in the sweep. That
shape is the grading key for the task, so getting it right is the point of the
oracle.

Everything here runs on CPU with no GPU and no network. The candidate really is
executed, in a real sandbox subprocess.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from crucible.capabilities import Capabilities
from crucible.config import Config
from crucible.oracles.base import OracleContext
from crucible.oracles.tolerance import (
    EPS,
    CompareResult,
    Tolerance,
    compare,
    derive,
    eps_for,
    fixed,
    from_tensor_scale,
    growth_factor,
    magnitude_of,
    normalize_dtype,
)
from crucible.schema import SeedRef, ShapeSpec, Task, sha256_text
from crucible.seeds.registry import SeedSpec

# --------------------------------------------------------------------------- #
# tolerance model
# --------------------------------------------------------------------------- #


def test_eps_are_the_real_unit_roundoffs() -> None:
    assert EPS["fp32"] == 2.0**-24
    assert EPS["tf32"] == 2.0**-11
    assert EPS["bf16"] == 2.0**-8
    assert EPS["fp16"] == 2.0**-11
    assert EPS["fp8"] == 2.0**-4
    # bf16 keeps 8 mantissa bits; that is the number the formula string prints.
    assert EPS["bf16"] == pytest.approx(3.90625e-3)
    # fp32's eps must match numpy's, halved (numpy reports 2**-23 spacing).
    assert EPS["fp32"] == pytest.approx(float(np.finfo(np.float32).eps) / 2.0)


def test_dtype_aliases_normalize() -> None:
    assert normalize_dtype("torch.bfloat16") == "bf16"
    assert normalize_dtype("float32") == "fp32"
    assert normalize_dtype("half") == "fp16"
    assert normalize_dtype("float8_e4m3fn") == "fp8"
    assert eps_for("float16") == EPS["fp16"]
    with pytest.raises(KeyError):
        normalize_dtype("int8")


def test_formula_shows_its_own_derivation() -> None:
    tol = derive("bf16", 1023, mode="stochastic", safety=4.0)
    # 4 * sqrt(1023) * 2**-8
    expected = 4.0 * math.sqrt(1023) * (2.0**-8)
    assert tol.rel == pytest.approx(expected)
    assert tol.rel == pytest.approx(0.4997, rel=1e-3)
    f = tol.formula
    assert "bf16" in f
    assert "eps=3.91e-03" in f
    assert "K=1023" in f
    assert "sqrt(K)=31.98" in f
    assert "safety=4" in f
    assert "rel=5.00e-01" in f
    assert tol.derived is True


def test_no_hardcoded_tolerance_anywhere_in_the_derivation() -> None:
    """Two different dtypes/depths must not land on the same number."""
    a = derive("fp32", 128)
    b = derive("bf16", 128)
    c = derive("fp32", 4096)
    assert a.rel != b.rel
    assert a.rel != c.rel
    assert a.rel == pytest.approx(4.0 * math.sqrt(128) * 2.0**-24)


@pytest.mark.parametrize("mode", ["stochastic", "deterministic"])
def test_relative_tolerance_is_monotone_in_accum_depth(mode: str) -> None:
    depths = [1, 2, 8, 127, 128, 129, 1023, 4096]
    rels = [derive("fp32", k, mode=mode).rel for k in depths]
    assert all(later > earlier for earlier, later in zip(rels, rels[1:]))


def test_deterministic_growth_bounds_stochastic_growth() -> None:
    for k in (1, 2, 128, 1023):
        assert growth_factor(k, "deterministic") >= growth_factor(k, "stochastic")
    assert growth_factor(1, "stochastic") == 1.0
    with pytest.raises(ValueError):
        growth_factor(8, "worst_case")


def test_accum_depth_below_one_is_clamped_and_noted() -> None:
    tol = derive("fp32", 0)
    assert tol.accum_depth == 1
    assert "clamped" in tol.note


def test_safety_factor_must_be_positive() -> None:
    with pytest.raises(ValueError):
        derive("fp32", 128, safety=0.0)
    with pytest.raises(ValueError):
        derive("fp32", 128, safety=float("nan"))


def test_from_tensor_scale_converts_relative_into_absolute() -> None:
    tol = derive("fp16", 256)
    ref = np.array([[1.0, -40.0], [3.0, 2.5]], dtype=np.float32)
    scaled = from_tensor_scale(tol, ref)
    assert magnitude_of(ref) == pytest.approx(40.0)
    assert scaled.rel == tol.rel
    assert scaled.abs == pytest.approx(tol.rel * 40.0)
    assert "scale=4.000e+01" in scaled.formula
    assert "max|reference|" in scaled.formula


def test_from_tensor_scale_ignores_non_finite_reference_entries() -> None:
    tol = derive("fp32", 4)
    ref = np.array([1.0, np.inf, np.nan, -7.0], dtype=np.float64)
    scaled = from_tensor_scale(tol, ref)
    assert scaled.scale == pytest.approx(7.0)


def test_zero_reference_falls_back_to_unit_scale_and_says_so() -> None:
    tol = derive("fp32", 16)
    scaled = from_tensor_scale(tol, np.zeros((3, 3), dtype=np.float32))
    assert scaled.scale == 1.0
    assert "unit fallback" in scaled.formula


def test_a_fixed_tolerance_is_recorded_as_a_decision_not_a_derivation() -> None:
    tol = fixed(1e-3, reason="legacy kernel contract")
    assert tol.derived is False
    assert tol.mode == "fixed"
    assert "FIXED by caller" in tol.formula
    assert "legacy kernel contract" in tol.formula
    assert "not derived" in tol.formula
    # And it must not be silently rescaled out from under the caller.
    same = from_tensor_scale(tol, np.array([1e6], dtype=np.float32))
    assert same.abs == tol.abs
    assert same is tol


# --------------------------------------------------------------------------- #
# compare()
# --------------------------------------------------------------------------- #


def _tol() -> Tolerance:
    return derive("fp32", 128)


def test_identical_arrays_compare_clean() -> None:
    x = np.linspace(-3.0, 3.0, 64, dtype=np.float32).reshape(8, 8)
    res = compare(x, x.copy(), _tol())
    assert isinstance(res, CompareResult)
    assert res.passed
    assert res.ok
    assert res.max_abs_err == 0.0
    assert res.max_rel_err == 0.0
    assert res.n_bad == 0
    assert res.first_bad_index is None
    assert res.n_total == 64


def test_small_perturbation_inside_the_budget_passes() -> None:
    tol = derive("fp32", 128).with_scale(1.0)
    ref = np.ones((4, 4), dtype=np.float64)
    cand = ref + tol.rel * 0.5
    assert compare(cand, ref, tol).passed


def test_perturbation_just_outside_the_budget_fails() -> None:
    tol = derive("fp32", 128).with_scale(1.0)
    ref = np.ones((4, 4), dtype=np.float64)
    cand = ref.copy()
    cand[2, 3] = 1.0 + (tol.abs + tol.rel) * 10.0
    res = compare(cand, ref, tol)
    assert not res.passed
    assert res.n_bad == 1
    assert res.first_bad_index == (2, 3)


def test_nan_in_candidate_where_reference_is_finite_always_fails() -> None:
    """Even with an absurdly loose budget: NaN is not 'within tolerance'."""
    huge = fixed(1e9, abs=1e9, reason="deliberately absurd")
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    cand = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    res = compare(cand, ref, huge)
    assert not res.passed
    assert res.n_bad == 1
    assert res.n_nan_mismatch == 1
    assert res.first_bad_index == (1,)
    assert "NaN" in res.detail
    # The reported magnitudes come only from positions that were comparable.
    assert res.n_comparable == 2
    assert res.max_abs_err == 0.0


def test_nan_in_reference_where_candidate_is_finite_also_fails() -> None:
    ref = np.array([1.0, np.nan], dtype=np.float64)
    cand = np.array([1.0, 2.0], dtype=np.float64)
    res = compare(cand, ref, fixed(1e9, abs=1e9))
    assert not res.passed
    assert res.n_nan_mismatch == 1


def test_nan_matching_nan_is_agreement() -> None:
    ref = np.array([np.nan, 1.0], dtype=np.float64)
    cand = np.array([np.nan, 1.0], dtype=np.float64)
    res = compare(cand, ref, _tol())
    assert res.passed
    assert res.n_nan_mismatch == 0
    assert res.n_comparable == 1


def test_infinities_must_match_in_sign() -> None:
    tol = fixed(1e9, abs=1e9)
    same = compare(
        np.array([np.inf, -np.inf]), np.array([np.inf, -np.inf]), tol
    )
    assert same.passed
    flipped = compare(np.array([np.inf]), np.array([-np.inf]), tol)
    assert not flipped.passed
    assert flipped.n_inf_mismatch == 1
    finite_vs_inf = compare(np.array([1.0]), np.array([np.inf]), tol)
    assert not finite_vs_inf.passed


def test_all_nan_candidate_reports_absent_magnitudes_not_zero_error() -> None:
    ref = np.array([1.0, 2.0], dtype=np.float64)
    cand = np.array([np.nan, np.nan], dtype=np.float64)
    res = compare(cand, ref, fixed(1e9, abs=1e9))
    assert not res.passed
    assert res.n_comparable == 0
    assert "not defined" in res.detail


def test_shape_mismatch_is_a_structural_failure() -> None:
    res = compare(np.zeros((2, 3)), np.zeros((3, 2)), _tol())
    assert not res.passed
    assert res.kind == "shape"
    assert "shape" in res.detail


def test_zero_reference_entries_use_the_absolute_budget_only() -> None:
    tol = fixed(0.5, abs=0.1)
    ref = np.array([0.0, 0.0], dtype=np.float64)
    ok = compare(np.array([0.05, -0.05]), ref, tol)
    assert ok.passed
    assert ok.n_zero_ref == 2
    assert ok.max_rel_err == 0.0  # undefined at ref==0, reported as 0, counted separately
    bad = compare(np.array([0.5, 0.0]), ref, tol)
    assert not bad.passed


def test_integer_outputs_are_compared_exactly() -> None:
    tol = fixed(1e9, abs=1e9)
    res = compare(np.array([1, 2, 3]), np.array([1, 2, 4]), tol)
    assert not res.passed
    assert res.kind == "exact"
    assert res.first_bad_index == (2,)


def test_compare_result_as_dict_is_json_safe() -> None:
    import json

    res = compare(np.array([1.0, 5.0]), np.array([1.0, 1.0]), _tol())
    payload = res.as_dict()
    text = json.dumps(payload)
    assert "Infinity" not in text and "NaN" not in text
    assert payload["first_bad_index"] == [1]
    assert payload["tolerance"]["formula"]


# --------------------------------------------------------------------------- #
# O1 fixture: a 128-wide block loop with a tail that only breaks at 129
# --------------------------------------------------------------------------- #

BLOCK = 128

#: Candidate that computes exactly what the reference computes.
EXACT_SOURCE = '''import torch


def blocksum(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).sum(dim=-1).to(x.dtype)
'''

#: Correct blocked implementation: same answer, different summation order.
GOOD_SOURCE = '''import torch

BLOCK = 128


def blocksum(x: torch.Tensor) -> torch.Tensor:
    x32 = x.to(torch.float32)
    n = x32.shape[-1]
    acc = x32[..., :BLOCK].sum(dim=-1)
    start = BLOCK
    while start < n:
        stop = min(start + BLOCK, n)
        acc = acc + x32[..., start:stop].sum(dim=-1)
        start += BLOCK
    return acc.to(x.dtype)
'''

#: The same code with one character changed: the continuation blocks drop their
#: last element. Sequence lengths <= 128 never enter the loop, so the bug is
#: invisible there and first becomes observable at 129.
BUGGY_SOURCE = GOOD_SOURCE.replace("x32[..., start:stop]", "x32[..., start:stop - 1]")

#: A candidate that raises unconditionally.
RAISING_SOURCE = '''import torch


def blocksum(x: torch.Tensor) -> torch.Tensor:
    raise ValueError("candidate is not implemented for this layout")
'''

#: A candidate that returns NaN on long sequences only.
NAN_SOURCE = '''import torch


def blocksum(x: torch.Tensor) -> torch.Tensor:
    out = x.to(torch.float32).sum(dim=-1)
    if x.shape[-1] >= 129:
        out = out * float("nan")
    return out.to(x.dtype)
'''

BLOCK_SHAPES = [
    ShapeSpec(name="n1", kwargs={"rows": 2, "cols": 1, "dtype": "float32"}),
    ShapeSpec(name="n127", kwargs={"rows": 2, "cols": 127, "dtype": "float32"}),
    ShapeSpec(name="n128", kwargs={"rows": 2, "cols": 128, "dtype": "float32"}),
    ShapeSpec(name="n129", kwargs={"rows": 2, "cols": 129, "dtype": "float32"}),
    ShapeSpec(name="n1023", kwargs={"rows": 2, "cols": 1023, "dtype": "float32"}),
]


def _block_make_inputs(shape: ShapeSpec, device: Any = "cpu", generator: Any = None) -> dict[str, Any]:
    import torch

    dtype = getattr(torch, str(shape.kwargs.get("dtype", "float32")))
    rows = int(shape.kwargs["rows"])
    cols = int(shape.kwargs["cols"])
    x = torch.randn(rows, cols, generator=generator, device=device, dtype=torch.float32)
    return {"x": x.to(dtype)}


def _block_reference(x: Any) -> Any:
    import torch

    return torch.sum(x.to(torch.float32), dim=-1).to(x.dtype)


@pytest.fixture
def block_seed() -> SeedSpec:
    """Inline seed: blocked reduction over the last dim, BLOCK=128."""
    return SeedSpec(
        id="test.blocksum",
        domain="pytorch",
        tiers=("T5",),
        description="Blocked last-dim reduction with a 128-wide block (test fixture).",
        entry="blocksum",
        source=GOOD_SOURCE,
        make_inputs=_block_make_inputs,
        reference=_block_reference,
        shape_sweep=list(BLOCK_SHAPES),
        accum_depth=lambda s: int(s.kwargs["cols"]),
        bytes_moved=lambda s: int(s.kwargs["rows"]) * int(s.kwargs["cols"]) * 4,
        flops=lambda s: int(s.kwargs["rows"]) * int(s.kwargs["cols"]),
        supports_cpu=True,
        module="tests.test_o1_numerics",
    )


@pytest.fixture
def block_task(make_task: Callable[..., Task], block_seed: SeedSpec) -> Task:
    return make_task(
        task_id="mut-blocksum-boundary_mask-0f0f",
        seed_source=SeedRef(
            seed_id=block_seed.id,
            module=block_seed.module,
            entry=block_seed.entry,
            content_sha256=sha256_text(block_seed.source),
        ),
        baseline_code=GOOD_SOURCE,
        mutant_code=BUGGY_SOURCE,
        ground_truth_diff="--- mutant\n+++ baseline\n@@\n-start:stop - 1\n+start:stop\n",
        witness=None,
        detect_shapes=[],
        decoy_shapes=[],
        prompt="",
        oracles=["O1"],
        calibration=None,
    )


@pytest.fixture
def make_ctx(
    block_task: Task,
    block_seed: SeedSpec,
    caps: Capabilities,
    cfg: Config,
    tmp_workdir: Path,
) -> Callable[[str], OracleContext]:
    def _make(candidate_src: str, **overrides: Any) -> OracleContext:
        kwargs: dict[str, Any] = dict(
            task=block_task,
            candidate_src=candidate_src,
            seed=block_seed,
            caps=caps,
            workdir=tmp_workdir,
            cfg=cfg,
            rng_seed=4242,
            device="cpu",
        )
        kwargs.update(overrides)
        return OracleContext(**kwargs)

    return _make


def _o1() -> Any:
    from crucible.oracles.o1_numerics import O1

    return O1


# --------------------------------------------------------------------------- #
# O1 behaviour
# --------------------------------------------------------------------------- #


def test_o1_is_registered_with_no_required_caps() -> None:
    o1 = _o1()
    assert o1.id == "O1"
    assert o1.required_caps == ()

    from crucible.oracles.base import ORACLES, _ensure_loaded

    _ensure_loaded()
    assert ORACLES.get("O1") is o1


def test_o1_sweep_is_the_seed_order_then_task_shapes(
    make_ctx: Callable[..., OracleContext], block_task: Task
) -> None:
    o1 = _o1()
    extra = ShapeSpec(name="n2048", kwargs={"rows": 1, "cols": 2048, "dtype": "float32"})
    task = block_task.model_copy(update={"detect_shapes": [BLOCK_SHAPES[3], extra]})
    ctx = make_ctx(GOOD_SOURCE, task=task)
    names = [s.name for s in o1.sweep(ctx)]
    assert names == ["n1", "n127", "n128", "n129", "n1023", "n2048"]


def test_identical_candidate_passes_every_shape(make_ctx: Callable[..., OracleContext]) -> None:
    res = _o1().run(make_ctx(EXACT_SOURCE))
    assert res.verdict == "PASS", res.reason
    assert res.evidence["first_break_shape"] is None
    assert res.evidence["summary"]["n_shapes"] == len(BLOCK_SHAPES)
    assert res.evidence["summary"]["n_passed"] == len(BLOCK_SHAPES)
    assert res.evidence["summary"]["n_not_run"] == 0
    assert res.evidence["max_abs_err"] == 0.0
    # Every shape carries its own derived tolerance and the derivation string.
    for rec in res.evidence["per_shape"]:
        assert rec["tolerance"]["derived"] is True
        assert f"K={rec['kwargs']['cols']}" in rec["tolerance_formula"]
        assert "fp32 eps=5.96e-08" in rec["tolerance_formula"]
    # And the candidate really ran out of process.
    assert "subprocess sandbox" in res.evidence["candidate_execution"]


def test_correct_blocked_candidate_passes_despite_reordering(
    make_ctx: Callable[..., OracleContext],
) -> None:
    """A different summation order is not an error; the budget must absorb it."""
    res = _o1().run(make_ctx(GOOD_SOURCE))
    assert res.verdict == "PASS", res.reason
    # The reordering does perturb the result, just far below the budget.
    assert res.evidence["max_abs_err"] < 1e-3


def test_off_by_one_on_the_tail_block_first_breaks_at_129(
    make_ctx: Callable[..., OracleContext],
) -> None:
    """THE headline: the bug is invisible up to 128 and n129 is the grading key."""
    res = _o1().run(make_ctx(BUGGY_SOURCE))
    assert res.verdict == "FAIL", res.reason

    assert res.evidence["first_break_shape"] == "n129"
    assert "n129" in res.reason

    status = {r["name"]: r["status"] for r in res.evidence["per_shape"]}
    assert status["n1"] == "pass"
    assert status["n127"] == "pass"
    assert status["n128"] == "pass"
    assert status["n129"] == "fail"
    assert status["n1023"] == "fail"

    broken = res.evidence["first_break"]
    assert broken["name"] == "n129"
    assert broken["max_abs_err"] > broken["tolerance"]["abs"]
    assert broken["n_bad"] >= 1
    # The failure reason quotes the derivation, not a bare number.
    assert "K=129" in broken["reason"]
    assert "sqrt(K)" in broken["reason"]
    assert res.evidence["summary"]["n_not_run"] == 0


def test_a_raising_candidate_is_a_fail_with_the_exception_in_evidence(
    make_ctx: Callable[..., OracleContext],
) -> None:
    res = _o1().run(make_ctx(RAISING_SOURCE))
    assert res.verdict == "FAIL", res.reason
    assert res.verdict != "ERROR"
    assert res.evidence["first_break_shape"] == "n1"
    exc = res.evidence["first_break"]["exception"]
    assert exc["where"] == "candidate"
    assert "ValueError" in exc["message"] or "ValueError" in exc["traceback"]
    assert "not implemented for this layout" in exc["message"]


def test_a_candidate_that_returns_nan_fails_at_the_first_nan_shape(
    make_ctx: Callable[..., OracleContext],
) -> None:
    res = _o1().run(make_ctx(NAN_SOURCE))
    assert res.verdict == "FAIL", res.reason
    assert res.evidence["first_break_shape"] == "n129"
    per_output = res.evidence["first_break"]["per_output"][0]
    assert per_output["n_nan_mismatch"] > 0


def test_a_candidate_with_the_wrong_entry_point_fails(
    make_ctx: Callable[..., OracleContext],
) -> None:
    res = _o1().run(make_ctx("import torch\n\n\ndef other(x):\n    return x\n"))
    assert res.verdict == "FAIL", res.reason
    assert "missing_entry" in res.evidence["first_break"]["exception"]["type"]


def test_a_shape_that_cannot_run_makes_the_whole_oracle_skip(
    make_ctx: Callable[..., OracleContext], block_seed: SeedSpec
) -> None:
    """A passing subset is not a pass."""
    bad_shape = ShapeSpec(name="n_oom", kwargs={"rows": 2, "cols": 8, "dtype": "float32"})

    def _boom(shape: ShapeSpec, device: Any = "cpu", generator: Any = None) -> dict[str, Any]:
        if shape.name == "n_oom":
            raise RuntimeError("CUDA out of memory. Tried to allocate 40.00 GiB")
        return _block_make_inputs(shape, device=device, generator=generator)

    block_seed.make_inputs = _boom
    block_seed.shape_sweep = [BLOCK_SHAPES[0], bad_shape]

    res = _o1().run(make_ctx(EXACT_SOURCE))
    assert res.verdict == "SKIP", res.reason
    assert "out of memory" in res.reason.lower()
    assert [s["name"] for s in res.evidence["shapes_not_run"]] == ["n_oom"]
    assert res.evidence["summary"]["n_passed"] == 1
    assert res.evidence["summary"]["n_not_run"] == 1
    assert res.evidence["first_break_shape"] is None


def test_a_real_failure_outranks_an_unrunnable_shape(
    make_ctx: Callable[..., OracleContext], block_seed: SeedSpec
) -> None:
    bad_shape = ShapeSpec(name="n_oom", kwargs={"rows": 2, "cols": 8, "dtype": "float32"})

    def _boom(shape: ShapeSpec, device: Any = "cpu", generator: Any = None) -> dict[str, Any]:
        if shape.name == "n_oom":
            raise RuntimeError("CUDA out of memory. Tried to allocate 40.00 GiB")
        return _block_make_inputs(shape, device=device, generator=generator)

    block_seed.make_inputs = _boom
    block_seed.shape_sweep = [BLOCK_SHAPES[3], bad_shape]

    res = _o1().run(make_ctx(BUGGY_SOURCE))
    assert res.verdict == "FAIL", res.reason
    assert res.evidence["first_break_shape"] == "n129"
    assert res.evidence["summary"]["n_not_run"] == 1


def test_cuda_device_without_a_cuda_capability_skips_with_a_real_reason(
    make_ctx: Callable[..., OracleContext],
) -> None:
    res = _o1().run(make_ctx(EXACT_SOURCE, device="cuda"))
    assert res.verdict == "SKIP"
    assert "cuda" in res.reason
    assert "is_available" in res.reason or "no CUDA device" in res.reason


def test_empty_sweep_skips_rather_than_passing(
    make_ctx: Callable[..., OracleContext], block_seed: SeedSpec
) -> None:
    block_seed.shape_sweep = []
    res = _o1().run(make_ctx(EXACT_SOURCE))
    assert res.verdict == "SKIP"
    assert "no shapes" in res.reason


def test_evidence_omits_error_magnitudes_when_nothing_was_compared(
    make_ctx: Callable[..., OracleContext], block_seed: SeedSpec
) -> None:
    """Absent counters are absent, not zero."""
    block_seed.shape_sweep = [BLOCK_SHAPES[0]]
    block_seed.reference = lambda **_kw: (_ for _ in ()).throw(RuntimeError("reference is broken"))
    res = _o1().run(make_ctx(EXACT_SOURCE))
    assert res.verdict == "SKIP"
    assert "max_abs_err" not in res.evidence
    assert "max_rel_err" not in res.evidence


def test_a_fixed_tolerance_override_is_carried_into_the_evidence(
    make_ctx: Callable[..., OracleContext], block_seed: SeedSpec
) -> None:
    block_seed.shape_sweep = [BLOCK_SHAPES[0]]
    ctx = make_ctx(EXACT_SOURCE)
    ctx.extras["fixed_tolerance"] = {"rel": 1e-3, "reason": "caller insisted"}
    res = _o1().run(ctx)
    assert res.verdict == "PASS", res.reason
    tol = res.evidence["per_shape"][0]["tolerance"]
    assert tol["derived"] is False
    assert tol["rel"] == 1e-3
    assert "FIXED by caller" in tol["formula"]
    assert "caller insisted" in tol["formula"]


def test_oracle_result_evidence_is_json_serialisable(
    make_ctx: Callable[..., OracleContext], block_seed: SeedSpec
) -> None:
    import json

    block_seed.shape_sweep = [BLOCK_SHAPES[0], BLOCK_SHAPES[3]]
    res = _o1().run(make_ctx(BUGGY_SOURCE))
    text = json.dumps(res.model_dump(mode="json"))
    assert "Infinity" not in text
    assert "n129" in text
