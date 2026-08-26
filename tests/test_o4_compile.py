"""Tests for O4, the compile-and-portability oracle.

The property under test throughout is the one the whole project rests on: an
unverifiable claim must never read as a verified one. Concretely that means

* every sub-check that cannot run says *why*, in words specific enough to act on
  (the triton DLL text, the jax import error, the missing C++ compiler),
* the aggregate verdict is SKIP - not PASS - when nothing could run, and
* the graph-break sub-check never reports a break count for a compilation that
  did not happen.

The graph-break path is not mocked. This machine has no inductor backend, but
dynamo traces without one, so the deliberate ``torch._dynamo.graph_break()``
below is really traced in a subprocess and really detected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from crucible.capabilities import CAP_NAMES, Capabilities
from crucible.config import Config
from crucible.oracles.base import OracleContext
from crucible.oracles.o4_compile import (
    CHECK_CAPS,
    CHECK_NAMES,
    CHECKS,
    ORACLE_ID,
    CheckResult,
    CompileOracle,
    check_graph_break,
    check_ir_diff,
    check_pallas_portability,
    check_triton_lowering,
    run_check,
    run_checks,
    select_shape,
    unified_ir_diff,
)
from crucible.schema import OracleResult, ShapeSpec, Task, TaskVerdict
from crucible.seeds.registry import SeedSpec

from .conftest import TINY_SOURCE

# --------------------------------------------------------------------------- #
# candidate sources
# --------------------------------------------------------------------------- #

#: A deliberate, unambiguous graph break. ``torch._dynamo.graph_break()`` is the
#: one construct guaranteed to break the graph across torch versions.
GRAPH_BREAK_SOURCE = '''import torch


def rowsum(x, block=32):
    acc = torch.zeros(x.shape[:-1], dtype=torch.float32, device=x.device)
    acc = acc + x.to(torch.float32).sum(dim=-1)
    torch._dynamo.graph_break()
    return acc.to(x.dtype)
'''

#: Two breaks, so a baseline that also breaks can be compared against.
DOUBLE_BREAK_SOURCE = '''import torch


def rowsum(x, block=32):
    acc = torch.zeros(x.shape[:-1], dtype=torch.float32, device=x.device)
    torch._dynamo.graph_break()
    acc = acc + x.to(torch.float32).sum(dim=-1)
    torch._dynamo.graph_break()
    return acc.to(x.dtype)
'''

UNPARSEABLE_SOURCE = '''import torch


def rowsum(x, block=32)
    return x
'''


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def make_ctx(
    *,
    task: Task,
    seed: SeedSpec | None,
    caps: Capabilities,
    cfg: Config,
    workdir: Path,
    candidate_src: str = TINY_SOURCE,
    device: str = "cpu",
    **extras: Any,
) -> OracleContext:
    workdir.mkdir(parents=True, exist_ok=True)
    return OracleContext(
        task=task,
        candidate_src=candidate_src,
        seed=seed,
        caps=caps,
        workdir=workdir,
        cfg=cfg,
        device=device,
        extras=dict(extras),
    )


#: Sub-checks that this machine (no triton, no jax, no inductor) cannot run.
UNAVAILABLE_HERE = ("triton_lowering", "pallas_portability", "ir_diff")


# --------------------------------------------------------------------------- #
# registration and wiring
# --------------------------------------------------------------------------- #


def test_oracle_registers_under_the_id_the_loader_expects() -> None:
    from crucible.oracles import ORACLE_MODULES, load_oracles

    registry = load_oracles(force=True)
    assert ORACLE_MODULES["o4_compile"] == ORACLE_ID
    assert ORACLE_ID in registry, "O4 must self-register at import time like O1 and O3"
    assert registry[ORACLE_ID].id == ORACLE_ID


def test_oracle_declares_no_top_level_caps_so_gating_is_per_check() -> None:
    oracle = CompileOracle()
    assert oracle.required_caps == ()


def test_check_registry_matches_the_declared_names() -> None:
    assert tuple(CHECKS) == CHECK_NAMES
    assert set(CHECK_CAPS) == set(CHECK_NAMES)


def test_declared_capability_names_are_real_capability_names() -> None:
    for name, caps in CHECK_CAPS.items():
        for cap in caps:
            assert cap in CAP_NAMES, f"{name} declares unknown capability {cap!r}"


def test_run_check_rejects_an_unknown_name(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(task=sample_task, seed=tiny_seed, caps=caps, cfg=cfg, workdir=tmp_workdir)
    with pytest.raises(ValueError, match="unknown compile sub-check"):
        run_check("no_such_check", ctx)


# --------------------------------------------------------------------------- #
# (a) graph breaks - genuinely executed on this machine
# --------------------------------------------------------------------------- #


def test_graph_break_detects_a_deliberate_break_here(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    """The headline case: a real dynamo trace, in a subprocess, on this machine."""
    ctx = make_ctx(
        task=sample_task,
        seed=tiny_seed,
        caps=caps,
        cfg=cfg,
        workdir=tmp_workdir / "brk",
        candidate_src=GRAPH_BREAK_SOURCE,
    )
    res = check_graph_break(ctx)

    assert res.verdict == "FAIL", res.detail
    assert res.evidence["outcome"] == "graph_breaks"
    assert res.evidence["graph_break_count"] >= 1
    assert res.evidence["baseline_graph_break_count"] == 0
    reasons = res.evidence["break_reasons"]
    assert reasons, "a detected graph break must carry its reason"
    assert any("graph_break" in r for r in reasons), reasons
    # The reason text is the deliverable, so it must reach the sub-check detail.
    assert "graph break" in res.detail
    assert reasons[0][:40] in res.detail


def test_graph_break_reports_clean_compilation_only_when_it_happened(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(
        task=sample_task,
        seed=tiny_seed,
        caps=caps,
        cfg=cfg,
        workdir=tmp_workdir / "clean",
        candidate_src=TINY_SOURCE,
    )
    res = check_graph_break(ctx)

    assert res.verdict == "PASS", res.detail
    assert res.evidence["outcome"] == "compiled_cleanly"
    assert res.evidence["graph_break_count"] == 0
    assert res.evidence["candidate"]["fullgraph_ok"] is True
    # No inductor here, so the evidence must not let a trace read as a lowering.
    assert res.evidence["backend"] == "eager"
    assert res.evidence["lowered"] is False
    assert "no kernel" in res.evidence["backend_note"]


def test_graph_break_never_reports_zero_breaks_for_a_trace_that_failed(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    """The whole invariant, at sub-check scale: absent is not zero."""
    ctx = make_ctx(
        task=sample_task,
        seed=tiny_seed,
        caps=caps,
        cfg=cfg,
        workdir=tmp_workdir / "unparseable",
        candidate_src=UNPARSEABLE_SOURCE,
    )
    res = check_graph_break(ctx)

    assert res.verdict == "SKIP"
    assert res.evidence["outcome"] == "unavailable"
    assert res.evidence["candidate"]["graph_break_count"] is None
    assert "0 graph breaks" not in res.detail
    assert "not zero" in res.detail
    # The real exception text, not a paraphrase.
    assert "SyntaxError" in res.evidence["candidate"]["error"]
    assert "SyntaxError" in res.detail


def test_graph_break_does_not_punish_a_break_the_baseline_also_has(
    make_task: Callable[..., Task],
    tiny_seed: SeedSpec,
    caps: Capabilities,
    cfg: Config,
    tmp_workdir: Path,
) -> None:
    """A break inherent to the seed is reported, not blamed on the candidate."""
    task = make_task(baseline_code=GRAPH_BREAK_SOURCE, mutant_code=GRAPH_BREAK_SOURCE)
    ctx = make_ctx(
        task=task,
        seed=tiny_seed,
        caps=caps,
        cfg=cfg,
        workdir=tmp_workdir / "equal",
        candidate_src=GRAPH_BREAK_SOURCE,
    )
    res = check_graph_break(ctx)

    assert res.verdict == "PASS", res.detail
    assert res.evidence["graph_break_count"] == res.evidence["baseline_graph_break_count"] >= 1
    # PASS, but the count is still stated - a silent pass would hide the break.
    assert "graph break" in res.detail


def test_graph_break_fails_a_candidate_that_breaks_more_than_the_baseline(
    make_task: Callable[..., Task],
    tiny_seed: SeedSpec,
    caps: Capabilities,
    cfg: Config,
    tmp_workdir: Path,
) -> None:
    task = make_task(baseline_code=GRAPH_BREAK_SOURCE, mutant_code=GRAPH_BREAK_SOURCE)
    ctx = make_ctx(
        task=task,
        seed=tiny_seed,
        caps=caps,
        cfg=cfg,
        workdir=tmp_workdir / "more",
        candidate_src=DOUBLE_BREAK_SOURCE,
    )
    res = check_graph_break(ctx)

    assert res.verdict == "FAIL", res.detail
    assert res.evidence["graph_break_count"] > res.evidence["baseline_graph_break_count"]


def test_graph_break_skips_specifically_when_there_is_no_seed(
    sample_task: Task, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(task=sample_task, seed=None, caps=caps, cfg=cfg, workdir=tmp_workdir / "noseed")
    res = check_graph_break(ctx)
    assert res.verdict == "SKIP"
    assert "no seed" in res.detail


# --------------------------------------------------------------------------- #
# (b) triton lowering / register spills
# --------------------------------------------------------------------------- #


def test_triton_lowering_skip_carries_the_capability_error_verbatim(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(task=sample_task, seed=tiny_seed, caps=caps, cfg=cfg, workdir=tmp_workdir / "tri")
    res = check_triton_lowering(ctx)

    assert res.verdict == "SKIP"
    assert caps.triton_error, "the fixture must model the real DLL failure"
    assert caps.triton_error in res.detail, "the probed error text must survive verbatim"
    assert res.evidence["triton_error"] == caps.triton_error
    assert "n_spills" not in res.evidence


def test_triton_lowering_does_not_invent_spill_counts_when_the_probe_fails(
    sample_task: Task,
    tiny_seed: SeedSpec,
    caps_full: Capabilities,
    cfg: Config,
    tmp_workdir: Path,
) -> None:
    """Even told triton works, an unloadable triton yields SKIP, never PASS.

    ``caps_full`` claims a working triton. The subprocess disagrees, because
    ``import triton`` really fails here. The counters are then ABSENT.
    """
    ctx = make_ctx(
        task=sample_task,
        seed=tiny_seed,
        caps=caps_full,
        cfg=cfg,
        workdir=tmp_workdir / "tri2",
    )
    res = check_triton_lowering(ctx)

    assert res.verdict == "SKIP", res.detail
    assert "import_triton" in res.detail
    # The probe must name the real failure, but WHICH failure is platform
    # specific: Windows has triton installed and its DLL fails to load
    # ("libtriton"), a CPU-only Linux wheel has no triton at all
    # ("No module named"). Asserting one machine's string made this test pass
    # only where it was written -- the same unverified-elsewhere claim this
    # project exists to catch. Assert the contract instead.
    assert any(tok in res.detail for tok in ("libtriton", "No module named", "ImportError")), res.detail
    assert "kernels" not in res.evidence


def test_triton_lowering_skips_when_triton_exists_but_no_cuda_device_does(
    sample_task: Task, tiny_seed: SeedSpec, cfg: Config, tmp_workdir: Path
) -> None:
    caps = Capabilities(
        cuda=False,
        cuda_device_count=0,
        device_name=None,
        cuda_version=None,
        triton=True,
        triton_error=None,
        ncu=None,
        can_lock_clocks=False,
        gloo=True,
        nccl=False,
        cxx_compiler=None,
        inductor_cpu=False,
        inductor_cuda=False,
        torch_version="2.6.0+cu124",
        platform="Windows-11-test",
        details={"cuda": "torch.cuda.is_available() returned False"},
    )
    ctx = make_ctx(task=sample_task, seed=tiny_seed, caps=caps, cfg=cfg, workdir=tmp_workdir / "tri3")
    res = check_triton_lowering(ctx)

    assert res.verdict == "SKIP"
    assert "no CUDA device" in res.detail
    assert "torch.cuda.is_available() returned False" in res.detail


# --------------------------------------------------------------------------- #
# (c) pallas portability
# --------------------------------------------------------------------------- #


def test_pallas_portability_skips_with_the_real_jax_import_error(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(task=sample_task, seed=tiny_seed, caps=caps, cfg=cfg, workdir=tmp_workdir / "pal")
    res = check_pallas_portability(ctx)

    assert res.verdict == "SKIP"
    assert "jax is not installed" in res.detail
    assert "No module named 'jax'" in res.detail
    assert res.evidence["probe"]["stage"] == "import_jax"
    assert "backends" not in res.evidence, "no backend may be reported when none was probed"


def test_pallas_portability_asks_for_every_backend_it_would_compare(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(task=sample_task, seed=tiny_seed, caps=caps, cfg=cfg, workdir=tmp_workdir / "pal2")
    res = check_pallas_portability(ctx)
    assert res.evidence["platforms_requested"] == ["tpu", "gpu", "cpu"]


# --------------------------------------------------------------------------- #
# (d) emitted IR diff
# --------------------------------------------------------------------------- #


def test_ir_diff_skips_naming_the_missing_backend(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(task=sample_task, seed=tiny_seed, caps=caps, cfg=cfg, workdir=tmp_workdir / "ir")
    res = check_ir_diff(ctx)

    assert res.verdict == "SKIP"
    assert "inductor_cpu" in res.detail
    assert caps.detail("inductor_cpu") in res.detail or "probed False" in res.detail
    assert "ir_diff" not in res.evidence, "no diff may be reported when nothing was emitted"


def test_ir_diff_reports_absence_rather_than_an_empty_diff(
    sample_task: Task,
    tiny_seed: SeedSpec,
    caps_full: Capabilities,
    cfg: Config,
    tmp_workdir: Path,
) -> None:
    """Two sides that both failed to lower are not "identical IR"."""
    ctx = make_ctx(
        task=sample_task,
        seed=tiny_seed,
        caps=caps_full,
        cfg=Config(**{**cfg.model_dump(), "sandbox_timeout_s": 180.0}),
        workdir=tmp_workdir / "ir2",
    )
    res = check_ir_diff(ctx)

    # Whether inductor can lower at all is an environment fact: a Linux runner
    # with gcc emits real output code, a box without a C compiler cannot. Both
    # are legitimate; what must hold either way is that absence is reported as
    # absence and never as an empty diff.
    if res.evidence.get("baseline_emitted") and res.evidence.get("candidate_emitted"):
        assert res.verdict in ("PASS", "FAIL"), res.detail
        assert "ir_identical" in res.evidence, "a real emission must say whether the IR matched"
        if res.evidence.get("ir_identical") is False:
            assert res.evidence.get("ir_diff"), "a differing emission must carry the diff"
    else:
        assert res.verdict == "SKIP", res.detail
        assert "ir_diff" not in res.evidence
        assert "ir_identical" not in res.evidence
        assert res.evidence["candidate_error"], "the lowering failure text must be kept"


def test_unified_ir_diff_is_empty_only_for_identical_emissions() -> None:
    code = "def call(args):\n    return args\n"
    lines, truncated = unified_ir_diff(code, code)
    assert lines == []
    assert truncated is False


def test_unified_ir_diff_shows_both_sides_of_a_change() -> None:
    base = "def call(args):\n    buf0 = empty((4, 8))\n    return buf0\n"
    cand = "def call(args):\n    buf0 = empty((4, 16))\n    return buf0\n"
    lines, truncated = unified_ir_diff(base, cand)

    assert truncated is False
    assert any(ln.startswith("-") and "(4, 8)" in ln for ln in lines)
    assert any(ln.startswith("+") and "(4, 16)" in ln for ln in lines)
    assert lines[0].startswith("--- baseline")
    assert lines[1].startswith("+++ candidate")


def test_unified_ir_diff_flags_its_own_truncation() -> None:
    base = "\n".join(f"line {i}" for i in range(500))
    cand = "\n".join(f"LINE {i}" for i in range(500))
    lines, truncated = unified_ir_diff(base, cand, max_lines=20)
    assert truncated is True
    assert len(lines) == 20


# --------------------------------------------------------------------------- #
# every sub-check that cannot run says why, specifically
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("check_name", UNAVAILABLE_HERE)
def test_each_unavailable_subcheck_skips_with_a_specific_reason(
    check_name: str,
    sample_task: Task,
    tiny_seed: SeedSpec,
    caps: Capabilities,
    cfg: Config,
    tmp_workdir: Path,
) -> None:
    ctx = make_ctx(
        task=sample_task, seed=tiny_seed, caps=caps, cfg=cfg, workdir=tmp_workdir / check_name
    )
    res = run_check(check_name, ctx)

    assert res.verdict == "SKIP", f"{check_name}: {res.detail}"
    assert res.detail.strip(), f"{check_name} skipped with an empty reason"
    assert len(res.detail) >= 40, f"{check_name} skip reason is too vague: {res.detail!r}"
    # A reason that names nothing concrete is not a reason.
    assert any(
        token in res.detail for token in ("triton", "jax", "inductor")
    ), f"{check_name} skip reason names no concrete missing component: {res.detail!r}"
    assert res.as_dict()["verdict"] == "SKIP"
    assert res.ran is False


def test_the_skip_helper_refuses_a_reasonless_skip() -> None:
    from crucible.oracles.o4_compile import _skip

    with pytest.raises(ValueError, match="without a reason"):
        _skip("graph_break", "   ", {}, 0.0)


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def test_aggregate_is_skip_not_pass_when_no_subcheck_could_run(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    """The invariant. Nothing compiled anywhere, so nothing is verified."""
    ctx = make_ctx(
        task=sample_task,
        seed=tiny_seed,
        caps=caps,
        cfg=cfg,
        workdir=tmp_workdir / "agg_skip",
        o4_checks=list(UNAVAILABLE_HERE),
    )
    res = CompileOracle().run(ctx)

    assert res.verdict == "SKIP"
    assert res.verdict != "PASS"
    assert res.reason.strip()
    assert "nothing about compilation was verified" in res.reason
    assert set(res.evidence["skipped"]) == set(UNAVAILABLE_HERE)
    assert res.evidence["passed"] == []
    assert res.evidence["ran"] == []
    # And SKIP must dominate PASS when this joins other oracles' verdicts.
    assert TaskVerdict.combine([res, OracleResult(oracle="O1", verdict="PASS")]) == "SKIP"


def test_aggregate_is_skip_when_the_context_carries_no_seed(
    sample_task: Task, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(task=sample_task, seed=None, caps=caps, cfg=cfg, workdir=tmp_workdir / "agg_noseed")
    res = CompileOracle().run(ctx)

    assert res.verdict == "SKIP"
    assert set(res.evidence["skipped"]) == set(CHECK_NAMES)
    for name, reason in res.evidence["skipped"].items():
        assert reason.strip(), f"{name} skipped with no reason"


def test_aggregate_passes_and_names_the_subcheck_that_actually_ran(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(
        task=sample_task, seed=tiny_seed, caps=caps, cfg=cfg, workdir=tmp_workdir / "agg_pass"
    )
    res = CompileOracle().run(ctx)

    assert res.verdict == "PASS", res.reason
    assert res.evidence["passed"] == ["graph_break"]
    assert res.evidence["ran"] == ["graph_break"]
    # The three that could not run are still individually accounted for.
    assert set(res.evidence["skipped"]) == set(UNAVAILABLE_HERE)
    assert res.evidence["checks"]["graph_break"]["outcome"] == "compiled_cleanly"
    assert res.evidence["checks"]["triton_lowering"]["verdict"] == "SKIP"


def test_aggregate_fails_and_names_the_subcheck_that_fired(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(
        task=sample_task,
        seed=tiny_seed,
        caps=caps,
        cfg=cfg,
        workdir=tmp_workdir / "agg_fail",
        candidate_src=GRAPH_BREAK_SOURCE,
        o4_checks=["graph_break"],
    )
    res = CompileOracle().run(ctx)

    assert res.verdict == "FAIL"
    assert res.evidence["fired"] == ["graph_break"]
    assert "graph_break" in res.reason


def test_aggregate_reports_error_when_a_subcheck_raises(
    monkeypatch: pytest.MonkeyPatch,
    sample_task: Task,
    tiny_seed: SeedSpec,
    caps: Capabilities,
    cfg: Config,
    tmp_workdir: Path,
) -> None:
    def boom(_ctx: OracleContext) -> CheckResult:
        raise RuntimeError("compiler probe exploded")

    monkeypatch.setitem(CHECKS, "graph_break", boom)
    ctx = make_ctx(
        task=sample_task,
        seed=tiny_seed,
        caps=caps,
        cfg=cfg,
        workdir=tmp_workdir / "agg_err",
        o4_checks=["graph_break"],
    )
    res = CompileOracle().run(ctx)

    assert res.verdict == "ERROR"
    assert "compiler probe exploded" in res.reason
    assert "traceback" in res.evidence["checks"]["graph_break"]


def test_oracle_result_evidence_is_json_serialisable(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    """A verdict nobody can serialise is a verdict nobody can audit."""
    ctx = make_ctx(
        task=sample_task,
        seed=tiny_seed,
        caps=caps,
        cfg=cfg,
        workdir=tmp_workdir / "agg_json",
        o4_checks=["triton_lowering", "pallas_portability"],
    )
    res = CompileOracle().run(ctx)
    text = json.dumps(res.model_dump(mode="json"))
    assert '"O4"' in text
    assert "libtriton" in text


def test_run_checks_defaults_to_every_declared_check(
    sample_task: Task, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(task=sample_task, seed=None, caps=caps, cfg=cfg, workdir=tmp_workdir / "all")
    results = run_checks(ctx)
    assert [r.name for r in results] == list(CHECK_NAMES)


# --------------------------------------------------------------------------- #
# shape selection
# --------------------------------------------------------------------------- #


def test_select_shape_prefers_the_seeds_own_sweep(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(task=sample_task, seed=tiny_seed, caps=caps, cfg=cfg, workdir=tmp_workdir)
    assert select_shape(ctx) == tiny_seed.shape_sweep[0]


def test_select_shape_falls_back_to_the_tasks_detect_shapes(
    sample_task: Task, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(task=sample_task, seed=None, caps=caps, cfg=cfg, workdir=tmp_workdir)
    assert select_shape(ctx) == sample_task.detect_shapes[0]


def test_select_shape_honours_an_explicit_override(
    sample_task: Task, tiny_seed: SeedSpec, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    forced = ShapeSpec(name="forced", kwargs={"rows": 1, "cols": 3, "dtype": "float32"})
    ctx = make_ctx(
        task=sample_task, seed=tiny_seed, caps=caps, cfg=cfg, workdir=tmp_workdir, o4_shape=forced
    )
    assert select_shape(ctx) == forced


def test_select_shape_returns_none_when_nothing_is_available(
    make_task: Callable[..., Task], caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    task = make_task(detect_shapes=[], decoy_shapes=[], witness=None, prompt="")
    ctx = make_ctx(task=task, seed=None, caps=caps, cfg=cfg, workdir=tmp_workdir)
    assert select_shape(ctx) is None
