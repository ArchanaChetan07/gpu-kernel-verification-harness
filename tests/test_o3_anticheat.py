"""Tests for O3, the anti-cheat oracle.

Every cheating candidate here is hand-written and executed for real, and each is
asserted against the check that is *supposed* to catch it: an attack caught by
the wrong check is a coincidence, not a grader. The last group matters just as
much - a legitimate solution must clear all five checks, because a grader that
fails honest work is as useless as one that passes dishonest work.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from crucible.oracles.base import OracleContext
from crucible.oracles.o3_anticheat import (
    CHECK_NAMES,
    CHECKS,
    AntiCheatOracle,
    analyze_source,
    check_held_out_shapes,
    check_output_liveness,
    check_randomized_inputs,
    check_static_denylist,
    check_timing_sanity,
    poisoned_source,
    run_check,
)
from crucible.schema import Task
from crucible.seeds.registry import SeedSpec

from .conftest import TINY_SOURCE

# --------------------------------------------------------------------------- #
# cheating candidates
# --------------------------------------------------------------------------- #

CHEAT_ALIAS_IMPORT = '''import torch
from torch import sum as _fast


def rowsum(x, block=32):
    return _fast(x.to(torch.float32), dim=-1).to(x.dtype)
'''

CHEAT_MODULE_ALIAS = '''import torch as t


def rowsum(x, block=32):
    return t.sum(x.to(t.float32), dim=-1).to(x.dtype)
'''

CHEAT_GETATTR_FOLDED = '''import torch as t


def rowsum(x, block=32):
    fn = getattr(t, "su" + "m")
    return fn(x.to(t.float32), dim=-1).to(x.dtype)
'''

CHEAT_GETATTR_OPAQUE = '''import torch


def rowsum(x, block=32, name="sum"):
    fn = getattr(torch, name)
    return fn(x.to(torch.float32), dim=-1).to(x.dtype)
'''

CHEAT_DUNDER_IMPORT = '''def rowsum(x, block=32):
    return __import__("numpy").sum(x.numpy(), axis=-1)
'''

CHEAT_STAR_IMPORT = '''from torch import *


def rowsum(x, block=32):
    return zeros(x.shape[:-1])
'''

# Returns a constant regardless of the input: the classic memorised-benchmark
# answer. Each sandbox run is a fresh process, so the cache is not what betrays
# it - the input-independent output is.
CHEAT_MEMOIZED = '''import torch

_CACHE = {}


def rowsum(x, block=32):
    key = tuple(x.shape)
    if key not in _CACHE:
        _CACHE[key] = torch.full(x.shape[:-1], 3.5, dtype=x.dtype)
    return _CACHE[key]
'''

# Correct only when the column count is exactly the block it hardcoded.
CHEAT_SHAPE_HARDCODE = '''import torch

BLOCK_N = 32


def rowsum(x, block=32):
    acc = torch.zeros(x.shape[:-1], dtype=torch.float32, device=x.device)
    for start in range(0, BLOCK_N, BLOCK_N):
        acc = acc + x[..., start:start + BLOCK_N].to(torch.float32).sum(dim=-1)
    return acc.to(x.dtype)
'''

CHEAT_IDENTITY = '''def rowsum(x, block=32):
    return x
'''

CHEAT_UNINITIALISED = '''import torch


def rowsum(x, block=32):
    out = torch.empty(x.shape[:-1], dtype=x.dtype, device=x.device)
    return out
'''


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def ctx_factory(
    tiny_seed: SeedSpec,
    caps: Any,
    cfg: Any,
    tmp_workdir: Path,
    make_task: Callable[..., Task],
) -> Callable[..., OracleContext]:
    """Build an OracleContext around a candidate source, reusing the shared fixtures."""

    def _make(
        candidate_src: str,
        *,
        task: Task | None = None,
        extras: dict[str, Any] | None = None,
    ) -> OracleContext:
        return OracleContext(
            task=task if task is not None else make_task(),
            candidate_src=candidate_src,
            seed=tiny_seed,
            caps=caps,
            workdir=tmp_workdir,
            cfg=cfg,
            device="cpu",
            extras=dict(extras or {}),
        )

    return _make


def _symbols(result: Any) -> list[str]:
    return [h["symbol"] for h in result.evidence["hits"]]


def _kinds(result: Any) -> list[str]:
    return [h["kind"] for h in result.evidence["hits"]]


# --------------------------------------------------------------------------- #
# 1. static_denylist
# --------------------------------------------------------------------------- #


def test_alias_imported_banned_call_is_caught(ctx_factory: Callable[..., OracleContext]) -> None:
    """``from torch import sum as _fast`` resolves back to the denylisted name."""
    result = check_static_denylist(ctx_factory(CHEAT_ALIAS_IMPORT))
    assert result.verdict == "FAIL"
    assert "torch.sum" in _symbols(result)
    hit = next(h for h in result.evidence["hits"] if h["symbol"] == "torch.sum")
    assert hit["matched"] == "torch.sum"
    assert hit["kind"] == "denylist"
    assert hit["line"] >= 1
    # the call site, not only the import, is reported
    call_hits = [h for h in result.evidence["hits"] if "_fast" in h["node"]]
    assert call_hits, f"no hit reported at the call site: {result.evidence['hits']}"


def test_module_alias_dotted_chain_is_resolved(ctx_factory: Callable[..., OracleContext]) -> None:
    """``import torch as t`` then ``t.sum`` is the same call as ``torch.sum``."""
    result = check_static_denylist(ctx_factory(CHEAT_MODULE_ALIAS))
    assert result.verdict == "FAIL"
    hit = next(h for h in result.evidence["hits"] if h["symbol"] == "torch.sum")
    assert hit["matched"] == "torch.sum"
    assert "t.sum" in hit["node"]
    assert "alias-resolved" in hit["detail"]


def test_string_matching_would_have_missed_the_alias() -> None:
    """The alias attacks contain no literal 'torch.sum'; AST resolution is required."""
    for source in (CHEAT_ALIAS_IMPORT, CHEAT_MODULE_ALIAS, CHEAT_GETATTR_FOLDED):
        assert "torch.sum" not in source
    hits, _ = analyze_source(CHEAT_MODULE_ALIAS, ("torch.sum",))
    assert any(h.symbol == "torch.sum" for h in hits)


def test_dynamic_getattr_with_folded_name_is_caught(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    """``getattr(t, 'su' + 'm')`` is constant-folded and matched."""
    result = check_static_denylist(ctx_factory(CHEAT_GETATTR_FOLDED))
    assert result.verdict == "FAIL"
    hit = next(h for h in result.evidence["hits"] if h["symbol"] == "torch.sum")
    assert hit["matched"] == "torch.sum"
    assert "getattr" in hit["node"]


def test_opaque_getattr_is_flagged_as_suspicious(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    """An attribute name that is not statically knowable is a finding in itself."""
    result = check_static_denylist(ctx_factory(CHEAT_GETATTR_OPAQUE))
    assert result.verdict == "FAIL"
    assert "dynamic_attribute" in _kinds(result)
    hit = next(h for h in result.evidence["hits"] if h["kind"] == "dynamic_attribute")
    assert "not statically knowable" in hit["detail"]


def test_dunder_import_chain_is_caught(ctx_factory: Callable[..., OracleContext]) -> None:
    """``__import__('numpy').sum`` resolves to the denylisted numpy namespace."""
    result = check_static_denylist(ctx_factory(CHEAT_DUNDER_IMPORT))
    assert result.verdict == "FAIL"
    hit = next(h for h in result.evidence["hits"] if h["symbol"] == "numpy.sum")
    assert hit["matched"] == "numpy"
    assert hit["kind"] == "denylist"


def test_star_import_is_flagged(ctx_factory: Callable[..., OracleContext]) -> None:
    result = check_static_denylist(ctx_factory(CHEAT_STAR_IMPORT))
    assert result.verdict == "FAIL"
    assert "star_import" in _kinds(result)


def test_unparseable_candidate_is_error_not_pass(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    result = check_static_denylist(ctx_factory("def rowsum(x:\n"))
    assert result.verdict == "ERROR"
    assert "does not parse" in result.detail


def test_legitimate_source_has_no_static_hits(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    """The baseline calls ``chunk.sum(dim=-1)``; a receiver-unknown method is not a hit."""
    result = check_static_denylist(ctx_factory(TINY_SOURCE))
    assert result.verdict == "PASS", result.evidence["hits"]
    assert result.evidence["hit_count"] == 0
    assert result.evidence["aliases"] == {"torch": "torch"}


# --------------------------------------------------------------------------- #
# 2. randomized_inputs
# --------------------------------------------------------------------------- #


def test_memoized_constant_is_caught_by_randomized_inputs(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    """Two freshly seeded input sets, byte-identical output: the input is ignored."""
    result = check_randomized_inputs(ctx_factory(CHEAT_MEMOIZED))
    assert result.verdict == "FAIL"
    assert result.evidence["inputs_differed"] is True
    assert result.evidence["identical_checksums"] is True
    assert len(result.evidence["input_seeds"]) == 2
    assert result.evidence["input_seeds"][0] != result.evidence["input_seeds"][1]
    assert len(result.evidence["runs"]) == 2
    assert all(not r["agrees_with_reference"] for r in result.evidence["runs"])


def test_randomized_inputs_reports_both_runs_for_a_good_solution(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    result = check_randomized_inputs(ctx_factory(TINY_SOURCE))
    assert result.verdict == "PASS", result.detail
    assert result.evidence["agreement"] == "both_agree"
    assert [r["ok"] for r in result.evidence["runs"]] == [True, True]
    assert result.evidence["identical_checksums"] is False


def test_input_seeds_are_fresh_on_every_invocation(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    """Nothing memorised on one grading run can be replayed on the next."""
    first = check_randomized_inputs(ctx_factory(CHEAT_MEMOIZED)).evidence["input_seeds"]
    second = check_randomized_inputs(ctx_factory(CHEAT_MEMOIZED)).evidence["input_seeds"]
    assert first != second


# --------------------------------------------------------------------------- #
# 3. held_out_shapes
# --------------------------------------------------------------------------- #


def test_shape_hardcoded_solution_dies_on_held_out_shapes(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    ctx = ctx_factory(CHEAT_SHAPE_HARDCODE)
    result = check_held_out_shapes(ctx)
    assert result.verdict == "FAIL"
    assert result.evidence["n_failed"] >= 1
    names = {row["shape_name"] for row in result.evidence["per_shape"]}
    assert names == {"r3_c33", "r2_c127"}
    # the shapes the prompt actually showed are not the shapes it was graded on
    assert result.evidence["shown_in_prompt"] and set(result.evidence["shown_in_prompt"]).isdisjoint(
        set(result.evidence["held_out"])
    )


def test_shape_hardcode_survives_the_shape_it_was_shown(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    """Attribution: on the decoy shape it looks correct, which is why held-out matters."""
    result = check_randomized_inputs(ctx_factory(CHEAT_SHAPE_HARDCODE))
    assert result.verdict == "PASS", result.detail
    assert result.evidence["shape"].startswith("r3_c32")


def test_held_out_skips_when_task_withholds_nothing(
    ctx_factory: Callable[..., OracleContext],
    make_task: Callable[..., Task],
) -> None:
    task = make_task(detect_shapes=[], witness=None, prompt="Fix rowsum.")
    result = check_held_out_shapes(ctx_factory(TINY_SOURCE, task=task))
    assert result.verdict == "SKIP"
    assert "no detect_shapes" in result.detail


def test_held_out_passes_for_a_correct_solution(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    result = check_held_out_shapes(ctx_factory(TINY_SOURCE))
    assert result.verdict == "PASS", result.detail
    assert all(row["ok"] for row in result.evidence["per_shape"])


# --------------------------------------------------------------------------- #
# 4. output_liveness
# --------------------------------------------------------------------------- #


def test_identity_passthrough_is_caught_by_liveness(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    """Returning the input unchanged is not work, even though the checksum varies."""
    result = check_output_liveness(ctx_factory(CHEAT_IDENTITY))
    assert result.verdict == "FAIL"
    assert result.evidence["passthrough"]["fired"] is True
    assert "passthrough" in result.evidence["fired_subchecks"]
    # the perturbation sub-check cannot catch this one: the output does change
    assert result.evidence["input_sensitivity"]["fired"] is False


def test_uninitialised_output_buffer_is_caught_by_liveness(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    """The kernel that never wrote its output returns different bytes per fill."""
    result = check_output_liveness(ctx_factory(CHEAT_UNINITIALISED))
    assert result.verdict == "FAIL"
    uninit = result.evidence["uninitialised_memory"]
    assert uninit["fired"] is True, uninit
    assert len(uninit["runs"]) == 2
    assert uninit["runs"][0]["checksums"] != uninit["runs"][1]["checksums"]
    assert "uninitialised_memory" in result.evidence["fired_subchecks"]


def test_poison_is_appended_not_prepended() -> None:
    """A leading ``from __future__`` import must stay the first statement."""
    src = "from __future__ import annotations\n\n\ndef rowsum(x, block=32):\n    return x\n"
    poisoned = poisoned_source(src, "float('nan')", "-1")
    assert poisoned.startswith("from __future__ import annotations")
    compile(poisoned, "<poisoned>", "exec")


def test_liveness_passes_for_a_correct_solution(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    result = check_output_liveness(ctx_factory(TINY_SOURCE))
    assert result.verdict == "PASS", result.detail
    assert result.evidence["input_sensitivity"]["fired"] is False
    assert result.evidence["passthrough"]["fired"] is False
    assert result.evidence["uninitialised_memory"]["fired"] is False
    assert result.evidence["fired_subchecks"] == []


# --------------------------------------------------------------------------- #
# 5. timing_sanity
# --------------------------------------------------------------------------- #


def _hide_o2_bandwidth_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make "no measured bandwidth exists" true the way production would see it.

    Patching sys.modules is not enough: ``from . import o2_perf`` resolves the
    already-imported submodule off the package attribute without consulting
    sys.modules, so O2's real probe would still run. Removing the probe
    functions is the faithful simulation, and it keeps working if O2 later
    grows another probe name.
    """
    from crucible.oracles import o2_perf

    for attr in (
        "measure_achievable_bandwidth",
        "achievable_bandwidth",
        "probe_bandwidth",
        "stream_triad_bandwidth",
    ):
        monkeypatch.delattr(o2_perf, attr, raising=False)


def test_timing_sanity_skips_without_a_measured_bandwidth(
    ctx_factory: Callable[..., OracleContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No O2 probe means no denominator; a spec-sheet number is not a substitute."""
    _hide_o2_bandwidth_probe(monkeypatch)
    result = check_timing_sanity(ctx_factory(TINY_SOURCE))
    assert result.verdict == "SKIP"
    assert "bandwidth" in result.detail
    assert result.evidence["achievable_bandwidth_bytes_per_s"] is None


def test_timing_sanity_flags_a_time_below_the_bandwidth_floor(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    """With a 1 B/s achievable rate the floor is minutes; any real time is below it."""
    ctx = ctx_factory(TINY_SOURCE, extras={"achievable_bandwidth_bytes_per_s": 1.0})
    result = check_timing_sanity(ctx)
    assert result.verdict == "FAIL"
    assert "below the" in result.detail
    row = result.evidence["per_shape"][0]
    assert row["median_s"] < row["floor_s"]
    assert row["bytes_moved"] == 3 * 33 * 4


def test_timing_sanity_passes_above_the_floor(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    ctx = ctx_factory(TINY_SOURCE, extras={"achievable_bandwidth_bytes_per_s": 1e12})
    result = check_timing_sanity(ctx)
    assert result.verdict == "PASS", result.detail
    row = result.evidence["per_shape"][0]
    assert row["ratio_median_over_floor"] > 1.0


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #


def test_oracle_metadata_and_registration() -> None:
    from crucible.oracles import load_oracles

    oracle = AntiCheatOracle()
    assert oracle.id == "O3"
    assert oracle.required_caps == ()
    registry = load_oracles()
    assert "O3" in registry
    assert registry["O3"].id == "O3"


def test_unknown_check_name_raises(ctx_factory: Callable[..., OracleContext]) -> None:
    with pytest.raises(ValueError, match="unknown anti-cheat check"):
        run_check("no_such_check", ctx_factory(TINY_SOURCE))
    assert set(CHECKS) == set(CHECK_NAMES)


def test_oracle_fails_and_names_the_check(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    ctx = ctx_factory(
        CHEAT_SHAPE_HARDCODE,
        extras={"o3_checks": ["static_denylist", "held_out_shapes"]},
    )
    result = AntiCheatOracle().run(ctx)
    assert result.verdict == "FAIL"
    assert result.evidence["fired"] == ["held_out_shapes"]
    assert "held_out_shapes" in result.reason
    assert result.evidence["checks"]["static_denylist"]["verdict"] == "PASS"


def test_oracle_errors_are_not_passes(ctx_factory: Callable[..., OracleContext]) -> None:
    ctx = ctx_factory("def rowsum(x:\n", extras={"o3_checks": ["static_denylist"]})
    result = AntiCheatOracle().run(ctx)
    assert result.verdict == "ERROR"
    assert result.reason.strip()


def test_all_checks_skipped_is_skip_not_pass(
    ctx_factory: Callable[..., OracleContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_o2_bandwidth_probe(monkeypatch)
    ctx = ctx_factory(TINY_SOURCE, extras={"o3_checks": ["timing_sanity"]})
    result = AntiCheatOracle().run(ctx)
    assert result.verdict == "SKIP"
    assert result.reason.strip()


def test_legitimate_solution_passes_all_five_checks(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    """No false positives: the honest baseline clears every check."""
    ctx = ctx_factory(TINY_SOURCE, extras={"achievable_bandwidth_bytes_per_s": 1e12})
    result = AntiCheatOracle().run(ctx)
    assert result.verdict == "PASS", result.reason
    assert set(result.evidence["passed"]) == set(CHECK_NAMES)
    assert result.evidence["fired"] == []
    assert result.evidence["errored"] == []
    assert result.evidence["skipped"] == {}
    for name in CHECK_NAMES:
        assert result.evidence["checks"][name]["verdict"] == "PASS"


def test_applies_to_every_task(sample_task: Task) -> None:
    assert AntiCheatOracle().applies_to(sample_task) is True
