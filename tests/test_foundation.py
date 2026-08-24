"""Tests for the foundation: capabilities, schema, taxonomy, sandbox, registry,
determinism, config and the oracle dispatcher.

These tests are written against the properties the contract actually promises:
SKIP never becomes PASS, a non-PASS verdict always carries a reason, a task
round-trips through YAML without loss, a hung candidate is really killed, and a
clock lock is always released.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from crucible import __version__, capabilities
from crucible.capabilities import CAP_NAMES, Capabilities
from crucible.config import Config
from crucible.errors import CapabilityError, CrucibleError, SandboxError
from crucible.oracles import base as obase
from crucible.runner import determinism, sandbox
from crucible.schema import (
    EnvironmentRecord,
    OracleResult,
    ShapeSpec,
    Task,
    TaskVerdict,
)
from crucible.seeds import registry
from crucible.seeds.registry import SeedSpec
from crucible.taxonomy import (
    CELLS,
    DOMAINS,
    TIER_ORDER,
    TIERS,
    CoverageGrid,
    all_cell_ids,
    cell_id,
    coverage,
)

# --------------------------------------------------------------------------- #
# package
# --------------------------------------------------------------------------- #


def test_version_and_exports() -> None:
    assert __version__ == "0.1.0"
    import crucible

    assert crucible.Task is Task
    assert crucible.TaskVerdict is TaskVerdict
    assert callable(crucible.detect)


# --------------------------------------------------------------------------- #
# capabilities
# --------------------------------------------------------------------------- #


def test_caps_missing_reports_only_falsy(caps: Capabilities) -> None:
    assert caps.missing(["gloo"]) == []
    assert caps.missing(["cuda", "triton", "gloo", "ncu"]) == ["cuda", "triton", "ncu"]
    # device_name is None and cuda_device_count is 0: both count as missing.
    assert caps.missing(["device_name", "cuda_device_count"]) == [
        "device_name",
        "cuda_device_count",
    ]


def test_caps_missing_is_empty_on_a_full_machine(caps_full: Capabilities) -> None:
    assert caps_full.missing(sorted(CAP_NAMES)) == []


def test_caps_missing_rejects_unknown_names(caps: Capabilities) -> None:
    with pytest.raises(CapabilityError) as exc:
        caps.missing(["cuda", "tensor_cores"])
    assert "tensor_cores" in str(exc.value)
    assert exc.value.missing == ["tensor_cores"]


def test_cap_names_cover_the_gating_attributes(caps: Capabilities) -> None:
    for name in CAP_NAMES:
        assert hasattr(caps, name), name
    # Diagnostics are not gates.
    assert "triton_error" not in CAP_NAMES
    assert "platform" not in CAP_NAMES


def test_caps_explain_quotes_the_probed_reason(caps: Capabilities) -> None:
    text = caps.explain(["triton", "cuda"])
    assert "DLL load failed" in text
    assert "cuda:" in text


def test_probe_subprocess_returns_payload(tmp_path: Path) -> None:
    src = capabilities._PROBE_PRELUDE + '\nemit({"ok": True, "answer": 41 + 1})\n'
    out = capabilities._run_probe(src, timeout_s=60.0)
    assert out["ok"] is True
    assert out["answer"] == 42


def test_probe_subprocess_reports_silence_as_failure_not_success() -> None:
    out = capabilities._run_probe("x = 1\n", timeout_s=60.0)
    assert out["ok"] is False
    assert "no result" in out["error"]


def test_probe_subprocess_times_out_without_hanging_the_parent() -> None:
    t0 = time.perf_counter()
    out = capabilities._run_probe("while True:\n    pass\n", timeout_s=3.0)
    elapsed = time.perf_counter() - t0
    assert out["ok"] is False
    assert "timed out" in out["error"]
    assert elapsed < 60.0


def test_detect_uses_the_cache_and_never_reprobes(tmp_path: Path, monkeypatch: Any) -> None:
    cache = tmp_path / "caps.json"
    monkeypatch.setattr(capabilities, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(capabilities, "CACHE_PATH", cache)
    monkeypatch.setattr(capabilities, "_MEMO", {})

    probed: list[int] = []
    synthetic = Capabilities(
        cuda=False,
        cuda_device_count=0,
        device_name=None,
        cuda_version=None,
        triton=False,
        triton_error="fabricated for test",
        ncu=None,
        can_lock_clocks=False,
        gloo=True,
        nccl=False,
        cxx_compiler=None,
        inductor_cpu=True,
        inductor_cuda=False,
        torch_version=capabilities._torch_version_hint(),
        platform=capabilities._platform.platform(),
        details={"triton": "fabricated for test"},
    )

    def fake_probe(_timeout: float) -> Capabilities:
        probed.append(1)
        return synthetic

    monkeypatch.setattr(capabilities, "_probe_all", fake_probe)

    first = capabilities.detect(refresh=True)
    assert first.triton_error == "fabricated for test"
    assert len(probed) == 1
    assert cache.exists()

    # In-process memo.
    assert capabilities.detect() == first
    assert len(probed) == 1

    # Cold process: memo cleared, on-disk cache still answers.
    monkeypatch.setattr(capabilities, "_MEMO", {})
    from_disk = capabilities.detect()
    assert from_disk == first
    assert len(probed) == 1

    # A stale key forces a re-probe rather than returning wrong answers.
    payload = json.loads(cache.read_text(encoding="utf-8"))
    payload["key"] = "torch=0.0.0|platform=other|py=3.0.0"
    cache.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(capabilities, "_MEMO", {})
    capabilities.detect()
    assert len(probed) == 2


def test_detect_survives_a_corrupt_cache(tmp_path: Path, monkeypatch: Any) -> None:
    cache = tmp_path / "caps.json"
    cache.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(capabilities, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(capabilities, "CACHE_PATH", cache)
    monkeypatch.setattr(capabilities, "_MEMO", {})
    assert capabilities._read_cache("any-key") is None


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def test_oracle_result_requires_a_reason_unless_pass() -> None:
    ok = OracleResult(oracle="O1", verdict="PASS")
    assert ok.reason == ""
    for bad in ("FAIL", "SKIP", "ERROR"):
        with pytest.raises(ValidationError):
            OracleResult(oracle="O1", verdict=bad)
        with pytest.raises(ValidationError):
            OracleResult(oracle="O1", verdict=bad, reason="   ")
    assert OracleResult(oracle="O1", verdict="SKIP", reason="no cuda").reason == "no cuda"


@pytest.mark.parametrize(
    "verdicts,expected",
    [
        ([], "SKIP"),
        (["PASS"], "PASS"),
        (["PASS", "PASS"], "PASS"),
        (["PASS", "SKIP"], "SKIP"),
        (["SKIP", "FAIL"], "FAIL"),
        (["FAIL", "ERROR"], "ERROR"),
        (["PASS", "SKIP", "FAIL", "ERROR"], "ERROR"),
        (["PASS", "SKIP", "FAIL"], "FAIL"),
        (["SKIP", "SKIP"], "SKIP"),
    ],
)
def test_verdict_precedence(verdicts: list[str], expected: str) -> None:
    assert TaskVerdict.combine(verdicts) == expected
    results = [
        OracleResult(oracle=f"O{i}", verdict=v, reason="" if v == "PASS" else "because")
        for i, v in enumerate(verdicts)
    ]
    assert TaskVerdict.combine(results) == expected


def test_skip_is_never_pass(sample_task: Task) -> None:
    results = [
        OracleResult(oracle="O1", verdict="PASS"),
        OracleResult(oracle="O3", verdict="SKIP", reason="requires cuda; missing cuda"),
    ]
    verdict = TaskVerdict(
        task_id=sample_task.task_id,
        verdict=TaskVerdict.combine(results),
        executed=True,
        oracle_results=results,
        environment=EnvironmentRecord.capture(None),
        duration_s=1.0,
        candidate_sha256="0" * 64,
    )
    assert verdict.verdict == "SKIP"
    assert len(verdict.skips()) == 1
    assert verdict.failures() == []


def test_task_yaml_round_trip_is_lossless(sample_task: Task, tmp_path: Path) -> None:
    text = sample_task.to_yaml()
    back = Task.from_yaml(text)
    assert back == sample_task
    assert back.model_dump() == sample_task.model_dump()

    path = sample_task.save(tmp_path / "bank" / "task.yaml")
    loaded = Task.load(path)
    assert loaded == sample_task
    # The tricky corners: int-keyed anchors and tuple thresholds survive.
    crit = loaded.rubric.criteria[0]
    assert set(crit.anchors) == {0, 3, 5}
    assert crit.probe_thresholds[0] == (1e-6, 5)
    assert loaded.witness is not None
    assert loaded.witness.shape.name == "r3_c33"
    assert loaded.calibration is not None
    assert loaded.calibration.route == "gold"
    # And a second round trip is a fixed point.
    assert Task.from_yaml(loaded.to_yaml()) == sample_task


def test_task_verdict_json_round_trip(sample_task: Task, tmp_path: Path) -> None:
    tv = TaskVerdict(
        task_id=sample_task.task_id,
        verdict="PASS",
        executed=True,
        oracle_results=[OracleResult(oracle="O1", verdict="PASS", evidence={"max_rel_err": 1e-7})],
        environment=EnvironmentRecord.capture(None),
        duration_s=2.5,
        candidate_sha256="f" * 64,
    )
    path = tv.save(tmp_path / "verdicts" / "v.json")
    assert TaskVerdict.load(path) == tv


def test_prompt_may_not_leak_a_detect_shape(make_task: Any) -> None:
    with pytest.raises(ValidationError) as exc:
        make_task(prompt="Reproduce with r3_c33 and report.")
    assert "detect" in str(exc.value)


def test_shape_key_is_order_independent_and_stable() -> None:
    a = ShapeSpec(name="one", kwargs={"rows": 4, "cols": 8})
    b = ShapeSpec(name="another_label", kwargs={"cols": 8, "rows": 4})
    c = ShapeSpec(name="one", kwargs={"rows": 4, "cols": 9})
    assert a.key() == b.key()
    assert a.key() != c.key()
    assert a.key() == ShapeSpec(name="x", kwargs={"rows": 4, "cols": 8}).key()


def test_rubric_criterion_requires_the_three_anchors() -> None:
    from crucible.schema import RubricCriterion

    with pytest.raises(ValidationError):
        RubricCriterion(id="c", weight=1.0, anchors={0: "a", 5: "b"})
    with pytest.raises(ValidationError):
        RubricCriterion(
            id="c",
            weight=1.0,
            anchors={0: "a", 3: "b", 5: "c"},
            probe_thresholds=[(1.0, 0), (1e-6, 5)],
        )
    with pytest.raises(ValidationError):
        RubricCriterion(id="c", weight=1.0, anchors={0: "a", 3: "b", 5: "c"}, machine_probed=True)


# --------------------------------------------------------------------------- #
# taxonomy
# --------------------------------------------------------------------------- #


def test_grid_is_exactly_48_cells() -> None:
    assert len(TIERS) == 6
    assert len(TIER_ORDER) == 6
    assert len(DOMAINS) == 8
    assert len(CELLS) == 48
    ids = all_cell_ids()
    assert len(ids) == 48
    assert len(set(ids)) == 48
    assert "T5/triton" in ids


def test_tier_table_matches_the_proposal() -> None:
    assert TIERS["T1"].failure_mode == "compile error"
    assert TIERS["T1"].signal_available == "full error message"
    assert TIERS["T1"].model_skill == "read the error"
    assert TIERS["T1"].data_value == "low"
    assert TIERS["T6"].failure_mode == "silent convergence divergence"
    assert TIERS["T6"].signal_available == "loss goes down, to the wrong place"
    assert TIERS["T6"].model_skill == "multi-rank differential reasoning"
    assert TIERS["T6"].data_value == "highest"
    ranks = [TIERS[t].value_rank for t in TIER_ORDER]
    assert ranks == sorted(ranks), "data value must rise monotonically with tier"
    assert {t for t in TIER_ORDER if TIERS[t].silent} == {"T5", "T6"}


def test_cell_id_rejects_unknown_axes() -> None:
    assert cell_id("T5", "triton") == "T5/triton"
    with pytest.raises(ValueError):
        cell_id("T7", "triton")
    with pytest.raises(ValueError):
        cell_id("T5", "opencl")


def test_coverage_of_an_empty_bank_is_all_gaps() -> None:
    grid = coverage([], target_per_cell=5)
    assert grid.total() == 0
    assert grid.filled() == []
    assert grid.fill_rate() == 0.0
    assert grid.silent_share() == 0.0
    gaps = grid.gaps()
    assert len(gaps) == 48
    assert all(shortfall == 5 for _, shortfall in gaps)
    assert len(grid.counts) == 48


def test_coverage_gap_math() -> None:
    tasks = (
        [{"failure_tier": "T5", "domain": "triton"}] * 5
        + [{"failure_tier": "T1", "domain": "cuda"}] * 3
        + [{"failure_tier": "T6", "domain": "pytorch"}] * 6
    )
    grid = coverage(tasks, target_per_cell=5)
    assert grid.total() == 14
    assert set(grid.filled()) == {"T5/triton", "T6/pytorch"}
    assert grid.fill_rate() == pytest.approx(2 / 48)
    gaps = dict(grid.gaps())
    assert "T5/triton" not in gaps
    assert gaps["T1/cuda"] == 2
    assert gaps["T2/jax"] == 5
    assert len(gaps) == 46
    # Worst shortfall first.
    assert grid.gaps()[0][1] == 5
    assert grid.silent_share() == pytest.approx(11 / 14)
    assert grid.by_tier()["T5"] == 5
    assert grid.by_domain()["cuda"] == 3


def test_coverage_accepts_task_objects(sample_task: Task) -> None:
    grid = coverage([sample_task], target_per_cell=1)
    assert grid.counts["T5/pytorch"] == 1
    assert grid.filled() == ["T5/pytorch"]
    assert grid.silent_share() == 1.0


def test_coverage_grid_serialises() -> None:
    grid = CoverageGrid(counts={"T5/triton": 2}, target_per_cell=5)
    d = grid.as_dict()
    assert d["total"] == 2
    assert d["n_filled"] == 0
    assert len(d["counts"]) == 48
    assert {"cell": "T5/triton", "shortfall": 3} in d["gaps"]


# --------------------------------------------------------------------------- #
# sandbox
# --------------------------------------------------------------------------- #

from tests.conftest import TINY_SOURCE  # noqa: E402  (fixture source reused as candidate)


def test_sandbox_calls_entry_and_returns_checksums(tmp_workdir: Path) -> None:
    import numpy as np
    import torch

    torch.manual_seed(0)
    x = torch.randn(4, 33)
    res = sandbox.call_entry(
        TINY_SOURCE, "rowsum", {"x": x}, workdir=tmp_workdir, timeout_s=180.0
    )
    assert res.ok, res.message + res.stderr[-2000:]
    assert res.timed_out is False
    assert res.exit_code == 0
    outputs = res.outputs()
    assert len(outputs) == 1
    rec = outputs[0]
    assert rec["shape"] == [4]
    assert rec["nan_count"] == 0
    assert len(res.checksums()["out"]) == 64

    got = np.load(rec["path"])
    want = x.sum(dim=-1).numpy()
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)


def test_sandbox_is_deterministic_for_identical_inputs(tmp_workdir: Path) -> None:
    import torch

    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    a = sandbox.call_entry(TINY_SOURCE, "rowsum", {"x": x}, workdir=tmp_workdir / "a")
    b = sandbox.call_entry(TINY_SOURCE, "rowsum", {"x": x}, workdir=tmp_workdir / "b")
    assert a.ok and b.ok
    assert a.checksums() == b.checksums()


def test_sandbox_reports_a_candidate_exception_structurally(tmp_workdir: Path) -> None:
    src = "def rowsum(x):\n    raise ValueError('boom in candidate')\n"
    res = sandbox.call_entry(src, "rowsum", {"x": [1.0]}, workdir=tmp_workdir)
    assert res.ok is False
    assert res.error is not None
    assert res.error["type"] == "candidate_exception"
    assert "boom in candidate" in res.error["message"]
    assert "ValueError" in res.traceback_text
    with pytest.raises(SandboxError):
        res.raise_for_status()


def test_sandbox_reports_a_missing_entry(tmp_workdir: Path) -> None:
    res = sandbox.call_entry("x = 1\n", "rowsum", {}, workdir=tmp_workdir)
    assert res.ok is False
    assert res.error is not None
    assert res.error["type"] == "missing_entry"


def test_sandbox_captures_a_non_zero_exit(tmp_workdir: Path) -> None:
    # os._exit bypasses the child's own error handling: nothing is written, and
    # the harness must surface the exit code rather than assume success.
    res = sandbox.run_source("import os\nos._exit(7)\n", workdir=tmp_workdir, timeout_s=60.0)
    assert res.ok is False
    assert res.exit_code == 7
    assert res.timed_out is False
    assert res.error is not None
    assert res.error["type"] == "no_result"
    assert "7" in res.message


def test_sandbox_surfaces_a_syntax_error(tmp_workdir: Path) -> None:
    res = sandbox.run_source("def (:\n", workdir=tmp_workdir, timeout_s=60.0)
    assert res.ok is False
    assert res.error is not None
    assert res.error["type"] == "import_error"
    assert "SyntaxError" in res.error["message"]


def test_sandbox_timeout_kills_the_process(tmp_workdir: Path) -> None:
    marker = tmp_workdir / "alive.txt"
    src = (
        "import time\n"
        "from pathlib import Path\n"
        f"p = Path(r'{marker}')\n"
        "while True:\n"
        "    with p.open('a', encoding='utf-8') as fh:\n"
        "        fh.write('x')\n"
        "    time.sleep(0.02)\n"
    )
    t0 = time.perf_counter()
    res = sandbox.run_source(src, workdir=tmp_workdir, timeout_s=3.0)
    elapsed = time.perf_counter() - t0

    assert res.timed_out is True
    assert res.ok is False
    assert res.error is not None and res.error["type"] == "timeout"
    assert 3.0 <= elapsed < 60.0
    assert marker.exists(), "the child should have started before being killed"

    # The decisive check: a killed process stops writing.
    size_at_kill = marker.stat().st_size
    time.sleep(1.0)
    assert marker.stat().st_size == size_at_kill


def test_sandbox_times_an_entry(tmp_workdir: Path) -> None:
    import torch

    x = torch.randn(8, 64)
    res = sandbox.time_entry(
        TINY_SOURCE, "rowsum", {"x": x}, workdir=tmp_workdir, reps=5, warmup=2, timeout_s=180.0
    )
    assert res.ok, res.message
    assert res.value is not None
    assert len(res.value["times_s"]) == 5
    assert res.value["warmup"] == 2
    assert res.value["median_s"] > 0.0
    assert all(t > 0.0 for t in res.value["times_s"])
    # A timing run still checksums its output so dead-code elision is detectable.
    assert len(res.value["checksums"]["out"]) == 64


def test_run_source_returns_the_result_global(tmp_workdir: Path) -> None:
    res = sandbox.run_source("RESULT = {'graph_breaks': 3}\n", workdir=tmp_workdir, timeout_s=60.0)
    assert res.ok is True
    assert res.value == {"result": {"graph_breaks": 3}}


def test_input_refs_never_pickle(tmp_workdir: Path) -> None:
    import torch

    refs = sandbox.input_refs_from(
        {"x": torch.zeros(2, 2, dtype=torch.bfloat16), "n": 128, "tag": "abc"},
        tmp_workdir / "in",
    )
    assert refs["x"]["kind"] == "npy"
    assert refs["x"]["torch_dtype"] == "bfloat16"
    assert Path(refs["x"]["path"]).exists()
    assert refs["n"] == {"kind": "json", "value": 128}
    assert refs["tag"] == {"kind": "json", "value": "abc"}


def test_sandbox_error_carries_context() -> None:
    err = SandboxError("boom", exit_code=3, stderr="trace", timed_out=False)
    assert isinstance(err, CrucibleError)
    assert err.exit_code == 3
    assert err.stderr == "trace"


# --------------------------------------------------------------------------- #
# seed registry
# --------------------------------------------------------------------------- #


def test_register_get_and_all_seeds(tiny_seed: SeedSpec) -> None:
    registry.unregister(tiny_seed.id)
    try:
        registered = registry.register(tiny_seed)
        assert registered is tiny_seed
        assert registry.get(tiny_seed.id) is tiny_seed
        assert tiny_seed.id in registry.seed_ids()
        assert tiny_seed in registry.all_seeds()
        assert registry.by_domain("pytorch") and registry.by_tier("T5")
        assert registry.resolve([tiny_seed.id]) == [tiny_seed]
    finally:
        registry.unregister(tiny_seed.id)
    assert tiny_seed.id not in registry.seed_ids()


def test_register_rejects_a_duplicate_id(tiny_seed: SeedSpec) -> None:
    import copy

    registry.unregister(tiny_seed.id)
    other = copy.copy(tiny_seed)
    try:
        registry.register(tiny_seed)
        with pytest.raises(ValueError):
            registry.register(other)
    finally:
        registry.unregister(tiny_seed.id)


def test_register_as_a_decorator_on_a_factory(tiny_seed: SeedSpec) -> None:
    registry.unregister("synthetic.decorated")

    def factory() -> SeedSpec:
        import dataclasses

        return dataclasses.replace(tiny_seed, id="synthetic.decorated")

    try:
        spec = registry.register(factory)
        assert isinstance(spec, SeedSpec)
        assert registry.get("synthetic.decorated") is spec
    finally:
        registry.unregister("synthetic.decorated")


def test_get_unknown_seed_lists_what_is_known() -> None:
    with pytest.raises(KeyError) as exc:
        registry.get("no.such.seed")
    assert "known seeds" in str(exc.value)


def test_discover_is_idempotent_and_reports_import_errors() -> None:
    first = registry.discover()
    second = registry.discover()
    assert first == second
    assert isinstance(registry.import_errors(), dict)


def test_seed_spec_rejects_an_entry_absent_from_its_source(tiny_seed: SeedSpec) -> None:
    import dataclasses

    with pytest.raises(ValueError):
        dataclasses.replace(tiny_seed, entry="not_in_source")
    with pytest.raises(ValueError):
        dataclasses.replace(tiny_seed, tiers=())


def test_seed_spec_helpers(tiny_seed: SeedSpec) -> None:
    assert tiny_seed.short_id() == "rowsum"
    assert len(tiny_seed.content_sha256()) == 64
    assert tiny_seed.shape("r3_c33").kwargs["cols"] == 33
    with pytest.raises(KeyError):
        tiny_seed.shape("nope")
    assert tiny_seed.accum_depth(tiny_seed.shape("r3_c33")) == 33
    assert tiny_seed.bytes_moved(tiny_seed.shape("r3_c33")) == 3 * 33 * 4


def test_seed_reference_is_not_the_baseline_code_path(tiny_seed: SeedSpec) -> None:
    import torch

    x = torch.randn(3, 33, generator=torch.Generator().manual_seed(0))
    got = tiny_seed.reference(x)
    assert got.shape == (3,)
    ns: dict[str, Any] = {}
    exec(compile(tiny_seed.source, "<seed>", "exec"), ns)  # trusted fixture source
    baseline = ns[tiny_seed.entry](x)
    cmp = tiny_seed.compare(baseline, got)
    assert cmp.ok, cmp.detail
    assert cmp.max_rel_err < 1e-5


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def test_seed_everything_makes_draws_reproducible() -> None:
    import torch

    determinism.seed_everything(7)
    a = torch.randn(5)
    determinism.seed_everything(7)
    b = torch.randn(5)
    assert torch.equal(a, b)
    determinism.seed_everything(8)
    c = torch.randn(5)
    assert not torch.equal(a, c)


def test_deterministic_ctx_restores_the_environment() -> None:
    import os

    import torch

    before_env = os.environ.get(determinism.CUBLAS_ENV)
    before_flag = torch.are_deterministic_algorithms_enabled()
    with determinism.deterministic_ctx(warn_only=True) as record:
        assert os.environ[determinism.CUBLAS_ENV] == determinism.CUBLAS_DETERMINISTIC
        assert record["applied"] is True
        assert torch.are_deterministic_algorithms_enabled() is True
    assert os.environ.get(determinism.CUBLAS_ENV) == before_env
    assert torch.are_deterministic_algorithms_enabled() == before_flag


def test_deterministic_ctx_restores_after_an_exception() -> None:
    import torch

    before = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError):
        with determinism.deterministic_ctx():
            raise RuntimeError("body failed")
    assert torch.are_deterministic_algorithms_enabled() == before


def test_lock_clocks_reports_failure_honestly(monkeypatch: Any) -> None:
    calls: list[list[str]] = []

    def fake_smi(args: list[str], timeout_s: float = 20.0) -> tuple[int, str, str]:
        calls.append(list(args))
        if args[0].startswith("--query"):
            return 0, "1395\n", ""
        if args[0] == "-lgc":
            return 4, "", "Insufficient Permissions"
        return 0, "reset", ""

    monkeypatch.setattr(determinism, "run_nvidia_smi", fake_smi)
    with determinism.lock_clocks() as lock:
        assert lock.locked is False
        assert "Insufficient Permissions" in lock.reason
        assert lock.requested_mhz == 1395
    assert ["-rgc"] in calls, "the reset must be issued even when the lock failed"
    assert lock.reset_ok is True
    assert lock.as_dict()["clock_locked"] is False


def test_lock_clocks_always_resets_even_on_exception(monkeypatch: Any) -> None:
    calls: list[list[str]] = []

    def fake_smi(args: list[str], timeout_s: float = 20.0) -> tuple[int, str, str]:
        calls.append(list(args))
        if args[0].startswith("--query"):
            return 0, "1395\n", ""
        return 0, "", ""

    monkeypatch.setattr(determinism, "run_nvidia_smi", fake_smi)
    with pytest.raises(ZeroDivisionError):
        with determinism.lock_clocks() as lock:
            assert lock.locked is True
            raise ZeroDivisionError("measurement blew up")
    assert ["-rgc"] in calls
    assert calls[-1] == ["-rgc"]


def test_lock_clocks_disabled_is_a_no_op(monkeypatch: Any) -> None:
    def fail_smi(args: list[str], timeout_s: float = 20.0) -> tuple[int, str, str]:
        raise AssertionError("nvidia-smi must not be called when locking is disabled")

    monkeypatch.setattr(determinism, "run_nvidia_smi", fail_smi)
    with determinism.lock_clocks(enabled=False) as lock:
        assert lock.locked is False
        assert lock.reason


def test_purge_autotune_cache_refuses_unrecognised_directories(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cache = tmp_path / "torchinductor_cache"
    cache.mkdir()
    (cache / "kernel.py").write_text("x=1", encoding="utf-8")
    innocent = tmp_path / "my_source_tree"
    innocent.mkdir()
    (innocent / "important.py").write_text("keep me", encoding="utf-8")

    monkeypatch.setattr(determinism, "_candidate_cache_dirs", lambda: [cache, innocent])
    report = determinism.purge_autotune_cache()
    assert not cache.exists()
    assert innocent.exists() and (innocent / "important.py").exists()
    assert str(cache.resolve()) in report["removed"]
    assert str(innocent.resolve()) in report["refused"]


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_config_defaults_match_the_contract() -> None:
    c = Config()
    assert c.alpha_gate == 0.67
    assert c.target_per_cell == 5
    assert c.bootstrap_resamples == 2000
    assert c.perf_reps >= 30
    assert c.nrank_k == 3.0
    assert c.timeout_for("O5") == 1200.0
    assert c.timeout_for("O9") == c.default_oracle_timeout_s


def test_config_loads_yaml_and_rejects_typos(tmp_path: Path) -> None:
    good = tmp_path / "cfg.yaml"
    good.write_text("perf_reps: 64\nalpha_gate: 0.8\n", encoding="utf-8")
    c = Config.load(good)
    assert c.perf_reps == 64
    assert c.alpha_gate == 0.8
    assert c.target_per_cell == 5  # untouched default

    bad = tmp_path / "bad.yaml"
    bad.write_text("perf_repz: 64\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Config.load(bad)

    assert Config.load(None) == Config()
    with pytest.raises(CrucibleError):
        Config.load(tmp_path / "missing.yaml")

    notmap = tmp_path / "list.yaml"
    notmap.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(CrucibleError):
        Config.load(notmap)


def test_config_validators_reject_nonsense() -> None:
    with pytest.raises(ValidationError):
        Config(tolerance_mode="vibes")
    with pytest.raises(ValidationError):
        Config(irr_metric="kappa")
    with pytest.raises(ValidationError):
        Config(nrank_world_sizes=[1])


def test_config_yaml_round_trip(tmp_path: Path) -> None:
    c = Config(perf_reps=41)
    p = c.save(tmp_path / "c.yaml")
    assert Config.load(p) == c


# --------------------------------------------------------------------------- #
# oracle dispatcher
# --------------------------------------------------------------------------- #


class _FakeOracle:
    """Minimal Oracle implementation used to exercise the dispatcher."""

    def __init__(
        self,
        oid: str = "OX",
        required: tuple[str, ...] = (),
        behaviour: str = "pass",
        applies: bool = True,
    ) -> None:
        self.id = oid
        self.name = f"fake-{oid}"
        self.required_caps = required
        self.behaviour = behaviour
        self._applies = applies
        self.runs = 0

    def applies_to(self, task: Task) -> bool:
        if self.behaviour == "applies_raises":
            raise RuntimeError("bad predicate")
        return self._applies

    def run(self, ctx: obase.OracleContext) -> OracleResult:
        self.runs += 1
        if self.behaviour == "raise":
            raise ZeroDivisionError("oracle exploded")
        if self.behaviour == "hang":
            time.sleep(30.0)
        if self.behaviour == "wrong_type":
            return "not a result"  # type: ignore[return-value]
        if self.behaviour == "fail":
            return OracleResult(oracle=self.id, verdict="FAIL", reason="witness reproduced")
        return OracleResult(oracle=self.id, verdict="PASS", evidence={"ran": True})


def _ctx(task: Task, caps: Capabilities, cfg: Config, workdir: Path) -> obase.OracleContext:
    return obase.OracleContext(
        task=task,
        candidate_src="def rowsum(x):\n    return x.sum(-1)\n",
        seed=None,
        caps=caps,
        workdir=workdir,
        cfg=cfg,
        rng_seed=1234,
        device="cpu",
    )


@pytest.fixture(autouse=True)
def _clean_oracle_registry() -> Any:
    saved = dict(obase.ORACLES)
    yield
    obase.ORACLES.clear()
    obase.ORACLES.update(saved)


def test_missing_capability_becomes_skip_not_pass(
    sample_task: Task, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    oracle = _FakeOracle("O2", required=("cuda", "can_lock_clocks"))
    obase.register_oracle(oracle)
    results = obase.run_all(_ctx(sample_task, caps, cfg, tmp_workdir), ids=["O2"])
    assert len(results) == 1
    r = results[0]
    assert r.verdict == "SKIP"
    assert "requires cuda, can_lock_clocks" in r.reason
    assert "missing cuda, can_lock_clocks" in r.reason
    assert "torch.cuda.is_available() returned False" in r.reason
    assert oracle.runs == 0, "a cap-gated oracle must not execute"
    assert TaskVerdict.combine(results) == "SKIP"


def test_present_capabilities_let_the_oracle_run(
    sample_task: Task, caps_full: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    oracle = _FakeOracle("O2", required=("cuda",))
    obase.register_oracle(oracle)
    results = obase.run_all(_ctx(sample_task, caps_full, cfg, tmp_workdir), ids=["O2"])
    assert results[0].verdict == "PASS"
    assert results[0].duration_s > 0.0
    assert oracle.runs == 1


def test_oracle_exception_becomes_error_with_a_traceback(
    sample_task: Task, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    obase.register_oracle(_FakeOracle("O1", behaviour="raise"))
    results = obase.run_all(_ctx(sample_task, caps, cfg, tmp_workdir), ids=["O1"])
    r = results[0]
    assert r.verdict == "ERROR"
    assert "ZeroDivisionError" in r.reason
    assert "oracle exploded" in r.evidence["traceback"]
    assert "Traceback" in r.evidence["traceback"]


def test_oracle_timeout_becomes_error(
    sample_task: Task, caps: Capabilities, tmp_workdir: Path
) -> None:
    fast = Config(oracle_timeouts_s={"O1": 0.5}, default_oracle_timeout_s=0.5)
    obase.register_oracle(_FakeOracle("O1", behaviour="hang"))
    t0 = time.perf_counter()
    results = obase.run_all(_ctx(sample_task, caps, fast, tmp_workdir), ids=["O1"])
    elapsed = time.perf_counter() - t0
    assert results[0].verdict == "ERROR"
    assert "budget" in results[0].reason
    assert elapsed < 25.0, "run_all must not wait for a hung oracle"


def test_unknown_oracle_id_is_an_error_not_a_silent_pass(
    sample_task: Task, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    results = obase.run_all(_ctx(sample_task, caps, cfg, tmp_workdir), ids=["O9"])
    assert results[0].verdict == "ERROR"
    assert "not registered" in results[0].reason


def test_inapplicable_oracle_is_not_run_and_cannot_force_skip(
    sample_task: Task, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    obase.register_oracle(_FakeOracle("O1", applies=False))
    obase.register_oracle(_FakeOracle("O3"))
    results = obase.run_all(_ctx(sample_task, caps, cfg, tmp_workdir), ids=["O1", "O3"])
    assert [r.oracle for r in results] == ["O3"]
    assert TaskVerdict.combine(results) == "PASS"


def test_broken_applies_to_is_an_error(
    sample_task: Task, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    obase.register_oracle(_FakeOracle("O1", behaviour="applies_raises"))
    results = obase.run_all(_ctx(sample_task, caps, cfg, tmp_workdir), ids=["O1"])
    assert results[0].verdict == "ERROR"
    assert "applies_to raised" in results[0].reason


def test_oracle_returning_the_wrong_type_is_an_error(
    sample_task: Task, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    obase.register_oracle(_FakeOracle("O1", behaviour="wrong_type"))
    results = obase.run_all(_ctx(sample_task, caps, cfg, tmp_workdir), ids=["O1"])
    assert results[0].verdict == "ERROR"
    assert "expected OracleResult" in results[0].reason


def test_run_all_defaults_to_the_tasks_own_oracle_list(
    sample_task: Task, caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    obase.register_oracle(_FakeOracle("O1"))
    obase.register_oracle(_FakeOracle("O3", behaviour="fail"))
    results = obase.run_all(_ctx(sample_task, caps, cfg, tmp_workdir))
    assert [r.oracle for r in results] == ["O1", "O3"]
    assert TaskVerdict.combine(results) == "FAIL"


def test_gate_capabilities_flags_an_unknown_declared_cap(caps: Capabilities) -> None:
    oracle = _FakeOracle("O1", required=("tensor_cores",))
    reason = obase.gate_capabilities(oracle, caps)
    assert reason is not None
    assert "unknown capability" in reason


def test_oracle_package_tolerates_missing_modules() -> None:
    import crucible.oracles as pkg

    registry_map = pkg.load_oracles(force=True)
    assert isinstance(registry_map, dict)
    errors = pkg.load_errors()
    # o1..o5 are other authors' files; whichever are absent must be reported,
    # never silently reduce the number of checks.
    for oid in pkg.ORACLE_MODULES.values():
        assert oid in registry_map or oid in errors
