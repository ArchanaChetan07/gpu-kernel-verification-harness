"""Tests for O2, the performance oracle.

The invariant under test is not "the oracle produces a number" but "the oracle
produces a number whose weakness is visible in the data". So the assertions here
are mostly about what O2 refuses to do: it refuses to run without CUDA, refuses
to report counters it did not collect, refuses to call a straddling interval a
win, and refuses to present a free-running-clock measurement as if it were as
strong as a pinned one.

Nothing here needs a GPU. The timing harness is substituted wherever hardware
would otherwise be required; the two places real execution is cheap and
meaningful -- the out-of-process CPU timing run and the CPU STREAM-triad probe
-- are executed for real, because a mocked measurement proves nothing about the
measurement.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pytest

from crucible.capabilities import Capabilities
from crucible.config import Config
from crucible.oracles import o2_perf
from crucible.oracles.base import OracleContext, run_one
from crucible.oracles.o2_perf import (
    MIN_TIMED_REPS,
    MIN_WARMUP_REPS,
    NCU_METRICS,
    UNLOCKED_CLOCK_CI_INFLATION,
    AchievablePeaks,
    BandwidthProbe,
    NcuResult,
    PerformanceOracle,
    TimingRun,
    bootstrap_median_ci,
    bootstrap_ratio_ci,
    classify_ratio,
    clear_peak_cache,
    collect_ncu_counters,
    measure_achievable_bandwidth,
    measure_achievable_peaks,
    parse_ncu_csv,
    time_in_sandbox,
    widen_ci,
)
from crucible.runner.determinism import ClockLock
from crucible.schema import ShapeSpec, Task
from crucible.seeds.registry import SeedSpec

from .conftest import TINY_SOURCE

# A candidate that is textually distinct from ``task.baseline_code`` so the
# oracle has two sides to compare; the difference is irrelevant to the fakes.
CANDIDATE_SOURCE = TINY_SOURCE.replace("block: int = 32", "block: int = 64")

RAISING_SOURCE = '''import torch


def rowsum(x, block=32):
    raise ZeroDivisionError("this candidate cannot be timed")
'''


# --------------------------------------------------------------------------- #
# fixtures and fakes
# --------------------------------------------------------------------------- #


@pytest.fixture
def ctx_factory(
    tiny_seed: SeedSpec,
    caps: Capabilities,
    caps_full: Capabilities,
    cfg: Config,
    tmp_workdir: Path,
    make_task: Callable[..., Task],
) -> Callable[..., OracleContext]:
    def _make(
        candidate_src: str = CANDIDATE_SOURCE,
        *,
        with_cuda: bool = True,
        device: str = "cpu",
        task: Task | None = None,
        extras: Mapping[str, Any] | None = None,
        config: Config | None = None,
    ) -> OracleContext:
        return OracleContext(
            task=task if task is not None else make_task(),
            candidate_src=candidate_src,
            seed=tiny_seed,
            caps=caps_full if with_cuda else caps,
            workdir=tmp_workdir,
            cfg=config if config is not None else cfg,
            device=device,
            extras=dict(extras or {}),
        )

    return _make


def lognormal_times(n: int, center: float, sigma: float, seed: int) -> list[float]:
    """Positive, right-skewed samples: what a latency sample actually looks like."""
    rng = np.random.default_rng(seed)
    return [float(center * np.exp(v)) for v in rng.normal(0.0, sigma, n)]


def fake_run(times: Sequence[float], device: str = "cpu", reps: int = 0, warmup: int = 0) -> TimingRun:
    return TimingRun(
        ok=True,
        times_s=list(times),
        timer="cuda_event" if device.startswith("cuda") else "perf_counter",
        device=device,
        reps=reps or len(times),
        warmup=warmup,
        checksums={"out": "f" * 64},
    )


@pytest.fixture
def stub_environment(monkeypatch: pytest.MonkeyPatch) -> Callable[..., dict[str, Any]]:
    """Replace every part of O2 that would otherwise touch real hardware.

    Returns a recorder so a test can assert what the oracle *asked* for -- the
    rep counts in particular, which is where the contract's floors live.
    """

    def _install(
        *,
        clock_locked: bool = True,
        clock_reason: str = "",
        candidate_times: Sequence[float] | None = None,
        baseline_times: Sequence[float] | None = None,
        candidate_run: TimingRun | None = None,
        peaks: AchievablePeaks | None = None,
    ) -> dict[str, Any]:
        recorder: dict[str, Any] = {"calls": [], "purged": 0, "lock_enabled": None}

        @contextlib.contextmanager
        def fake_lock(mhz: int | None = None, enabled: bool = True) -> Iterator[ClockLock]:
            recorder["lock_enabled"] = enabled
            yield ClockLock(
                locked=clock_locked,
                requested_mhz=1695 if clock_locked else None,
                reason=clock_reason,
                reset_ok=True,
            )

        def fake_purge() -> dict[str, Any]:
            recorder["purged"] += 1
            return {"removed": [r"C:\fake\torchinductor_cache"], "refused": [], "errors": []}

        def fake_time(
            source: str,
            entry: str,
            inputs: Mapping[str, Any] | None = None,
            *,
            workdir: Path | str,
            device: str = "cpu",
            reps: int = MIN_TIMED_REPS,
            warmup: int = MIN_WARMUP_REPS,
            timeout_s: float = 600.0,
            seed: int = 0,
            call_style: str = "kwargs",
            arg_order: Sequence[str] | None = None,
            input_refs: Mapping[str, Any] | None = None,
        ) -> TimingRun:
            is_candidate = source == CANDIDATE_SOURCE
            recorder["calls"].append(
                {
                    "role": "candidate" if is_candidate else "baseline",
                    "entry": entry,
                    "reps": reps,
                    "warmup": warmup,
                    "device": device,
                    "workdir": Path(workdir),
                    "seed": seed,
                }
            )
            if is_candidate:
                if candidate_run is not None:
                    return candidate_run
                return fake_run(
                    candidate_times if candidate_times is not None
                    else lognormal_times(reps, 1e-3, 0.05, 11),
                    device,
                    reps,
                    warmup,
                )
            return fake_run(
                baseline_times if baseline_times is not None
                else lognormal_times(reps, 1e-3, 0.05, 12),
                device,
                reps,
                warmup,
            )

        measured = peaks if peaks is not None else AchievablePeaks(
            device="cpu",
            dtype="float32",
            bandwidth=BandwidthProbe(
                bytes_per_s=1.2e11, method="stream_triad", device="cpu", detail={"dtype": "float32"}
            ),
            flops_per_s=1.5e12,
            measured_utc="2026-01-01T00:00:00+00:00",
        )

        def fake_peaks(
            device: str = "cuda",
            dtype: str = "float32",
            *,
            caps: Capabilities | None = None,
            cfg: Config | None = None,
            refresh: bool = False,
        ) -> AchievablePeaks:
            return measured

        monkeypatch.setattr(o2_perf, "lock_clocks", fake_lock)
        monkeypatch.setattr(o2_perf, "purge_autotune_cache", fake_purge)
        monkeypatch.setattr(o2_perf, "time_in_sandbox", fake_time)
        monkeypatch.setattr(o2_perf, "measure_achievable_peaks", fake_peaks)
        return recorder

    return _install


# --------------------------------------------------------------------------- #
# registration and capability gating
# --------------------------------------------------------------------------- #


def test_oracle_metadata_and_registration() -> None:
    from crucible.oracles import load_oracles

    oracle = PerformanceOracle()
    assert oracle.id == "O2"
    assert oracle.required_caps == ("cuda",)
    registry = load_oracles()
    assert "O2" in registry
    assert registry["O2"].id == "O2"


def test_run_skips_cleanly_without_cuda(ctx_factory: Callable[..., OracleContext]) -> None:
    """No CUDA is a SKIP with the probed reason, never a PASS by omission."""
    ctx = ctx_factory(with_cuda=False)
    result = PerformanceOracle().run(ctx)

    assert result.verdict == "SKIP"
    assert "cuda" in result.reason.lower()
    assert "torch.cuda.is_available() returned False" in result.reason
    assert result.capabilities_used == ["cuda"]
    # Nothing was measured, so nothing about a measurement may appear.
    for forbidden in ("per_shape", "clock_locked", "peaks", "ncu"):
        assert forbidden not in result.evidence


def test_dispatcher_gate_also_skips_without_cuda(ctx_factory: Callable[..., OracleContext]) -> None:
    result = run_one(o2_perf.O2, ctx_factory(with_cuda=False), timeout_s=30.0)
    assert result.verdict == "SKIP"
    assert "requires cuda" in result.reason
    assert "missing cuda" in result.reason


# --------------------------------------------------------------------------- #
# bootstrap statistics
# --------------------------------------------------------------------------- #


def test_bootstrap_median_ci_covers_the_true_median_at_about_the_nominal_rate() -> None:
    """A nominal 95% interval must actually cover about 95% of the time.

    Lognormal, because latency samples are right-skewed and a symmetric
    normal-theory interval would flatter the method here.
    """
    rng = np.random.default_rng(20260818)
    n, trials = 31, 300
    true_median = 1.0  # lognormal(mu=0) has median exp(0) = 1
    hits = 0
    for trial in range(trials):
        sample = rng.lognormal(0.0, 0.5, n)
        ci = bootstrap_median_ci(sample, resamples=300, level=0.95, seed=trial)
        assert ci.low <= ci.point <= ci.high
        hits += int(ci.contains(true_median))
    coverage = hits / trials
    assert 0.85 <= coverage <= 0.995, f"coverage {coverage:.3f} is not near the nominal 0.95"


def test_bootstrap_median_ci_is_reproducible_and_records_its_method() -> None:
    sample = lognormal_times(40, 1e-3, 0.2, 5)
    first = bootstrap_median_ci(sample, resamples=500, seed=7)
    second = bootstrap_median_ci(sample, resamples=500, seed=7)
    assert (first.low, first.high) == (second.low, second.high)
    assert first.as_dict()["method"] == "percentile bootstrap"
    assert first.as_dict()["resamples"] == 500
    assert first.as_dict()["n_samples"] == 40
    assert first.as_dict()["inflation_factor"] == 1.0


def test_bootstrap_median_ci_refuses_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        bootstrap_median_ci([])


def test_bootstrap_median_ci_calls_a_single_sample_degenerate() -> None:
    """One rep carries no dispersion, and the interval must say so, not imply width."""
    ci = bootstrap_median_ci([1.5])
    assert ci.low == ci.high == ci.point == 1.5
    assert "degenerate" in ci.method


def test_ratio_ci_straddling_one_is_inconclusive() -> None:
    """Two draws from the same distribution are not a speedup."""
    a = lognormal_times(40, 1e-3, 0.10, 101)
    b = lognormal_times(40, 1e-3, 0.10, 202)
    ci = bootstrap_ratio_ci(a, b, resamples=800, seed=3)
    assert ci.straddles_one
    assert classify_ratio(ci) == "inconclusive"


def test_ratio_ci_detects_a_real_win_and_a_real_loss() -> None:
    baseline = lognormal_times(40, 2e-3, 0.05, 11)
    fast = lognormal_times(40, 1e-3, 0.05, 12)
    win = bootstrap_ratio_ci(baseline, fast, resamples=800, seed=4)
    assert classify_ratio(win) == "faster"
    assert win.low > 1.0
    loss = bootstrap_ratio_ci(fast, baseline, resamples=800, seed=4)
    assert classify_ratio(loss) == "slower"
    assert loss.high < 1.0


def test_ratio_ci_refuses_a_non_positive_denominator() -> None:
    with pytest.raises(ValueError, match="positive denominator"):
        bootstrap_ratio_ci([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])


def test_widen_ci_stretches_the_interval_and_keeps_the_original() -> None:
    ci = bootstrap_median_ci(lognormal_times(40, 1e-3, 0.2, 9), resamples=500, seed=1)
    wide = widen_ci(ci, 2.0, "because the clocks were free-running")

    assert wide.point == ci.point
    assert wide.width > ci.width
    assert wide.low < ci.low
    assert wide.high > ci.high
    assert wide.low > 0.0, "a geometric widening of a positive quantity must stay positive"
    assert wide.inflation == 2.0
    assert "free-running" in wide.inflation_reason
    payload = wide.as_dict()
    assert payload["uninflated_low"] == pytest.approx(ci.low)
    assert payload["uninflated_high"] == pytest.approx(ci.high)


def test_widen_ci_is_a_no_op_at_factor_one() -> None:
    ci = bootstrap_median_ci(lognormal_times(20, 1e-3, 0.2, 13), resamples=200, seed=2)
    assert widen_ci(ci, 1.0, "not applicable") is ci


# --------------------------------------------------------------------------- #
# ncu: absence is recorded, never zero-filled
# --------------------------------------------------------------------------- #


NCU_CSV = '''==PROF== Connected to process 4242
==PROF== Profiling "rowsum_kernel" - 0: 0%....50%....100% - 8 passes
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
"0","4242","python.exe","127.0.0.1","rowsum_kernel","1","7","Command line profiler metrics","sm__warps_active.avg.pct_of_peak_sustained_active","%","41.37"
"0","4242","python.exe","127.0.0.1","rowsum_kernel","1","7","Command line profiler metrics","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","72.10"
"0","4242","python.exe","127.0.0.1","rowsum_kernel","1","7","Command line profiler metrics","l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum","byte","1,024"
"0","4242","python.exe","127.0.0.1","rowsum_kernel","1","7","Command line profiler metrics","l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum","byte","0"
"1","4242","python.exe","127.0.0.1","epilogue_kernel","1","7","Command line profiler metrics","sm__warps_active.avg.pct_of_peak_sustained_active","%","18.50"
'''


def test_parse_ncu_csv_reads_the_metric_table() -> None:
    parsed = parse_ncu_csv(NCU_CSV)

    occupancy = parsed["sm__warps_active.avg.pct_of_peak_sustained_active"]
    assert occupancy["unit"] == "%"
    assert occupancy["label"] == "achieved_occupancy_pct"
    assert occupancy["n_kernels"] == 2
    assert occupancy["max"] == pytest.approx(41.37)
    assert occupancy["min"] == pytest.approx(18.50)
    assert occupancy["kernels"] == ["rowsum_kernel", "epilogue_kernel"]

    spills = parsed["l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum"]
    assert spills["values"] == [1024.0], "thousands separators must be parsed, not dropped"
    # A genuinely measured zero is a fact and is kept; an *unmeasured* counter is
    # a different thing entirely, and is tested below by its absence.
    assert parse_ncu_csv(NCU_CSV)["l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"]["values"] == [0.0]


def test_parse_ncu_csv_without_a_metric_table_is_empty() -> None:
    assert parse_ncu_csv("==ERROR== ERR_NVGPUCTRPERM: permission denied\n") == {}
    assert parse_ncu_csv("") == {}


def test_ncu_absent_omits_counters_rather_than_zeroing_them(tmp_path: Path) -> None:
    result = collect_ncu_counters(None, ["python", "target.py"], workdir=tmp_path)

    assert result.available is False
    assert result.counters is None
    assert "ncu" in result.reason
    payload = result.as_dict()
    assert "counters" not in payload, "an unavailable profiler must not contribute a counter dict"
    assert payload["available"] is False
    assert payload["reason"]
    assert payload["metrics_requested"] == list(NCU_METRICS)


def test_ncu_launch_failure_records_the_reason_and_omits_counters(tmp_path: Path) -> None:
    """A bogus ncu path is the elevation-failure path in miniature: no counters."""
    missing = tmp_path / "definitely-not-ncu.bat"
    result = collect_ncu_counters(str(missing), ["python", "-c", "pass"], workdir=tmp_path)

    assert result.available is False
    assert result.counters is None
    assert "could not be launched" in result.reason
    assert "counters" not in result.as_dict()


def test_ncu_result_with_counters_exposes_them() -> None:
    parsed = parse_ncu_csv(NCU_CSV)
    payload = NcuResult(available=True, counters=parsed, returncode=0).as_dict()
    assert "counters" in payload
    assert payload["counters"]["dram__throughput.avg.pct_of_peak_sustained_elapsed"]["max"] == (
        pytest.approx(72.10)
    )


def test_oracle_skips_ncu_when_capabilities_report_none(
    ctx_factory: Callable[..., OracleContext], caps: Capabilities, tmp_path: Path
) -> None:
    ctx = ctx_factory(with_cuda=False)
    ctx.caps = caps  # ncu is None here, with the probed reason attached
    result = PerformanceOracle().collect_counters(ctx, "cuda", tmp_path / "job.json")
    assert result.available is False
    assert "ncu" in result.reason
    assert "counters" not in result.as_dict()


# --------------------------------------------------------------------------- #
# measured peaks (the roofline denominator)
# --------------------------------------------------------------------------- #


def test_bandwidth_probe_measures_a_real_rate_on_this_device() -> None:
    """The denominator is measured here and now, on the CPU if that is all there is."""
    probe = measure_achievable_bandwidth(device="cpu", dtype="float32")

    assert probe.bytes_per_s is not None
    assert probe.bytes_per_s > 0.0
    assert probe.method == "stream_triad"
    payload = probe.as_dict()
    assert payload["measured"] is True
    assert payload["bytes_per_s"] == pytest.approx(probe.bytes_per_s)
    assert payload["bytes_per_rep"] == 3 * payload["elements"] * 4
    assert payload["median_s"] > 0.0


def test_peaks_probe_measures_both_denominators_and_is_cached() -> None:
    first = measure_achievable_peaks("cpu", "float32")
    second = measure_achievable_peaks("cpu", "float32")
    assert first is second, "the peak probes must run once per session, not per shape"
    assert first.flops_per_s is not None and first.flops_per_s > 0.0
    payload = first.as_dict()
    assert payload["compute"]["measured"] is True
    assert payload["compute"]["flops_per_rep"] == 2.0 * payload["compute"]["size"] ** 3
    assert "spec" not in payload["source"].lower() or "no spec-sheet" in payload["source"]


def test_unmeasurable_bandwidth_is_absent_not_zero(caps: Capabilities) -> None:
    """No CUDA device means no CUDA bandwidth number at all."""
    clear_peak_cache()
    try:
        peaks = measure_achievable_peaks("cuda", "float32", caps=caps, refresh=True)
        assert peaks.bytes_per_s is None
        assert peaks.flops_per_s is None
        payload = peaks.as_dict()
        assert payload["bandwidth"]["measured"] is False
        assert "bytes_per_s" not in payload["bandwidth"]
        assert "no CUDA device" in payload["bandwidth"]["reason"]
        assert payload["compute"]["measured"] is False
        assert "flops_per_s" not in payload["compute"]
    finally:
        clear_peak_cache()


def test_o3_timing_sanity_can_consume_the_measured_bandwidth(
    ctx_factory: Callable[..., OracleContext],
) -> None:
    """The published probe is the one O3 actually looks for."""
    from crucible.oracles.o3_anticheat import achievable_bandwidth

    value, source = achievable_bandwidth(ctx_factory(with_cuda=False, device="cpu"))
    assert value is not None and value > 0.0
    assert "o2_perf" in source
    assert "measure_achievable_bandwidth" in source


# --------------------------------------------------------------------------- #
# the timing harness, executed for real out of process
# --------------------------------------------------------------------------- #


def test_timing_harness_runs_the_candidate_out_of_process(tmp_path: Path) -> None:
    import torch

    run = time_in_sandbox(
        TINY_SOURCE,
        "rowsum",
        {"x": torch.randn(3, 33)},
        workdir=tmp_path / "timed",
        device="cpu",
        reps=12,
        warmup=3,
        timeout_s=180.0,
    )

    assert run.ok, run.reason
    assert run.timer == "perf_counter"
    assert len(run.times_s) == 12, "warmup reps must be discarded, never folded into the sample"
    assert run.warmup == 3
    assert all(t > 0.0 for t in run.times_s)
    assert run.median_s > 0.0
    assert run.checksums, "the output must be checksummed so the timed work cannot be elided"
    # The candidate was executed in a child interpreter, which left its job behind.
    assert (tmp_path / "timed" / "o2_timing_job.json").exists()


def test_timing_harness_reports_a_raising_candidate_without_crashing(tmp_path: Path) -> None:
    import torch

    run = time_in_sandbox(
        RAISING_SOURCE,
        "rowsum",
        {"x": torch.randn(2, 4)},
        workdir=tmp_path / "boom",
        device="cpu",
        reps=4,
        warmup=1,
        timeout_s=180.0,
    )

    assert run.ok is False
    assert "ZeroDivisionError" in run.reason
    assert run.times_s == []
    with pytest.raises(ValueError, match="no samples"):
        _ = run.median_s


def test_timing_harness_refuses_cuda_it_does_not_have(tmp_path: Path) -> None:
    """CUDA hidden from the child: the harness must refuse, not silently use the CPU.

    ``CUDA_VISIBLE_DEVICES=-1`` makes this path reachable on a GPU machine too,
    so the assertion is not quietly skipped on the hardware it matters most on.
    (An empty string would not do: Windows treats ``VAR=`` as unsetting VAR.)
    """
    import torch

    run = time_in_sandbox(
        TINY_SOURCE,
        "rowsum",
        {"x": torch.randn(2, 4)},
        workdir=tmp_path / "nocuda",
        device="cuda",
        reps=4,
        warmup=1,
        timeout_s=180.0,
        env_extra={"CUDA_VISIBLE_DEVICES": "-1"},
    )
    assert run.ok is False
    assert "torch.cuda.is_available() is False" in run.reason
    assert run.times_s == []
    # And the oracle reads that message as a machine limit, not a candidate bug.
    assert o2_perf._looks_like_infrastructure(run.reason) is not None


# --------------------------------------------------------------------------- #
# the oracle end to end, with the harness substituted
# --------------------------------------------------------------------------- #


def test_pass_records_the_conditions_of_the_measurement(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
) -> None:
    recorder = stub_environment(clock_locked=True)
    result = PerformanceOracle().run(ctx_factory())

    assert result.verdict == "PASS", result.reason
    ev = result.evidence
    assert ev["clock_locked"] is True
    assert ev["ci_inflation_factor"] == 1.0
    assert ev["cache_purge"]["performed"] is True
    assert recorder["purged"] == 1
    assert ev["cache_purge"]["removed"]
    assert ev["ci_method"] == "percentile bootstrap over the per-rep times"
    assert "can_lock_clocks" in result.capabilities_used

    shape = ev["per_shape"][0]
    assert shape["status"] == "pass"
    ci = shape["candidate"]["ci"]
    assert ci["low"] <= shape["median_s"] <= ci["high"]
    assert ci["inflation_factor"] == 1.0
    assert len(shape["candidate"]["times_s"]) == MIN_TIMED_REPS
    assert shape["roofline"]["achieved_bw_frac"] > 0.0
    assert shape["roofline"]["achieved_flops_frac"] > 0.0
    assert shape["roofline"]["bytes_moved"] == 2 * 127 * 4  # the largest detect shape


def test_reps_and_warmup_floors_are_enforced_over_a_permissive_config(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
    cfg: Config,
) -> None:
    """The test config asks for 3 reps; the oracle's claim needs 30."""
    assert cfg.perf_reps < MIN_TIMED_REPS and cfg.perf_warmup < MIN_WARMUP_REPS
    recorder = stub_environment()
    result = PerformanceOracle().run(ctx_factory())

    assert result.verdict == "PASS", result.reason
    assert result.evidence["reps"] == MIN_TIMED_REPS
    assert result.evidence["warmup"] == MIN_WARMUP_REPS
    policy = result.evidence["rep_policy"]
    assert policy["reps_requested"] == cfg.perf_reps
    assert policy["floor_applied"] is True
    assert "never lower them" in policy["floor_reason"]
    assert [c["reps"] for c in recorder["calls"]] == [MIN_TIMED_REPS, MIN_TIMED_REPS]
    assert [c["warmup"] for c in recorder["calls"]] == [MIN_WARMUP_REPS, MIN_WARMUP_REPS]


def test_unlocked_clocks_widen_the_reported_ci_and_say_why(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
) -> None:
    """Same samples, weaker conditions: the interval must be visibly wider."""
    times = lognormal_times(MIN_TIMED_REPS, 1e-3, 0.08, 77)

    stub_environment(clock_locked=True, candidate_times=times)
    locked = PerformanceOracle().run(ctx_factory())

    stub_environment(
        clock_locked=False,
        clock_reason="The current user does not have permission to change clocks",
        candidate_times=times,
    )
    unlocked = PerformanceOracle().run(ctx_factory())

    assert locked.verdict == "PASS" and unlocked.verdict == "PASS", unlocked.reason

    assert unlocked.evidence["clock_locked"] is False
    assert unlocked.evidence["ci_inflation_factor"] == UNLOCKED_CLOCK_CI_INFLATION
    assert "widened" in unlocked.evidence["ci_inflation_reason"]
    assert "permission to change clocks" in unlocked.evidence["clock_lock"]["reason"]
    assert "weaker claim" in unlocked.evidence["measurement_strength"]
    assert "can_lock_clocks" not in unlocked.capabilities_used

    locked_ci = locked.evidence["per_shape"][0]["candidate"]["ci"]
    wide_ci = unlocked.evidence["per_shape"][0]["candidate"]["ci"]
    raw_ci = unlocked.evidence["per_shape"][0]["candidate"]["ci_unwidened"]

    assert wide_ci["point"] == pytest.approx(locked_ci["point"])
    assert wide_ci["width"] > locked_ci["width"]
    assert wide_ci["width"] > raw_ci["width"]
    assert raw_ci["width"] == pytest.approx(locked_ci["width"])
    assert wide_ci["low"] > 0.0
    assert wide_ci["uninflated_low"] == pytest.approx(raw_ci["low"])
    assert "CI widened" in unlocked.reason


def test_require_clock_lock_skips_rather_than_timing_on_free_clocks(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
    cfg: Config,
) -> None:
    recorder = stub_environment(clock_locked=False, clock_reason="no permission")
    strict = cfg.model_copy(update={"require_clock_lock": True})
    result = PerformanceOracle().run(ctx_factory(config=strict))

    assert result.verdict == "SKIP"
    assert "require_clock_lock" in result.reason
    assert result.evidence["clock_locked"] is False
    assert recorder["calls"] == [], "nothing may be timed once the run is known to be invalid"


def test_speedup_that_straddles_one_is_reported_as_inconclusive(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
) -> None:
    stub_environment(
        candidate_times=lognormal_times(MIN_TIMED_REPS, 1e-3, 0.10, 21),
        baseline_times=lognormal_times(MIN_TIMED_REPS, 1e-3, 0.10, 22),
    )
    result = PerformanceOracle().run(ctx_factory())

    assert result.verdict == "PASS", result.reason
    speedup = result.evidence["per_shape"][0]["speedup"]
    assert speedup["classification"] == "inconclusive"
    assert speedup["ci"]["low"] < 1.0 < speedup["ci"]["high"]
    assert "inconclusive" in result.reason
    assert "faster" not in result.reason


def test_a_real_win_is_claimed_with_its_interval(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
) -> None:
    stub_environment(
        candidate_times=lognormal_times(MIN_TIMED_REPS, 1e-3, 0.04, 31),
        baseline_times=lognormal_times(MIN_TIMED_REPS, 2e-3, 0.04, 32),
    )
    result = PerformanceOracle().run(ctx_factory())

    speedup = result.evidence["per_shape"][0]["speedup"]
    assert speedup["classification"] == "faster"
    assert speedup["ci"]["low"] > 1.0
    assert speedup["point"] == pytest.approx(2.0, rel=0.1)
    assert speedup["definition"].startswith("median(baseline) / median(candidate)")


def test_a_demanded_speedup_that_is_only_inconclusive_does_not_pass(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
) -> None:
    stub_environment(
        candidate_times=lognormal_times(MIN_TIMED_REPS, 1e-3, 0.10, 41),
        baseline_times=lognormal_times(MIN_TIMED_REPS, 1e-3, 0.10, 42),
    )
    result = PerformanceOracle().run(ctx_factory(extras={"min_speedup": 1.0}))

    assert result.verdict == "SKIP"
    assert "inconclusive rather than demonstrated" in result.reason


def test_a_demanded_speedup_that_is_out_of_reach_fails(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
) -> None:
    stub_environment(
        candidate_times=lognormal_times(MIN_TIMED_REPS, 2e-3, 0.04, 51),
        baseline_times=lognormal_times(MIN_TIMED_REPS, 1e-3, 0.04, 52),
    )
    result = PerformanceOracle().run(ctx_factory(extras={"min_speedup": 2.0}))

    assert result.verdict == "FAIL"
    assert "cannot reach the demanded" in result.reason
    assert result.evidence["min_speedup"] == 2.0


def test_a_candidate_that_cannot_be_timed_fails(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
) -> None:
    stub_environment(
        candidate_run=TimingRun(
            ok=False,
            reason="candidate raised during the timed loop: ZeroDivisionError: division by zero",
        )
    )
    result = PerformanceOracle().run(ctx_factory())

    assert result.verdict == "FAIL"
    assert "ZeroDivisionError" in result.reason
    assert result.evidence["per_shape"][0]["status"] == "fail"


def test_a_machine_limit_skips_instead_of_blaming_the_candidate(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
) -> None:
    stub_environment(
        candidate_run=TimingRun(
            ok=False,
            reason="the timing sandbox did not complete: CUDA error: out of memory",
        )
    )
    result = PerformanceOracle().run(ctx_factory())

    assert result.verdict == "SKIP"
    assert "out of memory" in result.reason
    assert result.evidence["shapes_timed"] == []
    assert result.evidence["shapes_not_timed"][0]["reason"]


def test_roofline_fractions_are_absent_when_the_peaks_were_not_measured(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
) -> None:
    """An unmeasured denominator yields no fraction at all -- not a zero."""
    stub_environment(
        peaks=AchievablePeaks(
            device="cpu",
            dtype="float32",
            bandwidth=BandwidthProbe(None, "nvidia-smi reported no device", device="cpu"),
            flops_per_s=None,
            flops_reason="the matmul probe could not allocate",
            measured_utc="2026-01-01T00:00:00+00:00",
        )
    )
    result = PerformanceOracle().run(ctx_factory())

    assert result.verdict == "PASS", result.reason
    roofline = result.evidence["per_shape"][0]["roofline"]
    assert "achieved_bw_frac" not in roofline
    assert "achieved_flops_frac" not in roofline
    assert "no device" in roofline["achieved_bw_frac_absent"]
    assert "could not allocate" in roofline["achieved_flops_frac_absent"]
    # The numerator is still a fact and stays.
    assert roofline["achieved_bw_bytes_per_s"] > 0.0


def test_ncu_counters_are_omitted_from_the_evidence_when_ncu_did_not_run(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
) -> None:
    stub_environment()
    result = PerformanceOracle().run(ctx_factory())

    ncu = result.evidence["ncu"]
    assert ncu["available"] is False
    assert "counters" not in ncu
    assert ncu["reason"]
    assert "ncu" not in result.capabilities_used


def test_shapes_without_a_sweep_skip_with_a_reason(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
    tiny_seed: SeedSpec,
    make_task: Callable[..., Task],
) -> None:
    stub_environment()
    tiny_seed.shape_sweep = []
    task = make_task(detect_shapes=[], decoy_shapes=[])
    result = PerformanceOracle().run(ctx_factory(task=task))

    assert result.verdict == "SKIP"
    assert "no shapes to time" in result.reason


def test_an_empty_candidate_is_skipped_not_timed(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
) -> None:
    recorder = stub_environment()
    result = PerformanceOracle().run(ctx_factory("   \n"))

    assert result.verdict == "SKIP"
    assert "candidate source is empty" in result.reason
    assert recorder["calls"] == []


def test_explicit_shapes_override_the_sweep(
    ctx_factory: Callable[..., OracleContext],
    stub_environment: Callable[..., dict[str, Any]],
) -> None:
    stub_environment()
    shape = ShapeSpec(name="tiny_probe", kwargs={"rows": 2, "cols": 5, "dtype": "float32"})
    result = PerformanceOracle().run(ctx_factory(extras={"perf_shapes": [shape]}))

    assert result.verdict == "PASS", result.reason
    assert result.evidence["shapes_timed"] == ["tiny_probe"]
    assert result.evidence["per_shape"][0]["roofline"]["bytes_moved"] == 2 * 5 * 4
