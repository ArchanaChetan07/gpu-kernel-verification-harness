"""O2 - performance, measured rather than asserted.

A performance claim is only as good as the conditions under which it was taken,
so this oracle reports the conditions with the same weight as the number:

* **A single number is not a measurement.** Every latency is a median over at
  least ``MIN_TIMED_REPS`` timed repetitions (after at least ``MIN_WARMUP_REPS``
  discarded warmup reps) with a percentile bootstrap confidence interval. The
  raw per-rep times stay in the evidence so the interval can be recomputed.
* **An unlocked clock is a weaker claim, and it must look weaker.** The bootstrap
  only sees the dispersion inside the measurement window. When ``nvidia-smi
  -lgc`` did not take -- no elevation, a laptop, a locked-down host -- the
  governor is free to drift on a timescale longer than that window, and the
  resampled interval therefore *understates* the uncertainty. The verdict may
  still be PASS, but ``clock_locked=False`` is recorded and the reported CI is
  widened by ``UNLOCKED_CLOCK_CI_INFLATION``; the un-inflated interval is kept
  beside it so nothing is hidden.
* **The roofline denominator is measured, never quoted.** ``achieved_bw_frac``
  and ``achieved_flops_frac`` divide by a STREAM-triad bandwidth probe and a
  large-matmul FLOP/s probe run on *this* device, cached for the session. Spec
  sheets are not admissible: on the sm_75 card this project runs on, fp16 is
  slower than fp32, so no dtype ordering is assumed either - each dtype is
  probed separately and the probe is recorded with the number it produced.
* **Absent counters are absent.** ``ncu`` usually needs elevation. When it
  cannot run, the failure reason is recorded and the ``counters`` key is simply
  not present. Emitting zeros would be indistinguishable from a kernel that
  really did spill nothing.
* **A speedup without an interval is not a claim.** The ratio of medians carries
  its own bootstrap CI; if that CI straddles 1.0 the result is reported as
  ``inconclusive`` and no win is claimed.

Candidate code is timed in a subprocess (``crucible.runner.sandbox.run_source``)
and never executed in the grading interpreter. The peak probes are first-party
code in this module and do run in-process; the evidence says so rather than
implying both were sandboxed.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..capabilities import Capabilities
from ..config import Config
from ..runner.determinism import ClockLock, lock_clocks, purge_autotune_cache
from ..runner.sandbox import input_refs_from, run_source
from ..schema import OracleResult, ShapeSpec, Task
from .base import OracleContext, register_oracle

logger = logging.getLogger(__name__)

ORACLE_ID = "O2"

#: Floors the contract commits to. Config may raise them, never lower them: the
#: oracle's claim is a claim about this many repetitions.
MIN_WARMUP_REPS = 10
MIN_TIMED_REPS = 30

#: How much the reported CI is stretched when the clocks were not pinned.
#:
#: The factor is not a statistical correction - there is no distributional model
#: for a DVFS governor - it is a deliberate, documented penalty. A Turing part
#: left free-running moves between its sustained-thermal floor and its boost
#: ceiling by roughly a factor of two, and a drift slower than the measurement
#: window contributes nothing to the resampled spread. Doubling the geometric
#: half-width makes an unpinned measurement quantitatively weaker than a pinned
#: one instead of merely annotated as such.
UNLOCKED_CLOCK_CI_INFLATION = 2.0
UNLOCKED_CLOCK_CI_REASON = (
    "clocks were not locked (nvidia-smi -lgc did not take), so DVFS drift on a timescale "
    "longer than the measurement window is invisible to the bootstrap; the interval is "
    f"widened geometrically by {UNLOCKED_CLOCK_CI_INFLATION}x to keep the claim honest"
)

#: ncu metrics: achieved occupancy, DRAM throughput, local (spill) traffic.
NCU_METRICS: tuple[str, ...] = (
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
)

NCU_LABELS: dict[str, str] = {
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct_of_peak",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum": "spill_load_bytes",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum": "spill_store_bytes",
}

NCU_TIMEOUT_S = 300.0
NCU_JOB_ENV = "CRUCIBLE_O2_JOB"
_DEFAULT_JOB_NAME = "o2_timing_job.json"
_NCU_JOB_NAME = "o2_ncu_job.json"
_NCU_TARGET_NAME = "o2_ncu_target.py"

#: How many shapes are timed by default. Timing is expensive and the sweep is
#: authored small-to-large, so the default is the largest shape only.
_DEFAULT_MAX_SHAPES = 1

#: Substrings meaning "this machine could not run the case", as opposed to "the
#: candidate is broken". These downgrade a shape to SKIP, never to PASS.
_INFRA_MARKERS: tuple[str, ...] = (
    "out of memory",
    "outofmemoryerror",
    "no cuda gpus are available",
    "cuda unavailable",
    "torch not compiled with cuda",
    "no kernel image is available",
    "dll load failed",
    "libtriton",
    "cannot allocate memory",
    "unable to allocate",
    "insufficient shared memory",
    "no such device",
    "torch could not be imported",
    "torch.cuda.is_available() is false",
)


def _looks_like_infrastructure(text: str) -> str | None:
    low = (text or "").lower()
    for marker in _INFRA_MARKERS:
        if marker in low:
            return marker
    return None


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def _accepted_kwargs(fn: Callable[..., Any], wanted: Mapping[str, Any]) -> dict[str, Any]:
    """Pass only the keyword arguments the callable actually declares."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(wanted)
    return {k: v for k, v in wanted.items() if k in params}


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BootstrapCI:
    """A point estimate with a percentile-bootstrap interval around it.

    ``inflation`` records a deliberate widening (see
    ``UNLOCKED_CLOCK_CI_INFLATION``) and ``raw_low``/``raw_high`` keep the
    un-inflated endpoints so a reader can see exactly what was done.
    """

    point: float
    low: float
    high: float
    level: float = 0.95
    resamples: int = 0
    n: int = 0
    statistic: str = "median"
    method: str = "percentile bootstrap"
    inflation: float = 1.0
    inflation_reason: str = ""
    raw_low: float | None = None
    raw_high: float | None = None

    @property
    def width(self) -> float:
        return float(self.high - self.low)

    @property
    def relative_width(self) -> float:
        return float(self.width / self.point) if self.point else float("inf")

    def contains(self, value: float) -> bool:
        return bool(self.low <= value <= self.high)

    @property
    def straddles_one(self) -> bool:
        return self.contains(1.0)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "point": float(self.point),
            "low": float(self.low),
            "high": float(self.high),
            "level": float(self.level),
            "resamples": int(self.resamples),
            "n_samples": int(self.n),
            "statistic": self.statistic,
            "method": self.method,
            "width": self.width,
            "relative_width": self.relative_width,
            "inflation_factor": float(self.inflation),
        }
        if self.inflation_reason:
            out["inflation_reason"] = self.inflation_reason
        if self.raw_low is not None and self.raw_high is not None:
            out["uninflated_low"] = float(self.raw_low)
            out["uninflated_high"] = float(self.raw_high)
        return out


def _percentile_ci(
    draws: np.ndarray, point: float, level: float, resamples: int, n: int, statistic: str
) -> BootstrapCI:
    alpha = (1.0 - float(level)) / 2.0
    low = float(np.quantile(draws, alpha))
    high = float(np.quantile(draws, 1.0 - alpha))
    return BootstrapCI(
        point=float(point),
        low=low,
        high=high,
        level=float(level),
        resamples=int(resamples),
        n=int(n),
        statistic=statistic,
    )


def bootstrap_median_ci(
    samples: Sequence[float],
    resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for the median of ``samples``.

    ``seed`` makes the interval reproducible: two runs of the report on the same
    times must not disagree about the interval they printed.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_median_ci requires at least one sample")
    point = float(np.median(arr))
    if arr.size == 1:
        return BootstrapCI(
            point=point,
            low=point,
            high=point,
            level=float(level),
            resamples=0,
            n=1,
            statistic="median",
            method="degenerate (a single sample carries no dispersion)",
        )
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(resamples), arr.size))
    draws = np.median(arr[idx], axis=1)
    return _percentile_ci(draws, point, level, int(resamples), int(arr.size), "median")


def bootstrap_ratio_ci(
    numerator: Sequence[float],
    denominator: Sequence[float],
    resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """CI for ``median(numerator) / median(denominator)``.

    The two sample sets are resampled independently, which is what the two
    timing runs actually were. A speedup reported without this interval is an
    anecdote.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if num.size == 0 or den.size == 0:
        raise ValueError("bootstrap_ratio_ci requires at least one sample on each side")
    den_median = float(np.median(den))
    if den_median <= 0.0:
        raise ValueError("bootstrap_ratio_ci requires a positive denominator median")
    point = float(np.median(num)) / den_median
    if num.size == 1 or den.size == 1:
        return BootstrapCI(
            point=point,
            low=point,
            high=point,
            level=float(level),
            resamples=0,
            n=int(min(num.size, den.size)),
            statistic="ratio of medians",
            method="degenerate (a single sample carries no dispersion)",
        )
    rng = np.random.default_rng(int(seed))
    num_draws = np.median(num[rng.integers(0, num.size, size=(int(resamples), num.size))], axis=1)
    den_draws = np.median(den[rng.integers(0, den.size, size=(int(resamples), den.size))], axis=1)
    safe = den_draws > 0.0
    if not bool(safe.any()):
        raise ValueError("every bootstrap denominator resample was non-positive")
    draws = num_draws[safe] / den_draws[safe]
    ci = _percentile_ci(
        draws, point, level, int(resamples), int(min(num.size, den.size)), "ratio of medians"
    )
    return ci


def widen_ci(ci: BootstrapCI, factor: float, reason: str) -> BootstrapCI:
    """Stretch an interval about its point estimate, recording why.

    Geometric for strictly positive quantities (times, rates and ratios all
    are), which keeps a widened ratio interval positive and symmetric in log
    space. The original endpoints are preserved in ``raw_low``/``raw_high``.
    """
    f = float(factor)
    if f <= 1.0:
        return ci
    raw_low = ci.raw_low if ci.raw_low is not None else ci.low
    raw_high = ci.raw_high if ci.raw_high is not None else ci.high
    point = ci.point
    if point > 0.0 and ci.low > 0.0 and ci.high > 0.0:
        low = point * (ci.low / point) ** f
        high = point * (ci.high / point) ** f
    else:
        low = point - f * (point - ci.low)
        high = point + f * (ci.high - point)
    return replace(
        ci,
        low=float(low),
        high=float(high),
        inflation=f,
        inflation_reason=reason,
        raw_low=float(raw_low),
        raw_high=float(raw_high),
    )


def classify_ratio(ci: BootstrapCI) -> str:
    """``faster`` | ``slower`` | ``inconclusive`` for a speedup interval."""
    if ci.low > 1.0:
        return "faster"
    if ci.high < 1.0:
        return "slower"
    return "inconclusive"


# --------------------------------------------------------------------------- #
# the sandbox timing harness
# --------------------------------------------------------------------------- #

#: Executed inside the sandbox child (op="exec"), and also invoked directly as a
#: script under ncu. The job path comes from an environment variable because
#: ``sys.argv[1]`` already belongs to the sandbox child protocol.
_TIMING_HARNESS = r'''"""CRUCIBLE O2 timing harness. Runs in a disposable interpreter."""
import hashlib
import json
import os
import time
import types
from pathlib import Path

import numpy as np

RESULT = {"ok": False, "reason": "harness did not complete", "times_s": []}


def _fail(reason):
    RESULT["ok"] = False
    RESULT["reason"] = str(reason)[:4000]


def _checksum(arr):
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()


def _to_numpy(value, torch):
    if torch is not None and isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        return t.contiguous().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bool, int, float, complex)):
        return np.asarray(value)
    return None


def _flatten(value):
    if isinstance(value, dict):
        return [(str(k), v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))]
    if isinstance(value, (list, tuple)):
        return [("out%d" % i, v) for i, v in enumerate(value)]
    return [("out", value)]


def _load_inputs(refs, torch, default_device):
    out = {}
    for name, ref in refs.items():
        if ref.get("kind", "json") == "json":
            out[name] = ref.get("value")
            continue
        arr = np.load(ref["path"], allow_pickle=False)
        if torch is None:
            out[name] = arr
            continue
        t = torch.from_numpy(arr)
        dt = ref.get("torch_dtype")
        if dt:
            t = t.to(getattr(torch, dt))
        t = t.to(ref.get("device", default_device))
        if ref.get("requires_grad"):
            t = t.detach().requires_grad_(True)
        out[name] = t
    return out


def _main():
    job_path = os.environ.get("CRUCIBLE_O2_JOB") or "o2_timing_job.json"
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    device = str(job.get("device", "cpu"))
    reps = int(job.get("reps", 30))
    warmup = int(job.get("warmup", 10))

    torch = None
    try:
        import torch as _torch
        torch = _torch
    except BaseException as exc:
        if device.startswith("cuda"):
            _fail("torch could not be imported, so a %r timing run is impossible: %r"
                  % (device, exc))
            return

    use_events = False
    if torch is not None and device.startswith("cuda"):
        if not torch.cuda.is_available():
            _fail("device %r was requested but torch.cuda.is_available() is False" % device)
            return
        use_events = True

    source = job.get("source", "")
    Path("o2_candidate.py").write_text(source, encoding="utf-8")
    mod = types.ModuleType("crucible_o2_candidate")
    mod.__file__ = "o2_candidate.py"
    try:
        exec(compile(source, "o2_candidate.py", "exec"), mod.__dict__)
    except BaseException as exc:
        _fail("candidate module failed to import: %s: %s" % (type(exc).__name__, exc))
        return

    entry = job.get("entry")
    fn = getattr(mod, entry, None)
    if not callable(fn):
        _fail("candidate does not define a callable named %r" % entry)
        return

    try:
        inputs = _load_inputs(job.get("inputs", {}), torch, device)
    except BaseException as exc:
        _fail("inputs could not be materialised: %s: %s" % (type(exc).__name__, exc))
        return

    style = job.get("call_style", "kwargs")
    order = job.get("arg_order") or sorted(inputs)

    def invoke():
        if style == "kwargs":
            return fn(**inputs)
        if style == "args":
            return fn(*[inputs[k] for k in order])
        if style == "single_dict":
            return fn(inputs)
        raise ValueError("unknown call_style %r" % style)

    # Warmup exists to pay for allocator growth, autotuning and lazy module
    # init. Its timings are discarded, never folded into the sample.
    try:
        for _ in range(warmup):
            invoke()
        if use_events:
            torch.cuda.synchronize()
    except BaseException as exc:
        _fail("candidate raised during warmup: %s: %s" % (type(exc).__name__, exc))
        return

    times = []
    out = None
    try:
        if use_events:
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
            for i in range(reps):
                torch.cuda.synchronize()
                starts[i].record()
                out = invoke()
                ends[i].record()
                torch.cuda.synchronize()
            times = [starts[i].elapsed_time(ends[i]) / 1000.0 for i in range(reps)]
        else:
            for _ in range(reps):
                t0 = time.perf_counter()
                out = invoke()
                times.append(time.perf_counter() - t0)
    except BaseException as exc:
        _fail("candidate raised during the timed loop: %s: %s" % (type(exc).__name__, exc))
        return

    # Checksumming the final output is the forced consumer: the bytes must
    # exist to be hashed, so the timed work cannot have been eliminated.
    checksums = {}
    try:
        for name, part in _flatten(out):
            arr = _to_numpy(part, torch)
            checksums[name] = (_checksum(arr) if arr is not None
                               else "opaque:%s" % type(part).__name__)
    except BaseException as exc:
        RESULT["checksum_error"] = "%s: %s" % (type(exc).__name__, exc)

    RESULT.update({
        "ok": True,
        "reason": "",
        "times_s": [float(t) for t in times],
        "timer": "cuda_event" if use_events else "perf_counter",
        "device": device,
        "reps": reps,
        "warmup": warmup,
        "checksums": checksums,
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
    })


try:
    _main()
except BaseException as exc:
    import traceback as _tb
    _fail("%s: %s | %s" % (type(exc).__name__, exc, _tb.format_exc()[-2000:]))
'''


@dataclass
class TimingRun:
    """What one timing attempt produced. ``ok=False`` always carries a reason."""

    ok: bool
    times_s: list[float] = field(default_factory=list)
    timer: str = ""
    device: str = "cpu"
    reps: int = 0
    warmup: int = 0
    checksums: dict[str, str] = field(default_factory=dict)
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def median_s(self) -> float:
        if not self.times_s:
            raise ValueError("timing run produced no samples; there is no median")
        return _median(self.times_s)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "timer": self.timer,
            "device": self.device,
            "reps": self.reps,
            "warmup": self.warmup,
            "n_samples": len(self.times_s),
        }
        if self.times_s:
            out["times_s"] = [float(t) for t in self.times_s]
            out["median_s"] = self.median_s
            out["min_s"] = float(min(self.times_s))
            out["max_s"] = float(max(self.times_s))
        if self.checksums:
            out["checksums"] = dict(self.checksums)
        if self.reason:
            out["reason"] = self.reason
        if self.detail:
            out["detail"] = dict(self.detail)
        return out


def build_timing_job(
    source: str,
    entry: str,
    input_refs: Mapping[str, Any],
    device: str,
    reps: int,
    warmup: int,
    call_style: str = "kwargs",
    arg_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "entry": entry,
        "inputs": dict(input_refs),
        "device": device,
        "reps": int(reps),
        "warmup": int(warmup),
        "call_style": call_style,
        "arg_order": list(arg_order) if arg_order else None,
    }


def time_in_sandbox(
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
    env_extra: Mapping[str, str] | None = None,
) -> TimingRun:
    """Time ``entry`` in a fresh interpreter. Never raises for candidate defects.

    CUDA runs are timed with ``torch.cuda.Event`` around a synchronised region;
    CPU runs with ``perf_counter``. Warmup reps are discarded inside the child,
    so they cannot leak into the sample.
    """
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    refs = dict(input_refs) if input_refs is not None else input_refs_from(
        dict(inputs or {}), wd / "in", device=device
    )
    job = build_timing_job(source, entry, refs, device, reps, warmup, call_style, arg_order)
    (wd / _DEFAULT_JOB_NAME).write_text(
        json.dumps(job, default=str, indent=2), encoding="utf-8"
    )

    child_env: dict[str, str] = {NCU_JOB_ENV: _DEFAULT_JOB_NAME}
    if env_extra:
        child_env.update({str(k): str(v) for k, v in env_extra.items()})
    res = run_source(
        _TIMING_HARNESS,
        workdir=wd,
        timeout_s=float(timeout_s),
        seed=int(seed),
        env_extra=child_env,
    )
    if not res.ok:
        return TimingRun(
            ok=False,
            device=device,
            reps=int(reps),
            warmup=int(warmup),
            reason=f"the timing sandbox did not complete: {res.message}",
            detail={"stderr_tail": res.stderr[-2000:], "timed_out": res.timed_out},
        )
    payload = res.value.get("result") if isinstance(res.value, dict) else None
    if not isinstance(payload, dict):
        return TimingRun(
            ok=False,
            device=device,
            reps=int(reps),
            warmup=int(warmup),
            reason=f"the timing harness returned {type(payload).__name__}, not a result mapping",
            detail={"stderr_tail": res.stderr[-2000:]},
        )
    if not payload.get("ok"):
        return TimingRun(
            ok=False,
            device=device,
            reps=int(reps),
            warmup=int(warmup),
            reason=str(payload.get("reason") or "the timing harness reported failure without a reason"),
            detail={"stderr_tail": res.stderr[-2000:]},
        )
    times = [float(t) for t in (payload.get("times_s") or [])]
    if not times:
        return TimingRun(
            ok=False,
            device=device,
            reps=int(reps),
            warmup=int(warmup),
            reason="the timing harness reported success but produced no samples",
        )
    detail: dict[str, Any] = {"workdir": str(wd)}
    if payload.get("torch_version"):
        detail["torch_version"] = payload["torch_version"]
    if payload.get("checksum_error"):
        detail["checksum_error"] = payload["checksum_error"]
    return TimingRun(
        ok=True,
        times_s=times,
        timer=str(payload.get("timer") or ""),
        device=str(payload.get("device") or device),
        reps=int(payload.get("reps") or reps),
        warmup=int(payload.get("warmup") or warmup),
        checksums={str(k): str(v) for k, v in (payload.get("checksums") or {}).items()},
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# measured achievable peaks (the roofline denominator)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BandwidthProbe:
    """A measured bandwidth, or an explicit absence with the reason."""

    bytes_per_s: float | None
    reason: str = ""
    method: str = "stream_triad"
    device: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # consumed verbatim by O3's SKIP reason
        return (
            f"BandwidthProbe(bytes_per_s={self.bytes_per_s!r}, device={self.device!r}, "
            f"method={self.method!r}, reason={self.reason!r})"
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": self.method, "device": self.device}
        if self.bytes_per_s is None:
            out["measured"] = False
            out["reason"] = self.reason or "bandwidth was not measured"
        else:
            out["measured"] = True
            out["bytes_per_s"] = float(self.bytes_per_s)
            out.update(self.detail)
        return out


@dataclass(frozen=True)
class AchievablePeaks:
    """Both roofline denominators for one (device, dtype), measured here."""

    device: str
    dtype: str
    bandwidth: BandwidthProbe
    flops_per_s: float | None
    flops_reason: str = ""
    flops_detail: dict[str, Any] = field(default_factory=dict)
    measured_utc: str = ""

    @property
    def bytes_per_s(self) -> float | None:
        return self.bandwidth.bytes_per_s

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "device": self.device,
            "dtype": self.dtype,
            "measured_utc": self.measured_utc,
            "source": (
                "measured on this device by crucible.oracles.o2_perf "
                "(STREAM-triad and large-matmul probes); no spec-sheet numbers are used"
            ),
            "bandwidth": self.bandwidth.as_dict(),
        }
        if self.flops_per_s is None:
            out["compute"] = {
                "measured": False,
                "reason": self.flops_reason or "FLOP/s was not measured",
                "method": "square matmul",
            }
        else:
            out["compute"] = {
                "measured": True,
                "flops_per_s": float(self.flops_per_s),
                "method": "square matmul",
                **self.flops_detail,
            }
        out["dtype_note"] = (
            "each dtype is probed separately; no ordering between fp32/fp16/bf16 is assumed "
            "(on sm_75 fp16 measures slower than fp32)"
        )
        return out


_PEAK_CACHE: dict[str, AchievablePeaks] = {}

#: Probe budgets. Small enough to run once per session without dominating a
#: grading run, large enough that a single kernel launch is not the measurement.
_BW_TARGET_BYTES = {"cuda": 192 * 1024 * 1024, "cpu": 48 * 1024 * 1024}
_MATMUL_SIZE = {"cuda": 2048, "cpu": 512}
_PROBE_WARMUP = 3
_PROBE_REPS = 11


def clear_peak_cache() -> None:
    """Drop the session cache of measured peaks (used by tests and ``--refresh``)."""
    _PEAK_CACHE.clear()


def _probe_times(call: Callable[[], Any], device: str, warmup: int, reps: int) -> tuple[list[float], str]:
    """Time a first-party callable in this process. CUDA events on CUDA."""
    import torch

    use_events = device.startswith("cuda") and torch.cuda.is_available()
    for _ in range(warmup):
        call()
    if use_events:
        torch.cuda.synchronize()
    times: list[float] = []
    if use_events:
        for _ in range(reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            call()
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)) / 1000.0)
        return times, "cuda_event"
    for _ in range(reps):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    return times, "perf_counter"


def _probe_bandwidth(device: str, dtype_name: str, target_bytes: int) -> BandwidthProbe:
    """STREAM triad ``a = b + s*c`` on this device; 3 arrays touched per rep."""
    try:
        import torch
    except ImportError as exc:
        return BandwidthProbe(None, f"torch is not importable: {exc}", device=device)

    try:
        dtype = getattr(torch, dtype_name)
        itemsize = int(torch.empty((), dtype=dtype).element_size())
    except (AttributeError, TypeError, RuntimeError) as exc:
        return BandwidthProbe(None, f"dtype {dtype_name!r} is unusable here: {exc}", device=device)

    budget = int(target_bytes)
    last_error = ""
    for _attempt in range(3):
        n = max(1024, budget // (3 * itemsize))
        try:
            b = torch.randn(n, device=device, dtype=torch.float32).to(dtype)
            c = torch.randn(n, device=device, dtype=torch.float32).to(dtype)
            a = torch.empty_like(b)
            scalar = 3.0

            def call() -> None:
                torch.add(b, c, alpha=scalar, out=a)

            times, timer = _probe_times(call, device, _PROBE_WARMUP, _PROBE_REPS)
        except (RuntimeError, MemoryError, TypeError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            budget //= 2
            if budget < 3 * itemsize * 1024:
                break
            continue
        median_s = _median(times)
        if median_s <= 0.0:
            return BandwidthProbe(
                None,
                "the STREAM-triad probe measured a non-positive median time; the timer "
                "resolution is coarser than the probe, so no rate can be reported",
                device=device,
            )
        moved = 3 * n * itemsize
        return BandwidthProbe(
            bytes_per_s=float(moved) / median_s,
            method="stream_triad",
            device=device,
            detail={
                "dtype": dtype_name,
                "elements": int(n),
                "bytes_per_rep": int(moved),
                "median_s": median_s,
                "timer": timer,
                "reps": _PROBE_REPS,
                "warmup": _PROBE_WARMUP,
                "kernel": "a = b + 3.0 * c (three arrays touched)",
            },
        )
    return BandwidthProbe(
        None,
        f"the STREAM-triad probe could not allocate its buffers on {device!r}: {last_error}",
        device=device,
    )


def _probe_flops(device: str, dtype_name: str, size: int) -> tuple[float | None, str, dict[str, Any]]:
    """Large square matmul FLOP/s on this device, for this dtype only."""
    try:
        import torch
    except ImportError as exc:
        return None, f"torch is not importable: {exc}", {}

    try:
        dtype = getattr(torch, dtype_name)
    except AttributeError as exc:
        return None, f"dtype {dtype_name!r} is unknown to torch: {exc}", {}

    n = int(size)
    last_error = ""
    for _attempt in range(3):
        try:
            a = torch.randn(n, n, device=device, dtype=torch.float32).to(dtype)
            b = torch.randn(n, n, device=device, dtype=torch.float32).to(dtype)
            out = torch.empty((n, n), device=device, dtype=dtype)

            def call() -> None:
                torch.matmul(a, b, out=out)

            times, timer = _probe_times(call, device, _PROBE_WARMUP, max(5, _PROBE_REPS // 2))
        except (RuntimeError, MemoryError, TypeError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            n //= 2
            if n < 128:
                break
            continue
        median_s = _median(times)
        if median_s <= 0.0:
            return (
                None,
                "the matmul probe measured a non-positive median time; the timer resolution "
                "is coarser than the probe, so no rate can be reported",
                {},
            )
        flops = 2.0 * float(n) ** 3
        detail: dict[str, Any] = {
            "dtype": dtype_name,
            "size": n,
            "flops_per_rep": flops,
            "median_s": median_s,
            "timer": timer,
            "reps": max(5, _PROBE_REPS // 2),
            "warmup": _PROBE_WARMUP,
        }
        try:
            detail["allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        except (AttributeError, RuntimeError) as exc:
            logger.debug("could not read allow_tf32: %s", exc)
        return flops / median_s, "", detail
    return None, f"the matmul probe could not allocate on {device!r}: {last_error}", {}


def measure_achievable_peaks(
    device: str = "cuda",
    dtype: str = "float32",
    *,
    caps: Capabilities | None = None,
    cfg: Config | None = None,
    refresh: bool = False,
) -> AchievablePeaks:
    """Measure and cache this device's achievable bandwidth and FLOP/s.

    Cached per ``(device, dtype)`` for the process: the probes are a fixed
    property of the machine within a session, and re-running them per shape
    would cost more than the measurements they normalise.
    """
    dev = str(device or "cpu")
    dt = str(dtype or "float32")
    key = f"{dev}|{dt}"
    if not refresh:
        cached = _PEAK_CACHE.get(key)
        if cached is not None:
            return cached

    if dev.startswith("cuda") and caps is not None and not caps.cuda:
        probe = BandwidthProbe(
            None,
            f"device {dev!r} was requested but capability detection reports no CUDA device "
            f"({caps.detail('cuda') or 'torch.cuda.is_available() returned False'})",
            device=dev,
        )
        peaks = AchievablePeaks(
            device=dev,
            dtype=dt,
            bandwidth=probe,
            flops_per_s=None,
            flops_reason=probe.reason,
            measured_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        _PEAK_CACHE[key] = peaks
        return peaks

    family = "cuda" if dev.startswith("cuda") else "cpu"
    bandwidth = _probe_bandwidth(dev, dt, _BW_TARGET_BYTES[family])
    flops, flops_reason, flops_detail = _probe_flops(dev, dt, _MATMUL_SIZE[family])
    peaks = AchievablePeaks(
        device=dev,
        dtype=dt,
        bandwidth=bandwidth,
        flops_per_s=flops,
        flops_reason=flops_reason,
        flops_detail=flops_detail,
        measured_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    _PEAK_CACHE[key] = peaks
    return peaks


def measure_achievable_bandwidth(
    device: str = "cuda",
    caps: Capabilities | None = None,
    cfg: Config | None = None,
    dtype: str = "float32",
) -> BandwidthProbe:
    """The public entry O3's ``timing_sanity`` consumes.

    O3 reads ``.bytes_per_s``; when the probe could not run that attribute is
    ``None`` and the reason travels in the repr, so O3 SKIPs with a specific
    explanation instead of inventing a spec-sheet denominator.
    """
    return measure_achievable_peaks(device, dtype, caps=caps, cfg=cfg).bandwidth


# --------------------------------------------------------------------------- #
# ncu counters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NcuResult:
    """ncu's answer, or an explicit absence. Counters are never fabricated."""

    available: bool
    counters: dict[str, Any] | None = None
    reason: str = ""
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    stderr_tail: str = ""

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "available": self.available,
            "metrics_requested": list(NCU_METRICS),
        }
        if self.command:
            out["command"] = list(self.command)
        if self.returncode is not None:
            out["returncode"] = int(self.returncode)
        if self.reason:
            out["reason"] = self.reason
        if self.stderr_tail:
            out["stderr_tail"] = self.stderr_tail
        # The whole point: when ncu could not run there is no "counters" key at
        # all. A dict of zeros would be indistinguishable from a real profile of
        # a kernel that spilled nothing.
        if self.available and self.counters is not None:
            out["counters"] = dict(self.counters)
        return out


def parse_ncu_csv(text: str) -> dict[str, dict[str, Any]]:
    """Parse ``ncu --csv`` output into ``{metric: {unit, values, max, ...}}``.

    Returns ``{}`` when no metric table is present; an empty parse is reported
    as a failure by the caller rather than as "all counters were zero".
    """
    rows = list(csv.reader(io.StringIO(text or "")))
    header: list[str] | None = None
    header_index = -1
    for i, row in enumerate(rows):
        if "Metric Name" in row and "Metric Value" in row:
            header = row
            header_index = i
            break
    if header is None:
        return {}
    cols = {name: j for j, name in enumerate(header)}
    out: dict[str, dict[str, Any]] = {}
    for row in rows[header_index + 1 :]:
        if len(row) < len(header):
            continue
        name = row[cols["Metric Name"]].strip()
        if not name:
            continue
        raw = row[cols["Metric Value"]].strip().replace(",", "")
        unit = row[cols["Metric Unit"]].strip() if "Metric Unit" in cols else ""
        kernel = row[cols["Kernel Name"]].strip() if "Kernel Name" in cols else ""
        rec = out.setdefault(
            name, {"unit": unit, "values": [], "kernels": [], "unparsed": []}
        )
        try:
            value = float(raw)
        except ValueError:
            rec["unparsed"].append(raw)
            continue
        rec["values"].append(value)
        rec["kernels"].append(kernel)
    for name, rec in out.items():
        values = rec["values"]
        if values:
            rec["n_kernels"] = len(values)
            rec["max"] = max(values)
            rec["min"] = min(values)
            rec["mean"] = sum(values) / len(values)
        if not rec["unparsed"]:
            rec.pop("unparsed")
        if NCU_LABELS.get(name):
            rec["label"] = NCU_LABELS[name]
    return out


def collect_ncu_counters(
    ncu_path: str | None,
    target_argv: Sequence[str],
    *,
    workdir: Path | str,
    metrics: Sequence[str] = NCU_METRICS,
    timeout_s: float = NCU_TIMEOUT_S,
    env_extra: Mapping[str, str] | None = None,
) -> NcuResult:
    """Run ncu over ``target_argv``. Absence is recorded, never zero-filled."""
    if not ncu_path:
        return NcuResult(
            available=False,
            reason=(
                "ncu was not found by capability detection, so no hardware counters exist "
                "for this run"
            ),
        )
    argv = [
        str(ncu_path),
        "--csv",
        "--target-processes",
        "all",
        "--metrics",
        ",".join(metrics),
        *[str(a) for a in target_argv],
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(timeout_s),
            cwd=str(workdir),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return NcuResult(
            available=False,
            reason=f"ncu did not finish within {float(timeout_s):.0f}s and was killed",
            command=argv,
        )
    except OSError as exc:
        return NcuResult(
            available=False,
            reason=f"ncu could not be launched: {exc}",
            command=argv,
        )

    stderr_tail = (proc.stderr or "").strip()[-2000:]
    if proc.returncode != 0:
        return NcuResult(
            available=False,
            reason=(
                f"ncu exited {proc.returncode}; profiling commonly requires elevation "
                f"(ERR_NVGPUCTRPERM). stderr: {stderr_tail or '<empty>'}"
            ),
            command=argv,
            returncode=proc.returncode,
            stderr_tail=stderr_tail,
        )
    counters = parse_ncu_csv(proc.stdout or "")
    if not counters:
        return NcuResult(
            available=False,
            reason=(
                "ncu exited 0 but emitted no metric table; the requested metrics were not "
                "collected on this device"
            ),
            command=argv,
            returncode=proc.returncode,
            stderr_tail=stderr_tail,
        )
    return NcuResult(
        available=True,
        counters=counters,
        command=argv,
        returncode=proc.returncode,
        stderr_tail=stderr_tail,
    )


# --------------------------------------------------------------------------- #
# input construction
# --------------------------------------------------------------------------- #


def _shape_rng_seed(base: int, shape: ShapeSpec) -> int:
    return (int(base) + int(shape.key()[:8], 16)) % (2**31 - 1)


def _make_inputs(seed: Any, shape: ShapeSpec, device: str, rng_seed: int) -> dict[str, Any]:
    """Build the case inputs, seeded, tolerating simpler make_inputs signatures."""
    generator: Any = None
    try:
        import torch

        generator = torch.Generator(device="cpu" if device == "cpu" else device)
        generator.manual_seed(int(rng_seed))
    except (ImportError, RuntimeError, TypeError) as exc:
        logger.debug("no seeded generator for shape %s: %s", shape.name, exc)
    for kwargs in ({"device": device, "generator": generator}, {"device": device}, {}):
        try:
            out = seed.make_inputs(shape, **kwargs)
        except TypeError as exc:
            logger.debug("make_inputs rejected %s for %s: %s", sorted(kwargs), shape.name, exc)
            continue
        if not isinstance(out, dict):
            raise TypeError(
                f"seed {getattr(seed, 'id', '?')}: make_inputs returned "
                f"{type(out).__name__}, expected a dict of keyword arguments"
            )
        return out
    raise TypeError(
        f"seed {getattr(seed, 'id', '?')}: make_inputs accepted none of the supported "
        "signatures (shape[, device][, generator])"
    )


def _shape_dtype(shape: ShapeSpec, inputs: Mapping[str, Any]) -> str:
    """The dtype the peaks must be probed at. Declared first, observed second."""
    declared = shape.kwargs.get("dtype")
    if declared is not None:
        return str(declared).replace("torch.", "")
    for value in inputs.values():
        dt = getattr(value, "dtype", None)
        if dt is None:
            continue
        name = str(getattr(dt, "name", dt)).replace("torch.", "")
        if name.startswith(("float", "bfloat", "half")):
            return "float16" if name == "half" else name
    return "float32"


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #


class PerformanceOracle:
    """Median latency with a bootstrap CI, a measured roofline, and its conditions."""

    id = ORACLE_ID
    name = "measured performance: median + bootstrap CI against a measured roofline"
    required_caps: tuple[str, ...] = ("cuda",)

    def applies_to(self, task: Task) -> bool:
        # Which oracles run is decided by ``task.oracles`` in the dispatcher.
        return True

    # -- plan -------------------------------------------------------------- #

    def reps_and_warmup(self, ctx: OracleContext) -> tuple[int, int, dict[str, Any]]:
        """Config values, floored at the contract minimums. Floors are recorded."""
        want_reps = int(ctx.cfg.perf_reps)
        want_warmup = int(ctx.cfg.perf_warmup)
        reps = max(want_reps, MIN_TIMED_REPS)
        warmup = max(want_warmup, MIN_WARMUP_REPS)
        note: dict[str, Any] = {
            "reps_requested": want_reps,
            "warmup_requested": want_warmup,
            "reps_floor": MIN_TIMED_REPS,
            "warmup_floor": MIN_WARMUP_REPS,
            "floor_applied": bool(reps != want_reps or warmup != want_warmup),
        }
        if note["floor_applied"]:
            note["floor_reason"] = (
                "the oracle's claim is a claim about at least "
                f"{MIN_WARMUP_REPS} discarded warmup reps and {MIN_TIMED_REPS} timed reps; "
                "config may raise those floors but never lower them"
            )
        return reps, warmup, note

    def perf_shapes(self, ctx: OracleContext) -> list[ShapeSpec]:
        """Shapes to time, largest last. Held-out detect shapes are preferred."""
        override = ctx.extras.get("perf_shapes") or ctx.extras.get("shapes")
        if override:
            return list(override)
        pool: list[ShapeSpec] = list(ctx.task.detect_shapes) or list(
            getattr(ctx.seed, "shape_sweep", []) or []
        )
        if not pool:
            return []
        limit = max(1, int(ctx.extras.get("o2_max_shapes", _DEFAULT_MAX_SHAPES)))
        return pool[-limit:]

    # -- one shape --------------------------------------------------------- #

    def _measure_shape(
        self,
        ctx: OracleContext,
        shape: ShapeSpec,
        device: str,
        reps: int,
        warmup: int,
        inflation: float,
        inflation_reason: str,
    ) -> dict[str, Any]:
        seed = ctx.seed
        rng_seed = _shape_rng_seed(ctx.rng_seed, shape)
        rec: dict[str, Any] = {
            "name": shape.name,
            "shape_key": shape.key(),
            "kwargs": dict(shape.kwargs),
            "status": "skip",
            "reason": "",
            "device": device,
            "rng_seed": rng_seed,
        }
        started = time.perf_counter()

        def finish(status: str, reason: str = "") -> dict[str, Any]:
            rec["status"] = status
            if reason:
                rec["reason"] = reason
            rec["duration_s"] = time.perf_counter() - started
            return rec

        try:
            inputs = _make_inputs(seed, shape, device, rng_seed)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            return finish(
                "skip",
                f"inputs for shape {shape.name!r} could not be built: "
                f"{type(exc).__name__}: {exc}",
            )

        dtype = _shape_dtype(shape, inputs)
        rec["dtype"] = dtype
        timeout_s = float(
            ctx.extras.get("o2_timing_timeout_s", max(float(ctx.cfg.sandbox_timeout_s) * 4.0, 120.0))
        )
        base_dir = ctx.sub_workdir(f"{ORACLE_ID}/{shape.key()}")
        cand_dir = base_dir / "candidate"
        # Where the profiler will find a ready-made job describing exactly the
        # call that was timed. Recorded whether or not ncu ever runs.
        rec["timing_job_path"] = str(cand_dir / _DEFAULT_JOB_NAME)

        cand = self.time_source(
            ctx,
            ctx.candidate_src,
            inputs,
            workdir=cand_dir,
            device=device,
            reps=reps,
            warmup=warmup,
            timeout_s=timeout_s,
            seed=rng_seed,
        )
        rec["candidate"] = cand.as_dict()
        if not cand.ok:
            marker = _looks_like_infrastructure(cand.reason)
            if marker is not None:
                return finish(
                    "skip",
                    f"shape {shape.name!r} could not be timed on this machine "
                    f"(matched {marker!r}): {cand.reason}",
                )
            return finish(
                "fail",
                f"the candidate could not be timed on shape {shape.name!r}: {cand.reason}",
            )

        resamples = int(ctx.cfg.bootstrap_resamples)
        level = float(ctx.cfg.ci_level)
        raw_ci = bootstrap_median_ci(cand.times_s, resamples=resamples, level=level, seed=rng_seed)
        ci = widen_ci(raw_ci, inflation, inflation_reason)
        rec["candidate"]["ci"] = ci.as_dict()
        rec["candidate"]["ci_unwidened"] = raw_ci.as_dict()
        median_s = cand.median_s
        rec["median_s"] = median_s

        # -- roofline, from measured peaks only ----------------------------- #
        peaks = measure_achievable_peaks(device, dtype, caps=ctx.caps, cfg=ctx.cfg)
        roofline: dict[str, Any] = {"dtype": dtype, "peaks_measured_utc": peaks.measured_utc}
        moved = _seed_quantity(seed, "bytes_moved", shape)
        work = _seed_quantity(seed, "flops", shape)
        if moved is None:
            roofline["bytes_moved_absent"] = f"seed.bytes_moved is unavailable for {shape.name!r}"
        else:
            roofline["bytes_moved"] = int(moved)
            if median_s > 0.0:
                achieved = float(moved) / median_s
                roofline["achieved_bw_bytes_per_s"] = achieved
                if peaks.bytes_per_s:
                    roofline["achievable_bw_bytes_per_s"] = float(peaks.bytes_per_s)
                    roofline["achieved_bw_frac"] = achieved / float(peaks.bytes_per_s)
                else:
                    roofline["achieved_bw_frac_absent"] = (
                        peaks.bandwidth.reason or "no measured achievable bandwidth"
                    )
        if work is None:
            roofline["flops_absent"] = f"seed.flops is unavailable for {shape.name!r}"
        else:
            roofline["flops"] = int(work)
            if median_s > 0.0:
                achieved_f = float(work) / median_s
                roofline["achieved_flops_per_s"] = achieved_f
                if peaks.flops_per_s:
                    roofline["achievable_flops_per_s"] = float(peaks.flops_per_s)
                    roofline["achieved_flops_frac"] = achieved_f / float(peaks.flops_per_s)
                else:
                    roofline["achieved_flops_frac_absent"] = (
                        peaks.flops_reason or "no measured achievable FLOP/s"
                    )
        rec["roofline"] = roofline

        # -- speedup against the task's known-good baseline ------------------ #
        baseline_src = str(ctx.extras.get("baseline_src") or ctx.task.baseline_code or "")
        if baseline_src.strip() and baseline_src != ctx.candidate_src:
            base = self.time_source(
                ctx,
                baseline_src,
                inputs,
                workdir=base_dir / "baseline",
                device=device,
                reps=reps,
                warmup=warmup,
                timeout_s=timeout_s,
                seed=rng_seed,
            )
            rec["baseline"] = base.as_dict()
            if base.ok:
                try:
                    raw_ratio = bootstrap_ratio_ci(
                        base.times_s,
                        cand.times_s,
                        resamples=resamples,
                        level=level,
                        seed=rng_seed + 1,
                    )
                except ValueError as exc:
                    rec["speedup_absent"] = f"a speedup interval could not be formed: {exc}"
                else:
                    ratio = widen_ci(raw_ratio, inflation, inflation_reason)
                    rec["speedup"] = {
                        "definition": "median(baseline) / median(candidate); >1 means faster",
                        "point": ratio.point,
                        "ci": ratio.as_dict(),
                        "ci_unwidened": raw_ratio.as_dict(),
                        "classification": classify_ratio(ratio),
                    }
            else:
                rec["speedup_absent"] = f"the baseline could not be timed: {base.reason}"
        else:
            rec["speedup_absent"] = (
                "no distinct baseline source was available, so no speedup was computed"
            )

        return finish("pass")

    def time_source(
        self,
        ctx: OracleContext,
        source: str,
        inputs: Mapping[str, Any],
        *,
        workdir: Path,
        device: str,
        reps: int,
        warmup: int,
        timeout_s: float,
        seed: int,
    ) -> TimingRun:
        """Seam for the sandboxed timing harness (tests substitute this)."""
        return time_in_sandbox(
            source,
            str(getattr(ctx.seed, "entry", "")),
            inputs,
            workdir=workdir,
            device=device,
            reps=reps,
            warmup=warmup,
            timeout_s=timeout_s,
            seed=seed,
        )

    # -- ncu ---------------------------------------------------------------- #

    def collect_counters(
        self, ctx: OracleContext, device: str, timing_job_path: str | Path | None
    ) -> NcuResult:
        """Profile the job that was just timed, if ncu can run here at all."""
        if not device.startswith("cuda"):
            return NcuResult(
                available=False,
                reason=f"ncu profiles CUDA kernels; this measurement ran on {device!r}",
            )
        ncu_path = ctx.caps.ncu
        if not ncu_path:
            return NcuResult(
                available=False,
                reason=(
                    "ncu is not available on this machine "
                    f"({ctx.caps.detail('ncu') or 'not found by capability detection'})"
                ),
            )
        if timing_job_path is None:
            return NcuResult(
                available=False,
                reason="no timed shape produced an ncu target, so no counters were collected",
            )
        job_file = Path(timing_job_path)
        if not job_file.exists():
            return NcuResult(
                available=False,
                reason=(
                    f"the timed call was not described on disk at {job_file}, so ncu had no "
                    "reproducible target to profile"
                ),
            )
        try:
            job = json.loads(job_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return NcuResult(
                available=False, reason=f"the timing job at {job_file} is unreadable: {exc}"
            )
        wd = job_file.parent
        # A profiled run is orders of magnitude slower than a timed one, so the
        # profile deliberately uses far fewer reps than the measurement did.
        job["reps"] = int(ctx.extras.get("o2_ncu_reps", 3))
        job["warmup"] = int(ctx.extras.get("o2_ncu_warmup", 1))
        try:
            (wd / _NCU_TARGET_NAME).write_text(_TIMING_HARNESS, encoding="utf-8")
            (wd / _NCU_JOB_NAME).write_text(
                json.dumps(job, default=str, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            return NcuResult(
                available=False, reason=f"the ncu target could not be written to {wd}: {exc}"
            )
        return collect_ncu_counters(
            ncu_path,
            [sys.executable, _NCU_TARGET_NAME],
            workdir=wd,
            timeout_s=float(ctx.extras.get("o2_ncu_timeout_s", NCU_TIMEOUT_S)),
            env_extra={NCU_JOB_ENV: _NCU_JOB_NAME},
        )

    # -- run ---------------------------------------------------------------- #

    def run(self, ctx: OracleContext) -> OracleResult:
        device = str(ctx.device or "cpu")
        caps_used = ["cuda"]

        def result(verdict: str, reason: str, evidence: dict[str, Any]) -> OracleResult:
            return OracleResult(
                oracle=ORACLE_ID,
                verdict=verdict,  # type: ignore[arg-type]
                reason=reason,
                evidence=evidence,
                capabilities_used=list(caps_used),
            )

        base_evidence: dict[str, Any] = {
            "device": device,
            "rng_seed": ctx.rng_seed,
            "candidate_execution": (
                "subprocess sandbox (crucible.runner.sandbox.run_source); candidate code is "
                "never executed in the grading interpreter"
            ),
            "probe_execution": (
                "in-process (the STREAM-triad and matmul peak probes are first-party code in "
                "crucible.oracles.o2_perf, not candidate input)"
            ),
        }

        if not ctx.caps.cuda:
            return result(
                "SKIP",
                f"O2 requires cuda; missing cuda "
                f"({ctx.caps.detail('cuda') or 'torch.cuda.is_available() returned False'}). "
                "A performance verdict taken on a device the task does not target is not a "
                "verdict about the task.",
                base_evidence,
            )
        if ctx.seed is None:
            return result(
                "SKIP",
                "no seed was supplied in the oracle context; there is nothing to time",
                base_evidence,
            )
        entry = str(getattr(ctx.seed, "entry", "") or "")
        base_evidence["seed_id"] = str(getattr(ctx.seed, "id", "?"))
        base_evidence["entry"] = entry
        if not entry:
            return result(
                "SKIP", f"seed {base_evidence['seed_id']!r} declares no entry point", base_evidence
            )
        if not ctx.candidate_src.strip():
            return result("SKIP", "candidate source is empty; nothing was executed", base_evidence)

        shapes = self.perf_shapes(ctx)
        if not shapes:
            return result(
                "SKIP",
                f"seed {base_evidence['seed_id']!r} and task {ctx.task.task_id!r} expose no "
                "shapes to time",
                base_evidence,
            )

        reps, warmup, rep_note = self.reps_and_warmup(ctx)
        base_evidence["reps"] = reps
        base_evidence["warmup"] = warmup
        base_evidence["rep_policy"] = rep_note
        base_evidence["ci_level"] = float(ctx.cfg.ci_level)
        base_evidence["ci_resamples"] = int(ctx.cfg.bootstrap_resamples)
        base_evidence["ci_method"] = "percentile bootstrap over the per-rep times"

        # Cold caches: an autotune cache carried over from an earlier run turns
        # the first timed shape into a measurement of the cache, not the kernel.
        purge = purge_autotune_cache()
        base_evidence["cache_purge"] = {
            "performed": True,
            "removed": purge.get("removed", []),
            "refused": purge.get("refused", []),
            "errors": purge.get("errors", []),
            "note": (
                "inductor/triton autotune caches were purged before timing so the first timed "
                "rep does not measure a warm cache from an earlier run"
            ),
        }

        lock_enabled = bool(ctx.extras.get("lock_clocks", True))
        records: list[dict[str, Any]] = []
        lock_record: dict[str, Any] = {}
        clock_locked = False
        inflation = UNLOCKED_CLOCK_CI_INFLATION
        inflation_reason = UNLOCKED_CLOCK_CI_REASON
        with lock_clocks(enabled=lock_enabled) as lock:
            clock_locked = bool(getattr(lock, "locked", False))
            if clock_locked:
                caps_used.append("can_lock_clocks")
            inflation = 1.0 if clock_locked else UNLOCKED_CLOCK_CI_INFLATION
            inflation_reason = "" if clock_locked else UNLOCKED_CLOCK_CI_REASON

            if ctx.cfg.require_clock_lock and not clock_locked:
                lock_record = _lock_record(lock, clock_locked)
                evidence = dict(base_evidence)
                evidence.update(
                    {
                        "clock_locked": False,
                        "clock_lock": lock_record,
                        "ci_inflation_factor": inflation,
                        "ci_inflation_reason": inflation_reason,
                    }
                )
                return result(
                    "SKIP",
                    "config sets require_clock_lock=True and the clocks could not be locked "
                    f"({lock_record.get('reason') or 'no reason reported'}); no timing was taken",
                    evidence,
                )

            for shape in shapes:
                records.append(
                    self._measure_shape(
                        ctx, shape, device, reps, warmup, inflation, inflation_reason
                    )
                )
            lock_record = _lock_record(lock, clock_locked)

        evidence = dict(base_evidence)
        evidence.update(
            {
                "clock_locked": clock_locked,
                "clock_lock": lock_record,
                "ci_inflation_factor": inflation,
                "ci_inflation_reason": inflation_reason,
                "measurement_strength": (
                    "clock-locked: the interval reflects the sampled dispersion"
                    if clock_locked
                    else "free-running clocks: a weaker claim, and the interval is widened to say so"
                ),
                "per_shape": records,
                "shapes_timed": [r["name"] for r in records if r["status"] == "pass"],
                "shapes_not_timed": [
                    {"name": r["name"], "reason": r["reason"]}
                    for r in records
                    if r["status"] != "pass"
                ],
            }
        )

        passed = [r for r in records if r["status"] == "pass"]
        failed = [r for r in records if r["status"] == "fail"]
        skipped = [r for r in records if r["status"] == "skip"]

        # ncu only after a timed shape exists to profile.
        target_rec: dict[str, Any] | None = passed[-1] if passed else None
        if target_rec is not None:
            ncu = self.collect_counters(ctx, device, target_rec.get("timing_job_path"))
        else:
            ncu = NcuResult(
                available=False,
                reason="no shape was timed, so there was nothing for ncu to profile",
            )
        if ncu.available:
            caps_used.append("ncu")
        evidence["ncu"] = ncu.as_dict()
        if target_rec is not None:
            evidence["ncu_target_shape"] = target_rec["name"]

        peak_dtypes = sorted({str(r.get("dtype")) for r in records if r.get("dtype")})
        evidence["peaks"] = [
            measure_achievable_peaks(device, dt, caps=ctx.caps, cfg=ctx.cfg).as_dict()
            for dt in peak_dtypes
        ]

        if failed:
            first = failed[0]
            return result(
                "FAIL",
                f"{len(failed)}/{len(records)} shape(s) could not be measured because the "
                f"candidate did not run: {first['name']}: {first['reason']}",
                evidence,
            )
        if not passed:
            reasons = "; ".join(f"{r['name']}: {r['reason']}" for r in skipped[:4])
            return result(
                "SKIP",
                f"no shape could be timed on this machine -- {reasons or 'no reason recorded'}",
                evidence,
            )
        if skipped:
            reasons = "; ".join(f"{r['name']}: {r['reason']}" for r in skipped[:4])
            return result(
                "SKIP",
                f"{len(skipped)}/{len(records)} shape(s) could not be timed, so the "
                f"performance picture is incomplete -- {reasons}",
                evidence,
            )

        # A demanded speedup that the interval cannot establish is not a pass.
        min_speedup = float(ctx.extras.get("min_speedup", 0.0) or 0.0)
        if min_speedup > 0.0:
            verdict, reason = _speedup_gate(passed, min_speedup)
            evidence["min_speedup"] = min_speedup
            if verdict != "PASS":
                return result(verdict, reason, evidence)

        head = passed[0]
        ci = head["candidate"]["ci"]
        claims = []
        for rec in passed:
            speed = rec.get("speedup")
            if speed:
                claims.append(f"{rec['name']}: speedup {speed['point']:.3f}x ({speed['classification']})")
        summary = "; ".join(claims)
        return result(
            "PASS",
            (
                f"{len(passed)} shape(s) timed over {reps} reps after {warmup} discarded warmup "
                f"reps; {head['name']} median {head['median_s']:.6e}s, "
                f"{int(float(ctx.cfg.ci_level) * 100)}% CI "
                f"[{ci['low']:.6e}, {ci['high']:.6e}]"
                + (f" (CI widened {ci['inflation_factor']:.1f}x: clocks not locked)" if not clock_locked else "")
                + (f"; {summary}" if summary else "")
            ),
            evidence,
        )


def _lock_record(lock: ClockLock | Any, locked: bool) -> dict[str, Any]:
    try:
        record = dict(lock.as_dict())
    except AttributeError:
        record = {"clock_locked": locked}
    record.setdefault("clock_locked", locked)
    if not locked:
        record.setdefault(
            "consequence",
            "the reported CI is widened; an unpinned measurement is a weaker claim",
        )
    return record


def _seed_quantity(seed: Any, name: str, shape: ShapeSpec) -> int | None:
    """``seed.bytes_moved(shape)`` / ``seed.flops(shape)``, or None with no guess."""
    fn = getattr(seed, name, None)
    if not callable(fn):
        return None
    try:
        value = int(fn(shape))
    except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError) as exc:
        logger.debug("seed.%s unavailable for shape %s: %s", name, shape.name, exc)
        return None
    return value if value > 0 else None


def _speedup_gate(records: Sequence[Mapping[str, Any]], threshold: float) -> tuple[str, str]:
    """Resolve a demanded speedup against every shape's ratio interval."""
    missing = [str(r["name"]) for r in records if not r.get("speedup")]
    if missing:
        return (
            "SKIP",
            f"a minimum speedup of {threshold:.3f}x was demanded but no ratio interval exists "
            f"for shape(s) {', '.join(missing)}: {records[0].get('speedup_absent', 'no baseline')}",
        )
    for rec in records:
        speed = rec["speedup"]
        ci = speed["ci"]
        if float(ci["high"]) < threshold:
            return (
                "FAIL",
                f"shape {rec['name']!r} cannot reach the demanded {threshold:.3f}x: the speedup "
                f"CI is [{ci['low']:.3f}, {ci['high']:.3f}] and its upper bound is below the "
                "threshold",
            )
    for rec in records:
        speed = rec["speedup"]
        ci = speed["ci"]
        if float(ci["low"]) < threshold:
            return (
                "SKIP",
                f"shape {rec['name']!r} did not establish the demanded {threshold:.3f}x: the "
                f"speedup CI is [{ci['low']:.3f}, {ci['high']:.3f}], which straddles the "
                "threshold, so the claim is inconclusive rather than demonstrated",
            )
    return "PASS", ""


O2 = PerformanceOracle()
register_oracle(O2)

__all__ = [
    "O2",
    "ORACLE_ID",
    "PerformanceOracle",
    "BootstrapCI",
    "bootstrap_median_ci",
    "bootstrap_ratio_ci",
    "widen_ci",
    "classify_ratio",
    "UNLOCKED_CLOCK_CI_INFLATION",
    "UNLOCKED_CLOCK_CI_REASON",
    "MIN_TIMED_REPS",
    "MIN_WARMUP_REPS",
    "TimingRun",
    "time_in_sandbox",
    "build_timing_job",
    "AchievablePeaks",
    "BandwidthProbe",
    "measure_achievable_peaks",
    "measure_achievable_bandwidth",
    "clear_peak_cache",
    "NcuResult",
    "NCU_METRICS",
    "NCU_LABELS",
    "collect_ncu_counters",
    "parse_ncu_csv",
]
