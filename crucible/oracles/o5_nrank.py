"""O5 - N-rank equivalence: does adding ranks change the run?

The claim a distributed training implementation makes is not "it converges". It
is **the world size is not an input to the result**. A correct data-parallel
step at 4 ranks must land on the same trajectory as the same step at 1 rank, up
to float non-associativity and nothing else. Every defect this oracle exists to
catch - a grad norm taken per shard, a dropped gradient all-reduce, a loss
averaged as a mean of per-rank means, optimizer state resharded wrongly on
resume - keeps the loss curve descending and violates exactly that claim.

The hard part is the word "up to". A comparison of two float training runs needs
a threshold, and a *guessed* threshold makes the whole oracle worthless in both
directions: too tight and every honest run is a failure, too loose and the
defects above sail through. So the tolerance is not a constant anywhere in this
file. It is **measured, every time, on the machine the grading runs on**:

1. The single-rank config is run ``nrank_noise_floor_runs`` times, identical in
   every respect except the order in which the microbatch is accumulated (the
   sample order is permuted by an index-space bijection, so the same numbers are
   summed in a different order and nothing else changes). Every pair of those
   runs differs by float non-associativity alone.
2. The largest per-step disagreement across those pairs **is** the noise floor,
   separately for the loss series, the global grad-norm series and the final
   parameter vector.
3. The N-rank run must stay within ``k`` times that floor, ``k`` from
   ``Config.nrank_k``.

The floor is reported with its provenance in every verdict, passing or failing,
because a threshold nobody can see is a threshold nobody can audit. Where a
measurement legitimately comes out at exactly zero - a run short enough that no
reordering changed a bit - the floor falls back to one unit-roundoff of the
quantity's own magnitude, and the evidence says ``ulp_fallback`` so that nobody
mistakes a derived bound for a measured one. It never falls back to a constant.

Three series are compared, not one, and the reason is timing: **the per-shard
grad-norm defect shows up in the norm series steps before it shows up in the
loss**, because the wrong norm has to first change a clipping decision and then
that decision has to move the weights far enough to move the loss. An oracle
that watched only the loss would report the same defect several steps later, or
- when clipping never triggers - not at all.

Two launchers, one marshalling path:

* ``spawn`` (production) - ``torch.multiprocessing.spawn`` over a real gloo
  process group rendezvoused through a ``FileStore``. gloo and not NCCL because
  the cheap path has to run on Windows too, and a ``FileStore`` and not TCP
  because it cannot collide with another run or be refused by a firewall.
  Every process is reaped in a ``finally``, including when a rank hangs.
* ``thread`` - the same candidate source over N threads with an exact,
  deterministic stand-in for ``torch.distributed``. This exists so the unit
  tests do not depend on process spawning under a test runner. It is a
  simulation and is labelled as one in the evidence; it shares
  :mod:`crucible.oracles.o5_worker` with the spawn path, so both launchers
  marshal arguments and summarise results through identical code.

A rank that hangs is a **finding**, not an infrastructure problem: a collective
reached by some ranks and not others is precisely the T3 defect this bank wants,
so a timeout is reported as FAIL with the ranks that never arrived, after the
process tree has been killed.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..schema import OracleResult, Task
from .base import OracleContext, register_oracle
from . import o5_worker

logger = logging.getLogger(__name__)

ORACLE_ID = "O5"

#: Series compared between the single-rank baseline and every N-rank run.
SERIES_KEYS: tuple[str, ...] = ("losses", "grad_norms")

#: The sub-checks reported per world size, in reporting order.
CHECK_NAMES: tuple[str, ...] = ("losses", "grad_norms", "final_params", "rank_agreement")

#: Domains for which N-rank equivalence is a meaningful question at all.
NRANK_DOMAINS: frozenset[str] = frozenset({"distributed", "checkpointing", "data_pipeline"})

#: A tiny model is a design requirement, not a convenience: the oracle has to
#: run several world sizes plus a repeated calibration inside one budget.
MAX_PARAMS = 1_000_000

_FP32_EPS = float(np.finfo(np.float32).eps)
_REAP_GRACE_S = 5.0

#: Failures that mean "this machine could not run it", not "the candidate is
#: wrong". These produce SKIP; everything else produces FAIL.
_INFRA_MARKERS: tuple[str, ...] = (
    "dll load failed",
    "cannot allocate memory",
    "unable to allocate",
    "out of memory",
    "no space left on device",
    "freeze_support",
    "the paging file is too small",
    "distributed package doesn't have",
    "gloo is not available",
    "not compiled with distributed",
)


def _looks_like_infrastructure(text: str) -> str | None:
    low = (text or "").lower()
    for marker in _INFRA_MARKERS:
        if marker in low:
            return marker
    return None


# --------------------------------------------------------------------------- #
# the workload
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NRankSpec:
    """The synthetic workload every rank count is run on.

    Fixed seed, synthetic data generated from a private ``torch.Generator`` (the
    global RNG is never touched, so nothing about the surrounding process can
    perturb a calibration), and a contiguous shard split by rank that is
    deliberately **uneven** whenever the sample count does not divide the world
    size - a mean-of-per-rank-means loss reduction is exactly correct on even
    shards and only wrong on uneven ones.

    The two weight matrices make the model two layers; ``w0`` is supplied as
    well so a one-layer candidate step (the collectives seed's gloo source) can
    be driven by the same payload. The signature of the candidate's entry
    selects which it receives.
    """

    n_samples: int = 50
    d: int = 12
    h: int = 16
    k: int = 4
    steps: int = 50
    lr: float = 0.5
    max_grad_norm: float = 0.05
    accum_dtype: str = "float32"
    seed: int = 1234
    init_scale: float = 0.3

    def n_params(self) -> int:
        return self.d * self.h + self.h * self.k + self.d * self.k

    def shard_sizes(self, world_size: int) -> list[int]:
        """Contiguous split; the remainder goes to the low ranks."""
        ws = int(world_size)
        if ws < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        base = self.n_samples // ws
        extra = self.n_samples % ws
        return [base + (1 if r < extra else 0) for r in range(ws)]

    def validate(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if min(self.d, self.h, self.k) <= 0:
            raise ValueError("d, h and k must be positive")
        if self.n_params() > MAX_PARAMS:
            raise ValueError(
                f"the O5 workload declares {self.n_params()} parameters, above the "
                f"{MAX_PARAMS} budget; N-rank equivalence is graded on a tiny model so "
                "that several world sizes and a repeated calibration fit one budget"
            )

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def permutation(n: int, salt: int) -> list[int]:
    """``i -> (a*i + b) mod n`` with ``gcd(a, n) == 1``: a bijection on
    ``range(n)`` derived from ``salt`` alone.

    An index-space bijection rather than an RNG shuffle so the calibration is
    reproducible on any machine and re-derivable from the evidence - the salts
    are recorded, and the permutation follows from them.
    """
    if n <= 1:
        return list(range(max(n, 0)))
    a = 1 + (int(salt) * 2246822519) % n
    while math.gcd(a, n) != 1:
        a += 1
        if a > n:
            a = 1
    b = (int(salt) * 2654435761) % n
    return [(a * i + b) % n for i in range(n)]


def build_payload(
    spec: NRankSpec, world_size: int, perm_salt: int = 0
) -> dict[str, Any]:
    """The JSON-able argument bundle one launch is driven with.

    ``perm_salt`` of 0 means the natural sample order. A non-zero salt permutes
    the samples, which reorders every reduction over the batch and changes
    nothing else - that is the noise-floor probe.
    """
    import torch

    spec.validate()
    gen = torch.Generator().manual_seed(int(spec.seed))
    n, d, h, k = spec.n_samples, spec.d, spec.h, spec.k
    x = torch.randn(n, d, generator=gen, dtype=torch.float32)
    y = torch.randint(0, k, (n,), generator=gen, dtype=torch.int64)
    w1_0 = spec.init_scale * torch.randn(d, h, generator=gen, dtype=torch.float32)
    w2_0 = spec.init_scale * torch.randn(h, k, generator=gen, dtype=torch.float32)
    w0 = spec.init_scale * torch.randn(d, k, generator=gen, dtype=torch.float32)

    if perm_salt:
        idx = torch.tensor(permutation(n, perm_salt), dtype=torch.int64)
        x = x[idx]
        y = y[idx]

    return {
        "x": [[float(v) for v in row] for row in x],
        "y": [int(v) for v in y],
        "w0": [[float(v) for v in row] for row in w0],
        "w1_0": [[float(v) for v in row] for row in w1_0],
        "w2_0": [[float(v) for v in row] for row in w2_0],
        "shard_sizes": spec.shard_sizes(world_size),
        "steps": int(spec.steps),
        "lr": float(spec.lr),
        "max_grad_norm": float(spec.max_grad_norm),
        "accum_dtype": str(spec.accum_dtype),
        "seed": int(spec.seed),
    }


# --------------------------------------------------------------------------- #
# launch results
# --------------------------------------------------------------------------- #


@dataclass
class LaunchResult:
    """One world-size run: what came back and, if nothing did, why."""

    world_size: int
    launcher: str
    ok: bool = False
    summaries: dict[int, dict[str, Any]] = field(default_factory=dict)
    duration_s: float = 0.0
    timed_out: bool = False
    error: str = ""
    #: "" | "candidate" | "hang" | "infrastructure" - decides FAIL vs SKIP.
    error_kind: str = ""
    pids: list[int] = field(default_factory=list)
    alive_after_reap: list[int] = field(default_factory=list)
    reaped: bool = True
    missing_ranks: list[int] = field(default_factory=list)
    rank_errors: dict[int, dict[str, Any]] = field(default_factory=dict)

    def series(self, key: str, rank: int = 0) -> list[float]:
        return [float(v) for v in self.summaries[rank][key]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "launcher": self.launcher,
            "ok": self.ok,
            "duration_s": round(self.duration_s, 6),
            "timed_out": self.timed_out,
            "error": self.error,
            "error_kind": self.error_kind,
            "pids": list(self.pids),
            "alive_after_reap": list(self.alive_after_reap),
            "reaped": self.reaped,
            "missing_ranks": list(self.missing_ranks),
            "rank_errors": {str(r): e for r, e in self.rank_errors.items()},
            "ranks_reporting": sorted(self.summaries),
        }


# --------------------------------------------------------------------------- #
# launcher 1: threads with an exact torch.distributed stand-in
# --------------------------------------------------------------------------- #


def _reduce_op_sum() -> Any:
    try:
        import torch.distributed as dist

        return dist.ReduceOp.SUM
    except (ImportError, AttributeError) as exc:  # pragma: no cover - gloo is gated on
        logger.debug("torch.distributed.ReduceOp unavailable: %s", exc)
        return "SUM"


class ThreadDist:
    """A deterministic ``torch.distributed`` for N threads in one process.

    Every collective is a two-phase barrier: deposit, rendezvous, reduce in rank
    order, rendezvous, write back in place. Reducing in rank order rather than in
    whatever order the threads arrive is what makes the simulation reproducible;
    it is a different summation order from gloo's, which is exactly why the noise
    floor is measured through the same launcher it will be applied to.

    A rank that does not enter a collective its peers entered breaks the barrier
    on the constructor timeout instead of hanging the test runner forever, and
    the resulting ``BrokenBarrierError`` is reported as the desync it is. Any
    collective this class does not implement raises rather than degrading
    silently - a simulated collective that quietly did nothing would manufacture
    a pass.
    """

    def __init__(self, world_size: int, timeout_s: float = 60.0) -> None:
        self.world_size = int(world_size)
        self.timeout_s = float(timeout_s)
        self.ReduceOp = type("_ReduceOpNS", (), {"SUM": _reduce_op_sum()})
        self._local = threading.local()
        self._barrier = threading.Barrier(self.world_size, timeout=self.timeout_s)
        self._slots: list[Any] = [None] * self.world_size
        self._collectives = 0

    # -- rank identity ----------------------------------------------------- #

    def bind(self, rank: int) -> None:
        self._local.rank = int(rank)

    def get_rank(self, group: Any = None) -> int:
        rank = getattr(self._local, "rank", None)
        if rank is None:
            raise RuntimeError("ThreadDist.get_rank called from a thread that was never bound")
        return int(rank)

    def get_world_size(self, group: Any = None) -> int:
        return self.world_size

    def is_initialized(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def get_backend(self, group: Any = None) -> str:
        return "gloo-threadsim"

    # -- collectives ------------------------------------------------------- #

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        self._barrier.wait()

    def all_reduce(
        self, tensor: Any, op: Any = None, group: Any = None, async_op: bool = False
    ) -> None:
        want = self.ReduceOp.SUM
        if op is not None and op is not want and str(op) != str(want):
            raise NotImplementedError(
                f"the O5 thread simulation implements SUM all-reduce only, not {op!r}; "
                "run with launcher='spawn' for the real gloo backend"
            )
        if async_op:
            raise NotImplementedError(
                "the O5 thread simulation has no async collectives; "
                "run with launcher='spawn' for the real gloo backend"
            )
        rank = self.get_rank()
        self._slots[rank] = tensor.detach().clone()
        self._barrier.wait()
        total = self._slots[0].clone()
        for r in range(1, self.world_size):
            total = total + self._slots[r]
        self._barrier.wait()
        tensor.copy_(total)
        self._collectives += 1

    def broadcast(self, tensor: Any, src: int = 0, group: Any = None, async_op: bool = False) -> None:
        rank = self.get_rank()
        if rank == int(src):
            self._slots[int(src)] = tensor.detach().clone()
        self._barrier.wait()
        source = self._slots[int(src)]
        self._barrier.wait()
        tensor.copy_(source)

    def abort(self) -> None:
        """Release peers blocked in a collective this rank will never reach."""
        self._barrier.abort()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise NotImplementedError(
            f"the O5 thread simulation does not implement torch.distributed.{name}; "
            "run with launcher='spawn' for the real gloo backend"
        )


#: Names a distributed source may have pulled directly into its own namespace.
_DIST_NAMES: tuple[str, ...] = (
    "all_reduce",
    "broadcast",
    "barrier",
    "get_rank",
    "get_world_size",
    "ReduceOp",
)


def inject_dist(module: Any, sim: ThreadDist) -> list[str]:
    """Point the candidate module's distributed references at ``sim``.

    Returns the names that were rebound. If none were, the source does not
    reference ``torch.distributed`` in a way this simulation can intercept, and
    that is an error rather than a run that silently never collected anything.
    """
    injected: list[str] = []
    if hasattr(module, "dist"):
        module.dist = sim
        injected.append("dist")
    for name in _DIST_NAMES:
        if hasattr(module, name):
            setattr(module, name, getattr(sim, name))
            injected.append(name)
    if not injected:
        raise RuntimeError(
            "the distributed source references no interceptable torch.distributed "
            "symbol (expected 'import torch.distributed as dist' or a direct import "
            "of all_reduce/get_rank); the thread simulation cannot drive it"
        )
    return injected


def _thread_body(
    rank: int,
    sim: ThreadDist,
    module: Any,
    entry: str,
    payload: dict[str, Any],
    results: dict[int, dict[str, Any]],
    errors: dict[int, dict[str, Any]],
) -> None:
    sim.bind(rank)
    try:
        results[rank] = o5_worker.call_entry(module, entry, payload, rank)
    except Exception as exc:  # noqa: BLE001 - recorded and reported, never swallowed
        errors[rank] = o5_worker.error_payload(rank, "candidate", exc)
        # Peers may be parked in a collective this rank will now never reach.
        sim.abort()


def launch_threads(
    source: str,
    entry: str,
    payload: dict[str, Any],
    world_size: int,
    job_dir: Path,
    timeout_s: float,
) -> LaunchResult:
    """Run every rank as a thread against :class:`ThreadDist`."""
    res = LaunchResult(world_size=int(world_size), launcher="thread")
    started = time.perf_counter()
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        src_path = o5_worker.write_source(source, job_dir / "o5_candidate.py")
        module = o5_worker.load_module_from_path(
            src_path, f"crucible_o5_threadsim_{job_dir.name}_{world_size}"
        )
        sim = ThreadDist(int(world_size), timeout_s=float(timeout_s))
        inject_dist(module, sim)
    except Exception as exc:  # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"
        res.error_kind = "infrastructure" if _looks_like_infrastructure(res.error) else "candidate"
        res.duration_s = time.perf_counter() - started
        return res

    results: dict[int, dict[str, Any]] = {}
    errors: dict[int, dict[str, Any]] = {}
    threads: list[threading.Thread] = []
    for rank in range(int(world_size)):
        t = threading.Thread(
            target=_thread_body,
            args=(rank, sim, module, entry, payload, results, errors),
            name=f"o5-rank{rank}",
            daemon=True,
        )
        threads.append(t)
        t.start()

    deadline = time.monotonic() + float(timeout_s)
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [t.name for t in threads if t.is_alive()]

    res.duration_s = time.perf_counter() - started
    res.summaries = dict(results)
    res.rank_errors = dict(errors)
    res.missing_ranks = [r for r in range(int(world_size)) if r not in results]

    if alive:
        res.timed_out = True
        # A python thread cannot be killed. This is precisely why the production
        # launcher is spawn, and the evidence says so rather than claiming a
        # clean teardown that did not happen.
        res.reaped = False
        res.alive_after_reap = []
        res.error = (
            f"rank thread(s) {alive} did not finish within {timeout_s:.0f}s; "
            "a rank that never reaches a collective its peers reached is a desync"
        )
        res.error_kind = "hang"
        return res
    if errors:
        first = sorted(errors)[0]
        detail = errors[first]
        text = f"{detail.get('type', 'error')}: {detail.get('message', '')}"
        marker = _looks_like_infrastructure(text)
        res.error = f"rank {first} failed -- {text}"
        res.error_kind = "infrastructure" if marker else "candidate"
        return res
    if res.missing_ranks:
        res.error = f"rank(s) {res.missing_ranks} produced no result and reported no error"
        res.error_kind = "candidate"
        return res
    res.ok = True
    return res


# --------------------------------------------------------------------------- #
# launcher 2: torch.multiprocessing.spawn over a real gloo process group
# --------------------------------------------------------------------------- #


def _reap(proc_ctx: Any) -> tuple[list[int], list[int]]:
    """Terminate, then kill, every child. Returns (pids, still-alive pids).

    Called from a ``finally`` on every path, including the hang path: a rank
    that wedged is a finding worth reporting, and an orphaned python process
    holding a FileStore is not something the next run should have to discover.
    """
    pids: list[int] = []
    alive: list[int] = []
    for proc in list(getattr(proc_ctx, "processes", ()) or ()):
        pid = int(getattr(proc, "pid", -1) or -1)
        pids.append(pid)
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(_REAP_GRACE_S)
            if proc.is_alive():
                proc.kill()
                proc.join(_REAP_GRACE_S)
            if proc.is_alive():
                alive.append(pid)
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning("could not reap O5 rank process %s: %s", pid, exc)
            alive.append(pid)
    return pids, alive


def _collect_rank_files(job_dir: Path, world_size: int) -> tuple[dict[int, Any], dict[int, Any]]:
    summaries: dict[int, Any] = {}
    errors: dict[int, Any] = {}
    for rank in range(int(world_size)):
        ok_path = job_dir / f"rank{rank}.json"
        err_path = job_dir / f"rank{rank}.error.json"
        for path, sink in ((ok_path, summaries), (err_path, errors)):
            if not path.exists():
                continue
            try:
                sink[rank] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors[rank] = {
                    "rank": rank,
                    "where": "parent",
                    "type": type(exc).__name__,
                    "message": f"could not read {path.name}: {exc}",
                }
    return summaries, errors


def launch_spawn(
    source: str,
    entry: str,
    payload: dict[str, Any],
    world_size: int,
    job_dir: Path,
    timeout_s: float,
    backend: str = "gloo",
) -> LaunchResult:
    """Run every rank as a spawned process under a real gloo process group."""
    res = LaunchResult(world_size=int(world_size), launcher="spawn")
    started = time.perf_counter()
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch.multiprocessing as torch_mp

        src_path = o5_worker.write_source(source, job_dir / "o5_candidate.py")
    except Exception as exc:  # noqa: BLE001
        res.error = f"the spawn launcher could not be prepared: {type(exc).__name__}: {exc}"
        res.error_kind = "infrastructure"
        res.duration_s = time.perf_counter() - started
        return res

    proc_ctx: Any = None
    join_error: str = ""
    joined = False
    try:
        try:
            proc_ctx = torch_mp.spawn(
                o5_worker.run_rank,
                args=(
                    int(world_size),
                    str(job_dir),
                    str(src_path),
                    str(entry),
                    dict(payload),
                    str(backend),
                    float(timeout_s),
                ),
                nprocs=int(world_size),
                join=False,
                daemon=False,
                start_method="spawn",
            )
        except Exception as exc:  # noqa: BLE001
            res.error = f"torch.multiprocessing.spawn failed to start: {type(exc).__name__}: {exc}"
            res.error_kind = "infrastructure"
            res.duration_s = time.perf_counter() - started
            return res

        deadline = time.monotonic() + float(timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                joined = bool(proc_ctx.join(timeout=min(0.5, remaining)))
            except Exception as exc:  # noqa: BLE001 - a rank raised or exited non-zero
                join_error = f"{type(exc).__name__}: {exc}"
                break
            if joined:
                break
    finally:
        res.pids, res.alive_after_reap = _reap(proc_ctx)
        res.reaped = not res.alive_after_reap

    res.duration_s = time.perf_counter() - started
    summaries, errors = _collect_rank_files(job_dir, int(world_size))
    res.summaries = summaries
    res.rank_errors = errors
    res.missing_ranks = [r for r in range(int(world_size)) if r not in summaries]

    if not joined and not join_error:
        res.timed_out = True
        res.error = (
            f"rank(s) {res.missing_ranks} did not finish within {timeout_s:.0f}s and the "
            "process tree was killed; a rank that never reaches a collective its peers "
            "reached is a desync, not an infrastructure fault"
        )
        res.error_kind = "hang"
        return res
    if join_error or errors:
        detail = join_error
        if errors:
            first = sorted(errors)[0]
            payload_err = errors[first]
            detail = (
                f"rank {first} {payload_err.get('where', '?')}: "
                f"{payload_err.get('type', 'error')}: {payload_err.get('message', '')}"
            ) + (f" | parent saw {join_error}" if join_error else "")
        marker = _looks_like_infrastructure(detail)
        res.error = detail
        res.error_kind = "infrastructure" if marker else "candidate"
        return res
    if res.missing_ranks:
        res.error = (
            f"rank(s) {res.missing_ranks} exited without writing a result and without "
            "recording an error"
        )
        res.error_kind = "candidate"
        return res
    res.ok = True
    return res


LAUNCHERS = ("spawn", "thread")


def run_world(
    source: str,
    entry: str,
    payload: dict[str, Any],
    world_size: int,
    job_dir: Path | str,
    timeout_s: float,
    launcher: str = "spawn",
    backend: str = "gloo",
) -> LaunchResult:
    """Run one world size through the requested launcher."""
    job = Path(job_dir)
    if launcher == "thread":
        return launch_threads(source, entry, payload, world_size, job, timeout_s)
    if launcher == "spawn":
        return launch_spawn(source, entry, payload, world_size, job, timeout_s, backend)
    raise ValueError(f"unknown O5 launcher {launcher!r}; known launchers: {list(LAUNCHERS)}")


# --------------------------------------------------------------------------- #
# comparison primitives
# --------------------------------------------------------------------------- #


def per_step_deviation(a: Sequence[float], b: Sequence[float]) -> list[float]:
    n = min(len(a), len(b))
    return [abs(float(a[i]) - float(b[i])) for i in range(n)]


def l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(n))))


def _argmax(values: Sequence[float]) -> int | None:
    if not values:
        return None
    best = 0
    for i, v in enumerate(values):
        if v > values[best]:
            best = i
    return best


def _scale_of(values: Iterable[float]) -> float:
    vals = [abs(float(v)) for v in values]
    return max(vals) if vals else 0.0


@dataclass(frozen=True)
class FloorEntry:
    """One quantity's noise floor, with everything needed to re-derive it."""

    quantity: str
    measured: float
    ulp_bound: float
    used: float
    source: str  # "measured" | "ulp_fallback"
    scale: float
    worst_pair: tuple[int, int] | None = None
    per_step: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "measured": self.measured,
            "ulp_bound": self.ulp_bound,
            "used": self.used,
            "source": self.source,
            "scale": self.scale,
            "worst_pair": list(self.worst_pair) if self.worst_pair else None,
            "per_step": self.per_step,
        }


def _floor_entry(
    quantity: str,
    measured: float,
    scale: float,
    worst_pair: tuple[int, int] | None,
    per_step: list[float],
) -> FloorEntry:
    """Combine the measurement with the unit-roundoff bound for the quantity.

    The floor used is the larger of the two. The measurement is the empirical
    answer and is what normally governs; the ulp bound exists because a run
    short enough that no reordering flipped a bit measures exactly zero, and
    zero is not a defensible threshold for a float comparison. The bound is one
    unit of fp32 roundoff at the quantity's own magnitude - derived, not chosen -
    and the entry records which of the two won.
    """
    ulp = _FP32_EPS * float(scale)
    used = max(float(measured), ulp)
    source = "measured" if float(measured) >= ulp else "ulp_fallback"
    return FloorEntry(
        quantity=quantity,
        measured=float(measured),
        ulp_bound=float(ulp),
        used=float(used),
        source=source,
        scale=float(scale),
        worst_pair=worst_pair,
        per_step=per_step,
    )


@dataclass
class NoiseFloor:
    """The calibration: what float non-associativity alone is worth here."""

    runs: int
    salts: list[int]
    entries: dict[str, FloorEntry]
    baseline: dict[str, list[float]]
    launcher: str
    method: str

    def used(self, quantity: str) -> float:
        return self.entries[quantity].used

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "permutation_salts": list(self.salts),
            "launcher": self.launcher,
            "method": self.method,
            "entries": {name: e.as_dict() for name, e in self.entries.items()},
            "baseline_series": {
                "losses": self.baseline.get("losses", []),
                "grad_norms": self.baseline.get("grad_norms", []),
            },
        }


def calibrate_noise_floor(results: Sequence[LaunchResult], salts: Sequence[int], launcher: str) -> NoiseFloor:
    """Turn repeated single-rank runs into a floor per graded quantity.

    Every pair of runs is compared, not just consecutive ones, and the maximum
    over pairs is taken: the floor has to bound the worst reordering the machine
    can produce, not an average one.
    """
    if len(results) < 2:
        raise ValueError("a noise floor needs at least two single-rank runs to compare")
    series: list[dict[str, list[float]]] = [
        {key: [float(v) for v in r.summaries[0][key]] for key in ("losses", "grad_norms", "w")}
        for r in results
    ]
    baseline = series[0]

    entries: dict[str, FloorEntry] = {}
    for quantity in SERIES_KEYS:
        worst = 0.0
        worst_pair: tuple[int, int] | None = None
        worst_series: list[float] = []
        for i in range(len(series)):
            for j in range(i + 1, len(series)):
                dev = per_step_deviation(series[i][quantity], series[j][quantity])
                peak = max(dev) if dev else 0.0
                if peak >= worst:
                    worst = peak
                    worst_pair = (i, j)
                    worst_series = dev
        entries[quantity] = _floor_entry(
            quantity, worst, _scale_of(baseline[quantity]), worst_pair, worst_series
        )

    worst_w = 0.0
    worst_w_pair: tuple[int, int] | None = None
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            d = l2_distance(series[i]["w"], series[j]["w"])
            if d >= worst_w:
                worst_w = d
                worst_w_pair = (i, j)
    w_scale = float(math.sqrt(sum(float(v) ** 2 for v in baseline["w"])))
    entries["final_params"] = _floor_entry("final_params", worst_w, w_scale, worst_w_pair, [])

    return NoiseFloor(
        runs=len(results),
        salts=[int(s) for s in salts],
        entries=entries,
        baseline=baseline,
        launcher=launcher,
        method=(
            "the single-rank config run repeatedly, identical except for the order in "
            "which the microbatch is accumulated (sample order permuted by i -> "
            "(a*i+b) mod n); the pairwise deviation is float non-associativity and "
            "nothing else"
        ),
    )


@dataclass
class CheckOutcome:
    """One graded sub-check at one world size."""

    name: str
    verdict: str  # PASS | FAIL
    observed: float
    floor: float
    k: float
    threshold: float
    margin: float
    first_bad_step: int | None
    peak_step: int | None
    detail: str
    per_step: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "verdict": self.verdict,
            "observed_max_deviation": self.observed,
            "noise_floor": self.floor,
            "k": self.k,
            "threshold": self.threshold,
            "margin": self.margin,
            "ratio_to_floor": (self.observed / self.floor) if self.floor > 0 else None,
            "first_bad_step": self.first_bad_step,
            "peak_step": self.peak_step,
            "detail": self.detail,
            "per_step_deviation": self.per_step,
        }


def _grade(
    name: str,
    per_step: list[float],
    floor: FloorEntry,
    k: float,
    units: str,
) -> CheckOutcome:
    threshold = float(k) * floor.used
    observed = max(per_step) if per_step else 0.0
    first_bad = next((i for i, v in enumerate(per_step) if v > threshold), None)
    peak = _argmax(per_step)
    passed = observed <= threshold
    if passed:
        detail = (
            f"{units} stayed within {k:g} x the measured noise floor "
            f"({floor.used:.3e}, {floor.source}); peak deviation {observed:.3e} at "
            f"step {peak}, margin {threshold - observed:.3e}"
        )
    else:
        detail = (
            f"{units} left the noise floor at step {first_bad}: deviation "
            f"{per_step[first_bad] if first_bad is not None else observed:.3e} > "
            f"threshold {threshold:.3e} (= k={k:g} x floor {floor.used:.3e}, "
            f"{floor.source}); peak {observed:.3e} at step {peak}"
        )
    return CheckOutcome(
        name=name,
        verdict="PASS" if passed else "FAIL",
        observed=observed,
        floor=floor.used,
        k=float(k),
        threshold=threshold,
        margin=threshold - observed,
        first_bad_step=first_bad,
        peak_step=peak,
        detail=detail,
        per_step=per_step,
    )


def compare_world(
    single: LaunchResult,
    multi: LaunchResult,
    floor: NoiseFloor,
    k: float,
) -> dict[str, Any]:
    """Grade one N-rank run against the single-rank baseline."""
    checks: list[CheckOutcome] = []

    for quantity, units in (
        ("losses", "per-step loss"),
        ("grad_norms", "global grad norm"),
    ):
        dev = per_step_deviation(single.series(quantity), multi.series(quantity))
        checks.append(_grade(quantity, dev, floor.entries[quantity], k, units))

    w_dev = l2_distance(single.series("w"), multi.series("w"))
    checks.append(_grade("final_params", [w_dev], floor.entries["final_params"], k, "final parameter L2 distance"))

    # Ranks that disagree with each other never even reached a common answer;
    # this is the loudest possible signature of a dropped or partial collective.
    disagreement: list[float] = []
    for rank in sorted(multi.summaries):
        if rank == 0:
            continue
        for quantity in SERIES_KEYS:
            dev = per_step_deviation(multi.series(quantity, 0), multi.series(quantity, rank))
            disagreement.append(max(dev) if dev else 0.0)
    loss_floor = floor.entries["losses"]
    norm_floor = floor.entries["grad_norms"]
    agreement_floor = FloorEntry(
        quantity="rank_agreement",
        measured=max(loss_floor.measured, norm_floor.measured),
        ulp_bound=max(loss_floor.ulp_bound, norm_floor.ulp_bound),
        used=max(loss_floor.used, norm_floor.used),
        source=(loss_floor if loss_floor.used >= norm_floor.used else norm_floor).source,
        scale=max(loss_floor.scale, norm_floor.scale),
    )
    checks.append(
        _grade(
            "rank_agreement",
            disagreement,
            agreement_floor,
            k,
            "disagreement between ranks after the collectives",
        )
    )

    failed = [c for c in checks if c.verdict == "FAIL"]
    return {
        "world_size": multi.world_size,
        "status": "fail" if failed else "pass",
        "shard_sizes": None,  # filled by the caller, which owns the spec
        "checks": {c.name: c.as_dict() for c in checks},
        "failed_checks": [c.name for c in failed],
        "final_params_l2": w_dev,
        "launch": multi.as_dict(),
    }


# --------------------------------------------------------------------------- #
# resolving what to run
# --------------------------------------------------------------------------- #


def resolve_source(ctx: OracleContext) -> tuple[str, str, str]:
    """(source, entry, provenance). Raises LookupError when there is nothing to run."""
    override = ctx.extras.get("nrank_source")
    if override:
        entry = str(ctx.extras.get("nrank_entry") or o5_worker.DEFAULT_STEP_ENTRY)
        return str(override), entry, "ctx.extras['nrank_source']"

    seed = ctx.seed
    extras = dict(getattr(seed, "extras", {}) or {})
    source = extras.get("gloo_source")
    if source:
        entry = str(
            ctx.extras.get("nrank_entry")
            or extras.get("gloo_step_entry")
            or o5_worker.DEFAULT_STEP_ENTRY
        )
        return str(source), entry, f"seed {getattr(seed, 'id', '?')!r} extras['gloo_source']"

    raise LookupError(
        f"seed {getattr(seed, 'id', '?')!r} exposes no distributed source "
        "(extras['gloo_source']) and none was supplied in ctx.extras['nrank_source']; "
        "there is no multi-rank program to run, so N-rank equivalence was not checked"
    )


_SPEC_FIELDS = {f.name for f in fields(NRankSpec)}


def resolve_spec(ctx: OracleContext) -> NRankSpec:
    """Build the workload spec from config, then apply explicit overrides."""
    spec = NRankSpec(steps=int(ctx.cfg.nrank_steps), seed=int(ctx.rng_seed))
    override = ctx.extras.get("nrank_spec")
    if override is None:
        return spec
    if isinstance(override, NRankSpec):
        return override
    if not isinstance(override, dict):
        raise TypeError(
            f"ctx.extras['nrank_spec'] must be an NRankSpec or a dict, got {type(override).__name__}"
        )
    unknown = sorted(set(override) - _SPEC_FIELDS)
    if unknown:
        raise ValueError(
            f"ctx.extras['nrank_spec'] has unknown key(s) {unknown}; valid keys: "
            f"{sorted(_SPEC_FIELDS)}"
        )
    return replace(spec, **override)


def resolve_world_sizes(ctx: OracleContext, spec: NRankSpec) -> tuple[list[int], list[dict[str, Any]]]:
    """(sizes to run, sizes excluded with the reason)."""
    requested = ctx.extras.get("nrank_world_sizes") or ctx.cfg.nrank_world_sizes
    sizes: list[int] = []
    excluded: list[dict[str, Any]] = []
    for raw in requested:
        n = int(raw)
        if n < 2:
            excluded.append({"world_size": n, "reason": "a world size below 2 is not a multi-rank run"})
            continue
        if n > spec.n_samples:
            excluded.append(
                {
                    "world_size": n,
                    "reason": (
                        f"{n} ranks over {spec.n_samples} samples would leave ranks with no "
                        "data; empty-shard behaviour is O1's sweep, not an equivalence claim"
                    ),
                }
            )
            continue
        if n not in sizes:
            sizes.append(n)
    return sizes, excluded


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #


class NRankOracle:
    """O5. One rank versus N, against an empirically calibrated noise floor."""

    id = ORACLE_ID
    name = "n-rank equivalence vs a measured non-associativity floor"
    required_caps: tuple[str, ...] = ("gloo",)

    def applies_to(self, task: Task) -> bool:
        return task.domain in NRANK_DOMAINS or ORACLE_ID in tuple(task.oracles or ())

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _result(verdict: str, reason: str, evidence: dict[str, Any]) -> OracleResult:
        return OracleResult(
            oracle=ORACLE_ID,
            verdict=verdict,  # type: ignore[arg-type]
            reason=reason,
            evidence=evidence,
            capabilities_used=["gloo"],
        )

    @staticmethod
    def _launch_failure(res: LaunchResult, stage: str) -> tuple[str, str]:
        """(verdict, reason) for a launch that produced no usable series."""
        if res.error_kind == "infrastructure":
            return (
                "SKIP",
                f"{stage} could not be executed on this machine: {res.error}",
            )
        if res.error_kind == "hang":
            return (
                "FAIL",
                f"{stage} hung: {res.error}. A rank that does not arrive at a collective "
                "its peers arrived at is the defect, not the harness"
                + ("" if res.reaped else f"; process(es) {res.alive_after_reap} survived reaping"),
            )
        return ("FAIL", f"{stage} failed: {res.error}")

    # -- run --------------------------------------------------------------- #

    def run(self, ctx: OracleContext) -> OracleResult:
        started = time.perf_counter()
        launcher = str(ctx.extras.get("nrank_launcher") or "spawn")
        backend = str(ctx.extras.get("nrank_backend") or "gloo")
        k = float(ctx.extras.get("nrank_k") or ctx.cfg.nrank_k)

        evidence: dict[str, Any] = {
            "launcher": launcher,
            "backend": backend,
            "k": k,
            "candidate_execution": (
                "every rank runs the candidate's distributed source out of the grading "
                "interpreter (torch.multiprocessing.spawn + gloo FileStore)"
                if launcher == "spawn"
                else "SIMULATION: every rank runs the candidate's distributed source on a "
                "thread in this interpreter against a deterministic torch.distributed "
                "stand-in; this is not the production launcher"
            ),
            "tolerance_model": (
                "max per-step |series_N - series_1| <= k * noise_floor, where the floor is "
                "measured on this machine from repeated single-rank runs under permuted "
                "microbatch accumulation order. No constant tolerance is used anywhere."
            ),
        }

        if launcher not in LAUNCHERS:
            return self._result(
                "ERROR",
                f"unknown O5 launcher {launcher!r}; known launchers: {list(LAUNCHERS)}",
                evidence,
            )
        if not ctx.caps.gloo:
            return self._result(
                "SKIP",
                "gloo is not available in this torch build "
                f"({ctx.caps.detail('gloo') or 'torch.distributed.is_gloo_available() returned False'}); "
                "no process group can be formed, so N-rank equivalence was not checked",
                evidence,
            )

        try:
            source, entry, provenance = resolve_source(ctx)
        except LookupError as exc:
            return self._result("SKIP", str(exc), evidence)
        evidence["source_provenance"] = provenance
        evidence["entry"] = entry

        try:
            spec = resolve_spec(ctx)
            spec.validate()
        except (TypeError, ValueError) as exc:
            return self._result("SKIP", f"the O5 workload could not be built: {exc}", evidence)
        evidence["spec"] = spec.as_dict()
        evidence["n_params"] = spec.n_params()

        world_sizes, excluded = resolve_world_sizes(ctx, spec)
        evidence["world_sizes_requested"] = list(world_sizes)
        evidence["world_sizes_excluded"] = excluded
        if not world_sizes:
            return self._result(
                "SKIP",
                "no runnable world size >= 2 was configured for this workload: "
                + "; ".join(f"{e['world_size']}: {e['reason']}" for e in excluded),
                evidence,
            )

        n_floor_runs = max(2, int(ctx.cfg.nrank_noise_floor_runs))
        budget = float(ctx.cfg.timeout_for(ORACLE_ID))
        per_launch = float(
            ctx.extras.get("nrank_launch_timeout_s")
            or max(30.0, budget / float(n_floor_runs + len(world_sizes) + 1))
        )
        evidence["per_launch_timeout_s"] = per_launch

        work = ctx.sub_workdir(ORACLE_ID)

        # ---- stage 1: calibrate the floor before comparing anything ------- #
        salts = [0] + list(range(1, n_floor_runs))
        single_runs: list[LaunchResult] = []
        for i, salt in enumerate(salts):
            res = run_world(
                source,
                entry,
                build_payload(spec, 1, perm_salt=salt),
                1,
                work / f"floor{i}",
                per_launch,
                launcher=launcher,
                backend=backend,
            )
            single_runs.append(res)
            evidence.setdefault("noise_floor_launches", []).append(res.as_dict())
            if not res.ok:
                verdict, reason = self._launch_failure(
                    res, f"the single-rank calibration run {i} (permutation salt {salt})"
                )
                return self._result(
                    verdict,
                    reason + " -- without a single-rank baseline there is nothing to compare "
                    "N ranks against, so no equivalence claim was verified",
                    evidence,
                )

        try:
            floor = calibrate_noise_floor(single_runs, salts, launcher)
        except (ValueError, KeyError) as exc:
            return self._result(
                "SKIP", f"the noise floor could not be calibrated: {type(exc).__name__}: {exc}", evidence
            )
        evidence["noise_floor"] = floor.as_dict()
        baseline = single_runs[0]

        # ---- stage 2: compare N ranks against it -------------------------- #
        records: dict[str, Any] = {}
        skipped: list[dict[str, Any]] = []
        failed_sizes: list[int] = []
        hung: list[int] = []
        for ws in world_sizes:
            res = run_world(
                source,
                entry,
                build_payload(spec, ws, perm_salt=0),
                ws,
                work / f"ws{ws}",
                per_launch,
                launcher=launcher,
                backend=backend,
            )
            if not res.ok:
                verdict, reason = self._launch_failure(res, f"the {ws}-rank run")
                entry_rec = {
                    "world_size": ws,
                    "status": "fail" if verdict == "FAIL" else "skip",
                    "shard_sizes": spec.shard_sizes(ws),
                    "reason": reason,
                    "launch": res.as_dict(),
                }
                records[str(ws)] = entry_rec
                if verdict == "FAIL":
                    failed_sizes.append(ws)
                    if res.error_kind == "hang":
                        hung.append(ws)
                else:
                    skipped.append({"world_size": ws, "reason": reason})
                continue

            rec = compare_world(baseline, res, floor, k)
            rec["shard_sizes"] = spec.shard_sizes(ws)
            records[str(ws)] = rec
            if rec["status"] == "fail":
                failed_sizes.append(ws)

        evidence["world_sizes"] = records
        evidence["duration_s"] = time.perf_counter() - started

        floor_summary = ", ".join(
            f"{name}={e.used:.3e} ({e.source})" for name, e in floor.entries.items()
        )

        if failed_sizes:
            first = records[str(failed_sizes[0])]
            if failed_sizes[0] in hung:
                detail = first.get("reason", "")
            else:
                bad = first.get("failed_checks", []) or []
                detail = "; ".join(first["checks"][name]["detail"] for name in bad)
            return self._result(
                "FAIL",
                f"the run is not invariant to the world size: {failed_sizes} rank(s) "
                f"diverged from the single-rank baseline. First failure at world size "
                f"{failed_sizes[0]} -- {detail}. Noise floor (measured over "
                f"{floor.runs} single-rank runs): {floor_summary}; k={k:g}",
                evidence,
            )
        if skipped or not records:
            reasons = "; ".join(f"world size {s['world_size']}: {s['reason']}" for s in skipped)
            return self._result(
                "SKIP",
                "N-rank equivalence is unverified because "
                + (reasons or "no world size produced a comparable run")
                + " -- a passing subset of world sizes is not a pass",
                evidence,
            )
        return self._result(
            "PASS",
            f"the run is invariant to the world size across {sorted(int(s) for s in records)}: "
            f"every per-step loss, global grad-norm and final-parameter deviation stayed "
            f"within k={k:g} x the measured non-associativity floor ({floor_summary})",
            evidence,
        )


O5 = NRankOracle()
ORACLE = register_oracle(O5)

__all__ = [
    "O5",
    "ORACLE",
    "ORACLE_ID",
    "NRankOracle",
    "NRankSpec",
    "NoiseFloor",
    "FloorEntry",
    "CheckOutcome",
    "LaunchResult",
    "ThreadDist",
    "CHECK_NAMES",
    "LAUNCHERS",
    "MAX_PARAMS",
    "NRANK_DOMAINS",
    "SERIES_KEYS",
    "build_payload",
    "calibrate_noise_floor",
    "compare_world",
    "inject_dist",
    "l2_distance",
    "launch_spawn",
    "launch_threads",
    "per_step_deviation",
    "permutation",
    "resolve_source",
    "resolve_spec",
    "resolve_world_sizes",
    "run_world",
]
