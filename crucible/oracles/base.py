"""Oracle protocol and the dispatcher that keeps every verdict honest.

``run_all`` is deliberately paranoid, because it is the only place where an
oracle's claim becomes a task's verdict:

* An oracle whose required capabilities are absent returns **SKIP** with the
  missing names and the probed reason. It never returns PASS by omission.
* An uncaught exception becomes **ERROR** with the traceback in the evidence,
  not a swallowed pass.
* An oracle that overruns its budget becomes **ERROR**; a hang is a defect in
  the oracle or the candidate, and either way it is not a pass.
* An oracle that does not apply to the task is simply not run; the invariant is
  "passed every *applicable* oracle", so inapplicable oracles must not be able
  to force a SKIP.
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from ..capabilities import Capabilities
from ..config import Config
from ..errors import CapabilityError
from ..schema import OracleResult, Task

logger = logging.getLogger(__name__)


@dataclass
class OracleContext:
    task: Task
    candidate_src: str
    seed: Any  # SeedSpec; typed loosely to keep oracles importable without seeds
    caps: Capabilities
    workdir: Path
    cfg: Config
    rng_seed: int = 1234
    device: str = "cpu"
    extras: dict[str, Any] = field(default_factory=dict)

    def sub_workdir(self, name: str) -> Path:
        p = self.workdir / name
        p.mkdir(parents=True, exist_ok=True)
        return p


@runtime_checkable
class Oracle(Protocol):
    id: str
    name: str
    required_caps: tuple[str, ...]

    def applies_to(self, task: Task) -> bool: ...

    def run(self, ctx: OracleContext) -> OracleResult: ...


#: Populated by ``crucible.oracles.load_oracles`` when the o1..o5 modules import.
ORACLES: dict[str, Oracle] = {}


def register_oracle(oracle: Oracle) -> Oracle:
    """Called at import time by each o*.py module."""
    oid = getattr(oracle, "id", "")
    if not oid:
        raise ValueError("oracle must define a non-empty id")
    existing = ORACLES.get(oid)
    if existing is not None and existing is not oracle:
        logger.warning("oracle id %s re-registered; replacing %r", oid, type(existing).__name__)
    ORACLES[oid] = oracle
    return oracle


def _skip(oracle_id: str, reason: str, caps_used: Iterable[str], duration: float) -> OracleResult:
    return OracleResult(
        oracle=oracle_id,
        verdict="SKIP",
        reason=reason,
        evidence={"skipped": True},
        duration_s=duration,
        capabilities_used=list(caps_used),
    )


def _error(oracle_id: str, reason: str, tb: str, duration: float) -> OracleResult:
    return OracleResult(
        oracle=oracle_id,
        verdict="ERROR",
        reason=reason,
        evidence={"traceback": tb},
        duration_s=duration,
    )


def gate_capabilities(oracle: Oracle, caps: Capabilities) -> str | None:
    """None if the oracle can run here, else the SKIP reason."""
    required = tuple(getattr(oracle, "required_caps", ()) or ())
    if not required:
        return None
    try:
        missing = caps.missing(required)
    except CapabilityError as exc:
        return f"oracle {oracle.id} declares an unknown capability: {exc}"
    if not missing:
        return None
    detail = caps.explain(missing)
    return f"requires {', '.join(required)}; missing {', '.join(missing)} ({detail})"


def run_one(oracle: Oracle, ctx: OracleContext, timeout_s: float | None = None) -> OracleResult:
    """Run a single oracle under cap gating, timeout and exception capture."""
    oid = getattr(oracle, "id", type(oracle).__name__)
    budget = ctx.cfg.timeout_for(oid) if timeout_s is None else timeout_s
    started = time.perf_counter()

    reason = gate_capabilities(oracle, ctx.caps)
    if reason is not None:
        return _skip(oid, reason, getattr(oracle, "required_caps", ()), time.perf_counter() - started)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"oracle-{oid}")
    future = executor.submit(oracle.run, ctx)
    try:
        result = future.result(timeout=budget)
    except FuturesTimeout:
        # The worker thread cannot be killed; the heavy lifting lives in
        # sandbox subprocesses, which have their own hard timeouts and are
        # reaped there. We stop waiting and record the overrun.
        executor.shutdown(wait=False, cancel_futures=True)
        return _error(
            oid,
            f"oracle exceeded its {budget:.0f}s budget",
            "",
            time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001 - re-reported, never swallowed
        executor.shutdown(wait=False)
        return _error(
            oid,
            f"{type(exc).__name__}: {exc}",
            traceback.format_exc(),
            time.perf_counter() - started,
        )
    executor.shutdown(wait=False)

    duration = time.perf_counter() - started
    if not isinstance(result, OracleResult):
        return _error(
            oid,
            f"oracle returned {type(result).__name__}, expected OracleResult",
            "",
            duration,
        )
    if not result.duration_s:
        result = result.model_copy(update={"duration_s": duration})
    if result.oracle != oid:
        result = result.model_copy(update={"oracle": oid})
    return result


def run_all(ctx: OracleContext, ids: Iterable[str] | None = None) -> list[OracleResult]:
    """Run the requested oracles (default: the task's own list) in id order."""
    _ensure_loaded()
    requested = list(ids) if ids is not None else list(ctx.task.oracles or sorted(ORACLES))
    results: list[OracleResult] = []
    for oid in requested:
        oracle = ORACLES.get(oid)
        if oracle is None:
            results.append(
                OracleResult(
                    oracle=oid,
                    verdict="ERROR",
                    reason=f"oracle {oid!r} is not registered; available: {sorted(ORACLES)}",
                    evidence={"registered": sorted(ORACLES)},
                )
            )
            continue
        try:
            applicable = bool(oracle.applies_to(ctx.task))
        except Exception as exc:  # noqa: BLE001 - a broken predicate is an oracle defect
            results.append(
                _error(oid, f"applies_to raised {type(exc).__name__}: {exc}", traceback.format_exc(), 0.0)
            )
            continue
        if not applicable:
            logger.debug("oracle %s does not apply to task %s; not run", oid, ctx.task.task_id)
            continue
        results.append(run_one(oracle, ctx))
    return results


def _ensure_loaded() -> None:
    """Import the concrete oracle modules on first use (avoids an import cycle).

    There is deliberately no "registry is already non-empty" short circuit: a
    partially populated registry is precisely the case that still needs
    completing, because every absent oracle is a check the task silently does
    not receive.
    """
    try:
        from . import load_oracles
    except ImportError as exc:
        logger.debug("oracle package not importable: %s", exc)
        return
    load_oracles()


__all__ = [
    "OracleContext",
    "Oracle",
    "ORACLES",
    "register_oracle",
    "run_all",
    "run_one",
    "gate_capabilities",
]
