"""Where a calibrated task goes, and why - decided on the interval.

The bands come from the proposal:

===========================  ==========  ==========================================
pass@1                       route       reason
===========================  ==========  ==========================================
pass@8 == 0 (any pass@1)     escalate    no evidence it is solvable at all
> 0.90                       reject      no gradient left; likely contaminated
in (0.70, 0.90]              reject      **decided here** - see below
in [0.10, 0.70]              gold        maximum training signal
< 0.10 with pass@8 > 0       frontier    hard but demonstrably reachable
===========================  ==========  ==========================================

**The band the proposal leaves implicit.** ``0.70 < pass@1 <= 0.90`` is not
covered by any of the four stated rules. It is resolved here as **reject**, and
the rationale distinguishes it from the ``> 0.90`` case by calling it *thin*
rather than *saturated*: at pass@1 = 0.85 roughly one sample in seven carries
any learning signal, and the same SME hours buy several times the signal from a
gold-band task. It is not evidence of contamination, so the contamination flag
is not raised for it. The choice also keeps the router consistent with
``crucible.report.metrics``, whose gold-band share is defined on ``[0.10, 0.70]``:
routing 0.85 to gold would make the two modules disagree about what gold means.

**Routing on the interval.** ``pass@1 = 3/8`` has a 95% Wilson interval of about
``[0.14, 0.69]``; ``4/8`` has ``[0.22, 0.79]``. Treating 0.375 as exact is the
error this module exists to avoid. When a CI is supplied, the route implied by
each endpoint is computed as well as the one implied by the point estimate. If
they disagree, the decision is a **straddle**: ``confident`` is False, the
rationale names the boundary crossed, and the route is picked by a fixed
conservatism order

    escalate  >  frontier  >  gold  >  reject          (most conservative first)

ordered by how irreversible the commitment is. ``escalate`` and ``frontier``
both mean "keep it and measure more"; ``gold`` spends SME time but a wrongly
gilded task is caught downstream by the rubric and IRR gates; ``reject``
destroys a task and is the only decision that cannot be walked back. So the
single rule is: **uncertainty never discards a task, and never buys a confident
gold label.** A straddle at the 0.10 boundary therefore lands on ``frontier``
(gold is not claimed), and a straddle at the 0.70 boundary lands on ``gold``
(the task is not thrown away on eight samples) with ``confident=False`` telling
the caller to resample before spending.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..config import Config
from ..errors import CrucibleError
from .passk import PassKEstimate, estimate

logger = logging.getLogger(__name__)

Route = Literal["reject", "gold", "frontier", "escalate"]

ROUTES: tuple[Route, ...] = ("reject", "gold", "frontier", "escalate")

#: Most conservative first. See the module docstring for the justification.
CONSERVATISM_ORDER: tuple[Route, ...] = ("escalate", "frontier", "gold", "reject")

__all__ = [
    "CONSERVATISM_ORDER",
    "ROUTES",
    "BankCalibration",
    "RouteBands",
    "RouteDecision",
    "Route",
    "TaskCalibration",
    "calibrate_bank",
    "route",
    "route_decision",
]


@dataclass(frozen=True)
class RouteBands:
    """Band edges. Policy inputs, never derived from the data."""

    gold_lo: float = 0.10
    gold_hi: float = 0.70
    saturated_lo: float = 0.90

    def __post_init__(self) -> None:
        if not 0.0 <= self.gold_lo < self.gold_hi <= self.saturated_lo <= 1.0:
            raise ValueError(
                "require 0 <= gold_lo < gold_hi <= saturated_lo <= 1; got "
                f"{self.gold_lo}, {self.gold_hi}, {self.saturated_lo}"
            )

    def band(self, pass1: float) -> str:
        """Descriptive band name for a pass@1 value."""
        if pass1 < self.gold_lo:
            return "frontier"
        if pass1 <= self.gold_hi:
            return "gold"
        if pass1 <= self.saturated_lo:
            return "thin"
        return "saturated"

    def as_dict(self) -> dict[str, float]:
        return {
            "gold_lo": self.gold_lo,
            "gold_hi": self.gold_hi,
            "saturated_lo": self.saturated_lo,
        }


DEFAULT_BANDS = RouteBands()

_BAND_TO_ROUTE: dict[str, Route] = {
    "frontier": "frontier",
    "gold": "gold",
    "thin": "reject",
    "saturated": "reject",
}


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    rationale: str
    confident: bool
    point_route: Route
    band: str
    pass_at_1: float
    pass_at_k: float
    k: int
    ci: tuple[float, float] | None = None
    endpoint_routes: tuple[Route, ...] = field(default_factory=tuple)
    bands: RouteBands = DEFAULT_BANDS

    def as_tuple(self) -> tuple[str, str]:
        return (self.route, self.rationale)

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "rationale": self.rationale,
            "confident": self.confident,
            "point_route": self.point_route,
            "band": self.band,
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "k": self.k,
            "ci": list(self.ci) if self.ci is not None else None,
            "endpoint_routes": list(self.endpoint_routes),
            "bands": self.bands.as_dict(),
        }


def _check_unit(name: str, value: float) -> float:
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]: {v}")
    return v


def _most_conservative(candidates: Iterable[Route]) -> Route:
    unique = set(candidates)
    for candidate in CONSERVATISM_ORDER:
        if candidate in unique:
            return candidate
    raise ValueError(f"no known route among {sorted(unique)}")


def route_decision(
    pass1: float,
    passk: float,
    k: int = 8,
    *,
    ci: tuple[float, float] | None = None,
    bands: RouteBands | None = None,
) -> RouteDecision:
    """Full routing decision, with the reasoning attached."""
    edges = bands or DEFAULT_BANDS
    p1 = _check_unit("pass1", pass1)
    pk = _check_unit("passk", passk)
    k_i = int(k)
    if k_i < 1:
        raise ValueError(f"k must be >= 1: {k_i}")
    if pk < p1 - 1e-9:
        raise ValueError(
            f"pass@{k_i}={pk:.6f} is below pass@1={p1:.6f}; pass@k is non-decreasing in k, "
            "so these estimates are inconsistent"
        )

    band = edges.band(p1)

    # Solvability dominates every band question: with zero successes at k there
    # is no evidence the task can be solved at all, and no interval on pass@1
    # can change that.
    if pk <= 0.0:
        return RouteDecision(
            route="escalate",
            rationale=(
                f"pass@{k_i}=0 over the sampled completions: not one succeeded, so there is no "
                "evidence this task is solvable. Escalate for a human solvability check before "
                "spending sampling or SME budget on it."
            ),
            confident=True,
            point_route="escalate",
            band=band,
            pass_at_1=p1,
            pass_at_k=pk,
            k=k_i,
            ci=tuple(ci) if ci is not None else None,
            endpoint_routes=(),
            bands=edges,
        )

    point_route = _BAND_TO_ROUTE[band]
    point_reason = _point_reason(band, p1, pk, k_i, edges)

    if ci is None:
        return RouteDecision(
            route=point_route,
            rationale=point_reason + " No interval was supplied, so this decision rests on the "
            "point estimate alone.",
            confident=False,
            point_route=point_route,
            band=band,
            pass_at_1=p1,
            pass_at_k=pk,
            k=k_i,
            ci=None,
            endpoint_routes=(),
            bands=edges,
        )

    lo, hi = _check_unit("ci low", ci[0]), _check_unit("ci high", ci[1])
    if lo > hi:
        raise ValueError(f"ci is inverted: [{lo}, {hi}]")
    lo_route = _BAND_TO_ROUTE[edges.band(lo)]
    hi_route = _BAND_TO_ROUTE[edges.band(hi)]
    endpoints = (lo_route, hi_route)

    if lo_route == hi_route == point_route:
        return RouteDecision(
            route=point_route,
            rationale=(
                point_reason
                + f" The whole [{lo:.3f}, {hi:.3f}] interval lies in the same band, so the "
                "decision does not depend on sampling noise."
            ),
            confident=True,
            point_route=point_route,
            band=band,
            pass_at_1=p1,
            pass_at_k=pk,
            k=k_i,
            ci=(lo, hi),
            endpoint_routes=endpoints,
            bands=edges,
        )

    chosen = _most_conservative((point_route, lo_route, hi_route))
    crossed = _crossed_edges(lo, hi, edges)
    rationale = (
        point_reason
        + f" The interval [{lo:.3f}, {hi:.3f}] straddles "
        + (f"the {crossed} boundary" if crossed else "a band boundary")
        + f" ({lo:.3f} routes {lo_route}, {hi:.3f} routes {hi_route}), so the point estimate does "
        "not settle it. Routed conservatively to "
        f"{chosen}: uncertainty never discards a task and never buys a confident gold label. "
        "Resample before acting on this."
    )
    return RouteDecision(
        route=chosen,
        rationale=rationale,
        confident=False,
        point_route=point_route,
        band=band,
        pass_at_1=p1,
        pass_at_k=pk,
        k=k_i,
        ci=(lo, hi),
        endpoint_routes=endpoints,
        bands=edges,
    )


def _point_reason(band: str, p1: float, pk: float, k: int, edges: RouteBands) -> str:
    if band == "saturated":
        return (
            f"pass@1={p1:.3f} exceeds {edges.saturated_lo:.2f}: the target model already solves "
            "this almost every time, so there is no gradient to learn from and the task is a "
            "contamination candidate."
        )
    if band == "thin":
        return (
            f"pass@1={p1:.3f} falls in ({edges.gold_hi:.2f}, {edges.saturated_lo:.2f}], the band "
            "the proposal leaves implicit. Policy: reject. Roughly one sample in "
            f"{max(1, round(1.0 / max(1e-9, 1.0 - p1)))} carries signal, which does not justify "
            "SME time against a gold-band alternative. This is thin, not saturated - no "
            "contamination is implied."
        )
    if band == "gold":
        return (
            f"pass@1={p1:.3f} sits inside the gold band [{edges.gold_lo:.2f}, {edges.gold_hi:.2f}]: "
            "the target both succeeds and fails often enough for the task to carry maximum "
            "training signal."
        )
    return (
        f"pass@1={p1:.3f} is below {edges.gold_lo:.2f} but pass@{k}={pk:.3f} > 0: hard, and "
        "demonstrably reachable. Frontier."
    )


def _crossed_edges(lo: float, hi: float, edges: RouteBands) -> str:
    names = []
    for value, label in (
        (edges.gold_lo, f"frontier/gold {edges.gold_lo:.2f}"),
        (edges.gold_hi, f"gold/reject {edges.gold_hi:.2f}"),
    ):
        if lo < value <= hi:
            names.append(label)
    return " and ".join(names)


def route(
    pass1: float,
    passk: float,
    k: int = 8,
    *,
    ci: tuple[float, float] | None = None,
    bands: RouteBands | None = None,
) -> tuple[str, str]:
    """``(route, rationale)``. Thin wrapper over :func:`route_decision`."""
    return route_decision(pass1, passk, k, ci=ci, bands=bands).as_tuple()


# --------------------------------------------------------------------------- #
# bank-level driver (the CLI entry point)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskCalibration:
    task_id: str
    path: str
    estimate: PassKEstimate
    decision: RouteDecision
    contamination: Any  # ContaminationSignal; typed loosely to avoid an import cycle
    trace_stats: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "path": self.path,
            "estimate": self.estimate.as_dict(),
            "decision": self.decision.as_dict(),
            "contamination": self.contamination.as_dict(),
            "trace_stats": dict(self.trace_stats),
        }


@dataclass(frozen=True)
class BankCalibration:
    model: str
    k: int
    n_samples: int
    temperature: float
    offline: bool
    generated_utc: str
    results: tuple[TaskCalibration, ...]
    errors: tuple[dict[str, str], ...] = ()

    @property
    def route_counts(self) -> dict[str, int]:
        counts = {r: 0 for r in ROUTES}
        for item in self.results:
            counts[item.decision.route] += 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "k": self.k,
            "n_samples": self.n_samples,
            "temperature": self.temperature,
            "offline": self.offline,
            "generated_utc": self.generated_utc,
            "n_tasks": len(self.results),
            "route_counts": self.route_counts,
            "n_confident": sum(1 for r in self.results if r.decision.confident),
            "n_contaminated": sum(1 for r in self.results if r.contamination.flagged),
            "n_contamination_unassessed": sum(
                1 for r in self.results if not r.contamination.assessed
            ),
            "tasks": [r.as_dict() for r in self.results],
            "errors": [dict(e) for e in self.errors],
        }


def _default_prompt(task: Any) -> str:
    mutation = getattr(task, "mutation", None)
    cls = getattr(mutation, "cls", "unknown") if mutation is not None else "unknown"
    return (
        f"The following {getattr(task, 'domain', 'pytorch')} implementation contains a defect "
        f"of class {cls!r}. Find it and return a corrected implementation.\n\n"
        f"{getattr(task, 'mutant_code', '')}\n"
    )


def _generate(model: Any, prompt: str, n: int, temperature: float, task_id: str) -> list[str]:
    """Call ``model.generate``, passing ``task_id`` only if it accepts one."""
    import inspect

    try:
        params = inspect.signature(model.generate).parameters
    except (TypeError, ValueError):
        params = {}
    if "task_id" in params:
        return list(model.generate(prompt, n=n, temperature=temperature, task_id=task_id))
    return list(model.generate(prompt, n, temperature))


def _measured_trace_stats(completions: Sequence[str]) -> dict[str, Any]:
    """Only what we actually measured. Nothing is invented here.

    ``mean_tokens`` is a whitespace token count, not a tokenizer count, and says
    so. Backtrack counts and entropy cannot be recovered from a finished
    completion string, so they are simply absent - which makes
    ``flag_contamination`` report the task as unassessed rather than clean.
    """
    if not completions:
        return {"token_counter": "whitespace", "n_completions": 0}
    counts = [len(c.split()) for c in completions]
    return {
        "mean_tokens": sum(counts) / len(counts),
        "token_counter": "whitespace",
        "n_completions": len(counts),
    }


def calibrate_bank(
    bank_dir: Path | str,
    *,
    model: str | Any = "stub",
    k: int = 8,
    cfg: Config | None = None,
    verify: Any = None,
    n_samples: int | None = None,
    temperature: float | None = None,
    trace_stats: Mapping[str, Mapping[str, Any]] | None = None,
    write_back: bool = False,
) -> BankCalibration:
    """Sample every task in ``bank_dir``, estimate pass@k, and route each one.

    ``model`` is a spec string (``"stub"`` by default, which never touches the
    network) or an already-constructed ``TargetModel``. ``verify`` is the hook
    into the real grading path; with the stub it is unnecessary because stub
    completions carry their own correctness marker.

    Per-task ``trace_stats`` may be supplied by the caller (keyed by task id) to
    feed the contamination detector the statistics that cannot be recovered from
    a completion string. Tasks with no such statistics are reported as
    contamination-unassessed, not contamination-free.
    """
    from ..schema import CalibrationRecord, Task
    from .irt import flag_contamination
    from .models import StubModel, grade_completion, resolve_model

    config = cfg or Config()
    n = int(n_samples if n_samples is not None else config.calibration_samples)
    temp = float(temperature if temperature is not None else config.calibration_temperature)
    k_i = int(k)
    if n < 1:
        raise ValueError(f"calibration needs at least one sample per task; got n={n}")
    if k_i < 1:
        raise ValueError(f"k must be >= 1: {k_i}")

    root = Path(bank_dir)
    if not root.exists():
        raise CrucibleError("bank directory not found", path=str(root))
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in (".yaml", ".yml"))

    target = resolve_model(model) if isinstance(model, str) else model
    target_name = str(getattr(target, "name", model))
    offline = isinstance(target, StubModel)

    results: list[TaskCalibration] = []
    errors: list[dict[str, str]] = []

    for path in paths:
        try:
            task = Task.load(path)
        except (OSError, ValueError, CrucibleError) as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue

        prompt = task.prompt or _default_prompt(task)
        try:
            completions = _generate(target, prompt, n, temp, task.task_id)
        except CrucibleError:
            raise
        except (RuntimeError, OSError, ValueError) as exc:
            errors.append(
                {"path": str(path), "error": f"generation failed: {type(exc).__name__}: {exc}"}
            )
            continue

        if len(completions) != n:
            errors.append(
                {
                    "path": str(path),
                    "error": f"model returned {len(completions)} completions, expected {n}",
                }
            )
            continue

        n_correct = sum(
            1 for c in completions if grade_completion(c, verify=verify, task=task).correct
        )
        est = estimate(n, n_correct, k_i, ci_level=config.ci_level)
        decision = route_decision(est.pass_at_1, est.pass_at_k, est.effective_k, ci=est.ci)

        measured = _measured_trace_stats(completions)
        supplied = dict((trace_stats or {}).get(task.task_id, {}))
        merged: dict[str, Any] = {**measured, **supplied}
        contamination = flag_contamination(task.task_id, est.pass_at_1, merged)

        results.append(
            TaskCalibration(
                task_id=task.task_id,
                path=str(path),
                estimate=est,
                decision=decision,
                contamination=contamination,
                trace_stats=merged,
            )
        )

        if write_back:
            record = CalibrationRecord(
                model=target_name,
                k=est.effective_k,
                n_samples=est.n_samples,
                n_correct=est.n_correct,
                pass_at_1=est.pass_at_1,
                pass_at_k=est.pass_at_k,
                route=decision.route,
                rationale=decision.rationale,
                trace_stats={
                    **merged,
                    "ci": list(est.ci),
                    "ci_level": est.ci_level,
                    "ci_method": est.ci_method,
                    "confident": decision.confident,
                    "contamination": contamination.as_dict(),
                },
            )
            task.model_copy(update={"calibration": record}).save(path)

    return BankCalibration(
        model=target_name,
        k=k_i,
        n_samples=n,
        temperature=temp,
        offline=offline,
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        results=tuple(results),
        errors=tuple(errors),
    )
