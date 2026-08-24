"""The seven headline metrics.

The proposal's central claim is that these numbers are *unmeasured* almost
everywhere, and the whole project exists to measure them. So the one rule this
module obeys above all others: **a metric with no input is ``unmeasured``, never
zero.** Reporting 0% catch rate because no red-team suite ran, or 0 SME hours
because no provenance was recorded, would reproduce exactly the error the
project was built to fix -- a fabricated number that looks like evidence.

``Metric.value is None`` means unmeasured; ``Metric.as_dict()["value"]`` is the
literal string ``"unmeasured"`` so that a JSON consumer cannot mistake it for a
measurement either.

Targets are *policy*: they are inputs (``MetricTargets``), editable, and never
derived from the data. Values are *measurements*: derived from the data only.
Keeping the two apart is what makes "met" meaningful.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..schema import Task, TaskVerdict
from ..taxonomy import CELLS, coverage as taxonomy_coverage

logger = logging.getLogger(__name__)

#: Sentinel emitted in JSON payloads in place of a number that was never measured.
UNMEASURED = "unmeasured"

STATUS_MET = "met"
STATUS_UNMET = "unmet"
STATUS_UNMEASURED = "unmeasured"


@dataclass(frozen=True)
class MetricTargets:
    """Policy thresholds. Inputs to the report, never inferred from the bank."""

    headline: float = 1.00
    gold_band: float = 0.60
    alpha_share: float = 1.00
    redteam_catch: float = 1.00
    coverage_fill: float = 1.00
    silent_share: float = 0.35
    #: SME hours per accepted task is a budget, not a floor; lower is better.
    sme_hours: float = 2.00

    # Band and threshold definitions the metrics are computed against.
    gold_band_lo: float = 0.10
    gold_band_hi: float = 0.70
    alpha_threshold: float = 0.70
    target_per_cell: int = 5


DEFAULT_TARGETS = MetricTargets()


@dataclass(frozen=True)
class Metric:
    """One headline number, its target, and whether it was measured at all."""

    key: str
    name: str
    value: float | None
    target: float | None
    n: int
    unit: str = "fraction"  # fraction | hours | count
    lower_is_better: bool = False
    detail: str = ""
    reason: str = ""  # why it is unmeasured; empty when measured
    numerator: float | None = None
    denominator: float | None = None

    @property
    def measured(self) -> bool:
        return self.value is not None

    @property
    def met(self) -> bool | None:
        """None when unmeasured. An unmeasured metric is not a met metric."""
        if self.value is None or self.target is None:
            return None
        if self.lower_is_better:
            return self.value <= self.target
        return self.value >= self.target

    @property
    def status(self) -> str:
        if not self.measured:
            return STATUS_UNMEASURED
        return STATUS_MET if self.met else STATUS_UNMET

    def display_value(self) -> str:
        if self.value is None:
            return UNMEASURED
        return _format(self.value, self.unit)

    def display_target(self) -> str:
        if self.target is None:
            return "n/a"
        prefix = "<= " if self.lower_is_better else ">= "
        return prefix + _format(self.target, self.unit)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "value": self.value if self.measured else UNMEASURED,
            "display_value": self.display_value(),
            "target": self.target,
            "display_target": self.display_target(),
            "met": self.met if self.measured else UNMEASURED,
            "status": self.status,
            "n": self.n,
            "unit": self.unit,
            "lower_is_better": self.lower_is_better,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "detail": self.detail,
            "reason": self.reason,
        }


def _format(value: float, unit: str) -> str:
    if unit == "fraction":
        return f"{100.0 * value:.1f}%"
    if unit == "hours":
        return f"{value:.2f} h"
    return f"{value:g}"


def _unmeasured(
    key: str,
    name: str,
    target: float | None,
    reason: str,
    *,
    unit: str = "fraction",
    lower_is_better: bool = False,
    n: int = 0,
) -> Metric:
    return Metric(
        key=key,
        name=name,
        value=None,
        target=target,
        n=n,
        unit=unit,
        lower_is_better=lower_is_better,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LoadError:
    path: str
    error: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "error": self.error}


@dataclass
class BankLoad:
    root: str
    tasks: list[Task] = field(default_factory=list)
    errors: list[LoadError] = field(default_factory=list)


@dataclass
class VerdictLoad:
    root: str
    verdicts: list[TaskVerdict] = field(default_factory=list)
    errors: list[LoadError] = field(default_factory=list)


def load_bank(bank_dir: Path | str) -> BankLoad:
    """Load every ``*.yaml`` task under ``bank_dir``.

    A file that fails to parse is *reported*, never dropped: a bank that
    silently shrank is a bank whose coverage numbers are wrong.
    """
    root = Path(bank_dir)
    out = BankLoad(root=str(root))
    if not root.exists():
        out.errors.append(LoadError(str(root), "bank directory does not exist"))
        return out
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(p for p in root.rglob("*") if p.suffix in (".yaml", ".yml") and p.is_file())
    for path in paths:
        if path.name.startswith("."):
            continue
        try:
            out.tasks.append(Task.load(path))
        except (OSError, ValueError, TypeError) as exc:
            out.errors.append(LoadError(str(path), f"{type(exc).__name__}: {exc}"))
            logger.warning("task %s could not be loaded: %s", path, exc)
    return out


def load_verdicts(verdicts_dir: Path | str) -> VerdictLoad:
    """Load every ``*.json`` TaskVerdict under ``verdicts_dir``."""
    root = Path(verdicts_dir)
    out = VerdictLoad(root=str(root))
    if not root.exists():
        out.errors.append(LoadError(str(root), "verdict directory does not exist"))
        return out
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(p for p in root.rglob("*.json") if p.is_file())
    for path in paths:
        if path.name.startswith("."):
            continue
        try:
            out.verdicts.append(TaskVerdict.load(path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            out.errors.append(LoadError(str(path), f"{type(exc).__name__}: {exc}"))
            logger.warning("verdict %s could not be loaded: %s", path, exc)
    return out


def index_verdicts(verdicts: Iterable[TaskVerdict]) -> dict[str, TaskVerdict]:
    """task_id -> verdict. A later file for the same id wins, and says so."""
    out: dict[str, TaskVerdict] = {}
    for v in verdicts:
        if v.task_id in out:
            logger.warning("duplicate verdict for task %s; keeping the last one", v.task_id)
        out[v.task_id] = v
    return out


def task_universe(tasks: Sequence[Task], verdicts: Sequence[TaskVerdict]) -> list[str]:
    """Every task id the report knows about: shipped tasks plus stray verdicts.

    A verdict with no task in the bank still counts (something was verified that
    is not in the bank -- worth seeing), and a task with no verdict counts
    against the headline (it was shipped unverified).
    """
    seen: dict[str, None] = {}
    for t in tasks:
        seen.setdefault(t.task_id, None)
    for v in verdicts:
        seen.setdefault(v.task_id, None)
    return list(seen)


def is_accepted(verdict: TaskVerdict | None) -> bool:
    """The one invariant, in code: executed AND passed every applicable oracle."""
    return verdict is not None and verdict.verdict == "PASS" and verdict.executed


# --------------------------------------------------------------------------- #
# metric 1 -- the headline
# --------------------------------------------------------------------------- #


def headline_execution_rate(
    tasks: Sequence[Task],
    verdicts: Sequence[TaskVerdict],
    targets: MetricTargets = DEFAULT_TARGETS,
) -> Metric:
    """Share of shipped tasks whose reference solution was executed and passed.

    Numerator: verdict == PASS *and* executed. Everything else -- SKIP, FAIL,
    ERROR, and tasks with no verdict at all -- counts against.
    """
    universe = task_universe(tasks, verdicts)
    if not universe:
        return _unmeasured(
            "headline_executed_and_passed",
            "Reference solutions executed and passed every oracle",
            targets.headline,
            "no tasks and no verdicts were found; nothing has been verified",
        )
    by_id = index_verdicts(verdicts)
    passed = 0
    pass_not_executed = 0
    unverified = 0
    for tid in universe:
        v = by_id.get(tid)
        if v is None:
            unverified += 1
            continue
        if v.verdict == "PASS":
            if v.executed:
                passed += 1
            else:
                # A pass that never ran is the exact defect being measured.
                pass_not_executed += 1
    n = len(universe)
    bits = [f"{passed}/{n} executed and passed"]
    if unverified:
        bits.append(f"{unverified} shipped with no verdict")
    if pass_not_executed:
        bits.append(f"{pass_not_executed} PASS without execution (counted against)")
    return Metric(
        key="headline_executed_and_passed",
        name="Reference solutions executed and passed every oracle",
        value=passed / n,
        target=targets.headline,
        n=n,
        numerator=float(passed),
        denominator=float(n),
        detail="; ".join(bits),
    )


# --------------------------------------------------------------------------- #
# metric 2 -- difficulty band
# --------------------------------------------------------------------------- #


def gold_band_rate(
    tasks: Sequence[Task],
    targets: MetricTargets = DEFAULT_TARGETS,
) -> Metric:
    """Share of calibrated tasks whose pass@1 sits in the gold band [0.1, 0.7]."""
    name = f"Tasks in the gold band pass@1 in [{targets.gold_band_lo:g},{targets.gold_band_hi:g}]"
    calibrated = [t for t in tasks if t.calibration is not None]
    if not tasks:
        return _unmeasured("gold_band", name, targets.gold_band, "the bank is empty")
    if not calibrated:
        return _unmeasured(
            "gold_band",
            name,
            targets.gold_band,
            f"none of the {len(tasks)} tasks carries a calibration record; run 'crucible calibrate'",
        )
    in_band = 0
    for t in calibrated:
        p1 = float(t.calibration.pass_at_1)  # type: ignore[union-attr]
        if targets.gold_band_lo <= p1 <= targets.gold_band_hi:
            in_band += 1
    n = len(calibrated)
    detail = f"{in_band}/{n} calibrated tasks in band"
    if n < len(tasks):
        detail += f"; {len(tasks) - n} of {len(tasks)} tasks are uncalibrated and excluded"
    return Metric(
        key="gold_band",
        name=name,
        value=in_band / n,
        target=targets.gold_band,
        n=n,
        numerator=float(in_band),
        denominator=float(n),
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# metric 3 -- rubric reliability
# --------------------------------------------------------------------------- #


def normalize_alphas(source: Any) -> dict[str, float]:
    """Accept the shapes an IRR report plausibly takes; drop what is not a number.

    Understood: ``{criterion: alpha}``, ``{"criteria": {...}}``,
    ``{"criteria": [{"id": ..., "alpha": ...}]}``, ``{"alphas": {...}}``.
    A NaN alpha means the criterion could not be scored and is dropped rather
    than counted as a failure to reach the threshold.
    """
    if source is None:
        return {}
    if isinstance(source, Mapping):
        for key in ("criteria", "alphas", "alpha_per_criterion"):
            if key in source:
                return normalize_alphas(source[key])
        out: dict[str, float] = {}
        for k, v in source.items():
            alpha = _as_alpha(v)
            if alpha is not None:
                out[str(k)] = alpha
        return out
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        out = {}
        for i, item in enumerate(source):
            if not isinstance(item, Mapping):
                continue
            cid = str(item.get("id") or item.get("criterion") or f"criterion_{i}")
            alpha = _as_alpha(item.get("alpha", item.get("value")))
            if alpha is not None:
                out[cid] = alpha
        return out
    return {}


def _as_alpha(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, Mapping):
        for key in ("alpha", "value"):
            if key in value:
                return _as_alpha(value[key])
    return None


def rubric_alpha_rate(
    alphas: Any,
    targets: MetricTargets = DEFAULT_TARGETS,
) -> Metric:
    """Share of rubric criteria whose Krippendorff alpha clears the threshold."""
    name = f"Rubric criteria with Krippendorff alpha >= {targets.alpha_threshold:g}"
    table = normalize_alphas(alphas)
    if not table:
        return _unmeasured(
            "rubric_alpha",
            name,
            targets.alpha_share,
            "no inter-rater ratings were supplied; alpha cannot be computed from the bank alone",
        )
    good = sum(1 for a in table.values() if a >= targets.alpha_threshold)
    n = len(table)
    failing = sorted(
        ((k, v) for k, v in table.items() if v < targets.alpha_threshold), key=lambda kv: kv[1]
    )
    detail = f"{good}/{n} criteria clear the threshold"
    if failing:
        # Naming them is the point: a criterion below alpha needs a rewrite.
        shown = ", ".join(f"{k}={v:.2f}" for k, v in failing[:4])
        detail += f"; below threshold: {shown}"
    return Metric(
        key="rubric_alpha",
        name=name,
        value=good / n,
        target=targets.alpha_share,
        n=n,
        numerator=float(good),
        denominator=float(n),
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# metric 4 -- red team
# --------------------------------------------------------------------------- #


def normalize_redteam(source: Any) -> tuple[int, int, list[str]]:
    """(n_caught, n_attacks, uncaught_ids) from a red-team report payload.

    Understood: ``{"attacks": [{"id", "caught"|"catcher"}]}`` and the
    pre-aggregated ``{"n_caught", "n_attacks"}``.
    """
    if not isinstance(source, Mapping):
        return 0, 0, []
    attacks = source.get("attacks")
    if isinstance(attacks, Sequence) and not isinstance(attacks, (str, bytes)):
        total = 0
        caught = 0
        uncaught: list[str] = []
        for i, item in enumerate(attacks):
            if not isinstance(item, Mapping):
                continue
            total += 1
            aid = str(item.get("id") or f"attack_{i}")
            if "caught" in item:
                hit = bool(item["caught"])
            else:
                catcher = item.get("catcher") or item.get("caught_by")
                hit = bool(catcher)
            if hit:
                caught += 1
            else:
                uncaught.append(aid)
        return caught, total, uncaught
    n_attacks = source.get("n_attacks", source.get("total"))
    n_caught = source.get("n_caught", source.get("caught"))
    if isinstance(n_attacks, (int, float)) and isinstance(n_caught, (int, float)):
        return int(n_caught), int(n_attacks), [str(x) for x in source.get("uncaught", [])]
    return 0, 0, []


def redteam_catch_rate(
    source: Any,
    targets: MetricTargets = DEFAULT_TARGETS,
) -> Metric:
    """Share of red-team attacks that some oracle caught."""
    name = "Reward-hack attacks caught by the harness"
    caught, total, uncaught = normalize_redteam(source)
    if total <= 0:
        return _unmeasured(
            "redteam_catch",
            name,
            targets.redteam_catch,
            "no red-team report was supplied; run 'crucible redteam --json'",
        )
    detail = f"{caught}/{total} attacks caught"
    if uncaught:
        # An attack caught by no oracle is a grader defect, so name it.
        detail += "; UNCAUGHT (grader defect): " + ", ".join(uncaught[:6])
    return Metric(
        key="redteam_catch",
        name=name,
        value=caught / total,
        target=targets.redteam_catch,
        n=total,
        numerator=float(caught),
        denominator=float(total),
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# metric 5 -- taxonomy coverage
# --------------------------------------------------------------------------- #


def taxonomy_coverage_rate(
    tasks: Sequence[Task],
    targets: MetricTargets = DEFAULT_TARGETS,
) -> Metric:
    """Share of the 48 taxonomy cells holding at least ``target_per_cell`` tasks."""
    name = f"Taxonomy cells with >= {targets.target_per_cell} tasks (of {len(CELLS)})"
    if not tasks:
        return _unmeasured(
            "taxonomy_coverage",
            name,
            targets.coverage_fill,
            "the bank is empty; no cell has been filled or measured",
            n=len(CELLS),
        )
    grid = taxonomy_coverage(tasks, target_per_cell=targets.target_per_cell)
    filled = len(grid.filled())
    empty = sum(1 for c, n in grid.counts.items() if n == 0)
    return Metric(
        key="taxonomy_coverage",
        name=name,
        value=grid.fill_rate(),
        target=targets.coverage_fill,
        n=len(CELLS),
        numerator=float(filled),
        denominator=float(len(CELLS)),
        detail=f"{filled}/{len(CELLS)} cells at target; {empty} cells still empty; "
        f"{grid.total()} tasks in the bank",
    )


# --------------------------------------------------------------------------- #
# metric 6 -- silent-failure share
# --------------------------------------------------------------------------- #


def silent_tier_share(
    tasks: Sequence[Task],
    targets: MetricTargets = DEFAULT_TARGETS,
) -> Metric:
    """T5 + T6 share of the bank -- the tiers where the failure leaks no signal."""
    name = "T5+T6 (silent failure) share of the bank"
    if not tasks:
        return _unmeasured(
            "silent_share", name, targets.silent_share, "the bank is empty"
        )
    grid = taxonomy_coverage(tasks, target_per_cell=targets.target_per_cell)
    share = grid.silent_share()
    by_tier = grid.by_tier()
    silent = by_tier.get("T5", 0) + by_tier.get("T6", 0)
    return Metric(
        key="silent_share",
        name=name,
        value=share,
        target=targets.silent_share,
        n=len(tasks),
        numerator=float(silent),
        denominator=float(len(tasks)),
        detail=f"{silent}/{len(tasks)} tasks are T5 or T6 "
        f"(T5={by_tier.get('T5', 0)}, T6={by_tier.get('T6', 0)})",
    )


# --------------------------------------------------------------------------- #
# metric 7 -- SME cost
# --------------------------------------------------------------------------- #

#: provenance key -> multiplier into hours
_SME_KEYS: tuple[tuple[str, float], ...] = (
    ("sme_hours", 1.0),
    ("sme_hours_total", 1.0),
    ("sme_minutes", 1.0 / 60.0),
    ("sme_seconds", 1.0 / 3600.0),
)


def sme_hours_from_provenance(provenance: Mapping[str, Any]) -> float | None:
    """Hours of SME time recorded for one task, or None if never recorded.

    None is not zero. A task with no recorded SME time tells us nothing about
    how long it took, and averaging it in as 0 would understate the real cost.
    """
    if not isinstance(provenance, Mapping):
        return None
    sources: list[Mapping[str, Any]] = [provenance]
    nested = provenance.get("sme")
    if isinstance(nested, Mapping):
        sources.append(nested)
        # {"sme": {"hours": 1.5}}
        for key, mult in (("hours", 1.0), ("minutes", 1.0 / 60.0), ("seconds", 1.0 / 3600.0)):
            v = nested.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v) * mult
    for src in sources:
        for key, mult in _SME_KEYS:
            v = src.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v) * mult
    return None


def sme_hours_per_accepted_task(
    tasks: Sequence[Task],
    verdicts: Sequence[TaskVerdict],
    targets: MetricTargets = DEFAULT_TARGETS,
) -> Metric:
    """Mean SME hours per *accepted* task, read from task provenance."""
    name = "SME hours per accepted task"
    by_id = index_verdicts(verdicts)
    accepted = [t for t in tasks if is_accepted(by_id.get(t.task_id))]
    if not tasks:
        return _unmeasured(
            "sme_hours",
            name,
            targets.sme_hours,
            "the bank is empty",
            unit="hours",
            lower_is_better=True,
        )
    if not accepted:
        return _unmeasured(
            "sme_hours",
            name,
            targets.sme_hours,
            "no task was executed and passed, so there is no accepted task to divide by",
            unit="hours",
            lower_is_better=True,
        )
    hours = [(t.task_id, sme_hours_from_provenance(t.provenance)) for t in accepted]
    recorded = [(tid, h) for tid, h in hours if h is not None]
    if not recorded:
        return _unmeasured(
            "sme_hours",
            name,
            targets.sme_hours,
            f"none of the {len(accepted)} accepted tasks records SME time in provenance "
            "(expected key 'sme_hours', 'sme_minutes' or 'sme_seconds')",
            unit="hours",
            lower_is_better=True,
            n=len(accepted),
        )
    total = sum(h for _, h in recorded)
    n = len(recorded)
    detail = f"{total:.2f} SME hours over {n} accepted tasks"
    if n < len(accepted):
        detail += f"; {len(accepted) - n} accepted tasks record no SME time and are excluded"
    return Metric(
        key="sme_hours",
        name=name,
        value=total / n,
        target=targets.sme_hours,
        n=n,
        unit="hours",
        lower_is_better=True,
        numerator=total,
        denominator=float(n),
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

METRIC_KEYS: tuple[str, ...] = (
    "headline_executed_and_passed",
    "gold_band",
    "rubric_alpha",
    "redteam_catch",
    "taxonomy_coverage",
    "silent_share",
    "sme_hours",
)


def all_metrics(
    tasks: Sequence[Task],
    verdicts: Sequence[TaskVerdict],
    *,
    irr: Any = None,
    redteam: Any = None,
    targets: MetricTargets = DEFAULT_TARGETS,
) -> list[Metric]:
    """The seven, in proposal order. The headline is always first."""
    return [
        headline_execution_rate(tasks, verdicts, targets),
        gold_band_rate(tasks, targets),
        rubric_alpha_rate(irr, targets),
        redteam_catch_rate(redteam, targets),
        taxonomy_coverage_rate(tasks, targets),
        silent_tier_share(tasks, targets),
        sme_hours_per_accepted_task(tasks, verdicts, targets),
    ]


def skip_breakdown(
    verdicts: Sequence[TaskVerdict],
    kinds: Sequence[str] = ("SKIP", "ERROR"),
) -> list[dict[str, Any]]:
    """Per (oracle, verdict, reason) counts -- why things were not verified.

    This is the most useful table in the report: it is the list of claims the
    machine could not check, which is exactly what a silent harness hides.
    """
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for v in verdicts:
        for r in v.oracle_results:
            if r.verdict not in kinds:
                continue
            key = (r.oracle, r.verdict, r.reason)
            row = groups.setdefault(
                key,
                {
                    "oracle": r.oracle,
                    "verdict": r.verdict,
                    "reason": r.reason,
                    "count": 0,
                    "task_ids": [],
                },
            )
            row["count"] += 1
            if len(row["task_ids"]) < 5:
                row["task_ids"].append(v.task_id)
    rows = list(groups.values())
    rows.sort(key=lambda r: (-r["count"], r["oracle"], r["verdict"]))
    return rows


def verdict_counts(tasks: Sequence[Task], verdicts: Sequence[TaskVerdict]) -> dict[str, int]:
    """Counts over the whole universe, including UNVERIFIED bank tasks."""
    by_id = index_verdicts(verdicts)
    counts: dict[str, int] = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0, "UNVERIFIED": 0}
    for tid in task_universe(tasks, verdicts):
        v = by_id.get(tid)
        if v is None:
            counts["UNVERIFIED"] += 1
        else:
            counts[v.verdict] = counts.get(v.verdict, 0) + 1
    return counts


@dataclass
class MetricsReport:
    bank_dir: str
    verdicts_dir: str
    generated_utc: str
    tasks: list[Task]
    verdicts: list[TaskVerdict]
    metrics: list[Metric]
    errors: list[LoadError] = field(default_factory=list)
    targets: MetricTargets = DEFAULT_TARGETS
    irr_source: str = ""
    redteam_source: str = ""

    def metric(self, key: str) -> Metric:
        for m in self.metrics:
            if m.key == key:
                return m
        raise KeyError(f"unknown metric {key!r}; known: {[m.key for m in self.metrics]}")

    @property
    def headline(self) -> Metric:
        return self.metric("headline_executed_and_passed")

    def verdict_counts(self) -> dict[str, int]:
        return verdict_counts(self.tasks, self.verdicts)

    def skip_breakdown(self) -> list[dict[str, Any]]:
        return skip_breakdown(self.verdicts)

    def combined_verdict(self) -> str:
        """Bank-wide verdict. A shipped task with no verdict is a SKIP, not a pass."""
        by_id = index_verdicts(self.verdicts)
        universe = task_universe(self.tasks, self.verdicts)
        if not universe:
            return "SKIP"
        states = [by_id[tid].verdict if tid in by_id else "SKIP" for tid in universe]
        return TaskVerdict.combine(states)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_utc": self.generated_utc,
            "bank_dir": self.bank_dir,
            "verdicts_dir": self.verdicts_dir,
            "n_tasks": len(self.tasks),
            "n_verdicts": len(self.verdicts),
            "verdict_counts": self.verdict_counts(),
            "combined_verdict": self.combined_verdict(),
            "metrics": [m.as_dict() for m in self.metrics],
            "skip_breakdown": self.skip_breakdown(),
            "load_errors": [e.as_dict() for e in self.errors],
            "irr_source": self.irr_source,
            "redteam_source": self.redteam_source,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=str)


def read_json_file(path: Path | str | None) -> tuple[Any, str]:
    """(payload, source_label). A missing or malformed file yields (None, why)."""
    if path is None:
        return None, ""
    p = Path(path)
    if not p.exists():
        return None, f"{p} (not found)"
    try:
        return json.loads(p.read_text(encoding="utf-8")), str(p)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("%s could not be read: %s", p, exc)
        return None, f"{p} ({type(exc).__name__}: {exc})"


def compute(
    bank_dir: Path | str,
    verdicts_dir: Path | str,
    *,
    irr_path: Path | str | None = None,
    redteam_path: Path | str | None = None,
    targets: MetricTargets = DEFAULT_TARGETS,
) -> MetricsReport:
    """Load a bank and its verdicts, then compute all seven metrics."""
    bank = load_bank(bank_dir)
    verds = load_verdicts(verdicts_dir)
    irr_payload, irr_src = read_json_file(irr_path)
    redteam_payload, redteam_src = read_json_file(redteam_path)
    return MetricsReport(
        bank_dir=str(bank_dir),
        verdicts_dir=str(verdicts_dir),
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        tasks=bank.tasks,
        verdicts=verds.verdicts,
        metrics=all_metrics(
            bank.tasks,
            verds.verdicts,
            irr=irr_payload,
            redteam=redteam_payload,
            targets=targets,
        ),
        errors=[*bank.errors, *verds.errors],
        targets=targets,
        irr_source=irr_src,
        redteam_source=redteam_src,
    )


__all__ = [
    "UNMEASURED",
    "STATUS_MET",
    "STATUS_UNMET",
    "STATUS_UNMEASURED",
    "Metric",
    "MetricTargets",
    "DEFAULT_TARGETS",
    "METRIC_KEYS",
    "LoadError",
    "BankLoad",
    "VerdictLoad",
    "MetricsReport",
    "load_bank",
    "load_verdicts",
    "index_verdicts",
    "task_universe",
    "is_accepted",
    "headline_execution_rate",
    "gold_band_rate",
    "rubric_alpha_rate",
    "redteam_catch_rate",
    "taxonomy_coverage_rate",
    "silent_tier_share",
    "sme_hours_per_accepted_task",
    "sme_hours_from_provenance",
    "normalize_alphas",
    "normalize_redteam",
    "all_metrics",
    "skip_breakdown",
    "verdict_counts",
    "read_json_file",
    "compute",
]
