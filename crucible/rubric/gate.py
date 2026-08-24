"""The reliability gate: alpha >= 0.67 keeps a criterion, below it demands a rewrite.

The wording of the failure matters as much as the threshold, so it is fixed
here rather than left to whoever reads the report.

**A criterion below the gate is an ambiguous criterion, not a set of bad
raters.** Two competent people reading the same anchor text and scoring the same
artefact differently have demonstrated that the anchor admits two readings. The
remedy is to rewrite the anchors until it does not - to name the observable that
separates a 3 from a 5 - and not to retrain, replace, or average over the
raters. Averaging an ambiguous criterion produces a precise number that measures
nothing, which is the failure mode this whole layer exists to catch.

The evidence attached to a rewrite flag is therefore the *disagreeing units*:
the specific items, with the specific pairs of ratings, ordered by how much they
contributed to the disagreement. Those items are the ambiguity, made concrete;
they are what a rewrite of the anchors has to disambiguate.

Machine-probed criteria are exempt. They carry no human variance at all, so a
missing alpha there is correct rather than a gap - see ``spec.MACHINE_PROBEABLE``
for the rule that puts every probeable criterion in that category.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from ..schema import RubricSpec
from .irr import (
    AlphaResult,
    Metric,
    UnitDisagreement,
    disagreeing_units,
    krippendorff_alpha,
    running_alpha,
)
from .sequences import MonitorTrace, monitor_alpha

logger = logging.getLogger(__name__)

#: The proposal's threshold. Krippendorff's own guidance: 0.800 for conclusions
#: you would act on, 0.667 as the floor below which even tentative conclusions
#: are unsafe. A rubric criterion is a measuring instrument, so the floor is the
#: right gate for "may this criterion stay in the rubric at all".
GATE_ALPHA = 0.67

Action = Literal["keep", "rewrite", "unmeasured", "exempt_machine_probed"]


@dataclass(frozen=True)
class CriterionGate:
    """One criterion's verdict, with the reason and the evidence behind it."""

    criterion_id: str
    alpha: float
    threshold: float
    action: Action
    message: str
    n_units: int = 0
    n_raters: int = 0
    metric: str = ""
    result: AlphaResult | None = None
    evidence: list[UnitDisagreement] = field(default_factory=list)
    trend: list[float] = field(default_factory=list)
    monitor: MonitorTrace | None = None

    @property
    def keep(self) -> bool:
        return self.action in ("keep", "exempt_machine_probed")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.criterion_id,
            "alpha": None if not math.isfinite(self.alpha) else float(self.alpha),
            "threshold": self.threshold,
            "action": self.action,
            "keep": self.keep,
            "message": self.message,
            "n_units": self.n_units,
            "n_raters": self.n_raters,
            "metric": self.metric,
            "trend": [None if not math.isfinite(a) else float(a) for a in self.trend],
            "evidence": [u.as_dict() for u in self.evidence],
            "result": self.result.as_dict() if self.result else None,
            "monitor": self.monitor.as_dict() if self.monitor else None,
        }


@dataclass(frozen=True)
class GateReport:
    threshold: float
    metric: str
    gates: list[CriterionGate] = field(default_factory=list)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.gates)

    def __len__(self) -> int:
        return len(self.gates)

    def by_id(self, criterion_id: str) -> CriterionGate:
        for g in self.gates:
            if g.criterion_id == criterion_id:
                return g
        raise KeyError(f"no gate for criterion {criterion_id!r}")

    @property
    def flagged(self) -> list[CriterionGate]:
        """Criteria that must be rewritten before the rubric ships."""
        return [g for g in self.gates if g.action == "rewrite"]

    @property
    def kept(self) -> list[CriterionGate]:
        return [g for g in self.gates if g.action == "keep"]

    @property
    def unmeasured(self) -> list[CriterionGate]:
        return [g for g in self.gates if g.action == "unmeasured"]

    @property
    def passed(self) -> bool:
        """No criterion is flagged for rewrite. Unmeasured criteria do not pass."""
        return not self.flagged and not self.unmeasured

    def alphas(self) -> dict[str, float]:
        return {g.criterion_id: g.alpha for g in self.gates if math.isfinite(g.alpha)}

    def trends(self) -> dict[str, list[float]]:
        """Running alpha per criterion, for the report's trend panel."""
        return {g.criterion_id: g.trend for g in self.gates if g.trend}

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "metric": self.metric,
            "passed": self.passed,
            "n_criteria": len(self.gates),
            "n_flagged": len(self.flagged),
            "n_unmeasured": len(self.unmeasured),
            "criteria": [g.as_dict() for g in self.gates],
        }


# --------------------------------------------------------------------------- #
# messages
# --------------------------------------------------------------------------- #


def _rewrite_message(
    criterion_id: str,
    alpha: float,
    threshold: float,
    evidence: Sequence[UnitDisagreement],
) -> str:
    head = (
        f"criterion {criterion_id!r} has Krippendorff alpha = {alpha:.3f}, below the "
        f"{threshold:.2f} gate: REWRITE THE CRITERION. This is a defect in the criterion, "
        "not in the raters -- competent people read the same anchors and scored the same "
        "artefacts differently, which means the anchors admit more than one reading. "
        "Rewrite them so each score names an observable a second reader can check; do not "
        "retrain the raters and do not average the disagreement away."
    )
    if not evidence:
        return head
    items = ", ".join(
        f"{u.unit_id} ("
        + "; ".join(f"r{p.rater_i}={p.value_i:g} vs r{p.rater_j}={p.value_j:g}" for p in u.pairs[:3])
        + ")"
        for u in evidence[:5]
    )
    return f"{head} The ambiguity is concrete in these items: {items}."


def _keep_message(criterion_id: str, alpha: float, threshold: float) -> str:
    return (
        f"criterion {criterion_id!r} has Krippendorff alpha = {alpha:.3f} >= {threshold:.2f}; "
        "raters reproduce each other well enough for the criterion to stay as written"
    )


# --------------------------------------------------------------------------- #
# gating
# --------------------------------------------------------------------------- #


def gate_criterion(
    criterion_id: str,
    matrix: Any,
    *,
    threshold: float = GATE_ALPHA,
    metric: Metric | str = "ordinal",
    orientation: str = "raters_x_units",
    unit_ids: Sequence[str] | None = None,
    bootstrap: int = 0,
    ci_level: float = 0.95,
    rng_seed: int = 1234,
    evidence_limit: int = 5,
    with_monitor: bool = False,
    cs_kind: str = "eb",
) -> CriterionGate:
    """Gate one criterion's ratings matrix (rows = raters, columns = items)."""
    result = krippendorff_alpha(
        matrix,
        metric,  # type: ignore[arg-type]
        orientation=orientation,  # type: ignore[arg-type]
        bootstrap=bootstrap,
        ci_level=ci_level,
        rng_seed=rng_seed,
    )
    trend = running_alpha(matrix, metric, orientation=orientation)  # type: ignore[arg-type]
    evidence = disagreeing_units(
        matrix,
        metric,  # type: ignore[arg-type]
        orientation=orientation,  # type: ignore[arg-type]
        unit_ids=unit_ids,
        limit=evidence_limit,
    )
    monitor = (
        monitor_alpha(
            matrix,
            metric,
            cs_kind=cs_kind,  # type: ignore[arg-type]
            orientation=orientation,
            criterion_id=criterion_id,
            unit_ids=unit_ids,
        )
        if with_monitor
        else None
    )

    if not math.isfinite(result.alpha):
        return CriterionGate(
            criterion_id=criterion_id,
            alpha=math.nan,
            threshold=threshold,
            action="unmeasured",
            message=(
                f"criterion {criterion_id!r}: alpha could not be computed -- "
                f"{result.reason or 'no reason recorded'}. An uncomputed alpha is not a pass; "
                "the criterion is ungated until more ratings exist"
            ),
            n_units=result.n_units,
            n_raters=result.n_raters,
            metric=result.metric,
            result=result,
            evidence=evidence,
            trend=trend,
            monitor=monitor,
        )

    if result.alpha >= threshold:
        return CriterionGate(
            criterion_id=criterion_id,
            alpha=result.alpha,
            threshold=threshold,
            action="keep",
            message=_keep_message(criterion_id, result.alpha, threshold),
            n_units=result.n_units,
            n_raters=result.n_raters,
            metric=result.metric,
            result=result,
            evidence=evidence,
            trend=trend,
            monitor=monitor,
        )

    return CriterionGate(
        criterion_id=criterion_id,
        alpha=result.alpha,
        threshold=threshold,
        action="rewrite",
        message=_rewrite_message(criterion_id, result.alpha, threshold, evidence),
        n_units=result.n_units,
        n_raters=result.n_raters,
        metric=result.metric,
        result=result,
        evidence=evidence,
        trend=trend,
        monitor=monitor,
    )


def gate_criteria(
    ratings: Mapping[str, Any],
    *,
    rubric: RubricSpec | None = None,
    threshold: float = GATE_ALPHA,
    metric: Metric | str = "ordinal",
    orientation: str = "raters_x_units",
    unit_ids: Sequence[str] | None = None,
    bootstrap: int = 0,
    ci_level: float = 0.95,
    rng_seed: int = 1234,
    evidence_limit: int = 5,
    with_monitor: bool = False,
    cs_kind: str = "eb",
) -> GateReport:
    """Gate every criterion in ``ratings`` (``{criterion_id: matrix}``).

    When ``rubric`` is supplied it also accounts for criteria that have no
    ratings at all: a machine-probed one is exempt (there is no human variance to
    measure), a human one is reported as ``unmeasured``, which does *not* pass.
    """
    gates: list[CriterionGate] = []
    for cid in ratings:
        gates.append(
            gate_criterion(
                str(cid),
                ratings[cid],
                threshold=threshold,
                metric=metric,
                orientation=orientation,
                unit_ids=unit_ids,
                bootstrap=bootstrap,
                ci_level=ci_level,
                rng_seed=rng_seed,
                evidence_limit=evidence_limit,
                with_monitor=with_monitor,
                cs_kind=cs_kind,
            )
        )

    if rubric is not None:
        rated = {str(c) for c in ratings}
        for criterion in rubric.criteria:
            if criterion.id in rated:
                continue
            if criterion.machine_probed:
                gates.append(
                    CriterionGate(
                        criterion_id=criterion.id,
                        alpha=math.nan,
                        threshold=threshold,
                        action="exempt_machine_probed",
                        message=(
                            f"criterion {criterion.id!r} is scored from {criterion.auto_probe!r} "
                            "and never reaches a human rater, so it carries no inter-rater "
                            "variance and no alpha is owed"
                        ),
                        metric=str(metric),
                    )
                )
            else:
                gates.append(
                    CriterionGate(
                        criterion_id=criterion.id,
                        alpha=math.nan,
                        threshold=threshold,
                        action="unmeasured",
                        message=(
                            f"criterion {criterion.id!r} is a human judgement and no ratings were "
                            "supplied for it; its reliability is unknown, which is not the same "
                            "as adequate"
                        ),
                        metric=str(metric),
                    )
                )

    return GateReport(threshold=float(threshold), metric=str(metric), gates=gates)


def rewrite_brief(gate: CriterionGate) -> str:
    """A paste-ready note for whoever rewrites a flagged criterion."""
    if gate.action != "rewrite":
        return gate.message
    lines = [gate.message, "", "Items to disambiguate first (worst disagreement first):"]
    for u in gate.evidence:
        pairs = "; ".join(
            f"rater {p.rater_i} gave {p.value_i:g}, rater {p.rater_j} gave {p.value_j:g}"
            for p in u.pairs
        )
        lines.append(f"  - {u.unit_id}: {pairs}")
    lines.append("")
    lines.append(
        "For each item above, decide which score is correct and write the anchor text that "
        "would have made that obvious to both raters."
    )
    return "\n".join(lines)


__all__ = [
    "GATE_ALPHA",
    "Action",
    "CriterionGate",
    "GateReport",
    "gate_criterion",
    "gate_criteria",
    "rewrite_brief",
]
