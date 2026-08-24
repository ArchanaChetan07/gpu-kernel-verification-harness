"""Resolve ``auto_probe`` strings against real oracle results.

A probe path names a measurement: ``oracle.O1.max_rel_err`` means "the
``max_rel_err`` key of O1's evidence". Resolution is deliberately unforgiving.

**Every failure to resolve raises :class:`~crucible.errors.ProbeError`.** There
is no default score, no fallback, no zero. A silently defaulted score is exactly
the invisible failure this project exists to prevent: it produces a number that
looks like a measurement, propagates into a weighted total, and is
indistinguishable in the report from a criterion that was actually probed.

The sharpest case is a **SKIP**. If O2 could not run, ``oracle.O2.speedup_ci``
does not resolve to 1.0, or to 0, or to "no speedup" - it does not resolve at
all, and the caller must decide, in the open, what to do about a criterion it
cannot score. That decision is :func:`score_rubric`'s ``on_missing`` policy, and
whatever it decides is recorded by name in the result.

Band semantics for ``probe_thresholds``
---------------------------------------
``[(b0, s0), (b1, s1), ..., (bn, sn)]`` with ascending bounds defines the bands
``(-inf, b0] -> s0``, ``(b0, b1] -> s1``, ... The **last band is terminal**:
anything above ``bn`` also scores ``sn``. That is not a default - it is the
rubric author's declared worst (or best) band, and it is what makes a value of
``inf`` (a candidate that produced infinities) score at the extreme rather than
raising. ``NaN`` still raises: a comparison that produced no number is not an
extreme value, it is an absent one.

Interval-valued probes
----------------------
``oracle.O2.speedup_ci`` resolves to an interval, not a scalar. Which end is
read is *derived from the rubric author's own thresholds*: if the scores ascend
with the bound then higher is better and the pessimistic end is the lower bound;
if they descend then lower is better and the pessimistic end is the upper bound.
The choice is recorded in :class:`ProbeValue.reduction`. A caller resolving a
path directly, with no thresholds to derive a direction from, gets a
``ProbeError`` rather than a guess.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..errors import ProbeError
from ..schema import OracleResult, RubricCriterion, RubricSpec

logger = logging.getLogger(__name__)

#: Verdicts that mean "this oracle produced no measurement". FAIL is absent on
#: purpose: a failing O1 measured a real error magnitude, and that magnitude is
#: precisely what the criterion should be scored on.
NO_MEASUREMENT_VERDICTS: frozenset[str] = frozenset({"SKIP", "ERROR"})

#: OracleResult fields (as opposed to evidence keys) a probe may name directly.
_RESULT_FIELDS: frozenset[str] = frozenset({"verdict", "duration_s"})

Reduction = Literal["scalar", "interval_low", "interval_high", "bool"]
MissingPolicy = Literal["raise", "unscored"]


@dataclass(frozen=True)
class ProbeValue:
    """A resolved measurement, with enough provenance to audit it later."""

    path: str
    oracle: str
    key: str
    value: float
    raw: Any = None
    reduction: Reduction = "scalar"
    derived: bool = False
    verdict: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "oracle": self.oracle,
            "key": self.key,
            "value": self.value,
            "reduction": self.reduction,
            "derived": self.derived,
            "verdict": self.verdict,
        }


# --------------------------------------------------------------------------- #
# path parsing and traversal
# --------------------------------------------------------------------------- #


def parse_probe_path(path: str) -> tuple[str, tuple[str, ...]]:
    """``"oracle.O1.max_rel_err"`` -> ``("O1", ("max_rel_err",))``."""
    parts = [p for p in str(path).split(".") if p]
    if len(parts) < 3 or parts[0] != "oracle":
        raise ProbeError(
            f"probe path {path!r} is malformed; expected 'oracle.<ORACLE_ID>.<key>[.<key>...]'",
            path=path,
        )
    return parts[1], tuple(parts[2:])


def _walk(container: Any, keys: Sequence[str], path: str, oracle_id: str) -> Any:
    cur = container
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        here = ".".join(walked)
        if isinstance(cur, Mapping):
            if key not in cur:
                raise ProbeError(
                    f"probe {path!r}: oracle {oracle_id} ran, but its evidence has no key "
                    f"{here!r}; available at this level: {sorted(str(k) for k in cur)[:20]}",
                    path=path,
                    oracle=oracle_id,
                    missing_key=here,
                )
            cur = cur[key]
            continue
        if isinstance(cur, Sequence) and not isinstance(cur, (str, bytes)):
            try:
                index = int(key)
            except ValueError as exc:
                raise ProbeError(
                    f"probe {path!r}: {here!r} indexes a sequence of length {len(cur)} but "
                    f"{key!r} is not an integer index",
                    path=path,
                    oracle=oracle_id,
                ) from exc
            if not -len(cur) <= index < len(cur):
                raise ProbeError(
                    f"probe {path!r}: index {index} is out of range for the {len(cur)}-element "
                    f"sequence at {'.'.join(walked[:-1]) or '<evidence>'}",
                    path=path,
                    oracle=oracle_id,
                )
            cur = cur[index]
            continue
        raise ProbeError(
            f"probe {path!r}: cannot look up {key!r} inside a "
            f"{type(cur).__name__} at {'.'.join(walked[:-1]) or '<evidence>'}",
            path=path,
            oracle=oracle_id,
        )
    return cur


def _derive_all_checks_passed(result: OracleResult, path: str) -> Any:
    """O3's ``all_checks_passed``, from the sub-check lists it actually reports.

    O3's contract is "any check firing fails the oracle", so this is a reading of
    the evidence, not an inference from the verdict. If the evidence does not
    carry the sub-check lists at all, we refuse rather than fall back to the
    verdict.
    """
    ev = result.evidence
    if not isinstance(ev, Mapping) or "fired" not in ev or "passed" not in ev:
        raise ProbeError(
            f"probe {path!r}: oracle O3 reported no sub-check lists ('fired'/'passed'), so "
            "whether every check passed cannot be read from its evidence",
            path=path,
            oracle="O3",
        )
    fired = list(ev.get("fired") or [])
    errored = list(ev.get("errored") or [])
    passed = list(ev.get("passed") or [])
    if not passed:
        raise ProbeError(
            f"probe {path!r}: no O3 sub-check ran, so 'every check passed' is vacuous, "
            "not true",
            path=path,
            oracle="O3",
        )
    return not fired and not errored


#: (oracle id, first key) -> reader used only when the evidence lacks the key.
_DERIVED: dict[tuple[str, str], Any] = {
    ("O3", "all_checks_passed"): _derive_all_checks_passed,
}


# --------------------------------------------------------------------------- #
# scalarisation
# --------------------------------------------------------------------------- #


def _interval_ends(raw: Any) -> tuple[float, float] | None:
    """``(lo, hi)`` if ``raw`` reads as an interval, else ``None``."""
    if isinstance(raw, Mapping):
        lo_key = next((k for k in ("lo", "low", "lower", "ci_lo", "l") if k in raw), None)
        hi_key = next((k for k in ("hi", "high", "upper", "ci_hi", "u") if k in raw), None)
        if lo_key is None or hi_key is None:
            return None
        lo, hi = raw[lo_key], raw[hi_key]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 2:
        lo, hi = raw[0], raw[1]
    else:
        return None
    if isinstance(lo, bool) or isinstance(hi, bool):
        return None
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return None
    return (float(lo), float(hi))


def _scalarize(
    raw: Any,
    path: str,
    oracle_id: str,
    prefer: Literal["lo", "hi"] | None,
) -> tuple[float, Reduction]:
    if isinstance(raw, bool):
        return (1.0 if raw else 0.0, "bool")
    if isinstance(raw, (int, float)):
        value = float(raw)
        if math.isnan(value):
            raise ProbeError(
                f"probe {path!r}: oracle {oracle_id} reported NaN; a comparison that produced "
                "no number is an absent measurement, not an extreme one",
                path=path,
                oracle=oracle_id,
            )
        return (value, "scalar")
    if raw is None:
        raise ProbeError(
            f"probe {path!r}: oracle {oracle_id} reported None for this key; the measurement "
            "is absent, and an absent measurement has no score",
            path=path,
            oracle=oracle_id,
        )
    ends = _interval_ends(raw)
    if ends is not None:
        lo, hi = ends
        if prefer is None:
            raise ProbeError(
                f"probe {path!r}: oracle {oracle_id} reported the interval [{lo!r}, {hi!r}], and "
                "nothing in this call says which end to read; append '.lo' or '.hi' to the path, "
                "or score it through a criterion whose thresholds fix the direction",
                path=path,
                oracle=oracle_id,
            )
        chosen = lo if prefer == "lo" else hi
        if math.isnan(chosen):
            raise ProbeError(
                f"probe {path!r}: the {prefer!r} end of oracle {oracle_id}'s interval is NaN",
                path=path,
                oracle=oracle_id,
            )
        return (chosen, "interval_low" if prefer == "lo" else "interval_high")
    raise ProbeError(
        f"probe {path!r}: oracle {oracle_id} reported a {type(raw).__name__}, which is not a "
        "number, a boolean or an interval; there is no honest way to score it",
        path=path,
        oracle=oracle_id,
    )


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #


def resolve_probe(
    path: str,
    results: Iterable[OracleResult],
    *,
    prefer: Literal["lo", "hi"] | None = None,
) -> ProbeValue:
    """Resolve one probe path, or raise :class:`ProbeError`. Never defaults."""
    oracle_id, keys = parse_probe_path(path)
    all_results = list(results)
    matches = [r for r in all_results if r.oracle == oracle_id]
    if not matches:
        raise ProbeError(
            f"probe {path!r}: no result for oracle {oracle_id!r}; the run reported "
            f"{sorted({r.oracle for r in all_results}) or 'no oracles at all'}. An oracle that "
            "did not run produced no measurement, and no measurement is not a score",
            path=path,
            oracle=oracle_id,
            available=sorted({r.oracle for r in all_results}),
        )
    if len(matches) > 1:
        raise ProbeError(
            f"probe {path!r}: {len(matches)} results claim to be oracle {oracle_id!r}; "
            "the run is ambiguous and picking one would be arbitrary",
            path=path,
            oracle=oracle_id,
        )
    result = matches[0]

    if result.verdict in NO_MEASUREMENT_VERDICTS:
        raise ProbeError(
            f"probe {path!r}: oracle {oracle_id} returned {result.verdict} "
            f"({result.reason.strip() or 'no reason recorded'}); a {result.verdict} oracle "
            "produced no measurement, and defaulting a score here is exactly the invisible "
            "failure this project exists to prevent",
            path=path,
            oracle=oracle_id,
            verdict=result.verdict,
        )

    derived = False
    if len(keys) == 1 and keys[0] in _RESULT_FIELDS:
        raw: Any = getattr(result, keys[0])
    else:
        reader = _DERIVED.get((oracle_id, keys[0]))
        evidence = result.evidence if isinstance(result.evidence, Mapping) else {}
        if reader is not None and keys[0] not in evidence:
            raw = reader(result, path)
            derived = True
            if len(keys) > 1:
                raw = _walk(raw, keys[1:], path, oracle_id)
        else:
            raw = _walk(evidence, keys, path, oracle_id)

    value, reduction = _scalarize(raw, path, oracle_id, prefer)
    return ProbeValue(
        path=path,
        oracle=oracle_id,
        key=".".join(keys),
        value=value,
        raw=raw,
        reduction=reduction,
        derived=derived,
        verdict=result.verdict,
    )


# --------------------------------------------------------------------------- #
# thresholds -> anchor score
# --------------------------------------------------------------------------- #


def threshold_direction(
    thresholds: Sequence[tuple[float, int]]
) -> Literal["higher_is_better", "lower_is_better", "mixed"]:
    """Read the author's intent out of the band table.

    Bounds ascend (the schema enforces it). If the scores ascend with them the
    author is saying "bigger is better"; if they descend, "smaller is better".
    """
    scores = [int(s) for _b, s in thresholds]
    if len(scores) < 2:
        return "mixed"
    if all(b >= a for a, b in zip(scores, scores[1:])):
        return "higher_is_better"
    if all(b <= a for a, b in zip(scores, scores[1:])):
        return "lower_is_better"
    return "mixed"


def preferred_end(
    thresholds: Sequence[tuple[float, int]]
) -> Literal["lo", "hi"] | None:
    """The pessimistic end of an interval, given what the thresholds reward."""
    direction = threshold_direction(thresholds)
    if direction == "higher_is_better":
        return "lo"
    if direction == "lower_is_better":
        return "hi"
    return None


def score_from_thresholds(
    value: float,
    thresholds: Sequence[tuple[float, int]],
    *,
    criterion_id: str = "",
) -> int:
    """Map a measurement to an anchor score. The last band is terminal."""
    if not thresholds:
        raise ProbeError(
            f"criterion {criterion_id!r} has no probe_thresholds; there is no rule mapping "
            f"the measured value {value!r} to a score",
            criterion=criterion_id,
        )
    if math.isnan(value):
        raise ProbeError(
            f"criterion {criterion_id!r}: cannot band a NaN measurement",
            criterion=criterion_id,
        )
    for bound, score in thresholds:
        if value <= float(bound):
            return int(score)
    return int(thresholds[-1][1])


# --------------------------------------------------------------------------- #
# criterion and rubric scoring
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CriterionScore:
    criterion_id: str
    weight: float
    score: int
    source: Literal["machine", "human"]
    anchor: str = ""
    probe: ProbeValue | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "weight": self.weight,
            "score": self.score,
            "source": self.source,
            "anchor": self.anchor,
            "probe": self.probe.as_dict() if self.probe else None,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class UnscoredCriterion:
    criterion_id: str
    weight: float
    source: Literal["machine", "human"]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "weight": self.weight,
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RubricScore:
    """A weighted score plus an explicit account of what could not be scored."""

    scored: list[CriterionScore] = field(default_factory=list)
    unscored: list[UnscoredCriterion] = field(default_factory=list)
    total_weight: float = 0.0
    scored_weight: float = 0.0

    @property
    def score(self) -> float:
        """Weighted mean on the 0-5 anchor scale over the criteria that scored."""
        if self.scored_weight <= 0:
            return math.nan
        return sum(c.weight * c.score for c in self.scored) / self.scored_weight

    @property
    def normalized(self) -> float:
        """The same number on 0-1. NaN when nothing could be scored."""
        s = self.score
        return s / 5.0 if math.isfinite(s) else math.nan

    @property
    def coverage(self) -> float:
        """Share of rubric weight that was actually scored."""
        if self.total_weight <= 0:
            return 0.0
        return self.scored_weight / self.total_weight

    def as_dict(self) -> dict[str, Any]:
        s = self.score
        return {
            "score": None if math.isnan(s) else s,
            "normalized": None if math.isnan(s) else self.normalized,
            "coverage": self.coverage,
            "total_weight": self.total_weight,
            "scored_weight": self.scored_weight,
            "scored": [c.as_dict() for c in self.scored],
            "unscored": [u.as_dict() for u in self.unscored],
        }


def score_criterion(
    criterion: RubricCriterion,
    results: Iterable[OracleResult],
) -> CriterionScore:
    """Score one machine-probed criterion. Raises if the probe does not resolve."""
    if not criterion.machine_probed or not criterion.auto_probe:
        raise ProbeError(
            f"criterion {criterion.id!r} is not machine-probed; there is nothing to resolve",
            criterion=criterion.id,
        )
    probe = resolve_probe(
        criterion.auto_probe,
        results,
        prefer=preferred_end(criterion.probe_thresholds),
    )
    score = score_from_thresholds(
        probe.value, criterion.probe_thresholds, criterion_id=criterion.id
    )
    return CriterionScore(
        criterion_id=criterion.id,
        weight=float(criterion.weight),
        score=score,
        source="machine",
        anchor=criterion.anchors.get(score, ""),
        probe=probe,
        detail=(
            f"{criterion.auto_probe} = {probe.value!r}"
            + (f" ({probe.reduction})" if probe.reduction != "scalar" else "")
        ),
    )


def score_rubric(
    rubric: RubricSpec,
    results: Iterable[OracleResult],
    human_scores: Mapping[str, int] | None = None,
    *,
    on_missing: MissingPolicy = "raise",
) -> RubricScore:
    """Score a whole rubric.

    ``on_missing`` governs criteria that cannot be scored - a probe that will not
    resolve, or a human criterion with no rating supplied:

    * ``"raise"`` (default) - propagate the :class:`ProbeError`. Use this when a
      task is supposed to be fully graded; a rubric that cannot be scored is a
      finding, not a lower score.
    * ``"unscored"`` - drop the criterion from the weighted mean and record it in
      ``RubricScore.unscored`` with the reason. The weighted mean is then taken
      over the remaining weight and ``coverage`` reports how much of the rubric
      that was. Nothing is defaulted; the gap is visible in the result.
    """
    if on_missing not in ("raise", "unscored"):
        raise ValueError(f"on_missing must be 'raise' or 'unscored', got {on_missing!r}")
    all_results = list(results)
    humans = dict(human_scores or {})

    scored: list[CriterionScore] = []
    unscored: list[UnscoredCriterion] = []

    for criterion in rubric.criteria:
        if criterion.machine_probed:
            try:
                scored.append(score_criterion(criterion, all_results))
            except ProbeError as exc:
                if on_missing == "raise":
                    raise
                unscored.append(
                    UnscoredCriterion(
                        criterion_id=criterion.id,
                        weight=float(criterion.weight),
                        source="machine",
                        reason=str(exc),
                    )
                )
            continue

        if criterion.id not in humans:
            reason = (
                f"criterion {criterion.id!r} is a human judgement and no rating was supplied"
            )
            if on_missing == "raise":
                raise ProbeError(reason, criterion=criterion.id)
            unscored.append(
                UnscoredCriterion(
                    criterion_id=criterion.id,
                    weight=float(criterion.weight),
                    source="human",
                    reason=reason,
                )
            )
            continue

        raw = humans[criterion.id]
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ProbeError(
                f"criterion {criterion.id!r}: human score {raw!r} is not an integer anchor",
                criterion=criterion.id,
            ) from exc
        if value not in criterion.anchors and not 0 <= value <= 5:
            raise ProbeError(
                f"criterion {criterion.id!r}: human score {value} is outside the 0-5 anchor "
                f"scale (anchors defined for {sorted(criterion.anchors)})",
                criterion=criterion.id,
            )
        scored.append(
            CriterionScore(
                criterion_id=criterion.id,
                weight=float(criterion.weight),
                score=value,
                source="human",
                anchor=criterion.anchors.get(value, ""),
                detail="human rating",
            )
        )

    return RubricScore(
        scored=scored,
        unscored=unscored,
        total_weight=float(rubric.total_weight()),
        scored_weight=float(sum(c.weight for c in scored)),
    )


__all__ = [
    "ProbeValue",
    "CriterionScore",
    "UnscoredCriterion",
    "RubricScore",
    "NO_MEASUREMENT_VERDICTS",
    "parse_probe_path",
    "resolve_probe",
    "threshold_direction",
    "preferred_end",
    "score_from_thresholds",
    "score_criterion",
    "score_rubric",
]
