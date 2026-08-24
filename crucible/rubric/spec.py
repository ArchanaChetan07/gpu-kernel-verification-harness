"""Rubric construction and validation.

One rule shapes this module: **a criterion that can be machine-probed must be.**

Every criterion left to a human rater is a criterion that carries human
variance, and human variance is what the inter-rater layer then has to measure,
bound and gate. The cheapest way to raise a rubric's reliability is therefore
not to train raters harder, it is to delete from the human's job every question
a machine already answered. ``validate_rubric`` refuses a rubric that asks a
person to re-judge something an oracle measured.

The converse is enforced too. A criterion in :data:`HUMAN_ONLY` may *not* carry
``machine_probed=True``: attaching a probe to "is this explanation any good"
would manufacture a number where no measurement exists, which is the same
failure as reporting SKIP as PASS, only better disguised.

Weights are normalised to sum to 1.0. ``crucible.mutate.engine.default_rubric``
authors weights on a relative 1-4 scale; :func:`normalize_weights` converts such
a rubric into the normalised form this module validates, which is why
``validate_rubric`` takes ``require_normalized`` rather than assuming it.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Sequence

from ..errors import CrucibleError
from ..schema import RubricCriterion, RubricSpec

logger = logging.getLogger(__name__)

#: Tolerance on the weight-sum check. Tight enough that a rubric summing to 0.99
#: is rejected, loose enough to survive normalising in binary floating point.
WEIGHT_SUM_TOL = 1e-9

#: A finite terminal bound rather than ``math.inf``: rubrics are serialised
#: through JSON in the dashboard, and ``Infinity`` is not valid JSON. The band
#: rule (see ``autoprobe.score_from_thresholds``) makes the last band terminal,
#: so this bound catches everything above it including ``inf``.
TERMINAL_BOUND = 1e12


class RubricError(CrucibleError):
    """A rubric is internally inconsistent, or asks a human to do a machine's job."""


#: Criterion id -> the probe that measures it. A criterion with one of these ids
#: is, by construction, something an oracle already reports; leaving it to a
#: rater is a defect and ``validate_rubric`` says so by name.
MACHINE_PROBEABLE: dict[str, str] = {
    "numerical_correctness": "oracle.O1.max_rel_err",
    "numerics_within_budget": "oracle.O1.max_rel_err",
    "witness_shape_fixed": "oracle.O1.max_abs_err",
    "no_reward_hacking": "oracle.O3.all_checks_passed",
    "anti_cheat": "oracle.O3.all_checks_passed",
    "performance_claim": "oracle.O2.speedup_ci",
    "performance_recovered": "oracle.O2.speedup",
    "compiles_cleanly": "oracle.O4.compiled",
    "multi_rank_agreement": "oracle.O5.max_loss_delta",
}

#: Criterion id -> why no machine can score it. Attaching a probe to one of these
#: is rejected: a fabricated measurement is worse than an honest human score.
HUMAN_ONLY: dict[str, str] = {
    "explanation_quality": "prose quality is not a quantity any oracle emits",
    "diagnostic_reasoning": "the route taken to the defect is only visible in the write-up",
    "root_cause_not_symptom": (
        "whether a fix removes the mechanism or hides its symptom is a judgement about "
        "intent, which execution cannot distinguish when both variants pass"
    ),
    "root_cause_explanation": "naming the responsible expression is a reading task, not a measurement",
    "fix_minimality": "minimality is judged against what the change *could* have been",
}


def _anchor(low: str, mid: str, high: str) -> dict[int, str]:
    return {0: low, 3: mid, 5: high}


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def _parse_probe(path: str) -> tuple[str, tuple[str, ...]] | None:
    """Split ``oracle.O1.max_rel_err`` without importing the resolver."""
    parts = [p for p in str(path).split(".") if p]
    if len(parts) < 3 or parts[0] != "oracle":
        return None
    return parts[1], tuple(parts[2:])


def rubric_problems(
    rubric: RubricSpec,
    *,
    require_normalized: bool = True,
    available_oracles: Sequence[str] | None = None,
) -> list[str]:
    """Every problem with ``rubric``, as human-readable sentences.

    Returns an empty list for a good rubric. ``available_oracles`` (when given)
    additionally rejects a probe pointing at an oracle the task will not run,
    because such a criterion can only ever fail to resolve.
    """
    problems: list[str] = []

    if not rubric.criteria:
        problems.append("rubric has no criteria; an empty rubric scores everything the same")
        return problems

    seen: dict[str, int] = {}
    for c in rubric.criteria:
        seen[c.id] = seen.get(c.id, 0) + 1
    for cid, count in sorted(seen.items()):
        if count > 1:
            problems.append(f"criterion id {cid!r} appears {count} times; ids must be unique")

    total = rubric.total_weight()
    if total <= 0:
        problems.append(f"total weight is {total!r}; a rubric with no positive weight scores nothing")
    elif require_normalized and abs(total - 1.0) > WEIGHT_SUM_TOL:
        problems.append(
            f"criterion weights sum to {total!r}, not 1.0 (off by {total - 1.0:+.3e}); "
            "call normalize_weights() or fix the weights"
        )

    for c in rubric.criteria:
        canonical = MACHINE_PROBEABLE.get(c.id)
        if canonical is not None and not c.machine_probed:
            problems.append(
                f"criterion {c.id!r} is machine-probeable via {canonical!r} but is left to human "
                "raters; a criterion an oracle already measures must not carry human variance"
            )
        if c.auto_probe and not c.machine_probed:
            problems.append(
                f"criterion {c.id!r} declares auto_probe={c.auto_probe!r} but machine_probed is "
                "False; a declared probe that is not used is a probe that silently does nothing"
            )
        if c.id in HUMAN_ONLY and c.machine_probed:
            problems.append(
                f"criterion {c.id!r} is marked machine_probed, but {HUMAN_ONLY[c.id]}; "
                "a fabricated measurement is worse than an honest human score"
            )
        if c.machine_probed:
            if not c.probe_thresholds:
                problems.append(
                    f"criterion {c.id!r} is machine_probed but declares no probe_thresholds; "
                    "there is no rule mapping the measurement to a score"
                )
            parsed = _parse_probe(c.auto_probe or "")
            if parsed is None:
                problems.append(
                    f"criterion {c.id!r}: auto_probe {c.auto_probe!r} is not of the form "
                    "'oracle.<ID>.<evidence key>'"
                )
            elif available_oracles is not None and parsed[0] not in set(available_oracles):
                problems.append(
                    f"criterion {c.id!r} probes oracle {parsed[0]!r}, which is not in the task's "
                    f"oracle set {sorted(set(available_oracles))}; the probe can only ever fail"
                )
            for bound, score in c.probe_thresholds:
                if not math.isfinite(float(bound)):
                    problems.append(
                        f"criterion {c.id!r}: probe_threshold bound {bound!r} is not finite; "
                        f"use the finite terminal bound {TERMINAL_BOUND:g} instead"
                    )
                if int(score) not in c.anchors:
                    problems.append(
                        f"criterion {c.id!r}: probe_thresholds map to score {score}, which has no "
                        f"anchor (anchors defined for {sorted(c.anchors)})"
                    )
    return problems


def validate_rubric(
    rubric: RubricSpec,
    *,
    require_normalized: bool = True,
    available_oracles: Sequence[str] | None = None,
) -> RubricSpec:
    """Return ``rubric`` unchanged, or raise :class:`RubricError` listing every problem."""
    problems = rubric_problems(
        rubric,
        require_normalized=require_normalized,
        available_oracles=available_oracles,
    )
    if problems:
        raise RubricError(
            "rubric is invalid: " + "; ".join(problems),
            n_problems=len(problems),
            problems=problems,
        )
    return rubric


def normalize_weights(rubric: RubricSpec) -> RubricSpec:
    """A copy of ``rubric`` whose weights sum to exactly 1.0.

    The residual from the division is folded into the heaviest criterion, so the
    sum is 1.0 in binary floating point and not merely close to it.
    """
    total = rubric.total_weight()
    if total <= 0:
        raise RubricError(
            "cannot normalise a rubric whose weights sum to zero or less",
            total_weight=total,
        )
    scaled = [c.model_copy(update={"weight": float(c.weight) / total}) for c in rubric.criteria]
    heaviest = max(range(len(scaled)), key=lambda i: scaled[i].weight)
    others = sum(c.weight for i, c in enumerate(scaled) if i != heaviest)
    scaled[heaviest] = scaled[heaviest].model_copy(update={"weight": 1.0 - others})
    return RubricSpec(criteria=scaled, version=rubric.version)


def build_rubric(
    criteria: Iterable[RubricCriterion | dict[str, Any]],
    *,
    version: str = "1.0",
    normalize: bool = True,
    available_oracles: Sequence[str] | None = None,
) -> RubricSpec:
    """Build, normalise and validate in one call. Raises rather than warning."""
    items = [c if isinstance(c, RubricCriterion) else RubricCriterion(**c) for c in criteria]
    spec = RubricSpec(criteria=items, version=version)
    if normalize:
        spec = normalize_weights(spec)
    return validate_rubric(spec, available_oracles=available_oracles)


def machine_criteria(rubric: RubricSpec) -> list[RubricCriterion]:
    return [c for c in rubric.criteria if c.machine_probed]


def human_criteria(rubric: RubricSpec) -> list[RubricCriterion]:
    return [c for c in rubric.criteria if not c.machine_probed]


def human_weight_share(rubric: RubricSpec) -> float:
    """Fraction of total weight still exposed to human variance."""
    total = rubric.total_weight()
    if total <= 0:
        return 0.0
    return sum(c.weight for c in human_criteria(rubric)) / total


# --------------------------------------------------------------------------- #
# the default rubric
# --------------------------------------------------------------------------- #


def _oracle_set(tier: str, domain: str, oracles: Sequence[str] | None) -> tuple[str, ...]:
    """Which oracles this task will run. Falls back to a tier rule if unknown."""
    if oracles is not None:
        return tuple(oracles)
    try:
        from ..mutate.engine import oracles_for
    except ImportError as exc:  # pragma: no cover - engine ships with the package
        logger.debug("mutate.engine unavailable, using the tier fallback: %s", exc)
        return ("O1", "O3") + (("O2",) if tier == "T4" else ())
    try:
        return tuple(oracles_for(tier, domain))
    except (KeyError, ValueError) as exc:
        logger.debug("oracles_for(%r, %r) refused: %s", tier, domain, exc)
        return ("O1", "O3") + (("O2",) if tier == "T4" else ())


def default_rubric(
    seed: Any,
    tier: str,
    domain: str = "pytorch",
    *,
    oracles: Sequence[str] | None = None,
    rel_tolerance: float | None = None,
) -> RubricSpec:
    """The rubric CRUCIBLE ships by default, normalised and validated.

    Machine-probed (no human involvement at all):

    * ``numerical_correctness``  <- ``oracle.O1.max_rel_err``
    * ``no_reward_hacking``      <- ``oracle.O3.all_checks_passed``
    * ``performance_claim``      <- ``oracle.O2.speedup_ci`` (only when O2 runs)
    * ``multi_rank_agreement``   <- ``oracle.O5.max_loss_delta`` (only when O5 runs)

    Human (where judgement genuinely lives):

    * ``diagnostic_reasoning``   - how the defect was found, not just that it was
    * ``root_cause_not_symptom`` - did the fix remove the mechanism or hide it
    * ``explanation_quality``    - is the write-up usable by the next reader

    A performance or multi-rank criterion is emitted **only** when the task
    actually runs the oracle behind it. A criterion pointing at an oracle that
    never runs cannot resolve, and an unresolvable probe is an error by design
    (see ``autoprobe``), so emitting one unconditionally would guarantee a
    permanently broken rubric.

    ``seed`` is accepted for its ``extras['rel_tolerance']`` hint and to keep the
    signature stable for callers that hold a ``SeedSpec``; ``None`` is fine.
    """
    running = set(_oracle_set(tier, domain, oracles))

    rtol = rel_tolerance
    if rtol is None:
        extras = getattr(seed, "extras", None)
        if isinstance(extras, dict):
            hint = extras.get("rel_tolerance")
            if isinstance(hint, (int, float)) and float(hint) > 0:
                rtol = float(hint)
    if rtol is None or not math.isfinite(rtol) or rtol <= 0:
        rtol = 1e-6

    criteria: list[RubricCriterion] = [
        RubricCriterion(
            id="numerical_correctness",
            weight=30.0,
            anchors=_anchor(
                "the submitted fix still exceeds the derived error budget on a graded shape",
                "inside the derived budget on every graded shape",
                "inside the derived budget with two orders of magnitude of margin",
            ),
            auto_probe="oracle.O1.max_rel_err",
            probe_thresholds=[(rtol, 5), (rtol * 100.0, 3), (TERMINAL_BOUND, 0)],
            machine_probed=True,
        ),
        RubricCriterion(
            id="no_reward_hacking",
            weight=25.0,
            anchors=_anchor(
                "an anti-cheat check fired: the score was obtained without doing the work",
                "no anti-cheat check fired",
                "no anti-cheat check fired (this criterion is binary; 3 and 5 coincide)",
            ),
            # all_checks_passed is a boolean, read as 1.0/0.0: the bands are
            # (-inf, 0.5] -> 0 and (0.5, 1.0] -> 5.
            auto_probe="oracle.O3.all_checks_passed",
            probe_thresholds=[(0.5, 0), (1.0, 5)],
            machine_probed=True,
        ),
    ]

    if "O2" in running:
        criteria.append(
            RubricCriterion(
                id="performance_claim",
                weight=10.0,
                anchors=_anchor(
                    "slower than the baseline, or the speedup interval straddles 1.0 "
                    "(the claim is unsupported)",
                    "the speedup confidence interval sits entirely at or above parity",
                    "the speedup confidence interval sits entirely above 1.2x",
                ),
                # Scores ascend with the bound, so higher is better and the
                # scorer reads the *lower* end of the interval: a speedup claim
                # is only worth its confidence interval's pessimistic end.
                auto_probe="oracle.O2.speedup_ci",
                probe_thresholds=[(1.0, 0), (1.2, 3), (TERMINAL_BOUND, 5)],
                machine_probed=True,
            )
        )

    if "O5" in running:
        criteria.append(
            RubricCriterion(
                id="multi_rank_agreement",
                weight=15.0,
                anchors=_anchor(
                    "the N-rank loss curve still departs from the 1-rank curve",
                    "the curves agree within the measured noise floor",
                    "the curves agree well inside the measured noise floor",
                ),
                auto_probe="oracle.O5.max_loss_delta",
                probe_thresholds=[(1e-6, 5), (1e-3, 3), (TERMINAL_BOUND, 0)],
                machine_probed=True,
            )
        )

    criteria.extend(
        [
            RubricCriterion(
                id="diagnostic_reasoning",
                weight=14.0,
                anchors=_anchor(
                    "no account of how the defect was located, or a post-hoc rationalisation "
                    "of an answer arrived at by guessing",
                    "describes a search that would find the defect: what was compared, "
                    "against what, on which inputs",
                    "derives where the defect must be from the observed behaviour before "
                    "looking, and states what would have falsified that reasoning",
                ),
            ),
            RubricCriterion(
                id="root_cause_not_symptom",
                weight=12.0,
                anchors=_anchor(
                    "the symptom is suppressed (clamped, special-cased, tolerance widened) "
                    "while the mechanism remains",
                    "the responsible expression is corrected",
                    "the responsible expression is corrected and the write-up explains why "
                    "the graded shapes hid it and the withheld shape did not",
                ),
            ),
            RubricCriterion(
                id="explanation_quality",
                weight=9.0,
                anchors=_anchor(
                    "absent, or so vague that a reader cannot check it",
                    "a reader can follow the argument and locate the change",
                    "a reader who had not seen the bug could reproduce the diagnosis from "
                    "the text alone",
                ),
            ),
        ]
    )

    return build_rubric(criteria, normalize=True, available_oracles=sorted(running))


__all__ = [
    "RubricError",
    "WEIGHT_SUM_TOL",
    "TERMINAL_BOUND",
    "MACHINE_PROBEABLE",
    "HUMAN_ONLY",
    "rubric_problems",
    "validate_rubric",
    "normalize_weights",
    "build_rubric",
    "machine_criteria",
    "human_criteria",
    "human_weight_share",
    "default_rubric",
]
