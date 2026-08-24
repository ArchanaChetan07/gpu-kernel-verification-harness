"""Tests for the rubric engine and the inter-rater reliability layer.

The alpha implementation is checked against Krippendorff's canonical worked
example, whose nominal / ordinal / interval values are published, and against a
hand derivation of ``D_o`` and ``D_e`` for that same matrix written out in the
test so the assertion carries its own proof rather than pointing at the code
that produced it. The behavioural properties every correct alpha must have
(perfect agreement -> 1, chance -> 0, systematic disagreement -> negative) are
asserted separately, because an implementation can match one worked example by
coincidence and still be wrong everywhere else.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from crucible.errors import ProbeError
from crucible.rubric.autoprobe import (
    RubricScore,
    parse_probe_path,
    preferred_end,
    resolve_probe,
    score_criterion,
    score_from_thresholds,
    score_rubric,
    threshold_direction,
)
from crucible.rubric.gate import GATE_ALPHA, gate_criteria, gate_criterion, rewrite_brief
from crucible.rubric.irr import (
    coincidence_matrix,
    disagreement_stream,
    krippendorff_alpha,
    running_alpha,
    to_units,
)
from crucible.rubric.sequences import (
    AlphaMonitor,
    BettingCS,
    EmpiricalBernsteinCS,
    HoeffdingCS,
    make_cs,
    monitor_alpha,
)
from crucible.rubric.spec import (
    RubricError,
    build_rubric,
    default_rubric,
    human_criteria,
    machine_criteria,
    normalize_weights,
    rubric_problems,
    validate_rubric,
)
from crucible.schema import OracleResult, RubricCriterion, RubricSpec

NAN = float("nan")

# Krippendorff's canonical reliability data: 3 observers, 15 units, "*" = not
# rated. Published alphas: nominal .691, ordinal .807, interval .811.
CANONICAL: list[list[float]] = [
    [NAN, NAN, NAN, NAN, NAN, 3, 4, 1, 2, 1, 1, 3, 3, NAN, 3],
    [1, NAN, 2, 1, 3, 3, 4, 3, NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [NAN, NAN, 2, 1, 3, 4, 4, NAN, 2, 1, 1, 3, 3, NAN, 4],
]


def _anchors(low: str = "low", mid: str = "mid", high: str = "high") -> dict[int, str]:
    return {0: low, 3: mid, 5: high}


# --------------------------------------------------------------------------- #
# Krippendorff's alpha
# --------------------------------------------------------------------------- #


def test_canonical_example_reproduces_the_published_alphas() -> None:
    """The published values for Krippendorff's own worked example."""
    assert krippendorff_alpha(CANONICAL, "nominal").alpha == pytest.approx(0.691, abs=5e-4)
    assert krippendorff_alpha(CANONICAL, "ordinal").alpha == pytest.approx(0.807, abs=5e-4)
    assert krippendorff_alpha(CANONICAL, "interval").alpha == pytest.approx(0.811, abs=5e-4)


def test_canonical_example_matches_a_hand_derivation_of_D_o_and_D_e() -> None:
    """The arithmetic, written out, so the test does not just re-run the code.

    Pairable values n = 26 (10 units rated twice, 2 rated three times; 3 units
    have fewer than two ratings and drop out). Marginals over the pairable
    values are n_1 = 7, n_2 = 4, n_3 = 10, n_4 = 5.

    Nominal: only three units disagree at all -- u6 (3,3,4), u8 (1,3) and
    u15 (3,4) -- contributing 4/2 + 2 + 2 = 6 disagreeing ordered pairs, so
    D_o = 6/26. D_e = (n^2 - sum n_c^2) / (n(n-1)) = (676 - 190) / 650.
    """
    res = krippendorff_alpha(CANONICAL, "nominal")
    assert res.n_pairable == pytest.approx(26.0)
    assert res.n_units == 12 and res.n_units_excluded == 3
    assert res.values == [1.0, 2.0, 3.0, 4.0]

    _values, o = coincidence_matrix(to_units(CANONICAL))
    assert list(o.sum(axis=1)) == pytest.approx([7.0, 4.0, 10.0, 5.0])

    assert res.d_observed == pytest.approx(6.0 / 26.0)
    assert res.d_expected == pytest.approx((676.0 - 190.0) / 650.0)
    assert res.alpha == pytest.approx(1.0 - (6.0 / 26.0) / ((676.0 - 190.0) / 650.0))

    # Interval: sum of squared differences over ordered pairs is
    # u6: (1+1+1+1)/2 = 2, u8: 4+4 = 8, u15: 1+1 = 2  ->  12.
    # D_e numerator = 2 * sum_{c<k} n_c n_k (c-k)^2
    #               = 2 * (28 + 280 + 315 + 40 + 80 + 50) = 1586.
    interval = krippendorff_alpha(CANONICAL, "interval")
    assert interval.d_observed == pytest.approx(12.0 / 26.0)
    assert interval.d_expected == pytest.approx(1586.0 / 650.0)

    # Ordinal delta2(c,k) = (sum_{g=c..k} n_g - (n_c+n_k)/2)^2, so
    # delta2(3,4) = (15 - 7.5)^2 = 56.25 and delta2(1,3) = (21 - 8.5)^2 = 156.25.
    # D_o numerator = 4*56.25/2 + 2*156.25 + 2*56.25 = 537.5.
    ordinal = krippendorff_alpha(CANONICAL, "ordinal")
    assert ordinal.d_observed == pytest.approx(537.5 / 26.0)
    assert ordinal.d_expected == pytest.approx(69524.0 / 650.0)


def test_perfect_agreement_is_exactly_one() -> None:
    matrix = [[1, 2, 3, 4, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3, 4]]
    for metric in ("nominal", "ordinal", "interval", "ratio"):
        res = krippendorff_alpha(matrix, metric)
        assert res.alpha == pytest.approx(1.0), metric
        assert res.d_observed == pytest.approx(0.0), metric


def test_chance_agreement_is_near_zero() -> None:
    rng = np.random.default_rng(20260818)
    matrix = rng.integers(0, 5, size=(4, 600)).tolist()
    for metric in ("nominal", "ordinal", "interval"):
        res = krippendorff_alpha(matrix, metric)
        assert abs(res.alpha) < 0.05, f"{metric}: alpha={res.alpha}"


def test_systematic_disagreement_is_negative() -> None:
    """Raters that mirror each other agree less than chance would predict."""
    matrix = [[0, 1] * 12, [1, 0] * 12]
    res = krippendorff_alpha(matrix, "nominal")
    assert res.alpha < 0
    # n = 48, marginals 24/24, D_o = 1, D_e = (48^2 - 2*24^2)/(48*47).
    assert res.alpha == pytest.approx(1.0 - 1.0 / ((48.0**2 - 2 * 24.0**2) / (48.0 * 47.0)))


def test_ordinal_and_nominal_diverge_in_the_expected_direction() -> None:
    """Nominal cannot see magnitude; ordinal can, and that is the whole point.

    Two data sets disagree on exactly the same units, so their nominal alphas
    are identical. In one the disagreements are one category wide, in the other
    they span the scale. Ordinal must rank them far apart.
    """
    base = [0, 1, 2, 3, 4, 5] * 6
    near = list(base)
    far = list(base)
    for i in range(0, len(base), 3):
        near[i] = min(5, base[i] + 1)
        far[i] = 5 - base[i]

    near_nominal = krippendorff_alpha([base, near], "nominal").alpha
    far_nominal = krippendorff_alpha([base, far], "nominal").alpha
    near_ordinal = krippendorff_alpha([base, near], "ordinal").alpha
    far_ordinal = krippendorff_alpha([base, far], "ordinal").alpha

    assert near_nominal == pytest.approx(far_nominal), "nominal is blind to magnitude"
    assert near_ordinal > near_nominal, "near-miss disagreement is forgiven by ordinal"
    assert far_ordinal < far_nominal, "scale-spanning disagreement is punished by ordinal"
    assert near_ordinal - far_ordinal > 0.5


def test_units_with_one_rating_are_excluded_not_counted_as_agreement() -> None:
    """A single rating cannot agree with anything, so it must not raise alpha."""
    two_raters = [[1, 2, 3, 4], [1, 2, 3, 4]]
    with_singleton = [[1, 2, 3, 4, 2], [1, 2, 3, 4, NAN]]
    base = krippendorff_alpha(two_raters, "nominal")
    plus = krippendorff_alpha(with_singleton, "nominal")
    assert plus.alpha == pytest.approx(base.alpha)
    assert plus.n_units == 4 and plus.n_units_excluded == 1
    assert "fewer than two ratings" in plus.reason


def test_missing_cells_are_honoured_as_missing_not_as_a_value() -> None:
    units = to_units([[1, None, 3], [1, 2, ""]])
    assert units.shape == (3, 2)
    assert math.isnan(units[1, 0]) and math.isnan(units[2, 1])
    # Only unit 0 is pairable, so alpha is undefined rather than 1.0.
    res = krippendorff_alpha([[1, None, 3], [1, 2, ""]], "nominal")
    assert math.isnan(res.alpha)
    assert res.reason


def test_no_variation_gives_an_undefined_alpha_with_a_reason() -> None:
    res = krippendorff_alpha([[2, 2, 2], [2, 2, 2]], "nominal")
    assert math.isnan(res.alpha)
    assert "zero" in res.reason and "0/0" in res.reason
    assert res.as_dict()["alpha"] is None


def test_bootstrap_ci_brackets_the_estimate_and_narrows_with_more_units() -> None:
    rng = np.random.default_rng(7)
    truth = rng.integers(0, 5, size=400)
    noisy = np.clip(truth + rng.integers(-1, 2, size=400), 0, 4)
    small = krippendorff_alpha(
        [truth[:25].tolist(), noisy[:25].tolist()], "ordinal", bootstrap=300, rng_seed=1
    )
    large = krippendorff_alpha(
        [truth.tolist(), noisy.tolist()], "ordinal", bootstrap=300, rng_seed=1
    )
    for res in (small, large):
        assert res.n_bootstrap == 300
        assert res.ci_lo <= res.alpha <= res.ci_hi
    assert (large.ci_hi - large.ci_lo) < (small.ci_hi - small.ci_lo)


def test_ratio_metric_refuses_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        krippendorff_alpha([[-1, 2, 3], [-1, 2, 4]], "ratio")


def test_unknown_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="metric must be one of"):
        krippendorff_alpha(CANONICAL, "cosine")


def test_running_alpha_reports_one_value_per_unit() -> None:
    trend = running_alpha(CANONICAL, "nominal")
    assert len(trend) == 15
    assert trend[-1] == pytest.approx(krippendorff_alpha(CANONICAL, "nominal").alpha)
    assert math.isnan(trend[0])  # unit 1 has a single rating


# --------------------------------------------------------------------------- #
# confidence sequences
# --------------------------------------------------------------------------- #


def _bernoulli_run(cls: Any, p: float, n: int, seed: int, **kwargs: Any) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    cs = cls(0.05, **kwargs)
    return [cs.update(float(rng.random() < p)) for _ in range(n)]


@pytest.mark.parametrize(
    "cls,n,trials,kwargs",
    [
        (HoeffdingCS, 200, 120, {}),
        (EmpiricalBernsteinCS, 200, 120, {}),
        (BettingCS, 120, 40, {"grid": 201}),
    ],
)
def test_confidence_sequences_hold_nominal_coverage(
    cls: Any, n: int, trials: int, kwargs: dict[str, Any]
) -> None:
    """Coverage is simultaneous over t: the mean must be inside at *every* step."""
    p = 0.3
    covered = 0
    for t in range(trials):
        intervals = _bernoulli_run(cls, p, n, seed=1000 + t, **kwargs)
        if all(lo - 1e-12 <= p <= hi + 1e-12 for lo, hi in intervals):
            covered += 1
    assert covered / trials >= 0.95, f"{cls.__name__} covered {covered}/{trials}"


@pytest.mark.parametrize("cls", [HoeffdingCS, EmpiricalBernsteinCS])
def test_confidence_sequences_narrow_as_evidence_accumulates(cls: Any) -> None:
    intervals = _bernoulli_run(cls, 0.3, 600, seed=11)
    widths = [hi - lo for lo, hi in intervals]
    assert widths[599] < widths[199] < widths[49] < widths[9]


def test_empirical_bernstein_is_tighter_than_hoeffding_on_a_low_variance_stream() -> None:
    """EB adapts to variance; on a nearly constant stream that is a large win."""
    eb = _bernoulli_run(EmpiricalBernsteinCS, 0.02, 500, seed=5)
    hoeffding = _bernoulli_run(HoeffdingCS, 0.02, 500, seed=5)
    eb_width = eb[-1][1] - eb[-1][0]
    hoeffding_width = hoeffding[-1][1] - hoeffding[-1][0]
    assert eb_width < hoeffding_width
    assert eb_width < 0.5 * hoeffding_width


def test_betting_cs_degrades_to_vacuous_when_every_candidate_is_rejected() -> None:
    """A hard drift breaks the fixed-mean assumption; the CS must say so."""
    cs = BettingCS(0.05, grid=201)
    for _ in range(300):
        cs.update(0.0)
    lo, hi = 0.0, 1.0
    for _ in range(300):
        lo, hi = cs.update(1.0)
    assert cs.failed is True
    assert (lo, hi) == (0.0, 1.0), "a failed capital process must not emit a tight interval"


def test_make_cs_rejects_an_unknown_estimator() -> None:
    with pytest.raises(ValueError, match="unknown confidence sequence"):
        make_cs("bayes")


def test_alpha_monitor_bounds_the_batch_alpha_and_states_its_approximation() -> None:
    rng = np.random.default_rng(31)
    truth = rng.integers(0, 5, size=300)
    noisy = np.clip(truth + rng.integers(-1, 2, size=300), 0, 4)
    matrix = [truth.tolist(), noisy.tolist()]

    batch = krippendorff_alpha(matrix, "ordinal").alpha
    trace = monitor_alpha(matrix, "ordinal", cs_kind="eb", criterion_id="c")
    final = trace.final
    assert final is not None
    assert trace.equal_ratings is True
    assert final.alpha_lo <= batch <= final.alpha_hi
    assert final.width < trace.bounds[9].width, "the bound must narrow with n"
    assert "plug-in" in final.approximation and "conditional" in final.approximation
    assert final.as_dict()["approximation"]


def test_alpha_monitor_refuses_a_degenerate_expected_disagreement() -> None:
    with pytest.raises(ValueError, match="positive"):
        AlphaMonitor(make_cs("eb"), d_expected=0.0, delta_max=1.0)
    with pytest.raises(ValueError, match="delta_max"):
        AlphaMonitor(make_cs("eb"), d_expected=0.5, delta_max=0.0)


def test_alpha_monitor_rejects_an_out_of_range_observation() -> None:
    monitor = AlphaMonitor(make_cs("hoeffding"), d_expected=0.5, delta_max=1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        monitor.update(1.5)


def test_monitor_on_constant_ratings_reports_why_there_is_nothing_to_monitor() -> None:
    trace = monitor_alpha([[3, 3, 3], [3, 3, 3]], "ordinal")
    assert trace.final is None
    assert "undefined" in trace.reason


def test_disagreement_stream_is_bounded_and_skips_unpairable_units() -> None:
    stream = disagreement_stream(CANONICAL, "ordinal")
    assert len(stream) == 12, "the three unpairable units emit no observation"
    assert all(0.0 <= x <= 1.0 for x in stream.observations)
    assert stream.delta_max > 0 and stream.d_expected > 0
    # Exactly three units disagreed in the canonical matrix.
    assert sum(1 for x in stream.observations if x > 0) == 3


# --------------------------------------------------------------------------- #
# autoprobe
# --------------------------------------------------------------------------- #


def _results(**evidence_by_oracle: Any) -> list[OracleResult]:
    return [
        OracleResult(oracle=oid, verdict="PASS", evidence=ev)
        for oid, ev in evidence_by_oracle.items()
    ]


def test_probe_resolves_a_scalar_out_of_oracle_evidence() -> None:
    probe = resolve_probe("oracle.O1.max_rel_err", _results(O1={"max_rel_err": 2.5e-7}))
    assert probe.value == pytest.approx(2.5e-7)
    assert probe.oracle == "O1" and probe.reduction == "scalar" and probe.derived is False


def test_probe_walks_nested_evidence_and_sequence_indices() -> None:
    results = _results(O1={"summary": {"per_shape": [{"max_rel_err": 3.0}]}})
    probe = resolve_probe("oracle.O1.summary.per_shape.0.max_rel_err", results)
    assert probe.value == pytest.approx(3.0)


def test_probe_raises_when_the_oracle_skipped_rather_than_defaulting() -> None:
    """The core invariant: a SKIP is not a score, not a zero, not a pass."""
    results = [OracleResult(oracle="O2", verdict="SKIP", reason="clocks are not lockable here")]
    with pytest.raises(ProbeError) as excinfo:
        resolve_probe("oracle.O2.speedup_ci", results, prefer="lo")
    message = str(excinfo.value)
    assert "SKIP" in message and "clocks are not lockable here" in message


def test_probe_raises_when_the_oracle_errored() -> None:
    results = [OracleResult(oracle="O1", verdict="ERROR", reason="budget exceeded")]
    with pytest.raises(ProbeError, match="ERROR"):
        resolve_probe("oracle.O1.max_rel_err", results)


def test_probe_reads_a_failing_oracles_measurement() -> None:
    """FAIL is not SKIP: a failing O1 measured a real error magnitude."""
    results = [
        OracleResult(
            oracle="O1", verdict="FAIL", reason="tolerance exceeded", evidence={"max_rel_err": 0.4}
        )
    ]
    assert resolve_probe("oracle.O1.max_rel_err", results).value == pytest.approx(0.4)


def test_probe_raises_when_the_oracle_never_ran() -> None:
    with pytest.raises(ProbeError, match="no result for oracle"):
        resolve_probe("oracle.O5.max_loss_delta", _results(O1={"max_rel_err": 0.0}))


def test_probe_raises_on_a_missing_evidence_key_and_names_what_is_there() -> None:
    with pytest.raises(ProbeError) as excinfo:
        resolve_probe("oracle.O1.max_abs_err", _results(O1={"max_rel_err": 0.0}))
    assert "max_rel_err" in str(excinfo.value)


def test_probe_raises_on_nan_and_on_none() -> None:
    with pytest.raises(ProbeError, match="NaN"):
        resolve_probe("oracle.O1.max_rel_err", _results(O1={"max_rel_err": NAN}))
    with pytest.raises(ProbeError, match="absent"):
        resolve_probe("oracle.O1.max_rel_err", _results(O1={"max_rel_err": None}))


def test_probe_rejects_a_malformed_path() -> None:
    for bad in ("O1.max_rel_err", "oracle.O1", "", "evidence.O1.x"):
        with pytest.raises(ProbeError, match="malformed"):
            resolve_probe(bad, _results(O1={"x": 1.0}))
    assert parse_probe_path("oracle.O1.a.b") == ("O1", ("a", "b"))


def test_probe_refuses_to_guess_which_end_of_an_interval_to_read() -> None:
    results = _results(O2={"speedup_ci": [0.9, 2.1]})
    with pytest.raises(ProbeError, match="which end"):
        resolve_probe("oracle.O2.speedup_ci", results)
    assert resolve_probe("oracle.O2.speedup_ci", results, prefer="lo").value == pytest.approx(0.9)
    assert resolve_probe("oracle.O2.speedup_ci", results, prefer="hi").value == pytest.approx(2.1)
    assert resolve_probe("oracle.O2.speedup_ci.1", results).value == pytest.approx(2.1)


def test_probe_reads_a_mapping_interval() -> None:
    results = _results(O2={"speedup_ci": {"lo": 1.1, "hi": 1.4}})
    assert resolve_probe("oracle.O2.speedup_ci", results, prefer="lo").value == pytest.approx(1.1)


def test_o3_all_checks_passed_is_read_from_the_subcheck_lists() -> None:
    ok = _results(O3={"fired": [], "errored": [], "passed": ["static_denylist"]})
    probe = resolve_probe("oracle.O3.all_checks_passed", ok)
    assert probe.value == 1.0 and probe.derived is True and probe.reduction == "bool"

    fired = _results(O3={"fired": ["shape_hardcode"], "errored": [], "passed": ["x"]})
    assert resolve_probe("oracle.O3.all_checks_passed", fired).value == 0.0

    nothing_ran = _results(O3={"fired": [], "errored": [], "passed": []})
    with pytest.raises(ProbeError, match="vacuous"):
        resolve_probe("oracle.O3.all_checks_passed", nothing_ran)

    with pytest.raises(ProbeError, match="sub-check lists"):
        resolve_probe("oracle.O3.all_checks_passed", _results(O3={"other": 1}))


def test_threshold_bands_are_inclusive_and_the_last_band_is_terminal() -> None:
    thresholds = [(1e-6, 5), (1e-4, 3), (1e12, 0)]
    assert score_from_thresholds(1e-6, thresholds) == 5
    assert score_from_thresholds(1.1e-6, thresholds) == 3
    assert score_from_thresholds(1.0, thresholds) == 0
    assert score_from_thresholds(math.inf, thresholds) == 0, "the terminal band catches inf"
    with pytest.raises(ProbeError, match="NaN"):
        score_from_thresholds(NAN, thresholds)
    with pytest.raises(ProbeError, match="no probe_thresholds"):
        score_from_thresholds(1.0, [])


def test_threshold_direction_picks_the_pessimistic_end_of_an_interval() -> None:
    assert threshold_direction([(1.0, 0), (1.2, 3), (1e12, 5)]) == "higher_is_better"
    assert threshold_direction([(1e-6, 5), (1e-4, 3), (1e12, 0)]) == "lower_is_better"
    assert preferred_end([(1.0, 0), (1.2, 3), (1e12, 5)]) == "lo"
    assert preferred_end([(1e-6, 5), (1e-4, 3), (1e12, 0)]) == "hi"
    assert preferred_end([(1.0, 3), (2.0, 0), (3.0, 5)]) is None


def test_score_criterion_records_the_measurement_it_used() -> None:
    criterion = RubricCriterion(
        id="performance_claim",
        weight=1.0,
        anchors=_anchors(),
        auto_probe="oracle.O2.speedup_ci",
        probe_thresholds=[(1.0, 0), (1.2, 3), (1e12, 5)],
        machine_probed=True,
    )
    scored = score_criterion(criterion, _results(O2={"speedup_ci": [1.05, 1.9]}))
    assert scored.score == 3, "a claim is worth its interval's pessimistic end"
    assert scored.source == "machine"
    assert scored.probe is not None and scored.probe.reduction == "interval_low"
    assert "1.05" in scored.detail


def test_score_rubric_raises_by_default_when_a_probe_cannot_resolve() -> None:
    rubric = default_rubric(None, "T4", "cuda")
    results = [
        OracleResult(oracle="O1", verdict="PASS", evidence={"max_rel_err": 0.0}),
        OracleResult(oracle="O3", verdict="PASS", evidence={"fired": [], "errored": [], "passed": ["a"]}),
        OracleResult(oracle="O2", verdict="SKIP", reason="ncu is not installed"),
    ]
    humans = {c.id: 4 for c in human_criteria(rubric)}
    with pytest.raises(ProbeError, match="SKIP"):
        score_rubric(rubric, results, humans)


def test_score_rubric_unscored_policy_records_the_gap_instead_of_inventing_a_score() -> None:
    rubric = default_rubric(None, "T4", "cuda")
    results = [
        OracleResult(oracle="O1", verdict="PASS", evidence={"max_rel_err": 0.0}),
        OracleResult(oracle="O3", verdict="PASS", evidence={"fired": [], "errored": [], "passed": ["a"]}),
        OracleResult(oracle="O2", verdict="SKIP", reason="ncu is not installed"),
    ]
    humans = {c.id: 4 for c in human_criteria(rubric)}
    score = score_rubric(rubric, results, humans, on_missing="unscored")
    assert [u.criterion_id for u in score.unscored] == ["performance_claim"]
    assert "SKIP" in score.unscored[0].reason
    assert score.coverage < 1.0
    assert 0.0 <= score.normalized <= 1.0
    # The unscored criterion contributes no weight at all, in either direction.
    assert score.scored_weight == pytest.approx(
        sum(c.weight for c in rubric.criteria if c.id != "performance_claim")
    )


def test_score_rubric_raises_when_a_human_rating_is_missing() -> None:
    rubric = default_rubric(None, "T5", "pytorch")
    results = [
        OracleResult(oracle="O1", verdict="PASS", evidence={"max_rel_err": 0.0}),
        OracleResult(oracle="O3", verdict="PASS", evidence={"fired": [], "errored": [], "passed": ["a"]}),
    ]
    with pytest.raises(ProbeError, match="no rating was supplied"):
        score_rubric(rubric, results, {})


def test_score_rubric_totals_are_a_weighted_mean_on_the_anchor_scale() -> None:
    rubric = build_rubric(
        [
            RubricCriterion(
                id="numerical_correctness",
                weight=3.0,
                anchors=_anchors(),
                auto_probe="oracle.O1.max_rel_err",
                probe_thresholds=[(1e-6, 5), (1e-4, 3), (1e12, 0)],
                machine_probed=True,
            ),
            RubricCriterion(id="explanation_quality", weight=1.0, anchors=_anchors()),
        ]
    )
    score = score_rubric(
        rubric,
        _results(O1={"max_rel_err": 1e-9}),
        {"explanation_quality": 1},
    )
    assert isinstance(score, RubricScore)
    assert score.score == pytest.approx(0.75 * 5 + 0.25 * 1)
    assert score.normalized == pytest.approx(score.score / 5.0)
    assert score.coverage == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# rubric spec
# --------------------------------------------------------------------------- #


def test_weight_sum_validator_rejects_a_rubric_that_does_not_sum_to_one() -> None:
    bad = RubricSpec(
        criteria=[
            RubricCriterion(id="explanation_quality", weight=0.4, anchors=_anchors()),
            RubricCriterion(id="diagnostic_reasoning", weight=0.4, anchors=_anchors()),
        ]
    )
    problems = rubric_problems(bad)
    assert any("sum to" in p for p in problems)
    with pytest.raises(RubricError, match="sum to"):
        validate_rubric(bad)
    # ...and it passes once normalised.
    fixed = normalize_weights(bad)
    assert fixed.total_weight() == pytest.approx(1.0, abs=1e-12)
    validate_rubric(fixed)


def test_normalize_weights_refuses_a_zero_weight_rubric() -> None:
    zero = RubricSpec(
        criteria=[RubricCriterion(id="explanation_quality", weight=0.0, anchors=_anchors())]
    )
    with pytest.raises(RubricError, match="zero or less"):
        normalize_weights(zero)


def test_a_machine_probeable_criterion_may_not_be_left_to_human_raters() -> None:
    rubric = RubricSpec(
        criteria=[RubricCriterion(id="numerical_correctness", weight=1.0, anchors=_anchors())]
    )
    problems = rubric_problems(rubric)
    assert any("oracle.O1.max_rel_err" in p and "human" in p for p in problems)


def test_a_human_judgement_may_not_be_dressed_up_as_a_measurement() -> None:
    rubric = RubricSpec(
        criteria=[
            RubricCriterion(
                id="explanation_quality",
                weight=1.0,
                anchors=_anchors(),
                auto_probe="oracle.O1.max_rel_err",
                probe_thresholds=[(1.0, 5)],
                machine_probed=True,
            )
        ]
    )
    problems = rubric_problems(rubric)
    assert any("fabricated measurement" in p for p in problems)


def test_validator_rejects_a_probe_for_an_oracle_the_task_never_runs() -> None:
    rubric = default_rubric(None, "T4", "cuda")
    problems = rubric_problems(rubric, available_oracles=["O1", "O3"])
    assert any("O2" in p and "can only ever fail" in p for p in problems)


def test_validator_rejects_a_threshold_score_with_no_anchor() -> None:
    rubric = RubricSpec(
        criteria=[
            RubricCriterion(
                id="numerical_correctness",
                weight=1.0,
                anchors=_anchors(),
                auto_probe="oracle.O1.max_rel_err",
                probe_thresholds=[(1e-6, 4), (1e12, 0)],
                machine_probed=True,
            )
        ]
    )
    assert any("has no anchor" in p for p in rubric_problems(rubric))


def test_validator_rejects_duplicate_ids_and_an_empty_rubric() -> None:
    dup = RubricSpec(
        criteria=[
            RubricCriterion(id="explanation_quality", weight=0.5, anchors=_anchors()),
            RubricCriterion(id="explanation_quality", weight=0.5, anchors=_anchors()),
        ]
    )
    assert any("appears 2 times" in p for p in rubric_problems(dup))
    assert any("no criteria" in p for p in rubric_problems(RubricSpec()))


def test_default_rubric_is_valid_normalised_and_mostly_machine_probed() -> None:
    for tier, domain in (("T5", "pytorch"), ("T4", "cuda"), ("T6", "distributed")):
        rubric = default_rubric(None, tier, domain)
        validate_rubric(rubric)
        assert rubric.total_weight() == pytest.approx(1.0, abs=1e-12)
        ids = {c.id for c in rubric.criteria}
        assert {"numerical_correctness", "no_reward_hacking"} <= ids
        assert {"explanation_quality", "diagnostic_reasoning", "root_cause_not_symptom"} <= ids
        probed = {c.id for c in machine_criteria(rubric)}
        assert probed >= {"numerical_correctness", "no_reward_hacking"}
        assert all(c.auto_probe and c.probe_thresholds for c in machine_criteria(rubric))


def test_default_rubric_only_emits_a_criterion_for_an_oracle_that_will_run() -> None:
    assert "performance_claim" in {c.id for c in default_rubric(None, "T4", "cuda").criteria}
    assert "performance_claim" not in {c.id for c in default_rubric(None, "T5", "pytorch").criteria}
    assert "multi_rank_agreement" in {
        c.id for c in default_rubric(None, "T6", "distributed").criteria
    }
    explicit = default_rubric(None, "T5", "pytorch", oracles=["O1", "O2", "O3"])
    assert "performance_claim" in {c.id for c in explicit.criteria}


def test_default_rubric_takes_its_tolerance_band_from_the_seed(tiny_seed: Any) -> None:
    tiny_seed.extras["rel_tolerance"] = 1e-3
    rubric = default_rubric(tiny_seed, "T5", "pytorch")
    numerics = next(c for c in rubric.criteria if c.id == "numerical_correctness")
    assert numerics.probe_thresholds[0][0] == pytest.approx(1e-3)


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


def _agreeing(n: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 6, size=n)
    return [truth.tolist(), truth.tolist()]


def _disagreeing(n: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 6, size=(2, n)).tolist()


def test_gate_flags_exactly_the_criteria_below_the_threshold() -> None:
    ratings = {
        "explanation_quality": _agreeing(40, 1),
        "diagnostic_reasoning": _disagreeing(40, 2),
        "root_cause_not_symptom": _agreeing(40, 3),
    }
    report = gate_criteria(ratings, threshold=GATE_ALPHA, metric="ordinal")
    flagged = {g.criterion_id for g in report.flagged}
    kept = {g.criterion_id for g in report.kept}
    assert flagged == {"diagnostic_reasoning"}
    assert kept == {"explanation_quality", "root_cause_not_symptom"}
    assert report.passed is False
    for gate in report.gates:
        assert (gate.alpha >= GATE_ALPHA) == (gate.action == "keep")


def test_a_flagged_criterion_blames_the_criterion_and_attaches_the_evidence() -> None:
    report = gate_criteria({"diagnostic_reasoning": _disagreeing(40, 9)}, metric="ordinal")
    gate = report.by_id("diagnostic_reasoning")
    assert gate.action == "rewrite" and gate.keep is False
    assert "REWRITE THE CRITERION" in gate.message
    assert "not in the raters" in gate.message
    assert gate.evidence, "the disagreeing items are the evidence for the rewrite"
    assert all(u.pairs for u in gate.evidence)
    top = gate.evidence[0]
    assert top.disagreement >= gate.evidence[-1].disagreement
    brief = rewrite_brief(gate)
    assert top.unit_id in brief and "anchor text" in brief


def test_gate_exposes_a_running_alpha_trend_per_criterion() -> None:
    report = gate_criteria({"explanation_quality": _agreeing(30, 4)}, metric="ordinal")
    trends = report.trends()
    assert set(trends) == {"explanation_quality"}
    assert len(trends["explanation_quality"]) == 30
    assert trends["explanation_quality"][-1] == pytest.approx(report.by_id("explanation_quality").alpha)


def test_gate_exempts_machine_probed_criteria_and_will_not_pass_unmeasured_ones() -> None:
    rubric = default_rubric(None, "T5", "pytorch")
    report = gate_criteria(
        {"explanation_quality": _agreeing(30, 5)}, rubric=rubric, metric="ordinal"
    )
    actions = {g.criterion_id: g.action for g in report.gates}
    assert actions["numerical_correctness"] == "exempt_machine_probed"
    assert actions["no_reward_hacking"] == "exempt_machine_probed"
    assert actions["diagnostic_reasoning"] == "unmeasured"
    assert actions["explanation_quality"] == "keep"
    assert report.flagged == []
    assert report.passed is False, "unknown reliability is not adequate reliability"
    assert report.by_id("numerical_correctness").keep is True


def test_gate_reports_an_uncomputable_alpha_as_unmeasured_not_as_a_pass() -> None:
    gate = gate_criterion("explanation_quality", [[3, 3, 3], [3, 3, 3]])
    assert gate.action == "unmeasured"
    assert math.isnan(gate.alpha)
    assert gate.keep is False
    assert "not a pass" in gate.message


def test_gate_report_serialises_into_the_shape_the_dashboard_reads() -> None:
    from crucible.report.metrics import normalize_alphas

    report = gate_criteria(
        {"explanation_quality": _agreeing(30, 6), "diagnostic_reasoning": _disagreeing(30, 7)},
        metric="ordinal",
    )
    payload = report.as_dict()
    assert payload["threshold"] == GATE_ALPHA
    alphas = normalize_alphas(payload)
    assert set(alphas) == {"explanation_quality", "diagnostic_reasoning"}
    assert alphas["explanation_quality"] > alphas["diagnostic_reasoning"]


def test_gate_can_attach_a_monitored_alpha_trace() -> None:
    report = gate_criteria(
        {"explanation_quality": _agreeing(40, 8)},
        metric="ordinal",
        with_monitor=True,
        cs_kind="hoeffding",
    )
    trace = report.by_id("explanation_quality").monitor
    assert trace is not None and trace.cs_kind == "hoeffding"
    assert len(trace.bounds) == 40
    assert trace.final is not None and trace.final.alpha_hi <= 1.0
