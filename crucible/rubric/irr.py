"""Krippendorff's alpha, computed from the coincidence matrix.

This is the number the proposal points at when it claims the rubric is
reliable, so it is implemented from the definition rather than borrowed, and the
tests check it against a hand-worked canonical example rather than against
itself.

The construction, in the order the code performs it:

1. **Reliability data.** A units x raters matrix; ``NaN`` is "this rater did not
   rate this unit". A unit with fewer than two ratings contributes nothing - not
   zero disagreement, *nothing* - because a single rating cannot agree or
   disagree with anything. Those units are excluded and counted.
2. **Coincidence matrix.** ``o[c][k] = sum over units of n_uc (n_uk - delta_ck)
   / (m_u - 1)``, where ``n_uc`` is how many raters gave unit ``u`` the value
   ``c`` and ``m_u`` is that unit's number of ratings. Dividing by ``m_u - 1``
   is what makes units with different numbers of raters commensurable; it is the
   whole reason the coincidence matrix exists rather than a plain pair count.
3. **Marginals.** ``n_c = sum_k o[c][k]``; ``n = sum_c n_c`` is the number of
   *pairable* values, which is smaller than the number of ratings whenever a
   unit was rated once.
4. **Difference function.** ``nominal``, ``ordinal``, ``interval``, ``ratio``.
   The ordinal metric is the cumulative one,
   ``delta2(c,k) = (sum_{g=c..k} n_g - (n_c + n_k)/2)^2``, so it depends on the
   marginals and not only on the values: two categories are "far apart" when a
   lot of the data lies between them.
5. ``D_o = sum o.delta2 / n``; ``D_e = sum n_c n_k delta2 / (n (n-1))``;
   ``alpha = 1 - D_o / D_e``.

Where alpha is undefined it is reported as ``NaN`` **with a reason**, not as 0
and not as 1. ``D_e = 0`` means every pairable value in the data is identical:
there is no variation to reproduce, so reliability is not a meaningful question,
and answering "perfect" would be a fabrication.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

Metric = Literal["nominal", "ordinal", "interval", "ratio"]
METRICS: tuple[str, ...] = ("nominal", "ordinal", "interval", "ratio")

Orientation = Literal["raters_x_units", "units_x_raters"]


# --------------------------------------------------------------------------- #
# data marshalling
# --------------------------------------------------------------------------- #


def to_units(
    matrix: Any,
    *,
    orientation: Orientation = "raters_x_units",
) -> np.ndarray:
    """Coerce ratings into a ``(n_units, n_raters)`` float array with NaN gaps.

    Accepts a nested sequence, a numpy array, or a mapping ``{rater: ratings}``
    (in which case each value is one rater's row and ``orientation`` is ignored).
    ``None`` and empty strings become NaN, so a spreadsheet with blanks loads
    without a separate cleaning pass.
    """
    if isinstance(matrix, Mapping):
        rows = [list(v) for _k, v in sorted(matrix.items(), key=lambda kv: str(kv[0]))]
        orientation = "raters_x_units"
    elif isinstance(matrix, np.ndarray):
        rows = [list(r) for r in np.atleast_2d(matrix)]
    else:
        rows = [list(r) for r in matrix]

    if not rows:
        raise ValueError("ratings are empty: no raters")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"ragged ratings matrix: rows have widths {sorted(widths)}")
    if widths == {0}:
        raise ValueError("ratings are empty: no units")

    cleaned: list[list[float]] = []
    for i, row in enumerate(rows):
        out: list[float] = []
        for j, cell in enumerate(row):
            if cell is None or (isinstance(cell, str) and not cell.strip()):
                out.append(math.nan)
                continue
            if isinstance(cell, bool):
                out.append(float(cell))
                continue
            try:
                out.append(float(cell))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"rating at row {i}, column {j} is not a number and not blank: {cell!r}"
                ) from exc
        cleaned.append(out)

    arr = np.asarray(cleaned, dtype=float)
    if orientation == "raters_x_units":
        arr = arr.T
    elif orientation != "units_x_raters":
        raise ValueError(
            f"orientation must be 'raters_x_units' or 'units_x_raters', got {orientation!r}"
        )
    return arr


def usable_units(units: np.ndarray) -> np.ndarray:
    """Boolean mask of units with at least two ratings."""
    return np.asarray((~np.isnan(units)).sum(axis=1) >= 2)


# --------------------------------------------------------------------------- #
# coincidence matrix and difference functions
# --------------------------------------------------------------------------- #


def coincidence_matrix(units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(values, o)``: the sorted distinct values and the coincidence matrix."""
    per_unit = [row[~np.isnan(row)] for row in np.atleast_2d(units)]
    pairable = [v for v in per_unit if v.size >= 2]
    if not pairable:
        return np.zeros(0), np.zeros((0, 0))
    values = np.unique(np.concatenate(pairable))
    index = {float(v): i for i, v in enumerate(values)}
    k = values.size
    o = np.zeros((k, k), dtype=float)
    for v in pairable:
        m = float(v.size)
        counts = np.zeros(k, dtype=float)
        for x in v:
            counts[index[float(x)]] += 1.0
        o += (np.outer(counts, counts) - np.diag(counts)) / (m - 1.0)
    return values, o


def difference_matrix(values: np.ndarray, n_c: np.ndarray, metric: Metric) -> np.ndarray:
    """``delta2[c][k]`` for the chosen metric.

    ``n_c`` (the coincidence marginals) is used by the ordinal metric only, but
    is required for every metric so that callers cannot accidentally build an
    ordinal matrix without it.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {list(METRICS)}, got {metric!r}")
    k = int(values.size)
    if k == 0:
        return np.zeros((0, 0))

    if metric == "nominal":
        return 1.0 - np.eye(k)

    if metric == "interval":
        diff = values[:, None] - values[None, :]
        return diff**2

    if metric == "ratio":
        if np.any(values < 0):
            raise ValueError(
                "the ratio metric is defined for non-negative values only; the data "
                f"contains {float(values.min())!r}"
            )
        total = values[:, None] + values[None, :]
        diff = values[:, None] - values[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(total > 0, (diff / np.where(total > 0, total, 1.0)) ** 2, 0.0)
        return np.asarray(out, dtype=float)

    # ordinal: (sum of marginals spanned, minus half the two endpoints)^2
    if n_c.size != k:
        raise ValueError(
            f"ordinal metric needs one marginal per value: got {n_c.size} marginals "
            f"for {k} values"
        )
    cum = np.cumsum(n_c)
    # span[c][k] = sum_{g=min..max} n_g, inclusive of both endpoints
    lo = np.minimum(np.arange(k)[:, None], np.arange(k)[None, :])
    hi = np.maximum(np.arange(k)[:, None], np.arange(k)[None, :])
    span = cum[hi] - cum[lo] + n_c[lo]
    endpoints = (n_c[:, None] + n_c[None, :]) / 2.0
    out = (span - endpoints) ** 2
    np.fill_diagonal(out, 0.0)
    return out


# --------------------------------------------------------------------------- #
# alpha
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AlphaResult:
    """Alpha with everything needed to audit or reproduce it."""

    alpha: float
    metric: str
    n_units: int
    n_units_excluded: int
    n_raters: int
    n_pairable: float
    d_observed: float
    d_expected: float
    values: list[float] = field(default_factory=list)
    reason: str = ""
    ci_lo: float = math.nan
    ci_hi: float = math.nan
    ci_level: float = math.nan
    n_bootstrap: int = 0
    n_bootstrap_dropped: int = 0

    @property
    def defined(self) -> bool:
        return math.isfinite(self.alpha)

    def as_dict(self) -> dict[str, Any]:
        def clean(x: float) -> float | None:
            return None if not math.isfinite(x) else float(x)

        return {
            "alpha": clean(self.alpha),
            "metric": self.metric,
            "n_units": self.n_units,
            "n_units_excluded": self.n_units_excluded,
            "n_raters": self.n_raters,
            "n_pairable": self.n_pairable,
            "d_observed": clean(self.d_observed),
            "d_expected": clean(self.d_expected),
            "values": self.values,
            "reason": self.reason,
            "ci": [clean(self.ci_lo), clean(self.ci_hi)] if self.n_bootstrap else None,
            "ci_level": clean(self.ci_level),
            "n_bootstrap": self.n_bootstrap,
            "n_bootstrap_dropped": self.n_bootstrap_dropped,
        }


@dataclass(frozen=True)
class _Core:
    alpha: float
    d_observed: float
    d_expected: float
    n_pairable: float
    values: np.ndarray
    reason: str


def _alpha_core(units: np.ndarray, metric: Metric) -> _Core:
    """Alpha and its parts for a units x raters block."""
    values, o = coincidence_matrix(units)
    if values.size == 0:
        return _Core(
            math.nan, math.nan, math.nan, 0.0, values,
            "no unit was rated by two or more raters",
        )
    n_c = o.sum(axis=1)
    n = float(n_c.sum())
    if n < 2:
        return _Core(math.nan, math.nan, math.nan, n, values, f"only {n:g} pairable value(s)")
    delta = difference_matrix(values, n_c, metric)
    d_o = float((o * delta).sum() / n)
    d_e = float((np.outer(n_c, n_c) * delta).sum() / (n * (n - 1.0)))
    if d_e <= 0.0:
        return _Core(
            math.nan,
            d_o,
            d_e,
            n,
            values,
            "expected disagreement is zero: every pairable value in the data is identical, so "
            "there is no variation for raters to reproduce and alpha is 0/0, not 1",
        )
    return _Core(1.0 - d_o / d_e, d_o, d_e, n, values, "")


def krippendorff_alpha(
    matrix: Any,
    metric: Metric | str = "nominal",
    *,
    orientation: Orientation = "raters_x_units",
    bootstrap: int = 0,
    ci_level: float = 0.95,
    rng_seed: int = 1234,
) -> AlphaResult:
    """Krippendorff's alpha over a ratings matrix (rows = raters by default).

    ``bootstrap > 0`` adds a percentile confidence interval by resampling
    **units** with replacement and recomputing alpha end to end. Resamples in
    which alpha is undefined (a draw with no variation at all) are dropped and
    counted rather than being scored as 1.0, which would bias the interval
    upward exactly where the data is weakest.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {list(METRICS)}, got {metric!r}")
    units = to_units(matrix, orientation=orientation)
    mask = usable_units(units)
    used = units[mask]
    n_excluded = int((~mask).sum())

    core = _alpha_core(units, metric)  # type: ignore[arg-type]
    reason = core.reason
    if n_excluded and not reason:
        reason = (
            f"{n_excluded} unit(s) had fewer than two ratings and were excluded; "
            "a single rating cannot agree or disagree with anything"
        )

    result = AlphaResult(
        alpha=core.alpha,
        metric=str(metric),
        n_units=int(used.shape[0]),
        n_units_excluded=n_excluded,
        n_raters=int(units.shape[1]),
        n_pairable=float(core.n_pairable),
        d_observed=core.d_observed,
        d_expected=core.d_expected,
        values=[float(v) for v in core.values],
        reason=reason,
    )
    if bootstrap <= 0 or used.shape[0] < 2:
        return result

    rng = np.random.default_rng(rng_seed)
    draws: list[float] = []
    dropped = 0
    for _ in range(int(bootstrap)):
        pick = rng.integers(0, used.shape[0], size=used.shape[0])
        drawn = _alpha_core(used[pick], metric)  # type: ignore[arg-type]
        if math.isfinite(drawn.alpha):
            draws.append(drawn.alpha)
        else:
            dropped += 1
    if not draws:
        return replace(
            result,
            n_bootstrap=int(bootstrap),
            n_bootstrap_dropped=dropped,
            reason=(result.reason + "; " if result.reason else "")
            + "every bootstrap resample was degenerate, so no interval could be formed",
        )
    tail = (1.0 - float(ci_level)) / 2.0
    lo, hi = np.percentile(draws, [100.0 * tail, 100.0 * (1.0 - tail)])
    return replace(
        result,
        ci_lo=float(lo),
        ci_hi=float(hi),
        ci_level=float(ci_level),
        n_bootstrap=int(bootstrap),
        n_bootstrap_dropped=dropped,
    )


#: Aliases the CLI probes for (``crucible.rubric.irr.alpha`` / ``compute_alpha``).
alpha = krippendorff_alpha
compute_alpha = krippendorff_alpha


def alpha_by_criterion(
    ratings: Mapping[str, Any],
    metric: Metric | str = "ordinal",
    **kwargs: Any,
) -> dict[str, AlphaResult]:
    """``{criterion_id: matrix}`` -> ``{criterion_id: AlphaResult}``."""
    return {
        str(cid): krippendorff_alpha(m, metric, **kwargs) for cid, m in ratings.items()
    }


def running_alpha(
    matrix: Any,
    metric: Metric | str = "ordinal",
    *,
    orientation: Orientation = "raters_x_units",
) -> list[float]:
    """Alpha recomputed after each additional unit, in the order given.

    This is the *trend* a reviewer looks at, not a sequential test: recomputing
    alpha at every step and stopping when it looks good is the peeking problem.
    ``sequences.AlphaMonitor`` is the version with a guarantee attached.
    """
    units = to_units(matrix, orientation=orientation)
    out: list[float] = []
    for i in range(1, units.shape[0] + 1):
        out.append(_alpha_core(units[:i], metric).alpha)  # type: ignore[arg-type]
    return out


# --------------------------------------------------------------------------- #
# disagreement, per unit and per pair
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RaterPair:
    """One disagreeing pair of ratings on one unit."""

    rater_i: int
    rater_j: int
    value_i: float
    value_j: float
    delta2: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "rater_i": self.rater_i,
            "rater_j": self.rater_j,
            "value_i": self.value_i,
            "value_j": self.value_j,
            "delta2": self.delta2,
        }


@dataclass(frozen=True)
class UnitDisagreement:
    """A unit, its normalised disagreement, and the pairs that caused it."""

    unit_index: int
    unit_id: str
    n_ratings: int
    disagreement: float  # mean delta2 over ordered pairs, normalised to [0, 1]
    pairs: list[RaterPair] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_index": self.unit_index,
            "unit_id": self.unit_id,
            "n_ratings": self.n_ratings,
            "disagreement": self.disagreement,
            "pairs": [p.as_dict() for p in self.pairs],
        }


@dataclass(frozen=True)
class DisagreementStream:
    """Per-unit disagreement in [0, 1], plus the constants used to scale it.

    ``delta_max`` and ``d_expected`` are **plug-in** quantities estimated once
    from the whole data set (they need the value set and its marginals). See
    ``sequences.AlphaMonitor`` for what that costs the guarantee.
    """

    observations: list[float]
    units: list[UnitDisagreement]
    delta_max: float
    d_expected: float
    metric: str
    values: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.observations)


def disagreement_stream(
    matrix: Any,
    metric: Metric | str = "ordinal",
    *,
    orientation: Orientation = "raters_x_units",
    unit_ids: Sequence[str] | None = None,
) -> DisagreementStream:
    """Per-unit observed disagreement, scaled into [0, 1].

    For unit ``u`` with ``m_u`` ratings the raw quantity is the mean of
    ``delta2`` over the ``m_u (m_u - 1)`` ordered rater pairs. Dividing by the
    largest ``delta2`` in the value set puts it in [0, 1], which is what the
    confidence sequences require.

    Units with fewer than two ratings emit no observation - they are absent from
    the stream rather than contributing a zero.
    """
    units_arr = to_units(matrix, orientation=orientation)
    values, o = coincidence_matrix(units_arr)
    if values.size == 0:
        return DisagreementStream([], [], 0.0, 0.0, str(metric), [])
    n_c = o.sum(axis=1)
    n = float(n_c.sum())
    delta = difference_matrix(values, n_c, metric)  # type: ignore[arg-type]
    delta_max = float(delta.max()) if delta.size else 0.0
    d_e = (
        float((np.outer(n_c, n_c) * delta).sum() / (n * (n - 1.0))) if n >= 2 else 0.0
    )
    index = {float(v): i for i, v in enumerate(values)}

    observations: list[float] = []
    records: list[UnitDisagreement] = []
    for u in range(units_arr.shape[0]):
        row = units_arr[u]
        present = [(r, float(row[r])) for r in range(row.size) if not math.isnan(row[r])]
        if len(present) < 2:
            continue
        pairs: list[RaterPair] = []
        total = 0.0
        for a in range(len(present)):
            for b in range(len(present)):
                if a == b:
                    continue
                ri, vi = present[a]
                rj, vj = present[b]
                d2 = float(delta[index[vi], index[vj]])
                total += d2
                if a < b and d2 > 0.0:
                    pairs.append(RaterPair(ri, rj, vi, vj, d2))
        m = len(present)
        raw = total / (m * (m - 1))
        scaled = 0.0 if delta_max <= 0 else min(max(raw / delta_max, 0.0), 1.0)
        observations.append(scaled)
        records.append(
            UnitDisagreement(
                unit_index=u,
                unit_id=str(unit_ids[u]) if unit_ids and u < len(unit_ids) else f"unit_{u}",
                n_ratings=m,
                disagreement=scaled,
                pairs=sorted(pairs, key=lambda p: p.delta2, reverse=True),
            )
        )
    return DisagreementStream(
        observations=observations,
        units=records,
        delta_max=delta_max,
        d_expected=d_e,
        metric=str(metric),
        values=[float(v) for v in values],
    )


def disagreeing_units(
    matrix: Any,
    metric: Metric | str = "ordinal",
    *,
    orientation: Orientation = "raters_x_units",
    unit_ids: Sequence[str] | None = None,
    limit: int = 10,
) -> list[UnitDisagreement]:
    """The units raters disagreed on most, worst first. Evidence for a rewrite."""
    stream = disagreement_stream(
        matrix, metric, orientation=orientation, unit_ids=unit_ids
    )
    ranked = [u for u in stream.units if u.disagreement > 0.0]
    ranked.sort(key=lambda u: u.disagreement, reverse=True)
    return ranked[: max(int(limit), 0)]


__all__ = [
    "METRICS",
    "Metric",
    "AlphaResult",
    "RaterPair",
    "UnitDisagreement",
    "DisagreementStream",
    "to_units",
    "usable_units",
    "coincidence_matrix",
    "difference_matrix",
    "krippendorff_alpha",
    "alpha",
    "compute_alpha",
    "alpha_by_criterion",
    "running_alpha",
    "disagreement_stream",
    "disagreeing_units",
]
