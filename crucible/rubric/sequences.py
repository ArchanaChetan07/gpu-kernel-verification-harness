"""Anytime-valid confidence sequences, and alpha monitored as ratings arrive.

The three estimators - :class:`HoeffdingCS`, :class:`EmpiricalBernsteinCS` and
:class:`BettingCS` - are ported, API and math intact, from the
**certified-sparse-attention** project (``csa/verify.py``, "Mechanism 2 - sampled
dense verification with anytime-valid bounds"), where they bound the diverged
step fraction of a sparse attention kernel. Credit for the construction, the
aGRAPA-style predictable bets and the "all candidates rejected means the capital
process failed" behaviour belongs there; this module reuses them unchanged and
adds the CRUCIBLE-specific layer below.

Why they are here
-----------------
Inter-rater reliability is watched while it accumulates. A reviewer computes
alpha after ten items, then twenty, then fifty, and stops when it clears 0.67 -
which is the peeking problem in its purest form: a fixed-sample interval
recomputed at every stopping time has no coverage guarantee at the time you
actually chose to stop. A confidence sequence is valid *simultaneously at every
t*, so "we stopped when the bound cleared the gate" is a legitimate statement.

What is monitored
-----------------
The stream is the per-unit **observed disagreement**
``d_u = mean over ordered rater pairs of delta2(x_i, x_j) / delta2_max``, using
the same difference function as the alpha metric (see
``irr.disagreement_stream``). It is bounded in [0, 1] by construction, which is
what every estimator here requires.

The approximation, stated plainly
---------------------------------
``alpha = 1 - D_o / D_e``. The confidence sequence bounds ``D_o`` only. ``D_e``
is supplied as a **plug-in**: it is estimated from the marginals of the data
seen so far and then treated as a known constant when the bound on ``D_o`` is
mapped through to a bound on alpha. Two consequences, both real:

1. The resulting alpha interval is anytime-valid **conditional on D_e being
   correct**. It is not a joint confidence statement about the pair
   ``(D_o, D_e)``, and its coverage is therefore not exactly ``1 - alpha_level``
   for alpha itself. Nothing here should be reported as "an anytime-valid
   confidence sequence for Krippendorff's alpha" without that qualifier.
2. ``D_e`` depends on the marginal distribution of ratings, which is itself
   still moving early on. The monitor exposes ``d_expected_history`` so a
   reader can see whether the plug-in had settled by the time the gate was
   crossed. A plug-in that is still drifting is a reason to keep rating, not a
   reason to trust the bound.

There is also a weighting approximation: ``D_o`` is the *pairable-value
weighted* mean of the per-unit disagreements, while the confidence sequence
bounds their *unweighted* mean. The two coincide exactly when every unit has the
same number of ratings, which is the normal case for a rating exercise; when
they do not, :class:`AlphaMonitor` records ``equal_ratings=False`` so the
discrepancy is visible rather than assumed away.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from .irr import Metric, disagreement_stream

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# ported from certified-sparse-attention/csa/verify.py
# --------------------------------------------------------------------------- #


class ConfidenceSequence:
    """Base: anytime-valid CS for the running mean of a [0,1]-bounded stream.

    NOTE on intersection: intersecting intervals across time is only valid when
    the target is a FIXED mean. Hoeffding/EB intervals below are simultaneously
    valid for the running (conditional) mean at every t via martingale
    arguments, so the raw interval is reported at each t. Set ``intersect=True``
    only for a stationary target.
    """

    def __init__(self, alpha: float = 0.05, intersect: bool = False) -> None:
        self.alpha = float(alpha)
        self.intersect = bool(intersect)
        self.n = 0
        self.lo = 0.0
        self.hi = 1.0

    def update(self, x: float) -> tuple[float, float]:
        raise NotImplementedError

    def _set(self, lo: float, hi: float) -> tuple[float, float]:
        lo, hi = max(0.0, lo), min(1.0, hi)
        if self.intersect:
            lo, hi = max(self.lo, lo), min(self.hi, hi)
            if lo > hi:
                lo = hi = 0.5 * (lo + hi)
        self.lo, self.hi = lo, hi
        return self.lo, self.hi

    @property
    def width(self) -> float:
        return self.hi - self.lo


class HoeffdingCS(ConfidenceSequence):
    """Union-bound Hoeffding: ``alpha_t = alpha / (t (t+1))``, which sums to alpha.

    Azuma-Hoeffding applies to the martingale sum of ``x_i - E[x_i | past]``, so
    each interval covers the running conditional mean even under drift; the
    union bound makes coverage simultaneous over all t.
    """

    def __init__(self, alpha: float = 0.05, intersect: bool = False) -> None:
        super().__init__(alpha, intersect)
        self.sum = 0.0

    def update(self, x: float) -> tuple[float, float]:
        self.n += 1
        self.sum += float(x)
        mean = self.sum / self.n
        eps = math.sqrt(
            math.log(2.0 * self.n * (self.n + 1) / self.alpha) / (2.0 * self.n)
        )
        return self._set(mean - eps, mean + eps)


class EmpiricalBernsteinCS(ConfidenceSequence):
    """Predictable plug-in empirical-Bernstein CS (Waudby-Smith & Ramdas 2023)."""

    def __init__(self, alpha: float = 0.05, c: float = 0.5, intersect: bool = False) -> None:
        super().__init__(alpha, intersect)
        self.c = float(c)
        self.sum_x = 0.0
        self.mu_prev = 0.5  # mu_hat_{t-1} with prior weight 1
        self.var_prev = 0.25  # sigma^2_hat_{t-1} with prior weight 1
        self.sum_sq_dev = 0.25
        self.S_lx = 0.0  # sum lambda_i * x_i
        self.S_l = 0.0  # sum lambda_i
        self.S_v = 0.0  # sum v_i * psi_E(lambda_i)

    @staticmethod
    def _psi_e(lam: float) -> float:
        # Fan's inequality form: valid with v_i = (x_i - mu_hat_{i-1})^2.
        # (WSR'23 state psi/4 with v = 4(.)^2; the factors cancel.)
        return -math.log1p(-lam) - lam

    def update(self, x: float) -> tuple[float, float]:
        x = float(x)
        t = self.n + 1
        lam = math.sqrt(
            2.0 * math.log(2.0 / self.alpha) / (self.var_prev * t * math.log(t + 1.0))
        )
        lam = min(lam, self.c)
        v = (x - self.mu_prev) ** 2
        self.S_lx += lam * x
        self.S_l += lam
        self.S_v += v * self._psi_e(lam)
        self.n = t
        # Predictable plug-in: accumulate sq-dev vs mu_hat_{t-1}, then update.
        self.sum_sq_dev += (x - self.mu_prev) ** 2
        self.sum_x += x
        self.mu_prev = (0.5 + self.sum_x) / (t + 1.0)
        self.var_prev = self.sum_sq_dev / (t + 1.0)
        if self.S_l <= 0:
            return self.lo, self.hi
        center = self.S_lx / self.S_l
        rad = (math.log(2.0 / self.alpha) + self.S_v) / self.S_l
        return self._set(center - rad, center + rad)


class BettingCS(ConfidenceSequence):
    """Hedged betting CS over a grid of candidate means.

    For each candidate mean ``m`` grow capital
    ``K+(m) = prod(1 + lam_t (x_t - m))`` and ``K-(m) = prod(1 - lam_t (x_t -
    m))``; reject ``m`` once ``max(K+, K-)`` ever reaches ``1/alpha`` (Ville's
    inequality makes rejection permanent). aGRAPA-style predictable bets.

    Scope: targets a FIXED mean. Permanent rejection is intrinsic to the capital
    process, so under a drifting target this CS can lock onto early behaviour.
    Use EB for bursty streams.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        grid: int = 401,
        c: float = 0.5,
        intersect: bool = True,
    ) -> None:
        super().__init__(alpha, intersect)
        self.m = np.linspace(0.0, 1.0, grid)
        self.logK_plus = np.zeros(grid)
        self.logK_minus = np.zeros(grid)
        self.rejected = np.zeros(grid, dtype=bool)
        self.failed = False  # True once every candidate mean is rejected
        self.c = float(c)
        self.sum_x = 0.0
        self.mu_prev = 0.5
        self.var_prev = 0.25
        self.sum_sq_dev = 0.25
        self.thresh = math.log(1.0 / alpha)

    def update(self, x: float) -> tuple[float, float]:
        x = float(x)
        t = self.n + 1
        m = self.m
        # predictable bet: approximate-GRAPA centred at the running mean
        lam = (self.mu_prev - m) / (self.var_prev + (self.mu_prev - m) ** 2)
        lam_plus = np.clip(lam, 0.0, self.c / np.maximum(m, 1e-4))
        lam_minus = np.clip(-lam, 0.0, self.c / np.maximum(1.0 - m, 1e-4))
        self.logK_plus += np.log1p(lam_plus * (x - m))
        self.logK_minus += np.log1p(-lam_minus * (x - m))
        self.rejected |= np.maximum(self.logK_plus, self.logK_minus) >= self.thresh
        self.n = t
        # Predictable plug-in: accumulate sq-dev vs mu_hat_{t-1}, then update.
        self.sum_sq_dev += (x - self.mu_prev) ** 2
        self.sum_x += x
        self.mu_prev = (0.5 + self.sum_x) / (t + 1.0)
        self.var_prev = self.sum_sq_dev / (t + 1.0)
        alive = ~self.rejected
        if alive.any():
            lo, hi = float(self.m[alive].min()), float(self.m[alive].max())
        else:
            # Every candidate mean has been rejected. That is not a zero-width
            # certificate -- it means the capital process has failed (its
            # fixed-mean assumption is violated, e.g. the target drifted).
            # Degrade to vacuous and say so, rather than emitting a confident
            # point estimate that is exactly what this project exists to
            # prevent.
            self.failed = True
            self.lo, self.hi = 0.0, 1.0
            return self.lo, self.hi
        return self._set(lo, hi)


CS_KINDS: dict[str, type[ConfidenceSequence]] = {
    "hoeffding": HoeffdingCS,
    "eb": EmpiricalBernsteinCS,
    "betting": BettingCS,
}


def make_cs(kind: str, alpha_level: float = 0.05, **kwargs: Any) -> ConfidenceSequence:
    """``"hoeffding" | "eb" | "betting"`` -> a fresh confidence sequence."""
    try:
        cls = CS_KINDS[kind]
    except KeyError as exc:
        raise ValueError(
            f"unknown confidence sequence {kind!r}; known: {sorted(CS_KINDS)}"
        ) from exc
    return cls(alpha_level, **kwargs)


# --------------------------------------------------------------------------- #
# the CRUCIBLE layer: alpha monitored as ratings accumulate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AlphaBound:
    """An anytime-valid bound on the disagreement, mapped through to alpha.

    ``alpha_lo``/``alpha_hi`` inherit the plug-in caveat documented at the top of
    this module; ``approximation`` restates it in the object so a bound that
    escapes into a report carries its own qualifier.
    """

    n: int
    d_obs_lo: float
    d_obs_hi: float
    alpha_lo: float
    alpha_hi: float
    d_expected: float
    delta_max: float
    vacuous: bool = False
    failed: bool = False
    approximation: str = (
        "D_e is a plug-in estimate held fixed; the interval is anytime-valid for alpha "
        "only conditional on that estimate being correct"
    )

    @property
    def width(self) -> float:
        return self.alpha_hi - self.alpha_lo

    def clears(self, gate: float) -> bool:
        """True only when the *whole* interval is above the gate."""
        return math.isfinite(self.alpha_lo) and self.alpha_lo >= gate

    def as_dict(self) -> dict[str, Any]:
        def clean(x: float) -> float | None:
            return None if not math.isfinite(x) else float(x)

        return {
            "n": self.n,
            "d_obs": [clean(self.d_obs_lo), clean(self.d_obs_hi)],
            "alpha": [clean(self.alpha_lo), clean(self.alpha_hi)],
            "d_expected": clean(self.d_expected),
            "delta_max": clean(self.delta_max),
            "vacuous": self.vacuous,
            "failed": self.failed,
            "approximation": self.approximation,
        }


class AlphaMonitor:
    """Feed per-unit disagreements in; get an anytime-valid alpha interval out.

    ``d_expected`` and ``delta_max`` are the plug-ins. Supply them from
    ``irr.disagreement_stream`` (which computes both from the full value set), or
    update them as data arrives via :meth:`set_expected`; every value used is
    kept in :attr:`d_expected_history` so the reader can see whether it settled.
    """

    def __init__(
        self,
        cs: ConfidenceSequence,
        *,
        d_expected: float,
        delta_max: float,
        metric: str = "ordinal",
        equal_ratings: bool = True,
    ) -> None:
        if not math.isfinite(d_expected) or d_expected <= 0.0:
            raise ValueError(
                "d_expected must be a positive, finite plug-in estimate of the expected "
                f"disagreement; got {d_expected!r}. With D_e = 0 there is no variation in the "
                "ratings and alpha is undefined, so there is nothing to monitor"
            )
        if not math.isfinite(delta_max) or delta_max <= 0.0:
            raise ValueError(
                f"delta_max must be positive and finite; got {delta_max!r}"
            )
        self.cs = cs
        self.d_expected = float(d_expected)
        self.delta_max = float(delta_max)
        self.metric = str(metric)
        self.equal_ratings = bool(equal_ratings)
        self.d_expected_history: list[float] = [float(d_expected)]
        self.history: list[AlphaBound] = []

    def set_expected(self, d_expected: float) -> None:
        """Refresh the plug-in ``D_e``; the new value is appended to the history."""
        if not math.isfinite(d_expected) or d_expected <= 0.0:
            raise ValueError(f"d_expected must be positive and finite; got {d_expected!r}")
        self.d_expected = float(d_expected)
        self.d_expected_history.append(float(d_expected))

    def update(self, disagreement: float) -> AlphaBound:
        """One unit's normalised disagreement in [0, 1] -> the current bound."""
        x = float(disagreement)
        if not math.isfinite(x) or not 0.0 <= x <= 1.0:
            raise ValueError(
                f"disagreement observations must lie in [0, 1]; got {disagreement!r}. "
                "Use irr.disagreement_stream, which normalises by the largest delta2"
            )
        lo, hi = self.cs.update(x)
        scale = self.delta_max / self.d_expected
        alpha_hi = min(1.0, 1.0 - lo * scale)
        alpha_lo = 1.0 - hi * scale
        bound = AlphaBound(
            n=self.cs.n,
            d_obs_lo=lo,
            d_obs_hi=hi,
            alpha_lo=alpha_lo,
            alpha_hi=alpha_hi,
            d_expected=self.d_expected,
            delta_max=self.delta_max,
            vacuous=(lo <= 0.0 and hi >= 1.0),
            failed=bool(getattr(self.cs, "failed", False)),
        )
        self.history.append(bound)
        return bound

    @property
    def current(self) -> AlphaBound | None:
        return self.history[-1] if self.history else None


@dataclass(frozen=True)
class MonitorTrace:
    """The whole monitoring run for one criterion."""

    criterion_id: str
    metric: str
    cs_kind: str
    bounds: list[AlphaBound] = field(default_factory=list)
    d_expected: float = math.nan
    delta_max: float = math.nan
    equal_ratings: bool = True
    reason: str = ""

    @property
    def final(self) -> AlphaBound | None:
        return self.bounds[-1] if self.bounds else None

    def first_clearing(self, gate: float) -> int | None:
        """1-based index of the first unit at which the interval cleared ``gate``."""
        for i, b in enumerate(self.bounds, start=1):
            if b.clears(gate):
                return i
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "metric": self.metric,
            "cs_kind": self.cs_kind,
            "d_expected": None if not math.isfinite(self.d_expected) else self.d_expected,
            "delta_max": None if not math.isfinite(self.delta_max) else self.delta_max,
            "equal_ratings": self.equal_ratings,
            "reason": self.reason,
            "n": len(self.bounds),
            "final": self.final.as_dict() if self.final else None,
        }


def monitor_alpha(
    matrix: Any,
    metric: Metric | str = "ordinal",
    *,
    cs_kind: Literal["hoeffding", "eb", "betting"] = "eb",
    alpha_level: float = 0.05,
    orientation: str = "raters_x_units",
    criterion_id: str = "",
    unit_ids: Sequence[str] | None = None,
) -> MonitorTrace:
    """Run a confidence sequence over one criterion's disagreement stream."""
    stream = disagreement_stream(
        matrix, metric, orientation=orientation, unit_ids=unit_ids  # type: ignore[arg-type]
    )
    if not stream.observations:
        return MonitorTrace(
            criterion_id=criterion_id,
            metric=str(metric),
            cs_kind=cs_kind,
            reason="no unit was rated by two or more raters; there is no stream to monitor",
        )
    if stream.d_expected <= 0.0 or stream.delta_max <= 0.0:
        return MonitorTrace(
            criterion_id=criterion_id,
            metric=str(metric),
            cs_kind=cs_kind,
            d_expected=stream.d_expected,
            delta_max=stream.delta_max,
            reason=(
                "expected disagreement is zero: every pairable rating is identical, so alpha "
                "is undefined and monitoring it would report a bound on nothing"
            ),
        )
    counts = {u.n_ratings for u in stream.units}
    monitor = AlphaMonitor(
        make_cs(cs_kind, alpha_level),
        d_expected=stream.d_expected,
        delta_max=stream.delta_max,
        metric=str(metric),
        equal_ratings=len(counts) == 1,
    )
    for x in stream.observations:
        monitor.update(x)
    return MonitorTrace(
        criterion_id=criterion_id,
        metric=str(metric),
        cs_kind=cs_kind,
        bounds=list(monitor.history),
        d_expected=stream.d_expected,
        delta_max=stream.delta_max,
        equal_ratings=monitor.equal_ratings,
        reason=(
            ""
            if monitor.equal_ratings
            else (
                "units carry different numbers of ratings, so the confidence sequence bounds the "
                "unweighted mean disagreement while D_o is the pairable-value weighted mean; the "
                "two differ and the alpha interval inherits that discrepancy"
            )
        ),
    )


__all__ = [
    "ConfidenceSequence",
    "HoeffdingCS",
    "EmpiricalBernsteinCS",
    "BettingCS",
    "CS_KINDS",
    "make_cs",
    "AlphaBound",
    "AlphaMonitor",
    "MonitorTrace",
    "monitor_alpha",
]
