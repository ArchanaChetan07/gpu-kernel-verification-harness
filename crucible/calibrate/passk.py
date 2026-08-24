"""Unbiased pass@k, and the uncertainty that comes with it.

Two commitments live in this module.

**The estimator is the Chen et al. one, not the naive one.** With ``n`` samples
of which ``c`` are correct::

    pass@k = 1 - C(n - c, k) / C(n, k)

and never ``(c/n)**k``. The naive form is the probability that *k independent
draws from a Bernoulli(c/n)* all fail, which is a different quantity: it ignores
that the ``n`` samples we actually drew are themselves a finite sample, and it is
biased low for small ``n``. The ratio of binomials is the probability that a
uniformly random ``k``-subset of the ``n`` draws we really made contains no
correct one, which is exactly what "pass@k from n samples" means. It is computed
in log space through ``lgamma`` so that ``n`` in the thousands does not overflow
a 64-bit integer factorial.

**A point estimate from eight samples is not a measurement.** ``estimate``
returns a ``PassKEstimate`` that carries a Wilson (default) or Jeffreys interval
on pass@1 alongside the point value, because 3/8 = 0.375 has a 95% Wilson
interval of roughly [0.14, 0.69] - it is compatible with almost the whole gold
band and with the frontier band below it. The router consumes that interval; see
``crucible.calibrate.router``.

Edge cases are decided, not defaulted:

* ``c == 0`` -> 0.0 exactly (no ``k``-subset can contain a correct sample).
* ``c == n`` -> 1.0 exactly (every ``k``-subset does).
* ``k > n`` -> **raises**. There is no unbiased estimator of pass@k from fewer
  than ``k`` samples, and quietly returning something would be a fabricated
  number. Callers that want the truncated answer pass ``clamp_k=True`` and get
  ``pass@min(k, n)`` plus a note saying so.
* ``n == 0`` -> raises. Nothing was measured.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Literal

logger = logging.getLogger(__name__)

CIMethod = Literal["wilson", "jeffreys"]

#: Default two-sided coverage for the interval on pass@1.
DEFAULT_CI_LEVEL = 0.95

__all__ = [
    "CIMethod",
    "DEFAULT_CI_LEVEL",
    "PassKEstimate",
    "confidence_interval",
    "estimate",
    "jeffreys_interval",
    "log_comb",
    "pass_at_1",
    "pass_at_k",
    "wilson_interval",
]


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #


def log_comb(n: int, k: int) -> float:
    """``log C(n, k)``, or ``-inf`` when the coefficient is zero.

    Uses ``lgamma`` rather than ``math.comb`` so the caller can subtract two of
    these without ever materialising a large integer.
    """
    if n < 0 or k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _validate(n: int, c: int, k: int) -> tuple[int, int, int]:
    n_i, c_i, k_i = int(n), int(c), int(k)
    if n_i < 1:
        raise ValueError(f"n must be >= 1 (nothing was sampled): n={n_i}")
    if c_i < 0 or c_i > n_i:
        raise ValueError(f"c must satisfy 0 <= c <= n: c={c_i}, n={n_i}")
    if k_i < 0:
        raise ValueError(f"k must be >= 0: k={k_i}")
    return n_i, c_i, k_i


def pass_at_k(n: int, c: int, k: int, *, clamp_k: bool = False) -> float:
    """Unbiased pass@k = ``1 - C(n-c, k) / C(n, k)``, computed in log space.

    ``clamp_k=False`` (the default) raises when ``k > n``. ``clamp_k=True``
    evaluates ``pass@min(k, n)`` instead; ``estimate`` records that substitution
    in its notes so it never disappears from the record.
    """
    n_i, c_i, k_i = _validate(n, c, k)
    if k_i > n_i:
        if not clamp_k:
            raise ValueError(
                f"pass@{k_i} is not estimable from {n_i} samples; "
                f"draw at least k samples or pass clamp_k=True to report pass@{n_i}"
            )
        k_i = n_i

    if c_i == 0:
        # C(n, k) / C(n, k) == 1 exactly; no floating point needed.
        return 0.0
    if n_i - c_i < k_i:
        # C(n-c, k) == 0: every k-subset must contain a correct sample.
        return 1.0

    ratio = math.exp(log_comb(n_i - c_i, k_i) - log_comb(n_i, k_i))
    return float(min(1.0, max(0.0, 1.0 - ratio)))


def pass_at_1(n: int, c: int) -> float:
    """``c / n``. Identical to ``pass_at_k(n, c, 1)``; kept for readability."""
    n_i, c_i, _ = _validate(n, c, 1)
    return c_i / n_i


# --------------------------------------------------------------------------- #
# intervals on pass@1
# --------------------------------------------------------------------------- #


def _z_for(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(f"ci level must be in (0, 1): {level}")
    return NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)


def wilson_interval(c: int, n: int, level: float = DEFAULT_CI_LEVEL) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval at the sample sizes calibration actually
    uses: Wald gives a zero-width interval at ``c == 0`` and ``c == n``, which
    would tell the router that 0/8 pins pass@1 at exactly zero.
    """
    n_i, c_i, _ = _validate(n, c, 1)
    z = _z_for(level)
    z2 = z * z
    denom = n_i + z2
    centre = (c_i + z2 / 2.0) / denom
    half = (z / denom) * math.sqrt(c_i * (n_i - c_i) / n_i + z2 / 4.0)
    return (max(0.0, centre - half), min(1.0, centre + half))


def jeffreys_interval(c: int, n: int, level: float = DEFAULT_CI_LEVEL) -> tuple[float, float]:
    """Equal-tailed Beta(c+1/2, n-c+1/2) credible interval.

    The standard degenerate-endpoint fix is applied: the lower bound is 0 when
    ``c == 0`` and the upper bound is 1 when ``c == n``.
    """
    n_i, c_i, _ = _validate(n, c, 1)
    if not 0.0 < level < 1.0:
        raise ValueError(f"ci level must be in (0, 1): {level}")
    try:
        from scipy.stats import beta as _beta
    except ImportError as exc:  # pragma: no cover - scipy is a hard dependency
        raise ImportError(
            "the Jeffreys interval needs scipy; use method='wilson' instead"
        ) from exc

    alpha = 1.0 - level
    lo = 0.0 if c_i == 0 else float(_beta.ppf(alpha / 2.0, c_i + 0.5, n_i - c_i + 0.5))
    hi = 1.0 if c_i == n_i else float(_beta.ppf(1.0 - alpha / 2.0, c_i + 0.5, n_i - c_i + 0.5))
    return (max(0.0, lo), min(1.0, hi))


def confidence_interval(
    c: int,
    n: int,
    level: float = DEFAULT_CI_LEVEL,
    method: CIMethod = "wilson",
) -> tuple[float, float]:
    if method == "wilson":
        return wilson_interval(c, n, level)
    if method == "jeffreys":
        return jeffreys_interval(c, n, level)
    raise ValueError(f"unknown interval method {method!r}; expected 'wilson' or 'jeffreys'")


# --------------------------------------------------------------------------- #
# the record a routing decision is made from
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PassKEstimate:
    """Everything a routing decision needs, including what it could not measure."""

    n_samples: int
    n_correct: int
    k: int
    effective_k: int
    pass_at_1: float
    pass_at_k: float
    ci_low: float
    ci_high: float
    ci_level: float
    ci_method: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ci(self) -> tuple[float, float]:
        return (self.ci_low, self.ci_high)

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_correct": self.n_correct,
            "k": self.k,
            "effective_k": self.effective_k,
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "ci_method": self.ci_method,
            "ci_width": self.ci_width,
            "notes": list(self.notes),
        }


def estimate(
    n: int,
    c: int,
    k: int = 8,
    *,
    ci_level: float = DEFAULT_CI_LEVEL,
    ci_method: CIMethod = "wilson",
    clamp_k: bool = True,
) -> PassKEstimate:
    """pass@1, pass@k and an interval on pass@1, with every caveat recorded."""
    n_i, c_i, k_i = _validate(n, c, k)
    notes: list[str] = []

    effective_k = k_i
    if k_i > n_i:
        if not clamp_k:
            raise ValueError(
                f"pass@{k_i} is not estimable from {n_i} samples; "
                "draw more samples or pass clamp_k=True"
            )
        effective_k = n_i
        notes.append(
            f"k={k_i} exceeds n={n_i}; reported pass@k is pass@{n_i} "
            "(pass@k is not estimable from fewer than k samples)"
        )

    p1 = pass_at_1(n_i, c_i)
    pk = pass_at_k(n_i, c_i, effective_k)
    lo, hi = confidence_interval(c_i, n_i, ci_level, ci_method)

    if n_i < 8:
        notes.append(f"n={n_i} is small; the interval on pass@1 is correspondingly wide")
    if hi - lo >= 0.5:
        notes.append(
            f"{ci_method} {ci_level:.0%} interval spans {hi - lo:.2f} of the unit range; "
            "the point estimate carries little information on its own"
        )

    return PassKEstimate(
        n_samples=n_i,
        n_correct=c_i,
        k=k_i,
        effective_k=effective_k,
        pass_at_1=p1,
        pass_at_k=pk,
        ci_low=lo,
        ci_high=hi,
        ci_level=ci_level,
        ci_method=ci_method,
        notes=tuple(notes),
    )
