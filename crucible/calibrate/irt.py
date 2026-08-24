"""2PL item-response theory over the task bank, by MAP.

A bank of tasks is an exam and the models we calibrate against are the
examinees. The two-parameter logistic model says respondent ``j`` answers item
``i`` correctly with probability::

    P(correct | theta_j, a_i, b_i) = sigmoid( a_i * (theta_j - b_i) )

``b_i`` is difficulty (the ability at which the item is a coin flip) and ``a_i``
is **discrimination** (how sharply the item separates ability just above ``b_i``
from ability just below). Pass rate alone cannot tell those apart: an item that
everything fails and an item that only strong models pass can have the same
mean, and only the second one is worth an SME's afternoon. ``discrimination_ranking``
is therefore the headline output of this module, not a footnote.

Why MAP rather than maximum likelihood. A bank of a few dozen tasks scored by a
handful of models is small, and plain 2PL MLE on small data is not identified:
the likelihood is invariant to shifting all ``theta`` and ``b`` by a constant and
to rescaling ``theta``/``a`` inversely, and items answered correctly by everyone
push ``a`` or ``b`` to infinity. Three weakly informative priors fix it:

* ``theta_j ~ Normal(0, 1)`` anchors both the location and the scale;
* ``log a_i ~ Normal(0, 0.7)`` (a lognormal on ``a``) keeps discrimination
  positive and finite;
* ``b_i ~ Normal(0, 2)`` keeps difficulty finite for degenerate items.

Optimisation is L-BFGS-B on the negative log posterior with an analytic
gradient, over ``(log a, b, theta)``. Nothing here calls a model or an oracle;
it consumes a response matrix somebody else measured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ContaminationSignal",
    "ContaminationThresholds",
    "IRTFit",
    "ItemStats",
    "Priors",
    "fit_2pl",
    "flag_contamination",
    "icc",
    "neg_log_posterior",
    "simulate_2pl",
    "unpack_params",
]


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Priors:
    """Weakly informative priors. Present so a caller can say what they used."""

    log_a_mean: float = 0.0
    log_a_sd: float = 0.7
    b_mean: float = 0.0
    b_sd: float = 2.0
    theta_mean: float = 0.0
    theta_sd: float = 1.0

    def __post_init__(self) -> None:
        for name in ("log_a_sd", "b_sd", "theta_sd"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "log_a_mean": self.log_a_mean,
            "log_a_sd": self.log_a_sd,
            "b_mean": self.b_mean,
            "b_sd": self.b_sd,
            "theta_mean": self.theta_mean,
            "theta_sd": self.theta_sd,
        }


def icc(a: float, b: float, theta: np.ndarray | Sequence[float] | float) -> np.ndarray:
    """Item characteristic curve ``sigmoid(a * (theta - b))``."""
    t = np.asarray(theta, dtype=float)
    return _sigmoid(float(a) * (t - float(b)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


@dataclass(frozen=True)
class ItemStats:
    """One task's fitted parameters and what they imply."""

    item_id: str
    index: int
    discrimination: float
    difficulty: float
    n_observations: int
    n_trials: int
    n_correct: int

    @property
    def observed_pass_rate(self) -> float:
        return self.n_correct / self.n_trials if self.n_trials else float("nan")

    @property
    def peak_information(self) -> float:
        """Fisher information at ``theta == b``: ``a**2 / 4``.

        The single number that says how much this task tells you about a model
        it is well matched to. Ranking by it is ranking by usefulness.
        """
        return (self.discrimination**2) / 4.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "index": self.index,
            "discrimination": self.discrimination,
            "difficulty": self.difficulty,
            "peak_information": self.peak_information,
            "observed_pass_rate": self.observed_pass_rate,
            "n_observations": self.n_observations,
            "n_trials": self.n_trials,
            "n_correct": self.n_correct,
        }


@dataclass(frozen=True)
class IRTFit:
    """Result of a 2PL MAP fit."""

    item_ids: tuple[str, ...]
    respondent_ids: tuple[str, ...]
    a: np.ndarray
    b: np.ndarray
    theta: np.ndarray
    neg_log_posterior: float
    converged: bool
    message: str
    n_observations: int
    priors: Priors = field(default_factory=Priors)
    _stats: tuple[ItemStats, ...] = field(default=(), repr=False)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_respondents(self) -> int:
        return len(self.respondent_ids)

    def item(self, item_id: str) -> ItemStats:
        for stat in self._stats:
            if stat.item_id == item_id:
                return stat
        raise KeyError(f"no such item {item_id!r}; known items: {list(self.item_ids)}")

    def items(self) -> tuple[ItemStats, ...]:
        return self._stats

    def discrimination_ranking(self) -> list[ItemStats]:
        """Items best-separating first.

        This is the quantity that decides which tasks earn their keep: one task
        that cleanly splits a strong model from a weak one is worth ten that
        both pass.
        """
        return sorted(self._stats, key=lambda s: (-s.discrimination, s.item_id))

    def item_response_curve(
        self, item_id: str, thetas: Sequence[float] | np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(thetas, P(correct))`` for one item, for plotting or inspection."""
        stat = self.item(item_id)
        grid = np.linspace(-4.0, 4.0, 81) if thetas is None else np.asarray(thetas, dtype=float)
        return grid, icc(stat.discrimination, stat.difficulty, grid)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "n_respondents": self.n_respondents,
            "n_observations": self.n_observations,
            "converged": self.converged,
            "message": self.message,
            "neg_log_posterior": self.neg_log_posterior,
            "priors": self.priors.as_dict(),
            "respondent_ids": list(self.respondent_ids),
            "ability": {rid: float(t) for rid, t in zip(self.respondent_ids, self.theta)},
            "items": [s.as_dict() for s in self.discrimination_ranking()],
        }


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #


def _as_matrix(values: Any, shape: tuple[int, int] | None, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        if shape is None:
            raise ValueError(f"{name} cannot be a scalar without a reference shape")
        arr = np.full(shape, float(arr))
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (respondents x items); got shape {arr.shape}")
    return arr


def unpack_params(
    x: np.ndarray, n_items: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the flat optimiser vector into ``(log a, b, theta)``."""
    return x[:n_items], x[n_items : 2 * n_items], x[2 * n_items :]


def neg_log_posterior(
    x: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    n_items: int,
    priors: Priors,
) -> tuple[float, np.ndarray]:
    """Negative log posterior and its analytic gradient at ``x``.

    ``x`` is ``concat(log a, b, theta)``. Missing cells are encoded as
    ``trials == successes == 0`` and contribute nothing to either term.

    The derivation, so the gradient can be checked by eye as well as by finite
    differences: with ``z = a(theta - b)`` and ``p = sigmoid(z)``, the
    log-likelihood derivative with respect to ``z`` is ``s - n*p``. Then
    ``dz/dtheta = a``, ``dz/db = -a`` and ``dz/d(log a) = z``.
    """
    alpha, b, theta = unpack_params(np.asarray(x, dtype=float), n_items)
    a = np.exp(alpha)
    z = a[None, :] * (theta[:, None] - b[None, :])

    # -LL through softplus so a large |z| cannot overflow.
    neg_ll = float(np.sum(successes * _softplus(-z) + (trials - successes) * _softplus(z)))

    p = _sigmoid(z)
    g = successes - trials * p  # d(LL)/dz; zero on missing cells

    d_alpha = -(g * z).sum(axis=0)
    d_b = a * g.sum(axis=0)
    d_theta = -(g * a[None, :]).sum(axis=1)

    pa = (alpha - priors.log_a_mean) / priors.log_a_sd
    pb = (b - priors.b_mean) / priors.b_sd
    pt = (theta - priors.theta_mean) / priors.theta_sd
    neg_log_prior = 0.5 * float(np.dot(pa, pa) + np.dot(pb, pb) + np.dot(pt, pt))

    d_alpha = d_alpha + pa / priors.log_a_sd
    d_b = d_b + pb / priors.b_sd
    d_theta = d_theta + pt / priors.theta_sd

    return neg_ll + neg_log_prior, np.concatenate([d_alpha, d_b, d_theta])


def _standardise_ability_scale(
    a: np.ndarray, b: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put the solution on a mean-0, sd-1 ability scale without changing any fit.

    2PL is invariant under ``theta -> (theta - m)/s``, ``b -> (b - m)/s``,
    ``a -> a*s``: every ``z = a(theta - b)``, and therefore every fitted
    probability, is unchanged. The transform is applied because a MAP point
    estimate of ``theta`` is shrunk toward the prior mean (exactly as a ridge
    estimate is), and the optimiser compensates by inflating ``a`` - so the raw
    optimum ranks discrimination correctly but reports it on a scale that
    depends on how many trials happened to be run. Re-expressing the same
    solution with the fitted abilities standardised restores the conventional
    IRT scale and makes ``a`` comparable across banks.

    With fewer than two respondents, or a degenerate ability spread, only the
    location is fixed; there is nothing to estimate a scale from.
    """
    m = float(theta.mean())
    s = float(theta.std())
    if theta.size < 2 or not np.isfinite(s) or s < 1e-6:
        return a, b - m, theta - m
    return a * s, (b - m) / s, (theta - m) / s


def fit_2pl(
    successes: Any,
    trials: Any = 1,
    *,
    item_ids: Sequence[str] | None = None,
    respondent_ids: Sequence[str] | None = None,
    priors: Priors | None = None,
    max_iter: int = 1000,
) -> IRTFit:
    """MAP fit of a 2PL model to a ``(respondents x items)`` response matrix.

    ``successes[j, i]`` is the number of correct responses respondent ``j`` gave
    to item ``i`` out of ``trials[j, i]`` attempts (``trials`` may be a scalar).
    A cell with zero trials, or a NaN, is **missing** and contributes nothing -
    it is not imputed and not counted as a failure.
    """
    prior = priors or Priors()

    s = _as_matrix(successes, None, "successes")
    n = _as_matrix(trials, s.shape, "trials")
    if n.shape != s.shape:
        raise ValueError(f"successes {s.shape} and trials {n.shape} must have the same shape")

    observed = np.isfinite(s) & np.isfinite(n) & (n > 0)
    s = np.where(observed, np.nan_to_num(s, nan=0.0), 0.0)
    n = np.where(observed, np.nan_to_num(n, nan=0.0), 0.0)
    if np.any(s < 0) or np.any(s > n):
        raise ValueError("every observed cell must satisfy 0 <= successes <= trials")

    n_resp, n_items = s.shape
    if n_items < 1 or n_resp < 1:
        raise ValueError(f"need at least one item and one respondent; got {s.shape}")
    n_obs = int(observed.sum())
    if n_obs == 0:
        raise ValueError("the response matrix is entirely missing; nothing to fit")

    iids = tuple(item_ids) if item_ids is not None else tuple(f"item{i}" for i in range(n_items))
    rids = (
        tuple(respondent_ids)
        if respondent_ids is not None
        else tuple(f"model{j}" for j in range(n_resp))
    )
    if len(iids) != n_items:
        raise ValueError(f"item_ids has {len(iids)} entries for {n_items} items")
    if len(rids) != n_resp:
        raise ValueError(f"respondent_ids has {len(rids)} entries for {n_resp} respondents")

    # --- initial values from the marginals ---------------------------------
    item_trials = n.sum(axis=0)
    item_correct = s.sum(axis=0)
    resp_trials = n.sum(axis=1)
    resp_correct = s.sum(axis=1)
    item_p = np.clip(
        np.divide(item_correct, item_trials, out=np.full(n_items, 0.5), where=item_trials > 0),
        0.05,
        0.95,
    )
    resp_p = np.clip(
        np.divide(resp_correct, resp_trials, out=np.full(n_resp, 0.5), where=resp_trials > 0),
        0.05,
        0.95,
    )
    alpha0 = np.zeros(n_items)  # a = 1
    b0 = np.clip(-np.log(item_p / (1.0 - item_p)), -3.0, 3.0)
    theta0 = np.clip(np.log(resp_p / (1.0 - resp_p)), -3.0, 3.0)
    x0 = np.concatenate([alpha0, b0, theta0])

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        return neg_log_posterior(x, s, n, n_items, prior)

    from scipy.optimize import minimize

    bounds = (
        [(-4.0, 4.0)] * n_items  # log a  -> a in [0.018, 54]
        + [(-8.0, 8.0)] * n_items  # b
        + [(-8.0, 8.0)] * n_resp  # theta
    )
    result = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-8},
    )
    alpha, b, theta = unpack_params(np.asarray(result.x, dtype=float), n_items)
    a = np.exp(alpha)

    if not bool(result.success):
        logger.warning("2PL MAP fit did not converge: %s", result.message)

    a, b, theta = _standardise_ability_scale(a, b, theta)

    stats = tuple(
        ItemStats(
            item_id=iids[i],
            index=i,
            discrimination=float(a[i]),
            difficulty=float(b[i]),
            n_observations=int(observed[:, i].sum()),
            n_trials=int(item_trials[i]),
            n_correct=int(round(float(item_correct[i]))),
        )
        for i in range(n_items)
    )

    return IRTFit(
        item_ids=iids,
        respondent_ids=rids,
        a=a,
        b=b,
        theta=theta,
        neg_log_posterior=float(result.fun),
        converged=bool(result.success),
        message=str(result.message),
        n_observations=n_obs,
        priors=prior,
        _stats=stats,
    )


def simulate_2pl(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    theta: Sequence[float] | np.ndarray,
    trials: int = 1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a synthetic ``(successes, trials)`` matrix from known 2PL parameters.

    Used by the tests to check that ``fit_2pl`` recovers what generated the
    data; exposed because a recovery check is the only honest way to trust a
    fitted discrimination ranking on a new bank.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    t_arr = np.asarray(theta, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must have the same length")
    if int(trials) < 1:
        raise ValueError("trials must be >= 1")
    rng = np.random.default_rng(seed)
    p = _sigmoid(a_arr[None, :] * (t_arr[:, None] - b_arr[None, :]))
    successes = rng.binomial(int(trials), p).astype(float)
    return successes, np.full(p.shape, float(int(trials)))


# --------------------------------------------------------------------------- #
# contamination
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ContaminationThresholds:
    """Policy for the contamination heuristic. Inputs, never inferred."""

    pass_at_1_min: float = 0.95
    min_mean_tokens: float = 600.0
    max_backtracks: float = 1.0
    max_entropy: float = 0.35
    min_signals: int = 2

    def as_dict(self) -> dict[str, float]:
        return {
            "pass_at_1_min": self.pass_at_1_min,
            "min_mean_tokens": self.min_mean_tokens,
            "max_backtracks": self.max_backtracks,
            "max_entropy": self.max_entropy,
            "min_signals": float(self.min_signals),
        }


@dataclass(frozen=True)
class ContaminationSignal:
    """Verdict of the contamination heuristic, including "cannot tell"."""

    task_id: str
    flagged: bool
    assessed: bool
    score: float
    pass_at_1: float
    signals: tuple[str, ...]
    missing_stats: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "flagged": self.flagged,
            "assessed": self.assessed,
            "score": self.score,
            "pass_at_1": self.pass_at_1,
            "signals": list(self.signals),
            "missing_stats": list(self.missing_stats),
            "reason": self.reason,
        }


#: Accepted spellings for each trace statistic. Caller vocabularies differ; the
#: heuristic should not silently treat a differently-named field as absent.
_TRACE_ALIASES: dict[str, tuple[str, ...]] = {
    "mean_tokens": ("mean_tokens", "tokens", "mean_completion_tokens", "token_count"),
    "backtracks": ("backtracks", "mean_backtracks", "backtrack_count", "revisions"),
    "entropy": ("entropy", "entropy_proxy", "mean_entropy", "token_entropy"),
}


def _read_stat(stats: Mapping[str, Any], key: str) -> float | None:
    for alias in _TRACE_ALIASES[key]:
        if alias in stats:
            value = stats[alias]
            try:
                as_float = float(value)
            except (TypeError, ValueError):
                logger.debug("trace statistic %s=%r is not numeric; treating as absent", alias, value)
                return None
            if as_float != as_float:  # NaN
                return None
            return as_float
    return None


def flag_contamination(
    task_id: str,
    pass_at_1: float,
    trace_stats: Mapping[str, Any] | None = None,
    thresholds: ContaminationThresholds | None = None,
) -> ContaminationSignal:
    """Contamination = saturated **and** the trace looks recalled, not solved.

    Both halves are required. A task everything passes may simply be easy; what
    distinguishes memorisation is that the model produces a long, confident,
    non-exploratory trace - many tokens, few backtracks, low entropy - instead
    of the shorter hesitant trace an easy-but-novel problem produces.

    The heuristic refuses to decide when it cannot. If pass@1 is saturated but
    fewer than ``min_signals`` trace statistics were supplied, the result is
    ``assessed=False, flagged=False`` and the reason says which statistics were
    missing. Absent evidence is reported as absent, never as a clean bill.
    """
    th = thresholds or ContaminationThresholds()
    p1 = float(pass_at_1)
    if not 0.0 <= p1 <= 1.0:
        raise ValueError(f"pass_at_1 must be in [0, 1]: {p1}")
    stats: Mapping[str, Any] = trace_stats or {}

    values = {key: _read_stat(stats, key) for key in _TRACE_ALIASES}
    missing = tuple(sorted(k for k, v in values.items() if v is None))
    available = tuple(sorted(k for k, v in values.items() if v is not None))

    if p1 < th.pass_at_1_min:
        return ContaminationSignal(
            task_id=task_id,
            flagged=False,
            assessed=True,
            score=0.0,
            pass_at_1=p1,
            signals=(),
            missing_stats=missing,
            reason=(
                f"pass@1={p1:.3f} is below the saturation threshold "
                f"{th.pass_at_1_min:.2f}; contamination requires saturation"
            ),
        )

    matched: list[str] = []
    if values["mean_tokens"] is not None and values["mean_tokens"] >= th.min_mean_tokens:
        matched.append(f"long trace (mean_tokens={values['mean_tokens']:.0f} >= {th.min_mean_tokens:.0f})")
    if values["backtracks"] is not None and values["backtracks"] <= th.max_backtracks:
        matched.append(f"non-exploratory (backtracks={values['backtracks']:.2f} <= {th.max_backtracks:.2f})")
    if values["entropy"] is not None and values["entropy"] <= th.max_entropy:
        matched.append(f"confident (entropy={values['entropy']:.3f} <= {th.max_entropy:.2f})")

    if len(available) < th.min_signals:
        return ContaminationSignal(
            task_id=task_id,
            flagged=False,
            assessed=False,
            score=0.0,
            pass_at_1=p1,
            signals=tuple(matched),
            missing_stats=missing,
            reason=(
                f"pass@1={p1:.3f} is saturated, but only {len(available)} of "
                f"{len(_TRACE_ALIASES)} trace statistics were supplied "
                f"(missing: {', '.join(missing) or 'none'}); "
                f"{th.min_signals} are needed to reach a verdict, so contamination is UNASSESSED"
            ),
        )

    score = len(matched) / len(available)
    flagged = len(matched) >= th.min_signals
    reason = (
        f"pass@1={p1:.3f} >= {th.pass_at_1_min:.2f} and {len(matched)}/{len(available)} "
        f"trace signals matched ({'; '.join(matched) or 'none'})"
    )
    if not flagged:
        reason += f"; fewer than {th.min_signals} signals, so not flagged"
    return ContaminationSignal(
        task_id=task_id,
        flagged=flagged,
        assessed=True,
        score=score,
        pass_at_1=p1,
        signals=tuple(matched),
        missing_stats=missing,
        reason=reason,
    )
