"""The error-budget model. There is no hardcoded 1e-3 anywhere in CRUCIBLE.

A numeric verdict is only as trustworthy as the threshold it was measured
against, so this module refuses to let a threshold be anonymous. Every
``Tolerance`` carries a ``formula`` string that shows its own derivation::

    bf16 eps=3.91e-03, K=1023, stochastic growth sqrt(K)=31.98, safety=4
        -> rel=5.00e-01; scale=1.000e+00 (unit: no reference magnitude observed)
        -> abs=5.00e-01

Three commitments shape the API:

1. **Derived, not chosen.** ``derive`` takes the dtype (which fixes the unit
   roundoff), the accumulation depth K (how many roundings compound), and a
   growth model. A caller that genuinely wants a fixed number must call
   ``fixed()``, and the resulting Tolerance says ``FIXED`` in its formula and
   ``derived=False`` in its evidence. A fixed tolerance is a decision, and a
   decision must be visible in the record.

2. **Relative until the data says otherwise.** ``derive`` alone knows nothing
   about magnitudes, so its absolute budget is the relative one at unit scale.
   ``from_tensor_scale`` replaces it using the observed magnitude of the
   *reference* output. The candidate's magnitude is never used: a candidate that
   returns huge numbers must not thereby buy itself a larger budget.

3. **NaN is never "within tolerance".** ``compare`` treats non-finite values
   structurally, not numerically. A NaN in the candidate where the reference is
   finite fails no matter how loose the budget is, and no arithmetic is
   performed that could quietly turn it into a passing zero.

This module imports numpy at module scope and torch only lazily, so it stays
importable on a machine with no CUDA and no torch.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

#: Unit roundoff (2**-(mantissa_bits+1)) per storage format.
#:
#: These are the *storage* epsilons. The accumulation format is modelled
#: separately by ``accum_depth``: a bf16 matmul with an fp32 accumulator still
#: rounds its inputs and its result at bf16, which is what dominates.
EPS: dict[str, float] = {
    "fp64": 2.0**-53,
    "fp32": 2.0**-24,
    "tf32": 2.0**-11,
    "bf16": 2.0**-8,
    "fp16": 2.0**-11,
    "fp8": 2.0**-4,
}

#: Spellings a seed, a shape kwarg or a torch dtype repr might use.
DTYPE_ALIASES: dict[str, str] = {
    "double": "fp64",
    "float64": "fp64",
    "f64": "fp64",
    "float": "fp32",
    "float32": "fp32",
    "f32": "fp32",
    "single": "fp32",
    "tensorfloat32": "tf32",
    "bfloat16": "bf16",
    "half": "fp16",
    "float16": "fp16",
    "f16": "fp16",
    "float8": "fp8",
    "float8_e4m3fn": "fp8",
    "float8_e4m3fnuz": "fp8",
    "float8_e5m2": "fp8",
    "float8_e5m2fnuz": "fp8",
}

MODES: tuple[str, ...] = ("stochastic", "deterministic")

__all__ = [
    "EPS",
    "DTYPE_ALIASES",
    "MODES",
    "Tolerance",
    "CompareResult",
    "normalize_dtype",
    "eps_for",
    "growth_factor",
    "derive",
    "fixed",
    "from_tensor_scale",
    "magnitude_of",
    "compare",
    "to_numpy",
]


# --------------------------------------------------------------------------- #
# dtype handling
# --------------------------------------------------------------------------- #


def normalize_dtype(dtype: Any) -> str:
    """Map any spelling of a float format onto a key of :data:`EPS`.

    Raises ``KeyError`` on an unknown format. Guessing here would silently pick
    an error budget for a dtype nobody modelled, which is exactly the kind of
    invented number this module exists to prevent.
    """
    name = str(getattr(dtype, "name", dtype)).strip().lower()
    if name.startswith("torch."):
        name = name[len("torch.") :]
    if name.startswith("numpy."):
        name = name[len("numpy.") :]
    if name in EPS:
        return name
    if name in DTYPE_ALIASES:
        return DTYPE_ALIASES[name]
    raise KeyError(
        f"no error budget is modelled for dtype {dtype!r} (normalized {name!r}); "
        f"known: {sorted(EPS)} plus aliases {sorted(DTYPE_ALIASES)}"
    )


def eps_for(dtype: Any) -> float:
    """Unit roundoff for ``dtype``."""
    return EPS[normalize_dtype(dtype)]


def growth_factor(accum_depth: int, mode: str = "stochastic") -> float:
    """How rounding error compounds over ``accum_depth`` dependent roundings.

    ``stochastic`` models errors as a random walk (sqrt(K)); ``deterministic``
    is the worst case where every rounding leans the same way (K).
    """
    if mode not in MODES:
        raise ValueError(f"unknown tolerance mode {mode!r}; expected one of {MODES}")
    k = max(int(accum_depth), 1)
    return math.sqrt(k) if mode == "stochastic" else float(k)


# --------------------------------------------------------------------------- #
# Tolerance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Tolerance:
    """A numeric budget that can explain where it came from.

    ``rel`` and ``abs`` are combined by :func:`compare` the standard way::

        |candidate - reference| <= abs + rel * |reference|

    so ``abs`` is what covers reference entries that are exactly zero.
    """

    rel: float
    abs: float
    dtype: str
    accum_depth: int
    mode: str
    formula: str
    safety: float = 0.0
    eps: float = 0.0
    growth: float = 1.0
    scale: float = 1.0
    scale_source: str = "unit: no reference magnitude observed"
    derived: bool = True
    note: str = ""

    def __str__(self) -> str:
        return self.formula

    def with_scale(self, scale: float, source: str = "max|reference|") -> Tolerance:
        """Same relative budget, absolute budget re-derived at ``scale``.

        A non-finite or non-positive scale falls back to unit scale rather than
        producing a zero (infinitely strict) or NaN budget, and says so in the
        formula.
        """
        s = float(scale)
        src = source
        if not math.isfinite(s) or s <= 0.0:
            src = f"unit fallback: observed scale {scale!r} is not usable"
            s = 1.0
        head = self.formula.split(";")[0].rstrip()
        formula = f"{head}; scale={s:.3e} ({src}) -> abs={self.rel * s:.2e}"
        return Tolerance(
            rel=self.rel,
            abs=self.rel * s,
            dtype=self.dtype,
            accum_depth=self.accum_depth,
            mode=self.mode,
            formula=formula,
            safety=self.safety,
            eps=self.eps,
            growth=self.growth,
            scale=s,
            scale_source=src,
            derived=self.derived,
            note=self.note,
        )

    def allows(self, abs_err: float, ref_magnitude: float = 0.0) -> bool:
        """Is ``abs_err`` acceptable against a reference entry of that magnitude?"""
        return float(abs_err) <= self.abs + self.rel * float(ref_magnitude)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rel": self.rel,
            "abs": self.abs,
            "dtype": self.dtype,
            "accum_depth": self.accum_depth,
            "mode": self.mode,
            "formula": self.formula,
            "safety": self.safety,
            "eps": self.eps,
            "growth": self.growth,
            "scale": self.scale,
            "scale_source": self.scale_source,
            "derived": self.derived,
            "note": self.note,
        }


def derive(
    dtype: Any,
    accum_depth: int,
    mode: str = "stochastic",
    safety: float = 4.0,
    scale: float | None = None,
) -> Tolerance:
    """Derive an error budget from the dtype, the accumulation depth and a model.

    ``rel = safety * growth(accum_depth) * EPS[dtype]``, with growth ``sqrt(K)``
    for the stochastic model and ``K`` for the deterministic worst case.

    ``scale`` is optional; when omitted the absolute budget is the relative one
    at unit scale and :func:`from_tensor_scale` is expected to refine it once a
    reference output exists.
    """
    key = normalize_dtype(dtype)
    eps = EPS[key]
    if not math.isfinite(safety) or safety <= 0.0:
        raise ValueError(f"safety factor must be a positive finite number, got {safety!r}")
    k = max(int(accum_depth), 1)
    growth = growth_factor(k, mode)
    rel = float(safety) * growth * eps
    gexpr = "sqrt(K)" if mode == "stochastic" else "K"
    gval = f"{growth:.2f}" if mode == "stochastic" else f"{growth:.0f}"
    formula = (
        f"{key} eps={eps:.2e}, K={k}, {mode} growth {gexpr}={gval}, "
        f"safety={float(safety):g} -> rel={rel:.2e}"
    )
    note = ""
    if int(accum_depth) < 1:
        note = f"accum_depth {accum_depth!r} clamped to 1 (a value is rounded at least once)"
    tol = Tolerance(
        rel=rel,
        abs=rel,
        dtype=key,
        accum_depth=k,
        mode=mode,
        formula=f"{formula}; scale=1.000e+00 (unit: no reference magnitude observed) -> abs={rel:.2e}",
        safety=float(safety),
        eps=eps,
        growth=growth,
        scale=1.0,
        derived=True,
        note=note,
    )
    if scale is not None:
        tol = tol.with_scale(scale, source="caller-supplied scale")
    return tol


def fixed(
    rel: float,
    abs: float | None = None,
    dtype: Any = "unspecified",
    reason: str = "",
) -> Tolerance:
    """A caller-declared constant budget, recorded as a decision.

    Nothing about this number is derived, so the formula says ``FIXED`` and
    ``derived`` is False. Downstream evidence therefore distinguishes "the model
    computed 2.7e-06" from "a human typed 1e-3".
    """
    r = float(rel)
    if not math.isfinite(r) or r < 0.0:
        raise ValueError(f"fixed relative tolerance must be finite and non-negative, got {rel!r}")
    a = r if abs is None else float(abs)
    if not math.isfinite(a) or a < 0.0:
        raise ValueError(f"fixed absolute tolerance must be finite and non-negative, got {abs!r}")
    try:
        key = normalize_dtype(dtype)
    except KeyError:
        key = str(dtype)
    why = f" ({reason})" if reason else ""
    formula = (
        f"FIXED by caller: rel={r:.2e}, abs={a:.2e}{why}; "
        f"not derived from dtype or accumulation depth"
    )
    return Tolerance(
        rel=r,
        abs=a,
        dtype=key,
        accum_depth=0,
        mode="fixed",
        formula=formula,
        safety=0.0,
        eps=0.0,
        growth=1.0,
        scale=1.0,
        scale_source="not applicable: absolute budget was supplied, not scaled",
        derived=False,
        note=reason,
    )


def magnitude_of(reference: Any) -> float:
    """Largest finite absolute value in ``reference`` (0.0 if there is none).

    Accepts a scalar, an array/tensor, or any iterable of those. Non-finite
    entries are ignored here on purpose: an inf in the reference must not
    inflate every other entry's budget to infinity.
    """
    best = 0.0
    for arr in _iter_arrays(reference):
        if arr.size == 0:
            continue
        try:
            f = np.asarray(arr, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        finite = np.isfinite(f)
        if not finite.any():
            continue
        m = float(np.abs(f[finite]).max())
        if m > best:
            best = m
    return best


def from_tensor_scale(
    tol: Tolerance,
    reference: Any,
    min_scale: float = 0.0,
    source: str = "max|reference|",
) -> Tolerance:
    """Turn a relative budget into an absolute one using the reference magnitude.

    ``reference`` may be the observed magnitude itself (a float) or the actual
    reference output(s), in which case the magnitude is measured here. Passing
    the candidate's output instead would let a wrong-but-large answer widen its
    own budget, so callers must pass the reference.
    """
    if not tol.derived:
        # A fixed budget is a decision about absolute numbers. Rescaling it here
        # would quietly turn the caller's stated constant into a different one.
        return tol
    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        scale = float(reference)
        src = source if source != "max|reference|" else "caller-supplied scale"
    else:
        scale = magnitude_of(reference)
        src = source
    floor = float(min_scale)
    if math.isfinite(floor) and floor > 0.0 and scale < floor:
        scale = floor
        src = f"{src}, floored at min_scale={floor:.3e}"
    if not math.isfinite(scale) or scale <= 0.0:
        return tol.with_scale(
            1.0,
            source="unit fallback: reference magnitude is zero or non-finite",
        )
    return tol.with_scale(scale, source=src)


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #


def _torch() -> Any:
    mod = sys.modules.get("torch")
    if mod is not None:
        return mod
    try:
        import torch as _t
    except ImportError:
        return None
    return _t


def to_numpy(value: Any) -> np.ndarray:
    """Best-effort conversion to a numpy array without pickling anything.

    bfloat16 is widened to float32 (exactly, it is a strict subset) because
    numpy has no bfloat16.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bool, int, float, complex)):
        return np.asarray(value)
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        return t.contiguous().numpy()
    return np.asarray(value)


def _iter_arrays(value: Any) -> Iterable[np.ndarray]:
    torch = _torch()
    if isinstance(value, np.ndarray) or isinstance(value, (bool, int, float, complex)):
        yield to_numpy(value)
        return
    if torch is not None and isinstance(value, torch.Tensor):
        yield to_numpy(value)
        return
    if isinstance(value, dict):
        for v in value.values():
            yield from _iter_arrays(v)
        return
    if isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_arrays(v)
        return
    yield to_numpy(value)


@dataclass(frozen=True)
class CompareResult:
    """Outcome of one candidate-vs-reference array comparison.

    ``max_abs_err`` and ``max_rel_err`` are measured over positions where *both*
    sides are finite. Positions where they are not are counted in
    ``n_nan_mismatch`` / ``n_inf_mismatch`` and always fail; they are deliberately
    not folded into an error magnitude, because there is no magnitude to report.

    ``max_rel_err`` skips positions where the reference is exactly zero (there is
    no relative error to define there); those positions are counted in
    ``n_zero_ref`` and are still checked against the absolute budget.
    """

    max_abs_err: float
    max_rel_err: float
    n_bad: int
    first_bad_index: tuple[int, ...] | None
    passed: bool
    n_total: int = 0
    n_comparable: int = 0
    n_nan_mismatch: int = 0
    n_inf_mismatch: int = 0
    n_zero_ref: int = 0
    kind: str = "numeric"
    detail: str = ""
    tolerance: Tolerance | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        """Alias for ``passed`` (matches ``crucible.seeds.registry.CompareResult``)."""
        return self.passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "max_abs_err": self.max_abs_err,
            "max_rel_err": self.max_rel_err,
            "n_bad": self.n_bad,
            "first_bad_index": None if self.first_bad_index is None else list(self.first_bad_index),
            "n_total": self.n_total,
            "n_comparable": self.n_comparable,
            "n_nan_mismatch": self.n_nan_mismatch,
            "n_inf_mismatch": self.n_inf_mismatch,
            "n_zero_ref": self.n_zero_ref,
            "kind": self.kind,
            "detail": self.detail,
            "tolerance": None if self.tolerance is None else self.tolerance.as_dict(),
        }


def _first_index(bad: np.ndarray) -> tuple[int, ...] | None:
    flat = np.flatnonzero(bad.ravel())
    if flat.size == 0:
        return None
    if bad.ndim == 0:
        return ()
    return tuple(int(i) for i in np.unravel_index(int(flat[0]), bad.shape))


def compare(a: Any, b: Any, tol: Tolerance) -> CompareResult:
    """Compare candidate ``a`` against reference ``b`` under ``tol``.

    Argument order is (candidate, reference) and it matters: the relative budget
    is scaled by ``|b|``, and the non-finite rules are asymmetric in intent even
    though both directions fail. A NaN produced by the candidate where the
    reference is finite is a failure at any tolerance.
    """
    ca = to_numpy(a)
    cb = to_numpy(b)

    if ca.shape != cb.shape:
        return CompareResult(
            max_abs_err=0.0,
            max_rel_err=0.0,
            n_bad=int(cb.size),
            first_bad_index=None,
            passed=False,
            n_total=int(cb.size),
            kind="shape",
            detail=f"candidate shape {tuple(ca.shape)} != reference shape {tuple(cb.shape)}",
            tolerance=tol,
        )

    n_total = int(cb.size)
    if n_total == 0:
        return CompareResult(
            max_abs_err=0.0,
            max_rel_err=0.0,
            n_bad=0,
            first_bad_index=None,
            passed=True,
            n_total=0,
            n_comparable=0,
            kind="numeric",
            detail="both outputs are empty; nothing to compare",
            tolerance=tol,
        )

    # Non-float outputs (indices, masks, counts) admit no error budget at all.
    if not (
        np.issubdtype(ca.dtype, np.floating)
        or np.issubdtype(ca.dtype, np.complexfloating)
        or np.issubdtype(cb.dtype, np.floating)
        or np.issubdtype(cb.dtype, np.complexfloating)
    ):
        bad = ca != cb
        n_bad = int(bad.sum())
        try:
            diff = np.abs(ca.astype(np.float64) - cb.astype(np.float64))
            max_abs = float(diff.max())
        except (TypeError, ValueError):
            max_abs = 0.0
        return CompareResult(
            max_abs_err=max_abs,
            max_rel_err=0.0,
            n_bad=n_bad,
            first_bad_index=_first_index(bad),
            passed=n_bad == 0,
            n_total=n_total,
            n_comparable=n_total,
            kind="exact",
            detail=(
                "non-floating dtype "
                f"{ca.dtype}/{cb.dtype}: compared exactly, no tolerance applies"
            ),
            tolerance=tol,
        )

    fa = np.asarray(ca, dtype=np.float64)
    fb = np.asarray(cb, dtype=np.float64)

    a_nan = np.isnan(fa)
    b_nan = np.isnan(fb)
    a_inf = np.isinf(fa)
    b_inf = np.isinf(fb)
    both_finite = np.isfinite(fa) & np.isfinite(fb)

    # Non-finite entries only "agree" structurally: NaN with NaN, or an infinity
    # of the same sign. Every other combination is a divergence.
    nonfinite_agree = (a_nan & b_nan) | (a_inf & b_inf & (np.sign(fa) == np.sign(fb)))
    nonfinite_bad = (~both_finite) & (~nonfinite_agree)

    n_nan_mismatch = int(((a_nan & ~b_nan) | (b_nan & ~a_nan)).sum())
    n_inf_mismatch = int((nonfinite_bad & ~((a_nan & ~b_nan) | (b_nan & ~a_nan))).sum())

    abs_err = np.zeros_like(fa)
    np.subtract(fa, fb, out=abs_err, where=both_finite)
    np.abs(abs_err, out=abs_err)
    abs_err[~both_finite] = 0.0

    budget = tol.abs + tol.rel * np.abs(fb)
    exceeds = both_finite & (abs_err > budget)

    bad = exceeds | nonfinite_bad
    n_bad = int(bad.sum())

    n_comparable = int(both_finite.sum())
    max_abs = float(abs_err[both_finite].max()) if n_comparable else 0.0

    ref_abs = np.abs(fb)
    rel_mask = both_finite & (ref_abs > 0.0)
    n_zero_ref = int((both_finite & (ref_abs == 0.0)).sum())
    if rel_mask.any():
        max_rel = float((abs_err[rel_mask] / ref_abs[rel_mask]).max())
    else:
        max_rel = 0.0

    details: list[str] = []
    if n_nan_mismatch:
        details.append(f"{n_nan_mismatch} NaN disagreement(s) (a NaN is never within tolerance)")
    if n_inf_mismatch:
        details.append(f"{n_inf_mismatch} infinity disagreement(s)")
    if int(exceeds.sum()):
        details.append(f"{int(exceeds.sum())}/{n_total} entries exceed abs+rel*|ref|")
    if n_comparable == 0:
        details.append("no position had finite values on both sides; error magnitudes are not defined")
    if n_zero_ref:
        details.append(f"{n_zero_ref} reference entries are exactly zero (absolute budget only)")

    return CompareResult(
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        n_bad=n_bad,
        first_bad_index=_first_index(bad),
        passed=n_bad == 0,
        n_total=n_total,
        n_comparable=n_comparable,
        n_nan_mismatch=n_nan_mismatch,
        n_inf_mismatch=n_inf_mismatch,
        n_zero_ref=n_zero_ref,
        kind="numeric",
        detail="; ".join(details),
        tolerance=tol,
    )


def align_for_compare(got: Any, want: Any) -> tuple[Any, Any]:
    """Put both sides on the host as torch tensors, preserving structure.

    A candidate's outputs come back from the sandbox on the host while the
    reference was computed on the active device, so any comparator doing tensor
    arithmetic raises "expected all tensors to be on the same device". Both O1
    and O3 hand user-supplied comparators these values, so this lives here
    rather than in either oracle: two copies of the alignment rule is how the
    two oracles came to disagree about correctness in the first place.

    Walks dicts and sequences so a seed returning several named outputs behaves
    like one returning a single tensor. Non-tensor values pass through.
    """
    from collections.abc import Mapping

    try:
        import torch
    except ImportError:
        return got, want

    def host(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: host(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(host(v) for v in value)
        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu")
        try:
            arr = to_numpy(value)
        except Exception:  # noqa: BLE001 - a non-array value is passed through
            return value
        if arr is None:
            return value
        try:
            return torch.as_tensor(arr)
        except (TypeError, RuntimeError, ValueError):
            return value

    return host(got), host(want)
