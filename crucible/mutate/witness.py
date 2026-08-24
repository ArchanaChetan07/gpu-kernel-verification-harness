"""Witness search: the step that makes a mutation into evidence.

A mutation is a hypothesis. This module tries to falsify the hypothesis "nothing
observable changed" by executing the baseline and the mutant on **identical**
seeded inputs across the seed's adversarial sweep, cheapest shape first, and
comparing against a derived error budget rather than a hardcoded tolerance.

Design commitments:

* **Identical inputs, not merely the same seed.** The inputs are materialised
  once per shape and the *same* ``.npy`` references are handed to both runs. A
  shared RNG seed would still leave the two runs exposed to any nondeterminism
  in input construction; shared files cannot diverge.
* **Everything runs in the sandbox.** A mutant that segfaults, allocates the
  machine, or spins forever cannot take the generator down with it. A timeout is
  not an error here: it is a legitimate witness of kind ``hang``.
* **No witness means discard.** A mutation that produces no observable
  difference on any shape is semantically neutral or untested-by-construction,
  and it is refused with a reason. That refusal is the structural guarantee that
  every shipped task has a real, reachable answer.
* **A broken baseline is never a witness.** If the *baseline* fails on a shape,
  that shape proves nothing about the mutation and is recorded as unusable.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..config import DEFAULT_CONFIG, Config
from ..runner.sandbox import SandboxResult, call_entry, input_refs_from
from ..schema import ShapeSpec, Witness
from .astutil import canonical

logger = logging.getLogger(__name__)

WitnessKind = str  # one of: numeric, exception, shape, hang, loss_curve

#: Unit roundoff per dtype, used only when ``crucible.oracles.tolerance`` is not
#: importable. Same values as the contract's table so the fallback is the same
#: model, not a different one.
_FALLBACK_EPS: dict[str, float] = {
    "fp64": 2.0**-53,
    "fp32": 2.0**-24,
    "tf32": 2.0**-11,
    "bf16": 2.0**-8,
    "fp16": 2.0**-11,
    "fp8": 2.0**-4,
}

_DTYPE_ALIASES: dict[str, str] = {
    "float64": "fp64",
    "double": "fp64",
    "float32": "fp32",
    "float": "fp32",
    "tfloat32": "tf32",
    "tf32": "tf32",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "fp16": "fp16",
    "float8_e4m3fn": "fp8",
    "float8_e5m2": "fp8",
    "fp8": "fp8",
}


def short_dtype(name: Any) -> str:
    """Normalise ``torch.bfloat16`` / ``"bfloat16"`` to the tolerance table key."""
    text = str(name).replace("torch.", "").strip().lower()
    return _DTYPE_ALIASES.get(text, text)


@dataclass(frozen=True)
class ToleranceUsed:
    """The error budget a comparison was made against, and where it came from."""

    rel: float
    abs: float
    dtype: str
    accum_depth: int
    mode: str
    formula: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rel": self.rel,
            "abs": self.abs,
            "dtype": self.dtype,
            "accum_depth": self.accum_depth,
            "mode": self.mode,
            "formula": self.formula,
            "source": self.source,
        }


def _fallback_tolerance(
    dtype: str, accum_depth: int, mode: str, safety: float, note: str
) -> ToleranceUsed:
    key = short_dtype(dtype)
    eps = _FALLBACK_EPS.get(key)
    if eps is None:
        # An unknown dtype gets the widest budget in the table, never a guess
        # that happens to be tight: a tight guess would manufacture witnesses.
        eps = max(_FALLBACK_EPS.values())
        key = f"{key}(unknown, using widest eps)"
    depth = max(int(accum_depth), 1)
    growth = math.sqrt(depth) if mode == "stochastic" else float(depth)
    rel = float(safety) * growth * eps
    formula = (
        f"rel = safety({safety:g}) * growth_{mode}({depth}) = {growth:.4g} * EPS[{key}]"
        f"={eps:.3g} -> {rel:.3g}"
    )
    return ToleranceUsed(
        rel=rel,
        abs=rel,
        dtype=key,
        accum_depth=depth,
        mode=mode,
        formula=formula,
        source=f"witness fallback ({note})",
    )


def derive_tolerance(dtype: Any, accum_depth: int, cfg: Config) -> ToleranceUsed:
    """The derived budget, preferring ``crucible.oracles.tolerance.derive``.

    That module is written by another author and may not exist yet; the import
    is therefore lazy and its absence is recorded in ``source`` rather than
    silently papered over.
    """
    key = short_dtype(dtype)
    try:
        from ..oracles.tolerance import derive  # type: ignore[attr-defined]
    except ImportError as exc:
        return _fallback_tolerance(
            key, accum_depth, cfg.tolerance_mode, cfg.tolerance_safety,
            f"crucible.oracles.tolerance unavailable: {exc}",
        )
    try:
        tol = derive(
            key,
            max(int(accum_depth), 1),
            mode=cfg.tolerance_mode,
            safety=cfg.tolerance_safety,
        )
        rel = float(tol.rel)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        return _fallback_tolerance(
            key, accum_depth, cfg.tolerance_mode, cfg.tolerance_safety,
            f"tolerance.derive unusable: {type(exc).__name__}: {exc}",
        )
    return ToleranceUsed(
        rel=rel,
        abs=float(getattr(tol, "abs", rel)),
        dtype=str(getattr(tol, "dtype", key)),
        accum_depth=int(getattr(tol, "accum_depth", max(int(accum_depth), 1))),
        mode=str(getattr(tol, "mode", cfg.tolerance_mode)),
        formula=str(getattr(tol, "formula", "")),
        source="crucible.oracles.tolerance.derive",
    )


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #


@dataclass
class ShapeOutcome:
    """What happened on one shape. Recorded whether or not it broke anything."""

    shape: ShapeSpec
    usable: bool
    differs: bool
    kind: WitnessKind
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    tolerance: float = 0.0
    baseline_checksum: str = ""
    mutant_checksum: str = ""
    detail: str = ""
    baseline_ok: bool = True
    mutant_ok: bool = True
    duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape.name,
            "kwargs": dict(self.shape.kwargs),
            "usable": self.usable,
            "differs": self.differs,
            "kind": self.kind,
            "max_abs_err": self.max_abs_err,
            "max_rel_err": self.max_rel_err,
            "tolerance": self.tolerance,
            "detail": self.detail,
            "baseline_ok": self.baseline_ok,
            "mutant_ok": self.mutant_ok,
            "duration_s": round(self.duration_s, 4),
        }


@dataclass
class WitnessSearchResult:
    witness: Witness | None = None
    decoys: list[ShapeSpec] = field(default_factory=list)
    detects: list[ShapeSpec] = field(default_factory=list)
    #: set when the sweep was cut short, so partial coverage is never reported
    #: as if the whole shape space had been classified
    truncated: str | None = None
    n_evaluated: int = 0
    discarded_reason: str | None = None
    outcomes: list[ShapeOutcome] = field(default_factory=list)
    tolerance: ToleranceUsed | None = None
    device: str = "cpu"

    @property
    def admitted(self) -> bool:
        return self.witness is not None and self.discarded_reason is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "witness_kind": self.witness.kind if self.witness else None,
            "witness_shape": self.witness.shape.name if self.witness else None,
            "n_decoys": len(self.decoys),
            "n_detects": len(self.detects),
            "truncated": self.truncated,
            "n_evaluated": self.n_evaluated,
            "discarded_reason": self.discarded_reason,
            "device": self.device,
            "tolerance": self.tolerance.as_dict() if self.tolerance else None,
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #


def _combined_checksum(result: SandboxResult) -> str:
    checks = {k: v for k, v in result.checksums().items() if v}
    if not checks:
        return ""
    if len(checks) == 1:
        return next(iter(checks.values()))
    payload = ";".join(f"{k}={v}" for k, v in sorted(checks.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _by_name(result: SandboxResult) -> dict[str, dict[str, Any]]:
    return {str(rec.get("name")): rec for rec in result.outputs()}


def _compare_outputs(
    baseline: SandboxResult, mutant: SandboxResult, tol: ToleranceUsed
) -> tuple[bool, WitnessKind, float, float, str]:
    """(differs, kind, max_abs_err, max_rel_err, detail).

    Non-finite values are reported as a structural mismatch with a count rather
    than folded into an ``inf`` error magnitude: "the mutant produced 12 inf
    where the baseline was finite" is evidence; ``max_abs_err = inf`` is not.
    """
    import numpy as np

    b_recs = _by_name(baseline)
    m_recs = _by_name(mutant)
    if set(b_recs) != set(m_recs):
        return (
            True,
            "shape",
            0.0,
            0.0,
            f"output names differ: baseline {sorted(b_recs)} vs mutant {sorted(m_recs)}",
        )

    worst_abs = 0.0
    worst_rel = 0.0
    differs = False
    kind: WitnessKind = "numeric"
    details: list[str] = []

    for name in sorted(b_recs):
        brec, mrec = b_recs[name], m_recs[name]
        if brec.get("kind") != mrec.get("kind"):
            return (True, "shape", 0.0, 0.0, f"output {name!r} changed kind")
        if brec.get("kind") == "opaque":
            if brec.get("repr") != mrec.get("repr"):
                differs = True
                details.append(f"opaque output {name!r} differs")
            continue
        if list(brec.get("shape") or []) != list(mrec.get("shape") or []):
            return (
                True,
                "shape",
                0.0,
                0.0,
                f"output {name!r} shape {brec.get('shape')} -> {mrec.get('shape')}",
            )
        if brec.get("checksum") and brec.get("checksum") == mrec.get("checksum"):
            continue

        bpath, mpath = brec.get("path"), mrec.get("path")
        if not bpath or not mpath:
            differs = True
            details.append(f"output {name!r} checksums differ and no array was saved")
            continue
        try:
            a = np.load(bpath, allow_pickle=False).astype(np.float64, copy=False)
            b = np.load(mpath, allow_pickle=False).astype(np.float64, copy=False)
        except (OSError, ValueError) as exc:
            differs = True
            details.append(f"output {name!r}: could not load saved arrays: {exc}")
            continue
        if a.shape != b.shape:
            return (True, "shape", 0.0, 0.0, f"output {name!r} shape {a.shape} -> {b.shape}")
        if a.size == 0:
            continue

        finite_mismatch = int(np.count_nonzero(np.isfinite(a) != np.isfinite(b)))
        nan_mismatch = int(np.count_nonzero(np.isnan(a) != np.isnan(b)))
        both_finite = np.isfinite(a) & np.isfinite(b)
        diff = np.zeros_like(a)
        np.subtract(a, b, out=diff, where=both_finite)
        diff = np.abs(diff)
        scale = float(np.abs(a[both_finite]).max()) if both_finite.any() else 0.0
        atol = max(tol.abs, tol.rel * scale)
        bad = np.zeros(a.shape, dtype=bool)
        np.greater(diff, atol + tol.rel * np.abs(a), out=bad, where=both_finite)

        max_abs = float(diff[both_finite].max()) if both_finite.any() else 0.0
        denom = np.maximum(np.abs(a), np.finfo(np.float64).tiny)
        rel = np.zeros_like(a)
        np.divide(diff, denom, out=rel, where=both_finite)
        max_rel = float(rel[both_finite].max()) if both_finite.any() else 0.0
        worst_abs = max(worst_abs, max_abs)
        worst_rel = max(worst_rel, max_rel)

        if finite_mismatch or nan_mismatch:
            differs = True
            details.append(
                f"output {name!r}: {finite_mismatch} element(s) finite in the baseline and "
                f"non-finite in the mutant ({nan_mismatch} NaN mismatches)"
            )
        elif bool(bad.any()):
            differs = True
            n_bad = int(np.count_nonzero(bad))
            details.append(
                f"output {name!r}: {n_bad}/{a.size} elements outside the budget "
                f"(max_abs={max_abs:.6g}, max_rel={max_rel:.6g}, rtol={tol.rel:.3g})"
            )
        if "loss" in name.lower() and differs:
            kind = "loss_curve"

    return differs, kind, worst_abs, worst_rel, "; ".join(details)


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


def _cost(seed: Any, shape: ShapeSpec) -> tuple[int, int, str]:
    """Sort key so the cheapest shape is executed first."""
    try:
        moved = int(seed.bytes_moved(shape))
    except Exception as exc:  # noqa: BLE001 - a seed's cost model must not stop the search
        logger.debug("seed %s bytes_moved failed for %s: %s", getattr(seed, "id", "?"), shape.name, exc)
        moved = 1 << 40
    try:
        flops = int(seed.flops(shape))
    except Exception as exc:  # noqa: BLE001
        logger.debug("seed %s flops failed for %s: %s", getattr(seed, "id", "?"), shape.name, exc)
        flops = 1 << 40
    return (moved, flops, shape.name)


def _accum_depth(seed: Any, shape: ShapeSpec) -> int:
    try:
        return max(int(seed.accum_depth(shape)), 1)
    except Exception as exc:  # noqa: BLE001
        logger.debug("seed %s accum_depth failed for %s: %s", getattr(seed, "id", "?"), shape.name, exc)
        return 1


def _error_text(result: SandboxResult) -> str:
    err = result.error or {}
    kind = str(err.get("type", "error"))
    message = str(err.get("message", result.message))
    return f"{kind}: {message}"[:1200]


def search(
    seed: Any,
    mutant_src: str,
    sweep: Sequence[ShapeSpec] | None = None,
    caps: Any = None,
    cfg: Config | None = None,
    *,
    baseline_src: str | None = None,
    workdir: Path | str | None = None,
    device: str | None = None,
    stop_on_first: bool | None = None,
    timeout_s: float | None = None,
) -> WitnessSearchResult:
    """Find the first shape on which the mutant and the baseline disagree.

    Returns a result whose ``witness`` is None and whose ``discarded_reason`` is
    set whenever no such shape exists in the sweep. The caller must treat that
    as a refusal to ship, not as a passing task.
    """
    cfg = cfg or DEFAULT_CONFIG
    shapes = list(sweep if sweep is not None else getattr(seed, "shape_sweep", []) or [])
    stop_first = cfg.witness_stop_on_first if stop_on_first is None else bool(stop_on_first)
    budget = float(cfg.witness_time_budget_s)
    per_run_timeout = float(cfg.sandbox_timeout_s if timeout_s is None else timeout_s)
    base_src = canonical(seed.source) if baseline_src is None else baseline_src

    if device is None:
        device = "cuda" if bool(getattr(caps, "cuda", False)) else "cpu"
    result = WitnessSearchResult(device=device)

    if not shapes:
        result.discarded_reason = (
            f"seed {getattr(seed, 'id', '?')!r} has no shapes to sweep; the mutation is "
            "untested-by-construction"
        )
        return result
    if device == "cpu" and not bool(getattr(seed, "supports_cpu", True)):
        result.discarded_reason = (
            f"seed {getattr(seed, 'id', '?')!r} declares supports_cpu=False and no CUDA device "
            "is available (caps.cuda is False); the mutation cannot be executed here and is "
            "untested-by-construction"
        )
        return result

    shapes.sort(key=lambda s: _cost(seed, s))
    shapes = shapes[: max(int(cfg.witness_max_shapes), 1)]

    root = Path(workdir) if workdir is not None else Path.cwd() / cfg.work_root / "witness"
    root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    input_failures: list[str] = []
    baseline_failures: list[str] = []

    for index, shape in enumerate(shapes):
        if time.perf_counter() - started > budget:
            if result.witness is None:
                result.discarded_reason = (
                    f"witness time budget of {budget:.0f}s exhausted after {result.n_evaluated} "
                    f"shape(s) with no disagreement found"
                )
                return result
            # A witness exists, so the task is admissible; the budget only cuts
            # the decoy/detect classification short. Record the truncation
            # rather than letting a partially swept task read as fully swept -
            # a silent cap is indistinguishable from full coverage.
            result.truncated = (
                f"sweep stopped after {result.n_evaluated}/{len(shapes)} shape(s): "
                f"witness time budget of {budget:.0f}s exhausted; "
                f"{len(shapes) - result.n_evaluated} shape(s) unclassified"
            )
            logger.info("witness sweep truncated: %s", result.truncated)
            break
        if result.witness is not None and stop_first:
            break

        tol = derive_tolerance(
            shape.kwargs.get("dtype", "float32"), _accum_depth(seed, shape), cfg
        )
        if result.tolerance is None:
            result.tolerance = tol

        shape_dir = root / f"s{index:02d}_{shape.key()[:8]}"
        shape_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        refs = _materialise(seed, shape, shape_dir, device, cfg)
        if isinstance(refs, str):
            input_failures.append(f"{shape.name}: {refs}")
            result.outcomes.append(
                ShapeOutcome(
                    shape=shape,
                    usable=False,
                    differs=False,
                    kind="exception",
                    detail=f"inputs could not be built: {refs}",
                    baseline_ok=False,
                    mutant_ok=False,
                    duration_s=time.perf_counter() - t0,
                )
            )
            continue

        baseline = call_entry(
            base_src,
            seed.entry,
            input_refs=refs,
            workdir=shape_dir / "baseline",
            device=device,
            timeout_s=per_run_timeout,
            seed=cfg.rng_seed,
            deterministic=True,
            save_outputs=True,
        )
        mutant = call_entry(
            mutant_src,
            seed.entry,
            input_refs=refs,
            workdir=shape_dir / "mutant",
            device=device,
            timeout_s=per_run_timeout,
            seed=cfg.rng_seed,
            deterministic=True,
            save_outputs=True,
        )
        result.n_evaluated += 1
        elapsed = time.perf_counter() - t0

        if not baseline.ok:
            reason = "timed out" if baseline.timed_out else _error_text(baseline)
            baseline_failures.append(f"{shape.name}: {reason}")
            result.outcomes.append(
                ShapeOutcome(
                    shape=shape,
                    usable=False,
                    differs=False,
                    kind="exception",
                    detail=f"baseline itself failed ({reason}); this shape proves nothing",
                    baseline_ok=False,
                    mutant_ok=mutant.ok,
                    duration_s=elapsed,
                )
            )
            continue

        b_sum = _combined_checksum(baseline)
        m_sum = _combined_checksum(mutant)

        if mutant.timed_out:
            outcome = ShapeOutcome(
                shape=shape,
                usable=True,
                differs=True,
                kind="hang",
                tolerance=tol.rel,
                baseline_checksum=b_sum,
                mutant_checksum="",
                detail=(
                    f"mutant exceeded the {per_run_timeout:.0f}s sandbox budget while the "
                    f"baseline completed in {baseline.duration_s:.2f}s"
                ),
                mutant_ok=False,
                duration_s=elapsed,
            )
        elif not mutant.ok:
            outcome = ShapeOutcome(
                shape=shape,
                usable=True,
                differs=True,
                kind="exception",
                tolerance=tol.rel,
                baseline_checksum=b_sum,
                mutant_checksum="",
                detail=f"mutant raised where the baseline succeeded: {_error_text(mutant)}",
                mutant_ok=False,
                duration_s=elapsed,
            )
        else:
            differs, kind, max_abs, max_rel, detail = _compare_outputs(baseline, mutant, tol)
            outcome = ShapeOutcome(
                shape=shape,
                usable=True,
                differs=differs,
                kind=kind if differs else "numeric",
                max_abs_err=max_abs,
                max_rel_err=max_rel,
                tolerance=tol.rel,
                baseline_checksum=b_sum,
                mutant_checksum=m_sum,
                detail=detail
                or (
                    "baseline and mutant agree within the derived budget "
                    f"(rtol={tol.rel:.3g}); {tol.formula}"
                ),
                duration_s=elapsed,
            )

        result.outcomes.append(outcome)
        if outcome.differs:
            result.detects.append(shape)
            if result.witness is None:
                # The reported budget is the one the witness was judged against,
                # not the one from whichever cheap shape happened to run first.
                result.tolerance = tol
                result.witness = Witness(
                    shape=shape,
                    max_abs_err=outcome.max_abs_err,
                    max_rel_err=outcome.max_rel_err,
                    tolerance=tol.rel,
                    baseline_checksum=outcome.baseline_checksum,
                    mutant_checksum=outcome.mutant_checksum,
                    kind=outcome.kind,  # type: ignore[arg-type]
                    detail=f"{outcome.detail} | budget: {tol.formula} [{tol.source}]"[:2000],
                )
        else:
            result.decoys.append(shape)

    if result.witness is None:
        # A T4 mutation is byte-identical to the baseline by definition -- the
        # whole tier is "correct output, wrong speed". Every correctness-based
        # witness above is blind to it, so before discarding as neutral, ask
        # whether the mutation is observable in TIME.
        result.witness = _performance_witness(
            seed, base_src, mutant_src, result, cfg, caps, root, device, per_run_timeout
        )
    if result.witness is None:
        result.discarded_reason = _discard_reason(
            result, input_failures, baseline_failures, len(shapes)
        )
    return result


def _performance_witness(
    seed: Any,
    baseline_src: str,
    mutant_src: str,
    result: WitnessSearchResult,
    cfg: Config,
    caps: Any,
    root: Path,
    device: str,
    timeout_s: float,
) -> Witness | None:
    """Admit a mutation that is slower but numerically identical.

    Timing is noisy, so admission keys on the LOWER bound of a bootstrap CI on
    the ratio, never the point estimate: a task must be reliably slower, not
    slower once. When clocks could not be locked the interval is widened first
    and the reason travels with it, so a weaker measurement looks weaker.
    Returning None here is the honest outcome -- it means the mutation is
    genuinely unobservable and will be discarded.
    """
    if not bool(getattr(cfg, "perf_witness_enabled", False)):
        return None
    usable = [o for o in result.outcomes if o.usable]
    if not usable:
        return None

    try:
        from ..oracles.o2_perf import (  # noqa: PLC0415 - optional at import time
            bootstrap_ratio_ci,
            time_in_sandbox,
            widen_ci,
        )
        from ..runner.determinism import lock_clocks  # noqa: PLC0415
    except ImportError as exc:
        logger.debug("performance witness unavailable: %s", exc)
        return None

    # The costliest shape that actually ran: a regression shows up there first.
    shape = max((o.shape for o in usable), key=lambda sh: _cost(seed, sh))
    work = Path(root) / "perf"
    work.mkdir(parents=True, exist_ok=True)
    refs = _materialise(seed, shape, work, device, cfg)
    if isinstance(refs, str):
        logger.debug("performance witness could not build inputs: %s", refs)
        return None

    reps = int(getattr(cfg, "perf_witness_reps", 30))
    warmup = int(getattr(cfg, "perf_witness_warmup", 10))
    min_ratio = float(getattr(cfg, "perf_witness_min_ratio", 1.15))

    with lock_clocks(enabled=bool(getattr(caps, "can_lock_clocks", False))) as lock:
        locked = bool(getattr(lock, "locked", False))
        runs = {}
        for label, src in (("baseline", baseline_src), ("mutant", mutant_src)):
            runs[label] = time_in_sandbox(
                src,
                seed.entry,
                input_refs=refs,
                workdir=work / label,
                device=device,
                reps=reps,
                warmup=warmup,
                timeout_s=timeout_s,
                seed=cfg.rng_seed,
            )

    base, mut = runs["baseline"], runs["mutant"]
    if not (base.ok and mut.ok and base.times_s and mut.times_s):
        logger.debug("performance witness: timing did not complete")
        return None

    ci = bootstrap_ratio_ci(mut.times_s, base.times_s, resamples=int(cfg.bootstrap_resamples))
    if not locked:
        ci = widen_ci(ci, 1.5, "GPU clocks could not be locked; interval inflated")
    # Deliberately NOT o2_perf.classify_ratio: that helper reads its interval as
    # a SPEEDUP (baseline/candidate), while this one is a SLOWDOWN
    # (mutant/baseline) because "how much slower" is the natural framing here.
    # Feeding a slowdown to it reports a 1.9x regression as "faster". The
    # admission rule below is the whole test anyway: the LOWER bound of the
    # interval must clear the threshold, so a mutation has to be reliably
    # slower rather than slower on one noisy run.
    if ci.low < min_ratio:
        return None

    return Witness(
        shape=shape,
        max_abs_err=0.0,
        max_rel_err=0.0,
        tolerance=min_ratio,
        baseline_checksum=str(sorted(base.checksums.values())[:1]),
        mutant_checksum=str(sorted(mut.checksums.values())[:1]),
        kind="performance",
        detail=(
            f"numerically identical but {ci.point:.2f}x slower "
            f"(95% CI [{ci.low:.2f}, {ci.high:.2f}] over {reps} reps on {device}; "
            f"admission needs CI low > {min_ratio:.2f}); clocks_locked={locked}"
            + (f"; {ci.inflation_reason}" if getattr(ci, "inflation_reason", "") else "")
        )[:2000],
    )


def _discard_reason(
    result: WitnessSearchResult,
    input_failures: list[str],
    baseline_failures: list[str],
    n_shapes: int,
) -> str:
    usable = [o for o in result.outcomes if o.usable]
    if not usable:
        parts = []
        if input_failures:
            parts.append("inputs could not be built for " + "; ".join(input_failures[:3]))
        if baseline_failures:
            parts.append("the baseline failed on " + "; ".join(baseline_failures[:3]))
        detail = "; ".join(parts) or "no shape produced a usable comparison"
        return (
            f"untested-by-construction: none of the {n_shapes} swept shape(s) yielded a valid "
            f"baseline/mutant comparison ({detail})"
        )
    tol = result.tolerance
    budget = f" (rtol={tol.rel:.3g}; {tol.formula})" if tol else ""
    return (
        f"semantically neutral: baseline and mutant agreed on all {len(usable)} executed "
        f"shape(s){budget}; there is no input in this sweep on which the mutation is observable"
    )


def _materialise(
    seed: Any, shape: ShapeSpec, shape_dir: Path, device: str, cfg: Config
) -> dict[str, Any] | str:
    """Build the inputs once and freeze them to disk. Returns refs or an error string."""
    try:
        import torch
    except ImportError as exc:
        return f"torch is not importable: {exc}"
    try:
        # The generator must live on the same device as the tensors it seeds:
        # torch rejects a CPU generator when building CUDA tensors, which would
        # fail every shape in the sweep and discard every mutation as
        # untested-by-construction -- a silent, total loss of task supply.
        generator = torch.Generator(device="cpu" if device == "cpu" else device)
        generator.manual_seed(int(cfg.rng_seed))
        inputs = seed.make_inputs(shape, device, generator)
    except Exception as exc:  # noqa: BLE001 - a seed defect must not abort the sweep
        logger.debug("make_inputs failed for %s/%s: %s", getattr(seed, "id", "?"), shape.name, exc)
        return f"{type(exc).__name__}: {exc}"
    if not isinstance(inputs, dict):
        return f"make_inputs returned {type(inputs).__name__}, expected a dict"
    try:
        return input_refs_from(inputs, shape_dir / "in", device=device)
    except Exception as exc:  # noqa: BLE001
        return f"inputs could not be serialised: {type(exc).__name__}: {exc}"


__all__ = [
    "search",
    "WitnessSearchResult",
    "ShapeOutcome",
    "ToleranceUsed",
    "derive_tolerance",
    "short_dtype",
]
