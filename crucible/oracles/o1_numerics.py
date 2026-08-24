"""O1 - differential numerics against the seed's independent reference.

The candidate is run in a subprocess for every shape in the adversarial sweep
and its outputs are compared against ``seed.reference`` under a tolerance
derived per shape from the dtype and the accumulation depth (never a constant).

What this oracle is careful about:

* **The shape where it first breaks is the grading key.** ``first_break_shape``
  is not a nicety in the evidence; it is the thing a task's answer is scored
  against, so the sweep is walked in a fixed order (the seed's authored order,
  small to large, then any extra detect/decoy shapes) and the first failing
  shape is reported by name.
* **A subset is not a pass.** PASS requires that every shape actually executed.
  A shape that could not run - out of memory, no CUDA device, a broken toolchain
  - makes the whole oracle SKIP with that specific reason. It never silently
  shrinks the sweep and reports PASS on what was left.
* **Candidate code never runs in this interpreter.** Every candidate call goes
  through ``crucible.runner.sandbox.call_entry``, which spawns a fresh
  interpreter, kills the whole process tree on timeout, and moves tensors as
  ``.npy`` files rather than pickles. ``seed.reference`` is first-party code
  from the seed bank, not candidate input, and is evaluated in this process; the
  evidence records that asymmetry explicitly rather than implying both sides
  were sandboxed.
* **A candidate exception is a FAIL, not an ERROR.** If the reference computed
  the shape and the candidate raised, that is a defect in the candidate and the
  exception goes into the evidence. ERROR is reserved for the harness breaking.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Sequence

import numpy as np

from ..runner.sandbox import SandboxResult, call_entry
from ..schema import OracleResult, ShapeSpec, Task
from .base import OracleContext, register_oracle
from .tolerance import (
    Tolerance,
    compare,
    derive,
    fixed,
    from_tensor_scale,
    magnitude_of,
    normalize_dtype,
    to_numpy,
)

logger = logging.getLogger(__name__)

ORACLE_ID = "O1"

#: Substrings that mean "this machine could not run the case", as opposed to
#: "the candidate is wrong". These produce SKIP for the shape, and therefore
#: SKIP for the oracle, because a partial sweep must never be reported as PASS.
_INFRA_MARKERS: tuple[str, ...] = (
    "out of memory",
    "outofmemoryerror",
    "cuda error: out of memory",
    "no cuda gpus are available",
    "cuda unavailable",
    "torch not compiled with cuda",
    "no kernel image is available",
    "dll load failed",
    "libtriton",
    "cannot allocate memory",
    "unable to allocate",
    "insufficient shared memory",
    "no such device",
    "device-side assert",
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _flatten(value: Any) -> list[tuple[str, Any]]:
    """Name the parts of an output exactly as the sandbox child names them."""
    if isinstance(value, dict):
        return [(str(k), v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))]
    if isinstance(value, (list, tuple)):
        return [(f"out{i}", v) for i, v in enumerate(value)]
    return [("out", value)]


def _looks_like_infrastructure(text: str) -> str | None:
    """The marker that matched, or None if this reads like a candidate defect."""
    low = (text or "").lower()
    for marker in _INFRA_MARKERS:
        if marker in low:
            return marker
    return None


def _safe_dir_name(name: str, fallback: str) -> str:
    cleaned = _SAFE_NAME.sub("_", name).strip("._")
    return cleaned or fallback


def _shape_rng_seed(base: int, shape: ShapeSpec) -> int:
    """Reproducible, shape-specific seed so shapes do not share input data."""
    return (int(base) + int(shape.key()[:8], 16)) % (2**31 - 1)


def _dtype_name(shape: ShapeSpec, inputs: dict[str, Any], ref_pairs: list[tuple[str, np.ndarray]]) -> str:
    """The storage dtype whose unit roundoff governs this shape.

    Preference order: what the shape says, then what the reference actually
    produced, then what the inputs actually were. We never default to fp32 for
    an unknown dtype; that would fabricate an error budget.
    """
    declared = shape.kwargs.get("dtype")
    if declared is not None:
        return normalize_dtype(declared)
    for _name, arr in ref_pairs:
        if arr.dtype.kind == "f":
            return normalize_dtype(arr.dtype.name)
    for value in inputs.values():
        dt = getattr(value, "dtype", None)
        if dt is None:
            continue
        try:
            return normalize_dtype(getattr(dt, "name", dt))
        except KeyError:
            continue
    raise KeyError(
        f"shape {shape.name!r} declares no dtype and neither the reference output "
        "nor the inputs expose a floating dtype to derive a budget from"
    )


def _make_inputs(seed: Any, shape: ShapeSpec, device: str, rng_seed: int) -> dict[str, Any]:
    """Build the case inputs, seeded, tolerating simpler make_inputs signatures."""
    generator: Any = None
    try:
        import torch

        generator = torch.Generator(device="cpu" if device == "cpu" else device)
        generator.manual_seed(rng_seed)
    except (ImportError, RuntimeError, TypeError) as exc:
        logger.debug("no seeded generator for shape %s: %s", shape.name, exc)
    for kwargs in (
        {"device": device, "generator": generator},
        {"device": device},
        {},
    ):
        try:
            out = seed.make_inputs(shape, **kwargs)
        except TypeError as exc:
            logger.debug("make_inputs rejected %s for %s: %s", sorted(kwargs), shape.name, exc)
            continue
        if not isinstance(out, dict):
            raise TypeError(
                f"seed {getattr(seed, 'id', '?')}: make_inputs returned "
                f"{type(out).__name__}, expected a dict of keyword arguments"
            )
        return out
    raise TypeError(
        f"seed {getattr(seed, 'id', '?')}: make_inputs accepted none of the "
        "supported signatures (shape[, device][, generator])"
    )


def _reference_pairs(value: Any) -> list[tuple[str, np.ndarray]]:
    return [(name, to_numpy(part)) for name, part in _flatten(value)]


def _load_candidate_pairs(res: SandboxResult) -> tuple[list[tuple[str, np.ndarray]], str]:
    """(pairs, problem). ``problem`` non-empty means the outputs are unusable."""
    pairs: list[tuple[str, np.ndarray]] = []
    for rec in res.outputs():
        name = str(rec.get("name", "out"))
        if rec.get("kind") != "array":
            return [], (
                f"candidate returned a non-array output {name!r} of python type "
                f"{rec.get('python_type', '?')}; numerics cannot be compared"
            )
        path = rec.get("path")
        if not path:
            return [], f"sandbox did not persist output {name!r} to disk; cannot compare values"
        try:
            arr = np.load(str(path), allow_pickle=False)
        except (OSError, ValueError) as exc:
            return [], f"could not read persisted output {name!r} from {path}: {exc}"
        pairs.append((name, arr))
    return pairs, ""


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #


class NumericsOracle:
    """Differential numerics across the adversarial sweep."""

    id = ORACLE_ID
    name = "differential numerics vs seed reference"
    required_caps: tuple[str, ...] = ()

    def applies_to(self, task: Task) -> bool:
        # Numerics apply to every task in the bank; which oracles actually run is
        # decided by ``task.oracles`` in the dispatcher, not here.
        return True

    # -- sweep ------------------------------------------------------------- #

    def sweep(self, ctx: OracleContext) -> list[ShapeSpec]:
        """The shapes to grade on, in evaluation order, de-duplicated.

        The seed's authored sweep comes first (it is ordered small to large and
        carries the adversarial boundaries), then any detect or decoy shapes the
        task added that are not already in it. Order is fixed because
        ``first_break_shape`` depends on it.
        """
        override = ctx.extras.get("shapes")
        if override:
            return list(override)
        ordered: list[ShapeSpec] = []
        seen: set[tuple[str, str]] = set()
        groups: Sequence[Sequence[ShapeSpec]] = (
            list(getattr(ctx.seed, "shape_sweep", []) or []),
            list(ctx.task.detect_shapes),
            list(ctx.task.decoy_shapes),
        )
        for group in groups:
            for shape in group:
                key = (shape.name, shape.key())
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(shape)
        return ordered

    # -- tolerance --------------------------------------------------------- #

    def tolerance_for(
        self,
        ctx: OracleContext,
        shape: ShapeSpec,
        inputs: dict[str, Any],
        ref_pairs: list[tuple[str, np.ndarray]],
    ) -> Tolerance:
        """Derive this shape's budget, or honour an explicitly fixed one."""
        override = ctx.extras.get("fixed_tolerance")
        if override is not None:
            if isinstance(override, Tolerance):
                return override
            if isinstance(override, dict):
                return fixed(
                    rel=float(override["rel"]),
                    abs=override.get("abs"),
                    dtype=override.get("dtype", "unspecified"),
                    reason=str(override.get("reason", "requested via ctx.extras")),
                )
            return fixed(
                rel=float(override),
                reason="requested via ctx.extras['fixed_tolerance']",
            )
        dtype = _dtype_name(shape, inputs, ref_pairs)
        depth_fn = getattr(ctx.seed, "accum_depth", None)
        depth = int(depth_fn(shape)) if callable(depth_fn) else 1
        tol = derive(
            dtype,
            depth,
            mode=ctx.cfg.tolerance_mode,
            safety=ctx.cfg.tolerance_safety,
        )
        return from_tensor_scale(tol, magnitude_of([arr for _n, arr in ref_pairs]))

    # -- one shape --------------------------------------------------------- #

    def _one_shape(self, ctx: OracleContext, shape: ShapeSpec, device: str) -> dict[str, Any]:
        seed = ctx.seed
        started = time.perf_counter()
        rng_seed = _shape_rng_seed(ctx.rng_seed, shape)
        rec: dict[str, Any] = {
            "name": shape.name,
            "shape_key": shape.key(),
            "kwargs": dict(shape.kwargs),
            "status": "skip",
            "reason": "",
            "device": device,
            "rng_seed": rng_seed,
        }

        def finish(status: str, reason: str = "") -> dict[str, Any]:
            rec["status"] = status
            if reason:
                rec["reason"] = reason
            rec["duration_s"] = time.perf_counter() - started
            return rec

        # 1. inputs -- a failure here is the harness's, not the candidate's.
        try:
            inputs = _make_inputs(seed, shape, device, rng_seed)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            marker = _looks_like_infrastructure(f"{type(exc).__name__}: {exc}")
            rec["exception"] = {"where": "make_inputs", "type": type(exc).__name__, "message": str(exc)[:2000]}
            detail = f" ({marker})" if marker else ""
            return finish(
                "skip",
                f"inputs for shape {shape.name!r} could not be built{detail}: "
                f"{type(exc).__name__}: {exc}",
            )

        # 2. reference -- first-party seed code, evaluated here.
        try:
            ref_out = seed.reference(**inputs)
            ref_pairs = _reference_pairs(ref_out)
        except Exception as exc:  # noqa: BLE001
            rec["exception"] = {"where": "reference", "type": type(exc).__name__, "message": str(exc)[:2000]}
            return finish(
                "skip",
                f"seed reference itself raised on shape {shape.name!r} "
                f"({type(exc).__name__}: {exc}); the candidate cannot be graded here",
            )
        rec["reference_outputs"] = [
            {"name": n, "dtype": str(a.dtype), "shape": list(a.shape)} for n, a in ref_pairs
        ]

        # 3. tolerance -- derived per shape, recorded with its derivation.
        try:
            tol = self.tolerance_for(ctx, shape, inputs, ref_pairs)
        except (KeyError, ValueError, TypeError) as exc:
            rec["exception"] = {"where": "tolerance", "type": type(exc).__name__, "message": str(exc)[:2000]}
            return finish(
                "skip",
                f"no error budget could be derived for shape {shape.name!r}: "
                f"{type(exc).__name__}: {exc}",
            )
        rec["dtype"] = tol.dtype
        rec["accum_depth"] = tol.accum_depth
        rec["tolerance"] = tol.as_dict()
        rec["tolerance_formula"] = tol.formula

        # 4. candidate -- always out of process.
        wd = ctx.sub_workdir(f"{ORACLE_ID}/{_safe_dir_name(shape.name, shape.key())}")
        res = call_entry(
            ctx.candidate_src,
            str(getattr(seed, "entry", "")),
            inputs=inputs,
            workdir=wd,
            device=device,
            timeout_s=float(ctx.cfg.sandbox_timeout_s),
            seed=rng_seed,
            deterministic=True,
            save_outputs=True,
            call_style="kwargs",
        )
        rec["sandbox_duration_s"] = res.duration_s

        if not res.ok:
            err = res.error or {}
            message = f"{err.get('type', 'error')}: {err.get('message', res.message)}"
            rec["exception"] = {
                "where": "candidate",
                "type": str(err.get("type", "error")),
                "message": str(err.get("message", res.message))[:4000],
                "traceback": res.traceback_text[-4000:],
                "stderr": res.stderr[-2000:],
            }
            if res.timed_out:
                return finish(
                    "fail",
                    f"candidate did not finish shape {shape.name!r} within "
                    f"{ctx.cfg.sandbox_timeout_s:.0f}s while the reference did; process tree killed",
                )
            marker = _looks_like_infrastructure(f"{message} {res.stderr}")
            if marker is not None:
                return finish(
                    "skip",
                    f"shape {shape.name!r} could not be executed on this machine "
                    f"(matched {marker!r}): {message}",
                )
            return finish(
                "fail",
                f"candidate raised on shape {shape.name!r} where the reference did not: {message}",
            )

        cand_pairs, problem = _load_candidate_pairs(res)
        if problem:
            marker = _looks_like_infrastructure(problem)
            return finish("skip" if marker else "fail", problem)

        rec["checksums"] = res.checksums()

        # 5. compare, output by output.
        ref_names = [n for n, _a in ref_pairs]
        cand_names = [n for n, _a in cand_pairs]
        if ref_names != cand_names:
            return finish(
                "fail",
                f"candidate returned outputs {cand_names} on shape {shape.name!r} "
                f"but the reference returned {ref_names}",
            )

        ref_by_name = dict(ref_pairs)
        worst_abs = 0.0
        worst_rel = 0.0
        n_bad = 0
        failed: list[str] = []
        per_output: list[dict[str, Any]] = []
        first_bad_index: list[int] | None = None
        for name, cand_arr in cand_pairs:
            cmp_res = compare(cand_arr, ref_by_name[name], tol)
            entry = cmp_res.as_dict()
            entry["name"] = name
            entry.pop("tolerance", None)  # already recorded once per shape
            per_output.append(entry)
            worst_abs = max(worst_abs, cmp_res.max_abs_err)
            worst_rel = max(worst_rel, cmp_res.max_rel_err)
            n_bad += cmp_res.n_bad
            if not cmp_res.passed:
                failed.append(name)
                if first_bad_index is None and cmp_res.first_bad_index is not None:
                    first_bad_index = list(cmp_res.first_bad_index)

        rec["per_output"] = per_output
        rec["max_abs_err"] = worst_abs
        rec["max_rel_err"] = worst_rel
        rec["n_bad"] = n_bad
        rec["first_bad_index"] = first_bad_index
        rec["comparable"] = True

        if failed:
            details = "; ".join(
                f"{e['name']}: {e['detail']}" for e in per_output if not e["passed"] and e["detail"]
            )
            return finish(
                "fail",
                f"shape {shape.name!r} exceeded its derived tolerance "
                f"[{tol.formula}]: max_abs_err={worst_abs:.3e}, max_rel_err={worst_rel:.3e}, "
                f"{n_bad} bad element(s) in {', '.join(failed)}"
                + (f" ({details})" if details else ""),
            )
        return finish("pass")

    # -- run --------------------------------------------------------------- #

    def run(self, ctx: OracleContext) -> OracleResult:
        seed = ctx.seed
        device = str(ctx.device or "cpu")

        def result(verdict: str, reason: str, evidence: dict[str, Any]) -> OracleResult:
            return OracleResult(
                oracle=ORACLE_ID,
                verdict=verdict,  # type: ignore[arg-type]
                reason=reason,
                evidence=evidence,
                capabilities_used=[],
            )

        base_evidence: dict[str, Any] = {
            "device": device,
            "rng_seed": ctx.rng_seed,
            "tolerance_mode": ctx.cfg.tolerance_mode,
            "tolerance_safety": ctx.cfg.tolerance_safety,
            "candidate_execution": (
                "subprocess sandbox (crucible.runner.sandbox.call_entry); "
                "candidate code is never executed in the grading interpreter"
            ),
            "reference_execution": (
                "in-process (seed.reference is first-party seed-bank code, not candidate input)"
            ),
        }

        if seed is None:
            return result(
                "SKIP",
                "no seed was supplied in the oracle context; there is no reference to compare against",
                base_evidence,
            )
        entry = str(getattr(seed, "entry", "") or "")
        base_evidence["seed_id"] = str(getattr(seed, "id", "?"))
        base_evidence["entry"] = entry
        if not entry:
            return result("SKIP", f"seed {base_evidence['seed_id']!r} declares no entry point", base_evidence)
        if not callable(getattr(seed, "reference", None)):
            return result(
                "SKIP",
                f"seed {base_evidence['seed_id']!r} has no callable reference implementation",
                base_evidence,
            )
        if not ctx.candidate_src.strip():
            return result("SKIP", "candidate source is empty; nothing was executed", base_evidence)

        if device.startswith("cuda") and not ctx.caps.cuda:
            return result(
                "SKIP",
                f"device {device!r} was requested but no CUDA device was detected "
                f"({ctx.caps.detail('cuda') or 'torch.cuda.is_available() returned False'})",
                base_evidence,
            )
        if device == "cpu" and not bool(getattr(seed, "supports_cpu", True)):
            return result(
                "SKIP",
                f"seed {base_evidence['seed_id']!r} declares supports_cpu=False and the "
                "context device is cpu; the sweep cannot be executed here",
                base_evidence,
            )

        shapes = self.sweep(ctx)
        if not shapes:
            return result(
                "SKIP",
                f"seed {base_evidence['seed_id']!r} and task {ctx.task.task_id!r} expose no shapes to sweep",
                base_evidence,
            )

        records: list[dict[str, Any]] = []
        for shape in shapes:
            records.append(self._one_shape(ctx, shape, device))

        passed = [r for r in records if r["status"] == "pass"]
        failed = [r for r in records if r["status"] == "fail"]
        skipped = [r for r in records if r["status"] == "skip"]
        comparable = [r for r in records if r.get("comparable")]
        first_break = failed[0] if failed else None

        evidence = dict(base_evidence)
        evidence.update(
            {
                "per_shape": records,
                "shapes_evaluated": [r["name"] for r in records],
                "shapes_not_run": [{"name": r["name"], "reason": r["reason"]} for r in skipped],
                "first_break_shape": first_break["name"] if first_break else None,
                "first_break": first_break,
                "summary": {
                    "n_shapes": len(records),
                    "n_passed": len(passed),
                    "n_failed": len(failed),
                    "n_not_run": len(skipped),
                    "n_compared": len(comparable),
                },
            }
        )
        # Error magnitudes exist only for shapes that were actually compared. If
        # nothing was compared they are absent from the evidence, not zero.
        if comparable:
            evidence["max_abs_err"] = max(float(r.get("max_abs_err", 0.0)) for r in comparable)
            evidence["max_rel_err"] = max(float(r.get("max_rel_err", 0.0)) for r in comparable)

        if failed:
            return result(
                "FAIL",
                f"{len(failed)}/{len(records)} shape(s) failed; first break at shape "
                f"{first_break['name']!r}: {first_break['reason']}",
                evidence,
            )
        if skipped:
            reasons = "; ".join(f"{r['name']}: {r['reason']}" for r in skipped[:4])
            more = "" if len(skipped) <= 4 else f" (+{len(skipped) - 4} more)"
            return result(
                "SKIP",
                f"{len(skipped)}/{len(records)} shape(s) could not be executed, so the sweep is "
                f"incomplete and the passing subset is not a pass -- {reasons}{more}",
                evidence,
            )
        return result(
            "PASS",
            f"all {len(records)} shape(s) matched the reference within their derived tolerance",
            evidence,
        )


O1 = NumericsOracle()
register_oracle(O1)

__all__ = ["O1", "NumericsOracle", "ORACLE_ID"]
