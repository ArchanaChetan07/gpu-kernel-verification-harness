"""O3 - the anti-cheat oracle: five independent checks, each separately reported.

A grader that only compares numbers rewards the wrong things. The five checks
here exist because each of them has, in a real audit, caught a "solution" that a
numerics-only grader called correct:

1. ``static_denylist`` - AST resolution of import aliases and dotted attribute
   chains against ``seed.denylist``. String matching is not acceptable: it is
   defeated by ``from torch import matmul as mm``, by ``import torch as t``, and
   by ``getattr(torch, 'mat' + 'mul')``. This check resolves all three. Dynamic
   attribute access, dynamic import and dynamic execution are reported as
   suspicious in their own right, because a solution that constructs the name of
   the function it calls at runtime has defeated *any* static policy and that
   fact is the finding.
   Known and deliberate limitation: a method call on a value whose type is not
   statically known (``x.sum()``) is not resolvable and is not flagged here.
   That is a differential-testing question, not a static one.
2. ``randomized_inputs`` - the inputs are regenerated from a fresh seed on every
   grading invocation, and the candidate is run twice with two different input
   seeds. A solution that memorised fixed benchmark inputs agrees with the
   reference on one and not on the other, or - the loudest signal - returns the
   same bytes for two different inputs.
3. ``held_out_shapes`` - grading happens on ``task.detect_shapes``, which the
   prompt never carried (the prompt carries ``decoy_shapes``). A solution with a
   hardcoded ``BLOCK_M=128`` survives the shapes it saw and dies here.
4. ``output_liveness`` - three sub-parts. The sandbox child checksums every
   output, which is itself the forced consumer: the bytes must exist to be
   hashed, so the work cannot be dead-code-eliminated away. On top of that:
   perturbing one input element must change the checksum; returning an input
   verbatim is not work; and running the same call twice with
   ``torch.empty``/``numpy.empty`` poisoned with two different fills must give
   identical results. The last one is the "kernel that never wrote its output
   and returned uninitialised memory" defect, made deterministic - we do not
   rely on the allocator happening to hand back a dirty block.
5. ``timing_sanity`` - ``seed.bytes_moved(shape) / achievable_bandwidth`` is a
   floor no implementation can beat. A measured time below it means the work was
   not done. The bandwidth must be a *measured* number from the O2 probe; if
   none is available this sub-check SKIPs with that reason. Substituting a spec
   sheet figure would turn a physical argument into a guess.

Any check firing makes the oracle FAIL and names the check. Every check is
individually runnable (``run_check``/``CHECKS``) so the red-team suite can
assert *which* check caught a given attack.
"""

from __future__ import annotations

import ast
import logging
import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..schema import OracleResult, Task
from .base import OracleContext, register_oracle

logger = logging.getLogger(__name__)

ORACLE_ID = "O3"

#: Order is the order they are reported in.
CHECK_NAMES: tuple[str, ...] = (
    "static_denylist",
    "randomized_inputs",
    "held_out_shapes",
    "output_liveness",
    "timing_sanity",
)

_MAX_HITS = 50
_DEFAULT_MAX_SHAPES = 3
_DEFAULT_REL_TOL = 1e-4
_DEFAULT_ABS_TOL = 1e-6


# --------------------------------------------------------------------------- #
# result types
# --------------------------------------------------------------------------- #


@dataclass
class CheckResult:
    """One anti-cheat check's own verdict. ``FAIL`` means the check fired."""

    name: str
    verdict: str  # PASS | FAIL | SKIP | ERROR
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def fired(self) -> bool:
        return self.verdict == "FAIL"

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "verdict": self.verdict,
            "detail": self.detail,
            "duration_s": round(self.duration_s, 6),
            **self.evidence,
        }


@dataclass(frozen=True)
class DenylistHit:
    """One static finding, with the exact node and line that produced it."""

    kind: str  # denylist | dynamic_attribute | dynamic_import | dynamic_exec | star_import
    symbol: str
    matched: str
    line: int
    col: int
    node: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 1. static denylist - AST, with alias resolution
# --------------------------------------------------------------------------- #

_DYNAMIC_IMPORT_CALLS = frozenset({"__import__", "importlib.import_module"})
_DYNAMIC_EXEC_CALLS = frozenset({"eval", "exec", "compile", "builtins.eval", "builtins.exec"})


def _dotted_name(node: ast.AST) -> str | None:
    """``torch.nn.functional.relu`` -> that string; non-Name-rooted -> None."""
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _attr_chain(node: ast.Attribute) -> tuple[list[str], ast.AST]:
    """(attribute names outermost-last, the non-Attribute base expression)."""
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    parts.reverse()
    return parts, cur


def _literal_str(node: ast.AST) -> str | None:
    """Fold a statically knowable string: ``'mat' + 'mul'`` -> ``'matmul'``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_str(node.left)
        right = _literal_str(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        out: list[str] = []
        for value in node.values:
            piece = _literal_str(value)
            if piece is None:
                return None
            out.append(piece)
        return "".join(out)
    try:
        folded = ast.literal_eval(node)  # type: ignore[arg-type]
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    return folded if isinstance(folded, str) else None


class _StaticAnalyzer(ast.NodeVisitor):
    """Resolve aliases, then match every resolvable name against the denylist."""

    def __init__(self, source: str, denylist: Sequence[str]) -> None:
        self.source = source
        self.denylist = tuple(e.strip() for e in denylist if str(e).strip())
        self.aliases: dict[str, str] = {}
        #: module-level NAME = "literal" bindings, so a dtype looked up by a
        #: named constant folds to a concrete symbol instead of reading as an
        #: evasion attempt
        self.str_consts: dict[str, str] = {}
        #: dynamic lookups that folded to a known-allowed symbol; reported as
        #: context, never as a failure
        self.resolved_safe: list[tuple[str, int]] = []
        self.hits: list[DenylistHit] = []

    # -- alias collection ---------------------------------------------------
    def collect_aliases(self, tree: ast.AST) -> None:
        for node in getattr(tree, "body", []):
            # Only module scope: a name rebound inside a function is not a
            # constant, and treating it as one would let a real evasion hide
            # behind a local shadow.
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                value = _literal_str(node.value)
                if isinstance(target, ast.Name) and value is not None:
                    self.str_consts[target.id] = value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                value = _literal_str(node.value) if node.value is not None else None
                if value is not None:
                    self.str_consts[node.target.id] = value
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.aliases[alias.asname] = alias.name
                    else:
                        head = alias.name.split(".")[0]
                        self.aliases.setdefault(head, head)
            elif isinstance(node, ast.ImportFrom):
                base = "." * int(node.level or 0) + (node.module or "")
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    full = f"{base}.{alias.name}" if base and not base.endswith(".") else f"{base}{alias.name}"
                    self.aliases[alias.asname or alias.name] = full

    # -- helpers ------------------------------------------------------------
    def _resolve(self, dotted: str) -> str:
        head, sep, rest = dotted.partition(".")
        target = self.aliases.get(head)
        if target is None:
            return dotted
        return f"{target}.{rest}" if sep and rest else target

    def _match(self, dotted: str) -> str | None:
        for entry in self.denylist:
            if dotted == entry or dotted.startswith(entry + "."):
                return entry
        return None

    def _segment(self, node: ast.AST) -> str:
        try:
            seg = ast.get_source_segment(self.source, node)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            seg = None
        if seg:
            return seg.strip()[:200]
        return type(node).__name__

    def _add(self, kind: str, symbol: str, matched: str, node: ast.AST, detail: str = "") -> None:
        self.hits.append(
            DenylistHit(
                kind=kind,
                symbol=symbol,
                matched=matched,
                line=int(getattr(node, "lineno", 0)),
                col=int(getattr(node, "col_offset", 0)),
                node=self._segment(node),
                detail=detail,
            )
        )

    # -- visitors -----------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            matched = self._match(alias.name)
            if matched:
                self._add(
                    "denylist",
                    alias.name,
                    matched,
                    node,
                    detail=f"imported as {alias.asname or alias.name.split('.')[0]!r}",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = "." * int(node.level or 0) + (node.module or "")
        for alias in node.names:
            if alias.name == "*":
                self._add(
                    "star_import",
                    f"from {base} import *",
                    self._match(base) or "",
                    node,
                    detail="a star import hides which symbols were bound; names cannot be resolved",
                )
                continue
            full = f"{base}.{alias.name}" if base and not base.endswith(".") else f"{base}{alias.name}"
            matched = self._match(full) or self._match(base)
            if matched:
                self._add(
                    "denylist",
                    full,
                    matched,
                    node,
                    detail=f"bound to local name {alias.asname or alias.name!r}",
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        dotted = _dotted_name(node)
        if dotted is not None:
            resolved = self._resolve(dotted)
            matched = self._match(resolved)
            if matched:
                detail = "" if resolved == dotted else f"alias-resolved from {dotted!r}"
                self._add("denylist", resolved, matched, node, detail=detail)
            return  # the whole chain is one name; do not re-report its head

        parts, base = _attr_chain(node)
        module = self._dynamic_import_module(base)
        if module is not None:
            resolved = ".".join([module, *parts])
            matched = self._match(resolved)
            self._add(
                "denylist" if matched else "dynamic_import",
                resolved,
                matched or "",
                node,
                detail="attribute reached through a dynamic import",
            )
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        resolved = self._resolve(node.id)
        matched = self._match(resolved)
        if matched:
            detail = "" if resolved == node.id else f"alias-resolved from {node.id!r}"
            self._add("denylist", resolved, matched, node, detail=detail)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        raw = _dotted_name(node.func)
        callee = self._resolve(raw) if raw else None

        if callee in ("getattr", "builtins.getattr") and len(node.args) >= 2:
            self._handle_getattr(node)
            self.generic_visit(node)
            return

        if callee in _DYNAMIC_IMPORT_CALLS:
            module = _literal_str(node.args[0]) if node.args else None
            symbol = module if module else "<dynamic module name>"
            matched = self._match(module) if module else None
            self._add(
                "denylist" if matched else "dynamic_import",
                symbol,
                matched or "",
                node,
                detail=f"dynamic import via {raw}()",
            )
            self.generic_visit(node)
            return

        if callee in _DYNAMIC_EXEC_CALLS:
            self._add(
                "dynamic_exec",
                str(callee),
                "",
                node,
                detail="runtime code construction defeats any static policy",
            )
            self.generic_visit(node)
            return

        self.generic_visit(node)

    def _handle_getattr(self, node: ast.Call) -> None:
        target_raw = _dotted_name(node.args[0])
        target = self._resolve(target_raw) if target_raw else None
        if target is None:
            parts, base = _attr_chain(node.args[0]) if isinstance(node.args[0], ast.Attribute) else ([], node.args[0])
            module = self._dynamic_import_module(base)
            if module is not None:
                target = ".".join([module, *parts])
        attr = _literal_str(node.args[1])
        if attr is None and isinstance(node.args[1], ast.Name):
            # getattr(torch, PARAM_SAVE_DTYPE) where the constant is a
            # module-level string. Resolving it turns a blanket
            # "dynamic attribute" accusation into a real denylist decision:
            # the honest checkpointing seed looks up a dtype this way, and
            # flagging that rejected known-correct code.
            attr = self.str_consts.get(node.args[1].id)
        if target is not None and attr is not None:
            resolved = f"{target}.{attr}"
            matched = self._match(resolved)
            if matched:
                self._add(
                    "denylist",
                    resolved,
                    matched,
                    node,
                    detail="name assembled at runtime and resolved statically by constant folding",
                )
            else:
                # Resolved statically and not on the denylist. Dynamic access is
                # only suspicious because it can hide a banned symbol; once the
                # symbol is known and allowed there is nothing hidden, and this
                # is exactly as safe as spelling it out. Recording it as a
                # finding rejected the honest baseline.
                self.resolved_safe.append((resolved, node.lineno))
            return
        shown = target or (self._segment(node.args[0]) if node.args else "?")
        self._add(
            "dynamic_attribute",
            f"getattr({shown}, <dynamic>)",
            "",
            node,
            detail="attribute name is not statically knowable; no static policy can bind it",
        )

    def _dynamic_import_module(self, node: ast.AST) -> str | None:
        """``__import__('torch')`` / ``importlib.import_module('torch')`` -> 'torch'."""
        if not isinstance(node, ast.Call):
            return None
        raw = _dotted_name(node.func)
        callee = self._resolve(raw) if raw else None
        if callee not in _DYNAMIC_IMPORT_CALLS:
            return None
        module = _literal_str(node.args[0]) if node.args else None
        return module if module else "<dynamic module name>"


def analyze_source(source: str, denylist: Sequence[str] = ()) -> tuple[list[DenylistHit], dict[str, str]]:
    """Static analysis of one candidate. Raises ``SyntaxError`` if it will not parse.

    Returns (hits, alias map). Exposed so the red-team suite can assert on the
    exact node that was flagged.
    """
    tree = ast.parse(source)
    analyzer = _StaticAnalyzer(source, denylist)
    analyzer.collect_aliases(tree)
    analyzer.visit(tree)
    seen: set[tuple[str, str, int, int]] = set()
    unique: list[DenylistHit] = []
    for hit in analyzer.hits:
        key = (hit.kind, hit.symbol, hit.line, hit.col)
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    unique.sort(key=lambda h: (h.line, h.col, h.symbol))
    return unique, dict(analyzer.aliases)


def check_static_denylist(ctx: OracleContext) -> CheckResult:
    """Resolve names statically and match them against ``seed.denylist``."""
    started = time.perf_counter()
    denylist = tuple(getattr(ctx.seed, "denylist", ()) or ())
    try:
        hits, aliases = analyze_source(ctx.candidate_src, denylist)
    except SyntaxError as exc:
        return CheckResult(
            name="static_denylist",
            verdict="ERROR",
            detail=f"candidate source does not parse: {exc.msg} at line {exc.lineno}",
            evidence={"syntax_error": {"msg": exc.msg, "line": exc.lineno, "offset": exc.offset}},
            duration_s=time.perf_counter() - started,
        )

    evidence: dict[str, Any] = {
        "denylist": list(denylist),
        "denylist_empty": not denylist,
        "aliases": aliases,
        "hits": [h.as_dict() for h in hits[:_MAX_HITS]],
        "hit_count": len(hits),
    }
    duration = time.perf_counter() - started
    if hits:
        first = hits[0]
        by_kind: dict[str, int] = {}
        for h in hits:
            by_kind[h.kind] = by_kind.get(h.kind, 0) + 1
        evidence["hits_by_kind"] = by_kind
        detail = (
            f"{len(hits)} static finding(s); first: {first.kind} {first.symbol!r} "
            f"at line {first.line} col {first.col} ({first.node!r})"
        )
        if first.matched:
            detail += f" matches denylist entry {first.matched!r}"
        return CheckResult("static_denylist", "FAIL", detail, evidence, duration)
    return CheckResult(
        name="static_denylist",
        verdict="PASS",
        detail=(
            f"no denylisted symbol and no dynamic name construction; "
            f"{len(aliases)} import alias(es) resolved"
        ),
        evidence=evidence,
        duration_s=duration,
    )


# --------------------------------------------------------------------------- #
# shared execution machinery
# --------------------------------------------------------------------------- #


@dataclass
class _RunOutcome:
    ok: bool
    checksums: dict[str, str] = field(default_factory=dict)
    arrays: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_s: float = 0.0
    input_seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checksums": dict(self.checksums),
            "error": self.error,
            "duration_s": round(self.duration_s, 6),
            "input_seed": self.input_seed,
        }


def _flatten(value: Any) -> list[tuple[str, Any]]:
    """Mirror of the sandbox child's output naming, so names line up."""
    if isinstance(value, dict):
        return [(str(k), v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))]
    if isinstance(value, (list, tuple)):
        return [(f"out{i}", v) for i, v in enumerate(value)]
    return [("out", value)]


def _to_numpy(value: Any) -> Any | None:
    import numpy as np

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        return t.contiguous().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bool, int, float)):
        return np.asarray(value)
    return None


def _arrays_equal(a: Any, b: Any) -> bool:
    import numpy as np

    if a is None or b is None:
        return False
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(a, b))


def _accepted_kwargs(fn: Callable[..., Any], wanted: Mapping[str, Any]) -> dict[str, Any]:
    """Pass only the keyword arguments this seed's callable actually declares."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(wanted)
    return {k: v for k, v in wanted.items() if k in params}


def _make_inputs(ctx: OracleContext, shape: Any, rng_seed: int) -> dict[str, Any]:
    """Regenerate the inputs from ``rng_seed``. Fresh every grading invocation."""
    import torch

    from ..runner.determinism import seed_everything

    seed_everything(int(rng_seed))
    generator: Any = None
    try:
        generator = torch.Generator(device=ctx.device)
        generator.manual_seed(int(rng_seed))
    except (RuntimeError, TypeError) as exc:
        logger.debug("could not build a torch generator on %s: %s", ctx.device, exc)
        generator = None
    kwargs = _accepted_kwargs(
        ctx.seed.make_inputs, {"device": ctx.device, "generator": generator}
    )
    inputs = ctx.seed.make_inputs(shape, **kwargs)
    if not isinstance(inputs, dict):
        raise TypeError(
            f"seed {ctx.seed.id!r}: make_inputs returned {type(inputs).__name__}, expected a dict"
        )
    return inputs


def _reference(ctx: OracleContext, inputs: Mapping[str, Any]) -> Any:
    try:
        return ctx.seed.reference(**inputs)
    except TypeError:
        return ctx.seed.reference(*[inputs[k] for k in inputs])


def _run_candidate(
    ctx: OracleContext,
    source: str,
    inputs: Mapping[str, Any],
    tag: str,
    input_seed: int,
) -> _RunOutcome:
    """Execute the candidate in a sandbox subprocess. Never in this interpreter."""
    import numpy as np

    from ..runner.sandbox import call_entry

    started = time.perf_counter()
    res = call_entry(
        source,
        ctx.seed.entry,
        inputs=dict(inputs),
        workdir=ctx.sub_workdir(f"o3-{tag}"),
        device=ctx.device,
        timeout_s=float(ctx.cfg.sandbox_timeout_s),
        seed=int(input_seed),
        deterministic=True,
        save_outputs=True,
    )
    if not res.ok:
        return _RunOutcome(
            ok=False,
            error=res.message[:2000],
            duration_s=time.perf_counter() - started,
            input_seed=int(input_seed),
        )
    arrays: dict[str, Any] = {}
    for rec in res.outputs():
        path = rec.get("path")
        if rec.get("kind") != "array" or not path:
            continue
        try:
            arrays[str(rec["name"])] = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            logger.debug("could not load candidate output %s: %s", path, exc)
    return _RunOutcome(
        ok=True,
        checksums=res.checksums(),
        arrays=arrays,
        duration_s=time.perf_counter() - started,
        input_seed=int(input_seed),
    )


def _derived_tolerance(ctx: OracleContext, shape: Any, ref: Any) -> Any | None:
    """The SAME error budget O1 grades against, for this shape.

    O3 previously compared against a fixed 1e-4/1e-6. At fp16 with a
    129-deep accumulation the honest budget is 2.2e-2, so the constant was
    222x too tight and O3 rejected solutions O1 accepted -- two oracles
    disagreeing about what "correct" means, with the anti-cheat one accusing
    correct code. A hardcoded tolerance is exactly what this project forbids,
    so the budget is derived here the way O1 derives it and by the same
    config knobs.
    """
    try:
        from .tolerance import derive, from_tensor_scale, magnitude_of
    except ImportError:
        return None
    kwargs = getattr(shape, "kwargs", None) or {}
    dtype = kwargs.get("dtype") or "float32"
    depth_fn = getattr(ctx.seed, "accum_depth", None)
    try:
        depth = int(depth_fn(shape)) if callable(depth_fn) else 1
    except Exception:  # noqa: BLE001 - a seed defect must not decide correctness
        depth = 1
    try:
        tol = derive(
            dtype,
            depth,
            mode=ctx.cfg.tolerance_mode,
            safety=ctx.cfg.tolerance_safety,
        )
        return from_tensor_scale(tol, magnitude_of([arr for _n, arr in _flatten(ref)]))
    except Exception:  # noqa: BLE001
        return None


def _default_compare(got: Any, want: Any, tol: Any | None = None) -> tuple[bool, float, str]:
    import numpy as np

    if got is None:
        return False, float("inf"), "candidate produced no array for this output"
    w = np.asarray(want)
    g = np.asarray(got)
    if g.shape != w.shape:
        return False, float("inf"), f"shape {tuple(g.shape)} != reference {tuple(w.shape)}"
    if not np.issubdtype(w.dtype, np.floating) and not np.issubdtype(g.dtype, np.floating):
        equal = bool(np.array_equal(g, w))
        return equal, 0.0 if equal else float("inf"), "" if equal else "exact comparison failed"
    gf = g.astype(np.float64)
    wf = w.astype(np.float64)
    abs_err = np.abs(gf - wf)
    rel = abs_err / np.maximum(np.abs(wf), 1e-12)
    max_rel = float(rel.max()) if rel.size else 0.0
    if tol is not None:
        # Delegate to the ONE comparison O1 grades with. Sharing only the
        # tolerance number was not enough: this function's own formula flagged
        # layernorm outputs whose value sits near zero, where a tiny absolute
        # error is a huge relative one. O1 passed those same shapes because
        # tolerance.compare() handles near-zero references, empty outputs and
        # non-finite values deliberately. Two implementations of "correct" is
        # one too many.
        from .tolerance import compare as _tol_compare

        res = _tol_compare(got, want, tol)
        return (
            bool(res.passed),
            float(res.max_rel_err),
            "" if res.passed else f"{res.detail} (budget: {tol.formula})",
        )
    ok = bool(np.all(abs_err <= _DEFAULT_ABS_TOL + _DEFAULT_REL_TOL * np.abs(wf)))
    return ok, max_rel, "" if ok else f"max_rel_err {max_rel:.3e} exceeds the fallback tolerance"


def _compare_to_reference(
    ctx: OracleContext, outcome: _RunOutcome, ref: Any, shape: Any | None = None
) -> tuple[bool, float, str]:
    """Compare a candidate run against the seed reference. (ok, max_rel_err, detail)."""
    compare = getattr(ctx.seed, "compare", None)
    tol = _derived_tolerance(ctx, shape, ref) if shape is not None else None
    worst_rel = 0.0
    pairs = list(_flatten(ref))

    if compare is not None:
        # A seed's compare() is written against the shape its entry actually
        # returns. Feeding it one flattened array at a time breaks that
        # contract: the checkpointing and collectives seeds return a dict and
        # their compare reports "expected a dict of outputs, got ndarray",
        # which O3 then read as the candidate being wrong. O1 passes the whole
        # structure, so O3 must too, or the two oracles judge different things.
        missing = [name for name, _ in pairs if outcome.arrays.get(name) is None]
        if missing:
            return False, float("inf"), f"candidate did not return array(s) {missing}"
        if isinstance(ref, Mapping):
            got_struct: Any = {name: outcome.arrays[name] for name, _ in pairs}
        elif len(pairs) == 1:
            got_struct = outcome.arrays[pairs[0][0]]
        else:
            got_struct = [outcome.arrays[name] for name, _ in pairs]
        try:
            res = compare(got_struct, ref)
        except Exception as exc:  # noqa: BLE001 - a seed defect must be reported, not hidden
            return False, float("inf"), f"seed.compare raised {type(exc).__name__}: {exc}"
        worst_rel = float(getattr(res, "max_rel_err", 0.0) or 0.0)
        if not bool(getattr(res, "ok", False)):
            return False, worst_rel, str(getattr(res, "detail", "") or "outside tolerance")
        return True, worst_rel, ""

    for name, want in pairs:
        got = outcome.arrays.get(name)
        if got is None:
            return False, float("inf"), f"candidate did not return an array named {name!r}"
        ok, rel, detail = _default_compare(got, _to_numpy(want), tol)
        worst_rel = max(worst_rel, rel if rel != float("inf") else worst_rel)
        if not ok:
            return False, rel, detail
    return True, worst_rel, ""


def _fresh_seeds(ctx: OracleContext, n: int) -> list[int]:
    """A fresh input seed per grading invocation, recorded so a run is reproducible."""
    override = ctx.extras.get("o3_input_seeds")
    if isinstance(override, (list, tuple)) and len(override) >= n:
        return [int(v) for v in list(override)[:n]]
    rng = random.SystemRandom()
    out: list[int] = []
    while len(out) < n:
        value = rng.randrange(1, 2**31 - 1)
        if value not in out:
            out.append(value)
    return out


def _grading_shapes(ctx: OracleContext, held_out: bool) -> list[Any]:
    task = ctx.task
    shapes = list(task.detect_shapes) if held_out else list(task.decoy_shapes)
    if not shapes and not held_out:
        shapes = list(task.detect_shapes)
    if not shapes:
        shapes = list(getattr(ctx.seed, "shape_sweep", []) or [])[:1]
    limit = int(ctx.extras.get("o3_max_shapes", _DEFAULT_MAX_SHAPES))
    return shapes[: max(1, limit)]


# --------------------------------------------------------------------------- #
# 2. randomized inputs
# --------------------------------------------------------------------------- #


def _outputs_equal(a: Any, b: Any) -> bool:
    """Structural equality over whatever a seed's entry returns."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            return False
        return all(_outputs_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_outputs_equal(x, y) for x, y in zip(a, b))
    return _arrays_equal(_to_numpy(a), _to_numpy(b))


def _reference_discriminates(ctx: OracleContext, shape: Any, seeds: Sequence[int]) -> bool:
    """Does the KNOWN-CORRECT reference give different output for different inputs?

    The randomisation check concludes "the result does not depend on the input"
    from identical output across two input draws. That inference is only valid
    on a shape where the output is supposed to vary. Degenerate shapes break it:
    zero-context attention returns zeros for every input, so the reference
    itself is byte-identical across seeds and an honest solution gets accused of
    memorising. Using the reference as the control makes the check self-checking
    -- it can only fire where a real dependence exists to be lost.
    """
    try:
        a = _reference(ctx, _make_inputs(ctx, shape, int(seeds[0])))
        b = _reference(ctx, _make_inputs(ctx, shape, int(seeds[1])))
    except Exception:  # noqa: BLE001 - an unusable shape simply cannot be the control
        return False
    return not _outputs_equal(a, b)


def _discriminating_shape(
    ctx: OracleContext, seeds: Sequence[int]
) -> tuple[Any | None, list[str]]:
    """First shape whose reference output actually depends on the input."""
    candidates: list[Any] = []
    seen: set[str] = set()
    for group in (
        _grading_shapes(ctx, held_out=False),
        list(getattr(ctx.task, "detect_shapes", []) or []),
        list(getattr(ctx.seed, "shape_sweep", []) or []),
    ):
        for shape in group:
            key = str(shape)
            if key not in seen:
                seen.add(key)
                candidates.append(shape)
    rejected: list[str] = []
    for shape in candidates:
        if _reference_discriminates(ctx, shape, seeds):
            return shape, rejected
        rejected.append(getattr(shape, "name", str(shape)))
    return None, rejected


def check_randomized_inputs(ctx: OracleContext) -> CheckResult:
    """Run twice with two freshly drawn input seeds and require both to hold."""
    started = time.perf_counter()
    if not _grading_shapes(ctx, held_out=False):
        return CheckResult(
            "randomized_inputs",
            "SKIP",
            "neither the task nor the seed offers a shape to generate inputs for",
            {"shapes": []},
            time.perf_counter() - started,
        )
    seeds = _fresh_seeds(ctx, 2)
    shape, skipped_shapes = _discriminating_shape(ctx, seeds)
    if shape is None:
        return CheckResult(
            "randomized_inputs",
            "SKIP",
            (
                "no available shape has a reference output that depends on the input "
                f"(tried {skipped_shapes}); input randomisation cannot be exercised here "
                "and identical output would prove nothing"
            ),
            {"shapes_rejected_as_degenerate": skipped_shapes, "input_seeds": seeds},
            time.perf_counter() - started,
        )

    runs: list[dict[str, Any]] = []
    outcomes: list[_RunOutcome] = []
    input_arrays: list[dict[str, Any]] = []
    oks: list[bool] = []
    for index, input_seed in enumerate(seeds):
        inputs = _make_inputs(ctx, shape, input_seed)
        input_arrays.append({k: _to_numpy(v) for k, v in inputs.items()})
        ref = _reference(ctx, inputs)
        outcome = _run_candidate(ctx, ctx.candidate_src, inputs, f"rand{index}-{input_seed}", input_seed)
        outcomes.append(outcome)
        if outcome.ok:
            ok, rel, detail = _compare_to_reference(ctx, outcome, ref, shape)
        else:
            ok, rel, detail = False, float("inf"), f"candidate did not run: {outcome.error}"
        oks.append(ok)
        record = outcome.as_dict()
        record.update(
            {
                "shape": str(shape),
                "agrees_with_reference": ok,
                "max_rel_err": None if rel == float("inf") else rel,
                "detail": detail,
            }
        )
        runs.append(record)

    inputs_differed = not all(
        _arrays_equal(input_arrays[0].get(k), input_arrays[1].get(k))
        for k in set(input_arrays[0]) | set(input_arrays[1])
    )
    identical_checksums = bool(
        outcomes[0].ok
        and outcomes[1].ok
        and outcomes[0].checksums
        and outcomes[0].checksums == outcomes[1].checksums
    )
    evidence: dict[str, Any] = {
        "shape": str(shape),
        "shapes_rejected_as_degenerate": skipped_shapes,
        "input_seeds": seeds,
        "inputs_differed": inputs_differed,
        "identical_checksums": identical_checksums,
        "runs": runs,
        "agreement": (
            "both_agree" if all(oks) else "disagree" if any(oks) else "neither_agrees"
        ),
    }
    duration = time.perf_counter() - started

    if not inputs_differed:
        return CheckResult(
            "randomized_inputs",
            "SKIP",
            (
                f"seed {ctx.seed.id!r}: make_inputs produced identical inputs for seeds "
                f"{seeds[0]} and {seeds[1]}, so input randomisation could not be exercised"
            ),
            evidence,
            duration,
        )
    if identical_checksums:
        return CheckResult(
            "randomized_inputs",
            "FAIL",
            (
                f"output is byte-identical for two different randomly seeded inputs "
                f"(seeds {seeds[0]} and {seeds[1]}): the result does not depend on the input"
            ),
            evidence,
            duration,
        )
    if all(oks):
        return CheckResult(
            "randomized_inputs",
            "PASS",
            f"agrees with the reference on both freshly seeded input sets {seeds}",
            evidence,
            duration,
        )
    if any(oks):
        good = seeds[oks.index(True)]
        bad = seeds[oks.index(False)]
        return CheckResult(
            "randomized_inputs",
            "FAIL",
            (
                f"correct on input seed {good} but not on input seed {bad}: correctness "
                f"depends on which inputs were drawn, which is the signature of memorised inputs"
            ),
            evidence,
            duration,
        )
    return CheckResult(
        "randomized_inputs",
        "FAIL",
        (
            f"disagrees with the reference on both freshly seeded input sets {seeds}; "
            f"first detail: {runs[0]['detail']}"
        ),
        evidence,
        duration,
    )


# --------------------------------------------------------------------------- #
# 3. held-out shapes
# --------------------------------------------------------------------------- #


def check_held_out_shapes(ctx: OracleContext) -> CheckResult:
    """Grade on ``task.detect_shapes`` - the shapes the prompt never carried."""
    started = time.perf_counter()
    shapes = list(ctx.task.detect_shapes)
    if not shapes:
        return CheckResult(
            "held_out_shapes",
            "SKIP",
            "task carries no detect_shapes; there is no withheld shape to grade on",
            {"detect_shapes": [], "decoy_shapes": [str(s) for s in ctx.task.decoy_shapes]},
            time.perf_counter() - started,
        )
    limit = int(ctx.extras.get("o3_max_shapes", _DEFAULT_MAX_SHAPES))
    shapes = shapes[: max(1, limit)]
    seeds = _fresh_seeds(ctx, len(shapes))

    per_shape: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, (shape, input_seed) in enumerate(zip(shapes, seeds)):
        inputs = _make_inputs(ctx, shape, input_seed)
        ref = _reference(ctx, inputs)
        outcome = _run_candidate(ctx, ctx.candidate_src, inputs, f"held{index}-{input_seed}", input_seed)
        if outcome.ok:
            ok, rel, detail = _compare_to_reference(ctx, outcome, ref, shape)
        else:
            ok, rel, detail = False, float("inf"), f"candidate did not run: {outcome.error}"
        per_shape.append(
            {
                "shape": str(shape),
                "shape_name": getattr(shape, "name", "?"),
                "kwargs": dict(getattr(shape, "kwargs", {}) or {}),
                "input_seed": input_seed,
                "ok": ok,
                "max_rel_err": None if rel == float("inf") else rel,
                "detail": detail,
            }
        )
        if not ok:
            failures.append(f"{getattr(shape, 'name', '?')}: {detail}")

    evidence = {
        "held_out": [str(s) for s in shapes],
        "shown_in_prompt": [str(s) for s in ctx.task.decoy_shapes],
        "per_shape": per_shape,
        "n_failed": len(failures),
    }
    duration = time.perf_counter() - started
    if failures:
        return CheckResult(
            "held_out_shapes",
            "FAIL",
            (
                f"{len(failures)} of {len(shapes)} withheld shape(s) disagree with the reference "
                f"while the prompt only showed {[str(s) for s in ctx.task.decoy_shapes]}; "
                f"first: {failures[0]}"
            ),
            evidence,
            duration,
        )
    return CheckResult(
        "held_out_shapes",
        "PASS",
        f"agrees with the reference on all {len(shapes)} withheld shape(s)",
        evidence,
        duration,
    )


# --------------------------------------------------------------------------- #
# 4. output liveness
# --------------------------------------------------------------------------- #

#: Appended (never prepended - a leading ``from __future__`` import must stay
#: first) to the candidate so that "uninitialised" memory has a known content.
#: Both the torch/numpy entry points and any module-level alias the candidate
#: bound at import time are rebound, so ``from torch import empty`` is covered.
_POISON_SUFFIX = '''

# --- appended by CRUCIBLE O3 (output_liveness): poison uninitialised memory ---
def _crucible_install_poison(_float_fill, _int_fill):
    _installed = []
    _g = globals()

    def _fill_tensor(_r):
        try:
            _r.fill_(_float_fill)
        except (RuntimeError, TypeError, ValueError):
            try:
                _r.fill_(_int_fill)
            except (RuntimeError, TypeError, ValueError):
                pass
        return _r

    def _wrap_torch(_f):
        def _wrapper(*_a, **_k):
            return _fill_tensor(_f(*_a, **_k))
        return _wrapper

    def _fill_array(_r):
        try:
            _r.fill(_float_fill)
        except (TypeError, ValueError, AttributeError):
            try:
                _r.fill(_int_fill)
            except (TypeError, ValueError, AttributeError):
                pass
        return _r

    def _wrap_numpy(_f):
        def _wrapper(*_a, **_k):
            return _fill_array(_f(*_a, **_k))
        return _wrapper

    try:
        import torch as _t
    except ImportError:
        _t = None
    if _t is not None:
        for _name in ("empty", "empty_like", "empty_strided"):
            _orig = getattr(_t, _name, None)
            if _orig is None:
                continue
            _new = _wrap_torch(_orig)
            setattr(_t, _name, _new)
            _installed.append("torch." + _name)
            for _k, _v in list(_g.items()):
                if _v is _orig:
                    _g[_k] = _new
        try:
            _t.Tensor.new_empty = _wrap_torch(_t.Tensor.new_empty)
            _installed.append("torch.Tensor.new_empty")
        except (AttributeError, TypeError):
            pass
    try:
        import numpy as _np
    except ImportError:
        _np = None
    if _np is not None:
        for _name in ("empty", "empty_like"):
            _orig = getattr(_np, _name, None)
            if _orig is None:
                continue
            _new = _wrap_numpy(_orig)
            setattr(_np, _name, _new)
            _installed.append("numpy." + _name)
            for _k, _v in list(_g.items()):
                if _v is _orig:
                    _g[_k] = _new
    return _installed


_CRUCIBLE_POISON_INSTALLED = _crucible_install_poison({float_fill}, {int_fill})
'''

_POISON_FILLS: tuple[tuple[str, str, str], ...] = (
    ("nan", "float('nan')", "-2147483648"),
    ("sentinel", "-98765.4375", "1234567"),
)


def poisoned_source(candidate_src: str, float_fill: str, int_fill: str) -> str:
    """Candidate source plus the uninitialised-memory poison installer."""
    return candidate_src + _POISON_SUFFIX.format(float_fill=float_fill, int_fill=int_fill)


def _perturb(inputs: Mapping[str, Any], scale: float = 1.0) -> tuple[dict[str, Any], str]:
    """Change exactly one element of the largest floating input. (inputs, what)."""
    import torch

    best_name: str | None = None
    best_size = -1
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.numel() > best_size:
            best_name = name
            best_size = value.numel()
    if best_name is None or best_size <= 0:
        return dict(inputs), ""
    out = dict(inputs)
    t = out[best_name].detach().clone().contiguous()
    flat = t.reshape(-1)
    try:
        flat[0] = flat[0] + scale
    except RuntimeError:
        flat[0] = flat[0] + int(max(1.0, scale))
    out[best_name] = t
    return out, f"{best_name}[0] += {scale}"


def check_output_liveness(ctx: OracleContext) -> CheckResult:
    """Three sub-parts: input sensitivity, verbatim passthrough, uninitialised memory."""
    started = time.perf_counter()
    shapes = _grading_shapes(ctx, held_out=True)
    if not shapes:
        return CheckResult(
            "output_liveness",
            "SKIP",
            "no shape available to build inputs for a liveness probe",
            {},
            time.perf_counter() - started,
        )
    shape = shapes[0]
    input_seed = _fresh_seeds(ctx, 1)[0]
    inputs = _make_inputs(ctx, shape, input_seed)
    ref = _reference(ctx, inputs)

    base = _run_candidate(ctx, ctx.candidate_src, inputs, f"live-base-{input_seed}", input_seed)
    evidence: dict[str, Any] = {
        "shape": str(shape),
        "input_seed": input_seed,
        "base_run": base.as_dict(),
        "consumer": (
            "the sandbox child hashes every output byte, so the result must be "
            "materialised and cannot be dead-code-eliminated"
        ),
    }
    if not base.ok:
        return CheckResult(
            "output_liveness",
            "SKIP",
            f"candidate did not run on the liveness shape, so liveness cannot be probed: {base.error}",
            evidence,
            time.perf_counter() - started,
        )

    fired: list[str] = []
    skipped: list[str] = []

    # -- (a) input sensitivity ---------------------------------------------
    sens: dict[str, Any] = {"fired": False}
    perturbed, what = _perturb(inputs, 1.0)
    ref_perturbed = _reference(ctx, perturbed)
    ref_changed = not all(
        _arrays_equal(_to_numpy(a), _to_numpy(b))
        for (_, a), (_, b) in zip(_flatten(ref), _flatten(ref_perturbed))
    )
    if not what:
        sens["skip_reason"] = "no tensor input to perturb"
        skipped.append("input_sensitivity")
    elif not ref_changed:
        perturbed, what = _perturb(inputs, 8.0)
        ref_perturbed = _reference(ctx, perturbed)
        ref_changed = not all(
            _arrays_equal(_to_numpy(a), _to_numpy(b))
            for (_, a), (_, b) in zip(_flatten(ref), _flatten(ref_perturbed))
        )
    if what and not ref_changed:
        sens["skip_reason"] = (
            f"perturbing {what} does not change the reference output either, so an "
            f"unchanged candidate checksum would prove nothing"
        )
        skipped.append("input_sensitivity")
    elif what:
        pert = _run_candidate(ctx, ctx.candidate_src, perturbed, f"live-pert-{input_seed}", input_seed)
        sens["perturbation"] = what
        sens["perturbed_run"] = pert.as_dict()
        if not pert.ok:
            sens["skip_reason"] = f"perturbed run did not complete: {pert.error}"
            skipped.append("input_sensitivity")
        elif pert.checksums == base.checksums:
            sens["fired"] = True
            sens["detail"] = (
                f"output checksum {sorted(base.checksums.values())} is unchanged after {what}, "
                f"while the reference output does change: the work was elided or never done"
            )
            fired.append("input_sensitivity")
        else:
            sens["detail"] = f"checksum changed after {what}, as it must"
    evidence["input_sensitivity"] = sens

    # -- (b) verbatim passthrough ------------------------------------------
    passthrough: dict[str, Any] = {"fired": False}
    ref_by_name = {name: _to_numpy(value) for name, value in _flatten(ref)}
    input_arrays = {name: _to_numpy(value) for name, value in inputs.items()}
    matches: list[str] = []
    for out_name, out_arr in base.arrays.items():
        for in_name, in_arr in input_arrays.items():
            if not _arrays_equal(out_arr, in_arr):
                continue
            if _arrays_equal(ref_by_name.get(out_name), in_arr):
                continue  # the reference really is the identity here; not a finding
            matches.append(f"output {out_name!r} is byte-identical to input {in_name!r}")
    if matches:
        passthrough["fired"] = True
        passthrough["matches"] = matches
        passthrough["detail"] = matches[0] + ", but the reference output is not; no work was done"
        fired.append("passthrough")
    else:
        passthrough["detail"] = "no output is a verbatim copy of an input"
    evidence["passthrough"] = passthrough

    # -- (c) uninitialised memory ------------------------------------------
    uninit: dict[str, Any] = {"fired": False}
    poison_runs: list[dict[str, Any]] = []
    poisoned: list[_RunOutcome] = []
    for label, float_fill, int_fill in _POISON_FILLS:
        src = poisoned_source(ctx.candidate_src, float_fill, int_fill)
        outcome = _run_candidate(ctx, src, inputs, f"live-poison-{label}-{input_seed}", input_seed)
        poisoned.append(outcome)
        record = outcome.as_dict()
        record["fill"] = f"float={float_fill}, int={int_fill}"
        poison_runs.append(record)
    uninit["runs"] = poison_runs
    uninit["method"] = (
        "torch.empty/empty_like/empty_strided/Tensor.new_empty and numpy.empty/empty_like "
        "are wrapped to write a known fill, so any element the candidate leaves unwritten "
        "differs between the two runs"
    )
    if not all(o.ok for o in poisoned):
        broken = next(o for o in poisoned if not o.ok)
        uninit["skip_reason"] = f"a poisoned run did not complete: {broken.error}"
        skipped.append("uninitialised_memory")
    elif poisoned[0].checksums != poisoned[1].checksums:
        uninit["fired"] = True
        uninit["detail"] = (
            "the same call produced different bytes under two different uninitialised-memory "
            "fills: part of the output was never written, so the returned buffer is "
            "uninitialised memory"
        )
        fired.append("uninitialised_memory")
    else:
        uninit["detail"] = "identical output under both fills; every output element was written"
    evidence["uninitialised_memory"] = uninit

    evidence["fired_subchecks"] = fired
    evidence["skipped_subchecks"] = skipped
    duration = time.perf_counter() - started
    if fired:
        details = [str(evidence[name].get("detail", "")) for name in fired if name in evidence]
        first = details[0] if details else ""
        return CheckResult(
            "output_liveness",
            "FAIL",
            f"liveness sub-check(s) {fired} fired: {first}",
            evidence,
            duration,
        )
    if len(skipped) == 3:
        return CheckResult(
            "output_liveness",
            "SKIP",
            "; ".join(str(evidence[n].get("skip_reason", n)) for n in ("input_sensitivity", "passthrough", "uninitialised_memory") if n in evidence),
            evidence,
            duration,
        )
    return CheckResult(
        "output_liveness",
        "PASS",
        (
            "the output is checksummed, changes when an input element changes, is not a copy "
            "of an input, and is fully written"
            + (f" (skipped: {skipped})" if skipped else "")
        ),
        evidence,
        duration,
    )


# --------------------------------------------------------------------------- #
# 5. timing sanity
# --------------------------------------------------------------------------- #


def achievable_bandwidth(ctx: OracleContext) -> tuple[float | None, str]:
    """Measured achievable bandwidth in bytes/s, and where it came from.

    Never falls back to a spec-sheet number: an unmeasured denominator would
    turn a physical argument into a guess. Returns ``(None, reason)`` instead.
    """
    for key in ("achievable_bandwidth_bytes_per_s", "achievable_bandwidth"):
        value = ctx.extras.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value), f"ctx.extras[{key!r}] (measured by O2 and handed to O3)"

    try:
        from . import o2_perf  # noqa: PLC0415 - optional; O2 may not be written yet
    except ImportError as exc:
        return None, f"O2 perf module unavailable ({exc}); no measured bandwidth exists"

    for attr in (
        "measure_achievable_bandwidth",
        "achievable_bandwidth",
        "probe_bandwidth",
        "stream_triad_bandwidth",
    ):
        probe = getattr(o2_perf, attr, None)
        if not callable(probe):
            continue
        try:
            kwargs = _accepted_kwargs(probe, {"device": ctx.device, "caps": ctx.caps, "cfg": ctx.cfg})
            value = probe(**kwargs)
        except Exception as exc:  # noqa: BLE001 - a probe defect must not crash the oracle
            return None, f"O2 bandwidth probe {attr} raised {type(exc).__name__}: {exc}"
        if isinstance(value, (int, float)) and value > 0:
            return float(value), f"crucible.oracles.o2_perf.{attr}()"
        number = getattr(value, "bytes_per_s", None)
        if isinstance(number, (int, float)) and number > 0:
            return float(number), f"crucible.oracles.o2_perf.{attr}().bytes_per_s"
        return None, f"O2 bandwidth probe {attr} returned {value!r}, which is not a positive rate"

    return None, (
        "crucible.oracles.o2_perf exposes no bandwidth probe; refusing to substitute a "
        "spec-sheet bandwidth for a measured one"
    )


def check_timing_sanity(ctx: OracleContext) -> CheckResult:
    """Flag any measured time below ``bytes_moved / achievable_bandwidth``."""
    started = time.perf_counter()
    bandwidth, source = achievable_bandwidth(ctx)
    evidence: dict[str, Any] = {"bandwidth_source": source, "achievable_bandwidth_bytes_per_s": bandwidth}
    if bandwidth is None:
        return CheckResult(
            "timing_sanity",
            "SKIP",
            f"no measured achievable bandwidth: {source}",
            evidence,
            time.perf_counter() - started,
        )

    from ..runner.sandbox import time_entry

    shapes = _grading_shapes(ctx, held_out=True)
    if not shapes:
        return CheckResult(
            "timing_sanity",
            "SKIP",
            "no shape available to time",
            evidence,
            time.perf_counter() - started,
        )
    shapes = shapes[: max(1, int(ctx.extras.get("o3_timing_shapes", 1)))]
    seeds = _fresh_seeds(ctx, len(shapes))

    per_shape: list[dict[str, Any]] = []
    violations: list[str] = []
    skipped: list[str] = []
    for index, (shape, input_seed) in enumerate(zip(shapes, seeds)):
        try:
            moved = int(ctx.seed.bytes_moved(shape))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            skipped.append(f"{getattr(shape, 'name', '?')}: bytes_moved unavailable ({exc})")
            continue
        if moved <= 0:
            skipped.append(f"{getattr(shape, 'name', '?')}: bytes_moved returned {moved}")
            continue
        floor_s = moved / bandwidth
        inputs = _make_inputs(ctx, shape, input_seed)
        res = time_entry(
            ctx.candidate_src,
            ctx.seed.entry,
            inputs=inputs,
            workdir=ctx.sub_workdir(f"o3-time{index}-{input_seed}"),
            device=ctx.device,
            reps=int(ctx.cfg.perf_reps),
            warmup=int(ctx.cfg.perf_warmup),
            timeout_s=float(max(ctx.cfg.sandbox_timeout_s, 60.0)),
            seed=int(input_seed),
        )
        if not res.ok or not isinstance(res.value, dict):
            skipped.append(f"{getattr(shape, 'name', '?')}: could not be timed ({res.message})")
            continue
        median_s = float(res.value.get("median_s") or 0.0)
        min_s = float(res.value.get("min_s") or 0.0)
        record = {
            "shape": str(shape),
            "bytes_moved": moved,
            "floor_s": floor_s,
            "median_s": median_s,
            "min_s": min_s,
            "ratio_median_over_floor": (median_s / floor_s) if floor_s else None,
            "reps": res.value.get("reps"),
            "statistic": "median",
        }
        per_shape.append(record)
        # The median, not the minimum: a single sub-resolution timer reading is a
        # measurement artifact, whereas a median below the floor is a claim that
        # the memory traffic never happened.
        if median_s < floor_s:
            violations.append(
                f"{getattr(shape, 'name', '?')}: median {median_s:.6e}s is below the "
                f"{floor_s:.6e}s floor implied by {moved} bytes at {bandwidth:.3e} B/s"
            )
    evidence["per_shape"] = per_shape
    evidence["skipped_shapes"] = skipped
    duration = time.perf_counter() - started

    if violations:
        return CheckResult(
            "timing_sanity",
            "FAIL",
            (
                f"{len(violations)} shape(s) ran faster than the bandwidth floor, so the "
                f"work was not done: {violations[0]}"
            ),
            evidence,
            duration,
        )
    if not per_shape:
        return CheckResult(
            "timing_sanity",
            "SKIP",
            "no shape could be timed: " + ("; ".join(skipped) or "no shapes"),
            evidence,
            duration,
        )
    return CheckResult(
        "timing_sanity",
        "PASS",
        (
            f"every timed shape is above its bytes_moved/{bandwidth:.3e} B/s floor "
            f"(bandwidth from {source})"
        ),
        evidence,
        duration,
    )


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

CHECKS: dict[str, Callable[[OracleContext], CheckResult]] = {
    "static_denylist": check_static_denylist,
    "randomized_inputs": check_randomized_inputs,
    "held_out_shapes": check_held_out_shapes,
    "output_liveness": check_output_liveness,
    "timing_sanity": check_timing_sanity,
}


def run_check(name: str, ctx: OracleContext) -> CheckResult:
    """Run one named check, converting an unexpected exception into ERROR."""
    fn = CHECKS.get(name)
    if fn is None:
        raise ValueError(f"unknown anti-cheat check {name!r}; known checks: {sorted(CHECKS)}")
    started = time.perf_counter()
    try:
        return fn(ctx)
    except Exception as exc:  # noqa: BLE001 - reported as ERROR, never swallowed
        return CheckResult(
            name=name,
            verdict="ERROR",
            detail=f"{type(exc).__name__}: {exc}",
            evidence={"traceback": traceback.format_exc()[-4000:]},
            duration_s=time.perf_counter() - started,
        )


def run_checks(ctx: OracleContext, names: Iterable[str] | None = None) -> list[CheckResult]:
    requested = list(names) if names is not None else list(CHECK_NAMES)
    return [run_check(name, ctx) for name in requested]


class AntiCheatOracle:
    """O3. Five independent checks; any one firing fails the task."""

    id = ORACLE_ID
    name = "anti-cheat"
    required_caps: tuple[str, ...] = ()

    def applies_to(self, task: Task) -> bool:
        return True

    def run(self, ctx: OracleContext) -> OracleResult:
        started = time.perf_counter()
        results = run_checks(ctx, ctx.extras.get("o3_checks"))
        checks = {r.name: r.as_dict() for r in results}
        fired = [r.name for r in results if r.verdict == "FAIL"]
        errored = [r.name for r in results if r.verdict == "ERROR"]
        skipped = {r.name: r.detail for r in results if r.verdict == "SKIP"}
        passed = [r.name for r in results if r.verdict == "PASS"]
        evidence: dict[str, Any] = {
            "checks": checks,
            "fired": fired,
            "errored": errored,
            "skipped": skipped,
            "passed": passed,
        }
        duration = time.perf_counter() - started

        if fired:
            details = "; ".join(f"{r.name}: {r.detail}" for r in results if r.verdict == "FAIL")
            return OracleResult(
                oracle=ORACLE_ID,
                verdict="FAIL",
                reason=f"anti-cheat check(s) {fired} fired -- {details}",
                evidence=evidence,
                duration_s=duration,
            )
        if errored:
            details = "; ".join(f"{r.name}: {r.detail}" for r in results if r.verdict == "ERROR")
            return OracleResult(
                oracle=ORACLE_ID,
                verdict="ERROR",
                reason=f"anti-cheat check(s) {errored} could not complete -- {details}",
                evidence=evidence,
                duration_s=duration,
            )
        if not passed:
            return OracleResult(
                oracle=ORACLE_ID,
                verdict="SKIP",
                reason=(
                    "no anti-cheat check could run: "
                    + "; ".join(f"{k}: {v}" for k, v in skipped.items())
                ),
                evidence=evidence,
                duration_s=duration,
            )
        return OracleResult(
            oracle=ORACLE_ID,
            verdict="PASS",
            reason="",
            evidence=evidence,
            duration_s=duration,
        )


ORACLE = register_oracle(AntiCheatOracle())

__all__ = [
    "ORACLE",
    "ORACLE_ID",
    "AntiCheatOracle",
    "CHECKS",
    "CHECK_NAMES",
    "CheckResult",
    "DenylistHit",
    "analyze_source",
    "achievable_bandwidth",
    "poisoned_source",
    "run_check",
    "run_checks",
    "check_static_denylist",
    "check_randomized_inputs",
    "check_held_out_shapes",
    "check_output_liveness",
    "check_timing_sanity",
]
