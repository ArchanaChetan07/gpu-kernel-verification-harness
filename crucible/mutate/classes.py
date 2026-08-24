"""The twelve mutation classes (CONTRACT.md section 6).

Every class finds its site **structurally**: it matches an AST pattern, never a
line number and never a substring of the source text. A string match would find
the word "mask" in a comment and would break the moment a seed is reformatted;
a structural match finds the ``Compare`` node whose operands are an offsets
vector and a bound, which is what the bug actually is.

Two invariants hold for every class in this module:

* ``sites()`` returns ``[]`` when the pattern is absent and **never raises**.
  A seed that happens not to carry a pattern must not stop the generator from
  trying the other eleven classes, so pattern-matching failures are logged and
  swallowed here and nowhere else.
* ``apply()`` returns a **new** tree. It may raise ``SiteError`` when the site
  does not resolve; that is a real defect and the engine records it.

The detectors are deliberately conservative. A false negative costs one task; a
false positive produces a mutation whose witness search wastes execution budget
and then discards it, which is merely slow. Neither can produce an unverified
task, because nothing ships without a witness.
"""

from __future__ import annotations

import ast
import copy
import logging
from typing import Any, Iterable, Protocol, runtime_checkable

from ..schema import Tier
from .astutil import (
    Site,
    SiteError,
    apply_to_copy,
    remove_node,
    replace_node,
    require_slot,
    site_for,
    walk_ordered,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# shared structural predicates
# --------------------------------------------------------------------------- #

#: Tensor methods that return a view (or a stride-only reinterpretation).
VIEW_OPS = frozenset(
    {
        "transpose",
        "permute",
        "view",
        "reshape",
        "t",
        "expand",
        "squeeze",
        "unsqueeze",
        "narrow",
        "select",
        "flatten",
        "movedim",
        "swapaxes",
        "as_strided",
    }
)

#: Allocation calls whose result is a buffer we may resize or re-dtype.
ALLOC_FUNCS = frozenset(
    {"zeros", "empty", "full", "ones", "zeros_like", "empty_like", "full_like", "ones_like"}
)

COLLECTIVES = frozenset(
    {
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "broadcast",
        "reduce",
        "barrier",
        "all_to_all",
        "gather",
        "scatter",
    }
)

#: In-place methods that write into an existing output buffer. ``zero_`` and
#: ``fill_`` are here because "define the whole buffer to a constant" is how a
#: degenerate path stores its result; a class that only recognised elementwise
#: index writes would not see the store that path actually makes.
STORE_METHODS = frozenset(
    {
        "index_copy_",
        "index_put_",
        "scatter_",
        "masked_scatter_",
        "copy_",
        "index_fill_",
        "put_",
        "zero_",
        "fill_",
    }
)

#: Broadcasts and stride reinterpretations. ``expand``/``broadcast_to`` are
#: separated out because they are the ops that *deliberately* set a stride to
#: zero, which is where a layout conflict is a choice rather than an accident.
BROADCAST_OPS = frozenset({"expand", "expand_as", "broadcast_to"})

#: Calls whose second positional argument is an index vector, not data. A view
#: op sitting in that position is reshaping indices, not choosing a layout.
INDEXING_CALLS = frozenset(
    {
        "gather",
        "scatter",
        "scatter_",
        "scatter_add_",
        "index_select",
        "index_copy_",
        "index_put_",
        "index_fill_",
        "take",
        "take_along_dim",
        "put_",
        "masked_scatter_",
    }
)

#: Roots that mark a call as a process-group collective rather than a tensor
#: method that happens to share a name with one (``probs.gather``, ``x.scatter_``).
_DIST_ROOTS = frozenset({"dist", "distributed", "c10d", "pg", "group"})

#: List/set mutators by which a rank contributes its piece to a later collective.
_CONTRIB_METHODS = frozenset({"append", "extend", "insert", "add"})

_OFFSET_TOKENS = ("offs", "offset", "idx", "index", "lane", "tid", "thread", "pos")
_BOUND_TOKENS = ("size", "count", "len", "limit", "bound", "total", "numel", "shape", "seq", "num", "dim")
_BOUND_EXACT = frozenset(
    {"n", "N", "m", "M", "k", "K", "end", "stop", "upper", "cols", "rows", "n_elements"}
)
_OUTPUT_NAMES = frozenset({"out", "output", "y", "result", "res", "dst", "o", "buf", "acc_out"})
_RESTORE_TOKENS = ("load", "restore", "resume", "from_checkpoint", "deserial")
_KEY_TOKENS = ("key", "signature", "sig", "cache_id")
_NORM_TOKENS = ("norm", "sq", "grad")


def _attr_name(func: ast.AST) -> str:
    """The final callable name: ``a.b.clone`` -> 'clone', ``arange`` -> 'arange'."""
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _call_name(node: ast.AST) -> str:
    return _attr_name(node.func) if isinstance(node, ast.Call) else ""


def _base_name(node: ast.AST) -> str:
    """Root identifier of a target expression: ``out[i]`` and ``out.copy_`` -> 'out'."""
    cur: ast.AST = node
    while True:
        if isinstance(cur, ast.Name):
            return cur.id
        if isinstance(cur, (ast.Subscript, ast.Attribute)):
            cur = cur.value
            continue
        if isinstance(cur, ast.Call):
            cur = cur.func
            continue
        if isinstance(cur, ast.Starred):
            cur = cur.value
            continue
        return ""


def _identifiers(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def _keyword(call: ast.Call, name: str) -> ast.keyword | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw
    return None


def _is_torch_dtype(node: ast.AST, names: Iterable[str]) -> bool:
    wanted = set(names)
    if isinstance(node, ast.Attribute) and node.attr in wanted:
        return True
    if isinstance(node, ast.Name) and node.id in wanted:
        return True
    return False


def _torch_attr(name: str) -> ast.Attribute:
    return ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()), attr=name, ctx=ast.Load())


def _contains(root: ast.AST, node: ast.AST) -> bool:
    return any(n is node for n in walk_ordered(root))


def _functions(tree: ast.AST) -> list[ast.AST]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _enclosing_function(tree: ast.AST, node: ast.AST) -> ast.AST | None:
    """Innermost enclosing def, or None for module-level nodes."""
    found: ast.AST | None = None
    for fn in _functions(tree):
        if _contains(fn, node):
            found = fn  # ast.walk is breadth-first, so the innermost match is last
    return found


def _looks_like_offsets(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and _call_name(node) in ("arange", "program_id", "tid", "lane_id"):
        return True
    return any(
        any(tok in name.lower() for tok in _OFFSET_TOKENS) for name in _identifiers(node)
    )


def _looks_like_bound(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return True
    if isinstance(node, ast.Call) and _call_name(node) in ("numel", "size", "len"):
        return True
    names = _identifiers(node)
    if names & _BOUND_EXACT:
        return True
    return any(any(tok in name.lower() for tok in _BOUND_TOKENS) for name in names)


def _is_accumulated(scope: ast.AST, name: str) -> bool:
    """Is ``name`` written in terms of itself anywhere in ``scope``?

    That is the operational definition of "accumulation target": ``acc += x``,
    ``acc = acc * s + p``, or ``acc.add_(x)``. A plain ``buf[i] = v`` store is
    not accumulation, which is what keeps ``accum_dtype`` off scratch buffers.
    """
    for node in ast.walk(scope):
        if isinstance(node, ast.AugAssign) and _base_name(node.target) == name:
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _base_name(target) != name:
                    continue
                if any(isinstance(x, ast.Name) and x.id == name for x in ast.walk(node.value)):
                    return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("add_", "mul_", "sub_", "addcmul_", "addmm_", "accumulate_"):
                if _base_name(node.func.value) == name:
                    return True
    return False


def _is_written(scope: ast.AST, name: str) -> bool:
    """Is ``name`` (or an element of it) assigned or mutated in ``scope``?"""
    for node in ast.walk(scope):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if _base_name(target) == name and not isinstance(target, ast.Name):
                    return True
                if isinstance(target, ast.Name) and target.id == name:
                    return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr.endswith("_") and not node.func.attr.startswith("_"):
                if _base_name(node.func.value) == name:
                    return True
    return False


def _is_subscripted(scope: ast.AST, name: str) -> bool:
    for node in ast.walk(scope):
        if isinstance(node, ast.Subscript) and _base_name(node.value) == name:
            if not isinstance(node.slice, ast.Constant):
                return True
    return False


def _store_statements(block: list[ast.stmt]) -> dict[str, ast.stmt]:
    """Statements in ``block`` that write into a named buffer, keyed by buffer."""
    out: dict[str, ast.stmt] = {}
    for stmt in block:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Subscript):
                out.setdefault(_base_name(target), stmt)
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Subscript):
            out.setdefault(_base_name(stmt.target), stmt)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute) and call.func.attr in STORE_METHODS:
                out.setdefault(_base_name(call.func.value), stmt)
    return out


def _blocks(tree: ast.AST) -> Iterable[tuple[ast.AST, str, list[ast.stmt]]]:
    for node in walk_ordered(tree):
        for fname in ("body", "orelse", "finalbody"):
            value = getattr(node, fname, None)
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                yield node, fname, value


# --------------------------------------------------------------------------- #
# protocol and base
# --------------------------------------------------------------------------- #


@runtime_checkable
class MutationClass(Protocol):
    id: str
    tier: Tier
    description: str

    def sites(self, tree: ast.AST, src: str) -> list[Site]: ...

    def apply(self, tree: ast.AST, site: Site) -> ast.AST: ...


class BaseMutation:
    """Shared plumbing. Subclasses implement ``_sites`` and ``_transform``."""

    id: str = ""
    tier: Tier = "T5"
    description: str = ""

    def sites(self, tree: ast.AST, src: str = "") -> list[Site]:
        try:
            found = self._sites(tree, src)
        except Exception as exc:  # noqa: BLE001 - a detector defect must not stop generation
            logger.warning("mutation class %s failed while scanning for sites: %s", self.id, exc)
            return []
        found.sort(key=lambda s: (s.lineno, s.col_offset, s.ordinal))
        return found

    def apply(self, tree: ast.AST, site: Site) -> ast.AST:
        return apply_to_copy(tree, site, lambda t, n: self._transform(t, n, site))

    def spec_params(self, site: Site) -> dict[str, Any]:
        """JSON-safe parameters recorded on the task's ``MutationSpec``."""
        params = dict(site.as_dict()["meta"])
        params["site"] = site.descriptor()
        return params

    # -- subclass hooks ----------------------------------------------------- #

    def _sites(self, tree: ast.AST, src: str) -> list[Site]:
        raise NotImplementedError

    def _transform(self, tree: ast.AST, node: ast.AST, site: Site) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1. boundary_mask
# --------------------------------------------------------------------------- #


class BoundaryMask(BaseMutation):
    id = "boundary_mask"
    tier: Tier = "T5"
    description = (
        "Relax a strict tail-guard comparison (offsets < bound) to an inclusive one, "
        "so the last partial block reads one element past the end."
    )

    _FLIP = {ast.Lt: ast.LtE, ast.Gt: ast.GtE}

    def _sites(self, tree: ast.AST, src: str) -> list[Site]:
        out: list[Site] = []
        for node in walk_ordered(tree):
            if not isinstance(node, ast.Compare) or len(node.ops) != 1:
                continue
            if type(node.ops[0]) not in self._FLIP:
                continue
            left, right = node.left, node.comparators[0]
            forward = _looks_like_offsets(left) and _looks_like_bound(right)
            reverse = _looks_like_bound(left) and _looks_like_offsets(right)
            if not (forward or reverse):
                continue
            op_name = type(node.ops[0]).__name__
            out.append(
                site_for(
                    tree,
                    node,
                    label=f"{ast.unparse(node)}",
                    meta={
                        "op": op_name,
                        "becomes": self._FLIP[type(node.ops[0])].__name__,
                        "orientation": "offsets_left" if forward else "offsets_right",
                        "expression": ast.unparse(node),
                    },
                )
            )
        return out

    def _transform(self, tree: ast.AST, node: ast.AST, site: Site) -> None:
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            raise SiteError("boundary_mask site is not a single-operator comparison")
        flipped = self._FLIP.get(type(node.ops[0]))
        if flipped is None:
            raise SiteError("boundary_mask site is not a strict comparison", op=type(node.ops[0]).__name__)
        node.ops[0] = flipped()


# --------------------------------------------------------------------------- #
# 2. accum_dtype
# --------------------------------------------------------------------------- #


class AccumDtype(BaseMutation):
    id = "accum_dtype"
    tier: Tier = "T5"
    description = (
        "Narrow the accumulator allocated for a reduction from fp32 to fp16, so the "
        "running sum loses precision as the accumulation depth grows."
    )

    to_dtype = "float16"
    _WIDE = ("float32", "float64", "double", "float")

    def _sites(self, tree: ast.AST, src: str) -> list[Site]:
        out: list[Site] = []
        for fn in _functions(tree):
            for stmt in ast.walk(fn):
                if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                call = stmt.value
                if not isinstance(call, ast.Call) or _call_name(call) not in ALLOC_FUNCS:
                    continue
                kw = _keyword(call, "dtype")
                if kw is None or not _is_torch_dtype(kw.value, self._WIDE):
                    continue
                if not _is_accumulated(fn, target.id):
                    continue
                out.append(
                    site_for(
                        tree,
                        call,
                        label=f"{target.id} = {ast.unparse(call)}",
                        meta={
                            "buffer": target.id,
                            "from_dtype": ast.unparse(kw.value),
                            "to_dtype": f"torch.{self.to_dtype}",
                            "allocator": _call_name(call),
                        },
                    )
                )
        return out

    def _transform(self, tree: ast.AST, node: ast.AST, site: Site) -> None:
        if not isinstance(node, ast.Call):
            raise SiteError("accum_dtype site is not a call")
        kw = _keyword(node, "dtype")
        if kw is None:
            raise SiteError("accum_dtype site has no dtype keyword")
        kw.value = _torch_attr(self.to_dtype)


# --------------------------------------------------------------------------- #
# 3. shmem_sizing
# --------------------------------------------------------------------------- #


class ShmemSizing(BaseMutation):
    id = "shmem_sizing"
    tier: Tier = "T2"
    description = (
        "Size a scratch buffer by a fixed hardware constant instead of the logical "
        "element count, while the index expression still uses the running index; "
        "out of bounds exactly when the count exceeds the fixed size."
    )

    fixed_size = 32

    def _sites(self, tree: ast.AST, src: str) -> list[Site]:
        out: list[Site] = []
        for fn in _functions(tree):
            for stmt in ast.walk(fn):
                if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                call = stmt.value
                if not isinstance(call, ast.Call) or _call_name(call) not in ALLOC_FUNCS:
                    continue
                if not call.args:
                    continue
                size_arg = call.args[0]
                # A logical count is a computed scalar, not a literal and not a
                # shape tuple: those are already correctly sized by construction.
                if not isinstance(size_arg, (ast.Name, ast.BinOp, ast.Call)):
                    continue
                if isinstance(size_arg, ast.Call) and _call_name(size_arg) in ("len", "numel", "size"):
                    pass
                elif isinstance(size_arg, ast.Call):
                    continue
                if not _is_subscripted(fn, target.id):
                    continue
                out.append(
                    site_for(
                        tree,
                        call,
                        label=f"{target.id} = {ast.unparse(call)}",
                        meta={
                            "buffer": target.id,
                            "logical_size": ast.unparse(size_arg),
                            "fixed_size": self.fixed_size,
                        },
                    )
                )
        return out

    def _transform(self, tree: ast.AST, node: ast.AST, site: Site) -> None:
        if not isinstance(node, ast.Call) or not node.args:
            raise SiteError("shmem_sizing site is not an allocation with a size argument")
        node.args[0] = ast.Constant(value=int(self.fixed_size))


# --------------------------------------------------------------------------- #
# 4. missing_barrier
# --------------------------------------------------------------------------- #


class MissingBarrier(BaseMutation):
    id = "missing_barrier"
    tier: Tier = "T5"
    description = (
        "Remove the explicit copy that separates a read of a buffer from a write to "
        "the same buffer in the next iteration, turning it into an alias."
    )

    _COPY_METHODS = frozenset({"clone", "copy"})

    def _sites(self, tree: ast.AST, src: str) -> list[Site]:
        out: list[Site] = []
        loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While, ast.AsyncFor))]
        for node in walk_ordered(tree):
            if not isinstance(node, ast.Call):
                continue
            receiver = self._receiver(node)
            if receiver is None:
                continue
            base = _base_name(receiver)
            if not base:
                continue
            in_loop = any(_contains(loop, node) for loop in loops)
            scope = _enclosing_function(tree, node) or tree
            if not in_loop and not _is_written(scope, base):
                continue
            out.append(
                site_for(
                    tree,
                    node,
                    label=ast.unparse(node),
                    meta={
                        "buffer": base,
                        "copy_call": ast.unparse(node),
                        "inside_loop": in_loop,
                    },
                )
            )
        return out

    def _receiver(self, call: ast.Call) -> ast.expr | None:
        """The expression whose copy is being taken, or None if this is not a copy."""
        if isinstance(call.func, ast.Attribute) and call.func.attr in self._COPY_METHODS:
            if isinstance(call.func.value, (ast.Name, ast.Subscript, ast.Attribute)):
                return call.func.value
            return None
        if _attr_name(call.func) == "clone" and len(call.args) == 1 and not isinstance(call.func, ast.Attribute):
            return call.args[0]
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "clone"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "torch"
            and len(call.args) == 1
        ):
            return call.args[0]
        return None

    def _transform(self, tree: ast.AST, node: ast.AST, site: Site) -> None:
        if not isinstance(node, ast.Call):
            raise SiteError("missing_barrier site is not a call")
        receiver = self._receiver(node)
        if receiver is None:
            raise SiteError("missing_barrier site is not a copy call")
        replace_node(tree, node, copy.deepcopy(receiver))


# --------------------------------------------------------------------------- #
# 5. cross_block_reduction
# --------------------------------------------------------------------------- #


class CrossBlockReduction(BaseMutation):
    id = "cross_block_reduction"
    tier: Tier = "T5"
    description = (
        "Replace the running-rescale combine of a blocked reduction with the current "
        "block's partial, so a per-block result is reported as the total."
    )

    def _sites(self, tree: ast.AST, src: str) -> list[Site]:
        out: list[Site] = []
        for node in walk_ordered(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = node.value
            if not isinstance(value, ast.BinOp) or not isinstance(value.op, (ast.Add, ast.Sub)):
                continue
            left_self = self._mentions(value.left, target.id)
            right_self = self._mentions(value.right, target.id)
            if left_self == right_self:
                continue  # both or neither carry the accumulator: not a combine
            carrier = value.left if left_self else value.right
            partial = value.right if left_self else value.left
            if not self._is_rescale(carrier):
                continue
            out.append(
                site_for(
                    tree,
                    node,
                    label=ast.unparse(node),
                    meta={
                        "accumulator": target.id,
                        "rescale": ast.unparse(carrier),
                        "partial": ast.unparse(partial),
                        "keeps": "partial",
                    },
                )
            )
        return out

    @staticmethod
    def _mentions(node: ast.AST, name: str) -> bool:
        return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))

    @staticmethod
    def _is_rescale(node: ast.AST) -> bool:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
            return True
        return isinstance(node, ast.Call) and _call_name(node) in ("mul", "div", "multiply")

    def _transform(self, tree: ast.AST, node: ast.AST, site: Site) -> None:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.BinOp):
            raise SiteError("cross_block_reduction site is not a combine assignment")
        target = node.targets[0]
        name = target.id if isinstance(target, ast.Name) else ""
        value = node.value
        if self._mentions(value.left, name):
            partial = value.right
        elif self._mentions(value.right, name):
            partial = value.left
        else:
            raise SiteError("cross_block_reduction site does not carry its accumulator")
        node.value = copy.deepcopy(partial)


# --------------------------------------------------------------------------- #
# 6. empty_input_guard
# --------------------------------------------------------------------------- #


class EmptyInputGuard(BaseMutation):
    id = "empty_input_guard"
    tier: Tier = "T2"
    description = (
        "Delete the guard that skips a zero-length block, so the degenerate case "
        "falls through into the indexing path."
    )

    _EXITS = (ast.Continue, ast.Return, ast.Pass, ast.Break)

    def _sites(self, tree: ast.AST, src: str) -> list[Site]:
        out: list[Site] = []
        for node in walk_ordered(tree):
            if not isinstance(node, ast.If) or node.orelse:
                continue
            if len(node.body) != 1 or not isinstance(node.body[0], self._EXITS):
                continue
            if not self._zero_length_test(node.test):
                continue
            out.append(
                site_for(
                    tree,
                    node,
                    label=f"if {ast.unparse(node.test)}: {ast.unparse(node.body[0]).strip()}",
                    meta={
                        "test": ast.unparse(node.test),
                        "guarded_exit": type(node.body[0]).__name__,
                    },
                )
            )
        return out

    @staticmethod
    def _zero_length_test(test: ast.AST) -> bool:
        for node in ast.walk(test):
            if isinstance(node, ast.Call) and _call_name(node) in ("numel", "len", "size"):
                return True
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return True
        if isinstance(test, ast.Compare):
            operands = [test.left, *test.comparators]
            for operand in operands:
                if isinstance(operand, ast.Constant) and operand.value in (0, 1):
                    return True
            names = {n.lower() for n in _identifiers(test)}
            if names & {"start", "begin"} and names & {"stop", "end"}:
                return True
        return False

    def _transform(self, tree: ast.AST, node: ast.AST, site: Site) -> None:
        if not isinstance(node, ast.If):
            raise SiteError("empty_input_guard site is not an if statement")
        remove_node(tree, node)


# --------------------------------------------------------------------------- #
# 7. uninit_output
# --------------------------------------------------------------------------- #


class UninitOutput(BaseMutation):
    id = "uninit_output"
    tier: Tier = "T5"
    description = (
        "Drop the store into the output buffer on one branch of a conditional, so "
        "that path leaves the output holding whatever it was allocated with."
    )

    def _sites(self, tree: ast.AST, src: str) -> list[Site]:
        out: list[Site] = []
        for node in walk_ordered(tree):
            if not isinstance(node, ast.If):
                continue
            body_stores = _store_statements(node.body)

            if node.orelse:
                else_stores = _store_statements(node.orelse)
                for name in sorted(set(body_stores) & set(else_stores)):
                    if not self._looks_like_output(name):
                        continue
                    stmt = else_stores[name]
                    out.append(
                        site_for(
                            tree,
                            stmt,
                            label=ast.unparse(stmt),
                            meta={
                                "buffer": name,
                                "branch": "orelse",
                                "surviving_store": ast.unparse(body_stores[name]),
                            },
                        )
                    )
                continue

            # Guard-with-continue: the idiom kernels actually use for a boundary
            # tile ("if nothing valid here: write the tile, continue"). There is
            # no orelse, so an if/else-only pattern finds nothing in exactly the
            # code this mutation exists to attack. Dropping the guard's store
            # leaves that tile holding whatever torch.empty handed back - the
            # "returned uninitialized memory" defect verbatim.
            if not any(isinstance(st, (ast.Continue, ast.Return)) for st in node.body):
                continue
            for name, stmt in sorted(body_stores.items()):
                if not self._looks_like_output(name):
                    continue
                elsewhere = self._other_stores(tree, name, stmt)
                if not elsewhere:
                    # Nothing else writes this buffer, so deleting the only
                    # store leaves it wholly unwritten - a different (and more
                    # obvious) defect than the partial hole this class models.
                    continue
                out.append(
                    site_for(
                        tree,
                        stmt,
                        label=ast.unparse(stmt),
                        meta={
                            "buffer": name,
                            "branch": "guard",
                            "surviving_store": ast.unparse(elsewhere[0]),
                        },
                    )
                )
        return out

    @staticmethod
    def _other_stores(tree: ast.AST, name: str, exclude: ast.stmt) -> list[ast.stmt]:
        """Every other statement that writes into ``name``."""
        found: list[ast.stmt] = []
        for node in walk_ordered(tree):
            if node is exclude:
                continue
            target = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
            elif isinstance(node, ast.AugAssign):
                target = node.target
            if isinstance(target, ast.Subscript) and _base_name(target) == name:
                found.append(node)
        return found

    @staticmethod
    def _looks_like_output(name: str) -> bool:
        lowered = name.lower()
        return lowered in _OUTPUT_NAMES or "out" in lowered

    def _transform(self, tree: ast.AST, node: ast.AST, site: Site) -> None:
        remove_node(tree, node)


# --------------------------------------------------------------------------- #
# 8. layout_conflict
# --------------------------------------------------------------------------- #


class LayoutConflict(BaseMutation):
    id = "layout_conflict"
    tier: Tier = "T5"
    description = (
        "Force a materialised, more-replicated layout at a stride-preserving op "
        "(contiguous copy then re-expand), so downstream writes no longer alias the "
        "source and the result changes without raising."
    )

    # SCOPE, stated plainly because it bounds what this class can ever produce:
    # `x.contiguous().expand_as(x)` is value-preserving. It changes only
    # ALIASING, so it is observable exactly when something later writes through
    # the view in place. A layout conflict that changes VALUES - the Triton
    # #10987 shape, where the compiler picks a more-replicated layout and
    # returns wrong numbers - is a property of a real compiler choosing
    # layouts, and pure-torch view ops preserve values by construction. So on
    # torch seeds this class is honestly narrow; reproducing the compiler
    # variant needs a Triton seed executing on a working Triton toolchain.
    #
    # Requiring the aliasing precondition drops the site count from 37 to
    # however many are real. That is the point: 37 sites that can never be
    # admitted make the catalogue look healthier than it is, which is the same
    # invisible failure this project exists to prevent.

    def _sites(self, tree: ast.AST, src: str) -> list[Site]:
        out: list[Site] = []
        for node in walk_ordered(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in VIEW_OPS:
                continue
            # Already materialised: forcing it again is a textual no-op.
            if isinstance(node.func.value, ast.Call) and _call_name(node.func.value) == "contiguous":
                continue
            if not self._aliased_downstream(tree, node):
                continue
            out.append(
                site_for(
                    tree,
                    node,
                    label=ast.unparse(node),
                    meta={
                        "view_op": node.func.attr,
                        "expression": ast.unparse(node),
                        "becomes": f"{ast.unparse(node)}.contiguous().expand_as(...)",
                    },
                )
            )
        return out

    @staticmethod
    def _aliased_downstream(tree: ast.AST, node: ast.Call) -> bool:
        """Is the viewed buffer written in place anywhere?

        Without such a write, materialising the view changes nothing an oracle
        can see, and the mutation is discarded as semantically neutral after
        paying for a full shape sweep.
        """
        base = _base_name(node.func.value if isinstance(node.func, ast.Attribute) else node)
        if not base:
            return False

        # The view is often BOUND first and mutated through that name
        # (`view = x.transpose(0, 1)` then `view.mul_(f)`). Watching only the
        # source buffer misses the most idiomatic aliasing there is.
        aliases = {base}
        for other in walk_ordered(tree):
            if (
                isinstance(other, ast.Assign)
                and other.value is node
                and len(other.targets) == 1
                and isinstance(other.targets[0], ast.Name)
            ):
                aliases.add(other.targets[0].id)

        for other in walk_ordered(tree):
            # Any trailing-underscore method is torch's in-place convention.
            if (
                isinstance(other, ast.Expr)
                and isinstance(other.value, ast.Call)
                and isinstance(other.value.func, ast.Attribute)
                and other.value.func.attr.endswith("_")
                and not other.value.func.attr.startswith("_")
                and _base_name(other.value.func.value) in aliases
            ):
                return True

        for other in walk_ordered(tree):
            target = None
            if isinstance(other, ast.Assign) and len(other.targets) == 1:
                target = other.targets[0]
            elif isinstance(other, ast.AugAssign):
                target = other.target
            if isinstance(target, ast.Subscript) and _base_name(target) in aliases:
                return True
            if (
                isinstance(other, ast.Expr)
                and isinstance(other.value, ast.Call)
                and isinstance(other.value.func, ast.Attribute)
                and other.value.func.attr in STORE_METHODS
                and _base_name(other.value.func.value) in aliases
            ):
                return True
        return False

    def _transform(self, tree: ast.AST, node: ast.AST, site: Site) -> None:
        if not isinstance(node, ast.Call):
            raise SiteError("layout_conflict site is not a call")
        # Both copies are of a pure view op, so re-evaluating the expression is
        # free of side effects; what changes is that the result is a copy.
        materialised = ast.Call(
            func=ast.Attribute(value=copy.deepcopy(node), attr="contiguous", ctx=ast.Load()),
            args=[],
            keywords=[],
        )
        replacement = ast.Call(
            func=ast.Attribute(value=materialised, attr="expand_as", ctx=ast.Load()),
            args=[copy.deepcopy(node)],
            keywords=[],
        )
        replace_node(tree, node, replacement)


# --------------------------------------------------------------------------- #
# 9. grad_norm_scope
# --------------------------------------------------------------------------- #


class GradNormScope(BaseMutation):
    id = "grad_norm_scope"
    tier: Tier = "T6"
    description = (
        "Remove the cross-rank reduction of the squared gradient norm, so clipping "
        "uses a per-shard norm and each rank scales its gradients differently."
    )

    _REDUCERS = frozenset({"all_reduce", "all_gather", "reduce", "all_gather_into_tensor"})

    @classmethod
    def _is_reducer(cls, name: str) -> bool:
        """Match the collective by token, not by exact identifier.

        Real code reaches a cross-rank sum two ways: the in-place
        ``dist.all_reduce(t)`` statement, and a functional wrapper that returns
        the total (``total = _all_reduce_sum(parts, world)``). A seed that has
        to run single-process for differential testing must use the second
        form, so an exact-name match against the torch.distributed API silently
        finds nothing in exactly the code this mutation exists to attack.
        """
        low = name.lower()
        return any(r in low for r in cls._REDUCERS)

    def _sites(self, tree: ast.AST, src: str) -> list[Site]:
        out: list[Site] = []
        for node in walk_ordered(tree):
            # in-place form: `dist.all_reduce(norm_sq)` as a bare statement
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if not self._is_reducer(_call_name(call)) or not call.args:
                    continue
                name = _base_name(call.args[0])
                if not name or not any(tok in name.lower() for tok in _NORM_TOKENS):
                    continue
                out.append(
                    site_for(
                        tree,
                        node,
                        label=ast.unparse(node),
                        meta={"tensor": name, "collective": _call_name(call), "form": "in_place"},
                    )
                )
                continue

            # functional form: `total_sq = _all_reduce_sum(local_sq, n_shards)`
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call = node.value
                if not self._is_reducer(_call_name(call)) or len(call.args) < 2:
                    continue
                if not isinstance(call.args[0], ast.Name):
                    continue
                source = _base_name(call.args[0])
                target = _base_name(node.targets[0]) if node.targets else ""
                blob = f"{source} {target}".lower()
                if not any(tok in blob for tok in _NORM_TOKENS):
                    continue
                out.append(
                    site_for(
                        tree,
                        node,
                        label=ast.unparse(node),
                        meta={
                            "tensor": source,
                            "collective": _call_name(call),
                            "form": "functional",
                        },
                    )
                )
        return out

    def _transform(self, tree: ast.AST, node: ast.AST, site: Site) -> None:
        if site.meta.get("form") == "functional":
            # Keep the binding, drop the cross-rank scope: the clipping norm
            # becomes this shard's own sum of squares. The loss curve still
            # descends, which is exactly why this class is T6.
            call = node.value  # type: ignore[attr-defined]
            node.value = ast.Subscript(  # type: ignore[attr-defined]
                value=call.args[0],
                slice=ast.Constant(value=0),
                ctx=ast.Load(),
            )
            ast.fix_missing_locations(node)
            return
        remove_node(tree, node)


# --------------------------------------------------------------------------- #
# 10. collective_ordering
# --------------------------------------------------------------------------- #


class CollectiveOrdering(BaseMutation):
    id = "collective_ordering"
    tier: Tier = "T3"
    description = (
        "Move a collective across a conditional so it is issued on only one branch; "
        "ranks that take the other branch never join, and the ranks diverge."
    )

    @staticmethod
    def _reduced_names(tree: ast.AST) -> set[str]:
        """Names that are later handed to a collective.

        A single-process seed cannot call dist.all_reduce, so it builds a list
        of per-rank contributions and reduces it functionally. The list IS the
        collective's participant set, which makes an append to it a
        contribution to the collective even though no collective API appears.
        """
        names: set[str] = set()
        for node in walk_ordered(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = _call_name(node).lower()
            if not any(c in fname for c in COLLECTIVES):
                continue
            for arg in node.args:
                base = _base_name(arg)
                if base:
                    names.add(base)
        return names

    @staticmethod
    def _contribution_target(stmt: ast.stmt, reduced: set[str]) -> str:
        """``parts.append(x)`` where ``parts`` is later reduced -> 'parts'."""
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return ""
        func = stmt.value.func
        if not isinstance(func, ast.Attribute) or func.attr != "append":
            return ""
        base = _base_name(func.value)
        return base if base in reduced else ""

    def _sites(self, tree: ast.AST, src: str) -> list[Site]:
        out: list[Site] = []
        reduced = self._reduced_names(tree)
        for _parent, _field, block in _blocks(tree):
            for i, stmt in enumerate(block):
                # Form A: a bare collective call adjacent to a conditional.
                if self._is_collective(stmt):
                    neighbour = self._neighbour_if(block, i)
                    if neighbour is None:
                        continue
                    direction = "after" if neighbour is block[min(i + 1, len(block) - 1)] else "before"
                    out.append(
                        site_for(
                            tree,
                            stmt,
                            label=ast.unparse(stmt),
                            meta={
                                "collective": _call_name(stmt.value) if isinstance(stmt, ast.Expr) else "",
                                "moved_into": "if.body",
                                "conditional_at_line": getattr(neighbour, "lineno", -1),
                                "direction": direction,
                                "form": "call",
                            },
                        )
                    )
                    continue

                # Form B: a contribution to a functionally-reduced list, sitting
                # after a guard. Moving it inside the guard means a rank that
                # takes the other branch never joins, which is the divergence
                # this class exists to inject. In-process that surfaces as a
                # participant-count desync rather than a hang - the same defect,
                # observable without a real process group.
                target = self._contribution_target(stmt, reduced)
                if not target:
                    continue
                if i - 1 < 0 or not isinstance(block[i - 1], ast.If):
                    continue
                guard = block[i - 1]
                out.append(
                    site_for(
                        tree,
                        stmt,
                        label=ast.unparse(stmt),
                        meta={
                            "collective": target,
                            "moved_into": "if.orelse" if guard.orelse else "if.body",
                            "conditional_at_line": getattr(guard, "lineno", -1),
                            "direction": "after",
                            "form": "contribution",
                        },
                    )
                )
        return out

    @staticmethod
    def _is_collective(stmt: ast.stmt) -> bool:
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and _call_name(stmt.value) in COLLECTIVES
        )

    @staticmethod
    def _neighbour_if(block: list[ast.stmt], i: int) -> ast.If | None:
        if i + 1 < len(block) and isinstance(block[i + 1], ast.If):
            return block[i + 1]  # type: ignore[return-value]
        if i - 1 >= 0 and isinstance(block[i - 1], ast.If):
            return block[i - 1]  # type: ignore[return-value]
        return None

    def _transform(self, tree: ast.AST, node: ast.AST, site: Site) -> None:
        slot = require_slot(tree, node)
        if slot.index is None:
            raise SiteError("collective_ordering site is not a statement in a block")
        block = getattr(slot.parent, slot.field)
        if site.meta.get("form") == "contribution":
            index = slot.index
            if index - 1 < 0 or not isinstance(block[index - 1], ast.If):
                raise SiteError("collective_ordering contribution site has no preceding guard")
            guard = block[index - 1]
            del block[index]
            # Prefer the else branch: the common path keeps contributing and
            # only the guarded path drops out, so small inputs still agree and
            # the divergence needs the guard to actually fire.
            (guard.orelse if guard.orelse else guard.body).append(node)
            if not block:
                block.append(ast.Pass())
            ast.fix_missing_locations(guard)
            return
        target = self._neighbour_if(block, slot.index)
        if target is None:
            raise SiteError("collective_ordering site has no adjacent conditional")
        del block[slot.index]
        target.body.insert(0, node)
        if not block:
            block.append(ast.Pass())


# --------------------------------------------------------------------------- #
# 11. resume_fidelity
# --------------------------------------------------------------------------- #


class ResumeFidelity(BaseMutation):
    id = "resume_fidelity"
    tier: Tier = "T6"
    description = (
        "Restore optimizer state in bf16 although it was saved in fp32, so a resumed "
        "run silently continues from a rounded moment estimate."
    )

    to_dtype = "bfloat16"
    _WIDE = ("float32", "float64", "double", "float")

    def _sites(self, tree: ast.AST, src: str) -> list[Site]:
        out: list[Site] = []
        for fn in _functions(tree):
            name = getattr(fn, "name", "").lower()
            if not any(tok in name for tok in _RESTORE_TOKENS):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                form = self._form(node)
                if form is None:
                    continue
                out.append(
                    site_for(
                        tree,
                        node,
                        label=ast.unparse(node),
                        meta={
                            "function": getattr(fn, "name", ""),
                            "form": form,
                            "to_dtype": f"torch.{self.to_dtype}",
                        },
                    )
                )
        return out

    def _form(self, call: ast.Call) -> str | None:
        name = _call_name(call)
        if name == "float" and isinstance(call.func, ast.Attribute) and not call.args:
            return "float_method"
        if name == "to":
            if call.args and _is_torch_dtype(call.args[0], self._WIDE):
                return "to_positional"
            kw = _keyword(call, "dtype")
            if kw is not None and _is_torch_dtype(kw.value, self._WIDE):
                return "to_keyword"
            return None
        kw = _keyword(call, "dtype")
        if kw is not None and _is_torch_dtype(kw.value, self._WIDE):
            return "dtype_keyword"
        return None

    def _transform(self, tree: ast.AST, node: ast.AST, site: Site) -> None:
        if not isinstance(node, ast.Call):
            raise SiteError("resume_fidelity site is not a call")
        form = self._form(node)
        if form == "float_method":
            if not isinstance(node.func, ast.Attribute):
                raise SiteError("resume_fidelity float() site has no receiver")
            node.func.attr = self.to_dtype
            return
        if form == "to_positional":
            node.args[0] = _torch_attr(self.to_dtype)
            return
        if form in ("to_keyword", "dtype_keyword"):
            kw = _keyword(node, "dtype")
            if kw is None:
                raise SiteError("resume_fidelity site lost its dtype keyword")
            kw.value = _torch_attr(self.to_dtype)
            return
        raise SiteError("resume_fidelity site is not a widening restore", call=ast.unparse(node))


# --------------------------------------------------------------------------- #
# 12. autotune_staleness
# --------------------------------------------------------------------------- #


class AutotuneStaleness(BaseMutation):
    id = "autotune_staleness"
    tier: Tier = "T4"
    description = (
        "Drop the shape components from the autotune cache key, so a configuration "
        "chosen for one shape class is reused for every later shape class."
    )

    def _sites(self, tree: ast.AST, src: str) -> list[Site]:
        out: list[Site] = []
        for fn in _functions(tree):
            for stmt in ast.walk(fn):
                if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if not any(tok in target.id.lower() for tok in _KEY_TOKENS):
                    continue
                value = stmt.value
                if not isinstance(value, ast.Tuple) or len(value.elts) < 2:
                    continue
                if not self._used_as_lookup(fn, target.id):
                    continue
                out.append(
                    site_for(
                        tree,
                        stmt,
                        label=ast.unparse(stmt),
                        meta={
                            "key": target.id,
                            "dropped": [ast.unparse(e) for e in value.elts[:-1]],
                            "kept": ast.unparse(value.elts[-1]),
                            "form": "assign",
                        },
                    )
                )

        # A key built by a dedicated helper that RETURNS the tuple. The lookup
        # then lives in a different function, so the same-scope check above can
        # never see it - and factoring the key out is the normal way real code
        # is written, which is precisely where this defect hides.
        for fn in _functions(tree):
            if not any(tok in fn.name.lower() for tok in _KEY_TOKENS):
                continue
            for stmt in ast.walk(fn):
                if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.Tuple):
                    continue
                elts = stmt.value.elts
                if len(elts) < 2:
                    continue
                out.append(
                    site_for(
                        tree,
                        stmt,
                        label=ast.unparse(stmt),
                        meta={
                            "key": fn.name,
                            "dropped": [ast.unparse(elts[-1])],
                            "kept": [ast.unparse(e) for e in elts[:-1]],
                            "form": "return",
                        },
                    )
                )
        return out

    @staticmethod
    def _used_as_lookup(scope: ast.AST, name: str) -> bool:
        for node in ast.walk(scope):
            if isinstance(node, ast.Subscript):
                if any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node.slice)):
                    return True
            if isinstance(node, ast.Call) and _call_name(node) in ("get", "setdefault", "pop"):
                if any(
                    isinstance(n, ast.Name) and n.id == name for arg in node.args for n in ast.walk(arg)
                ):
                    return True
        return False

    def _transform(self, tree: ast.AST, node: ast.AST, site: Site) -> None:
        if site.meta.get("form") == "return":
            if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Tuple):
                raise SiteError("autotune_staleness site is not a tuple key return")
            elts = node.value.elts
            if len(elts) < 2:
                raise SiteError("autotune_staleness site has nothing to drop")
            # Drop exactly one component. Collapsing the key entirely would make
            # every problem share one config, which is too loud to be a T4; the
            # realistic defect is a key that omits ONE dimension and so cannot
            # tell two genuinely different shape classes apart.
            node.value = ast.Tuple(
                elts=[copy.deepcopy(e) for e in elts[:-1]], ctx=ast.Load()
            )
            ast.fix_missing_locations(node)
            return
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Tuple):
            raise SiteError("autotune_staleness site is not a tuple key assignment")
        elts = node.value.elts
        if len(elts) < 2:
            raise SiteError("autotune_staleness site has nothing to drop")
        node.value = ast.Tuple(elts=[copy.deepcopy(elts[-1])], ctx=ast.Load())


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

ALL_CLASSES: tuple[BaseMutation, ...] = (
    BoundaryMask(),
    AccumDtype(),
    ShmemSizing(),
    MissingBarrier(),
    CrossBlockReduction(),
    EmptyInputGuard(),
    UninitOutput(),
    LayoutConflict(),
    GradNormScope(),
    CollectiveOrdering(),
    ResumeFidelity(),
    AutotuneStaleness(),
)

MUTATION_CLASSES: dict[str, BaseMutation] = {c.id: c for c in ALL_CLASSES}


def get(class_id: str) -> BaseMutation:
    try:
        return MUTATION_CLASSES[class_id]
    except KeyError:
        raise KeyError(
            f"unknown mutation class {class_id!r}; known: {', '.join(sorted(MUTATION_CLASSES))}"
        ) from None


def resolve_classes(class_ids: Iterable[str] | None) -> list[BaseMutation]:
    """Resolve ids (or already-constructed classes) to instances; None means all."""
    if class_ids is None:
        return list(ALL_CLASSES)
    out: list[BaseMutation] = []
    for item in class_ids:
        out.append(get(item) if isinstance(item, str) else item)
    return out


def class_ids() -> list[str]:
    return [c.id for c in ALL_CLASSES]


__all__ = [
    "MutationClass",
    "BaseMutation",
    "BoundaryMask",
    "AccumDtype",
    "ShmemSizing",
    "MissingBarrier",
    "CrossBlockReduction",
    "EmptyInputGuard",
    "UninitOutput",
    "LayoutConflict",
    "GradNormScope",
    "CollectiveOrdering",
    "ResumeFidelity",
    "AutotuneStaleness",
    "ALL_CLASSES",
    "MUTATION_CLASSES",
    "get",
    "resolve_classes",
    "class_ids",
    "VIEW_OPS",
    "ALLOC_FUNCS",
    "COLLECTIVES",
    "STORE_METHODS",
]
