"""AST plumbing for the mutation engine.

Three properties this module exists to guarantee:

* **Round trip.** ``unparse(parse(src))`` must itself re-parse. Every mutation is
  applied to a tree and read back as text, so a lossy round trip would corrupt
  tasks silently. ``canonical()`` is the normal form: both the baseline and the
  mutant are produced by it, so the only textual difference between them is the
  mutation itself rather than incidental formatting.
* **Stable site addressing.** A site is ``(lineno, col_offset, node_type,
  ordinal)``. Position alone is ambiguous (``a < b and c < d`` puts two
  ``Compare`` nodes on one line), so the ordinal disambiguates within a
  deterministic pre-order walk. Addresses survive ``deepcopy`` because they
  depend only on positions, which the copy preserves.
* **Non-destructive application.** ``apply_to_copy`` never touches the caller's
  tree. A mutation class that raised halfway through an in-place edit would
  leave the baseline tree corrupted for every subsequent class.

``apply_unified_diff`` is here rather than in the tests because the round trip
"diff applied to the mutant reproduces the baseline" is a property the engine
must be able to check on itself, not only a property a test asserts.
"""

from __future__ import annotations

import ast
import copy
import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ..errors import CrucibleError

logger = logging.getLogger(__name__)


class SiteError(CrucibleError):
    """A site could not be resolved, or a transform was illegal for that node."""


# --------------------------------------------------------------------------- #
# sites
# --------------------------------------------------------------------------- #


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


@dataclass(frozen=True)
class Site:
    """A stable address for one node inside one parse of one source text.

    ``meta`` carries whatever the finding mutation class wants to record (which
    buffer, which branch, what the original expression read). It is excluded
    from equality and hashing so that two addresses of the same node compare
    equal regardless of the annotations attached to them.
    """

    lineno: int
    col_offset: int
    node_type: str
    ordinal: int
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def address(self) -> tuple[int, int, str, int]:
        return (self.lineno, self.col_offset, self.node_type, self.ordinal)

    def descriptor(self) -> str:
        return f"{self.node_type}@{self.lineno}:{self.col_offset}#{self.ordinal}"

    def __str__(self) -> str:
        return f"{self.descriptor()} {self.label}".strip()

    def as_dict(self) -> dict[str, Any]:
        return {
            "lineno": self.lineno,
            "col_offset": self.col_offset,
            "node_type": self.node_type,
            "ordinal": self.ordinal,
            "label": self.label,
            "meta": _json_safe(self.meta),
        }


def walk_ordered(node: ast.AST) -> Iterator[ast.AST]:
    """Deterministic pre-order walk (``ast.walk`` is breadth-first and coarser)."""
    yield node
    for child in ast.iter_child_nodes(node):
        yield from walk_ordered(child)


def _pos(node: ast.AST) -> tuple[int, int]:
    lineno = getattr(node, "lineno", None)
    col = getattr(node, "col_offset", None)
    return (-1 if lineno is None else int(lineno), -1 if col is None else int(col))


def site_for(
    tree: ast.AST,
    node: ast.AST,
    label: str = "",
    meta: dict[str, Any] | None = None,
) -> Site:
    """Address ``node`` within ``tree``. Raises if the node is not in the tree."""
    lineno, col = _pos(node)
    key = (lineno, col, type(node).__name__)
    ordinal = 0
    for cand in walk_ordered(tree):
        if cand is node:
            return Site(
                lineno=lineno,
                col_offset=col,
                node_type=key[2],
                ordinal=ordinal,
                label=label,
                meta=dict(meta or {}),
            )
        if (*_pos(cand), type(cand).__name__) == key:
            ordinal += 1
    raise SiteError(
        "node does not belong to the given tree",
        node_type=type(node).__name__,
        lineno=lineno,
    )


def resolve(tree: ast.AST, site: Site) -> ast.AST | None:
    """The node ``site`` addresses in ``tree``, or None if it is not there."""
    seen = 0
    for cand in walk_ordered(tree):
        lineno, col = _pos(cand)
        if lineno == site.lineno and col == site.col_offset and type(cand).__name__ == site.node_type:
            if seen == site.ordinal:
                return cand
            seen += 1
    return None


def require(tree: ast.AST, site: Site) -> ast.AST:
    node = resolve(tree, site)
    if node is None:
        raise SiteError("site does not resolve against this tree", site=site.descriptor())
    return node


# --------------------------------------------------------------------------- #
# structural edits
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Slot:
    """Where a node sits in its parent: ``parent.field`` or ``parent.field[index]``."""

    parent: ast.AST
    field: str
    index: int | None

    @property
    def is_list(self) -> bool:
        return self.index is not None


def find_slot(tree: ast.AST, target: ast.AST) -> Slot | None:
    for parent in walk_ordered(tree):
        for fname, value in ast.iter_fields(parent):
            if value is target:
                return Slot(parent=parent, field=fname, index=None)
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if item is target:
                        return Slot(parent=parent, field=fname, index=i)
    return None


def require_slot(tree: ast.AST, target: ast.AST) -> Slot:
    slot = find_slot(tree, target)
    if slot is None:
        raise SiteError("node has no parent in this tree", node_type=type(target).__name__)
    return slot


def replace_node(tree: ast.AST, target: ast.AST, new_node: ast.AST) -> None:
    slot = require_slot(tree, target)
    if slot.index is None:
        setattr(slot.parent, slot.field, new_node)
    else:
        getattr(slot.parent, slot.field)[slot.index] = new_node


#: Fields where an empty list is a syntax error and must be padded with ``pass``.
_MUST_BE_NONEMPTY = frozenset({"body", "finalbody"})


def remove_node(tree: ast.AST, target: ast.AST) -> None:
    """Delete a statement from its parent block.

    An emptied ``body`` gets a ``pass`` because an empty body does not unparse;
    an emptied ``orelse`` is left empty because "no else branch" is exactly the
    semantics a deletion on one path is supposed to produce.
    """
    slot = require_slot(tree, target)
    if slot.index is None:
        raise SiteError(
            "cannot remove a node held in a non-list field",
            field=slot.field,
            node_type=type(target).__name__,
        )
    block = getattr(slot.parent, slot.field)
    del block[slot.index]
    if not block and slot.field in _MUST_BE_NONEMPTY:
        block.append(ast.Pass())


def insert_into(block: list[ast.stmt], index: int, stmt: ast.stmt) -> None:
    block.insert(max(0, min(index, len(block))), stmt)


def apply_to_copy(
    tree: ast.AST,
    site: Site,
    transform: Callable[[ast.AST, ast.AST], None],
) -> ast.AST:
    """Deepcopy ``tree``, resolve ``site`` in the copy, transform, fix locations.

    The caller's tree is never observed by ``transform``, so a failed mutation
    cannot leave the baseline in a half-edited state.
    """
    new_tree = copy.deepcopy(tree)
    node = require(new_tree, site)
    transform(new_tree, node)
    ast.fix_missing_locations(new_tree)
    return new_tree


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #


def parse(src: str, filename: str = "<crucible-seed>") -> ast.Module:
    return ast.parse(src, filename=filename)


def unparse(tree: ast.AST) -> str:
    return ast.unparse(tree)


def ensure_trailing_newline(text: str) -> str:
    if not text or text.endswith("\n"):
        return text
    return text + "\n"


def roundtrip(src: str) -> str:
    return unparse(parse(src))


def canonical(src: str) -> str:
    """The normal form both the baseline and the mutant are stored in.

    Comments and original spacing are lost. That is the point: a diff between
    two canonical forms contains the mutation and nothing else.
    """
    return ensure_trailing_newline(roundtrip(src))


def check_roundtrip(src: str) -> bool:
    """True if the source survives parse -> unparse -> parse."""
    try:
        once = roundtrip(src)
        twice = roundtrip(once)
    except SyntaxError as exc:
        logger.debug("round trip failed to parse: %s", exc)
        return False
    return once == twice


def unified_diff(
    a: str,
    b: str,
    fromfile: str = "mutant",
    tofile: str = "baseline",
    n: int = 3,
) -> str:
    """Unified diff turning ``a`` into ``b``.

    Both sides are newline-terminated first so that no hunk ends with a line
    lacking its terminator; that is what makes ``apply_unified_diff`` exact
    rather than approximate.
    """
    a_lines = ensure_trailing_newline(a).splitlines(keepends=True)
    b_lines = ensure_trailing_newline(b).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(a_lines, b_lines, fromfile=fromfile, tofile=tofile, n=n)
    )


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_unified_diff(source: str, diff: str) -> str:
    """Apply a unified diff produced by :func:`unified_diff`.

    Strict on purpose: every context and removal line must match the source
    exactly. A patch that "mostly" applies would produce a baseline that is not
    the baseline, and the whole ground-truth-diff guarantee would be a guess.
    """
    src = ensure_trailing_newline(source).splitlines(keepends=True)
    lines = ensure_trailing_newline(diff).splitlines(keepends=True)
    out: list[str] = []
    idx = 0
    i = 0
    applied = 0

    while i < len(lines):
        match = _HUNK_RE.match(lines[i])
        if match is None:
            i += 1
            continue
        old_start = int(match.group(1))
        old_count = 1 if match.group(2) is None else int(match.group(2))
        # A zero-length old range names the line *before* the insertion point.
        target = old_start if old_count == 0 else old_start - 1
        if target < idx:
            raise SiteError(
                "unified diff hunks overlap or are out of order",
                hunk=lines[i].strip(),
                position=idx,
            )
        out.extend(src[idx:target])
        idx = target
        i += 1
        applied += 1

        while i < len(lines):
            line = lines[i]
            if _HUNK_RE.match(line):
                break
            tag, body = line[:1], line[1:]
            if tag == "\\":  # "\ No newline at end of file"
                i += 1
                continue
            if tag not in (" ", "-", "+"):
                break
            if tag in (" ", "-"):
                if idx >= len(src) or src[idx] != body:
                    found = src[idx] if idx < len(src) else "<end of file>"
                    raise SiteError(
                        "unified diff does not apply: context mismatch",
                        line_number=idx + 1,
                        expected=body.rstrip("\n"),
                        found=found.rstrip("\n"),
                    )
                idx += 1
            if tag in (" ", "+"):
                out.append(body)
            i += 1

    if applied == 0 and diff.strip():
        raise SiteError("diff text contains no hunks")
    out.extend(src[idx:])
    return "".join(out)


__all__ = [
    "Site",
    "SiteError",
    "Slot",
    "walk_ordered",
    "site_for",
    "resolve",
    "require",
    "find_slot",
    "require_slot",
    "replace_node",
    "remove_node",
    "insert_into",
    "apply_to_copy",
    "parse",
    "unparse",
    "roundtrip",
    "canonical",
    "check_roundtrip",
    "ensure_trailing_newline",
    "unified_diff",
    "apply_unified_diff",
]
