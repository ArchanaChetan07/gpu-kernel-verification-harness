"""Seed registry: the known-good baselines every task is mutated from.

Seed modules are discovered, not listed. ``discover()`` imports every sibling
module in this package, and each one registers its ``SEED`` on import. Adding a
seed therefore means adding one file, and a seed that fails to import is
reported by name rather than silently vanishing from the bank.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..schema import Domain, ShapeSpec, Tier, sha256_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompareResult:
    """Outcome of comparing a candidate output against the reference.

    ``ok`` is the verdict; the error magnitudes are kept even when ok is True so
    that a report can show the margin, not just the pass.
    """

    ok: bool
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    detail: str = ""
    kind: str = "numeric"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "max_abs_err": self.max_abs_err,
            "max_rel_err": self.max_rel_err,
            "detail": self.detail,
            "kind": self.kind,
        }


@dataclass
class SeedSpec:
    id: str
    domain: Domain
    tiers: tuple[Tier, ...]
    description: str
    entry: str
    source: str
    make_inputs: Callable[..., dict[str, Any]]
    reference: Callable[..., Any]
    shape_sweep: list[ShapeSpec]
    accum_depth: Callable[[ShapeSpec], int]
    bytes_moved: Callable[[ShapeSpec], int]
    flops: Callable[[ShapeSpec], int]
    denylist: tuple[str, ...] = ()
    compare: Callable[[Any, Any], CompareResult] | None = None
    supports_cpu: bool = True
    module: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("SeedSpec.id must be non-empty")
        if not self.entry:
            raise ValueError(f"seed {self.id!r}: entry must be non-empty")
        if self.entry not in self.source:
            raise ValueError(
                f"seed {self.id!r}: entry {self.entry!r} does not appear in source; "
                "the baseline must actually define the function candidates must write"
            )
        if not self.tiers:
            raise ValueError(f"seed {self.id!r}: must advertise at least one tier")

    def content_sha256(self) -> str:
        return sha256_text(self.source)

    def shape(self, name: str) -> ShapeSpec:
        for s in self.shape_sweep:
            if s.name == name:
                return s
        raise KeyError(f"seed {self.id!r} has no shape named {name!r}")

    def short_id(self) -> str:
        """Compact token used in task ids: 'attention.blocked_fwd' -> 'blocked_fwd'."""
        return self.id.split(".")[-1]


_REGISTRY: dict[str, SeedSpec] = {}
_DISCOVERED = False
_IMPORT_ERRORS: dict[str, str] = {}


def register(obj: SeedSpec | Callable[[], SeedSpec]) -> SeedSpec:
    """Register a seed. Usable as ``SEED = register(SeedSpec(...))`` or as a
    decorator on a zero-argument factory returning a SeedSpec."""
    spec = obj() if callable(obj) and not isinstance(obj, SeedSpec) else obj
    if not isinstance(spec, SeedSpec):
        raise TypeError(f"register expected a SeedSpec, got {type(spec).__name__}")
    existing = _REGISTRY.get(spec.id)
    if existing is not None and existing is not spec:
        raise ValueError(f"duplicate seed id {spec.id!r}")
    _REGISTRY[spec.id] = spec
    return spec


def unregister(seed_id: str) -> None:
    """Remove a seed (tests register synthetic seeds and must clean up)."""
    _REGISTRY.pop(seed_id, None)


def get(seed_id: str) -> SeedSpec:
    discover()
    try:
        return _REGISTRY[seed_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise KeyError(f"unknown seed {seed_id!r}; known seeds: {known}") from None


def all_seeds() -> list[SeedSpec]:
    discover()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def seed_ids() -> list[str]:
    discover()
    return sorted(_REGISTRY)


def import_errors() -> dict[str, str]:
    """Modules in the seeds package that could not be imported, and why."""
    discover()
    return dict(_IMPORT_ERRORS)


def discover(force: bool = False) -> list[str]:
    """Import every sibling module so seeds self-register. Returns seed ids."""
    global _DISCOVERED
    if _DISCOVERED and not force:
        return sorted(_REGISTRY)

    package = importlib.import_module(__package__ or "crucible.seeds")
    for info in pkgutil.iter_modules(list(getattr(package, "__path__", []))):
        name = info.name
        if name.startswith("_") or name == "registry":
            continue
        full = f"{package.__name__}.{name}"
        try:
            module = importlib.import_module(full)
        except ImportError as exc:
            _IMPORT_ERRORS[full] = f"{type(exc).__name__}: {exc}"
            logger.warning("seed module %s could not be imported: %s", full, exc)
            continue
        spec = getattr(module, "SEED", None)
        if isinstance(spec, SeedSpec):
            if not spec.module:
                spec.module = full
            if spec.id not in _REGISTRY:
                register(spec)
        elif spec is not None:
            _IMPORT_ERRORS[full] = "module defines SEED but it is not a SeedSpec"
    _DISCOVERED = True
    return sorted(_REGISTRY)


def by_domain(domain: str) -> list[SeedSpec]:
    return [s for s in all_seeds() if s.domain == domain]


def by_tier(tier: str) -> list[SeedSpec]:
    return [s for s in all_seeds() if tier in s.tiers]


def resolve(seed_ids_in: Iterable[str] | None) -> list[SeedSpec]:
    """Resolve a possibly-None id list to seeds; None means every seed."""
    if seed_ids_in is None:
        return all_seeds()
    return [get(sid) for sid in seed_ids_in]


__all__ = [
    "SeedSpec",
    "CompareResult",
    "register",
    "unregister",
    "get",
    "all_seeds",
    "seed_ids",
    "discover",
    "import_errors",
    "by_domain",
    "by_tier",
    "resolve",
]
