"""Known-good baselines. Seed modules self-register; see ``registry.discover``."""

from __future__ import annotations

from .registry import (
    CompareResult,
    SeedSpec,
    all_seeds,
    by_domain,
    by_tier,
    discover,
    get,
    import_errors,
    register,
    resolve,
    seed_ids,
    unregister,
)

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
