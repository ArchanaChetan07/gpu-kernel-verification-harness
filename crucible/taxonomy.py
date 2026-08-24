"""The 6x8 failure grid.

The tier axis is the whole thesis: a compile error hands the model the answer in
the error message, while a silent convergence divergence hands it nothing. Data
value rises monotonically with how little signal the failure leaks. Coverage is
therefore reported per cell, not in aggregate, and the T5+T6 share is a headline
number rather than a footnote.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .schema import Domain, Tier

TIER_ORDER: tuple[Tier, ...] = ("T1", "T2", "T3", "T4", "T5", "T6")


@dataclass(frozen=True)
class TierInfo:
    tier: Tier
    failure_mode: str
    signal_available: str
    model_skill: str
    data_value: str
    value_rank: int  # 1 (lowest) .. 6 (highest)

    @property
    def silent(self) -> bool:
        """Does the failure leave no error and no obviously wrong output?"""
        return self.value_rank >= 5


TIERS: dict[Tier, TierInfo] = {
    "T1": TierInfo(
        tier="T1",
        failure_mode="compile error",
        signal_available="full error message",
        model_skill="read the error",
        data_value="low",
        value_rank=1,
    ),
    "T2": TierInfo(
        tier="T2",
        failure_mode="runtime crash (illegal access, assert, OOB)",
        signal_available="stack trace and faulting operation",
        model_skill="map a crash back to an indexing or sizing bug",
        data_value="low-medium",
        value_rank=2,
    ),
    "T3": TierInfo(
        tier="T3",
        failure_mode="loud wrong answer (NaN, inf, wrong shape)",
        signal_available="output is visibly wrong on any input",
        model_skill="compare against a reference and localize",
        data_value="medium",
        value_rank=3,
    ),
    "T4": TierInfo(
        tier="T4",
        failure_mode="performance regression, results still correct",
        signal_available="it works, it is just slow",
        model_skill="reason about memory traffic, occupancy and cache reuse",
        data_value="medium-high",
        value_rank=4,
    ),
    "T5": TierInfo(
        tier="T5",
        failure_mode="silent numerical divergence",
        signal_available="correct on typical shapes, wrong on adversarial ones",
        model_skill="derive an error budget and find the witness shape",
        data_value="high",
        value_rank=5,
    ),
    "T6": TierInfo(
        tier="T6",
        failure_mode="silent convergence divergence",
        signal_available="loss goes down, to the wrong place",
        model_skill="multi-rank differential reasoning",
        data_value="highest",
        value_rank=6,
    ),
}

DOMAINS: tuple[Domain, ...] = (
    "triton",
    "pallas",
    "cuda",
    "jax",
    "pytorch",
    "distributed",
    "data_pipeline",
    "checkpointing",
)

CELLS: list[tuple[Tier, Domain]] = [(t, d) for t in TIER_ORDER for d in DOMAINS]

SILENT_TIERS: frozenset[str] = frozenset(t for t, info in TIERS.items() if info.silent)


def cell_id(tier: Tier | str, domain: Domain | str) -> str:
    """`"T5/triton"`. Raises on an unknown tier or domain rather than inventing a cell."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {list(TIER_ORDER)}")
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {list(DOMAINS)}")
    return f"{tier}/{domain}"


def parse_cell(cid: str) -> tuple[Tier, Domain]:
    tier, _, domain = cid.partition("/")
    if tier not in TIERS or domain not in DOMAINS:
        raise ValueError(f"malformed cell id {cid!r}")
    return tier, domain  # type: ignore[return-value]


def all_cell_ids() -> list[str]:
    return [cell_id(t, d) for t, d in CELLS]


@dataclass
class CoverageGrid:
    counts: dict[str, int] = field(default_factory=dict)
    target_per_cell: int = 5

    def __post_init__(self) -> None:
        # Every cell is present, including the empty ones. A gap that is missing
        # from the report is a gap that never gets filled.
        for cid in all_cell_ids():
            self.counts.setdefault(cid, 0)

    def total(self) -> int:
        return int(sum(self.counts.values()))

    def filled(self) -> list[str]:
        """Cells that have reached the per-cell target."""
        return [cid for cid in all_cell_ids() if self.counts.get(cid, 0) >= self.target_per_cell]

    def gaps(self) -> list[tuple[str, int]]:
        """(cell, shortfall) for every cell below target, worst shortfall first."""
        out = [
            (cid, self.target_per_cell - self.counts.get(cid, 0))
            for cid in all_cell_ids()
            if self.counts.get(cid, 0) < self.target_per_cell
        ]
        out.sort(key=lambda kv: (-kv[1], kv[0]))
        return out

    def fill_rate(self) -> float:
        """Fraction of the 48 cells that have reached the target."""
        return len(self.filled()) / len(CELLS)

    def silent_share(self) -> float:
        """(T5 + T6) share of all tasks. 0.0 when the bank is empty."""
        total = self.total()
        if total == 0:
            return 0.0
        silent = sum(n for cid, n in self.counts.items() if parse_cell(cid)[0] in SILENT_TIERS)
        return silent / total

    def by_tier(self) -> dict[str, int]:
        out: dict[str, int] = {t: 0 for t in TIER_ORDER}
        for cid, n in self.counts.items():
            out[parse_cell(cid)[0]] += n
        return out

    def by_domain(self) -> dict[str, int]:
        out: dict[str, int] = {d: 0 for d in DOMAINS}
        for cid, n in self.counts.items():
            out[parse_cell(cid)[1]] += n
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "target_per_cell": self.target_per_cell,
            "total": self.total(),
            "n_filled": len(self.filled()),
            "fill_rate": self.fill_rate(),
            "silent_share": self.silent_share(),
            "gaps": [{"cell": c, "shortfall": s} for c, s in self.gaps()],
        }


def _tier_domain(task: Any) -> tuple[str, str]:
    if isinstance(task, dict):
        return str(task.get("failure_tier", "")), str(task.get("domain", ""))
    return str(getattr(task, "failure_tier", "")), str(getattr(task, "domain", ""))


def coverage(tasks: Iterable[Any], target_per_cell: int = 5) -> CoverageGrid:
    """Count tasks per cell. Accepts Task objects or plain mappings."""
    counter: Counter[str] = Counter()
    for task in tasks:
        tier, domain = _tier_domain(task)
        counter[cell_id(tier, domain)] += 1
    return CoverageGrid(counts=dict(counter), target_per_cell=target_per_cell)


__all__ = [
    "TierInfo",
    "TIERS",
    "TIER_ORDER",
    "DOMAINS",
    "CELLS",
    "SILENT_TIERS",
    "CoverageGrid",
    "cell_id",
    "parse_cell",
    "all_cell_ids",
    "coverage",
]
