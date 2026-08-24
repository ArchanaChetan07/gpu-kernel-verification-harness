"""48-cell coverage, and which gap to fill next.

``taxonomy.CoverageGrid`` already answers "how many tasks are in each cell". The
question this module answers is the one that actually drives work: *which empty
cell is worth filling first?* An empty T6 cell is not worth the same as an empty
T1 cell -- T1 hands the model the answer in the error message, T6 hands it
nothing -- so gaps are ranked by

    priority = tier data-value rank  x  emptiness

where emptiness is the shortfall as a fraction of the per-cell target. A cell
that is half full and highly valuable can outrank a worthless empty one, which
is the intended behaviour: it is the cheapest remaining high-value work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..taxonomy import (
    CELLS,
    DOMAINS,
    SILENT_TIERS,
    TIER_ORDER,
    TIERS,
    CoverageGrid,
    cell_id,
    coverage as taxonomy_coverage,
)


@dataclass(frozen=True)
class CellRow:
    """One of the 48 cells, with everything the report needs about it."""

    cell: str
    tier: str
    domain: str
    count: int
    target: int
    value_rank: int
    silent: bool

    @property
    def shortfall(self) -> int:
        return max(0, self.target - self.count)

    @property
    def fill(self) -> float:
        """Fraction of the per-cell target reached, clamped to 1.0."""
        if self.target <= 0:
            return 1.0
        return min(1.0, self.count / self.target)

    @property
    def emptiness(self) -> float:
        return 1.0 - self.fill

    @property
    def priority(self) -> float:
        """Tier data-value x emptiness. 0.0 for a cell that is already at target."""
        return self.value_rank * self.emptiness

    @property
    def at_target(self) -> bool:
        return self.count >= self.target

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "tier": self.tier,
            "domain": self.domain,
            "count": self.count,
            "target": self.target,
            "shortfall": self.shortfall,
            "fill": self.fill,
            "value_rank": self.value_rank,
            "silent": self.silent,
            "priority": self.priority,
            "at_target": self.at_target,
        }


@dataclass
class CoverageReport:
    grid: CoverageGrid
    cells: list[CellRow] = field(default_factory=list)

    # ---- aggregates -------------------------------------------------------
    def total(self) -> int:
        return self.grid.total()

    def fill_rate(self) -> float:
        return self.grid.fill_rate()

    def silent_share(self) -> float:
        return self.grid.silent_share()

    def n_filled(self) -> int:
        return sum(1 for c in self.cells if c.at_target)

    def n_empty(self) -> int:
        return sum(1 for c in self.cells if c.count == 0)

    # ---- views ------------------------------------------------------------
    def gaps_ranked(self, limit: int | None = None) -> list[CellRow]:
        """Cells below target, most valuable-and-emptiest first.

        Ties break on data value, then on cell id, so the order is stable and
        reproducible across runs.
        """
        gaps = [c for c in self.cells if not c.at_target]
        gaps.sort(key=lambda c: (-c.priority, -c.value_rank, c.cell))
        return gaps[:limit] if limit is not None else gaps

    def heatmap(self) -> list[dict[str, Any]]:
        """Rows for the dashboard table: one per tier, one column per domain."""
        by_cell = {c.cell: c for c in self.cells}
        rows: list[dict[str, Any]] = []
        for tier in TIER_ORDER:
            info = TIERS[tier]
            cells = [by_cell[cell_id(tier, d)] for d in DOMAINS]
            rows.append(
                {
                    "tier": tier,
                    "failure_mode": info.failure_mode,
                    "data_value": info.data_value,
                    "value_rank": info.value_rank,
                    "silent": info.silent,
                    "total": sum(c.count for c in cells),
                    "cells": [c.as_dict() for c in cells],
                }
            )
        return rows

    def by_tier(self) -> dict[str, int]:
        return self.grid.by_tier()

    def by_domain(self) -> dict[str, int]:
        return self.grid.by_domain()

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_per_cell": self.grid.target_per_cell,
            "n_cells": len(CELLS),
            "n_filled": self.n_filled(),
            "n_empty": self.n_empty(),
            "total_tasks": self.total(),
            "fill_rate": self.fill_rate(),
            "silent_share": self.silent_share(),
            "by_tier": self.by_tier(),
            "by_domain": self.by_domain(),
            "cells": [c.as_dict() for c in self.cells],
            "gaps_ranked": [c.as_dict() for c in self.gaps_ranked()],
        }


def analyze(tasks: Sequence[Any], target_per_cell: int = 5) -> CoverageReport:
    """Build a coverage report from Task objects (or plain mappings)."""
    grid = taxonomy_coverage(tasks, target_per_cell=target_per_cell)
    rows: list[CellRow] = []
    for tier, domain in CELLS:
        cid = cell_id(tier, domain)
        rows.append(
            CellRow(
                cell=cid,
                tier=tier,
                domain=domain,
                count=int(grid.counts.get(cid, 0)),
                target=int(grid.target_per_cell),
                value_rank=TIERS[tier].value_rank,
                silent=tier in SILENT_TIERS,
            )
        )
    return CoverageReport(grid=grid, cells=rows)


def analyze_bank(bank_dir: Path | str, target_per_cell: int = 5) -> tuple[CoverageReport, list[Any]]:
    """Load a bank directory and analyze it. Returns (report, load_errors)."""
    from .metrics import load_bank

    bank = load_bank(bank_dir)
    return analyze(bank.tasks, target_per_cell=target_per_cell), bank.errors


__all__ = ["CellRow", "CoverageReport", "analyze", "analyze_bank"]
