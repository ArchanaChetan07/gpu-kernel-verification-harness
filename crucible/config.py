"""Runtime configuration.

One immutable-ish object carries every tunable so that no module hardcodes a
timeout, a resample count, or a tolerance factor. Defaults are the values the
proposal commits to; a YAML file may override any of them, and an unknown key is
an error rather than a silent no-op (a typo in a config is a silent experiment
change otherwise).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import CrucibleError

DEFAULT_ORACLE_TIMEOUTS: dict[str, float] = {
    "O1": 300.0,
    "O2": 900.0,
    "O3": 300.0,
    "O4": 600.0,
    "O5": 1200.0,
}


class Config(BaseModel):
    """Every knob in the system. Constructed with defaults; overridden from YAML."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # --- execution -----------------------------------------------------------
    oracle_timeouts_s: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_ORACLE_TIMEOUTS))
    default_oracle_timeout_s: float = 300.0
    sandbox_timeout_s: float = 120.0
    probe_timeout_s: float = 60.0
    rng_seed: int = 1234

    # --- statistics ----------------------------------------------------------
    bootstrap_resamples: int = 2000
    ci_level: float = 0.95

    # --- performance measurement (O2) ---------------------------------------
    perf_reps: int = 30
    perf_warmup: int = 10
    perf_min_duration_s: float = 0.0
    require_clock_lock: bool = False

    # --- witness search (mutate/witness.py) ---------------------------------
    witness_max_shapes: int = 24
    witness_time_budget_s: float = 600.0
    # Continue past the first disagreement by default. Stopping there finds a
    # witness but leaves decoy_shapes empty, and decoys ARE the difficulty
    # dial: shapes where the bug does not reproduce are what force genuine
    # hypothesis formation instead of trial-and-error. Completing the sweep
    # also yields more than one detect shape, so a solution that special-cases
    # the single breaking shape cannot pass the held-out-shape check.
    witness_stop_on_first: bool = False
    # Performance-witness search (T4: correct output, wrong speed). A regression
    # is admitted only when the CI's LOWER bound clears this ratio, so noise
    # cannot manufacture a task.
    perf_witness_enabled: bool = True
    perf_witness_min_ratio: float = 1.15
    perf_witness_reps: int = 30
    perf_witness_warmup: int = 10
    tolerance_safety: float = 4.0
    tolerance_mode: str = "stochastic"

    # --- multi-rank differential (O5) ---------------------------------------
    nrank_k: float = 3.0
    nrank_world_sizes: list[int] = Field(default_factory=lambda: [2, 4])
    nrank_steps: int = 50
    nrank_noise_floor_runs: int = 2

    # --- calibration ---------------------------------------------------------
    pass_at_k: int = 8
    calibration_samples: int = 16
    calibration_temperature: float = 0.8

    # --- rubric / IRR --------------------------------------------------------
    alpha_gate: float = 0.67
    irr_metric: str = "ordinal"

    # --- coverage ------------------------------------------------------------
    target_per_cell: int = 5

    # --- paths ---------------------------------------------------------------
    bank_dir: str = "bank"
    work_root: str = ".crucible/work"

    @field_validator("tolerance_mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        if v not in ("stochastic", "deterministic"):
            raise ValueError("tolerance_mode must be 'stochastic' or 'deterministic'")
        return v

    @field_validator("irr_metric")
    @classmethod
    def _check_metric(cls, v: str) -> str:
        if v not in ("nominal", "ordinal", "interval", "ratio"):
            raise ValueError("irr_metric must be one of nominal|ordinal|interval|ratio")
        return v

    @field_validator("nrank_world_sizes")
    @classmethod
    def _check_world_sizes(cls, v: list[int]) -> list[int]:
        if not v or any(int(n) < 2 for n in v):
            raise ValueError("nrank_world_sizes must be non-empty and every size >= 2")
        return [int(n) for n in v]

    def timeout_for(self, oracle_id: str) -> float:
        """Per-oracle wall-clock budget, falling back to the global default."""
        return float(self.oracle_timeouts_s.get(oracle_id, self.default_oracle_timeout_s))

    def work_dir(self, root: Path | str | None = None) -> Path:
        base = Path(root) if root is not None else Path.cwd()
        return base / self.work_root

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        """Defaults, optionally overlaid with a YAML mapping.

        A missing path yields pure defaults. A malformed or non-mapping document
        is an error: silently falling back to defaults would hide a broken run
        configuration.
        """
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            raise CrucibleError("config file not found", path=str(p))
        try:
            raw: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise CrucibleError(f"config file is not valid YAML: {exc}", path=str(p)) from exc
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise CrucibleError("config file must contain a YAML mapping", path=str(p))
        return cls(**raw)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True, allow_unicode=True)

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(), encoding="utf-8")
        return p


DEFAULT_CONFIG = Config()

__all__ = ["Config", "DEFAULT_CONFIG", "DEFAULT_ORACLE_TIMEOUTS"]
