"""Execution plumbing: out-of-process candidate execution and determinism control."""

from __future__ import annotations

from .determinism import (
    ClockLock,
    deterministic_ctx,
    lock_clocks,
    purge_autotune_cache,
    seed_everything,
    unlock_clocks,
)
from .sandbox import (
    SandboxResult,
    call_entry,
    input_refs_from,
    run_job,
    run_source,
    time_entry,
)

__all__ = [
    "SandboxResult",
    "run_job",
    "run_source",
    "call_entry",
    "time_entry",
    "input_refs_from",
    "seed_everything",
    "deterministic_ctx",
    "purge_autotune_cache",
    "lock_clocks",
    "unlock_clocks",
    "ClockLock",
]
