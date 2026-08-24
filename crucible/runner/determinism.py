"""Make runs repeatable, and say so honestly when they cannot be.

Every helper here returns or reports what it actually achieved. ``lock_clocks``
is the sharp edge: it yields a record saying whether the lock took effect, and
it issues ``nvidia-smi -rgc`` in a ``finally`` so an exception in the timed
region can never leave the GPU pinned for the next user of the machine.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..capabilities import run_nvidia_smi

logger = logging.getLogger(__name__)

CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_DETERMINISTIC = ":4096:8"


def seed_everything(n: int) -> int:
    """Seed python, numpy and torch (host and device). Returns the seed used."""
    random.seed(n)
    os.environ["PYTHONHASHSEED"] = str(n)  # affects child processes only
    try:
        import numpy as np

        np.random.seed(n % (2**32))
    except ImportError as exc:
        logger.debug("numpy unavailable while seeding: %s", exc)
    try:
        import torch

        torch.manual_seed(n)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(n)
    except ImportError as exc:
        logger.debug("torch unavailable while seeding: %s", exc)
    return n


@contextmanager
def deterministic_ctx(warn_only: bool = True) -> Iterator[dict[str, Any]]:
    """Turn on deterministic kernels for the duration of the block.

    ``CUBLAS_WORKSPACE_CONFIG`` is honoured by cuBLAS only if it is set before
    the handle is created, so setting it here helps subprocesses and any handle
    created later; the returned record says whether it was already set, which is
    what a report should quote rather than assuming determinism was achieved.
    """
    record: dict[str, Any] = {
        "requested": True,
        "warn_only": warn_only,
        "cublas_env_preset": os.environ.get(CUBLAS_ENV),
        "applied": False,
        "reason": "",
    }
    prev_env = os.environ.get(CUBLAS_ENV)
    os.environ[CUBLAS_ENV] = CUBLAS_DETERMINISTIC
    prev_flag: bool | None = None
    torch: Any = None
    try:
        import torch as _torch

        torch = _torch
    except ImportError as exc:
        record["reason"] = f"torch unavailable: {exc}"

    if torch is not None:
        try:
            prev_flag = bool(torch.are_deterministic_algorithms_enabled())
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            record["applied"] = True
        except (RuntimeError, AttributeError) as exc:
            record["reason"] = f"use_deterministic_algorithms failed: {exc}"

    try:
        yield record
    finally:
        if torch is not None and prev_flag is not None:
            try:
                torch.use_deterministic_algorithms(prev_flag, warn_only=warn_only)
            except (RuntimeError, AttributeError) as exc:
                logger.warning("could not restore deterministic-algorithms flag: %s", exc)
        if prev_env is None:
            os.environ.pop(CUBLAS_ENV, None)
        else:
            os.environ[CUBLAS_ENV] = prev_env


def _candidate_cache_dirs() -> list[Path]:
    dirs: list[Path] = []
    for env_name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_COMPILE_CACHE_DIR"):
        val = os.environ.get(env_name)
        if val:
            dirs.append(Path(val))
    try:
        from torch._inductor.runtime.runtime_utils import cache_dir as _inductor_cache_dir

        dirs.append(Path(_inductor_cache_dir()))
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("inductor cache_dir unavailable: %s", exc)
    dirs.append(Path.home() / ".triton" / "cache")
    return dirs


def purge_autotune_cache() -> dict[str, Any]:
    """Delete inductor/triton autotune caches so a timing run starts cold.

    Only paths that are recognisably a triton or inductor cache are touched; a
    mis-set environment variable pointing at a source tree must not become a
    delete. The refusals are reported, not swallowed.
    """
    removed: list[str] = []
    refused: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    for d in _candidate_cache_dirs():
        try:
            resolved = d.resolve()
        except OSError as exc:
            errors.append(f"{d}: {exc}")
            continue
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if not resolved.exists():
            continue
        if not any(tok in key for tok in ("triton", "inductor", "torchinductor")):
            refused.append(str(resolved))
            continue
        try:
            shutil.rmtree(resolved, ignore_errors=False)
            removed.append(str(resolved))
        except OSError as exc:
            errors.append(f"{resolved}: {exc}")
    return {"removed": removed, "refused": refused, "errors": errors}


@dataclass
class ClockLock:
    """What the clock lock attempt actually achieved."""

    locked: bool = False
    requested_mhz: int | None = None
    reason: str = ""
    reset_ok: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "clock_locked": self.locked,
            "requested_mhz": self.requested_mhz,
            "reason": self.reason,
            "reset_ok": self.reset_ok,
            **self.details,
        }


def unlock_clocks() -> tuple[bool, str]:
    """Reset application graphics clocks. (ok, message)."""
    rc, out, err = run_nvidia_smi(["-rgc"])
    if rc == 0:
        return True, "clocks reset"
    return False, (err or out or f"nvidia-smi -rgc exited {rc}").strip()[:400]


@contextmanager
def lock_clocks(mhz: int | None = None, enabled: bool = True) -> Iterator[ClockLock]:
    """Pin the graphics clock for the duration of the block.

    The yielded ``ClockLock`` reports the truth: on a machine without admin
    rights or without nvidia-smi, ``locked`` is False with the reason attached,
    and the caller is expected to downgrade its verdict rather than pretend the
    measurement was clock-stable. The reset runs in ``finally`` unconditionally.
    """
    lock = ClockLock()
    if not enabled:
        lock.reason = "clock locking disabled by caller"
        yield lock
        return

    target = mhz
    if target is None:
        rc, out, err = run_nvidia_smi(
            ["--query-gpu=clocks.max.graphics", "--format=csv,noheader,nounits"]
        )
        if rc != 0:
            lock.reason = (err or out or f"nvidia-smi query exited {rc}").strip()[:400]
            yield lock
            return
        first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
        try:
            target = int(float(first.split()[0]))
        except (ValueError, IndexError):
            lock.reason = f"could not parse max graphics clock from {first!r}"
            yield lock
            return

    lock.requested_mhz = target
    rc, out, err = run_nvidia_smi(["-lgc", f"{target},{target}"])
    lock.locked = rc == 0
    if not lock.locked:
        lock.reason = (err or out or f"nvidia-smi -lgc exited {rc}").strip()[:400]
    try:
        yield lock
    finally:
        # Always restore, even if the body raised and even if we believe the
        # lock never took: a partially applied lock still needs resetting.
        ok, msg = unlock_clocks()
        lock.reset_ok = ok
        if lock.locked and not ok:
            logger.error("clocks were locked at %s MHz and could not be reset: %s", target, msg)


@contextlib.contextmanager
def cuda_cache_cleared() -> Iterator[None]:
    """Empty the CUDA caching allocator around a measurement, if CUDA exists."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
        torch = None  # type: ignore[assignment]
    if has_cuda and torch is not None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        yield
    finally:
        if has_cuda and torch is not None:
            torch.cuda.synchronize()


__all__ = [
    "seed_everything",
    "deterministic_ctx",
    "purge_autotune_cache",
    "lock_clocks",
    "unlock_clocks",
    "cuda_cache_cleared",
    "ClockLock",
    "CUBLAS_ENV",
]
