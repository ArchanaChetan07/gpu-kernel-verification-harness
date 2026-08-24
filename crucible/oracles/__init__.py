"""Oracle package. Concrete oracles are imported lazily and self-register.

Importing this package must not require every oracle module to exist yet, and a
module that fails to import must be visible (``load_errors()``) rather than
quietly reducing the number of checks a task is graded against.
"""

from __future__ import annotations

import importlib
import logging

from .base import (
    ORACLES,
    Oracle,
    OracleContext,
    gate_capabilities,
    register_oracle,
    run_all,
    run_one,
)

logger = logging.getLogger(__name__)

#: module suffix -> oracle id it is expected to provide
ORACLE_MODULES: dict[str, str] = {
    "o1_numerics": "O1",
    "o2_perf": "O2",
    "o3_anticheat": "O3",
    "o4_compile": "O4",
    "o5_nrank": "O5",
}

_LOAD_ERRORS: dict[str, str] = {}
_LOADED = False


def _missing_ids() -> set[str]:
    """Expected oracle ids that are not currently in the registry."""
    return {oid for oid in ORACLE_MODULES.values() if oid not in ORACLES}


def _module_oracle(module: object, oracle_id: str) -> Oracle | None:
    """Recover an already-imported module's oracle instance.

    Oracles register themselves at module scope, so registration runs exactly
    once per interpreter. If the registry is cleared or rebuilt after that
    first import, re-importing is a no-op and the oracle stays missing --
    which would grade tasks against fewer checks than they declare, without
    saying so. Fetching the instance off the module lets the loader converge
    on a complete registry instead of trusting that it already did.
    """
    for attr in (oracle_id, "ORACLE"):
        obj = getattr(module, attr, None)
        if obj is not None and getattr(obj, "id", None) == oracle_id:
            return obj  # type: ignore[return-value]
    return None


def load_oracles(force: bool = False) -> dict[str, Oracle]:
    """Import o1..o5 and return the populated registry.

    The fast path is guarded by registry COMPLETENESS, not by a one-shot flag.
    An oracle absent for any reason other than a recorded load error is a
    missing check, and a missing check must never be silent -- that is the
    same class of failure this project exists to detect in the tasks it ships.
    """
    global _LOADED
    if _LOADED and not force and _missing_ids() <= set(_LOAD_ERRORS):
        return ORACLES
    for mod_name, oracle_id in ORACLE_MODULES.items():
        full = f"{__name__}.{mod_name}"
        try:
            module = importlib.import_module(full)
        except Exception as exc:  # noqa: BLE001
            # Deliberately broader than ImportError. A module with a syntax
            # error, or one that raises while executing at import time, is a
            # missing check like any other -- and it must be RECORDED, not
            # allowed to abort the whole harness and take the working oracles
            # down with it.
            _LOAD_ERRORS[oracle_id] = f"{type(exc).__name__}: {exc}"
            logger.debug("oracle module %s unavailable: %s", full, exc)
            continue
        _LOAD_ERRORS.pop(oracle_id, None)
        if oracle_id not in ORACLES:
            recovered = _module_oracle(module, oracle_id)
            if recovered is not None:
                register_oracle(recovered)
            else:
                _LOAD_ERRORS[oracle_id] = f"{full} imported but did not register {oracle_id}"
                logger.warning("%s imported but did not register %s", full, oracle_id)
    _LOADED = True
    return ORACLES


def load_errors() -> dict[str, str]:
    """Oracle ids that could not be loaded, with the reason."""
    load_oracles()
    return dict(_LOAD_ERRORS)


def available() -> list[str]:
    return sorted(load_oracles())


__all__ = [
    "ORACLES",
    "Oracle",
    "OracleContext",
    "register_oracle",
    "run_all",
    "run_one",
    "gate_capabilities",
    "load_oracles",
    "load_errors",
    "available",
    "ORACLE_MODULES",
]
