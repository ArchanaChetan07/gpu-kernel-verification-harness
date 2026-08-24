"""CRUCIBLE: an execution-grounded task foundry for ML-systems training data.

The invariant the whole package exists to enforce: a task ships only if its
reference solution was executed on real hardware and passed every applicable
oracle. SKIP is never PASS.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .capabilities import Capabilities, detect
from .config import Config
from .errors import (
    CapabilityError,
    CrucibleError,
    ProbeError,
    SandboxError,
    WitnessNotFound,
)
from .schema import OracleResult, Task, TaskVerdict

__all__ = [
    "__version__",
    "Task",
    "TaskVerdict",
    "OracleResult",
    "detect",
    "Capabilities",
    "Config",
    "CrucibleError",
    "SandboxError",
    "CapabilityError",
    "WitnessNotFound",
    "ProbeError",
]
