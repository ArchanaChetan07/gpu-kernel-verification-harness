"""Exception hierarchy for CRUCIBLE.

Every failure that we can attribute to a specific subsystem gets its own type so
that callers can distinguish "the machine cannot do this" (CapabilityError,
ProbeError) from "the candidate did something wrong" (SandboxError) from "the
mutation was semantically neutral" (WitnessNotFound). Bare ``except`` is banned
project-wide, so these types are the contract for what may be caught.
"""

from __future__ import annotations

from typing import Any


class CrucibleError(Exception):
    """Base class for every CRUCIBLE-raised error."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context)

    def __str__(self) -> str:
        if not self.context:
            return self.message
        extra = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({extra})"


class SandboxError(CrucibleError):
    """A subprocess execution of candidate code failed, timed out, or died.

    ``exit_code`` is None when the process was killed by the harness.
    """

    def __init__(
        self,
        message: str,
        *,
        exit_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
        **context: Any,
    ) -> None:
        super().__init__(message, **context)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


class CapabilityError(CrucibleError):
    """A required capability is missing, or an unknown capability was named."""

    def __init__(self, message: str, *, missing: list[str] | None = None, **context: Any) -> None:
        super().__init__(message, **context)
        self.missing = list(missing or [])


class WitnessNotFound(CrucibleError):
    """A mutation produced no observable difference on any swept shape.

    This is not a bug; it is the discard signal that keeps unverifiable tasks
    out of the bank.
    """


class ProbeError(CrucibleError):
    """A capability probe could not be executed (not: probed and found absent)."""


__all__ = [
    "CrucibleError",
    "SandboxError",
    "CapabilityError",
    "WitnessNotFound",
    "ProbeError",
]
