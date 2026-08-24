"""Mutation engine: turn a known-good seed into a task with a proven witness.

Modules (see docs/CONTRACT.md section 6): ``astutil``, ``classes``, ``witness``,
``engine``. A mutation with no witness is discarded; that discard is the
structural guarantee that every shipped task has a real, reachable answer.

Nothing is re-exported here so that a missing sibling module cannot break an
import of the package.
"""

from __future__ import annotations

__all__: list[str] = []
