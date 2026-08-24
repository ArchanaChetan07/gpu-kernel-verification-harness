"""Difficulty calibration: pass@k, routing, and 2PL item-response fitting.

Modules (see docs/CONTRACT.md section 9): ``passk``, ``models``, ``irt``,
``router``. The default target model is the offline ``StubModel``: no network
call may happen unless the user explicitly passes a non-stub ``--model``.
"""

from __future__ import annotations

__all__: list[str] = []
