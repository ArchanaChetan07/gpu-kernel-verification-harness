"""Run every applicable attack through the real oracle harness.

This module answers one question per attack: *did the grader catch it, and was
it caught by the check that was supposed to catch it?* Nothing is simulated -
each attack is generated as source, handed to
:func:`crucible.oracles.base.run_all`, and executed in the same subprocess
sandbox a real candidate would be.

Three outcomes, and they are not the same thing:

* **Caught by the expected check.** The intended reading.
* **Caught by a different check.** A *warning*. The harness still refuses the
  hack, but the check that was designed for it did not fire, which usually
  means the attack drifted or two checks overlap more than we think. Worth
  reporting; not a failure.
* **Caught by nothing.** A **grader defect**. The suite fails loudly and says
  so in those words. The failure is not in the attack: the attack is a
  faithful description of something a model will do, and the harness has no
  check that can see it. Every candidate shipping that hack would be recorded
  as a pass. The fix is a new or repaired oracle check - never a weakened,
  retuned or deleted attack.

The honest control is part of the suite for the mirror-image reason. A grader
that fails the seed's own baseline is not a strict grader, it is a broken one,
and a red-team run that only measured catch rate would report that as perfect.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..capabilities import Capabilities
from ..config import DEFAULT_CONFIG, Config
from ..errors import CrucibleError
from ..oracles.base import OracleContext, run_all
from ..schema import MutationSpec, OracleResult, SeedRef, ShapeSpec, Task, sha256_text
from . import attacks as attacks_mod
from .attacks import DEFAULT_SUBCHECK, Attack, AttackError

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")

#: Default number of shapes each attack is graded on. The task's shown shape
#: plus its held-out shapes is enough to separate "wrong everywhere" from
#: "right only on the shape it saw", and a full seed sweep costs one subprocess
#: per shape per check. Set to None to grade the seed's whole sweep.
DEFAULT_SHAPE_BUDGET = 3


class GraderDefect(CrucibleError):
    """An attack no oracle caught. The grader is the defect, not the attack."""


def _safe(name: str, fallback: str = "x") -> str:
    cleaned = _UNSAFE.sub("_", str(name)).strip("._")
    return cleaned or fallback


# --------------------------------------------------------------------------- #
# reading a verdict: which oracle, and which of its sub-checks, fired
# --------------------------------------------------------------------------- #


def fired_checks(result: OracleResult) -> list[str]:
    """The sub-check names that fired inside one oracle's FAIL.

    O3 reports its five checks individually in ``evidence["fired"]``; oracles
    that carry a single verdict get their conventional sub-check name so that
    an ``expected_catcher`` is always comparable.
    """
    evidence = result.evidence or {}
    fired = evidence.get("fired")
    if isinstance(fired, Sequence) and not isinstance(fired, (str, bytes)) and fired:
        return [str(item) for item in fired]
    checks = evidence.get("checks")
    if isinstance(checks, Mapping):
        named = [
            str(name)
            for name, body in checks.items()
            if isinstance(body, Mapping) and body.get("verdict") in ("FAIL", "ERROR")
        ]
        if named:
            return named
    return [DEFAULT_SUBCHECK.get(result.oracle, result.oracle.lower())]


def catchers_of(results: Iterable[OracleResult]) -> list[str]:
    """``["O3/static_denylist", ...]`` for every oracle that returned FAIL.

    Only FAIL counts as a catch. SKIP is not a pass and it is not a catch
    either - an oracle that could not run saw nothing. ERROR is the harness
    breaking, which is reported separately rather than credited as a detection.
    """
    out: list[str] = []
    for result in results:
        if result.verdict != "FAIL":
            continue
        for check in fired_checks(result):
            token = f"{result.oracle}/{check}"
            if token not in out:
                out.append(token)
    return out


# --------------------------------------------------------------------------- #
# report types
# --------------------------------------------------------------------------- #


@dataclass
class AttackOutcome:
    """One attack, run against one (task, seed), through the real oracles."""

    attack_id: str
    seed_id: str
    task_id: str
    expected_catcher: str
    caught: bool
    catchers: list[str] = field(default_factory=list)
    expected_matched: bool = False
    verdict: str = "SKIP"
    oracle_verdicts: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    errored: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    source_sha256: str = ""
    source_path: str = ""
    generation_error: str = ""

    @property
    def mismatched(self) -> bool:
        """Caught, but not by the check that was supposed to catch it."""
        return self.caught and not self.expected_matched

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.attack_id,
            "seed_id": self.seed_id,
            "task_id": self.task_id,
            "expected_catcher": self.expected_catcher,
            "caught": self.caught,
            "catcher": self.catchers[0] if self.catchers else None,
            "catchers": list(self.catchers),
            "expected_matched": self.expected_matched,
            "mismatched": self.mismatched,
            "verdict": self.verdict,
            "oracle_verdicts": dict(self.oracle_verdicts),
            "reasons": dict(self.reasons),
            "errored": list(self.errored),
            "duration_s": round(self.duration_s, 4),
            "source_sha256": self.source_sha256,
            "source_path": self.source_path,
            "generation_error": self.generation_error,
        }


@dataclass
class ControlOutcome:
    """The seed's own baseline, graded by the same harness."""

    seed_id: str
    task_id: str
    verdict: str
    flagged_by: list[str] = field(default_factory=list)
    oracle_verdicts: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def false_positive(self) -> bool:
        return bool(self.flagged_by)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "task_id": self.task_id,
            "verdict": self.verdict,
            "flagged_by": list(self.flagged_by),
            "false_positive": self.false_positive,
            "oracle_verdicts": dict(self.oracle_verdicts),
            "reasons": dict(self.reasons),
            "duration_s": round(self.duration_s, 4),
        }


@dataclass
class SkippedAttack:
    attack_id: str
    seed_id: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.attack_id, "seed_id": self.seed_id, "reason": self.reason}


@dataclass
class SuiteReport:
    """What the red team found. ``ok`` is the ship gate."""

    attacks: list[AttackOutcome] = field(default_factory=list)
    controls: list[ControlOutcome] = field(default_factory=list)
    skipped: list[SkippedAttack] = field(default_factory=list)
    seeds: list[str] = field(default_factory=list)
    oracles_run: list[str] = field(default_factory=list)
    shape_budget: int | None = DEFAULT_SHAPE_BUDGET
    device: str = "cpu"
    started_utc: str = ""
    duration_s: float = 0.0

    # -- aggregates -------------------------------------------------------- #

    @property
    def n_attacks(self) -> int:
        return len(self.attacks)

    @property
    def n_caught(self) -> int:
        return sum(1 for a in self.attacks if a.caught)

    @property
    def catch_rate(self) -> float | None:
        """None, not 1.0, when nothing ran: an unmeasured rate is not a perfect one."""
        if not self.attacks:
            return None
        return self.n_caught / len(self.attacks)

    @property
    def uncaught(self) -> list[AttackOutcome]:
        return [a for a in self.attacks if not a.caught]

    @property
    def mismatched(self) -> list[AttackOutcome]:
        return [a for a in self.attacks if a.mismatched]

    @property
    def generation_failures(self) -> list[AttackOutcome]:
        return [a for a in self.attacks if a.generation_error]

    @property
    def false_positives(self) -> list[ControlOutcome]:
        return [c for c in self.controls if c.false_positive]

    @property
    def ok(self) -> bool:
        """A template ships only at a 100% catch rate with no honest-work failures."""
        if not self.attacks:
            return False
        return not self.uncaught and not self.generation_failures and not self.false_positives

    # -- messages ---------------------------------------------------------- #

    def grader_defect_message(self) -> str:
        """The loud message. It names the grader, because the grader is the defect."""
        if not self.uncaught:
            return ""
        lines = [
            f"GRADER DEFECT: {len(self.uncaught)} red-team attack(s) were caught by NO oracle."
        ]
        for outcome in self.uncaught:
            verdicts = ", ".join(f"{k}={v}" for k, v in sorted(outcome.oracle_verdicts.items()))
            lines.append(
                f"  - {outcome.attack_id} on seed {outcome.seed_id}: "
                f"expected {outcome.expected_catcher}, got {verdicts or 'no oracle verdicts'}"
                + (f" [generation error: {outcome.generation_error}]" if outcome.generation_error else "")
            )
        lines.append(
            "This is a defect in the GRADER, not in the attacks. Each attack above is a "
            "faithful description of a hack a model will produce, and the harness has no "
            "check that can see it, so any candidate shipping that hack would be recorded "
            "as a pass. Fix or add the oracle check. Do not weaken, retune or delete the "
            "attack to make this message go away."
        )
        return "\n".join(lines)

    def false_positive_message(self) -> str:
        if not self.false_positives:
            return ""
        lines = [
            f"GRADER DEFECT: the honest baseline was flagged on "
            f"{len(self.false_positives)} seed(s)."
        ]
        for control in self.false_positives:
            lines.append(
                f"  - {control.seed_id}: flagged by {control.flagged_by}; "
                + "; ".join(f"{k}: {v}" for k, v in sorted(control.reasons.items()))
            )
        lines.append(
            "A grader that refuses the seed's own known-good baseline is not strict, it is "
            "wrong, and its catch rate means nothing until this is fixed."
        )
        return "\n".join(lines)

    def summary(self) -> str:
        rate = self.catch_rate
        shown = "unmeasured" if rate is None else f"{100.0 * rate:.1f}%"
        return (
            f"{self.n_caught}/{self.n_attacks} attacks caught ({shown}); "
            f"{len(self.mismatched)} caught by an unexpected check; "
            f"{len(self.skipped)} not applicable; "
            f"{len(self.false_positives)} honest-baseline false positive(s)"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary(),
            "n_attacks": self.n_attacks,
            "n_caught": self.n_caught,
            "catch_rate": self.catch_rate,
            "uncaught": [a.attack_id for a in self.uncaught],
            "mismatched": [
                {"id": a.attack_id, "expected": a.expected_catcher, "actual": a.catchers}
                for a in self.mismatched
            ],
            "attacks": [a.as_dict() for a in self.attacks],
            "controls": [c.as_dict() for c in self.controls],
            "skipped": [s.as_dict() for s in self.skipped],
            "seeds": list(self.seeds),
            "oracles_run": list(self.oracles_run),
            "shape_budget": self.shape_budget,
            "device": self.device,
            "started_utc": self.started_utc,
            "duration_s": round(self.duration_s, 4),
            "grader_defect": self.grader_defect_message(),
            "false_positive_report": self.false_positive_message(),
        }


# --------------------------------------------------------------------------- #
# targets: a task to attack, and the seed it came from
# --------------------------------------------------------------------------- #


def synthetic_task(
    seed: Any,
    *,
    oracles: Sequence[str] = ("O1", "O3"),
    n_detect: int = 2,
) -> Task:
    """A grading target built straight from a seed, for red-teaming only.

    This is not a bank task and carries no mutation: its purpose is to give the
    oracles a shown shape and a set of withheld shapes so that the checks which
    depend on that distinction (``held_out_shapes`` above all) are actually
    exercised. The last shapes of the sweep are withheld because a seed's sweep
    is authored small-to-large and the large end is where a hardcoded block size
    stops being right.
    """
    sweep = list(getattr(seed, "shape_sweep", []) or [])
    if len(sweep) < 2:
        raise AttackError(
            f"seed {getattr(seed, 'id', '?')!r} exposes {len(sweep)} shape(s); a red-team "
            "target needs at least one shown and one withheld shape"
        )
    n_hold = max(1, min(int(n_detect), len(sweep) - 1))
    decoy: list[ShapeSpec] = [sweep[0]]
    detect: list[ShapeSpec] = sweep[-n_hold:]
    seed_id = str(getattr(seed, "id", "?"))
    source = str(getattr(seed, "source", "") or "")
    return Task(
        task_id=f"redteam-{_safe(seed_id)}",
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed_source=SeedRef(
            seed_id=seed_id,
            module=str(getattr(seed, "module", "") or ""),
            entry=str(getattr(seed, "entry", "") or ""),
            content_sha256=sha256_text(source),
        ),
        domain=getattr(seed, "domain", "pytorch"),
        failure_tier=(list(getattr(seed, "tiers", ())) or ["T2"])[0],
        mutation=MutationSpec(
            cls="redteam.none",
            site=f"{seed_id}:{getattr(seed, 'entry', '?')}",
            description="red-team target: the baseline is unmutated; the candidate is the attack",
        ),
        baseline_code=source,
        mutant_code=source,
        ground_truth_diff="",
        detect_shapes=detect,
        decoy_shapes=decoy,
        prompt="Red-team target. The withheld grading shapes are deliberately not named here.",
        oracles=list(oracles),
    )


def _resolve_seeds(seed_id: str | None, seeds: Iterable[Any] | None) -> list[Any]:
    if seeds is not None:
        return list(seeds)
    from ..seeds import registry

    if seed_id:
        return [registry.get(seed_id)]
    return registry.all_seeds()


def _seed_for_task(task: Task, seeds: Mapping[str, Any] | None) -> Any:
    if seeds and task.seed_source.seed_id in seeds:
        return seeds[task.seed_source.seed_id]
    from ..seeds import registry

    return registry.get(task.seed_source.seed_id)


def _budget_extras(
    task: Task,
    shape_budget: int | None,
    extras: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Grade on the shown shape plus the withheld ones, not the whole sweep.

    The truncation is recorded in the report; it is a cost decision, not a
    claim that the untouched shapes would have passed.
    """
    out: dict[str, Any] = dict(extras or {})
    if shape_budget is None:
        return out
    budget = max(1, int(shape_budget))
    out.setdefault("o3_max_shapes", budget)
    if "shapes" not in out:
        ordered: list[ShapeSpec] = []
        seen: set[tuple[str, str]] = set()
        for shape in list(task.decoy_shapes) + list(task.detect_shapes):
            key = (shape.name, shape.key())
            if key in seen:
                continue
            seen.add(key)
            ordered.append(shape)
        if ordered:
            out["shapes"] = ordered[: budget + 1]
    return out


# --------------------------------------------------------------------------- #
# running one candidate
# --------------------------------------------------------------------------- #


def _grade(
    candidate_src: str,
    task: Task,
    seed: Any,
    caps: Capabilities,
    cfg: Config,
    *,
    workdir: Path,
    device: str,
    oracle_ids: Sequence[str] | None,
    extras: Mapping[str, Any] | None,
) -> tuple[list[OracleResult], float]:
    ctx = OracleContext(
        task=task,
        candidate_src=candidate_src,
        seed=seed,
        caps=caps,
        workdir=workdir,
        cfg=cfg,
        rng_seed=int(cfg.rng_seed),
        device=device,
        extras=dict(extras or {}),
    )
    started = time.perf_counter()
    results = run_all(ctx, ids=list(oracle_ids) if oracle_ids is not None else None)
    return results, time.perf_counter() - started


def run_attack(
    attack: Attack,
    task: Task,
    seed: Any,
    caps: Capabilities,
    cfg: Config | None = None,
    *,
    workdir: Path | str | None = None,
    device: str = "cpu",
    oracle_ids: Sequence[str] | None = None,
    extras: Mapping[str, Any] | None = None,
    shape_budget: int | None = DEFAULT_SHAPE_BUDGET,
    keep_source: bool = True,
) -> AttackOutcome:
    """Generate one attack and put it through the real oracle harness."""
    cfg = cfg or DEFAULT_CONFIG
    seed_id = str(getattr(seed, "id", "?"))
    expected = attack.expected_catcher
    root = Path(workdir) if workdir is not None else cfg.work_dir() / "redteam"
    run_dir = root / _safe(attack.id) / _safe(seed_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    outcome = AttackOutcome(
        attack_id=attack.id,
        seed_id=seed_id,
        task_id=task.task_id,
        expected_catcher=str(expected),
        caught=False,
    )

    try:
        source = attack.source(task, seed)
    except (AttackError, KeyError, TypeError, ValueError, AttributeError) as exc:
        outcome.generation_error = f"{type(exc).__name__}: {exc}"
        outcome.verdict = "ERROR"
        return outcome

    outcome.source_sha256 = sha256_text(source)
    if keep_source:
        path = run_dir / "attack_candidate.py"
        path.write_text(source, encoding="utf-8")
        outcome.source_path = str(path)

    results, duration = _grade(
        source,
        task,
        seed,
        caps,
        cfg,
        workdir=run_dir,
        device=device,
        oracle_ids=oracle_ids,
        extras=_budget_extras(task, shape_budget, extras),
    )
    outcome.duration_s = duration
    outcome.oracle_verdicts = {r.oracle: r.verdict for r in results}
    outcome.reasons = {r.oracle: r.reason for r in results if r.reason}
    outcome.errored = [r.oracle for r in results if r.verdict == "ERROR"]
    outcome.catchers = catchers_of(results)
    outcome.caught = bool(outcome.catchers)
    outcome.expected_matched = str(expected) in outcome.catchers
    outcome.verdict = "FAIL" if outcome.caught else ("ERROR" if outcome.errored else "PASS")
    if outcome.caught and not outcome.expected_matched:
        logger.warning(
            "attack %s on seed %s was caught by %s, not by the expected %s",
            attack.id,
            seed_id,
            outcome.catchers,
            expected,
        )
    return outcome


def run_control(
    task: Task,
    seed: Any,
    caps: Capabilities,
    cfg: Config | None = None,
    *,
    workdir: Path | str | None = None,
    device: str = "cpu",
    oracle_ids: Sequence[str] | None = None,
    extras: Mapping[str, Any] | None = None,
    shape_budget: int | None = DEFAULT_SHAPE_BUDGET,
    candidate_src: str | None = None,
) -> ControlOutcome:
    """Grade honest work. Anything but PASS/SKIP here is a false positive."""
    cfg = cfg or DEFAULT_CONFIG
    seed_id = str(getattr(seed, "id", "?"))
    root = Path(workdir) if workdir is not None else cfg.work_dir() / "redteam"
    run_dir = root / "_control" / _safe(seed_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    source = candidate_src if candidate_src is not None else task.baseline_code

    results, duration = _grade(
        source,
        task,
        seed,
        caps,
        cfg,
        workdir=run_dir,
        device=device,
        oracle_ids=oracle_ids,
        extras=_budget_extras(task, shape_budget, extras),
    )
    flagged = catchers_of(results)
    verdicts = {r.oracle: r.verdict for r in results}
    return ControlOutcome(
        seed_id=seed_id,
        task_id=task.task_id,
        verdict="FAIL" if flagged else ("ERROR" if "ERROR" in verdicts.values() else "PASS"),
        flagged_by=flagged,
        oracle_verdicts=verdicts,
        reasons={r.oracle: r.reason for r in results if r.reason},
        duration_s=duration,
    )


# --------------------------------------------------------------------------- #
# the suite
# --------------------------------------------------------------------------- #


def run_suite(
    seed_id: str | None = None,
    caps: Capabilities | None = None,
    cfg: Config | None = None,
    *,
    seeds: Iterable[Any] | None = None,
    tasks: Iterable[Task] | None = None,
    attacks: Iterable[Attack] | None = None,
    bank_dir: Path | str | None = None,
    oracle_ids: Sequence[str] | None = None,
    extras: Mapping[str, Any] | None = None,
    workdir: Path | str | None = None,
    device: str = "cpu",
    shape_budget: int | None = DEFAULT_SHAPE_BUDGET,
    include_control: bool = True,
    strict: bool = False,
) -> SuiteReport:
    """Attack the grader with every applicable attack, on every requested seed.

    ``seed_id``/``caps``/``cfg`` are positional-compatible with the CLI. Pass
    ``seeds`` (SeedSpec objects) or ``tasks`` to target something other than the
    registered seed bank - tests and one-off investigations use that path.

    ``strict=True`` raises :class:`GraderDefect` instead of returning a report
    whose ``ok`` is False.
    """
    cfg = cfg or DEFAULT_CONFIG
    if caps is None:
        from ..capabilities import detect

        caps = detect(probe_timeout_s=cfg.probe_timeout_s)
    if bank_dir is None:
        bank_dir = cfg.bank_dir
    pool = list(attacks) if attacks is not None else attacks_mod.all_attacks(bank_dir)
    root = Path(workdir) if workdir is not None else cfg.work_dir() / "redteam"

    seed_by_id: dict[str, Any] = {}
    if seeds is not None:
        for spec in seeds:
            seed_by_id[str(getattr(spec, "id", "?"))] = spec

    targets: list[tuple[Task, Any]] = []
    if tasks is not None:
        for task in tasks:
            targets.append((task, _seed_for_task(task, seed_by_id)))
    else:
        for spec in _resolve_seeds(seed_id, seeds):
            targets.append((synthetic_task(spec), spec))

    report = SuiteReport(
        oracles_run=list(oracle_ids) if oracle_ids is not None else [],
        shape_budget=shape_budget,
        device=device,
        started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    started = time.perf_counter()

    for task, seed in targets:
        spec_id = str(getattr(seed, "id", "?"))
        report.seeds.append(spec_id)
        for attack in pool:
            reason = attack.why_not(seed)
            if reason:
                report.skipped.append(SkippedAttack(attack.id, spec_id, reason))
                continue
            report.attacks.append(
                run_attack(
                    attack,
                    task,
                    seed,
                    caps,
                    cfg,
                    workdir=root,
                    device=device,
                    oracle_ids=oracle_ids,
                    extras=extras,
                    shape_budget=shape_budget,
                )
            )
        if include_control:
            report.controls.append(
                run_control(
                    task,
                    seed,
                    caps,
                    cfg,
                    workdir=root,
                    device=device,
                    oracle_ids=oracle_ids,
                    extras=extras,
                    shape_budget=shape_budget,
                )
            )

    report.duration_s = time.perf_counter() - started
    if not report.oracles_run:
        seen: list[str] = []
        for outcome in report.attacks:
            for oid in outcome.oracle_verdicts:
                if oid not in seen:
                    seen.append(oid)
        report.oracles_run = seen

    for outcome in report.mismatched:
        logger.warning(
            "attack %s: expected catcher %s, actual %s",
            outcome.attack_id,
            outcome.expected_catcher,
            outcome.catchers,
        )
    if strict:
        assert_all_caught(report)
    return report


def assert_all_caught(report: SuiteReport) -> SuiteReport:
    """Raise :class:`GraderDefect` unless the harness caught everything.

    A mismatched catcher is *not* raised here: the hack was refused, which is
    what the ship gate is about. It is logged and carried in the report so a
    human can decide whether two checks have quietly merged.
    """
    if report.uncaught or report.generation_failures:
        raise GraderDefect(report.grader_defect_message() or "red-team suite failed")
    if report.false_positives:
        raise GraderDefect(report.false_positive_message())
    if not report.attacks:
        raise GraderDefect(
            "GRADER DEFECT: the red-team suite ran zero attacks, so the catch rate is "
            "unmeasured. An unmeasured catch rate is not a 100% catch rate and no "
            "template may ship on it."
        )
    return report


def catch_rate(report: SuiteReport) -> float | None:
    return report.catch_rate


# Alias for the CLI's entry-point probe order (run_suite, run, ...).
run = run_suite


__all__ = [
    "AttackOutcome",
    "ControlOutcome",
    "DEFAULT_SHAPE_BUDGET",
    "GraderDefect",
    "SkippedAttack",
    "SuiteReport",
    "assert_all_caught",
    "catch_rate",
    "catchers_of",
    "fired_checks",
    "run",
    "run_attack",
    "run_control",
    "run_suite",
    "synthetic_task",
]
