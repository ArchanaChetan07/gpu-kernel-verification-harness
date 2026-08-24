"""Turn seeds into tasks: mutate, prove, admit or discard.

The pipeline is deliberately linear and deliberately unforgiving::

    seed source -> canonical form -> sites -> apply -> mutant source
                -> witness search -> admit (build Task) | discard (with a reason)

Three things are worth stating explicitly because they are where the rigour
lives:

* **The baseline stored on the task is the canonical form, not the seed file.**
  Both sides of the diff come out of ``ast.unparse``, so the ground-truth diff
  contains the mutation and nothing else. The original file's hash is kept in
  ``provenance`` so the derivation stays auditable.
* **The diff is the fix, not the bug.** ``ground_truth_diff`` transforms the
  mutant into the baseline. The engine verifies that round trip on itself before
  it will emit a task; a diff that does not reproduce the baseline exactly is a
  generator defect, not a task.
* **The prompt is checked for leaks, not assumed clean.** The mutation class id,
  the diff, the withheld detect shapes and the baseline's own version of the
  mutated line are each asserted absent. A leaked line turns a discovery task
  into a copy task and the whole calibration story collapses.
"""

from __future__ import annotations

import logging
import platform
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import DEFAULT_CONFIG, Config
from ..errors import CrucibleError
from ..schema import (
    Domain,
    MutationSpec,
    RubricCriterion,
    RubricSpec,
    SeedRef,
    ShapeSpec,
    Task,
    Tier,
    Witness,
    sha256_text,
)
from . import witness as witness_mod
from .astutil import (
    SiteError,
    Site,
    apply_unified_diff,
    canonical,
    ensure_trailing_newline,
    parse,
    unified_diff,
    unparse,
)
from .classes import BaseMutation, resolve_classes

logger = logging.getLogger(__name__)

GENERATOR = "crucible.mutate.engine"
GENERATOR_VERSION = "1.0"


class PromptLeak(CrucibleError):
    """The generated prompt contains something that is supposed to be withheld."""


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


@dataclass
class GenerationOutcome:
    seed_id: str
    cls: str
    site: str
    admitted: bool
    task_id: str | None = None
    path: str | None = None
    discarded_reason: str | None = None
    witness_kind: str | None = None
    n_evaluated: int = 0
    duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "class": self.cls,
            "site": self.site,
            "admitted": self.admitted,
            "task_id": self.task_id,
            "path": self.path,
            "discarded_reason": self.discarded_reason,
            "witness_kind": self.witness_kind,
            "n_evaluated": self.n_evaluated,
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class GenerationReport:
    outcomes: list[GenerationOutcome] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    started_utc: str = ""
    finished_utc: str = ""
    device: str = "cpu"

    def admitted(self) -> list[GenerationOutcome]:
        return [o for o in self.outcomes if o.admitted]

    def discarded(self) -> list[GenerationOutcome]:
        return [o for o in self.outcomes if not o.admitted]

    def discard_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.discarded():
            head = (o.discarded_reason or "unknown").split(":", 1)[0]
            counts[head] = counts.get(head, 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_admitted": len(self.admitted()),
            "n_discarded": len(self.discarded()),
            "device": self.device,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "discard_reasons": self.discard_reasons(),
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


# --------------------------------------------------------------------------- #
# task assembly
# --------------------------------------------------------------------------- #


def oracles_for(tier: str, domain: str) -> list[str]:
    """Which oracles apply to a task in this cell.

    Membership here is about *applicability*, not availability: an oracle that
    applies but cannot run on this machine must still be listed so that it SKIPs
    loudly rather than disappearing from the verdict.
    """
    ids = {"O1", "O3"}
    if domain in ("triton", "pallas", "cuda", "jax", "pytorch"):
        ids.add("O4")
    if tier in ("T1", "T4"):
        ids.add("O4")
    if tier == "T4":
        ids.add("O2")
    if tier == "T6" or domain in ("distributed", "checkpointing"):
        ids.add("O5")
    return sorted(ids)


def refine_tier(class_tier: Tier, witness_kind: str) -> Tier:
    """The tier records the failure that was *observed*, not the one intended.

    A mutation designed to be a silent numerical divergence that in practice
    raises on the witness shape is a T2 crash task, and filing it as T5 would
    overstate the bank's silent-failure share -- the headline number this whole
    project is judged on.
    """
    if witness_kind in ("exception", "hang"):
        return "T2"
    if witness_kind == "shape":
        return "T3"
    if witness_kind == "loss_curve":
        return "T6"
    return class_tier


def _anchor(low: str, mid: str, high: str) -> dict[int, str]:
    return {0: low, 3: mid, 5: high}


def default_rubric(w: Witness | None, tier: str, domain: str) -> RubricSpec:
    """A rubric whose objective criteria are machine-probed, not re-judged.

    Anything an oracle already measures is scored from that measurement. Human
    raters are asked only about things a machine cannot read: whether the
    explanation identifies the mechanism, and whether the fix is minimal.
    """
    rtol = max(float(w.tolerance) if w else 1e-6, 1e-12)
    abs_ref = float(w.max_abs_err) if (w and w.max_abs_err > 0) else 0.0
    abs_pass = abs_ref * 1e-3 if abs_ref > 0 else 1e-6
    abs_partial = abs_ref * 1e-1 if abs_ref > 0 else 1e-3

    criteria = [
        RubricCriterion(
            id="numerics_within_budget",
            weight=4.0,
            anchors=_anchor(
                "the fixed implementation still exceeds the derived error budget",
                "within the budget on every graded shape",
                "within the budget with margin, on the witness shape included",
            ),
            auto_probe="oracle.O1.max_rel_err",
            probe_thresholds=[(rtol, 5), (rtol * 100.0, 3), (1e12, 0)],
            machine_probed=True,
        ),
        RubricCriterion(
            id="witness_shape_fixed",
            weight=3.0,
            anchors=_anchor(
                "the withheld shape still reproduces the defect",
                "the withheld shape is fixed",
                "the withheld shape is fixed and the absolute error is at reference level",
            ),
            auto_probe="oracle.O1.max_abs_err",
            probe_thresholds=[(abs_pass, 5), (abs_partial, 3), (1e12, 0)],
            machine_probed=True,
        ),
        RubricCriterion(
            id="root_cause_explanation",
            weight=2.0,
            anchors=_anchor(
                "no mechanism given, or the wrong expression blamed",
                "names the responsible expression",
                "names the expression and derives why the given shapes hide it",
            ),
        ),
        RubricCriterion(
            id="fix_minimality",
            weight=1.0,
            anchors=_anchor(
                "rewrites unrelated code or changes the interface",
                "a local fix with some incidental change",
                "the minimal edit that removes the defect",
            ),
        ),
    ]

    if tier == "T4":
        criteria.append(
            RubricCriterion(
                id="performance_recovered",
                weight=2.0,
                anchors=_anchor(
                    "slower than the baseline or unmeasured",
                    "at parity with the baseline",
                    "at or above baseline throughput with a confidence interval clear of 1.0",
                ),
                auto_probe="oracle.O2.speedup",
                probe_thresholds=[(0.9, 0), (1.0, 3), (1e12, 5)],
                machine_probed=True,
            )
        )
    if tier == "T6" or domain in ("distributed", "checkpointing"):
        criteria.append(
            RubricCriterion(
                id="multi_rank_agreement",
                weight=3.0,
                anchors=_anchor(
                    "the N-rank loss curve still diverges from the 1-rank curve",
                    "the curves agree within the measured noise floor",
                    "the curves agree well inside the measured noise floor",
                ),
                auto_probe="oracle.O5.max_loss_delta",
                probe_thresholds=[(1e-06, 5), (1e-03, 3), (1e12, 0)],
                machine_probed=True,
            )
        )
    return RubricSpec(criteria=criteria)


def _shape_line(shape: ShapeSpec) -> str:
    kwargs = ", ".join(f"{k}={v!r}" for k, v in sorted(shape.kwargs.items()))
    return f"- `{shape.name}`: {kwargs}" if kwargs else f"- `{shape.name}`"


def build_prompt(
    seed: Any,
    mutant_src: str,
    decoys: Sequence[ShapeSpec],
    domain: str,
) -> str:
    """The task text handed to a candidate. Contains the mutant and the decoys only."""
    entry = seed.entry
    denylist = ", ".join(f"`{d}`" for d in getattr(seed, "denylist", ()) or ()) or "(none)"
    if decoys:
        shape_block = "\n".join(_shape_line(s) for s in decoys)
        shape_note = (
            "Each of these was executed and agreed with an independent reference to "
            "within the derived error budget, so none of them exposes the defect:"
        )
    else:
        shape_block = "- (no shape in the recorded sweep agreed with the reference)"
        shape_note = "The recorded sweep produced no agreeing shape:"

    return f"""# Debugging task: `{entry}`

The `{entry}` implementation below is a {domain} routine that is believed to be
correct. It is not. On at least one input it silently disagrees with an
independent reference implementation, and the disagreement is not visible on the
shapes it was checked on.

## Implementation under test

```python
{mutant_src.rstrip()}
```

## Shapes it was checked on

{shape_note}

{shape_block}

## What to produce

1. **Root cause.** Which expression is wrong, and what it does wrong.
2. **Trigger class.** What property an input must have for the defect to show,
   and why every shape above lacks that property.
3. **Fix.** A corrected `{entry}` with the same name and the same signature.

## Rules

- Pure PyTorch. Do not import or call: {denylist}.
- Do not special-case shapes, memoise inputs, or return a precomputed answer.
- Your submission is executed on withheld shapes chosen specifically to expose
  this defect, with freshly seeded inputs. Reproducing the behaviour on the
  shapes above is necessary and not sufficient.
"""


def assert_no_leak(
    prompt: str,
    cls_id: str,
    detect_shapes: Sequence[ShapeSpec],
    diff: str,
    baseline_src: str,
    mutant_src: str,
) -> None:
    """Refuse to emit a prompt that hands over any part of the answer."""
    if cls_id and cls_id in prompt:
        raise PromptLeak("prompt names the mutation class", cls=cls_id)
    for shape in detect_shapes:
        if shape.name and shape.name in prompt:
            raise PromptLeak("prompt leaks a withheld detect shape", shape=shape.name)
    for hunk_line in diff.splitlines():
        if hunk_line.startswith("@@") and hunk_line in prompt:
            raise PromptLeak("prompt contains the ground-truth diff", hunk=hunk_line)
    # The baseline's version of a mutated line is the answer. It is a leak only
    # if that exact line is absent from the mutant, since the prompt legitimately
    # embeds every line the mutant still has.
    mutant_lines = {ln.strip() for ln in mutant_src.splitlines()}
    prompt_lines = {ln.strip() for ln in prompt.splitlines()}
    for line in diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        stripped = line[1:].strip()
        if len(stripped) < 8 or stripped in mutant_lines:
            continue
        if stripped in prompt_lines:
            raise PromptLeak("prompt contains the baseline's version of a mutated line", line=stripped)


def _provenance(
    seed: Any,
    cls: BaseMutation,
    site: Site,
    result: witness_mod.WitnessSearchResult,
    baseline_src: str,
    mutant_src: str,
    caps: Any,
    device: str,
) -> dict[str, Any]:
    try:
        host = socket.gethostname()
    except OSError:
        host = "unknown"
    tol = result.tolerance
    return {
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "host": host,
        "python": platform.python_version(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "capabilities": caps.summary() if hasattr(caps, "summary") else str(caps),
        "seed_module": getattr(seed, "module", "") or "",
        "seed_tiers": list(getattr(seed, "tiers", ()) or ()),
        "class_tier": cls.tier,
        "site": site.as_dict(),
        "seed_source_sha256": sha256_text(getattr(seed, "source", "")),
        "canonical_baseline_sha256": sha256_text(baseline_src),
        "mutant_sha256": sha256_text(mutant_src),
        "shapes_evaluated": result.n_evaluated,
        "shape_outcomes": [o.as_dict() for o in result.outcomes],
        "tolerance": tol.as_dict() if tol else None,
    }


def build_task(
    seed: Any,
    cls: BaseMutation,
    site: Site,
    baseline_src: str,
    mutant_src: str,
    result: witness_mod.WitnessSearchResult,
    caps: Any = None,
    device: str = "cpu",
) -> Task:
    """Assemble a Task from a mutation whose witness has already been proven."""
    if result.witness is None:
        raise CrucibleError(
            "refusing to build a task from a mutation with no witness",
            seed=getattr(seed, "id", "?"),
            cls=cls.id,
        )
    diff = unified_diff(
        mutant_src,
        baseline_src,
        fromfile=f"a/{seed.entry}.py (mutant)",
        tofile=f"b/{seed.entry}.py (baseline)",
    )
    # The diff is the answer key; if it does not reproduce the baseline exactly,
    # the task would be graded against a fix that does not exist.
    reconstructed = apply_unified_diff(mutant_src, diff)
    if reconstructed != baseline_src:
        raise CrucibleError(
            "ground-truth diff does not reproduce the baseline; refusing to emit the task",
            seed=getattr(seed, "id", "?"),
            cls=cls.id,
        )

    tier = refine_tier(cls.tier, result.witness.kind)
    domain: Domain = seed.domain
    prompt = build_prompt(seed, mutant_src, result.decoys, domain)
    assert_no_leak(prompt, cls.id, result.detects, diff, baseline_src, mutant_src)

    task_id = f"mut-{seed.short_id()}-{cls.id}-{sha256_text(mutant_src)[:4]}"
    return Task(
        task_id=task_id,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed_source=SeedRef(
            seed_id=seed.id,
            module=getattr(seed, "module", "") or seed.id,
            entry=seed.entry,
            content_sha256=sha256_text(getattr(seed, "source", "")),
        ),
        domain=domain,
        failure_tier=tier,
        mutation=MutationSpec(
            cls=cls.id,
            site=f"{seed.id}:{site.descriptor()}",
            params=cls.spec_params(site),
            description=cls.description,
        ),
        baseline_code=baseline_src,
        mutant_code=mutant_src,
        ground_truth_diff=diff,
        witness=result.witness,
        detect_shapes=list(result.detects),
        decoy_shapes=list(result.decoys),
        prompt=prompt,
        oracles=oracles_for(tier, domain),
        rubric=default_rubric(result.witness, tier, domain),
        provenance=_provenance(
            seed, cls, site, result, baseline_src, mutant_src, caps, device
        ),
    )


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


def _resolve_seeds(seed_ids: Iterable[Any] | None) -> list[Any]:
    """Accept seed ids or already-constructed SeedSpec objects."""
    if seed_ids is None:
        from ..seeds.registry import all_seeds

        return all_seeds()
    out: list[Any] = []
    for item in seed_ids:
        if isinstance(item, str):
            from ..seeds.registry import get

            out.append(get(item))
        else:
            out.append(item)
    return out


def generate(
    seed_ids: Iterable[Any] | None = None,
    classes: Iterable[Any] | None = None,
    caps: Any = None,
    cfg: Config | None = None,
    out_dir: Path | str | None = None,
    limit: int | None = None,
    *,
    device: str | None = None,
    sites_per_class: int = 1,
    workdir: Path | str | None = None,
    stop_on_first: bool | None = None,
) -> GenerationReport:
    """Mutate every (seed, class, site) and admit only what a witness proves.

    ``caps`` is required in spirit but optional in signature: when it is None the
    machine is probed, because a generator that guessed at capabilities could
    emit a task whose witness was never actually executed.
    """
    cfg = cfg or DEFAULT_CONFIG
    if caps is None:
        from ..capabilities import detect

        caps = detect()
    seeds = _resolve_seeds(seed_ids)
    mutations = resolve_classes(list(classes) if classes is not None else None)
    chosen_device = device or ("cuda" if bool(getattr(caps, "cuda", False)) else "cpu")
    root = Path(workdir) if workdir is not None else Path.cwd() / cfg.work_root / "mutate"
    out_path = Path(out_dir) if out_dir is not None else None

    report = GenerationReport(
        started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        device=chosen_device,
    )

    for seed in seeds:
        try:
            baseline_src = canonical(seed.source)
            tree = parse(baseline_src, filename=f"<seed {seed.id}>")
        except SyntaxError as exc:
            report.outcomes.append(
                GenerationOutcome(
                    seed_id=getattr(seed, "id", "?"),
                    cls="-",
                    site="-",
                    admitted=False,
                    discarded_reason=f"seed source does not parse: {exc}",
                )
            )
            continue

        for cls in mutations:
            found = cls.sites(tree, baseline_src)
            if not found:
                logger.debug("class %s finds no site in seed %s", cls.id, seed.id)
                continue
            for site in found[: max(int(sites_per_class), 1)]:
                if limit is not None and len(report.admitted()) >= limit:
                    report.finished_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    return report
                outcome = _one(
                    seed,
                    cls,
                    site,
                    tree,
                    baseline_src,
                    caps,
                    cfg,
                    root,
                    out_path,
                    chosen_device,
                    stop_on_first,
                    report,
                )
                report.outcomes.append(outcome)

    report.finished_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return report


def _one(
    seed: Any,
    cls: BaseMutation,
    site: Site,
    tree: Any,
    baseline_src: str,
    caps: Any,
    cfg: Config,
    root: Path,
    out_path: Path | None,
    device: str,
    stop_on_first: bool | None,
    report: GenerationReport,
) -> GenerationOutcome:
    started = time.perf_counter()
    outcome = GenerationOutcome(
        seed_id=getattr(seed, "id", "?"),
        cls=cls.id,
        site=site.descriptor(),
        admitted=False,
    )

    try:
        mutant_tree = cls.apply(tree, site)
        mutant_src = ensure_trailing_newline(unparse(mutant_tree))
        parse(mutant_src, filename=f"<mutant {seed.id}/{cls.id}>")
    except (SiteError, SyntaxError, ValueError, TypeError, AttributeError) as exc:
        outcome.discarded_reason = f"mutation could not be applied: {type(exc).__name__}: {exc}"
        outcome.duration_s = time.perf_counter() - started
        return outcome

    if mutant_src == baseline_src:
        outcome.discarded_reason = (
            "semantically neutral: the mutation produced a textually identical program"
        )
        outcome.duration_s = time.perf_counter() - started
        return outcome

    result = witness_mod.search(
        seed,
        mutant_src,
        getattr(seed, "shape_sweep", None),
        caps,
        cfg,
        baseline_src=baseline_src,
        workdir=root / f"{seed.short_id()}-{cls.id}-{site.lineno}",
        device=device,
        stop_on_first=stop_on_first,
    )
    outcome.n_evaluated = result.n_evaluated
    if result.witness is None:
        outcome.discarded_reason = result.discarded_reason or "no witness found"
        outcome.duration_s = time.perf_counter() - started
        return outcome

    outcome.witness_kind = result.witness.kind
    try:
        task = build_task(seed, cls, site, baseline_src, mutant_src, result, caps, device)
    except (CrucibleError, ValueError) as exc:
        outcome.discarded_reason = f"task assembly refused it: {type(exc).__name__}: {exc}"
        outcome.duration_s = time.perf_counter() - started
        return outcome

    outcome.admitted = True
    outcome.task_id = task.task_id
    report.tasks.append(task)
    if out_path is not None:
        outcome.path = str(task.save(out_path / f"{task.task_id}.yaml"))
    outcome.duration_s = time.perf_counter() - started
    return outcome


__all__ = [
    "generate",
    "build_task",
    "build_prompt",
    "assert_no_leak",
    "default_rubric",
    "oracles_for",
    "refine_tier",
    "GenerationReport",
    "GenerationOutcome",
    "PromptLeak",
]
