"""The command line: everything a user can ask this machine to do.

Two rules shape this file.

**Lazy imports.** Every subsystem (mutate, oracles, calibrate, rubric, redteam)
is imported *inside* the command that needs it. A half-built or broken module
must not stop ``crucible doctor`` from running, because doctor is precisely the
command you reach for when something is broken.

**Honest exit codes.** 0 ok, 1 FAIL, 2 SKIP-blocked, 3 ERROR. A run whose
oracles could not execute exits 2, not 0: a green pipeline that verified nothing
is the failure mode this project exists to remove.

``print`` is permitted here and in ``crucible/report/`` only.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import typer
from click import exceptions as click_exceptions
from rich import box
from rich.console import Console
from rich.table import Table

from . import __version__, capabilities
from .config import Config
from .errors import CrucibleError
from .schema import EnvironmentRecord, OracleResult, Task, TaskVerdict, sha256_text

logger = logging.getLogger("crucible.cli")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_SKIP_BLOCKED = 2
EXIT_ERROR = 3

_EXIT_FOR_VERDICT: dict[str, int] = {
    "PASS": EXIT_OK,
    "FAIL": EXIT_FAIL,
    "SKIP": EXIT_SKIP_BLOCKED,
    "ERROR": EXIT_ERROR,
}

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="CRUCIBLE - execution-grounded task foundry. A task ships only if its "
    "reference solution was executed and passed every applicable oracle.",
)
seeds_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Inspect the seed registry.")
app.add_typer(seeds_app, name="seeds")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _console(stderr: bool = False) -> Console:
    """Rich console. ASCII-only tables so redirected Windows output cannot fail."""
    stream = sys.stderr if stderr else sys.stdout
    try:
        tty = bool(stream.isatty())
    except (AttributeError, ValueError):
        tty = False
    return Console(stderr=stderr, width=None if tty else 160, highlight=False)


def _table(title: str, *columns: str) -> Table:
    t = Table(title=title, box=box.ASCII, header_style="bold", title_justify="left", pad_edge=False)
    for c in columns:
        t.add_column(c, overflow="fold")
    return t


def _emit_json(payload: Any) -> None:
    """Machine output goes to stdout alone; nothing else may share the stream."""
    print(json.dumps(payload, indent=2, default=str))


def _fail(message: str, code: int = EXIT_ERROR, *, json_out: bool = False, **extra: Any) -> None:
    if json_out:
        _emit_json({"status": "error", "error": message, **extra})
    else:
        _console(stderr=True).print(f"[red]error:[/red] {message}")
    raise typer.Exit(code)


def _cfg(ctx: typer.Context) -> Config:
    path = (ctx.obj or {}).get("config_path") if ctx.obj else None
    try:
        return Config.load(path)
    except CrucibleError as exc:
        _fail(str(exc))
        raise  # unreachable; keeps the type checker honest


def _split_list(value: str | None) -> list[str] | None:
    """``"a,b"`` -> ``["a","b"]``; ``None``/``"all"``/``""`` -> ``None`` (means every)."""
    if value is None:
        return None
    v = value.strip()
    if not v or v.lower() == "all":
        return None
    return [part.strip() for part in v.split(",") if part.strip()]


def _resolve_callable(candidates: Sequence[tuple[str, str]]) -> tuple[Any, str] | None:
    """First importable ``(module, attribute)`` pair, or None.

    Used for subsystems written by other authors: the CLI dispatches to whatever
    entry point exists rather than guessing at an implementation.
    """
    for module_name, attr in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            logger.debug("%s not importable: %s", module_name, exc)
            continue
        fn = getattr(module, attr, None)
        if callable(fn):
            return fn, f"{module_name}.{attr}"
    return None


def _describe(value: Any) -> Any:
    """Best-effort JSON view of a report object from another module."""
    for attr in ("as_dict", "model_dump", "to_dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return fn()
            except (TypeError, ValueError) as exc:
                logger.debug("%s.%s() failed: %s", type(value).__name__, attr, exc)
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return repr(value)


# --------------------------------------------------------------------------- #
# root callback
# --------------------------------------------------------------------------- #


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", help="YAML config overriding defaults."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging on stderr."),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ctx.obj = {"config_path": config}


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #

#: Capability fields shown in the doctor table, in reading order.
_CAP_ROWS: tuple[str, ...] = (
    "platform",
    "torch_version",
    "cuda",
    "cuda_device_count",
    "device_name",
    "cuda_version",
    "triton",
    "ncu",
    "can_lock_clocks",
    "gloo",
    "nccl",
    "cxx_compiler",
    "inductor_cpu",
    "inductor_cuda",
)


def _oracle_readiness(caps: Any) -> list[dict[str, Any]]:
    """For each of the five oracles: can it run here, and if not what is missing."""
    try:
        from .oracles import ORACLE_MODULES, load_errors, load_oracles
        from .oracles.base import gate_capabilities
    except ImportError as exc:
        return [
            {
                "oracle": "O1..O5",
                "name": "oracle package",
                "ready": False,
                "required_caps": [],
                "reason": f"oracle package could not be imported: {exc}",
            }
        ]

    registry = load_oracles()
    errors = load_errors()
    rows: list[dict[str, Any]] = []
    for oracle_id in sorted(set(ORACLE_MODULES.values())):
        oracle = registry.get(oracle_id)
        if oracle is None:
            rows.append(
                {
                    "oracle": oracle_id,
                    "name": "(not implemented)",
                    "ready": False,
                    "required_caps": [],
                    "reason": errors.get(oracle_id, "module not present in crucible/oracles/"),
                }
            )
            continue
        reason = gate_capabilities(oracle, caps)
        rows.append(
            {
                "oracle": oracle_id,
                "name": str(getattr(oracle, "name", oracle_id)),
                "ready": reason is None,
                "required_caps": list(getattr(oracle, "required_caps", ()) or ()),
                "reason": reason or "",
            }
        )
    return rows


_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("seeds", "crucible.seeds.registry"),
    ("mutate", "crucible.mutate.engine"),
    ("oracles", "crucible.oracles.base"),
    ("calibrate", "crucible.calibrate.passk"),
    ("rubric", "crucible.rubric.irr"),
    ("redteam", "crucible.redteam.suite"),
    ("report", "crucible.report.metrics"),
)


def _component_status() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for label, module_name in _COMPONENTS:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            out[label] = {"module": module_name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        out[label] = {"module": module_name, "ok": True, "error": ""}
    return out


def _seed_status() -> dict[str, Any]:
    try:
        from .seeds import registry
    except ImportError as exc:
        return {"count": 0, "ids": [], "import_errors": {"crucible.seeds.registry": str(exc)}}
    try:
        ids = registry.seed_ids()
        errors = registry.import_errors()
    except (ImportError, ValueError, TypeError) as exc:
        return {"count": 0, "ids": [], "import_errors": {"discover": f"{type(exc).__name__}: {exc}"}}
    return {"count": len(ids), "ids": ids, "import_errors": errors}


@app.command()
def doctor(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Emit the capability report as JSON."),
    refresh: bool = typer.Option(False, "--refresh", help="Re-probe instead of using the cache."),
) -> None:
    """Report what this machine can honestly verify, oracle by oracle."""
    cfg = _cfg(ctx)
    caps = capabilities.detect(refresh=refresh, probe_timeout_s=cfg.probe_timeout_s)
    cap_dict = caps.as_dict()
    details = dict(cap_dict.pop("details", {}) or {})
    oracles = _oracle_readiness(caps)
    components = _component_status()
    seeds = _seed_status()

    if json_out:
        # Key order matters: pipeline.sh flattens this payload and the last write
        # of a repeated key wins, so the real capability values go LAST and can
        # never be shadowed by a same-named key inside `capability_details`.
        payload: dict[str, Any] = {
            "crucible_version": __version__,
            "oracles": {r["oracle"]: r for r in oracles},
            "components": components,
            "seeds": seeds,
            "capability_details": details,
            "capabilities": dict(cap_dict),
            **cap_dict,
        }
        _emit_json(payload)
        raise typer.Exit(EXIT_OK)

    out = _console()
    out.print(f"[bold]CRUCIBLE {__version__}[/bold] - capability report")
    out.print(caps.summary())

    cap_table = _table("capabilities", "capability", "value", "why not")
    for name in _CAP_ROWS:
        value = cap_dict.get(name)
        shown = "-" if value is None else str(value)
        cap_table.add_row(name, shown, details.get(name, ""))
    out.print(cap_table)

    orc_table = _table("oracles on this machine", "oracle", "name", "can run", "requires", "what is missing")
    for row in oracles:
        orc_table.add_row(
            str(row["oracle"]),
            str(row["name"]),
            "yes" if row["ready"] else "NO",
            ", ".join(row["required_caps"]) or "-",
            str(row["reason"]) or "-",
        )
    out.print(orc_table)

    ready = [r["oracle"] for r in oracles if r["ready"]]
    blocked = [r["oracle"] for r in oracles if not r["ready"]]
    out.print(f"can verify here : {', '.join(ready) if ready else 'nothing'}")
    out.print(f"cannot verify   : {', '.join(blocked) if blocked else 'nothing'}")
    if blocked:
        out.print(
            "Tasks graded by a blocked oracle will be SKIP, and SKIP counts "
            "against the headline metric, never toward it."
        )

    comp_table = _table("subsystems", "component", "module", "status")
    for label, info in components.items():
        comp_table.add_row(label, str(info["module"]), "ok" if info["ok"] else str(info["error"]))
    out.print(comp_table)

    out.print(f"seeds registered: {seeds['count']}")
    for module_name, err in (seeds.get("import_errors") or {}).items():
        out.print(f"  seed module failed to import: {module_name}: {err}")
    raise typer.Exit(EXIT_OK)


# --------------------------------------------------------------------------- #
# seeds
# --------------------------------------------------------------------------- #


@seeds_app.command("list")
def seeds_list(
    json_out: bool = typer.Option(False, "--json", help="Emit the seed list as JSON."),
) -> None:
    """List every registered seed."""
    try:
        from .seeds import registry
    except ImportError as exc:
        _fail(f"seed registry unavailable: {exc}", json_out=json_out)
        return
    specs = registry.all_seeds()
    errors = registry.import_errors()
    rows = [
        {
            "id": s.id,
            "domain": s.domain,
            "tiers": list(s.tiers),
            "entry": s.entry,
            "supports_cpu": s.supports_cpu,
            "shapes": len(s.shape_sweep),
            "description": s.description,
        }
        for s in specs
    ]
    if json_out:
        _emit_json({"seeds": rows, "import_errors": errors})
        raise typer.Exit(EXIT_OK)
    out = _console()
    table = _table("seeds", "id", "domain", "tiers", "entry", "cpu", "shapes", "description")
    for r in rows:
        table.add_row(
            r["id"],
            r["domain"],
            ",".join(r["tiers"]),
            r["entry"],
            "yes" if r["supports_cpu"] else "no",
            str(r["shapes"]),
            r["description"],
        )
    out.print(table)
    for module_name, err in errors.items():
        out.print(f"[yellow]seed module failed to import:[/yellow] {module_name}: {err}")
    raise typer.Exit(EXIT_OK if rows else EXIT_FAIL)


@seeds_app.command("show")
def seeds_show(
    seed_id: str = typer.Argument(..., help="Seed id, e.g. attention.blocked_fwd"),
    json_out: bool = typer.Option(False, "--json", help="Emit the seed as JSON."),
    source: bool = typer.Option(False, "--source", help="Also print the baseline source."),
) -> None:
    """Show one seed: its shapes, tiers, denylist and (optionally) its source."""
    try:
        from .seeds import registry
    except ImportError as exc:
        _fail(f"seed registry unavailable: {exc}", json_out=json_out)
        return
    try:
        spec = registry.get(seed_id)
    except KeyError as exc:
        _fail(str(exc), json_out=json_out)
        return
    payload = {
        "id": spec.id,
        "domain": spec.domain,
        "tiers": list(spec.tiers),
        "entry": spec.entry,
        "module": spec.module,
        "description": spec.description,
        "supports_cpu": spec.supports_cpu,
        "denylist": list(spec.denylist),
        "content_sha256": spec.content_sha256(),
        "shape_sweep": [{"name": s.name, "kwargs": s.kwargs} for s in spec.shape_sweep],
    }
    if source:
        payload["source"] = spec.source
    if json_out:
        _emit_json(payload)
        raise typer.Exit(EXIT_OK)
    out = _console()
    out.print(f"[bold]{spec.id}[/bold]  domain={spec.domain}  tiers={','.join(spec.tiers)}")
    out.print(f"entry={spec.entry}  supports_cpu={spec.supports_cpu}  sha256={payload['content_sha256'][:16]}")
    out.print(spec.description)
    table = _table("adversarial shape sweep", "name", "kwargs")
    for s in spec.shape_sweep:
        table.add_row(s.name, json.dumps(s.kwargs, default=str))
    out.print(table)
    if spec.denylist:
        out.print(f"denylist: {', '.join(spec.denylist)}")
    if source:
        out.print(spec.source)
    raise typer.Exit(EXIT_OK)


# --------------------------------------------------------------------------- #
# mutate
# --------------------------------------------------------------------------- #


@app.command()
def mutate(
    ctx: typer.Context,
    seed: str = typer.Option("all", "--seed", help="Comma list of seed ids, or 'all'."),
    classes: str = typer.Option("all", "--classes", help="Comma list of mutation classes, or 'all'."),
    limit: int = typer.Option(200, "--limit", help="Maximum number of tasks to admit."),
    sites_per_class: int = typer.Option(
        1,
        "--sites-per-class",
        help=(
            "Mutation sites to use per (seed, class) pair. The default of 1 caps the "
            "bank at seeds x classes regardless of --limit, which is why raising --limit "
            "alone does not grow coverage."
        ),
    ),
    out: Path = typer.Option(Path("bank"), "--out", "-o", help="Directory to write task YAML into."),
    json_out: bool = typer.Option(False, "--json", help="Emit the generation report as JSON."),
) -> None:
    """Manufacture tasks: seed -> mutation -> witness search -> admit or discard."""
    cfg = _cfg(ctx)
    try:
        from .mutate.engine import generate
    except ImportError as exc:
        _fail(
            f"mutation engine unavailable (crucible.mutate.engine): {exc}",
            json_out=json_out,
            stage="mutate",
        )
        return
    caps = capabilities.detect(probe_timeout_s=cfg.probe_timeout_s)
    kwargs = {
        "seed_ids": _split_list(seed),
        "classes": _split_list(classes),
        "caps": caps,
        "cfg": cfg,
        "out_dir": out,
        "limit": limit,
        "sites_per_class": sites_per_class,
    }
    try:
        result = generate(**kwargs)
    except TypeError as exc:
        # Signature drift is a contract break, not something to paper over.
        _fail(
            f"crucible.mutate.engine.generate does not accept the contract signature "
            f"(seed_ids, classes, caps, cfg, out_dir, limit): {exc}",
            json_out=json_out,
            stage="mutate",
        )
        return
    except KeyError as exc:
        # An unknown seed or mutation class id: a usage error, not a crash.
        _fail(f"generation failed: {exc.args[0] if exc.args else exc}", json_out=json_out, stage="mutate")
        return
    except (ValueError, OSError, CrucibleError) as exc:
        _fail(f"generation failed: {type(exc).__name__}: {exc}", json_out=json_out, stage="mutate")
        return

    payload = _describe(result)
    written = sorted(p for p in Path(out).glob("*.yaml"))
    n_written = len(written)
    if json_out:
        _emit_json({"status": "ok", "report": payload, "n_task_files": n_written, "out_dir": str(out)})
    else:
        console = _console()
        if isinstance(payload, dict):
            table = _table("generation report", "field", "value")
            for k, v in payload.items():
                table.add_row(str(k), _short(v))
            console.print(table)
        console.print(f"{n_written} task files in {out}")
    raise typer.Exit(EXIT_OK if n_written else EXIT_FAIL)


def _short(value: Any, limit: int = 400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #


def _harness_error(task_id: str, reason: str, candidate_src: str, caps: Any) -> TaskVerdict:
    """A verdict for a failure of the harness itself. Never a PASS."""
    return TaskVerdict(
        task_id=task_id,
        verdict="ERROR",
        executed=False,
        oracle_results=[OracleResult(oracle="harness", verdict="ERROR", reason=reason)],
        environment=EnvironmentRecord.capture(caps),
        duration_s=0.0,
        candidate_sha256=sha256_text(candidate_src),
    )


def _verify_task(
    task: Task,
    candidate_src: str,
    oracle_ids: list[str] | None,
    caps: Any,
    cfg: Config,
    workdir: Path,
) -> TaskVerdict:
    """Execute the candidate against the task's oracles and combine the verdicts."""
    started = time.perf_counter()
    try:
        from .oracles.base import OracleContext, run_all
    except ImportError as exc:
        return _harness_error(task.task_id, f"oracle dispatcher unavailable: {exc}", candidate_src, caps)
    try:
        from .seeds import registry
    except ImportError as exc:
        return _harness_error(task.task_id, f"seed registry unavailable: {exc}", candidate_src, caps)
    try:
        seed = registry.get(task.seed_source.seed_id)
    except KeyError as exc:
        return _harness_error(task.task_id, f"seed not resolvable: {exc}", candidate_src, caps)

    task_dir = workdir / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    ctx = OracleContext(
        task=task,
        candidate_src=candidate_src,
        seed=seed,
        caps=caps,
        workdir=task_dir,
        cfg=cfg,
        rng_seed=cfg.rng_seed,
        device="cuda" if getattr(caps, "cuda", False) else "cpu",
    )
    results = run_all(ctx, ids=oracle_ids)
    # "Executed" means the candidate actually ran: only PASS and FAIL are
    # conclusions drawn from running it. SKIP and ERROR are not.
    executed = any(r.verdict in ("PASS", "FAIL") for r in results)
    return TaskVerdict(
        task_id=task.task_id,
        verdict=TaskVerdict.combine(results),
        executed=executed,
        oracle_results=results,
        environment=EnvironmentRecord.capture(caps),
        duration_s=time.perf_counter() - started,
        candidate_sha256=sha256_text(candidate_src),
    )


def _verdict_table(verdicts: Sequence[TaskVerdict]) -> Table:
    table = _table("verdicts", "task", "verdict", "executed", "oracles", "not verified by")
    for v in verdicts:
        not_verified = sorted({r.oracle for r in v.oracle_results if r.verdict != "PASS"})
        table.add_row(
            v.task_id,
            v.verdict,
            "yes" if v.executed else "no",
            ",".join(r.oracle for r in v.oracle_results) or "-",
            ",".join(not_verified) or "-",
        )
    return table


def _print_skip_panel(console: Console, verdicts: Sequence[TaskVerdict]) -> None:
    from .report.metrics import skip_breakdown

    rows = skip_breakdown(verdicts)
    if not rows:
        return
    table = _table(
        "NOT VERIFIED (these count against the headline, not toward it)",
        "oracle",
        "verdict",
        "count",
        "reason",
    )
    for row in rows:
        table.add_row(str(row["oracle"]), str(row["verdict"]), str(row["count"]), str(row["reason"]))
    console.print(table)


@app.command()
def verify(
    ctx: typer.Context,
    task_path: Path = typer.Argument(..., help="Task YAML to verify."),
    candidate: Optional[Path] = typer.Option(
        None, "--candidate", help="Candidate source. Default: the task's reference solution."
    ),
    oracles: Optional[str] = typer.Option(None, "--oracles", help="Comma list, e.g. O1,O3. Default: the task's own."),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Write the TaskVerdict JSON here."),
    json_out: bool = typer.Option(False, "--json", help="Emit the verdict as JSON on stdout."),
) -> None:
    """Execute one task's candidate against its oracles."""
    cfg = _cfg(ctx)
    try:
        task = Task.load(task_path)
    except (OSError, ValueError) as exc:
        _fail(f"{task_path}: {exc}", json_out=json_out)
        return
    if candidate is not None:
        try:
            candidate_src = candidate.read_text(encoding="utf-8")
        except OSError as exc:
            _fail(f"{candidate}: {exc}", json_out=json_out)
            return
    else:
        # The reference solution is the fixed baseline; that is what the one
        # invariant demands be executed before a task may ship.
        candidate_src = task.baseline_code

    caps = capabilities.detect(probe_timeout_s=cfg.probe_timeout_s)
    workdir = cfg.work_dir()
    verdict = _verify_task(task, candidate_src, _split_list(oracles), caps, cfg, workdir)
    if out is not None:
        verdict.save(out)
    if json_out:
        _emit_json(verdict.model_dump(mode="json"))
    else:
        console = _console()
        console.print(_verdict_table([verdict]))
        _print_skip_panel(console, [verdict])
        if out is not None:
            console.print(f"verdict written to {out}")
    raise typer.Exit(_EXIT_FOR_VERDICT.get(verdict.verdict, EXIT_ERROR))


@app.command("verify-bank")
def verify_bank(
    ctx: typer.Context,
    bank: Path = typer.Argument(..., help="Directory of task YAML files."),
    out: Path = typer.Option(..., "-o", "--out", help="Directory to write TaskVerdict JSON into."),
    oracles: Optional[str] = typer.Option(None, "--oracles", help="Comma list overriding each task's own."),
    json_out: bool = typer.Option(False, "--json", help="Emit the summary as JSON."),
) -> None:
    """Execute every task's reference solution. This is the product."""
    cfg = _cfg(ctx)
    from .report.metrics import load_bank

    loaded = load_bank(bank)
    if not loaded.tasks:
        _fail(
            f"no tasks loaded from {bank}",
            code=EXIT_SKIP_BLOCKED,
            json_out=json_out,
            load_errors=[e.as_dict() for e in loaded.errors],
        )
        return

    caps = capabilities.detect(probe_timeout_s=cfg.probe_timeout_s)
    workdir = cfg.work_dir()
    out.mkdir(parents=True, exist_ok=True)
    ids = _split_list(oracles)
    console = _console()
    verdicts: list[TaskVerdict] = []
    for task in loaded.tasks:
        verdict = _verify_task(task, task.baseline_code, ids, caps, cfg, workdir)
        verdict.save(out / f"{task.task_id}.json")
        verdicts.append(verdict)
        if not json_out:
            console.print(f"{task.task_id}: {verdict.verdict}")

    combined = TaskVerdict.combine([v.verdict for v in verdicts])
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    n_pass = counts.get("PASS", 0)

    if json_out:
        from .report.metrics import skip_breakdown

        _emit_json(
            {
                "status": "ok",
                "bank": str(bank),
                "out": str(out),
                "n_tasks": len(verdicts),
                "verdict_counts": counts,
                "combined_verdict": combined,
                "executed_and_passed": n_pass,
                "headline": n_pass / len(verdicts),
                "skip_breakdown": skip_breakdown(verdicts),
                "load_errors": [e.as_dict() for e in loaded.errors],
            }
        )
    else:
        console.print(_verdict_table(verdicts))
        _print_skip_panel(console, verdicts)
        console.print(
            f"executed and passed every oracle: {n_pass}/{len(verdicts)} "
            f"= {100.0 * n_pass / len(verdicts):.1f}%"
        )
        for err in loaded.errors:
            console.print(f"[yellow]not loaded:[/yellow] {err.path}: {err.error}")
    raise typer.Exit(_EXIT_FOR_VERDICT.get(combined, EXIT_ERROR))


# --------------------------------------------------------------------------- #
# calibrate / irr / redteam - dispatched to sibling subsystems
# --------------------------------------------------------------------------- #

_CALIBRATE_ENTRIES: tuple[tuple[str, str], ...] = (
    ("crucible.calibrate", "calibrate_bank"),
    ("crucible.calibrate.router", "calibrate_bank"),
    ("crucible.calibrate.passk", "calibrate_bank"),
    ("crucible.calibrate.models", "calibrate_bank"),
)


@app.command()
def calibrate(
    ctx: typer.Context,
    bank: Path = typer.Argument(..., help="Directory of task YAML files."),
    model: str = typer.Option("stub", "--model", help="Target model. 'stub' is offline and default."),
    k: int = typer.Option(8, "--k", help="k for pass@k."),
    json_out: bool = typer.Option(False, "--json", help="Emit the calibration report as JSON."),
) -> None:
    """Estimate pass@1 and pass@k per task and route each one."""
    cfg = _cfg(ctx)
    entry = _resolve_callable(_CALIBRATE_ENTRIES)
    if entry is None:
        _fail(
            "no calibration entry point found; expected one of "
            + ", ".join(f"{m}.{a}" for m, a in _CALIBRATE_ENTRIES),
            json_out=json_out,
            stage="calibrate",
            looked_for=[f"{m}.{a}" for m, a in _CALIBRATE_ENTRIES],
        )
        return
    fn, where = entry
    try:
        result = fn(bank_dir=bank, model=model, k=k, cfg=cfg)
    except TypeError as exc:
        _fail(
            f"{where} does not accept (bank_dir, model, k, cfg): {exc}",
            json_out=json_out,
            stage="calibrate",
        )
        return
    except (KeyError, ValueError, OSError, CrucibleError) as exc:
        _fail(f"calibration failed: {type(exc).__name__}: {exc}", json_out=json_out, stage="calibrate")
        return

    payload = _describe(result)
    if json_out:
        _emit_json({"status": "ok", "entry_point": where, "model": model, "k": k, "report": payload})
    else:
        console = _console()
        console.print(f"calibration via {where} (model={model}, k={k})")
        console.print(_short(payload, limit=4000))
    raise typer.Exit(EXIT_OK)


_IRR_ENTRIES: tuple[tuple[str, str], ...] = (
    ("crucible.rubric.irr", "krippendorff_alpha"),
    ("crucible.rubric.irr", "alpha"),
    ("crucible.rubric.irr", "compute_alpha"),
)


def _read_ratings_csv(path: Path) -> tuple[list[list[float]], list[str]]:
    """Ratings matrix (rows = raters, columns = items). Blank cells become NaN.

    A leading non-numeric header row and a leading non-numeric id column are
    tolerated and dropped, because that is how a human writes the file.
    """
    import csv
    import math

    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.reader(fh) if any(cell.strip() for cell in r)]
    if not rows:
        raise ValueError("ratings file is empty")

    def numeric(cell: str) -> float | None:
        cell = cell.strip()
        if not cell or cell.upper() in ("NA", "NAN", "NONE", "-"):
            return math.nan
        try:
            return float(cell)
        except ValueError:
            return None

    header: list[str] = []
    if all(numeric(c) is None for c in rows[0]):
        header = [c.strip() for c in rows[0]]
        rows = rows[1:]
    if not rows:
        raise ValueError("ratings file has a header but no rating rows")
    if all(numeric(r[0]) is None for r in rows):
        rows = [r[1:] for r in rows]
        header = header[1:] if header else header

    matrix: list[list[float]] = []
    for i, raw in enumerate(rows):
        parsed: list[float] = []
        for j, cell in enumerate(raw):
            value = numeric(cell)
            if value is None:
                raise ValueError(f"cell at row {i + 1}, column {j + 1} is not a number: {cell!r}")
            parsed.append(value)
        matrix.append(parsed)
    widths = {len(r) for r in matrix}
    if len(widths) != 1:
        raise ValueError(f"ragged ratings matrix: rows have widths {sorted(widths)}")
    return matrix, header


@app.command()
def irr(
    ctx: typer.Context,
    ratings: Path = typer.Argument(..., help="CSV of ratings: rows = raters, columns = items."),
    metric: Optional[str] = typer.Option(None, "--metric", help="nominal|ordinal|interval|ratio."),
    json_out: bool = typer.Option(False, "--json", help="Emit alpha as JSON."),
) -> None:
    """Krippendorff's alpha over a ratings matrix, against the configured gate."""
    cfg = _cfg(ctx)
    level = metric or cfg.irr_metric
    try:
        matrix, header = _read_ratings_csv(ratings)
    except (OSError, ValueError) as exc:
        _fail(f"{ratings}: {exc}", json_out=json_out, stage="irr")
        return
    entry = _resolve_callable(_IRR_ENTRIES)
    if entry is None:
        _fail(
            "no Krippendorff implementation found; expected one of "
            + ", ".join(f"{m}.{a}" for m, a in _IRR_ENTRIES),
            json_out=json_out,
            stage="irr",
            looked_for=[f"{m}.{a}" for m, a in _IRR_ENTRIES],
        )
        return
    fn, where = entry
    try:
        result = fn(matrix, metric=level)
    except TypeError as exc:
        _fail(f"{where} does not accept (matrix, metric=...): {exc}", json_out=json_out, stage="irr")
        return
    except (ValueError, CrucibleError) as exc:
        _fail(f"alpha could not be computed: {exc}", json_out=json_out, stage="irr")
        return

    value = result if isinstance(result, (int, float)) else _describe(result)
    alpha_value = float(value) if isinstance(value, (int, float)) else None
    if alpha_value is None and isinstance(value, dict):
        raw = value.get("alpha")
        alpha_value = float(raw) if isinstance(raw, (int, float)) else None
    met = None if alpha_value is None else alpha_value >= cfg.alpha_gate
    payload = {
        "status": "ok",
        "entry_point": where,
        "metric": level,
        "n_raters": len(matrix),
        "n_items": len(matrix[0]) if matrix else 0,
        "items": header,
        "alpha": alpha_value,
        "gate": cfg.alpha_gate,
        "met": met,
        "result": value,
    }
    if json_out:
        _emit_json(payload)
    else:
        console = _console()
        shown = "unmeasured" if alpha_value is None else f"{alpha_value:.4f}"
        console.print(
            f"Krippendorff alpha ({level}) = {shown} over "
            f"{payload['n_raters']} raters x {payload['n_items']} items; gate {cfg.alpha_gate:.2f}"
        )
        if met is False:
            console.print("below the gate: this criterion needs a rewrite, not more raters")
    if met is None:
        raise typer.Exit(EXIT_ERROR)
    raise typer.Exit(EXIT_OK if met else EXIT_FAIL)


_REDTEAM_ENTRIES: tuple[tuple[str, str], ...] = (
    ("crucible.redteam.suite", "run_suite"),
    ("crucible.redteam.suite", "run"),
    ("crucible.redteam", "run_suite"),
)


@app.command()
def redteam(
    ctx: typer.Context,
    seed: Optional[str] = typer.Option(None, "--seed", help="Restrict to one seed id."),
    json_out: bool = typer.Option(False, "--json", help="Emit the suite result as JSON."),
) -> None:
    """Attack the grader. A template ships only at a 100% catch rate."""
    cfg = _cfg(ctx)
    entry = _resolve_callable(_REDTEAM_ENTRIES)
    if entry is None:
        _fail(
            "no red-team suite found; expected one of "
            + ", ".join(f"{m}.{a}" for m, a in _REDTEAM_ENTRIES),
            json_out=json_out,
            stage="redteam",
            looked_for=[f"{m}.{a}" for m, a in _REDTEAM_ENTRIES],
        )
        return
    fn, where = entry
    caps = capabilities.detect(probe_timeout_s=cfg.probe_timeout_s)
    try:
        result = fn(seed_id=seed, caps=caps, cfg=cfg)
    except TypeError as exc:
        _fail(f"{where} does not accept (seed_id, caps, cfg): {exc}", json_out=json_out, stage="redteam")
        return
    except (KeyError, ValueError, OSError, CrucibleError) as exc:
        _fail(f"red team failed: {type(exc).__name__}: {exc}", json_out=json_out, stage="redteam")
        return

    payload = _describe(result)
    from .report.metrics import normalize_redteam

    caught, total, uncaught = normalize_redteam(payload if isinstance(payload, dict) else {})
    body: dict[str, Any] = {"status": "ok", "entry_point": where}
    if isinstance(payload, dict):
        body.update(payload)
    else:
        body["report"] = payload
    if json_out:
        _emit_json(body)
    else:
        console = _console()
        if total:
            console.print(f"attacks caught: {caught}/{total} = {100.0 * caught / total:.1f}%")
        else:
            console.print("the suite reported no attacks; catch rate is unmeasured, not 100%")
        for aid in uncaught:
            console.print(f"[red]UNCAUGHT (grader defect):[/red] {aid}")
    if total == 0:
        raise typer.Exit(EXIT_SKIP_BLOCKED)
    raise typer.Exit(EXIT_OK if not uncaught else EXIT_FAIL)


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #


@app.command()
def coverage(
    ctx: typer.Context,
    bank: Path = typer.Argument(..., help="Directory of task YAML files."),
    top: int = typer.Option(12, "--top", help="How many ranked gaps to show."),
    json_out: bool = typer.Option(False, "--json", help="Emit the coverage grid as JSON."),
) -> None:
    """48-cell fill rates, the ranked gaps, and the T5+T6 share."""
    cfg = _cfg(ctx)
    from .report.coverage import analyze_bank

    report, errors = analyze_bank(bank, target_per_cell=cfg.target_per_cell)
    if json_out:
        payload = report.as_dict()
        payload["load_errors"] = [e.as_dict() for e in errors]
        payload["bank"] = str(bank)
        _emit_json(payload)
        raise typer.Exit(EXIT_OK)

    from .taxonomy import DOMAINS

    console = _console()
    grid = _table(f"coverage ({report.total()} tasks, target {cfg.target_per_cell}/cell)", "tier", *DOMAINS)
    for row in report.heatmap():
        grid.add_row(str(row["tier"]), *[str(c["count"]) for c in row["cells"]])
    console.print(grid)
    console.print(
        f"cells at target: {report.n_filled()}/48 ({100.0 * report.fill_rate():.1f}%); "
        f"empty cells: {report.n_empty()}; T5+T6 share: {100.0 * report.silent_share():.1f}%"
    )
    gaps = _table("highest-value gaps (tier value x emptiness)", "cell", "have", "need", "tier value", "priority")
    for cell in report.gaps_ranked(limit=top):
        gaps.add_row(cell.cell, str(cell.count), str(cell.shortfall), str(cell.value_rank), f"{cell.priority:.2f}")
    console.print(gaps)
    for err in errors:
        console.print(f"[yellow]not loaded:[/yellow] {err.path}: {err.error}")
    raise typer.Exit(EXIT_OK)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def _sibling(verdicts_dir: Path, name: str) -> Path | None:
    """Look for ``<run dir>/<name>`` next to the verdicts dir, as pipeline.sh writes it."""
    candidate = verdicts_dir.parent / name
    return candidate if candidate.is_file() else None


@app.command()
def report(
    ctx: typer.Context,
    bank: Path = typer.Argument(..., help="Directory of task YAML files."),
    verdicts: Path = typer.Argument(..., help="Directory of TaskVerdict JSON files."),
    out: Path = typer.Option(Path("report.html"), "-o", "--out", help="HTML file to write."),
    irr_path: Optional[Path] = typer.Option(None, "--irr", help="JSON of alpha per rubric criterion."),
    redteam_path: Optional[Path] = typer.Option(None, "--redteam", help="JSON red-team suite result."),
    json_out: bool = typer.Option(False, "--json", help="Also emit the metrics as JSON on stdout."),
) -> None:
    """Compute the seven metrics and write the self-contained HTML report."""
    cfg = _cfg(ctx)
    from .report import dashboard
    from .report.metrics import MetricTargets

    # pipeline.sh writes redteam.json and calibration.json beside the verdicts
    # dir, so pick them up without making the user repeat the paths.
    irr_file = irr_path or _sibling(verdicts, "irr.json")
    redteam_file = redteam_path or _sibling(verdicts, "redteam.json")
    targets = MetricTargets(target_per_cell=cfg.target_per_cell)

    try:
        html, metrics_report = dashboard.build(
            bank,
            verdicts,
            irr_path=irr_file,
            redteam_path=redteam_file,
            targets=targets,
            alpha_gate=cfg.alpha_gate,
        )
    except (OSError, ValueError) as exc:
        _fail(f"report could not be built: {exc}", json_out=json_out, stage="report")
        return

    violations = dashboard.check_self_contained(html)
    dashboard.write(html, out)
    combined = metrics_report.combined_verdict()

    if json_out:
        payload = metrics_report.as_dict()
        payload["report_html"] = str(out)
        payload["self_contained"] = not violations
        payload["external_references"] = violations
        _emit_json(payload)
    else:
        console = _console()
        table = _table("the seven metrics", "#", "metric", "value", "target", "state", "n", "detail")
        for i, metric in enumerate(metrics_report.metrics, start=1):
            table.add_row(
                str(i),
                metric.name,
                metric.display_value(),
                metric.display_target(),
                metric.status.upper(),
                str(metric.n),
                metric.detail or metric.reason,
            )
        console.print(table)
        _print_skip_panel(console, metrics_report.verdicts)
        console.print(f"bank-wide verdict: {combined}")
        console.print(f"report written to {out}")
        if violations:
            console.print(f"[yellow]warning:[/yellow] report references external resources: {violations}")
        for err in metrics_report.errors:
            console.print(f"[yellow]not loaded:[/yellow] {err.path}: {err.error}")
    raise typer.Exit(_EXIT_FOR_VERDICT.get(combined, EXIT_ERROR))


@app.command()
def version() -> None:
    """Print the CRUCIBLE version."""
    print(__version__)
    raise typer.Exit(EXIT_OK)


def run(argv: Iterable[str] | None = None) -> int:
    """Programmatic entry point used by tests and by ``python -m crucible.cli``.

    With ``standalone_mode=False`` click *returns* the exit code of a
    ``typer.Exit`` instead of raising it, so the return value is the exit code
    and must not be discarded -- doing so turns every failure into a silent 0.
    """
    try:
        rv = app(args=list(argv) if argv is not None else None, standalone_mode=False)
    except typer.Exit as exc:  # older click raises rather than returns
        return int(exc.exit_code)
    except click_exceptions.UsageError as exc:
        _console(stderr=True).print(f"[red]usage error:[/red] {exc}")
        return EXIT_ERROR
    except SystemExit as exc:
        return int(exc.code or 0)
    return int(rv) if isinstance(rv, int) else EXIT_OK


if __name__ == "__main__":
    sys.exit(run())


__all__ = [
    "app",
    "run",
    "EXIT_OK",
    "EXIT_FAIL",
    "EXIT_SKIP_BLOCKED",
    "EXIT_ERROR",
]
