"""The self-contained HTML report.

Self-contained means exactly that: one file, inline CSS, no CDN, no external
font, no external script, no network access of any kind. A report that phones
home is a report that stops working the moment it is copied off the pod, and
this file is the artifact people actually keep.

``check_self_contained`` enforces the property by byte-level scan rather than by
good intentions, and ``render`` runs it on its own output.

Two rendering notes:

* Every free-text field that comes from evidence (skip reasons, load errors)
  passes through the ``noscheme`` filter, which entity-encodes ``://``. The
  browser renders the text identically; what changes is that a byte scan for an
  external reference cannot be confused by a URL that merely *appears inside*
  quoted evidence.
* The page is styled for light and dark via ``prefers-color-scheme`` only. No
  script means no theme toggle, and a toggle is not worth a script tag here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from jinja2 import Environment, StrictUndefined

from ..schema import Task, TaskVerdict
from ..taxonomy import DOMAINS
from . import coverage as coverage_mod
from . import metrics as metrics_mod

logger = logging.getLogger(__name__)

#: Substrings that would mean the page is not self-contained.
EXTERNAL_MARKERS: tuple[str, ...] = (
    "http://",
    "https://",
    "<script src=",
    "<script src ",
    "<link rel=\"stylesheet\"",
    "<link rel='stylesheet'",
    "@import",
    "<iframe",
    "url(//",
)


def check_self_contained(html: str) -> list[str]:
    """Markers of an external dependency found in the rendered page (empty = clean)."""
    lowered = html.lower()
    return [m for m in EXTERNAL_MARKERS if m in lowered]


def _noscheme(text: Any) -> str:
    """Entity-encode ``://`` so evidence text cannot look like a resource link."""
    return str(text).replace("://", ":&#47;&#47;")


def _fill_class(fill: float) -> str:
    if fill <= 0.0:
        return "f0"
    if fill < 0.25:
        return "f1"
    if fill < 0.5:
        return "f2"
    if fill < 1.0:
        return "f3"
    return "f4"


def _truncate(text: str, limit: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #


@dataclass
class DashboardData:
    title: str
    generated_utc: str
    bank_dir: str
    verdicts_dir: str
    metrics: list[dict[str, Any]] = field(default_factory=list)
    verdict_counts: dict[str, int] = field(default_factory=dict)
    combined_verdict: str = "SKIP"
    n_tasks: int = 0
    n_verdicts: int = 0
    heatmap: list[dict[str, Any]] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    coverage_summary: dict[str, Any] = field(default_factory=dict)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    skips: list[dict[str, Any]] = field(default_factory=list)
    n_skips: int = 0
    irr_rows: list[dict[str, Any]] = field(default_factory=list)
    irr_gate: float = 0.67
    irr_reason: str = ""
    environment: dict[str, str] = field(default_factory=dict)
    load_errors: list[dict[str, str]] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)


def _task_rows(
    tasks: Sequence[Task],
    verdicts: Sequence[TaskVerdict],
) -> list[dict[str, Any]]:
    by_id = metrics_mod.index_verdicts(verdicts)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in tasks:
        seen.add(task.task_id)
        v = by_id.get(task.task_id)
        rows.append(_task_row(task.task_id, task, v))
    for v in verdicts:
        if v.task_id not in seen:
            # Verified something the bank does not contain. Worth seeing.
            rows.append(_task_row(v.task_id, None, v))
    rows.sort(key=lambda r: (r["verdict_rank"], r["task_id"]))
    return rows


_VERDICT_RANK = {"ERROR": 0, "FAIL": 1, "SKIP": 2, "UNVERIFIED": 3, "PASS": 4}


def _task_row(task_id: str, task: Task | None, v: TaskVerdict | None) -> dict[str, Any]:
    verdict = v.verdict if v is not None else "UNVERIFIED"
    cal = task.calibration if task is not None else None
    not_verified_by = (
        ", ".join(sorted({r.oracle for r in v.oracle_results if r.verdict != "PASS"}))
        if v is not None
        else ""
    )
    return {
        "task_id": task_id,
        "tier": task.failure_tier if task is not None else "-",
        "domain": task.domain if task is not None else "-",
        "mutation": task.mutation.cls if task is not None else "-",
        "verdict": verdict,
        "verdict_rank": _VERDICT_RANK.get(verdict, 5),
        "executed": ("yes" if v.executed else "no") if v is not None else "-",
        "pass_at_1": f"{cal.pass_at_1:.2f}" if cal is not None else "-",
        "route": cal.route if cal is not None else "-",
        "in_bank": task is not None,
        "not_verified_by": not_verified_by or "-",
    }


def _irr_rows(
    tasks: Sequence[Task],
    irr: Any,
    gate: float,
) -> tuple[list[dict[str, Any]], str]:
    """Alpha per criterion against the gate, plus criteria that have no alpha.

    A machine-probed criterion carries no human variance, so a missing alpha
    there is expected rather than a hole; the panel says which is which.
    """
    alphas = metrics_mod.normalize_alphas(irr)
    known: dict[str, dict[str, Any]] = {}
    for task in tasks:
        for crit in task.rubric.criteria:
            entry = known.setdefault(
                crit.id, {"criterion": crit.id, "n_tasks": 0, "machine_probed": False}
            )
            entry["n_tasks"] += 1
            entry["machine_probed"] = entry["machine_probed"] or bool(crit.machine_probed)
    for cid in alphas:
        known.setdefault(cid, {"criterion": cid, "n_tasks": 0, "machine_probed": False})

    rows: list[dict[str, Any]] = []
    for cid in sorted(known):
        entry = known[cid]
        alpha = alphas.get(cid)
        if alpha is None:
            status = "unmeasured"
            display = metrics_mod.UNMEASURED
        elif alpha >= gate:
            status = "met"
            display = f"{alpha:.3f}"
        else:
            status = "unmet"
            display = f"{alpha:.3f}"
        rows.append(
            {
                **entry,
                "alpha": alpha,
                "alpha_display": display,
                "status": status,
                "gate": gate,
            }
        )
    if not alphas:
        reason = (
            "no inter-rater ratings were supplied, so no alpha could be computed. "
            "Run 'crucible irr ratings.csv --json' and pass the result with --irr."
        )
    else:
        reason = ""
    return rows, reason


def _environment(verdicts: Sequence[TaskVerdict]) -> dict[str, str]:
    for v in verdicts:
        env = v.environment
        if env.host or env.python or env.torch:
            return {
                "host": env.host or "unknown",
                "python": env.python or "unknown",
                "torch": env.torch or "unknown",
                "recorded_utc": env.utc or "unknown",
                "device": str(env.caps.get("device_name") or "no CUDA device"),
                "platform": str(env.caps.get("platform") or "unknown"),
                "triton": "yes" if env.caps.get("triton") else "no",
                "gloo": "yes" if env.caps.get("gloo") else "no",
                "ncu": "yes" if env.caps.get("ncu") else "no",
            }
    return {}


def build_data(
    report: metrics_mod.MetricsReport,
    *,
    coverage_report: coverage_mod.CoverageReport | None = None,
    irr: Any = None,
    alpha_gate: float = 0.67,
    title: str = "CRUCIBLE report",
    gap_limit: int = 12,
) -> DashboardData:
    cov = coverage_report or coverage_mod.analyze(
        report.tasks, target_per_cell=report.targets.target_per_cell
    )
    heatmap = cov.heatmap()
    for row in heatmap:
        for cell in row["cells"]:
            cell["fill_class"] = _fill_class(float(cell["fill"]))
    skips = report.skip_breakdown()
    for row in skips:
        row["reason_display"] = _truncate(row["reason"])
        row["task_ids_display"] = ", ".join(row["task_ids"])
    irr_rows, irr_reason = _irr_rows(report.tasks, irr, alpha_gate)
    return DashboardData(
        title=title,
        generated_utc=report.generated_utc,
        bank_dir=report.bank_dir,
        verdicts_dir=report.verdicts_dir,
        metrics=[m.as_dict() for m in report.metrics],
        verdict_counts=report.verdict_counts(),
        combined_verdict=report.combined_verdict(),
        n_tasks=len(report.tasks),
        n_verdicts=len(report.verdicts),
        heatmap=heatmap,
        domains=list(DOMAINS),
        gaps=[g.as_dict() for g in cov.gaps_ranked(limit=gap_limit)],
        coverage_summary=cov.as_dict(),
        tasks=_task_rows(report.tasks, report.verdicts),
        skips=skips,
        n_skips=sum(int(r["count"]) for r in skips),
        irr_rows=irr_rows,
        irr_gate=alpha_gate,
        irr_reason=irr_reason,
        environment=_environment(report.verdicts),
        load_errors=[e.as_dict() for e in report.errors],
        sources={
            "irr": report.irr_source or "not supplied",
            "redteam": report.redteam_source or "not supplied",
        },
    )


# --------------------------------------------------------------------------- #
# template
# --------------------------------------------------------------------------- #

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ d.title }}</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #ffffff;
  --panel: #f7f8fa;
  --fg: #14171c;
  --muted: #5c6472;
  --border: #d9dee6;
  --accent: #2b4f8a;
  --ok-fg: #145c35; --ok-bg: #e3f3ea; --ok-br: #9fd3b6;
  --bad-fg: #8f1d17; --bad-bg: #fbe9e7; --bad-br: #eeaea7;
  --unk-fg: #7a5a10; --unk-bg: #fbf1da; --unk-br: #e5cd92;
  --f0: #eef0f3; --f1: #dbe7f6; --f2: #b7d1ee; --f3: #86b3e2; --f4: #3d7fc4;
  --cellfg: #14171c;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1216;
    --panel: #161b22;
    --fg: #e6e9ee;
    --muted: #99a2b0;
    --border: #2a323d;
    --accent: #8ab4f8;
    --ok-fg: #7ee0a8; --ok-bg: #12291d; --ok-br: #2c5f42;
    --bad-fg: #ff9b91; --bad-bg: #2b1614; --bad-br: #6b2c26;
    --unk-fg: #f0cd7e; --unk-bg: #2a2312; --unk-br: #6a5620;
    --f0: #1b2129; --f1: #1e3552; --f2: #234a76; --f3: #2f6299; --f4: #4b87c8;
    --cellfg: #e6e9ee;
  }
}
* { box-sizing: border-box; }
html, body { max-width: 100%; overflow-x: hidden; }
body {
  margin: 0;
  padding: 0 0 4rem;
  background: var(--bg);
  color: var(--fg);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.5;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 1rem; }
header { border-bottom: 1px solid var(--border); padding: 1.6rem 0 1.2rem; margin-bottom: 1.6rem; }
h1 { font-size: 1.5rem; margin: 0 0 .3rem; letter-spacing: -0.01em; }
h2 { font-size: 1.05rem; margin: 2.2rem 0 .5rem; letter-spacing: .02em; text-transform: uppercase; color: var(--muted); }
h2:first-of-type { margin-top: 1rem; }
p.sub { margin: .2rem 0; color: var(--muted); font-size: .85rem; }
code, .mono { font-family: var(--mono); font-size: .85em; }

.headline {
  border: 1px solid var(--border); border-left: 4px solid var(--accent);
  background: var(--panel); border-radius: 8px; padding: 1rem 1.1rem; margin: 1rem 0 0;
}
.headline .big { font-size: 2.1rem; font-weight: 650; line-height: 1.1; }
.headline .lbl { color: var(--muted); font-size: .85rem; }

.tiles { display: grid; gap: .75rem; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
.tile { border: 1px solid var(--border); background: var(--panel); border-radius: 8px; padding: .8rem .85rem; }
.tile .name { font-size: .8rem; color: var(--muted); min-height: 2.4em; }
.tile .val { font-size: 1.6rem; font-weight: 620; line-height: 1.2; margin: .25rem 0 .1rem; }
.tile .tgt { font-size: .78rem; color: var(--muted); }
.tile .det { font-size: .76rem; color: var(--muted); margin-top: .45rem; }

.badge {
  display: inline-block; padding: .05rem .45rem; border-radius: 999px;
  font-size: .68rem; font-weight: 700; letter-spacing: .06em; border: 1px solid transparent;
}
.st-met    { color: var(--ok-fg);  background: var(--ok-bg);  border-color: var(--ok-br); }
.st-unmet  { color: var(--bad-fg); background: var(--bad-bg); border-color: var(--bad-br); }
.st-unmeasured { color: var(--unk-fg); background: var(--unk-bg); border-color: var(--unk-br); }

.scroll { overflow-x: auto; max-width: 100%; border: 1px solid var(--border); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: .84rem; }
th, td { padding: .38rem .55rem; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }
th { position: sticky; top: 0; background: var(--panel); font-weight: 620; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.nowrap { white-space: nowrap; }
td.reason { min-width: 24rem; }

.grid td.cell { text-align: center; font-variant-numeric: tabular-nums; color: var(--cellfg); font-weight: 600; }
.f0 { background: var(--f0); } .f1 { background: var(--f1); } .f2 { background: var(--f2); }
.f3 { background: var(--f3); } .f4 { background: var(--f4); }
.grid th.dom { font-size: .72rem; }
.legend { font-size: .76rem; color: var(--muted); margin-top: .4rem; }
.legend span { display: inline-block; width: 1.6rem; height: .8rem; vertical-align: -1px;
  border: 1px solid var(--border); margin: 0 .15rem 0 .6rem; }

.v-PASS { color: var(--ok-fg); font-weight: 650; }
.v-FAIL, .v-ERROR { color: var(--bad-fg); font-weight: 650; }
.v-SKIP, .v-UNVERIFIED { color: var(--unk-fg); font-weight: 650; }

.note { font-size: .82rem; color: var(--muted); margin: .5rem 0 .2rem; }
.empty { border: 1px dashed var(--border); border-radius: 8px; padding: .9rem;
  color: var(--muted); font-size: .85rem; background: var(--panel); }
footer { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--border);
  color: var(--muted); font-size: .78rem; }
dl.kv { display: grid; grid-template-columns: max-content 1fr; gap: .1rem .8rem; margin: 0; font-size: .82rem; }
dl.kv dt { color: var(--muted); }
dl.kv dd { margin: 0; }
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>{{ d.title }}</h1>
  <p class="sub">generated {{ d.generated_utc }} &middot; bank <span class="mono">{{ d.bank_dir }}</span>
     &middot; verdicts <span class="mono">{{ d.verdicts_dir }}</span></p>
  <p class="sub">{{ d.n_tasks }} tasks in the bank, {{ d.n_verdicts }} verdicts on disk.
     Bank-wide verdict: <span class="v-{{ d.combined_verdict }}">{{ d.combined_verdict }}</span>.</p>

  {% set h = d.metrics[0] %}
  <div class="headline">
    <div class="lbl">HEADLINE &mdash; {{ h.name }}</div>
    <div class="big">{{ h.display_value }}
      <span class="badge st-{{ h.status }}">{{ h.status | upper }}</span></div>
    <div class="lbl">target {{ h.display_target }}
      {%- if h.detail %} &middot; {{ h.detail }}{% endif %}
      {%- if h.reason %} &middot; {{ h.reason | noscheme }}{% endif %}</div>
    <div class="lbl">SKIP is not PASS: an oracle that could not run counts against this number,
      and so does a shipped task with no verdict at all.</div>
  </div>
</header>

<h2>The seven metrics</h2>
<div class="tiles">
{% for m in d.metrics %}
  <div class="tile">
    <div class="name">{{ loop.index }}. {{ m.name }}</div>
    <div class="val">{{ m.display_value }}</div>
    <div class="tgt">target {{ m.display_target }}
      <span class="badge st-{{ m.status }}">{{ m.status | upper }}</span></div>
    <div class="det">n = {{ m.n }}{% if m.detail %} &middot; {{ m.detail | noscheme }}{% endif %}
      {%- if m.reason %} &middot; {{ m.reason | noscheme }}{% endif %}</div>
  </div>
{% endfor %}
</div>
<p class="note">An unmeasured metric is not a zero. Targets are policy inputs; only the values are
   measurements.</p>

<h2>Why tasks were not verified &mdash; {{ d.n_skips }} non-passing oracle results</h2>
{% if d.skips %}
<div class="scroll">
<table>
  <thead><tr>
    <th>Oracle</th><th>Verdict</th><th class="num">Count</th><th>Reason</th><th>Example tasks</th>
  </tr></thead>
  <tbody>
  {% for row in d.skips %}
    <tr>
      <td class="nowrap mono">{{ row.oracle }}</td>
      <td class="nowrap v-{{ row.verdict }}">{{ row.verdict }}</td>
      <td class="num">{{ row.count }}</td>
      <td class="reason">{{ row.reason_display | noscheme }}</td>
      <td class="mono">{{ row.task_ids_display | noscheme }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</div>
<p class="note">This is the list of claims this machine could not check. Each row is a capability
   that was absent, not a check that passed.</p>
{% else %}
<div class="empty">No SKIP or ERROR oracle results were recorded.
  {% if d.n_verdicts == 0 %}No verdicts were loaded at all, so nothing was verified either.{% endif %}</div>
{% endif %}

<h2>Coverage &mdash; 48 cells, {{ d.coverage_summary.n_filled }} at target of
    {{ d.coverage_summary.target_per_cell }}</h2>
<div class="scroll">
<table class="grid">
  <thead><tr>
    <th>Tier</th><th>Failure mode</th>
    {% for dom in d.domains %}<th class="dom num">{{ dom }}</th>{% endfor %}
    <th class="num">Total</th>
  </tr></thead>
  <tbody>
  {% for row in d.heatmap %}
    <tr>
      <td class="nowrap"><strong>{{ row.tier }}</strong></td>
      <td class="nowrap">{{ row.failure_mode }}{% if row.silent %} (silent){% endif %}</td>
      {% for c in row.cells %}<td class="cell {{ c.fill_class }}" title="{{ c.cell }}">{{ c.count }}</td>{% endfor %}
      <td class="num">{{ row.total }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</div>
<p class="legend">fill vs the per-cell target:
  <span class="f0"></span>empty <span class="f1"></span>&lt;25%
  <span class="f2"></span>&lt;50% <span class="f3"></span>&lt;100% <span class="f4"></span>at target
  &nbsp;&middot;&nbsp; T5+T6 share {{ '%.1f' | format(100 * d.coverage_summary.silent_share) }}%</p>

<h2>Highest-value gaps</h2>
{% if d.gaps %}
<div class="scroll">
<table>
  <thead><tr>
    <th>Cell</th><th class="num">Have</th><th class="num">Need</th>
    <th class="num">Tier value</th><th class="num">Priority</th>
  </tr></thead>
  <tbody>
  {% for g in d.gaps %}
    <tr>
      <td class="nowrap mono">{{ g.cell }}</td>
      <td class="num">{{ g.count }}</td>
      <td class="num">{{ g.shortfall }}</td>
      <td class="num">{{ g.value_rank }}</td>
      <td class="num">{{ '%.2f' | format(g.priority) }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</div>
<p class="note">priority = tier data-value rank x emptiness. A half-full T6 cell outranks an empty
   T1 cell on purpose: T1 hands the model the answer in the error message.</p>
{% else %}
<div class="empty">Every cell is at target.</div>
{% endif %}

<h2>Inter-rater reliability &mdash; alpha vs the {{ '%.2f' | format(d.irr_gate) }} gate</h2>
{% if d.irr_reason %}<div class="empty">{{ d.irr_reason | noscheme }}</div>{% endif %}
{% if d.irr_rows %}
<div class="scroll">
<table>
  <thead><tr>
    <th>Criterion</th><th class="num">Krippendorff alpha</th><th>State</th>
    <th class="num">Tasks using it</th><th>Machine-probed</th>
  </tr></thead>
  <tbody>
  {% for r in d.irr_rows %}
    <tr>
      <td class="nowrap mono">{{ r.criterion }}</td>
      <td class="num">{{ r.alpha_display }}</td>
      <td><span class="badge st-{{ r.status }}">{{ r.status | upper }}</span></td>
      <td class="num">{{ r.n_tasks }}</td>
      <td>{{ 'yes' if r.machine_probed else 'no' }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</div>
<p class="note">A machine-probed criterion carries no human variance, so a missing alpha there is
   expected rather than a hole.</p>
{% elif not d.irr_reason %}
<div class="empty">No rubric criteria were found in the bank.</div>
{% endif %}

<h2>Tasks</h2>
{% if d.tasks %}
<div class="scroll">
<table>
  <thead><tr>
    <th>Task</th><th>Tier</th><th>Domain</th><th>Mutation class</th>
    <th>Verdict</th><th>Executed</th><th class="num">pass@1</th><th>Route</th>
    <th>Not verified by</th>
  </tr></thead>
  <tbody>
  {% for t in d.tasks %}
    <tr>
      <td class="nowrap mono">{{ t.task_id }}{% if not t.in_bank %} (not in bank){% endif %}</td>
      <td class="nowrap">{{ t.tier }}</td>
      <td class="nowrap">{{ t.domain }}</td>
      <td class="nowrap mono">{{ t.mutation }}</td>
      <td class="nowrap v-{{ t.verdict }}">{{ t.verdict }}</td>
      <td class="nowrap">{{ t.executed }}</td>
      <td class="num">{{ t.pass_at_1 }}</td>
      <td class="nowrap">{{ t.route }}</td>
      <td class="nowrap mono">{{ t.not_verified_by }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</div>
{% else %}
<div class="empty">The bank is empty and no verdicts were found.</div>
{% endif %}

{% if d.environment %}
<h2>Environment</h2>
<dl class="kv">
{% for k, v in d.environment.items() %}
  <dt>{{ k }}</dt><dd class="mono">{{ v | noscheme }}</dd>
{% endfor %}
</dl>
<p class="note">Recorded by the verifier at run time, not probed when this page was rendered.</p>
{% endif %}

{% if d.load_errors %}
<h2>Files that could not be loaded</h2>
<div class="scroll">
<table>
  <thead><tr><th>Path</th><th>Error</th></tr></thead>
  <tbody>
  {% for e in d.load_errors %}
    <tr><td class="mono">{{ e.path | noscheme }}</td><td>{{ e.error | noscheme }}</td></tr>
  {% endfor %}
  </tbody>
</table>
</div>
<p class="note">These files are excluded from every number on this page.</p>
{% endif %}

<footer>
  Verdict counts over the whole bank:
  {% for k, v in d.verdict_counts.items() %}{{ k }}={{ v }}{% if not loop.last %}, {% endif %}{% endfor %}.
  IRR source: <span class="mono">{{ d.sources.irr | noscheme }}</span>.
  Red-team source: <span class="mono">{{ d.sources.redteam | noscheme }}</span>.
  <br>Self-contained: inline CSS only, no external script, style, font or image.
</footer>

</div>
</body>
</html>
"""


def _environment_jinja() -> Environment:
    env = Environment(autoescape=True, undefined=StrictUndefined, trim_blocks=False)
    env.filters["noscheme"] = _noscheme
    return env


def render(
    report: metrics_mod.MetricsReport,
    *,
    coverage_report: coverage_mod.CoverageReport | None = None,
    irr: Any = None,
    alpha_gate: float = 0.67,
    title: str = "CRUCIBLE report",
) -> str:
    """Render the whole report to one self-contained HTML string."""
    data = build_data(
        report,
        coverage_report=coverage_report,
        irr=irr,
        alpha_gate=alpha_gate,
        title=title,
    )
    html = _environment_jinja().from_string(TEMPLATE).render(d=data)
    violations = check_self_contained(html)
    if violations:
        # Never silently ship a page that reaches out to the network.
        logger.warning(
            "rendered report contains external references %s; the page is not self-contained",
            violations,
        )
    return html


def build(
    bank_dir: Path | str,
    verdicts_dir: Path | str,
    *,
    irr_path: Path | str | None = None,
    redteam_path: Path | str | None = None,
    targets: metrics_mod.MetricTargets = metrics_mod.DEFAULT_TARGETS,
    alpha_gate: float = 0.67,
    title: str = "CRUCIBLE report",
) -> tuple[str, metrics_mod.MetricsReport]:
    """Load, compute and render in one call. Returns (html, metrics report)."""
    report = metrics_mod.compute(
        bank_dir,
        verdicts_dir,
        irr_path=irr_path,
        redteam_path=redteam_path,
        targets=targets,
    )
    irr_payload, _ = metrics_mod.read_json_file(irr_path)
    html = render(report, irr=irr_payload, alpha_gate=alpha_gate, title=title)
    return html, report


def write(html: str, out_path: Path | str) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p


def render_and_write(
    bank_dir: Path | str,
    verdicts_dir: Path | str,
    out_path: Path | str,
    **kwargs: Any,
) -> tuple[Path, metrics_mod.MetricsReport]:
    html, report = build(bank_dir, verdicts_dir, **kwargs)
    return write(html, out_path), report


__all__ = [
    "TEMPLATE",
    "EXTERNAL_MARKERS",
    "DashboardData",
    "check_self_contained",
    "build_data",
    "render",
    "build",
    "write",
    "render_and_write",
]
