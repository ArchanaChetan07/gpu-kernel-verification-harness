"""Tests for the reporting layer and the CLI.

The properties under test are the ones the reporting layer exists to guarantee:

* an absent input yields ``unmeasured``, never ``0`` -- a fabricated zero is the
  exact failure the project was built to remove;
* the headline counts SKIP *against* the numerator, so a run that verified
  nothing cannot look like a run that verified everything;
* the HTML report is genuinely self-contained, asserted by byte scan rather than
  by inspection;
* the CLI's exit codes distinguish "passed" from "could not check".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
from typer.testing import CliRunner

from crucible.capabilities import Capabilities
from crucible.cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, EXIT_SKIP_BLOCKED, app
from crucible.report import coverage as coverage_mod
from crucible.report import dashboard, metrics
from crucible.schema import EnvironmentRecord, OracleResult, Task, TaskVerdict

runner = CliRunner()


# --------------------------------------------------------------------------- #
# local helpers
# --------------------------------------------------------------------------- #


def make_verdict(
    task_id: str,
    verdict: str,
    *,
    executed: bool | None = None,
    oracle_results: list[OracleResult] | None = None,
) -> TaskVerdict:
    """A TaskVerdict whose oracle results are consistent with its verdict."""
    if oracle_results is None:
        if verdict == "PASS":
            oracle_results = [OracleResult(oracle="O1", verdict="PASS")]
        elif verdict == "FAIL":
            oracle_results = [
                OracleResult(
                    oracle="O1",
                    verdict="FAIL",
                    reason="max_rel_err 4.1e-2 exceeds derived tolerance 6.0e-5 on r3_c33",
                )
            ]
        elif verdict == "SKIP":
            oracle_results = [
                OracleResult(
                    oracle="O2",
                    verdict="SKIP",
                    reason="requires cuda; missing cuda (torch.cuda.is_available() returned False)",
                )
            ]
        else:
            oracle_results = [
                OracleResult(oracle="O4", verdict="ERROR", reason="oracle exceeded its 600s budget")
            ]
    if executed is None:
        executed = verdict in ("PASS", "FAIL")
    return TaskVerdict(
        task_id=task_id,
        verdict=verdict,
        executed=executed,
        oracle_results=oracle_results,
        environment=EnvironmentRecord(
            caps={"device_name": "NVIDIA T1000 8GB", "triton": False, "gloo": True},
            python="3.11.5",
            torch="2.6.0+cu124",
            host="test-host",
            utc="2026-01-01T00:00:00+00:00",
        ),
        duration_s=1.25,
        candidate_sha256="c" * 64,
    )


@pytest.fixture
def bank_and_verdicts(
    tmp_path: Path, make_task: Callable[..., Task]
) -> Callable[..., tuple[Path, Path]]:
    """Factory writing a bank dir and a verdicts dir from (task_id, verdict) pairs.

    Each call gets its own run directory so two banks built in one test cannot
    contaminate each other.
    """
    counter = {"n": 0}

    def _build(pairs: list[tuple[str, str]], **task_overrides: Any) -> tuple[Path, Path]:
        counter["n"] += 1
        run = tmp_path / f"run{counter['n']}"
        bank = run / "bank"
        verdicts = run / "verdicts"
        bank.mkdir(parents=True, exist_ok=True)
        verdicts.mkdir(parents=True, exist_ok=True)
        for task_id, verdict in pairs:
            task = make_task(task_id=task_id, **task_overrides)
            task.save(bank / f"{task_id}.yaml")
            if verdict != "UNVERIFIED":
                make_verdict(task_id, verdict).save(verdicts / f"{task_id}.json")
        return bank, verdicts

    return _build


# --------------------------------------------------------------------------- #
# metrics: unmeasured is not zero
# --------------------------------------------------------------------------- #


def test_every_metric_is_unmeasured_on_empty_input(tmp_path: Path) -> None:
    report = metrics.compute(tmp_path / "no-bank", tmp_path / "no-verdicts")

    assert [m.key for m in report.metrics] == list(metrics.METRIC_KEYS)
    for metric in report.metrics:
        assert metric.value is None, f"{metric.key} invented a value from nothing"
        assert metric.value != 0
        assert metric.status == metrics.STATUS_UNMEASURED
        assert metric.met is None
        assert metric.reason, f"{metric.key} is unmeasured without saying why"
        payload = metric.as_dict()
        assert payload["value"] == "unmeasured"
        assert payload["met"] == "unmeasured"
        assert payload["display_value"] == "unmeasured"

    # A missing directory is reported, not silently treated as an empty bank.
    assert len(report.errors) == 2
    assert report.combined_verdict() == "SKIP"


def test_unmeasured_reasons_are_specific(tmp_path: Path, make_task: Callable[..., Task]) -> None:
    """A populated bank with no ratings and no red team still reports honestly."""
    bank = tmp_path / "bank"
    task = make_task()
    task.save(bank / "t.yaml")
    report = metrics.compute(bank, tmp_path / "verdicts")

    assert report.metric("rubric_alpha").value is None
    assert "ratings" in report.metric("rubric_alpha").reason
    assert report.metric("redteam_catch").value is None
    assert "redteam" in report.metric("redteam_catch").reason
    # Coverage and silent share ARE measurable from the bank alone.
    assert report.metric("taxonomy_coverage").value == 0.0
    assert report.metric("silent_share").value == 1.0


# --------------------------------------------------------------------------- #
# metrics: the headline
# --------------------------------------------------------------------------- #


def test_headline_counts_skip_against_the_numerator(make_task: Callable[..., Task]) -> None:
    tasks = [make_task(task_id=f"t-{v.lower()}") for v in ("PASS", "SKIP", "FAIL")]
    verdicts = [
        make_verdict("t-pass", "PASS"),
        make_verdict("t-skip", "SKIP"),
        make_verdict("t-fail", "FAIL"),
    ]

    metric = metrics.headline_execution_rate(tasks, verdicts)

    assert metric.numerator == 1.0
    assert metric.denominator == 3.0
    assert metric.value == pytest.approx(1.0 / 3.0)
    assert metric.n == 3
    assert metric.met is False


def test_headline_counts_an_unverified_task_against_itself(make_task: Callable[..., Task]) -> None:
    tasks = [make_task(task_id="t-pass"), make_task(task_id="t-none")]
    metric = metrics.headline_execution_rate(tasks, [make_verdict("t-pass", "PASS")])

    assert metric.value == pytest.approx(0.5)
    assert "1 shipped with no verdict" in metric.detail


def test_headline_rejects_a_pass_that_never_executed(make_task: Callable[..., Task]) -> None:
    """PASS with executed=False is a contradiction, and counts against."""
    tasks = [make_task(task_id="t-ghost")]
    verdicts = [make_verdict("t-ghost", "PASS", executed=False)]

    metric = metrics.headline_execution_rate(tasks, verdicts)

    assert metric.value == 0.0
    assert "without execution" in metric.detail


# --------------------------------------------------------------------------- #
# metrics: the other six
# --------------------------------------------------------------------------- #


def test_gold_band_uses_only_calibrated_tasks(make_task: Callable[..., Task]) -> None:
    from crucible.schema import CalibrationRecord

    def cal(p1: float) -> CalibrationRecord:
        return CalibrationRecord(
            model="stub",
            k=8,
            n_samples=16,
            n_correct=int(round(p1 * 16)),
            pass_at_1=p1,
            pass_at_k=min(1.0, p1 * 2),
            route="gold",
            rationale="fixture",
        )

    tasks = [
        make_task(task_id="in-band", calibration=cal(0.25)),
        make_task(task_id="too-easy", calibration=cal(0.95)),
        make_task(task_id="uncalibrated", calibration=None),
    ]
    metric = metrics.gold_band_rate(tasks)

    assert metric.n == 2
    assert metric.value == pytest.approx(0.5)
    assert "uncalibrated" in metric.detail


def test_rubric_alpha_accepts_several_report_shapes() -> None:
    flat = metrics.rubric_alpha_rate({"numerics": 0.81, "explanation": 0.44})
    listed = metrics.rubric_alpha_rate(
        {"criteria": [{"id": "numerics", "alpha": 0.81}, {"id": "explanation", "alpha": 0.44}]}
    )
    assert flat.value == pytest.approx(0.5) == listed.value
    assert "below threshold: explanation=0.44" in flat.detail
    # A NaN alpha means "could not be scored"; it must not be counted as a miss.
    assert metrics.rubric_alpha_rate({"numerics": 0.81, "broken": float("nan")}).n == 1


def test_redteam_names_the_attacks_no_oracle_caught() -> None:
    metric = metrics.redteam_catch_rate(
        {
            "attacks": [
                {"id": "cublas_smuggling", "caught": True},
                {"id": "dce_elision", "caught": False},
            ]
        }
    )
    assert metric.value == pytest.approx(0.5)
    assert "UNCAUGHT" in metric.detail and "dce_elision" in metric.detail
    assert metrics.redteam_catch_rate({"attacks": []}).value is None


def test_sme_hours_is_unmeasured_without_provenance(make_task: Callable[..., Task]) -> None:
    tasks = [make_task(task_id="t1", provenance={"generator": "crucible"})]
    verdicts = [make_verdict("t1", "PASS")]

    metric = metrics.sme_hours_per_accepted_task(tasks, verdicts)
    assert metric.value is None
    assert "provenance" in metric.reason

    timed = [make_task(task_id="t1", provenance={"sme_minutes": 90})]
    measured = metrics.sme_hours_per_accepted_task(timed, verdicts)
    assert measured.value == pytest.approx(1.5)
    assert measured.lower_is_better is True
    assert measured.met is True  # 1.5 h is inside the 2.0 h budget


def test_sme_hours_unmeasured_when_nothing_was_accepted(make_task: Callable[..., Task]) -> None:
    tasks = [make_task(task_id="t1", provenance={"sme_hours": 3.0})]
    metric = metrics.sme_hours_per_accepted_task(tasks, [make_verdict("t1", "SKIP")])
    assert metric.value is None
    assert "no accepted task" in metric.reason


def test_skip_breakdown_groups_by_oracle_and_reason(make_task: Callable[..., Task]) -> None:
    verdicts = [make_verdict(f"t{i}", "SKIP") for i in range(3)] + [make_verdict("t9", "FAIL")]
    rows = metrics.skip_breakdown(verdicts)

    assert len(rows) == 1
    assert rows[0]["oracle"] == "O2"
    assert rows[0]["count"] == 3
    assert "missing cuda" in rows[0]["reason"]
    assert rows[0]["task_ids"] == ["t0", "t1", "t2"]


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #


def _cov_tasks(make_task: Callable[..., Task]) -> list[Task]:
    tasks: list[Task] = []
    for i in range(5):  # T1/cuda at target
        tasks.append(make_task(task_id=f"t1c{i}", failure_tier="T1", domain="cuda"))
    for i in range(4):  # T6/cuda one short of target
        tasks.append(make_task(task_id=f"t6c{i}", failure_tier="T6", domain="cuda"))
    return tasks


def test_coverage_ranks_gaps_by_value_times_emptiness(make_task: Callable[..., Task]) -> None:
    report = coverage_mod.analyze(_cov_tasks(make_task), target_per_cell=5)

    assert report.total() == 9
    assert report.n_filled() == 1
    assert report.fill_rate() == pytest.approx(1 / 48)

    ranked = report.gaps_ranked()
    cells = [c.cell for c in ranked]
    assert "T1/cuda" not in cells, "a cell at target is not a gap"
    assert ranked[0].tier == "T6", "the emptiest highest-value tier ranks first"

    # The load-bearing property: a nearly-full T6 cell outranks a wholly empty
    # T1 cell, because T1 hands the model the answer in the error message.
    assert cells.index("T6/cuda") < cells.index("T1/pytorch")
    priorities = [c.priority for c in ranked]
    assert priorities == sorted(priorities, reverse=True)


def test_coverage_heatmap_is_the_full_grid(make_task: Callable[..., Task]) -> None:
    report = coverage_mod.analyze(_cov_tasks(make_task), target_per_cell=5)
    rows = report.heatmap()

    assert [r["tier"] for r in rows] == ["T1", "T2", "T3", "T4", "T5", "T6"]
    assert all(len(r["cells"]) == 8 for r in rows)
    assert sum(len(r["cells"]) for r in rows) == 48
    assert report.silent_share() == pytest.approx(4 / 9)


def test_coverage_of_an_empty_bank_has_all_48_cells() -> None:
    report = coverage_mod.analyze([], target_per_cell=5)
    assert len(report.cells) == 48
    assert report.n_empty() == 48
    assert len(report.gaps_ranked()) == 48


# --------------------------------------------------------------------------- #
# dashboard
# --------------------------------------------------------------------------- #


def test_dashboard_html_is_self_contained(
    bank_and_verdicts: Callable[..., tuple[Path, Path]]
) -> None:
    bank, verdicts = bank_and_verdicts([("t-pass", "PASS"), ("t-skip", "SKIP")])
    html, _ = dashboard.build(bank, verdicts)

    lowered = html.lower()
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "<script src=" not in lowered
    assert "@import" not in lowered
    assert dashboard.check_self_contained(html) == []

    # It really is a styled page, not an empty shell that trivially passes.
    assert "<style>" in html
    assert "prefers-color-scheme: dark" in html


def test_self_containment_check_actually_detects_a_violation() -> None:
    assert "https://" in dashboard.check_self_contained('<img src="https://x/y.png">')
    assert "<script src=" in dashboard.check_self_contained('<script src="a.js"></script>')


def test_dashboard_shows_metrics_coverage_tasks_and_skip_reasons(
    bank_and_verdicts: Callable[..., tuple[Path, Path]]
) -> None:
    bank, verdicts = bank_and_verdicts([("t-pass", "PASS"), ("t-skip", "SKIP"), ("t-fail", "FAIL")])
    html, report = dashboard.build(bank, verdicts)

    assert "HEADLINE" in html
    assert "33.3%" in html  # 1 of 3 executed and passed
    assert "UNMEASURED" in html  # red team and IRR were never supplied
    assert "missing cuda" in html  # the skip-reason panel names the real cause
    for task_id in ("t-pass", "t-skip", "t-fail"):
        assert task_id in html
    assert "boundary_mask" in html  # mutation class column
    assert "T5/triton" in html  # coverage heatmap cell ids
    assert report.headline.value == pytest.approx(1 / 3)


def test_dashboard_irr_panel_uses_the_gate(
    tmp_path: Path, bank_and_verdicts: Callable[..., tuple[Path, Path]]
) -> None:
    bank, verdicts = bank_and_verdicts([("t-pass", "PASS")])
    irr_file = tmp_path / "irr.json"
    irr_file.write_text(json.dumps({"numerics": 0.91, "explanation": 0.40}), encoding="utf-8")

    html, report = dashboard.build(bank, verdicts, irr_path=irr_file, alpha_gate=0.67)

    assert "0.910" in html and "0.400" in html
    assert report.metric("rubric_alpha").value == pytest.approx(0.5)

    data = dashboard.build_data(report, irr={"numerics": 0.91, "explanation": 0.40}, alpha_gate=0.67)
    states = {r["criterion"]: r["status"] for r in data.irr_rows}
    assert states["numerics"] == "met"
    assert states["explanation"] == "unmet"


def test_dashboard_renders_an_empty_bank_without_inventing_numbers(tmp_path: Path) -> None:
    html, report = dashboard.build(tmp_path / "nope", tmp_path / "also-nope")
    assert "unmeasured" in html.lower()
    assert dashboard.check_self_contained(html) == []
    assert all(m.value is None for m in report.metrics)


def test_dashboard_entity_encodes_urls_inside_evidence(
    tmp_path: Path, make_task: Callable[..., Task]
) -> None:
    """A URL quoted in a skip reason must not read as an external reference."""
    bank = tmp_path / "bank"
    verdicts = tmp_path / "verdicts"
    make_task(task_id="t-url").save(bank / "t.yaml")
    make_verdict(
        "t-url",
        "SKIP",
        oracle_results=[
            OracleResult(
                oracle="O4",
                verdict="SKIP",
                reason="triton unavailable; see https://example.invalid/install for the DLL fix",
            )
        ],
    ).save(verdicts / "t.json")

    html, _ = dashboard.build(bank, verdicts)

    assert dashboard.check_self_contained(html) == []
    assert "example.invalid" in html  # the evidence is still there, verbatim


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@pytest.fixture
def stub_caps(monkeypatch: pytest.MonkeyPatch, caps: Capabilities) -> Capabilities:
    """Never probe the real machine from a test."""
    from crucible import capabilities as caps_mod

    monkeypatch.setattr(caps_mod, "detect", lambda **_kwargs: caps)
    return caps


def test_doctor_json_reports_capabilities_and_every_oracle(stub_caps: Capabilities) -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == EXIT_OK, result.output

    payload = json.loads(result.stdout)
    # Flat capability keys, as scripts/pipeline.sh expects to read them.
    assert payload["cuda"] is False
    assert payload["gloo"] is True
    assert payload["triton"] is False
    assert payload["torch_version"] == "2.6.0+cu124"
    # The detail dict must not shadow the real capability values when flattened.
    assert isinstance(payload["capability_details"]["triton"], str)

    assert sorted(payload["oracles"]) == ["O1", "O2", "O3", "O4", "O5"]
    for oracle_id, row in payload["oracles"].items():
        assert row["reason"] or row["ready"], f"{oracle_id} is not ready and did not say why"


def test_doctor_human_output_names_what_is_missing(stub_caps: Capabilities) -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == EXIT_OK, result.output
    assert "capabilities" in result.stdout
    assert "cannot verify" in result.stdout
    assert "DLL load failed" in result.stdout  # the verbatim triton probe error


def test_report_exits_zero_on_a_fully_verified_bank(
    tmp_path: Path, bank_and_verdicts: Callable[..., tuple[Path, Path]]
) -> None:
    bank, verdicts = bank_and_verdicts([("t-a", "PASS"), ("t-b", "PASS")])
    out = tmp_path / "report.html"

    result = runner.invoke(app, ["report", str(bank), str(verdicts), "-o", str(out)])

    assert result.exit_code == EXIT_OK, result.output
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "100.0%" in html
    assert dashboard.check_self_contained(html) == []


def test_report_exits_one_when_a_reference_solution_failed(
    tmp_path: Path, bank_and_verdicts: Callable[..., tuple[Path, Path]]
) -> None:
    bank, verdicts = bank_and_verdicts([("t-a", "PASS"), ("t-b", "FAIL")])
    out = tmp_path / "report.html"

    result = runner.invoke(app, ["report", str(bank), str(verdicts), "-o", str(out)])

    assert result.exit_code == EXIT_FAIL, result.output
    assert out.exists()


def test_report_exits_skip_blocked_when_nothing_could_be_verified(
    tmp_path: Path, bank_and_verdicts: Callable[..., tuple[Path, Path]]
) -> None:
    bank, verdicts = bank_and_verdicts([("t-a", "PASS"), ("t-b", "SKIP")])
    out = tmp_path / "report.html"

    result = runner.invoke(app, ["report", str(bank), str(verdicts), "-o", str(out)])

    assert result.exit_code == EXIT_SKIP_BLOCKED, result.output


def test_report_exits_error_when_an_oracle_errored(
    tmp_path: Path, bank_and_verdicts: Callable[..., tuple[Path, Path]]
) -> None:
    bank, verdicts = bank_and_verdicts([("t-a", "ERROR")])
    result = runner.invoke(
        app, ["report", str(bank), str(verdicts), "-o", str(tmp_path / "r.html")]
    )
    assert result.exit_code == EXIT_ERROR, result.output


def test_report_json_is_machine_readable_and_says_unmeasured(
    tmp_path: Path, bank_and_verdicts: Callable[..., tuple[Path, Path]]
) -> None:
    bank, verdicts = bank_and_verdicts([("t-a", "PASS")])
    result = runner.invoke(
        app, ["report", str(bank), str(verdicts), "-o", str(tmp_path / "r.html"), "--json"]
    )

    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.stdout)
    assert payload["combined_verdict"] == "PASS"
    assert payload["self_contained"] is True
    by_key = {m["key"]: m for m in payload["metrics"]}
    assert by_key["headline_executed_and_passed"]["value"] == 1.0
    assert by_key["redteam_catch"]["value"] == "unmeasured"
    assert by_key["rubric_alpha"]["value"] == "unmeasured"


def test_report_picks_up_a_sibling_redteam_file(
    tmp_path: Path, bank_and_verdicts: Callable[..., tuple[Path, Path]]
) -> None:
    """pipeline.sh writes redteam.json beside the verdicts dir; find it there."""
    bank, verdicts = bank_and_verdicts([("t-a", "PASS")])
    (verdicts.parent / "redteam.json").write_text(
        json.dumps({"attacks": [{"id": "dce_elision", "caught": True}]}), encoding="utf-8"
    )

    result = runner.invoke(
        app, ["report", str(bank), str(verdicts), "-o", str(tmp_path / "r.html"), "--json"]
    )

    payload = json.loads(result.stdout)
    by_key = {m["key"]: m for m in payload["metrics"]}
    assert by_key["redteam_catch"]["value"] == 1.0
    assert by_key["redteam_catch"]["status"] == "met"


def test_coverage_command_emits_the_grid(
    bank_and_verdicts: Callable[..., tuple[Path, Path]]
) -> None:
    bank, _ = bank_and_verdicts([("t-a", "PASS"), ("t-b", "PASS")])

    result = runner.invoke(app, ["coverage", str(bank), "--json"])

    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.stdout)
    assert payload["n_cells"] == 48
    assert payload["total_tasks"] == 2
    assert len(payload["cells"]) == 48
    assert payload["gaps_ranked"][0]["shortfall"] > 0
    assert payload["by_tier"]["T5"] == 2


def test_verify_bank_refuses_an_empty_bank(tmp_path: Path, stub_caps: Capabilities) -> None:
    """No tasks means nothing was verified, which is SKIP-blocked, not success."""
    empty = tmp_path / "empty"
    empty.mkdir()

    result = runner.invoke(app, ["verify-bank", str(empty), "-o", str(tmp_path / "v")])

    assert result.exit_code == EXIT_SKIP_BLOCKED, result.output


def test_verify_reports_error_when_the_seed_cannot_be_resolved(
    tmp_path: Path, make_task: Callable[..., Task], stub_caps: Capabilities
) -> None:
    """The conftest seed is not in the real registry, so this must ERROR loudly."""
    task_path = tmp_path / "task.yaml"
    make_task().save(task_path)
    out = tmp_path / "verdict.json"

    result = runner.invoke(app, ["verify", str(task_path), "-o", str(out), "--json"])

    assert result.exit_code == EXIT_ERROR, result.output
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "ERROR"
    assert payload["executed"] is False
    assert payload["oracle_results"][0]["reason"]
    assert out.exists()


def test_missing_subsystem_still_yields_valid_json(tmp_path: Path) -> None:
    """A stage whose module is absent must fail with a machine-readable reason.

    pipeline.sh redirects stdout straight into a file, so an error that is not
    JSON silently corrupts the run artifact.
    """
    from crucible import cli

    if cli._resolve_callable(cli._REDTEAM_ENTRIES) is not None:
        pytest.skip("the red-team suite is implemented; the missing-module path cannot be exercised")

    result = runner.invoke(app, ["redteam", "--json"])

    assert result.exit_code == EXIT_ERROR
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["stage"] == "redteam"
    assert payload["looked_for"]


def test_run_shim_propagates_the_exit_code(
    tmp_path: Path, bank_and_verdicts: Callable[..., tuple[Path, Path]]
) -> None:
    """``run()`` must return the real code, not swallow it into a silent zero.

    click returns a typer.Exit's code rather than raising it when
    standalone_mode is off, so discarding the return value would make every
    failing pipeline stage look successful.
    """
    from crucible import cli

    bank, verdicts = bank_and_verdicts([("t-a", "PASS"), ("t-b", "FAIL")])
    out = tmp_path / "r.html"

    assert cli.run(["report", str(bank), str(verdicts), "-o", str(out), "--json"]) == EXIT_FAIL

    ok_bank, ok_verdicts = bank_and_verdicts([("t-a", "PASS")])
    assert cli.run(["report", str(ok_bank), str(ok_verdicts), "-o", str(out), "--json"]) == EXIT_OK


def test_ratings_csv_parser_handles_headers_and_gaps(tmp_path: Path) -> None:
    import math

    from crucible import cli

    path = tmp_path / "ratings.csv"
    path.write_text(
        "rater,item1,item2,item3\nalice,1,2,\nbob,1,3,5\n",
        encoding="utf-8",
    )
    matrix, header = cli._read_ratings_csv(path)

    assert header == ["item1", "item2", "item3"]
    assert matrix[0][0] == 1.0 and matrix[1][2] == 5.0
    assert math.isnan(matrix[0][2]), "a blank cell is missing data, not a zero rating"

    bad = tmp_path / "bad.csv"
    bad.write_text("1,2\n3,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a number"):
        cli._read_ratings_csv(bad)
