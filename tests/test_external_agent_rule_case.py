"""完整案例事实、证据引用和只消费快照的边界回归。"""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from mars.reporting import load_report

SCRIPT = Path(__file__).resolve().parents[1] / "docs/snippets/external_agent_rule_case.py"


def test_independent_case_runs_in_new_process_and_replays_evidence(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    cross = load_report(tmp_path / "score-cross.marsreport")
    rules = load_report(tmp_path / "rules.marsreport")
    assert cross.describe()["parameters"]["bin_definitions"]["x"]["direction"] == "lower_risk"
    assert rules.describe()["parameters"]["profile"] == "production"
    assert rules.describe()["parameters"]["validation_status"] == "independent"
    assert "validation_filter" in rules.get_table("candidates")["rejection_stage"].to_list()
    notes = json.loads((tmp_path / "case-notes.json").read_text(encoding="utf-8"))
    assert notes["independent_rows"] and not notes["raw_data_saved"]
    assert not list(tmp_path.glob("*.parquet")) and not list(tmp_path.glob("*.csv"))
    trace = json.loads((tmp_path / "query-trace.json").read_text(encoding="utf-8"))
    reports = {r.report_id: r for r in (cross, rules)}
    for entry in trace["queries"]:
        ref = entry["reference"]
        replay = reports[ref["report_id"]].get_table(ref["table"], **ref["query"])
        assert replay.to_dicts() == entry["rows"]
    assert trace["unavailable"]["answer"] == "当前报告无法回答"
    assert len((tmp_path / "agent-context.json").read_text(encoding="utf-8")) <= 12000
    review = json.loads((tmp_path / "evidence-review.json").read_text(encoding="utf-8"))
    assert review["kind"] == "deterministic_evidence_review" and review["simulated"]
    assert review["independent_validation"] and review["findings"]
    assert "{{OUTPUT_DIR}}" not in (tmp_path / "external-agent-task.md").read_text(encoding="utf-8")


def test_consumer_never_calls_computation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    namespace = runpy.run_path(str(SCRIPT))
    namespace["produce"](tmp_path)
    consumer = namespace["consume"]

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("消费阶段禁止生成原数据或计算")

    for name in ("_sample", "produce", "cross_scores", "mine_rules"):
        monkeypatch.setitem(consumer.__globals__, name, forbidden)
    review = consumer(tmp_path)
    assert review["missing_information"]["answer"] == "当前报告无法回答"


def test_minimal_rule_report_snippet(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(SCRIPT.with_name("rule_portable_report.py")))
    namespace["run"](tmp_path)
    assert (
        load_report(tmp_path / "rules.marsreport").get_feature("debt")["evidence_status"]
        == "available"
    )
