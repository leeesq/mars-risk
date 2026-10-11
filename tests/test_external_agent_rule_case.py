"""Home Credit 规则发现、独立复核与只消费快照的回归。"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from home_credit_case_fixture import write_case_fixture
from mars.reporting import load_report

SCRIPT = Path(__file__).resolve().parents[1] / "docs/snippets/external_agent_rule_case.py"


@pytest.fixture(scope="module")
def rule_case(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("home-credit-rule-fixture")
    raw = write_case_fixture(root / "raw")
    output = root / "case"
    code = (
        "from pathlib import Path; import sys; "
        f"sys.path.insert(0,{str(SCRIPT.parent)!r}); "
        "from home_credit_data import prepare_home_credit_data,HomeCreditConfig; "
        "from external_agent_rule_case import produce; "
        f"inputs=prepare_home_credit_data({str(raw)!r},config=HomeCreditConfig(target_group_rows=200),require_full_dataset=False); "
        f"produce(Path({str(output)!r}),inputs=inputs)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True,
                   encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    subprocess.run([sys.executable, str(SCRIPT), "--consume-only", "--output-dir", str(output)],
                   check=True, capture_output=True, text=True,
                   encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return output


def test_independent_case_runs_and_replays_evidence(rule_case: Path) -> None:
    cross = load_report(rule_case / "score-cross.marsreport")
    rules = load_report(rule_case / "rules.marsreport")
    fixed = load_report(rule_case / "rules-independent.marsreport")
    assert cross.describe()["parameters"]["bin_definitions"]["x"]["direction"] == "lower_risk"
    assert rules.describe()["parameters"]["profile"] == "production"
    assert rules.describe()["parameters"]["validation_status"] == "independent"
    assert rules.describe()["parameters"]["time_col"] is None
    notes = json.loads((rule_case / "case-notes.json").read_text(encoding="utf-8"))
    assert notes["independent_rows"] and not notes["raw_data_saved"]
    assert notes["real_features"] and notes["simulated_dates"]
    assert not list(rule_case.glob("*.parquet")) and not list(rule_case.glob("*.csv"))
    trace = json.loads((rule_case / "query-trace.json").read_text(encoding="utf-8"))
    reports = {r.report_id: r for r in (cross, rules, fixed)}
    for entry in trace["queries"]:
        ref = entry["reference"]
        assert reports[ref["report_id"]].get_table(ref["table"], **ref["query"]).to_dicts() == entry["rows"]
    assert trace["unavailable"]["answer"] == "当前报告无法回答"
    assert len((rule_case / "rule-agent-context.json").read_text(encoding="utf-8")) <= 12000
    review = json.loads((rule_case / "evidence-review.json").read_text(encoding="utf-8"))
    assert review["kind"] == "deterministic_evidence_review"
    assert review["independent_validation"] and review["findings"]
    for row in fixed.get_table("slices", filters={"slice": "4_oot_undue"}).to_dicts():
        assert row["sample_count"] == 0 and row["event_count"] is None
        assert row["status"] in {"unobserved", "empty"}


def test_consumer_never_calls_computation(rule_case: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    consumer = runpy.run_path(str(SCRIPT))["consume"]

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("消费阶段禁止数据准备或计算")

    for name in ("prepare_home_credit_data", "produce", "cross_scores", "mine_rules"):
        monkeypatch.setitem(consumer.__globals__, name, forbidden)
    review = consumer(rule_case)
    assert review["missing_information"]["answer"] == "当前报告无法回答"


def test_minimal_rule_report_snippet(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(SCRIPT.with_name("rule_portable_report.py")))
    namespace["run"](tmp_path)
    assert load_report(tmp_path / "rules.marsreport").get_feature("debt")["evidence_status"] == "available"
