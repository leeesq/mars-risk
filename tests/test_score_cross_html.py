"""真实快照的 HTML 证据、离线约束及 JS/Python policy 口径一致性。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from mars.analysis import cross_scores, evaluate_score_policy, write_score_cross_html
from mars.analysis._score_cross_expression import _EXPRESSION_JAVASCRIPT
from mars.analysis._score_cross_html import SCORE_CROSS_HTML
from mars.reporting import load_report


def _scoped_report(
    weighted: bool = False, labels: bool = True, confidence_level: float = 0.90
) -> Any:
    """构造聚合验收夹具，覆盖相反轴、特殊箱、任意组、周期及未观测。"""
    frame = pl.DataFrame(
        {
            "old <score>": [0.9, 0.8, 0.5, 0.4, 0.1, 0.0, None, -1.0, 1.1, 0.9, 0.9, 0.1],
            "new </script>": [0.1, 0.2, 0.7, 0.6, 0.9, None, 0.5, -1.0, 0.1, 0.9, 0.1, 0.9],
            "bad": [0, 1, None, 1, 1, 0, 1, 0, 1, None, 0, 0],
            "late": [None, 0, 0, 1, 1, 0, None, 0, 1, 1, None, 0],
            "sample": ["dev <&>"] * 9 + ["holdout 非 TRAIN"] * 3,
            "month": ["2025-01-01"] * 9 + ["2025-02-01"] * 3,
            "weight": [0.0, 0.0, 5.0, 2.0, 3.0, 1.0, 6.0, 3.0, 4.0, 1.0, 0.0, 7.0],
            "amount": [10.0, 20.0, 30.0, 5.0, 2.0, 4.0, 8.0, 2.0, 6.0, 1.0, 2.0, 7.0],
        }
    )
    return cross_scores(
        frame,
        score_x="old <score>",
        score_y="new </script>",
        score_directions={"old <score>": "lower_risk", "new </script>": "higher_risk"},
        probability_scores=["old <score>", "new </script>"],
        cutpoints={"old <score>": [0.3, 0.6], "new </script>": [0.5]},
        special_values={"old <score>": [-1.0], "new </script>": [-1.0]},
        targets=["bad", "late"] if labels else None,
        group_col="sample",
        time_col="month",
        time_grain="month",
        weights_col="weight" if weighted else None,
        amount_col="amount",
        min_observed=2,
        confidence_level=confidence_level,
    )


def _payload(path: Path) -> dict[str, Any]:
    """只解析真实导出的 JSON 证据，不模拟浏览器执行。"""
    text = path.read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="data">(.*?)</script>', text, re.S)
    assert match is not None
    payload: dict[str, Any] = json.loads(match.group(1))
    return payload


def test_html_snapshot_keeps_actual_tables_scopes_directions_and_safe_text(tmp_path: Path) -> None:
    report = _scoped_report(weighted=True)
    archive = tmp_path / "scoped.marsreport"
    report.save(archive)
    restored = load_report(archive)
    replay = evaluate_score_policy(restored, {"type": "expression", "expression": "X <= X2"})
    path = tmp_path / "cross.html"
    write_score_cross_html(
        restored, path, report_name="真实 <script>alert(1)</script>", policy_reports=[replay]
    )
    text = path.read_text(encoding="utf-8")
    payload = _payload(path)
    assert payload["description"]["report_id"] == report.report_id
    assert payload["tables"]["cells"] == report.get_table("cells").to_dicts()
    assert payload["tables"]["overall"] == report.get_table("overall").to_dicts()
    assert payload["tables"]["row_summary"] == report.get_table("row_summary").to_dicts()
    assert payload["tables"]["column_summary"] == report.get_table("column_summary").to_dicts()
    assert {row["group"] for row in payload["tables"]["overall"]} == {
        "dev <&>", "holdout 非 TRAIN"
    }
    assert payload["bin_definitions"]["x"]["direction"] == "lower_risk"
    assert payload["rule_contract"]["weighted"] is True
    assert "including special bins" in payload["rule_contract"]["sample_share_denominator"]
    assert payload["policies"][0]["description"]["report_id"] == replay.report_id
    assert "<script>alert(1)</script>" not in text and "new </script>" not in text
    assert "connect-src 'none'" in text and "default-src 'none'" in text
    assert not re.search(r"(?:src|href)\s*=\s*['\"](?:https?://|//)", text)
    assert "fetch(" not in text and "XMLHttpRequest" not in text
    assert "eval(" not in text and "new Function" not in text
    assert "row-chart" in text and "column-chart" in text
    assert "ArrowUp" in text and "aria-pressed" in text
    assert "未加权 " in text and "weighted CI unsupported" in text
    assert "min(2,xs.length)" in text and "min(3,ys.length)" in text
    assert "repeat(5" not in text


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("labels", [False, True])
def test_offline_rule_aggregation_matches_public_policy_for_every_scope(
    tmp_path: Path, weighted: bool, labels: bool
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 未安装，JS/Python 差分未运行；浏览器检查另行执行。")
    report = _scoped_report(weighted=weighted, labels=labels)
    path = tmp_path / "cross.html"
    write_score_cross_html(report, path)
    payload = _payload(path)
    start = SCORE_CROSS_HTML.index("function aggregateScoreRule(")
    end = SCORE_CROSS_HTML.index("\nfunction renderRule()", start)
    aggregation = SCORE_CROSS_HTML[start:end]
    expressions = [
        "X <= X2 AND Y <= Y1",
        "X = X1 OR X = X1 OR Y >= Y2",
        "X != 2 AND Y = 1",
        "X < 1",
        "X = X1 AND Y = Y1",
        "X = X3 AND Y = Y1",
    ]
    script = (
        "const fs=require('fs'),input=JSON.parse(fs.readFileSync(0,'utf8'));\n"
        "const finite=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const ratio=(n,d)=>finite(n)&&finite(d)&&d>0?n/d:null;\n"
        + _EXPRESSION_JAVASCRIPT
        + aggregation
        + "\nconst rows=input.payload.tables.overall.map(scope=>{"
        "const ast=parseScoreExpression(input.expression,3,2);"
        "const cells=input.payload.tables.cells.filter(c=>"
        "['target','group','period'].every(k=>c[k]===scope[k])&&"
        "evaluateScoreExpression(ast,c.x_risk_rank,c.y_risk_rank));"
        "return {...scope,...aggregateScoreRule(cells,scope,input.payload.description.parameters,"
        "input.payload.rule_contract)};});process.stdout.write(JSON.stringify(rows));"
    )
    for expression in expressions:
        completed = subprocess.run(
            [node, "-e", script],
            input=json.dumps({"payload": payload, "expression": expression}),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        actual: list[dict[str, Any]] = json.loads(completed.stdout)
        policy = evaluate_score_policy(report, {"type": "expression", "expression": expression})
        expected = policy.get_table(
            "summary", filters={"rule": "candidate", "retained": True}
        ).to_dicts()
        assert len(actual) == len(expected)
        for row in actual:
            other = next(
                item for item in expected
                if all(item[key] == row[key] for key in ("target", "group", "period"))
            )
            for field in [
                *payload["rule_contract"]["count_fields"],
                "status", "bad_rate", "sample_share", "lift_vs_overall", "lift_status",
                "observed_coverage",
            ]:
                if isinstance(other[field], float):
                    assert row[field] == pytest.approx(other[field]), (expression, field, row)
                else:
                    assert row[field] == other[field], (expression, field, row)


def test_export_rejects_policy_from_another_parent(tmp_path: Path) -> None:
    report = _scoped_report()
    other = _scoped_report()
    policy = evaluate_score_policy(other, {"type": "expression", "expression": "X = X1"})
    with pytest.raises(ValueError, match="derive from this report_id"):
        write_score_cross_html(report, tmp_path / "wrong.html", policy_reports=[policy])


def test_scope_switch_retains_target_when_period_is_unavailable() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 未安装，scope 切换回归未运行。")
    start = SCORE_CROSS_HTML.index("function selectScoreScope(")
    end = SCORE_CROSS_HTML.index("\nfunction changeScope(", start)
    script = SCORE_CROSS_HTML[start:end] + "\n" + (
        "const scopes=[{target:'bad',group:'dev',period:'Jan'},"
        "{target:'bad',group:'hold',period:'Feb'},"
        "{target:'late',group:'dev',period:'Jan'},"
        "{target:'late',group:'hold',period:'Feb'},"
        "{target:'only',group:'custom',period:null}];"
        "process.stdout.write(JSON.stringify(["
        "selectScoreScope(scopes,scopes[2],{group:'hold'}),"
        "selectScoreScope(scopes,scopes[3],{target:'bad'}),"
        "selectScoreScope(scopes,scopes[3],{target:'only'}),"
        "selectScoreScope(scopes,scopes[2],{target:'unknown'})]));"
    )
    completed = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, encoding="utf-8", check=True
    )
    assert json.loads(completed.stdout) == [3, 1, 4, -1]


def test_html_title_markers_are_literal_and_script_is_valid(tmp_path: Path) -> None:
    report = _scoped_report(confidence_level=0.975)
    title = "真实 __DATA__ / __TITLE__ / __EXPRESSION_JS__ <&>"
    path = tmp_path / "marker.html"
    write_score_cross_html(report, path, report_name=title)
    text = path.read_text(encoding="utf-8")
    assert "<title>真实 __DATA__ / __TITLE__ / __EXPRESSION_JS__ &lt;&amp;&gt;</title>" in text
    assert _payload(path)["description"]["parameters"]["confidence_level"] == 0.975
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 未安装，完整导出脚本语法与 Wilson 标签检查未运行。")
    script = re.search(r"<script>(.*?)</script>", text, re.S)
    assert script is not None
    subprocess.run(
        [node, "--check"], input=script.group(1), capture_output=True, text=True,
        encoding="utf-8", check=True,
    )
    match = re.search(r"function confidenceLabel\(.*?\}\n", text)
    assert match is not None
    completed = subprocess.run(
        [node, "-e", match.group(0) + "process.stdout.write(confidenceLabel(.975));"],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    assert completed.stdout == "97.5%"
