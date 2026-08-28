"""规则高级分析与报告测试。"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from mars.rule import MarsRule, MarsRuleMiningSpec, MarsRuleReport, MarsRuleSet, mine_rules


def _frame() -> pl.DataFrame:
    """构造交互分析样本。"""
    return pl.DataFrame(
        {
            "x": list(range(100)),
            "z": [value % 2 for value in range(100)],
            "target": [int(value >= 80) for value in range(100)],
            "amount": [float(value + 1) for value in range(100)],
            "customer": [f"c{value // 2}" for value in range(100)],
        }
    )


def test_on_demand_analysis_has_interaction_and_cumulative_tables() -> None:
    """默认挖掘不做 O(n²)，显式 analyze 才生成两类表。"""
    frame = _frame()
    result = mine_rules(
        frame,
        target="target",
        validation_df=frame,
        seed_rules=["x >= 80", "x >= 90"],
        generators=[],
        spec=MarsRuleMiningSpec(iou_threshold=1.0),
    )
    analysis = result.analyze(
        frame,
        amount_col="amount",
        customer_col="customer",
        max_pairs=1,
        bootstrap_repeats=20,
        random_state=7,
    )
    assert analysis.interaction_table.height == 1
    assert analysis.cumulative_table.height == len(result.rule_set.rules)
    assert analysis.metadata["pair_count"] == 1
    assert analysis.bootstrap_table.height == len(result.rule_set.rules)
    assert analysis.bootstrap_table["repeat_count"].min() > 0
    assert analysis.interaction_table["intersection_amount_total"][0] == sum(
        float(value + 1) for value in range(90, 100)
    )
    assert analysis.interaction_table["intersection_customer_count"][0] == 5
    assert "cumulative_amount_lift" in analysis.cumulative_table.columns
    assert "marginal_customer_event_rate" in analysis.cumulative_table.columns
    assert "bootstrap" in result.to_report(analysis).detail_tables
    with pytest.raises(ValueError, match="max_pairs"):
        result.analyze(frame, max_pairs=0)


def test_report_omits_analysis_until_explicitly_supplied(tmp_path: Path) -> None:
    """报告应结构化导出并安全转义用户内容。"""
    frame = _frame()
    result = mine_rules(
        frame,
        target="target",
        validation_df=frame,
        seed_rules=["x >= 80"],
        generators=[],
    )
    report = result.to_report()
    assert "interactions" not in report.detail_tables
    report_with_analysis = result.to_report(result.analyze(frame))
    html_path = tmp_path / "nested" / "rule.html"
    excel_path = tmp_path / "nested" / "rule.xlsx"
    escaped_report = report_with_analysis.__class__(
        summary_table=report_with_analysis.summary_table,
        detail_tables={
            **report_with_analysis.detail_tables,
            "bad[]:*?/\\name": pl.DataFrame({"unsafe": ["<script>alert(1)</script>"]}),
            "bad[]:*?/\\name-two-with-a-very-long-suffix": pl.DataFrame({"x": [1]}),
        },
        metadata={"unsafe": "<script>alert(2)</script>"},
    )
    escaped_report.write_html(html_path)
    escaped_report.write_excel(excel_path)
    html = html_path.read_text(encoding="utf-8")

    assert html_path.exists() and excel_path.exists()
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html
    assert "cumulative" in report_with_analysis.detail_tables
    assert "interactions" not in report_with_analysis.detail_tables
    assert "rule_explanations" in report_with_analysis.detail_tables
    assert "Lift" in report_with_analysis.detail_tables["rule_explanations"]["explanation"][0]


def test_benchmark_report_renders_without_writing() -> None:
    """benchmark 记录应通过 MarsRuleReport 构造并安全渲染 HTML。"""
    report = MarsRuleReport.from_benchmark(
        [{"engine": "mars<script>", "seconds": 1.25}],
    )
    rendered = report.render_html()

    assert report.summary_table["benchmark_rows"][0] == 1
    assert "benchmark" in report.detail_tables
    assert "mars<script>" not in rendered
    assert "mars&lt;script&gt;" in rendered


def test_analysis_rejects_all_null_target() -> None:
    """高级分析不应把全空目标解释成零事件样本。"""
    rule_set = MarsRuleSet([MarsRule("x > 0")])
    with pytest.raises(ValueError, match="没有可分析"):
        from mars.rule.analysis import analyze_rule_set

        analyze_rule_set(
            rule_set,
            pl.DataFrame({"x": [1], "target": [None]}),
            target="target",
        )
