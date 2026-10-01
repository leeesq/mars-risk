"""分箱 DSL 解释器、镜像规则、完整聚合证据与保存后独立回放的行为验证。"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.analysis import cross_scores, evaluate_score_policy
from mars.analysis._score_cross_expression import (
    _EXPRESSION_JAVASCRIPT,
    _evaluate_score_expression,
    _parse_score_expression,
    _score_rule_examples,
)
from mars.reporting import ReportSnapshot, load_report


@pytest.mark.parametrize(
    ("expression", "matches"),
    [
        ("x <= x2 and y <= y3", [(x, y) for x in (1, 2) for y in (1, 2, 3)]),
        ("X < 2", [(1, y) for y in range(1, 6)]),
        ("X == X1", [(1, y) for y in range(1, 6)]),
        ("X != 1", [(x, y) for x in range(2, 6) for y in range(1, 6)]),
        ("X > 4", [(5, y) for y in range(1, 6)]),
        ("Y >= Y5", [(x, 5) for x in range(1, 6)]),
        ("X=1 OR X=2 AND Y=1", [(1, y) for y in range(1, 6)] + [(2, 1)]),
        ("(X=1 OR X=2) AND Y=1", [(1, 1), (2, 1)]),
        ("X=1 OR X=1 OR Y=1", [(1, y) for y in range(1, 6)] + [(x, 1) for x in range(2, 6)]),
        ("X > X5", []),
    ],
)
def test_safe_expression_operators_precedence_and_overlap(
    expression: str, matches: list[tuple[int, int]]
) -> None:
    node = _parse_score_expression(expression, 5, 5)
    actual = {
        (x, y) for x in range(1, 6) for y in range(1, 6) if _evaluate_score_expression(node, x, y)
    }
    assert actual == set(matches)
    assert not _evaluate_score_expression(node, None, 1)
    assert not _evaluate_score_expression(node, 1, None)


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        ("", "不能为空"),
        ("X <= Y2", "轴不匹配"),
        ("X X2", "缺少比较符"),
        ("X <= 1.5", "箱号必须是整数"),
        ("X <= 0", "越界"),
        ("Y <= Y6", "越界"),
        ("(X=1", "括号不成对"),
        ("X=1)", "括号不成对"),
        ("X=1 AND ()", "括号不成对"),
        ("X=1 Y=1", "缺少 AND 或 OR"),
        ("X=1 && Y=1", "非法词元"),
        ("X=1;alert(1)", "非法词元"),
        ("__import__('os')", "非法词元"),
        ("X=1 OR Function('return true')()", "非法词元"),
        ("X=1 OR eval(1)", "非法词元"),
        ("X=" + "1" * 240, "长度超限"),
        ("X=1 " * 33, "复杂度超限"),
        ("(" * 13 + "X=1" + ")" * 13, "嵌套超限"),
    ],
)
def test_safe_expression_rejects_invalid_and_injected_input(expression: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _parse_score_expression(expression, 5, 5)


def test_expression_complexity_boundaries() -> None:
    expression = "(" * 12 + "X=1" + ")" * 12
    assert _evaluate_score_expression(_parse_score_expression(expression, 5, 5), 1, 1)
    assert _parse_score_expression("X=1" + " " * 237, 1, 1)["rank"] == 1


@pytest.mark.parametrize(("x_count", "y_count"), [(5, 5), (3, 7), (7, 3), (1, 1), (1, 4), (2, 1), (2, 2), (3, 1)])
def test_mirror_examples_follow_actual_counts_and_accurate_description(
    x_count: int, y_count: int
) -> None:
    examples = _score_rule_examples(x_count, y_count)
    sets: list[set[tuple[int, int]]] = []
    for example in examples:
        ast = _parse_score_expression(example["expression"], x_count, y_count)
        matched = {
            (x, y)
            for x in range(1, x_count + 1)
            for y in range(1, y_count + 1)
            if _evaluate_score_expression(ast, x, y)
        }
        assert matched == {tuple(cell) for cell in example["cells"]}
        assert len(example["cells"]) == len(matched)
        assert f"共 {len(matched)} 格" in example["description"]
        sets.append(matched)
    assert sets[1] == {(x_count + 1 - x, y_count + 1 - y) for x, y in sets[0]}
    assert "不是审批规则" in examples[1]["description"]
    if (x_count, y_count) == (5, 5):
        assert len(sets[0]) == len(sets[1]) == 8
        assert examples[0]["expression"] == "(X = X1) OR (X = X2 AND Y <= Y2) OR (X = X3 AND Y = Y1)"
        assert examples[1]["expression"] == "(X = X5) OR (X = X4 AND Y >= Y4) OR (X = X3 AND Y = Y5)"


def _weighted_report() -> Any:
    rows: list[dict[str, Any]] = []
    for group, period in [("开发<&", "2026-01-01"), ("回看", "2026-02-01")]:
        for x in range(1, 6):
            for y in range(1, 6):
                rows.append(
                    {"x": float(x), "y": float(y), "bad": (x + y) % 2, "late": None if y == 5 else x % 2, "w": float(x * y), "a": float(10 * y), "g": group, "t": period}
                )
        for x, y in [(None, 1.0), (5.0, None), (-99.0, 1.0), (5.0, float("inf"))]:
            rows.append({"x": x, "y": y, "bad": 1, "late": None, "w": 9.0, "a": 90.0, "g": group, "t": period})
    return cross_scores(
        pl.DataFrame(rows),
        score_x="x",
        score_y="y",
        score_directions={"x": "lower_risk", "y": "higher_risk"},
        cutpoints={"x": [1.5, 2.5, 3.5, 4.5], "y": [1.5, 2.5, 3.5, 4.5]},
        special_values={"x": [-99.0]},
        targets=["bad", "late"],
        group_col="g",
        time_col="t",
        weights_col="w",
        amount_col="a",
        min_observed=2,
    )


def test_expression_replay_uses_stable_ranks_real_weighted_scope_and_persists(tmp_path: Path) -> None:
    report = _weighted_report()
    before = deepcopy(report.describe())
    rule = {"type": "expression", "expression": _score_rule_examples(5, 5)[0]["expression"]}
    policy = report.evaluate_policy(rule)
    scope = {"target": "bad", "group": "开发<&", "period": "202601"}
    result = policy.get_table("summary", filters={**scope, "rule": "candidate", "retained": True}).row(0, named=True)
    # 高分低风险 X1 实际为原始 x=5；三段按真实原始数据给出 5+2+1 个人。
    expected = [(5, y) for y in range(1, 6)] + [(4, 1), (4, 2), (3, 1)]
    weight = sum(x * y for x, y in expected)
    bad_weight = sum(x * y for x, y in expected if (x + y) % 2)
    assert result["sample_count"] == result["observed_sample_count"] == 8
    assert result["bad_sample_count"] == sum((x + y) % 2 for x, y in expected)
    assert result["weight_sum"] == result["observed_weight_sum"] == weight
    assert result["bad_weight_sum"] == bad_weight
    assert result["tot_amt"] == sum(10 * y for _, y in expected)
    assert result["bad_rate"] == pytest.approx(bad_weight / weight)
    assert result["sample_share"] == pytest.approx(8 / 29)
    overall = report.get_table("overall", filters=scope).row(0, named=True)
    assert result["lift_vs_overall"] == pytest.approx(result["bad_rate"] / overall["bad_rate"])
    assert result["weighted_ci_status"] == "unsupported" and result["lift_status"] == "valid"
    decisions = policy.get_table("cell_decisions", filters={**scope, "candidate_pass": True})
    assert decisions.height == 8 and decisions["sample_count"].sum() == 8
    for target in ("bad", "late"):
        for group, period in (("开发<&", "202601"), ("回看", "202602")):
            selected = policy.get_table("summary", filters={"target": target, "group": group, "period": period, "rule": "candidate", "retained": True}).row(0, named=True)
            assert selected["sample_count"] == 8
            assert selected["observed_sample_count"] == (8 if target == "bad" else 7)
            regions = policy.get_table("regions", filters={"target": target, "group": group, "period": period})
            assert regions.height == 4 and regions["sample_count"].sum() == 29
    assert report.describe() == before
    source = tmp_path / "cross.marsreport"
    report.save(source)
    snapshot = load_report(source)
    assert snapshot.report_id == report.report_id
    restored = evaluate_score_policy(snapshot, rule)
    for table in policy.describe()["tables"]:
        assert_frame_equal(policy.get_table(table), restored.get_table(table))
    policy_path = tmp_path / "expression-policy.marsreport"
    policy.save(policy_path)
    saved = load_report(policy_path)
    parameters = saved.describe()["parameters"]
    assert parameters["parent_report_id"] == report.report_id
    assert parameters["candidate_expression_ast"] == _parse_score_expression(rule["expression"], 5, 5)
    assert parameters["expression_syntax"]["normal_bins_only"] is True
    assert evaluate_score_policy(snapshot, parameters["candidate"]).get_table("summary").equals(policy.get_table("summary"))
    page = saved.query_page("summary", filters={**scope, "rule": "candidate", "retained": True}, columns=["sample_count", "bad_rate", "lift_vs_overall", "lift_status"], limit=1)
    assert page["data"]["lift_vs_overall"][0] == result["lift_vs_overall"]
    assert page["reference"]["report_id"] == saved.report_id


@pytest.mark.skipif(sys.version_info < (3, 10), reason="mars.agent 支持 Python 3.10 及以上。")
def test_saved_expression_policy_is_independently_queryable_by_agent(tmp_path: Path) -> None:
    from mars.agent import MarsAgentSession, MarsRiskAgent

    report = _weighted_report()
    policy = report.evaluate_policy({"type": "expression", "expression": "X <= X2 AND Y <= Y3"})
    path = tmp_path / "agent-policy.marsreport"
    policy.save(path)
    saved = load_report(path)
    scope = {"target": "bad", "group": "开发<&", "period": "202601"}
    page = saved.query_page("summary", filters={**scope, "rule": "candidate", "retained": True}, columns=["sample_count", "bad_rate", "lift_vs_overall", "lift_status"], limit=1)
    session = MarsAgentSession()
    handle = session.register_report(saved)
    query = MarsRiskAgent().execute_tool(
        "get_report_table",
        {"report_id": handle, "table": "summary", "filters": {**scope, "rule": "candidate", "retained": True}, "columns": ["sample_count", "bad_rate", "lift_vs_overall", "lift_status"]},
        session=session,
    )
    assert query.success and query.data["rows"] == page["data"].to_dicts()


def test_expression_replays_prior_snapshot_without_new_display_or_lift_fields(tmp_path: Path) -> None:
    report = _weighted_report()
    old_tables = {
        name: report.get_table(name).drop([column for column in ("display_label", "overall_status", "lift_status") if column in report.get_table(name).columns])
        for name in report.describe()["tables"]
    }
    description = report.describe()
    for name, table in old_tables.items():
        description["tables"][name]["fields"] = {column: definition for column, definition in description["tables"][name]["fields"].items() if column in table.columns}
    old = ReportSnapshot(old_tables, description)
    old.save(tmp_path / "prior.marsreport")
    saved = load_report(tmp_path / "prior.marsreport")
    candidate = {"type": "expression", "expression": "X>=X4 OR Y=Y5"}
    baseline = {"type": "expression", "expression": "X=X5"}
    expected = evaluate_score_policy(report, candidate, baseline=baseline)
    actual = evaluate_score_policy(saved, candidate, baseline=baseline)
    assert_frame_equal(actual.get_table("summary"), expected.get_table("summary"))
    assert_frame_equal(actual.get_table("regions"), expected.get_table("regions"))
    assert actual.describe()["parameters"]["baseline_expression_ast"] == _parse_score_expression("X=X5", 5, 5)
    description_before = actual.describe()
    with pytest.raises(ValueError, match="轴不匹配"):
        evaluate_score_policy(saved, {"type": "expression", "expression": "X <= Y2"})
    assert actual.describe() == description_before


@pytest.mark.parametrize(("target", "weights", "expression", "status", "lift"), [
    ([0, 1], [1.0, 3.0], "X=X1", "valid", 0.0),
    ([0, 0], [1.0, 3.0], "X=X1", "invalid_denominator", None),
    ([None, 1], [1.0, 3.0], "X=X1", "unobserved", None),
    ([0, 1], [0.0, 3.0], "X=X1", "invalid_denominator", None),
    ([0, 1], [1.0, 3.0], "X>X2", "empty", None),
])
def test_policy_lift_zero_missing_zero_weight_and_no_matches(
    target: list[int | None], weights: list[float], expression: str, status: str, lift: float | None
) -> None:
    report = cross_scores(
        pl.DataFrame({"x": [1, 4], "y": [1, 4], "bad": target, "w": weights}),
        score_x="x", score_y="y", score_directions={"x": "higher_risk", "y": "higher_risk"},
        cutpoints={"x": [2], "y": [2]}, targets=["bad"], weights_col="w", min_observed=1,
    )
    policy = evaluate_score_policy(report, {"type": "expression", "expression": expression})
    row = policy.get_table("summary", filters={"rule": "candidate", "retained": True}).row(0, named=True)
    assert row["lift_status"] == status
    assert row["lift_vs_overall"] == lift
    if expression == "X=X1":
        assert report.get_table("cells", filters={"x_bin": "b0", "y_bin": "b0"})["lift_status"][0] == status


def test_expression_empty_unobserved_distribution_and_explicit_special_boundaries() -> None:
    report = cross_scores(
        pl.DataFrame({"x": [1, None], "y": [1, 1]}), score_x="x", score_y="y",
        score_directions={"x": "higher_risk", "y": "higher_risk"}, cutpoints={"x": [2], "y": [2]},
    )
    replay = evaluate_score_policy(report, {"type": "expression", "expression": "X=X2"})
    row = replay.get_table("summary", filters={"rule": "candidate", "retained": True}).row(0, named=True)
    assert row["sample_count"] == 0 and row["sample_status"] == "empty"
    assert row["status"] == row["lift_status"] == "not_requested"
    assert row["bad_rate"] is None and row["observed_sample_count"] is None
    explicit = evaluate_score_policy(report, {"type": "x_only", "x_max_risk_rank": 1, "accepted_special_bins": {"x": ["missing"]}})
    assert explicit.get_table("summary", filters={"rule": "candidate", "retained": True})["sample_count"][0] == 2
    with pytest.raises(ValueError, match="特殊箱"):
        evaluate_score_policy(report, {"type": "expression", "expression": "X=X1", "accepted_special_bins": {"x": ["missing"]}})


def test_python_and_offline_javascript_parser_and_interpreter_parity(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js 不可用，未运行 Python/离线 JS 差分验证。")
    expressions = [
        "x <= x2 and y <= 3", "X=1 OR X=2 AND Y=1", "(X=1 OR X=2) AND Y=1",
        "X!=2 OR Y>=4", "X<2 OR X>4",
        *[example["expression"] for example in _score_rule_examples(5, 5)],
        "X<=Y2", "X X2", "X<=1.5", "X=0", "X=1)", "(X=1", "X=1;alert(1)",
        "(" * 13 + "X=1" + ")" * 13, "X=1 " * 33, "X=1" + " " * 238,
    ]
    script = tmp_path / "expression-parity.cjs"
    script.write_text(
        _EXPRESSION_JAVASCRIPT + "\nconst expressions=" + json.dumps(expressions) + ";\n"
        "process.stdout.write(JSON.stringify(expressions.map(text=>{try{const ast=parseScoreExpression(text,5,5);const matches=[];for(let x=1;x<=5;x++)for(let y=1;y<=5;y++)if(evaluateScoreExpression(ast,x,y))matches.push([x,y]);return {ast,matches};}catch(error){return {error:error.message};}})));",
        encoding="utf-8",
    )
    result = subprocess.run([node, str(script)], capture_output=True, text=True, encoding="utf-8", check=True)
    js_results = json.loads(result.stdout)
    for expression, actual in zip(expressions, js_results):
        try:
            ast = _parse_score_expression(expression, 5, 5)
        except ValueError as error:
            # 非法词元的引号格式不是语法契约，比较准确错误类别和其余错误文本。
            assert "error" in actual
            if "非法词元" in str(error):
                assert "非法词元" in actual["error"]
            else:
                assert actual["error"] == str(error)
        else:
            assert actual["ast"] == ast
            assert actual["matches"] == [[x, y] for x in range(1, 6) for y in range(1, 6) if _evaluate_score_expression(ast, x, y)]
