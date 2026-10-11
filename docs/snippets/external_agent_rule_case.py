"""Home Credit 外部来源分、固定规则与报告快照的独立消费；不调用 LLM。"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import polars as pl
from home_credit_data import DEFAULT_DATA_DIR, HomeCreditInputs, prepare_home_credit_data

from mars.analysis import cross_scores, get_score_bin_definitions
from mars.compute import missing_condition_expr
from mars.reporting import Report, load_report
from mars.reporting._serialization import json_safe
from mars.rule import (
    MarsRule,
    MarsRuleEvaluator,
    MarsRuleMiningSpec,
    MarsRuleReport,
    MarsRuleSet,
    mine_rules,
)

QUESTION = "同一外部来源分 2 等级内，来源分 3 的风险有何差异？固定组合规则能否通过独立复核？"
DIRECTIONS = {"EXT_SOURCE_2": "lower_risk", "EXT_SOURCE_3": "lower_risk"}


def _write_json(path: Path, value: Any) -> None:
    """只写有限聚合统计，不写原始记录或成员 ID。"""
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _write_agent_task(output_dir: Path) -> None:
    """让计算和快照消费共同刷新完整规则审计任务。"""
    prompt = (
        "你是具备 Python 工具的外部分析消费者。先 load_report，再 describe，按 query_page 分页取证。\n"
        "读取 score-cross.marsreport、rules.marsreport、rules-independent.marsreport；不访问原始 CSV。\n"
        "先核对 describe()['tables'] 中实际表名、字段、粒度和状态，再读取 parameters 中实际配置。\n"
        "EXT_SOURCE_2/3 是公开外部来源分；Train 决定切点与候选，Val 参与选择。\n"
        "在 rules.marsreport 查询 candidates 表的 rule_id、status、rejection_stage、reason，"
        "逐条解释实际保留或拒绝阶段；再查询 rules 表核对最终入选规则及表达式。\n"
        "本批验证指标保存在 evaluation 表，不存在名为 validation 的独立表。"
        "用 query_page('evaluation', filters={'dataset': 'validation', 'target': 'TARGET', 'group': 'hit'}, limit=8) "
        "读取 1val 的 sample_count、event_count、coverage、event_rate、lift、置信区间和 q_value；"
        "dataset='validation' 表示参与选择的 Val，并非最终独立 Test/OOT。\n"
        "读取 summary 与 parameters.resolved_spec 的 candidate_filter、validation_filter、max_fdr 等真实门槛，"
        "结合 candidates 的实际拒绝原因作判断；不得把未执行的门禁写成通过。"
        "temporal_assessed 为 false，本批没有真实跨期验证。\n"
        "在 score-cross.marsreport 的 cells 查询 Train 格子人数、风险与切点定义，解释候选的实际来源；"
        "规则筛选之后查询 rules-independent.marsreport 的 slices 表，"
        "用 slice='2test'/'3oot'、group='hit' 获取固定规则独立证据。\n"
        "在同一 slices 表查询 slice='4_oot_undue' 的分布、total_count、observed_count、"
        "unobserved_count 和 status；监督指标 null 是未观测，不能改写成零坏率。\n"
        "query_page 若返回 next_offset，将它作为下一页 offset 继续分页；引用每条结论的实际 reference，"
        "区分 total_count 与 sample_count（已观测人数），报告表缺失或未计算时明确指出。\n"
        "APPLY_DATE 是模拟日期；不能证明真实跨期泛化。币种、损失、利润、因果效果未提供。\n"
        "to_ai_context 的 max_chars 是 Unicode 字符预算，不是 token。无需内置 Agent 或 SDK。\n"
    )
    for name in ("external-agent-task.md", "external-agent-rule-task.txt"):
        (output_dir / name).write_text(prompt, encoding="utf-8")


def _rows(frame: Any) -> list[dict[str, Any]]:
    """只物化已受限的报告查询。"""
    return json_safe(frame.to_dicts() if isinstance(frame, pl.DataFrame)
                     else frame.to_dict("records"))


def _condition(feature: str, definition: dict[str, Any], index: int) -> str:
    """按真实定义写右闭正常箱；排除缺失、特殊值和非有限数。"""
    cuts = definition["cutpoints"]
    identifier = '"' + feature.replace('"', '""') + '"'
    parts = [f"{identifier} IS NOT MISSING", f"{identifier} >= {-sys.float_info.max!r}",
             f"{identifier} <= {sys.float_info.max!r}"]
    excluded = [*definition.get("special_values", []), *definition.get("missing_values", [])]
    for value in excluded:
        if isinstance(value, (int, float)) and math.isfinite(value):
            parts.append(f"{identifier} != {value!r}")
    if definition.get("probability"):
        parts.extend([f"{identifier} >= 0", f"{identifier} <= 1"])
    if index:
        parts.append(f"{identifier} > {cuts[index - 1]!r}")
    if index < len(cuts):
        parts.append(f"{identifier} <= {cuts[index]!r}")
    return " AND ".join(parts)


def _check_cell_members(
    data: pl.DataFrame, rule: MarsRule, definitions: dict[str, Any], cell: dict[str, Any],
) -> dict[str, Any]:
    """在内存核对正常箱的精确成员，仅公开汇总，不保存 SK_ID_CURR。"""
    expected = pl.lit(True)
    for axis in ("x", "y"):
        definition = definitions[axis]
        feature = definition["score"]
        values = pl.col(feature)
        normal = values.is_finite() & ~missing_condition_expr(
            feature, dtype=data.schema[feature], missing_values=definition.get("missing_values"),
        )
        if definition["special_values"]:
            normal &= ~values.is_in(definition["special_values"])
        if definition["probability"]:
            normal &= values.is_between(0, 1, closed="both")
        index = int(cell[f"{axis}_bin"][1:])
        cuts = definition["cutpoints"]
        if index:
            normal &= values > cuts[index - 1]
        if index < len(cuts):
            normal &= values <= cuts[index]
        expected &= normal
    actual = MarsRuleSet([rule]).transform(data)
    expected_ids = set(data.filter(expected)["SK_ID_CURR"])
    actual_ids = set(actual.filter(pl.col(f"rule__{rule.rule_id}") == 1)["SK_ID_CURR"])
    if expected_ids != actual_ids:
        raise AssertionError(f"正常交叉箱与规则 {rule.rule_id} 成员不同。")
    return {
        "verified": True, "scope": "all supplied rows; exact SK_ID_CURR membership; IDs not saved",
        "member_count": len(expected_ids),
        "members_by_dataset": data.filter(expected).group_by("dataset_flag").agg(
            pl.len().alias("total_count"), pl.col("TARGET").count().alias("observed_count"),
        ).sort("dataset_flag").to_dicts(),
    }


def _page(report: Report, table: str, question: str, **query: Any) -> dict[str, Any]:
    """保存真实分页内容和可重放引用。"""
    page = report.query_page(table, **query)
    return {"question": question, "reference": page["reference"],
            "total_rows": page["total_rows"], "returned_rows": page["returned_rows"],
            "omitted_rows": page["omitted_rows"], "next_offset": page["next_offset"],
            "rows": _rows(page["data"])}


def produce(
    output_dir: Path, *, inputs: HomeCreditInputs | None = None,
    cross_report: Report | None = None,
) -> None:
    """在 Train 发现、Val 筛选规则，再用固定 RuleSet 独立评估全部分区。

    Parameters
    ----------
    output_dir : Path
        产物目录，自动创建；只保存聚合报告和规则定义。
    inputs : HomeCreditInputs | None
        同批共享准备结果；None 时按公共默认数据目录全量读取三文件。
    cross_report : Report | None
        同批 Train 固定切点的五分区报告；None 时使用真实公开入口计算。

    Returns
    -------
    None
        写入规则、独立评估报告、可重放证据和外部消费说明。

    Examples
    --------
    >>> produce(Path("output/agent-rule-case"))  # doctest: +SKIP
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_home_credit_data() if inputs is None else inputs
    data = prepared.df
    train = prepared.benchmark_df
    validation = data.filter(pl.col("dataset_flag") == "1val")
    context: dict[str, Any] = {
        **prepared.business_context, "question": QUESTION, "score_directions": DIRECTIONS,
        "selection_scope": "0train discovery; 1val selection; 2test/3oot untouched until fixed evaluation",
        "binning_origin": "0train only; fixed quantile cutpoints",
    }
    if cross_report is None:
        initial = cross_scores(train, score_x="EXT_SOURCE_2", score_y="EXT_SOURCE_3",
                               targets=["TARGET"], score_directions=DIRECTIONS, n_bins=4,
                               feature_metadata=prepared.feature_metadata, business_context=context)
        cross_report = cross_scores(
            data, score_x="EXT_SOURCE_2", score_y="EXT_SOURCE_3", targets=["TARGET"],
            score_directions=DIRECTIONS, bin_definitions=get_score_bin_definitions(initial),
            group_col="dataset_flag", amount_col="AMT_CREDIT",
            feature_metadata=prepared.feature_metadata, business_context=context,
        )
    cross_report.save(output_dir / "score-cross.marsreport", overwrite=True)
    definitions = get_score_bin_definitions(cross_report)
    train_scope: dict[str, Any] = {"group": "0train", "target": "TARGET"}
    cells = cross_report.get_table(
        "cells", filters={**train_scope, "x_bin": {"op": "in", "value": ["b0", "b1", "b2", "b3"]},
                          "y_bin": {"op": "in", "value": ["b0", "b1", "b2", "b3"]}},
        sort_by="lift_vs_overall", descending=True,
    )
    # 候选只取 Train 真实格子；低 Lift 格子作为反证，不人为造空覆盖规则。
    picks = [*cells.head(3).to_dicts(), cells.tail(1).row(0, named=True)]
    seeds: list[MarsRule] = []
    origins: dict[str, Any] = {}
    for cell in picks:
        expression = (
            f"({_condition('EXT_SOURCE_2', definitions['x'], int(cell['x_bin'][1:]))}) AND "
            f"({_condition('EXT_SOURCE_3', definitions['y'], int(cell['y_bin'][1:]))})"
        )
        rule = MarsRule(expression, source="cross_train")
        if rule.rule_id in origins:
            continue
        seeds.append(rule)
        origins[rule.rule_id] = {
            "kind": "development_cell", "region": {"x_bin": cell["x_bin"], "y_bin": cell["y_bin"]},
            "member_check": _check_cell_members(data, rule, definitions, cell),
            "reference": cross_report.query_page(
                "cells", filters={**train_scope, "period": cell["period"],
                                  "x_bin": cell["x_bin"], "y_bin": cell["y_bin"]}, limit=1,
            )["reference"],
        }
    result = mine_rules(
        train, target="TARGET", validation_df=validation, features=list(DIRECTIONS),
        seed_rules=seeds, generators=[], amount_col="AMT_CREDIT",
        spec=MarsRuleMiningSpec.production(),
    )
    report = result.to_report(feature_metadata=prepared.feature_metadata, business_context={
        **context, "candidate_origins": origins,
        "validation_role": "1val participates in selection; API dataset='validation' is not final independent test",
        "analysis_scope": "production selection; no bootstrap/interactions computed",
    })
    report.save(output_dir / "rules.marsreport", overwrite=True)
    report.write_html(output_dir / "rules.html")
    report.write_excel(output_dir / "rules.xlsx")
    result.rule_set.save_json(output_dir / "ruleset.json")
    # 独立复核不调用 mine_rules、不重选；固定 RuleSet 对五分区一次混合评估。
    independent = MarsRuleEvaluator().evaluate(
        data, result.rule_set, target="TARGET", dataset="fixed_all_partitions",
        group_col="dataset_flag", amount_col="AMT_CREDIT", compute_statistics=True,
    )
    fixed_report = MarsRuleReport(
        summary_table=pl.DataFrame({"selected_count": [len(result.rule_set.rules)],
                                    "status": ["evaluated" if result.rule_set.rules else "no_rules"]}),
        detail_tables={"rules": report.get_table("rules"), "evaluation": independent.overall_table,
                       "slices": independent.slice_table},
        metadata={**dict(independent.metadata), "features": list(DIRECTIONS),
                  "selection_report_id": report.report_id, "fixed_rule_set": True},
        feature_metadata=prepared.feature_metadata, business_context={
            **context, "evaluation_roles": "2test/3oot independent; 4_oot_undue distribution only",
        }, caption="Home Credit 固定规则的独立复核",
    )
    fixed_report.save(output_dir / "rules-independent.marsreport", overwrite=True)
    fixed_report.write_html(output_dir / "rules-independent.html")
    fixed_report.write_excel(output_dir / "rules-independent.xlsx")
    _write_json(output_dir / "case-notes.json", {
        "question": QUESTION, "real_features": True, "simulated_dates": True,
        "independent_rows": True, "raw_data_saved": False,
        "partition_stats": prepared.summary["partition_stats"],
        "selection_report_id": report.report_id, "independent_report_id": fixed_report.report_id,
        "agent_execution": "deterministic acceptance; no LLM",
    })
    _write_agent_task(output_dir)


def consume(output_dir: Path) -> dict[str, Any]:
    """只消费既有快照，在新进程复核引用，不调用原始数据或分析函数。

    Parameters
    ----------
    output_dir : Path
        含交叉、规则与独立评估快照的目录。

    Returns
    -------
    dict[str, Any]
        有限查询证据与可回答边界，另写 JSON。

    Examples
    --------
    >>> consume(Path("output/agent-rule-case"))  # doctest: +SKIP
    """
    cross = load_report(output_dir / "score-cross.marsreport")
    rules = load_report(output_dir / "rules.marsreport")
    fixed = load_report(output_dir / "rules-independent.marsreport")
    _write_agent_task(output_dir)
    trace = [
        _page(cross, "cells", "Train 同一来源分 2 箱内的来源分 3 风险与人数",
              filters={"group": "0train", "target": "TARGET", "x_bin": "b1"},
              sort_by="y_risk_rank", limit=8),
        _page(rules, "candidates", "实际选择/拒绝阶段与原因", limit=8),
        _page(fixed, "slices", "固定规则的独立 test/oot 复核",
              filters={"slice": {"op": "in", "value": ["2test", "3oot"]}, "group": "hit"}, limit=8),
        _page(fixed, "slices", "无标签组保留全部分布并显示未观测状态",
              filters={"slice": "4_oot_undue", "group": "hit"}, limit=8),
    ]
    reports = {r.report_id: r for r in (cross, rules, fixed)}
    for entry in trace:
        ref = entry["reference"]
        assert _rows(reports[ref["report_id"]].get_table(ref["table"], **ref["query"])) == entry["rows"]
    context = rules.to_ai_context(queries={"summary": {"limit": 1}, "candidates": {"limit": 3}}, max_chars=12000)
    assert len(context) <= 12000
    (output_dir / "rule-agent-context.json").write_text(context + "\n", encoding="utf-8")
    unavailable = {"question": "规则的真实线上收益与真实时间外泛化如何？", "answer": "当前报告无法回答",
                   "reason": "没有真实申请日期、损失利润、线上或因果验证。"}
    review: dict[str, Any] = {
        "kind": "deterministic_evidence_review", "question": QUESTION,
        "real_features": True, "simulated_dates": True,
        "selection": trace[1], "independent_validation": trace[2], "unobserved": trace[3],
        "findings": trace[0]["rows"], "missing_information": unavailable,
    }
    _write_json(output_dir / "query-trace.json", {"consumer": "deterministic public queries; no LLM",
                                                "queries": trace, "unavailable": unavailable})
    _write_json(output_dir / "evidence-review.json", review)
    (output_dir / "review.md").write_text(
        "# Home Credit 确定性证据复核\n\n" + QUESTION + "\n\n"
        "实际分页内容与引用见 query-trace.json；Val 参与筛选，独立证据来自固定规则的 test/oot。\n\n"
        "July 无标签分布保留；模拟日期不能证明真实跨期表现。\n", encoding="utf-8",
    )
    return review


def main() -> None:
    """复用共享数据准备；消费参数先于准备路径，保证无需 CSV。"""
    parser = argparse.ArgumentParser(description=QUESTION)
    parser.add_argument("--output-dir", type=Path, default=Path("output/agent-rule-case"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--columns-description-csv", type=Path)
    parser.add_argument("--consume-only", action="store_true")
    args = parser.parse_args()
    if args.consume_only:
        consume(args.output_dir)
        return
    inputs = prepare_home_credit_data(args.data_dir, train_csv=args.train_csv, test_csv=args.test_csv,
                                     columns_description_csv=args.columns_description_csv)
    produce(args.output_dir, inputs=inputs)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--consume-only",
                    "--output-dir", str(args.output_dir.resolve())], check=True)


if __name__ == "__main__":
    main()
