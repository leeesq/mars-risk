"""规则证据目录与桥接定义；指标仍保留原表粒度。"""

from __future__ import annotations

from typing import Any

import polars as pl

from mars.reporting._artifact import ReportSnapshot
from mars.reporting._metadata import FeatureMetadata
from mars.reporting._result import _result_report
from mars.rule._dsl import expression_features, parse_expression

_GRAINS = {
    "summary": "one mining run (or benchmark summary); no feature-level metrics",
    "rules": "final rule_id / selection rank; grades are rule membership, not metrics",
    "candidates": "candidate rule_id / generation_round when present; generator audit",
    "evaluation": "dataset / rule_id / target / slice / group (hit, miss, total)",
    "slices": "dataset / rule_id / target / slice / group; slice encodes supplied time or segment",
    "rule_explanations": "final rule_id / rank / dataset / target / slice / group",
    "rule_features": "distinct rule_id / English feature; includes rejected candidates",
    "cumulative_features": "added_rule_id / feature of any rule in the ordered prefix",
    "interactions": "ordered rule_a / rule_b pair on explicitly supplied analysis population",
    "cumulative": "rank / added_rule_id; ordered union and new marginal hits",
    "bootstrap": "rule_id on explicitly supplied analysis population; resampling interval",
    "benchmark": "one supplied benchmark record; no rule qualification or rule evidence",
}
_FIELDS = {
    "sample_count": ("count", "当前 target 已表现样本数；group/slice/dataset 定义统计范围"),
    "candidate_count": (
        "count",
        "候选审计行数；cascade 包含同一 rule_id 在不同轮次的审计，不是去重规则数",
    ),
    "selected_count": ("count", "最终 RuleSet 中入选规则数；no_rules 为 0"),
    "event_count": ("count", "当前范围已表现 label=1 样本数"),
    "coverage": (
        "ratio",
        "当前范围 sample_count / 同 dataset、target、slice 的 base sample_count；只含已表现样本",
    ),
    "event_rate": ("ratio", "event_count / sample_count；零分母 null"),
    "lift": (
        "dimensionless",
        "event_rate / 同 dataset、target、slice 的 base event_rate；零分母 null",
    ),
    "amount_total": (
        "unknown_business_unit",
        "当前范围已表现样本金金额和；amount_col 未提供则 null；币种见 business_context",
    ),
    "event_amount": (
        "unknown_business_unit",
        "当前范围已表现 label=1 金额和；amount_col 未提供则 null",
    ),
    "amount_coverage": ("ratio", "当前 amount_total / 同范围 base amount_total"),
    "amount_event_rate": ("ratio", "event_amount / amount_total；零分母 null"),
    "amount_lift": ("dimensionless", "amount_event_rate / 同范围 base amount_event_rate"),
    "customer_count": ("count", "当前已表现范围去重客户数；customer_col 未提供则 null"),
    "event_customer_count": ("count", "当前已表现 label=1 范围去重客户数"),
    "customer_coverage": ("ratio", "customer_count / 同范围 base customer_count；不可跨规则相加"),
    "customer_event_rate": ("ratio", "event_customer_count / customer_count；不可跨规则相加"),
    "customer_lift": ("dimensionless", "customer_event_rate / 同范围 base customer_event_rate"),
    "event_rate_ci_lower": ("ratio", "Wilson 事件率区间下界；confidence_level 见实际参数"),
    "event_rate_ci_upper": ("ratio", "Wilson 事件率区间上界；confidence_level 见实际参数"),
    "lift_ci_lower": ("dimensionless", "规则 Wilson 事件率下界 / 同范围总体事件率；保守 Lift 下界"),
    "lift_ci_upper": ("dimensionless", "规则 Wilson 事件率上界 / 同范围总体事件率；保守 Lift 上界"),
    "p_value": ("probability", "按 direction 的单侧事件富集检验 p 值；方法见 evaluator"),
    "q_value": ("probability", "同数据范围、target、slice、group 候选检验族 BH 校正 q 值"),
    "iou": ("ratio", "已表现交集样本数 / 两规则已表现并集样本数"),
    "union_count": ("count", "两端规则已表现命中的并集人数"),
    "combo_gain_lift": ("dimensionless", "intersection Lift 减两端最大 Lift"),
    "rank": ("ordinal", "最终规则集的 1 起始顺序；cumulative 使用此前全部规则的并集"),
    "rule_id": ("identifier", "规范 DSL 的稳定哈希 ID；同 ID 可出现在不同 dataset、target 或轮次"),
    "added_rule_id": ("identifier", "累计顺序本步新增规则 ID；累计指标还包含此前规则"),
    "rule_a": ("identifier", "规则交互左端点 ID；与 rule_b 共同确定规则对"),
    "rule_b": ("identifier", "规则交互右端点 ID；与 rule_a 共同确定规则对"),
    "expression": ("definition", "既有 DSL 规范表达式；不执行报告文本"),
    "feature": ("identifier", "AST 提取的英文输入特征 ID；中文名不替代此身份"),
    "sources": ("category", "候选生成器／seed 来源；不是 feature_metadata.data_source"),
    "source": ("category", "规则生成来源；不是业务特征来源"),
    "dataset": (
        "dimension",
        "实际保存的 train/validation/in_sample 范围；不推断未保存的逐轮剩余样本统计",
    ),
    "target": ("identifier", "真实标签列 ID；各 target 独立排除未表现标签"),
    "slice": ("dimension", "__overall__ 或 evaluator 实际构造的时间／客群切片值"),
    "group": ("dimension", "hit / miss / total；此列不是原始客群，客群见 slice"),
    "status": (
        "state",
        "summary success/no_rules；候选 selected/rejected/candidate/deferred；结合 reason 和 rejection_stage",
    ),
    "qualification": ("state", "计算时 RuleSet 资格；快照恢复不会重建部署资格"),
    "validation_status": ("state", "independent 或 in_sample；验证样本独立性由调用方保证"),
    "profile": ("category", "实际 explore / production 门禁；报告不改阈值"),
    "generation_round": ("ordinal", "候选生成轮次；相同规则在不同轮次的审计不可合并"),
    "selection_round": ("ordinal", "实际 cascade 入选轮次；未入选为 null"),
    "selection_rank": ("ordinal", "实际选择顺序；未入选为 null"),
    "grades": ("category", "既有 RuleSet 等级成员列表"),
    "explanation": ("text", "当前行所引用验证范围的中文说明；缺指标显示 N/A"),
    "reason": ("state_reason", "真实候选淘汰原因；错误不转换为 no_rules"),
    "rejection_stage": ("state_reason", "实际淘汰门禁阶段"),
    "seconds": ("seconds", "调用方提供的 benchmark 耗时；不是本次报告实测"),
}


def _field_definition(column: str, name: str, dtype: Any) -> tuple[str, str]:
    """按规则真实字段及高级分析前缀映射单位，未知业务口径不推断。"""
    metric = column
    for prefix in ("rule_a_", "rule_b_", "intersection_", "cumulative_", "marginal_"):
        if metric.startswith(prefix):
            metric = metric[len(prefix) :]
            break
    if metric in _FIELDS:
        return _FIELDS[metric]
    if column.startswith("lift_bootstrap_"):
        return "dimensionless", "实际 bootstrap Lift 分位数界；配置及分析范围见 advanced_analysis"
    if column.endswith("_count") or column in {"complexity", "repeat_count", "benchmark_rows"}:
        return "count", f"实际 {column} 整数计数；粒度见本表 grain"
    if column.endswith("_passed") or column.startswith("within_") or dtype == pl.Boolean:
        return "boolean", f"实际 {column} 门禁结果；null 表示未执行或不适用"
    if column.endswith("_position"):
        return "ordinal", f"实际 {column} 候选顺序；null 表示不适用"
    if column.endswith("pass_rate"):
        return "ratio", "通过时间切片数 / 已评估时间切片数"
    if name == "benchmark" and (
        "memory" in column or "rss" in column or column.endswith(("_mb", "_bytes"))
    ):
        return "bytes" if column.endswith("bytes") else "MB" if column.endswith(
            "mb"
        ) else "unknown", "调用方提供的 benchmark 内存；仅显式后缀声明单位"
    return (
        "category" if dtype == pl.String else "unknown",
        f"已保存的 {column}；未登记业务含义保持 unknown",
    )


def _rule_snapshot(
    summary: pl.DataFrame,
    details: dict[str, pl.DataFrame],
    parameters: dict[str, Any],
    metadata: FeatureMetadata | None,
    context: dict[str, Any] | None,
) -> ReportSnapshot:
    """从已有统计和 DSL 定义构造公共快照，不访问原始样本或运行分析。"""
    benchmark = parameters.get("report_type") == "benchmark"
    tables = {"summary": summary, **details}
    # 合法零行仍有领域身份字段，消费者可以正常查询空结果；不伪造指标或行。
    empty_dimensions = {
        "candidates": ["rule_id", "expression", "sources", "status"],
        "evaluation": ["dataset", "rule_id", "target", "slice", "group"],
        "slices": ["dataset", "rule_id", "target", "slice", "group"],
        "interactions": ["rule_a", "rule_b"],
        "cumulative": ["added_rule_id"],
        "bootstrap": ["rule_id"],
    }
    for name, columns in empty_dimensions.items():
        if name in tables and not tables[name].columns:
            tables[name] = pl.DataFrame(schema={column: pl.String for column in columns})
    pairs: set[tuple[str, str]] = set()
    for name in ("rules", "candidates", "rule_explanations"):
        frame = tables.get(name, pl.DataFrame())
        if {"rule_id", "expression"}.issubset(frame.columns):
            for rule_id, expression in frame.select("rule_id", "expression").unique().iter_rows():
                pairs.update(
                    (rule_id, f) for f in expression_features(parse_expression(expression))
                )
    features = list(
        dict.fromkeys([*parameters.get("features", []), *sorted({f for _, f in pairs})])
    )
    if not benchmark:
        tables["rule_features"] = pl.DataFrame(
            sorted(pairs), schema={"rule_id": pl.String, "feature": pl.String}, orient="row"
        )
        cumulative = tables.get("cumulative", pl.DataFrame())
        if "added_rule_id" in cumulative.columns:
            prefix_features: set[str] = set()
            prefix_pairs: list[tuple[str, str]] = []
            for rule_id in cumulative["added_rule_id"]:
                prefix_features.update(f for r, f in pairs if r == rule_id)
                prefix_pairs.extend((rule_id, f) for f in sorted(prefix_features))
            tables["cumulative_features"] = pl.DataFrame(
                prefix_pairs, schema={"rule_id": pl.String, "feature": pl.String}, orient="row"
            )
    snapshot = _result_report(
        "rule_benchmark" if benchmark else "rule",
        tables,
        parameters,
        features,
        metadata,
        context,
        grains=_GRAINS,
    )
    description = snapshot.describe()
    description["source"].update(
        producer="mars.rule.MarsRuleReport",
        mars_version=parameters.get("mars_version", "unknown"),
        origin_project=parameters.get("source_project", "unknown"),
    )
    description["maturity"] = "Experimental"
    if "rule_explanations" in tables:
        preferred = [
            "rank",
            "rule_id",
            "dataset",
            "target",
            "slice",
            "group",
            "sample_count",
            "event_count",
            "event_rate",
            "lift",
        ]
        description["default_ai_queries"] = {
            "summary": {"limit": 1},
            "rule_explanations": {
                "limit": 3,
                "columns": [c for c in preferred if c in tables["rule_explanations"].columns],
            },
        }
    description["limitations"].extend(
        [
            "Feature matching uses rule membership union; features AND sources may match different features of the same rule. sources is business data_source, not candidate generators.",
            "summary and benchmark do not support features/sources queries. Advanced analysis scope is advanced_analysis; no dataset identity is inferred for supplied analysis samples.",
            "Absent analyses are not computed; present zero-row tables are computed_empty unless explicitly not_computed. Null may be undefined or unconfigured; inspect parameters and denominators.",
            "Snapshot is aggregate evidence, not RuleSet JSON, transform capability or deployment permission. No original samples or models are saved.",
        ]
    )
    for name, frame in tables.items():
        entry = description["tables"][name]
        entry["state"] = "computed" if frame.height else "computed_empty"
        if name in parameters.get("analysis_states", {}):
            entry["state"] = parameters["analysis_states"][name]
        if name in {"interactions", "cumulative", "bootstrap", "cumulative_features"}:
            entry["scope"] = parameters.get("advanced_analysis", "unknown")
        roles = {
            c: _FIELDS[c][1]
            for c in ("rule_id", "rule_a", "rule_b", "added_rule_id")
            if c in frame.columns
        }
        if roles and "feature" not in frame.columns and not benchmark:
            entry["feature_relation"] = {
                "table": "cumulative_features" if name == "cumulative" else "rule_features",
                "key": "rule_id",
                "feature": "feature",
                "roles": roles,
            }
        entry["feature_query"] = (
            "direct" if "feature" in frame.columns else "relation" if roles else "unsupported"
        )
        for column, dtype in frame.schema.items():
            unit, meaning = _field_definition(column, name, dtype)
            if name == "bootstrap" and column.startswith("lift_"):
                unit, meaning = (
                    "dimensionless",
                    "实际 bootstrap Lift 分位数界或中位数；重采样配置与样本范围见 advanced_analysis",
                )
            entry["fields"][column].update(unit=unit, meaning=meaning)
    return ReportSnapshot(tables, description)
