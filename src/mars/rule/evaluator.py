"""规则命中评估与类型化筛选。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

import polars as pl

from mars.compute import FrameLike, to_polars_frame
from mars.rule._dsl import expression_to_polars, parse_expression
from mars.rule._statistics import add_empty_statistical_metrics, add_statistical_metrics
from mars.rule.contracts import (
    MarsRuleFilter,
    MarsRuleMetricCondition,
    MarsRuleSet,
    RuleDirection,
)


@dataclass(frozen=True)
class MarsRuleEvaluation:
    """保存一次规则评估的固定长表。

    Parameters
    ----------
    overall_table : polars.DataFrame
        ``slice="__overall__"`` 的整体评估长表。
    slice_table : polars.DataFrame
        按业务分组或时间切片展开的评估长表。
    metadata : Mapping[str, Any]
        数据集角色、目标和列配置等运行元数据。
    """

    overall_table: pl.DataFrame = field(default_factory=pl.DataFrame)
    slice_table: pl.DataFrame = field(default_factory=pl.DataFrame)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class MarsRuleEvaluator:
    """在样本、金额和客户维度评估规则。

    评估器接受 Pandas 或 Polars 输入，输出固定 Polars 长表。每个目标只使用该目标非空且
    可转换为二分类数值的样本，未定义比率保留为 null。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.rule import MarsRule, MarsRuleEvaluator, MarsRuleSet
    >>> frame = pl.DataFrame({"age": [20, 40], "y": [1, 0]})
    >>> rules = MarsRuleSet([MarsRule("age < 30")])
    >>> result = MarsRuleEvaluator().evaluate(frame, rules, target="y")
    >>> result.overall_table.filter(pl.col("group") == "hit")["event_count"][0]
    1
    """

    def evaluate(
        self,
        df: FrameLike,
        rule_set: MarsRuleSet,
        *,
        target: str,
        aux_targets: Sequence[str] | None = None,
        dataset: str = "evaluation",
        group_col: str | None = None,
        time_col: str | None = None,
        time_grain: str | None = None,
        amount_col: str | None = None,
        customer_col: str | None = None,
        batch_size: int = 100,
        direction: RuleDirection = "high_risk",
        confidence_level: float = 0.95,
        compute_statistics: bool = False,
    ) -> MarsRuleEvaluation:
        """评估规则整体与切片指标。

        Parameters
        ----------
        df : FrameLike
            待评估样本。
        rule_set : MarsRuleSet
            不含样本指标的规则定义。
        target : str
            生成和主要筛选使用的二分类目标。
        aux_targets : Sequence[str] | None
            仅参与评估的辅助目标。
        dataset : str
            写入长表的样本角色标签。
        group_col : str | None
            已存在的切片列。
        time_col : str | None
            用于生成时间切片的日期列。
        time_grain : str | None
            ``day``、``week``、``month`` 或 ``year``；不传时保留原始日期文本。
        amount_col : str | None
            金额指标列。
        customer_col : str | None
            客户去重指标列。
        batch_size : int
            单批物化和聚合的规则数量。
        direction : RuleDirection
            精确检验使用的风险方向。
        confidence_level : float
            Wilson 单侧置信水平。
        compute_statistics : bool
            是否计算 Wilson、精确检验和 BH-FDR；关闭时仍保留固定可空列。

        Returns
        -------
        MarsRuleEvaluation
            整体表、切片表和运行元数据。

        Raises
        ------
        ValueError
            缺列、目标非二分类、时间粒度非法或规则引用缺失时抛出。
        """
        frame: pl.DataFrame = to_polars_frame(df)
        if batch_size < 1:
            raise ValueError("batch_size 必须至少为 1。")
        if direction not in {"high_risk", "low_risk"}:
            raise ValueError("direction 必须是 'high_risk' 或 'low_risk'。")
        if not 0.5 < confidence_level < 1:
            raise ValueError("confidence_level 必须位于 (0.5, 1)。")
        targets: List[str] = list(dict.fromkeys([target, *(aux_targets or [])]))
        required: List[str] = list(rule_set.required_features) + targets
        required.extend(column for column in (group_col, time_col, amount_col, customer_col) if column)
        missing: List[str] = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"规则评估缺少必需列：{sorted(set(missing))}。")

        # 评估结果不返回原始宽表；只保留实际依赖列，避免每个规则批次重复携带无关特征。
        required_columns: List[str] = list(dict.fromkeys(required))
        frame = frame.select(required_columns)

        work: pl.DataFrame = frame
        slice_column: str | None = None
        if group_col is not None:
            slice_column = "__mars_rule_slice"
            work = work.with_columns(
                pl.col(group_col).cast(pl.Utf8).fill_null("Missing").alias(slice_column)
            )
        elif time_col is not None:
            slice_column = "__mars_rule_slice"
            work = work.with_columns(
                _time_slice_expression(time_col, time_grain).alias(slice_column)
            )

        overall_rows: List[Dict[str, Any]] = []
        slice_rows: List[Dict[str, Any]] = []
        for target_name in targets:
            base_target_frame: pl.DataFrame = _validated_target_frame(work, target_name)
            for start in range(0, len(rule_set.rules), batch_size):
                batch_rules = rule_set.rules[start : start + batch_size]
                batch_set = MarsRuleSet(batch_rules)
                mask_expressions: List[pl.Expr] = [
                    expression_to_polars(
                        parse_expression(rule.expression),
                        base_target_frame.schema,
                    )
                    .fill_null(False)
                    .alias(f"__mars_rule_{rule.rule_id}")
                    for rule in batch_rules
                ]
                target_frame: pl.DataFrame = base_target_frame.with_columns(mask_expressions)
                overall_rows.extend(
                    _evaluate_partition(
                        target_frame,
                        batch_set,
                        target_name,
                        dataset,
                        "__overall__",
                        amount_col,
                        customer_col,
                    )
                )
                if slice_column is None:
                    continue
                slice_rows.extend(
                    _evaluate_slices(
                        target_frame,
                        batch_set,
                        target_name,
                        dataset,
                        slice_column,
                        amount_col,
                        customer_col,
                    )
                )
        metadata: Dict[str, Any] = {
            "dataset": dataset,
            "primary_target": target,
            "targets": targets,
            "group_col": group_col,
            "time_col": time_col,
            "time_grain": time_grain,
            "amount_col": amount_col,
            "customer_col": customer_col,
            "batch_size": batch_size,
            "direction": direction,
            "confidence_level": confidence_level,
            "compute_statistics": compute_statistics,
        }
        overall_table: pl.DataFrame = _rows_to_table(overall_rows)
        slice_table: pl.DataFrame = _rows_to_table(slice_rows)
        if compute_statistics:
            overall_table = add_statistical_metrics(
                overall_table,
                direction=direction,
                confidence_level=confidence_level,
            )
            slice_table = add_statistical_metrics(
                slice_table,
                direction=direction,
                confidence_level=confidence_level,
            )
        else:
            overall_table = add_empty_statistical_metrics(overall_table)
            slice_table = add_empty_statistical_metrics(slice_table)
        return MarsRuleEvaluation(
            overall_table=overall_table,
            slice_table=slice_table,
            metadata=metadata,
        )


def select_rule_ids(
    evaluation: MarsRuleEvaluation,
    rule_filter: MarsRuleFilter,
    *,
    primary_target: str,
) -> List[str]:
    """按类型化筛选器返回稳定 rule_id 顺序。"""
    overall: pl.DataFrame = evaluation.overall_table
    if overall.is_empty():
        return []
    hit_rows: pl.DataFrame = overall.filter(pl.col("group") == "hit")
    targets: List[str] = _resolve_filter_targets(hit_rows, rule_filter, primary_target)
    rule_ids: List[str] = list(dict.fromkeys(hit_rows["rule_id"].to_list()))
    selected: List[str] = []
    for rule_id in rule_ids:
        per_target: List[bool] = []
        for target in targets:
            rows: pl.DataFrame = hit_rows.filter(
                (pl.col("rule_id") == rule_id) & (pl.col("target") == target)
            )
            passed: bool = rows.height == 1 and _row_passes(
                rows.row(0, named=True),
                rule_filter.conditions,
            )
            if passed and rule_filter.slice_pass_rate is not None and not evaluation.slice_table.is_empty():
                target_slices: pl.DataFrame = evaluation.slice_table.filter(
                    (pl.col("rule_id") == rule_id)
                    & (pl.col("target") == target)
                    & (pl.col("group") == "hit")
                )
                if target_slices.is_empty():
                    passed = False
                else:
                    slice_flags: List[bool] = [
                        _row_passes(row, rule_filter.conditions)
                        for row in target_slices.iter_rows(named=True)
                    ]
                    passed = sum(slice_flags) / len(slice_flags) >= rule_filter.slice_pass_rate
            per_target.append(passed)
        target_pass: bool = all(per_target) if rule_filter.target_scope == "all" else any(per_target)
        if target_pass:
            selected.append(rule_id)
    return selected


def select_production_rule_ids(
    evaluation: MarsRuleEvaluation,
    rule_filter: MarsRuleFilter,
    *,
    primary_target: str,
    direction: RuleDirection,
    max_fdr: float,
    min_time_slices: int,
) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
    """执行不可被自定义筛选器移除的生产统计与时间门禁。"""
    overall: pl.DataFrame = evaluation.overall_table
    if overall.is_empty():
        return [], {}
    hit_rows: pl.DataFrame = overall.filter(pl.col("group") == "hit")
    targets: List[str] = _resolve_filter_targets(hit_rows, rule_filter, primary_target)
    rule_ids: List[str] = list(dict.fromkeys(hit_rows["rule_id"].to_list()))
    diagnostics: Dict[str, Dict[str, Any]] = {}
    selected: List[str] = []
    for rule_id in rule_ids:
        per_target_pass: List[bool] = []
        target_diagnostics: List[Dict[str, Any]] = []
        for target in targets:
            rows: pl.DataFrame = hit_rows.filter(
                (pl.col("rule_id") == rule_id) & (pl.col("target") == target)
            )
            if rows.height != 1:
                per_target_pass.append(False)
                continue
            row: Dict[str, Any] = dict(rows.row(0, named=True))
            point_pass: bool = _row_passes(row, rule_filter.conditions)
            statistical_pass: bool = _production_row_passes(
                row,
                rule_filter.conditions,
                direction=direction,
                max_fdr=max_fdr,
            )
            temporal_assessed, temporal_passed, slice_count, slice_pass_rate = (
                _temporal_gate(
                    evaluation,
                    rule_id=rule_id,
                    target=target,
                    conditions=rule_filter.conditions,
                    required_pass_rate=rule_filter.slice_pass_rate,
                    min_time_slices=min_time_slices,
                )
            )
            passed: bool = point_pass and statistical_pass and temporal_passed
            per_target_pass.append(passed)
            target_diagnostics.append(
                {
                    "target": target,
                    "point_passed": point_pass,
                    "statistical_passed": statistical_pass,
                    "q_value": row.get("q_value"),
                    "lift_ci_lower": row.get("lift_ci_lower"),
                    "lift_ci_upper": row.get("lift_ci_upper"),
                    "temporal_assessed": temporal_assessed,
                    "temporal_passed": temporal_passed,
                    "time_slice_count": slice_count,
                    "time_slice_pass_rate": slice_pass_rate,
                }
            )
        target_pass: bool = (
            all(per_target_pass)
            if rule_filter.target_scope == "all"
            else any(per_target_pass)
        )
        diagnostics[rule_id] = {
            "passed": target_pass,
            "targets": target_diagnostics,
            "temporal_assessed": bool(target_diagnostics)
            and all(item["temporal_assessed"] for item in target_diagnostics),
            "temporal_passed": bool(target_diagnostics)
            and all(item["temporal_passed"] for item in target_diagnostics),
            "time_slice_count": min(
                (int(item["time_slice_count"]) for item in target_diagnostics),
                default=0,
            ),
            "time_slice_pass_rate": min(
                (
                    float(item["time_slice_pass_rate"])
                    for item in target_diagnostics
                    if item["time_slice_pass_rate"] is not None
                ),
                default=None,
            ),
        }
        if target_pass:
            selected.append(rule_id)
    return selected, diagnostics


def _production_row_passes(
    row: Mapping[str, Any],
    conditions: Sequence[MarsRuleMetricCondition],
    *,
    direction: RuleDirection,
    max_fdr: float,
) -> bool:
    """把 Lift 点估计条件替换为方向一致的保守置信界。"""
    q_value: Any = row.get("q_value")
    if q_value is None or float(q_value) > max_fdr:
        return False
    conservative_conditions: List[MarsRuleMetricCondition] = []
    lift_metric: str = "lift_ci_lower" if direction == "high_risk" else "lift_ci_upper"
    for condition in conditions:
        metric: str = lift_metric if condition.metric == "lift" else condition.metric
        conservative_conditions.append(
            MarsRuleMetricCondition(metric, condition.operator, condition.value)
        )
    return _row_passes(row, conservative_conditions)


def _temporal_gate(
    evaluation: MarsRuleEvaluation,
    *,
    rule_id: str,
    target: str,
    conditions: Sequence[MarsRuleMetricCondition],
    required_pass_rate: float | None,
    min_time_slices: int,
) -> tuple[bool, bool, int, float | None]:
    """在切片足够时执行硬门禁，否则保留独立验证资格。"""
    if not evaluation.metadata.get("time_col"):
        return False, True, 0, None
    slices: pl.DataFrame = evaluation.slice_table
    if slices.is_empty():
        return False, True, 0, None
    target_slices: pl.DataFrame = slices.filter(
        (pl.col("rule_id") == rule_id)
        & (pl.col("target") == target)
        & (pl.col("group") == "hit")
        & (pl.col("slice") != "Missing")
    )
    slice_count: int = target_slices.height
    if slice_count < min_time_slices:
        return False, True, slice_count, None
    flags: List[bool] = [
        _row_passes(row, conditions) for row in target_slices.iter_rows(named=True)
    ]
    pass_rate: float = sum(flags) / len(flags)
    threshold: float = required_pass_rate if required_pass_rate is not None else 0.8
    return True, pass_rate >= threshold, slice_count, pass_rate


def _resolve_filter_targets(
    table: pl.DataFrame,
    rule_filter: MarsRuleFilter,
    primary_target: str,
) -> List[str]:
    """解析筛选器的目标范围并校验显式目标。"""
    available: List[str] = list(dict.fromkeys(table["target"].to_list()))
    if rule_filter.targets == "primary":
        targets = [primary_target]
    elif rule_filter.targets == "all":
        targets = available
    else:
        targets = list(rule_filter.targets)
    missing: List[str] = [target for target in targets if target not in available]
    if missing:
        raise ValueError(f"规则筛选引用未知目标：{missing}。")
    return targets


def _row_passes(row: Mapping[str, Any], conditions: Sequence[MarsRuleMetricCondition]) -> bool:
    """判断评估行是否满足全部结构化条件。"""
    for condition in conditions:
        value: Any = row.get(condition.metric)
        if value is None:
            return False
        threshold: float = condition.value
        passed = {
            "<": value < threshold,
            "<=": value <= threshold,
            "==": value == threshold,
            "!=": value != threshold,
            ">=": value >= threshold,
            ">": value > threshold,
        }[condition.operator]
        if not passed:
            return False
    return True


def _validated_target_frame(frame: pl.DataFrame, target: str) -> pl.DataFrame:
    """排除空标签、转换为浮点并校验二分类取值。"""
    target_alias: str = "__mars_rule_target"
    converted: pl.DataFrame = (
        frame
        .with_columns(pl.col(target).cast(pl.Float64, strict=False).alias(target_alias))
        .filter(pl.col(target_alias).is_not_null() & pl.col(target_alias).is_not_nan())
    )
    values: List[float] = sorted(float(value) for value in converted[target_alias].unique())
    if not values:
        raise ValueError(f"目标列 {target!r} 没有可评估的非空标签。")
    if any(value not in {0.0, 1.0} for value in values):
        raise ValueError(f"目标列 {target!r} 必须是 0/1 二分类，实际取值：{values}。")
    return converted


def _evaluate_partition(
    frame: pl.DataFrame,
    rule_set: MarsRuleSet,
    target: str,
    dataset: str,
    slice_value: str,
    amount_col: str | None,
    customer_col: str | None,
) -> List[Dict[str, Any]]:
    """计算单目标、单切片下的命中/未命中/总体指标。"""
    base: Dict[str, Any] = _base_totals(frame, amount_col, customer_col)
    if not rule_set.rules:
        return []
    expressions: List[pl.Expr] = []
    for index, rule in enumerate(rule_set.rules):
        mask_column: str = f"__mars_rule_{rule.rule_id}"
        expressions.extend(
            _group_aggregate_expressions(
                f"r{index}__hit",
                pl.col(mask_column),
                amount_col,
                customer_col,
            )
        )
        if customer_col is not None:
            expressions.extend(
                _group_aggregate_expressions(
                    f"r{index}__miss",
                    ~pl.col(mask_column),
                    amount_col,
                    customer_col,
                )
            )
    aggregates: Dict[str, Any] = frame.select(expressions).row(0, named=True)

    rows: List[Dict[str, Any]] = []
    for index, rule in enumerate(rule_set.rules):
        rows.append(
            _metric_row_from_aggregates(
                aggregates,
                f"r{index}__hit",
                base,
                rule.rule_id,
                target,
                dataset,
                slice_value,
                "hit",
                amount_col,
                customer_col,
            )
        )
        if customer_col is None:
            rows.append(
                _metric_row_from_complement(
                    aggregates,
                    f"r{index}__hit",
                    base,
                    rule.rule_id,
                    target,
                    dataset,
                    slice_value,
                    amount_col,
                )
            )
        else:
            rows.append(
                _metric_row_from_aggregates(
                    aggregates,
                    f"r{index}__miss",
                    base,
                    rule.rule_id,
                    target,
                    dataset,
                    slice_value,
                    "miss",
                    amount_col,
                    customer_col,
                )
            )
        rows.append(
            _metric_row_from_base(
                base,
                rule.rule_id,
                target,
                dataset,
                slice_value,
            )
        )
    return rows


def _evaluate_slices(
    frame: pl.DataFrame,
    rule_set: MarsRuleSet,
    target: str,
    dataset: str,
    slice_column: str,
    amount_col: str | None,
    customer_col: str | None,
) -> List[Dict[str, Any]]:
    """一次分组聚合一个规则批次的全部切片，避免逐切片反复扫描。"""
    if not rule_set.rules:
        return []
    target_alias: str = "__mars_rule_target"
    expressions: List[pl.Expr] = [
        pl.len().alias("base__sample_count"),
        pl.col(target_alias).sum().alias("base__event_count"),
    ]
    if amount_col is not None:
        expressions.extend(
            [
                pl.col(amount_col).sum().alias("base__amount_total"),
                pl.col(amount_col)
                .filter(pl.col(target_alias) == 1)
                .sum()
                .alias("base__event_amount"),
            ]
        )
    if customer_col is not None:
        expressions.extend(
            [
                pl.col(customer_col).n_unique().alias("base__customer_count"),
                pl.col(customer_col)
                .filter(pl.col(target_alias) == 1)
                .n_unique()
                .alias("base__event_customer_count"),
            ]
        )
    for index, rule in enumerate(rule_set.rules):
        mask_column: str = f"__mars_rule_{rule.rule_id}"
        expressions.extend(
            _group_aggregate_expressions(
                f"r{index}__hit",
                pl.col(mask_column),
                amount_col,
                customer_col,
            )
        )
        if customer_col is not None:
            expressions.extend(
                _group_aggregate_expressions(
                    f"r{index}__miss",
                    ~pl.col(mask_column),
                    amount_col,
                    customer_col,
                )
            )
    grouped: pl.DataFrame = frame.group_by(slice_column).agg(expressions).sort(slice_column)
    rows: List[Dict[str, Any]] = []
    for aggregate_row in grouped.iter_rows(named=True):
        slice_value: str = str(aggregate_row[slice_column])
        base: Dict[str, Any] = _base_from_grouped_row(
            aggregate_row,
            amount_col=amount_col,
            customer_col=customer_col,
        )
        for index, rule in enumerate(rule_set.rules):
            rows.append(
                _metric_row_from_aggregates(
                    aggregate_row,
                    f"r{index}__hit",
                    base,
                    rule.rule_id,
                    target,
                    dataset,
                    slice_value,
                    "hit",
                    amount_col,
                    customer_col,
                )
            )
            if customer_col is None:
                rows.append(
                    _metric_row_from_complement(
                        aggregate_row,
                        f"r{index}__hit",
                        base,
                        rule.rule_id,
                        target,
                        dataset,
                        slice_value,
                        amount_col,
                    )
                )
            else:
                rows.append(
                    _metric_row_from_aggregates(
                        aggregate_row,
                        f"r{index}__miss",
                        base,
                        rule.rule_id,
                        target,
                        dataset,
                        slice_value,
                        "miss",
                        amount_col,
                        customer_col,
                    )
                )
            rows.append(
                _metric_row_from_base(
                    base,
                    rule.rule_id,
                    target,
                    dataset,
                    slice_value,
                )
            )
    return rows


def _base_from_grouped_row(
    row: Mapping[str, Any],
    *,
    amount_col: str | None,
    customer_col: str | None,
) -> Dict[str, Any]:
    """把切片分组聚合行恢复成统一分母结构。"""
    sample_count: int = int(row["base__sample_count"] or 0)
    event_count: int = int(row["base__event_count"] or 0)
    amount_total: float | None = (
        float(row["base__amount_total"] or 0.0) if amount_col is not None else None
    )
    event_amount: float | None = (
        float(row["base__event_amount"] or 0.0) if amount_col is not None else None
    )
    customer_count: int | None = (
        int(row["base__customer_count"] or 0) if customer_col is not None else None
    )
    event_customer_count: int | None = (
        int(row["base__event_customer_count"] or 0)
        if customer_col is not None
        else None
    )
    return {
        "sample_count": sample_count,
        "event_count": event_count,
        "event_rate": _safe_div(event_count, sample_count),
        "amount_total": amount_total,
        "event_amount": event_amount,
        "amount_event_rate": _safe_div(event_amount, amount_total),
        "customer_count": customer_count,
        "event_customer_count": event_customer_count,
        "customer_event_rate": _safe_div(event_customer_count, customer_count),
    }


def _group_aggregate_expressions(
    prefix: str,
    mask: pl.Expr,
    amount_col: str | None,
    customer_col: str | None,
) -> List[pl.Expr]:
    """构造单个命中分组的批量聚合表达式。"""
    target_alias: str = "__mars_rule_target"
    safe_mask: pl.Expr = mask.fill_null(False)
    expressions: List[pl.Expr] = [
        safe_mask.cast(pl.Int64).sum().alias(f"{prefix}__sample_count"),
        pl.col(target_alias).filter(safe_mask).sum().alias(f"{prefix}__event_count"),
    ]
    if amount_col is not None:
        expressions.extend(
            [
                pl.col(amount_col).filter(safe_mask).sum().alias(f"{prefix}__amount_total"),
                pl.col(amount_col)
                .filter(safe_mask & (pl.col(target_alias) == 1))
                .sum()
                .alias(f"{prefix}__event_amount"),
            ]
        )
    if customer_col is not None:
        expressions.extend(
            [
                pl.col(customer_col)
                .filter(safe_mask)
                .n_unique()
                .alias(f"{prefix}__customer_count"),
                pl.col(customer_col)
                .filter(safe_mask & (pl.col(target_alias) == 1))
                .n_unique()
                .alias(f"{prefix}__event_customer_count"),
            ]
        )
    return expressions


def _base_totals(
    frame: pl.DataFrame,
    amount_col: str | None,
    customer_col: str | None,
) -> Dict[str, Any]:
    """计算分母口径。"""
    target_alias: str = "__mars_rule_target"
    sample_count: int = frame.height
    event_count: int = int(frame[target_alias].sum() or 0)
    amount_total: float | None = None
    event_amount: float | None = None
    if amount_col is not None:
        amount_total = float(frame[amount_col].sum() or 0.0)
        event_amount = float(
            frame.filter(pl.col(target_alias) == 1)[amount_col].sum() or 0.0
        )
    customer_count: int | None = None
    event_customer_count: int | None = None
    if customer_col is not None:
        customer_count = int(frame[customer_col].n_unique())
        event_customer_count = int(
            frame.filter(pl.col(target_alias) == 1)[customer_col].n_unique()
        )
    return {
        "sample_count": sample_count,
        "event_count": event_count,
        "event_rate": _safe_div(event_count, sample_count),
        "amount_total": amount_total,
        "event_amount": event_amount,
        "amount_event_rate": _safe_div(event_amount, amount_total),
        "customer_count": customer_count,
        "event_customer_count": event_customer_count,
        "customer_event_rate": _safe_div(event_customer_count, customer_count),
    }


def _metric_row_from_aggregates(
    aggregates: Mapping[str, Any],
    prefix: str,
    base: Mapping[str, Any],
    rule_id: str,
    target: str,
    dataset: str,
    slice_value: str,
    group: str,
    amount_col: str | None,
    customer_col: str | None,
) -> Dict[str, Any]:
    """把单组批量聚合值组装为固定 schema 指标行。"""
    sample_count: int = int(aggregates[f"{prefix}__sample_count"] or 0)
    event_count: int = int(aggregates[f"{prefix}__event_count"] or 0)
    amount_total: float | None = (
        float(aggregates[f"{prefix}__amount_total"] or 0.0)
        if amount_col is not None
        else None
    )
    event_amount: float | None = (
        float(aggregates[f"{prefix}__event_amount"] or 0.0)
        if amount_col is not None
        else None
    )
    customer_count: int | None = (
        int(aggregates[f"{prefix}__customer_count"] or 0)
        if customer_col is not None
        else None
    )
    event_customer_count: int | None = (
        int(aggregates[f"{prefix}__event_customer_count"] or 0)
        if customer_col is not None
        else None
    )
    return _build_metric_row(
        sample_count=sample_count,
        event_count=event_count,
        amount_total=amount_total,
        event_amount=event_amount,
        customer_count=customer_count,
        event_customer_count=event_customer_count,
        base=base,
        rule_id=rule_id,
        target=target,
        dataset=dataset,
        slice_value=slice_value,
        group=group,
    )


def _metric_row_from_complement(
    aggregates: Mapping[str, Any],
    hit_prefix: str,
    base: Mapping[str, Any],
    rule_id: str,
    target: str,
    dataset: str,
    slice_value: str,
    amount_col: str | None,
) -> Dict[str, Any]:
    """用总体减命中聚合值构造未命中行，避免重复扫描布尔补集。"""
    amount_total: float | None = None
    event_amount: float | None = None
    if amount_col is not None:
        amount_total = float(base["amount_total"] or 0.0) - float(
            aggregates[f"{hit_prefix}__amount_total"] or 0.0
        )
        event_amount = float(base["event_amount"] or 0.0) - float(
            aggregates[f"{hit_prefix}__event_amount"] or 0.0
        )
    return _build_metric_row(
        sample_count=int(base["sample_count"])
        - int(aggregates[f"{hit_prefix}__sample_count"] or 0),
        event_count=int(base["event_count"])
        - int(aggregates[f"{hit_prefix}__event_count"] or 0),
        amount_total=amount_total,
        event_amount=event_amount,
        customer_count=None,
        event_customer_count=None,
        base=base,
        rule_id=rule_id,
        target=target,
        dataset=dataset,
        slice_value=slice_value,
        group="miss",
    )


def _metric_row_from_base(
    base: Mapping[str, Any],
    rule_id: str,
    target: str,
    dataset: str,
    slice_value: str,
) -> Dict[str, Any]:
    """把分区分母复制为规则 total 行。"""
    return _build_metric_row(
        sample_count=int(base["sample_count"]),
        event_count=int(base["event_count"]),
        amount_total=base["amount_total"],
        event_amount=base["event_amount"],
        customer_count=base["customer_count"],
        event_customer_count=base["event_customer_count"],
        base=base,
        rule_id=rule_id,
        target=target,
        dataset=dataset,
        slice_value=slice_value,
        group="total",
    )


def _build_metric_row(
    *,
    sample_count: int,
    event_count: int,
    amount_total: float | None,
    event_amount: float | None,
    customer_count: int | None,
    event_customer_count: int | None,
    base: Mapping[str, Any],
    rule_id: str,
    target: str,
    dataset: str,
    slice_value: str,
    group: str,
) -> Dict[str, Any]:
    """根据已聚合分子和分区分母计算全部固定指标。"""
    event_rate: float | None = _safe_div(event_count, sample_count)
    amount_event_rate: float | None = _safe_div(event_amount, amount_total)
    customer_event_rate: float | None = _safe_div(
        event_customer_count,
        customer_count,
    )
    row: Dict[str, Any] = {
        "dataset": dataset,
        "rule_id": rule_id,
        "target": target,
        "slice": slice_value,
        "group": group,
        "sample_count": sample_count,
        "event_count": event_count,
        "coverage": _safe_div(sample_count, base["sample_count"]),
        "event_rate": event_rate,
        "lift": _safe_div(event_rate, base["event_rate"]),
        "amount_total": amount_total,
        "event_amount": event_amount,
        "amount_coverage": _safe_div(amount_total, base["amount_total"]),
        "amount_event_rate": amount_event_rate,
        "amount_lift": _safe_div(amount_event_rate, base["amount_event_rate"]),
        "customer_count": customer_count,
        "event_customer_count": event_customer_count,
        "customer_coverage": _safe_div(customer_count, base["customer_count"]),
        "customer_event_rate": customer_event_rate,
        "customer_lift": _safe_div(
            customer_event_rate,
            base["customer_event_rate"],
        ),
    }
    return row


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    """执行语义安全除法，未定义时返回 null。"""
    if numerator is None or denominator is None:
        return None
    denominator_value: float = float(denominator)
    if denominator_value == 0:
        return None
    return float(numerator) / denominator_value


def _time_slice_expression(time_col: str, time_grain: str | None) -> pl.Expr:
    """构造兼容字符串和日期输入的时间切片表达式。"""
    if time_grain is None:
        return pl.col(time_col).cast(pl.Utf8).fill_null("Missing")
    formats: Dict[str, str] = {
        "day": "%Y-%m-%d",
        "week": "%G-W%V",
        "month": "%Y-%m",
        "year": "%Y",
    }
    if time_grain not in formats:
        raise ValueError("time_grain 必须是 day、week、month、year 或 None。")
    parsed: pl.Expr = pl.col(time_col).cast(pl.Utf8).str.strptime(pl.Date, strict=False)
    return parsed.dt.strftime(formats[time_grain]).fill_null("Missing")


def _rows_to_table(rows: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """把指标行稳定转换为 Polars 表。"""
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame([dict(row) for row in rows])
