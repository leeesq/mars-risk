"""规则挖掘高层工作流。"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple, Union

import numpy as np
import polars as pl

from mars import __version__ as mars_version
from mars.compute import FrameLike, to_polars_frame
from mars.rule._dsl import expression_to_polars, parse_expression
from mars.rule.analysis import MarsRuleAnalysis, analyze_rule_set
from mars.rule.contracts import (
    MarsRule,
    MarsRuleMiningSpec,
    MarsRuleSet,
    RuleQualification,
)
from mars.rule.evaluator import (
    MarsRuleEvaluation,
    MarsRuleEvaluator,
    select_production_rule_ids,
    select_rule_ids,
)
from mars.rule.generators import (
    MarsCombinationRuleGenerator,
    MarsRuleGenerator,
    MarsTreeRuleGenerator,
)
from mars.rule.report import MarsRuleReport

_POPCOUNT_TABLE: np.ndarray = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None],
    axis=1,
).sum(axis=1)


@dataclass(frozen=True)
class MarsRuleMiningResult:
    """规则挖掘的可审计结构化结果。

    Parameters
    ----------
    status : {"success", "no_rules"}
        最终业务状态；合法零入选使用 ``no_rules``。
    rule_set : MarsRuleSet
        最终可部署规则定义。
    candidate_table : polars.DataFrame
        候选来源、淘汰阶段和原因审计表。
    evaluation : MarsRuleEvaluation
        训练与验证整体、切片长表。
    spec : MarsRuleMiningSpec
        完整解析后的挖掘策略。
    metadata : Mapping[str, Any]
        数据角色、验证状态、版本和阶段耗时。
    """

    status: str
    rule_set: MarsRuleSet
    candidate_table: pl.DataFrame
    evaluation: MarsRuleEvaluation
    spec: MarsRuleMiningSpec
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def analyze(
        self,
        df: FrameLike,
        *,
        target: str | None = None,
        amount_col: str | None = None,
        customer_col: str | None = None,
        max_pairs: int = 5000,
        bootstrap_repeats: int = 0,
        confidence_level: float | None = None,
        random_state: int | None = None,
    ) -> MarsRuleAnalysis:
        """按需计算最终规则的交互和累计贡献。

        Parameters
        ----------
        df : FrameLike
            分析样本；结果对象不会保存该数据。
        target : str | None
            分析目标；不传时使用挖掘主目标。
        amount_col : str | None
            金额列；不传时沿用挖掘阶段配置。
        customer_col : str | None
            客户列；不传时沿用挖掘阶段配置。
        max_pairs : int
            交互分析最大规则对数。
        bootstrap_repeats : int
            最终规则 Lift 重采样次数；默认关闭。
        confidence_level : float | None
            bootstrap 区间置信水平；不传时沿用挖掘 spec。
        random_state : int | None
            bootstrap 种子；不传时沿用挖掘 spec。

        Returns
        -------
        MarsRuleAnalysis
            交互、累计表和分析元数据。
        """
        resolved_target: str = target or str(self.metadata["target"])
        resolved_amount: str | None = amount_col or _optional_metadata_column(
            self.metadata,
            "amount_col",
        )
        resolved_customer: str | None = customer_col or _optional_metadata_column(
            self.metadata,
            "customer_col",
        )
        return analyze_rule_set(
            self.rule_set,
            df,
            target=resolved_target,
            amount_col=resolved_amount,
            customer_col=resolved_customer,
            max_pairs=max_pairs,
            bootstrap_repeats=bootstrap_repeats,
            confidence_level=confidence_level or self.spec.confidence_level,
            random_state=(
                self.spec.random_state if random_state is None else random_state
            ),
        )

    def to_report(
        self,
        analysis: MarsRuleAnalysis | None = None,
    ) -> MarsRuleReport:
        """构造不产生文件副作用的结构化报告。

        Parameters
        ----------
        analysis : MarsRuleAnalysis | None
            显式执行的高级分析；不传时省略相关 section。

        Returns
        -------
        MarsRuleReport
            可进一步导出 HTML 或 Excel 的报告。
        """
        summary: pl.DataFrame = pl.DataFrame(
            [
                {
                    "status": self.status,
                    "candidate_count": self.candidate_table.height,
                    "selected_count": len(self.rule_set.rules),
                    "validation_status": self.metadata.get("validation_status"),
                    "profile": self.spec.profile,
                    "qualification": self.rule_set.qualification,
                }
            ]
        )
        detail_tables: Dict[str, pl.DataFrame] = {
            "candidates": self.candidate_table,
            "evaluation": self.evaluation.overall_table,
        }
        explanations: pl.DataFrame = _build_rule_explanations(self)
        if not explanations.is_empty():
            detail_tables["rule_explanations"] = explanations
        if not self.evaluation.slice_table.is_empty():
            detail_tables["slices"] = self.evaluation.slice_table
        if analysis is not None:
            if not analysis.interaction_table.is_empty():
                detail_tables["interactions"] = analysis.interaction_table
            if not analysis.cumulative_table.is_empty():
                detail_tables["cumulative"] = analysis.cumulative_table
            if not analysis.bootstrap_table.is_empty():
                detail_tables["bootstrap"] = analysis.bootstrap_table
        return MarsRuleReport(
            summary_table=summary,
            detail_tables=detail_tables,
            metadata=dict(self.metadata),
        )


def mine_rules(
    train_df: FrameLike,
    *,
    target: str,
    validation_df: FrameLike | None = None,
    aux_targets: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
    group_col: str | None = None,
    time_col: str | None = None,
    time_grain: str | None = None,
    amount_col: str | None = None,
    customer_col: str | None = None,
    seed_rules: Sequence[Union[str, MarsRule]] | None = None,
    spec: MarsRuleMiningSpec | None = None,
    generators: Sequence[MarsRuleGenerator] | None = None,
) -> MarsRuleMiningResult:
    """生成、评估、筛选并组装可部署规则集。

    Parameters
    ----------
    train_df : FrameLike
        候选生成和训练筛选样本。
    target : str
        主二分类目标列。
    validation_df : FrameLike | None
        独立验证样本；不传时明确降级为样本内探索。
    aux_targets : Sequence[str] | None
        仅参与评估和类型化筛选的辅助目标。
    features : Sequence[str] | None
        自动生成使用的数值特征；不传时排除所有数据角色列后推断。
    group_col : str | None
        已存在的验证切片列。
    time_col : str | None
        用于构造验证时间切片的日期列。
    time_grain : str | None
        ``day``、``week``、``month`` 或 ``year``。
    amount_col : str | None
        金额指标列。
    customer_col : str | None
        客户去重指标列。
    seed_rules : Sequence[Union[str, MarsRule]] | None
        用户提供的 DSL 规则候选。
    spec : MarsRuleMiningSpec | None
        类型化挖掘策略；不传时使用默认高风险策略。
    generators : Sequence[MarsRuleGenerator] | None
        自定义生成器；不传时使用组合规则和浅层树，空序列表示只评估 seed rules。

    Returns
    -------
    MarsRuleMiningResult
        不保存原始 DataFrame 的可审计挖掘结果。

    Raises
    ------
    TypeError
        spec、generators 或 seed_rules 未遵循类型化公开契约时抛出。
    ValueError
        production 缺少独立验证集或输入配置非法时抛出。
    """
    start_time: float = time.perf_counter()
    if spec is not None and not isinstance(spec, MarsRuleMiningSpec):
        raise TypeError("spec 必须是 MarsRuleMiningSpec。")
    resolved_spec: MarsRuleMiningSpec = spec or MarsRuleMiningSpec()
    train: pl.DataFrame = to_polars_frame(train_df)
    validation_status: str = "independent"
    if validation_df is None:
        if resolved_spec.profile == "production":
            raise ValueError("production profile 必须提供独立 validation_df。")
        warnings.warn(
            "未提供 validation_df；最终规则仅经过样本内筛选，不能视为独立验证结果。",
            UserWarning,
            stacklevel=2,
        )
        validation: pl.DataFrame = train.clone()
        validation_status = "in_sample"
    else:
        validation = to_polars_frame(validation_df)
    targets: List[str] = list(dict.fromkeys([target, *(aux_targets or [])]))
    role_columns: Set[str] = set(targets)
    role_columns.update(
        column for column in (group_col, time_col, amount_col, customer_col) if column
    )
    resolved_features: List[str] = (
        list(features)
        if features is not None
        else [column for column in train.columns if column not in role_columns]
    )

    active_generators: Sequence[MarsRuleGenerator]
    if generators is None:
        active_generators = (
            MarsCombinationRuleGenerator(random_state=resolved_spec.random_state),
            MarsTreeRuleGenerator(random_state=resolved_spec.random_state),
        )
    else:
        if any(not isinstance(generator, MarsRuleGenerator) for generator in generators):
            raise TypeError("generators 只接受 MarsRuleGenerator 实例。")
        active_generators = tuple(generators)
    if resolved_spec.selection_strategy == "cascade":
        return _mine_rules_cascade(
            train=train,
            validation=validation,
            validation_status=validation_status,
            target=target,
            aux_targets=aux_targets,
            features=resolved_features,
            group_col=group_col,
            time_col=time_col,
            time_grain=time_grain,
            amount_col=amount_col,
            customer_col=customer_col,
            seed_rules=seed_rules,
            spec=resolved_spec,
            generators=active_generators,
            start_time=start_time,
        )

    (
        unique_rules,
        sources,
        duplicate_counts,
        generation_errors,
        budget_metadata,
    ) = _generate_candidates(
        train,
        target=target,
        features=resolved_features,
        seed_rules=seed_rules,
        generators=active_generators,
        on_generator_error=resolved_spec.on_generator_error,
        max_candidates=resolved_spec.max_candidates,
    )
    audit: Dict[str, Dict[str, Any]] = {
        rule.rule_id: {
            "rule_id": rule.rule_id,
            "expression": rule.expression,
            "sources": ",".join(sorted(sources[rule.rule_id])),
            "distinct_source_count": len(sources[rule.rule_id]),
            "exact_duplicate_count": duplicate_counts[rule.rule_id],
            "within_candidate_budget": None,
            "candidate_filter_passed": None,
            "validation_filter_passed": None,
            "iou_filter_passed": None,
            "selection_rank": None,
            "status": "candidate",
            "rejection_stage": None,
            "reason": None,
            "source_position": budget_metadata[rule.rule_id]["source_position"],
            "budget_position": budget_metadata[rule.rule_id]["budget_position"],
            "budget_strategy": "seed_then_round_robin",
            "q_value": None,
            "lift_ci_lower": None,
            "lift_ci_upper": None,
            "temporal_assessed": None,
            "temporal_passed": None,
            "time_slice_count": None,
            "time_slice_pass_rate": None,
        }
        for rule in unique_rules
    }
    budgeted_rules: List[MarsRule] = unique_rules[: resolved_spec.max_candidates]
    for rule in budgeted_rules:
        audit[rule.rule_id]["within_candidate_budget"] = True
    for rule in unique_rules[resolved_spec.max_candidates :]:
        audit[rule.rule_id]["within_candidate_budget"] = False
        _reject(audit, rule.rule_id, "candidate_budget", "超过 max_candidates")

    evaluator = MarsRuleEvaluator()
    candidate_set = MarsRuleSet(budgeted_rules)
    train_evaluation: MarsRuleEvaluation = evaluator.evaluate(
        train,
        candidate_set,
        target=target,
        aux_targets=aux_targets,
        dataset="train",
        amount_col=amount_col,
        customer_col=customer_col,
        batch_size=resolved_spec.batch_size,
        direction=resolved_spec.direction,
        confidence_level=resolved_spec.confidence_level,
    )
    candidate_ids: List[str] = select_rule_ids(
        train_evaluation,
        resolved_spec.candidate_filter,
        primary_target=target,
    )
    candidate_id_set: Set[str] = set(candidate_ids)
    for rule in budgeted_rules:
        audit[rule.rule_id]["candidate_filter_passed"] = rule.rule_id in candidate_id_set
        if rule.rule_id not in candidate_id_set:
            _reject(audit, rule.rule_id, "candidate_filter", "未通过训练集候选阈值")

    candidate_rules: List[MarsRule] = [
        rule for rule in budgeted_rules if rule.rule_id in candidate_id_set
    ]
    validation_set = MarsRuleSet(candidate_rules)
    validation_evaluation: MarsRuleEvaluation = evaluator.evaluate(
        validation,
        validation_set,
        target=target,
        aux_targets=aux_targets,
        dataset="validation" if validation_status == "independent" else "in_sample",
        group_col=group_col,
        time_col=time_col,
        time_grain=time_grain,
        amount_col=amount_col,
        customer_col=customer_col,
        batch_size=resolved_spec.batch_size,
        direction=resolved_spec.direction,
        confidence_level=resolved_spec.confidence_level,
        compute_statistics=True,
    )
    validated_ids, validation_diagnostics = _select_validation_ids(
        validation_evaluation,
        spec=resolved_spec,
        primary_target=target,
    )
    _attach_validation_diagnostics(
        audit,
        validation_evaluation,
        validation_diagnostics,
        primary_target=target,
    )
    validated_id_set: Set[str] = set(validated_ids)
    for rule in candidate_rules:
        audit[rule.rule_id]["validation_filter_passed"] = rule.rule_id in validated_id_set
        if rule.rule_id not in validated_id_set:
            _reject(audit, rule.rule_id, "validation_filter", "未通过验证阈值或切片通过率")

    ranked_ids: List[str] = _rank_rule_ids(
        validation_evaluation.overall_table,
        validated_ids,
        target,
        resolved_spec.direction,
    )
    ranked_rules: List[MarsRule] = _rules_by_ids(candidate_rules, ranked_ids)
    deduplicated_rules, duplicate_ids = _iou_deduplicate(
        validation,
        ranked_rules,
        resolved_spec.iou_threshold,
        batch_size=resolved_spec.iou_batch_size,
    )
    for rule_id in duplicate_ids:
        audit[rule_id]["iou_filter_passed"] = False
        _reject(audit, rule_id, "iou_deduplication", "命中人群与更高优先级规则重叠")
    for rule in deduplicated_rules:
        audit[rule.rule_id]["iou_filter_passed"] = True

    selected_rules: List[MarsRule] = deduplicated_rules[: resolved_spec.top_k]
    selected_ids: Set[str] = {rule.rule_id for rule in selected_rules}
    for rule in deduplicated_rules:
        if rule.rule_id not in selected_ids:
            _reject(audit, rule.rule_id, "top_k", "超过最终规则预算")
    for rank, rule in enumerate(selected_rules, start=1):
        audit[rule.rule_id]["status"] = "selected"
        audit[rule.rule_id]["selection_rank"] = rank

    grades: Dict[str, Tuple[str, ...]] = {}
    for grade, grade_filter in resolved_spec.grade_filters.items():
        grade_ids: List[str] = select_rule_ids(
            validation_evaluation,
            grade_filter,
            primary_target=target,
        )
        grades[str(grade)] = tuple(rule_id for rule_id in grade_ids if rule_id in selected_ids)
    qualification: RuleQualification = _resolve_qualification(
        resolved_spec,
        selected_ids,
        validation_diagnostics,
    )
    validation_summary: Dict[str, Any] = _build_validation_summary(
        resolved_spec,
        qualification,
        validation_status,
        selected_ids,
        validation_diagnostics,
    )
    rule_set = MarsRuleSet(
        selected_rules,
        grades=grades,
        metadata={
            "direction": resolved_spec.direction,
            "target": target,
            "validation_status": validation_status,
        },
        qualification=qualification,
        validation_summary=validation_summary,
    )
    evaluation = MarsRuleEvaluation(
        overall_table=_concat_tables(
            train_evaluation.overall_table,
            validation_evaluation.overall_table,
        ),
        slice_table=_concat_tables(
            train_evaluation.slice_table,
            validation_evaluation.slice_table,
        ),
        metadata={"primary_target": target, "targets": targets},
    )
    elapsed: float = time.perf_counter() - start_time
    status: str = "success" if selected_rules else "no_rules"
    if status == "no_rules":
        warnings.warn("没有候选规则通过最终筛选；返回可审计空结果。", UserWarning, stacklevel=2)
    metadata: Dict[str, Any] = {
        "mars_version": mars_version,
        "source_project": "deimos-rule",
        "source_commit": "e6714c5e795054e44f0c58ad7097668b4117b4a2",
        "target": target,
        "aux_targets": list(aux_targets or []),
        "features": resolved_features,
        "validation_status": validation_status,
        "profile": resolved_spec.profile,
        "qualification": qualification,
        "group_col": group_col,
        "time_col": time_col,
        "time_grain": time_grain,
        "amount_col": amount_col,
        "customer_col": customer_col,
        "generation_errors": generation_errors,
        "generator_diagnostics": _generator_diagnostics(active_generators),
        "elapsed_seconds": elapsed,
        "resolved_spec": resolved_spec.to_dict(),
    }
    return MarsRuleMiningResult(
        status=status,
        rule_set=rule_set,
        candidate_table=pl.DataFrame(list(audit.values())) if audit else pl.DataFrame(),
        evaluation=evaluation,
        spec=resolved_spec,
        metadata=metadata,
    )


def _generate_candidates(
    frame: pl.DataFrame,
    *,
    target: str,
    features: Sequence[str],
    seed_rules: Sequence[Union[str, MarsRule]] | None,
    generators: Sequence[MarsRuleGenerator],
    on_generator_error: str,
    max_candidates: int,
) -> Tuple[
    List[MarsRule],
    Dict[str, Set[str]],
    Dict[str, int],
    List[Dict[str, str]],
    Dict[str, Dict[str, int | None]],
]:
    """生成候选并按 seed 优先、生成器轮询顺序分配预算。"""
    generator_streams: List[List[MarsRule]] = []
    generation_errors: List[Dict[str, str]] = []
    for generator in generators:
        try:
            generator_streams.append(
                list(
                    generator.generate(
                        frame,
                        target=target,
                        features=features,
                    )
                )
            )
        except Exception as exc:
            if on_generator_error == "raise":
                raise
            generator_streams.append([])
            generation_errors.append(
                {
                    "generator": type(generator).__name__,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    seeds: List[MarsRule] = []
    for seed in seed_rules or ():
        if not isinstance(seed, (str, MarsRule)):
            raise TypeError("seed_rules 只接受 DSL 字符串或 MarsRule。")
        seeds.append(seed if isinstance(seed, MarsRule) else MarsRule(seed, source="seed"))

    unique_seeds, _, _ = _merge_rule_sources(seeds)
    if len(unique_seeds) > max_candidates:
        raise ValueError(
            f"唯一 seed 规则数 {len(unique_seeds)} 超过 max_candidates={max_candidates}。"
        )
    generated: List[MarsRule] = [
        rule for stream in generator_streams for rule in stream
    ] + seeds
    unique, sources, duplicate_counts = _merge_rule_sources(generated)
    unique_map: Dict[str, MarsRule] = {rule.rule_id: rule for rule in unique}
    selected_ids: Set[str] = {rule.rule_id for rule in unique_seeds}
    ordered_ids: List[str] = [rule.rule_id for rule in unique_seeds]
    stream_positions: List[int] = [0] * len(generator_streams)
    source_positions: Dict[str, int] = {}
    for stream in generator_streams:
        for position, rule in enumerate(stream, start=1):
            source_positions.setdefault(rule.rule_id, position)
    for position, rule in enumerate(seeds, start=1):
        source_positions[rule.rule_id] = min(
            position,
            source_positions.get(rule.rule_id, position),
        )

    while len(ordered_ids) < max_candidates:
        progressed: bool = False
        for stream_index, stream in enumerate(generator_streams):
            while stream_positions[stream_index] < len(stream):
                selected_rule: MarsRule = stream[stream_positions[stream_index]]
                stream_positions[stream_index] += 1
                if selected_rule.rule_id in selected_ids:
                    continue
                selected_ids.add(selected_rule.rule_id)
                ordered_ids.append(selected_rule.rule_id)
                progressed = True
                break
            if len(ordered_ids) >= max_candidates:
                break
        if not progressed:
            break
    ordered_ids.extend(rule.rule_id for rule in unique if rule.rule_id not in selected_ids)
    ordered: List[MarsRule] = [unique_map[rule_id] for rule_id in ordered_ids]
    budget_metadata: Dict[str, Dict[str, int | None]] = {
        rule_id: {
            "source_position": source_positions.get(rule_id),
            "budget_position": index + 1 if index < max_candidates else None,
        }
        for index, rule_id in enumerate(ordered_ids)
    }
    return ordered, sources, duplicate_counts, generation_errors, budget_metadata


def _mine_rules_cascade(
    *,
    train: pl.DataFrame,
    validation: pl.DataFrame,
    validation_status: str,
    target: str,
    aux_targets: Sequence[str] | None,
    features: Sequence[str],
    group_col: str | None,
    time_col: str | None,
    time_grain: str | None,
    amount_col: str | None,
    customer_col: str | None,
    seed_rules: Sequence[Union[str, MarsRule]] | None,
    spec: MarsRuleMiningSpec,
    generators: Sequence[MarsRuleGenerator],
    start_time: float,
) -> MarsRuleMiningResult:
    """在训练与验证剩余人群上逐轮重新生成并选择一条规则。"""
    active_train: pl.DataFrame = train
    active_validation: pl.DataFrame = validation
    selected: List[MarsRule] = []
    audit_rows: List[Dict[str, Any]] = []
    generation_errors: List[Dict[str, Any]] = []
    evaluator = MarsRuleEvaluator()
    max_rounds: int = min(spec.max_rounds, spec.top_k)

    for round_index in range(1, max_rounds + 1):
        if active_train.is_empty() or active_validation.is_empty():
            break
        (
            unique_rules,
            sources,
            duplicate_counts,
            round_errors,
            budget_metadata,
        ) = _generate_candidates(
            active_train,
            target=target,
            features=features,
            seed_rules=seed_rules,
            generators=generators,
            on_generator_error=spec.on_generator_error,
            max_candidates=spec.max_candidates,
        )
        generation_errors.extend(
            {**error, "round": round_index} for error in round_errors
        )
        selected_ids: Set[str] = {rule.rule_id for rule in selected}
        unique_rules = [rule for rule in unique_rules if rule.rule_id not in selected_ids]
        if not unique_rules:
            break

        audit: Dict[str, Dict[str, Any]] = {
            rule.rule_id: _new_audit_row(
                rule,
                sources=sources[rule.rule_id],
                exact_duplicate_count=duplicate_counts[rule.rule_id],
                generation_round=round_index,
                source_position=budget_metadata[rule.rule_id]["source_position"],
                budget_position=budget_metadata[rule.rule_id]["budget_position"],
            )
            for rule in unique_rules
        }
        budgeted_rules: List[MarsRule] = unique_rules[: spec.max_candidates]
        for rule in budgeted_rules:
            audit[rule.rule_id]["within_candidate_budget"] = True
        for rule in unique_rules[spec.max_candidates :]:
            audit[rule.rule_id]["within_candidate_budget"] = False
            _reject(audit, rule.rule_id, "candidate_budget", "超过 max_candidates")

        train_evaluation: MarsRuleEvaluation = evaluator.evaluate(
            active_train,
            MarsRuleSet(budgeted_rules),
            target=target,
            aux_targets=aux_targets,
            dataset=f"cascade_train_round_{round_index}",
            amount_col=amount_col,
            customer_col=customer_col,
            batch_size=spec.batch_size,
            direction=spec.direction,
            confidence_level=spec.confidence_level,
        )
        candidate_ids: List[str] = select_rule_ids(
            train_evaluation,
            spec.candidate_filter,
            primary_target=target,
        )
        candidate_id_set: Set[str] = set(candidate_ids)
        for rule in budgeted_rules:
            passed: bool = rule.rule_id in candidate_id_set
            audit[rule.rule_id]["candidate_filter_passed"] = passed
            if not passed:
                _reject(audit, rule.rule_id, "candidate_filter", "未通过本轮训练候选阈值")

        candidate_rules: List[MarsRule] = [
            rule for rule in budgeted_rules if rule.rule_id in candidate_id_set
        ]
        validation_evaluation: MarsRuleEvaluation = evaluator.evaluate(
            active_validation,
            MarsRuleSet(candidate_rules),
            target=target,
            aux_targets=aux_targets,
            dataset=f"cascade_validation_round_{round_index}",
            group_col=group_col,
            time_col=time_col,
            time_grain=time_grain,
            amount_col=amount_col,
            customer_col=customer_col,
            batch_size=spec.batch_size,
            direction=spec.direction,
            confidence_level=spec.confidence_level,
            compute_statistics=True,
        )
        validated_ids, validation_diagnostics = _select_validation_ids(
            validation_evaluation,
            spec=spec,
            primary_target=target,
        )
        _attach_validation_diagnostics(
            audit,
            validation_evaluation,
            validation_diagnostics,
            primary_target=target,
        )
        validated_id_set: Set[str] = set(validated_ids)
        for rule in candidate_rules:
            passed = rule.rule_id in validated_id_set
            audit[rule.rule_id]["validation_filter_passed"] = passed
            if not passed:
                _reject(
                    audit,
                    rule.rule_id,
                    "validation_filter",
                    "未通过本轮验证阈值或切片通过率",
                )

        ranked_ids: List[str] = _rank_rule_ids(
            validation_evaluation.overall_table,
            validated_ids,
            target,
            spec.direction,
        )
        ranked_rules: List[MarsRule] = _rules_by_ids(candidate_rules, ranked_ids)
        deduplicated, duplicate_ids = _iou_deduplicate(
            active_validation,
            ranked_rules,
            spec.iou_threshold,
            batch_size=spec.iou_batch_size,
        )
        for rule_id in duplicate_ids:
            audit[rule_id]["iou_filter_passed"] = False
            _reject(audit, rule_id, "iou_deduplication", "本轮命中人群与更高优先级规则重叠")
        for rule in deduplicated:
            audit[rule.rule_id]["iou_filter_passed"] = True

        if not deduplicated:
            audit_rows.extend(audit.values())
            break
        chosen: MarsRule = deduplicated[0]
        selected.append(chosen)
        audit[chosen.rule_id].update(
            {
                "status": "selected",
                "selection_rank": len(selected),
                "selection_round": round_index,
            }
        )
        for deferred in deduplicated[1:]:
            audit[deferred.rule_id].update(
                {
                    "status": "deferred",
                    "rejection_stage": "cascade_round",
                    "reason": "本轮仅选择排名第一的规则，下一轮重新生成候选",
                }
            )
        audit_rows.extend(audit.values())

        chosen_mask: pl.Expr = expression_to_polars(
            parse_expression(chosen.expression),
            active_train.schema,
        ).fill_null(False)
        active_train = active_train.filter(~chosen_mask)
        active_validation = active_validation.filter(~chosen_mask)

    final_set: MarsRuleSet = MarsRuleSet(selected)
    train_evaluation = evaluator.evaluate(
        train,
        final_set,
        target=target,
        aux_targets=aux_targets,
        dataset="train",
        amount_col=amount_col,
        customer_col=customer_col,
        batch_size=spec.batch_size,
        direction=spec.direction,
        confidence_level=spec.confidence_level,
    )
    validation_dataset: str = "validation" if validation_status == "independent" else "in_sample"
    validation_evaluation = evaluator.evaluate(
        validation,
        final_set,
        target=target,
        aux_targets=aux_targets,
        dataset=validation_dataset,
        group_col=group_col,
        time_col=time_col,
        time_grain=time_grain,
        amount_col=amount_col,
        customer_col=customer_col,
        batch_size=spec.batch_size,
        direction=spec.direction,
        confidence_level=spec.confidence_level,
        compute_statistics=True,
    )
    grades: Dict[str, Tuple[str, ...]] = {}
    selected_id_set: Set[str] = {rule.rule_id for rule in selected}
    for grade, grade_filter in spec.grade_filters.items():
        grade_ids: List[str] = select_rule_ids(
            validation_evaluation,
            grade_filter,
            primary_target=target,
        )
        grades[str(grade)] = tuple(
            rule_id for rule_id in grade_ids if rule_id in selected_id_set
        )
    final_selected_ids: Set[str] = {rule.rule_id for rule in selected}
    _, final_validation_diagnostics = _select_validation_ids(
        validation_evaluation,
        spec=spec,
        primary_target=target,
    )
    qualification: RuleQualification = _resolve_qualification(
        spec,
        final_selected_ids,
        final_validation_diagnostics,
    )
    validation_summary: Dict[str, Any] = _build_validation_summary(
        spec,
        qualification,
        validation_status,
        final_selected_ids,
        final_validation_diagnostics,
    )
    rule_set = MarsRuleSet(
        selected,
        grades=grades,
        metadata={
            "direction": spec.direction,
            "target": target,
            "validation_status": validation_status,
            "selection_strategy": "cascade",
        },
        qualification=qualification,
        validation_summary=validation_summary,
    )
    evaluation = MarsRuleEvaluation(
        overall_table=_concat_tables(
            train_evaluation.overall_table,
            validation_evaluation.overall_table,
        ),
        slice_table=_concat_tables(
            train_evaluation.slice_table,
            validation_evaluation.slice_table,
        ),
        metadata={"primary_target": target, "targets": [target, *(aux_targets or [])]},
    )
    status: str = "success" if selected else "no_rules"
    if status == "no_rules":
        warnings.warn(
            "没有候选规则通过 cascade 最终筛选；返回可审计空结果。",
            UserWarning,
            stacklevel=2,
        )
    elapsed: float = time.perf_counter() - start_time
    metadata: Dict[str, Any] = {
        "mars_version": mars_version,
        "source_project": "deimos-rule",
        "source_commit": "e6714c5e795054e44f0c58ad7097668b4117b4a2",
        "target": target,
        "aux_targets": list(aux_targets or []),
        "features": list(features),
        "validation_status": validation_status,
        "profile": spec.profile,
        "qualification": qualification,
        "group_col": group_col,
        "time_col": time_col,
        "time_grain": time_grain,
        "amount_col": amount_col,
        "customer_col": customer_col,
        "generation_errors": generation_errors,
        "generator_diagnostics": _generator_diagnostics(generators),
        "elapsed_seconds": elapsed,
        "cascade_rounds": len(selected),
        "resolved_spec": spec.to_dict(),
    }
    return MarsRuleMiningResult(
        status=status,
        rule_set=rule_set,
        candidate_table=pl.DataFrame(audit_rows) if audit_rows else pl.DataFrame(),
        evaluation=evaluation,
        spec=spec,
        metadata=metadata,
    )


def _new_audit_row(
    rule: MarsRule,
    *,
    sources: Set[str],
    exact_duplicate_count: int,
    generation_round: int,
    source_position: int | None,
    budget_position: int | None,
) -> Dict[str, Any]:
    """构造单轮 cascade 候选的完整审计行。"""
    return {
        "rule_id": rule.rule_id,
        "expression": rule.expression,
        "sources": ",".join(sorted(sources)),
        "distinct_source_count": len(sources),
        "exact_duplicate_count": exact_duplicate_count,
        "generation_round": generation_round,
        "source_position": source_position,
        "budget_position": budget_position,
        "budget_strategy": "seed_then_round_robin",
        "within_candidate_budget": None,
        "candidate_filter_passed": None,
        "validation_filter_passed": None,
        "iou_filter_passed": None,
        "selection_rank": None,
        "selection_round": None,
        "status": "candidate",
        "rejection_stage": None,
        "reason": None,
        "q_value": None,
        "lift_ci_lower": None,
        "lift_ci_upper": None,
        "temporal_assessed": None,
        "temporal_passed": None,
        "time_slice_count": None,
        "time_slice_pass_rate": None,
    }


def _merge_rule_sources(
    rules: Sequence[MarsRule],
) -> Tuple[List[MarsRule], Dict[str, Set[str]], Dict[str, int]]:
    """精确去重候选并合并来源。"""
    unique: Dict[str, MarsRule] = {}
    sources: Dict[str, Set[str]] = {}
    seen_counts: Dict[str, int] = {}
    for rule in rules:
        existing: MarsRule | None = unique.get(rule.rule_id)
        if existing is not None and existing.expression != rule.expression:
            raise ValueError(f"规则 ID 哈希冲突：{rule.rule_id}。")
        unique.setdefault(rule.rule_id, rule)
        sources.setdefault(rule.rule_id, set()).add(rule.source)
        seen_counts[rule.rule_id] = seen_counts.get(rule.rule_id, 0) + 1
    duplicate_counts: Dict[str, int] = {
        rule_id: count - 1 for rule_id, count in seen_counts.items()
    }
    return list(unique.values()), sources, duplicate_counts


def _reject(
    audit: Dict[str, Dict[str, Any]],
    rule_id: str,
    stage: str,
    reason: str,
) -> None:
    """只记录候选第一次被淘汰的阶段。"""
    row: Dict[str, Any] = audit[rule_id]
    if row["status"] == "candidate":
        row.update({"status": "rejected", "rejection_stage": stage, "reason": reason})


def _rank_rule_ids(
    table: pl.DataFrame,
    rule_ids: Sequence[str],
    target: str,
    direction: str,
) -> List[str]:
    """按方向和确定性 ID 排序验证规则。"""
    if not rule_ids:
        return []
    rows: pl.DataFrame = table.filter(
        pl.col("rule_id").is_in(list(rule_ids))
        & (pl.col("target") == target)
        & (pl.col("group") == "hit")
    )
    records: List[Dict[str, Any]] = rows.to_dicts()

    def key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        """生成高/低风险稳定排序键。"""
        lift: float = float(row["lift"]) if row.get("lift") is not None else float("inf")
        if direction == "high_risk":
            return (-lift, -int(row["event_count"]), str(row["rule_id"]))
        return (lift, -int(row["sample_count"]), str(row["rule_id"]))

    return [str(row["rule_id"]) for row in sorted(records, key=key)]


def _rules_by_ids(rules: Sequence[MarsRule], rule_ids: Sequence[str]) -> List[MarsRule]:
    """按 rule_id 顺序恢复规则对象。"""
    rule_map: Dict[str, MarsRule] = {rule.rule_id: rule for rule in rules}
    return [rule_map[rule_id] for rule_id in rule_ids]


def _select_validation_ids(
    evaluation: MarsRuleEvaluation,
    *,
    spec: MarsRuleMiningSpec,
    primary_target: str,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """按 profile 选择验证规则并返回生产门禁诊断。"""
    if spec.profile == "production":
        return select_production_rule_ids(
            evaluation,
            spec.validation_filter,
            primary_target=primary_target,
            direction=spec.direction,
            max_fdr=spec.max_fdr,
            min_time_slices=spec.min_time_slices,
        )
    selected: List[str] = select_rule_ids(
        evaluation,
        spec.validation_filter,
        primary_target=primary_target,
    )
    return selected, {}


def _attach_validation_diagnostics(
    audit: Dict[str, Dict[str, Any]],
    evaluation: MarsRuleEvaluation,
    diagnostics: Mapping[str, Mapping[str, Any]],
    *,
    primary_target: str,
) -> None:
    """把主目标统计量与时间门禁结果写入候选审计。"""
    if evaluation.overall_table.is_empty():
        return
    rows: pl.DataFrame = evaluation.overall_table.filter(
        (pl.col("target") == primary_target) & (pl.col("group") == "hit")
    )
    metric_rows: Dict[str, Dict[str, Any]] = {
        str(row["rule_id"]): dict(row) for row in rows.iter_rows(named=True)
    }
    for rule_id, audit_row in audit.items():
        metric_row: Mapping[str, Any] = metric_rows.get(rule_id, {})
        diagnostic: Mapping[str, Any] = diagnostics.get(rule_id, {})
        audit_row.update(
            {
                "q_value": metric_row.get("q_value"),
                "lift_ci_lower": metric_row.get("lift_ci_lower"),
                "lift_ci_upper": metric_row.get("lift_ci_upper"),
                "temporal_assessed": diagnostic.get("temporal_assessed"),
                "temporal_passed": diagnostic.get("temporal_passed"),
                "time_slice_count": diagnostic.get("time_slice_count"),
                "time_slice_pass_rate": diagnostic.get("time_slice_pass_rate"),
            }
        )


def _resolve_qualification(
    spec: MarsRuleMiningSpec,
    selected_ids: Set[str],
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> RuleQualification:
    """根据 profile 和最终规则的时间门禁状态确定部署资格。"""
    if spec.profile == "explore":
        return "exploratory"
    temporal_assessed: bool = bool(selected_ids) and all(
        bool(diagnostics.get(rule_id, {}).get("temporal_assessed"))
        for rule_id in selected_ids
    )
    return "temporally_validated" if temporal_assessed else "validated"


def _build_validation_summary(
    spec: MarsRuleMiningSpec,
    qualification: RuleQualification,
    validation_status: str,
    selected_ids: Set[str],
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """构造可序列化的资格验证摘要。"""
    slice_counts: List[int] = [
        int(diagnostics.get(rule_id, {}).get("time_slice_count") or 0)
        for rule_id in selected_ids
    ]
    pass_rates: List[float] = [
        float(value)
        for rule_id in selected_ids
        if (value := diagnostics.get(rule_id, {}).get("time_slice_pass_rate"))
        is not None
    ]
    return {
        "profile": spec.profile,
        "qualification": qualification,
        "validation_status": validation_status,
        "confidence_level": spec.confidence_level,
        "max_fdr": spec.max_fdr,
        "min_time_slices": spec.min_time_slices,
        "selected_count": len(selected_ids),
        "temporal_assessed": qualification == "temporally_validated",
        "minimum_time_slice_count": min(slice_counts) if slice_counts else 0,
        "minimum_time_slice_pass_rate": min(pass_rates) if pass_rates else None,
    }


def _iou_deduplicate(
    frame: pl.DataFrame,
    rules: Sequence[MarsRule],
    threshold: float,
    *,
    batch_size: int,
) -> Tuple[List[MarsRule], List[str]]:
    """使用压缩位图按优先级执行命中人群 IoU 去重。"""
    if batch_size < 1:
        raise ValueError("IoU batch_size 必须至少为 1。")
    kept: List[MarsRule] = []
    kept_masks: List[np.ndarray] = []
    kept_counts: List[int] = []
    duplicate_ids: List[str] = []
    for start in range(0, len(rules), batch_size):
        batch_rules: Sequence[MarsRule] = rules[start : start + batch_size]
        expressions: List[pl.Expr] = [
            expression_to_polars(parse_expression(rule.expression), frame.schema)
            .fill_null(False)
            .alias(f"mask_{index}")
            for index, rule in enumerate(batch_rules)
        ]
        matrix: np.ndarray = np.asarray(frame.select(expressions).to_numpy(), dtype=bool)
        packed_matrix: np.ndarray = np.packbits(matrix, axis=0)
        for index, rule in enumerate(batch_rules):
            mask: np.ndarray = packed_matrix[:, index]
            mask_count: int = int(_POPCOUNT_TABLE[mask].sum())
            duplicate: bool = False
            for kept_mask, kept_count in zip(kept_masks, kept_counts):
                intersection: int = int(
                    _POPCOUNT_TABLE[np.bitwise_and(mask, kept_mask)].sum()
                )
                union: int = mask_count + kept_count - intersection
                iou: float = intersection / union if union else 0.0
                if iou >= threshold:
                    duplicate = True
                    break
            if duplicate:
                duplicate_ids.append(rule.rule_id)
            else:
                kept.append(rule)
                kept_masks.append(mask.copy())
                kept_counts.append(mask_count)
    return kept, duplicate_ids


def _concat_tables(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    """安全拼接可能为空的同 schema 表。"""
    if left.is_empty():
        return right
    if right.is_empty():
        return left
    return pl.concat([left, right], how="vertical")


def _optional_metadata_column(metadata: Mapping[str, Any], key: str) -> str | None:
    """从结果元数据读取可选列名。"""
    value: Any = metadata.get(key)
    return str(value) if value else None


def _build_rule_explanations(result: MarsRuleMiningResult) -> pl.DataFrame:
    """把最终验证指标整理为稳定的中文规则解释表。"""
    if not result.rule_set.rules or result.evaluation.overall_table.is_empty():
        return pl.DataFrame()
    dataset: str = (
        "validation"
        if result.metadata.get("validation_status") == "independent"
        else "in_sample"
    )
    target: str = str(result.metadata["target"])
    hit_rows: pl.DataFrame = result.evaluation.overall_table.filter(
        (pl.col("dataset") == dataset)
        & (pl.col("target") == target)
        & (pl.col("group") == "hit")
    )
    metrics_by_rule: Dict[str, Dict[str, Any]] = {
        str(row["rule_id"]): row for row in hit_rows.to_dicts()
    }
    rows: List[Dict[str, Any]] = []
    for rank, rule in enumerate(result.rule_set.rules, start=1):
        metrics: Dict[str, Any] = metrics_by_rule.get(rule.rule_id, {})
        sample_count: Any = metrics.get("sample_count")
        event_count: Any = metrics.get("event_count")
        coverage: Any = metrics.get("coverage")
        event_rate: Any = metrics.get("event_rate")
        lift: Any = metrics.get("lift")
        lift_ci_lower: Any = metrics.get("lift_ci_lower")
        lift_ci_upper: Any = metrics.get("lift_ci_upper")
        q_value: Any = metrics.get("q_value")
        text: str = (
            f"第 {rank} 条规则命中 {_format_metric(sample_count, 0)} 个样本，"
            f"覆盖率 {_format_metric(coverage, 2, percent=True)}，"
            f"事件数 {_format_metric(event_count, 0)}，"
            f"事件率 {_format_metric(event_rate, 2, percent=True)}，"
            f"Lift {_format_metric(lift, 3)}，"
            f"95% 保守区间 [{_format_metric(lift_ci_lower, 3)}, "
            f"{_format_metric(lift_ci_upper, 3)}]，"
            f"q 值 {_format_metric(q_value, 4)}。"
        )
        rows.append(
            {
                "rank": rank,
                "rule_id": rule.rule_id,
                "expression": rule.expression,
                "source": rule.source,
                "complexity": rule.complexity,
                "sample_count": sample_count,
                "event_count": event_count,
                "coverage": coverage,
                "event_rate": event_rate,
                "lift": lift,
                "lift_ci_lower": lift_ci_lower,
                "lift_ci_upper": lift_ci_upper,
                "p_value": metrics.get("p_value"),
                "q_value": q_value,
                "qualification": result.rule_set.qualification,
                "amount_total": metrics.get("amount_total"),
                "amount_event_rate": metrics.get("amount_event_rate"),
                "customer_count": metrics.get("customer_count"),
                "customer_event_rate": metrics.get("customer_event_rate"),
                "explanation": text,
            }
        )
    return pl.DataFrame(rows)


def _format_metric(value: Any, digits: int, *, percent: bool = False) -> str:
    """把可空数值格式化为解释文本。"""
    if value is None:
        return "N/A"
    number: float = float(value) * (100.0 if percent else 1.0)
    suffix: str = "%" if percent else ""
    return f"{number:.{digits}f}{suffix}"


def _generator_diagnostics(
    generators: Sequence[MarsRuleGenerator],
) -> List[Dict[str, Any]]:
    """收集生成器公开诊断属性，避免结果对象持有生成器实例。"""
    diagnostics: List[Dict[str, Any]] = []
    for generator in generators:
        tree_metadata: Any = getattr(generator, "tree_metadata_", None)
        if tree_metadata:
            diagnostics.append(
                {
                    "generator": type(generator).__name__,
                    "tree_metadata": list(tree_metadata),
                }
            )
    return diagnostics
