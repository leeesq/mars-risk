"""规则评估与高层挖掘工作流测试。"""

from __future__ import annotations

from typing import List, Sequence

import polars as pl
import pytest

from mars.compute import FrameLike
from mars.rule import (
    MarsRule,
    MarsRuleEvaluator,
    MarsRuleFilter,
    MarsRuleGenerator,
    MarsRuleMetricCondition,
    MarsRuleMiningSpec,
    MarsRuleSet,
    mine_rules,
)


def _signal_frame() -> pl.DataFrame:
    """构造具有稳定高低风险区间的确定性样本。"""
    values = list(range(200))
    return pl.DataFrame(
        {
            "x": values,
            "cycle": [value % 100 for value in values],
            "target": [int(value % 100 >= 80) for value in values],
            "aux": [int(value % 100 >= 70) for value in values],
            "month": ["2026-01-15"] * 100 + ["2026-02-15"] * 100,
            "amount": [100.0] * 200,
            "customer": [f"c{value}" for value in values],
        }
    )


def test_evaluator_fixed_long_table_metrics_and_null_targets() -> None:
    """评估器应排除各目标空值并计算样本、金额和客户指标。"""
    frame = _signal_frame().with_columns(
        pl.when(pl.col("x") == 199).then(None).otherwise(pl.col("target")).alias("target")
    )
    rule = MarsRule("x >= 180")
    evaluation = MarsRuleEvaluator().evaluate(
        frame,
        MarsRuleSet([rule]),
        target="target",
        aux_targets=["aux"],
        dataset="validation",
        time_col="month",
        time_grain="month",
        amount_col="amount",
        customer_col="customer",
    )
    hit = evaluation.overall_table.filter(
        (pl.col("target") == "target") & (pl.col("group") == "hit")
    ).row(0, named=True)

    assert evaluation.overall_table.height == 6
    assert evaluation.slice_table.height == 12
    assert hit["sample_count"] == 19
    assert hit["event_count"] == 19
    assert hit["amount_total"] == 1900.0
    assert hit["customer_count"] == 19
    assert hit["lift"] == pytest.approx(199 / 39)


def test_evaluator_zero_denominators_are_null_and_fail_filter() -> None:
    """空命中产生 null 比率，结构化筛选必须视为不通过。"""
    frame = _signal_frame()
    rule = MarsRule("x > 999")
    evaluation = MarsRuleEvaluator().evaluate(frame, MarsRuleSet([rule]), target="target")
    hit = evaluation.overall_table.filter(pl.col("group") == "hit").row(0, named=True)
    strict_filter = MarsRuleFilter((MarsRuleMetricCondition("lift", ">=", 1.0),))

    assert hit["sample_count"] == 0
    assert hit["event_rate"] is None
    from mars.rule.evaluator import select_rule_ids

    assert select_rule_ids(evaluation, strict_filter, primary_target="target") == []


def test_mine_rules_uses_validation_and_returns_auditable_result() -> None:
    """seed 候选应依次通过训练、验证、排序并进入可部署 RuleSet。"""
    frame = _signal_frame()
    result = mine_rules(
        frame,
        target="target",
        validation_df=frame,
        aux_targets=["aux"],
        seed_rules=["x >= 180", "x < 100"],
        generators=[],
    )

    assert result.status == "success"
    assert [rule.expression for rule in result.rule_set.rules] == ['"x" >= 180']
    assert result.metadata["validation_status"] == "independent"
    assert "train" in set(result.evaluation.overall_table["dataset"])
    assert "validation" in set(result.evaluation.overall_table["dataset"])
    rejected = result.candidate_table.filter(pl.col("status") == "rejected")
    assert rejected["rejection_stage"].to_list() == ["candidate_filter"]
    assert not any(value.__class__.__name__.endswith("DataFrame") for value in result.metadata.values())


def test_mine_rules_without_validation_warns_and_marks_in_sample() -> None:
    """缺少验证集必须显式降级，但仍允许返回结果。"""
    with pytest.warns(UserWarning, match="validation_df"):
        result = mine_rules(
            _signal_frame(),
            target="target",
            seed_rules=["x >= 180"],
            generators=[],
        )
    assert result.status == "success"
    assert result.metadata["validation_status"] == "in_sample"
    assert result.rule_set.qualification == "exploratory"


def test_production_requires_independent_validation() -> None:
    """production profile 不得退化为样本内资格。"""
    with pytest.raises(ValueError, match="validation_df"):
        mine_rules(
            _signal_frame(),
            target="target",
            seed_rules=["x >= 180"],
            generators=[],
            spec=MarsRuleMiningSpec.production(),
        )


def test_production_applies_statistical_gate_and_qualifies_ruleset() -> None:
    """独立验证上的显著高风险规则应通过 FDR 和保守 Lift 门禁。"""
    frame = _signal_frame()
    result = mine_rules(
        frame,
        target="target",
        validation_df=frame.clone(),
        seed_rules=["x >= 180"],
        generators=[],
        spec=MarsRuleMiningSpec.production(),
    )

    assert result.status == "success"
    assert result.rule_set.qualification == "validated"
    selected = result.candidate_table.filter(pl.col("status") == "selected").row(
        0,
        named=True,
    )
    assert selected["q_value"] <= 0.05
    assert selected["lift_ci_lower"] >= 2.0


def test_production_time_slices_upgrade_qualification() -> None:
    """至少三个稳定时间切片应把规则集升级为 temporally_validated。"""
    values = list(range(300))
    frame = pl.DataFrame(
        {
            "x": values,
            "cycle": [value % 100 for value in values],
            "target": [int(value % 100 >= 80) for value in values],
            "month": [f"2026-{value // 100 + 1:02d}-15" for value in values],
        }
    )
    result = mine_rules(
        frame,
        target="target",
        validation_df=frame.clone(),
        time_col="month",
        time_grain="month",
        seed_rules=["cycle >= 80"],
        generators=[],
        spec=MarsRuleMiningSpec.production(),
    )

    assert result.rule_set.qualification == "temporally_validated"


def test_validation_data_controls_final_selection() -> None:
    """训练集通过不能替代独立验证集的最终筛选。"""
    train = _signal_frame()
    validation = train.with_columns((1 - pl.col("target")).alias("target"))
    with pytest.warns(UserWarning, match="没有候选"):
        result = mine_rules(
            train,
            target="target",
            validation_df=validation,
            seed_rules=["x >= 180"],
            generators=[],
        )
    assert result.status == "no_rules"
    assert result.candidate_table["candidate_filter_passed"][0]
    assert not result.candidate_table["validation_filter_passed"][0]
    assert result.candidate_table["rejection_stage"][0] == "validation_filter"


def test_time_slice_pass_rate_rejects_unstable_rule() -> None:
    """提供时间切片后，默认至少 80% 切片必须通过验证阈值。"""
    frame = _signal_frame()
    with pytest.warns(UserWarning, match="没有候选"):
        result = mine_rules(
            frame,
            target="target",
            validation_df=frame,
            time_col="month",
            time_grain="month",
            seed_rules=["x >= 180"],
            generators=[],
        )
    assert result.status == "no_rules"
    assert result.candidate_table["rejection_stage"][0] == "validation_filter"


def test_mine_rules_legal_no_rules_result() -> None:
    """业务零入选应返回空 RuleSet 和淘汰审计而不是异常。"""
    with pytest.warns(UserWarning, match="没有候选"):
        result = mine_rules(
            _signal_frame(),
            target="target",
            validation_df=_signal_frame(),
            seed_rules=["x > 999"],
            generators=[],
        )
    assert result.status == "no_rules"
    assert not result.rule_set.rules
    assert result.rule_set.transform(pl.DataFrame({"x": [1]}))["rule_hit_count"][0] == 0
    assert result.candidate_table["rejection_stage"][0] == "candidate_filter"


class _FailingGenerator(MarsRuleGenerator):
    """用于验证生成器失败策略的测试生成器。"""

    def generate(
        self,
        df: FrameLike,
        *,
        target: str,
        features: Sequence[str] | None = None,
    ) -> List[MarsRule]:
        """始终抛出确定性生成错误。"""
        raise RuntimeError("boom")


class _RegeneratingGenerator(MarsRuleGenerator):
    """根据剩余人群生成不同候选，用于锁定 cascade 再生成语义。"""

    def __init__(self) -> None:
        self.sample_counts: List[int] = []

    def generate(
        self,
        df: FrameLike,
        *,
        target: str,
        features: Sequence[str] | None = None,
    ) -> List[MarsRule]:
        """首轮和后续轮次返回不同规则。"""
        sample_count: int = len(df)
        self.sample_counts.append(sample_count)
        expression: str = "x >= 180" if sample_count == 200 else "x >= 80 AND x < 100"
        return [MarsRule(expression, source="regenerating")]


class _StaticGenerator(MarsRuleGenerator):
    """按固定优先级返回候选，用于验证生成器间预算公平性。"""

    def __init__(self, expressions: Sequence[str], source: str) -> None:
        self.expressions = list(expressions)
        self.source = source

    def generate(
        self,
        df: FrameLike,
        *,
        target: str,
        features: Sequence[str] | None = None,
    ) -> List[MarsRule]:
        """返回固定顺序候选。"""
        return [MarsRule(expression, source=self.source) for expression in self.expressions]


def test_generator_errors_raise_by_default_or_are_recorded() -> None:
    """部分成功只能由 on_generator_error=record 显式开启。"""
    frame = _signal_frame()
    with pytest.raises(RuntimeError, match="boom"):
        mine_rules(
            frame,
            target="target",
            validation_df=frame,
            generators=[_FailingGenerator()],
        )

    record_spec = MarsRuleMiningSpec(on_generator_error="record")
    with pytest.warns(UserWarning, match="没有候选"):
        result = mine_rules(
            frame,
            target="target",
            validation_df=frame,
            spec=record_spec,
            generators=[_FailingGenerator()],
        )
    assert result.metadata["generation_errors"][0]["generator"] == "_FailingGenerator"


def test_candidate_budget_prioritizes_seed_and_round_robins_generators() -> None:
    """seed 应优先占位，剩余额度不得被第一个生成器垄断。"""
    first = _StaticGenerator(["x >= 180", "x >= 181"], "first")
    second = _StaticGenerator(["x >= 185", "x >= 186"], "second")
    result = mine_rules(
        _signal_frame(),
        target="target",
        validation_df=_signal_frame(),
        seed_rules=["x >= 190"],
        generators=[first, second],
        spec=MarsRuleMiningSpec(max_candidates=3, iou_threshold=1.0),
    )

    budgeted = result.candidate_table.filter(pl.col("within_candidate_budget")).sort(
        "budget_position"
    )
    assert budgeted["sources"].to_list() == ["seed", "first", "second"]
    assert budgeted["budget_position"].to_list() == [1, 2, 3]


def test_seed_count_cannot_silently_exceed_candidate_budget() -> None:
    """唯一 seed 超出总预算时必须报错而不是截断用户输入。"""
    with pytest.raises(ValueError, match="seed"):
        mine_rules(
            _signal_frame(),
            target="target",
            validation_df=_signal_frame(),
            seed_rules=["x >= 180", "x >= 190"],
            generators=[],
            spec=MarsRuleMiningSpec(max_candidates=1),
        )


def test_low_risk_spec_selects_low_lift_rule() -> None:
    """低风险工厂应按 Lift 升序筛选和排序。"""
    frame = _signal_frame()
    result = mine_rules(
        frame,
        target="target",
        validation_df=frame,
        seed_rules=["x < 80", "x < 60"],
        generators=[],
        spec=MarsRuleMiningSpec.low_risk(top_k=1),
    )
    assert result.status == "success"
    assert result.rule_set.rules[0].expression == '"x" < 80'


def test_cascade_re_evaluates_remaining_population() -> None:
    """cascade 应逐轮删除已命中人群并受 max_rounds/top_k 约束。"""
    frame = _signal_frame()
    result = mine_rules(
        frame,
        target="target",
        validation_df=frame,
        seed_rules=["x >= 180", "x >= 190"],
        generators=[],
        spec=MarsRuleMiningSpec(
            selection_strategy="cascade",
            iou_threshold=1.0,
            max_rounds=2,
            top_k=2,
        ),
    )
    assert result.status == "success"
    assert len(result.rule_set.rules) == 1
    assert result.candidate_table.filter(pl.col("status") == "selected")[
        "selection_rank"
    ].to_list() == [1]


def test_cascade_regenerates_candidates_on_each_remaining_population() -> None:
    """cascade 每轮必须重新调用生成器，而不是只重评首轮固定候选。"""
    frame = _signal_frame()
    generator = _RegeneratingGenerator()
    result = mine_rules(
        frame,
        target="target",
        validation_df=frame,
        generators=[generator],
        spec=MarsRuleMiningSpec(
            selection_strategy="cascade",
            iou_threshold=1.0,
            max_rounds=2,
            top_k=2,
        ),
    )

    assert generator.sample_counts == [200, 180]
    assert [rule.expression for rule in result.rule_set.rules] == [
        '"x" >= 180',
        '("x" < 100 AND "x" >= 80)',
    ]
    selected = result.candidate_table.filter(pl.col("status") == "selected")
    assert selected["generation_round"].to_list() == [1, 2]
    assert selected["selection_round"].to_list() == [1, 2]
