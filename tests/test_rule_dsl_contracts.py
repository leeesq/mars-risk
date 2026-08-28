"""规则 DSL、RuleSet 和 artifact 契约测试。"""

from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import mars
from mars.rule import (
    MarsRule,
    MarsRuleFilter,
    MarsRuleMetricCondition,
    MarsRuleMiningSpec,
    MarsRuleSet,
)
from mars.rule.exceptions import (
    MarsRuleArtifactError,
    MarsRuleDeploymentError,
    MarsRuleSchemaError,
    MarsRuleSyntaxError,
)


def test_dsl_normalization_is_deterministic_and_simplifies_duplicates() -> None:
    """逻辑顺序和重复条件不应改变规则 ID。"""
    left = MarsRule("score > 10 AND age >= 18 AND score > 10")
    right = MarsRule("age >= 18 and score > 10")

    assert left.expression == right.expression
    assert left.rule_id == right.rule_id
    assert left.complexity == 2
    bounded = MarsRule("age > 20 AND age >= 30 AND age <= 50 AND age < 40")
    assert bounded.expression == '("age" < 40 AND "age" >= 30)'


@pytest.mark.parametrize(
    "expression",
    [
        "income > 10000 AND income < 5000",
        "flag = 1 AND flag = 2",
        "value IS NULL AND value > 0",
        "value IS NULL AND value IS NOT NULL",
    ],
)
def test_dsl_rejects_static_contradictions(expression: str) -> None:
    """可静态证明不可能成立的规则必须 fail closed。"""
    with pytest.raises(MarsRuleSyntaxError, match="矛盾"):
        MarsRule(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "abs(score) > 1",
        "score + 1 > 3",
        "score IN (1, 2)",
        "score >",
        "SELECT score FROM data",
        "",
    ],
)
def test_dsl_rejects_non_v2_syntax(expression: str) -> None:
    """函数、算术、集合、子查询和不完整表达式不属于 DSL v2。"""
    with pytest.raises(MarsRuleSyntaxError):
        MarsRule(expression)


def test_dsl_precedence_null_and_escaping_execute_in_polars() -> None:
    """AND 优先级、引号转义和 null 语义应统一编译。"""
    frame = pl.DataFrame(
        {
            "customer name": ["O'Brien", "Alice", "O'Brien"],
            "score": [None, 5, 20],
            "active": [False, True, False],
        }
    )
    rule = MarsRule(
        '"customer name" = \'O\'\'Brien\' AND score IS NULL OR active = TRUE'
    )
    result = MarsRuleSet([rule]).transform(frame)

    assert result[f"rule__{rule.rule_id}"].to_list() == [1, 1, 0]
    assert "'O''Brien'" in rule.expression


def test_dsl_distinguishes_null_nan_and_missing() -> None:
    """NULL 只匹配 null，MISSING 额外匹配浮点 NaN，普通比较不得命中缺失。"""
    frame = pl.DataFrame({"x": [None, float("nan"), 2.0]})
    null_rule = MarsRule("x IS NULL")
    missing_rule = MarsRule("x IS MISSING")
    comparison_rule = MarsRule("x > 1")

    result = MarsRuleSet([null_rule, missing_rule, comparison_rule]).transform(frame)

    assert result[f"rule__{null_rule.rule_id}"].to_list() == [1, 0, 0]
    assert result[f"rule__{missing_rule.rule_id}"].to_list() == [1, 1, 0]
    assert result[f"rule__{comparison_rule.rule_id}"].to_list() == [0, 0, 1]


def test_dsl_schema_type_mismatch_fails_closed() -> None:
    """规则字面量与部署列类型不兼容时必须在计算前失败。"""
    with pytest.raises(MarsRuleSchemaError, match="不兼容"):
        MarsRuleSet([MarsRule("x > 1")]).transform(pl.DataFrame({"x": ["high"]}))


def test_dsl_rejects_pathological_nesting_and_length() -> None:
    """表达式资源上限必须在编译前 fail closed。"""
    nested = "NOT (" * 33 + "x > 1" + ")" * 33
    with pytest.raises(MarsRuleSyntaxError, match="深度"):
        MarsRule(nested)
    with pytest.raises(MarsRuleSyntaxError, match="长度"):
        MarsRule("x = '" + "a" * 16_384 + "'")


def test_ruleset_transform_preserves_input_type_and_counts() -> None:
    """RuleSet 应返回同类型输入并生成稳定命中列。"""
    first = MarsRule("x >= 2")
    second = MarsRule("x IS NULL")
    rule_set = MarsRuleSet(
        [first, second],
        grades={"A/high": [first.rule_id], "B": [second.rule_id]},
    )
    polars_result = rule_set.transform(pl.DataFrame({"x": [1, 2, None]}))
    pandas_result = rule_set.transform(pd.DataFrame({"x": [1, 2, None]}))

    assert isinstance(polars_result, pl.DataFrame)
    assert isinstance(pandas_result, pd.DataFrame)
    assert polars_result["rule_hit_count"].to_list() == [0, 1, 1]
    assert pandas_result["grade__A_high__hit_count"].tolist() == [0, 1, 0]
    assert polars_result.schema["rule_hit_count"] == pl.Int32


def test_empty_ruleset_still_emits_total_hit_count() -> None:
    """合法空 RuleSet 也应具有固定部署输出 schema。"""
    result = MarsRuleSet().transform(pl.DataFrame({"x": [1, 2]}))
    assert result["rule_hit_count"].to_list() == [0, 0]


def test_ruleset_missing_columns_fail_closed() -> None:
    """部署数据缺少规则列时不能静默当作未命中。"""
    with pytest.raises(ValueError, match="缺少"):
        MarsRuleSet([MarsRule("missing > 1")]).transform(pl.DataFrame({"x": [1]}))


def test_ruleset_sql_matches_reference_database() -> None:
    """ANSI CASE WHEN 命中列和计数应可在 SQLite 参考执行器运行。"""
    high = MarsRule("x >= 2 AND name = 'O''Brien'")
    missing = MarsRule("x IS NULL")
    rule_set = MarsRuleSet([high, missing], grades={"A": [high.rule_id]})
    connection = sqlite3.connect(":memory:")
    connection.execute('CREATE TABLE data ("x" REAL, "name" TEXT)')
    connection.executemany(
        "INSERT INTO data VALUES (?, ?)",
        [(1, "O'Brien"), (2, "O'Brien"), (None, "Alice")],
    )

    rows = connection.execute(
        f"SELECT {rule_set.generate_sql(table_alias='d', minimum_qualification=None)} "
        "FROM data AS d ORDER BY rowid"
    ).fetchall()
    assert rows == [(0, 0, 0, 0), (1, 0, 1, 1), (0, 1, 1, 0)]


def test_ruleset_sql_blocks_exploratory_and_missing_without_normalization() -> None:
    """探索规则和 MISSING 规则必须分别通过显式部署前提才能导出。"""
    rule_set = MarsRuleSet([MarsRule("x IS MISSING")])
    with pytest.raises(MarsRuleDeploymentError, match="资格"):
        rule_set.generate_sql()
    with pytest.raises(MarsRuleDeploymentError, match="MISSING"):
        rule_set.generate_sql(minimum_qualification=None)

    sql = rule_set.generate_sql(
        minimum_qualification=None,
        missing_policy="normalized_to_null",
    )
    assert '"x" IS NULL' in sql


def test_ruleset_json_round_trip_and_strict_validation(tmp_path: Path) -> None:
    """JSON 往返后保持 ID，并拒绝未知字段、版本和篡改 ID。"""
    rule = MarsRule("x >= 2")
    original = MarsRuleSet([rule], grades={"A": [rule.rule_id]}, metadata={"owner": "risk"})
    path = tmp_path / "rules.json"
    original.save_json(path)
    loaded = MarsRuleSet.load_json(path)
    assert loaded.to_dict() == original.to_dict()

    payload = original.to_dict()
    for mutate in ("version", "id", "unknown", "qualification", "summary"):
        invalid = copy.deepcopy(payload)
        if mutate == "version":
            invalid["expression_version"] = 1
        elif mutate == "id":
            invalid["rules"][0]["rule_id"] = "mr_tampered"
        elif mutate == "qualification":
            invalid["qualification"] = "validated"
        elif mutate == "summary":
            invalid["validation_summary"] = {"profile": "production"}
        else:
            invalid["legacy"] = True
        with pytest.raises(MarsRuleArtifactError):
            MarsRuleSet.from_dict(invalid)


def test_ruleset_rejects_grade_output_name_collision() -> None:
    """不同等级不得映射到同一部署列。"""
    rule = MarsRule("x > 0")
    with pytest.raises(MarsRuleArtifactError, match="重复"):
        MarsRuleSet(
            [rule],
            grades={"A/B": [rule.rule_id], "A?B": [rule.rule_id]},
        )


def test_rule_api_is_only_exported_from_mars_rule() -> None:
    """Experimental 规则入口不能污染根 mars 稳定导出面。"""
    assert not hasattr(mars, "mine_rules")
    assert "mine_rules" not in mars.__all__


def test_filters_and_specs_reject_untyped_legacy_values() -> None:
    """tuple、mapping 和指标字符串不能绕过类型化筛选契约。"""
    with pytest.raises(TypeError, match="MarsRuleMetricCondition"):
        MarsRuleFilter(conditions=(("lift", ">=", 2.0),))
    with pytest.raises(TypeError, match="MarsRuleFilter"):
        MarsRuleMiningSpec(candidate_filter={"lift": 2.0})
    with pytest.raises(ValueError, match="不支持"):
        MarsRuleMetricCondition("lift >= 2", ">=", 2.0)
