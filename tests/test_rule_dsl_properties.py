"""规则 DSL 的属性测试与参考数据库差分测试。"""

from __future__ import annotations

import sqlite3
from typing import Any, List

import polars as pl
import pytest

from mars.rule import MarsRule, MarsRuleSet

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
settings = hypothesis.settings
st = hypothesis.strategies


@st.composite
def _expressions(draw: Any) -> str:
    """生成不会触发静态矛盾消除的受限随机表达式。"""
    identifier: str = draw(st.sampled_from(["x", "y"]))
    operator: str = draw(st.sampled_from(["<", "<=", "=", "!=", ">=", ">"]))
    value: int = draw(st.integers(min_value=-10, max_value=10))
    atom: st.SearchStrategy[str] = st.just(f"{identifier} {operator} {value}")
    expression: str = draw(
        st.recursive(
            atom,
            lambda child: st.one_of(
                child.map(lambda item: f"NOT ({item})"),
                st.tuples(child, child).map(
                    lambda pair: f"({pair[0]}) OR ({pair[1]})"
                ),
            ),
            max_leaves=8,
        )
    )
    return expression


@settings(max_examples=60, deadline=None)
@given(
    expression=_expressions(),
    rows=st.lists(
        st.tuples(
            st.one_of(st.none(), st.integers(-12, 12)),
            st.one_of(st.none(), st.integers(-12, 12)),
        ),
        min_size=1,
        max_size=30,
    ),
)
def test_random_dsl_is_idempotent_and_matches_sqlite(
    expression: str,
    rows: List[tuple[int | None, int | None]],
) -> None:
    """随机合法 AST 的规范化、Polars 和 ANSI SQL 命中结果必须一致。"""
    rule = MarsRule(expression)
    normalized = MarsRule(rule.expression)
    assert normalized.expression == rule.expression
    assert normalized.rule_id == rule.rule_id

    frame = pl.DataFrame(
        rows,
        schema={"x": pl.Int64, "y": pl.Int64},
        orient="row",
    )
    polars_hits = MarsRuleSet([rule]).transform(frame)[f"rule__{rule.rule_id}"].to_list()
    connection = sqlite3.connect(":memory:")
    connection.execute('CREATE TABLE data ("x" INTEGER, "y" INTEGER)')
    connection.executemany("INSERT INTO data VALUES (?, ?)", rows)
    sql = MarsRuleSet([rule]).generate_sql(minimum_qualification=None)
    sqlite_hits = [item[0] for item in connection.execute(f"SELECT {sql} FROM data")]

    assert sqlite_hits == polars_hits
