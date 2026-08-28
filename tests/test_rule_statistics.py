"""规则置信界、精确检验与 FDR 校正测试。"""

from __future__ import annotations

import polars as pl
import pytest

from mars.rule._statistics import add_statistical_metrics, benjamini_hochberg


def test_known_high_risk_contingency_table_metrics() -> None:
    """已知 2×2 表应锁定单侧精确检验和 Wilson Lift 下界。"""
    table = pl.DataFrame(
        {
            "dataset": ["validation", "validation"],
            "rule_id": ["mr_known", "mr_known"],
            "target": ["target", "target"],
            "slice": ["__overall__", "__overall__"],
            "group": ["total", "hit"],
            "sample_count": [10, 2],
            "event_count": [5, 2],
            "event_rate": [0.5, 1.0],
        }
    )

    result = add_statistical_metrics(
        table,
        direction="high_risk",
        confidence_level=0.95,
    )
    hit = result.filter(pl.col("group") == "hit").row(0, named=True)

    assert hit["p_value"] == pytest.approx(2.0 / 9.0)
    assert hit["q_value"] == pytest.approx(2.0 / 9.0)
    assert hit["event_rate_ci_lower"] == pytest.approx(0.425030609, rel=1e-8)
    assert hit["lift_ci_lower"] == pytest.approx(0.850061218, rel=1e-8)


def test_benjamini_hochberg_preserves_input_order() -> None:
    """BH 校正应保持输入顺序并执行反向单调化。"""
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.002]) == pytest.approx(
        [0.02, 0.04, 0.04, 0.008]
    )
