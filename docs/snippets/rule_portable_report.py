"""模拟规则证据的公共查询与保存恢复最小示例。"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from mars.reporting import load_report
from mars.rule import mine_rules


def run(output_dir: Path) -> None:
    """生成模拟证据，完整保存后按成员特征与来源继续查询。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pl.DataFrame(
        {"income": list(range(200)), "debt": [1] * 200, "bad": [int(i >= 160) for i in range(200)]}
    )
    # 独立样本 ID 未保存；这里独立生成第二组模拟行，分布仅用于接口演示。
    validation = pl.DataFrame(
        {"income": list(range(240)), "debt": [1] * 240, "bad": [int(i >= 170) for i in range(240)]}
    )
    result = mine_rules(
        train,
        target="bad",
        validation_df=validation,
        features=["income", "debt"],
        seed_rules=["income >= 170 AND debt > 0"],
        generators=[],
    )
    report = result.to_report(
        feature_metadata={
            "income": {"display_name": "收入", "data_source": "application"},
            "debt": {"display_name": "负债", "data_source": "credit"},
        },
        business_context={
            "dataset_id": "synthetic-rule-example",
            "labels": {"bad": {"definition": "模拟违约标签"}},
        },
    )
    report.save(output_dir / "rules.marsreport", overwrite=True)
    restored = load_report(output_dir / "rules.marsreport")
    page = restored.query_page(
        "evaluation",
        features="income",
        sources="credit",
        filters={"dataset": "validation", "target": "bad", "group": "hit"},
        limit=2,
    )
    assert page["reference"]["report_id"] == report.report_id
    assert page["returned_rows"] == 1
    restored.to_ai_context(
        queries={"evaluation": {"features": "debt", "limit": 2}}, max_chars=12000
    )


if __name__ == "__main__":
    run(Path("output/rule-report"))
