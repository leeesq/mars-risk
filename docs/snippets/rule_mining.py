"""规则挖掘文档中的可执行最小示例。"""

import polars as pl

from mars.rule import mine_rules

train_df = pl.DataFrame(
    {
        "income": list(range(100)),
        "target": [int(value >= 80) for value in range(100)],
    }
)

result = mine_rules(
    train_df,
    target="target",
    validation_df=train_df,
    seed_rules=["income >= 80"],
    generators=[],
)
scored_df = result.rule_set.transform(train_df)

assert result.status == "success"
assert scored_df["rule_hit_count"].sum() == 20
