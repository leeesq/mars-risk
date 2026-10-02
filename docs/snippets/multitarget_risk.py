"""多目标省略 features，使用同一组首目标参考边界。"""

import polars as pl

from mars.analysis import profile_risk

df = pl.DataFrame(
    {
        "income": [2600, 3100, 3500, 3900, 4500, 5200, 6100, 7200],
        "bad30": [1, 1, 1, 0, 1, 0, 0, 0],
        "bad60": [1, None, 1, 0, 1, 1, 0, None],
        "period": ["2024-01"] * 4 + ["2024-02"] * 4,
        "apply_dt": ["2024-01-01"] * 4 + ["2024-02-01"] * 4,
        "weight": [1.0, 2.0] * 4,
    }
)

profile = profile_risk(
    df,
    target=["bad30", "bad60"],
    group_col="period",
    time_col="apply_dt",
    weights_col="weight",
    method="cart",
    n_bins=2,
    n_jobs=1,
)

# 标签与声明角色列均被排除，只对 income 拟合一次。
assert profile.metadata["features"] == ["income"]
assert profile.targets == ["bad30", "bad60"]
summary = profile.report.get_table("summary")
# trend_tables 保持首目标 bad30 的既有趋势语义。
trends = profile.report.trend_tables
