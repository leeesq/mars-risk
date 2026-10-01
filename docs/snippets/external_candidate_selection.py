"""外部候选选择示例；模拟评估表不是 MARS 自动生成的 TEST 字段。"""

from __future__ import annotations

import math
from typing import Literal

import pandas as pd


def select_candidates(
    evaluation: pd.DataFrame,
    *,
    metric: str,
    direction: Literal["maximize", "minimize"],
    candidate_count: int = 10,
    slice_name: str = "TEST",
) -> tuple[list[int], pd.DataFrame]:
    """在明确切片选取合格候选，并列按 trial_num 升序打破。

    Parameters
    ----------
    evaluation : pd.DataFrame
        外部已计算表，含 trial_num、slice、eligible 和指标；每 trial/切片一行。
    metric : str
        已计算指标列，不能将 val 指标重命名为 TEST 指标。
    direction : Literal["maximize", "minimize"]
        本次候选选择的指标方向。
    candidate_count : int
        候选数，默认示例配置为 10。
    slice_name : str
        本次选择使用的真实评估切片。

    Returns
    -------
    tuple[list[int], pd.DataFrame]
        确定顺序的编号和包括选择口径的审计表。

    Raises
    ------
    ValueError
        配置、列、重复候选或评估状态不满足要求时抛出。
    """
    if (
        direction not in {"maximize", "minimize"}
        or type(candidate_count) is not int
        or candidate_count < 1
    ):
        raise ValueError("provide direction and positive candidate_count")
    required = {"trial_num", "slice", "eligible", metric}
    if not required.issubset(evaluation.columns):
        raise ValueError("first obtain actual candidate evaluation on the requested slice")
    scope = evaluation.loc[evaluation["slice"] == slice_name].copy()
    if scope.empty or scope["trial_num"].duplicated().any():
        raise ValueError("require one evaluated row per trial in the requested slice")
    if not scope["eligible"].isin([True, False]).all():
        raise ValueError("eligibility must be explicitly recorded")
    eligible = scope.loc[scope["eligible"] & scope[metric].notna()].copy()
    eligible = eligible.loc[eligible[metric].map(math.isfinite)]
    ranked = eligible.sort_values(
        [metric, "trial_num"],
        ascending=[direction == "minimize", True],
        kind="stable",
    ).head(candidate_count)
    ranked["selection_slice"] = slice_name
    ranked["selection_metric"] = metric
    ranked["selection_direction"] = direction
    ranked["tie_policy"] = "trial_num ascending"
    return ranked["trial_num"].astype(int).tolist(), ranked


# 已取得候选在 TEST 上的评估；eligible 由外部预先声明的资格规则计算。
evaluation = pd.DataFrame(
    {
        "trial_num": [1, 2, 3, 4],
        "slice": ["TEST"] * 4,
        "ks": [35.0, 40.0, 40.0, 50.0],
        "eligible": [True, True, True, False],
    }
)
trial_nums, selection_record = select_candidates(
    evaluation,
    metric="ks",
    direction="maximize",
    candidate_count=10,
)
assert trial_nums == [2, 3, 1]
print(selection_record.to_string(index=False))

# 实际工作中，已有 tuning_result 和 df 时按上面的顺序调用：
# replay = runner.replay(tuning_result, df, trial_nums=trial_nums,
#                        metric_directions={"ks": "maximize"}, retrain=False)
# retrain=False 要求 tune 已保留这些模型；否则使用 retrain=True。
# 外部代码再读取 replay.reports 的真实 OOT、客群及成本证据，记录最终选择及理由。
