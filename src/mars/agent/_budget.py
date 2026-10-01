"""内置 Agent 的计算预检查；规模估算不等同于 CPU 或内存硬限额。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import polars as pl

from mars.utils.date import MarsDate

from ._session import _Dataset


@dataclass(frozen=True)
class MarsAgentComputeBudget:
    """内置 Agent 单次计算的规模上限，不限制公共分析 API 或报告查询。

    Parameters
    ----------
    max_features : int
        解析默认值后的特征数上限，默认沿用工具的 200 特征边界。
    max_current_rows : int
        当前样本行数上限。
    max_benchmark_rows : int
        基准样本行数上限。
    max_input_cells : int
        当前与基准行数之和乘以有效特征数的上限。
    max_groups : int
        当前与基准分组基数之和的保守上限，不包含整体行。
    max_time_windows : int
        当前与基准有效日期基数之和的上限，覆盖按日缺失明细。
    max_bins : int
        数值分箱参数或类别特征估算正常箱数的上限。
    max_estimated_rows : int
        汇总、趋势、分箱和按日明细的保守总行数上限。
    max_estimated_cells : int
        按表宽度估算的结果单元格总数上限。

    Raises
    ------
    ValueError
        任一上限不是正整数时抛出。

    Notes
    -----
    默认值用于交互式分析；选择依据和估算公式见 Agent 用户指南。
    估算不扫描数值特征，也不复制宽表；仅统计角色列和类别列的基数。

    Examples
    --------
    >>> MarsAgentComputeBudget(max_features=20).max_features
    20
    """

    max_features: int = 200
    max_current_rows: int = 2_000_000
    max_benchmark_rows: int = 2_000_000
    max_input_cells: int = 40_000_000
    max_groups: int = 120
    max_time_windows: int = 366
    max_bins: int = 50
    max_estimated_rows: int = 500_000
    max_estimated_cells: int = 10_000_000

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class _ComputeBudgetExceeded(ValueError):
    """携带纯规模信息的结构化错误，不包含样本值。"""

    def __init__(self, dimension: str, actual: int, limit: int, *, estimated: bool) -> None:
        self.details = {
            "dimension": dimension,
            "actual": actual,
            "limit": limit,
            "estimated": estimated,
            "suggestion": "Request fewer features/bins or register a smaller sample/window; "
            "the caller may explicitly raise compute_budget after reviewing the workload.",
        }
        super().__init__(f"Compute budget exceeded: {dimension}={actual}, limit={limit}.")


def check_compute_budget(
    budget: MarsAgentComputeBudget,
    name: str,
    dataset: _Dataset,
    benchmark: _Dataset | None,
    features: list[str],
    group_col: str | None,
    n_bins: int,
    metrics: list[str],
) -> dict[str, Any]:
    """先检查 O(1) 规模，再投影角色列与类别列统计基数，拒绝超预算计算。"""
    scale: dict[str, int] = {}

    def check(dimension: str, actual: int, *, estimated: bool = False) -> None:
        """记录实际或估算规模，超限时终止预检查。"""
        limit = getattr(budget, f"max_{dimension}")
        scale[dimension] = actual
        if actual > limit:
            raise _ComputeBudgetExceeded(dimension, actual, limit, estimated=estimated)

    check("features", len(features))
    check("current_rows", dataset.frame.height)
    check("benchmark_rows", benchmark.frame.height if benchmark else 0)
    check("input_cells", (scale["current_rows"] + scale["benchmark_rows"]) * len(features))
    has_bins = name != "profile_data" or "psi" in metrics
    # profiler 的 PSI 默认 10 箱，另外两个工具默认 5 箱。
    normal_bins = n_bins if name != "profile_data" else 10 if has_bins else 0
    check("bins", normal_bins)

    groups = windows = 0
    categorical_bins: dict[str, int] = {}
    for source in (dataset, benchmark):
        if source is None:
            continue
        # 只投影必要列，日期解析沿用 MARS 共享语义；null 组也计入规模。
        time_col = dataset.time_col
        if name == "profile_data" and time_col and group_col and MarsDate.is_time_grain(group_col):
            groups += int(
                source.frame.select(MarsDate.from_grain(time_col, group_col).n_unique()).item()
            )
        elif group_col and group_col in source.frame.columns:
            groups += int(source.frame.select(pl.col(group_col).cast(pl.String).n_unique()).item())
        elif time_col and time_col in source.frame.columns:
            grain = group_col if group_col and MarsDate.is_time_grain(group_col) else "month"
            groups += int(
                source.frame.select(MarsDate.from_grain(time_col, grain).n_unique()).item()
            )
        if time_col and time_col in source.frame.columns:
            windows += int(
                source.frame.select(
                    MarsDate.smart_parse_expr(time_col).drop_nulls().n_unique()
                ).item()
            )
        check("groups", groups)
        check("time_windows", windows)
        if has_bins:
            categorical = [f for f in features if not source.frame.schema[f].is_numeric()]
            if categorical:
                counts = source.frame.select(
                    [pl.col(f).n_unique().alias(f) for f in categorical]
                ).row(0, named=True)
                for feature, count in counts.items():
                    categorical_bins[feature] = categorical_bins.get(feature, 0) + int(count)
                check("bins", max(normal_bins, *categorical_bins.values()), estimated=True)

    # 上界包括整体行、benchmark、特殊/缺失箱和按日缺失表；不承诺精确表大小。
    scopes = 1 + groups + int(benchmark is not None)
    bin_rows = sum(max(normal_bins, categorical_bins.get(f, 0)) + 2 for f in features)
    metric_count = len(metrics) + 2 if name == "profile_data" else 16
    summary_rows = len(features) * scopes * metric_count
    detail_rows = bin_rows * scopes * (2 if name == "monitor_data" else 1) if has_bins else 0
    daily_rows = len(features) * windows * 2 if name != "profile_data" else 0
    check("estimated_rows", summary_rows + detail_rows + daily_rows, estimated=True)
    # 趋势为宽表；同时对 scopes 和 windows 计宽度，覆盖透视后的展开。
    cells = summary_rows * (scopes + 12) + detail_rows * 32 + daily_rows * 8
    check("estimated_cells", cells, estimated=True)
    return {"scale": scale, "limits": asdict(budget), "estimation": "conservative_v1"}
