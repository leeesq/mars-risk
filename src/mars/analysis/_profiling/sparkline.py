"""数据画像 sparkline 字符分布图。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import polars as pl

from mars.analysis._profiling.metrics import excluded_values, is_numeric_feature
from mars.analysis._profiling.types import ProfileComputeOptions, ProfileRunContext
from mars.utils.logger import logger


def compute_sparklines(context: ProfileRunContext, options: ProfileComputeOptions) -> pl.DataFrame:
    """批量计算数值特征的字符分布图。"""
    num_cols = [col for col in context.features if is_numeric_feature(context, col)]
    if not num_cols:
        return pl.DataFrame(
            {"feature": [], "distribution": []},
            schema={"feature": pl.String, "distribution": pl.String},
        )

    df_subset = context.df.select(num_cols)
    if df_subset.height > options.sparkline_sample_size:
        sample_df = df_subset.sample(n=options.sparkline_sample_size, with_replacement=False)
    else:
        sample_df = df_subset

    max_workers = max(1, pl.thread_pool_size() - 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                lambda col: _sparkline_for_column(context, options, sample_df, col),
                num_cols,
            )
        )

    return pl.DataFrame(results, schema={"feature": pl.String, "distribution": pl.String})


def _sparkline_for_column(
    context: ProfileRunContext,
    options: ProfileComputeOptions,
    sample_df: pl.DataFrame,
    col: str,
) -> dict[str, str]:
    """计算单字段 sparkline。"""
    bars = ["_", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588"]
    try:
        target_s = sample_df[col]
        if target_s.dtype in [pl.Float32, pl.Float64]:
            target_s = target_s.filter(target_s.is_not_nan())

        exclude_vals = excluded_values(context, col, options)
        if exclude_vals:
            target_s = target_s.filter(~target_s.is_in(exclude_vals))

        series = target_s.drop_nulls()
        if series.len() == 0:
            distribution = "_" * options.sparkline_bins
        elif series.len() == 1 or series.min() == series.max():
            distribution = _constant_distribution(options.sparkline_bins)
        else:
            distribution = _histogram_distribution(series, options.sparkline_bins, bars)
    except (pl.exceptions.PolarsError, ValueError, TypeError) as exc:
        logger.error("Sparkline calculation failed for feature '%s': %s", col, exc)
        distribution = "ERR"

    return {"feature": col, "distribution": distribution}


def _constant_distribution(n_bins: int) -> str:
    """生成常量分布的居中字符图。"""
    if n_bins <= 1:
        return "\u2588"
    center = n_bins // 2
    chars = ["_"] * n_bins
    chars[center] = "\u2588"
    return "".join(chars)


def _histogram_distribution(series: pl.Series, n_bins: int, bars: list[str]) -> str:
    """根据直方图计数生成字符图。"""
    hist_df = series.hist(bin_count=n_bins)
    counts = hist_df.get_column(hist_df.columns[-1]).to_list()
    max_count = max(counts)
    if max_count == 0:
        return "_" * n_bins

    chars: list[str] = []
    for count in counts:
        if count == 0:
            chars.append(bars[0])
        else:
            idx = int(count / max_count * (len(bars) - 2)) + 1
            chars.append(bars[min(idx, len(bars) - 1)])
    return "".join(chars)

