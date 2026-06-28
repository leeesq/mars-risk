"""分箱表现表内部整理逻辑。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol

import numpy as np
import polars as pl


class _PerformanceBinnerState(Protocol):
    """描述表现表整理所需的最小分箱器状态。"""

    bin_mappings_: dict[str, dict[int, str]]
    bin_woes_: dict[str, dict[int, float]]

    @staticmethod
    def _build_trend_shape_frame(
        df_woe_lists: pl.DataFrame,
        *,
        trend_col_name: str,
    ) -> pl.DataFrame:
        """构造特征趋势形态表。"""
        ...


def _update_woe_cache_from_stats(
    binner: _PerformanceBinnerState,
    stats_df: pl.DataFrame,
) -> None:
    """把表现表中的 WOE 列回写到分箱器缓存。"""
    woe_data = stats_df.select(["feature", "bin_index", "woe"]).to_dict(as_series=False)
    temp_woe_map: defaultdict[str, dict[int, float]] = defaultdict(dict)
    for feature, bin_index, woe in zip(
        woe_data["feature"],
        woe_data["bin_index"],
        woe_data["woe"],
        strict=False,
    ):
        if bin_index is not None and not (
            isinstance(bin_index, float) and np.isnan(bin_index)
        ) and woe is not None:
            temp_woe_map[feature][int(bin_index)] = float(woe)
    binner.bin_woes_.update(temp_woe_map)


def _build_profile_mapping_frame(binner: _PerformanceBinnerState) -> pl.DataFrame:
    """把分箱标签映射缓存转为表现表可 join 的长表。"""
    mapping_rows: list[dict[str, Any]] = []
    for col, map_dict in binner.bin_mappings_.items():
        for idx, label in map_dict.items():
            mapping_rows.append(
                {
                    "feature": col,
                    "bin_index": idx,
                    "bin_label": label,
                },
            )

    if not mapping_rows:
        return pl.DataFrame()
    return pl.DataFrame(
        mapping_rows,
        schema={
            "feature": pl.Utf8,
            "bin_index": pl.Int16,
            "bin_label": pl.Utf8,
        },
    )


def _finalize_bin_performance_table(
    binner: _PerformanceBinnerState,
    stats_df: pl.DataFrame,
    *,
    include_bin_index: bool,
) -> pl.DataFrame:
    """补充分箱标签、趋势形态和稳定列顺序。"""
    mapping_df = _build_profile_mapping_frame(binner)
    if mapping_df.is_empty():
        return stats_df

    final_df = (
        stats_df
        .join(mapping_df, on=["feature", "bin_index"], how="left")
        .with_columns((pl.col("bin_index") < 0).alias("_is_special"))
        .sort(["feature", "_is_special", "bin_index"])
        .drop("_is_special")
        .select(
            [
                pl.col("feature"),
                *([pl.col("bin_index")] if include_bin_index else []),
                pl.col("bin_label").fill_null(pl.col("bin_index").cast(pl.Utf8)),
                pl.all().exclude(["feature", "bin_index", "bin_label"]),
            ]
        )
    )

    trend_df = binner._build_trend_shape_frame(
        stats_df.lazy()
        .filter(pl.col("bin_index") >= 0)
        .sort(["feature", "bin_index"])
        .group_by("feature")
        .agg(pl.col("woe"))
        .collect(),
        trend_col_name="trend_shape",
    )
    final_df = (
        final_df
        .join(trend_df, on="feature", how="left")
        .with_columns(pl.col("trend_shape").fill_null("undefined"))
    )

    base_cols = ["feature"]
    if include_bin_index:
        base_cols.append("bin_index")
    base_cols.extend(["bin_label", "trend_shape"])
    other_cols = [col for col in final_df.columns if col not in base_cols]
    return final_df.select(base_cols + other_cols)
