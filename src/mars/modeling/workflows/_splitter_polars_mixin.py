"""Polars 样本切分实现。"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


class PolarsSplitterMixin:
    """封装 Polars 路径下的样本切分逻辑。"""

    df: pl.DataFrame
    time_col: str
    label_col: str
    dataset_flag_col: str

    def _validate_hybrid_keys(
        self,
        split_ratios: dict[str, float],
        train_key: str,
        val_key: str,
    ) -> tuple[float, float, float]:
        """校验 hybrid 切分键。"""
        raise NotImplementedError

    def _init_polars(self) -> None:
        """初始化 Polars 工作副本中的辅助列。"""
        assert isinstance(self.df, pl.DataFrame)
        clean_dt = pl.col(self.time_col).cast(pl.Utf8).str.to_datetime(strict=False)
        self.df = self.df.with_columns(
            clean_dt.alias("__clean_dt__"),
            clean_dt.dt.date().alias("__date__"),
            pl.lit("unassigned").alias(self.dataset_flag_col),
        )

    def _reset_and_mark_other_polars(self) -> None:
        """重置 Polars 切片列，并标记无效样本为 ``other``。"""
        assert isinstance(self.df, pl.DataFrame)
        self.df = self.df.with_columns(
            pl.when(
                pl.col(self.label_col).is_in(pl.Series([0, 1]).implode())
                & pl.col("__date__").is_not_null()
            )
            .then(pl.lit("unassigned"))
            .otherwise(pl.lit("other"))
            .alias(self.dataset_flag_col)
        )

    def _get_date_cutoffs_polars(self, split_ratios: dict[str, float]) -> dict[str, Any]:
        """按自然日累计样本量计算各切片截止日。"""
        assert isinstance(self.df, pl.DataFrame)
        valid_df = self.df.filter(pl.col(self.dataset_flag_col) == "unassigned")
        if valid_df.is_empty():
            return {}

        daily_stats = (
            valid_df.group_by("__date__")
            .agg(pl.len().alias("row_count"))
            .sort("__date__")
            .with_columns(pl.col("row_count").cum_sum().alias("cum_count"))
        )
        total_valid = int(daily_stats.select(pl.col("row_count").sum()).item())
        if total_valid <= 0:
            return {}

        cutoffs: dict[str, Any] = {}
        cum_ratio = 0.0
        for flag, ratio in split_ratios.items():
            cum_ratio += float(ratio)
            target_count = total_valid * cum_ratio
            cutoff_df = daily_stats.filter(pl.col("cum_count") >= target_count)
            cutoffs[flag] = (
                daily_stats.select(pl.col("__date__").last()).item()
                if cutoff_df.is_empty()
                else cutoff_df.select(pl.col("__date__").first()).item()
            )
        return cutoffs

    def _assign_until_cutoff_polars(self, flag: str, cutoff_date: Any) -> None:
        """给 Polars 未分配样本按截止日打切片标记。"""
        assert isinstance(self.df, pl.DataFrame)
        if cutoff_date is None:
            return
        self.df = self.df.with_columns(
            pl.when(
                (pl.col("__date__") <= pl.lit(cutoff_date))
                & (pl.col(self.dataset_flag_col) == "unassigned")
            )
            .then(pl.lit(flag))
            .otherwise(pl.col(self.dataset_flag_col))
            .alias(self.dataset_flag_col)
        )

    def _split_by_time_strictly_polars(self, split_ratios: dict[str, float]) -> pl.DataFrame:
        """执行 Polars 严格时间切分。"""
        self._reset_and_mark_other_polars()
        cutoffs = self._get_date_cutoffs_polars(split_ratios)
        for flag in split_ratios:
            self._assign_until_cutoff_polars(flag, cutoffs.get(flag))
        return self._get_result_polars()

    def _split_hybrid_random_val_polars(
        self,
        split_ratios: dict[str, float],
        train_key: str,
        val_key: str,
        random_seed: int,
    ) -> pl.DataFrame:
        """执行 Polars hybrid 随机验证集切分。"""
        assert isinstance(self.df, pl.DataFrame)
        _, val_ratio, modeling_ratio = self._validate_hybrid_keys(split_ratios, train_key, val_key)
        self._reset_and_mark_other_polars()

        timeline_ratios: dict[str, float] = {"__modeling__": modeling_ratio}
        for flag, ratio in split_ratios.items():
            if flag not in {train_key, val_key}:
                timeline_ratios[flag] = float(ratio)

        cutoffs = self._get_date_cutoffs_polars(timeline_ratios)
        modeling_cutoff = cutoffs.get("__modeling__")
        self.df = self.df.with_columns(pl.Series("__rand__", np.random.default_rng(random_seed).random(self.df.height)))
        val_internal_prob = val_ratio / modeling_ratio

        if modeling_cutoff is not None:
            mask_modeling = (
                (pl.col("__date__") <= pl.lit(modeling_cutoff))
                & (pl.col(self.dataset_flag_col) == "unassigned")
            )
            self.df = self.df.with_columns(
                pl.when(mask_modeling & (pl.col("__rand__") < val_internal_prob))
                .then(pl.lit(val_key))
                .when(mask_modeling)
                .then(pl.lit(train_key))
                .otherwise(pl.col(self.dataset_flag_col))
                .alias(self.dataset_flag_col)
            )

        for flag in split_ratios:
            if flag not in {train_key, val_key}:
                self._assign_until_cutoff_polars(flag, cutoffs.get(flag))
        return self._get_result_polars()

    def _get_result_polars(self) -> pl.DataFrame:
        """返回清理辅助列后的 Polars 结果。"""
        assert isinstance(self.df, pl.DataFrame)
        result = self.df.with_columns(
            pl.when(pl.col(self.dataset_flag_col) == "unassigned")
            .then(pl.lit("other"))
            .otherwise(pl.col(self.dataset_flag_col))
            .alias(self.dataset_flag_col)
        )
        cols_to_drop = [col for col in ["__clean_dt__", "__date__", "__rand__"] if col in result.columns]
        return result.drop(cols_to_drop)
