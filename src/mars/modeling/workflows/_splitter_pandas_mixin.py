"""Pandas 样本切分实现。"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl

from mars.modeling.workflows._splitter_protocols import _HybridKeyValidator


class PandasSplitterMixin:
    """封装 Pandas 路径下的样本切分逻辑。"""

    df: pd.DataFrame | pl.DataFrame
    time_col: str
    label_col: str
    dataset_flag_col: str

    def _init_pandas(self) -> None:
        """初始化 Pandas 工作副本中的辅助列。"""
        assert isinstance(self.df, pd.DataFrame)
        clean_dt = pd.to_datetime(self.df[self.time_col], errors="coerce")
        self.df["__clean_dt__"] = clean_dt
        self.df["__date__"] = clean_dt.dt.date
        self.df[self.dataset_flag_col] = "unassigned"

    def _reset_and_mark_other_pandas(self) -> None:
        """重置 Pandas 切片列，并标记无效样本为 ``other``。"""
        assert isinstance(self.df, pd.DataFrame)
        self.df[self.dataset_flag_col] = "unassigned"
        valid_mask = self.df[self.label_col].isin([0, 1]) & self.df["__date__"].notna()
        self.df.loc[~valid_mask, self.dataset_flag_col] = "other"

    def _get_date_cutoffs_pandas(self, split_ratios: dict[str, float]) -> dict[str, Any]:
        """按自然日累计样本量计算各切片截止日。"""
        assert isinstance(self.df, pd.DataFrame)
        valid_df = self.df.loc[self.df[self.dataset_flag_col] == "unassigned"]
        if valid_df.empty:
            return {}

        daily_stats = (
            valid_df.groupby("__date__", sort=True)
            .size()
            .rename("row_count")
            .reset_index()
            .sort_values("__date__")
        )
        daily_stats["cum_count"] = daily_stats["row_count"].cumsum()
        total_valid = int(daily_stats["row_count"].sum())
        if total_valid <= 0:
            return {}

        cutoffs: dict[str, Any] = {}
        cum_ratio = 0.0
        for flag, ratio in split_ratios.items():
            cum_ratio += float(ratio)
            target_count = total_valid * cum_ratio
            cutoff_rows = daily_stats.loc[daily_stats["cum_count"] >= target_count]
            cutoffs[flag] = (
                daily_stats["__date__"].iloc[-1]
                if cutoff_rows.empty
                else cutoff_rows["__date__"].iloc[0]
            )
        return cutoffs

    def _assign_until_cutoff_pandas(self, flag: str, cutoff_date: Any) -> None:
        """给 Pandas 未分配样本按截止日打切片标记。"""
        assert isinstance(self.df, pd.DataFrame)
        if cutoff_date is None:
            return
        mask = (self.df["__date__"] <= cutoff_date) & (self.df[self.dataset_flag_col] == "unassigned")
        self.df.loc[mask, self.dataset_flag_col] = flag

    def _split_by_time_strictly_pandas(self, split_ratios: dict[str, float]) -> pd.DataFrame:
        """执行 Pandas 严格时间切分。"""
        self._reset_and_mark_other_pandas()
        cutoffs = self._get_date_cutoffs_pandas(split_ratios)
        for flag in split_ratios:
            self._assign_until_cutoff_pandas(flag, cutoffs.get(flag))
        return self._get_result_pandas()

    def _split_hybrid_random_val_pandas(
        self,
        split_ratios: dict[str, float],
        train_key: str,
        val_key: str,
        random_seed: int,
    ) -> pd.DataFrame:
        """执行 Pandas hybrid 随机验证集切分。"""
        assert isinstance(self.df, pd.DataFrame)
        _, val_ratio, modeling_ratio = cast(_HybridKeyValidator, self)._validate_hybrid_keys(
            split_ratios,
            train_key,
            val_key,
        )
        self._reset_and_mark_other_pandas()

        timeline_ratios: dict[str, float] = {"__modeling__": modeling_ratio}
        for flag, ratio in split_ratios.items():
            if flag not in {train_key, val_key}:
                timeline_ratios[flag] = float(ratio)

        cutoffs = self._get_date_cutoffs_pandas(timeline_ratios)
        modeling_cutoff = cutoffs.get("__modeling__")
        rng = np.random.default_rng(random_seed)
        self.df["__rand__"] = rng.random(len(self.df))
        val_internal_prob = val_ratio / modeling_ratio

        if modeling_cutoff is not None:
            mask_modeling = (
                (self.df["__date__"] <= modeling_cutoff)
                & (self.df[self.dataset_flag_col] == "unassigned")
            )
            self.df.loc[
                mask_modeling & (self.df["__rand__"] < val_internal_prob),
                self.dataset_flag_col,
            ] = val_key
            self.df.loc[
                mask_modeling & (self.df[self.dataset_flag_col] == "unassigned"),
                self.dataset_flag_col,
            ] = train_key

        for flag in split_ratios:
            if flag not in {train_key, val_key}:
                self._assign_until_cutoff_pandas(flag, cutoffs.get(flag))
        return self._get_result_pandas()

    def _get_result_pandas(self) -> pd.DataFrame:
        """返回清理辅助列后的 Pandas 结果。"""
        assert isinstance(self.df, pd.DataFrame)
        result = self.df.copy()
        result.loc[result[self.dataset_flag_col] == "unassigned", self.dataset_flag_col] = "other"
        cols_to_drop = [col for col in ["__clean_dt__", "__date__", "__rand__"] if col in result.columns]
        return result.drop(columns=cols_to_drop)
