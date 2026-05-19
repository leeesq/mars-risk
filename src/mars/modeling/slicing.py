"""建模样本切分工具。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import polars as pl

from mars.modeling.utils import FrameLike, is_polars_dataframe


class MarsModelDataSlicer:
    """
    按输入数据引擎切分二分类建模样本。

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame
        原始建模样本。
    time_col : str
        时间列名，切分时按自然日保持完整。
    label_col : str
        二分类标签列名，仅 ``0``/``1`` 参与切分。
    dataset_flag_col : str, default "dataset_flag"
        输出的数据集标识列名。

    Notes
    -----
    Pandas 输入全程走 Pandas，Polars 输入全程走 Polars，避免无收益的跨框架转换。
    """

    def __init__(
        self,
        df: FrameLike,
        time_col: str,
        label_col: str,
        dataset_flag_col: str = "dataset_flag",
    ) -> None:
        self._input_is_polars: bool = is_polars_dataframe(df)
        self._engine: str

        if isinstance(df, pl.DataFrame):
            self._engine = "polars"
            self.df: pl.DataFrame | pd.DataFrame = df.clone()
        elif isinstance(df, pd.DataFrame):
            self._engine = "pandas"
            self.df = df.copy()
        else:
            raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")

        self.time_col: str = time_col
        self.label_col: str = label_col
        self.dataset_flag_col: str = dataset_flag_col

        missing_cols = {time_col, label_col}.difference(self.df.columns)
        if missing_cols:
            raise ValueError(f"Input data is missing required columns: {sorted(missing_cols)}")

        if self._engine == "pandas":
            self._init_pandas()
        else:
            self._init_polars()

    @property
    def engine_(self) -> str:
        """返回由输入类型自动选择的切分引擎。"""
        return self._engine

    def _init_pandas(self) -> None:
        assert isinstance(self.df, pd.DataFrame)
        clean_dt = pd.to_datetime(self.df[self.time_col], errors="coerce")
        self.df["__clean_dt__"] = clean_dt
        self.df["__date__"] = clean_dt.dt.date
        self.df[self.dataset_flag_col] = "unassigned"

    def _init_polars(self) -> None:
        assert isinstance(self.df, pl.DataFrame)
        clean_dt = pl.col(self.time_col).cast(pl.Utf8).str.to_datetime(strict=False)
        self.df = self.df.with_columns(
            clean_dt.alias("__clean_dt__"),
            clean_dt.dt.date().alias("__date__"),
            pl.lit("unassigned").alias(self.dataset_flag_col),
        )

    def _validate_ratios(self, split_ratios: Dict[str, float]) -> None:
        if not split_ratios:
            raise ValueError("split_ratios must not be empty.")

        negative_ratios = {name: value for name, value in split_ratios.items() if float(value) < 0.0}
        if negative_ratios:
            raise ValueError(f"Split ratios must be non-negative, got {negative_ratios}.")

        total = float(sum(split_ratios.values()))
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")

    def _validate_hybrid_keys(
        self,
        split_ratios: Dict[str, float],
        train_key: str,
        val_key: str,
    ) -> tuple[float, float, float]:
        if train_key not in split_ratios:
            raise ValueError(f"train_key {train_key!r} is missing from split_ratios.")
        if val_key not in split_ratios:
            raise ValueError(f"val_key {val_key!r} is missing from split_ratios.")
        if train_key == val_key:
            raise ValueError("train_key and val_key must be different.")

        train_ratio = float(split_ratios.get(train_key, 0.0))
        val_ratio = float(split_ratios.get(val_key, 0.0))
        modeling_ratio = train_ratio + val_ratio
        if modeling_ratio <= 0.0:
            raise ValueError("train_ratio + val_ratio must be greater than 0.")
        return train_ratio, val_ratio, modeling_ratio

    def split_by_time_strictly(self, split_ratios: Dict[str, float]) -> FrameLike:
        """
        按时间顺序严格切分，并保证同一天不被拆到多个数据集。

        Parameters
        ----------
        split_ratios : dict of str to float
            切分名称到比例的映射，比例合计必须为 1。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            与输入类型一致的切分结果。
        """
        self._validate_ratios(split_ratios)
        if self._engine == "pandas":
            return self._split_by_time_strictly_pandas(split_ratios)
        return self._split_by_time_strictly_polars(split_ratios)

    def split_hybrid_random_val(
        self,
        split_ratios: Dict[str, float],
        train_key: str = "train",
        val_key: str = "val",
        random_seed: int = 42,
    ) -> FrameLike:
        """
        在建模窗口内随机切分 train/val，其余切片保持时间顺序。

        Parameters
        ----------
        split_ratios : dict of str to float
            切分名称到比例的映射，比例合计必须为 1。
        train_key : str, default "train"
            训练集标识。
        val_key : str, default "val"
            验证集标识。
        random_seed : int, default 42
            随机种子。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            与输入类型一致的切分结果。
        """
        self._validate_ratios(split_ratios)
        self._validate_hybrid_keys(split_ratios, train_key, val_key)
        if self._engine == "pandas":
            return self._split_hybrid_random_val_pandas(split_ratios, train_key, val_key, random_seed)
        return self._split_hybrid_random_val_polars(split_ratios, train_key, val_key, random_seed)

    def _reset_and_mark_other_pandas(self) -> None:
        assert isinstance(self.df, pd.DataFrame)
        self.df[self.dataset_flag_col] = "unassigned"
        valid_mask = self.df[self.label_col].isin([0, 1]) & self.df["__date__"].notna()
        self.df.loc[~valid_mask, self.dataset_flag_col] = "other"

    def _get_date_cutoffs_pandas(self, split_ratios: Dict[str, float]) -> Dict[str, Any]:
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

        cutoffs: Dict[str, Any] = {}
        cum_ratio = 0.0
        for flag, ratio in split_ratios.items():
            cum_ratio += float(ratio)
            target_count = total_valid * cum_ratio
            cutoff_rows = daily_stats.loc[daily_stats["cum_count"] >= target_count]
            if cutoff_rows.empty:
                cutoff_date = daily_stats["__date__"].iloc[-1]
            else:
                cutoff_date = cutoff_rows["__date__"].iloc[0]
            cutoffs[flag] = cutoff_date
        return cutoffs

    def _assign_until_cutoff_pandas(self, flag: str, cutoff_date: Any) -> None:
        assert isinstance(self.df, pd.DataFrame)
        if cutoff_date is None:
            return
        mask = (self.df["__date__"] <= cutoff_date) & (self.df[self.dataset_flag_col] == "unassigned")
        self.df.loc[mask, self.dataset_flag_col] = flag

    def _split_by_time_strictly_pandas(self, split_ratios: Dict[str, float]) -> pd.DataFrame:
        self._reset_and_mark_other_pandas()
        cutoffs = self._get_date_cutoffs_pandas(split_ratios)
        for flag in split_ratios:
            self._assign_until_cutoff_pandas(flag, cutoffs.get(flag))
        return self._get_result_pandas()

    def _split_hybrid_random_val_pandas(
        self,
        split_ratios: Dict[str, float],
        train_key: str,
        val_key: str,
        random_seed: int,
    ) -> pd.DataFrame:
        assert isinstance(self.df, pd.DataFrame)
        _, val_ratio, modeling_ratio = self._validate_hybrid_keys(split_ratios, train_key, val_key)
        self._reset_and_mark_other_pandas()

        timeline_ratios: Dict[str, float] = {"__modeling__": modeling_ratio}
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
            self.df.loc[mask_modeling & (self.df["__rand__"] < val_internal_prob), self.dataset_flag_col] = val_key
            self.df.loc[mask_modeling & (self.df[self.dataset_flag_col] == "unassigned"), self.dataset_flag_col] = train_key

        for flag in split_ratios:
            if flag in {train_key, val_key}:
                continue
            self._assign_until_cutoff_pandas(flag, cutoffs.get(flag))

        return self._get_result_pandas()

    def _get_result_pandas(self) -> pd.DataFrame:
        assert isinstance(self.df, pd.DataFrame)
        result = self.df.copy()
        result.loc[result[self.dataset_flag_col] == "unassigned", self.dataset_flag_col] = "other"
        cols_to_drop = [col for col in ["__clean_dt__", "__date__", "__rand__"] if col in result.columns]
        return result.drop(columns=cols_to_drop)

    def _reset_and_mark_other_polars(self) -> None:
        assert isinstance(self.df, pl.DataFrame)
        label_is_valid = pl.col(self.label_col).is_in([0, 1])
        date_is_valid = pl.col("__date__").is_not_null()
        self.df = self.df.with_columns(
            pl.when(label_is_valid & date_is_valid)
            .then(pl.lit("unassigned"))
            .otherwise(pl.lit("other"))
            .alias(self.dataset_flag_col)
        )

    def _get_date_cutoffs_polars(self, split_ratios: Dict[str, float]) -> Dict[str, Any]:
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

        cutoffs: Dict[str, Any] = {}
        cum_ratio = 0.0
        for flag, ratio in split_ratios.items():
            cum_ratio += float(ratio)
            target_count = total_valid * cum_ratio
            cutoff_df = daily_stats.filter(pl.col("cum_count") >= target_count)
            if cutoff_df.is_empty():
                cutoff_date = daily_stats.select(pl.col("__date__").last()).item()
            else:
                cutoff_date = cutoff_df.select(pl.col("__date__").first()).item()
            cutoffs[flag] = cutoff_date
        return cutoffs

    def _assign_until_cutoff_polars(self, flag: str, cutoff_date: Any) -> None:
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

    def _split_by_time_strictly_polars(self, split_ratios: Dict[str, float]) -> pl.DataFrame:
        self._reset_and_mark_other_polars()
        cutoffs = self._get_date_cutoffs_polars(split_ratios)
        for flag in split_ratios:
            self._assign_until_cutoff_polars(flag, cutoffs.get(flag))
        return self._get_result_polars()

    def _split_hybrid_random_val_polars(
        self,
        split_ratios: Dict[str, float],
        train_key: str,
        val_key: str,
        random_seed: int,
    ) -> pl.DataFrame:
        assert isinstance(self.df, pl.DataFrame)
        _, val_ratio, modeling_ratio = self._validate_hybrid_keys(split_ratios, train_key, val_key)
        self._reset_and_mark_other_polars()

        timeline_ratios: Dict[str, float] = {"__modeling__": modeling_ratio}
        for flag, ratio in split_ratios.items():
            if flag not in {train_key, val_key}:
                timeline_ratios[flag] = float(ratio)

        cutoffs = self._get_date_cutoffs_polars(timeline_ratios)
        modeling_cutoff = cutoffs.get("__modeling__")

        rng = np.random.default_rng(random_seed)
        self.df = self.df.with_columns(pl.Series("__rand__", rng.random(self.df.height)))
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
            if flag in {train_key, val_key}:
                continue
            self._assign_until_cutoff_polars(flag, cutoffs.get(flag))

        return self._get_result_polars()

    def _get_result_polars(self) -> pl.DataFrame:
        assert isinstance(self.df, pl.DataFrame)
        result = self.df.with_columns(
            pl.when(pl.col(self.dataset_flag_col) == "unassigned")
            .then(pl.lit("other"))
            .otherwise(pl.col(self.dataset_flag_col))
            .alias(self.dataset_flag_col)
        )
        cols_to_drop = [col for col in ["__clean_dt__", "__date__", "__rand__"] if col in result.columns]
        return result.drop(cols_to_drop)
