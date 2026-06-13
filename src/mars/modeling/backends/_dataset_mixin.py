"""建模后端数据切片与取数 mixin。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from mars.compute import FrameLike
from mars.modeling.backends.common import (
    normalize_dataset_flags,
    split_name_sort_key,
    validate_dataset_flag_roles,
)


class BackendDatasetMixin:
    """封装后端共享的数据切片和特征取数逻辑。"""

    _input_is_polars: bool
    df_pl: pl.DataFrame | None
    df_pd: pd.DataFrame | None
    features: list[str]
    target: str
    dataset_flag_col: str
    categorical_features: list[str]
    data_dict: dict[str, FrameLike]
    category_levels: dict[str, list[Any]]

    @property
    def split_names(self) -> list[str]:
        """返回当前可用的数据切片名。"""
        return list(self.data_dict.keys())

    def _prepare_data(self) -> None:
        """根据 ``dataset_flag_col`` 构造 train/val/oot 切片。"""
        if self._input_is_polars:
            assert self.df_pl is not None
            flags_pd = normalize_dataset_flags(self.df_pl.get_column(self.dataset_flag_col))
            validate_dataset_flag_roles(flags_pd)
            train_mask_pd = flags_pd.str.contains("train", na=False)
            val_mask_pd = flags_pd.str.contains("val", na=False)

            train_mask = pl.Series("__mask__", train_mask_pd.to_numpy())
            val_mask = pl.Series("__mask__", val_mask_pd.to_numpy())
            train_df = self.df_pl.filter(train_mask)
            val_df = self.df_pl.filter(val_mask)

            if train_df.is_empty():
                raise ValueError("No training rows were found from dataset_flag contains 'train'.")
            if val_df.is_empty():
                raise ValueError("No validation rows were found from dataset_flag contains 'val'.")

            self.data_dict = {"train": train_df, "val": val_df}
            original_flags = self.df_pl.get_column(self.dataset_flag_col).cast(pl.Utf8).to_list()
            oot_flags = sorted(
                {
                    original_flag
                    for original_flag in original_flags
                    if "oot" in str(original_flag).lower()
                },
                key=split_name_sort_key,
            )
            for flag in oot_flags:
                self.data_dict[str(flag)] = self.df_pl.filter(
                    pl.col(self.dataset_flag_col).cast(pl.Utf8) == str(flag)
                )
            return

        assert self.df_pd is not None
        flags_pd = normalize_dataset_flags(self.df_pd[self.dataset_flag_col])
        validate_dataset_flag_roles(flags_pd)
        train_mask = flags_pd.str.contains("train", na=False)
        val_mask = flags_pd.str.contains("val", na=False)
        train_df = self.df_pd.loc[train_mask].copy()
        val_df = self.df_pd.loc[val_mask].copy()

        if train_df.empty:
            raise ValueError("No training rows were found from dataset_flag contains 'train'.")
        if val_df.empty:
            raise ValueError("No validation rows were found from dataset_flag contains 'val'.")

        self.data_dict = {"train": train_df, "val": val_df}
        original_flags = self.df_pd[self.dataset_flag_col].astype(str).tolist()
        oot_flags = sorted(
            {
                original_flag
                for original_flag in original_flags
                if "oot" in str(original_flag).lower()
            },
            key=split_name_sort_key,
        )
        for flag in oot_flags:
            self.data_dict[str(flag)] = self.df_pd.loc[
                self.df_pd[self.dataset_flag_col].astype(str) == str(flag)
            ].copy()

    def _initialize_category_levels(self) -> None:
        """从训练切片提取稳定类别水平。"""
        if not self.categorical_features or "train" not in self.data_dict:
            self.category_levels = {}
            return

        train_df = self.data_dict["train"]
        levels: dict[str, list[Any]] = {}
        for feature in self.categorical_features:
            if isinstance(train_df, pd.DataFrame):
                if feature not in train_df.columns:
                    continue
                values = pd.Series(train_df[feature]).dropna()
                levels[feature] = list(pd.unique(values))
                continue
            if isinstance(train_df, pl.DataFrame) and feature in train_df.columns:
                levels[feature] = (
                    train_df.get_column(feature)
                    .drop_nulls()
                    .unique(maintain_order=True)
                    .to_list()
                )
        self.category_levels = levels

    def _apply_category_levels(self, X: pd.DataFrame) -> pd.DataFrame:
        """将稳定类别水平应用到 Pandas 特征表。"""
        for feature in self.categorical_features:
            if feature not in X.columns:
                continue
            categories = self.category_levels.get(feature)
            if categories is not None:
                X[feature] = X[feature].astype(pd.CategoricalDtype(categories=categories))
            else:
                X[feature] = X[feature].astype("category")
        return X

    def _get_feature_frame(
        self,
        df: FrameLike,
        *,
        for_categorical_backend: bool,
    ) -> pd.DataFrame:
        """返回 Pandas 形态的特征表。"""
        if isinstance(df, pd.DataFrame):
            X = df.loc[:, self.features].copy()
        elif isinstance(df, pl.DataFrame):
            X = df.select(self.features).to_pandas()
        else:
            raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")
        if for_categorical_backend:
            X = self._apply_category_levels(X)
        return X

    def _get_feature_polars(self, df: FrameLike) -> pl.DataFrame:
        """返回 Polars 形态的特征表。"""
        if isinstance(df, pl.DataFrame):
            return df.select(self.features)
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df.loc[:, self.features])
        raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")

    def _get_feature_arrow(self, df: FrameLike) -> Any:
        """返回 Arrow 形态的特征表。"""
        return self._get_feature_polars(df).to_arrow()

    def _has_categorical_backend_features(self) -> bool:
        """判断当前后端是否走原生类别特征路径。"""
        return bool(self.categorical_features)

    def _get_target_array(self, df: FrameLike) -> np.ndarray:
        """提取切片中的目标数组。"""
        if isinstance(df, pd.DataFrame):
            return df[self.target].to_numpy()
        if isinstance(df, pl.DataFrame):
            return df.get_column(self.target).to_numpy()
        raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")
