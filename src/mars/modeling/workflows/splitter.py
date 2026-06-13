"""建模样本切分工具。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from mars.compute import FrameLike, is_polars_dataframe
from mars.modeling.workflows._splitter_pandas_mixin import PandasSplitterMixin
from mars.modeling.workflows._splitter_polars_mixin import PolarsSplitterMixin


class MarsModelDataSplitter(PandasSplitterMixin, PolarsSplitterMixin):
    """无状态的建模样本切分入口。"""

    def __init__(self) -> None:
        """初始化无状态切分器。"""
        self._input_is_polars = False
        self._engine = "stateless"
        self.df: pl.DataFrame | pd.DataFrame = pd.DataFrame()
        self.time_col = ""
        self.label_col = ""
        self.dataset_flag_col = "dataset_flag"

    @classmethod
    def _from_data(
        cls,
        df: FrameLike,
        *,
        time_col: str,
        target: str,
        dataset_flag_col: str,
    ) -> MarsModelDataSplitter:
        """创建绑定单次切分上下文的工作副本。"""
        slicer = cls()
        slicer._bind_data(
            df,
            time_col=time_col,
            target=target,
            dataset_flag_col=dataset_flag_col,
        )
        return slicer

    def _bind_data(
        self,
        df: FrameLike,
        *,
        time_col: str,
        target: str,
        dataset_flag_col: str,
    ) -> None:
        """绑定单次切分所需的数据和列名。"""
        self._input_is_polars = is_polars_dataframe(df)
        if isinstance(df, pl.DataFrame):
            self._engine = "polars"
            self.df = df.clone()
        elif isinstance(df, pd.DataFrame):
            self._engine = "pandas"
            self.df = df.copy()
        else:
            raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")

        self.time_col = time_col
        self.label_col = target
        self.dataset_flag_col = dataset_flag_col

        missing_cols = {time_col, target}.difference(self.df.columns)
        if missing_cols:
            raise ValueError(f"Input data is missing required columns: {sorted(missing_cols)}")

        if self._engine == "pandas":
            self._init_pandas()
        else:
            self._init_polars()

    @property
    def engine_(self) -> str:
        """返回当前工作副本使用的数据引擎。"""
        return self._engine

    def _validate_ratios(self, split_ratios: dict[str, float]) -> None:
        """校验切分比例合法。"""
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
        split_ratios: dict[str, float],
        train_key: str,
        val_key: str,
    ) -> tuple[float, float, float]:
        """校验 hybrid 模式所需的 train/val 键。"""
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

    def split_by_time_strictly(
        self,
        df: FrameLike,
        *,
        time_col: str,
        target: str,
        split_ratios: dict[str, float],
        dataset_flag_col: str = "dataset_flag",
    ) -> FrameLike:
        """按时间顺序严格切分 train/val/oot。"""
        slicer = self._from_data(
            df,
            time_col=time_col,
            target=target,
            dataset_flag_col=dataset_flag_col,
        )
        slicer._validate_ratios(split_ratios)
        if slicer._engine == "pandas":
            return slicer._split_by_time_strictly_pandas(split_ratios)
        return slicer._split_by_time_strictly_polars(split_ratios)

    def split_hybrid_random_val(
        self,
        df: FrameLike,
        *,
        time_col: str,
        target: str,
        split_ratios: dict[str, float],
        dataset_flag_col: str = "dataset_flag",
        train_key: str = "train",
        val_key: str = "val",
        random_seed: int = 42,
    ) -> FrameLike:
        """在建模窗口内随机切 train/val，其余切片仍按时间顺序。"""
        slicer = self._from_data(
            df,
            time_col=time_col,
            target=target,
            dataset_flag_col=dataset_flag_col,
        )
        slicer._validate_ratios(split_ratios)
        slicer._validate_hybrid_keys(split_ratios, train_key, val_key)
        if slicer._engine == "pandas":
            return slicer._split_hybrid_random_val_pandas(
                split_ratios,
                train_key,
                val_key,
                random_seed,
            )
        return slicer._split_hybrid_random_val_polars(
            split_ratios,
            train_key,
            val_key,
            random_seed,
        )

    def split_by_target_observation(
        self,
        df: FrameLike,
        *,
        time_col: str,
        target: str,
        split_ratios: dict[str, float],
        aux_targets: Sequence[str] | None = None,
        dataset_flag_col: str = "dataset_flag",
        aux_dataset_flag_suffix: str = "__dataset_flag",
    ) -> FrameLike:
        """为主目标和辅助目标生成各自独立的切片列。"""
        is_polars_input = isinstance(df, pl.DataFrame)
        if is_polars_input:
            working_df = df.to_pandas()
        elif isinstance(df, pd.DataFrame):
            working_df = df.copy()
        else:
            raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")

        required_cols = {time_col, target, *list(aux_targets or [])}
        missing_cols = required_cols.difference(working_df.columns)
        if missing_cols:
            raise ValueError(f"Input data is missing required columns: {sorted(missing_cols)}")

        primary_split = self.split_by_time_strictly(
            working_df,
            time_col=time_col,
            target=target,
            split_ratios=dict(split_ratios),
            dataset_flag_col=dataset_flag_col,
        )
        assert isinstance(primary_split, pd.DataFrame)
        result = working_df.copy()
        result[dataset_flag_col] = primary_split[dataset_flag_col]

        for aux_target in aux_targets or []:
            aux_flag_col = f"{aux_target}{aux_dataset_flag_suffix}"
            aux_split = self.split_by_time_strictly(
                working_df,
                time_col=time_col,
                target=aux_target,
                split_ratios=dict(split_ratios),
                dataset_flag_col=aux_flag_col,
            )
            assert isinstance(aux_split, pd.DataFrame)
            result[aux_flag_col] = aux_split[aux_flag_col]

        return pl.from_pandas(result) if is_polars_input else result
