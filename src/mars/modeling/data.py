"""MARS 建模数据切分工具。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mars.modeling.base import FrameLike, is_polars_dataframe, restore_frame_type, to_pandas_frame


class MarsModelDataSlicer:
    """
    具备按天对齐边界的风控建模数据切分器。

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame
        原始输入数据集。
    time_col : str
        用于切分的时间列名。
    label_col : str
        二分类目标列名。取值不在 ``{0, 1}`` 内的样本会被归入 ``other``。
    dataset_flag_col : str, default "dataset_flag"
        输出切分标识列名。

    Attributes
    ----------
    df : pandas.DataFrame
        内部统一处理用的 Pandas 数据框。
    dataset_flag_col : str
        切分结果写入列名。
    """

    def __init__(
        self,
        df: FrameLike,
        time_col: str,
        label_col: str,
        dataset_flag_col: str = "dataset_flag",
    ) -> None:
        self._input_is_polars: bool = is_polars_dataframe(df)
        self.df: pd.DataFrame = to_pandas_frame(df)
        self.time_col: str = time_col
        self.label_col: str = label_col
        self.dataset_flag_col: str = dataset_flag_col

        missing_cols = {time_col, label_col}.difference(self.df.columns)
        if missing_cols:
            raise ValueError(f"Input data is missing required columns: {sorted(missing_cols)}")

        self.df["__clean_dt__"] = pd.to_datetime(self.df[time_col], errors="coerce")
        self.df["__date__"] = self.df["__clean_dt__"].dt.date
        self.df[self.dataset_flag_col] = "unassigned"

    def _restore(self, df: pd.DataFrame) -> FrameLike:
        """
        将内部 Pandas 结果恢复为调用方期望的公开类型。

        Parameters
        ----------
        df : pandas.DataFrame
            内部处理结果。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            与输入类型一致的结果数据框。
        """
        return restore_frame_type(df, self._input_is_polars)

    def _validate_ratios(self, split_ratios: Dict[str, float]) -> None:
        """
        校验切分比例之和是否合法。

        Parameters
        ----------
        split_ratios : dict of str to float
            切分名称与占比映射。

        Raises
        ------
        ValueError
            当比例和不为 1.0 时抛出。
        """
        if not split_ratios:
            raise ValueError("split_ratios must not be empty.")

        negative_ratios = {name: value for name, value in split_ratios.items() if float(value) < 0.0}
        if negative_ratios:
            raise ValueError(f"Split ratios must be non-negative, got {negative_ratios}.")

        total = float(sum(split_ratios.values()))
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")

    def _reset_and_mark_other(self) -> None:
        """
        清空旧切分状态，并将异常标签样本标记为 ``other``。
        """
        self.df[self.dataset_flag_col] = "unassigned"
        mask_other = ~self.df[self.label_col].isin([0, 1])
        self.df.loc[mask_other, self.dataset_flag_col] = "other"

    def _get_date_cutoffs(self, split_ratios: Dict[str, float]) -> Dict[str, Any]:
        """
        按天累计样本量，计算各切片的日期截止点。

        Parameters
        ----------
        split_ratios : dict of str to float
            时间轴上的顺序切分占比。

        Returns
        -------
        dict of str to Any
            每个切片对应的截止日期。
        """
        valid_df = self.df.dropna(subset=["__date__"]).copy()
        valid_df = valid_df[valid_df[self.dataset_flag_col] == "unassigned"]
        if valid_df.empty:
            return {}

        daily_stats = (
            valid_df.groupby("__date__", dropna=False)
            .size()
            .reset_index(name="row_count")
            .sort_values("__date__")
        )
        # 切分边界始终按“整天”为最小单位推进，避免同一天样本被拆到不同集合。
        daily_stats["cum_count"] = daily_stats["row_count"].cumsum()
        total_valid = int(daily_stats["row_count"].sum())

        cutoffs: Dict[str, Any] = {}
        cum_ratio = 0.0
        for flag, ratio in split_ratios.items():
            cum_ratio += float(ratio)
            target_count = total_valid * cum_ratio
            cutoff_df = daily_stats[daily_stats["cum_count"] >= target_count]
            if cutoff_df.empty:
                cutoff_date = daily_stats.iloc[-1]["__date__"]
            else:
                cutoff_date = cutoff_df.iloc[0]["__date__"]
            cutoffs[flag] = cutoff_date
        return cutoffs

    def split_by_time_strictly(self, split_ratios: Dict[str, float]) -> FrameLike:
        """
        按时间顺序严格按天切分整个数据集。

        Parameters
        ----------
        split_ratios : dict of str to float
            输出切片名称及其比例。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            附带 ``dataset_flag_col`` 的切分结果，类型与输入保持一致。
        """
        self._validate_ratios(split_ratios)
        self._reset_and_mark_other()

        cutoffs = self._get_date_cutoffs(split_ratios)
        for flag in split_ratios:
            cutoff_date = cutoffs.get(flag)
            if cutoff_date is None:
                continue
            mask = (
                (self.df["__date__"] <= cutoff_date)
                & (self.df[self.dataset_flag_col] == "unassigned")
            )
            self.df.loc[mask, self.dataset_flag_col] = flag

        return self._get_result()

    def split_hybrid_random_val(
        self,
        split_ratios: Dict[str, float],
        train_key: str = "train",
        val_key: str = "val",
        random_seed: int = 42,
    ) -> FrameLike:
        """
        在建模时间窗内随机切分训练/验证集，并对后续 OOT 保持时序顺延。

        Parameters
        ----------
        split_ratios : dict of str to float
            输出切片名称及其比例。
        train_key : str, default "train"
            训练集名称。
        val_key : str, default "val"
            验证集名称。
        random_seed : int, default 42
            建模区内部随机切分的随机种子。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            附带 ``dataset_flag_col`` 的切分结果，类型与输入保持一致。

        Raises
        ------
        ValueError
            当 `train_key` 或 `val_key` 缺失，或建模区比例无效时抛出。
        """
        self._validate_ratios(split_ratios)
        self._reset_and_mark_other()

        if train_key not in split_ratios:
            raise ValueError(f"train_key {train_key!r} is missing from split_ratios.")
        if val_key not in split_ratios:
            raise ValueError(f"val_key {val_key!r} is missing from split_ratios.")
        if train_key == val_key:
            raise ValueError("train_key and val_key must be different.")

        rng = np.random.default_rng(random_seed)
        train_ratio = float(split_ratios.get(train_key, 0.0))
        val_ratio = float(split_ratios.get(val_key, 0.0))
        modeling_ratio = train_ratio + val_ratio
        if modeling_ratio <= 0.0:
            raise ValueError("train_ratio + val_ratio must be greater than 0.")

        # 先把 train/val 合并成一个连续的建模窗口，再在窗口内部随机拆分。
        timeline_ratios: Dict[str, float] = {"__modeling__": modeling_ratio}
        for flag, ratio in split_ratios.items():
            if flag not in {train_key, val_key}:
                timeline_ratios[flag] = float(ratio)

        cutoffs = self._get_date_cutoffs(timeline_ratios)
        modeling_cutoff = cutoffs.get("__modeling__")

        self.df["__rand__"] = rng.random(len(self.df))
        val_internal_prob = val_ratio / modeling_ratio if modeling_ratio > 0 else 0.0

        if modeling_cutoff is not None:
            mask_modeling = (
                (self.df["__date__"] <= modeling_cutoff)
                & (self.df[self.dataset_flag_col] == "unassigned")
            )
            mask_val = mask_modeling & (self.df["__rand__"] < val_internal_prob)
            mask_train = mask_modeling & ~mask_val
            self.df.loc[mask_val, self.dataset_flag_col] = val_key
            self.df.loc[mask_train, self.dataset_flag_col] = train_key

        # 建模窗口之后的集合继续按时间顺延分配，保留典型 OOT 时序结构。
        for flag in split_ratios:
            if flag in {train_key, val_key}:
                continue
            cutoff_date = cutoffs.get(flag)
            if cutoff_date is None:
                continue
            mask = (
                (self.df["__date__"] <= cutoff_date)
                & (self.df[self.dataset_flag_col] == "unassigned")
            )
            self.df.loc[mask, self.dataset_flag_col] = flag

        return self._get_result()

    def _get_result(self) -> FrameLike:
        """
        清理内部临时列并返回最终结果。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            清理后的切分结果。
        """
        cols_to_drop = [col for col in ["__clean_dt__", "__date__", "__rand__"] if col in self.df.columns]
        return self._restore(self.df.drop(columns=cols_to_drop))
