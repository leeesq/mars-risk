"""建模样本切分工具。"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import polars as pl

from mars.utils.frame import FrameLike, is_polars_dataframe


class MarsModelDataSplitter:
    """
    无状态的二分类建模样本切分工具。

    构造函数不绑定样本表和列名；每次切分都通过
    `split_by_time_strictly` 或 `split_hybrid_random_val` 传入数据、时间列、
    目标列和输出切片列名。方法内部会创建临时工作副本，并根据输入类型选择
    Pandas 或 Polars 实现，返回结果尽量保持输入表类型。

    Attributes
    ----------
    df : pandas.DataFrame or polars.DataFrame
        单次切分期间使用的内部工作副本。
    engine_ : str
        当前工作副本对应的数据引擎；未绑定数据时为 `"stateless"`。
    time_col : str
        单次切分使用的时间列名。
    label_col : str
        单次切分使用的目标列名。

    Notes
    -----
    Pandas 输入全程走 Pandas，Polars 输入全程走 Polars，避免无收益的跨框架转换。
    时间严格切分会保证同一天不被拆到多个样本切片；hybrid 切分会先确定建模窗口，
    再在建模窗口内随机拆分 train/val。

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"apply_dt": ["2026-01-01", "2026-01-02"], "y": [0, 1]}
    ... )
    >>> splitter = MarsModelDataSplitter()
    >>> out = splitter.split_by_time_strictly(
    ...     df,
    ...     time_col="apply_dt",
    ...     target="y",
    ...     split_ratios={"train": 0.5, "val": 0.5},
    ... )
    >>> "dataset_flag" in out.columns
    True
    """

    def __init__(
        self,
    ) -> None:
        """
        初始化无状态建模样本切分器。

        数据、时间列、目标列和输出切片列名都在具体切分方法中传入。实例属性只用于
        单次切分内部工作副本，方法结束后调用方不应依赖这些临时状态。
        """
        self._input_is_polars: bool = False
        self._engine: str = "stateless"
        self.df: pl.DataFrame | pd.DataFrame = pd.DataFrame()
        self.time_col: str = ""
        self.label_col: str = ""
        self.dataset_flag_col: str = "dataset_flag"

    @classmethod
    def _from_data(
        cls,
        df: FrameLike,
        *,
        time_col: str,
        target: str,
        dataset_flag_col: str,
    ) -> MarsModelDataSplitter:
        """创建绑定单次切分上下文的内部工作副本。"""
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
        """绑定单次切分使用的数据和列名。"""
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
        """
        返回由输入类型自动选择的切分引擎。

        Returns
        -------
        str
            ``"pandas"`` 或 ``"polars"``。

        Examples
        --------
        >>> splitter = MarsModelDataSplitter()
        >>> splitter.engine_
        'stateless'
        """
        return self._engine

    def _init_pandas(self) -> None:
        """初始化 Pandas 工作副本中的清洗日期和切片标识列。"""
        assert isinstance(self.df, pd.DataFrame)
        clean_dt = pd.to_datetime(self.df[self.time_col], errors="coerce")
        self.df["__clean_dt__"] = clean_dt
        self.df["__date__"] = clean_dt.dt.date
        self.df[self.dataset_flag_col] = "unassigned"

    def _init_polars(self) -> None:
        """初始化 Polars 工作副本中的清洗日期和切片标识列。"""
        assert isinstance(self.df, pl.DataFrame)
        clean_dt = pl.col(self.time_col).cast(pl.Utf8).str.to_datetime(strict=False)
        self.df = self.df.with_columns(
            clean_dt.alias("__clean_dt__"),
            clean_dt.dt.date().alias("__date__"),
            pl.lit("unassigned").alias(self.dataset_flag_col),
        )

    def _validate_ratios(self, split_ratios: Dict[str, float]) -> None:
        """校验切分比例非空、非负且总和为 1。"""
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
        """
        校验 hybrid 模式的 train/val 键并返回建模窗口比例。

        返回值依次为 train 比例、val 比例和二者之和；当键缺失、重名或
        建模窗口比例非正时抛出异常。
        """
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
        split_ratios: Dict[str, float],
        dataset_flag_col: str = "dataset_flag",
    ) -> FrameLike:
        """
        按时间顺序严格切分，并保证同一天不被拆到多个数据集。

        Parameters
        ----------
        df : FrameLike
            原始建模样本。
        time_col : str
            原始时间列名，切分时按自然日保持完整。
        target : str
            二分类目标列名，仅 `0`/`1` 样本参与切分。
        split_ratios : Dict[str, float]
            切分名称到比例的映射，比例合计必须为 1。
        dataset_flag_col : str
            输出的数据集切片列名。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            与输入类型一致的切分结果。

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"apply_dt": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], "y": [0, 1, 0, 1]}
        ... )
        >>> splitter = MarsModelDataSplitter()
        >>> out = splitter.split_by_time_strictly(
        ...     df,
        ...     time_col="apply_dt",
        ...     target="y",
        ...     split_ratios={"train": 0.5, "val": 0.5},
        ... )
        >>> sorted(out["dataset_flag"].unique())
        ['train', 'val']
        """
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
        split_ratios: Dict[str, float],
        dataset_flag_col: str = "dataset_flag",
        train_key: str = "train",
        val_key: str = "val",
        random_seed: int = 42,
    ) -> FrameLike:
        """
        在建模窗口内随机切分 train/val，其余切片保持时间顺序。

        Parameters
        ----------
        df : FrameLike
            原始建模样本。
        time_col : str
            原始时间列名，切分时按自然日保持完整。
        target : str
            二分类目标列名，仅 `0`/`1` 样本参与切分。
        split_ratios : Dict[str, float]
            切分名称到比例的映射，比例合计必须为 1。
        dataset_flag_col : str
            输出的数据集切片列名。
        train_key : str
            训练集标识。
        val_key : str
            验证集标识。
        random_seed : int
            随机种子。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            与输入类型一致的切分结果。

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"apply_dt": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], "y": [0, 1, 0, 1]}
        ... )
        >>> splitter = MarsModelDataSplitter()
        >>> out = splitter.split_hybrid_random_val(
        ...     df,
        ...     time_col="apply_dt",
        ...     target="y",
        ...     split_ratios={"train": 0.5, "val": 0.5},
        ... )
        >>> sorted(out["dataset_flag"].unique())
        ['train', 'val']
        """
        slicer = self._from_data(
            df,
            time_col=time_col,
            target=target,
            dataset_flag_col=dataset_flag_col,
        )
        slicer._validate_ratios(split_ratios)
        slicer._validate_hybrid_keys(split_ratios, train_key, val_key)
        if slicer._engine == "pandas":
            return slicer._split_hybrid_random_val_pandas(split_ratios, train_key, val_key, random_seed)
        return slicer._split_hybrid_random_val_polars(split_ratios, train_key, val_key, random_seed)

    def split_by_target_observation(
        self,
        df: FrameLike,
        *,
        time_col: str,
        target: str,
        split_ratios: Dict[str, float],
        aux_targets: Sequence[str] | None = None,
        dataset_flag_col: str = "dataset_flag",
        aux_dataset_flag_suffix: str = "__dataset_flag",
    ) -> FrameLike:
        """
        按每个 target 的已表现样本生成独立切片列。

        Parameters
        ----------
        df : FrameLike
            原始建模样本。
        time_col : str
            原始时间列名。
        target : str
            训练使用的主目标列名。
        split_ratios : Dict[str, float]
            切分名称到比例的映射，比例合计必须为 1。
        aux_targets : Sequence[str] | None
            只参与评估的辅助目标列名。
        dataset_flag_col : str
            主目标生成的切片列名。
        aux_dataset_flag_suffix : str
            辅助目标切片列后缀，默认生成 ``<target>__dataset_flag``。

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            与输入类型一致、已追加主目标和辅助目标切片列的数据框。

        Raises
        ------
        TypeError
            当输入不是 Pandas 或 Polars DataFrame 时抛出。
        ValueError
            当输入缺少时间列、主目标或辅助目标列时抛出。

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     "apply_dt": ["2026-01-01", "2026-01-02", "2026-02-01", "2026-03-01"],
        ...     "long_y": [0, 1, None, None],
        ...     "short_y": [0, 1, 0, 1],
        ... })
        >>> splitter = MarsModelDataSplitter()
        >>> out = splitter.split_by_target_observation(
        ...     df,
        ...     time_col="apply_dt",
        ...     target="long_y",
        ...     aux_targets=["short_y"],
        ...     split_ratios={"train": 0.5, "val": 0.5},
        ... )
        >>> "short_y__dataset_flag" in out.columns
        True
        """
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

        if is_polars_input:
            return pl.from_pandas(result)
        return result

    def _reset_and_mark_other_pandas(self) -> None:
        """重置 Pandas 切片列，并将无效标签或日期样本标为 other。"""
        assert isinstance(self.df, pd.DataFrame)
        self.df[self.dataset_flag_col] = "unassigned"
        valid_mask = self.df[self.label_col].isin([0, 1]) & self.df["__date__"].notna()
        self.df.loc[~valid_mask, self.dataset_flag_col] = "other"

    def _get_date_cutoffs_pandas(self, split_ratios: Dict[str, float]) -> Dict[str, Any]:
        """
        基于 Pandas 日粒度累计样本量计算各切片截止日期。

        截止日期按输入比例顺序累积生成，保证同一天样本不会被拆到多个
        数据集切片中。
        """
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
        """将 Pandas 中尚未分配且不晚于截止日期的样本标为指定切片。"""
        assert isinstance(self.df, pd.DataFrame)
        if cutoff_date is None:
            return
        mask = (self.df["__date__"] <= cutoff_date) & (self.df[self.dataset_flag_col] == "unassigned")
        self.df.loc[mask, self.dataset_flag_col] = flag

    def _split_by_time_strictly_pandas(self, split_ratios: Dict[str, float]) -> pd.DataFrame:
        """按 Pandas 路径执行严格时间顺序切分。"""
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
        """
        在 Pandas 路径下执行建模窗口内随机 val 切分。

        训练集和验证集共享同一个时间窗口，窗口外切片仍按时间顺序分配；
        无效标签或无效日期样本会进入 ``other``。
        """
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
        """返回清理辅助列后的 Pandas 切分结果。"""
        assert isinstance(self.df, pd.DataFrame)
        result = self.df.copy()
        result.loc[result[self.dataset_flag_col] == "unassigned", self.dataset_flag_col] = "other"
        cols_to_drop = [col for col in ["__clean_dt__", "__date__", "__rand__"] if col in result.columns]
        return result.drop(columns=cols_to_drop)

    def _reset_and_mark_other_polars(self) -> None:
        """重置 Polars 切片列，并将无效标签或日期样本标为 other。"""
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
        """
        基于 Polars 日粒度累计样本量计算各切片截止日期。

        该实现保持 Polars 原生执行路径，避免为切分边界计算额外转换到
        Pandas。
        """
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
        """将 Polars 中尚未分配且不晚于截止日期的样本标为指定切片。"""
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
        """按 Polars 路径执行严格时间顺序切分。"""
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
        """
        在 Polars 路径下执行建模窗口内随机 val 切分。

        随机列仅在切分过程中临时存在，返回前会与清洗日期辅助列一起删除，
        保证输出 schema 只包含业务列和 dataset flag。
        """
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
        """返回清理辅助列后的 Polars 切分结果。"""
        assert isinstance(self.df, pl.DataFrame)
        result = self.df.with_columns(
            pl.when(pl.col(self.dataset_flag_col) == "unassigned")
            .then(pl.lit("other"))
            .otherwise(pl.col(self.dataset_flag_col))
            .alias(self.dataset_flag_col)
        )
        cols_to_drop = [col for col in ["__clean_dt__", "__date__", "__rand__"] if col in result.columns]
        return result.drop(cols_to_drop)
