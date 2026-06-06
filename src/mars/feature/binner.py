"""MARS 分箱器实现模块。"""

import gc
import multiprocessing
from collections.abc import Iterator
from typing import Any, Dict, List, Literal, Set, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from optbinning import OptimalBinning
from sklearn.tree import DecisionTreeClassifier

from mars.core.base import MarsTransformer
from mars.utils.decorators import time_it
from mars.utils.logger import logger


class MarsBinnerBase(MarsTransformer):
    """
    MARS 分箱器抽象基类。

    该基类定义了分箱器的公共状态、索引协议以及数值/类别特征的统一转换行为。
    子类负责学习切点或类别分组规则，基类则负责缓存管理、映射导出、
    WOE 物化、统计报告和 SQL 生成。

    Attributes
    ----------
    bin_cuts_ : dict of str to list of float
        数值型特征的物理切点，每个列表均以 ``[-inf, ..., inf]`` 闭合。
    cat_cuts_ : dict of str to list of list
        类别型特征的分组映射规则。将零散的字符串/分类标签聚类为逻辑组。
    bin_mappings_ : dict of str to dict of int to str
        分箱可视化映射表。将物理索引映射为业务可读标签。
    bin_woes_ : dict of str to dict of int to float
        分箱权重字典。存储每个分箱索引对应的 WOE 值。
    feature_names_in_ : list of str
        拟合时输入的原始特征列名。
    fit_failures_ : dict of str to str
        拟合过程中失败的特征及其诊断信息。

    Notes
    -----
    索引协议如下：
    ``Missing`` 为 ``-1``，
    ``Other`` 为 ``-2``，
    ``Special`` 从 ``-3`` 开始向负方向扩展，
    正常分箱索引从 ``0`` 开始递增。

    Examples
    --------
    >>> issubclass(MarsNativeBinner, MarsBinnerBase)
    True
    """

    # 类型常量: 用于快速判定数值列
    NUMERIC_DTYPES: Set[pl.DataType] = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    }

    # 索引协议常量
    IDX_MISSING: int = -1
    IDX_OTHER: int = -2
    IDX_SPECIAL_START: int = -3

    def __init__(
        self,
        n_bins: int = 10,
        special_values: List[Union[int, float, str]] | None = None,
        missing_values: List[Union[int, float, str]] | None = None,
        join_threshold: int = 100,
        n_jobs: int = -1
   ) -> None:
        """
        初始化分箱器基类, 配置全局业务规则与并行策略。

        Parameters
        ----------
        n_bins : int
            期望的最大分箱数量。最终生成的箱数可能少于此值 (受单调性约束或样本量影响)。
        special_values : List[Union[int, float, str]] | None
            特殊值列表。
            - 在部分场景中, 某些特定取值 (如 -999, -1)代表特定含义, 会被强制分配到独立的负数索引分箱中, 不参与正常区间的切分。
        missing_values : List[Union[int, float, str]] | None
            自定义缺失值列表。除了原生的 `null` 和 `NaN` 外, 用户可指定其他代表缺失的值。
        join_threshold : int
            在 `transform` 阶段, 为防止因构建过深的逻辑分支树 (When-Then Tree)导致的计算图解析缓慢:
            - 当类别特征的基数 (Unique Values) 低于此值时, 使用内存级 `replace` 映射。
            - 当基数超过此值时, 自动切换为 `Hash Join` 模式。
        n_jobs : int
            并行计算的核心数:
            - `-1`: 自动使用 `CPU核心数 - 1`, 预留一个核心保证系统响应。
            - `1`: 强制单线程模式, 便于调试。
            - `N`: 使用指定的核心数。

        Notes
        -----
        初始化阶段不执行任何重型计算。所有计算资源 (进程池、线程池) 均在 `fit` 阶段按需按需申请。
        """
        super().__init__()
        self.features: list[str] = []
        self.cat_features: list[str] = []
        self.n_bins = n_bins
        self.special_values = special_values if special_values is not None else []
        self.missing_values = missing_values if missing_values is not None else []
        self.join_threshold = join_threshold
        self.n_jobs = max(1, multiprocessing.cpu_count() - 1) if n_jobs == -1 else n_jobs

        # 状态属性初始化
        self.bin_cuts_: Dict[str, List[float]] = {}
        self.cat_cuts_: Dict[str, List[List[Any]]] = {}
        self.bin_mappings_: Dict[str, Dict[int, str]] = {}
        self.bin_woes_: Dict[str, Dict[int, float]] = {}

        # 缓存引用
        self._cache_X: pl.DataFrame | None = None
        self._cache_y: Any | None = None

        self.fit_failures_: Dict[str, str] = {}

    def transform(
        self,
        X: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame],
        *,
        return_type: Literal["index", "label", "woe"] = "index",
        woe_batch_size: int = 200,
        lazy: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame]:
        """
        按当前分箱规则将输入特征映射为索引、标签或 WOE 值。

        Parameters
        ----------
        X : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame]
            待转换的数据集。
        return_type : Literal['index', 'label', 'woe']
            输出形式。``"index"`` 返回分箱索引，``"label"`` 返回分箱标签，
            ``"woe"`` 返回对应分箱的 WOE 值。
        woe_batch_size : int
            当 ``return_type="woe"`` 且当前实例尚未物化 WOE 映射时，
            计算 WOE 的批处理特征数。
        lazy : bool
            是否保持延迟执行。为 ``True`` 时返回 ``pl.LazyFrame``。

        Returns
        -------
        pl.DataFrame or pd.DataFrame or pl.LazyFrame
            分箱转换结果。若设置了 ``set_output("pandas")`` 且结果为 eager
            DataFrame，则返回 Pandas 对象。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, features=["age"])
        >>> binner.transform(X).columns
        ['age_bin']
        """
        self._check_is_fitted()

        X_pl = self._ensure_polars_dataframe(X)
        X_new = self._transform_impl(
            X_pl,
            return_type=return_type,
            woe_batch_size=woe_batch_size,
            lazy=lazy,
        )
        return self._format_output(X_new)

    def fit_transform(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Any | None = None,
        *,
        features: List[str] | None = None,
        cat_features: List[str] | None = None,
        return_type: Literal["index", "label", "woe"] = "index",
        woe_batch_size: int = 200,
        lazy: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame]:
        """
        先拟合分箱器，再返回分箱转换结果。

        Parameters
        ----------
        X : Union[pl.DataFrame, pd.DataFrame]
            输入特征矩阵。
        y : Any | None
            目标变量。无监督分箱场景下可为空。
        features : List[str] | None
            本次拟合和转换的特征列；不传时使用全部候选列。
        cat_features : List[str] | None
            明确指定的类别特征列。
        return_type : Literal['index', 'label', 'woe']
            转换结果形式。
        woe_batch_size : int
            计算 WOE 映射时的批处理特征数。
        lazy : bool
            是否返回 ``pl.LazyFrame``。

        Returns
        -------
        pl.DataFrame or pd.DataFrame or pl.LazyFrame
            分箱转换结果。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2)
        >>> binner.fit_transform(X, return_type="label").columns
        ['age_bin']
        """
        return self.fit(X, y, features=features, cat_features=cat_features).transform(
            X,
            return_type=return_type,
            woe_batch_size=woe_batch_size,
            lazy=lazy,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        将分箱器状态序列化为 Python 字典。

        Returns
        -------
        dict of str to Any
            包含 ``params`` 与 ``state`` 两部分的可序列化字典。

        Notes
        -----
        返回结果只包含分箱规则和必要状态，不包含缓存的训练数据本体。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, features=["age"])
        >>> sorted(binner.to_dict().keys())
        ['params', 'state']
        """
        return {
            "params": {
                "n_bins": self.n_bins,
                "special_values": self.special_values,
                "missing_values": self.missing_values,
                "join_threshold": self.join_threshold,
                # 注意: 子类可能还有额外的 params (如 solver), 子类可以考虑重写
            },
            "state": {
                "bin_cuts_": self.bin_cuts_,
                "cat_cuts_": getattr(self, "cat_cuts_", {}), # 兼容可能没有 cat_cuts_ 的情况
                "bin_mappings_": self.bin_mappings_,
                "bin_woes_": self.bin_woes_,
                # 保存失败记录, 使用 getattr 防止未 fit 时报错
                "fit_failures_": getattr(self, "fit_failures_", {})
            }
        }

    @staticmethod
    def _build_trend_shape_frame(
        grouped_woe_df: pl.DataFrame,
        trend_col_name: str = "trend_shape"
    ) -> pl.DataFrame:
        """
        基于聚合后的 WOE 列表构建趋势形态结果表。

        Parameters
        ----------
        grouped_woe_df : pl.DataFrame
            至少包含 `feature` 与 `woe` 列的聚合结果。
        trend_col_name : str
            输出趋势列名。

        Returns
        -------
        pl.DataFrame
            包含特征名与趋势标签的结果表。
        """
        rows = []
        for row in grouped_woe_df.iter_rows(named=True):
            rows.append({
                "feature": row["feature"],
                trend_col_name: MarsBinnerBase._detect_trend_scientific(row["woe"]),
            })

        if not rows:
            return pl.DataFrame(schema={"feature": pl.Utf8, trend_col_name: pl.Utf8})

        return pl.DataFrame(rows)

    @classmethod
    def from_dict(cls: type["MarsBinnerBase"], data: Dict[str, Any]) -> "MarsBinnerBase":
        """
        从字典恢复分箱器实例。

        Parameters
        ----------
        data : Dict[str, Any]
            由 ``to_dict`` 生成的状态字典。

        Returns
        -------
        MarsBinnerBase
            恢复后的已拟合分箱器实例。

        Examples
        --------
        >>> class DemoBinner(MarsBinnerBase):
        ...     def _fit_impl(self, X: pl.DataFrame, y: Any | None = None) -> None:
        ...         return None
        ...     def _transform_impl(self, X: pl.DataFrame) -> pl.DataFrame:
        ...         return X
        >>> state = {
        ...     "params": {"features": ["age"], "n_bins": 2},
        ...     "state": {"bin_mappings_": {"age": {0: "young"}}},
        ... }
        >>> DemoBinner.from_dict(state).get_bin_mapping("age")
        {0: 'young'}
        """
        # 实例化一个空对象
        instance = cls(**data["params"])

        # 恢复训练后的状态
        state: Dict[str, Any] = data["state"]
        instance.bin_cuts_ = state.get("bin_cuts_", {})
        instance.cat_cuts_ = state.get("cat_cuts_", {})
        instance.bin_mappings_ = state.get("bin_mappings_", {})
        instance.bin_woes_ = state.get("bin_woes_", {})

        # 恢复失败记录
        instance.fit_failures_ = state.get("fit_failures_", {})

        instance._is_fitted = True
        return instance

    def __getstate__(self) -> dict[str, Any]:
        """
        Pickle 序列化时的钩子。

        在保存模型时, 自动剔除巨大的训练数据缓存, 只保留配置和计算结果。
        """
        state = self.__dict__.copy()
        # 移除大数据缓存, 防止模型文件变成几百 MB
        state["_cache_X"] = None
        state["_cache_y"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Pickle 反序列化时的钩子。

        恢复模型状态, 并将缓存初始化为 None。
        """
        self.__dict__.update(state)
        # 确保属性存在, 防止 AttributeError
        if "_cache_X" not in self.__dict__:
            self._cache_X = None
        if "_cache_y" not in self.__dict__:
            self._cache_y = None

    def clear_cache(self) -> None:
        """
        清理缓存的训练数据引用。

        Notes
        -----
        该方法会清空 ``_cache_X`` 和 ``_cache_y``，并主动触发一次垃圾回收。
        适合在模型训练完成、无需再次即时重算统计量时调用。

        Returns
        -------
        None
            函数仅清理训练数据缓存。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, features=["age"])
        >>> binner.clear_cache()
        >>> binner._cache_X is None
        True
        """
        self._cache_X = None
        self._cache_y = None
        gc.collect()

    def _get_safe_values(self, dtype: pl.DataType, values: List[Any]) -> List[Any]:
        """
        根据列类型筛选可安全参与比较的配置值。

        在强类型引擎 (如 Polars)中, 类型不匹配是导致崩溃的主要原因。该方法通过预扫描
        Schema, 确保用户定义的业务逻辑 (缺失值、特殊值)与数据的物理存储类型保持绝对兼容。

        Parameters
        ----------
        dtype : pl.DataType
            当前处理列的原始数据类型。
        values : List[Any]
            用户在配置中指定的数值列表 (如 [-999, 'unknown', None])。

        Returns
        -------
        List[Any]
            经过物理类型对齐后的清洗列表。

        Notes
        -----
        1. 严格过滤机制：
        若目标列为数值型, 系统会剔除所有非数值项。特别地, 由于 Python 中 `True == 1`,
        系统会显式排除布尔类型, 防止逻辑误判导致的异常成箱。

        2. 宽容转换机制：
        若目标列为非数值型, 系统会将所有配置项强制转换为字符串。这保证了在进行
        `is_in` 操作或 `join` 操作时, 比较操作发生在相同的物理类型之上。

        3. 空值剥离：
        `None` 和 `np.nan` 会在此阶段被剥离, 转由 `is_null()` 和 `is_nan()` 算子在
        Polars 内核中进行更高效率的处理。
        """
        if not values:
            return []

        is_numeric = dtype in self.NUMERIC_DTYPES
        safe_vals = []

        for v in values:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue

            if is_numeric:
                # 数值列: 严格保留数值, 剔除 bool (True==1 歧义) 和字符串
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    safe_vals.append(v)
            else:
                # 非数值列: 宽容处理, 全部转为字符串以匹配 Categorical/String 列
                safe_vals.append(str(v))

        return safe_vals

    def get_bin_mapping(self, col: str) -> Dict[int, str]:
        """
        获取指定特征的分箱标签映射。

        Parameters
        ----------
        col : str
            特征名称。

        Returns
        -------
        dict of int to str
            分箱索引到标签的映射字典。若该特征不存在，则返回空字典。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, features=["age"])
        >>> isinstance(binner.get_bin_mapping("age"), dict)
        True
        """
        return self.bin_mappings_.get(col, {})

    def _is_numeric(self, series: pl.Series) -> bool:
        """判断输入序列是否为数值类型。"""
        if series.dtype == pl.Null:
            return False
        return series.dtype in self.NUMERIC_DTYPES

    def _materialize_woe(self, batch_size: int = 200) -> None:
        """
        将分箱统计分布转化为 WOE 映射表。

        该方法按特征逐列聚合分箱统计量，避免将宽表物化成超长表，
        从而显著降低大样本、超宽表场景下的峰值内存占用。

        Parameters
        ----------
        batch_size : int
            分批处理的特征数量。
        """
        if self._cache_X is None or self._cache_y is None:
            logger.warning("No training data cached. WOE cannot be computed.")
            return

        n_cols = len(self.bin_cuts_) + len(self.cat_cuts_)
        logger.info(f"Materializing WOE mappings for {n_cols} features.")

        y_name = "_y_tmp"
        y_series = pl.Series(name=y_name, values=self._cache_y)
        epsilon: float = 1e-6
        total_bads: float = float(y_series.sum() or 0.0)
        total_goods: float = float(len(y_series)) - total_bads

        # 涵盖数值和类别特征
        bin_cols_orig = [
            c for c in self.bin_cuts_.keys()] + (list(self.cat_cuts_.keys())
            if hasattr(self, 'cat_cuts_') else []
        )

        for i in range(0, len(bin_cols_orig), batch_size):
            batch_features = bin_cols_orig[i: i + batch_size]

            X_batch_bin: pl.DataFrame = self.transform(
                self._cache_X.select(batch_features),
                return_type="index",
                lazy=False
            )
            X_batch_bin = X_batch_bin.with_columns(y_series)

            # 逐列聚合分箱统计量，避免宽转长物化出 N 行 * M 特征的长表。
            stats_list = []
            for feature in batch_features:
                target_bin_col = f"{feature}_bin"
                if target_bin_col not in X_batch_bin.columns:
                    continue

                stats_list.append(
                    X_batch_bin.group_by(target_bin_col)
                    .agg([
                        pl.col(y_name).sum().alias("bin_bads"),
                        pl.len().alias("bin_total")
                    ])
                    .rename({target_bin_col: "bin_index"})
                    .with_columns(pl.lit(feature).alias("feature"))
                    .select(["feature", "bin_index", "bin_bads", "bin_total"])
                )

            if not stats_list:
                del X_batch_bin
                gc.collect()
                continue

            stats_df = (
                pl.concat(stats_list)
                .with_columns(
                    (
                        ((pl.col("bin_bads") + epsilon) / (total_bads + epsilon))
                        /
                        (
                            (pl.col("bin_total") - pl.col("bin_bads") + epsilon)
                            / (total_goods + epsilon)
                        )
                    )
                    .log()
                    .cast(pl.Float32)
                    .alias("woe")
                )
            )

            woe_data = stats_df.select(["feature", "bin_index", "woe"]).to_dict(as_series=False)

            from collections import defaultdict
            temp_woe_map = defaultdict(dict)

            for f, b, w in zip(
                woe_data["feature"],
                woe_data["bin_index"],
                woe_data["woe"],
                strict=False,
            ):
                # 严格过滤: 只有合法的索引 (-1, 0, 1...) 允许进入 WOE 映射表
                if b is not None and not (isinstance(b, float) and np.isnan(b)):
                    temp_woe_map[f][int(b)] = w

            self.bin_woes_.update(temp_woe_map)

            del X_batch_bin, stats_list, stats_df
            gc.collect()

    def _transform_impl(
        self,
        X: Union[pl.DataFrame, pl.LazyFrame],
        return_type: Literal["index", "label", "woe"] = "index",
        woe_batch_size: int = 200,
        lazy: bool = False
   ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        [分箱转换]

        兼容数值与类别特征, 支持 Eager 与 Lazy 模式。

        该方法采用了“表达式瀑布流 (Expression Waterfall)”设计, 通过 Polars 的原生算子实现
        了高效的向量化转换。针对高基数类别特征, 采用了 Join 优化策略以规避深层逻辑树带来的性能损耗。

        Parameters
        ----------
        X : Union[pl.DataFrame, pl.LazyFrame]
            待转换的数据集。支持延迟计算流 (LazyFrame) 以优化长流水线性能。
        return_type : Literal['index', 'label', 'woe']
            转换后的输出格式:
            - 'index': 输出分箱索引 (Int16 类型)。
            - 'label': 输出分箱的可读标签 (Utf8 类型, 如 "01_[10.5, 20.0)")。
            - 'woe': 输出对应的 WOE 值 (Float32 类型)。
        woe_batch_size : int
            仅在 return_type='woe' 且未预计算 WOE 时有效。指定并行计算 WOE 的批大小。
            - 若遇到内存溢出 (OOM)，请将此值调小 (如 50)；若内存充足，调大此值可提升吞吐量。
        lazy : bool
            是否保持延迟执行状态。若为 True, 则无论输入是 Eager 还是 Lazy, 均返回 LazyFrame。

        Returns
        -------
        Union[pl.DataFrame, pl.LazyFrame]
            转换后的数据集。原列保持不变, 新增以 `_bin` 或 `_woe` 为后缀的转换列。

        Notes
        -----
        1. 分箱索引协议，为了确保与下游 Profiler 和 PSI 计算算子对齐, 系统采用以下固定索引:
        - `IDX_MISSING (-1)`: 缺失值及自定义缺失值。
        - `IDX_OTHER (-2)`: 类别型特征中的未见类别 (Unseen categories)。
        - `IDX_SPECIAL_START (-3)`: 特殊值分箱起始索引 (向负无穷延伸)。
        - `[0, N]`: 正常数值区间或类别分组索引。

        2. 数值型转换：
        - 预处理: 利用 `_get_safe_values` 确保缺失值/特殊值的类型与列 Schema 严格一致。
        - core: 使用 `pl.cut` 进行向量化区间划分。
        - 组合: 通过 `pl.when().then()` 瀑布流, 按照 "缺失值 -> 特殊值 -> 正常区间" 的优先级进行合并。

        3. 类别型转换：
        - **路径 A (低基数)**: 使用 `replace` 算子进行内存级映射, 速度极快。
        - **路径 B (高基数)**: 当类别数超过 `join_threshold` 时, 自动转为 `Join` 模式。
            这避免了构建数千个 `when-then` 分支导致的逻辑树深度爆炸 (Stack Overflow 风险),
            将逻辑判断转化为哈希连接操作, 极大提升了宽表转换效率。

        4. 自动路由与路由安全：
        - 在进行 Utf8 类型操作 (如类别分组)前, 系统会自动创建临时 Utf8 缓存列。
        - 转换结束后, 会自动清理所有产生的中间 Join 列和临时缓存列, 保证输出 Schema 纯净。
        """
        exprs = []
        temp_join_cols = []

        # 索引协议常量: 与下游 Profiler 对齐
        IDX_MISSING = -1
        IDX_OTHER   = -2
        IDX_SPECIAL_START = -3

        # 自动触发 WOE 计算
        if return_type == "woe" and not self.bin_woes_:
            self._materialize_woe(woe_batch_size)

        # 获取 Schema
        schema_map = X.collect_schema() if isinstance(X, pl.LazyFrame) else X.schema
        current_columns = schema_map.names()

        all_train_cols = list(set(
            list(self.bin_cuts_.keys()) +
            (list(self.cat_cuts_.keys()) if hasattr(self, 'cat_cuts_') else [])
       ))

        for col in all_train_cols:
            if col not in current_columns:
                continue

            # 计算类型安全值, 防止例如在 Int 列上查询 "unknown" 导致的崩溃
            col_dtype = schema_map[col]
            safe_missing_vals: List[int|float] = self._get_safe_values(col_dtype, self.missing_values)
            safe_special_vals: List[int|float] = self._get_safe_values(col_dtype, self.special_values)
            is_numeric_col = col_dtype in self.NUMERIC_DTYPES

            # Part A: 数值型分箱 (Numeric Binning)
            if col in self.bin_cuts_:
                cuts = self.bin_cuts_[col]

                # 缺失值逻辑: Is Null OR Is Missing Val
                missing_cond = pl.col(col).is_null()
                if is_numeric_col:
                    missing_cond |= pl.col(col).cast(pl.Float64).is_nan()
                for v in safe_missing_vals:
                    missing_cond |= (pl.col(col) == v)
                # 先将缺失值映射到统一索引，再叠加特殊值和常规分箱逻辑。
                layer_missing = pl.when(missing_cond).then(pl.lit(IDX_MISSING, dtype=pl.Int16))

                # 正常分箱逻辑: Cut
                raw_breaks = cuts[1:-1] if len(cuts) > 2 else []
                # `pl.cut` 要求切点严格递增；高偏态数据可能产生重复分位点，这里统一去重并排序。
                breaks = sorted(list(set(raw_breaks)))

                col_mapping: Dict[int, str] = {IDX_MISSING: "Missing", IDX_OTHER: "Other"} # 分箱标签映射表 IDX -> Label

                # 无切点逻辑
                if not breaks:
                    col_mapping[0] = "00_[-inf, inf)"
                    layer_normal = pl.lit(0, dtype=pl.Int16) # 全部归为 0 号箱
                else:
                    for i in range(len(cuts) - 1):
                        low, high = cuts[i], cuts[i+1]
                        # 调用智能格式化函数
                        low_str = self._format_cut_point(low)
                        high_str = self._format_cut_point(high)
                        col_mapping[i] = f"{i:02d}_[{low_str}, {high_str})"

                    # 显式生成标签，确保后续转换稳定地落回业务索引。
                    bin_labels: List[str] = [str(i) for i in range(len(breaks) + 1)]
                    layer_normal = (
                        pl.col(col)
                        .cut(breaks, labels=bin_labels, left_closed=True)
                        # 先还原逻辑标签，再转回业务索引，避免直接读取底层物理 ID。
                        .cast(pl.Utf8)
                        .cast(pl.Int16)
                   )

                # 特殊值逻辑: 瀑布流覆盖
                current_branch = layer_normal
                if safe_special_vals:
                    for i in range(len(safe_special_vals)-1, -1, -1):
                        v = safe_special_vals[i]
                        idx = IDX_SPECIAL_START - i
                        col_mapping[idx] = f"Special_{v}"
                        # 注意这里的覆盖顺序: 后定义的优先级更高
                        current_branch = pl.when(pl.col(col) == v).then(pl.lit(idx, dtype=pl.Int16)).otherwise(current_branch)

                # 优先级为 Missing -> Special -> Normal。
                final_idx_expr = layer_missing.otherwise(current_branch)
                self.bin_mappings_[col] = col_mapping

            # Part B: 类别型分箱 (Categorical Binning)
            elif hasattr(self, 'cat_cuts_') and col in self.cat_cuts_:
                splits = self.cat_cuts_[col]
                cat_to_idx: Dict[str, int] = {}
                idx_to_label: Dict[int, str] = {IDX_MISSING: "Missing", IDX_OTHER: "Other"}

                # 未命中任何已知类别时，默认落入 Other 箱。
                default_bin_idx = IDX_OTHER

                # 更新映射表
                if safe_special_vals:
                    for i, val in enumerate(safe_special_vals):
                        idx_to_label[IDX_SPECIAL_START - i] = f"Special_{val}"

                for i, group in enumerate(splits):
                    disp_grp = group[:3] if len(group) > 3 else group
                    suffix = ",..." if len(group) > 3 else ""
                    idx_to_label[i] = f"{i:02d}_[{','.join(str(g) for g in disp_grp) + suffix}]"
                    for val in group:
                        val_str = str(val)
                        cat_to_idx[val_str] = i
                        # 若训练阶段预聚合了 Other 占位类别，则沿用该箱作为默认去向。
                        if val_str == "__Mars_Other_Pre__":
                            default_bin_idx = i

                self.bin_mappings_[col] = idx_to_label
                # 强转 String, 确保类别匹配安全
                target_col = pl.col(col).cast(pl.Utf8)

                # 缺失值
                missing_cond = target_col.is_null() | (target_col == "nan") # Polars 中 NaN 的字符串表现形式
                for v in safe_missing_vals:
                    missing_cond |= (target_col == str(v))
                layer_missing = pl.when(missing_cond).then(pl.lit(IDX_MISSING, dtype=pl.Int16))

                # 特殊值
                current_branch = pl.lit(default_bin_idx, dtype=pl.Int16)
                if safe_special_vals:
                    for i in range(len(safe_special_vals)-1, -1, -1):
                        v = safe_special_vals[i]
                        idx = IDX_SPECIAL_START - i
                        # 如果是特殊值则赋予 -3，否则掉入上一层的 current_branch (即 default_bin_idx)
                        current_branch = (
                            pl.when(target_col == str(v))
                            .then(pl.lit(idx, dtype=pl.Int16))
                            .otherwise(current_branch)
                       )

                target_col_name = col
                if col_dtype != pl.Utf8:
                    target_col_name = f"_{col}_utf8_tmp"
                    X = X.with_columns(pl.col(col).cast(pl.Utf8).alias(target_col_name))

                # 路由: Join (高基数) vs Replace (低基数)
                if len(cat_to_idx) > self.join_threshold:
                    map_df = pl.DataFrame({
                        "_k": list(cat_to_idx.keys()),
                        f"_idx_{col}": list(cat_to_idx.values())
                    }).cast({"_k": pl.Utf8, f"_idx_{col}": pl.Int16})

                    # 跟随输入数据的 eager/lazy 形态，避免额外的执行模式切换。
                    join_tbl = map_df.lazy() if isinstance(X, pl.LazyFrame) else map_df
                    X = X.join(join_tbl, left_on=target_col_name, right_on="_k", how="left")
                    temp_join_cols.append(f"_idx_{col}")
                    if target_col_name != col:
                        temp_join_cols.append(target_col_name)

                    layer_normal = pl.col(f"_idx_{col}")
                else:
                    # 类别型特征的 Replace 逻辑
                    # 1. 显式转 String 确保匹配安全
                    str_map = {k: str(v) for k, v in cat_to_idx.items()}

                    # 保持 default=None，让特殊值和未知类别都进入后续兜底分支统一处理。
                    layer_normal = (
                        target_col
                        .replace_strict(str_map, default=None)
                        .cast(pl.Int16)
                    )

                # 最终的分箱表达式: Missing -> Normal (Join/Replace Result) -> Special/Other
                final_idx_expr = layer_missing.otherwise(
                    pl.when(layer_normal.is_not_null()).then(layer_normal).otherwise(current_branch)
                )
            else:
                continue

            # 输出分发
            if return_type == "index":
                exprs.append(final_idx_expr.alias(f"{col}_bin"))
            elif return_type == "woe":
                woe_map = self.bin_woes_.get(col, {})
                if woe_map:
                    # 对旧模型或外部加载映射做一次键类型清洗，避免脏键污染 replace。
                    clean_woe_map = {
                        int(k): float(v) for k, v in woe_map.items()
                        if k is not None and not (isinstance(k, float) and np.isnan(k))
                    }

                    # 未命中的索引统一记为 0.0，避免原始分箱索引泄露到 WOE 输出。
                    expr = final_idx_expr.replace_strict(clean_woe_map, default=0.0).cast(pl.Float32)
                else:
                    # 如果压根没映射表, 保持原样的全列 0.0
                    expr = pl.lit(0.0)
                    logger.warning(f"WOE mapping for column '{col}' not found. Defaulting to 0.0.")
                exprs.append(expr.alias(f"{col}_woe"))
            else:
                str_map = {str(k): v for k, v in self.bin_mappings_.get(col, {}).items()}
                exprs.append(final_idx_expr.cast(pl.Utf8).replace(str_map).alias(f"{col}_bin"))

        return X.with_columns(exprs).drop(temp_join_cols).lazy() if lazy else X.with_columns(exprs).drop(temp_join_cols)

    @staticmethod
    def _detect_trend_scientific(woes: List[float]) -> str:
        """基于差分的严格单调性与峰谷检测"""
        y = np.array([w for w in woes if w is not None and not np.isnan(w)])
        n = len(y)

        if n < 2:
            return "scanty"

        # 计算差分
        diff = np.diff(y)

        # 严格单调性 (Ascending / Descending)
        if np.all(diff >= 0):
            return "ascending"
        if np.all(diff <= 0):
            return "descending"

        if n < 3:
            return "undefined" # 非单调且点数少于3，无法构成峰谷

        # Peak (倒U型)
        #    Max 必须在中间 (0 < t < n-1)
        t_max = np.argmax(y)
        if 0 < t_max < n - 1:
            # 左侧单调增，右侧单调减
            if np.all(diff[:t_max] >= 0) and np.all(diff[t_max:] <= 0):
                return "peak"

        # Valley (U型)
        #    Min 必须在中间 (0 < t < n-1)
        t_min = np.argmin(y)
        if 0 < t_min < n - 1:
            # 左侧单调减，右侧单调增
            if np.all(diff[:t_min] <= 0) and np.all(diff[t_min:] >= 0):
                return "valley"

        return "undefined"

    @time_it
    def profile_bin_performance(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: pl.Series | pd.Series,
        update_woe: bool = True,
        batch_size: int = 100
    ) -> pl.DataFrame | pd.DataFrame:
        """
        计算分箱表现统计报告。

        Parameters
        ----------
        X : pl.DataFrame | pd.DataFrame
            原始特征数据集。
        y : pl.Series | pd.Series
            二分类目标标签。
        update_woe : bool
            是否将本次计算得到的 WOE 同步回写到 ``bin_woes_``。
        batch_size : int
            特征分批处理大小。减小该值可进一步降低内存峰值。

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            包含各特征各分箱统计量的明细表，通常包括样本数、坏样本数、
            分布占比、WOE、IV、KS、AUC 和 Lift 等指标。

        Notes
        -----
        该方法依赖当前分箱器已完成拟合，并会复用 ``transform(return_type="index")``
        的输出结果来执行聚合统计。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> y = pl.Series("target", [0, 0, 1, 1])
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, y, features=["age"])
        >>> stats = binner.profile_bin_performance(X, y)
        >>> "feature" in stats.columns
        True
        """
        X = self._ensure_polars_dataframe(X)

        raw_name = getattr(y, "name", None)
        if raw_name is None or raw_name == "":
            y_name = "target"
        else:
            y_name = str(raw_name)
        y = self._ensure_polars_series(y, name=y_name)

        X_bin_lazy: pl.LazyFrame = self.transform(X, return_type="index", lazy=True)
        X_bin_lazy = X_bin_lazy.with_columns(pl.lit(np.array(y)).alias(y_name))

        # 获取全局统计量
        meta = X_bin_lazy.select([
            pl.len().alias("total_counts"),
            pl.col(y_name).sum().alias("total_bads")
        ]).collect()

        epsilon: float = 1e-6
        total_counts: float = float(meta[0, "total_counts"] or 0.0)
        total_bads: float = float(meta[0, "total_bads"] or 0.0)
        total_goods: float = total_counts - total_bads
        global_bad_rate = (total_bads / total_counts) if total_counts > 0 else 0

        current_cols = X_bin_lazy.collect_schema().names()
        bin_cols = [c for c in current_cols if c.endswith("_bin")]

        agg_results: List[pl.DataFrame] = []
        for i in range(0, len(bin_cols), batch_size):
            batch_cols = bin_cols[i : i + batch_size]

            # 构建仅针对当前批次的查询计划，并在聚合后立即 collect() 物化为极小的表
            batch_stats = (
                X_bin_lazy
                .select([y_name] + batch_cols)
                .unpivot(
                    index=[y_name],
                    on=batch_cols,
                    variable_name="feature",
                    value_name="bin_index"
                )
                .group_by(["feature", "bin_index"])
                .agg([
                    pl.len().alias("count"),
                    pl.col(y_name).sum().alias("bad")
                ])
                .with_columns(
                    pl.col("feature").str.replace("_bin", "")
                )
                .collect(engine="streaming")
            )
            agg_results.append(batch_stats)

        if not agg_results:
            return self._format_output(pl.DataFrame())

        stats_df = pl.concat(agg_results)
        del agg_results
        gc.collect()

        # 基础计算
        stats_df = stats_df.with_columns([
            (pl.col("count") - pl.col("bad")).alias("good")
        ]).with_columns([
            (pl.col("count") / total_counts).cast(pl.Float32).alias("count_dist"),
            (pl.col("bad") / pl.col("count")).cast(pl.Float32).alias("bad_rate"),
            (pl.col("bad") / (total_bads + epsilon)).cast(pl.Float32).alias("bad_dist"),
            (pl.col("good") / (total_goods + epsilon)).cast(pl.Float32).alias("good_dist")
        ])

        # 计算 WOE 与 IV
        stats_df = (
            stats_df
            .with_columns([
                (
                    ((pl.col("bad") + epsilon) / (total_bads + epsilon))
                    /
                    ((pl.col("good") + epsilon) / (total_goods + epsilon))
                )
                .log()
                .cast(pl.Float32)
                .alias("woe")
            ])
            .with_columns([
                ((pl.col("bad_dist") - pl.col("good_dist")) * pl.col("woe")).cast(pl.Float32).alias("bin_iv")
            ])
        )

        # 计算 KS 和 AUC
        stats_df = (
            stats_df
            .with_columns(pl.col("woe").fill_null(-999.0).alias("_woe_sort_key"))
            .sort(["feature", "_woe_sort_key", "bin_index"])
            .with_columns([
                pl.col("bad_dist").cum_sum().over("feature").alias("cum_bad_dist"),
                pl.col("good_dist").cum_sum().over("feature").alias("cum_good_dist")
            ])
            .with_columns([
                (pl.col("cum_bad_dist") - pl.col("cum_good_dist")).abs().alias("bin_ks"),
                (
                    (pl.col("cum_good_dist") - pl.col("cum_good_dist").shift(1, fill_value=0).over("feature"))
                    *
                    (pl.col("cum_bad_dist") + pl.col("cum_bad_dist").shift(1, fill_value=0).over("feature"))
                    / 2
                ).alias("bin_auc_contrib")
            ])
            .with_columns([
                pl.col("bin_iv").sum().over("feature").alias("IV"),
                pl.col("bin_ks").max().over("feature").alias("KS"),
                pl.col("bin_auc_contrib").sum().over("feature").alias("AUC"),
                (pl.col("bad_rate") / (global_bad_rate + 1e-6)).alias("Lift")
            ])
            .with_columns([
                pl.when(pl.col("AUC") < 0.5).then(1 - pl.col("AUC")).otherwise(pl.col("AUC")).alias("AUC")
            ])
            .drop(["bin_auc_contrib", "_woe_sort_key"])
        )

        if update_woe:
            woe_data = stats_df.select(["feature", "bin_index", "woe"]).to_dict(as_series=False)
            from collections import defaultdict
            temp_woe_map = defaultdict(dict)

            for f, b, w in zip(
                woe_data["feature"],
                woe_data["bin_index"],
                woe_data["woe"],
                strict=False,
            ):
                if b is not None and not (isinstance(b, float) and np.isnan(b)):
                    temp_woe_map[f][int(b)] = w
            self.bin_woes_.update(temp_woe_map)

        mapping_rows = []
        for col, map_dict in self.bin_mappings_.items():
            for idx, label in map_dict.items():
                mapping_rows.append({"feature": col, "bin_index": idx, "bin_label": label})

        if not mapping_rows:
            return stats_df

        mapping_df = pl.DataFrame(mapping_rows, schema={
            "feature": pl.Utf8,
            "bin_index": pl.Int16,
            "bin_label": pl.Utf8
        })

        final_df = (
            stats_df
            .join(mapping_df, on=["feature", "bin_index"], how="left")
            .with_columns((pl.col("bin_index") < 0).alias("_is_special"))
            .sort(["feature", "_is_special", "bin_index"])
            .drop("_is_special")
            .select([
                pl.col("feature"),
                pl.col("bin_label").fill_null(pl.col("bin_index").cast(pl.Utf8)),
                pl.all().exclude(["feature", "bin_index", "bin_label"])
            ])
        )

        trend_df = self._build_trend_shape_frame(
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

        base_cols = ["feature", "bin_label", "trend_shape"]
        other_cols = [c for c in final_df.columns if c not in base_cols]

        out_df = final_df.select(base_cols + other_cols)
        return self._format_output(out_df)

    def update_bins(
        self,
        bin_rules: Dict[str, Union[List[Union[int, float]], List[List[Any]]]],
        X: Union[pl.DataFrame, pd.DataFrame] | None = None,
        y: Any | None = None,
    ) -> pl.DataFrame | None:
        """
        批量更新分箱规则并即时重算相关统计量。

        允许用户批量传入需要强行修改切点的特征字典，系统将自动更新内部规则，
        并在单次扫描中重新计算所有被修改特征的 WOE 和分箱统计量。

        Parameters
        ----------
        bin_rules : Dict[str, Union[List[Union[int, float]], List[List[Any]]]]
            待修改的特征分箱规则字典。
            - 数值型特征：传入内部切点列表，如 {'age': [25, 30, 45]} (系统会自动补齐 -inf 和 inf)。
            - 类别型特征：传入二维分组列表，如 {'city': [['北京', '上海'], ['广州', '深圳'], ['其他']]}。
        X : Union[pl.DataFrame, pd.DataFrame] | None
            用于重新计算 WOE 的数据。若为 None，将尝试使用 fit 时缓存的 _cache_X。
        y : Any | None
            目标标签。若为 None，将尝试使用 fit 时缓存的 _cache_y。

        Returns
        -------
        pl.DataFrame or pd.DataFrame or None
            返回被修改特征的最新分箱统计分布表；若没有任何有效特征被更新，则返回 ``None``。

        Raises
        ------
        ValueError
            当缺少用于重算 WOE 的 ``X``/``y``，且缓存也不可用时抛出。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> y = pl.Series("target", [0, 0, 1, 1])
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, y, features=["age"])
        >>> updated = binner.update_bins({"age": [35]})
        >>> "feature" in updated.columns
        True
        """
        self._check_is_fitted()

        # 提取计算上下文
        calc_X = self._ensure_polars_dataframe(X) if X is not None else self._cache_X
        calc_y = self._ensure_polars_series(y) if y is not None else self._cache_y

        if calc_X is None or calc_y is None:
            raise ValueError(
                "Missing data for WOE recalculation. "
                "Either provide X and y explicitly, or ensure the binner cache is not cleared."
            )

        updated_features = []

        # 遍历更新物理切点状态
        for feature, splits in bin_rules.items():
            if feature not in self.feature_names_in_:
                logger.warning(f"Feature '{feature}' is not recognized by this binner. Skipped.")
                continue

            # 智能推断类型：如果列表里的元素还是列表，说明是类别型分组；否则是数值型切点
            is_categorical = len(splits) > 0 and isinstance(splits[0], list)

            if not is_categorical:
                # 数值型特征：补齐边界并去重排序
                clean_splits = sorted(list(set(splits)))
                new_cuts = [float('-inf')] + clean_splits + [float('inf')]
                self.bin_cuts_[feature] = new_cuts
            else:
                # 类别型特征
                if not hasattr(self, "cat_cuts_"):
                    self.cat_cuts_ = {}
                self.cat_cuts_[feature] = splits

            # 清理旧的映射与 WOE 缓存
            if feature in self.bin_mappings_:
                del self.bin_mappings_[feature]
            if feature in self.bin_woes_:
                del self.bin_woes_[feature]

            updated_features.append(feature)

        if not updated_features:
            logger.warning("No valid features were updated.")
            return None

        # 执行即时重算 (Batch 模式)
        logger.info(f"Recalculating WOE and statistics for {len(updated_features)} modified features.")

        # 仅截取被更新的特征列送入 profile 引擎，实现单次极速扫描
        stats_df = self.profile_bin_performance(
            X=calc_X.select(updated_features),
            y=calc_y,
            update_woe=True
        )

        return stats_df

    def prune(self, keep_features: List[str]) -> "MarsBinnerBase":
        """
        裁剪分箱器内部状态，仅保留指定特征。

        Parameters
        ----------
        keep_features : List[str]
            需要保留状态的特征列表。

        Returns
        -------
        MarsBinnerBase
            裁剪完成后的当前实例。

        Notes
        -----
        该方法会同步裁剪切点、类别分组、标签映射、WOE 映射以及 ``feature_names_in_``，
        常用于特征筛选完成后缩小序列化模型体积。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50], "income": [5, 6, 7, 8]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, features=["age", "income"])
        >>> binner.prune(["age"]).feature_names_in_
        ['age']
        """
        keep_set = set(keep_features)

        # 过滤字典
        self.bin_cuts_ = {k: v for k, v in self.bin_cuts_.items() if k in keep_set}
        if hasattr(self, "cat_cuts_"):
            self.cat_cuts_ = {k: v for k, v in self.cat_cuts_.items() if k in keep_set}

        self.bin_mappings_ = {k: v for k, v in self.bin_mappings_.items() if k in keep_set}
        self.bin_woes_ = {k: v for k, v in self.bin_woes_.items() if k in keep_set}

        # 更新输入特征名单
        self.feature_names_in_ = [f for f in self.feature_names_in_ if f in keep_set]

        logger.info(f"Pruned binner down to {len(self.feature_names_in_)} features.")
        return self

    def generate_sql(
        self,
        features: Union[str, List[str]] | None = None,
        table_prefix: str = "t",
        return_type: Literal["woe", "index", "label"] = "woe",
        map_missing: bool = True,
        map_special: bool = True
    ) -> str:
        """
        将分箱规则导出为 SQL ``CASE WHEN`` 片段。

        Parameters
        ----------
        features : Union[str, List[str]] | None
            特征名称或特征列表。若为 None，则自动导出所有已拟合的特征。
        table_prefix : str
            表别名前缀。例如 "t" 会生成 "t.age"。若为空则直接使用特征名。
        return_type : Literal['woe', 'index', 'label']
            生成 SQL 的目标值类型：
            - 'woe': 输出 WOE 浮点数 (适合 LR 逻辑回归模型部署)
            - 'index': 输出分箱序号 (适合 XGBoost/LightGBM 树模型部署)
            - 'label': 输出分箱的中文/字符标签 (适合 BI 看板、数据分析或规则引擎)
        map_missing : bool
            是否将缺失值映射为对应的 WOE/Index/Label。
        map_special : bool
            是否将特殊值映射为对应的 WOE/Index/Label。

        Returns
        -------
        str
            标准 SQL 脚本（多字段间已用逗号安全分隔，可直接嵌入 SELECT 子句）。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, features=["age"])
        >>> sql = binner.generate_sql(features="age", return_type="index")
        >>> "age_index" in sql
        True
        """
        self._check_is_fitted()

        # 1. 入参类型归一化
        if features is None:
            # 默认导出所有包含在 bin_mappings_ 中的特征
            target_features = list(self.bin_mappings_.keys())
        elif isinstance(features, str):
            target_features = [features]
        else:
            target_features = features

        if not target_features:
            return ""

        # 2. 定义内部核心处理函数：生成单列的 CASE WHEN
        def _generate_single_sql(feature: str) -> str:
            """为单个特征生成 SQL CASE WHEN 片段。"""
            if feature not in self.bin_mappings_:
                raise ValueError(f"Feature '{feature}' not found or not fitted.")

            col_name = f"{table_prefix}.{feature}" if table_prefix else feature
            lines = ["CASE"]

            mappings = self.bin_mappings_.get(feature, {})
            woes = self.bin_woes_.get(feature, {})

            def _get_output_val(idx: int) -> str:
                """内部函数：根据输出契约动态格式化 THEN 后置结果"""
                if return_type == "woe":
                    return f"{woes.get(idx, 0.0):.4f}"
                elif return_type == "index":
                    return str(idx)
                else:  # 输出分箱标签。
                    label_str = mappings.get(idx, "Unknown")
                    return f"'{label_str}'"

            # 处理缺失值
            if map_missing:
                lines.append(f"  WHEN {col_name} IS NULL THEN {_get_output_val(self.IDX_MISSING)}")
            else:
                lines.append(f"  WHEN {col_name} IS NULL THEN NULL")

            # 处理特殊值 (逆序保证优先级)
            special_idx = [k for k in mappings.keys() if k <= self.IDX_SPECIAL_START]
            for idx in sorted(special_idx, reverse=True):
                label = mappings[idx]
                val_str = label.replace("Special_", "")

                try:
                    float(val_str)
                    sql_val = val_str
                except ValueError:
                    sql_val = f"'{val_str}'"

                if map_special:
                    lines.append(f"  WHEN {col_name} = {sql_val} THEN {_get_output_val(idx)}")
                else:
                    lines.append(f"  WHEN {col_name} = {sql_val} THEN {col_name}")

            # 处理数值型特征切点逻辑
            if hasattr(self, "bin_cuts_") and feature in self.bin_cuts_:
                cuts = self.bin_cuts_[feature]
                for i in range(len(cuts) - 1):
                    upper_bound = cuts[i+1]
                    if upper_bound != float('inf'):
                        lines.append(f"  WHEN {col_name} < {upper_bound} THEN {_get_output_val(i)}")
                    else:
                        lines.append(f"  ELSE {_get_output_val(i)}")

            # 处理类别型特征逻辑
            elif hasattr(self, "cat_cuts_") and feature in self.cat_cuts_:
                groups = self.cat_cuts_[feature]
                for i, group in enumerate(groups):
                    if "__Mars_Other_Pre__" in group:
                        continue
                    in_clause = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in group])
                    lines.append(f"  WHEN {col_name} IN ({in_clause}) THEN {_get_output_val(i)}")

                lines.append(f"  ELSE {_get_output_val(self.IDX_OTHER)}")

            # 兜底逻辑
            if "ELSE" not in "\n".join(lines):
                lines.append(f"  ELSE {_get_output_val(self.IDX_OTHER)}")

            lines.append(f"END AS {feature}_{return_type}")
            return "\n".join(lines)

        # 3. 遍历拼接多个特征的 SQL 代码块
        sql_blocks = [_generate_single_sql(feat) for feat in target_features]

        # 使用逗号加两个换行符拼接，使其满足 SELECT 多列的语法格式
        return ",\n\n".join(sql_blocks)

    @staticmethod
    def _format_cut_point(val: float) -> str:
        """将切点格式化为适合图表和 SQL 展示的字符串。"""
        if val == float('inf'):
            return 'inf'
        if val == float('-inf'):
            return '-inf'
        if val == 0:
            return '0'

        abs_val = abs(val)

        # 超大数字 (>=10000)，使用千分位逗号，如 1,000,000
        if abs_val >= 10000:
            if val == int(val):
                return f"{int(val):,}"
            else:
                # 保留两位小数并剔除末尾多余的 0
                return f"{val:,.2f}".rstrip('0').rstrip('.')

        # 极小数字 (<0.001)，强制使用定点数避免科学计数法, 保留 6 位小数，并动态剔除尾部无效的 0
        elif abs_val < 0.001:
            return f"{val:.6f}".rstrip('0').rstrip('.')

        # 常规数字，最多保留 4 位小数并去掉多余的 0
        else:
            if val == int(val):
                return str(int(val))
            else:
                return f"{val:.4f}".rstrip('0').rstrip('.')

class MarsNativeBinner(MarsBinnerBase):
    """
    原生高性能特征分箱器。

    基于 Polars 向量化计算与 Scikit-Learn 决策树算法构建。支持针对连续型变量的等频、
    等宽与决策树（CART）离散化策略，以及针对类别型变量的头部频数保留策略。特殊值与
    缺失值在执行核心分箱逻辑前会被强制物理隔离。

    Attributes
    ----------
    bin_cuts_ : dict of str to list of float
        针对连续型特征拟合生成的物理切点映射字典。数组形态为 `[-inf, cut_1, ..., cut_n, inf]`。

    cat_cuts_ : dict of str to list of str
        针对类别型特征拟合生成的高频类别保留映射字典。

    fit_failures_ : dict of str to str
        记录在拟合过程中触发严重计算异常的特征名称及其内部堆栈报错信息。

    feature_names_in_ : list of str
        实际参与拟合管道的全局特征名称列表。

    Notes
    -----
    当数据集体量极大且连续变量呈严重长尾或零膨胀分布时，无监督的等频与等宽算法极易生成
    绝对占比极低甚至频数为 0 的异常物理箱。开启 `merge_small_bins` 选项可在不显著增加
    计算开销的前提下，强制修复连续区间的碎化问题，保障后续 WOE 映射的稳定性。

    Examples
    --------
    >>> import polars as pl
    >>> binner = MarsNativeBinner(method="quantile", n_bins=2)
    >>> df = pl.DataFrame({"age": [20, 30, 40, 50]})
    >>> binner.fit_transform(df).columns
    ['age_bin']
    """

    def __init__(
        self,
        *,
        method: Literal["cart", "quantile", "uniform"] = "quantile",
        n_bins: int = 10,
        special_values: List[Union[int, float, str]] | None = None,
        missing_values: List[Union[int, float, str]] | None = None,
        min_bin_size: float = 0.02,
        merge_small_bins: bool = False,
        cart_params: Dict[str, Any] | None = None,
        remove_empty_bins: bool = False,
        n_jobs: int = -1,
    ) -> None:
        """
        初始化原生分箱器。

        Parameters
        ----------
        method : Literal['cart', 'quantile', 'uniform']
            数值特征的分箱策略。
        n_bins : int
            最大分箱数量，不含缺失值箱和特殊值箱。
        special_values : List[Union[int, float, str]] | None
            需要独立隔离的特殊值集合。
        missing_values : List[Union[int, float, str]] | None
            需要额外识别为缺失的值集合。
        min_bin_size : float
            单箱最小样本占比约束。
        merge_small_bins : bool
            是否在无监督分箱后自动合并小样本箱。
        cart_params : Dict[str, Any] | None
            透传给 ``DecisionTreeClassifier`` 的参数。
        remove_empty_bins : bool
            是否在 ``uniform`` 分箱时清理空箱。
        n_jobs : int
            并行计算使用的核心数限制。
        """
        super().__init__(
            n_bins=n_bins,
            special_values=special_values,
            missing_values=missing_values,
            n_jobs=n_jobs
       )
        self.method = method
        self.min_bin_size = min_bin_size
        self.merge_small_bins = merge_small_bins # 挂载到实例
        self.remove_empty_bins = remove_empty_bins

        self.cart_params = cart_params if cart_params is not None else {}

    def fit(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: pl.Series | pd.Series | np.ndarray | list[Any] | None = None,
        *,
        features: List[str] | None = None,
        cat_features: List[str] | None = None,
    ) -> "MarsNativeBinner":
        """
        拟合原生分箱器。

        Parameters
        ----------
        X : pl.DataFrame | pd.DataFrame
            输入特征矩阵。
        y : pl.Series | pd.Series | np.ndarray | list[Any] | None
            目标变量。仅当 ``method="cart"`` 时必填。
        features : List[str] | None
            本次拟合的特征列；不传时使用全部候选列。
        cat_features : List[str] | None
            明确指定的类别特征列。

        Returns
        -------
        MarsNativeBinner
            拟合完成后的原生分箱器实例。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2)
        >>> binner.fit(X).feature_names_in_
        ['age']
        """
        if self.method == "cart" and y is None:
            raise ValueError("Decision Tree Binning ('cart') requires y.")

        self.features = list(features or [])
        self.cat_features = list(cat_features or [])
        super().fit(X, y)
        return self

    def _fit_impl(self, X: pl.DataFrame, y: Any | None = None) -> None:
        """
        [Core Dispatcher] 原生分箱核心拟合与路由引擎。

        该方法充当整个分箱流程的“交通指挥枢纽”。它通过一次完整的 Schema 扫描，
        将特征划分为三大阵营（全空、数值、类别），并自动过滤掉零方差特征，
        最后将有效特征分发给对应的底层算法进行拟合。

        Parameters
        ----------
        X : pl.DataFrame
            训练数据集 (特征矩阵)。
        y : Any | None
            目标变量 (Label)。
            - 无监督分箱 (quantile, uniform, categorical) 时可为 None。
            - 有监督分箱 (cart) 时必须提供。

        Process Flow
        ------------
        1. **特征探查与分流**:
           - 全空列 (Null): 直接赋予空切点，安全熔断。
           - 类别列 (Categorical/String/Bool): 放入 `cat_cols` 队列。
           - 数值列 (Numeric): 放入 `num_cols` 队列。
        2. **零方差前置拦截 (Numeric)**:
           - 向量化极速提取所有数值列的 `min` 和 `max`。
           - 拦截 `min == max` (单一值) 的特征，直接赋予 `[-inf, inf]` 兜底。
        3. **算法分发**:
           - 数值列分发至 `_fit_quantile`, `_fit_uniform`, 或 `_fit_cart_parallel`。
           - 类别列分发至 `_fit_categorical_native`。
        4. **异常容错**:
           - 捕获无法分箱的特征并存入 `self.fit_failures_`，不阻塞全局流程。
        """
        self._cache_X = X
        self._cache_y = y
        self.fit_failures_: Dict[str, str] = {}

        y_name = getattr(y, "name", None)
        all_target_cols = self.features if self.features else [c for c in X.columns if c != y_name]
        cat_set = set(self.cat_features) if self.cat_features else set()

        num_cols = []
        cat_cols = []
        null_cols = []

        for c in all_target_cols:
            if c not in X.columns:
                continue

            # 判定全空
            if X[c].dtype == pl.Null or X[c].null_count() == X.height:
                null_cols.append(c)
                continue

            # 分流：类别 vs 数值
            if c in cat_set or X[c].dtype in [pl.Utf8, pl.Categorical, pl.Boolean]:
                cat_cols.append(c)
            elif self._is_numeric(X[c]):
                num_cols.append(c)

        for c in null_cols:
            self.bin_cuts_[c] = []

        if not num_cols and not cat_cols:
            logger.warning("No valid columns found for binning.")
            return

        # ---------------- 1. 处理数值型特征 ----------------
        if num_cols:
            float_cols = [c for c in num_cols if X.schema[c] in [pl.Float32, pl.Float64]]
            int_cols = [c for c in num_cols if X.schema[c] not in [pl.Float32, pl.Float64]]

            stats_exprs = []

            #利用 fill_nan(None) 将 NaN 转为自动被忽略的 Null
            if float_cols:
                stats_exprs.extend([
                    pl.col(float_cols).fill_nan(None).min().name.suffix("_min"),
                    pl.col(float_cols).fill_nan(None).max().name.suffix("_max")
                ])

            if int_cols:
                stats_exprs.extend([
                    pl.col(int_cols).min().name.suffix("_min"),
                    pl.col(int_cols).max().name.suffix("_max")
                ])

            # 使用 named=True 会返回类似 {'age_min': 18, 'age_max': 60} 的字典
            stats_dict = X.select(stats_exprs).row(0, named=True)

            valid_num_cols = []

            for c in num_cols:
                min_val = stats_dict[f"{c}_min"]
                max_val = stats_dict[f"{c}_max"]

                # 防御全空列或零方差常量列
                if min_val is None or max_val is None or min_val == max_val:
                    self.bin_cuts_[c] = [float('-inf'), float('inf')]
                    continue

                valid_num_cols.append(c)

            if valid_num_cols:
                if y is None and self.method == "cart":
                    raise ValueError("Decision Tree Binning ('cart') requires target 'y'.")

                if self.method == "quantile":
                    self._fit_quantile(X, valid_num_cols)
                elif self.method == "uniform":
                    self._fit_uniform(X, valid_num_cols)
                elif self.method == "cart":
                    self._fit_cart_parallel(X, y, valid_num_cols)

                # CART 已经内建叶子样本量约束；仅机械切分策略需要额外合并小箱。
                if self.merge_small_bins and self.method in ["quantile", "uniform"]:
                    self._apply_min_bin_size(X, valid_num_cols)

        # ---------------- 2. 处理类别型特征 ----------------
        if cat_cols:
            self._fit_categorical_native(X, cat_cols)

        if self.fit_failures_:
            logger.warning(
                f"{len(self.fit_failures_)} features failed during fitting and fell back to default handling. "
                f"Check `.fit_failures_` for details."
            )

    def _apply_min_bin_size(self, X: pl.DataFrame, valid_num_cols: List[str]) -> None:
        """
        [Algorithm] 单趟 CDF 前向贪心合并 (One-Pass CDF Greedy Merge).

        用于消除等频/等宽分箱产生的、样本占比小于 min_bin_size 的微型碎片箱。
        """
        if not self.merge_small_bins or self.min_bin_size <= 0:
            return

        raw_exclude = self.special_values + self.missing_values
        total_rows = X.height  # [核心修复 1] 提取数据集的绝对总行数作为全局分母

        for col in valid_num_cols:
            raw_cuts = self.bin_cuts_.get(col, [])
            if len(raw_cuts) <= 2:
                continue # 只有 [-inf, inf] 兜底，无需合并

            # 取出中间切点
            inner_cuts = sorted(raw_cuts[1:-1])

            # 获取剔除特殊值/空值后的干净数据
            col_dtype = X.schema[col]
            safe_exclude = self._get_safe_values(col_dtype, raw_exclude)

            series = X.get_column(col)
            valid_mask = series.is_not_null()
            if col_dtype in [pl.Float32, pl.Float64]:
                valid_mask &= series.is_not_nan()
            if safe_exclude:
                valid_mask &= ~series.is_in(safe_exclude)

            clean_series = series.filter(valid_mask)
            clean_total = clean_series.len()

            if clean_total == 0:
                continue

            # 计算 CDF -
            # 构造表达式：每个切点包含的样本数
            exprs = [(pl.col(col) < c).sum().alias(f"cut_{i}") for i, c in enumerate(inner_cuts)]

            # 一次 Select 查出所有切点包含的绝对样本数
            cdf_row = clean_series.to_frame().select(exprs).row(0)

            # 使用全局 total_rows 计算全局占比，而不是干净数据的占比
            cdf_vals = [val / total_rows for val in cdf_row]

            # 前向贪心合并
            kept_cuts = []
            last_cdf = 0.0

            for cut_val, cdf in zip(inner_cuts, cdf_vals, strict=False):
                # 只有当当前切点与上一个保留切点的全局区间占比达标时，才保留该切点
                if cdf - last_cdf >= self.min_bin_size:
                    kept_cuts.append(cut_val)
                    last_cdf = cdf

            # 尾部反悔
            # 尾部剩余比例 = 干净数据的全局总占比 - 最后一个切点的累计占比
            clean_ratio = clean_total / total_rows
            if kept_cuts and (clean_ratio - last_cdf < self.min_bin_size):
                # 尾部不达标，直接踢掉最后一个保留切点，它会自动与倒数第二个箱子合并
                # 合并后的新尾部占比必然 >= min_bin_size
                kept_cuts.pop()

            # 重新装载合并后的切点
            self.bin_cuts_[col] = [float('-inf')] + kept_cuts + [float('inf')]

    def _fit_categorical_native(self, X: pl.DataFrame, cols: List[str]) -> None:
        """
        [Algorithm] 类别特征极速分箱 (Top-N Truncation)。

        这是针对高基数类别特征（High Cardinality Categorical Features）的极速降维算法。
        它基于频率统计，保留样本量最大的 Top-K 个类别独立成箱，其余长尾类别予以截断。

        Parameters
        ----------
        X : pl.DataFrame
            训练数据集。
        cols : List[str]
            被判定为类别型的特征列表。

        Architectural Note (架构精要)
        -----------------------------
        1. **参数复用**: 这里的 `K` 值直接复用基类的 `self.n_bins` (即最大箱数)。
        2. **隐式长尾归集**: 本方法只需将 Top-K 类别写入 `self.cat_cuts_`。
           在 `transform` 阶段，任何未命中 `cat_cuts_` 字典的长尾类别（或新出现的类别），
           都会极其优雅地触发基类的 `otherwise(default_bin)` 机制，
           自动跌入 `IDX_OTHER` (-2) 兜底箱中，无需在 fit 阶段手动将它们替换为 "Other"。
        """
        if not hasattr(self, "cat_cuts_"):
            self.cat_cuts_ = {}

        raw_exclude = self.special_values + self.missing_values

        for c in cols:
            col_dtype = X.schema[c]
            safe_exclude = self._get_safe_values(col_dtype, raw_exclude)

            series = X.get_column(c)

            # 构建过滤掩码：剔除空值与业务指定的特殊值
            valid_mask = series.is_not_null()
            if safe_exclude:
                valid_mask &= (~series.is_in(safe_exclude))

            clean_series = series.filter(valid_mask)

            # 异常熔断：全部是空值或特殊值
            if clean_series.len() == 0:
                self.fit_failures_[c] = "All values are missing or special."
                self.cat_cuts_[c] = []
                continue

            # 核心：使用 Polars 极速统计频次，取前 n_bins 个
            # 例如 n_bins=10，则保留最多 10 个独立类别
            top_k_df = clean_series.value_counts(sort=True).head(self.n_bins)
            top_vals = top_k_df.get_column(c).to_list()

            # cat_cuts_ 要求是二维列表 (List[List[Any]])，每个子列表代表一个箱
            # 这里为每个 Top 类别分配一个独立的箱
            self.cat_cuts_[c] = [[val] for val in top_vals]

    def _fit_quantile(self, X: pl.DataFrame, cols: List[str]) -> None:
        """
        执行等频分箱 (One-Shot Quantile Query)。

        该方法摒弃了传统的“循环、筛选、计算”模式, 转而利用 Polars 的延迟计算特性,
        将数千个特征的分位数计算合并为一个单一的原子查询计划 (Atomic Query Plan)。

        Parameters
        ----------
        X : pl.DataFrame
            训练数据集。
        cols : List[str]
            需要执行等频分箱的数值型特征列名列表。

        Notes
        -----
        1. 查询计划合并：
        - 传统实现: 针对 N 个特征执行 N 次 `quantile()` 调用, 触发 N 次内存扫描。
        - Mars 实现: 构建一个扁平化的表达式列表 `[col1_q1, col1_q2, ..., colN_qM]`。
          通过 `X.select(q_exprs)` 将该列表一次性喂给 Rust 引擎。引擎会优化执行路径,
          在单次 (或极少数次) 内存扫描中并行完成所有特征的切点计算。

        2. 数据质量控：
        - 源头隔离: 在计算分位数前, 利用 `pl.when().then(None)` 将 `special_values` 和
          `missing_values` 临时替换为 `Null`, 确保切点的分布仅由业务层面的“正常值”决定。
        - 自动去重: 针对高偏态数据 (如某些取值极度集中的分位数一致), 会自动执行 `set()`
          去重并重新排序, 防止生成重复切点导致的 `Cut Error`。

        3. 低基数优化：
        - 针对二值/离散整数 (如 0/1), Quantile 往往会切出 [0.0, 1.0] 这种尴尬边界。
        - 优化逻辑: 若特征唯一值数量 <= n_bins, 自动降级为"中点切分", 例如 [0, 1] 会被切在 0.5。
        """
        # 构建分位点
        if self.n_bins <= 1:
            quantiles = [0.5]
        else:
            quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1].tolist()

        # 预处理排除值
        raw_exclude = self.special_values + self.missing_values

        # 批量计算 n_unique, 用于路由低基数逻辑
        # 这一步开销很小, Polars 针对数值列的 n_unique 有极速优化
        unique_exprs = []
        for c in cols:
            col_dtype = X.schema[c]
            safe_exclude = self._get_safe_values(col_dtype, raw_exclude)

            # 非 Null
            keep_mask = pl.col(c).is_not_null()
            # 非 NaN (仅浮点)
            if col_dtype in [pl.Float32, pl.Float64]:
                keep_mask &= ~pl.col(c).is_nan()
            # 非特殊值
            if safe_exclude:
                keep_mask &= ~pl.col(c).is_in(safe_exclude)

            target_col = pl.col(c).filter(keep_mask)
            unique_exprs.append(target_col.n_unique().alias(c))

        unique_counts = X.select(unique_exprs).row(0)
        col_unique_map = dict(zip(cols, unique_counts, strict=False))

        # 分流: 哪些列走 Quantile, 哪些列走 Midpoint (中点)
        quantile_cols = []
        low_card_cols = []

        for c in cols:
            # 如果唯一值比箱数还少, 算分位数没有意义, 直接切中点
            if col_unique_map[c] <= self.n_bins:
                low_card_cols.append(c)
            else:
                quantile_cols.append(c)

        # 处理高基数列 (标准 Quantile 逻辑)
        if quantile_cols:
            q_exprs = []
            for c in quantile_cols:
                col_dtype = X.schema[c]
                safe_exclude = self._get_safe_values(col_dtype, raw_exclude)

                # 统一组装“可参与分位数计算”的过滤条件。
                valid_cond = pl.col(c).is_not_null()

                # 叠加: 非 NaN (仅浮点)
                if col_dtype in [pl.Float32, pl.Float64]:
                    valid_cond &= ~pl.col(c).is_nan()

                # 叠加: 非 Special Values
                if safe_exclude:
                    valid_cond &= ~pl.col(c).is_in(safe_exclude)

                # 应用过滤
                target_col = pl.col(c).filter(valid_cond)

                for i, q in enumerate(quantiles):
                    # 别名技巧: col:::idx, 便于后续解析
                    alias_name = f"{c}:::{i}"
                    q_exprs.append(target_col.quantile(q).alias(alias_name))

            # 计算 (One-Shot Query)
            if q_exprs:
                stats = X.select(q_exprs)
                row = stats.row(0)

                # 解析结果并去重排序
                temp_cuts: Dict[str, List[float]] = {c: [] for c in quantile_cols}

                for val, name in zip(row, stats.columns, strict=False):
                    c_name, _ = name.split(":::")
                    if val is not None and not np.isnan(val):
                        temp_cuts[c_name].append(val)

                for c in quantile_cols:
                    cuts = sorted(list(set(temp_cuts[c])))

                    if len(cuts) < 1:
                        # 极端情况：所有分位数都一样（例如全是0）
                        # 强制退化为全区间，防止后续 cut 算子切出空箱或单箱
                        self.bin_cuts_[c] = [float('-inf'), float('inf')]
                        if not hasattr(self, "fit_failures_"):
                            self.fit_failures_ = {}
                        self.fit_failures_[c] = "Degenerate feature: all quantiles are identical."
                    else:
                        self.bin_cuts_[c] = [float('-inf')] + cuts + [float('inf')]

        # 处理低基数列 (中点切分优化)
        if low_card_cols:
            for c in low_card_cols:
                safe_exclude = self._get_safe_values(X.schema[c], raw_exclude)

                # 获取唯一值并排序
                # 这里的 unique 已经是全量 unique 减去 null, 但还需要排除 safe_exclude
                unique_vals = (
                    X.select(pl.col(c).unique())
                    .to_series()
                    .sort()
                    .to_list()
               )

                # 清洗, 因为唯一值极少, 速度很快
                clean_vals = [v for v in unique_vals if v is not None and (not isinstance(v, float) or not np.isnan(v))]
                if safe_exclude:
                    clean_vals = [v for v in clean_vals if v not in safe_exclude]

                if len(clean_vals) <= 1:
                    # 只有一个值, 无法切分
                    self.bin_cuts_[c] = [float('-inf'), float('inf')]
                    if not hasattr(self, "fit_failures_"):
                        self.fit_failures_ = {}
                    self.fit_failures_[c] = "Degenerate feature: single unique value."
                else:
                    # 计算中点: (a+b)/2
                    # 例如 [0, 1] -> 切点 0.5 -> [-inf, 0.5, inf]
                    mid_points = [(clean_vals[k] + clean_vals[k+1])/2 for k in range(len(clean_vals)-1)]
                    self.bin_cuts_[c] = [float('-inf')] + mid_points + [float('inf')]

    def _fit_uniform(self, X: pl.DataFrame, cols: List[str]) -> None:
        """
        执行等宽分箱 (Uniform/Step Binning)。

        该方法利用 Polars 的向量化算子, 将所有特征的统计信息提取和切点生成分为两个物理阶段,
        在保证统计严谨性的同时, 最大程度减少对原始数据的扫描次数。

        Parameters
        ----------
        X : pl.DataFrame
            训练数据集。
        cols : List[str]
            需要执行等宽分箱的数值型特征列名列表。

        Notes
        -----
        1. 基础统计量聚合：
        - 构建一个全局查询计划, 一次性计算所有目标列的 `min` (最小值)、`max` (最大值)
          和 `n_unique` (唯一值个数)。
        - 排除逻辑: 在计算极值前, 会自动过滤用户定义的 `special_values` 和 `missing_values`,
          确保切点仅基于“正常”数值分布生成。
        - 低基数处理: 若特征唯一值个数小于目标箱数 (`n_unique <= n_bins`), 则自动退化为
          基于唯一值中点的精确切分, 防止生成重复切点。

        2. 空箱动态优化：
        - 仅在 `remove_empty_bins=True` 时触发。
        - 机制: 利用 Polars 的 `cut` 和 `value_counts` 算子, 在主进程中并行嗅探初始等宽
          切点下的样本分布。
        - 压缩逻辑: 识别样本量为 0 的区间, 并将相邻的空箱进行物理合并。这在数据分布极端
          偏态 (如长尾分布)时, 能有效防止产生毫无意义的无效分箱。
        """
        raw_exclude = self.special_values + self.missing_values

        # 基础统计量
        exprs = []
        col_safe_excludes = {}

        for c in cols:
            col_dtype = X.schema[c]
            safe_exclude = self._get_safe_values(col_dtype, raw_exclude)
            col_safe_excludes[c] = safe_exclude

            # 统一组装“可参与统计”的过滤条件。
            keep_mask = pl.lit(True)
            if col_dtype in [pl.Float32, pl.Float64]:
                keep_mask &= ~pl.col(c).is_nan()
            if safe_exclude:
                keep_mask &= ~pl.col(c).is_in(safe_exclude)

            target_col = pl.col(c).filter(keep_mask)

            exprs.append(target_col.min().alias(f"{c}_min"))
            exprs.append(target_col.max().alias(f"{c}_max"))
            exprs.append(target_col.n_unique().alias(f"{c}_n_unique"))

        stats = X.select(exprs)
        row = stats.row(0)

        initial_cuts_map = {}
        pending_optimization_cols = []

        # 解析统计量, 生成等距切点
        for i, c in enumerate(cols):
            base_idx = i * 3
            min_val, max_val, n_unique = row[base_idx], row[base_idx + 1], row[base_idx + 2]
            safe_exclude = col_safe_excludes[c]

            if min_val is None or max_val is None:
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue

            # 低基数检查 (Unique <= N_Bins), 直接取中点切分
            if n_unique <= self.n_bins:
                unique_vals = X.select(pl.col(c).unique().sort()).to_series().to_list()
                clean_vals = [v for v in unique_vals if v not in safe_exclude and v is not None]

                if len(clean_vals) <= 1:
                    self.bin_cuts_[c] = [float('-inf'), float('inf')]
                else:
                    mid_points = [(clean_vals[k] + clean_vals[k+1])/2 for k in range(len(clean_vals)-1)]
                    self.bin_cuts_[c] = [float('-inf')] + mid_points + [float('inf')]
                continue

            if min_val == max_val:
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue

            # 生成等宽切点
            raw_cuts = np.linspace(min_val, max_val, self.n_bins + 1)[1:-1].tolist()
            full_cuts = [float('-inf')] + raw_cuts + [float('inf')]
            initial_cuts_map[c] = full_cuts

            if self.remove_empty_bins:
                pending_optimization_cols.append(c)
            else:
                self.bin_cuts_[c] = full_cuts

        # 空箱优化
        if pending_optimization_cols:
            batch_exprs = []
            for c in pending_optimization_cols:
                cuts = initial_cuts_map[c]
                breaks = cuts[1:-1]
                safe_exclude = col_safe_excludes[c]

                # 空箱优化必须使用单个组合掩码；先过滤 NaN 再叠加 special_values
                # 会让第二个掩码仍保持原始行数，宽表场景下触发 Polars ShapeError。
                col_dtype = X.schema[c]
                keep_mask = pl.lit(True)
                if col_dtype in [pl.Float32, pl.Float64]:
                    keep_mask &= ~pl.col(c).is_nan()
                if safe_exclude:
                    keep_mask &= ~pl.col(c).is_in(safe_exclude)
                target_col = pl.col(c).filter(keep_mask)

                labels = [str(i) for i in range(len(breaks)+1)]

                # 批量计算直方图
                batch_exprs.append(
                    target_col.cut(breaks, labels=labels, left_closed=True)
                    .value_counts().implode().alias(f"{c}_counts")
               )

            batch_counts_df = X.select(batch_exprs)

            # 解析并剔除 Count=0 的箱
            for c in pending_optimization_cols:
                inner_series: pl.Series = batch_counts_df.get_column(f"{c}_counts")[0]
                # [动态解析] value_counts 返回的 Struct 字段名取决于原始列名 (例如: {"age": 25, "count": 10})
                # 不能硬编码 keys["count"]，必须通过 struct.fields 动态获取第 0 个 (Value) 和第 1 个 (Count) 字段名
                keys = inner_series.struct.fields
                dist_list = inner_series.to_list()

                valid_indices = set()
                for row in dist_list:
                    # row 是 {'brk': '0', 'counts': 100} 格式
                    idx_val = row.get(keys[0])
                    cnt_val = row.get(keys[1])
                    if idx_val is not None and cnt_val > 0:
                        valid_indices.add(int(idx_val))

                cuts = initial_cuts_map[c]
                breaks = cuts[1:-1]
                new_cuts = [cuts[0]]
                for i in range(len(breaks) + 1):
                    if i in valid_indices:
                        new_cuts.append(cuts[i + 1])

                if new_cuts[-1] != float('inf'):
                    new_cuts.append(float('inf'))
                self.bin_cuts_[c] = sorted(list(set(new_cuts)))

    def _fit_cart_parallel(self, X: pl.DataFrame, y: pl.Series, cols: List[str]) -> None:
        """
        执行并行的决策树分箱。

        该方法是 Mars 库的“动力心脏”, 专门针对高 PCR (计算传输比) 任务设计。
        它通过“生产-消费”流水线模式, 将 Polars 的预处理能力与 Sklearn 的拟合能力深度耦合。

        Parameters
        ----------
        X : pl.DataFrame
            特征数据集。
        y : pl.Series
            目标变量。要求已在基类中完成类型对齐 (pl.Series)。
        cols : List[str]
            需要执行决策树分箱的特征列名列表。

        Notes
        -----
        1. 计算重心前置：
        - 在 `cart_task_gen` 生成器中, 利用 Polars 的位运算内核极速完成空值和特殊值的过滤。
        - 异构对齐: 使用生成的 Numpy 掩码 (Mask) 同时对 x 和 y 进行物理切片,
          确保两端数据行索引在没有任何显式 Join 操作的情况下实现绝对对齐。

        2. 混合并行调度：
        - 后端选择: 采用 `threading` 后端配合 `n_jobs`。
        - 依据: 由于 `x_clean` 和 `y_clean` 切片已在主进程内存中完成, 使用多线程可实现
          **零拷贝** 传递给 Worker, 规避了多进程频繁序列化大数据块的物流负担。
        - 锁优化: 利用 Sklearn 底层在拟合过程中会释放 GIL 的物理特性, 实现真正的多核利用。

        3. 内存防护：
        - 异常追踪: 引入 `fit_failures_` 属性。任何由于数据极端分布或内存溢出导致的
          单特征失败将被捕获并记录原因, 而不会触发主任务的中断 (Fail-Soft 机制)。
        """
        y_np = np.ascontiguousarray(y.to_numpy())

        if len(y_np) != X.height:
            raise ValueError(f"Target 'y' length mismatch: X({X.height}) vs y({len(y_np)})")

        n_total_samples = X.height
        def worker(col_name: str, x_clean_np: np.ndarray, y_clean_np: np.ndarray) -> Tuple[str, List[float]]:
            """对单个数值特征执行 CART 分箱拟合。"""
            try:
                # 如果 min_bin_size 是浮点数 (如 0.05), 则基于 总行数(n_total_samples) 计算
                # 而不是基于 过滤后的行数(len(x_clean_np)) 计算
                if isinstance(self.min_bin_size, float):
                    min_bin_size_abs = int(np.ceil(self.min_bin_size * n_total_samples))
                else:
                    min_bin_size_abs = self.min_bin_size

                # 安全检查: 如果清洗后的数据量甚至不足以支撑 2 个最小叶子节点
                # 说明该特征在有效值范围内过于稀疏, 不应强行分箱
                if len(x_clean_np) < 2 * min_bin_size_abs:
                     return col_name, [float('-inf'), float('inf')], "Insufficient clean samples to satisfy global min_bin_size."

                cart = DecisionTreeClassifier(
                    max_leaf_nodes=self.n_bins,
                    min_samples_leaf=min_bin_size_abs,
                    **self.cart_params
               )
                cart.fit(x_clean_np, y_clean_np)
                cuts = cart.tree_.threshold[cart.tree_.threshold != -2]
                cuts = np.sort(np.unique(cuts)).tolist()
                return col_name, [float('-inf')] + cuts + [float('inf')], None # 成功
            except Exception as e:
                error_info = f"{type(e).__name__}: {str(e)}"
                return col_name, [float('-inf'), float('inf')], error_info

        raw_exclude = self.special_values + self.missing_values

        def cart_task_gen() -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
            """逐列生成 CART 分箱任务。"""
            for c in cols:
                col_dtype = X.schema[c]
                safe_exclude = self._get_safe_values(col_dtype, raw_exclude)

                series = X.get_column(c)

                # 先在列级别构建有效样本掩码，减少后续 numpy 清洗成本。
                valid_mask = series.is_not_null()
                if col_dtype in self.NUMERIC_DTYPES:
                    valid_mask &= (~series.is_nan())
                if safe_exclude:
                    valid_mask &= (~series.is_in(safe_exclude))

                if not valid_mask.any():
                    continue

                # x 端尽量采用零拷贝转换，减轻并行任务间的数据搬运开销。
                x_clean: np.ndarray = (
                    series
                    .filter(valid_mask)
                    .cast(pl.Float32)
                    .to_numpy(writable=False)
                    .reshape(-1, 1)
                )
                if not x_clean.flags['C_CONTIGUOUS']:
                    x_clean = np.ascontiguousarray(x_clean)

                # y 端利用 Numpy 的视图切片
                y_clean = y_np[valid_mask.to_numpy()]

                yield c, x_clean, y_clean

        # Backend 选型:
        # 如果数据量极大, threading 会受限于 GIL。
        # 但因为 Sklearn 的树拟合大部分是在 C++ 层释放了 GIL 的,
        # 且任务分发开销 (PCR) 在第一阶段很低, 所以 threading 是合理的。
        results = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=0)(
            delayed(worker)(name, x, y) for name, x, y in cart_task_gen()
       )

        for col_name, cuts, error_msg in results:
            self.bin_cuts_[col_name] = cuts
            if error_msg:
                self.fit_failures_[col_name] = error_msg

        # fit 结束后统一警告
        if self.fit_failures_:
            logger.warning(
                f"{len(self.fit_failures_)} features failed during CART binning and fell back to a single bin. "
                f"Check `self.fit_failures_` for details. Sample fails: {list(self.fit_failures_.items())[:3]}"
           )

class MarsOptimalBinner(MarsBinnerBase):
    """
    基于数学规划的最优分箱器。

    该组件集成了空间降维预分箱技术与 OptBinning 核心规划算法。通过在指定的目标事件率
    （Event Rate）单调性约束、最小区间占比及最小事件数等边界条件下，求解信息值（IV）最大化
    的混合整数规划或约束编程问题，生成具备极高鲁棒性与严格业务逻辑解释性的特征切点。

    Attributes
    ----------
    bin_cuts_ : dict of str to list of float
        针对连续型特征求解生成的物理切点映射字典。

    cat_cuts_ : dict of str to list of list
        针对类别型特征求解生成的离散组合分类映射字典。

    fit_failures_ : dict of str to str
        记录在拟合过程中触发求解器异常、数据类型不支持或超时熔断的特征名称及其内部诊断原因。

    Notes
    -----
    该离散化评估器在执行过程中高度依赖底层预分箱产生的搜索边界。在面临极度偏态的特征分布时，
    求解器可能因无法在给定的 `min_bin_size` 与单调性约束下找到可行解 (Infeasible) 而崩溃。
    为此，引擎内部构建了完备的异常隔离与降级回退机制，确保流水线在处理高噪超宽表时具备
    绝对的稳定性。

    Examples
    --------
    >>> import polars as pl
    >>> binner = MarsOptimalBinner(n_bins=2, min_bin_n_event=1)
    >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
    >>> y = pl.Series("y", [0, 0, 1, 1])
    >>> binner.fit(X, y).transform(X).columns
    ['age_bin']
    """

    def __init__(
        self,
        *,
        n_bins: int = 10,
        min_n_bins: int = 1,
        min_bin_size: float = 0.02,
        min_bin_n_event: int = 3,
        prebinning_method: Literal["quantile", "uniform", "cart"] = "cart",
        n_prebins: int = 50,
        min_prebin_size: float = 0.01,
        monotonic_trend: Literal["ascending", "descending", "auto", "auto_asc_desc"] = "auto_asc_desc",
        solver: Literal["cp", "mip"] = "cp",
        time_limit: int = 10,
        max_cats_to_solver: int | None = 100,
        min_cat_fraction: float = 0.05,
        special_values: List[Any] | None = None,
        missing_values: List[Any] | None = None,
        cart_params: Dict[str, Any] | None = None,
        join_threshold: int = 100,
        n_jobs: int = -1
    ) -> None:
        """
        初始化最优分箱器。

        Parameters
        ----------
        n_bins : int
            最大分箱数量，不含缺失值箱和特殊值箱。
        min_n_bins : int
            允许求解器返回的最小分箱数。
        min_bin_size : float
            单箱最小样本占比约束。
        min_bin_n_event : int
            单箱最少事件数约束。
        prebinning_method : Literal['quantile', 'uniform', 'cart']
            求解前的预分箱策略。
        n_prebins : int
            预分箱数量上限。
        min_prebin_size : float
            预分箱阶段的最小样本占比。
        monotonic_trend : Literal['ascending', 'descending', 'auto', 'auto_asc_desc']
            目标事件率的单调性约束方向。
        solver : Literal['cp', 'mip']
            数学规划求解器类型。
        time_limit : int
            单个特征的求解时间上限，单位为秒。
        max_cats_to_solver : int | None
            进入求解器搜索空间的最大类别数。
        min_cat_fraction : float
            类别特征单一类别的最小样本占比。
        special_values : List[Any] | None
            需要独立隔离的特殊值集合。
        missing_values : List[Any] | None
            需要额外识别为缺失的值集合。
        cart_params : Dict[str, Any] | None
            透传给预分箱决策树的参数。
        join_threshold : int
            高基数类别映射时切换到 Join 模式的阈值。
        n_jobs : int
            并行计算使用的核心数限制。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        """
        super().__init__(
            n_bins=n_bins,
            special_values=special_values, missing_values=missing_values,
            join_threshold=join_threshold, n_jobs=n_jobs
       )
        self.min_n_bins = min_n_bins
        self.min_bin_size = min_bin_size
        self.min_bin_n_event = min_bin_n_event
        self.n_prebins = n_prebins
        self.prebinning_method = prebinning_method

        if self.prebinning_method not in ["cart", "quantile", "uniform"]:
            raise ValueError("prebinning_method must be one of {'cart', 'quantile', 'uniform'}")

        self.min_prebin_size = min_prebin_size
        self.monotonic_trend = monotonic_trend
        self.solver = solver
        self.time_limit = time_limit
        self.max_cats_to_solver = max_cats_to_solver
        self.min_cat_fraction = min_cat_fraction
        self.cart_params = cart_params if cart_params is not None else {}

        self.OptimalBinning = OptimalBinning

    def fit(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: pl.Series | pd.Series | np.ndarray | list[Any],
        *,
        features: List[str] | None = None,
        cat_features: List[str] | None = None,
    ) -> "MarsOptimalBinner":
        """
        拟合最优分箱器。

        Parameters
        ----------
        X : pl.DataFrame | pd.DataFrame
            输入特征矩阵。
        y : pl.Series | pd.Series | np.ndarray | list[Any]
            目标变量。最优分箱依赖监督信息，必须提供。
        features : List[str] | None
            本次拟合的特征列；不传时使用全部候选列。
        cat_features : List[str] | None
            明确指定的类别特征列。

        Returns
        -------
        MarsOptimalBinner
            拟合完成后的最优分箱器实例。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> y = pl.Series("target", [0, 0, 1, 1])
        >>> binner = MarsOptimalBinner(n_bins=2, min_bin_n_event=1)
        >>> binner.fit(X, y).feature_names_in_
        ['age']
        """
        if y is None:
            raise ValueError("MarsOptimalBinner.fit requires y.")

        self.features = list(features or [])
        self.cat_features = list(cat_features or [])
        super().fit(X, y)
        return self

    def _fit_impl(self, X: pl.DataFrame, y: pl.Series = None) -> None:
        """
        自动执行特征识别与任务流分发。

        Parameters
        ----------
        X : pl.DataFrame
            训练集特征数据。
        y : pl.Series
            目标变量。要求必须可转换为二分类的 int32 数组。
        """
        # 缓存数据引用, 仅用于 transform 阶段请求 return_type='woe' 时的延迟计算
        self._cache_X = X
        self._cache_y = y

        if y is None:
            raise ValueError("Optimal Binning requires target 'y' to calculate IV/WOE.")

        y_np = np.ascontiguousarray(y.to_numpy()).astype(np.int32)

        # 获取 y 的名称 (如果 y 是 Series)
        y_name = getattr(y, "name", None)

        # 确定目标列: 如果没有指定 features, 则获取 X 的所有列, 但必须排除掉 y 所在的列
        if not self.features:
            all_target_cols = [c for c in X.columns if c != y_name]
        else:
            all_target_cols = self.features
        cat_set = set(self.cat_features)

        num_cols = []
        cat_cols = []
        null_cols = []

        for c in all_target_cols:
            if c not in X.columns:
                continue

            # 优先判定类别
            if c in cat_set or X[c].dtype in [pl.Utf8, pl.Categorical, pl.Boolean] :
                cat_cols.append(c)
                continue

            # 判定全空
            if X[c].dtype == pl.Null or X[c].null_count() == X.height:
                null_cols.append(c)
                continue

            # 判定数值
            if self._is_numeric(X[c]):
                num_cols.append(c)

        if not num_cols and not cat_cols and not null_cols:
            logger.warning("No valid numeric or categorical columns found.")
            return

        self.fit_failures_: Dict[str, str] = {}

        for c in null_cols:
            self.bin_cuts_[c] = []

        if num_cols:
            self._fit_numerical_impl(X, y_np, num_cols)

        if cat_cols:
            self._fit_categorical_impl(X, y_np, cat_cols)

        if self.fit_failures_:
            num_fails = len([k for k in self.fit_failures_ if k in num_cols])
            cat_fails = len([k for k in self.fit_failures_ if k in cat_cols])
            logger.warning(
                f"MarsOptimalBinner: {len(self.fit_failures_)} features encountered issues "
                f"({num_fails} num, {cat_fails} cat). Fallback applied. "
                f"Check `.fit_failures_` for details. Sample: {list(self.fit_failures_.items())[:2]}"
           )

    def _fit_numerical_impl(self, X: pl.DataFrame, y_np: np.ndarray, num_cols: List[str]) -> None:
        """
        拟合数值特征的最优分箱规则。

        Parameters
        ----------
        X : pl.DataFrame
            特征数据。
        y_np : np.ndarray
            已经过内存对齐和类型转换的标签数组。
        num_cols : List[str]
            待处理的数值列名。

        Notes
        -----
        - 1. 计算重心前置: 在 `num_task_gen` 内部利用 Polars进行极速过滤, Worker 仅接收经过净化的 Numpy 视图。

        - 2. 两阶段联动: 先调用 `MarsNativeBinner` 获取粗粒度切点,
          随后将其作为 `user_splits` 注入 `optbinning`, 极大缩小了数学规划的搜索空间。

        - 3. 并发控制: 使用 `loky` 后端。由于单个特征的最优求解耗时较长 PCR,
          支付跨进程通讯成本以换取独立 CPU 核心的满载运行是非常合算的。
        """
        # 包装为具名 Series，便于下游复用统一的目标列读取逻辑。
        y_series = pl.Series(name="target", values=y_np)

        pre_binner = MarsNativeBinner(
            method=self.prebinning_method,
            n_bins=self.n_prebins,
            special_values=self.special_values,
            missing_values=self.missing_values,
            min_bin_size=self.min_prebin_size,
            cart_params=self.cart_params,
            n_jobs=self.n_jobs,
            remove_empty_bins=False
       )
        pre_binner.fit(X, y_series, features=num_cols)
        pre_cuts_map = pre_binner.bin_cuts_

        # 筛选需要优化的列
        active_cols: list[str] = []
        for col, cuts in pre_cuts_map.items():
            if len(cuts) > 2:
                active_cols.append(col)
            else:
                self.bin_cuts_[col] = cuts

        if not active_cols:
            return

        # 获取全局样本总数
        n_total_samples = X.height

        def num_worker(
            col: str,
            pre_cuts: List[float],
            col_data: np.ndarray,
            y_data: np.ndarray
        ) -> Tuple[str, List[float], str | None]:
            """对单个数值特征执行最优分箱求解。"""
            fallback_res = (col, pre_cuts, None)
            try:
                # 计算基于"总体"的绝对 min_bin_size
                if isinstance(self.min_bin_size, float):
                    min_bin_size_abs = int(np.ceil(self.min_bin_size * n_total_samples))
                else:
                    min_bin_size_abs = self.min_bin_size # 如果用户初始化时就传了整数

                # 绝对值检查
                # 如果当前数据量 < 最小分箱数 * 最小单箱大小, 直接回退
                if len(col_data) < self.min_n_bins * min_bin_size_abs:
                     return fallback_res

                if len(col_data) < 10 or np.var(col_data) < 1e-6:
                    return col, pre_cuts, "Low variance or insufficient samples"

                # 将绝对值转换回当前数据的相对比例
                # OptBinning 源码限制 min_bin_size 必须在 (0, 0.5] 之间
                    # 如果占比 > 0.5，意味着无法分出 2 个箱子，求解器会无解报错。
                    # 强制截断为 0.5 是为了让求解器至少能尝试分出 2 个箱 (或证明不可分)。
                current_ratio = min_bin_size_abs / len(col_data)

                # 如果比例超过 0.5, 说明当前数据甚至无法切分出两个满足要求的箱子
                # (例如: 要求每箱至少500人, 但当前只有800人, 500/800 = 0.625 > 0.5)
                if current_ratio > 0.5:
                    return fallback_res

                # 为了防止浮点精度问题导致正好等于 0.50000001 报错, 做个截断保护
                current_ratio = min(current_ratio, 0.5)

                raw_splits = np.array(pre_cuts[1:-1])
                if len(raw_splits) > 1:
                    diffs = np.diff(raw_splits)
                    # 剔除过于接近的切点, 防止求解器报错
                    mask = np.concatenate(([True], diffs > 1e-6))
                    user_splits = raw_splits[mask]
                else:
                    user_splits = raw_splits

                if len(user_splits) == 0:
                    return fallback_res

                opt = self.OptimalBinning(
                    name=col,
                    dtype="numerical",
                    solver=self.solver,
                    monotonic_trend=self.monotonic_trend,
                    user_splits=user_splits,
                    min_n_bins=self.min_n_bins,
                    max_n_bins=self.n_bins,
                    time_limit=self.time_limit,
                    min_bin_size=current_ratio,
                    min_bin_n_event=self.min_bin_n_event,
                    verbose=False
               )
                opt.fit(col_data, y_data)

                if opt.status in ["OPTIMAL", "FEASIBLE"]:
                    res_cuts = [float('-inf')] + list(opt.splits) + [float('inf')]
                    return col, res_cuts, None

                # 捕获求解器非最优状态 (如 TIMEOUT)
                return col, pre_cuts, f"Solver status: {opt.status}"

            except Exception as e:
                # 捕获代码级异常
                return col, pre_cuts, f"{type(e).__name__}: {str(e)}"

        # 预处理排除值
        raw_exclude = self.special_values + self.missing_values
        def num_task_gen() -> Iterator[tuple[str, list[float], np.ndarray, np.ndarray]]:
            """通过 yield 纯净的 NumPy 数组, 触发 joblib 的 mmap 共享内存优化。"""
            for c in active_cols:
                # 类型感知与安全过滤列表获取
                col_dtype = X.schema[c]
                safe_exclude = self._get_safe_values(col_dtype, raw_exclude)

                # 获取 Series 指针, 不使用 select, 避免 DataFrame 物化开销
                series = X.get_column(c)

                # 构建 Polars 过滤掩码
                # 基础过滤: 非 null
                valid_mask = series.is_not_null()

                # 针对数值特征增加: 非 NaN 过滤
                if col_dtype in [pl.Float32, pl.Float64]:  # 仅对浮点数检查 NaN
                    valid_mask &= (~series.is_nan())

                # 针对业务特殊值进行排除, 如 -999, -998
                if safe_exclude:
                    valid_mask &= (~series.is_in(safe_exclude))

                # 将位掩码转换为 NumPy 布尔数组, 用于 y 的快速切片
                mask_np = valid_mask.to_numpy()

                # 如果过滤后样本量不足, 直接跳过此列, 减少并行开销
                if not mask_np.any():
                    continue

                # 特征列 X 处理
                col_np: np.ndarray = (
                    series.filter(valid_mask)
                    .cast(pl.Float32)
                    .to_numpy(writable=False)
                )

                # `loky` 传输连续数组更稳定，也能减少额外的内存重排。
                if not col_np.flags['C_CONTIGUOUS']:
                    col_np = np.ascontiguousarray(col_np)

                clean_y = y_np[mask_np]
                yield c, pre_cuts_map[c], col_np, clean_y

        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(num_worker)(c, cuts, data, y) for c, cuts, data, y in num_task_gen()
       )

        for col, cuts, error_msg in results:
            self.bin_cuts_[col] = cuts
            if error_msg:
                self.fit_failures_[col] = error_msg

    def _fit_categorical_impl(self, X: pl.DataFrame, y_np: np.ndarray, cat_cols: List[str]) -> None:
        """
        拟合类别特征的最优分箱规则。

        Parameters
        ----------
        X : pl.DataFrame
            特征数据。
        y_np : np.ndarray
            标签数组。
        cat_cols : List[str]
            待处理的类别列名。

        Notes
        -----
        - 1. 长尾截断路由 (__Mars_Other_Pre__): 针对频数极低或基数极大的类别, 自动执行
          `Top-K` 截断, 并将长尾数据归并为特殊的 `__Mars_Other_Pre__` 类别。

        - 2. 数据源头净化: 在任务生成器中完成字符串映射和空值隔离,
          Worker 进程拿到的直接是满足 `optbinning` 输入要求的 `pl.Utf8` 映射数据。
        """
        raw_exclude = self.special_values + self.missing_values

        def cat_worker(
            col: str,
            clean_data: np.ndarray,
            clean_y: np.ndarray
        ) -> Tuple[str, List[List[Any]] | None, str | None]:
            """对单个类别特征执行最优分箱求解。"""
            try:
                opt = self.OptimalBinning(
                    name=col, dtype="categorical",
                    solver=self.solver,
                    max_n_bins=self.n_bins,
                    time_limit=self.time_limit,
                    cat_cutoff=self.min_cat_fraction,
                    verbose=False
                )
                opt.fit(clean_data, clean_y)

                if opt.status in ["OPTIMAL", "FEASIBLE"]:
                    return col, opt.splits, None
                return col, None, f"Solver status: {opt.status}"
            except Exception as e:
                return col, None, f"{type(e).__name__}: {str(e)}"

        def cat_task_gen() -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
            """逐列生成类别型最优分箱任务。"""
            for c in cat_cols:
                series = X.get_column(c)
                col_dtype = series.dtype

                # [核心提速] Top-K 预处理使用 Polars 原生操作
                if self.max_cats_to_solver is not None:
                    top_k_df = series.value_counts(sort=True).head(self.max_cats_to_solver)
                    top_vals = top_k_df.get_column(c)

                    truncated_expr: pl.Expr = (
                        pl.when(pl.col(c).is_in(top_vals))
                        .then(pl.col(c))
                        .otherwise(pl.lit("__Mars_Other_Pre__"))
                    )
                    series = X.select(truncated_expr).to_series()

                # 获取该列的安全排除列表
                safe_exclude = self._get_safe_values(col_dtype, raw_exclude)

                # 过滤条件: 非空 且 不在排除列表中
                valid_mask = series.is_not_null()
                if safe_exclude:
                    valid_mask &= (~series.is_in(safe_exclude))

                # 执行过滤
                clean_series = series.filter(valid_mask)
                if clean_series.len() == 0:
                    continue

                valid_mask_np = valid_mask.to_numpy() # 预转 Numpy 掩码
                # 强制转为 Utf8 确保传给 optbinning 的绝对是字符串
                col_data = clean_series.cast(pl.Utf8).to_numpy()
                clean_y = y_np[valid_mask_np]

                yield c, col_data, clean_y


        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(cat_worker)(c, data, y) for c, data, y in cat_task_gen()
        )

        for col, splits, error_msg in results:
            if splits is not None:
                self.cat_cuts_[col] = splits
            if error_msg:
                self.fit_failures_[col] = error_msg
