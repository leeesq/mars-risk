"""MARS 最优分箱器。"""

from collections.abc import Iterator
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from optbinning import OptimalBinning

from mars.core.constants import MIN_VARIANCE
from mars.feature.base import MarsBinnerBase
from mars.feature.native_binner import MarsNativeBinner
from mars.utils.logger import logger


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
    >>> binner = MarsOptimalBinner(n_bins=2, min_bin_n_event=30)
    >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
    >>> y = pl.Series("y", [0, 0, 1, 1])
    >>> binner.fit(X, y).transform(X).columns
    ['age_bin']
    """

    def __init__(
        self,
        *,
        n_bins: int = 10,
        min_n_bins: int = 2,
        min_bin_size: float = 0.05,
        min_bin_n_event: int = 30,
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
        >>> binner = MarsOptimalBinner(n_bins=2, min_bin_n_event=30)
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

                if len(col_data) < 10 or np.var(col_data) < MIN_VARIANCE:
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
                    mask = np.concatenate(([True], diffs > MIN_VARIANCE))
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
