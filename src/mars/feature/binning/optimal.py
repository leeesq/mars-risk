"""MARS 最优分箱器。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from optbinning import OptimalBinning

from mars._compat import polars_is_in
from mars.core.constants import MIN_VARIANCE
from mars.feature.binning.base import MarsBinnerBase
from mars.feature.binning.native import MarsNativeBinner
from mars.utils.logger import logger

_FALLBACK_BINNER_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "method",
        "n_bins",
        "min_bin_size",
        "merge_small_bins",
        "remove_empty_bins",
        "cart_params",
    }
)


class MarsOptimalBinner(MarsBinnerBase):
    """
    基于数学规划的最优分箱器。

    数值特征先由 ``MarsNativeBinner`` 预分箱，再交给 OptBinning 求解。
    求解器在箱数、事件率趋势、最小样本量和事件数约束下优化 IV。
    类别特征使用 OptBinning 的类别合并。

    Attributes
    ----------
    bin_cuts_ : dict of str to list of float
        各数值特征的切点列表。

    cat_cuts_ : dict of str to list of list
        各类别特征的分组列表。

    fit_failures_ : dict of str to str
        求解失败、超时或回退的特征及诊断原因。

    Notes
    -----
    预分箱切点限定数值求解的搜索范围。样本不足、预分箱不可用、求解异常
    或无可行解时，数值特征回退到原生分箱，并在 ``fit_failures_`` 中记录原因。
    部分特征仍有可用规则时允许拟合成功；全部特征无可用规则时抛出异常。

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
        fallback_binner_params: Dict[str, Any] | None = None,
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
            数值最优求解的最小正常箱数，默认为 2；回退规则可少于该值。
        min_bin_size : float
            数值正常箱占拟合表总行数的最小比例，默认为 0.05。
            排除该特征的缺失值和特殊值后，仍按总行数计算最小样本数。
        min_bin_n_event : int
            数值最优求解中每箱最少的目标事件数，默认为 30。
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
            类别预处理保留的高频类别数，默认为 100；其余合并为
            ``__Mars_Other_Pre__``。``None`` 表示关闭这一步截断。
        min_cat_fraction : float
            传给 OptBinning ``cat_cutoff`` 的低频类别合并阈值，默认为 0.05。
        special_values : List[Any] | None
            需要独立隔离的特殊值集合。
        missing_values : List[Any] | None
            需要额外识别为缺失的值集合。
        cart_params : Dict[str, Any] | None
            透传给预分箱决策树的参数。
        fallback_binner_params : Dict[str, Any] | None
            数值最优求解失败后使用的原生分箱配置。默认使用等频分箱，
            继承 ``n_bins``、``min_bin_size``，并开启小箱合并。只允许覆盖
            ``method``、``n_bins``、``min_bin_size``、``merge_small_bins``、
            ``remove_empty_bins`` 和 ``cart_params``。
        join_threshold : int
            高基数类别映射时切换到 Join 模式的阈值。
        n_jobs : int
            预分箱与逐特征求解的并行任务数。``-1`` 使用 CPU 核心数减 1，
            至少为 1；``1`` 使用单任务。

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
        self.fallback_binner_params = self._validate_fallback_binner_params(
            fallback_binner_params,
        )

        self.OptimalBinning = OptimalBinning

    @staticmethod
    def _validate_fallback_binner_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
        """校验最优分箱失败后的原生分箱回退参数。"""
        if params is None:
            return {}

        invalid_keys = sorted(set(params) - _FALLBACK_BINNER_ALLOWED_KEYS)
        if invalid_keys:
            allowed_text = ", ".join(sorted(_FALLBACK_BINNER_ALLOWED_KEYS))
            invalid_text = ", ".join(invalid_keys)
            raise ValueError(
                "`fallback_binner_params` contains invalid keys: "
                f"{invalid_text}. Allowed keys are: {allowed_text}. "
                "`special_values`, `missing_values` and `n_jobs` are inherited "
                "from MarsOptimalBinner and must not be provided here.",
            )

        return dict(params)

    def _resolve_fallback_binner_params(self) -> Dict[str, Any]:
        """生成当前特征失败时使用的原生分箱回退参数。"""
        fallback_params: Dict[str, Any] = {
            "method": "quantile",
            "n_bins": self.n_bins,
            "min_bin_size": self.min_bin_size,
            "merge_small_bins": True,
            "remove_empty_bins": False,
            "cart_params": {},
        }
        fallback_params.update(self.fallback_binner_params)

        cart_params = fallback_params.get("cart_params")
        fallback_params["cart_params"] = dict(cart_params) if cart_params is not None else {}
        return fallback_params

    def _fit_numeric_native_fallbacks(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        features: list[str],
    ) -> dict[str, list[float]]:
        """使用单个原生分箱器为全部最优求解失败特征批量生成切点。"""
        if not features:
            return {}

        fallback_params: dict[str, Any] = self._resolve_fallback_binner_params()
        fallback_binner = MarsNativeBinner(
            **fallback_params,
            special_values=self.special_values,
            missing_values=self.missing_values,
            n_jobs=self.n_jobs,
        )
        fallback_binner.fit(X.select(features), y, features=features)
        return {
            feature: list(fallback_binner.bin_cuts_.get(feature, []))
            for feature in features
        }

    def _serialization_params(self) -> dict[str, Any]:
        """返回最优分箱器的完整构造配置。"""
        params = super()._serialization_params()
        params.update(
            {
                "min_n_bins": self.min_n_bins,
                "min_bin_size": self.min_bin_size,
                "min_bin_n_event": self.min_bin_n_event,
                "prebinning_method": self.prebinning_method,
                "n_prebins": self.n_prebins,
                "min_prebin_size": self.min_prebin_size,
                "monotonic_trend": self.monotonic_trend,
                "solver": self.solver,
                "time_limit": self.time_limit,
                "max_cats_to_solver": self.max_cats_to_solver,
                "min_cat_fraction": self.min_cat_fraction,
                "cart_params": self.cart_params,
                "fallback_binner_params": self.fallback_binner_params,
            }
        )
        return params

    def _fit_impl(self, X: pl.DataFrame, y: pl.Series | None = None) -> None:
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
            raise ValueError("MarsOptimalBinner.fit requires y.")

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
        - 1. 计算重心前置: 在 `num_task_gen` 内部用 Polars 过滤无效值，worker 只接收正常值的 NumPy 数组。

        - 2. 两阶段联动: 先调用 `MarsNativeBinner` 获取预分箱切点,
          随后将其作为 `user_splits` 注入 `optbinning`, 限定数学规划搜索的候选边界。

        - 3. 并发控制: 使用 `loky` 后端。各特征由独立进程求解；
          任务传入已过滤的连续数组，减少数据转换。
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

        # 先收集预分箱不可用的特征，避免在循环内重复构造回退分箱器。
        active_cols: list[str] = []
        fallback_reasons: dict[str, str] = {}
        for col in num_cols:
            cuts = pre_cuts_map.get(col, [])
            if len(cuts) > 2:
                active_cols.append(col)
            else:
                fallback_reasons[col] = (
                    "Prebinning produced no usable optimization splits"
                )

        # 获取全局样本总数
        n_total_samples = X.height

        def num_worker(
            col: str,
            pre_cuts: List[float],
            col_data: np.ndarray,
            y_data: np.ndarray
        ) -> Tuple[str, List[float] | None, str | None]:
            """对单个数值特征执行最优分箱求解。"""
            try:
                # 计算基于"总体"的绝对 min_bin_size
                if isinstance(self.min_bin_size, float):
                    min_bin_size_abs = int(np.ceil(self.min_bin_size * n_total_samples))
                else:
                    min_bin_size_abs = self.min_bin_size # 如果用户初始化时就传了整数

                # 绝对值检查
                # 如果当前数据量 < 最小分箱数 * 最小单箱大小, 直接回退
                if len(col_data) < self.min_n_bins * min_bin_size_abs:
                    return col, None, "Insufficient samples for optimal constraints"

                if len(col_data) < 10 or np.var(col_data) < MIN_VARIANCE:
                    return col, None, "Low variance or insufficient samples"

                # 把总样本量约束换算为正常值样本中的比例。
                # OptBinning 要求比例在 (0, 0.5] 内，超出时直接回退。
                current_ratio = min_bin_size_abs / len(col_data)

                # 如果比例超过 0.5, 说明当前数据甚至无法切分出两个满足要求的箱子
                # (例如: 要求每箱至少500人, 但当前只有800人, 500/800 = 0.625 > 0.5)
                if current_ratio > 0.5:
                    return col, None, "min_bin_size is too large for the clean sample"

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
                    return col, None, "No valid user_splits after prebinning"

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
                return col, None, f"Solver status: {opt.status}"

            except Exception as e:
                # 捕获代码级异常
                return col, None, f"{type(e).__name__}: {str(e)}"

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
                    valid_mask &= ~polars_is_in(series, pl.Series(safe_exclude))

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

        results: list[tuple[str, list[float] | None, str | None]] = []
        if active_cols:
            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(num_worker)(c, cuts, data, y)
                for c, cuts, data, y in num_task_gen()
            )

        for col, fitted_cuts, error_msg in results:
            if fitted_cuts is None:
                fallback_reasons[col] = error_msg or "Optimal binning did not return cuts"
                continue

            self.bin_cuts_[col] = fitted_cuts
            if error_msg:
                self.fit_failures_[col] = error_msg

        # NativeBinner 能在一次 Polars 查询中处理宽表，因此所有失败特征统一批量回退。
        fallback_features = [col for col in num_cols if col in fallback_reasons]
        fallback_cuts = self._fit_numeric_native_fallbacks(
            X,
            y_series,
            fallback_features,
        )
        for col in fallback_features:
            self.bin_cuts_[col] = fallback_cuts.get(col, [])
            self.fit_failures_[col] = (
                f"{fallback_reasons[col]}; native fallback applied."
            )

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
        unique_values = {col: X[col].drop_nulls().unique() for col in cat_cols}
        original_values: dict[str, dict[str, Any]] = {
            col: {
                str(key): value
                for key, value in zip(
                    unique_values[col].cast(pl.Utf8).to_list(), unique_values[col].to_list()
                )
                if value is not None
            }
            for col in cat_cols
        }

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
                original_series = X.get_column(c)
                col_dtype = original_series.dtype
                series = original_series.cast(pl.Utf8)

                # [核心提速] Top-K 预处理使用 Polars 原生操作
                if self.max_cats_to_solver is not None:
                    top_k_df = series.value_counts(sort=True).head(self.max_cats_to_solver)
                    top_vals = top_k_df.get_column(c)

                    truncated_expr: pl.Expr = (
                        pl.when(pl.col(c).is_null())
                        .then(None)
                        .when(polars_is_in(pl.col(c).cast(pl.Utf8), top_vals))
                        .then(pl.col(c).cast(pl.Utf8))
                        .otherwise(pl.lit("__Mars_Other_Pre__"))
                    )
                    series = X.select(truncated_expr).to_series()

                # 获取该列的安全排除列表
                safe_exclude = (
                    pl.Series(self._get_safe_values(col_dtype, raw_exclude), dtype=col_dtype)
                    .cast(pl.Utf8)
                    .to_list()
                )

                # 过滤条件: 非空 且 不在排除列表中
                valid_mask = series.is_not_null()
                if safe_exclude:
                    valid_mask &= ~polars_is_in(series, pl.Series(safe_exclude))

                # 执行过滤
                clean_series = series.filter(valid_mask)
                if clean_series.len() == 0:
                    continue

                valid_mask_np = valid_mask.to_numpy() # 预转 Numpy 掩码
                # 求解器接收字符串类别，正式规则再恢复原值类型。
                col_data = clean_series.cast(pl.Utf8).to_numpy()
                clean_y = y_np[valid_mask_np]

                yield c, col_data, clean_y


        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(cat_worker)(c, data, y) for c, data, y in cat_task_gen()
        )

        for col, splits, error_msg in results:
            if splits is not None:
                # 求解器使用字符串类别，正式规则保留原值类型以供转换与 JSON 重放。
                self.cat_cuts_[col] = [
                    [original_values[col].get(str(value), value) for value in group]
                    for group in splits
                ]
            if error_msg:
                self.fit_failures_[col] = error_msg
