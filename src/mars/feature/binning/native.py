"""MARS 原生分箱器。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from mars._compat import polars_is_in
from mars.feature.binning.base import MarsBinnerBase
from mars.utils.logger import logger


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
        min_bin_size: float = 0.05,
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
        features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> MarsNativeBinner:
        """
        拟合原生分箱器。

        Parameters
        ----------
        X : pl.DataFrame | pd.DataFrame
            输入特征矩阵。
        y : pl.Series | pd.Series | np.ndarray | list[Any] | None
            目标变量。仅当 ``method="cart"`` 时必填。
        features : list[str] | None
            本次拟合的特征列；不传时使用全部候选列。
        cat_features : list[str] | None
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

        super().fit(X, y, features=features, cat_features=cat_features)
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
                valid_mask &= ~polars_is_in(series, pl.Series(safe_exclude))

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

            for cut_val, cdf in zip(inner_cuts, cdf_vals):
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
                valid_mask &= ~polars_is_in(series, pl.Series(safe_exclude))

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
                keep_mask &= ~polars_is_in(pl.col(c), pl.Series(safe_exclude))

            target_col = pl.col(c).filter(keep_mask)
            unique_exprs.append(target_col.n_unique().alias(c))

        unique_counts = X.select(unique_exprs).row(0)
        col_unique_map = dict(zip(cols, unique_counts))

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
                    valid_cond &= ~polars_is_in(pl.col(c), pl.Series(safe_exclude))

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

                for val, name in zip(row, stats.columns):
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
                keep_mask &= ~polars_is_in(pl.col(c), pl.Series(safe_exclude))

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
                    keep_mask &= ~polars_is_in(pl.col(c), pl.Series(safe_exclude))
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
                    valid_mask &= ~polars_is_in(series, pl.Series(safe_exclude))

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
