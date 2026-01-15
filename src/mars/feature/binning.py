from joblib import Parallel, delayed
from typing import List, Dict, Optional, Union, Any, Literal, Tuple, Set
import multiprocessing
import gc

import numpy as np
import pandas as pd
import polars as pl
from sklearn.tree import DecisionTreeClassifier

from mars.core.base import MarsTransformer
from mars.utils.logger import logger
from mars.utils.decorators import time_it

class MarsNativeBinner(MarsTransformer):
    """
    [极速分箱引擎] MarsNativeBinner
    
    完全基于 Polars 和 Sklearn 原生实现的高性能分箱器。
    针对大规模宽表 (如 2000+ 特征, 20万+ 样本) 进行了内存与速度的极致优化。
    
    核心优化策略 (Performance Strategies)
    -------------------------------------
    1. **Quantile/Uniform**: 
       利用纯 Polars 表达式进行标量聚合计算，避免了 Python 循环和数据复制，Fit 速度提升 100x。
    2. **Decision Tree (DT)**: 
       使用 `joblib` 进行多进程并行训练，通过生成器惰性传输数据，大幅降低内存峰值。
    3. **Transform**: 
       使用 Polars 的 `cut` 和 `when-then` 表达式进行映射，支持全链路 Lazy 模式。
    4. **Type Safety**:
       内置类型安全层，自动过滤混合类型配置（如 Int 列混入 String 缺失值），防止 Schema Error。

    Attributes
    ----------
    bin_cuts_ : Dict[str, List[float]]
        数值型特征的分箱切点字典。
        格式: ``{col_name: [-inf, split1, split2, ..., inf]}``。
    bin_mappings_ : Dict[str, Dict[int, str]]
        分箱索引到标签的映射字典。
        格式: ``{col_name: {0: "00_[-inf, 1.5)", -1: "Missing", ...}}``。
    bin_woes_ : Dict[str, Dict[int, float]]
        分箱索引到 WOE 值的映射字典（用于 WOE 编码）。
    """

    # 类属性：定义数值类型集合，用于快速判定列类型，避免硬编码
    NUMERIC_DTYPES: Set[pl.DataType] = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, 
        pl.Float32, pl.Float64
    }
    
    def __init__(
        self,
        features: Optional[List[str]] = None,
        method: Literal["cart", "quantile", "uniform"] = "quantile",
        n_bins: int = 5,
        special_values: Optional[List[Union[int, float, str]]] = None,
        missing_values: Optional[List[Union[int, float, str]]] = None,
        min_samples: float = 0.05,
        n_jobs: int = -1,
        remove_empty_bins: bool = False,
        join_threshold: int = 100
    ) -> None:
        """
        初始化分箱器。

        Parameters
        ----------
        features : List[str], optional
            需要分箱的特征名称列表。如果不传，fit 时会自动识别所有数值型列。
        method : Literal["cart", "quantile", "uniform"], default="quantile"
            分箱方法：
            - 'cart': 决策树分箱 (Decision Tree)，最大化信息增益。
            - 'quantile': 等频分箱 (Quantile)。
            - 'uniform': 等宽分箱 (Uniform)。
        n_bins : int, default=5
            期望的分箱数量 (不包含特殊值和缺失值箱)。
        special_values : List[Union[int, float, str]], optional
            特殊值列表 (如 -999, -998)。将被单独分为独立箱 (Index <= -3)。
        missing_values : List[Union[int, float, str]], optional
            缺失值列表 (如 -1, "unknown")。将被归类为 "Missing" (Index = -1)。
        min_samples : float, default=0.05
            仅对 method='cart' 有效。叶子节点最小样本比例。
        n_jobs : int, default=-1
            仅对 method='cart' 有效。并行核心数，-1 表示使用所有可用核心。
        remove_empty_bins : bool, default=False
            仅对 method='uniform' 有效。是否扫描全表以剔除样本数为0的空箱。
            在大宽表场景下关闭此项可显著提升速度。
        join_threshold : int, default=100
            类别特征路由阈值。基数超过此值时，Transform 阶段将由 `replace` 模式切换为 `join` 模式以提升性能。
        """
        super().__init__()
        self.features: Optional[List[str]] = features
        self.method: str = method
        self.n_bins: int = n_bins
        # 初始化列表，避免 None 导致的迭代错误
        self.special_values: List[Any] = special_values if special_values is not None else []
        self.missing_values: List[Any] = missing_values if missing_values is not None else []
        self.min_samples: float = min_samples
        # 智能设置 CPU 核心数，保留一个核心给系统
        self.n_jobs: int = max(1, multiprocessing.cpu_count() - 1) if n_jobs == -1 else n_jobs
        self.remove_empty_bins: bool = remove_empty_bins
        self.join_threshold: int = join_threshold
        
        # 状态存储初始化
        self.bin_cuts_: Dict[str, List[float]] = {}
        self.bin_mappings_: Dict[str, Dict[int, str]] = {}
        self.bin_woes_: Dict[str, Dict[int, float]] = {}
        
        # 缓存引用 (用于延迟计算 WOE)
        self._cache_X: Optional[pl.DataFrame] = None
        self._cache_y: Optional[Any] = None
        
    def _get_safe_values(self, dtype: pl.DataType, values: List[Any]) -> List[Any]:
        """
        [Helper] 类型安全清洗函数。
        
        **为什么这样做？**
        Polars 是强类型的。如果列是 Int64，但 `values` 列表中包含字符串 "unknown"，
        直接调用 `pl.col(c).is_in(values)` 会导致 Schema Error 或崩溃。
        
        **这行代码运行后有啥用？**
        根据列的物理类型，自动剔除不兼容的值。例如 Int 列只保留 Int 配置项，
        String 列则将所有配置项转为 String。

        Parameters
        ----------
        dtype : pl.DataType
            当前处理列的 Polars 数据类型。
        values : List[Any]
            用户配置的缺失值或特殊值列表。

        Returns
        -------
        List[Any]
            清洗后的类型安全列表。
        """
        if not values:
            return []
            
        is_numeric = dtype in self.NUMERIC_DTYPES
        safe_vals = []
        
        for v in values:
            if v is None: continue # None 由 is_null() 单独处理，不需要在此列表中
            
            if is_numeric:
                # 数值列：严格保留数值，剔除 bool (True==1 歧义) 和字符串
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    safe_vals.append(v)
            else:
                # 非数值列：宽容处理，全部转为字符串以匹配 Categorical/String 列
                safe_vals.append(str(v))
                
        return safe_vals

    @time_it
    def _fit_impl(self, X: pl.DataFrame, y: Optional[Any] = None, **kwargs) -> None:
        """
        训练实现的入口函数。

        Parameters
        ----------
        X : pl.DataFrame
            训练数据。
        y : Optional[Any]
            目标变量 (仅 CART 分箱需要)。
        """
        # 1. 缓存数据引用，仅用于 transform 阶段请求 return_type='woe' 时的延迟计算
        self._cache_X = X
        self._cache_y = y

        # 2. 确定目标列 (仅筛选数值列，忽略全空列)
        all_target_cols = self.features if self.features else X.columns
        target_cols: List[str] = []
        null_cols: List[str] = [] 

        for c in all_target_cols:
            if c not in X.columns: continue
            
            # 判定全空/Null类型列，记录下来以便直接注册为空箱
            if X[c].dtype == pl.Null or X[c].null_count() == X.height:
                null_cols.append(c)
                continue

            # 仅处理数值类型
            if self._is_numeric(X[c]):
                target_cols.append(c)

        # 注册全空列为空切点，防止 transform 时漏列
        for c in null_cols:
            self.bin_cuts_[c] = []

        if not target_cols:
            if not null_cols:
                logger.warning("No numeric columns found for binning.")
            return

        # ========================================================
        # [优化] 极速预过滤 (常量特征剔除)
        # ========================================================
        valid_cols: List[str] = []
        
        # 构建聚合表达式，一次性扫描全表获取 Min/Max
        stats_exprs = []
        for c in target_cols:
            stats_exprs.append(pl.col(c).min().alias(f"{c}_min"))
            stats_exprs.append(pl.col(c).max().alias(f"{c}_max"))
            
        # 触发计算 (Eager 模式下立即执行，速度极快)
        stats_row = X.select(stats_exprs).row(0)
        
        for i, c in enumerate(target_cols):
            min_val = stats_row[i * 2]
            max_val = stats_row[i * 2 + 1]
            
            # 如果 Min == Max，说明是常量列，无需分箱，直接设为全区间
            if min_val == max_val:
                logger.warning(f"Feature '{c}' is constant. Skipped.")
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue
            
            valid_cols.append(c)

        if not valid_cols:
            return

        # 3. 检查 CART 方法的依赖
        if y is None and self.method == "cart":
            raise ValueError("Decision Tree Binning ('cart') requires target 'y'.")

        logger.info(f"⚙️ Fitting bins for {len(valid_cols)} features (Native Mode: {self.method})...")

        # 4. 策略分发
        if self.method == "quantile":
            self._fit_quantile(X, valid_cols)
        elif self.method == "uniform":
            self._fit_uniform(X, valid_cols)
        elif self.method == "cart":
            self._fit_cart_parallel(X, y, valid_cols)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_quantile(self, X: pl.DataFrame, cols: List[str]) -> None:
        """
        执行极速等频分箱 (Quantile Binning)。
        
        **核心优化**:
        不使用 Python 循环逐列计算，而是构建一个包含所有列分位数计算的
        巨大 Polars 表达式列表，发送给 Rust 引擎一次性执行。
        """
        # 1. 构建分位点
        if self.n_bins <= 1:
            quantiles = [0.5]
        else:
            quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1].tolist()
            
        raw_exclude = self.special_values + self.missing_values
        
        # 2. 构建表达式列表 (Flattened)
        q_exprs = []
        for c in cols:
            # 获取当前列安全的排除值
            safe_exclude = self._get_safe_values(X.schema[c], raw_exclude)
            target_col = pl.col(c)
            # 如果有需要排除的值，在计算分位数前先置为 Null (不参与计算)
            if safe_exclude:
                target_col = pl.when(pl.col(c).is_in(safe_exclude)).then(None).otherwise(pl.col(c))
            
            for i, q in enumerate(quantiles):
                # 别名技巧: col:::idx，便于后续解析
                alias_name = f"{c}:::{i}"
                q_exprs.append(target_col.quantile(q).alias(alias_name))
        
        # 3. 触发计算 (One-Shot Query)
        stats = X.select(q_exprs)
        row = stats.row(0)
        
        # 4. 解析结果并去重排序
        temp_cuts: Dict[str, List[float]] = {c: [] for c in cols}
        
        for val, name in zip(row, stats.columns):
            c_name, _ = name.split(":::")
            if val is not None and not np.isnan(val):
                temp_cuts[c_name].append(val)

        for c in cols:
            cuts = sorted(list(set(temp_cuts[c]))) 
            self.bin_cuts_[c] = [float('-inf')] + cuts + [float('inf')]

    def _fit_uniform(self, X: pl.DataFrame, cols: List[str]) -> None:
        """
        执行极速等宽分箱 (Uniform/Step Binning)。
        
        **核心优化**:
        分为两阶段。第一阶段批量计算 Min/Max/Unique。
        第二阶段（可选）批量计算 Histogram 以剔除空箱。
        """
        raw_exclude = self.special_values + self.missing_values
        
        # --- 阶段 1: 基础统计量 ---
        exprs = []
        col_safe_excludes = {} 

        for c in cols:
            safe_exclude = self._get_safe_values(X.schema[c], raw_exclude)
            col_safe_excludes[c] = safe_exclude # 缓存供后续使用

            target_col = pl.col(c)
            if safe_exclude:
                target_col = target_col.filter(~pl.col(c).is_in(safe_exclude))
            
            exprs.append(target_col.min().alias(f"{c}_min"))
            exprs.append(target_col.max().alias(f"{c}_max"))
            exprs.append(target_col.n_unique().alias(f"{c}_n_unique"))

        stats = X.select(exprs)
        row = stats.row(0)
        
        initial_cuts_map = {}
        pending_optimization_cols = []

        # 解析统计量，生成等距切点
        for i, c in enumerate(cols):
            base_idx = i * 3
            min_val, max_val, n_unique = row[base_idx], row[base_idx + 1], row[base_idx + 2]
            safe_exclude = col_safe_excludes[c]

            if min_val is None or max_val is None:
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue
            
            # 优化: 低基数检查 (Unique <= N_Bins)，直接取中点切分
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

        # --- 阶段 2: 空箱优化 (可选) ---
        if pending_optimization_cols:
            batch_exprs = []
            for c in pending_optimization_cols:
                cuts = initial_cuts_map[c]
                breaks = cuts[1:-1]
                target_col = pl.col(c)
                safe_exclude = col_safe_excludes[c]
                
                if safe_exclude:
                    target_col = target_col.filter(~pl.col(c).is_in(safe_exclude))
                
                labels = [str(i) for i in range(len(breaks)+1)]
                
                # 批量计算直方图 (Value Counts)
                batch_exprs.append(
                    target_col.cut(breaks, labels=labels, left_closed=True)
                    .value_counts().implode().alias(f"{c}_counts")
                )

            logger.info(f"⚡ Batch scanning {len(pending_optimization_cols)} columns for empty bins...")
            batch_counts_df = X.select(batch_exprs)
            
            # 解析并剔除 Count=0 的箱
            for c in pending_optimization_cols:
                inner_series = batch_counts_df.get_column(f"{c}_counts")[0]
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
                    if i in valid_indices: new_cuts.append(cuts[i+1])
                
                if new_cuts[-1] != float('inf'): new_cuts.append(float('inf'))
                self.bin_cuts_[c] = sorted(list(set(new_cuts)))

    def _fit_cart_parallel(self, X: pl.DataFrame, y: Any, cols: List[str]) -> None:
        """
        执行并行的决策树分箱 (Decision Tree Binning)。
        
        **核心优化**:
        1. 使用 Generator (task_generator) 惰性产出数据，避免一次性复制所有列数据到内存。
        2. Worker 函数只接收 Numpy 数组，减少序列化开销。
        3. 在 Generator 内部利用 Polars Rust 内核进行极速过滤和类型转换 (Float32)。
        """
        y_np = np.array(y)
        if len(y_np) != X.height:
            raise ValueError(f"Target 'y' length mismatch: X({X.height}) vs y({len(y_np)})")

        # 定义 Worker 逻辑：纯 Sklearn 拟合
        def worker(col_name: str, x_clean_np: np.ndarray, y_clean_np: np.ndarray) -> Tuple[str, List[float]]:
            try:
                if len(x_clean_np) < self.n_bins * 10: 
                    return col_name, [float('-inf'), float('inf')]
                
                cart = DecisionTreeClassifier(
                    max_leaf_nodes=self.n_bins,
                    min_samples_leaf=self.min_samples,
                    random_state=42
                )
                cart.fit(x_clean_np, y_clean_np)
                cuts = cart.tree_.threshold[cart.tree_.threshold != -2]
                cuts = np.sort(np.unique(cuts)).tolist()
                return col_name, [float('-inf')] + cuts + [float('inf')]
            except Exception:
                return col_name, [float('-inf'), float('inf')]

        logger.info(f"⚙️ Pre-processing {len(cols)} features for CART...")
        
        raw_exclude = self.special_values + self.missing_values
        
        # 任务生成器：按需生成数据
        def task_generator():
            for c in cols:
                # ✅ [Critical Fix] 类型安全过滤，防止在 Int 列上查询 "unknown" 报错
                safe_exclude = self._get_safe_values(X.schema[c], raw_exclude)

                # Polars 极速预处理：过滤 -> 转换 Float32
                valid_df = X.select([
                    pl.col(c).alias("x"),
                    pl.lit(y_np).alias("y")
                ]).filter(
                    pl.col("x").is_not_null() & 
                    ~pl.col("x").is_nan() & 
                    ~pl.col("x").is_in(safe_exclude)
                )
                
                if valid_df.height == 0: continue

                # Zero-copy (如果可能) 转 Numpy
                x_clean = valid_df["x"].cast(pl.Float32).to_numpy(writable=False).reshape(-1, 1)
                y_clean = valid_df["y"].to_numpy(writable=False)
                
                yield c, x_clean, y_clean

        logger.info(f"🚀 Starting parallel DT fitting with n_jobs={self.n_jobs}...")
        
        results = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=0)(
            delayed(worker)(name, x, y) for name, x, y in task_generator()
        )
        
        for col_name, cuts in results:
            self.bin_cuts_[col_name] = cuts

    @time_it
    def _materialize_woe(self) -> None:
        """
        [极速 WOE 物化引擎 v7.0]
        
        **为什么这样做？**
        Transform 阶段如果每次都实时计算 WOE，对于 2000+ 特征的宽表，
        Polars 可能会构建过大的计算图导致内存飙升。
        
        **做了什么？**
        1. 将 WOE 计算拆分为 batch (200列一组)。
        2. 使用 Eager 模式立即计算并回收内存 (`gc.collect`)。
        3. 使用 group_by 极速聚合而不是构建复杂的 when-then 逻辑。
        """
        if self._cache_X is None or self._cache_y is None:
            logger.warning("No training data cached. WOE cannot be computed.")
            return

        logger.info("⚡ [Auto-Trigger] Materializing WOE (Eager Cross-Grouping Mode)...")
        y_name = "_y_tmp"
        
        y_series = pl.Series(name=y_name, values=self._cache_y)
        total_bads = y_series.sum()
        total_goods = len(y_series) - total_bads
        
        # 涵盖数值和类别特征
        bin_cols_orig = [c for c in self.bin_cuts_.keys()] + \
                        (list(self.cat_cuts_.keys()) if hasattr(self, 'cat_cuts_') else [])

        batch_size = 200 
        for i in range(0, len(bin_cols_orig), batch_size):
            batch_features = bin_cols_orig[i : i + batch_size]
            
            # Step A: 局部 Eager 转换 (获取 Index)
            X_batch_bin = self.transform(
                self._cache_X.select(batch_features), 
                return_type="index", 
                lazy=False
            )
            X_batch_bin = X_batch_bin.with_columns(y_series)
            
            # Step B: 逐列聚合计算 bad/good
            for c in batch_features:
                c_bin = f"{c}_bin"
                stats = (
                    X_batch_bin.group_by(c_bin)
                    .agg([
                        pl.col(y_name).sum().alias("b"),
                        pl.count().alias("n")
                    ])
                )
                
                # Step C: 向量化计算 WOE 并存入字典
                idxs = stats.get_column(c_bin)
                b = stats.get_column("b")
                n = stats.get_column("n")
                
                woe_vals = (((b + 1e-6) / (total_bads + 1e-6)) / 
                            (((n - b) + 1e-6) / (total_goods + 1e-6))).log()
                
                self.bin_woes_[c] = dict(zip(idxs.to_list(), woe_vals.to_list()))
            
            # Step D: 强制内存断层
            del X_batch_bin
            gc.collect()
            
        logger.info(f"✅ [V7.0] Materialization finished for {len(self.bin_woes_)} features.")

    def _transform_impl(
        self, 
        X: Union[pl.DataFrame, pl.LazyFrame], 
        return_type: Literal["index", "label", "woe"] = "index"
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        [混合动力分箱转换实现] 
        核心转换逻辑，兼容数值与类别特征，支持 Eager/Lazy。

        Parameters
        ----------
        X : Union[pl.DataFrame, pl.LazyFrame]
            待转换数据。
        return_type : Literal["index", "label", "woe"]
            输出格式。

        Returns
        -------
        Union[pl.DataFrame, pl.LazyFrame]
            转换后的数据。
        """
        exprs = []
        temp_join_cols = []
        
        # 索引协议常量: 与下游 Profiler 对齐
        IDX_MISSING = -1
        IDX_OTHER   = -2
        IDX_SPECIAL_START = -3

        # 自动触发 WOE 计算
        if return_type == "woe" and not self.bin_woes_:
            self._materialize_woe()

        # 获取 Schema (Lazy/Eager 兼容写法)
        schema_map = X.collect_schema() if isinstance(X, pl.LazyFrame) else X.schema
        current_columns = schema_map.names()
        
        all_train_cols = list(set(
            list(self.bin_cuts_.keys()) + 
            (list(self.cat_cuts_.keys()) if hasattr(self, 'cat_cuts_') else [])
        ))

        for col in all_train_cols:
            if col not in current_columns: continue
            
            # --- [关键] 计算类型安全值 ---
            # 这一步至关重要，防止例如在 Int 列上查询 "unknown" 导致的崩溃
            col_dtype = schema_map[col]
            safe_missing_vals = self._get_safe_values(col_dtype, self.missing_values)
            safe_special_vals = self._get_safe_values(col_dtype, self.special_values)
            is_numeric_col = col_dtype in self.NUMERIC_DTYPES

            # =========================================================
            # Part A: 数值型分箱 (Numeric Binning)
            # =========================================================
            if col in self.bin_cuts_:
                cuts = self.bin_cuts_[col]
                
                # 1. 缺失值逻辑: Is Null OR Is Missing Val
                missing_cond = pl.col(col).is_null() 
                if is_numeric_col: missing_cond |= pl.col(col).cast(pl.Float64).is_nan()
                for v in safe_missing_vals: missing_cond |= (pl.col(col) == v)
                
                layer_missing = pl.when(missing_cond).then(pl.lit(IDX_MISSING, dtype=pl.Int16))
                
                # 2. 正常分箱逻辑: Cut
                breaks = cuts[1:-1] if len(cuts) > 2 else []
                col_mapping = {IDX_MISSING: "Missing", IDX_OTHER: "Other"}
                
                if not breaks:
                    col_mapping[0] = "00_[-inf, inf)"
                    layer_normal = pl.lit(0, dtype=pl.Int16)
                else:
                    for i in range(len(cuts) - 1):
                        low, high = cuts[i], cuts[i+1]
                        col_mapping[i] = f"{i:02d}_[{low:.3g}, {high:.3g})"
                    # 显式生成 labels 确保 cast(Int16) 成功，修复 PSI=0 Bug
                    bin_labels = [str(i) for i in range(len(breaks) + 1)]
                    layer_normal = pl.col(col).cut(
                        breaks, labels=bin_labels, left_closed=True
                    ).cast(pl.Int16)
                
                # 3. 特殊值逻辑: 瀑布流覆盖
                current_branch = layer_normal
                if safe_special_vals:
                    for i in range(len(safe_special_vals)-1, -1, -1):
                        v = safe_special_vals[i]
                        idx = IDX_SPECIAL_START - i 
                        col_mapping[idx] = f"Special_{v}"
                        current_branch = pl.when(pl.col(col) == v).then(pl.lit(idx, dtype=pl.Int16)).otherwise(current_branch)
                
                # 组合: Missing -> Special -> Normal
                final_idx_expr = layer_missing.otherwise(current_branch)
                self.bin_mappings_[col] = col_mapping
                
            # =========================================================
            # Part B: 类别型分箱 (Categorical Binning)
            # =========================================================
            elif hasattr(self, 'cat_cuts_') and col in self.cat_cuts_:
                splits = self.cat_cuts_[col]
                cat_to_idx: Dict[str, int] = {}
                idx_to_label: Dict[int, str] = {IDX_MISSING: "Missing", IDX_OTHER: "Other"}
                
                # 更新映射表
                if safe_special_vals:
                    for i, val in enumerate(safe_special_vals):
                        idx_to_label[IDX_SPECIAL_START - i] = f"Special_{val}"

                for i, group in enumerate(splits):
                    disp_grp = group[:3] if len(group) > 3 else group
                    suffix = ",..." if len(group) > 3 else ""
                    idx_to_label[i] = f"{i:02d}_[{','.join(str(g) for g in disp_grp) + suffix}]"
                    for val in group: cat_to_idx[str(val)] = i
                
                self.bin_mappings_[col] = idx_to_label
                # 强转 String，确保类别匹配安全
                target_col = pl.col(col).cast(pl.Utf8)
                
                # 1. 缺失值
                missing_cond = target_col.is_null()
                for v in safe_missing_vals:
                    missing_cond |= (target_col == str(v))
                layer_missing = pl.when(missing_cond).then(pl.lit(IDX_MISSING, dtype=pl.Int16))
                
                # 2. 特殊值
                current_branch = pl.lit(IDX_OTHER, dtype=pl.Int16)
                if safe_special_vals:
                    for i in range(len(safe_special_vals)-1, -1, -1):
                        v = safe_special_vals[i]
                        idx = IDX_SPECIAL_START - i
                        current_branch = pl.when(target_col == str(v)).then(pl.lit(idx, dtype=pl.Int16)).otherwise(current_branch)
                
                # 3. 路由: Join (高基数) vs Replace (低基数)
                # Join 模式避免了在表达式中构建巨大的 when-then 树，极大提升性能
                if len(cat_to_idx) > self.join_threshold:
                    map_df = pl.DataFrame({
                        "_k": list(cat_to_idx.keys()), 
                        f"_idx_{col}": list(cat_to_idx.values())
                    }).with_columns([
                        pl.col("_k").cast(pl.Utf8),
                        pl.col(f"_idx_{col}").cast(pl.Int16)
                    ])
                    # 兼容 Lazy 模式的 Join
                    join_tbl = map_df.lazy() if isinstance(X, pl.LazyFrame) else map_df
                    X = X.join(join_tbl, left_on=target_col, right_on="_k", how="left")
                    
                    temp_join_cols.append(f"_idx_{col}")
                    layer_normal = pl.col(f"_idx_{col}")
                else:
                    layer_normal = target_col.replace(cat_to_idx, default=None).cast(pl.Int16)
                
                # 组合: Missing -> Normal (Join Result) -> Special/Other
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
                exprs.append(final_idx_expr.replace(woe_map).cast(pl.Float64).alias(f"{col}_woe") if woe_map else pl.lit(0.0).alias(f"{col}_woe"))
            else:
                str_map = {str(k): v for k, v in self.bin_mappings_.get(col, {}).items()}
                exprs.append(final_idx_expr.cast(pl.Utf8).replace(str_map).alias(f"{col}_bin"))

        # 清理 Join 产生的临时列
        return X.with_columns(exprs).drop(temp_join_cols)

    def transform(
        self, 
        X: Any, 
        return_type: Literal["index", "label", "woe"] = "index", 
        lazy: bool = False
    ) -> Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame]:
        """
        对数据应用分箱转换。

        Parameters
        ----------
        X : Any
            输入数据 (Pandas/Polars DataFrame)。
        return_type : Literal["index", "label", "woe"], default="index"
            返回类型：
            - 'index': 返回 Int16 的箱索引 (-1=Missing, 0, 1...)。最快。
            - 'label': 返回字符串标签 (如 "01_[0.5, 1.2)")。
            - 'woe': 返回 Float64 的 WOE 编码值。
        lazy : bool, default=False
            是否返回 LazyFrame。如果为 True，不会触发计算，适合构建计算图。

        Returns
        -------
        Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame]
            转换后的数据框。
        """
        # 1. 智能输入处理：确保是 Polars 对象
        if isinstance(X, pl.LazyFrame):
            X_pl = X
        else:
            X_pl = self._ensure_polars(X)
        
        # 2. 模式切换：如果需要 Lazy，转为 LazyFrame
        if lazy and isinstance(X_pl, pl.DataFrame):
            X_pl = X_pl.lazy()
        
        # 3. 执行核心逻辑
        res = self._transform_impl(X_pl, return_type=return_type)
        
        # 4. 输出格式控制
        if not lazy:
            if isinstance(res, pl.LazyFrame): res = res.collect()
            if isinstance(X, pd.DataFrame): return res.to_pandas()
        return res

    def get_bin_mapping(self, col: str) -> Dict[int, str]:
        """获取指定列的分箱映射字典。"""
        return self.bin_mappings_.get(col, {})

    def _is_numeric(self, series: pl.Series) -> bool:
        """Helper: 判断 Series 是否为数值类型。"""
        if series.dtype == pl.Null:
            return False
        return series.dtype in self.NUMERIC_DTYPES
    
    @time_it
    def compute_bin_stats(self, X: pl.DataFrame, y: Any) -> pl.DataFrame:
        """
        [极速指标引擎 v2.0] 计算全量分箱指标。
        
        **核心优化**:
        使用 `unpivot` + `group_by` 实现矩阵化聚合，而非循环逐列聚合。
        
        Parameters
        ----------
        X : pl.DataFrame
            特征数据。
        y : Any
            目标标签。

        Returns
        -------
        pl.DataFrame
            包含 feature, bin_index, count, bad_rate, woe, iv, ks 等指标的宽表。
        """
        y_name = "_target_tmp"
        # 强制开启 Lazy 转换以合并查询计划
        X_bin_lazy = self.transform(X, return_type="index", lazy=True)
        X_bin_lazy = X_bin_lazy.with_columns(pl.lit(np.array(y)).alias(y_name))
        
        # 获取全局统计量 (T, B)
        meta = X_bin_lazy.select([
            pl.count().alias("total_counts"),
            pl.col(y_name).sum().alias("total_bads")
        ]).collect()
        
        total_counts = meta[0, "total_counts"]
        total_bads = meta[0, "total_bads"]
        total_goods = total_counts - total_bads
        global_bad_rate = (total_bads / total_counts) if total_counts > 0 else 0
        
        current_cols = X_bin_lazy.collect_schema().names()
        bin_cols = [c for c in current_cols if c.endswith("_bin")]
        logger.info(f"📊 Lazily computing risk metrics for {len(bin_cols)} features...")

        # 利用 unpivot 实现矩阵化并行聚合计划
        # (rows * cols) -> (rows * features, 2)
        lf_stats = (
            X_bin_lazy.unpivot(
                index=[y_name],
                on=bin_cols,
                variable_name="feature",
                value_name="bin_index"
            )
            .group_by(["feature", "bin_index"])
            .agg([
                pl.count().alias("count"),
                pl.col(y_name).sum().alias("bad")
            ])
            .with_columns([
                (pl.col("count") - pl.col("bad")).alias("good")
            ])
        )

        # 向量化计算各项指标 (WOE, IV, Lift)
        lf_stats = lf_stats.with_columns([
            (pl.col("count") / total_counts).alias("count_dist"),
            (pl.col("bad") / pl.col("count")).alias("bad_rate"),
            (pl.col("bad") / (total_bads + 1e-6)).alias("bad_dist"),
            (pl.col("good") / (total_goods + 1e-6)).alias("good_dist")
        ]).with_columns([
            ((pl.col("bad_dist") + 1e-6) / (pl.col("good_dist") + 1e-6)).log().alias("woe")
        ]).with_columns([
            ((pl.col("bad_dist") - pl.col("good_dist")) * pl.col("woe")).alias("bin_iv"),
            (pl.col("bad_rate") / (global_bad_rate + 1e-6)).alias("lift")
        ])

        # Window Function 计算 KS (基于特征内累计)
        lf_stats = lf_stats.sort(["feature", "bin_index"]).with_columns([
            pl.col("bad_dist").cum_sum().over("feature").alias("cum_bad_dist"),
            pl.col("good_dist").cum_sum().over("feature").alias("cum_good_dist")
        ]).with_columns([
            (pl.col("cum_bad_dist") - pl.col("cum_good_dist")).abs().alias("bin_ks")
        ])

        # 最终物理物化
        stats_df = lf_stats.collect(streaming=True)
        
        # 关联标签并计算总 IV
        final_list = []
        for feat_name in bin_cols:
            orig_name = feat_name.replace("_bin", "")
            mapping = self.get_bin_mapping(orig_name)
            
            feat_stats = stats_df.filter(pl.col("feature") == feat_name).with_columns([
                pl.col("bin_index").cast(pl.Utf8).replace({str(k): v for k, v in mapping.items()}).alias("bin_label")
            ])
            
            # 同时同步 WOE 字典，方便后续 transform 使用
            self.bin_woes_[orig_name] = dict(zip(feat_stats["bin_index"].to_list(), feat_stats["woe"].to_list()))
            final_list.append(feat_stats)

        result_df = pl.concat(final_list)
        iv_sum = result_df.group_by("feature").agg(pl.col("bin_iv").sum().alias("total_iv"))
        return result_df.join(iv_sum, on="feature")


class MarsOptimalBinner(MarsNativeBinner):
    """
    [混合动力分箱引擎] MarsOptimalBinner

    该类实现了基于混合动力架构 (Hybrid Engine) 的最优分箱算法。
    
    设计目标
    -------
    解决传统 OptBinning 在大规模数据（如 20万行 x 2000列）上直接求解 MIP (混合整数规划) 
    导致的计算性能瓶颈，同时保留其数学规划带来的最优性和单调性约束能力。

    核心架构 (Architecture)
    -----------------------
    1. **Numeric Pipeline (数值型特征)**: "两阶段火箭" 模式
       - **Stage 1 (Pre-binning)**: 利用 Polars 进行极速分位数/等宽预分箱 (O(N))。
         将原始数据离散化为细粒度 (如 50 箱) 的候选区间。
       - **Stage 2 (Optimization)**: 将预分箱切点注入 OptBinning (MIP Solver)。
         利用约束编程 (CP) 求解满足单调性约束的最优合并方案 (O(1))。
    
    2. **Categorical Pipeline (类别型特征)**:
       - **Pre-filtering**: 对高基数特征进行 Top-K 过滤，将长尾类别归并为 "Other_Pre"。
       - **Optimization**: 调用 OptBinning 处理类别合并。

    Attributes
    ----------
    bin_cuts_ : Dict[str, List[float]]
        数值型特征的最优切点字典。
    cat_cuts_ : Dict[str, List[List[Any]]]
        类别型特征的分箱规则字典。
        格式: ``{col: [['A', 'B'], ['C'], ['D']]}``，表示 A和B 归为箱0，C 归为箱1...
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        cat_features: Optional[List[str]] = None,
        n_bins: int = 5,
        n_prebins: int = 50,
        prebinning_method: Literal["quantile", "uniform", "cart"] = "quantile",
        monotonic_trend: str = "auto_asc_desc",
        solver: str = "cp",
        time_limit: int = 10,
        special_values: Optional[List[Union[int, float, str]]] = None,
        missing_values: Optional[List[Union[int, float, str]]] = None,
        cat_cutoff: Optional[int] = 100,  
        join_threshold: int = 1000,       
        n_jobs: int = -1  
    ) -> None:
        """
        初始化混合动力分箱器。
        
        Parameters
        ----------
        cat_features : List[str], optional
            需要处理的类别型特征列表。
        n_prebins : int, default=50
            第一阶段预分箱的数量。数量越多，第二阶段优化的空间越大，但速度越慢。
        prebinning_method : str, default="quantile"
            第一阶段预分箱的方法。
        monotonic_trend : str, default="auto_asc_desc"
            单调性约束: 'auto', 'ascending', 'descending'。
        solver : str, default="cp"
            OptBinning 求解器: 'cp' (Constraint Programming) 或 'mip'。
        time_limit : int, default=10
            求解器超时时间 (秒)。
        cat_cutoff : Optional[int], default=100
            类别特征 Top-K 截断阈值。保留频数最高的 K 个类别，其余归为 Other。
        
        (其余参数参见父类 MarsNativeBinner)
        """
        # 初始化父类 MarsNativeBinner (负责 Stage 1)
        super().__init__(
            features=features,
            method=prebinning_method,
            n_bins=n_bins,
            special_values=special_values,
            missing_values=missing_values,
            n_jobs=n_jobs
        )
        self.cat_features: List[str] = cat_features if cat_features is not None else []
        self.n_prebins: int = n_prebins
        self.monotonic_trend: str = monotonic_trend
        self.solver: str = solver
        self.time_limit: int = time_limit
        self.cat_cutoff: Optional[int] = cat_cutoff
        self.join_threshold: int = join_threshold
        
        # 专门存储类别特征的分箱规则
        # 结构: {col_name: [['A', 'B'], ['C'], ['D']]}
        self.cat_cuts_: Dict[str, List[List[Any]]] = {}

        # 检查 OptBinning 依赖
        try:
            import optbinning
        except ImportError:
            logger.warning("⚠️ 'optbinning' not installed. Optimal binning will fallback to pre-binning.")

    def _fit_impl(self, X: pl.DataFrame, y: Optional[Any] = None, **kwargs) -> None:
        """
        训练入口：分流数值型和类别型特征到不同的 Pipeline。
        """
        if y is None:
            raise ValueError("Optimal Binning requires target 'y' to calculate IV/WOE.")

        y_np = np.array(y)
        
        all_target_cols = self.features if self.features else X.columns
        cat_set = set(self.cat_features)
        
        num_cols = []
        cat_cols = []
        null_cols = [] 

        for c in all_target_cols:
            if c not in X.columns: continue
            
            # 1. 优先判定类别
            if c in cat_set:
                cat_cols.append(c)
                continue
            
            # 2. 判定全空
            if X[c].dtype == pl.Null or X[c].null_count() == X.height:
                null_cols.append(c)
                continue

            # 3. 判定数值
            if self._is_numeric(X[c]):
                num_cols.append(c)

        if not num_cols and not cat_cols and not null_cols:
            logger.warning("No valid numeric or categorical columns found.")
            return
        
        logger.info(f"📊 Features identified: {len(num_cols)} Numeric, {len(cat_cols)} Categorical, {len(null_cols)} All-Null.")

        # 注册全空列
        for c in null_cols:
            self.bin_cuts_[c] = []

        if num_cols:
            self._fit_numerical_pipeline(X, y_np, num_cols)

        if cat_cols:
            self._fit_categorical_pipeline(X, y_np, cat_cols)

    def _fit_numerical_pipeline(self, X: pl.DataFrame, y_np: np.ndarray, num_cols: List[str]) -> None:
        """
        [Pipeline] 数值型特征混合动力处理流水线。
        """
        logger.info(f"🚀 [Numeric Pipeline] Starting Hybrid Engine for {len(num_cols)} features...")
        
        # --- Stage 1: 极速预分箱 (Pre-binning) ---
        logger.info(f"   [Stage 1] Pre-binning with Polars (Method: {self.method}, Pre-bins: {self.n_prebins})...")
        
        pre_binner = MarsNativeBinner(
            features=num_cols,
            method=self.method, 
            n_bins=self.n_prebins, 
            special_values=self.special_values,
            missing_values=self.missing_values,
            n_jobs=self.n_jobs,
            remove_empty_bins=False 
        )
        pre_binner.fit(X, y_np)
        pre_cuts_map = pre_binner.bin_cuts_

        # 筛选出有意义的列 (切点数 > 2 表示不仅仅是 inf)
        active_cols = []
        for col, cuts in pre_cuts_map.items():
            if len(cuts) > 2: 
                active_cols.append(col)
            else:
                self.bin_cuts_[col] = cuts 

        if not active_cols:
            logger.info("   [Stage 1] Completed. No features require further optimization.")
            return

        logger.info(f"   [Stage 1] Completed. {len(active_cols)} features passed to Solver.")
        
        # --- Stage 2: 并行优化 (Optimization) ---
        logger.info(f"   [Stage 2] Optimizing with Solver (Engine: {self.solver}, TimeLimit: {self.time_limit}s)...")
        
        def num_worker(col: str, pre_cuts: List[float], col_data: np.ndarray) -> Tuple[str, List[float]]:
            fallback_res = (col, pre_cuts)
            try:
                from optbinning import OptimalBinning
                
                # 1. 基础检查
                valid_mask = ~np.isnan(col_data)
                valid_data = col_data[valid_mask]
                if len(valid_data) < 10 or np.var(valid_data) < 1e-8:
                    return fallback_res

                # 2. 注入 Stage 1 切点 (User Splits)
                user_splits = np.array(pre_cuts[1:-1]) 
                if len(user_splits) == 0:
                    return fallback_res
                
                opt = OptimalBinning(
                    name=col, dtype="numerical", solver=self.solver,
                    monotonic_trend=self.monotonic_trend,
                    user_splits=user_splits,  
                    max_n_bins=self.n_bins,   
                    time_limit=self.time_limit, 
                    min_bin_size=0.05,
                    verbose=False
                )
                opt.fit(valid_data, y_np[valid_mask])
                
                if opt.status in ["OPTIMAL", "FEASIBLE"]:
                    res_cuts = [float('-inf')] + list(opt.splits) + [float('inf')]
                    # Bug Fix: 防止 Solver 优化过度
                    if len(res_cuts) <= 2 and len(pre_cuts) > 2:
                        return fallback_res
                    return col, res_cuts
                
                return fallback_res 
            except Exception:
                return fallback_res

        task_gen = (
            (c, pre_cuts_map[c], X.select(c).to_series().to_numpy()) 
            for c in active_cols
        )
        
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(num_worker)(c, cuts, data) for c, cuts, data in task_gen
        )
        
        for col, cuts in results:
            self.bin_cuts_[col] = cuts
        
        logger.info(f"✅ [Numeric Pipeline] Optimization finished for {len(active_cols)} features.")

    def _fit_categorical_pipeline(self, X: pl.DataFrame, y_np: np.ndarray, cat_cols: List[str]) -> None:
        """
        [Pipeline] 类别型特征处理流水线 (带 Top-K 优化)。
        """
        logger.info(f"🚀 [Categorical Pipeline] Optimizing {len(cat_cols)} features (Top-K Cutoff: {self.cat_cutoff})...")

        def cat_worker(col: str, col_data_raw: np.ndarray) -> Tuple[str, Optional[List[List[Any]]]]:
            try:
                from optbinning import OptimalBinning
                
                # Bug Fix: 剔除 None 防止 astype(str) 生成 "None"
                mask_valid = pd.notnull(col_data_raw)
                valid_data = col_data_raw[mask_valid].astype(str)
                valid_y = y_np[mask_valid]
                
                # Top-K 预处理: 将长尾类别归为 "Other_Pre"
                if self.cat_cutoff is not None:
                    unique_vals, counts = np.unique(valid_data, return_counts=True)
                    if len(unique_vals) > self.cat_cutoff:
                        top_indices = np.argsort(-counts)[:self.cat_cutoff]
                        top_vals = set(unique_vals[top_indices])
                        mask_keep = np.isin(valid_data, list(top_vals))
                        valid_data = np.where(mask_keep, valid_data, "Other_Pre")

                opt = OptimalBinning(
                    name=col, dtype="categorical", solver=self.solver,
                    max_n_bins=self.n_bins, 
                    time_limit=self.time_limit,
                    cat_cutoff=0.05, 
                    verbose=False
                )
                opt.fit(valid_data, valid_y)
                
                if opt.status in ["OPTIMAL", "FEASIBLE"]:
                    return col, opt.splits
                
                return col, None
            except Exception:
                return col, None

        task_gen = (
            (c, X.select(c).to_series().to_numpy()) 
            for c in cat_cols
        )
        
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(cat_worker)(c, data) for c, data in task_gen
        )
        
        for col, splits in results:
            if splits is not None:
                self.cat_cuts_[col] = splits
        
        logger.info(f"✅ [Categorical Pipeline] Finished. Rules generated for {len(self.cat_cuts_)} features.")

    def _transform_impl(
        self, 
        X: Union[pl.DataFrame, pl.LazyFrame], 
        return_type: Literal["index", "label", "woe"] = "index"
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        [Override] 混合动力分箱的转换实现。
        
        必须重写以确保使用 MarsOptimalBinner 的上下文（如 bin_cuts_ 和 cat_cuts_），
        并且为了安全起见，再次应用类型安全检查。
        
        (注意：由于继承关系，如果 MarsNativeBinner 的 _transform_impl 已经支持了 cat_cuts_，
         实际上可以直接调用 super，但为了 Explicit Safety 和独立的扩展性，这里保留独立实现，
         并修复了类型安全逻辑)
        """
        exprs = []
        temp_join_cols = []
        
        IDX_MISSING = -1
        IDX_OTHER   = -2
        IDX_SPECIAL_START = -3

        if return_type == "woe" and not self.bin_woes_:
            self._materialize_woe()
            
        # 获取 Schema
        schema_map = X.collect_schema() if isinstance(X, pl.LazyFrame) else X.schema
        current_columns = schema_map.names()

        # 遍历所有已训练的列 (Numeric + Categorical)
        for col in self.bin_cuts_.keys() | self.cat_cuts_.keys():
            if col not in current_columns: continue
            
            # --- [关键] 计算安全值 ---
            col_dtype = schema_map[col]
            safe_missing_vals = self._get_safe_values(col_dtype, self.missing_values)
            safe_special_vals = self._get_safe_values(col_dtype, self.special_values)
            is_numeric_col = col_dtype in self.NUMERIC_DTYPES

            # =====================================================
            # Part A: 数值型特征
            # =====================================================
            if col in self.bin_cuts_:
                cuts = self.bin_cuts_[col]
                col_mapping = {IDX_MISSING: "Missing", IDX_OTHER: "Other"}
                
                # 缺失值
                missing_cond = pl.col(col).is_null() 
                if is_numeric_col: missing_cond |= pl.col(col).cast(pl.Float64).is_nan()
                for val in safe_missing_vals: 
                    missing_cond |= (pl.col(col) == val)
                
                layer_missing = pl.when(missing_cond).then(pl.lit(IDX_MISSING, dtype=pl.Int16))
                
                # 正常分箱
                breaks = cuts[1:-1] if len(cuts) > 2 else []
                if not breaks:
                    col_mapping[0] = "00_[-inf, inf)"
                    layer_normal = pl.lit(0, dtype=pl.Int16)
                else:
                    for i in range(len(cuts) - 1):
                        low, high = cuts[i], cuts[i+1]
                        col_mapping[i] = f"{i:02d}_[{low:.3g}, {high:.3g})"
                    bin_labels = [str(i) for i in range(len(breaks) + 1)]
                    layer_normal = pl.col(col).cut(
                        breaks, labels=bin_labels, left_closed=True
                    ).cast(pl.Int16)

                # 特殊值
                current_branch = layer_normal
                if safe_special_vals:
                    for i in range(len(safe_special_vals) - 1, -1, -1):
                        val = safe_special_vals[i]
                        idx = IDX_SPECIAL_START - i
                        col_mapping[idx] = f"Special_{val}"
                        current_branch = pl.when(pl.col(col) == val).then(pl.lit(idx, dtype=pl.Int16)).otherwise(current_branch)
                
                final_idx_expr = layer_missing.otherwise(current_branch)
                self.bin_mappings_[col] = col_mapping

                if return_type == "index":
                    exprs.append(final_idx_expr.alias(f"{col}_bin"))
                elif return_type == "woe":
                    woe_map = self.bin_woes_.get(col, {})
                    exprs.append(final_idx_expr.replace(woe_map).cast(pl.Float64).alias(f"{col}_woe") if woe_map else pl.lit(0.0).alias(f"{col}_woe"))
                else:
                    str_map = {str(k): v for k, v in col_mapping.items()}
                    exprs.append(final_idx_expr.cast(pl.Utf8).replace(str_map).alias(f"{col}_bin"))

            # =====================================================
            # Part B: 类别型特征
            # =====================================================
            elif col in self.cat_cuts_:
                splits = self.cat_cuts_[col]
                cat_to_idx = {}
                idx_to_label = {IDX_MISSING: "Missing", IDX_OTHER: "Other"}
                
                if safe_special_vals:
                    for i, val in enumerate(safe_special_vals):
                        idx_to_label[IDX_SPECIAL_START - i] = f"Special_{val}"

                for i, group in enumerate(splits):
                    disp_grp = group[:3] if len(group) > 3 else group
                    suffix = ",..." if len(group) > 3 else ""
                    label = f"{i:02d}_[{','.join(str(g) for g in disp_grp) + suffix}]"
                    idx_to_label[i] = label
                    for val in group:
                        cat_to_idx[str(val)] = i
                
                self.bin_mappings_[col] = idx_to_label
                target_col = pl.col(col).cast(pl.Utf8) 
                
                # 缺失值
                missing_cond = target_col.is_null()
                for val in safe_missing_vals: 
                    missing_cond |= (target_col == str(val))
                layer_missing = pl.when(missing_cond).then(pl.lit(IDX_MISSING, dtype=pl.Int16))
                
                # 特殊值
                current_branch = pl.lit(IDX_OTHER, dtype=pl.Int16)
                if safe_special_vals:
                    for i in range(len(safe_special_vals) - 1, -1, -1):
                        val = safe_special_vals[i]
                        idx = IDX_SPECIAL_START - i
                        current_branch = pl.when(target_col == str(val)).then(pl.lit(idx, dtype=pl.Int16)).otherwise(current_branch)

                # 路由: Join vs Replace
                if len(cat_to_idx) > self.join_threshold:
                    map_df = pl.DataFrame({
                        "_k": list(cat_to_idx.keys()), 
                        f"_idx_{col}": list(cat_to_idx.values())
                    }).with_columns([
                        pl.col("_k").cast(pl.Utf8),
                        pl.col(f"_idx_{col}").cast(pl.Int16)
                    ])
                    join_tbl = map_df.lazy() if isinstance(X, pl.LazyFrame) else map_df
                    X = X.join(join_tbl, left_on=target_col, right_on="_k", how="left")
                    temp_join_cols.append(f"_idx_{col}")
                    layer_normal = pl.col(f"_idx_{col}")
                else:
                    layer_normal = target_col.replace(cat_to_idx, default=None).cast(pl.Int16)
                
                final_idx_expr = layer_missing.otherwise(
                    pl.when(layer_normal.is_not_null()).then(layer_normal).otherwise(current_branch)
                )

                if return_type == "index":
                    exprs.append(final_idx_expr.alias(f"{col}_bin"))
                elif return_type == "woe":
                    woe_map = self.bin_woes_.get(col, {})
                    exprs.append(final_idx_expr.replace(woe_map).cast(pl.Float64).alias(f"{col}_woe") if woe_map else pl.lit(0.0).alias(f"{col}_woe"))
                else:
                    str_map = {str(k): v for k, v in idx_to_label.items()}
                    exprs.append(final_idx_expr.cast(pl.Utf8).replace(str_map).alias(f"{col}_bin"))

        return X.with_columns(exprs).drop(temp_join_cols)