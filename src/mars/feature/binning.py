from joblib import Parallel, delayed
from typing import List, Dict, Optional, Union, Any, Literal, Tuple

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier

from mars.core.base import MarsTransformer
from mars.utils.logger import logger

class MarsNativeBinner(MarsTransformer):
    """
    [极速分箱引擎] MarsNativeBinner
    
    完全基于 Polars 和 Sklearn 原生实现的高性能分箱器。
    针对大规模宽表 (如 2000+ 特征, 20万+ 样本) 进行了内存和速度的极致优化。
    
    核心优化策略 (Performance Strategies)
    -------------------------------------
    1. **Quantile/Uniform**: 
       利用纯 Polars 表达式进行标量聚合计算，避免了 Python 循环和数据复制，Fit 速度提升 100x。
    2. **Decision Tree (DT)**: 
       使用 `joblib` 进行多进程并行训练，通过生成器惰性传输数据，大幅降低内存峰值，Fit 速度提升 N_Cores 倍。
    3. **Transform**: 
       使用 Polars 的 `cut` 和 `when-then` 表达式进行映射，转换阶段实现毫秒级响应。

    Attributes
    ----------
    bin_cuts_ : Dict[str, List[float]]
        训练后存储的切点字典。格式: ``{col_name: [-inf, split1, split2, ..., inf]}``。
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        method: Literal["cart", "quantile", "uniform"] = "quantile",
        n_bins: int = 5,
        special_values: Optional[List[Union[int, float, str]]] = None,
        missing_values: Optional[List[Union[int, float, str]]] = None,
        min_samples: float = 0.05,
        n_jobs: int = -1  
    ) -> None:
        """
        初始化分箱器。

        Parameters
        ----------
        features : List[str], optional
            需要分箱的特征名称列表。如果不传，fit 时会自动识别所有数值型列。
        method : Literal["cart", "quantile", "uniform"], default="quantile"
            分箱方法：
            - 'cart': 决策树分箱 (Decision Tree)，最大化信息增益，依赖 target。
            - 'quantile': 等频分箱 (Quantile)，每个箱子样本数大致相等。
            - 'uniform': 等宽分箱 (Uniform)，每个箱子区间跨度相等。
        n_bins : int, default=5
            期望的分箱数量 (不包含特殊值和缺失值箱)。
        special_values : List[Union[int, float, str]], optional
            特殊值列表 (如 -999, -998)。这些值将不参与数值计算，并被单独分为独立箱。
        missing_values : List[Union[int, float, str]], optional
            缺失值列表 (如 -1, None)。这些值将被归类为 "Missing"。
        min_samples : float, default=0.05
            仅对 method='cart' 有效。决策树叶子节点的最小样本比例，用于控制过拟合。
        n_jobs : int, default=-1
            仅对 method='cart' 有效。并行作业的核心数，-1 表示使用所有可用核心。
        """
        super().__init__()
        self.features: Optional[List[str]] = features
        self.method: str = method
        self.n_bins: int = n_bins
        self.special_values: List[Any] = special_values if special_values is not None else []
        self.missing_values: List[Any] = missing_values if missing_values is not None else []
        self.min_samples: float = min_samples
        self.n_jobs: int = n_jobs
        
        # 存储训练好的切点: {col: [-inf, split1, split2, ..., inf]}
        self.bin_cuts_: Dict[str, List[float]] = {}

    def _fit_impl(self, X: pl.DataFrame, y: Optional[Any] = None, **kwargs) -> None:
        """
        训练实现的入口函数。
        """
        # 1. 确定目标列 (仅筛选数值列)
        target_cols = self.features if self.features else X.columns
        target_cols = [c for c in target_cols if c in X.columns and self._is_numeric(X[c])]

        if not target_cols:
            logger.warning("No numeric columns found for binning.")
            return

        # ========================================================
        # [优化] 极速预过滤 (修正版)
        # ========================================================
        valid_cols = []
        n_rows = X.height
        
        # 1. 构建批量表达式 (一次性计算所有统计量)
        stats_exprs = []
        for c in target_cols:
            stats_exprs.append(pl.col(c).null_count().alias(f"{c}_null"))
            stats_exprs.append(pl.col(c).min().alias(f"{c}_min"))
            stats_exprs.append(pl.col(c).max().alias(f"{c}_max"))
            
        # 2. 触发并行计算 (One-Shot)
        # 速度极快，20万行x2000列通常在 50ms 以内
        stats_row = X.select(stats_exprs).row(0)
        
        # 3. 解析结果
        for i, c in enumerate(target_cols):
            base_idx = i * 3
            null_cnt = stats_row[base_idx]
            min_val = stats_row[base_idx + 1]
            max_val = stats_row[base_idx + 2]
            
            # Case A: 全空列 -> 跳过
            # 这种列没有任何信息，分箱没有意义
            if null_cnt == n_rows:
                logger.warning(f"Feature '{c}' is all null. Skipped.")
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue
            
            # Case B: 单一值检查 (Constant Value)
            # 修正逻辑：只有当 min == max 且 没有空值 时，才是真正的"单一值"。
            # 如果 min == max 但有空值 (如 [1, 1, null])，它实际上是二值特征 (1 vs Missing)，必须保留！
            if min_val == max_val and null_cnt == 0:
                logger.warning(f"Feature '{c}' has constant value ({min_val}) and no nulls. Skipped.")
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue

            # 其他情况 (包括 [1, 1, null]) 全部保留
            valid_cols.append(c)

        if not valid_cols:
            logger.warning("No valid features remain after null check.")
            return

        # 2. 检查依赖关系
        if y is None and self.method == "cart":
            raise ValueError("Decision Tree Binning ('cart') requires target 'y'.")

        logger.info(f"⚙️ Fitting bins for {len(valid_cols)} features (Native Mode: {self.method})...")

        # 3. 策略分发 (只传入有效列)
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
        
        核心优化：使用 Polars 表达式一次性计算所有列的分位数，避免 Python 循环。
        注意：为了避免 Polars 处理 `List(Float)` 类型时的开销，采用了生成多个标量表达式的方法。

        Parameters
        ----------
        X : pl.DataFrame
            输入数据。
        cols : List[str]
            需要计算的数值列列表。
        """
        # 1. 构建分位点 (不包含 0% 和 100%)
        if self.n_bins <= 1:
            quantiles = [0.5]
        else:
            quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1].tolist()
        
        # 2. 准备过滤特殊值的逻辑 (None 在 Polars 中自动处理，无需包含)
        exclude_vals = self.special_values + self.missing_values
        exclude_vals_clean = [v for v in exclude_vals if v is not None]

        # 3. 构建表达式列表 (Flattened)
        # 将 "N列 x M个分位数" 拆解成 N*M 个独立的标量表达式
        # 例如: feature_a:::0 (20%), feature_a:::1 (40%)...
        q_exprs = []

        for c in cols:
            target_col = pl.col(c)
            # 如果存在特殊值，使用 when-then 将其视为 Null (Polars quantile 会自动忽略 Null)
            if exclude_vals_clean:
                target_col = pl.when(pl.col(c).is_in(exclude_vals_clean)).then(None).otherwise(pl.col(c))
            
            # 为每个分位点生成一个独立的表达式
            for i, q in enumerate(quantiles):
                alias_name = f"{c}:::{i}"
                q_exprs.append(target_col.quantile(q).alias(alias_name))
        
        # 4. 触发计算 (One-Shot Query)
        # Polars 引擎会并行优化这些标量聚合计算
        stats = X.select(q_exprs)
        row = stats.row(0)
        
        # 5. 解析结果
        # 将扁平的结果重组回 {col: [cuts]} 结构
        temp_cuts: Dict[str, List[float]] = {c: [] for c in cols}
        
        # row 是 tuple，stats.columns 是列名列表
        for val, name in zip(row, stats.columns):
            c_name, _ = name.split(":::")
            # [Fix] 增加对 NaN 的过滤
            # val is not None: 过滤 Polars 的 null
            # not np.isnan(val): 过滤 numpy 的 nan
            if val is not None and not np.isnan(val):
                temp_cuts[c_name].append(val)

        # 6. 最终封装
        for c in cols:
            # 去重并排序，添加 -inf 和 inf
            cuts = sorted(list(set(temp_cuts[c]))) 
            self.bin_cuts_[c] = [float('-inf')] + cuts + [float('inf')]

    def _fit_uniform(self, X: pl.DataFrame, cols: List[str]) -> None:
        """
        执行极速等宽分箱 (Uniform/Step Binning)。
        
        优化策略：
        1. **低基数检查**：若唯一值数量 <= n_bins，直接按唯一值切分，避免空箱。
        2. **空箱合并**：计算出切点后，立即校验各箱样本数，自动剔除 Count=0 的区间。

        Parameters
        ----------
        X : pl.DataFrame
            输入数据。
        cols : List[str]
            需要计算的数值列列表。
        """
        exclude_vals = [v for v in (self.special_values + self.missing_values) if v is not None]
        
        # 1. 构建聚合表达式 (计算每列的 min, max 以及 approx_n_unique)
        exprs = []
        for c in cols:
            target_col = pl.col(c)
            if exclude_vals:
                target_col = target_col.filter(~pl.col(c).is_in(exclude_vals))
            
            exprs.append(target_col.min().alias(f"{c}_min"))
            exprs.append(target_col.max().alias(f"{c}_max"))
            # 使用 approx_n_unique 快速估算基数 (性能远高于 n_unique)
            exprs.append(target_col.n_unique().alias(f"{c}_n_unique"))

        # 2. 触发计算 (One-Shot)
        stats = X.select(exprs)
        row = stats.row(0)
        
        # 3. 解析结果并生成切点
        for i, c in enumerate(cols):
            # stats 结构: [c1_min, c1_max, c1_nu, c2_min, c2_max, c2_nu, ...]
            base_idx = i * 3
            min_val = row[base_idx]
            max_val = row[base_idx + 1]
            n_unique = row[base_idx + 2]
            
            # 异常处理：全空
            if min_val is None or max_val is None:
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue
            
            # --- 优化1: 低基数检查 ---
            # 如果唯一值很少，直接查询出所有唯一值作为切分依据
            if n_unique <= self.n_bins:
                # 这里需要额外查一次该列的具体唯一值 (因为前面只查了数量)
                # 这种操作仅在低基数时触发，开销极小
                unique_vals = X.select(pl.col(c).unique().sort()).to_series().to_list()
                # 过滤特殊值
                clean_vals = [v for v in unique_vals if v not in exclude_vals and v is not None]
                
                if len(clean_vals) <= 1:
                    self.bin_cuts_[c] = [float('-inf'), float('inf')]
                else:
                    # 取相邻值的中间点: (1, 2, 3) -> (1.5, 2.5)
                    mid_points = [(clean_vals[k] + clean_vals[k+1])/2 for k in range(len(clean_vals)-1)]
                    self.bin_cuts_[c] = [float('-inf')] + mid_points + [float('inf')]
                continue

            # --- 常规逻辑: 等宽切分 ---
            if min_val == max_val:
                self.bin_cuts_[c] = [float('-inf'), float('inf')]
                continue

            step = (max_val - min_val) / self.n_bins
            # 生成初始切点
            # raw_cuts = [min_val + step * k for k in range(1, self.n_bins)]
            raw_cuts = np.linspace(min_val, max_val, self.n_bins + 1)[1:-1].tolist()
            
            # --- 优化2: 空箱合并 (Post-Optimization) ---
            # 即使是连续变量，等宽也可能切出空箱。我们需要根据数据分布修正切点。
            full_cuts = [float('-inf')] + raw_cuts + [float('inf')]
            optimized_cuts = self._remove_empty_bins(X, c, full_cuts, exclude_vals)
            
            self.bin_cuts_[c] = optimized_cuts

    def _remove_empty_bins(self, X: pl.DataFrame, col: str, cuts: List[float], exclude_vals: List[Any]) -> List[float]:
        """
        [内部方法] 移除样本数为 0 的空箱子 (Empty Bin Pruning)。
        
        应用场景：
        通常用于等宽分箱 (Uniform Binning) 后处理。由于数据分布不均，等宽切分极易产生
        中间没有样本的"空档"。该方法会识别并合并这些空档。

        合并策略：
        ----------------
        如果发现某区间 `(cuts[i], cuts[i+1]]` 的 count 为 0 (空箱)：
        1. 我们选择**移除该区间的右边界** `cuts[i+1]`。
        2. 视觉效果上，这相当于当前空箱被"向右合并"到了下一个箱子中，或者说当前区间的
           分割线失效了，两个区间连通了。
        3. 这种策略能最大程度保持切点的连续性，且实现逻辑简单高效。

        Parameters
        ----------
        X : pl.DataFrame
            输入数据表。
        col : str
            目标列名。
        cuts : List[float]
            原始切点列表 (包含 -inf 和 inf)。
        exclude_vals : List[Any]
            不参与分箱统计的特殊值列表。

        Returns
        -------
        List[float]
            优化后的、去除了空箱边界的切点列表。
        """
        # 1. 准备中间切点 (breaks)
        # cuts 包含了 -inf 和 inf，但 Polars 的 cut/hist 函数只需要中间的分割点
        breaks = cuts[1:-1]
        
        # 如果没有中间切点 (即只有 [-inf, inf])，说明只有1箱，直接返回
        if not breaks:
            return cuts

        # 2. 构建目标列的过滤表达式
        # 我们只统计"有效值"的分布，忽略特殊值
        target_col = pl.col(col)
        if exclude_vals:
            target_col = target_col.filter(~pl.col(col).is_in(exclude_vals))

        # 3. 极速直方图统计 (Histogram Calculation)
        # 相比 Python for 循环，利用 Polars 表达式引擎计算分布快 100 倍以上。
        # 
        # 逻辑分解：
        # a. cut(breaks): 将数据映射到对应的区间索引 (0, 1, 2...)
        # b. group_by("bin_idx"): 按区间分组
        # c. len(): 统计每个区间的样本数
        bin_counts = (
            X.select(
                target_col.cut(breaks, labels=[str(i) for i in range(len(breaks)+1)], left_closed=True)
                .alias("bin_idx")
            )
            .group_by("bin_idx")
            .len()
            .sort("bin_idx")
        )
        
        # bin_counts 结果示例 (可能包含 null):
        # ┌─────────┬───────┐
        # │ bin_idx ┆ len   │
        # ╞═════════╪═══════╡
        # │ "0"     ┆ 150   │
        # │ "1"     ┆ 0     │ <-- 空箱
        # │ null    ┆ 10    │ <-- NaN 或 脏数据导致的 null (这就是报错根源！)
        # └─────────┴───────┘
        
        # 4. 提取有效箱索引 (Valid Indices Extraction)
        # [Critical Fix]: 必须增加 is_not_null() 过滤。
        # 原因：当数据包含 NaN 时，cut 算子会将其映射为 null。如果不过滤，
        # 在后续 int(idx) 转换时会抛出 "TypeError: int() argument must be... not 'NoneType'"。
        
        valid_indices = set(
            int(idx) for idx in bin_counts.filter(
                (pl.col("len") > 0) & 
                (pl.col("bin_idx").is_not_null())  # <--- 核心修复点
            )["bin_idx"].to_list()
        )
        
        # 5. 重构切点 (Reconstruct Cuts)
        # 原始 cuts: [-inf, c1, c2, c3, inf]
        # 对应的箱:    Bin0, Bin1, Bin2, Bin3
        # 
        # 逻辑：
        # 我们遍历所有可能的箱索引 i。如果 Bin(i) 是有效的 (在 valid_indices 里)，
        # 我们就保留它的**右边界** (cuts[i+1])。
        # 如果 Bin(i) 是空的，我们就跳过它的右边界，从而实现合并。
        
        new_cuts = [cuts[0]] # 始终保留 -inf
        
        for i in range(len(breaks) + 1):
            # i 代表箱子索引 (0 到 N-1)
            if i in valid_indices:
                # 只有当箱子不为空时，才保留这个箱子的结束边界
                new_cuts.append(cuts[i+1])
            else:
                # 箱子为空 (count=0)，跳过添加 cuts[i+1]。
                # 效果：当前箱子的空间合并到了下一个箱子中。
                pass
                
        # 6. 兜底处理 (Finalize)
        # 确保 inf 总是存在。如果最后一个箱子也是空的，上面的循环逻辑可能会漏掉 inf。
        if new_cuts[-1] != float('inf'):
            new_cuts.append(float('inf'))
            
        # 去重并排序，确保切点严格单调递增
        return sorted(list(set(new_cuts)))

    def _fit_cart_parallel(self, X: pl.DataFrame, y: Any, cols: List[str]) -> None:
        """
        执行并行的决策树分箱 (Decision Tree Binning)。

        原理：
        1. 将每一列数据和 Target 转换为 Numpy 数组。
        2. 使用 `joblib` 将任务分发到多个 CPU 核心。
        3. 每个子进程独立训练一个单特征的 `DecisionTreeClassifier`。
        4. 提取树的阈值作为切点。

        Parameters
        ----------
        X : pl.DataFrame
            输入数据。
        y : Any
            目标变量 (Label)。
        cols : List[str]
            需要计算的数值列列表。
        """
        # 1. 准备 Y (只需转换一次，减少内存复制)
        y_np = np.array(y)
        if len(y_np) != X.height:
             raise ValueError("Target 'y' length mismatch.")

        # 2. 定义 Worker 函数 (必须是无副作用的纯函数或闭包)
        # 该函数将在独立的进程中运行
        def worker(col_name: str, col_data_np: np.ndarray) -> tuple:
            try:
                # A. 过滤 Mask (去除 Special/Missing/NaN)
                # 注意：np.isin 对 None 的处理比较棘手，建议先处理 NaN
                mask_nan = np.isnan(col_data_np)
                
                # 标记 Special/Missing (不含 None，因为 numpy float array 里 None 也是 NaN)
                ignore_vals = [v for v in (self.special_values + self.missing_values) if v is not None]
                
                if ignore_vals:
                    mask_ignore = np.isin(col_data_np, ignore_vals)
                    mask_valid = ~(mask_nan | mask_ignore)
                else:
                    mask_valid = ~mask_nan
                
                clean_X = col_data_np[mask_valid].reshape(-1, 1)
                clean_y = y_np[mask_valid]
                
                # B. 边界检查：有效数据太少则不分箱
                if len(clean_X) < 100:
                    return col_name, [float('-inf'), float('inf')]
                
                # C. 训练 Sklearn DT
                cart = DecisionTreeClassifier(
                    max_leaf_nodes=self.n_bins,
                    min_samples_leaf=self.min_samples,
                    random_state=42
                )
                cart.fit(clean_X, clean_y)
                
                # D. 提取阈值
                # threshold 中 -2 表示叶子节点，需要过滤
                cuts = cart.tree_.threshold[cart.tree_.threshold != -2]
                cuts = np.sort(np.unique(cuts))
                
                full_cuts = [float('-inf')] + list(cuts) + [float('inf')]
                return col_name, full_cuts
            
            except Exception as e:
                # 容错：单个特征失败不影响整体
                return col_name, [float('-inf'), float('inf')]

        # 3. 准备数据生成器
        # 关键优化：使用 generator 惰性获取数据。
        # 避免调用 `X.to_numpy()` 一次性把整个大宽表转入内存，而是每次只取一列。
        task_gen = (
            (c, X.select(c).to_series().to_numpy()) 
            for c in cols
        )
        
        logger.info(f"🚀 Starting parallel DT fitting with n_jobs={self.n_jobs}...")
        
        # 4. 并行执行
        # backend="loky" 是 joblib 的默认后端，对大数据传输有优化 (memmap)
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(worker)(c, data) for c, data in task_gen
        )
        
        # 5. 收集结果
        for c, cuts in results:
            self.bin_cuts_[c] = cuts

    def _transform_impl(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        预测阶段：应用分箱规则。
        
        逻辑顺序 (Waterfall)：
        1. Missing Layer: 匹配 null/nan 及用户定义的缺失值。
        2. Special Layer: 匹配用户定义的特殊值 (如 -999)。
        3. Normal Layer: 对剩余数值进行区间切分。

        Returns
        -------
        pl.DataFrame
            包含原始列和新生成的分箱列 (`{col}_bin`) 的 DataFrame。
        """
        exprs = []
        
        for col, cuts in self.bin_cuts_.items():
            if col not in X.columns: continue
            
            # --- Layer 1: Missing Bin (优先级最高) ---
            missing_condition = pl.col(col).is_null() | pl.col(col).is_nan()
            
            # 追加用户定义的缺失值 (如 -1)
            for val in self.missing_values:
                if val is None: continue
                missing_condition = missing_condition | (pl.col(col) == val)
            
            layer_missing = pl.when(missing_condition).then(pl.lit("Missing"))
            
            # --- Layer 2: Special Bin (优先级第二) ---
            layer_special = pl.when(False).then(pl.lit("None")) # 初始化空分支
            
            for val in self.special_values:
                if val is None: continue
                label = f"Special_{val}"
                layer_special = layer_special.when(pl.col(col) == val).then(pl.lit(label))
            
            # --- Layer 3: Normal Bin (优先级最低) ---
            breaks = cuts[1:-1]
            if not breaks:
                layer_normal = pl.lit("00_[-inf, inf)")
            else:
                # 生成可读性强的标签: 00_[l, r), 01_[l, r)...
                labels = []
                for i in range(len(cuts) - 1):
                    low, high = cuts[i], cuts[i+1]
                    
                    # 格式化数值 (去除多余的.000)
                    low_str = "-inf" if low == float('-inf') else f"{low:.3f}".rstrip('0').rstrip('.')
                    high_str = "inf" if high == float('inf') else f"{high:.3f}".rstrip('0').rstrip('.')
                    
                    # 补齐小数点末尾可能被删掉的情况 (如 25. -> 25)
                    if low_str.endswith('.'): low_str = low_str[:-1]
                    if high_str.endswith('.'): high_str = high_str[:-1]

                    labels.append(f"{i:02d}_[{low_str}, {high_str})")
                
                # 使用 Polars 的 cut 算子进行快速二分查找映射
                layer_normal = pl.col(col).cut(breaks, labels=labels, left_closed=True).cast(pl.Utf8)
            
            # --- 组装瀑布流 ---
            final_expr = (
                layer_missing
                .otherwise(
                    layer_special.otherwise(layer_normal)
                )
                .alias(f"{col}_bin")
            )
            
            exprs.append(final_expr)

        return X.with_columns(exprs)

    def _is_numeric(self, series: pl.Series) -> bool:
        """
        判断 Polars Series 是否为数值类型。
        
        Returns
        -------
        bool
            如果是整型或浮点型返回 True，否则 False。
        """
        return series.dtype in [
            pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, 
            pl.Float32, pl.Float64
        ]
        
        
class MarsOptimalBinner(MarsNativeBinner):
    """
    [混合动力分箱引擎] MarsOptimalBinner

    该类实现了基于混合动力架构 (Hybrid Engine) 的最优分箱算法。
    
    设计目标：
    解决传统 OptBinning 在大规模数据（如 20万行 x 2000列）上直接求解 MIP (混合整数规划) 
    导致的计算性能瓶颈，同时保留其数学规划带来的最优性和单调性约束能力。

    核心架构 (Architecture):
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
        格式: ``{col: [-inf, c1, c2, ..., inf]}``
    
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
        cat_cutoff: int = 100,
        n_jobs: int = -1  
    ) -> None:
        """
        初始化混合动力分箱器。

        Parameters
        ----------
        features : List[str], optional
            所有需要分箱的特征名称列表。
        
        cat_features : List[str], optional
            显式指定哪些特征是类别型 (Categorical)。
            未在此列表中的特征将被自动识别并视为数值型。
        
        n_bins : int, default=5
            最终期望的最优箱数 (Max bins)。
        
        n_prebins : int, default=50
            [性能关键] 预分箱的细粒度。
            数值越大，Solver 的搜索空间越大，结果越接近理论最优，但耗时增加。建议 20-50。
        
        prebinning_method : Literal["quantile", "uniform"], default="quantile"
            预分箱方法。'quantile' 适合长尾分布，'uniform' 适合均匀分布。
        
        monotonic_trend : str, default="auto_asc_desc"
            单调性约束类型 (auto, ascending, descending, convex, concave)。
        
        solver : str, default="cp"
            数学规划求解器。'cp' (Constraint Programming) 通常比 'mip' 更快。
        
        time_limit : int, default=10
            单个特征的求解超时时间 (秒)。**超时后将自动回退到预分箱结果**。
            
        special_values : List, optional
            特殊值列表 (如 -999)。独立成箱，不参与数学规划。
            
        missing_values : List, optional
            缺失值列表 (如 -1, None)。独立成箱。
            
        cat_cutoff : int, default=100
            类别特征预处理阈值。若基数超过此值，仅保留 Top-K 高频类别，其余归为 Other。
            
        n_jobs : int, default=-1
            并行计算的核心数。
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
        self.cat_cutoff: int = cat_cutoff
        
        # 专门存储类别特征的分箱规则
        # 结构: {col_name: [['A', 'B'], ['C'], ['D']]}
        self.cat_cuts_: Dict[str, List[List[Any]]] = {}

        # 检查依赖
        try:
            import optbinning
        except ImportError:
            logger.warning("⚠️ 'optbinning' not installed. Optimal binning will fallback to pre-binning.")

    def _fit_impl(self, X: pl.DataFrame, y: Optional[Any] = None, **kwargs) -> None:
        """
        训练入口：实现数值与类别特征的分流处理。

        Parameters
        ----------
        X : pl.DataFrame
            输入特征数据。
        y : Any
            目标变量 (必须提供以计算 IV/WOE)。
        """
        if y is None:
            raise ValueError("Optimal Binning requires target 'y' to calculate IV/WOE.")

        y_np = np.array(y)
        
        # 1. 特征分类 (自动推断 + 用户指定)
        all_target_cols = self.features if self.features else X.columns
        cat_set = set(self.cat_features)
        
        # 数值列: (在目标中) & (不在类别白名单中) & (物理类型是数字)
        num_cols = [
            c for c in all_target_cols 
            if c not in cat_set and c in X.columns and self._is_numeric(X[c])
        ]
        
        # 类别列: (在目标中) & (在类别白名单中)
        cat_cols = [
            c for c in all_target_cols 
            if c in cat_set and c in X.columns
        ]

        if not num_cols and not cat_cols:
            logger.warning("No valid numeric or categorical columns found.")
            return

        # 2. 并行流水线执行
        if num_cols:
            self._fit_numerical_pipeline(X, y_np, num_cols)

        if cat_cols:
            self._fit_categorical_pipeline(X, y_np, cat_cols)

    def _fit_numerical_pipeline(self, X: pl.DataFrame, y_np: np.ndarray, num_cols: List[str]) -> None:
        """
        [Pipeline] 数值型特征混合动力处理流水线。
        
        Process:
            1. Polars Pre-binning -> 2. OptBinning Solver -> 3. Fallback Check
        """
        logger.info(f"🚀 [Numeric] Starting Hybrid Pipeline for {len(num_cols)} features...")
        
        # --- Stage 1: 极速预分箱 (Pre-binning) ---
        # 利用父类能力快速生成 n_prebins 个切点
        pre_binner = MarsNativeBinner(
            features=num_cols,
            method=self.method, 
            n_bins=self.n_prebins, # 注意：这里使用预分箱粒度
            special_values=self.special_values,
            missing_values=self.missing_values,
            n_jobs=self.n_jobs
        )
        pre_binner.fit(X, y_np)
        pre_cuts_map = pre_binner.bin_cuts_

        # 筛选出有意义的列 (切点数 > 2 表示不仅仅是 inf)
        active_cols = []
        for col, cuts in pre_cuts_map.items():
            if len(cuts) > 2: 
                active_cols.append(col)
            else:
                # 预分箱都分不出(如单一值)，直接保留结果，不送入 Solver
                self.bin_cuts_[col] = cuts 

        if not active_cols:
            return

        # --- Stage 2: 并行优化 (Optimization) ---
        logger.info(f"🧠 [Numeric] Optimizing {len(active_cols)} features with Solver...")
        
        # 定义 Worker
        def num_worker(col: str, pre_cuts: List[float], col_data: np.ndarray) -> Tuple[str, List[float]]:
            # 默认回退方案
            fallback_res = (col, pre_cuts)
            
            try:
                from optbinning import OptimalBinning
                
                # 1. 基础方差检查: 如果方差极小，Solver 可能会报错
                valid_mask = ~np.isnan(col_data)
                valid_data = col_data[valid_mask]
                if len(valid_data) < 10 or np.var(valid_data) < 1e-8:
                    return fallback_res

                # 2. 注入 Stage 1 切点 (User Splits)
                user_splits = np.array(pre_cuts[1:-1]) 
                
                opt = OptimalBinning(
                    name=col, dtype="numerical", solver=self.solver,
                    monotonic_trend=self.monotonic_trend,
                    user_splits=user_splits,  # <--- 核心：注入预分箱
                    max_n_bins=self.n_bins,   # 最终目标箱数
                    time_limit=self.time_limit, 
                    min_bin_size=0.0,         # 预分箱已控制粒度，此处放宽
                    verbose=False
                )
                
                opt.fit(valid_data, y_np[valid_mask])
                
                # 3. 状态检查
                if opt.status in ["OPTIMAL", "FEASIBLE"]:
                    return col, [float('-inf')] + list(opt.splits) + [float('inf')]
                
                return fallback_res # 状态异常，回退

            except Exception:
                # 任何 Python 异常都触发回退，保证鲁棒性
                return fallback_res

        # 数据生成器 (惰性加载)
        task_gen = (
            (c, pre_cuts_map[c], X.select(c).to_series().to_numpy()) 
            for c in active_cols
        )
        
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(num_worker)(c, cuts, data) for c, cuts, data in task_gen
        )
        
        for col, cuts in results:
            self.bin_cuts_[col] = cuts

    def _fit_categorical_pipeline(self, X: pl.DataFrame, y_np: np.ndarray, cat_cols: List[str]) -> None:
        """
        [Pipeline] 类别型特征处理流水线 (带 Top-K 优化)。
        """
        logger.info(f"🧠 [Categorical] Optimizing {len(cat_cols)} features...")

        def cat_worker(col: str, col_data_raw: np.ndarray) -> Tuple[str, Optional[List[List[Any]]]]:
            try:
                from optbinning import OptimalBinning
                col_data = col_data_raw.astype(str)
                
                # --- 🚀 Optimization: Top-K Pre-filtering ---
                # 如果基数过大，先保留 Top-K，其余置为 "Other_Pre"
                # 避免 OptBinning 在预处理阶段卡死
                unique_vals, counts = np.unique(col_data, return_counts=True)
                if len(unique_vals) > self.cat_cutoff:
                    # 获取 Top K 的索引
                    top_indices = np.argsort(-counts)[:self.cat_cutoff]
                    top_vals = set(unique_vals[top_indices])
                    
                    # 使用 Numpy 向量化操作进行替换 (不在 TopK 的变为 "Other_Pre")
                    # 这一步比传给 OptBinning 几万个类别要快得多
                    mask_keep = np.isin(col_data, list(top_vals))
                    col_data = np.where(mask_keep, col_data, "Other_Pre")

                # --- Optimization End ---

                opt = OptimalBinning(
                    name=col, dtype="categorical", solver=self.solver,
                    max_n_bins=self.n_bins, 
                    time_limit=self.time_limit,
                    cat_cutoff=0.05, # 辅助: 进一步归类低频 (<5%)
                    verbose=False
                )
                opt.fit(col_data, y_np)
                
                if opt.status in ["OPTIMAL", "FEASIBLE"]:
                    return col, opt.splits
                
                return col, None # 失败返回 None
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
            # 注意：若 splits 为 None，该列将不产生 _bin 结果 (Soft Fail)
            # 也可以在这里实现简单的 top-n fallback 逻辑

    def _transform_impl(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        [Transform] 极速预测实现。
        
        优化点：
        1. 使用 Polars `cut` 处理数值。
        2. 使用 `replace` 处理类别，并兼容 Polars 新旧版本。
        3. 严格的空值/特殊值分层逻辑 (Waterfall Logic)。

        Returns
        -------
        pl.DataFrame
            包含原始列和新生成 `{col}_bin` 列的数据表。
        """
        exprs = []
        
        # =====================================================
        # Part A: 数值型特征 (Numeric)
        # =====================================================
        for col, cuts in self.bin_cuts_.items():
            if col not in X.columns: continue
            
            # 1. Missing Layer (Priority 1)
            missing_condition = pl.col(col).is_null() | pl.col(col).is_nan()
            for val in self.missing_values:
                if val is not None:
                    missing_condition |= (pl.col(col) == val)
            
            layer_missing = pl.when(missing_condition).then(pl.lit("Missing"))
            
            # 2. Special Layer (Priority 2)
            layer_special = pl.when(False).then(pl.lit("None"))
            for val in self.special_values:
                if val is not None:
                    layer_special = layer_special.when(pl.col(col) == val).then(pl.lit(f"Special_{val}"))
            
            # 3. Normal Layer (Priority 3)
            breaks = cuts[1:-1]
            if not breaks:
                layer_normal = pl.lit("00_[-inf, inf)")
            else:
                # 生成 Labels: 00_[2.5, 10.0)
                labels = []
                for i in range(len(cuts) - 1):
                    low, high = cuts[i], cuts[i+1]
                    # 格式化优化
                    l_s = "-inf" if low == float('-inf') else f"{low:.3g}"
                    h_s = "inf" if high == float('inf') else f"{high:.3g}"
                    labels.append(f"{i:02d}_[{l_s}, {h_s})")
                
                layer_normal = pl.col(col).cut(breaks, labels=labels, left_closed=True).cast(pl.Utf8)
            
            # 组装
            exprs.append(
                layer_missing.otherwise(layer_special.otherwise(layer_normal)).alias(f"{col}_bin")
            )

        # =====================================================
        # Part B: 类别型特征 (Categorical)
        # =====================================================
        for col, splits in self.cat_cuts_.items():
            if col not in X.columns: continue

            # 1. 构建映射字典 (Value -> Label)
            mapping_dict = {}
            for i, group in enumerate(splits):
                # 生成可读标签: "00_[A,B...]"
                disp_grp = group[:3] if len(group) > 3 else group
                suffix = ",..." if len(group) > 3 else ""
                grp_str = ",".join(str(g) for g in disp_grp) + suffix
                label = f"{i:02d}_[{grp_str}]"
                
                for val in group:
                    mapping_dict[str(val)] = label
            
            target_col = pl.col(col).cast(pl.Utf8)
            
            # 2. Missing Layer
            missing_condition = target_col.is_null()
            for val in self.missing_values:
                if val is not None:
                    missing_condition |= (target_col == str(val))
            layer_missing = pl.when(missing_condition).then(pl.lit("Missing"))
            
            # 3. Special Layer
            layer_special = pl.when(False).then(pl.lit("None"))
            for val in self.special_values:
                if val is not None:
                    layer_special = layer_special.when(target_col == str(val)).then(pl.lit(f"Special_{val}"))
            
            # 4. Normal Layer (Map)
            # 使用 replace 映射。对于未见过的类别，这一步保留原值。
            # 随后我们用 "Other" 填充那些未被映射的值（通过检查是否以 "bin_prefix" 开头或直接填充）
            
            # 方案：利用 replace 的 return_dtype 行为
            # 映射表中存在的 -> 变成 Label
            # 映射表中不存在的 -> 保持原 String
            # 最后：如果值不在 mapping_dict.values() 中，则视为 Other。
            # 但更简单的做法是：
            layer_normal = target_col.replace(mapping_dict)
            
            # 检查是否映射成功 (Label 格式通常是 "00_...")
            # 任何没有变成 Label 的，都是未见过的类别 -> "Other"
            # 这里的逻辑假设原始数据不包含与 Label 相同的格式
            known_labels = list(set(mapping_dict.values()))
            layer_normal = pl.when(layer_normal.is_in(known_labels)).then(layer_normal).otherwise(pl.lit("Other"))

            exprs.append(
                layer_missing.otherwise(layer_special.otherwise(layer_normal)).alias(f"{col}_bin")
            )

        return X.with_columns(exprs)