import numpy as np
import polars as pl
import pandas as pd
import dataclasses
from typing import List, Union, Optional, Any, Dict
from mars.utils.logger import logger
from mars.utils.decorators import time_it
from mars.core.base import MarsBaseEstimator
from mars.analysis.report import MarsProfileReport
from mars.core.config import MarsProfileConfig

class MarsDataProfiler(MarsBaseEstimator):
    """
    [全能版] MarsDataProfiler - 基于 Polars 的高性能多维数据体检工具。
    
    专为大规模风控建模数据集设计。它利用 Polars 的高性能计算引擎，
    不仅能进行基础的数据质量 (DQ) 体检，还能对特征的统计分布进行
    跨时间/跨客群的稳定性监控，并提供字符画形式的迷你分布图可视化。

    Attributes
    ----------
    df : pl.DataFrame
        内部存储的 Polars DataFrame。
    config : MarsProfileConfig
        全局配置对象，包含指标列表、报警阈值和可视化参数。
    custom_missing : List[Any]
        自定义缺失值列表 (如 -999, 'null')，在计算 missing_rate 时会被视为 Null。
    """

    def __init__(
        self, 
        df: Union[pl.DataFrame, pd.DataFrame], 
        config: Optional[MarsProfileConfig] = None,
        custom_missing_values: Optional[List[Union[int, float, str]]] = None
    ) -> None:
        """
        初始化数据分析器。

        Parameters
        ----------
        df : Union[pl.DataFrame, pd.DataFrame]
            输入数据集。会自动转换为 Polars 格式以利用其向量化计算优势。
        config : MarsProfileConfig, optional
            配置对象。如果为 None，则使用默认配置 (包含默认指标和阈值)。
        custom_missing_values : List[Union[int, float, str]], optional
            指定自定义缺失值列表。例如: [-999, "unknown", "\\N"]。
        """
        super().__init__()
        self.df: pl.DataFrame = self._ensure_polars(df)
        self.config: MarsProfileConfig = config if config else MarsProfileConfig()
        self.custom_missing: List[Any] = custom_missing_values if custom_missing_values else []

    @time_it
    def generate_profile(
        self, 
        profile_by: Optional[str] = None, 
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> MarsProfileReport:
        """
        [核心接口] 执行数据体检管道，生成完整分析报告。

        该方法会自动计算两类指标：
        1. **Overview (全量概览)**: 包含数据分布(Sparkline)、DQ指标、统计指标。不涉及分组。
        2. **Trends (分组趋势)**: 如果指定了 `profile_by`，会计算各项指标随该维度的变化。

        Parameters
        ----------
        profile_by : str, optional
            分组维度字段名 (如 'month', 'vintage')。
            
            * ``None``: 仅生成 Overview 和 Total 趋势列。
            * ``Str``: 生成分组透视表 (Pivot Table)。

        config_overrides : Dict[str, Any], optional
            临时覆盖 `MarsProfileConfig` 中的默认配置。支持以下配置项：

            **1. 计算范围 (Metrics)**
            
            * ``stat_metrics`` (List[str]): 需要计算的统计指标。
              可选值: "mean", "std", "min", "max", "median", "p25", "p75"。
            * ``dq_metrics`` (List[str]): 需要计算的数据质量指标。
              可选值: "missing", "zeros", "unique", "top1"。

            **2. 可视化 (Visualization)**
            
            * ``enable_sparkline`` (bool): 是否计算字符画形式的迷你分布图 (默认 True)。
              (建议在极大数据量或追求极致性能时设为 False)。
            * ``sparkline_sample_size`` (int): 计算分布图时的采样行数。
            * ``sparkline_bins`` (int): 分布图的分箱精度。

            **3. 报警阈值 (Thresholds)**
            
            * ``threshold_missing_danger`` (float): 缺失率报警阈值。
            * ``threshold_top1_danger`` (float): 单一值占比报警阈值。

        Returns
        -------
        MarsProfileReport
            包含概览表和趋势表的报告对象容器。

        Examples
        --------
        >>> # 场景 1: 仅计算均值和最大值，并关闭 Sparkline 以加速
        >>> profiler.generate_profile(
        ...     config_overrides={
        ...         "stat_metrics": ["mean", "max"], 
        ...         "enable_sparkline": False
        ...     }
        ... )
        
        >>> # 场景 2: 按月份分组分析
        >>> report = profiler.generate_profile(profile_by="month")
        """
        # 1. 动态配置合并 (只影响本次运行，不污染 self.config)
        run_config: MarsProfileConfig = self.config
        if config_overrides:
            run_config = dataclasses.replace(self.config, **config_overrides)

        logger.info(f"Starting profiling (Group: {profile_by}, Sparkline: {run_config.enable_sparkline})...")
        
        # 2. 计算全量概览 (Overview) - 核心基础表
        #    包含: Sparkline, DQ, Stats
        overview_df: pl.DataFrame = self._calculate_overview(run_config)

        # 3. 计算趋势表 (Trend Tables)
        #    利用 Polars 的表达式引擎进行 Pivot 操作
        dq_tables: Dict[str, pl.DataFrame] = {}
        for m in run_config.dq_metrics:
            dq_tables[m] = self._generate_pivot_report(m, profile_by)

        stat_tables: Dict[str, pl.DataFrame] = {}
        for m in run_config.stat_metrics:
            # a. 基础透视 (Pivot)
            pivot: pl.DataFrame = self._generate_pivot_report(m, profile_by)
            
            # b. 如果有分组 (且列数足够)，计算稳定性指标 (CV/Var)
            #    exclude_cols 排除掉非数据列，剩下的即为分组列 (如 '202301', '202302')
            if profile_by:
                pivot = self._add_stability_metrics(pivot, exclude_cols=["feature", "dtype", "total"])
            
            stat_tables[m] = pivot

        logger.info("Profile generated successfully.")

        # 4. 统一封装 
        #    利用基类的 _format_output，如果我们被 set_output("pandas")，
        #    这里的 Polars DF 会被自动转为 Pandas DF 存入 Report 对象
        return MarsProfileReport(
            overview=self._format_output(overview_df),
            dq_tables=self._format_output(dq_tables),
            stats_tables=self._format_output(stat_tables)
        )

    # =========================================================================
    # 核心实现逻辑 (Internal Core Logic)
    # =========================================================================

    def _calculate_overview(self, config: MarsProfileConfig) -> pl.DataFrame:
        """
        [Internal] 计算全量概览大宽表 (不分组)。
        
        流程:
        1. 向量化计算所有基础指标 (DQ + Stats)。
        2. 拼接特征元数据 (Dtype)。
        3. (可选) 计算并合并迷你分布图 (Sparkline)。
        4. 调整列顺序以符合阅读习惯。

        Parameters
        ----------
        config : MarsProfileConfig
            当前运行的配置对象。

        Returns
        -------
        pl.DataFrame
            包含 feature, dtype, distribution, missing_rate, mean... 等所有指标的宽表。
        """
        cols = self.df.columns
        
        # 1. 向量化计算所有基础指标
        stats: pl.DataFrame = self._analyze_cols_vectorized(cols, config)
        
        # 2. 拼上 dtype 信息
        dtype_df: pl.DataFrame = self._get_feature_dtypes()
        stats = stats.join(dtype_df, on="feature", how="left")
        
        # 3. [Feature] 计算迷你分布图 (Sparklines)
        #    为了不影响性能，这里只对数值列计算，且内部包含采样逻辑
        if config.enable_sparkline:
            sparkline_df: pl.DataFrame = self._compute_all_sparklines_polars(cols, config)
            if not sparkline_df.is_empty():
                stats = stats.join(sparkline_df, on="feature", how="left")
        
        # 4. 显式指定列顺序：Feature -> Dtype -> Distribution -> DQ -> Stats
        ideal_order: List[str] = [
            "feature", "dtype", 
            "distribution",  # <--- 新增列放在这里
            # DQ Rates (优先展示)
            "missing_rate", "zeros_rate", "unique_rate", "top1_ratio"
        ] + config.stat_metrics # Mean, Std, etc.
        
        # 容错：只选择实际存在的列 (final_cols) 并保持 ideal_order 的顺序
        final_cols: List[str] = []
        seen = set()
        for c in ideal_order:
            if c in stats.columns and c not in seen:
                final_cols.append(c)
                seen.add(c)
        
        # 如果还有其他未定义的列，放到最后
        remaining_cols = [c for c in stats.columns if c not in seen]
        
        return stats.select(final_cols + remaining_cols).sort(["dtype", "feature"])

    def _compute_all_sparklines_polars(self, cols: List[str], config: MarsProfileConfig) -> pl.DataFrame:
        """
        [Internal] 批量计算数值列的迷你分布图 (Polars Native V2)。

        该方法使用 Polars 原生 API (`series.hist`) 进行直方图统计，并映射为 Unicode 字符画。
        相比 Numpy 方案，它减少了数据在 Python 和 Rust 内存间的拷贝，并在处理缺失值和
        边缘情况（如常量列、全空列）时更加鲁棒。

        Parameters
        ----------
        cols : List[str]
            待计算的列名列表。方法内部会自动筛选出数值型列。
        config : MarsProfileConfig
            配置对象。使用其中的 `sparkline_sample_size` 控制采样行数，
            `sparkline_bins` 控制直方图的分箱数量（即字符画长度）。

        Returns
        -------
        pl.DataFrame
            包含两列的 DataFrame：
            - feature (str): 特征名称。
            - distribution (str): 分布图字符串 (如 "  ▂▅▇█ ")。
              特殊状态说明:
              - "-": 无有效数据。
              - "ERR": 计算过程发生异常。
              - "    █    ": 数据为常量（方差为0）。
        """
        # 1. 筛选数值列 (非数值列无法画分布图)
        # -------------------------------------------------------------------------
        num_cols: List[str] = [c for c in cols if self._is_numeric(c)]
        if not num_cols:
            return pl.DataFrame(
                {"feature": [], "distribution": []}, 
                schema={"feature": pl.String, "distribution": pl.String}
            )

        # 2. 采样 (Sampling) - 性能关键点
        # -------------------------------------------------------------------------
        # 如果数据量超过配置上限 (默认20w)，则进行不放回采样以加速直方图计算。
        # 对于分布概览而言，大样本采样足以拟合整体形态。
        sample_n: int = min(self.df.height, config.sparkline_sample_size)
        sample_df: pl.DataFrame = self.df.select(num_cols)
        
        if sample_df.height > sample_n:
            sample_df = sample_df.sample(sample_n, with_replacement=False)

        # 3. 准备计算参数
        # -------------------------------------------------------------------------
        sparkline_data: List[Dict[str, str]] = []
        # Unicode 方块字符集: 0/8 到 8/8 高度
        # bars: List[str] = list(" ▂▃▄▅▆▇█") 
        bars = ['_', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588']
        n_bins: int = config.sparkline_bins
        
        # 4. 逐列计算
        # -------------------------------------------------------------------------
        for col in num_cols:
            dist_str: str = "-" # 默认显示短横线，代表无数据
            try:
                valid_missing: List[Any] = self._get_valid_missing(col)
                
                # [A] 数据清洗 (Data Cleaning)
                # 获取 Series 引用 (Polars 是 Lazy/Copy-on-Write 的，此处开销很小)
                target_s: pl.Series = sample_df[col]
                
                # A.1 过滤自定义缺失值 (如 -999, -1 等)
                if valid_missing:
                    # 对于浮点数，先剔除 NaN，因为 Polars 的 is_in 无法匹配 NaN
                    if target_s.dtype in [pl.Float32, pl.Float64]:
                         target_s = target_s.filter(target_s.is_not_nan())
                    
                    # 剔除配置中定义的缺失值
                    target_s = target_s.filter(~target_s.is_in(valid_missing))

                # A.2 过滤 Null (物理缺失)
                s: pl.Series = target_s.drop_nulls()
                
                # [B] 边界情况检查 (Edge Cases)
                if s.len() == 0:
                    dist_str = "-" # 清洗后无数据
                elif s.len() == 1 or s.min() == s.max():
                     dist_str = "    █    " # 常量 (居中显示)
                else:
                    # [C] 核心计算: Polars 原生直方图
                    # s.hist() 返回一个小型的 DataFrame
                    # 列结构通常为: [break_point, category, count]
                    # 注意: 不同 Polars 版本的 count 列名可能不同 (count, bin_count 等)
                    hist_df: pl.DataFrame = s.hist(bin_count=n_bins)
                    
                    # 鲁棒性获取: 永远取最后一列作为计数列，规避列名变更问题
                    counts: List[int] = hist_df.get_column(hist_df.columns[-1]).to_list()
                    
                    # [D] 字符映射 (Visualization Mapping)
                    max_count = max(counts)
                    if max_count == 0:
                        dist_str = "-"
                    else:
                        # 归一化公式: count / max * 7 (索引 0-7)
                        # 使用 int() 向下取整，确保索引不越界
                        indices = [int(c / max_count * (len(bars) - 1)) for c in counts]
                        dist_str = "".join([bars[i] for i in indices])
                        
            except Exception as e:
                # 捕获单列计算异常，防止拖累整个报告生成
                # 使用 error 级别日志以便在排查时可见
                logger.error(f"Sparkline calculation failed for feature '{col}': {str(e)}")
                dist_str = "ERR" 
            
            sparkline_data.append({"feature": col, "distribution": dist_str})

        # 5. 返回结果表
        # -------------------------------------------------------------------------
        return pl.DataFrame(
            sparkline_data, 
            schema={"feature": pl.String, "distribution": pl.String}
        )

    def _generate_pivot_report(self, metric: str, group_col: Optional[str]) -> pl.DataFrame:
        """
        [Internal] 生成透视表 (Pivot Table)。
        
        逻辑：先计算 Total 列 (全局聚合)，如果有分组字段，再计算 GroupBy 聚合，最后横向拼接。

        Parameters
        ----------
        metric : str
            指标类型 ('missing', 'mean', 'max' 等)。
        group_col : str, optional
            分组列名。如果为 None，则只生成 Total 列。

        Returns
        -------
        pl.DataFrame
            形状为 [Features x (Metadata + Groups + Total)] 的透视表。
        """
        target_cols = [c for c in self.df.columns if c != group_col]
        if not target_cols: return pl.DataFrame()

        # 1. 计算 Total 列 (全局聚合)
        total_exprs = [self._get_single_metric_expr(c, metric).alias(c) for c in target_cols]
        total_row = self.df.select(total_exprs)
        # Transpose: [1, n_feats] -> [n_feats, 1]
        total_df = total_row.transpose(include_header=True, header_name="feature", column_names=["total"])

        # 2. 准备基础表 (feature + dtype + total)
        dtype_df = self._get_feature_dtypes()
        base_df = total_df.join(dtype_df, on="feature", how="left")
        
        # Case A: 没有分组 -> 直接返回 Total 表
        if group_col is None:
            return base_df.select(["feature", "dtype", "total"]).sort(["dtype", "feature"])

        # Case B: 有分组 -> 计算 Pivot 并 Join
        agg_exprs = [self._get_single_metric_expr(c, metric).alias(c) for c in target_cols]
        # GroupBy -> Agg -> Sort
        grouped = self.df.group_by(group_col).agg(agg_exprs).sort(group_col)
        # Transpose
        pivot_df = grouped.transpose(include_header=True, header_name="feature", column_names=group_col)

        # 3. Join Together
        result = base_df.join(pivot_df, on="feature", how="left")
        
        # 4. 调整列顺序: feature, dtype, ...groups..., total
        fixed = {"feature", "dtype", "total"}
        groups = [c for c in result.columns if c not in fixed]
        final_order = ["feature", "dtype"] + groups + ["total"]
        
        return result.select(final_order).sort(["dtype", "feature"])

    def _analyze_cols_vectorized(self, cols: List[str], config: Optional[MarsProfileConfig] = None) -> pl.DataFrame:
        """
        [Internal] 全量指标向量化计算引擎 (用于 Overview)。
        
        通过构建巨大的 Polars 表达式列表，实现 One-Pass (一次扫描) 计算所有特征的所有指标。
        相比循环计算，这种方式在 Polars 中能获得显著的性能提升。
        
        Returns
        -------
        pl.DataFrame
            统计结果宽表: [feature, metric1, metric2...]
        """
        if not cols: return pl.DataFrame()
        all_exprs = []
        
        for col in cols:
            base_exprs = self._build_expressions(col)
            for expr in base_exprs:
                # 别名格式: feature:::metric，后续通过 split 拆解
                metric_name = expr.meta.output_name()
                all_exprs.append(expr.alias(f"{col}:::{metric_name}"))

        # 执行计算
        raw_row = self.df.select(all_exprs)
        
        # Reshape: Wide -> Long -> Wide
        # unpivot 是 Polars > 1.0 的新 API (旧版本为 melt)
        long_df = raw_row.unpivot(variable_name="temp_id", value_name="value")
        
        return (
            long_df
            .with_columns(
                pl.col("temp_id").str.split_exact(":::", 1)
                .struct.rename_fields(["feature", "metric"])
                .alias("meta")
            )
            .unnest("meta")
            .pivot(on="metric", index="feature", values="value", aggregate_function="first")
        )

    def _add_stability_metrics(self, df: pl.DataFrame, exclude_cols: List[str]) -> pl.DataFrame:
        """
        [Internal] 计算行级稳定性指标：方差 (Var) 和 变异系数 (CV)。
        
        利用 Polars 的 list 算子进行水平聚合 (Horizontal Aggregation)。
        
        Parameters
        ----------
        df : pl.DataFrame
            包含分组数据的透视表。
        exclude_cols : List[str]
            需要排除的非数据列 (如 feature, dtype)。

        Returns
        -------
        pl.DataFrame
            增加了 group_var 和 group_cv 列的 DataFrame。
        """
        if df.is_empty(): return df
        
        # 锁定纯分组列 (排除 feature, dtype, total)
        calc_cols = [
            c for c in df.columns 
            if c not in exclude_cols and df[c].dtype in [pl.Float64, pl.Float32]
        ]
        if not calc_cols: return df

        epsilon = 1e-9 # 防止除以0
        
        return (
            df
            .with_columns(pl.concat_list(calc_cols).alias("_tmp")) # 将分组列压缩为 List
            .with_columns([
                # 计算方差
                pl.col("_tmp").list.var().fill_null(0).alias("group_var"),
                # 计算变异系数: Std / Mean
                (pl.col("_tmp").list.std() / (pl.col("_tmp").list.mean().abs() + epsilon)).fill_null(0).alias("group_cv")
            ])
            .drop("_tmp")
            # 调整列顺序: feature, dtype, groups..., total, var, cv
            .select(["feature", "dtype"] + calc_cols + ["total", "group_var", "group_cv"])
        )

    # --- Expression Factories (表达式工厂) ---
    
    def _build_expressions(self, col: str) -> List[pl.Expr]:
        """
        [Factory] 为单个列生成所有 Overview 指标的计算表达式。
        """
        return self._get_full_stats_exprs(col)

    def _get_single_metric_expr(self, col: str, metric_type: str) -> pl.Expr:
        """
        [Factory] 为单个列生成指定指标的计算表达式 (用于 Pivot)。
        """
        return self._get_metric_expr(col, metric_type)

    # --- Helpers (辅助方法) ---
    
    def _get_feature_dtypes(self) -> pl.DataFrame:
        """获取 Schema 信息表"""
        schema = {"feature": [], "dtype": []}
        for n, d in self.df.schema.items():
            schema["feature"].append(n)
            schema["dtype"].append(str(d))
        return pl.DataFrame(schema)

    def _is_numeric(self, col: str) -> bool:
        """判断列是否为数值类型"""
        # 兼容 Polars 这里的类型判断
        return self.df[col].dtype in [
            pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, 
            pl.Float32, pl.Float64
        ]

    def _get_valid_missing(self, col: str) -> List[Any]:
        """类型安全的缺失值匹配 (防止类型不匹配报错)"""
        if not self.custom_missing: return []
        is_num = self._is_numeric(col)
        is_str = self.df[col].dtype == pl.String
        return [v for v in self.custom_missing if (is_num and isinstance(v, (int, float))) or (is_str and isinstance(v, str))]

    # --- Detailed Expr Implementations (具体计算逻辑) ---
    
    def _get_full_stats_exprs(self, col: str) -> List[pl.Expr]:
        """[Helper] 生成全量统计表达式列表"""
        native_null = pl.col(col).null_count()
        total_len = pl.len()
        valid_missing = self._get_valid_missing(col)
        # 总缺失 = 原生 Null + 自定义 Null
        total_missing = native_null + pl.col(col).is_in(valid_missing).sum() if valid_missing else native_null
        is_num = self._is_numeric(col)
        zeros_c = (pl.col(col) == 0).sum() if is_num else pl.lit(0, dtype=pl.UInt32)
        
        # 基础 DQ 指标
        exprs = [
            (total_missing / total_len).alias("missing_rate"),
            (zeros_c / total_len).alias("zeros_rate"),
            (pl.col(col).n_unique() / total_len).alias("unique_rate"),
            (pl.col(col).value_counts(sort=True).first().struct.field("count") / total_len).alias("top1_ratio")
        ]
        
        # 数值统计指标 (非数值列填充 Null)
        if is_num:
            exprs.extend([
                pl.col(col).mean().alias("mean"), pl.col(col).std().alias("std"),
                pl.col(col).min().alias("min"), pl.col(col).max().alias("max"),
                pl.col(col).median().alias("median"),
                pl.col(col).quantile(0.25).alias("p25"), pl.col(col).quantile(0.75).alias("p75")
            ])
        else:
            null_lit = pl.lit(None, dtype=pl.Float64)
            exprs.extend([null_lit.alias(n) for n in ["mean", "std", "min", "max", "median", "p25", "p75"]])
        return exprs

    def _get_metric_expr(self, col: str, metric_type: str) -> pl.Expr:
        """[Helper] 生成单个指标的表达式"""
        if metric_type == "missing":
            native_null = pl.col(col).null_count()
            valid_missing = self._get_valid_missing(col)
            total_missing = native_null + pl.col(col).is_in(valid_missing).sum() if valid_missing else native_null
            return total_missing / pl.len()
        elif metric_type == "zeros":
            return (pl.col(col) == 0).sum() / pl.len() if self._is_numeric(col) else pl.lit(0, dtype=pl.UInt32)
        elif metric_type == "unique":
            return pl.col(col).n_unique() / pl.len()
        elif metric_type == "top1":
            return pl.col(col).value_counts(sort=True).first().struct.field("count") / pl.len()
        
        if not self._is_numeric(col): return pl.lit(None)
        
        mapper = {
            "mean": pl.col(col).mean(),
            "std": pl.col(col).std(),
            "min": pl.col(col).min(),
            "max": pl.col(col).max(),
            "median": pl.col(col).median(),
            "sum": pl.col(col).sum()
        }
        return mapper.get(metric_type, pl.lit(None))