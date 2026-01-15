import numpy as np
import polars as pl
import pandas as pd
import dataclasses
from typing import List, Union, Optional, Any, Dict, Literal
from mars.utils.logger import logger
from mars.utils.decorators import time_it, monitor_os_memory
from mars.core.base import MarsBaseEstimator
from mars.analysis.report import MarsProfileReport
from mars.analysis.config import MarsProfileConfig
from mars.feature.binning import MarsNativeBinner

class MarsDataProfiler(MarsBaseEstimator):
    """
    MarsDataProfiler - 基于 Polars 的高性能多维数据画像工具。

    专为大规模风控建模数据集设计。它作为分析流程的入口，封装了从
    数据质量诊断、统计值计算到可视化生成的全链路逻辑。

    主要功能 (Key Features)
    -----------------------
    1. **全量指标概览 (Overview)**:
       - 计算 Missing/Zero/Unique 等基础 DQ 指标。
       - 自动识别并计算数值列的统计分布 (Mean, Std, Quantiles)。
    
    2. **迷你分布图 (Sparklines)**:
       - 在报告中生成 Unicode 字符画 (如  ▂▅▇█)。
       - **可视化逻辑**: 自动采样 (默认20w行) -> 剔除异常值 -> 等宽分箱 -> 字符映射。
       - 支持通过 Config 调整分箱精度和采样率。

    3. **多维趋势分析 (Trend Analysis)**:
       - 支持按时间 (Month/Vintage) 或客群 (Segment) 进行分组分析。
       - 自动计算组间稳定性指标 (Variance/CV)。

    Attributes
    ----------
    df : pl.DataFrame
        内部存储的 Polars DataFrame。
    config : MarsProfileConfig
        全局配置对象。控制计算哪些指标、是否画图等。
        详见 `mars.analysis.config.MarsProfileConfig`。
    custom_missing : List[Any]
        自定义缺失值列表 (如 -999, 'null')。在计算 missing_rate 和画分布图时，
        这些值会被自动识别并剔除，确保统计准确性。

    Examples
    --------
    >>> # 1. 基础用法
    >>> profiler = MarsDataProfiler(df)
    >>> report = profiler.generate_profile()
    >>> report.show_overview()

    >>> # 2. 高级用法：自定义缺失值 + 按月分组 + 关闭画图(提速)
    >>> profiler = MarsDataProfiler(df, custom_missing_values=[-999, "unknown"])
    >>> report = profiler.generate_profile(
    ...     profile_by="month",
    ...     config_overrides={"enable_sparkline": False}
    ... )
    """

    def __init__(
        self, 
        df: Union[pl.DataFrame, pd.DataFrame], 
        features: Optional[List[str]] = None,
        *,
        exclude_features: Optional[List[str]] = None,
        include_dtypes: Union[type, pl.DataType, List[Union[type, pl.DataType]], None] = None,
        
        custom_missing_values: Optional[List[Union[int, float, str]]] = None,
        custom_special_values: Optional[List[Any]] = None,
        
        psi_batch_size: int = 50, 
        psi_n_bins: int = 10,           
        psi_bin_method: Literal["quantile", "uniform"] = "quantile", 
        
        sample_frac: Optional[float] = None, 
        
        config: Optional[MarsProfileConfig] = None,
    ) -> None:
        """
        初始化数据分析器。

        Parameters
        ----------
        df : Union[pl.DataFrame, pd.DataFrame]
            输入数据集。会自动转换为 Polars 格式以利用其向量化计算优势。
        features : List[str], optional
            指定要分析的列名列表。如果为 None，则分析所有列。
        exclude_features : List[str], optional
            [黑名单] 指定要排除的列名列表。
            逻辑优先级: final_features = (features or all_cols) - exclude_features
        include_dtypes : List[pl.DataType], optional
            [类型白名单] 仅包含指定数据类型的列进行分析。
            例如: [pl.Int64, pl.Float64] 只分析数值列。
        custom_missing_values : List[Union[int, float, str]], optional
            指定自定义缺失值列表。例如: [-999, "unknown", "\\N"]。
        custom_special_values : List[Any], optional
            指定自定义特殊值列表 (如极端值)。这些值在计算分布图时会被单独处理。
        psi_batch_size : int, optional
            计算 PSI 时的特征批处理大小。默认为 50。
        psi_n_bins : int, optional
            计算 PSI 时的分箱数量。默认为 10。
        psi_bin_method : str, optional
            计算 PSI 时的分箱方法。支持 "quantile" 或 "uniform"。默认为 "quantile"。
        config : MarsProfileConfig, optional
            配置对象。如果为 None，则使用默认配置。
        
        """
        super().__init__()
        # 1. 数据接入与采样
        self.df = self._ensure_polars(df)
        if sample_frac is not None and 0 < sample_frac < 1.0:
            logger.warning(f"🎲 Data is sampled (frac={sample_frac}). Metrics are estimates.")
            self.df = self.df.sample(fraction=sample_frac, shuffle=True)

        self.config = config if config else MarsProfileConfig()
        
        # 2. 值处理配置
        self.custom_missing = custom_missing_values if custom_missing_values else []
        self.special_values = custom_special_values if custom_special_values else []
        
        # 3. PSI 配置
        self.psi_batch_size = psi_batch_size
        self.psi_n_bins = psi_n_bins
        self.psi_bin_method = psi_bin_method

        # 4. 特征筛选逻辑 (核心漏斗)
        # Step A: 初始范围
        candidates = features if features else self.df.columns
            
        # Step B: 黑名单剔除
        if exclude_features:
            exclude_set = set(exclude_features)
            candidates = [c for c in candidates if c not in exclude_set]

        # Step C: 类型白名单 (支持 Python原生类型 + Polars类型)
        # ---------------------------------------------------------
        if include_dtypes:
            import polars.selectors as cs
            
            # 1. 归一化为列表
            if not isinstance(include_dtypes, list):
                raw_dtypes = [include_dtypes]
            else:
                raw_dtypes = include_dtypes
            
            # 2. 类型映射：Python Type -> Polars Abstract Type
            target_dtypes = []
            for t in raw_dtypes:
                # --- Python Native Mapping ---
                if t is int:
                    target_dtypes.append(pl.Integer) # 匹配所有整型 (Int8~64, UInt)
                elif t is float:
                    target_dtypes.append(pl.Float)   # 匹配所有浮点 (Float32/64)
                elif t is str:
                    target_dtypes.append(pl.String)  # 匹配 String/Utf8
                elif t is bool:
                    target_dtypes.append(pl.Boolean)
                elif t is list:
                    target_dtypes.append(pl.List)
                # --- Polars Type Pass-through ---
                else:
                    target_dtypes.append(t)
            
            # 3. 智能选择
            try:
                # 利用 Selectors 进行宽容匹配
                dtype_selector = cs.by_dtype(target_dtypes)
                # 只在 candidates 范围内筛选
                matched_cols = self.df.select(pl.col(candidates)).select(dtype_selector).columns
                candidates = matched_cols
                
            except Exception as e:
                logger.error(f"Type filtering failed: {e}")
                # 降级策略: 简单的包含判断 (仅对 Polars 类型有效)
                candidates = [c for c in candidates if self.df.schema[c] in target_dtypes]

        if not candidates:
            raise ValueError("No features selected after filtering.")
            
        self.features = candidates
    @time_it
    def generate_profile(
        self, 
        profile_by: Optional[str] = None, 
        *,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> MarsProfileReport:
        """
        [核心接口] 执行数据画像分析管道，生成完整分析报告。

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
              可选值: "psi", "mean", "std", "min", "max", "p25", "median", "p75", "skew", "kurtosis"。
            * ``dq_metrics`` (List[str]): 需要计算的数据质量指标。
              可选值: "missing", "zeros", "unique", "top1"。

            **2. 可视化 (Visualization)**
            
            * ``enable_sparkline`` (bool): 是否计算字符画形式的迷你分布图 (默认 True)。
            * ``sparkline_sample_size`` (int): 计算分布图时的采样行数。
            * ``sparkline_bins`` (int): 分布图的分箱精度。

        Returns
        -------
        MarsProfileReport
            包含概览表和趋势表的报告对象容器。

        Examples
        --------
        >>> # 1. 基础用法：生成并查看报告
        >>> profiler = MarsDataProfiler(df)
        >>> report = profiler.generate_profile()
        
        >>> # 拿到 report 后怎么用？
        >>> report # 在 Jupyter 中显示报告的用法
        >>> report.show_overview()  # 在 Jupyter 中显示数据概览
        >>> report.write_excel("my_analysis.xlsx")  # 导出 Excel

        >>> # 2. 高级用法：按月分组分析
        >>> report = profiler.generate_profile(profile_by="month")
        >>> report.show_trend("mean") # 查看均值随月份的变化趋势
        
        >>> # 3. 获取底层数据 (可以用于自动化特征筛选)
        >>> # 返回值结构:
        >>> # overview: DataFrame (全量概览)
        >>> # dq_tables: Dict[str, DataFrame] (DQ 指标趋势表字典)
        >>> # stat_tables: Dict[str, DataFrame] (统计指标趋势表字典)
        >>> overview, dq_tables, stat_tables = report.get_profile_data()
        
        >>> # 示例: 筛选出缺失率 > 90% 的特征列表
        >>> high_missing_cols = overview.filter(pl.col("missing_rate") > 0.9)["feature"].to_list()
        """
        # 1. 动态配置合并 (只影响本次运行，不污染 self.config)
        run_config: MarsProfileConfig = self.config
        if config_overrides:
            run_config = dataclasses.replace(self.config, **config_overrides)

        logger.info(f"Starting profiling (Profile by: {profile_by}, Sparkline: {run_config.enable_sparkline})...")
        
        # 2. 计算全量概览 (Overview) - 核心基础表
        #    包含: Sparkline, DQ, Stats
        overview_df: pl.DataFrame = self._calculate_overview(run_config)

        # 3. 计算趋势表 (Trend Tables)
        #    利用 Polars 的表达式引擎进行 Pivot 操作
        dq_tables: Dict[str, pl.DataFrame] = {}
        # 3.1 计算 DQ 指标透视表
        for m in run_config.dq_metrics:
            dq_tables[m] = self._generate_pivot_report(m, profile_by)

        # 3.2 计算 统计指标透视表
        stat_tables: Dict[str, pl.DataFrame] = {}
        for m in run_config.stat_metrics:
            # a. 基础透视 (Pivot)
            pivot: pl.DataFrame = self._generate_pivot_report(m, profile_by)
            
            # b. 如果有分组 (且列数足够)，计算稳定性指标 (CV/Var)
            #    exclude_cols 排除掉非数据列，剩下的即为分组列 (如 '202301', '202302')
            if profile_by:
                pivot = self._add_stability_metrics(pivot, exclude_cols=["feature", "dtype", "total"])
            
            stat_tables[m] = pivot
        # 3.2 计算 PSI (仅当有分组时)
        if profile_by and ("psi" in run_config.stat_metrics):
            try:
                # 调用独立的 PSI 计算逻辑
                psi_df = self._get_psi_trend(profile_by)
                if not psi_df.is_empty():
                    # 将结果作为 "psi" 指标加入 stat_tables
                    # 这样在 Report 中就可以通过 report.show_trend("psi") 查看
                    stat_tables["psi"] = psi_df
            except Exception as e:
                logger.warning(f"⚠️ PSI calculation skipped due to error: {e}")

        logger.info("Profile generated successfully.")

        # 4. 统一封装 
        #    利用基类的 _format_output 自动处理格式转换
        return MarsProfileReport(
            overview=self._format_output(overview_df),
            dq_tables=self._format_output(dq_tables),
            stats_tables=self._format_output(stat_tables)
        )
        
    @time_it
    @monitor_os_memory
    def _get_psi_trend(self, group_col: str, features: Optional[List[str]] = None) -> pl.DataFrame:
        """
        计算分组 PSI 趋势。
        
        Fix: Binner.transform 输入改为 Eager Slice (Polars 列切片是零拷贝的)，
             配合 lazy=True 参数实现流式计算，解决类型报错同时保持高性能。
        Parameters
        ----------
        group_col : str
            分组字段名 (如 'month', 'vintage')。
        features : List[str], optional
            指定计算 PSI 的列名列表。如果为 None，则使用初始化时的所有列。
        """
        # 1. 确定计算范围
        candidates = features if features else self.features
        candidates = [c for c in candidates if c != group_col]
        
        if not candidates:
            return pl.DataFrame()

        num_cols = [c for c in candidates if self._is_numeric(c)]
        cat_cols = [c for c in candidates if c not in num_cols]
        
        logger.info(f"🚀 PSI Calc: {len(num_cols)} numeric + {len(cat_cols)} categorical.")

        try:
            baseline_group = self.df.select(pl.col(group_col).min()).item()
        except Exception:
            return pl.DataFrame()

        psi_result_parts = []
        common_schema_order = [group_col, "feature", "total", "psi"]
        
        BATCH_SIZE = self.psi_batch_size 
        # ==============================================================================
        # 🟢 路一：数值特征 PSI (分批 Int16 高速路)
        # ==============================================================================
        if num_cols:
            try:
                numeric_missing = [v for v in self.custom_missing if isinstance(v, (int, float)) and not isinstance(v, bool)]
                numeric_special = [v for v in self.special_values if isinstance(v, (int, float)) and not isinstance(v, bool)]
                
                # 1. Fit Global
                binner = MarsNativeBinner(
                    features=num_cols,
                    method=self.psi_bin_method, 
                    n_bins=self.psi_n_bins,          
                    special_values=numeric_special,
                    missing_values=numeric_missing,
                    remove_empty_bins=False     
                )
                binner.fit(self.df)
                
                # 预构建骨架所需的 Bin IDs
                possible_bins = list(range(self.psi_n_bins)) + [-1]
                if numeric_special:
                    possible_bins.extend([-3 - i for i in range(len(numeric_special))])
                b_ids = pl.DataFrame({"bin_id": possible_bins}, schema={"bin_id": pl.Int16})

                # 2. 分批处理 Loop
                for i in range(0, len(num_cols), BATCH_SIZE):
                    batch_cols = num_cols[i : i + BATCH_SIZE]
                    
                    # --- Local Scope Start ---
                    
                    # A. Transform (Input: Eager, Output: Lazy)
                    cols_needed = batch_cols + [group_col]
                    
                    # 这里直接传入 Eager DataFrame Slice
                    # Polars 的 select 是零拷贝的，不会复制数据，所以这里很快且内存安全
                    df_batch_input = self.df.select(cols_needed)
                    
                    # 开启 lazy=True，让 transform 内部转为 lazy 模式执行逻辑，避免生成巨大的中间结果矩阵
                    lf_binned: pl.LazyFrame = binner.transform(df_batch_input, return_type='index', lazy=True)
                    
                    # B. 构建当前批次的 Rename Map
                    feat_map_batch = {idx: name for idx, name in enumerate(batch_cols)}
                    bin_cols_batch = [f"{c}_bin" for c in batch_cols]
                    rename_map = {old: str(idx) for idx, old in enumerate(bin_cols_batch)}
                    
                    # C. Streaming Unpivot & Aggregation
                    agg_stats_batch = (
                        lf_binned
                        .rename(rename_map)
                        .select([group_col] + list(rename_map.values()))
                        .unpivot(
                            index=[group_col],
                            on=list(rename_map.values()),
                            variable_name="feat_idx", 
                            value_name="bin_id"
                        )
                        .with_columns([
                            # ✅ [FIX] 双重保险：强制转为 Int16
                            # 这行代码是 PSI 计算稳定性的“定海神针”
                            pl.col("feat_idx").cast(pl.Int16),
                            pl.col("bin_id").cast(pl.Int16) 
                        ])
                        .group_by([group_col, "feat_idx", "bin_id"])
                        .len()
                        .collect(streaming=True) 
                    )
                    
                    # D. 构建当前批次的骨架
                    f_ids = pl.DataFrame({"feat_idx": list(feat_map_batch.keys())}, schema={"feat_idx": pl.Int16})
                    unique_bins_skel = f_ids.join(b_ids, how="cross")
                    
                    unique_groups = agg_stats_batch.select(group_col).unique()
                    skeleton = unique_bins_skel.join(unique_groups, how="cross")
                    
                    # E. 计算 PSI
                    psi_num_raw = self._calc_psi_from_stats(agg_stats_batch, skeleton, unique_bins_skel, group_col, baseline_group)
                    
                    # F. 还原特征名 & 格式化
                    mapping_df = pl.DataFrame({
                        "feat_idx": list(feat_map_batch.keys()),
                        "feature": list(feat_map_batch.values())
                    }, schema={"feat_idx": pl.Int16, "feature": pl.String})
                    
                    psi_num_final = (
                        psi_num_raw
                        .join(mapping_df, on="feat_idx", how="left")
                        .select(common_schema_order)
                    )
                    
                    psi_result_parts.append(psi_num_final)
                    # --- Local Scope End ---

            except Exception as e:
                logger.error(f"Numeric PSI failed: {e}")

        # ==============================================================================
        # 🟡 路二：类别特征 PSI (常规 Lazy)
        # ==============================================================================
        if cat_cols:
            try:
                long_cat = (
                    self.df.lazy()
                    .select(cat_cols + [group_col])
                    .unpivot(
                        index=[group_col],
                        on=cat_cols,
                        variable_name="feature",
                        value_name="bin_id_raw"
                    )
                    .with_columns(
                        pl.col("bin_id_raw").fill_null("Missing").cast(pl.Utf8).alias("bin_id")
                    )
                    .group_by([group_col, "feature", "bin_id"])
                    .len()
                    .collect(streaming=True)
                )

                unique_bins_cat = long_cat.select(["feature", "bin_id"]).unique()
                unique_groups_cat = long_cat.select(group_col).unique()
                skeleton_cat = unique_bins_cat.join(unique_groups_cat, how="cross")

                psi_cat_raw = self._calc_psi_from_stats(long_cat, skeleton_cat, unique_bins_cat, group_col, baseline_group)
                psi_result_parts.append(psi_cat_raw.select(common_schema_order))

            except Exception as e:
                logger.error(f"Categorical PSI failed: {e}")

        # ==============================================================================
        # 🏁 合并与整形
        # ==============================================================================
        if not psi_result_parts:
            return pl.DataFrame()

        final_long_psi: pl.DataFrame = pl.concat(psi_result_parts)

        # Pivot
        pivot_df = (
            final_long_psi
            .pivot(on=group_col, index=["feature", "total"], values="psi")
        )

        dtype_df = self._get_feature_dtypes()
        result = pivot_df.join(dtype_df, on="feature", how="left")
        
        raw_group_cols = [c for c in result.columns if c not in ["feature", "dtype", "total"]]
        psi_data_cols = sorted(raw_group_cols)
        
        if psi_data_cols:
            epsilon_stat = 1e-9
            result = (
                result
                .with_columns(pl.concat_list(psi_data_cols).alias("_tmp_psi_list"))
                .with_columns([
                    pl.col("_tmp_psi_list").list.mean().alias("group_mean"),
                    pl.col("_tmp_psi_list").list.var().fill_null(0).alias("group_var"),
                    (pl.col("_tmp_psi_list").list.std() / (pl.col("_tmp_psi_list").list.mean() + epsilon_stat)).fill_null(0).alias("group_cv")
                ])
                .drop("_tmp_psi_list")
            )
            
            final_order = ["feature", "dtype"] + psi_data_cols + ["total", "group_mean", "group_var", "group_cv"]
            return result.select(final_order).sort("feature")
        else:
            return result.sort("feature")

    def _calc_psi_from_stats(self, stats_df: pl.DataFrame, skeleton: pl.DataFrame, unique_bins_skel: pl.DataFrame, group_col: str, baseline_group: Any) -> pl.DataFrame:
        """
        [Helper] 基于聚合统计表 (Count Table) 计算 PSI。
        不再处理原始明细数据，速度极快。
        
        Args:
            stats_df: 聚合后的统计表 [group_col, feature(or idx), bin_id, len]
            skeleton: 骨架表
            unique_bins_skel: 特征-箱号对应表 (用于 Total 聚合)
        """
        # 识别特征列名 (可能是 "feature" 也可能是 "feat_idx")
        feat_col = "feat_idx" if "feat_idx" in stats_df.columns else "feature"
        
        # 1. 计算基准分布 (Expected)
        # 从统计表中直接 filter，无需再次 aggregation
        base_stats = stats_df.filter(pl.col(group_col) == baseline_group)
        
        expected = (
            base_stats
            .with_columns((pl.col("len") / pl.col("len").sum().over(feat_col)).alias("E"))
            .select([feat_col, "bin_id", "E"])
        )

        # 2. 计算实际分布 (Actual) - 分组统计
        actual = (
            stats_df
            .with_columns((pl.col("len") / pl.col("len").sum().over([group_col, feat_col])).alias("A"))
            .select([group_col, feat_col, "bin_id", "A"])
        )

        # 3. 计算全量实际分布 (Global Actual)
        # 直接在统计表上再次 group_by 即可，无需回溯 9000万行数据
        global_stats = (
            stats_df
            .group_by([feat_col, "bin_id"])
            .agg(pl.col("len").sum().alias("total_len"))
        )
        
        global_actual = (
            global_stats
            .with_columns((pl.col("total_len") / pl.col("total_len").sum().over(feat_col)).alias("A_global"))
            .select([feat_col, "bin_id", "A_global"])
        )

        epsilon = 1e-6

        # ==========================================
        # 计算 PSI
        # ==========================================
        
        # Part A: Group PSI
        psi_group_df = (
            skeleton
            .join(actual, on=[group_col, feat_col, "bin_id"], how="left")
            .with_columns(pl.col("A").fill_null(epsilon)) 
            .join(expected, on=[feat_col, "bin_id"], how="left")
            .with_columns(pl.col("E").fill_null(epsilon))
            .with_columns([
                ((pl.col("A") - pl.col("E")) * (pl.col("A") / pl.col("E")).log()).alias("psi_contrib")
            ])
            .group_by([group_col, feat_col])
            .agg(pl.col("psi_contrib").sum().alias("psi"))
        )

        # Part B: Total PSI
        psi_total_df = (
            unique_bins_skel
            .join(global_actual, on=[feat_col, "bin_id"], how="left")
            .with_columns(pl.col("A_global").fill_null(epsilon))
            .join(expected, on=[feat_col, "bin_id"], how="left")
            .with_columns(pl.col("E").fill_null(epsilon))
            .with_columns([
                ((pl.col("A_global") - pl.col("E")) * (pl.col("A_global") / pl.col("E")).log()).alias("psi_contrib_total")
            ])
            .group_by(feat_col)
            .agg(pl.col("psi_contrib_total").sum().alias("total"))
        )

        return psi_group_df.join(psi_total_df, on=feat_col, how="left")

    # =========================================================================
    # Internal Core Logic (核心实现)
    # =========================================================================

    def _calculate_overview(self, config: MarsProfileConfig) -> pl.DataFrame:
        """
        [Internal] 计算全量概览大宽表 (不分组)。
        
        Pipeline:
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
            包含 feature, dtype, distribution, missing_rate... 等所有指标的宽表。
        """
        cols = self.features
        
        # 1. 向量化计算所有基础指标 (One-Pass)
        stats: pl.DataFrame = self._analyze_cols_vectorized(cols, config)
        
        # 2. 拼接 dtype 信息
        dtype_df: pl.DataFrame = self._get_feature_dtypes()
        stats = stats.join(dtype_df, on="feature", how="left")
        
        # 3. [Feature] 计算迷你分布图 (Sparklines)
        if config.enable_sparkline:
            sparkline_df: pl.DataFrame = self._compute_all_sparklines(cols, config)
            if not sparkline_df.is_empty():
                stats = stats.join(sparkline_df, on="feature", how="left")
        
        # 4. 显式指定列顺序：Feature -> Dtype -> Distribution -> DQ -> Stats
        ideal_order: List[str] = [
            "feature", "dtype", 
            "distribution",  
            "missing_rate", "zeros_rate", "unique_rate", "top1_ratio"
        ] + config.stat_metrics
        
        # 容错：只选择实际存在的列并保持 ideal_order 的顺序
        final_cols: List[str] = []
        seen = set()
        for c in ideal_order:
            if c in stats.columns and c not in seen:
                final_cols.append(c)
                seen.add(c)
        
        # 如果还有其他未定义的列，放到最后
        remaining_cols = [c for c in stats.columns if c not in seen]
        
        return stats.select(final_cols + remaining_cols).sort(["dtype", "feature"])

    def _compute_all_sparklines(self, cols: List[str], config: MarsProfileConfig) -> pl.DataFrame:
        """
        [Internal] 批量计算数值列的迷你分布图 (Polars Native V3)。

        使用 Polars 原生 API (`series.hist`) 进行直方图统计，并映射为 Unicode 字符画。
        相比 Numpy 方案，减少了数据拷贝，并在处理缺失值和边缘情况时更加鲁棒。

        **分布图符号说明 (Visual Representation)**:
        -----------------------------------------
        * **正常分布**: 使用 Unicode 方块字符表示频率高低 (如 ``_ ▂▅▇█``)。
          - **0 值**: 强制使用下划线 ``_`` 作为基准线，确保视觉占位。
          - **非 0 值**: 使用 2/8 到 8/8 高度的方块 (``▂`` 到 ``█``)，跳过 1/8 块以增强可视性。
        
        * **无有效数据**: 显示全下划线 ``________``。
          - 场景: 原始列全为 Null/NaN，或者所有值都在 `custom_missing` 列表中。
        
        * **逻辑无分布**: 显示全下划线 ``________`` (并记录 Debug 日志)。
          - 场景: 数据存在 (len>0) 但无法构建直方图 (如全为无穷大 Inf)。
        
        * **单一值 (Constant)**: 显示居中方块 ``____█____``。
          - 场景: 方差为 0，所有有效值都相等。
        
        * **计算异常**: 显示 ``ERR``。

        Parameters
        ----------
        cols : List[str]
            待计算的列名列表。方法内部会自动筛选出数值型列。
        config : MarsProfileConfig
            包含 `sparkline_sample_size` (采样数) 和 `sparkline_bins` (字符画长度/分箱数) 配置。

        Returns
        -------
        pl.DataFrame
            包含 [feature, distribution] 的两列 DataFrame。
        """
        # 1. 筛选数值列 (非数值列无法画分布图)
        num_cols: List[str] = [c for c in cols if self._is_numeric(c)]
        if not num_cols:
            return pl.DataFrame(
                {"feature": [], "distribution": []}, 
                schema={"feature": pl.String, "distribution": pl.String}
            )

        # 2. 采样 (Sampling) - 性能优化
        #    如果数据量超过上限，则进行不放回采样以加速直方图计算
        limit_n = config.sparkline_sample_size
        
        # 策略：先 Select (减少列宽) -> 再判断是否需要 Sample
        # Polars 的 select 操作通常是零拷贝的，非常轻量
        df_subset = self.df.select(num_cols)
        
        if df_subset.height > limit_n:
            # Case A: 数据量过大 -> 随机采样 (触发数据拷贝和洗牌)
            sample_df = df_subset.sample(n=limit_n, with_replacement=False)
        else:
            # Case B: 数据量在限制内 -> 直接使用 (Zero-Copy，速度极快)
            # 避免了对小数据集进行无意义的 shuffle
            sample_df = df_subset

        # 3. 准备字符集
        #    0值使用下划线，非0值使用 Block Elements，确保视觉对比度和可见性
        sparkline_data: List[Dict[str, str]] = []
        bars = ['_', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588']
        n_bins: int = config.sparkline_bins
        
        for col in num_cols:
            dist_str: str = "-" # 默认显示短横线，代表无数据
            try:
                # 获取清洗逻辑（排除 -999 等自定义缺失值）
                exclude_vals = self._get_values_to_exclude(col)
                target_s: pl.Series = sample_df[col]
                
                # --- A. 数据清洗 ---
                if exclude_vals:
                    # 对于浮点数，先剔除 NaN (Polars is_in 不处理 NaN)
                    if target_s.dtype in [pl.Float32, pl.Float64]:
                         target_s = target_s.filter(target_s.is_not_nan())
                    target_s = target_s.filter(~target_s.is_in(exclude_vals))
                s: pl.Series = target_s.drop_nulls()
                
                # --- B. 边界检查 ---
                if s.len() == 0:
                    # [Case 1] 物理无数据: 清洗后什么都不剩了
                    dist_str = "_" * n_bins 
                    logger.debug(f"ℹ️ Sparkline skipped for '{col}': No valid data after cleaning (All Null/NaN or in custom_missing).")
                elif s.len() == 1 or s.min() == s.max():
                     # 常量: 居中显示
                     dist_str = "____█____" 
                     logger.debug(f"ℹ️ Sparkline constant for '{col}': All values are {s.min()}.")
                else:
                    # --- C. 核心计算 (Polars Hist) ---
                    # hist 返回 [break_point, category, count]，最后一列是 count
                    hist_df: pl.DataFrame = s.hist(bin_count=n_bins)
                    counts: List[int] = hist_df.get_column(hist_df.columns[-1]).to_list()
                    
                    # --- D. 字符映射算法 ---
                    # 找出最高的桶有多少个数据，作为分母
                    max_count = max(counts)
                    
                    if max_count == 0:
                        # [Case 2] 逻辑无分布: 有数据，但直方图没算出来 (如全 Inf)
                        dist_str = "_" * n_bins
                        # [Log] 记录特殊情况，方便排查为什么有数据却没图
                        logger.debug(f"⚠️ Sparkline empty for '{col}': Data len={s.len()}, but histogram counts are 0. (Possible cause: all values are Infinite)")
                    else:
                        chars = []
                        for c in counts:
                            if c == 0:
                                chars.append(bars[0]) # 0 -> 下划线
                            else:
                                # 计算高度比例，非0值映射到 bars 的索引 (1到7)
                                idx = int(c / max_count * (len(bars) - 2)) + 1
                                idx = min(idx, len(bars) - 1)
                                chars.append(bars[idx])
                        dist_str = "".join(chars)
                        
            except Exception as e:
                logger.error(f"Sparkline calculation failed for feature '{col}': {str(e)}")
                dist_str = "ERR" 
            
            sparkline_data.append({"feature": col, "distribution": dist_str})

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
        target_cols = [c for c in self.features if c != group_col]
        if not target_cols: 
            return pl.DataFrame()

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
        #   结果形状: M个分组 x N个特征
        grouped = self.df.group_by(group_col).agg(agg_exprs).sort(group_col)
        
        # 再次转置
        #   输入: 
        #   month  | age | income
        #   202301 | 25  | 10000
        #   202302 | 26  | 12000
        #
        #   输出 (Transpose后):
        #   feature | 202301 | 202302
        #   age     | 25     | 26
        #   income  | 10000  | 12000
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
        
        # 确保 config 存在
        cfg = config if config else self.config
        
        # 1. 构建表达式列表
        for col in cols:
            # 获取该列需要计算的所有指标表达式 (Mean, Missing, Max...)
            base_exprs = self._build_expressions(col, cfg)
            for expr in base_exprs:
                # 给表达式起个特殊名字，包含 特征名 和 指标名，中间用 ::: 分隔
                # 比如: "age:::mean", "salary:::missing_rate"，后续通过 split 拆解
                metric_name = expr.meta.output_name()
                all_exprs.append(expr.alias(f"{col}:::{metric_name}"))

        # 2. 执行计算 (One-Shot)
        #    因为全是聚合表达式
        #    于是，结果 raw_row 形状: 1 行, (N_features * N_metrics) 列
        raw_row = self.df.select(all_exprs)
        
        # 3. 变形 (Reshape): 宽变长 (Wide -> Long)
        #    unpivot 会把那几千列变成两列：
        #    temp_id (原来的列名, 如 "age:::mean")
        #    value   (计算出的值)
        long_df = raw_row.unpivot(variable_name="temp_id", value_name="value")
        
        # 4. 解析 & 再次透视 (Pivot): 长变宽
        return (
            long_df
            # 把 "age:::mean" 拆分成 "age" 和 "mean"
            .with_columns(
                pl.col("temp_id").str.split_exact(":::", 1)
                .struct.rename_fields(["feature", "metric"])
                .alias("meta")
            )
            .unnest("meta") # 把 struct 拆成两列 feature, metric
            # pivot 操作：行索引是 feature，列名是 metric，值是 value
            .pivot(on="metric", index="feature", values="value", aggregate_function="first")
        )

    def _add_stability_metrics(self, df: pl.DataFrame, exclude_cols: List[str]) -> pl.DataFrame:
        """
        [Internal] 计算行级稳定性指标：方差 (Var) 和 变异系数 (CV)。
        
        利用 Polars 的 list 算子进行水平聚合 (Horizontal Aggregation)。
        
        我们想计算每一行 (每个特征) 在不同 group 间的波动。
        Polars 主要是列式计算，行计算比较麻烦。
        这里用了一个技巧：`concat_list`。
        把所有月份列的值，合并成一列 list: [25, 26, ...]
        然后直接对这个 list 列算 std 和 mean。
        
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
                # [新增] group_mean: 指标在不同分组间的平均值
                pl.col("_tmp").list.mean().fill_null(0).alias("group_mean"),
                # 计算方差
                pl.col("_tmp").list.var().fill_null(0).alias("group_var"),
                # 计算变异系数: Std / Mean
                (pl.col("_tmp").list.std() / (pl.col("_tmp").list.mean().abs() + epsilon)).fill_null(0).alias("group_cv")
            ])
            .drop("_tmp")
            # 调整列顺序: feature, dtype, groups..., total, var, cv
            .select(["feature", "dtype"] + calc_cols + ["total", "group_mean", "group_var", "group_cv"])
        )

    # =========================================================================
    # Expression Factories (表达式工厂)
    # =========================================================================
    
    def _build_expressions(self, col: str, config: MarsProfileConfig) -> List[pl.Expr]:
        """[Factory] 为单个列生成所有 Overview 指标的计算表达式。"""
        return self._get_full_stats_exprs(col, config)

    def _get_single_metric_expr(self, col: str, metric_type: str) -> pl.Expr:
        """[Factory] 为单个列生成指定指标的计算表达式 (用于 Pivot)。"""
        return self._get_metric_expr(col, metric_type)

    # =========================================================================
    # Helpers (辅助方法)
    # =========================================================================
    
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
        # Polars 很严格，如果你拿字符串 "unknown" 去过滤整数列，程序会崩。
        # 这个函数会检查当前列的类型，只返回类型匹配的自定义缺失值。
        if not self.custom_missing: 
            return []
        is_num = self._is_numeric(col)
        is_str = self.df[col].dtype == pl.String
        return [v for v in self.custom_missing if (is_num and isinstance(v, (int, float))) or (is_str and isinstance(v, str))]
    
    def _get_values_to_exclude(self, col: str) -> List[Any]:
        """
        [Helper] 获取当前列需要剔除的所有特定值 (类型安全)。
        
        逻辑：
        1. 合并 custom_missing (视为缺失) 和 special_values (视为特殊)。
        2. 根据当前列的类型 (Numeric/String) 过滤掉不匹配的值，防止 Polars 报错。
           (例如：不能拿着字符串 'unknown' 去过滤整数类型的 age 列)
        
        Returns
        -------
        List[Any]
            当前列中应当被视为“非正常数值”的列表。
        """
        # 1. 合并两个列表 (假设你在 __init__ 里已经定义了 self.special_values)
        # 如果 self.special_values 还没定义，就用空列表代替
        special_vals = getattr(self, "special_values", [])
        candidates = self.custom_missing + special_vals
        
        if not candidates: 
            return []

        # 2. 获取列类型
        is_num = self._is_numeric(col)
        is_str = self.df[col].dtype == pl.String

        # 3. 类型安全过滤 (Type-Safe Filtering)
        valid_values = []
        for v in candidates:
            # 只有当 值类型 与 列类型 匹配时，才加入列表
            if is_num and isinstance(v, (int, float)) and not isinstance(v, bool):
                valid_values.append(v)
            elif is_str and isinstance(v, str):
                valid_values.append(v)
                
        return valid_values

    # --- Detailed Expr Implementations (具体计算逻辑) ---
    
    def _get_full_stats_exprs(self, col: str, config: MarsProfileConfig) -> List[pl.Expr]:
        """
        [Helper] 生成单个列的全量统计表达式列表
        这里返回的是一堆 pl.Expr 对象，还没有真正执行计算。
        """
        
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
            (
                pl.col(col)                         # 1. 选中目标列 (假设列名叫 "city")
                
                .value_counts(sort=True)            # 2. 统计每个值出现的次数，并按次数从多到少排序
                                                    #    返回数据格式 (List[Struct]): 
                                                    #    [{"city": "北京", "count": 100},  <-- 第1行 (次数最多)
                                                    #     {"city": "上海", "count": 80},   <-- 第2行
                                                    #     ...]

                .first()                            # 3. 只取排序后的第 1 行数据 (也就是众数的那一行)
                                                    #    返回数据格式 (Struct): 
                                                    #    {"city": "北京", "count": 100}

                .struct.field("count")              # 4. 从这个结构体(Struct)中，只提取 "count" 这个字段的值
                                                    #    返回数据: 100

                / total_len                         # 5. 除以总行数 (例如总共有 1000 行)
                                                    #    计算: 100 / 1000 = 0.1

            ).alias("top1_ratio")                   # 6. 给这个计算结果起个名字叫 "top1_ratio"
        ]
        
        # 动态统计指标 (基于 Config)
        # 数值统计指标
        if is_num:
            # [修改] 遍历 config.stat_metrics 动态生成
            for metric in config.stat_metrics:
                expr = self._get_metric_expr(col, metric)
                if expr is not None:
                    exprs.append(expr.alias(metric))
        # 非数值列，直接填充 Null
        else:
            null_lit = pl.lit(None, dtype=pl.Float64)
            for metric in config.stat_metrics:
                exprs.append(null_lit.alias(metric))
        return exprs

    def _get_metric_expr(self, col: str, metric_type: str) -> pl.Expr:
        """
        [Helper] 生成单个指标的计算表达式。

        **特殊值处理逻辑 (Special Values Handling)**:
        1. **DQ 指标 (Missing/Unique/Top1)**: 基于全量数据计算。特殊值会被视为“值”参与 Unique/Top1 统计，或被归为 Missing。
        2. **统计指标 (Mean/Std/Quantile)**: 基于剔除特殊值后的“净数据”计算。防止 -999 拉低均值或扭曲分布。

        Parameters
        ----------
        col : str
            目标列名。
        metric_type : str
            指标名称。

        Returns
        -------
        pl.Expr
            Polars 表达式对象。
        """
        # 1. 获取该列对应的特殊值/缺失值列表
        valid_missing = self._get_valid_missing(col)
        
        # 2. 定义基础列对象 (Raw Data)
        raw_col = pl.col(col)

        # ---------------------------------------------------------
        # Group A: 数据质量指标 (基于 Raw Data 计算)
        # ---------------------------------------------------------
        if metric_type == "missing":
            # 缺失率 = (原生 Null + 自定义特殊值) / 总行数
            native_null = raw_col.null_count()
            custom_missing_count = raw_col.is_in(valid_missing).sum() if valid_missing else pl.lit(0, dtype=pl.UInt32)
            return (native_null + custom_missing_count) / pl.len()
        
        elif metric_type == "zeros":
            # 零值率 (物理意义上的 0)
            return (raw_col == 0).sum() / pl.len() if self._is_numeric(col) else pl.lit(0, dtype=pl.UInt32)
        
        elif metric_type == "unique":
            # 唯一值数量 (包含特殊值)
            return raw_col.n_unique() / pl.len()
        
        elif metric_type == "top1":
            # 众数占比 (特殊值也可能成为众数，需暴露风险)
            return raw_col.value_counts(sort=True).first().struct.field("count") / pl.len()

        # ---------------------------------------------------------
        # Group B: 数值统计指标 (基于 Clean Data 计算)
        # ---------------------------------------------------------
        if not self._is_numeric(col): 
            return pl.lit(None)

        # 计算统计均时，要把 Missing 和 Special 全部踢走
        exclude_vals = self._get_values_to_exclude(col)
        
        if exclude_vals:
            # 过滤掉不需要的值
            clean_col = raw_col.filter(~raw_col.is_in(exclude_vals))
        else:
            clean_col = raw_col

        mapper = {
            # 集中度
            "mean": clean_col.mean(),
            "median": clean_col.median(),
            "sum": clean_col.sum(),
            
            # 离散度
            "std": clean_col.std(),
            
            # 极值 (最小值如果是 -999 就没意义了，所以要用 clean_col)
            "min": clean_col.min(),
            "max": clean_col.max(),
            
            # 分位数
            "p25": clean_col.quantile(0.25),
            "p75": clean_col.quantile(0.75),
            
            # 分布形态
            "skew": clean_col.skew(),
            "kurtosis": clean_col.kurtosis()
        }
        
        return mapper.get(metric_type, pl.lit(None))