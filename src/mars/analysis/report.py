import polars as pl
import pandas as pd
from typing import Dict, Tuple, Optional, Union, List, Any
from mars.utils.logger import logger

try:
    from IPython.display import display, HTML
except ImportError:
    display = None

class MarsProfileReport:
    """
    [报告容器] MarsProfileReport - 统一管理数据分析结果的展示与导出。
    
    该类作为 MarsDataProfiler 的输出容器，负责将原始的统计数据 (DataFrame)
    转换为适合人类阅读的格式。它支持两种主要的输出渠道：
    1. **Jupyter Notebook**: 生成富文本 HTML，包含交互式表格、热力图和迷你分布图。
    2. **Excel 文件**: 导出带格式 (条件格式、数据条、百分比) 的 Excel 报表。

    Attributes
    ----------
    overview_table : Union[pl.DataFrame, pd.DataFrame]
        全量概览大宽表，包含所有特征的统计指标。
    dq_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
        数据质量 (DQ) 指标的分组趋势表字典，key 为指标名 (如 'missing')。
    stats_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
        统计指标的分组趋势表字典，key 为指标名 (如 'mean')。
    """

    def __init__(
        self, 
        overview: Union[pl.DataFrame, pd.DataFrame],
        dq_tables: Dict[str, Union[pl.DataFrame, pd.DataFrame]],
        stats_tables: Dict[str, Union[pl.DataFrame, pd.DataFrame]]
    ) -> None:
        """
        初始化报告容器。

        Parameters
        ----------
        overview : Union[pl.DataFrame, pd.DataFrame]
            全量概览表。
        dq_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            DQ 指标趋势表字典。
        stats_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            统计指标趋势表字典。
        """
        self.overview_table: Union[pl.DataFrame, pd.DataFrame] = overview
        self.dq_tables: Dict[str, Union[pl.DataFrame, pd.DataFrame]] = dq_tables
        self.stats_tables: Dict[str, Union[pl.DataFrame, pd.DataFrame]] = stats_tables

    def get_profile_data(self) -> Tuple[
        Union[pl.DataFrame, pd.DataFrame], 
        Dict[str, Union[pl.DataFrame, pd.DataFrame]], 
        Dict[str, Union[pl.DataFrame, pd.DataFrame]]
    ]:
        """
        [API] 获取纯净的原始数据对象。
        
        用于后续的特征筛选 (Selector)、自定义分析或将数据传入其他系统。

        Returns
        -------
        Tuple
            包含三个元素的元组: (overview_df, dq_tables_dict, stats_tables_dict)。
        """
        return self.overview_table, self.dq_tables, self.stats_tables

    def _repr_html_(self) -> str:
        """
        [Magic Method] Jupyter Notebook 的富文本展示接口。
        
        当在 Jupyter 中直接打印该对象时，会显示一个包含概览信息和快捷方法的 HTML 面板。
        """
        df_ov = self.overview_table
        # 兼容 Polars/Pandas 获取行数
        n_feats = len(df_ov) if hasattr(df_ov, "__len__") else df_ov.height
        
        # 简单推断分组数量 (通过检查 missing 指标表的列数)
        sample_dq = self.dq_tables.get('missing')
        if sample_dq is not None:
            cols = sample_dq.columns
            n_cols = len(cols)
            # 减去固定列: feature, dtype, total
            n_groups = max(0, n_cols - 3)
        else:
            n_groups = 0

        # 构建面板 HTML 内容
        lines = []
        lines.append('<code>.show_overview()</code> 👈 <b>Start Here (Full Stats)</b>')
        
        dq_keys = list(self.dq_tables.keys())
        dq_links = [f"<code>.show_dq('{k}')</code>" for k in dq_keys]
        lines.append(f'DQ Trends: {", ".join(dq_links)}')
        
        stats_keys = list(self.stats_tables.keys())
        if stats_keys:
            display_keys = stats_keys
            suffix = ""
            stat_links = [f"<code>.show_trend('{k}')</code>" for k in display_keys]
            lines.append(f'Stat Trends: {", ".join(stat_links)}{suffix}')
        
        lines.append('<code>.write_excel()</code>')
        lines.append('<code>.get_profile_data()</code> <i>(For Feature Selection)</i>')

        return f"""
        <div style="border-left: 5px solid #2980b9; background-color: #f4f6f7; padding: 15px; border-radius: 0 5px 5px 0;">
            <h3 style="margin:0 0 10px 0; color:#2c3e50;">📊 Mars Profile Report</h3>
            <div style="display: flex; gap: 20px; margin-bottom: 10px; color: #555;">
                <div><strong>🏷️ Features:</strong> {n_feats}</div>
                <div><strong>📅 Groups:</strong> {n_groups}</div>
            </div>
            <div style="font-size:0.9em; line-height:1.8; color:#7f8c8d; border-top: 1px solid #e0e0e0; padding-top: 8px;">
                { "<br>".join(lines) }
            </div>
        </div>
        """

    def show_overview(self) -> "pd.io.formats.style.Styler":
        """
        展示全量概览大宽表。
        
        采用 'RdYlGn_r' (红-黄-绿 反转) 色系：
        - 高缺失率/高单一值率 -> 红色 (警示)
        - 低缺失率 -> 绿色 (健康)

        Returns
        -------
        pd.io.formats.style.Styler
            应用了热力图、Sparkline 字体样式的 Pandas Styler 对象。
        """
        return self._get_styler(
            self.overview_table, 
            title="Dataset Overview", 
            cmap="RdYlGn_r", 
            # 仅对特定的 DQ 指标应用热力图，避免污染其他数值列
            subset_cols=["missing_rate", "zeros_rate", "unique_rate", "top1_ratio"],
            fmt_as_pct=False # Overview 表混合了百分比和普通数值，需自动判断
        )

    def show_dq(self, metric: str) -> "pd.io.formats.style.Styler":
        """
        展示数据质量 (DQ) 指标趋势。
        
        Parameters
        ----------
        metric : str
            DQ 指标名 ('missing', 'zeros', 'unique', 'top1')。

        Returns
        -------
        pd.io.formats.style.Styler
            应用了格式化的 Pandas Styler 对象。
        """
        if metric not in self.dq_tables: raise ValueError(f"Unknown DQ metric: {metric}")
        return self._get_styler(
            self.dq_tables[metric], 
            title=f"DQ Trend: {metric}", 
            cmap="RdYlGn_r",
            fmt_as_pct=True # DQ 指标全为百分比，强制格式化
        )

    def show_trend(self, metric: str) -> "pd.io.formats.style.Styler":
        """
        展示统计指标趋势 (含稳定性监控)。
        
        Parameters
        ----------
        metric : str
            统计指标名 ('mean', 'std', 'max' 等)。

        Returns
        -------
        pd.io.formats.style.Styler
            应用了 Data Bars (用于 CV/Stability) 的 Pandas Styler 对象。
        """
        if metric not in self.stats_tables: raise ValueError(f"Unknown Stat metric: {metric}")
        return self._get_styler(
            self.stats_tables[metric], 
            title=f"Stat Trend: {metric}", 
            cmap="Blues", 
            add_bars=True, # 启用 Data Bars 显示 CV/Stability
            fmt_as_pct=False
        )

    def write_excel(self, path: str = "mars_report.xlsx") -> None:
        """
        将完整报告导出为 Excel 文件。
        
        该方法不仅导出数据，还会保留所有的视觉样式，包括：
        - 条件格式 (热力图)
        - 数据条 (Data Bars)
        - **百分比数字格式** (关键修复点，确保 Excel 中是数值而非文本)
        - 列宽自适应

        Parameters
        ----------
        path : str, default "mars_report.xlsx"
            导出文件的路径。
        """
        logger.info(f"📊 Exporting to {path}...")
        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                # 1. 导出 Overview Sheet
                if (styler := self.show_overview()) is not None:
                    styler.to_excel(writer, sheet_name="Overview", index=False)
                
                # 2. 导出 DQ Sheets (循环所有 DQ 指标)
                for name in self.dq_tables:
                    if (styler := self.show_dq(name)) is not None:
                        styler.to_excel(writer, sheet_name=f"DQ_{name}", index=False)
                
                # 3. 导出 Stat Sheets (循环所有统计指标)
                for name in self.stats_tables:
                    if (styler := self.show_trend(name)) is not None:
                        sheet_name = f"Trend_{name.capitalize()}"
                        styler.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # 特殊处理：使用 xlsxwriter 原生接口添加 Data Bars (Pandas Styler 对 DataBars 的导出支持有限)
                        # 我们需要重新获取 DataFrame 来定位 group_cv 列
                        df = self._to_pd(self.stats_tables[name])
                        if "group_cv" in df.columns:
                            worksheet = writer.sheets[sheet_name]
                            # 获取列索引 (Pandas 导出默认无 index，所以直接是 DataFrame 的列序)
                            col_idx = df.columns.get_loc("group_cv")
                            # 应用红色数据条到 group_cv 列
                            worksheet.conditional_format(1, col_idx, len(df), col_idx, {
                                'type': 'data_bar', 'bar_color': '#FF9999', 'bar_solid': True,
                                'min_type': 'num', 'min_value': 0, 'max_type': 'num', 'max_value': 1
                            })
                            
                # 4. 自动调整列宽
                for sheet in writer.sheets.values():
                    sheet.autofit()
            logger.info("✅ Done.")
        except Exception as e:
            logger.error(f"Failed to export Excel: {e}")

    # --- Internal Helpers ---
    
    def _to_pd(self, df: Any) -> pd.DataFrame:
        """
        [Helper] 确保转换为 Pandas DataFrame。
        
        Pandas Styler 只能工作在 Pandas DataFrame 上，因此如果是 Polars 对象需转换。
        """
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        return df

    def _get_styler(
        self, 
        df_input: Any, 
        title: str, 
        cmap: str, 
        subset_cols: List[str] = None, 
        add_bars: bool = False, 
        fmt_as_pct: bool = False
    ) -> "pd.io.formats.style.Styler":
        """
        [Helper] 通用样式生成器。
        
        负责生成统一风格的 Pandas Styler 对象，包含热力图、数字格式化和特殊字体设置。

        Parameters
        ----------
        df_input : Any
            输入 DataFrame (Polars 或 Pandas)。
        title : str
            表格标题 (Caption)。
        cmap : str
            热力图颜色映射 (如 'RdYlGn_r', 'Blues')。
        subset_cols : List[str], optional
            指定应用热力图的列名列表。如果为 None，则自动对所有数值列(排除元数据)应用。
        add_bars : bool, default False
            是否为 'group_cv' 列添加数据条 (Data Bars)。
        fmt_as_pct : bool, default False
            - True: 强制将除元数据外的所有数值列格式化为百分比 (DQ 趋势表模式)。
            - False: 智能判断，仅对列名包含 'rate'/'ratio' 的列应用百分比 (Overview/Stats 模式)。

        Returns
        -------
        pd.io.formats.style.Styler
            配置好的 Styler 对象。
        """
        if df_input is None: return None
        df = self._to_pd(df_input)
        if df.empty: return None

        # 元数据列，不参与热力图也不参与格式化
        # [修改] 增加 "distribution" 到排除列表，防止 Sparkline 被当作数值处理
        exclude = ["feature", "dtype", "group_var", "group_cv", "distribution"]
        
        # 1. 确定应用热力图的列
        if subset_cols:
            gradient_cols = [c for c in subset_cols if c in df.columns]
        else:
            gradient_cols = [c for c in df.columns if c not in exclude]

        styler = df.style.set_caption(f"<b>{title}</b>").hide(axis="index")
        
        # 2. 应用热力图 (Gradient)
        if gradient_cols:
            styler = styler.background_gradient(cmap=cmap, subset=gradient_cols, axis=None)
        
        # 3. 应用数据条 (Data Bars for Stability)
        if add_bars and "group_cv" in df.columns:
            styler = styler.bar(subset=["group_cv"], color='#ff9999', vmin=0, vmax=1, width=90)
            styler = styler.format("{:.4f}", subset=["group_cv", "group_var"])

        # 4. 数值格式化逻辑
        #    注意：这里定义的 format string 会被传入 Excel，使其显示为真正的数字而非文本。
        num_cols = df.select_dtypes(include=['number']).columns
        # 排除非数据列
        data_cols = [c for c in num_cols if c not in ["group_var", "group_cv", "distribution"]]

        # [关键修复] 使用 Pandas Styler 支持的格式化字符串
        # "{:.2%}" 在导出 Excel 时会被正确映射为百分比格式 (0.00%)
        pct_format = "{:.2%}"  
        float_format = "{:.2f}"

        if fmt_as_pct:
            # 强制模式 (DQ Trend): 所有数据列都是百分比
            if data_cols:
                styler = styler.format(pct_format, subset=data_cols)
        else:
            # 自动模式 (Overview / Stats): 根据列名智能判断
            pct_cols = [c for c in df.columns if "rate" in c or "ratio" in c]
            if pct_cols:
                styler = styler.format(pct_format, subset=pct_cols)
            
            float_cols = [c for c in data_cols if c not in pct_cols]
            if float_cols:
                styler = styler.format(float_format, subset=float_cols)
        
        # 5. 针对 Sparkline (distribution) 列的特殊样式
        #    强制使用 Monospace 字体，确保字符画在 Jupyter 中对齐；设置颜色为深蓝色
        if "distribution" in df.columns:
            styler = styler.set_table_styles([
                {'selector': '.col_distribution', 'props': [
                    ('font-family', 'monospace'), 
                    ('color', '#1f77b4'),
                    ('font-weight', 'bold'),
                    ('text-align', 'left')
                ]}
            ], overwrite=False)

        # 全局样式 (表头对齐、字体大小)
        styler = styler.set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'left'), ('background-color', '#f0f2f5'), ('color', '#333')]},
            {'selector': 'caption', 'props': [('font-size', '1.2em'), ('padding', '10px 0'), ('color', '#2c3e50')]}
        ], overwrite=False)

        return styler