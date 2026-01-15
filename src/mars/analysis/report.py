import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union, List, Any
from mars.utils.logger import logger

try:
    from IPython.display import display, HTML
except ImportError:
    display = None

class MarsProfileReport:
    """
    [报告容器] MarsProfileReport - 统一管理数据画像结果的展示与导出。
    
    该类作为 MarsDataProfiler 的输出容器，负责将原始的统计数据 (DataFrame)
    转换为适合阅读分析的格式。它支持两种主要的输出渠道：
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
            全量概览表，包含特征名、类型、分布图及各类统计指标。
        dq_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            数据质量 (DQ) 指标趋势表字典，包含缺失率、零值率等随分组维度的变化。
        stats_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            统计指标趋势表字典，包含均值、标准差等随分组维度的变化。
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
        overview_df : Union[pl.DataFrame, pd.DataFrame]
            全量概览大宽表。
        dq_tables_dict : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            DQ 指标趋势字典。
        stats_tables_dict : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            统计指标趋势字典。
        """
        return self.overview_table, self.dq_tables, self.stats_tables

    def _repr_html_(self) -> str:
        """
        [Internal] Jupyter Notebook 的富文本展示接口。
        
        当在 Jupyter 环境中直接打印此对象时，生成一个交互式的 HTML 控制面板。

        Returns
        -------
        str
            包含概览统计信息和操作指南的 HTML 字符串。
        """
        df_ov: Union[pl.DataFrame, pd.DataFrame] = self.overview_table
        
        # 统计特征总数
        n_feats: int = len(df_ov) if hasattr(df_ov, "__len__") else df_ov.height
        
        # 推断分组数量
        sample_dq: Optional[Union[pl.DataFrame, pd.DataFrame]] = self.dq_tables.get('missing')
        n_groups: int = 0
        if sample_dq is not None:
            n_cols: int = len(sample_dq.columns)
            # 减去固定列: feature, dtype, total
            n_groups = max(0, n_cols - 3)

        # 构建控制面板内容
        lines: List[str] = []
        lines.append('<code>.show_overview()</code> 👈 <b>Full Overview (Recommended)</b>')
        
        dq_keys: List[str] = list(self.dq_tables.keys())
        dq_links: List[str] = [f"<code>.show_dq('{k}')</code>" for k in dq_keys]
        lines.append(f'DQ Trends: {", ".join(dq_links)}')
        
        stats_keys: List[str] = list(self.stats_tables.keys())
        if stats_keys:
            stat_links: List[str] = [f"<code>.show_trend('{k}')</code>" for k in stats_keys]
            lines.append(f'Stats Trends: {", ".join(stat_links)}')
        
        lines.append('<code>.write_excel()</code> Export formatted report')
        lines.append('<code>.get_profile_data()</code> Get raw data for feature selection')

        return f"""
        <div style="border-left: 5px solid #2980b9; background-color: #f4f6f7; padding: 15px; border-radius: 0 5px 5px 0;">
            <h3 style="margin:0 0 10px 0; color:#2c3e50;">📊 Mars Data Profile Report</h3>
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
        
        采用 'RdYlGn_r' (红-黄-绿 反转) 色系展示数据质量指标：
        - 高缺失率/高单一值率 -> 红色 (警示风险)
        - 低缺失率 -> 绿色 (健康状态)

        Returns
        -------
        pd.io.formats.style.Styler
            配置了热力图、迷你图样式和数值格式化的 Pandas Styler 对象。
        """
        return self._get_styler(
            self.overview_table, 
            title="Dataset Overview", 
            cmap="RdYlGn_r", 
            subset_cols=["missing_rate", "zeros_rate", "unique_rate", "top1_ratio"],
            fmt_as_pct=False
        )

    def show_dq(self, metric: str) -> "pd.io.formats.style.Styler":
        """
        展示指定数据质量 (DQ) 指标的趋势表。
        
        Parameters
        ----------
        metric : str
            DQ 指标名称，可选：'missing', 'zeros', 'unique', 'top1'。

        Returns
        -------
        pd.io.formats.style.Styler
            针对百分比指标优化的 Pandas Styler 对象。

        Raises
        ------
        ValueError
            当输入的指标名称不在 dq_tables 中时抛出。
        """
        if metric not in self.dq_tables:
            raise ValueError(f"Unknown DQ metric: {metric}")
        return self._get_styler(
            self.dq_tables[metric], 
            title=f"DQ Trends: {metric}", 
            cmap="RdYlGn_r",
            fmt_as_pct=True
        )

    def show_trend(self, metric: str) -> "pd.io.formats.style.Styler":
        """
        展示指定统计指标的趋势表。
        
        针对稳定性指标 (group_cv) 会自动添加数据条 (Data Bars) 可视化。
        [新增] 针对 PSI 指标，使用红绿灯色系 (Red-Yellow-Green) 警示风险。

        Parameters
        ----------
        metric : str
            统计指标名称，例如：'mean', 'std', 'max', 'p50' 等。

        Returns
        -------
        pd.io.formats.style.Styler
            包含稳定性数据条展示的 Pandas Styler 对象。

        Raises
        ------
        ValueError
            当输入的指标名称不在 stats_tables 中时抛出。
        """
        if metric not in self.stats_tables:
            raise ValueError(f"Unknown stats metric: {metric}")
        
        # [修改] 针对 PSI 的特殊可视化逻辑
        cmap = "Blues"
        vmin, vmax = None, None
        
        if metric == "psi":
            # 使用红黄绿反转色系: 低(绿) -> 中(黄) -> 高(红)
            cmap = "RdYlGn_r"
            # [关键] 锚定 PSI 的绝对值范围，确保 0.25 以上呈现明显的红色
            # 0.0 (Green) --- 0.1 (Yellow) --- 0.25 (Red) --- 0.5 (Dark Red)
            vmin = 0.0
            vmax = 0.5 
        
        return self._get_styler(
            self.stats_tables[metric], 
            title=f"Stats Trend: {metric}", 
            cmap=cmap, 
            add_bars=True,
            fmt_as_pct=False,
            # [新增] 传入范围参数
            vmin=vmin,
            vmax=vmax
        )

    def write_excel(self, path: str = "mars_report.xlsx") -> None:
        """
        将分析结果完整导出为带视觉格式的 Excel 文件。
        
        导出内容包括：
        1. Overview (概览页): 包含特征分布热力图。
        2. DQ_{Metric} (质量趋势页): 包含缺失率等趋势。
        3. Trend_{Metric} (分布趋势页): 包含稳定性分析及数据条展示。

        Excel 特性：
        - 百分比数字格式。
        - 自动列宽适配。
        - 冻结表头样式。

        Parameters
        ----------
        path : str, default "mars_report.xlsx"
            导出文件的目标路径。
        """
        logger.info(f"📊 Exporting report to: {path}...")
        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                # 1. 导出概览页
                overview_styler: Optional["pd.io.formats.style.Styler"] = self.show_overview()
                if overview_styler is not None:
                    overview_styler.to_excel(writer, sheet_name="Overview", index=False)
                
                # 2. 导出 DQ 指标页
                for name in self.dq_tables:
                    dq_styler = self.show_dq(name)
                    if dq_styler is not None:
                        dq_styler.to_excel(writer, sheet_name=f"DQ_{name}", index=False)
                
                # 3. 导出统计指标页 (特别处理 Data Bars 和 PSI Colors)
                for name in self.stats_tables:
                    trend_styler = self.show_trend(name)
                    if trend_styler is not None:
                        sheet_name: str = f"Trend_{name.capitalize()}"
                        trend_styler.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        df_pd: pd.DataFrame = self._to_pd(self.stats_tables[name])
                        worksheet = writer.sheets[sheet_name]
                        
                        # [新增] 针对 PSI 的 Excel 条件格式 (3色阶)
                        # 规则: 0(绿) -> 0.1(黄) -> 0.25(红)
                        if name == "psi":
                            # 确定应用范围: 除去 feature, dtype, total, group_var, group_cv 之外的数据列
                            # 或者简单点，对所有数值格应用 (Excel 会自动忽略文本)
                            start_col_idx = 0
                            end_col_idx = len(df_pd.columns) - 1
                            
                            worksheet.conditional_format(1, start_col_idx, len(df_pd), end_col_idx, {
                                'type': '3_color_scale',
                                'min_type': 'num', 'min_value': 0,    'min_color': '#63BE7B', # Green
                                'mid_type': 'num', 'mid_value': 0.1,  'mid_color': '#FFEB84', # Yellow
                                'max_type': 'num', 'max_value': 0.25, 'max_color': '#F8696B'  # Red
                            })

                        # 通过 xlsxwriter 原生接口补全 Data Bars 导出支持 (针对 group_cv)
                        if "group_cv" in df_pd.columns:
                            col_idx: int = df_pd.columns.get_loc("group_cv")
                            # 应用红色渐变数据条
                            worksheet.conditional_format(1, col_idx, len(df_pd), col_idx, {
                                'type': 'data_bar', 
                                'bar_color': '#FF9999', 
                                'bar_solid': True,
                                'min_type': 'num', 'min_value': 0, 
                                'max_type': 'num', 'max_value': 1
                            })
                            
                # 4. 自动列宽调整
                for sheet in writer.sheets.values():
                    sheet.autofit()
                    
            logger.info("✅ Report exported successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to export Excel: {e}")

    def _to_pd(self, df: Any) -> pd.DataFrame:
        """
        [辅助方法] 确保数据转换为 Pandas DataFrame 格式。

        Parameters
        ----------
        df : Any
            输入数据，支持 Polars DataFrame 或 Pandas DataFrame。

        Returns
        -------
        pd.DataFrame
            转换后的 Pandas 对象。
        """
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        return df

    def _get_styler(
        self, 
        df_input: Any, 
        title: str, 
        cmap: str, 
        subset_cols: Optional[List[str]] = None, 
        add_bars: bool = False, 
        fmt_as_pct: bool = False,
        vmin: Optional[float] = None, # [新增] 支持自定义最小值
        vmax: Optional[float] = None  # [新增] 支持自定义最大值
    ) -> Optional["pd.io.formats.style.Styler"]:
        """
        [Internal] 通用样式生成器。
        
        负责构建统一的 Pandas Styler 对象，处理色彩映射、数值格式和 CSS 样式。

        Parameters
        ----------
        df_input : Any
            待格式化的 DataFrame。
        title : str
            表格标题 (Caption)。
        cmap : str
            色彩映射方案 (Matplotlib colormap)。
        subset_cols : List[str], optional
            指定应用渐变色的列。若为 None 则对所有可用数值列应用。
        add_bars : bool, default False
            是否在 'group_cv' 列上绘制数据条。
        fmt_as_pct : bool, default False
            是否强制将数值列显示为百分比。
        vmin : float, optional
            渐变色的下限值。
        vmax : float, optional
            渐变色的上限值。

        Returns
        -------
        Optional[pd.io.formats.style.Styler]
            配置完成的 Styler 对象；若输入为空则返回 None。
        """
        if df_input is None:
            return None
        df: pd.DataFrame = self._to_pd(df_input)
        if df.empty:
            return None

        # 元数据排除列表：不参与热力图染色和百分比格式化
        exclude_meta: List[str] = ["feature", "dtype", "group_mean", "group_var", "group_cv", "distribution"]
        
        # 1. 确定色彩渐变范围
        if subset_cols:
            gradient_cols: List[str] = [c for c in subset_cols if c in df.columns]
        else:
            gradient_cols = [c for c in df.columns if c not in exclude_meta]

        styler = df.style.set_caption(f"<b>{title}</b>").hide(axis="index")
        
        # 2. 应用热力图 (支持 vmin/vmax)
        if gradient_cols:
            # [修改] 传递 vmin/vmax 到 background_gradient
            styler = styler.background_gradient(
                cmap=cmap, 
                subset=gradient_cols, 
                axis=None,
                vmin=vmin,
                vmax=vmax
            )
        
        # 3. 应用数据条 (稳定性专用)
        if add_bars and "group_cv" in df.columns:
            styler = styler.bar(subset=["group_cv"], color='#ff9999', vmin=0, vmax=1, width=90)
            styler = styler.format("{:.4f}", subset=["group_cv", "group_var"])

        # 4. 数值格式化逻辑
        num_cols: pd.Index = df.select_dtypes(include=['number']).columns
        data_cols: List[str] = [c for c in num_cols if c not in ["group_var", "group_cv", "distribution"]]

        pct_format: str = "{:.2%}"  
        float_format: str = "{:.2f}"

        if fmt_as_pct:
            # 强制百分比模式 (DQ 模式)
            if data_cols:
                styler = styler.format(pct_format, subset=data_cols)
        else:
            # 智能判断模式 (Overview/Stats 模式)
            pct_cols: List[str] = [c for c in df.columns if "rate" in c or "ratio" in c]
            if pct_cols:
                styler = styler.format(pct_format, subset=pct_cols)
            
            float_cols: List[str] = [c for c in data_cols if c not in pct_cols]
            if float_cols:
                styler = styler.format(float_format, subset=float_cols)
        
        # 5. 分布迷你图 (Sparkline) 样式配置
        if "distribution" in df.columns:
            # 注入 CSS 确保等宽字体和颜色一致性
            styler = styler.set_table_styles([
                {'selector': '.col_distribution', 'props': [
                    ('font-family', 'monospace'), 
                    ('color', '#1f77b4'),
                    ('font-weight', 'bold'),
                    ('text-align', 'left')
                ]}
            ], overwrite=False)

        # 6. 全局表格外观配置
        styler = styler.set_table_styles([
            {
                'selector': 'th', 
                'props': [('text-align', 'left'), ('background-color', '#f0f2f5'), ('color', '#333')]
            },
            {
                'selector': 'caption', 
                'props': [('font-size', '1.2em'), ('padding', '10px 0'), ('color', '#2c3e50')]
            }
        ], overwrite=False)

        return styler