"""分箱评估报告对象。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from mars.compute import RiskCorrBaseline, to_pandas_frame
from mars.reporting._binning_excel import _BinningExcelWriter
from mars.reporting._binning_html import _BinningHtmlRenderer
from mars.reporting._binning_plot import _BinningPlotRenderer
from mars.reporting._types import MarsHtmlRenderResult

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class MarsBinningReport:
    """
    特征效能与稳定性评估报告容器。

    管理 `MarsBinEvaluator` 产出的特征级汇总、分箱明细和指标趋势表，并提供表格读取、
    富文本视图、Excel 和 HTML 导出方法。

    Attributes
    ----------
    summary_table : DataFrame
        特征级汇总评估表。

    trend_tables : dict of str to DataFrame
        核心评估指标趋势表字典。

    detail_table : DataFrame
        分箱明细表。

    group_col : str
        内部挂载的分组维度标识。

    Notes
    -----
    `show_summary` 与 `show_trend` 使用 Pandas Styler 显示指标色带和数据条。`write_excel()`
    和 `write_html()` 分别生成电子表格和交互式 HTML 报告。

    Examples
    --------
    >>> import polars as pl
    >>> summary = pl.DataFrame({"feature": ["age"], "iv": [0.12], "ks": [18.0]})
    >>> detail = pl.DataFrame({"feature": ["age"], "bin_index": [0], "count": [100]})
    >>> trend_tables = {"psi": pl.DataFrame({"feature": ["age"], "202601": [0.01]})}
    >>> report = MarsBinningReport(summary, trend_tables, detail, group_col="month")
    >>> report.get_evaluation_data()[0].height
    1
    """

    def __init__(
        self,
        summary_table: Union[pl.DataFrame, pd.DataFrame],
        trend_tables: Dict[str, Union[pl.DataFrame, pd.DataFrame]],
        detail_table: Union[pl.DataFrame, pd.DataFrame],
        group_col: str | None = None,
        detail_group_col: str | None = None,
        feature_data_source: Dict[str, str] | None = None,
        dt_col: str | None = None,
        missing_by_day_table: Union[pl.DataFrame, pd.DataFrame] | None = None,
        risk_corr_reference_table: Union[pl.DataFrame, pd.DataFrame] | None = None,
        report_meta: Dict[str, Any] | None = None,
    ) -> None:
        """
        初始化报告容器。

        Parameters
        ----------
        summary_table : Union[pl.DataFrame, pd.DataFrame]
            特征级汇总表。
        trend_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            指标趋势表字典。
        detail_table : Union[pl.DataFrame, pd.DataFrame]
            最细粒度的分箱明细表。
        group_col : str | None
            公开分组语义列名（例如 `'month'` 或 `'vintage'`）。
        detail_group_col : str | None
            明细表内部实际使用的分组列名。未显式传入时默认沿用 ``group_col``。
        feature_data_source : Dict[str, str] | None
            特征到数据源标签的映射。
        dt_col : str | None
            原始日期列名。
        missing_by_day_table : Union[pl.DataFrame, pd.DataFrame] | None
            按日汇总的缺失率明细表。
        risk_corr_reference_table : Union[pl.DataFrame, pd.DataFrame] | None
            报告内部保存的 RC 参考坏率表，供图表与明细复用同一口径。
        report_meta : Dict[str, Any] | None
            报告元信息，例如目标列、绘图配置或上下文标签。
        """
        # 直接存储原始数据，不再强制命名为 _pl，以支持多种类型
        self._summary = summary_table
        self._trend_dict = trend_tables
        self._detail = detail_table
        self.group_col = group_col
        self._detail_group_col = detail_group_col or group_col
        self.feature_data_source = feature_data_source or {}
        self.dt_col = dt_col
        self._missing_by_day = missing_by_day_table
        self._risk_corr_reference = risk_corr_reference_table
        self._report_meta = report_meta or {}

    @property
    def summary_table(self) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        返回特征汇总评估表。

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            与构造时输入类型一致的汇总表。

        Examples
        --------
        >>> import polars as pl
        >>> summary = pl.DataFrame({"feature": ["age"], "iv": [0.12]})
        >>> report = MarsBinningReport(summary, {}, pl.DataFrame())
        >>> report.summary_table.height
        1
        """
        return self._summary

    @property
    def trend_tables(self) -> Dict[str, Union[pl.DataFrame, pd.DataFrame]]:
        """
        返回指标趋势表字典。

        Returns
        -------
        dict of str to pl.DataFrame or pd.DataFrame
            键为指标名称，值为对应趋势宽表。

        Examples
        --------
        >>> import polars as pl
        >>> trend = {"psi": pl.DataFrame({"feature": ["age"], "2026-01": [0.01]})}
        >>> report = MarsBinningReport(pl.DataFrame(), trend, pl.DataFrame())
        >>> sorted(report.trend_tables)
        ['psi']
        """
        return self._trend_dict

    @property
    def detail_table(self) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        返回分箱明细表。

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            与构造时输入类型一致的分箱明细表。

        Examples
        --------
        >>> import polars as pl
        >>> detail = pl.DataFrame({"feature": ["age"], "bin_index": [0]})
        >>> report = MarsBinningReport(pl.DataFrame(), {}, detail)
        >>> report.detail_table.height
        1
        """
        return self._detail

    @property
    def missing_by_day_table(self) -> Union[pl.DataFrame, pd.DataFrame] | None:
        """
        返回按日聚合的缺失明细表。

        Returns
        -------
        pl.DataFrame or pd.DataFrame or None
            若评估流程生成了按日缺失统计，则返回对应表；否则返回 ``None``。

        Examples
        --------
        >>> import polars as pl
        >>> missing = pl.DataFrame({"feature": ["age"], "date": ["2026-01-01"], "missing_rate": [0.0]})
        >>> report = MarsBinningReport(pl.DataFrame(), {}, pl.DataFrame(), missing_by_day_table=missing)
        >>> report.missing_by_day_table.height
        1
        """
        return self._missing_by_day

    @property
    def report_meta(self) -> Dict[str, Any]:
        """
        返回报告元信息字典。

        Returns
        -------
        dict of str to Any
            生成报告时记录的辅助元数据。

        Examples
        --------
        >>> import polars as pl
        >>> report = MarsBinningReport(pl.DataFrame(), {}, pl.DataFrame(), report_meta={"target": "y"})
        >>> report.report_meta["target"]
        'y'
        """
        return self._report_meta

    @property
    def risk_corr_reference_table(self) -> Union[pl.DataFrame, pd.DataFrame] | None:
        """
        返回 RC 参考坏率表。

        Returns
        -------
        pl.DataFrame or pd.DataFrame or None
            报告生成时保存的 RC 参考表。
        """
        return self._risk_corr_reference

    @property
    def detail_group_col(self) -> str | None:
        """
        返回明细表内部使用的分组列名。

        Returns
        -------
        str | None
            分箱明细表中的真实分组列名。
        """
        return self._detail_group_col

    def get_evaluation_data(self) -> Tuple[
        Union[pl.DataFrame, pd.DataFrame],
        Dict[str, Union[pl.DataFrame, pd.DataFrame]],
        Union[pl.DataFrame, pd.DataFrame]
    ]:
        """
        获取评估报告的原始数据。

        Returns
        -------
        tuple
            依次返回 ``(summary_table, trend_tables, detail_table)``，
            且各对象类型与构造时输入保持一致。

        Examples
        --------
        >>> import polars as pl
        >>> summary = pl.DataFrame({"feature": ["age"], "iv": [0.12]})
        >>> detail = pl.DataFrame({"feature": ["age"], "bin_index": [0]})
        >>> report = MarsBinningReport(summary, {}, detail)
        >>> report.get_evaluation_data()[0].height
        1
        """
        return self.summary_table, self.trend_tables, self.detail_table

    def _repr_html_(self) -> str:
        """返回 Jupyter 环境下的报告摘要 HTML。"""
        return _BinningHtmlRenderer(self)._repr_html_()

    def write_excel(
        self,
        path: str = "mars_bin_report.xlsx",
        engine: str = "openpyxl",
    ) -> None:
        """
        导出分箱评估 Excel 报告。

        Parameters
        ----------
        path : str
            输出 Excel 文件路径。
        engine : str
            Excel 写入引擎，支持 ``"auto"``、``"xlwings"`` 和 ``"openpyxl"``。

        Returns
        -------
        None
            方法仅产生文件写入副作用。

        Raises
        ------
        ValueError
            当 ``engine`` 不受支持，或模板列与 ``detail_table`` 不一致时抛出。
        RuntimeError
            当底层 Excel 写入流程失败时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> report = MarsBinningReport(
        ...     pl.DataFrame({"feature": ["age"], "iv": [0.1]}),
        ...     {},
        ...     pl.DataFrame({"feature": ["age"], "bin_index": [0], "count": [10]}),
        ... )
        >>> with TemporaryDirectory() as tmp:
        ...     report.write_excel(str(Path(tmp) / "report.xlsx"), engine="openpyxl") is None
        True
        """
        try:
            _BinningExcelWriter(self).write_excel(path=path, engine=engine)
        except (ValueError, RuntimeError):
            raise

    def write_html(
        self,
        path: str = "mars_bin_report.html",
        *,
        report_name: str = "MARS Evaluation Report",
        max_plots: int = 500,
        chart_embed_mode: Literal["auto", "inline", "asset"] = "auto",
        sort_by: str = "iv",
        ascending: bool = False,
        include_summary: bool = True,
        include_trends: bool = True,
        include_detail: bool = True,
        include_charts: bool = True,
    ) -> None:
        """
        导出支持页面切换和大规模图表懒加载的交互式 HTML 报告。

        Parameters
        ----------
        path : str
            输出 HTML 文件路径。
        report_name : str
            HTML 页面标题和报告名称。
        max_plots : int
            每个 target 的图表区域最多展示的特征数量，默认 500。
        chart_embed_mode : Literal["auto", "inline", "asset"]
            图表图片嵌入模式。``auto`` 在图表数量超过 50 张时写入旁路资产并懒加载；
            ``inline`` 强制内嵌；``asset`` 强制写入资产目录。
        sort_by : str
            图表和汇总视图使用的排序指标。
        ascending : bool
            是否按 ``sort_by`` 升序排序。
        include_summary : bool
            是否包含汇总表区域。
        include_trends : bool
            是否包含趋势分析区域。
        include_detail : bool
            是否包含明细区域。
        include_charts : bool
            是否包含图表区域。

        Returns
        -------
        None
            方法仅产生文件写入副作用。

        Examples
        --------
        >>> import polars as pl
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> report = MarsBinningReport(pl.DataFrame({"feature": ["age"], "iv": [0.1]}), {}, pl.DataFrame())
        >>> with TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "report.html"
        ...     report.write_html(str(path), include_charts=False)
        ...     path.exists()
        True
        """
        _BinningHtmlRenderer(self).write_html(
            path=path,
            report_name=report_name,
            max_plots=max_plots,
            chart_embed_mode=chart_embed_mode,
            sort_by=sort_by,
            ascending=ascending,
            include_summary=include_summary,
            include_trends=include_trends,
            include_detail=include_detail,
            include_charts=include_charts,
        )

    def build_risk_trend_figures(
        self,
        features: str | List[str] | None = None,
        *,
        target: str | List[str] | None = None,
        risk_corr_baseline: RiskCorrBaseline | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        sort_by: str = "iv",
        ascending: bool = False,
        max_plots: int = 20,
        dpi: int = 150,
    ) -> list[Figure]:
        """
        构建风险趋势图对象。

        Parameters
        ----------
        features : str | List[str] | None
            需要绘图的特征名称。传入 ``None`` 时，按 ``sort_by`` 和 ``max_plots`` 自动选择。
        target : str | List[str] | None
            多目标报告下需要生成的目标列名。传入 ``None`` 时生成全部目标。
        risk_corr_baseline : RiskCorrBaseline | None
            绘图阶段使用的 RC 基准；传入 ``None`` 时沿用报告生成时保存的口径。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式，分别表示件数口径、金额口径和双线同屏。
        sort_by : str
            未显式指定 ``features`` 时的特征排序字段。
        ascending : bool
            是否按 ``sort_by`` 升序选择特征。
        max_plots : int
            未显式指定 ``features`` 时最多生成的特征数量。
        dpi : int
            设置到返回 figure 上的图像分辨率。

        Returns
        -------
        list[Figure]
            Matplotlib 图对象列表。调用方负责在不再使用后关闭 figure。

        Raises
        ------
        ValueError
            当报告明细为空、分组列缺失、目标不存在或风险线模式无效时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> report = MarsBinningReport(pl.DataFrame({"feature": ["age"], "iv": [0.1]}), {}, pl.DataFrame())
        >>> callable(report.build_risk_trend_figures)
        True
        """
        try:
            return _BinningPlotRenderer(self).build_risk_trend_figures(
                features=features,
                target=target,
                risk_corr_baseline=risk_corr_baseline,
                show_risk=show_risk,
                sort_by=sort_by,
                ascending=ascending,
                max_plots=max_plots,
                dpi=dpi,
            )
        except ValueError:
            raise

    def save_risk_trend_images(
        self,
        output_dir: str | Path,
        features: str | List[str] | None = None,
        *,
        target: str | List[str] | None = None,
        image_format: Literal["svg", "png"] = "svg",
        filename_prefix: str = "risk_trend",
        overwrite: bool = True,
        dpi: int = 150,
        risk_corr_baseline: RiskCorrBaseline | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        sort_by: str = "iv",
        ascending: bool = False,
        max_plots: int = 20,
    ) -> list[Path]:
        """
        保存风险趋势图为图片文件。

        Parameters
        ----------
        output_dir : str | Path
            图片输出目录；目录不存在时会自动创建。
        features : str | List[str] | None
            需要绘图的特征名称。传入 ``None`` 时，按 ``sort_by`` 和 ``max_plots`` 自动选择。
        target : str | List[str] | None
            多目标报告下需要保存的目标列名。传入 ``None`` 时保存全部目标。
        image_format : Literal["svg", "png"]
            输出图片格式。
        filename_prefix : str
            输出文件名前缀。
        overwrite : bool
            当目标文件已存在时是否覆盖。为 ``False`` 时抛出 ``FileExistsError``。
        dpi : int
            PNG 输出分辨率。
        risk_corr_baseline : RiskCorrBaseline | None
            绘图阶段使用的 RC 基准；传入 ``None`` 时沿用报告生成时保存的口径。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式，分别表示件数口径、金额口径和双线同屏。
        sort_by : str
            未显式指定 ``features`` 时的特征排序字段。
        ascending : bool
            是否按 ``sort_by`` 升序选择特征。
        max_plots : int
            未显式指定 ``features`` 时最多保存的特征数量。

        Returns
        -------
        list[Path]
            已写出的图片路径列表。

        Raises
        ------
        FileExistsError
            当 ``overwrite=False`` 且目标文件已存在时抛出。
        ValueError
            当图像格式、报告明细、分组列或目标参数无效时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> report = MarsBinningReport(pl.DataFrame({"feature": ["age"], "iv": [0.1]}), {}, pl.DataFrame())
        >>> callable(report.save_risk_trend_images)
        True
        """
        try:
            return _BinningPlotRenderer(self).save_risk_trend_images(
                output_dir=output_dir,
                features=features,
                target=target,
                image_format=image_format,
                filename_prefix=filename_prefix,
                overwrite=overwrite,
                dpi=dpi,
                risk_corr_baseline=risk_corr_baseline,
                show_risk=show_risk,
                sort_by=sort_by,
                ascending=ascending,
                max_plots=max_plots,
            )
        except (FileExistsError, ValueError):
            raise

    def render_risk_trends_html(
        self,
        features: str | List[str] | None = None,
        *,
        target: str | List[str] | None = None,
        image_format: Literal["svg", "png"] = "svg",
        embed_mode: Literal["inline", "asset"] = "inline",
        output_dir: str | Path | None = None,
        relative_to: str | Path | None = None,
        include_title: bool = True,
        include_caption: bool = True,
        return_figures: bool = False,
        dpi: int = 150,
        filename_prefix: str = "risk_trend",
        overwrite: bool = True,
        risk_corr_baseline: RiskCorrBaseline | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        sort_by: str = "iv",
        ascending: bool = False,
        max_plots: int = 20,
    ) -> MarsHtmlRenderResult:
        """
        渲染可嵌入外部 HTML 报告的风险趋势图片段。

        Parameters
        ----------
        features : str | List[str] | None
            需要渲染的特征名称。传入 ``None`` 时，按 ``sort_by`` 和 ``max_plots`` 自动选择。
        target : str | List[str] | None
            多目标报告下需要渲染的目标列名。传入 ``None`` 时渲染全部目标。
        image_format : Literal["svg", "png"]
            图片格式。``"svg"`` 可直接内嵌 XML，``"png"`` 可生成 data URI 或资产文件。
        embed_mode : Literal["inline", "asset"]
            HTML 嵌入模式。``"inline"`` 生成单文件片段，``"asset"`` 写出图片并引用路径。
        output_dir : str | Path | None
            ``embed_mode="asset"`` 时的图片输出目录。
        relative_to : str | Path | None
            asset 模式下计算 ``img src`` 相对路径的基准目录。
        include_title : bool
            是否在片段中包含标题。
        include_caption : bool
            是否为每张图包含 target 和 feature 说明。
        return_figures : bool
            是否在结果中返回 Matplotlib figure。为 ``True`` 时调用方负责关闭 figure。
        dpi : int
            PNG 输出分辨率。
        filename_prefix : str
            asset 模式下的输出文件名前缀。
        overwrite : bool
            asset 模式下目标文件已存在时是否覆盖。
        risk_corr_baseline : RiskCorrBaseline | None
            绘图阶段使用的 RC 基准；传入 ``None`` 时沿用报告生成时保存的口径。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式，分别表示件数口径、金额口径和双线同屏。
        sort_by : str
            未显式指定 ``features`` 时的特征排序字段。
        ascending : bool
            是否按 ``sort_by`` 升序选择特征。
        max_plots : int
            未显式指定 ``features`` 时最多渲染的特征数量。

        Returns
        -------
        MarsHtmlRenderResult
            HTML 片段、写出的资产路径和可选 figure 列表。

        Raises
        ------
        FileExistsError
            当 asset 模式下 ``overwrite=False`` 且目标文件已存在时抛出。
        ValueError
            当嵌入模式、图像格式、报告明细、分组列或目标参数无效时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> report = MarsBinningReport(pl.DataFrame({"feature": ["age"], "iv": [0.1]}), {}, pl.DataFrame())
        >>> callable(report.render_risk_trends_html)
        True
        """
        try:
            return _BinningPlotRenderer(self).render_risk_trends_html(
                features=features,
                target=target,
                image_format=image_format,
                embed_mode=embed_mode,
                output_dir=output_dir,
                relative_to=relative_to,
                include_title=include_title,
                include_caption=include_caption,
                return_figures=return_figures,
                dpi=dpi,
                filename_prefix=filename_prefix,
                overwrite=overwrite,
                risk_corr_baseline=risk_corr_baseline,
                show_risk=show_risk,
                sort_by=sort_by,
                ascending=ascending,
                max_plots=max_plots,
            )
        except (FileExistsError, ValueError):
            raise

    def plot_risk_trends(
        self,
        features: str | List[str] | None = None,
        *,
        target: str | List[str] | None = None,
        risk_corr_baseline: RiskCorrBaseline | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        sort_by: str = "iv",
        ascending: bool = False,
        max_plots: int = 20,
        dpi: int = 150,
        return_figures: bool = False,
    ) -> list[Figure] | None:
        """
        直接展示分箱风险趋势图。

        Parameters
        ----------
        features : str | List[str] | None
            需要绘图的特征名称。传入 ``None`` 时，按 ``sort_by`` 和 ``max_plots`` 自动选择。
        target : str | List[str] | None
            多目标报告下需要展示的目标列名。传入 ``None`` 时展示全部目标。
        risk_corr_baseline : RiskCorrBaseline | None
            绘图阶段使用的 RC 基准；传入 ``None`` 时沿用报告生成时保存的口径。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式，分别表示件数口径、金额口径和双线同屏。
        sort_by : str
            未显式指定 ``features`` 时的特征排序字段。
        ascending : bool
            是否按 ``sort_by`` 升序选择特征。
        max_plots : int
            未显式指定 ``features`` 时最多展示的特征数量。
        dpi : int
            图像显示分辨率。
        return_figures : bool
            是否返回 Matplotlib figure。为 ``True`` 时调用方负责关闭 figure。

        Returns
        -------
        list[Figure] or None
            默认返回 ``None`` 并关闭展示用图像；``return_figures=True`` 时返回 figure 列表。

        Raises
        ------
        ValueError
            当报告明细为空、分组列缺失或目标不存在时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> report = MarsBinningReport(pl.DataFrame({"feature": ["age"], "iv": [0.1]}), {}, pl.DataFrame())
        >>> callable(report.plot_risk_trends)
        True
        """
        try:
            return _BinningPlotRenderer(self).plot_risk_trends(
                features=features,
                target=target,
                risk_corr_baseline=risk_corr_baseline,
                show_risk=show_risk,
                sort_by=sort_by,
                ascending=ascending,
                max_plots=max_plots,
                dpi=dpi,
                return_figures=return_figures,
            )
        except ValueError:
            raise

    def show_summary(self,
                     features: Union[str, List[str]] | None = None
                     ) -> pd.io.formats.style.Styler:
        """
        展示特征汇总评分表。

        Parameters
        ----------
        features : Union[str, List[str]] | None
            需要展示的特征名称。若为 ``None``，展示全部特征。

        Returns
        -------
        pd.io.formats.style.Styler
            样式化后的特征汇总表。

        Examples
        --------
        >>> import polars as pl
        >>> summary = pl.DataFrame({"feature": ["age"], "iv": [0.12], "ks": [18.0]})
        >>> report = MarsBinningReport(summary, {}, pl.DataFrame())
        >>> hasattr(report.show_summary(features="age"), "to_html")
        True
        """
        df: pd.DataFrame = to_pandas_frame(self.summary_table).copy()

        # 特征筛选逻辑
        if features is not None:
            if isinstance(features, str):
                features = [features]
            df = df[df["feature"].isin(features)]

        # 多目标模式下，将 target 列提前，便于快速按目标查看结果。
        for t_col in ["target", "target_col", "y"]:
            if t_col in df.columns:
                cols = [t_col] + [c for c in df.columns if c != t_col]
                df = df[cols]
                break

        styler = df.style.set_caption("<b>Feature Performance Summary</b>").hide(axis="index")

        # 异常熔断：如果筛选后为空，直接返回表框架，避免底图渲染报错
        if df.empty:
            return styler

        if "psi_max" in df.columns:
            styler = styler.background_gradient(cmap="RdYlGn_r", subset=["psi_max"], vmin=0, vmax=0.25)

        if "iv" in df.columns:
            styler = styler.background_gradient(cmap="RdYlGn", subset=["iv"], vmin=0.02, vmax=0.2)
        if "auc" in df.columns:
            styler = styler.background_gradient(cmap="RdYlGn", subset=["auc"], vmin=0.5, vmax=0.65)
        if "ks" in df.columns:
            styler = styler.background_gradient(cmap="RdYlGn", subset=["ks"], vmin=5, vmax=20)

        if "rc_min" in df.columns:
            styler = styler.background_gradient(cmap="RdYlGn", subset=["rc_min"], vmin=0.5, vmax=1.0)

        if "mono" in df.columns:
            # coolwarm 色带: -1 为深蓝(单调递减)，0 为灰白(无单调性)，1 为深红(单调递增)
            styler = styler.background_gradient(cmap="coolwarm", subset=["mono"], vmin=-1, vmax=1)

        return styler.format("{:.4f}", subset=df.select_dtypes("number").columns)

    def show_trend(self,
                   metric: str,
                   features: Union[str, List[str]] | None = None,
                   group_ascending: bool = True,
                   sort_by: Union[str, List[str]] = "Total",
                   sort_ascending: bool = False) -> pd.io.formats.style.Styler:
        """
        展示指定指标的时间趋势热力图。

        渲染并返回一个带条件格式 (Conditional Formatting) 的 Pandas Styler 对象，
        用于直观分析特征在不同时间切片（或客群分组）下的指标波动趋势。内置了针对
        风控业务语义优化的专属色盘 (Colormap)。

        Parameters
        ----------
        metric : str
            需要展示的指标名称。支持的选项可通过 `self.trend_tables.keys()` 查看
            (通常包含 'psi', 'auc', 'ks', 'iv', 'bad_rate', 'risk_corr')。
        features : Union[str, List[str]] | None
            需要展示的特征名列表。若为 None，则展示所有特征。
        group_ascending : bool
            分组/时间切片列的排序方向 (横向)。True 表示正序（从左到右由旧到新 / 由小到大）。
        sort_by : Union[str, List[str]]
            特征行的排序依据列。默认按照全局表现 (Total) 排序。
        sort_ascending : bool
            特征行的排序方向 (纵向)。默认降序 (False)，即把表现最差/最好的特征排在最上面。

        Returns
        -------
        pd.io.formats.style.Styler
            渲染完成的热力图对象。在 Jupyter Notebook 环境下会自动渲染为精美表格。

        Raises
        ------
        ValueError
            当 ``metric`` 不在当前报告支持的趋势指标集合中时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> trend = pl.DataFrame({"feature": ["age"], "2026-01": [0.01], "Total": [0.01]})
        >>> report = MarsBinningReport(pl.DataFrame(), {"psi": trend}, pl.DataFrame())
        >>> hasattr(report.show_trend("psi", features="age"), "to_html")
        True
        """
        if metric not in self.trend_tables:
            raise ValueError(f"Unknown metric: {metric}. Options: {list(self.trend_tables.keys())}")

        # 转换为 Pandas 副本进行安全的样式处理
        df: pd.DataFrame = to_pandas_frame(self.trend_tables[metric]).copy()

        # 特征筛选逻辑
        if features is not None:
            if isinstance(features, str):
                features = [features]
            df = df[df["feature"].isin(features)]

        # 行排序：紧跟 sort_by 和 sort_ascending 语义
        if sort_by in df.columns or (isinstance(sort_by, list) and all(c in df.columns for c in sort_by)):
            df = df.sort_values(by=sort_by, ascending=sort_ascending)

        # 识别列类型并重排时间切片列
        meta_cols = ["feature", "dtype"]
        special_cols = ["Total"]
        time_cols = [c for c in df.columns if c not in meta_cols + special_cols]

        # 列排序：受 group_ascending 控制
        time_cols_sorted = sorted(time_cols, reverse=not group_ascending)

        # 组装最终的列顺序：元数据 -> 时间切片 -> 汇总列
        final_cols = (
            [c for c in meta_cols if c in df.columns]
            + time_cols_sorted
            + [c for c in special_cols if c in df.columns]
        )
        df = df[final_cols]

        # 基础表格样式初始化
        styler = df.style.set_caption(f"<b>Trend Analysis: {metric.upper()}</b>").hide(axis="index")
        styler = styler.set_properties(subset=["feature"], **{'text-align': 'left', 'font-weight': 'bold'})

        if df.empty:
            return styler # 如果筛选后为空，直接返回空表格框架，避免报错

        # 根据不同业务指标的阈值与方向，映射专属渐变色盘
        if metric == "psi":
            styler = styler.background_gradient(
                cmap="RdYlGn_r", subset=time_cols_sorted, vmin=0, vmax=0.25, axis=None
            )
        elif metric in ["auc", "ks", "iv"]:
            styler = styler.background_gradient(
                cmap="RdYlGn", subset=time_cols_sorted, axis=None
            )
        elif metric == "bad_rate":
            styler = styler.background_gradient(
                cmap="Blues", subset=time_cols_sorted, axis=None
            )
        elif metric == "risk_corr":
            styler = styler.background_gradient(
                cmap="RdYlGn", subset=time_cols_sorted, vmin=0.5, vmax=1.0, axis=None
            )

        # 统一数值精度
        format_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
        return styler.format("{:.4f}", subset=format_cols)
