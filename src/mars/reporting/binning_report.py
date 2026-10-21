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
from mars.reporting._metadata import FeatureMetadata
from mars.reporting._query import ReportFrame, _ReportQuery
from mars.reporting._types import MarsHtmlRenderResult

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class MarsBinningReport(_ReportQuery):
    """
    保存特征区分度、分布稳定性和分箱风险的评估结果。

    保存 `MarsBinEvaluator` 已计算的特征汇总、分箱明细和指标趋势表。
    支持查询、Notebook 展示及 Excel、HTML 导出，查询时不重新分箱或计算指标。

    Attributes
    ----------
    summary_table : DataFrame
        特征级汇总评估表。

    trend_tables : dict of str to DataFrame
        指标名到趋势宽表的映射，分组值或时间段作为列。

    detail_table : DataFrame
        分箱明细表。

    group_col : str
        报告对外使用的分组列名；明细中的实际列名见 ``detail_group_col``。

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
        feature_metadata: FeatureMetadata | None = None,
        business_context: dict[str, Any] | None = None,
    ) -> None:
        """
        初始化报告容器。

        Parameters
        ----------
        summary_table : Union[pl.DataFrame, pd.DataFrame]
            已计算的特征级汇总表；多目标报告按目标和特征分别汇总。
        trend_tables : Dict[str, Union[pl.DataFrame, pd.DataFrame]]
            指标名到趋势宽表的映射，只提供已计算的指标，不补算缺少的表。
        detail_table : Union[pl.DataFrame, pd.DataFrame]
            按目标、特征、分组和箱保存的明细表，供查询、绘图和导出复用。
        group_col : str | None
            报告对外使用的分组列名，例如 ``"month"`` 或 ``"vintage"``；默认不指定。
        detail_group_col : str | None
            明细表内部实际使用的分组列名。未显式传入时默认沿用 ``group_col``。
        feature_data_source : Dict[str, str] | None
            旧版来源参数，格式为特征 ID 到数据源的映射，默认不额外登记来源。
            与 ``feature_metadata`` 的来源合并；同一特征来源冲突时报错。
        dt_col : str | None
            评估时使用的原始日期列名，默认不指定。绘图所需日期范围从报告元数据读取。
        missing_by_day_table : Union[pl.DataFrame, pd.DataFrame] | None
            按自然日统计的原始缺失率宽表，默认不包含。
        risk_corr_reference_table : Union[pl.DataFrame, pd.DataFrame] | None
            报告内部保存的 RC 参考坏率表，供图表与明细复用同一口径。
        report_meta : Dict[str, Any] | None
            本次评估的实际参数、目标、分箱来源、指标口径和诊断，默认使用空字典。
        feature_metadata : FeatureMetadata | None
            以英文特征 ID 为键的业务字典，或含 ``feature`` 列的 Pandas/Polars 表。
            支持 ``display_name``、``description``、``data_source`` 和 ``unit``；
            默认不补充业务解释，已有身份列中的有效信息仍会保留。
        business_context : dict[str, Any] | None
            JSON 业务上下文，仅用于解释结果，不改变统计。
            多标签定义按 ``labels`` 中的目标字段组织，未提供的信息标为 unknown。
        """
        # 保留输入表的 Pandas/Polars 类型，身份列在语义初始化时统一补齐。
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
        self._calculation_status: ReportFrame | None = None
        self._initialize_semantics(feature_metadata, business_context, feature_data_source)

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
            评估时记录的实际参数、计算口径、来源和诊断。

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

    def _query_tables(self) -> dict[str, ReportFrame]:
        """按公共目录命名分箱表，仅加入已生成的可选附表。"""
        tables: dict[str, ReportFrame] = {"summary": self.summary_table, "detail": self.detail_table}
        tables.update({f"trend.{k}": v for k, v in self.trend_tables.items()})
        if getattr(self, "_calculation_status", None) is not None:
            tables["calculation_status"] = self._calculation_status
        for name, table in [("missing_by_day", self.missing_by_day_table),
                            ("risk_corr_reference", self.risk_corr_reference_table)]:
            if table is not None:
                tables[name] = table
        return tables

    def _query_metadata(self) -> dict[str, Any]:
        """返回分箱评估运行元数据。"""
        return self.report_meta

    def get_evaluation_data(self) -> Tuple[
        Union[pl.DataFrame, pd.DataFrame],
        Dict[str, Union[pl.DataFrame, pd.DataFrame]],
        Union[pl.DataFrame, pd.DataFrame]
    ]:
        """
        返回报告中已计算的汇总、趋势和分箱明细。

        Returns
        -------
        tuple
            依次返回 ``(summary_table, trend_tables, detail_table)``，
            且各对象类型与构造时输入保持一致。返回报告中的对象，不包含原始样本。

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

        raw 与 binned 复用各引擎的同一 22 列模板，主展示统一为分箱 KS（0～100 点）。
        导出副本的 ``ks`` 按特征、目标和实际分组取 ``max(detail.ks_bin)``，供 KS 与
        综合指标透视读取；最终汇总使用对应 Total 的分箱 KS，最终 KS 趋势只保留首个目标。
        综合指标保留模板的 average 聚合，全部箱级 KS 未定义的组保持空值。
        原报告的 ``ks``、逐箱 ``ks_bin``、元数据和计算状态保持原样，不重新拟合或评估。
        口径说明分别记录源报告的 ``ks_method`` 与导出的 binned 口径；raw 诊断仍属于源报告。
        openpyxl 不计算公式或刷新透视缓存，透视由原生 Excel 打开后刷新。

        Parameters
        ----------
        path : str
            输出 Excel 文件路径。
        engine : str
            Excel 写入引擎，默认使用 ``"openpyxl"``，也支持 ``"auto"`` 和 ``"xlwings"``。

        Returns
        -------
        None
            写入报告文件，不返回数据。

        Raises
        ------
        ValueError
            当 ``engine`` 不受支持、缺少分箱 KS 的必要字段，或模板列与
            ``detail_table`` 不一致时抛出。
        RuntimeError
            当底层 Excel 写入流程失败时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> from mars.analysis import profile_risk
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> data = pl.DataFrame({"age": [20, 25, 30, 35], "target": [0, 1, 0, 1]})
        >>> report = profile_risk(
        ...     data, target="target", features=["age"], method="quantile", n_bins=2,
        ... ).report
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
        include_total: bool = True,
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
            图表图片嵌入模式。默认 ``auto`` 在图表数量超过 50 张时写入同级图片目录并懒加载；
            ``inline`` 强制内嵌；``asset`` 强制写入资产目录。
        sort_by : str
            图表特征与汇总视图的排序指标，默认按 ``"iv"`` 排序。
        ascending : bool
            是否按 ``sort_by`` 升序排序，默认降序。
        include_summary : bool
            是否包含汇总表区域，默认包含。
        include_trends : bool
            是否包含已生成的指标趋势区域，默认包含。
        include_detail : bool
            是否生成 Grouped Pivot 分箱明细区域及其导航、控件和数据。
            默认生成；不影响原报告明细、公共查询或图表内部读取明细。
        include_charts : bool
            是否生成原生趋势图区域，默认生成，与 ``include_detail`` 独立。
            启用时需要有效时间信息；False 时不解析绘图时间。
        include_total : bool
            是否在风险图中显示 Total 面板，默认显示；不改变全局统计或 RC 基准。

        Returns
        -------
        None
            写入报告文件，不返回数据。

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
            include_total=include_total,
        )

    def build_risk_trend_figures(
        self,
        features: str | List[str] | None = None,
        *,
        target: str | List[str] | None = None,
        risk_corr_baseline: RiskCorrBaseline | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        include_total: bool = True,
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
            显式指定时按传入顺序去重，只保留明细中的特征，不受 ``max_plots`` 限制。
        target : str | List[str] | None
            多目标报告下需要生成的目标列名。传入 ``None`` 时生成全部目标。
        risk_corr_baseline : RiskCorrBaseline | None
            绘图使用的 RC 基准。默认沿用报告保存的口径，未保存时使用 ``"total"``。
            ``"benchmark"`` 要求报告含已保存的参考表；显式传入时只影响图中 RC。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式，分别表示件数口径、金额口径和双线同屏。
        include_total : bool
            是否显示 Total 面板；不改变全局统计或 RC 基准。
        sort_by : str
            未显式指定 ``features`` 时的特征排序字段。
        ascending : bool
            是否按 ``sort_by`` 升序选择特征。
        max_plots : int
            未显式指定 ``features`` 时，每个目标最多选择的特征数，默认 20。
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
                include_total=include_total,
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
        include_total: bool = True,
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
            显式指定时按传入顺序去重，只保留明细中的特征，不受 ``max_plots`` 限制。
        target : str | List[str] | None
            多目标报告下需要保存的目标列名。传入 ``None`` 时保存全部目标。
        image_format : Literal["svg", "png"]
            输出图片格式，默认 SVG；支持 PNG。
        filename_prefix : str
            输出文件名前缀。
        overwrite : bool
            当目标文件已存在时是否覆盖。为 ``False`` 时抛出 ``FileExistsError``。
        dpi : int
            PNG 输出分辨率。
        risk_corr_baseline : RiskCorrBaseline | None
            绘图使用的 RC 基准。默认沿用报告保存的口径，未保存时使用 ``"total"``。
            ``"benchmark"`` 要求报告含已保存的参考表；显式传入时只影响图中 RC。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式，分别表示件数口径、金额口径和双线同屏。
        include_total : bool
            是否显示 Total 面板；不改变全局统计或 RC 基准。
        sort_by : str
            未显式指定 ``features`` 时的特征排序字段。
        ascending : bool
            是否按 ``sort_by`` 升序选择特征。
        max_plots : int
            未显式指定 ``features`` 时，每个目标最多选择的特征数，默认 20。

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
                include_total=include_total,
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
        include_total: bool = True,
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
            显式指定时按传入顺序去重，只保留明细中的特征，不受 ``max_plots`` 限制。
        target : str | List[str] | None
            多目标报告下需要渲染的目标列名。传入 ``None`` 时渲染全部目标。
        image_format : Literal["svg", "png"]
            图片格式。``"svg"`` 可直接内嵌 XML，``"png"`` 可生成 data URI 或资产文件。
        embed_mode : Literal["inline", "asset"]
            HTML 嵌入模式。``"inline"`` 生成单文件片段，``"asset"`` 写出图片并引用路径。
        output_dir : str | Path | None
            图片输出目录，``embed_mode="asset"`` 时必须提供；目录不存在时会自动创建。
        relative_to : str | Path | None
            asset 模式下计算 ``img src`` 相对路径的基准目录；默认直接使用图片路径。
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
            绘图使用的 RC 基准。默认沿用报告保存的口径，未保存时使用 ``"total"``。
            ``"benchmark"`` 要求报告含已保存的参考表；显式传入时只影响图中 RC。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式，分别表示件数口径、金额口径和双线同屏。
        include_total : bool
            是否显示 Total 面板；不改变全局统计或 RC 基准。
        sort_by : str
            未显式指定 ``features`` 时的特征排序字段。
        ascending : bool
            是否按 ``sort_by`` 升序选择特征。
        max_plots : int
            未显式指定 ``features`` 时，每个目标最多选择的特征数，默认 20。

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
                include_total=include_total,
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
        include_total: bool = True,
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
            显式指定时按传入顺序去重，只保留明细中的特征，不受 ``max_plots`` 限制。
        target : str | List[str] | None
            多目标报告下需要展示的目标列名。传入 ``None`` 时展示全部目标。
        risk_corr_baseline : RiskCorrBaseline | None
            绘图使用的 RC 基准。默认沿用报告保存的口径，未保存时使用 ``"total"``。
            ``"benchmark"`` 要求报告含已保存的参考表；显式传入时只影响图中 RC。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式，分别表示件数口径、金额口径和双线同屏。
        include_total : bool
            是否显示 Total 面板；不改变全局统计或 RC 基准。
        sort_by : str
            未显式指定 ``features`` 时的特征排序字段。
        ascending : bool
            是否按 ``sort_by`` 升序选择特征。
        max_plots : int
            未显式指定 ``features`` 时，每个目标最多选择的特征数，默认 20。
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
                include_total=include_total,
                sort_by=sort_by,
                ascending=ascending,
                max_plots=max_plots,
                dpi=dpi,
                return_figures=return_figures,
            )
        except ValueError:
            raise

    def show_summary(self,
                     features: Union[str, List[str]] | None = None,
                     *, sort_by: str | list[str] | None = None,
                     sort_ascending: bool = False,
                     columns: list[str] | None = None,
                     limit: int | None = None,
                     sources: str | list[str] | None = None,
                     ) -> pd.io.formats.style.Styler:
        """
        展示特征级评估汇总表。

        Parameters
        ----------
        features : Union[str, List[str]] | None
            需要展示的特征名称。若为 ``None``，展示全部特征。
        sort_by : str | list[str] | None
            特征行的排序字段，支持单列或多列；默认保留原顺序，排序先于列投影和行数限制。
        sort_ascending : bool
            是否按 ``sort_by`` 升序排列，默认降序；仅指定排序字段时生效。
        columns : list[str] | None
            展示字段。默认保留全部列；显式指定时只返回所列字段，并保持指定顺序。
        limit : int | None
            排序后的最大展示行数，默认不限制；筛选后或截取后为空时抛出 ``ValueError``。
        sources : str | list[str] | None
            按已登记来源筛选特征，默认不限制；与 ``features`` 取交集，未知来源报错。

        Returns
        -------
        pd.io.formats.style.Styler
            样式化后的特征汇总表。

        Raises
        ------
        ValueError
            汇总表为空或指定特征没有匹配行时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> summary = pl.DataFrame({"feature": ["age"], "iv": [0.12], "ks": [18.0]})
        >>> report = MarsBinningReport(summary, {}, pl.DataFrame())
        >>> hasattr(report.show_summary(features="age"), "to_html")
        True
        """
        df: pd.DataFrame = to_pandas_frame(self.get_table(
            "summary", features=features, columns=columns, sort_by=sort_by,
            descending=not sort_ascending, limit=limit, sources=sources,
        ))

        df = self._display_frame(df)
        # 多目标模式下，将 target 列提前，便于快速按目标查看结果。
        if columns is None:
            for t_col in ["target", "target_col", "y"]:
                if t_col in df.columns:
                    cols = [t_col] + [c for c in df.columns if c != t_col]
                    df = df[cols]
                    break

        styler = df.style.set_caption("<b>Feature Performance Summary</b>").hide(axis="index")

        if df.empty:
            raise ValueError("Requested binning summary contains no matching rows.")

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
            # mono 是秩相关系数：负值偏递减，正值偏递增，缺失值保留未定义状态。
            styler = styler.background_gradient(cmap="coolwarm", subset=["mono"], vmin=-1, vmax=1)

        return styler.format("{:.4f}", subset=df.select_dtypes("number").columns)

    def show_trend(self,
                   metric: str,
                   features: Union[str, List[str]] | None = None,
                   group_ascending: bool = True,
                   sort_by: Union[str, List[str]] = "Total",
                   sort_ascending: bool = False,
                   *, columns: list[str] | None = None,
                   limit: int | None = None,
                   sources: str | list[str] | None = None) -> pd.io.formats.style.Styler:
        """
        展示指定指标在时间切片或客群分组中的变化。

        返回带条件格式的 Pandas Styler，色阶方向按指标设置。
        查询只读取已有趋势表，不重新聚合分箱明细。

        Parameters
        ----------
        metric : str
            已生成的趋势指标名，可用 ``self.trend_tables.keys()`` 查看。
            新报告通常包含 ``psi``、``auc``、``ks``、``iv`` 和 ``risk_corr``。
            ``bad_rate`` 读取分箱明细；历史报告已有的同名趋势表仍可展示。
        features : Union[str, List[str]] | None
            原始英文特征 ID，默认展示所有特征。
        group_ascending : bool
            分组或时间切片列的横向排序方向，默认按列名升序。
            仅在 ``columns=None`` 时生效；身份列置前，``Total`` 置后。
        sort_by : Union[str, List[str]]
            特征行的排序字段，默认按 ``Total`` 排序，指定字段必须存在于趋势表。
        sort_ascending : bool
            是否按 ``sort_by`` 升序排列特征行，默认降序；指标大小的业务含义须单独判断。
        columns : list[str] | None
            展示字段。默认保留全部列并排列分组列；显式指定时保持所列字段及其顺序。
        limit : int | None
            排序后的最大展示行数，默认不限制；筛选后或截取后为空时抛出 ``ValueError``。
        sources : str | list[str] | None
            按已登记来源筛选特征，默认不限制；与 ``features`` 取交集，未知来源报错。

        Returns
        -------
        pd.io.formats.style.Styler
            可在 Jupyter Notebook 中显示的样式化趋势表。

        Raises
        ------
        ValueError
            指标、查询字段或来源无效，或查询结果为空时抛出。

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

        ordering = [sort_by] if isinstance(sort_by, str) else sort_by
        effective_sort = ordering
        df: pd.DataFrame = to_pandas_frame(self.get_table(
            f"trend.{metric}", features=features, columns=columns, sort_by=effective_sort,
            descending=not sort_ascending, limit=limit, sources=sources,
        ))

        df = self._display_frame(df)
        # 识别列类型并重排时间切片列
        meta_cols = ["target", "y", "feature", "display_name", "dtype", "data_source"]
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
        if columns is None:
            df = df[final_cols]

        # 基础表格样式初始化
        styler = df.style.set_caption(f"<b>Trend Analysis: {metric.upper()}</b>").hide(axis="index")
        if "feature" in df.columns:
            styler = styler.set_properties(subset=["feature"], **{'text-align': 'left', 'font-weight': 'bold'})

        if df.empty:
            raise ValueError("Requested binning trend contains no matching rows.")

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
