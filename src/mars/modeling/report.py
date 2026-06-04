"""建模评估报告容器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


class MarsModelingReport:
    """
    建模评估报告的数据容器。

    Parameters
    ----------
    summary_table : pandas.DataFrame
        分数据集的核心指标汇总表。
    caption : str, default "MARS Model Evaluation"
        Notebook 样式展示标题。
    detail_tables : dict of str to pandas.DataFrame, optional
        ROC、KS、PSI、风险水位图等轻量明细表。
    metadata : dict, optional
        训练配置、版本、特征重要性等报告元数据。

    Attributes
    ----------
    summary_table : pandas.DataFrame
        分数据集的核心指标汇总表。
    caption : str
        Notebook 样式展示标题。
    detail_tables : dict of str to pandas.DataFrame
        ROC、KS、PSI、风险水位图等轻量明细表。
    metadata : dict
        训练配置、版本、特征重要性等报告元数据。

    Examples
    --------
    >>> import pandas as pd
    >>> report = MarsModelingReport(pd.DataFrame({"metric": [1.0]}))
    >>> report.caption
    'MARS Model Evaluation'
    """

    def __init__(
        self,
        summary_table: pd.DataFrame,
        caption: str = "MARS Model Evaluation",
        detail_tables: Dict[str, pd.DataFrame] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        self.summary_table: pd.DataFrame = summary_table
        self.caption: str = caption
        self.detail_tables: Dict[str, pd.DataFrame] = dict(detail_tables or {})
        self.metadata: Dict[str, Any] = dict(metadata or {})

    @property
    def styled_summary(self) -> Any:
        """
        返回适合 Notebook 展示的 Pandas Styler。

        Returns
        -------
        Any
            Pandas Styler 对象。

        Examples
        --------
        >>> report = MarsModelingReport(pd.DataFrame({"metric": [1.0]}))
        >>> report.styled_summary.caption
        'MARS Model Evaluation'
        """
        all_cols = list(self.summary_table.columns)
        numeric_cols = [
            col
            for col in all_cols
            if isinstance(col, tuple)
            and col[1]
            in {
                "New AUC",
                "New KS",
                "Bench AUC",
                "Bench KS",
                "AUC Diff",
                "KS Diff",
                "LogLoss",
                "Brier",
                "Score PSI",
                "Top 10% Capture",
                "Top 20% Capture",
            }
        ]
        rate_cols = [
            col
            for col in all_cols
            if isinstance(col, tuple) and col[1] == "Bad Rate"
        ]
        count_cols = [
            col
            for col in all_cols
            if isinstance(col, tuple) and col[1] in {"Total Count", "Good", "Bad"}
        ]
        time_cols = [
            col
            for col in all_cols
            if isinstance(col, tuple) and col[1] in {"Start Time", "End Time"}
        ]

        styler = self.summary_table.style
        if numeric_cols:
            styler = styler.format("{:.2f}", subset=numeric_cols, na_rep="-")
        if rate_cols:
            styler = styler.format("{:.2%}", subset=rate_cols, na_rep="-")
        if count_cols:
            styler = styler.format("{:,.0f}", subset=count_cols, na_rep="-")
        if time_cols:
            styler = styler.format("{:%Y-%m-%d}", subset=time_cols, na_rep="-")

        return styler.set_caption(self.caption)

    def show_summary(self) -> Any:
        """
        返回样式化汇总表。

        Returns
        -------
        Any
            Pandas Styler 对象。

        Examples
        --------
        >>> report = MarsModelingReport(pd.DataFrame({"metric": [1.0]}))
        >>> report.show_summary().caption
        'MARS Model Evaluation'
        """
        return self.styled_summary

    def to_pandas(self) -> pd.DataFrame:
        """
        返回汇总表副本。

        Returns
        -------
        pandas.DataFrame
            汇总表副本。

        Examples
        --------
        >>> report = MarsModelingReport(pd.DataFrame({"metric": [1.0]}))
        >>> report.to_pandas().equals(report.summary_table)
        True
        """
        return self.summary_table.copy()

    def write_excel(self, path: str = "mars_model_evaluation.xlsx", engine: str | None = None) -> None:
        """
        将汇总表和明细表写入 Excel 工作簿。

        Parameters
        ----------
        path : str, default "mars_model_evaluation.xlsx"
            输出文件路径。
        engine : str, optional
            Pandas ExcelWriter 引擎。

        Returns
        -------
        None
            函数仅产生 Excel 文件写入副作用。

        Examples
        --------
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> report = MarsModelingReport(pd.DataFrame({"metric": [1.0]}))
        >>> with TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "report.xlsx"
        ...     report.write_excel(str(path))
        ...     path.exists()
        True
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if not self.detail_tables:
            self.summary_table.to_excel(path_obj, engine=engine)
            return
        with pd.ExcelWriter(path_obj, engine=engine) as writer:
            self.summary_table.to_excel(writer, sheet_name="summary")
            for name, table in self.detail_tables.items():
                table.to_excel(writer, sheet_name=str(name)[:31], index=False)

    def to_html(
        self,
        path: str = "mars_model_report.html",
        *,
        title: str | None = None,
        run: Any | None = None,
        scorecard: Any | None = None,
        importance_table: pd.DataFrame | None = None,
        history_table: pd.DataFrame | None = None,
        top_features: int = 20,
        dpi: int = 150,
    ) -> Path:
        """
        生成单文件 HTML 模型报告。

        Parameters
        ----------
        path : str, default "mars_model_report.html"
            输出路径。
        title : str, optional
            HTML 报告标题。
        run : Any, optional
            调参结果对象，用于补充审计元数据。
        scorecard : Any, optional
            评分卡对象，用于展示评分刻度。
        importance_table : pandas.DataFrame, optional
            特征重要性表。
        history_table : pandas.DataFrame, optional
            调参历史表。
        top_features : int, default 20
            HTML 中展示的重要特征数量。
        dpi : int, default 150
            Matplotlib 图片分辨率。

        Returns
        -------
        pathlib.Path
            写出的 HTML 文件路径。

        Examples
        --------
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> report = MarsModelingReport(pd.DataFrame({"metric": [1.0]}))
        >>> with TemporaryDirectory() as tmp:
        ...     path = report.to_html(str(Path(tmp) / "report.html"))
        ...     path.name
        'report.html'
        """
        from mars.modeling.html_report import _ModelReportHtmlRenderer
        renderer = _ModelReportHtmlRenderer(
            report=self,
            title=title or self.caption,
            run=run,
            scorecard=scorecard,
            importance_table=importance_table,
            history_table=history_table,
            top_features=top_features,
            dpi=dpi,
        )
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(renderer.render(), encoding="utf-8")
        return path_obj
