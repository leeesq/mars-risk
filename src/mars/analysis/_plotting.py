"""分箱评估画像的内部绘图辅助。"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl

from mars.analysis.report import MarsEvaluationReport
from mars.utils.logger import logger

if TYPE_CHECKING:
    from mars.analysis.evaluator import MarsBinEvaluator


def _plot_report_helper(
    evaluator: MarsBinEvaluator,
    report: MarsEvaluationReport,
    target_list: list[str],
    sort_by: str,
    ascending: bool,
    max_plots: int,
    dpi: int,
) -> None:
    """根据多 target 报告筛选 Top-N 特征并调用评估器绘图方法。"""
    summary_all = cast(pl.DataFrame, evaluator._ensure_polars_dataframe(report.summary_table))
    detail_all = cast(pl.DataFrame, evaluator._ensure_polars_dataframe(report.detail_table))

    sort_map = {
        "iv": "iv",
        "psi": "psi_max",
        "ks": "ks",
        "auc": "auc",
        "rc": "rc_min",
        "mono": "mono",
    }
    sort_key = sort_map.get(sort_by.lower(), "iv")

    for current_target in target_list:
        logger.info(f"Plotting target '{current_target}'.")

        if "target" in summary_all.columns:
            curr_summary = summary_all.filter(pl.col("target") == current_target)
        else:
            curr_summary = summary_all

        plot_features: list[str] | None = None
        if sort_key in curr_summary.columns:
            sorted_feats = curr_summary.sort(sort_key, descending=not ascending)["feature"].to_list()
            plot_features = sorted_feats[:max_plots] if len(sorted_feats) > max_plots else sorted_feats
            if len(sorted_feats) > max_plots:
                logger.info(f"Selecting top {max_plots} features by '{sort_key}' for plotting.")

        curr_detail = detail_all.filter(pl.col("y") == current_target)
        if curr_detail.is_empty():
            logger.warning(f"No detail data found for target '{current_target}'. Skipping plotting.")
            continue

        evaluator.plot_feature_binning_risk_trends(
            report=None,
            df_detail=curr_detail,
            features=plot_features,
            group_col=report.group_col,
            target_name=current_target,
            sort_by=sort_by,
            ascending=ascending,
            dpi=dpi,
        )

        if len(target_list) > 1:
            logger.info(f"{'-' * 40}")
