"""高层风险画像工作流入口。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
import polars as pl

from mars.analysis.report import MarsEvaluationReport
from mars.feature.base import MarsBinnerBase
from mars.reporting.plotting import _plot_report_helper
from mars.utils.logger import logger

if TYPE_CHECKING:
    from mars.analysis.evaluator import MarsRiskProfile


def profile_risk(
    df: pl.DataFrame | pd.DataFrame,
    *,
    target: str | list[str] | None = None,
    features: list[str] | None = None,
    feature_data_source: dict[str, list[str]] | None = None,
    group_col: str | None = None,
    time_col: str | None = None,
    time_grain: str | None = None,
    feature_start_aware_baseline: bool = False,
    psi_include_missing: bool = False,
    psi_include_special: bool = False,
    binning_type: Literal["native", "optimal", "lite_opt"] = "native",
    binner: MarsBinnerBase | None = None,
    binner_params: dict[str, Any] | None = None,
    benchmark_df: pl.DataFrame | pd.DataFrame | None = None,
    weights_col: str | None = None,
    plot: bool = True,
    plot_target: str | list[str] | None = None,
    max_plots: int = 10,
    sort_by: str = "iv",
    ascending: bool = False,
    dpi: int = 300,
    batch_size: int = 100,
) -> MarsRiskProfile:
    """
    运行高层风险画像工作流。

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        待画像样本表。
    target : str | list[str] | None
        二分类目标列名或目标列列表；``None`` 表示无标签画像。
    features : list[str] | None
        本次参与画像的特征列。
    feature_data_source : dict[str, list[str]] | None
        特征来源映射，键为来源名称，值为该来源下的特征列表。
    group_col : str | None
        已存在的分组列名。
    time_col : str | None
        原始日期列名。
    time_grain : str | None
        时间聚合粒度，例如 ``"day"``、``"week"``、``"month"`` 或 ``"7d"``。
    feature_start_aware_baseline : bool
        是否按特征首次出现的分组选择 PSI 基准。
    psi_include_missing : bool
        计算 PSI 时是否纳入缺失值箱。
    psi_include_special : bool
        计算 PSI 时是否纳入特殊值箱。
    binning_type : Literal["native", "optimal", "lite_opt"]
        未显式传入 ``binner`` 时使用的分箱器类型。
    binner : MarsBinnerBase | None
        显式复用的分箱器；传入后不允许再传 ``binner_params``。
    binner_params : dict[str, Any] | None
        构造默认分箱器时使用的参数。
    benchmark_df : pl.DataFrame | pd.DataFrame | None
        外部 benchmark 样本。
    weights_col : str | None
        样本权重列名。
    plot : bool
        是否生成图表明细。
    plot_target : str | list[str] | None
        指定需要绘图的目标列；``"all"`` 表示全部目标，``"primary"`` 表示主目标。
    max_plots : int
        最多绘制的特征数量。
    sort_by : str
        绘图特征排序指标。
    ascending : bool
        是否按 ``sort_by`` 升序排序。
    dpi : int
        图表分辨率。
    batch_size : int
        批量评估时的特征批大小。

    Returns
    -------
    MarsRiskProfile
        单次风险画像结果，包含报告、分箱器、目标列列表和运行元数据。

    Raises
    ------
    ValueError
        ``binner`` 与 ``binner_params`` 同时传入，或输入列配置不合法时抛出。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.analysis import profile_risk
    >>> df = pl.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
    >>> profile = profile_risk(df, target="y", features=["age"], plot=False)
    >>> profile.targets
    ['y']
    """
    from mars.analysis.evaluator import MarsBinEvaluator, MarsRiskProfile

    input_is_pandas = isinstance(df, pd.DataFrame)
    if binner is not None and binner_params:
        raise ValueError("`binner` and `binner_params` cannot be provided together.")

    effective_binner_params = dict(binner_params or {})
    if target is None or target == []:
        target_list: list[str] = []
        primary_target: str | None = None
        is_multi_target = False
        if binning_type in {"optimal", "lite_opt"} or effective_binner_params.get("method") == "cart":
            logger.warning("No target provided. Forcing `binning_type='native'` and `method='quantile'.")
            binning_type = "native"
            effective_binner_params["method"] = "quantile"
    else:
        target_list = [target] if isinstance(target, str) else list(target)
        primary_target = target_list[0]
        is_multi_target = len(target_list) > 1

    primary_evaluator = MarsBinEvaluator(
        binning_type=binning_type,
        binner_params=effective_binner_params,
    )
    primary_run = primary_evaluator.evaluate(
        df=df,
        target=primary_target,
        features=features,
        binner=binner,
        feature_data_source=feature_data_source,
        group_col=group_col,
        time_col=time_col,
        time_grain=time_grain,
        feature_start_aware_baseline=feature_start_aware_baseline,
        psi_include_missing=psi_include_missing,
        psi_include_special=psi_include_special,
        benchmark_df=benchmark_df,
        weights_col=weights_col,
        batch_size=batch_size,
    )
    primary_report = primary_run.report
    trained_binner = primary_run.binner

    if not is_multi_target:
        final_report = primary_report
        final_targets = primary_run.targets
    else:

        def to_pl(data: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
            """将报告中间表统一转换为 Polars DataFrame。"""
            return pl.from_pandas(data) if isinstance(data, pd.DataFrame) else data

        p_summary = to_pl(primary_report.summary_table).with_columns(pl.lit(primary_target).alias("target"))
        p_detail = to_pl(primary_report.detail_table)
        all_details: list[pl.DataFrame] = [p_detail]
        all_summaries: list[pl.DataFrame] = [p_summary]

        for sec_target in target_list[1:]:
            sec_run = MarsBinEvaluator(binning_type=binning_type).evaluate(
                df=df,
                target=sec_target,
                features=features,
                binner=trained_binner,
                feature_data_source=feature_data_source,
                group_col=group_col,
                time_col=time_col,
                time_grain=time_grain,
                feature_start_aware_baseline=feature_start_aware_baseline,
                psi_include_missing=psi_include_missing,
                psi_include_special=psi_include_special,
                benchmark_df=benchmark_df,
                weights_col=weights_col,
                batch_size=batch_size,
            )
            all_details.append(to_pl(sec_run.report.detail_table))
            all_summaries.append(
                to_pl(sec_run.report.summary_table).with_columns(pl.lit(sec_target).alias("target"))
            )

        final_detail: pl.DataFrame | pd.DataFrame = pl.concat(all_details, how="vertical_relaxed")
        final_summary: pl.DataFrame | pd.DataFrame = pl.concat(all_summaries, how="vertical_relaxed")
        if input_is_pandas:
            final_detail = final_detail.to_pandas()
            final_summary = final_summary.to_pandas()

        logger.info("`trend_tables` in the merged report contains primary-target data only.")
        merged_meta = dict(primary_report.report_meta or {})
        meta_df = cast(pl.DataFrame, primary_evaluator._ensure_polars_dataframe(df))
        merged_meta["targets"] = [str(t) for t in target_list]
        event_rate_map: dict[str, float | None] = {}
        for target_name in target_list:
            if target_name in meta_df.columns:
                try:
                    event_rate_map[str(target_name)] = float(
                        meta_df.select(pl.col(target_name).cast(pl.Float64).mean()).item()
                    )
                except Exception:
                    event_rate_map[str(target_name)] = None
        merged_meta["event_rate_by_target"] = event_rate_map

        final_report = MarsEvaluationReport(
            summary_table=final_summary,
            trend_tables=primary_report.trend_tables,
            detail_table=final_detail,
            group_col=primary_report.group_col,
            feature_data_source=primary_report.feature_data_source,
            dt_col=primary_report.dt_col,
            missing_by_day_table=primary_report.missing_by_day_table,
            report_meta=merged_meta,
        )
        final_targets = [str(t) for t in target_list]

    if plot and final_targets:
        targets_to_plot: list[str] = []
        if plot_target is None or plot_target == "all":
            targets_to_plot = final_targets
        elif plot_target == "primary" and final_targets:
            targets_to_plot = [final_targets[0]]
        elif isinstance(plot_target, str):
            targets_to_plot = [plot_target]
        elif isinstance(plot_target, list):
            targets_to_plot = plot_target

        targets_to_plot = [t for t in targets_to_plot if t in final_targets]
        if not targets_to_plot:
            logger.warning("No valid targets specified for plotting.")
        else:
            _plot_report_helper(
                evaluator=primary_evaluator,
                report=final_report,
                target_list=targets_to_plot,
                sort_by=sort_by,
                ascending=ascending,
                max_plots=max_plots,
                dpi=dpi,
            )

    return MarsRiskProfile(
        report=final_report,
        binner=trained_binner,
        targets=final_targets,
        metadata=dict(final_report.report_meta or {}),
    )
