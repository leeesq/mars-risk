"""高层风险画像工作流入口。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
import polars as pl

from mars.analysis.report import MarsBinningReport
from mars.compute import RiskCorrBaseline
from mars.feature.base import MarsBinnerBase
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
    feature_start_aware_reference: bool = False,
    risk_corr_baseline: RiskCorrBaseline = "total",
    psi_include_missing: bool = False,
    psi_include_special: bool = False,
    binning_type: Literal["native", "optimal", "lite_opt"] = "native",
    binner: MarsBinnerBase | None = None,
    binner_params: dict[str, Any] | None = None,
    benchmark_df: pl.DataFrame | pd.DataFrame | None = None,
    weights_col: str | None = None,
    batch_size: int = 100,
) -> MarsRiskProfile:
    """
    运行高层分箱风险评估工作流。

    `profile_risk` 会统一编排分箱器构建、分箱评估、多目标合并和结果对象组装，
    最终返回 `MarsRiskProfile`。函数本身不再承担绘图副作用；需要查看趋势图时，
    请直接调用 `report.plot_risk_trends(...)`。

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        待评估样本表。
    target : str | list[str] | None
        目标列名或目标列列表。传入 `None` 时进入无标签模式，只计算分布类指标与 PSI。
    features : list[str] | None
        参与本次评估的特征列。传入 `None` 时自动从输入表中推断。
    feature_data_source : dict[str, list[str]] | None
        特征来源映射，用于在报告中保留数据源维度。
    group_col : str | None
        已存在的分组列名。
    time_col : str | None
        原始日期列名。
    time_grain : str | None
        时间聚合粒度，例如 `"day"`、`"week"`、`"month"` 或 `"7d"`。
    feature_start_aware_reference : bool
        RC 鍙傜収鍙ｅ緞锛岄粯璁ゆ寜 `Total` 鍒嗙鍧忕巼璁＄畻涓€鑷存€с€?
    risk_corr_baseline : {"total", "first_group", "benchmark"}
        是否按特征首次出现的分组选择 PSI 基准。
    psi_include_missing : bool
        计算 PSI 时是否纳入缺失值箱。
    psi_include_special : bool
        计算 PSI 时是否纳入特殊值箱。
    binning_type : Literal["native", "optimal", "lite_opt"]
        未显式传入 `binner` 时使用的分箱器类型。
    binner : MarsBinnerBase | None
        显式复用的分箱器。传入后不允许再同时传 `binner_params`。
    binner_params : dict[str, Any] | None
        自动构建分箱器时使用的参数。
    benchmark_df : pl.DataFrame | pd.DataFrame | None
        外部 benchmark 样本。
    weights_col : str | None
        样本权重列名。
    batch_size : int
        批量评估时的特征批大小。

    Returns
    -------
    MarsRiskProfile
        单次风险评估结果，包含 `MarsBinningReport`、分箱器、目标列表和元数据。

    Raises
    ------
    ValueError
        当 `binner` 与 `binner_params` 同时传入，或输入配置不满足要求时抛出。

    Examples
    --------
    >>> import polars as pl
    >>> from mars.analysis import profile_risk
    >>> df = pl.DataFrame({"age": [20, 30, 40, 50], "y": [0, 0, 1, 1]})
    >>> profile = profile_risk(df, target="y", features=["age"])
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
        feature_start_aware_reference=feature_start_aware_reference,
        risk_corr_baseline=risk_corr_baseline,
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
        feature_start_aware_reference=feature_start_aware_reference,
        risk_corr_baseline=risk_corr_baseline,
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
        p_reference = to_pl(primary_report.risk_corr_reference_table)
        all_details: list[pl.DataFrame] = [p_detail]
        all_summaries: list[pl.DataFrame] = [p_summary]
        all_references: list[pl.DataFrame] = [p_reference]

        for sec_target in target_list[1:]:
            sec_run = MarsBinEvaluator(
                binning_type=binning_type,
                feature_start_aware_reference=feature_start_aware_reference,
                risk_corr_baseline=risk_corr_baseline,
            ).evaluate(
                df=df,
                target=sec_target,
                features=features,
                binner=trained_binner,
                feature_data_source=feature_data_source,
                group_col=group_col,
                time_col=time_col,
                time_grain=time_grain,
                feature_start_aware_reference=feature_start_aware_reference,
                risk_corr_baseline=risk_corr_baseline,
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
            all_references.append(to_pl(sec_run.report.risk_corr_reference_table))

        final_detail: pl.DataFrame | pd.DataFrame = pl.concat(all_details, how="vertical_relaxed")
        final_summary: pl.DataFrame | pd.DataFrame = pl.concat(all_summaries, how="vertical_relaxed")
        final_reference: pl.DataFrame | pd.DataFrame = pl.concat(
            all_references,
            how="vertical_relaxed",
        )
        if input_is_pandas:
            final_detail = final_detail.to_pandas()
            final_summary = final_summary.to_pandas()
            final_reference = final_reference.to_pandas()

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

        final_report = MarsBinningReport(
            summary_table=final_summary,
            trend_tables=primary_report.trend_tables,
            detail_table=final_detail,
            group_col=primary_report.group_col,
            detail_group_col=primary_report.detail_group_col,
            feature_data_source=primary_report.feature_data_source,
            dt_col=primary_report.dt_col,
            missing_by_day_table=primary_report.missing_by_day_table,
            risk_corr_reference_table=final_reference,
            report_meta=merged_meta,
        )
        final_targets = [str(t) for t in target_list]

    return MarsRiskProfile(
        report=final_report,
        binner=trained_binner,
        targets=final_targets,
        metadata=dict(final_report.report_meta or {}),
    )
