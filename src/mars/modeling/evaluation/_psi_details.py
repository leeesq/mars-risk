"""PSI 明细构建 helper。"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from mars.compute import to_pandas_table


def evaluate_psi_with_binner(
    df: pd.DataFrame,
    *,
    group_col: str,
    feature: str,
    target_col: str,
    psi_include_missing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """复用分箱评估器计算单字段 PSI。"""
    from mars.analysis.evaluator import MarsBinEvaluator

    evaluator = MarsBinEvaluator(
        binning_type="native",
        binner_params={"method": "quantile", "n_bins": 10},
    )
    run = evaluator.evaluate(
        df,
        target=target_col,
        features=[feature],
        group_col=group_col,
        psi_include_missing=psi_include_missing,
    )
    return (
        to_pandas_table(run.report.detail_table),
        to_pandas_table(run.report.summary_table),
        to_pandas_table(run.report.trend_tables.get("psi")),
    )


def psi_detail_group_col(detail: pd.DataFrame, group_col: str) -> str:
    """识别 PSI 明细表里的分组列名。"""
    if group_col in detail.columns:
        return group_col
    if "mars_group" in detail.columns:
        return "mars_group"
    return group_col


def build_score_psi_detail(
    df: pd.DataFrame,
    *,
    group_col: str,
    pred_col: str,
    target_col: str,
    ordered_groups: Sequence[str],
    psi_include_missing: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """构建模型分数 PSI 明细与分组 PSI 映射。"""
    if not ordered_groups:
        return pd.DataFrame(), {}
    detail, _, trend = evaluate_psi_with_binner(
        df,
        group_col=group_col,
        feature=pred_col,
        target_col=target_col,
        psi_include_missing=psi_include_missing,
    )
    psi_map: dict[str, float] = {}
    if not trend.empty:
        score_rows = trend.loc[trend["feature"] == pred_col]
        if not score_rows.empty:
            score_row = score_rows.iloc[0]
            for group in ordered_groups:
                if str(group) in score_row.index:
                    psi_map[str(group)] = float(score_row[str(group)])
    if detail.empty:
        return pd.DataFrame(), psi_map

    detail_group_col = psi_detail_group_col(detail, group_col)
    score_detail = detail.loc[
        (detail["feature"] == pred_col)
        & (detail[detail_group_col].astype(str).isin([str(group) for group in ordered_groups]))
    ].copy()
    if detail_group_col != group_col:
        score_detail = score_detail.rename(columns={detail_group_col: group_col})
    if "psi_bin" in score_detail.columns:
        score_detail["psi"] = score_detail["psi_bin"]
    if "bin_label" in score_detail.columns:
        score_detail["score_range"] = score_detail["bin_label"]
    if "bin_index" in score_detail.columns:
        score_detail["bin"] = score_detail["bin_index"]
    return score_detail, psi_map


def build_feature_psi_detail(
    df: pd.DataFrame,
    *,
    group_col: str,
    target_col: str,
    feature_cols: Sequence[str],
    ordered_groups: Sequence[str],
    psi_include_missing: bool,
) -> pd.DataFrame:
    """构建特征级 PSI 明细表。"""
    valid_feature_cols = [col for col in feature_cols if col in df.columns]
    if not ordered_groups or not valid_feature_cols:
        return pd.DataFrame()

    detail_frames: list[pd.DataFrame] = []
    for feature in valid_feature_cols:
        detail, _, _ = evaluate_psi_with_binner(
            df,
            group_col=group_col,
            feature=feature,
            target_col=target_col,
            psi_include_missing=psi_include_missing,
        )
        if detail.empty:
            continue
        detail_group_col = psi_detail_group_col(detail, group_col)
        feature_detail = detail.loc[
            (detail["feature"] == feature)
            & (detail[detail_group_col].astype(str).isin([str(group) for group in ordered_groups]))
        ].copy()
        if feature_detail.empty:
            continue
        if detail_group_col != group_col:
            feature_detail = feature_detail.rename(columns={detail_group_col: group_col})
        if "psi_bin" in feature_detail.columns:
            feature_detail["psi"] = feature_detail["psi_bin"]
            feature_detail["feature_psi"] = feature_detail.groupby(
                ["feature", group_col]
            )["psi_bin"].transform("sum")
        detail_frames.append(feature_detail)
    if not detail_frames:
        return pd.DataFrame()
    return pd.concat(detail_frames, ignore_index=True).sort_values(
        ["feature_psi", "feature", group_col],
        ascending=[False, True, True],
    )
