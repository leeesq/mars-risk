"""建模评估明细表构建工具。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from mars.core.constants import METRIC_EPSILON


def build_decile_lift_detail(
    df: pd.DataFrame,
    *,
    group_col: str,
    pred_col: str,
    target_col: str,
    ordered_groups: Sequence[str],
) -> pd.DataFrame:
    """
    按模型分降序构建分组十分位 Lift 明细。

    Parameters
    ----------
    df : pd.DataFrame
        已包含分组、模型分和目标列的样本表。
    group_col : str
        分组列名。
    pred_col : str
        模型分列名。
    target_col : str
        二分类目标列名。
    ordered_groups : Sequence[str]
        需要输出的分组顺序。

    Returns
    -------
    pd.DataFrame
        十分位 Lift 明细表。
    """
    rows: list[dict[str, Any]] = []
    for group in ordered_groups:
        sub_df = df[df[group_col].astype(str) == str(group)].copy()
        y_true = pd.to_numeric(sub_df[target_col], errors="coerce")
        y_pred = pd.to_numeric(sub_df[pred_col], errors="coerce")
        valid = sub_df.loc[y_true.notna() & y_pred.notna() & (y_true >= 0)].copy()
        if valid.empty:
            continue
        valid["_target"] = pd.to_numeric(valid[target_col], errors="coerce")
        valid["_score"] = pd.to_numeric(valid[pred_col], errors="coerce")
        valid = valid.sort_values("_score", ascending=False).reset_index(drop=True)
        decile_count = min(10, max(int(valid.shape[0]), 1))
        valid["_decile"] = np.floor(
            np.arange(valid.shape[0]) * decile_count / valid.shape[0]
        ).astype(int) + 1
        base_bad_rate = float(valid["_target"].mean()) if valid.shape[0] else np.nan
        total_bad = float(valid["_target"].sum())
        for decile, part in valid.groupby("_decile", sort=True):
            bad = float(part["_target"].sum())
            count = int(part.shape[0])
            bad_rate = float(bad / count) if count else np.nan
            rows.append(
                {
                    group_col: group,
                    "decile": int(decile),
                    "count": count,
                    "bad": bad,
                    "bad_rate": bad_rate,
                    "lift": (
                        bad_rate / base_bad_rate
                        if base_bad_rate and pd.notna(base_bad_rate)
                        else np.nan
                    ),
                    "capture_rate": bad / total_bad if total_bad > 0 else np.nan,
                    "min_score": float(part["_score"].min()),
                    "max_score": float(part["_score"].max()),
                }
            )
    return pd.DataFrame(rows)


def build_roc_curve_detail(
    df: pd.DataFrame,
    *,
    group_col: str,
    pred_col: str,
    target_col: str,
    ordered_groups: Sequence[str],
    max_points: int = 500,
) -> pd.DataFrame:
    """
    构建 ROC 曲线明细表。

    Parameters
    ----------
    df : pd.DataFrame
        已包含分组、模型分和目标列的样本表。
    group_col : str
        分组列名。
    pred_col : str
        模型分列名。
    target_col : str
        二分类目标列名。
    ordered_groups : Sequence[str]
        需要输出的分组顺序。
    max_points : int
        每个分组最多保留的曲线点数量。

    Returns
    -------
    pd.DataFrame
        ROC 曲线明细表。
    """
    rows: list[dict[str, Any]] = []
    for group in ordered_groups:
        sub_df = df[df[group_col].astype(str) == str(group)]
        y, score = _valid_score_arrays(sub_df, pred_col=pred_col, target_col=target_col)
        if len(y) == 0 or np.unique(y).size < 2:
            continue
        fpr, tpr, thresholds = roc_curve(y, score)
        thinned = _thin_arrays(max_points, fpr=fpr, tpr=tpr, threshold=thresholds)
        for fpr_val, tpr_val, threshold_val in zip(
            thinned["fpr"],
            thinned["tpr"],
            thinned["threshold"],
        ):
            rows.append(
                {
                    group_col: group,
                    "fpr": float(fpr_val),
                    "tpr": float(tpr_val),
                    "threshold": (
                        float(threshold_val) if np.isfinite(threshold_val) else threshold_val
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_ks_curve_detail(
    df: pd.DataFrame,
    *,
    group_col: str,
    pred_col: str,
    target_col: str,
    ordered_groups: Sequence[str],
    max_points: int = 500,
) -> pd.DataFrame:
    """
    构建 KS 曲线明细表。

    Parameters
    ----------
    df : pd.DataFrame
        已包含分组、模型分和目标列的样本表。
    group_col : str
        分组列名。
    pred_col : str
        模型分列名。
    target_col : str
        二分类目标列名。
    ordered_groups : Sequence[str]
        需要输出的分组顺序。
    max_points : int
        每个分组最多保留的曲线点数量。

    Returns
    -------
    pd.DataFrame
        KS 曲线明细表。
    """
    rows: list[dict[str, Any]] = []
    for group in ordered_groups:
        sub_df = df[df[group_col].astype(str) == str(group)]
        y, score = _valid_score_arrays(sub_df, pred_col=pred_col, target_col=target_col)
        pos = float(y.sum())
        neg = float(len(y) - pos)
        if len(y) == 0 or pos <= 0 or neg <= 0:
            continue
        order = np.argsort(-score)
        y_sorted = y[order]
        bad_cum = np.cumsum(y_sorted) / pos
        good_cum = np.cumsum(1.0 - y_sorted) / neg
        sample_pct = np.arange(1, len(y_sorted) + 1, dtype=float) / len(y_sorted)
        ks = np.abs(bad_cum - good_cum)
        thinned = _thin_arrays(
            max_points,
            sample_pct=sample_pct,
            bad_cum_rate=bad_cum,
            good_cum_rate=good_cum,
            ks=ks,
        )
        for idx in range(len(thinned["sample_pct"])):
            rows.append(
                {
                    group_col: group,
                    "sample_pct": float(thinned["sample_pct"][idx]),
                    "bad_cum_rate": float(thinned["bad_cum_rate"][idx]),
                    "good_cum_rate": float(thinned["good_cum_rate"][idx]),
                    "ks": float(thinned["ks"][idx]),
                }
            )
    return pd.DataFrame(rows)


def build_calibration_curve_detail(
    df: pd.DataFrame,
    *,
    group_col: str,
    pred_col: str,
    target_col: str,
    ordered_groups: Sequence[str],
) -> pd.DataFrame:
    """
    按分位箱构建校准曲线明细表。

    Parameters
    ----------
    df : pd.DataFrame
        已包含分组、模型分和目标列的样本表。
    group_col : str
        分组列名。
    pred_col : str
        模型分列名。
    target_col : str
        二分类目标列名。
    ordered_groups : Sequence[str]
        需要输出的分组顺序。

    Returns
    -------
    pd.DataFrame
        校准曲线明细表。
    """
    rows: list[dict[str, Any]] = []
    for group in ordered_groups:
        sub_df = df[df[group_col].astype(str) == str(group)]
        y, score = _valid_score_arrays(sub_df, pred_col=pred_col, target_col=target_col)
        if len(y) == 0:
            continue
        valid = pd.DataFrame({"target": y, "score": score})
        bin_count = min(10, max(int(valid["score"].nunique()), 1))
        if bin_count <= 1:
            valid["_bin"] = 1
        else:
            try:
                valid["_bin"] = (
                    pd.qcut(valid["score"], q=bin_count, duplicates="drop", labels=False) + 1
                )
            except ValueError:
                valid["_bin"] = (
                    pd.cut(valid["score"], bins=bin_count, duplicates="drop", labels=False) + 1
                )
        for bin_idx, part in valid.groupby("_bin", sort=True):
            rows.append(
                {
                    group_col: group,
                    "bin": int(bin_idx) if pd.notna(bin_idx) else np.nan,
                    "count": int(part.shape[0]),
                    "pred_mean": float(part["score"].mean()),
                    "bad_rate": float(part["target"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_score_distribution_detail(
    df: pd.DataFrame,
    *,
    group_col: str,
    pred_col: str,
    target_col: str,
    ordered_groups: Sequence[str],
) -> pd.DataFrame:
    """
    按目标取值构建分箱后的分数分布明细表。

    Parameters
    ----------
    df : pd.DataFrame
        已包含分组、模型分和目标列的样本表。
    group_col : str
        分组列名。
    pred_col : str
        模型分列名。
    target_col : str
        二分类目标列名。
    ordered_groups : Sequence[str]
        需要输出的分组顺序。

    Returns
    -------
    pd.DataFrame
        分数分布明细表。
    """
    scores = pd.to_numeric(df[pred_col], errors="coerce").dropna()
    if scores.empty:
        return pd.DataFrame()
    min_score = float(scores.min())
    max_score = float(scores.max())
    if min_score == max_score:
        min_score -= METRIC_EPSILON
        max_score += METRIC_EPSILON
    bins = np.linspace(min_score, max_score, 31)
    rows: list[dict[str, Any]] = []
    for group in ordered_groups:
        sub_df = df[df[group_col].astype(str) == str(group)].copy()
        sub_df["_score"] = pd.to_numeric(sub_df[pred_col], errors="coerce")
        sub_df["_target"] = pd.to_numeric(sub_df[target_col], errors="coerce")
        sub_df = sub_df[
            sub_df["_score"].notna() & sub_df["_target"].notna() & (sub_df["_target"] >= 0)
        ]
        for target_value, target_part in sub_df.groupby("_target", sort=True):
            counts = pd.cut(target_part["_score"], bins=bins, include_lowest=True).value_counts(
                sort=False
            )
            denom = max(float(counts.sum()), 1.0)
            for idx, interval in enumerate(counts.index):
                rows.append(
                    {
                        group_col: group,
                        "target_value": int(target_value),
                        "bin": idx + 1,
                        "score_min": float(interval.left),
                        "score_max": float(interval.right),
                        "bin_center": float((interval.left + interval.right) / 2.0),
                        "count": int(counts.iloc[idx]),
                        "pct": float(counts.iloc[idx] / denom),
                    }
                )
    return pd.DataFrame(rows)


def _valid_score_arrays(
    sub_df: pd.DataFrame,
    *,
    pred_col: str,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """返回图表明细使用的干净二分类标签和分数数组。"""
    y_true = pd.to_numeric(sub_df[target_col], errors="coerce")
    y_pred = pd.to_numeric(sub_df[pred_col], errors="coerce")
    mask = y_true.notna() & y_pred.notna() & (y_true >= 0)
    return y_true[mask].to_numpy(dtype=float), y_pred[mask].to_numpy(dtype=float)


def _thin_arrays(max_points: int, **arrays: np.ndarray) -> dict[str, np.ndarray]:
    """对齐下采样数组，控制报告明细表体积。"""
    if not arrays:
        return {}
    size = len(next(iter(arrays.values())))
    if size <= max_points:
        return arrays
    idx = np.unique(np.linspace(0, size - 1, max_points).astype(int))
    return {name: values[idx] for name, values in arrays.items()}
