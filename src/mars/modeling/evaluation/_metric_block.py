"""评估指标块组装 helper。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mars.core.constants import PROBABILITY_EPSILON
from mars.modeling.evaluation.metrics import calculate_auc, calculate_f1, calculate_ks


def calculate_metric_block(
    sub_df: pd.DataFrame,
    *,
    pred_col: str,
    target_col: str,
    benchmark_cols: list[str],
    section_label: str | None = None,
    score_psi: float | None = None,
) -> dict[tuple[str, str], Any]:
    """构建单个目标在单个分组上的指标块。"""
    y_true = pd.to_numeric(sub_df[target_col], errors="coerce")
    y_pred = pd.to_numeric(sub_df[pred_col], errors="coerce")
    valid_mask = y_true.notna() & y_pred.notna() & (y_true >= 0)
    valid_y = y_true[valid_mask]
    valid_pred = y_pred[valid_mask]

    total_count = int(valid_y.shape[0])
    if total_count > 0:
        bad_count = float(valid_y.sum())
        good_count = float(total_count - bad_count)
        bad_rate = float(bad_count / total_count)
    else:
        bad_count = np.nan
        good_count = np.nan
        bad_rate = np.nan

    section = section_label or f"Target: {target_col}"
    block: dict[tuple[str, str], Any] = {
        (section, "Total Count"): total_count,
        (section, "Good"): good_count,
        (section, "Bad"): bad_count,
        (section, "Bad Rate"): bad_rate,
    }

    if total_count > 0 and valid_y.nunique() >= 2:
        block[(section, "New AUC")] = calculate_auc(valid_y.to_numpy(), valid_pred.to_numpy())
        block[(section, "New KS")] = calculate_ks(valid_y.to_numpy(), valid_pred.to_numpy())
        block[(section, "New F1")] = calculate_f1(valid_y.to_numpy(), valid_pred.to_numpy())
        clipped_pred = np.clip(
            valid_pred.to_numpy(dtype=float),
            PROBABILITY_EPSILON,
            1 - PROBABILITY_EPSILON,
        )
        y_arr = valid_y.to_numpy(dtype=float)
        block[(section, "LogLoss")] = float(
            -np.mean(y_arr * np.log(clipped_pred) + (1.0 - y_arr) * np.log(1.0 - clipped_pred))
        )
        block[(section, "Brier")] = float(np.mean((clipped_pred - y_arr) ** 2))
        total_bad = float(y_arr.sum())
        order = np.argsort(-clipped_pred)
        for pct, label in [(0.10, "Top 10% Capture"), (0.20, "Top 20% Capture")]:
            top_n = max(int(np.ceil(len(order) * pct)), 1)
            block[(section, label)] = float(y_arr[order[:top_n]].sum() / total_bad) if total_bad > 0 else np.nan
    else:
        for column_name in [
            "New AUC",
            "New KS",
            "New F1",
            "LogLoss",
            "Brier",
            "Top 10% Capture",
            "Top 20% Capture",
        ]:
            block[(section, column_name)] = np.nan

    block[(section, "Score PSI")] = score_psi if score_psi is not None else np.nan

    for benchmark_col in benchmark_cols:
        bench_pred = pd.to_numeric(sub_df[benchmark_col], errors="coerce")
        bench_mask = valid_mask & bench_pred.notna()
        bench_y = y_true[bench_mask]
        bench_scores = bench_pred[bench_mask]
        if bench_y.shape[0] > 0 and bench_y.nunique() >= 2:
            bench_auc = calculate_auc(bench_y.to_numpy(), bench_scores.to_numpy())
            bench_ks = calculate_ks(bench_y.to_numpy(), bench_scores.to_numpy())
            bench_f1 = calculate_f1(bench_y.to_numpy(), bench_scores.to_numpy())
        else:
            bench_auc = np.nan
            bench_ks = np.nan
            bench_f1 = np.nan

        use_short_name = len(benchmark_cols) == 1
        auc_name = "Bench AUC" if use_short_name else f"Bench {benchmark_col} AUC"
        ks_name = "Bench KS" if use_short_name else f"Bench {benchmark_col} KS"
        f1_name = "Bench F1" if use_short_name else f"Bench {benchmark_col} F1"
        auc_diff_name = "AUC Diff" if use_short_name else f"{benchmark_col} AUC Diff"
        ks_diff_name = "KS Diff" if use_short_name else f"{benchmark_col} KS Diff"
        f1_diff_name = "F1 Diff" if use_short_name else f"{benchmark_col} F1 Diff"

        block[(section, auc_name)] = bench_auc
        block[(section, ks_name)] = bench_ks
        block[(section, f1_name)] = bench_f1
        block[(section, auc_diff_name)] = (
            block[(section, "New AUC")] - bench_auc
            if pd.notna(block[(section, "New AUC")]) and pd.notna(bench_auc)
            else np.nan
        )
        block[(section, ks_diff_name)] = (
            block[(section, "New KS")] - bench_ks
            if pd.notna(block[(section, "New KS")]) and pd.notna(bench_ks)
            else np.nan
        )
        block[(section, f1_diff_name)] = (
            block[(section, "New F1")] - bench_f1
            if pd.notna(block[(section, "New F1")]) and pd.notna(bench_f1)
            else np.nan
        )

    return block
