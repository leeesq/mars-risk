"""建模评估指标汇总与明细表构建。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from mars.modeling.metrics import calculate_auc, calculate_f1, calculate_ks
from mars.modeling.report import MarsModelingReport
from mars.modeling.utils import FrameLike, split_name_sort_key, to_pandas_frame


class MarsModelEvaluator:
    """
    构建二分类模型分组评估报告。

    评估器在 public API 层面按无状态工具使用：样本数据、预测分、目标列、分组列、
    benchmark 分数、时间列、特征 PSI 列和重要性表都由每次 :meth:`evaluate`
    调用传入。同一个实例可以连续评估不同模型、不同样本或不同切片，不会复用上一次
    评估上下文中的列名。

    Notes
    -----
    `group_col` 表示已经存在的样本分组列，例如 `dataset_flag`；`time_col`
    只用于报告中补充每个分组的时间边界；`feature_cols` 只影响特征 PSI 明细，
    不参与模型指标计算。

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"dataset_flag": ["train", "val"], "y": [0, 1], "score": [0.1, 0.9]})
    >>> evaluator = MarsModelEvaluator()
    >>> report = evaluator.evaluate(df, pred_col="score", group_col="dataset_flag", target="y")
    >>> report.summary_table is not None
    True
    """

    COLUMN_ORDER = [
        "Total Count",
        "Good",
        "Bad",
        "Bad Rate",
        "New AUC",
        "New KS",
        "New F1",
        "LogLoss",
        "Brier",
        "Score PSI",
        "Top 10% Capture",
        "Top 20% Capture",
        "Bench AUC",
        "Bench KS",
        "Bench F1",
        "AUC Diff",
        "KS Diff",
        "F1 Diff",
    ]

    def __init__(
        self,
    ) -> None:
        """
        初始化一个空评估器。

        构造函数不接收数据列名，也不绑定某一次评估数据。列名和可选元数据会在
        `evaluate` 调用期间写入实例，用于复用内部 helper；下一次调用会重新覆盖这些
        临时上下文。
        """
        self.group_col: str = ""
        self.target_col: str = ""
        self.benchmark_col: str | None = None
        self.benchmark_cols: List[str] = []
        self.time_col: str | None = None
        self.val_target: str | None = None
        self.aux_targets: List[str] = []
        self.target_group_cols: Dict[str, str] = {}
        self.feature_cols: List[str] = []
        self.importance_table: pd.DataFrame | None = None
        self.psi_include_missing: bool = False

    def _validate_frame(self, df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
        """校验必需列，并在配置时间列时统一转换为 datetime。"""
        required = {self.group_col, pred_col, self.target_col}
        if self.time_col:
            required.add(self.time_col)
        required.update(self.benchmark_cols)
        required.update(self.aux_targets)
        required.update(self.target_group_cols.values())

        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Evaluation data is missing required columns: {sorted(missing)}")

        normalized = df.copy()
        if self.time_col is not None:
            normalized[self.time_col] = pd.to_datetime(normalized[self.time_col], errors="coerce")
        return normalized

    def _calc_metric_block(
        self,
        sub_df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        section_label: str | None = None,
        score_psi: float | None = None,
    ) -> Dict[Tuple[str, str], Any]:
        """构建分组评估报告中的单个目标指标块。"""
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

        block: Dict[Tuple[str, str], Any] = {}
        section = section_label or f"Target: {target_col}"
        block[(section, "Total Count")] = total_count
        block[(section, "Good")] = good_count
        block[(section, "Bad")] = bad_count
        block[(section, "Bad Rate")] = bad_rate

        if total_count > 0 and valid_y.nunique() >= 2:
            block[(section, "New AUC")] = calculate_auc(valid_y.to_numpy(), valid_pred.to_numpy())
            block[(section, "New KS")] = calculate_ks(valid_y.to_numpy(), valid_pred.to_numpy())
            block[(section, "New F1")] = calculate_f1(valid_y.to_numpy(), valid_pred.to_numpy())
            clipped_pred = np.clip(valid_pred.to_numpy(dtype=float), 1e-15, 1 - 1e-15)
            y_arr = valid_y.to_numpy(dtype=float)
            block[(section, "LogLoss")] = float(
                -np.mean(y_arr * np.log(clipped_pred) + (1.0 - y_arr) * np.log(1.0 - clipped_pred))
            )
            block[(section, "Brier")] = float(np.mean((clipped_pred - y_arr) ** 2))
            total_bad = float(y_arr.sum())
            order = np.argsort(-clipped_pred)
            for pct, label in [(0.10, "Top 10% Capture"), (0.20, "Top 20% Capture")]:
                top_n = max(int(np.ceil(len(order) * pct)), 1)
                block[(section, label)] = (
                    float(y_arr[order[:top_n]].sum() / total_bad)
                    if total_bad > 0
                    else np.nan
                )
        else:
            block[(section, "New AUC")] = np.nan
            block[(section, "New KS")] = np.nan
            block[(section, "New F1")] = np.nan
            block[(section, "LogLoss")] = np.nan
            block[(section, "Brier")] = np.nan
            block[(section, "Top 10% Capture")] = np.nan
            block[(section, "Top 20% Capture")] = np.nan

        block[(section, "Score PSI")] = score_psi if score_psi is not None else np.nan

        for benchmark_col in self.benchmark_cols:
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

            use_short_name = len(self.benchmark_cols) == 1
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

    def _get_ordered_groups(self, df: pd.DataFrame) -> List[str]:
        """按 MARS 稳定顺序返回数据切片名称。"""
        return self._get_ordered_groups_for_col(df, self.group_col)

    @staticmethod
    def _get_ordered_groups_for_col(df: pd.DataFrame, group_col: str) -> List[str]:
        """按 MARS 稳定顺序返回指定切片列的名称。"""
        groups = df[group_col].dropna().astype(str).unique().tolist()
        return sorted(groups, key=split_name_sort_key)

    def _get_ordered_columns(self, available_columns: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """为最终报告返回稳定的列布局。"""
        ordered_columns: List[Tuple[str, str]] = []
        if self.time_col:
            for time_name in ("Start Time", "End Time"):
                candidate = ("Time Period", time_name)
                if candidate in available_columns:
                    ordered_columns.append(candidate)

        sections = [f"Target: {self.target_col}"]
        sections.extend([f"Aux Target: {target}" for target in self.aux_targets])

        for section in sections:
            for column_name in self.COLUMN_ORDER:
                candidate = (section, column_name)
                if candidate in available_columns:
                    ordered_columns.append(candidate)

        remaining_columns = [col for col in available_columns if col not in ordered_columns]
        return ordered_columns + remaining_columns

    @staticmethod
    def _as_pandas_table(table: Any) -> pd.DataFrame:
        """将 Polars/Pandas 报表统一转成 Pandas，便于建模报告拼装。"""
        if table is None:
            return pd.DataFrame()
        if isinstance(table, pd.DataFrame):
            return table.copy()
        if hasattr(table, "to_pandas"):
            return table.to_pandas()
        return pd.DataFrame(table)

    def _evaluate_psi_with_binner(
        self,
        df: pd.DataFrame,
        *,
        feature: str,
        target_col: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """复用分箱评估器计算单个字段的 PSI 明细、汇总和趋势表。"""
        from mars.analysis.evaluator import MarsBinEvaluator

        evaluator = MarsBinEvaluator(
            binning_type="native",
            binner_params={
                "method": "quantile",
                "n_bins": 10,
            },
        )
        run = evaluator.evaluate(
            df,
            target=target_col,
            features=[feature],
            group_col=self.group_col,
            psi_include_missing=self.psi_include_missing,
        )
        return (
            self._as_pandas_table(run.report.detail_table),
            self._as_pandas_table(run.report.summary_table),
            self._as_pandas_table(run.report.trend_tables.get("psi")),
        )

    def _psi_detail_group_col(self, detail: pd.DataFrame) -> str:
        """识别分箱评估器明细表中的实际分组列。"""
        if self.group_col in detail.columns:
            return self.group_col
        if "mars_group" in detail.columns:
            return "mars_group"
        return self.group_col

    def _build_score_psi_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        ordered_groups: Sequence[str],
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """复用风险分箱评估器生成模型分 PSI 明细。"""
        if not ordered_groups:
            return pd.DataFrame(), {}
        detail, _, trend = self._evaluate_psi_with_binner(
            df,
            feature=pred_col,
            target_col=self.target_col,
        )
        psi_map: Dict[str, float] = {}
        if not trend.empty:
            score_rows = trend.loc[trend["feature"] == pred_col]
            if not score_rows.empty:
                score_row = score_rows.iloc[0]
                for group in ordered_groups:
                    if str(group) in score_row.index:
                        psi_map[str(group)] = float(score_row[str(group)])
        if detail.empty:
            return pd.DataFrame(), psi_map
        detail_group_col = self._psi_detail_group_col(detail)
        score_detail = detail.loc[
            (detail["feature"] == pred_col)
            & (detail[detail_group_col].astype(str).isin([str(group) for group in ordered_groups]))
        ].copy()
        if detail_group_col != self.group_col:
            score_detail = score_detail.rename(columns={detail_group_col: self.group_col})
        if "psi_bin" in score_detail.columns:
            score_detail["psi"] = score_detail["psi_bin"]
        if "bin_label" in score_detail.columns:
            score_detail["score_range"] = score_detail["bin_label"]
        if "bin_index" in score_detail.columns:
            score_detail["bin"] = score_detail["bin_index"]
        return score_detail, psi_map

    def _build_decile_lift_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        ordered_groups: Sequence[str],
    ) -> pd.DataFrame:
        """按模型分数降序构建分组十分位 Lift 明细。"""
        rows: List[Dict[str, Any]] = []
        for group in ordered_groups:
            sub_df = df[df[self.group_col].astype(str) == str(group)].copy()
            y_true = pd.to_numeric(sub_df[target_col], errors="coerce")
            y_pred = pd.to_numeric(sub_df[pred_col], errors="coerce")
            valid = sub_df.loc[y_true.notna() & y_pred.notna() & (y_true >= 0)].copy()
            if valid.empty:
                continue
            valid["_target"] = pd.to_numeric(valid[target_col], errors="coerce")
            valid["_score"] = pd.to_numeric(valid[pred_col], errors="coerce")
            valid = valid.sort_values("_score", ascending=False).reset_index(drop=True)
            decile_count = min(10, max(int(valid.shape[0]), 1))
            valid["_decile"] = np.floor(np.arange(valid.shape[0]) * decile_count / valid.shape[0]).astype(int) + 1
            base_bad_rate = float(valid["_target"].mean()) if valid.shape[0] else np.nan
            total_bad = float(valid["_target"].sum())
            for decile, part in valid.groupby("_decile", sort=True):
                bad = float(part["_target"].sum())
                count = int(part.shape[0])
                bad_rate = float(bad / count) if count else np.nan
                rows.append(
                    {
                        self.group_col: group,
                        "decile": int(decile),
                        "count": count,
                        "bad": bad,
                        "bad_rate": bad_rate,
                        "lift": bad_rate / base_bad_rate if base_bad_rate and pd.notna(base_bad_rate) else np.nan,
                        "capture_rate": bad / total_bad if total_bad > 0 else np.nan,
                        "min_score": float(part["_score"].min()),
                        "max_score": float(part["_score"].max()),
                    }
                )
        return pd.DataFrame(rows)

    def _valid_score_arrays(
        self,
        sub_df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """返回图表明细使用的干净二分类标签和分数数组。"""
        y_true = pd.to_numeric(sub_df[target_col], errors="coerce")
        y_pred = pd.to_numeric(sub_df[pred_col], errors="coerce")
        mask = y_true.notna() & y_pred.notna() & (y_true >= 0)
        return y_true[mask].to_numpy(dtype=float), y_pred[mask].to_numpy(dtype=float)

    @staticmethod
    def _thin_arrays(max_points: int, **arrays: np.ndarray) -> Dict[str, np.ndarray]:
        """对齐下采样数组，控制报告明细表体积。"""
        if not arrays:
            return {}
        size = len(next(iter(arrays.values())))
        if size <= max_points:
            return arrays
        idx = np.unique(np.linspace(0, size - 1, max_points).astype(int))
        return {name: values[idx] for name, values in arrays.items()}

    def _build_roc_curve_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        ordered_groups: Sequence[str],
    ) -> pd.DataFrame:
        """为每个切片构建 ROC 曲线明细行。"""
        rows: List[Dict[str, Any]] = []
        for group in ordered_groups:
            sub_df = df[df[self.group_col].astype(str) == str(group)]
            y, score = self._valid_score_arrays(sub_df, pred_col=pred_col, target_col=target_col)
            pos = float(y.sum())
            neg = float(len(y) - pos)
            if len(y) == 0 or pos <= 0 or neg <= 0:
                continue
            order = np.argsort(-score)
            y_sorted = y[order]
            score_sorted = score[order]
            tpr = np.r_[0.0, np.cumsum(y_sorted) / pos, 1.0]
            fpr = np.r_[0.0, np.cumsum(1.0 - y_sorted) / neg, 1.0]
            threshold = np.r_[np.inf, score_sorted, -np.inf]
            thinned = self._thin_arrays(500, fpr=fpr, tpr=tpr, threshold=threshold)
            for fpr_val, tpr_val, threshold_val in zip(
                thinned["fpr"],
                thinned["tpr"],
                thinned["threshold"],
                strict=False,
            ):
                rows.append(
                    {
                        self.group_col: group,
                        "fpr": float(fpr_val),
                        "tpr": float(tpr_val),
                        "threshold": float(threshold_val) if np.isfinite(threshold_val) else threshold_val,
                    }
                )
        return pd.DataFrame(rows)

    def _build_ks_curve_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        ordered_groups: Sequence[str],
    ) -> pd.DataFrame:
        """为每个切片构建 KS 曲线明细行。"""
        rows: List[Dict[str, Any]] = []
        for group in ordered_groups:
            sub_df = df[df[self.group_col].astype(str) == str(group)]
            y, score = self._valid_score_arrays(sub_df, pred_col=pred_col, target_col=target_col)
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
            thinned = self._thin_arrays(
                500,
                sample_pct=sample_pct,
                bad_cum_rate=bad_cum,
                good_cum_rate=good_cum,
                ks=ks,
            )
            for idx in range(len(thinned["sample_pct"])):
                rows.append(
                    {
                        self.group_col: group,
                        "sample_pct": float(thinned["sample_pct"][idx]),
                        "bad_cum_rate": float(thinned["bad_cum_rate"][idx]),
                        "good_cum_rate": float(thinned["good_cum_rate"][idx]),
                        "ks": float(thinned["ks"][idx]),
                    }
                )
        return pd.DataFrame(rows)

    def _build_calibration_curve_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        ordered_groups: Sequence[str],
    ) -> pd.DataFrame:
        """按分位箱构建校准曲线明细行。"""
        rows: List[Dict[str, Any]] = []
        for group in ordered_groups:
            sub_df = df[df[self.group_col].astype(str) == str(group)]
            y, score = self._valid_score_arrays(sub_df, pred_col=pred_col, target_col=target_col)
            if len(y) == 0:
                continue
            valid = pd.DataFrame({"target": y, "score": score})
            bin_count = min(10, max(int(valid["score"].nunique()), 1))
            if bin_count <= 1:
                valid["_bin"] = 1
            else:
                try:
                    valid["_bin"] = pd.qcut(valid["score"], q=bin_count, duplicates="drop", labels=False) + 1
                except ValueError:
                    valid["_bin"] = pd.cut(valid["score"], bins=bin_count, duplicates="drop", labels=False) + 1
            for bin_idx, part in valid.groupby("_bin", sort=True):
                rows.append(
                    {
                        self.group_col: group,
                        "bin": int(bin_idx) if pd.notna(bin_idx) else np.nan,
                        "count": int(part.shape[0]),
                        "pred_mean": float(part["score"].mean()),
                        "bad_rate": float(part["target"].mean()),
                    }
                )
        return pd.DataFrame(rows)

    def _build_score_distribution_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        ordered_groups: Sequence[str],
    ) -> pd.DataFrame:
        """按目标取值构建分箱后的分数分布明细行。"""
        scores = pd.to_numeric(df[pred_col], errors="coerce").dropna()
        if scores.empty:
            return pd.DataFrame()
        min_score = float(scores.min())
        max_score = float(scores.max())
        if min_score == max_score:
            min_score -= 1e-6
            max_score += 1e-6
        bins = np.linspace(min_score, max_score, 31)
        rows: List[Dict[str, Any]] = []
        for group in ordered_groups:
            sub_df = df[df[self.group_col].astype(str) == str(group)].copy()
            sub_df["_score"] = pd.to_numeric(sub_df[pred_col], errors="coerce")
            sub_df["_target"] = pd.to_numeric(sub_df[target_col], errors="coerce")
            sub_df = sub_df[sub_df["_score"].notna() & sub_df["_target"].notna() & (sub_df["_target"] >= 0)]
            for target_value, target_part in sub_df.groupby("_target", sort=True):
                counts = pd.cut(target_part["_score"], bins=bins, include_lowest=True).value_counts(sort=False)
                denom = max(float(counts.sum()), 1.0)
                for idx, interval in enumerate(counts.index):
                    rows.append(
                        {
                            self.group_col: group,
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

    def _build_feature_psi_detail(
        self,
        df: pd.DataFrame,
        *,
        ordered_groups: Sequence[str],
    ) -> pd.DataFrame:
        """复用风险分箱评估器构建特征级 PSI 明细行。"""
        feature_cols = [col for col in self.feature_cols if col in df.columns]
        if not ordered_groups or not feature_cols:
            return pd.DataFrame()
        detail_frames: list[pd.DataFrame] = []
        for feature in feature_cols:
            detail, _, _ = self._evaluate_psi_with_binner(
                df,
                feature=feature,
                target_col=self.target_col,
            )
            if detail.empty:
                continue
            detail_group_col = self._psi_detail_group_col(detail)
            feature_detail = detail.loc[
                (detail["feature"] == feature)
                & (detail[detail_group_col].astype(str).isin([str(group) for group in ordered_groups]))
            ].copy()
            if feature_detail.empty:
                continue
            if detail_group_col != self.group_col:
                feature_detail = feature_detail.rename(columns={detail_group_col: self.group_col})
            if "psi_bin" in feature_detail.columns:
                feature_detail["psi"] = feature_detail["psi_bin"]
                feature_detail["feature_psi"] = feature_detail.groupby(
                    ["feature", self.group_col]
                )["psi_bin"].transform("sum")
            detail_frames.append(feature_detail)
        if not detail_frames:
            return pd.DataFrame()
        feature_psi = pd.concat(detail_frames, ignore_index=True)
        return feature_psi.sort_values(
            ["feature_psi", "feature", self.group_col],
            ascending=[False, True, True],
        )

    def evaluate(
        self,
        df: FrameLike,
        *,
        pred_col: str,
        group_col: str,
        target: str,
        benchmark_col: str | None = None,
        benchmark_cols: Sequence[str] | None = None,
        time_col: str | None = None,
        val_target: str | None = None,
        aux_targets: Sequence[str] | None = None,
        target_group_cols: Mapping[str, str] | None = None,
        feature_cols: Sequence[str] | None = None,
        importance_table: pd.DataFrame | None = None,
        psi_include_missing: bool = False,
    ) -> MarsModelingReport:
        """
        对一次已打分样本构建模型评估报告。

        Parameters
        ----------
        df : FrameLike
            已包含预测分和目标列的样本表。
        pred_col : str
            当前模型预测分列名。
        group_col : str
            已存在的样本分组列名，常见取值是 `dataset_flag`。
        target : str
            二分类目标列名。
        benchmark_col : str | None
            benchmark 或 champion 模型分数列名；传入后会计算 AUC/KS 差异。
        benchmark_cols : Sequence[str] | None
            多个 benchmark 或 champion 模型分数列名；会分别输出 AUC、KS、F1
            和差异指标。
        time_col : str | None
            原始时间列名；传入后会在汇总表中展示每个分组的起止时间。
        val_target : str | None
            替代验证目标列名；适合在主目标之外同时观察另一个口径。
        aux_targets : Sequence[str] | None
            辅助验证目标列名；不参与训练，只按各自已表现样本生成评估区块。
        target_group_cols : Mapping[str, str] | None
            每个目标对应的独立样本切片列名，用于长短 y 表现期不一致的评估。
        feature_cols : Sequence[str] | None
            用于计算特征 PSI 明细的特征列名。
        importance_table : pd.DataFrame | None
            特征重要性表；会复制进报告元数据，便于导出和追踪。
        psi_include_missing : bool
            计算 `score_psi` 和 `feature_psi` 时是否纳入缺失值箱。

        Returns
        -------
        MarsModelingReport
            包含汇总指标、曲线/分布明细表和报告元数据的模型评估报告。

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "pred": [0.1, 0.8, 0.3, 0.7],
        ...     "target": [0, 1, 0, 1],
        ...     "sample": ["train", "train", "val", "val"],
        ... })
        >>> evaluator = MarsModelEvaluator()
        >>> report = evaluator.evaluate(df, pred_col="pred", group_col="sample", target="target")
        >>> report.summary_table.shape[0]
        2
        """
        self.group_col = group_col
        self.target_col = target
        self.benchmark_col = benchmark_col
        benchmark_col_list = list(benchmark_cols or [])
        if benchmark_col is not None and benchmark_col not in benchmark_col_list:
            benchmark_col_list.insert(0, benchmark_col)
        self.benchmark_cols = benchmark_col_list
        self.time_col = time_col
        self.val_target = val_target
        aux_target_list = list(aux_targets or [])
        if val_target is not None and val_target not in aux_target_list:
            aux_target_list.insert(0, val_target)
        self.aux_targets = aux_target_list
        self.target_group_cols = dict(target_group_cols or {})
        self.feature_cols = list(feature_cols or [])
        self.importance_table = None if importance_table is None else importance_table.copy()
        self.psi_include_missing = psi_include_missing

        df_pd = self._validate_frame(to_pandas_frame(df), pred_col)
        rows: List[Dict[Any, Any]] = []
        target_group_map = {
            self.target_col: self.group_col,
            **{
                target_name: self.target_group_cols.get(target_name, self.group_col)
                for target_name in self.aux_targets
            },
        }
        primary_ordered_groups = self._get_ordered_groups(df_pd)
        ordered_groups = sorted(
            {
                group
                for group_col_name in target_group_map.values()
                for group in self._get_ordered_groups_for_col(df_pd, group_col_name)
            },
            key=split_name_sort_key,
        )
        score_psi_detail, score_psi_map = self._build_score_psi_detail(
            df_pd,
            pred_col=pred_col,
            ordered_groups=primary_ordered_groups,
        )

        for group in ordered_groups:
            sub_df = df_pd[df_pd[self.group_col].astype(str) == str(group)].copy()
            row: Dict[Any, Any] = {self.group_col: group}

            if self.time_col:
                row[("Time Period", "Start Time")] = sub_df[self.time_col].min()
                row[("Time Period", "End Time")] = sub_df[self.time_col].max()

            row.update(
                self._calc_metric_block(
                    sub_df,
                    pred_col=pred_col,
                    target_col=self.target_col,
                    score_psi=score_psi_map.get(str(group)),
                )
            )
            for aux_target in self.aux_targets:
                aux_group_col = target_group_map[aux_target]
                aux_sub_df = df_pd[df_pd[aux_group_col].astype(str) == str(group)].copy()
                row.update(
                    self._calc_metric_block(
                        aux_sub_df,
                        pred_col=pred_col,
                        target_col=aux_target,
                        section_label=f"Aux Target: {aux_target}",
                    )
                )
            rows.append(row)

        summary = pd.DataFrame(rows).set_index(self.group_col)
        tuple_cols = [col for col in summary.columns if isinstance(col, tuple)]
        ordered_tuple_cols = self._get_ordered_columns(tuple_cols)
        summary = summary.reindex(columns=ordered_tuple_cols)
        summary.columns = pd.MultiIndex.from_tuples(ordered_tuple_cols)
        detail_tables = {
            "decile_lift": self._build_decile_lift_detail(
                df_pd,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=primary_ordered_groups,
            ),
            "score_psi": score_psi_detail,
            "roc_curve": self._build_roc_curve_detail(
                df_pd,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=primary_ordered_groups,
            ),
            "ks_curve": self._build_ks_curve_detail(
                df_pd,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=primary_ordered_groups,
            ),
            "calibration_curve": self._build_calibration_curve_detail(
                df_pd,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=primary_ordered_groups,
            ),
            "score_distribution": self._build_score_distribution_detail(
                df_pd,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=primary_ordered_groups,
            ),
        }
        feature_psi = self._build_feature_psi_detail(df_pd, ordered_groups=primary_ordered_groups)
        if not feature_psi.empty:
            detail_tables["feature_psi"] = feature_psi
        metadata: Dict[str, Any] = {
            "group_col": self.group_col,
            "target_col": self.target_col,
            "pred_col": pred_col,
            "benchmark_col": self.benchmark_col,
            "benchmark_cols": list(self.benchmark_cols),
            "time_col": self.time_col,
            "val_target": self.val_target,
            "aux_targets": list(self.aux_targets),
            "target_group_cols": dict(self.target_group_cols),
            "feature_cols": [col for col in self.feature_cols if col in df_pd.columns],
            "psi_include_missing": self.psi_include_missing,
        }
        if self.importance_table is not None:
            metadata["importance_table"] = self.importance_table.copy()
        return MarsModelingReport(
            summary,
            caption=f"Model Evaluation by [{self.group_col}]",
            detail_tables=detail_tables,
            metadata=metadata,
        )
