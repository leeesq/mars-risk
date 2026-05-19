"""Evaluation reports and prediction helpers for ``mars.modeling``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import importlib

import numpy as np
import pandas as pd
import polars as pl

from mars.modeling.base import (
    FrameLike,
    calculate_auc,
    calculate_ks,
    is_polars_dataframe,
    restore_frame_type,
    split_name_sort_key,
    to_pandas_frame,
)


def _optional_import(module_name: str) -> Any:
    """Import an optional dependency and return ``None`` when unavailable."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


class MarsModelingReport:
    """Container for grouped model evaluation summaries."""

    def __init__(
        self,
        summary_table: pd.DataFrame,
        caption: str = "MARS Model Evaluation",
        detail_tables: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        self.summary_table: pd.DataFrame = summary_table
        self.caption: str = caption
        self.detail_tables: Dict[str, pd.DataFrame] = dict(detail_tables or {})

    @property
    def styled_summary(self) -> Any:
        """Return a styled summary suitable for notebook rendering."""
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
        """Return the styled summary view."""
        return self.styled_summary

    def to_pandas(self) -> pd.DataFrame:
        """Return a copy of the underlying summary table."""
        return self.summary_table.copy()

    def write_excel(self, path: str = "mars_model_evaluation.xlsx", engine: Optional[str] = None) -> None:
        """Write the summary table to an Excel workbook."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if not self.detail_tables:
            self.summary_table.to_excel(path_obj, engine=engine)
            return
        with pd.ExcelWriter(path_obj, engine=engine) as writer:
            self.summary_table.to_excel(writer, sheet_name="summary")
            for name, table in self.detail_tables.items():
                table.to_excel(writer, sheet_name=str(name)[:31], index=False)


class MarsModelEvaluator:
    """Reusable evaluation tool for scored binary risk datasets."""

    COLUMN_ORDER = [
        "Total Count",
        "Good",
        "Bad",
        "Bad Rate",
        "New AUC",
        "New KS",
        "LogLoss",
        "Brier",
        "Score PSI",
        "Top 10% Capture",
        "Top 20% Capture",
        "Bench AUC",
        "Bench KS",
        "AUC Diff",
        "KS Diff",
    ]

    def __init__(
        self,
        *,
        group_col: str,
        target_col: str,
        benchmark_col: Optional[str] = None,
        time_col: Optional[str] = None,
        val_target_col: Optional[str] = None,
    ) -> None:
        self.group_col: str = group_col
        self.target_col: str = target_col
        self.benchmark_col: Optional[str] = benchmark_col
        self.time_col: Optional[str] = time_col
        self.val_target_col: Optional[str] = val_target_col

    def _validate_frame(self, df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
        """Validate required columns and normalize time columns when configured."""
        required = {self.group_col, pred_col, self.target_col}
        if self.time_col:
            required.add(self.time_col)
        if self.benchmark_col:
            required.add(self.benchmark_col)
        if self.val_target_col:
            required.add(self.val_target_col)

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
        section_label: Optional[str] = None,
        score_psi: Optional[float] = None,
    ) -> Dict[Tuple[str, str], Any]:
        """Build one target block for the grouped evaluation report."""
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
            block[(section, "LogLoss")] = np.nan
            block[(section, "Brier")] = np.nan
            block[(section, "Top 10% Capture")] = np.nan
            block[(section, "Top 20% Capture")] = np.nan

        block[(section, "Score PSI")] = score_psi if score_psi is not None else np.nan

        if self.benchmark_col:
            bench_pred = pd.to_numeric(sub_df[self.benchmark_col], errors="coerce")
            bench_mask = valid_mask & bench_pred.notna()
            bench_y = y_true[bench_mask]
            bench_scores = bench_pred[bench_mask]
            if bench_y.shape[0] > 0 and bench_y.nunique() >= 2:
                bench_auc = calculate_auc(bench_y.to_numpy(), bench_scores.to_numpy())
                bench_ks = calculate_ks(bench_y.to_numpy(), bench_scores.to_numpy())
            else:
                bench_auc = np.nan
                bench_ks = np.nan

            block[(section, "Bench AUC")] = bench_auc
            block[(section, "Bench KS")] = bench_ks
            block[(section, "AUC Diff")] = (
                block[(section, "New AUC")] - bench_auc
                if pd.notna(block[(section, "New AUC")]) and pd.notna(bench_auc)
                else np.nan
            )
            block[(section, "KS Diff")] = (
                block[(section, "New KS")] - bench_ks
                if pd.notna(block[(section, "New KS")]) and pd.notna(bench_ks)
                else np.nan
            )

        return block

    def _get_ordered_groups(self, df: pd.DataFrame) -> List[str]:
        """Return grouped split names in stable MARS order."""
        groups = df[self.group_col].astype(str).unique().tolist()
        return sorted(groups, key=split_name_sort_key)

    def _get_ordered_columns(self, available_columns: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Return a stable column layout for the final report."""
        ordered_columns: List[Tuple[str, str]] = []
        if self.time_col:
            for time_name in ("Start Time", "End Time"):
                candidate = ("Time Period", time_name)
                if candidate in available_columns:
                    ordered_columns.append(candidate)

        sections = [f"Target: {self.target_col}"]
        if self.val_target_col:
            sections.append(f"Val Target: {self.val_target_col}")

        for section in sections:
            for column_name in self.COLUMN_ORDER:
                candidate = (section, column_name)
                if candidate in available_columns:
                    ordered_columns.append(candidate)

        remaining_columns = [col for col in available_columns if col not in ordered_columns]
        return ordered_columns + remaining_columns

    def _build_score_bins(self, baseline_scores: pd.Series) -> Optional[np.ndarray]:
        """Build stable decile cut points from the first available group."""
        clean_scores = pd.to_numeric(baseline_scores, errors="coerce").dropna()
        if clean_scores.nunique() < 2:
            return None
        quantiles = np.linspace(0.0, 1.0, 11)
        bins = np.unique(np.quantile(clean_scores.to_numpy(dtype=float), quantiles))
        if bins.size < 2:
            return None
        bins[0] = -np.inf
        bins[-1] = np.inf
        return bins

    def _build_score_psi_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        ordered_groups: Sequence[str],
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Calculate score distribution PSI for each group against the first group."""
        if not ordered_groups:
            return pd.DataFrame(), {}
        baseline_group = ordered_groups[0]
        baseline_scores = df.loc[df[self.group_col].astype(str) == str(baseline_group), pred_col]
        bins = self._build_score_bins(baseline_scores)
        if bins is None:
            return pd.DataFrame(), {str(group): np.nan for group in ordered_groups}

        rows: List[Dict[str, Any]] = []
        expected_counts = pd.cut(
            pd.to_numeric(baseline_scores, errors="coerce"),
            bins=bins,
            include_lowest=True,
            duplicates="drop",
        ).value_counts(sort=False)
        expected_dist = expected_counts / max(float(expected_counts.sum()), 1.0)
        psi_map: Dict[str, float] = {}

        for group in ordered_groups:
            group_scores = pd.to_numeric(
                df.loc[df[self.group_col].astype(str) == str(group), pred_col],
                errors="coerce",
            )
            actual_counts = pd.cut(
                group_scores,
                bins=bins,
                include_lowest=True,
                duplicates="drop",
            ).value_counts(sort=False)
            actual_dist = actual_counts / max(float(actual_counts.sum()), 1.0)
            psi_values = (actual_dist - expected_dist) * np.log(
                (actual_dist + 1e-6) / (expected_dist + 1e-6)
            )
            psi_map[str(group)] = float(psi_values.sum())
            for idx, interval in enumerate(expected_dist.index):
                rows.append(
                    {
                        self.group_col: group,
                        "bin": idx + 1,
                        "score_range": str(interval),
                        "expected_pct": float(expected_dist.iloc[idx]),
                        "actual_pct": float(actual_dist.iloc[idx]),
                        "psi": float(psi_values.iloc[idx]),
                    }
                )

        return pd.DataFrame(rows), psi_map

    def _build_decile_lift_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        ordered_groups: Sequence[str],
    ) -> pd.DataFrame:
        """Build grouped decile lift details ordered by descending model score."""
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

    def evaluate(self, df: FrameLike, *, pred_col: str) -> MarsModelingReport:
        """Evaluate a scored dataset and return a structured report object."""
        df_pd = self._validate_frame(to_pandas_frame(df), pred_col)
        rows: List[Dict[Any, Any]] = []
        ordered_groups = self._get_ordered_groups(df_pd)
        score_psi_detail, score_psi_map = self._build_score_psi_detail(
            df_pd,
            pred_col=pred_col,
            ordered_groups=ordered_groups,
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
            if self.val_target_col:
                row.update(
                    self._calc_metric_block(
                        sub_df,
                        pred_col=pred_col,
                        target_col=self.val_target_col,
                        section_label=f"Val Target: {self.val_target_col}",
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
                ordered_groups=ordered_groups,
            ),
            "score_psi": score_psi_detail,
        }
        return MarsModelingReport(
            summary,
            caption=f"Model Evaluation by [{self.group_col}]",
            detail_tables=detail_tables,
        )

class _ModelPredictor:
    """Internal prediction helper for trained tree models."""

    def __init__(
        self,
        model: Any,
        feature_list: Sequence[str],
        categorical_features: Optional[Sequence[str]] = None,
        category_levels: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> None:
        self.model: Any = model
        self.features: List[str] = list(feature_list)
        self.categorical_features: List[str] = list(categorical_features or [])
        self.category_levels: Dict[str, List[Any]] = {
            str(feature): list(levels)
            for feature, levels in dict(category_levels or {}).items()
        }

    def _safe_predict_logic(self, df: pd.DataFrame) -> np.ndarray:
        """Dispatch prediction logic to the correct backend implementation."""
        missing = sorted(set(self.features).difference(df.columns))
        if missing:
            raise ValueError(f"Input data is missing required features: {missing}")

        X = df.loc[:, self.features].copy()
        for feature in self.categorical_features:
            if feature in X.columns:
                categories = self.category_levels.get(feature)
                if categories is not None:
                    X[feature] = X[feature].astype(pd.CategoricalDtype(categories=categories))
                else:
                    X[feature] = X[feature].astype("category")

        xgb = _optional_import("xgboost")
        lgb = _optional_import("lightgbm")
        catboost = _optional_import("catboost")

        if xgb is not None and isinstance(self.model, getattr(xgb, "Booster", tuple())):
            dtest = xgb.DMatrix(X, enable_categorical=bool(self.categorical_features))
            best_iteration = getattr(self.model, "best_iteration", None)
            if best_iteration is None:
                return np.asarray(self.model.predict(dtest))
            return np.asarray(self.model.predict(dtest, iteration_range=(0, best_iteration + 1)))

        if xgb is not None and isinstance(self.model, getattr(xgb, "XGBModel", tuple())):
            preds = self.model.predict_proba(X)
            return np.asarray(preds[:, 1])

        if lgb is not None and isinstance(self.model, getattr(lgb, "Booster", tuple())):
            best_iteration = getattr(self.model, "best_iteration", None)
            return np.asarray(self.model.predict(X, num_iteration=best_iteration or None))

        if lgb is not None and isinstance(self.model, getattr(lgb, "LGBMModel", tuple())):
            preds = self.model.predict_proba(X)
            return np.asarray(preds[:, 1])

        if catboost is not None and isinstance(self.model, getattr(catboost, "CatBoost", tuple())):
            preds = self.model.predict_proba(X)
            return np.asarray(preds[:, 1])

        raise TypeError(f"Unsupported model type: {type(self.model)!r}")

    def _safe_predict_logic_polars(self, df: pl.DataFrame) -> np.ndarray:
        """Predict directly from Polars/Arrow for numeric-only booster paths."""
        if self.categorical_features:
            return self._safe_predict_logic(df.to_pandas())

        missing = sorted(set(self.features).difference(df.columns))
        if missing:
            raise ValueError(f"Input data is missing required features: {missing}")

        X_arrow = df.select(self.features).to_arrow()
        xgb = _optional_import("xgboost")
        lgb = _optional_import("lightgbm")

        if xgb is not None and isinstance(self.model, getattr(xgb, "Booster", tuple())):
            dtest = xgb.DMatrix(X_arrow)
            best_iteration = getattr(self.model, "best_iteration", None)
            if best_iteration is None:
                return np.asarray(self.model.predict(dtest))
            return np.asarray(self.model.predict(dtest, iteration_range=(0, best_iteration + 1)))

        if lgb is not None and isinstance(self.model, getattr(lgb, "Booster", tuple())):
            best_iteration = getattr(self.model, "best_iteration", None)
            return np.asarray(self.model.predict(X_arrow, num_iteration=best_iteration or None))

        return self._safe_predict_logic(df.to_pandas())

    def predict(
        self,
        df: FrameLike,
        pred_col_name: str = "pred_score",
        inplace: bool = False,
    ) -> FrameLike:
        """Score a dataset and append the prediction column."""
        prefer_polars = is_polars_dataframe(df)
        if prefer_polars and not inplace and isinstance(df, pl.DataFrame):
            preds = self._safe_predict_logic_polars(df)
            return df.with_columns(pl.Series(pred_col_name, preds))
        df_pd = df if isinstance(df, pd.DataFrame) and inplace else to_pandas_frame(df)
        df_pd[pred_col_name] = self._safe_predict_logic(df_pd)
        return restore_frame_type(df_pd, prefer_polars)

    def evaluate(
        self,
        df: FrameLike,
        group_col: str,
        target_col: str,
        *,
        time_col: Optional[str] = None,
        val_target_col: Optional[str] = None,
        benchmark_col: Optional[str] = None,
        pred_col_name: str = "pred_score",
    ) -> MarsModelingReport:
        """Score a dataset and immediately return an evaluation report."""
        scored = self.predict(df, pred_col_name=pred_col_name, inplace=False)
        evaluator = MarsModelEvaluator(
            group_col=group_col,
            target_col=target_col,
            time_col=time_col,
            benchmark_col=benchmark_col,
            val_target_col=val_target_col,
        )
        return evaluator.evaluate(scored, pred_col=pred_col_name)
