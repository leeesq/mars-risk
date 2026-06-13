"""建模评估报告构建器。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from mars.compute import FrameLike, to_pandas_frame
from mars.modeling.contracts.report import MarsModelingReport
from mars.modeling.evaluation._metric_block import calculate_metric_block
from mars.modeling.evaluation._ordering import split_name_sort_key
from mars.modeling.evaluation._psi_details import (
    build_feature_psi_detail,
    build_score_psi_detail,
)
from mars.modeling.evaluation.tables import (
    build_calibration_curve_detail,
    build_decile_lift_detail,
    build_ks_curve_detail,
    build_roc_curve_detail,
    build_score_distribution_detail,
)


class MarsModelEvaluator:
    """构建二分类模型分组评估报告。"""

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

    def __init__(self) -> None:
        """初始化一个空评估器。"""
        self.group_col = ""
        self.target_col = ""
        self.benchmark_col: str | None = None
        self.benchmark_cols: list[str] = []
        self.time_col: str | None = None
        self.val_target: str | None = None
        self.aux_targets: list[str] = []
        self.target_group_cols: dict[str, str] = {}
        self.feature_cols: list[str] = []
        self.importance_table: pd.DataFrame | None = None
        self.psi_include_missing = False

    def _validate_frame(self, df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
        """校验必需列，并统一时间列 dtype。"""
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

    def _get_ordered_groups(self, df: pd.DataFrame) -> list[str]:
        """返回主分组列的稳定顺序。"""
        return self._get_ordered_groups_for_col(df, self.group_col)

    @staticmethod
    def _get_ordered_groups_for_col(df: pd.DataFrame, group_col: str) -> list[str]:
        """返回任意分组列的稳定顺序。"""
        groups = df[group_col].dropna().astype(str).unique().tolist()
        return sorted(groups, key=split_name_sort_key)

    def _get_ordered_columns(self, available_columns: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """返回汇总表的稳定列顺序。"""
        ordered_columns: list[tuple[str, str]] = []
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

        return ordered_columns + [col for col in available_columns if col not in ordered_columns]

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
        """对已打分样本构建模型评估报告。"""
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
        score_psi_detail, score_psi_map = build_score_psi_detail(
            df_pd,
            group_col=self.group_col,
            pred_col=pred_col,
            target_col=self.target_col,
            ordered_groups=primary_ordered_groups,
            psi_include_missing=self.psi_include_missing,
        )

        rows: list[dict[Any, Any]] = []
        for group in ordered_groups:
            sub_df = df_pd[df_pd[self.group_col].astype(str) == str(group)].copy()
            row: dict[Any, Any] = {self.group_col: group}
            if self.time_col:
                row[("Time Period", "Start Time")] = sub_df[self.time_col].min()
                row[("Time Period", "End Time")] = sub_df[self.time_col].max()

            row.update(
                calculate_metric_block(
                    sub_df,
                    pred_col=pred_col,
                    target_col=self.target_col,
                    benchmark_cols=self.benchmark_cols,
                    score_psi=score_psi_map.get(str(group)),
                )
            )
            for aux_target in self.aux_targets:
                aux_group_col = target_group_map[aux_target]
                aux_sub_df = df_pd[df_pd[aux_group_col].astype(str) == str(group)].copy()
                row.update(
                    calculate_metric_block(
                        aux_sub_df,
                        pred_col=pred_col,
                        target_col=aux_target,
                        benchmark_cols=self.benchmark_cols,
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
            "decile_lift": build_decile_lift_detail(
                df_pd,
                group_col=self.group_col,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=primary_ordered_groups,
            ),
            "score_psi": score_psi_detail,
            "roc_curve": build_roc_curve_detail(
                df_pd,
                group_col=self.group_col,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=primary_ordered_groups,
            ),
            "ks_curve": build_ks_curve_detail(
                df_pd,
                group_col=self.group_col,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=primary_ordered_groups,
            ),
            "calibration_curve": build_calibration_curve_detail(
                df_pd,
                group_col=self.group_col,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=primary_ordered_groups,
            ),
            "score_distribution": build_score_distribution_detail(
                df_pd,
                group_col=self.group_col,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=primary_ordered_groups,
            ),
        }
        feature_psi = build_feature_psi_detail(
            df_pd,
            group_col=self.group_col,
            target_col=self.target_col,
            feature_cols=self.feature_cols,
            ordered_groups=primary_ordered_groups,
            psi_include_missing=self.psi_include_missing,
        )
        if not feature_psi.empty:
            detail_tables["feature_psi"] = feature_psi

        metadata: dict[str, Any] = {
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
