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
        """按样本分组评估已有预测分，并生成指标及曲线明细。

        Parameters
        ----------
        df : FrameLike
            已包含预测分、主目标和分组列的样本表，不在此方法中训练或预测。
        pred_col : str
            待评估的预测分列。计算 LogLoss、Brier 和捕获率时，将分数裁剪到概率范围。
        group_col : str
            主目标的样本分组列，按 train、val、oot 和其他名称的顺序展示。
        target : str
            主目标列，非缺失标签必须为 0/1；空值表示未表现，不参与表现指标。
            全局有效标签只有一类时，PSI 分箱评估会报错。
        benchmark_col : str | None
            单个对照模型分列，默认不提供；列表中未包含时，将其插入 ``benchmark_cols`` 首位。
        benchmark_cols : Sequence[str] | None
            多个对照模型分列，默认无对照。只比较已有分数，不用这些列拟合 PSI 分箱。
        time_col : str | None
            用于汇总各主分组起止日期的时间列，默认不统计日期。
            解析失败的日期转为缺失值，不用于自动生成分组。
        val_target : str | None
            兼容的单个辅助目标入口，默认不提供；列表中未包含时，将其插入 ``aux_targets`` 首位。
        aux_targets : Sequence[str] | None
            额外评估的目标列，默认无辅助目标；不改变主目标的分箱和 PSI 基准。
        target_group_cols : Mapping[str, str] | None
            辅助目标到专用分组列的映射。未配置的辅助目标使用 ``group_col``。
        feature_cols : Sequence[str] | None
            需要计算特征 PSI 的列，默认不计算；不存在于输入中的列会跳过。
        importance_table : pd.DataFrame | None
            已计算的特征重要性表，默认不提供。只复制到报告元数据，不重新计算重要性。
        psi_include_missing : bool
            模型分和特征 PSI 是否纳入缺失箱，默认 ``False``，排除后重新归一化分布。

        Returns
        -------
        MarsModelingReport
            分组指标汇总、模型分及可选特征 PSI、ROC、KS、校准和分数分布明细。

        Notes
        -----
        本模块的 AUC、KS 和 F1 使用 0～100 百分制，坏率、LogLoss 和 Brier 保留原数值。
        表现指标要求目标和当前预测分均非空。单个分组无有效样本或标签只有一类时，区分度指标为 NaN。
        对照模型还会排除对照分缺失的行，因此对照指标与当前模型的样本数可能不同。
        PSI 复用原生等频 10 箱，在本次全量输入上拟合，并按主分组比较。
        缺少必需列时抛出 ValueError。
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
