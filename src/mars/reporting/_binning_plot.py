"""分箱风险趋势图入口实现。"""

from __future__ import annotations

from typing import Any, List, Literal

import pandas as pd

from mars.compute import RiskCorrBaseline, normalize_risk_corr_baseline, to_pandas_frame
from mars.reporting.plotter import MarsPlotter
from mars.utils.logger import logger


class _BinningPlotRenderer:
    """分箱报告绘图能力。"""

    def __init__(self, report: Any) -> None:
        self._report = report

    def __getattr__(self, name: str) -> Any:
        """将只读数据访问委托给 report 容器。"""
        return getattr(self._report, name)

    @staticmethod
    def _normalize_name_list(values: str | List[str] | None) -> List[str] | None:
        """将单个名称或名称列表统一规范为字符串列表。"""
        if values is None:
            return None
        if isinstance(values, str):
            return [values]
        return [str(value) for value in values]

    @staticmethod
    def _resolve_plot_sort_key(sort_by: str) -> str:
        """将公开排序字段映射为汇总表中的实际列名。"""
        sort_key_map = {
            "iv": "iv",
            "ks": "ks",
            "auc": "auc",
            "psi": "psi_max",
            "rc": "rc_min",
            "risk_corr": "rc_min",
            "mono": "mono",
            "missing": "missing_max",
            "lift": "lift_max",
        }
        return sort_key_map.get(str(sort_by).lower(), str(sort_by))

    def _resolve_plot_risk_corr_baseline(
        self: Any,
        risk_corr_baseline: RiskCorrBaseline | None,
    ) -> RiskCorrBaseline:
        """解析绘图阶段生效的 RC 基准。"""
        meta_baseline = self.report_meta.get("risk_corr_baseline")
        return normalize_risk_corr_baseline(risk_corr_baseline or meta_baseline or "total")

    @staticmethod
    def _build_plot_risk_corr_reference(
        detail_pd: pd.DataFrame,
        *,
        group_col: str,
        baseline: RiskCorrBaseline,
        current_target: str | None,
        saved_reference: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """按目标和基准模式解析绘图所需的 RC 参考表。"""
        if baseline == "benchmark":
            if saved_reference is None or saved_reference.empty:
                raise ValueError(
                    "`risk_corr_baseline='benchmark'` requires a saved reference table in the report.",
                )
            reference_pd = saved_reference.copy()
            if current_target is not None and "y" in reference_pd.columns:
                reference_pd = reference_pd[reference_pd["y"].astype(str) == current_target].copy()
            if reference_pd.empty:
                raise ValueError(
                    f"Target {current_target!r} does not have benchmark RC reference data.",
                )
            return reference_pd[["feature", "bin_index", "base_br"]].copy()

        normal_detail = detail_pd[detail_pd["bin_index"] >= 0].copy()
        if normal_detail.empty:
            return pd.DataFrame(columns=["feature", "bin_index", "base_br"])

        if baseline == "total":
            total_reference = normal_detail[normal_detail[group_col].astype(str) == "Total"].copy()
            return total_reference.rename(columns={"bad_rate": "base_br"})[
                ["feature", "bin_index", "base_br"]
            ].copy()

        groups = sorted(
            group
            for group in normal_detail[group_col].astype(str).drop_duplicates().tolist()
            if group != "Total"
        )
        if not groups:
            return pd.DataFrame(columns=["feature", "bin_index", "base_br"])
        first_group = groups[0]
        return (
            normal_detail[normal_detail[group_col].astype(str) == first_group]
            .rename(columns={"bad_rate": "base_br"})[["feature", "bin_index", "base_br"]]
            .copy()
        )

    def _resolve_plot_features(
        self: Any,
        *,
        summary_pd: pd.DataFrame,
        detail_pd: pd.DataFrame,
        features: str | List[str] | None,
        target: str | None,
        sort_by: str,
        ascending: bool,
        max_plots: int,
    ) -> List[str]:
        """根据显式特征、目标筛选和排序规则确定最终绘图特征列表。"""
        available_features = detail_pd["feature"].astype(str).drop_duplicates().tolist()
        requested_features = self._normalize_name_list(features)
        if requested_features is not None:
            seen_features: set[str] = set()
            resolved_features: List[str] = []
            for feature in requested_features:
                if feature in available_features and feature not in seen_features:
                    resolved_features.append(feature)
                    seen_features.add(feature)
            return resolved_features

        scoped_summary = summary_pd.copy()
        if target is not None and "target" in scoped_summary.columns:
            scoped_summary = scoped_summary[scoped_summary["target"].astype(str) == target]

        sort_key = self._resolve_plot_sort_key(sort_by)
        if sort_key in scoped_summary.columns:
            scoped_summary = scoped_summary.sort_values(
                by=sort_key,
                ascending=ascending,
                na_position="last",
            )

        if not scoped_summary.empty and "feature" in scoped_summary.columns:
            ordered_features = [
                str(feature)
                for feature in scoped_summary["feature"].astype(str).drop_duplicates().tolist()
            ]
        else:
            ordered_features = available_features
        return ordered_features[:max_plots]

    def plot_risk_trends(
        self: Any,
        features: str | List[str] | None = None,
        *,
        target: str | List[str] | None = None,
        risk_corr_baseline: RiskCorrBaseline | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        sort_by: str = "iv",
        ascending: bool = False,
        max_plots: int = 20,
        dpi: int = 150,
    ) -> None:
        """
        直接展示分箱风险趋势图。

        Parameters
        ----------
        features : str | List[str] | None
            需要绘图的特征名。传入 ``None`` 时，会按 ``sort_by`` 和
            ``max_plots`` 从汇总表中自动挑选特征。
        target : str | List[str] | None
            多目标报告下需要展示的目标列名。传入 ``None`` 时，默认展示报告中的全部目标。
        risk_corr_baseline : RiskCorrBaseline | None
            绘图阶段使用的 RC 基准；传入 `None` 时沿用报告生成时保存的默认口径。
        show_risk : Literal["count", "amt", "both"]
            风险线展示模式。`count` 仅展示件数坏率，`amt` 仅展示金额坏率，
            `both` 同时展示两条风险线。
        sort_by : str
            未显式指定 ``features`` 时的特征排序字段。支持 ``iv``、``ks``、``auc``、
            ``psi``、``rc``、``risk_corr``、``mono``、``missing`` 和 ``lift``。
        ascending : bool
            是否按 ``sort_by`` 升序选择特征。
        max_plots : int
            未显式指定 ``features`` 时，最多展示的特征数量。
        dpi : int
            图像显示分辨率。

        Returns
        -------
        None
            图像会直接显示在当前交互环境中，函数本身不返回图形对象。

        Raises
        ------
        ValueError
            当 ``detail_table`` 为空、缺少分组列，或 ``target`` 指向不存在的目标时抛出。

        Examples
        --------
        >>> import polars as pl
        >>> summary = pl.DataFrame({"feature": ["age"], "iv": [0.12]})
        >>> detail = pl.DataFrame(
        ...     {
        ...         "y": ["target"],
        ...         "feature": ["age"],
        ...         "month": ["2026-01"],
        ...         "bin_index": [0],
        ...         "bin_label": ["[20, 40)"],
        ...         "count": [100],
        ...         "observed_count": [100],
        ...         "bad": [12],
        ...         "good": [88],
        ...         "pct": [1.0],
        ...         "bad_rate": [0.12],
        ...         "lift": [1.0],
        ...         "cum_count": [100],
        ...         "cum_observed_count": [100],
        ...         "cum_bad": [12],
        ...         "cum_bad_rate": [0.12],
        ...         "psi_bin": [0.0],
        ...         "ks_bin": [12.0],
        ...         "auc_bin": [0.61],
        ...         "iv_bin": [0.12],
        ...         "total_count": [100],
        ...         "bin_type": ["正常组"],
        ...     }
        ... )
        >>> report = MarsBinningReport(summary, {}, detail, group_col="month")
        >>> report.plot_risk_trends(features="age", dpi=80) is None
        True
        """
        detail_pd = to_pandas_frame(self.detail_table).copy()
        if detail_pd.empty:
            raise ValueError("detail_table is empty. Cannot plot risk trends.")
        show_risk = MarsPlotter._normalize_show_risk(show_risk)

        plot_group_col = self.detail_group_col or "mars_group"
        if plot_group_col not in detail_pd.columns:
            raise ValueError(
                f"Group column '{plot_group_col}' was not found in detail_table."
            )

        summary_pd = to_pandas_frame(self.summary_table).copy()
        reference_pd = (
            to_pandas_frame(self.risk_corr_reference_table).copy()
            if self.risk_corr_reference_table is not None
            else None
        )
        requested_targets = self._normalize_name_list(target)
        if "y" in detail_pd.columns and detail_pd["y"].notna().any():
            available_targets = detail_pd["y"].astype(str).drop_duplicates().tolist()
        else:
            available_targets = []

        if requested_targets is None:
            target_list: List[str | None] = available_targets if available_targets else [None]
        else:
            if not available_targets:
                target_list = [None]
            else:
                target_list = [item for item in requested_targets if item in available_targets]
                if not target_list:
                    raise ValueError(
                        f"Targets {requested_targets!r} were not found in this binning report."
                    )

        for current_target in target_list:
            current_detail = detail_pd.copy()
            if current_target is not None and "y" in current_detail.columns:
                current_detail = current_detail[
                    current_detail["y"].astype(str) == current_target
                ].copy()
            if current_detail.empty:
                logger.warning(
                    "Target '%s' has no detail rows in the current binning report.",
                    current_target,
                )
                continue

            plot_features = self._resolve_plot_features(
                summary_pd=summary_pd,
                detail_pd=current_detail,
                features=features,
                target=current_target,
                sort_by=sort_by,
                ascending=ascending,
                max_plots=max_plots,
            )
            if not plot_features:
                logger.warning("No features were available for risk trend plotting.")
                continue

            display_target = (
                current_target
                if current_target not in {None, "", "dummy_target"}
                else "Target"
            )
            effective_risk_corr_baseline = self._resolve_plot_risk_corr_baseline(
                risk_corr_baseline,
            )
            current_reference = self._build_plot_risk_corr_reference(
                current_detail,
                group_col=plot_group_col,
                baseline=effective_risk_corr_baseline,
                current_target=current_target,
                saved_reference=reference_pd,
            )
            MarsPlotter.plot_feature_binning_risk_trend_batch(
                df_detail=current_detail,
                features=plot_features,
                group_col=plot_group_col,
                target_name=display_target or "Target",
                target_key=current_target,
                dpi=dpi,
                show_risk=show_risk,
                sort_by="",
                ascending=ascending,
                risk_corr_reference_df=current_reference,
                risk_corr_baseline=effective_risk_corr_baseline,
            )
