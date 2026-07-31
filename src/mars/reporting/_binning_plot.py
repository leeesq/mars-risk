"""分箱风险趋势图入口实现。"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Literal, cast

import pandas as pd

from mars.compute import RiskCorrBaseline, normalize_risk_corr_baseline, to_pandas_frame
from mars.reporting._binning_html_helpers import slugify
from mars.reporting._matplotlib import require_pyplot
from mars.reporting._time_range import TimeRange, resolve_report_time_range
from mars.reporting._types import MarsHtmlRenderResult
from mars.reporting.plotter import MarsPlotter
from mars.utils.logger import logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class _RiskTrendPlotContext:
    """保存单个 target 下风险趋势图生成所需的上下文。"""

    detail: pd.DataFrame
    features: list[str]
    group_col: str
    target_name: str
    target_key: str | None
    risk_corr_reference: pd.DataFrame
    risk_corr_baseline: RiskCorrBaseline
    time_range: TimeRange


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

    def _resolve_risk_trend_time_range(self: Any) -> TimeRange:
        """解析报告风险趋势图使用的原始时间范围。"""
        return resolve_report_time_range(
            report_meta=self.report_meta,
            dt_col=self.dt_col,
        )

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

    def _resolve_risk_trend_contexts(
        self: Any,
        *,
        features: str | List[str] | None = None,
        target: str | List[str] | None = None,
        risk_corr_baseline: RiskCorrBaseline | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        sort_by: str = "iv",
        ascending: bool = False,
        max_plots: int = 20,
    ) -> tuple[list[_RiskTrendPlotContext], Literal["count", "amt", "both"]]:
        """解析多 target 报告的绘图上下文。"""
        time_range = self._resolve_risk_trend_time_range()
        detail_pd = to_pandas_frame(self.detail_table).copy()
        if detail_pd.empty:
            raise ValueError("detail_table is empty. Cannot plot risk trends.")
        normalized_show_risk = MarsPlotter._normalize_show_risk(show_risk)

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

        contexts: list[_RiskTrendPlotContext] = []
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
            contexts.append(
                _RiskTrendPlotContext(
                    detail=current_detail,
                    features=plot_features,
                    group_col=plot_group_col,
                    target_name=display_target or "Target",
                    target_key=current_target,
                    risk_corr_reference=current_reference,
                    risk_corr_baseline=effective_risk_corr_baseline,
                    time_range=time_range,
                )
            )
        return contexts, normalized_show_risk

    def build_risk_trend_figures(
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
    ) -> list[Figure]:
        """生成风险趋势图对象，不展示也不写文件。"""
        contexts, normalized_show_risk = self._resolve_risk_trend_contexts(
            features=features,
            target=target,
            risk_corr_baseline=risk_corr_baseline,
            show_risk=show_risk,
            sort_by=sort_by,
            ascending=ascending,
            max_plots=max_plots,
        )
        figures: list[Figure] = []
        for context in contexts:
            for feature in context.features:
                figure = MarsPlotter._build_feature_binning_risk_figure(
                    df_detail=context.detail,
                    feature=feature,
                    group_col=context.group_col,
                    target_name=context.target_name,
                    risk_corr_reference_df=context.risk_corr_reference,
                    show_risk=normalized_show_risk,
                    time_range=context.time_range,
                )
                if figure is not None:
                    figure.set_dpi(dpi)
                    figures.append(figure)
        return figures

    @staticmethod
    def _close_figure(figure: Figure) -> None:
        """关闭 Matplotlib figure，避免批量渲染后资源累积。"""
        pyplot = require_pyplot(feature_name="MarsBinningReport risk trend rendering")
        pyplot.close(figure)

    @staticmethod
    def _figure_to_svg(figure: Figure, *, close: bool) -> str:
        """将 figure 转换为可嵌入 HTML 的 SVG 字符串。"""
        buffer = BytesIO()
        figure.savefig(buffer, format="svg", bbox_inches="tight")
        buffer.seek(0)
        svg_text = buffer.read().decode("utf-8")
        svg_start = svg_text.find("<svg")
        if svg_start >= 0:
            svg_text = svg_text[svg_start:]
        if close:
            _BinningPlotRenderer._close_figure(figure)
        return svg_text

    @staticmethod
    def _figure_to_png_data_uri(figure: Figure, *, dpi: int, close: bool) -> str:
        """将 figure 转换为 PNG data URI。"""
        image_text = MarsPlotter._figure_to_base64(figure, dpi=dpi, close=close)
        return f"data:image/png;base64,{image_text}"

    @staticmethod
    def _build_asset_filename(
        *,
        index: int,
        target_name: str,
        feature: str,
        image_format: Literal["svg", "png"],
        filename_prefix: str,
    ) -> str:
        """生成稳定的图表资产文件名。"""
        target_slug = slugify(target_name)
        feature_slug = slugify(feature)
        return f"{filename_prefix}_{index:03d}_{target_slug}_{feature_slug}.{image_format}"

    @staticmethod
    def _write_figure_image(
        figure: Figure,
        path: Path,
        *,
        image_format: Literal["svg", "png"],
        dpi: int,
        close: bool,
    ) -> None:
        """将 figure 写为 SVG 或 PNG 文件。"""
        save_kwargs: dict[str, Any] = {"format": image_format, "bbox_inches": "tight"}
        if image_format == "png":
            save_kwargs["dpi"] = dpi
        figure.savefig(path, **save_kwargs)
        if close:
            _BinningPlotRenderer._close_figure(figure)

    @staticmethod
    def _relative_asset_src(asset_path: Path, relative_to: str | Path | None) -> str:
        """计算 HTML asset 模式使用的相对路径。"""
        if relative_to is None:
            return asset_path.as_posix()
        try:
            relative_path = asset_path.resolve().relative_to(Path(relative_to).resolve())
        except ValueError:
            relative_path = Path(os.path.relpath(asset_path.resolve(), Path(relative_to).resolve()))
        return relative_path.as_posix()

    @staticmethod
    def _validate_image_format(image_format: str) -> Literal["svg", "png"]:
        """校验并规范化图像格式。"""
        normalized = str(image_format).strip().lower()
        if normalized not in {"svg", "png"}:
            raise ValueError("`image_format` only supports 'svg' or 'png'.")
        return cast(Literal["svg", "png"], normalized)

    @staticmethod
    def _validate_embed_mode(embed_mode: str) -> Literal["inline", "asset"]:
        """校验并规范化 HTML 嵌入模式。"""
        normalized = str(embed_mode).strip().lower()
        if normalized not in {"inline", "asset"}:
            raise ValueError("`embed_mode` only supports 'inline' or 'asset'.")
        return cast(Literal["inline", "asset"], normalized)

    def save_risk_trend_images(
        self: Any,
        output_dir: str | Path,
        features: str | List[str] | None = None,
        *,
        target: str | List[str] | None = None,
        image_format: Literal["svg", "png"] = "svg",
        filename_prefix: str = "risk_trend",
        overwrite: bool = True,
        dpi: int = 150,
        risk_corr_baseline: RiskCorrBaseline | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        sort_by: str = "iv",
        ascending: bool = False,
        max_plots: int = 20,
    ) -> list[Path]:
        """保存风险趋势图为图片文件，并返回资产路径。"""
        image_format = self._validate_image_format(image_format)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        contexts, normalized_show_risk = self._resolve_risk_trend_contexts(
            features=features,
            target=target,
            risk_corr_baseline=risk_corr_baseline,
            show_risk=show_risk,
            sort_by=sort_by,
            ascending=ascending,
            max_plots=max_plots,
        )
        assets: list[Path] = []
        index = 1
        for context in contexts:
            for feature in context.features:
                figure = MarsPlotter._build_feature_binning_risk_figure(
                    df_detail=context.detail,
                    feature=feature,
                    group_col=context.group_col,
                    target_name=context.target_name,
                    risk_corr_reference_df=context.risk_corr_reference,
                    show_risk=normalized_show_risk,
                    time_range=context.time_range,
                )
                if figure is None:
                    continue
                filename = self._build_asset_filename(
                    index=index,
                    target_name=context.target_name,
                    feature=feature,
                    image_format=image_format,
                    filename_prefix=filename_prefix,
                )
                asset_path = output_path / filename
                if asset_path.exists() and not overwrite:
                    self._close_figure(figure)
                    raise FileExistsError(f"Risk trend image already exists: {asset_path}")
                self._write_figure_image(
                    figure,
                    asset_path,
                    image_format=image_format,
                    dpi=dpi,
                    close=True,
                )
                assets.append(asset_path)
                index += 1
        return assets

    def render_risk_trends_html(
        self: Any,
        features: str | List[str] | None = None,
        *,
        target: str | List[str] | None = None,
        image_format: Literal["svg", "png"] = "svg",
        embed_mode: Literal["inline", "asset"] = "inline",
        output_dir: str | Path | None = None,
        relative_to: str | Path | None = None,
        include_title: bool = True,
        include_caption: bool = True,
        return_figures: bool = False,
        dpi: int = 150,
        filename_prefix: str = "risk_trend",
        overwrite: bool = True,
        risk_corr_baseline: RiskCorrBaseline | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        sort_by: str = "iv",
        ascending: bool = False,
        max_plots: int = 20,
    ) -> MarsHtmlRenderResult:
        """渲染可嵌入外部 HTML 报告的风险趋势图片段。"""
        image_format = self._validate_image_format(image_format)
        embed_mode = self._validate_embed_mode(embed_mode)
        if embed_mode == "asset" and output_dir is None:
            raise ValueError("`output_dir` is required when `embed_mode='asset'`.")

        contexts, normalized_show_risk = self._resolve_risk_trend_contexts(
            features=features,
            target=target,
            risk_corr_baseline=risk_corr_baseline,
            show_risk=show_risk,
            sort_by=sort_by,
            ascending=ascending,
            max_plots=max_plots,
        )
        output_path = Path(output_dir) if output_dir is not None else None
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

        blocks: list[str] = []
        assets: list[Path] = []
        returned_figures: list[Figure] = []
        index = 1
        for context in contexts:
            for feature in context.features:
                figure = MarsPlotter._build_feature_binning_risk_figure(
                    df_detail=context.detail,
                    feature=feature,
                    group_col=context.group_col,
                    target_name=context.target_name,
                    risk_corr_reference_df=context.risk_corr_reference,
                    show_risk=normalized_show_risk,
                    time_range=context.time_range,
                )
                if figure is None:
                    continue

                if embed_mode == "inline" and image_format == "svg":
                    image_html = self._figure_to_svg(figure, close=not return_figures)
                elif embed_mode == "inline":
                    image_src = self._figure_to_png_data_uri(
                        figure,
                        dpi=dpi,
                        close=not return_figures,
                    )
                    image_html = (
                        f'<img class="mars-risk-trend-image" src="{image_src}" '
                        f'alt="{html.escape(feature)} risk trend" />'
                    )
                else:
                    assert output_path is not None
                    filename = self._build_asset_filename(
                        index=index,
                        target_name=context.target_name,
                        feature=feature,
                        image_format=image_format,
                        filename_prefix=filename_prefix,
                    )
                    asset_path = output_path / filename
                    if asset_path.exists() and not overwrite:
                        if not return_figures:
                            self._close_figure(figure)
                        raise FileExistsError(f"Risk trend image already exists: {asset_path}")
                    self._write_figure_image(
                        figure,
                        asset_path,
                        image_format=image_format,
                        dpi=dpi,
                        close=not return_figures,
                    )
                    assets.append(asset_path)
                    image_src = self._relative_asset_src(asset_path, relative_to)
                    image_html = (
                        f'<img class="mars-risk-trend-image" src="{html.escape(image_src)}" '
                        f'alt="{html.escape(feature)} risk trend" />'
                    )

                if return_figures:
                    returned_figures.append(figure)

                caption = (
                    f'<figcaption>{html.escape(context.target_name)} / '
                    f'{html.escape(feature)}</figcaption>'
                    if include_caption
                    else ""
                )
                blocks.append(
                    '<figure class="mars-risk-trend" '
                    f'data-target="{html.escape(context.target_name)}" '
                    f'data-feature="{html.escape(feature)}">{image_html}{caption}</figure>'
                )
                index += 1

        title_html = "<h3>MARS Risk Trends</h3>" if include_title else ""
        fragment = f'<div class="mars-risk-trends">{title_html}{"".join(blocks)}</div>'
        return MarsHtmlRenderResult(
            html=fragment,
            assets=assets,
            figures=returned_figures if return_figures else None,
        )

    @staticmethod
    def _display_figure(figure: Figure, *, dpi: int, close: bool) -> None:
        """在交互环境中展示 figure，可选择是否关闭原图。"""
        from IPython.display import HTML, display

        display(
            HTML(
                MarsPlotter._build_image_html(
                    MarsPlotter._figure_to_base64(figure, dpi=dpi, close=close),
                ),
            ),
        )

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
        return_figures: bool = False,
    ) -> list[Figure] | None:
        """直接展示分箱风险趋势图，并可选返回 figure 对象。"""
        figures: list[Figure] = self.build_risk_trend_figures(
            features=features,
            target=target,
            risk_corr_baseline=risk_corr_baseline,
            show_risk=show_risk,
            sort_by=sort_by,
            ascending=ascending,
            max_plots=max_plots,
        )
        for figure in figures:
            self._display_figure(figure, dpi=dpi, close=not return_figures)
        if return_figures:
            return figures
        return None
