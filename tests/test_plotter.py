from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text

from mars.compute import RiskCorrBaseline
from mars.reporting import MarsBinningReport
from mars.reporting.plotter import MarsPlotter


def _build_group_date_report(
    *,
    ks_method: Literal["binned", "raw"] = "binned",
    explicit_group: bool = True,
    risk_corr_baseline: RiskCorrBaseline = "total",
) -> MarsBinningReport:
    """不同真实组日期、未表现样本与缺日期组共同覆盖原生绘图上下文。"""
    from mars.analysis import profile_risk

    frame = pl.DataFrame({
        "income": [0.1, 0.2, 0.4, 0.5, 0.4, 0.7, 0.8, None, None, 0.9, 0.2, 0.5],
        "segment": ["Train"] * 4 + ["Total loans"] * 4 + ["Undue"] * 2 + ["No dates"] * 2,
        "date": [
            "2026-01-01", "20260103", "2026/01/02", "2026-01-20",
            "2026-02-01", "2026-02-03", "2026-02-05", "2026-02-28",
            "2026-03-01", "2026-03-09", None, "invalid",
        ],
        "target": [0, 1, 0, None, 1, 0, 0, 1, None, None, None, None],
        "target_aux": [1, 0, 1, None, 0, 1, 1, 0, None, None, None, None],
        "amount": [100.0, 200.0, 300.0, 400.0] * 3,
    })
    benchmark = frame.head(4).with_columns(pl.lit("2020-01-01").alias("date"))
    return profile_risk(
        frame, target=["target", "target_aux"], features=["income"],
        group_col="segment" if explicit_group else None, time_col="date",
        binning_type="native", n_bins=3, ks_method=ks_method,
        feature_metadata={"income": {"display_name": "家庭收入（元）"}},
        benchmark_df=benchmark, amount_col="amount",
        risk_corr_baseline=risk_corr_baseline,
    ).report


def _build_panel_title_frame(
    *,
    bad: int,
    observed: int,
    total: int,
    group: str = "2026-01",
) -> pd.DataFrame:
    count_left = total // 2
    observed_left = observed // 2
    bad_left = bad // 2
    counts = [count_left, total - count_left]
    observed_counts = [observed_left, observed - observed_left]
    bad_counts = [bad_left, bad - bad_left]
    assert len(bad_counts) == len(observed_counts)
    bad_rates = [
        bad_count / observed_count if observed_count > 0 else float("nan")
        for bad_count, observed_count in zip(bad_counts, observed_counts)
    ]
    return pd.DataFrame(
        {
            "feature": ["income", "income"],
            "month": [group, group],
            "bin_index": [0, 1],
            "bin_label": ["bin0", "bin1"],
            "count": counts,
            "observed_count": observed_counts,
            "bad": bad_counts,
            "bad_rate": bad_rates,
            "lift": [1.0, 1.0],
            "psi_bin": [0.01, 0.02],
            "ks_bin": [0.1, 0.2],
            "auc_bin": [0.55, 0.0],
            "iv_bin": [0.03, 0.02],
            "total_count": [total, total],
        }
    )


def _group_date_text(axis: Axes) -> Text:
    """查找独立日期行，避免将日期和旧标题混成多行文本。"""
    return next(text for text in axis.texts if text.get_gid() == "mars-group-date")


def _old_text_styles(figure: Figure) -> list[tuple[Any, ...]]:
    """同时记录声明属性和实际解析字体，覆盖粗体回退导致的样式变化。"""
    return [
        (
            text.get_text(),
            tuple(text.get_fontfamily()),
            findfont(text.get_fontproperties()),
            text.get_fontsize(),
            text.get_fontweight(),
            mpl.colors.to_rgba(text.get_color()),
            text.get_rotation(),
            text.get_horizontalalignment(),
            text.get_verticalalignment(),
        )
        for text in figure.findobj(match=Text)
        if text.get_text()
        and text.get_gid() not in {"mars-group-date", "mars-display-name"}
    ]


def _axes_geometry_inches(figure: Figure) -> list[tuple[float, ...]]:
    """以物理尺寸比较每个轴和轴间距，避免画布加高掩盖绘图区压缩。"""
    width, height = figure.get_size_inches()
    return [
        (box.x0 * width, box.y0 * height, box.width * width, box.height * height)
        for axis in figure.axes
        for box in [axis.get_position()]
    ]


def _check_risk_trend_header_layout(
    figure: Figure, renderer: Any, legacy: Figure,
) -> tuple[float | None, float]:
    """用本次绘制的 renderer 对照旧布局净距，同时检查全部新增行和背景边界。"""
    dpi = renderer.points_to_pixels(72)
    points_per_pixel = 72 / dpi
    axis = next(axis for axis in figure.axes if axis.get_title())
    title_box = axis.title.get_window_extent(renderer)
    metrics = next(text for text in axis.texts if text.get_text().startswith("IV:"))
    metrics_box = metrics.get_window_extent(renderer)
    date_box = _group_date_text(axis).get_window_extent(renderer)
    old_gap = (title_box.y0 - metrics_box.y1) * points_per_pixel
    assert old_gap > 0
    assert (date_box.y0 - title_box.y1) * points_per_pixel == pytest.approx(
        old_gap, abs=points_per_pixel,
    )

    # 旧路径已有独立的字体和英寸几何合同；只在同 renderer 更新局部 bbox，不触发绘制。
    legacy.set_dpi(dpi)
    legacy.texts[0].update_bbox_position_size(renderer)
    legacy_patch = legacy.texts[0].get_bbox_patch()
    assert legacy_patch is not None
    legacy_axis = next(axis for axis in legacy.axes if axis.get_title())
    expected_summary_gap = max(
        # 既有布局对极小旧净距保留 2pt 的最小可见距离。
        2.0,
        (
            legacy_patch.get_window_extent(renderer).y0
            - legacy_axis.title.get_window_extent(renderer).y1
        ) * points_per_pixel,
    )

    summary = figure.texts[0]
    display_texts = [text for text in figure.texts if text.get_gid() == "mars-display-name"]
    name_gap: float | None = None
    if display_texts:
        assert len(display_texts) == 1
        display_box = display_texts[0].get_window_extent(renderer)
        english_box = summary.get_window_extent(renderer)
        line_probe = Text(
            text=summary.get_text().splitlines()[0],
            fontproperties=summary.get_fontproperties(),
            linespacing=1.6,
        )
        line_probe.set_figure(figure)
        line_probe.set_parse_math(False)
        original_line_gap = (
            english_box.height - 2 * line_probe.get_window_extent(renderer).height
        ) * points_per_pixel
        name_gap = (display_box.y0 - english_box.y1) * points_per_pixel
        assert name_gap >= -1e-9  # tight 平移可能留下约 1e-14pt 的浮点误差。
        assert name_gap == pytest.approx(original_line_gap, abs=points_per_pixel)
        summary_patch = next(patch for patch in figure.artists
                             if patch.get_gid() == "mars-summary-box")
    else:
        summary_patch = summary.get_bbox_patch()
    assert summary_patch is not None
    summary_box = summary_patch.get_window_extent(renderer)
    for text in [summary, *display_texts]:
        box = text.get_window_extent(renderer)
        assert summary_box.x0 <= box.x0 <= box.x1 <= summary_box.x1
        assert summary_box.y0 <= box.y0 <= box.y1 <= summary_box.y1
    summary_gap = (summary_box.y0 - date_box.y1) * points_per_pixel
    # 一像素只覆盖渲染量化，仍锁定旧可见净距，不能吸收 6pt 对 3.312pt 的漂移。
    assert summary_gap == pytest.approx(expected_summary_gap, abs=points_per_pixel)
    assert summary_gap > 0
    assert summary_box.y1 <= figure.bbox.y1
    return name_gap, summary_gap


def test_plot_feature_binning_risk_trend_uses_shared_figure_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fig = plt.figure()
    captured = {}

    def fake_build_feature_binning_risk_figure(
        *,
        df_detail: pd.DataFrame,
        feature: str,
        group_col: str,
        target_name: str,
        risk_corr_reference_df: pd.DataFrame | None = None,
        show_risk: Literal["count", "amt", "both"] = "both",
        time_range: tuple[str, str] | None = None,
        include_total: bool = True,
    ) -> Figure:
        captured["feature"] = feature
        captured["group_col"] = group_col
        captured["target_name"] = target_name
        captured["risk_corr_reference_df"] = risk_corr_reference_df
        captured["show_risk"] = show_risk
        captured["time_range"] = time_range
        captured["include_total"] = include_total
        return fig

    def fake_show_scrollable(passed_fig: Figure, dpi: int = 150) -> None:
        captured["fig"] = passed_fig
        captured["dpi"] = dpi

    monkeypatch.setattr(
        MarsPlotter,
        "_build_feature_binning_risk_figure",
        staticmethod(fake_build_feature_binning_risk_figure),
    )
    monkeypatch.setattr(
        MarsPlotter,
        "_show_scrollable",
        staticmethod(fake_show_scrollable),
    )

    MarsPlotter.plot_feature_binning_risk_trend(
        pd.DataFrame({"feature": []}),
        feature="income",
        group_col="month",
        target_name="target",
        dpi=180,
        time_range=("2024-01-01", "2024-03-31"),
        include_total=False,
    )

    assert captured["feature"] == "income"
    assert captured["group_col"] == "month"
    assert captured["target_name"] == "target"
    assert captured["risk_corr_reference_df"] is None
    assert captured["show_risk"] == "both"
    assert captured["time_range"] == ("2024-01-01", "2024-03-31")
    assert captured["include_total"] is False
    assert captured["fig"] is fig
    assert captured["dpi"] == 180

    plt.close(fig)


def test_shared_figure_builder_excludes_total_bin_rows():
    df = pd.DataFrame(
        {
            "feature": ["income"] * 8,
            "month": ["2024-01"] * 4 + ["Total"] * 4,
            "bin_index": [0, 1, -1, 9999, 0, 1, -1, 9999],
            "bin_label": ["bin0", "bin1", "Missing", "Total", "bin0", "bin1", "Missing", "Total"],
            "bin_type": ["正常组", "正常组", "空值组", "汇总组", "正常组", "正常组", "空值组", "汇总组"],
            "count": [40, 30, 10, 80, 40, 30, 10, 80],
            "observed_count": [40, 30, 10, 80, 40, 30, 10, 80],
            "bad": [8, 6, 3, 17, 8, 6, 3, 17],
            "bad_rate": [0.20, 0.20, 0.30, 0.2125, 0.20, 0.20, 0.30, 0.2125],
            "lift": [1.2, 1.0, 0.8, 1.2, 1.2, 1.0, 0.8, 1.2],
            "psi_bin": [0.01, 0.02, 0.0, 0.03, 0.01, 0.02, 0.0, 0.03],
            "ks_bin": [0.1, 0.2, 0.0, 0.2, 0.1, 0.2, 0.0, 0.2],
            "auc_bin": [0.55, 0.0, 0.0, 0.55, 0.55, 0.0, 0.0, 0.55],
            "iv_bin": [0.03, 0.02, 0.01, 0.06, 0.03, 0.02, 0.01, 0.06],
            "trend": ["asc"] * 8,
            "total_count": [80] * 8,
        }
    )

    fig = MarsPlotter._build_feature_binning_risk_figure(
        df_detail=df,
        feature="income",
        group_col="month",
        target_name="target",
        time_range=("2024-01-01", "2024-03-31"),
    )

    labels = []
    for ax in fig.axes:
        labels.extend([tick.get_text() for tick in ax.get_xticklabels()])

    assert "Total" not in labels
    plt.close(fig)


def test_shared_figure_builder_uses_explicit_time_range_for_summary() -> None:
    df = pd.DataFrame(
        {
            "feature": ["income"] * 4,
            "month": ["segment_a", "segment_a", "segment_b", "Total"],
            "bin_index": [0, 1, 0, 9999],
            "bin_label": ["bin0", "bin1", "bin0", "Total"],
            "bin_type": ["normal", "normal", "normal", "汇总组"],
            "count": [40, 30, 30, 100],
            "observed_count": [40, 30, 30, 100],
            "bad": [8, 6, 3, 17],
            "bad_rate": [0.20, 0.20, 0.10, 0.17],
            "lift": [1.2, 1.0, 0.8, 1.0],
            "psi_bin": [0.01, 0.02, 0.01, 0.03],
            "ks_bin": [0.1, 0.2, 0.1, 0.2],
            "auc_bin": [0.55, 0.55, 0.55, 0.55],
            "iv_bin": [0.03, 0.02, 0.01, 0.06],
            "trend": ["asc"] * 4,
            "total_count": [100] * 4,
        }
    )

    fig = MarsPlotter._build_feature_binning_risk_figure(
        df_detail=df,
        feature="income",
        group_col="month",
        target_name="target",
        time_range=("2024-01-03", "2024-03-31"),
    )

    assert fig is not None
    summary_text = "\n".join(text.get_text() for text in fig.texts)
    assert "[2024-01-03 ~ 2024-03-31]" in summary_text
    assert "[segment_a ~ segment_b]" not in summary_text
    plt.close(fig)


def test_shared_figure_builder_requires_time_range() -> None:
    with pytest.raises(ValueError, match="time_range"):
        MarsPlotter._build_feature_binning_risk_figure(
            df_detail=pd.DataFrame({"feature": []}),
            feature="income",
            group_col="month",
        )


def test_shared_figure_builder_uses_amount_color_semantics() -> None:
    df = pd.DataFrame(
        {
            "feature": ["income"] * 8,
            "month": ["2024-01"] * 4 + ["Total"] * 4,
            "bin_index": [0, 1, -1, 9999, 0, 1, -1, 9999],
            "bin_label": ["bin0", "bin1", "Missing", "Total", "bin0", "bin1", "Missing", "Total"],
            "bin_type": ["正常组", "正常组", "空值组", "汇总组", "正常组", "正常组", "空值组", "汇总组"],
            "count": [40, 30, 10, 80, 40, 30, 10, 80],
            "observed_count": [40, 30, 10, 80, 40, 30, 10, 80],
            "bad": [8, 3, 2, 13, 8, 3, 2, 13],
            "bad_rate": [0.20, 0.10, 0.20, 0.1625, 0.20, 0.10, 0.20, 0.1625],
            "lift": [1.2, 0.8, 1.0, 1.0, 1.2, 0.8, 1.0, 1.0],
            "amt_bad_rate": [0.17, 0.13, 0.09, 0.14, 0.17, 0.13, 0.09, 0.14],
            "lift_amt": [4.4, 3.3, 2.2, 3.5, 4.4, 3.3, 2.2, 3.5],
            "good_amt": [830.0, 640.0, 250.0, 1720.0, 830.0, 640.0, 250.0, 1720.0],
            "bad_amt": [170.0, 130.0, 90.0, 390.0, 170.0, 130.0, 90.0, 390.0],
            "psi_bin": [0.01, 0.02, 0.0, 0.03, 0.01, 0.02, 0.0, 0.03],
            "ks_bin": [0.1, 0.2, 0.0, 0.2, 0.1, 0.2, 0.0, 0.2],
            "auc_bin": [0.55, 0.0, 0.0, 0.55, 0.55, 0.0, 0.0, 0.55],
            "iv_bin": [0.03, 0.02, 0.01, 0.06, 0.03, 0.02, 0.01, 0.06],
            "trend": ["asc"] * 8,
            "total_count": [80] * 8,
        }
    )

    fig = MarsPlotter._build_feature_binning_risk_figure(
        df_detail=df,
        feature="income",
        group_col="month",
        target_name="target",
        show_risk="both",
        time_range=("2024-01-01", "2024-03-31"),
    )

    text_colors = {}
    for ax in fig.axes:
        for text in ax.texts:
            text_colors.setdefault(text.get_text(), set()).add(str(text.get_color()).lower())
    line_colors = {
        str(line.get_color()).lower()
        for ax in fig.axes
        for line in ax.lines
    }

    assert "#6a0dad" in text_colors["4.4"]
    assert "#b57edc" in text_colors["17.0%"]
    assert "#355cde" in text_colors["2.2"]
    assert "#7f8cff" in text_colors["9.0%"]
    assert "#d4a017" in line_colors
    plt.close(fig)


@pytest.mark.parametrize(
    ("bad", "observed", "total", "group", "expected_title"),
    [
        (12, 1_000, 5_000, "2026-01", "2026-01 (12/1,000: 1.2%, n: 5,000)"),
        (60, 5_000, 5_000, "2026-01", "2026-01 (60/5,000: 1.2%, n: 5,000)"),
        (0, 0, 5_000, "2026-01", "2026-01 (0/0: n.a., n: 5,000)"),
        (
            12_000,
            1_000_000,
            5_000_000,
            "2026-01",
            "2026-01 (12,000/1,000,000: 1.2%, n: 5,000,000)",
        ),
        (12, 1_000, 5_000, "Total", "Total (12/1,000: 1.2%, n: 5,000)"),
    ],
)
def test_panel_title_uses_observed_bad_rate_and_total_count(
    bad: int,
    observed: int,
    total: int,
    group: str,
    expected_title: str,
) -> None:
    df = _build_panel_title_frame(
        bad=bad,
        observed=observed,
        total=total,
        group=group,
    )

    fig = MarsPlotter._build_feature_binning_risk_figure(
        df_detail=df,
        feature="income",
        group_col="month",
        target_name="target",
        show_risk="count",
        time_range=("2026-01-01", "2026-01-31"),
    )

    assert fig is not None
    assert [axis.get_title() for axis in fig.axes if axis.get_title()] == [expected_title]
    plt.close(fig)


def test_panel_average_bad_rate_line_uses_observed_count() -> None:
    df = _build_panel_title_frame(bad=12, observed=1_000, total=5_000)

    fig = MarsPlotter._build_feature_binning_risk_figure(
        df_detail=df,
        feature="income",
        group_col="month",
        target_name="target",
        show_risk="count",
        time_range=("2026-01-01", "2026-01-31"),
    )

    assert fig is not None
    average_lines = [
        line
        for axis in fig.axes
        for line in axis.lines
        if line.get_linestyle() == "--"
    ]
    assert len(average_lines) == 1
    assert list(average_lines[0].get_ydata()) == pytest.approx([0.012, 0.012])
    plt.close(fig)


def test_unobserved_panel_has_no_bad_rate_line_and_is_not_label_free() -> None:
    df = _build_panel_title_frame(bad=0, observed=0, total=5_000)

    fig = MarsPlotter._build_feature_binning_risk_figure(
        df_detail=df,
        feature="income",
        group_col="month",
        target_name="target",
        show_risk="count",
        time_range=("2026-01-01", "2026-01-31"),
    )

    assert fig is not None
    assert not any(axis.lines for axis in fig.axes)
    assert "Label-Free Mode" not in "\n".join(text.get_text() for text in fig.texts)
    plt.close(fig)


def test_label_free_panel_keeps_total_title() -> None:
    df = _build_panel_title_frame(bad=0, observed=0, total=5_000)
    df[["observed_count", "bad", "bad_rate"]] = float("nan")

    fig = MarsPlotter._build_feature_binning_risk_figure(
        df_detail=df,
        feature="income",
        group_col="month",
        target_name="target",
        show_risk="count",
        time_range=("2026-01-01", "2026-01-31"),
    )

    assert fig is not None
    assert [axis.get_title() for axis in fig.axes if axis.get_title()] == [
        "2026-01   (Total: 5000)"
    ]
    plt.close(fig)


def test_labeled_panel_requires_observed_count() -> None:
    df = _build_panel_title_frame(bad=12, observed=1_000, total=5_000).drop(
        columns="observed_count"
    )

    with pytest.raises(ValueError, match="requires `observed_count`"):
        MarsPlotter._build_feature_binning_risk_figure(
            df_detail=df,
            feature="income",
            group_col="month",
            target_name="target",
            show_risk="count",
            time_range=("2026-01-01", "2026-01-31"),
        )


def test_legacy_figure_without_added_information_keeps_old_geometry_and_bold_font() -> None:
    detail = _build_panel_title_frame(bad=12, observed=1_000, total=5_000)
    with mpl.rc_context({"font.family": ["DejaVu Serif"], "font.weight": "normal"}):
        figure = MarsPlotter._build_feature_binning_risk_figure(
            detail, "income", time_range=("2026-01-01", "2026-01-31"), show_risk="count",
        )
        assert figure is not None
        try:
            figure.canvas.draw()
            assert figure.get_size_inches() == pytest.approx([MarsPlotter.UNIT_WIDTH, 3.45])
            assert figure.texts[0].get_position() == pytest.approx((0.04, 0.94))
            assert figure.texts[0].get_fontsize() == pytest.approx(7.35)
            assert len(figure.texts[0].get_text().splitlines()) == 2
            assert figure.texts[0].get_bbox_patch() is not None
            for geometry in _axes_geometry_inches(figure):
                assert geometry == pytest.approx((
                    MarsPlotter.UNIT_WIDTH * 0.05,
                    3.45 * 0.15,
                    MarsPlotter.UNIT_WIDTH * 0.9,
                    3.45 * 0.6,
                ))
            lift_text = next(text for axis in figure.axes for text in axis.texts
                             if text.get_text() == "1.0")
            assert lift_text.get_fontsize() == pytest.approx(7.35)
            assert lift_text.get_fontweight() == "bold"
            assert lift_text.get_fontfamily() == ["DejaVu Serif"]
            assert findfont(lift_text.get_fontproperties()) == findfont(
                FontProperties(family=["DejaVu Serif"], weight="bold"),
            )
            assert findfont(lift_text.get_fontproperties()) != findfont(
                FontProperties(family=["DejaVu Serif"], weight="normal"),
            )
        finally:
            plt.close(figure)


@pytest.mark.parametrize("panels", [1, 2, 4])
@pytest.mark.parametrize("show_risk", ["count", "amt", "both"])
def test_added_display_name_and_dates_preserve_old_text_and_plot_geometry(
    panels: int, show_risk: Literal["count", "amt", "both"],
) -> None:
    detail = pd.concat([
        _build_panel_title_frame(bad=12, observed=1_000, total=5_000, group=f"group_{index}")
        for index in range(panels)
    ], ignore_index=True)
    detail["amt_bad_rate"] = np.tile([0.02, 0.04], panels)
    detail["lift_amt"] = np.tile([0.7, 1.3], panels)
    detail["good_amt"] = np.tile([980.0, 960.0], panels)
    detail["bad_amt"] = np.tile([20.0, 40.0], panels)
    with mpl.rc_context({"font.family": ["DejaVu Serif"], "font.weight": "normal"}):
        original_rc = mpl.rcParams.copy()
        old = MarsPlotter._build_feature_binning_risk_figure(
            detail, "income", time_range=("2026-01-01", "2026-01-31"), show_risk=show_risk,
        )
        added = MarsPlotter._build_feature_binning_risk_figure(
            detail, "income", time_range=("2026-01-01", "2026-01-31"), show_risk=show_risk,
            display_name="家庭收入（元）",
            group_time_ranges={f"group_{index}": ("2026-01-01", "2026-01-31")
                               for index in range(panels)},
        )
        assert old is not None and added is not None
        try:
            old.canvas.draw()
            added.canvas.draw()
            assert mpl.rcParams == original_rc
            assert _old_text_styles(added) == _old_text_styles(old)
            np.testing.assert_allclose(_axes_geometry_inches(added), _axes_geometry_inches(old))
            for old_axis, added_axis in zip(old.axes, added.axes):
                assert old_axis.get_ylim() == added_axis.get_ylim()
                assert old_axis.get_xlim() == added_axis.get_xlim()
                assert len(old_axis.lines) == len(added_axis.lines)
                for old_line, added_line in zip(old_axis.lines, added_axis.lines):
                    np.testing.assert_array_equal(old_line.get_xydata(), added_line.get_xydata())
                    assert old_line.get_color() == added_line.get_color()
                    assert old_line.get_linewidth() == added_line.get_linewidth()
                    assert old_line.get_linestyle() == added_line.get_linestyle()
                    assert old_line.get_marker() == added_line.get_marker()
                assert len(old_axis.patches) == len(added_axis.patches)
                for old_bar, added_bar in zip(old_axis.patches, added_axis.patches):
                    assert old_bar.get_bbox().bounds == added_bar.get_bbox().bounds
                    assert old_bar.get_facecolor() == added_bar.get_facecolor()
                assert len(old_axis.collections) == len(added_axis.collections)
                for old_points, added_points in zip(old_axis.collections, added_axis.collections):
                    np.testing.assert_array_equal(
                        old_points.get_offsets(), added_points.get_offsets(),
                    )
                    np.testing.assert_array_equal(old_points.get_sizes(), added_points.get_sizes())
                    np.testing.assert_array_equal(
                        old_points.get_facecolors(), added_points.get_facecolors(),
                    )
        finally:
            plt.close(old)
            plt.close(added)


@pytest.mark.parametrize("display_name", [None, "家庭收入（元）", "家庭收入与日常支出" * 12])
@pytest.mark.parametrize("dpi", [100, 300])
def test_added_rows_use_visible_gaps_without_wasting_header_space(
    display_name: str | None, dpi: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    detail = _build_panel_title_frame(bad=12, observed=1_000, total=5_000)
    figure = MarsPlotter._build_feature_binning_risk_figure(
        detail, "income", time_range=("2026-01-01", "2026-01-31"), show_risk="count",
        display_name=display_name, group_time_ranges={"2026-01": ("2026-01-01", "2026-01-31")},
    )
    legacy = MarsPlotter._build_feature_binning_risk_figure(
        detail, "income", time_range=("2026-01-01", "2026-01-31"), show_risk="count",
    )
    assert figure is not None and legacy is not None
    try:
        figure.set_dpi(dpi)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        _check_risk_trend_header_layout(figure, renderer, legacy)
        if display_name:
            summary_patch = next(patch for patch in figure.artists
                                 if patch.get_gid() == "mars-summary-box")
        else:
            summary_patch = figure.texts[0].get_bbox_patch()
        assert summary_patch is not None
        output = tmp_path / f"layout-{dpi}.png"
        export_draws: list[tuple[float, ...]] = []
        original_draw = FancyBboxPatch.draw

        def check_exported_header(patch: FancyBboxPatch, export_renderer: Any) -> None:
            """检查 tight 保存期间的真实变换，防止背景框未随裁剪平移。"""
            original_draw(patch, export_renderer)
            if patch is not summary_patch:
                return
            _check_risk_trend_header_layout(figure, export_renderer, legacy)
            export_draws.append(figure.bbox.bounds)

        monkeypatch.setattr(FancyBboxPatch, "draw", check_exported_header)
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        assert len(set(export_draws)) >= 2  # 包括裁剪前测量与裁剪后最终绘制。
        assert output.stat().st_size > 1_000
    finally:
        plt.close(figure)
        plt.close(legacy)


@pytest.mark.parametrize("display_name", [None, "收入"])
@pytest.mark.parametrize(
    ("feature", "target"),
    [
        ("income", "target"),
        (r"cost_$\unknown$", "target"),
        (r"cost_$\frac{$", "target"),
        (r"cost_$\frac{", "target"),
        ("income", r"target_$\unknown$"),
    ],
)
def test_public_risk_trend_treats_feature_and_target_names_as_literals(
    feature: str, target: str, display_name: str | None, tmp_path: Path,
) -> None:
    from mars.analysis import profile_risk

    frame = pl.DataFrame({
        feature: [0.1, 0.2, 0.4, 0.5, 0.7, 0.9],
        target: [0, 1, None, 1, 0, None],
        "segment": ["Train"] * 3 + ["Current"] * 3,
        "date": ["2026-01-01", "2026-01-02", "2026-01-09",
                 "2026-02-01", "2026-02-02", "2026-02-09"],
    })
    report = profile_risk(
        frame, target=target, features=[feature], group_col="segment", time_col="date",
        binning_type="native", n_bins=2,
        feature_metadata={feature: {"display_name": display_name}},
    ).report
    before = report.detail_table.clone()
    before_meta = deepcopy(report.report_meta)
    before_feature_meta = deepcopy(report.feature_metadata)
    figures = report.build_risk_trend_figures(feature, target=target, dpi=300)
    assert len(figures) == 1
    figure = figures[0]
    try:
        figure.canvas.draw()
        assert figure.texts[0].get_text().startswith(f"{feature},  {target},")
        assert [text.get_text() for text in figure.texts
                if text.get_gid() == "mars-display-name"] == ([display_name] if display_name else [])
        assert set(before["feature"].to_list()) == {feature}
        for image_format in ("png", "svg"):
            output = tmp_path / f"literal.{image_format}"
            figure.savefig(output, format=image_format, dpi=300, bbox_inches="tight")
            assert output.stat().st_size > 1_000
        assert report.detail_table.equals(before)
        assert report.report_meta == before_meta
        assert report.feature_metadata == before_feature_meta
    finally:
        plt.close(figure)


def test_public_risk_trend_builds_and_saves_with_direct_svg_backend(tmp_path: Path) -> None:
    # 后端必须在导入 pyplot 前由独立进程选择，避免默认 Agg 的 SVG 导出掩盖构图问题。
    script = textwrap.dedent(r"""
        import sys
        from pathlib import Path
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import polars as pl
        from mars.analysis import profile_risk

        assert mpl.get_backend().lower() == "svg"
        frame = pl.DataFrame({
            "income": [0.1, 0.2, 0.4, 0.5, 0.7, 0.9],
            "target": [0, 1, None, 1, 0, None],
            "segment": ["Train"] * 3 + ["Current"] * 3,
            "date": ["2026-01-01", "2026-01-02", "2026-01-09",
                     "2026-02-01", "2026-02-02", "2026-02-09"],
        })
        before_font = list(mpl.rcParams["font.family"])
        report = profile_risk(
            frame, target="target", features=["income"], group_col="segment", time_col="date",
            binning_type="native", n_bins=2,
            feature_metadata={"income": {"display_name": "家庭收入（元）"}},
        ).report
        figure = report.build_risk_trend_figures("income", show_risk="count", dpi=300)[0]
        canvas, manager = figure.canvas, figure.canvas.manager
        figure.canvas.draw()
        assert figure.texts[0].get_text().startswith("income,  target,  Total: 6,")
        assert [text.get_text() for text in figure.texts
                if text.get_gid() == "mars-display-name"] == ["家庭收入（元）"]
        assert {axis.get_title(): [text.get_text() for text in axis.texts
                                  if text.get_gid() == "mars-group-date"]
                for axis in figure.axes if axis.get_title()} == {
            "Current (1/2: 50.0%, n: 3)": ["[2026-02-01 ~ 2026-02-09]"],
            "Train (1/2: 50.0%, n: 3)": ["[2026-01-01 ~ 2026-01-09]"],
            "Total (2/4: 50.0%, n: 6)": ["[2026-01-01 ~ 2026-02-09]"],
        }
        assert any(text.get_text().startswith("IV:")
                   for axis in figure.axes for text in axis.texts)
        figure.savefig(Path(sys.argv[1]) / "direct.svg", bbox_inches="tight")
        assert figure.canvas is canvas and figure.canvas.manager is manager
        assert mpl.get_backend().lower() == "svg"
        assert list(mpl.rcParams["font.family"]) == before_font
        plt.close(figure)
    """)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "svg"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], env=environment,
        capture_output=True, text=True, timeout=90, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    root = ElementTree.parse(tmp_path / "direct.svg").getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    # 字体路径编码时用 gid 对应的实际向量 glyph 验证中文和日期均有绘制内容。
    for gid in ("mars-display-name", "mars-group-date"):
        groups = root.findall(f".//svg:g[@id='{gid}']", namespace)
        assert groups
        assert all(group.findall(".//svg:use", namespace) for group in groups)


@pytest.mark.parametrize("force_fallback", [False, True])
def test_risk_trend_dpi_and_repeated_tight_exports_do_not_accumulate_layout(
    force_fallback: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if force_fallback:
        monkeypatch.setattr("mars.reporting.plotter._resolve_cjk_font_family", lambda: None)
    detail = _build_panel_title_frame(bad=12, observed=1_000, total=5_000)
    figure = MarsPlotter._build_feature_binning_risk_figure(
        detail, "income", time_range=("2026-01-01", "2026-01-31"), show_risk="count",
        display_name="家庭收入与日常支出" * 12,
        group_time_ranges={"2026-01": ("2026-01-01", "2026-01-31")},
    )
    legacy = MarsPlotter._build_feature_binning_risk_figure(
        detail, "income", time_range=("2026-01-01", "2026-01-31"), show_risk="count",
    )
    assert figure is not None and legacy is not None
    original_canvas = figure.canvas
    original_manager = figure.canvas.manager
    original_layout = figure.get_layout_engine()
    snapshots: dict[int, tuple[tuple[float, ...], list[tuple[float, ...]]]] = {}
    counts = (len(figure.texts), len(figure.artists),
              tuple(len(axis.texts) for axis in figure.axes))
    try:
        for dpi in (100, 300, 100, 300):
            figure.set_dpi(dpi)
            for _ in range(2):
                figure.canvas.draw()
                _check_risk_trend_header_layout(figure, figure.canvas.get_renderer(), legacy)
            snapshot = (tuple(figure.get_size_inches()), _axes_geometry_inches(figure))
            if dpi in snapshots:
                np.testing.assert_allclose(snapshot[0], snapshots[dpi][0], atol=1e-10)
                np.testing.assert_allclose(snapshot[1], snapshots[dpi][1], atol=1e-10)
            snapshots[dpi] = snapshot

        summary_patch = next(patch for patch in figure.artists
                             if patch.get_gid() == "mars-summary-box")
        original_draw = FancyBboxPatch.draw
        export_draws: list[tuple[str, tuple[float, ...]]] = []

        def check_exported_header(patch: FancyBboxPatch, renderer: Any) -> None:
            """在每种后端的最终 tight 绘制阶段检查实际文字净距。"""
            original_draw(patch, renderer)
            if patch is summary_patch:
                _check_risk_trend_header_layout(figure, renderer, legacy)
                export_draws.append((type(renderer).__name__, figure.bbox.bounds))

        monkeypatch.setattr(FancyBboxPatch, "draw", check_exported_header)
        export_sizes: dict[str, tuple[float, ...]] = {}
        for index, image_format in enumerate(("png", "svg", "png", "svg", "png")):
            before_draw_count = len(export_draws)
            output = tmp_path / f"roundtrip-{index}.{image_format}"
            figure.savefig(output, dpi=300, format=image_format, bbox_inches="tight")
            assert output.stat().st_size > 1_000
            current_draws = export_draws[before_draw_count:]
            assert len({bounds for _, bounds in current_draws}) >= 2
            assert figure.canvas is original_canvas and figure.canvas.manager is original_manager
            assert figure.get_layout_engine() is original_layout
            assert figure.dpi == 300
            # 无中途 draw 连续导出；各后端可有真实字形度量差，但同格式高度必须稳定。
            export_size = tuple(figure.get_size_inches())
            if image_format in export_sizes:
                np.testing.assert_allclose(export_size, export_sizes[image_format], atol=1e-10)
            export_sizes[image_format] = export_size
            np.testing.assert_allclose(_axes_geometry_inches(figure), snapshots[300][1], atol=1e-10)
            assert (len(figure.texts), len(figure.artists),
                    tuple(len(axis.texts) for axis in figure.axes)) == counts
        assert any(renderer_name != "RendererAgg" for renderer_name, _ in export_draws)
        figure.canvas.draw()
        np.testing.assert_allclose(figure.get_size_inches(), snapshots[300][0], atol=1e-10)
    finally:
        plt.close(figure)
        plt.close(legacy)


@pytest.mark.parametrize("ks_method", ["binned", "raw"])
def test_group_dates_use_current_all_rows_and_survive_multi_target_raw_and_snapshot(
    ks_method: Literal["binned", "raw"], tmp_path: Path,
) -> None:
    from mars.reporting import load_report

    report = _build_group_date_report(ks_method=ks_method)
    context = report.report_meta["group_context"]
    assert context["input_group_col"] == "segment"
    assert context["internal_group_col"] == "mars_group"
    dates = {group["internal_value"]: group for group in context["groups"]}
    assert dates["Train"]["end_dt"] == "2026-01-20"  # 此行标签未表现。
    assert dates["Total loans"]["end_dt"] == "2026-02-28"  # 此行特征缺失。
    assert dates["Undue"]["start_dt"] == "2026-03-01"  # 整组标签均未表现。
    assert dates["No dates"]["start_dt"] is None
    assert context["total_range"] == {"start_dt": "2026-01-01", "end_dt": "2026-03-09"}
    assert report.report_meta["start_dt"] == "2026-01-01"
    archive = tmp_path / "groups.marsreport"
    report.save(archive)
    restored = load_report(archive)
    assert restored.describe()["parameters"]["group_context"] == context


@pytest.mark.parametrize("show_risk", ["count", "amt", "both"])
@pytest.mark.parametrize("baseline", ["total", "first_group", "benchmark"])
def test_include_total_preserves_global_summary_rc_and_other_panels(
    show_risk: Literal["count", "amt", "both"], baseline: RiskCorrBaseline,
) -> None:
    import matplotlib as mpl

    report = _build_group_date_report(risk_corr_baseline=baseline)
    before = report.detail_table.clone()
    original_font = list(mpl.rcParams["font.family"])
    with_total = report.build_risk_trend_figures(
        "income", target="target", show_risk=show_risk, risk_corr_baseline=baseline,
    )[0]
    without_total = report.build_risk_trend_figures(
        "income", target="target", show_risk=show_risk, risk_corr_baseline=baseline,
        include_total=False,
    )[0]
    try:
        summary = with_total.texts[0].get_text()
        assert summary.startswith("income")
        assert [text.get_text() for text in with_total.texts
                if text.get_gid() == "mars-display-name"] == ["家庭收入（元）"]
        assert without_total.texts[0].get_text() == summary
        full_axes = {axis.get_title().split(" (")[0]: axis
                     for axis in with_total.axes if axis.get_title()}
        reduced_axes = {axis.get_title().split(" (")[0]: axis
                        for axis in without_total.axes if axis.get_title()}
        assert "Total" in full_axes and "Total" not in reduced_axes
        assert "Total loans" in reduced_axes
        assert len(reduced_axes) == len(full_axes) - 1
        for key, axis in reduced_axes.items():
            assert axis.get_title() == full_axes[key].get_title()
            assert axis.get_ylim() == full_axes[key].get_ylim()
            assert [text.get_text() for text in axis.texts] == [
                text.get_text() for text in full_axes[key].texts
            ]
        assert _group_date_text(full_axes["Train"]).get_text() == "[2026-01-01 ~ 2026-01-20]"
        assert _group_date_text(full_axes["Undue"]).get_text() == "[2026-03-01 ~ 2026-03-09]"
        assert _group_date_text(full_axes["No dates"]).get_text() == "[Date range: unknown]"
        assert report.detail_table.equals(before)
        assert list(mpl.rcParams["font.family"]) == original_font
    finally:
        plt.close(with_total)
        plt.close(without_total)


def test_total_only_can_not_hide_last_panel() -> None:
    detail = _build_panel_title_frame(bad=12, observed=1_000, total=5_000, group="Total")
    with pytest.raises(ValueError, match="leaves no"):
        MarsPlotter._build_feature_binning_risk_figure(
            detail, "income", time_range=("2026-01-01", "2026-01-31"),
            include_total=False,
        )


def test_implicit_time_groups_and_legacy_reports_do_not_invent_group_dates() -> None:
    report = _build_group_date_report(explicit_group=False, risk_corr_baseline="first_group")
    assert "group_context" not in report.report_meta
    figures = report.build_risk_trend_figures("income", target="target", include_total=False)
    try:
        assert not any(text.get_gid() == "mars-group-date"
                       for axis in figures[0].axes for text in axis.texts)
    finally:
        plt.close(figures[0])
    report = _build_group_date_report()
    report._report_meta.pop("group_context")
    report.feature_metadata["income"]["display_name"] = None
    figure = report.build_risk_trend_figures("income", target="target")[0]
    try:
        assert figure.texts[0].get_text().startswith("income")
        assert not any(text.get_gid() == "mars-group-date"
                       for axis in figure.axes for text in axis.texts)
    finally:
        plt.close(figure)


def test_plot_export_entries_share_total_choice_and_long_display_name(tmp_path: Path) -> None:
    report = _build_group_date_report()
    report.feature_metadata["income"]["display_name"] = "家庭收入与日常支出" * 12
    fragment = report.render_risk_trends_html(
        "income", target="target", image_format="svg", include_total=False,
        return_figures=True,
    )
    figure = fragment.figures[0]
    try:
        display_text = next(text for text in figure.texts
                            if text.get_gid() == "mars-display-name")
        assert len(display_text.get_text().splitlines()) > 1
        assert len(figure.texts[0].get_text().splitlines()) == 2
        assert all(not axis.get_title().startswith("Total (")
                   for axis in figure.axes)
        assert "<svg" in fragment.html
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        summary_patch = next(patch for patch in figure.artists
                             if patch.get_gid() == "mars-summary-box")
        date_boxes = [_group_date_text(axis).get_window_extent(renderer)
                      for axis in figure.axes if axis.get_title()]
        assert summary_patch.get_window_extent(renderer).y0 > max(box.y1 for box in date_boxes)
    finally:
        plt.close(figure)
    for image_format in ("png", "svg"):
        paths = report.save_risk_trend_images(
            tmp_path / image_format, "income", target="target", image_format=image_format,
            include_total=False, dpi=300,
        )
        assert len(paths) == 1 and paths[0].stat().st_size > 1_000


def test_write_html_charts_use_same_context_and_total_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mars.reporting._binning_plot import _BinningPlotRenderer, _RiskTrendPlotContext

    report = _build_group_date_report()
    captured: list[dict[str, object]] = []
    original = _BinningPlotRenderer._build_context_figure

    def capture_figure(
        self: _BinningPlotRenderer,
        context: _RiskTrendPlotContext,
        feature: str,
        *,
        show_risk: Literal["count", "amt", "both"],
        include_total: bool,
    ) -> Figure | None:
        figure = original(self, context, feature, show_risk=show_risk, include_total=include_total)
        assert figure is not None
        captured.append({
            "include_total": include_total,
            "summary": figure.texts[0].get_text(),
            "display_names": [text.get_text() for text in figure.texts
                              if text.get_gid() == "mars-display-name"],
            "dates": [_group_date_text(axis).get_text() for axis in figure.axes
                      if axis.get_title()],
            "titles": [axis.get_title() for axis in figure.axes if axis.get_title()],
        })
        return figure

    monkeypatch.setattr(_BinningPlotRenderer, "_build_context_figure", capture_figure)
    report.write_html(str(tmp_path / "report.html"), max_plots=1, include_total=False)
    assert len(captured) == 2  # 每个目标各一张图，不重复按特征解析全量数据。
    for data in captured:
        assert data["include_total"] is False
        assert data["summary"].startswith("income")
        assert data["display_names"] == ["家庭收入（元）"]
        assert "[2026-01-01 ~ 2026-01-20]" in data["dates"]
        assert all(not title.startswith("Total (") for title in data["titles"])
