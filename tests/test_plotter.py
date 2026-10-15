from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import pytest
from matplotlib.figure import Figure

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
        assert summary.startswith("家庭收入（元）\nincome")
        assert without_total.texts[0].get_text() == summary
        full_axes = {axis.get_title().split("\n")[-1].split(" (")[0]: axis
                     for axis in with_total.axes if axis.get_title()}
        reduced_axes = {axis.get_title().split("\n")[-1].split(" (")[0]: axis
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
        assert full_axes["Train"].get_title().startswith("[2026-01-01 ~ 2026-01-20]\n")
        assert full_axes["Undue"].get_title().startswith("[2026-03-01 ~ 2026-03-09]\n")
        assert full_axes["No dates"].get_title().startswith("[Date range: unknown]\n")
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
        assert all(" ~ " not in axis.get_title() for axis in figures[0].axes)
    finally:
        plt.close(figures[0])
    report = _build_group_date_report()
    report._report_meta.pop("group_context")
    report.feature_metadata["income"]["display_name"] = None
    figure = report.build_risk_trend_figures("income", target="target")[0]
    try:
        assert figure.texts[0].get_text().startswith("income")
        assert all(" ~ " not in axis.get_title() for axis in figure.axes)
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
        summary = figure.texts[0].get_text()
        assert len(summary.splitlines()) > 3
        assert all(not axis.get_title().split("\n")[-1].startswith("Total (")
                   for axis in figure.axes)
        assert "<svg" in fragment.html
        renderer = figure.canvas.get_renderer()
        figure.canvas.draw()
        summary_box = figure.texts[0].get_window_extent(renderer)
        title_boxes = [axis.title.get_window_extent(renderer) for axis in figure.axes
                       if axis.get_title()]
        assert summary_box.y0 > max(box.y1 for box in title_boxes)
    finally:
        plt.close(figure)
    for image_format in ("png", "svg"):
        paths = report.save_risk_trend_images(
            tmp_path / image_format, "income", target="target", image_format=image_format,
            include_total=False,
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
            "titles": [axis.get_title() for axis in figure.axes if axis.get_title()],
        })
        return figure

    monkeypatch.setattr(_BinningPlotRenderer, "_build_context_figure", capture_figure)
    report.write_html(str(tmp_path / "report.html"), max_plots=1, include_total=False)
    assert len(captured) == 2  # 每个目标各一张图，不重复按特征解析全量数据。
    for data in captured:
        assert data["include_total"] is False
        assert data["summary"].startswith("家庭收入（元）\nincome")
        assert any(title.startswith("[2026-01-01 ~ 2026-01-20]\n") for title in data["titles"])
        assert all(not title.split("\n")[-1].startswith("Total (") for title in data["titles"])
