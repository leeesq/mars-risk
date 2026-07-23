import matplotlib.pyplot as plt
import pandas as pd
import pytest

from mars.reporting.plotter import MarsPlotter


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
    bad_rates = [
        bad_count / observed_count if observed_count > 0 else float("nan")
        for bad_count, observed_count in zip(bad_counts, observed_counts, strict=True)
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


def test_plot_feature_binning_risk_trend_uses_shared_figure_builder(monkeypatch):
    fig = plt.figure()
    captured = {}

    def fake_build_feature_binning_risk_figure(
        *,
        df_detail,
        feature,
        group_col,
        target_name,
        risk_corr_reference_df=None,
        show_risk="both",
        time_range=None,
    ):
        captured["feature"] = feature
        captured["group_col"] = group_col
        captured["target_name"] = target_name
        captured["risk_corr_reference_df"] = risk_corr_reference_df
        captured["show_risk"] = show_risk
        captured["time_range"] = time_range
        return fig

    def fake_show_scrollable(passed_fig, dpi=150):
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
    )

    assert captured["feature"] == "income"
    assert captured["group_col"] == "month"
    assert captured["target_name"] == "target"
    assert captured["risk_corr_reference_df"] is None
    assert captured["show_risk"] == "both"
    assert captured["time_range"] == ("2024-01-01", "2024-03-31")
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
