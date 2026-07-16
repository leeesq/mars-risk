import matplotlib.pyplot as plt
import pandas as pd
import pytest

from mars.reporting.plotter import MarsPlotter


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
