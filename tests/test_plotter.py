import matplotlib.pyplot as plt
import pandas as pd

from mars.reporting.plotter import MarsPlotter


def test_plot_feature_binning_risk_trend_uses_shared_figure_builder(monkeypatch):
    fig = plt.figure()
    captured = {}

    def fake_build_feature_binning_risk_figure(*, df_detail, feature, group_col, target_name):
        captured["feature"] = feature
        captured["group_col"] = group_col
        captured["target_name"] = target_name
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
    )

    assert captured["feature"] == "income"
    assert captured["group_col"] == "month"
    assert captured["target_name"] == "target"
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
    )

    labels = []
    for ax in fig.axes:
        labels.extend([tick.get_text() for tick in ax.get_xticklabels()])

    assert "Total" not in labels
    plt.close(fig)
