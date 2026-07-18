import polars as pl

from mars.feature import MarsStatsSelector

df = pl.DataFrame(
    {
        "income": [2600, 3100, 3500, 3900, 4500, 5200, 6100, 7200] * 3,
        "utilization": [0.72, 0.64, 0.55, 0.48, 0.36, 0.28, 0.19, 0.11] * 3,
        "constant": [1.0] * 24,
        "target": [1, 1, 1, 0, 1, 0, 0, 0] * 3,
    }
)

selector = MarsStatsSelector(
    missing_thr=0.95,
    iv_thr=0.0,
    lift_thr=None,
    psi_thr=None,
    rc_thr=None,
    corr_thr=None,
    skip_fine_scan=True,
    n_jobs=1,
)
selector.fit(
    df,
    target="target",
    features=["income", "utilization", "constant"],
    white_list=["income"],
)

selected_features = selector.selected_features_
selection_report = selector.get_report()
