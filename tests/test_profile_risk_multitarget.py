"""多目标入口共享特征集和参考分箱的回归。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.analysis import profile_risk
from mars.feature import MarsNativeBinner


def _sample() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": list(range(8)),
            "bad30": [0, 0, 0, 0, 1, 1, 1, 1],
            "bad60": [0, None, 1, 0, 1, 1, None, 0],
            "bad90": [None, 0, 0, 1, 0, 1, 1, None],
            "segment": ["a"] * 4 + ["b"] * 4,
            "day": ["2024-01-01"] * 4 + ["2024-02-01"] * 4,
            "weight": [1.0, 2.0] * 4,
            "amount": [10.0, 20.0] * 4,
        }
    )


@pytest.mark.parametrize("pandas", [False, True])
@pytest.mark.parametrize("target_count", [2, 3])
@pytest.mark.parametrize("ks_method", ["binned", "raw"])
@pytest.mark.parametrize("benchmark", [False, True])
def test_default_features_match_explicit_shared_features(
    pandas: bool, target_count: int, ks_method: str, benchmark: bool
) -> None:
    targets = ["bad30", "bad60", "bad90"][:target_count]
    frame = _sample()
    if target_count == 2:
        frame = frame.drop("bad90")
    data = frame.to_pandas() if pandas else frame
    options: dict[str, Any] = {
        "target": targets,
        "group_col": "segment",
        "time_col": "day",
        "weights_col": "weight",
        "amount_col": "amount",
        "method": "cart",
        "n_bins": 2,
        "ks_method": ks_method,
        "max_raw_ks_features": 1,
        "n_jobs": 1,
    }
    if benchmark:
        options["benchmark_df"] = data
    actual = profile_risk(data, **options)
    expected = profile_risk(data, features=["x"], **options)
    assert actual.metadata["features"] == ["x"]
    assert actual.targets == targets
    assert actual.metadata["binning_fit_source"] == ("benchmark_df" if benchmark else "df")
    assert actual.binner.bin_cuts_ == expected.binner.bin_cuts_
    for name in expected.report._query_tables():
        left = actual.report.get_table(name)
        right = expected.report.get_table(name)
        if isinstance(left, pd.DataFrame):
            left, right = pl.from_pandas(left), pl.from_pandas(right)
        assert_frame_equal(left, right)
    # 每个标签独立排除未观测值，且趋势继续仅描述首目标。
    detail = actual.report.get_table("detail")
    if isinstance(detail, pd.DataFrame):
        detail = pl.from_pandas(detail)
    assert set(detail["y"].to_list()) == set(targets)
    totals = detail.filter((pl.col("mars_group") == "Total") & (pl.col("bin_label") == "Total"))
    for target in targets:
        assert totals.filter(pl.col("y") == target)["observed_count"].sum() == (
            frame.filter(pl.col(target).is_not_null())["weight"].sum()
        )
    primary = profile_risk(data, target="bad30", features=["x"], **{
        key: value for key, value in options.items() if key != "target"
    })
    for name, table in actual.report.trend_tables.items():
        other = primary.report.trend_tables[name]
        if isinstance(table, pd.DataFrame):
            table, other = pl.from_pandas(table), pl.from_pandas(other)
        assert_frame_equal(table, other)


@pytest.mark.parametrize("pandas", [False, True])
def test_reference_target_is_fitted_once(
    pandas: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _sample()
    data = frame.to_pandas() if pandas else frame
    calls: list[tuple[list[str], list[Any]]] = []
    original = MarsNativeBinner.fit

    def record_fit(self: MarsNativeBinner, X: Any, y: Any = None, **kwargs: Any) -> Any:
        calls.append((list(kwargs["features"]), y.to_list()))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(MarsNativeBinner, "fit", record_fit)
    result = profile_risk(
        data, target=["bad30", "bad60", "bad90"], method="cart", n_bins=2,
        group_col="segment", time_col="day", weights_col="weight", amount_col="amount",
        ks_method="raw", max_raw_ks_features=1, n_jobs=1,
    )
    assert calls == [(["x"], frame["bad30"].to_list())]
    assert result.metadata["target_requested"] == "bad30"
    assert set(result.binner.bin_cuts_) == {"x"}


@pytest.mark.parametrize("pandas", [False, True])
@pytest.mark.parametrize("targets", [None, [], "bad30"])
def test_default_roles_are_excluded_in_single_and_label_free_modes(
    pandas: bool, targets: Any
) -> None:
    frame = _sample().drop("bad60", "bad90")
    if targets is None or targets == []:
        frame = frame.drop("bad30")
    data = frame.to_pandas() if pandas else frame
    result = profile_risk(
        data, target=targets, group_col="segment", time_col="day",
        weights_col="weight", amount_col="amount", n_bins=2, n_jobs=1,
    )
    assert result.metadata["features"] == ["x"]


@pytest.mark.parametrize("role", ["group_col", "time_col", "weights_col", "amount_col"])
def test_target_cannot_also_be_a_declared_role(role: str) -> None:
    with pytest.raises(ValueError, match="Target.*roles"):
        profile_risk(_sample(), target=["bad30", "bad60"], **{role: "bad60"})


@pytest.mark.parametrize("pandas", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_benchmark_reference_and_unobserved_secondary_target(
    pandas: bool, grouped: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _sample().drop("bad90", "segment", "weight", "amount")
    benchmark = frame.select("x", "bad30").with_columns(1 - pl.col("bad30"))
    frame = frame.with_columns(pl.lit(None).cast(pl.Int8).alias("bad60"))
    data = frame.to_pandas() if pandas else frame
    baseline = benchmark.to_pandas() if pandas else benchmark
    calls: list[list[Any]] = []
    original = MarsNativeBinner.fit

    def record_fit(self: MarsNativeBinner, X: Any, y: Any = None, **kwargs: Any) -> Any:
        calls.append(y.to_list())
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(MarsNativeBinner, "fit", record_fit)
    result = profile_risk(
        data, target=["bad30", "bad60"], benchmark_df=baseline,
        time_col="day", time_grain="month" if grouped else None,
        method="cart", n_bins=2, n_jobs=1,
    )
    assert calls == [benchmark["bad30"].to_list()]
    assert result.metadata["features"] == ["x"]
    summary = result.report.get_table("summary")
    if isinstance(summary, pd.DataFrame):
        summary = pl.from_pandas(summary)
    assert summary.filter(pl.col("target") == "bad60")["ks"].null_count() == 1


def test_no_inferred_features_fails_before_fitting() -> None:
    with pytest.raises(ValueError, match="No feature columns remain"):
        profile_risk(pl.DataFrame({"bad30": [0, 1], "bad60": [0, 1]}), target=["bad30", "bad60"])


def test_explicit_feature_selection_retains_existing_priority() -> None:
    result = profile_risk(
        _sample(), target=["bad30", "bad60"], features=["x"], n_bins=2, n_jobs=1,
    )
    assert result.metadata["features"] == ["x"]
