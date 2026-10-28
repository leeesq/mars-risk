"""客群风险画像的独立数值、固定规则和参考人口回归。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.analysis import MarsBinEvaluator, MarsRiskProfile, profile_risk
from mars.feature._scopes import resolve_scopes, scope_filter
from mars.feature.binning.base import MarsBinnerBase


def population() -> pl.DataFrame:
    return pl.DataFrame({
        "s": ["seg1", "seg2", "seg3"] * 24,
        "g": ["01"] * 36 + ["02"] * 36,
        "dt": ["2026-01-01"] * 36 + ["2026-02-01"] * 36,
        "x": list(range(72)), "c": ["a", "b", "c", "d"] * 18,
        "y": [0, 1, 1, 0] * 18, "w": [1.0, 2.0, 0.0, 1.0] * 18,
        "amt": [10.0, 20.0, 30.0, 40.0] * 18,
    })


@pytest.mark.parametrize("mode", ["shared", "per_segment"])
@pytest.mark.parametrize("batch_size", [1, 16, 100])
def test_scope_counts_and_metrics_match_fixed_public_reference(mode: str, batch_size: int) -> None:
    frame = population()
    benchmark = frame.filter(pl.col("g") == "01")
    run = profile_risk(
        frame, target="y", features=["x", "c"], segment_col="s", group_col="g",
        benchmark_df=benchmark, binning_scope=mode, n_bins=3, n_jobs=1,
        weights_col="w", amount_col="amt", batch_size=batch_size,
    )
    assert run.scopes is not None
    for identity in run.scopes["scope_id"].to_list():
        current = scope_filter(frame, run.scopes, identity)
        expected = MarsBinEvaluator().evaluate(
            current, target="y", features=["x", "c"], group_col="g",
            binner=MarsBinnerBase.from_dict(run.binners_by_scope[identity].to_dict()),
            benchmark_df=scope_filter(benchmark, run.scopes, identity),
            weights_col="w", amount_col="amt", batch_size=batch_size,
        ).report
        actual = run.report.detail_table.filter(pl.col("scope_id") == identity).drop("scope_id")
        assert_frame_equal(actual, expected.detail_table, check_row_order=False)
        totals = actual.filter((pl.col("mars_group") == "Total") & (pl.col("feature") == "x") & (pl.col("bin_index") != 9999))
        assert totals["count"].sum() == current["w"].sum()
        assert totals["bad"].sum() == current.select((pl.col("y") * pl.col("w")).sum()).item()
        assert run.binners_by_scope[identity]._cache_X is None
    if mode == "shared":
        assert len({id(b) for b in run.binners_by_scope.values()}) == 1


def test_missing_benchmark_segment_is_local_uncomputed_psi() -> None:
    frame = population()
    reference = frame.filter((pl.col("g") == "01") & (pl.col("s") != "seg2"))
    run = profile_risk(frame, target="y", features=["x", "c"], segment_col="s",
                       group_col="g", benchmark_df=reference, n_jobs=1)
    assert run.scopes is not None
    identity = resolve_scopes(run.scopes, ["seg2"], False)[0]
    summary = run.report.summary_table.filter(pl.col("scope_id") == identity)
    assert summary["psi_max"].null_count() == 2
    assert summary["ks"].null_count() == 0
    status = run.report.get_table("calculation_status").filter(
        (pl.col("scope_id") == identity) & (pl.col("metric") == "psi")
    )
    assert set(status["reason"]) == {"missing_reference_population"}
    trend = run.report.trend_tables["psi"]
    assert trend.schema["data_source"] == pl.String


def test_no_benchmark_never_advances_to_segments_own_first_group() -> None:
    frame = population().filter(~((pl.col("s") == "seg2") & (pl.col("g") == "01")))
    run = profile_risk(frame, target="y", features=["x"], segment_col="s", group_col="g",
                       risk_corr_baseline="first_group", n_jobs=1)
    assert run.scopes is not None
    identity = resolve_scopes(run.scopes, ["seg2"], False)[0]
    summary = run.report.summary_table.filter(pl.col("scope_id") == identity)
    assert summary["psi_max"].item() is None
    assert summary["rc_min"].item() is None
    assert run.report.report_meta["scope_metadata"][identity]["reference_group"] == "01"


def test_single_class_scope_keeps_bad_rate_and_marks_separation_undefined() -> None:
    frame = population().with_columns(
        pl.when(pl.col("s") == "seg2").then(0).otherwise(pl.col("y")).alias("y")
    )
    run = profile_risk(frame, target="y", features=["x"], segment_col="s", group_col="g", n_jobs=1)
    assert run.scopes is not None
    identity = resolve_scopes(run.scopes, ["seg2"], False)[0]
    detail = run.report.detail_table.filter(pl.col("scope_id") == identity)
    assert set(detail["bad_rate"].drop_nulls()) == {0.0}
    statuses = run.report.get_table("calculation_status").filter(pl.col("scope_id") == identity)
    assert "single_observed_class" in statuses["reason"].to_list()


def test_supervised_fit_failure_is_local_and_repeated_run_clears_rules() -> None:
    frame = population().with_columns(
        pl.when(pl.col("s") == "seg2").then(0).otherwise(pl.col("y")).alias("y")
    )
    evaluator = MarsBinEvaluator(binning_scope="per_segment", binner_params={"method": "cart", "n_jobs": 1})
    run = evaluator.evaluate(frame, target="y", features=["x"], segment_col="s", group_col="g")
    assert run.scopes is not None
    identity = resolve_scopes(run.scopes, ["seg2"], False)[0]
    assert identity not in run.binners_by_scope
    assert run.scopes.filter(pl.col("scope_id") == identity)["reason"].item() == "single_class_fit"
    next_run = evaluator.evaluate(population().filter(pl.col("s") != "seg2"), target="y",
                                  features=["x"], segment_col="s", group_col="g")
    assert identity not in next_run.binners_by_scope


def test_multitarget_reuses_primary_scope_failure_without_refitting_secondary() -> None:
    frame = population().with_columns(
        pl.col("y").alias("secondary"),
        pl.when(pl.col("s") == "seg2").then(0).otherwise(pl.col("y")).alias("y"),
    )
    run = profile_risk(frame, target=["y", "secondary"], features=["x"], segment_col="s",
                       group_col="g", binning_scope="per_segment", method="cart", n_jobs=1)
    assert run.scopes is not None
    identity = resolve_scopes(run.scopes, ["seg2"], False)[0]
    assert identity not in run.binners_by_scope
    failed = run.report.get_table("calculation_status").filter(
        (pl.col("scope_id") == identity) & (pl.col("metric") == "binning")
    )
    assert set(failed["target"].to_list()) == {"y", "secondary"}
    assert set(failed["reason"].to_list()) == {"single_class_fit"}


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_projection_multichunk_and_scope_identity_preserve_population(backend: str) -> None:
    frame = population().with_columns(pl.Series("s", ["All", "Total", "Missing", "null", "", "NaN", None, "seg2"] * 9))
    frame = pl.concat([frame.slice(0, 10), frame.slice(10, 19), frame.slice(29)], rechunk=False)
    source = frame.to_pandas() if backend == "pandas" else frame
    run = profile_risk(source, target="y", features=["x", "c"], segment_col="s", group_col="g", n_jobs=1)
    assert run.scopes is not None
    assert run.scopes.height == 9
    for value in ["All", "Total", "Missing", "null", "", "NaN", None, "seg2"]:
        identity = resolve_scopes(run.scopes, [value], False)[0]
        detail = run.report.get_table("detail")
        detail = pl.from_pandas(detail) if backend == "pandas" else detail
        count = detail.filter((pl.col("scope_id") == identity) & (pl.col("feature") == "x") & (pl.col("mars_group") == "Total") & (pl.col("bin_index") == 9999))["count"].item()
        assert count == 9


def test_fixed_rule_round_trip_never_refits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = population()
    run = profile_risk(frame, target="y", features=["x", "c"], segment_col="s",
                       group_col="g", binning_scope="per_segment", n_jobs=1)
    path = tmp_path / "rules.json"
    run.save_binning_rules(path)
    fixed = MarsRiskProfile.load_binning_rules(path)

    def forbidden_fit(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Fixed rules must never fit")

    monkeypatch.setattr(MarsBinnerBase, "fit", forbidden_fit)
    before = {identity: b.to_dict() for identity, b in fixed.items()}
    restored = MarsBinEvaluator(binning_scope="per_segment").evaluate(
        frame, target="y", features=["x", "c"], segment_col="s", group_col="g", segment_binners=fixed,
    )
    keys = ["scope_id", "y", "feature", "mars_group", "bin_index"]
    assert_frame_equal(run.report.detail_table.sort(keys), restored.report.detail_table.sort(keys))
    assert before == {identity: b.to_dict() for identity, b in fixed.items()}
    missing = dict(fixed)
    missing.pop(next(identity for identity in fixed if identity != "all"))
    with pytest.raises(ValueError, match="missing current scope"):
        MarsBinEvaluator(binning_scope="per_segment").evaluate(
            frame, target="y", features=["x"], segment_col="s", segment_binners=missing,
        )


@pytest.mark.parametrize("invalid", [-1.0, float("inf"), None])
def test_invalid_weights_are_global_contract_errors(invalid: float | None) -> None:
    frame = population().with_columns(pl.Series("w", [invalid] + [1.0] * 71))
    with pytest.raises(ValueError, match="finite non-negative"):
        profile_risk(frame, target="y", features=["x"], segment_col="s", weights_col="w")


@pytest.mark.parametrize("mode", ["shared", "per_segment"])
def test_raw_ks_multi_target_is_joined_by_scope(mode: str) -> None:
    frame = population().with_columns((1 - pl.col("y")).alias("y2"))
    run = profile_risk(frame, target=["y", "y2"], features=["x", "c"], segment_col="s",
                       group_col="g", binning_scope=mode, ks_method="raw", n_jobs=1)
    assert run.scopes is not None
    assert run.report.summary_table.height == 16
    for identity in run.scopes["scope_id"].to_list():
        independent = profile_risk(scope_filter(frame, run.scopes, identity), target=["y", "y2"],
                                   features=["x", "c"], group_col="g", ks_method="raw", n_jobs=1)
        actual = run.report.summary_table.filter((pl.col("scope_id") == identity) & (pl.col("feature") == "x"))
        expected = independent.report.summary_table.filter(pl.col("feature") == "x")
        assert actual["ks"].to_list() == expected["ks"].to_list()


def test_null_group_reference_and_feature_empty_expected_population() -> None:
    frame = population().with_columns(pl.lit(None, dtype=pl.String).alias("g"))
    run = profile_risk(frame, target="y", features=["x", "c"], segment_col="s",
                       group_col="g", risk_corr_baseline="first_group", n_jobs=1)
    assert run.report.summary_table["psi_max"].null_count() == 0
    assert all(meta["psi_reference_row_count"] > 0 for meta in run.report.report_meta["scope_metadata"].values())
    reference = population().with_columns(
        pl.when(pl.col("s") == "seg2").then(None).otherwise(pl.col("x")).alias("x")
    )
    run = profile_risk(population(), target="y", features=["x", "c"], segment_col="s",
                       group_col="g", benchmark_df=reference, n_jobs=1)
    assert run.scopes is not None
    identity = resolve_scopes(run.scopes, ["seg2"], False)[0]
    summary = run.report.summary_table.filter(pl.col("scope_id") == identity)
    assert summary.filter(pl.col("feature") == "x")["psi_max"].item() is None
    assert summary.filter(pl.col("feature") == "c")["psi_max"].item() is not None
