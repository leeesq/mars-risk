from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mars.feature import MarsStatsSelector
from mars.feature._scopes import resolve_scopes, scope_filter
from mars.feature.binning.base import MarsBinnerBase
from mars.reporting import get_correlation_matrix, load_report


def _data() -> pl.DataFrame:
    rows = []
    for segment in ("seg1", "seg2", "seg3"):
        for group in ("01", "02"):
            for index in range(80):
                target = index % 2
                x = target if segment == "seg1" else 1 - target if segment == "seg2" else index % 4 // 2
                rows.append({
                    "customer": segment, "month": group, "target": target,
                    "x": float(x), "copy": float(x), "noise": float(index % 5),
                    "local": float(target) if segment == "seg1" else None,
                })
    return pl.DataFrame(rows)


def _selector(**options: object) -> MarsStatsSelector:
    parameters = {
        "selection_scope": "per_segment", "binning_scope": "shared",
        "skip_fine_scan": True, "psi_thr": None, "rc_thr": None,
        "corr_thr": None, "rough_iv_thr": 0.2, "rough_lift_thr": 1e6,
        "missing_thr": 0.6, "n_jobs": 1,
        "rough_binning_params": {"n_bins": 2},
    }
    parameters.update(options)
    return MarsStatsSelector(**parameters)


@pytest.mark.parametrize("binning_scope", ["shared", "per_segment"])
@pytest.mark.parametrize("batch_size", [1, 16, 100])
def test_segment_candidates_are_independent_and_transform_preserves_rows(
    binning_scope: str, batch_size: int,
) -> None:
    frame = _data()
    selector = _selector(binning_scope=binning_scope, batch_size=batch_size)
    selector.fit(frame, target="target", segment_col="customer", group_col="month")
    assert "customer" not in selector._corr_input_features
    assert "local" not in selector.selected_features_
    first = resolve_scopes(selector.scopes_, ["seg1"], False)[0]
    second = resolve_scopes(selector.scopes_, ["seg2"], False)[0]
    assert "local" in selector.selected_features_by_segment_[first]
    assert "local" not in selector.selected_features_by_segment_[second]
    actual = selector.transform(frame, segments=["seg1"], include_all=False)
    expected = frame.filter(pl.col("customer") == "seg1").select([
        *selector.selected_features_by_segment_[first], "target",
    ])
    assert actual.equals(expected)
    decisions = selector.get_report(segments=["seg1"], include_all=False)
    assert decisions["scope_id"].unique().to_list() == [first]
    assert selector.binning_scope == binning_scope
    with pytest.raises(ValueError, match="exactly one"):
        selector.transform(frame, segments=["seg1"])


def test_shared_rules_and_mappings_survive_other_scope_pruning() -> None:
    frame = _data()
    selector = _selector().fit(
        frame, target="target", segment_col="customer", features=["x", "local"],
    )
    first = resolve_scopes(selector.scopes_, ["seg1"], False)[0]
    child = selector.selectors_by_scope_[first]
    assert "local" in child._stage3_binner.features
    assert child._stage3_binner is not selector._stage3_binner
    assert "local" not in selector.selected_features_
    artifact = child._stage3_binner.to_dict()
    restored = MarsBinnerBase.from_dict(artifact)
    current = frame.filter(pl.col("customer") == "seg1")
    assert restored.transform(current, features=["local"], return_type="woe").equals(
        child._stage3_binner.transform(current, features=["local"], return_type="woe")
    )
    second = resolve_scopes(selector.scopes_, ["seg2"], False)[0]
    assert child._stage3_binner.bin_woes_["x"] == selector.selectors_by_scope_[second]._stage3_binner.bin_woes_["x"]


def test_per_segment_matches_independent_public_fit() -> None:
    frame = _data()
    selector = _selector(binning_scope="per_segment").fit(
        frame, target="target", segment_col="customer", features=["x", "local", "noise"],
    )
    for value in ("seg1", "seg2", "seg3"):
        identity = resolve_scopes(selector.scopes_, [value], False)[0]
        reference = _selector(selection_scope="global", binning_scope="shared").fit(
            frame.filter(pl.col("customer") == value), target="target",
            features=["x", "local", "noise"],
        )
        assert selector.selected_features_by_segment_[identity] == reference.selected_features_
        actual = selector.get_report(segments=[value], include_all=False).drop("scope_id")
        assert actual.equals(reference.get_report())


def test_global_registration_keeps_original_schema_and_rejects_roles() -> None:
    frame = _data()
    selector = _selector(selection_scope="global").fit(
        frame, target="target", segment_col="customer", group_col="month",
    )
    assert selector.scopes_ is None
    assert "scope_id" not in selector.get_report().columns
    assert "customer" not in selector._corr_input_features
    with pytest.raises(ValueError, match="explicit feature"):
        selector.fit(frame, target="target", segment_col="customer", features=["customer"])
    with pytest.raises(ValueError, match="not fitted"):
        selector.transform(frame)
    with pytest.raises(ValueError, match="global selection"):
        _selector(selection_scope="global", binning_scope="per_segment")
    with pytest.raises(ValueError, match="requires segment_col"):
        _selector().fit(frame, target="target", features=["x"])


def test_refit_removes_disappeared_scopes_and_failed_fit_has_no_old_results() -> None:
    frame = _data()
    selector = _selector().fit(frame, target="target", segment_col="customer", features=["x"])
    selector.fit(frame.filter(pl.col("customer") != "seg3"), target="target", segment_col="customer", features=["x"])
    with pytest.raises(ValueError, match="Unknown segment"):
        selector.get_report(segments=["seg3"], include_all=False)
    with pytest.raises(ValueError):
        selector.fit(frame.with_columns(pl.lit(-1).alias("target")), target="target", segment_col="customer")
    assert selector.selected_features_by_segment_ == {}
    with pytest.raises(ValueError, match="not fitted"):
        selector.get_correlation_report()


def test_early_failed_refit_clears_old_rules_and_public_decisions() -> None:
    frame = _data()
    selector = _selector().fit(
        frame, target="target", segment_col="customer", features=["x", "local"],
    )
    assert selector._stage3_binner is not None
    assert selector.report_records_
    with pytest.raises(ValueError, match="explicit feature"):
        selector.fit(
            frame, target="target", segment_col="customer", features=["customer"],
        )
    assert selector._stage3_binner is None
    assert selector._rough_binner is None
    assert selector._stability_report is None
    assert selector.report_records_ == []
    assert selector._corr_report is None
    assert selector.selected_features_by_segment_ == {}
    selector.fit(
        frame.filter(pl.col("customer") == "seg1"), target="target",
        segment_col="customer", features=["x"],
    )
    assert selector.selected_features_ == ["x"]
    assert selector.scopes_.height == 2


def test_scope_correlation_uses_saved_matrix_and_survives_snapshot(tmp_path: Path) -> None:
    frame = _data()
    selector = _selector(corr_thr=0.95).fit(
        frame, target="target", segment_col="customer", features=["x", "copy", "local"],
    )
    report = selector.get_correlation_report(segments=["seg1"], include_all=False)
    identity = resolve_scopes(selector.scopes_, ["seg1"], False)[0]
    assert report.get_table("pairs")["scope_id"].unique().to_list() == [identity]
    reference = get_correlation_matrix(selector.selectors_by_scope_[identity].get_correlation_report())
    actual = get_correlation_matrix(report)
    np.testing.assert_allclose(actual, reference, equal_nan=True)
    path = tmp_path / "correlation.marsreport"
    report.save(path)
    restored = load_report(path)
    np.testing.assert_allclose(get_correlation_matrix(restored), actual, equal_nan=True)
    context = json.loads(restored.to_ai_context(tables=["pairs"], max_chars=16000))
    assert context["description"]["report_id"] == report.report_id
    output = subprocess.check_output([
        sys.executable, "-c",
        "import json,sys,polars as pl; from mars.reporting import load_report,get_correlation_matrix; "
        "report=load_report(sys.argv[1]); "
        "print(json.dumps({'python':sys.version.split()[0],'polars':pl.__version__,"
        "'report_id':report.report_id,'matrix':get_correlation_matrix(report).to_numpy().tolist()}))",
        str(path),
    ], text=True)
    fresh = json.loads(output)
    assert fresh["python"] == sys.version.split()[0]
    assert fresh["polars"] == pl.__version__
    assert fresh["report_id"] == report.report_id
    np.testing.assert_allclose(fresh["matrix"], actual, equal_nan=True)


def test_missing_benchmark_segment_does_not_use_overall_expected() -> None:
    frame = _data()
    benchmark = frame.filter((pl.col("month") == "01") & (pl.col("customer") != "seg3"))
    selector = _selector(psi_thr=0.25, rc_thr=None, rough_iv_thr=-1.0).fit(
        frame, target="target", segment_col="customer", features=["x"],
        group_col="month", benchmark_df=benchmark,
    )
    identity = resolve_scopes(selector.scopes_, ["seg3"], False)[0]
    if identity in selector.selectors_by_scope_:
        child = selector.selectors_by_scope_[identity]
        assert child._stability_report.summary_table["psi_max"].null_count() == 1
        assert any(row["reason"] == "metric_unavailable" for row in child.report_records_)
    else:
        pytest.fail("Shared rules must allow a scope with missing PSI reference to finish its funnel.")


def test_fit_transform_uses_same_single_scope_selection() -> None:
    frame = _data().to_pandas()
    selector = _selector().set_output("pandas")
    result = selector.fit_transform(
        frame, target="target", segment_col="customer", features=["x", "local"],
        segments=["seg1"], include_all=False,
    )
    assert len(result) == 160
    assert result["target"].tolist() == frame.loc[frame.customer == "seg1", "target"].tolist()


def test_null_and_reserved_segment_values_keep_distinct_scopes() -> None:
    base = _data().filter(pl.col("customer") == "seg1")
    frame = pl.concat([base.with_columns(pl.lit(value, dtype=pl.String).alias("customer"))
                       for value in ("All", "Total", "Missing", "NaN", None, "")])
    selector = _selector().fit(frame, target="target", segment_col="customer", features=["x"])
    assert selector.scopes_.height == 7
    for value in ("All", "Total", "Missing", "NaN", None, ""):
        identity = resolve_scopes(selector.scopes_, [value], False)[0]
        assert scope_filter(frame, selector.scopes_, identity).height == base.height


@pytest.mark.parametrize("binning_scope", ["shared", "per_segment"])
def test_fine_funnel_keeps_scope_rules_and_correlation(binning_scope: str) -> None:
    frame = _data()
    selector = _selector(
        binning_scope=binning_scope, skip_fine_scan=False, corr_thr=0.95,
        rough_iv_thr=-1.0, iv_thr=-1.0, lift_thr=None,
        binning_params={"n_bins": 2, "n_prebins": 4, "min_bin_n_event": 1},
    ).fit(
        frame, target="target", segment_col="customer", group_col="month",
        features=["x", "copy", "local"],
        benchmark_df=frame.filter(pl.col("month") == "01"),
    )
    first = resolve_scopes(selector.scopes_, ["seg1"], False)[0]
    assert selector.scopes_["status"].to_list() == ["computed"] * 4
    assert "local" in selector.selected_features_by_segment_[first]
    report = selector.get_binning_report(
        frame, benchmark_df=frame.filter(pl.col("month") == "01"),
        segments=["seg1"], include_all=False,
    )
    assert report.get_table("summary")["scope_id"].unique().to_list() == [first]
    correlation = selector.get_correlation_report(segments=["seg1"], include_all=False)
    assert correlation.describe()["parameters"]["correlation_row_count"] == 160


def test_single_class_supervised_scope_is_unavailable_without_fallback() -> None:
    frame = _data().with_columns(
        pl.when(pl.col("customer") == "seg3").then(pl.lit(0))
        .otherwise(pl.col("target")).alias("target"),
    )
    selector = _selector(
        binning_scope="per_segment", rough_binning_params={"method": "cart"},
    ).fit(frame, target="target", segment_col="customer", features=["x"])
    third = resolve_scopes(selector.scopes_, ["seg3"], False)[0]
    assert selector.selected_features_by_segment_[third] is None
    assert selector.scopes_.filter(pl.col("scope_id") == third)["reason"].item() == "single_class_fit"
    with pytest.raises(ValueError, match="single_class_fit"):
        selector.transform(frame, segments=["seg3"], include_all=False)


def test_no_benchmark_uses_same_global_reference_group() -> None:
    frame = _data().filter(~((pl.col("customer") == "seg3") & (pl.col("month") == "01")))
    selector = _selector(psi_thr=0.25, rough_iv_thr=-1.0).fit(
        frame, target="target", segment_col="customer", group_col="month", features=["x"],
    )
    identity = resolve_scopes(selector.scopes_, ["seg3"], False)[0]
    report = selector.selectors_by_scope_[identity]._stability_report
    assert report.summary_table["psi_max"].null_count() == 1
    assert selector.selected_features_by_segment_[identity] == []


def test_unlabelled_benchmark_shared_unsupervised_rules_use_one_mapping_population() -> None:
    frame = _data()
    selector = _selector(rough_iv_thr=-1.0).fit(
        frame, target="target", segment_col="customer", features=["x"],
        benchmark_df=frame.drop("target"),
    )
    first, second = [resolve_scopes(selector.scopes_, [value], False)[0] for value in ("seg1", "seg2")]
    left, right = selector.selectors_by_scope_[first], selector.selectors_by_scope_[second]
    assert left._stage3_binner.bin_woes_["x"] == right._stage3_binner.bin_woes_["x"]
    assert "benchmark contains no target" in left._corr_parameters["woe_mapping_population"]


def test_shared_single_class_statistics_are_not_a_successful_empty_list() -> None:
    frame = _data().with_columns(
        pl.when(pl.col("customer") == "seg3").then(pl.lit(None, dtype=pl.Int64))
        .otherwise(pl.col("target")).alias("target"),
    )
    selector = _selector().fit(frame, target="target", segment_col="customer", features=["x"])
    identity = resolve_scopes(selector.scopes_, ["seg3"], False)[0]
    assert selector.selected_features_by_segment_[identity] is None
    assert selector.scopes_.filter(pl.col("scope_id") == identity)["reason"].item() == "no_observed_labels"


@pytest.mark.parametrize("benchmark_backend", ["pandas", "polars"])
def test_default_pandas_scope_output_follows_current_input(benchmark_backend: str) -> None:
    frame = _data().to_pandas()
    benchmark_pd = frame.loc[frame.month == "01"]
    benchmark = benchmark_pd if benchmark_backend == "pandas" else pl.from_pandas(benchmark_pd)
    selector = _selector().fit(
        frame, target="target", segment_col="customer", features=["x", "local"],
        benchmark_df=benchmark,
    )
    assert isinstance(selector.get_report(), pd.DataFrame)
    actual = selector.transform(frame, segments=["seg1"], include_all=False)
    assert isinstance(actual, pd.DataFrame)
    assert actual["target"].tolist() == frame.loc[frame.customer == "seg1", "target"].tolist()
    report = selector.get_binning_report(
        frame, benchmark_df=benchmark, segments=["seg1"], include_all=False,
    )
    assert isinstance(report.summary_table, pd.DataFrame)
    assert isinstance(selector.get_report(), pd.DataFrame)
    selector.set_output("polars")
    assert isinstance(selector.transform(frame, segments=["seg1"], include_all=False), pl.DataFrame)


def test_default_pandas_scope_fit_transform_keeps_dataframe_type() -> None:
    frame = _data().to_pandas()
    selector = _selector()
    actual = selector.fit_transform(
        frame, target="target", segment_col="customer", features=["x", "local"],
        segments=["seg1"], include_all=False,
    )
    assert isinstance(actual, pd.DataFrame)
