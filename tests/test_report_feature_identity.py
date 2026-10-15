from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from mars.analysis import profile_risk, profile_stats
from mars.reporting import (
    MarsBinningReport,
    MarsProfileReport,
    ReportSnapshot,
    load_report,
    snapshot_report,
)
from mars.reporting._metadata import normalize_feature_frame


def _sample() -> pl.DataFrame:
    return pl.DataFrame({
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, None],
        "s": ["a", "b", "a", "b", "a", "b", None, "b"],
        "b": [True, False] * 4,
        "e": pl.Series(["a", "b"] * 4, dtype=pl.Enum(["a", "b"])),
        "y": [0, 1, 0, 1, 1, 0, 1, None],
        "y2": [1, 0, 1, 0, 0, 1, None, None],
        "g": ["train"] * 4 + ["oot"] * 4,
    })


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_identity_lifecycle_and_exact_projection(backend: str, tmp_path: Path) -> None:
    source = _sample()
    data = source.to_pandas() if backend == "pandas" else source
    metadata = {f: {"display_name": "重复中文名", "data_source": "application"} for f in ["x", "s", "b", "e"]}
    profile = profile_stats(data, features=["x", "s", "b", "e"], metrics=["missing", "mean"], group_col="g", feature_metadata=metadata, enable_sparkline=False)
    risk = profile_risk(data, target=["y", "y2"], features=["x", "s", "b", "e"], group_col="g", feature_metadata=metadata, ks_method="raw", n_bins=2).report
    assert risk.show_summary(columns=["iv", "feature", "target"]).data.columns.tolist() == ["iv", "feature", "target"]
    assert "bad_rate" not in risk.trend_tables
    assert "bad_rate" in risk.detail_table.columns
    for report in (profile, risk):
        path = tmp_path / f"{report.report_type}.marsreport"
        report.save(path)
        for saved in (report, snapshot_report(report), load_report(path)):
            for name in saved.describe()["tables"]:
                frame = saved.get_table(name)
                if "feature" not in frame.columns:
                    continue
                columns = list(frame.columns)
                start = columns.index("feature")
                assert columns[start:start + 4] == ["feature", "display_name", "dtype", "data_source"]
                assert len(frame) == len(report.get_table(name))
                identity = saved.get_table(name, columns=["data_source", "feature", "display_name"], features="x")
                assert list(identity.columns) == ["data_source", "feature", "display_name"]
                assert saved.show_table(name, columns=["data_source", "feature"]).columns.tolist() == ["data_source", "feature"] if hasattr(saved, "show_table") else True
            definitions = saved.describe()["tables"]
            wide_name = "stats.mean" if report is profile else "trend.ks"
            assert definitions[wide_name]["fields"]["display_name"]["unit"] == "category"
            payload = json.loads(saved.to_ai_context(tables=[wide_name], features="x", max_chars=16000))
            assert payload["description"]["tables"][wide_name]["value_definition"]["unit"] != "category"
    assert risk.detail_table["dtype"].n_unique() >= 3 if backend == "polars" else risk.detail_table["dtype"].nunique() >= 3
    assert source.equals(_sample())


def test_original_types_are_current_input_and_not_benchmark() -> None:
    data = _sample()
    benchmark = data.with_columns(pl.col("x").cast(pl.Int32))
    report = profile_stats(data, features=["x", "e"], benchmark_df=benchmark, metrics=["schema", "missing"], enable_sparkline=False)
    assert report.get_table("overview", features="x")["dtype"].item() == "Float64"
    comparison = report.get_table("comparison.schema", features="x")
    assert comparison["dtype"].item() == "Float64"
    assert comparison["benchmark_dtype"].item() == "Int32"
    assert report.get_table("overview", features="e")["dtype"].item().startswith("Enum")


def test_pandas_normalization_preserves_index_nullable_metrics_and_empty_schema() -> None:
    frame = pd.DataFrame({"feature": ["x", "x"], "value": pd.array([1, None], dtype="Int64")}, index=pd.Index([8, 2], name="row"))
    before = frame.copy(deep=True)
    normalized = normalize_feature_frame(frame, {"x": {"display_name": "中文"}}, {"x": "Boolean"})
    assert_frame_equal(normalized[["feature", "value"]], before)
    assert_frame_equal(frame, before)
    assert_frame_equal(normalize_feature_frame(normalized, {"x": {"display_name": "中文"}}, {"x": "Boolean"}), normalized)
    empty = normalize_feature_frame(frame.iloc[:0], {}, {})
    assert empty["value"].dtype == frame["value"].dtype
    assert empty.index.equals(frame.index[:0])
    with pytest.raises(ValueError, match="Conflicting display_name"):
        normalize_feature_frame(normalized, {"x": {"display_name": "矛盾"}}, {"x": "Boolean"})
    with pytest.raises(ValueError, match="Conflicting dtype"):
        normalize_feature_frame(normalized, {}, {"x": "String"})


def test_explicit_show_sort_is_global_and_projection_is_exact() -> None:
    report = MarsProfileReport(pl.DataFrame({"feature": ["a", "b", "c"], "dtype": ["String", "Int64", "Boolean"], "missing_rate": [0.1, 0.9, 0.2]}), {}, {})
    styled = report.show_overview(sort_by="missing_rate", sort_ascending=False, columns=["missing_rate", "feature"])
    assert styled.data.columns.tolist() == ["missing_rate", "feature"]
    assert styled.data["feature"].tolist() == ["b", "c", "a"]
    with pytest.raises(ValueError, match="Unknown columns"):
        report.show_overview(sort_by="missing")
    trend = MarsBinningReport(pl.DataFrame({"feature": ["x"]}), {"psi": pl.DataFrame({"feature": ["x"], "B": [0.2], "A": [0.1], "Total": [0.3]})}, pl.DataFrame())
    assert trend.show_trend("psi", columns=["Total", "feature", "B"], sort_by="A").data.columns.tolist() == ["Total", "feature", "B"]


def test_legacy_binning_snapshot_preserves_tables_and_does_not_guess_original_type() -> None:
    restored = load_report(Path(__file__).parent / "fixtures" / "legacy_e990813_no_target.marsreport")
    for name in restored.describe()["tables"]:
        if name.startswith("trend."):
            frame = restored.get_table(name)
            assert "display_name" in frame.columns
            assert frame["dtype"].null_count() == frame.height


@pytest.mark.parametrize("group_value", ["display_name", "dtype", "data_source", "feature"])
@pytest.mark.parametrize("analysis", ["profile", "risk"])
def test_group_labels_cannot_overwrite_identity_columns(group_value: str, analysis: str) -> None:
    data = _sample().with_columns(pl.lit(group_value).alias("g"))
    with pytest.raises(ValueError, match="reserved report columns"):
        if analysis == "profile":
            profile_stats(data, features=["x"], metrics=["missing"], group_col="g", enable_sparkline=False)
        else:
            profile_risk(data, features=["x"], target="y", group_col="g")


def test_enum_uses_existing_categorical_statistics_and_original_type() -> None:
    data = _sample()
    enum_report = profile_risk(data, target="y", features=["e"], n_bins=2).report
    string_report = profile_risk(data.with_columns(pl.col("e").cast(pl.String)), target="y", features=["e"], n_bins=2).report
    columns = ["feature", "iv", "ks", "auc", "missing"]
    assert enum_report.get_table("summary", columns=columns).equals(string_report.get_table("summary", columns=columns))
    assert enum_report.get_table("summary")["dtype"].item().startswith("Enum")


def test_existing_names_are_unique_and_join_the_searchable_dictionary() -> None:
    report = MarsBinningReport(pl.DataFrame({"feature": ["x", "x"], "display_name": ["甲", "甲"], "iv": [0.1, 0.2]}), {}, pl.DataFrame())
    assert report.feature_metadata["x"]["display_name"] == "甲"
    assert report.search_features("甲")[0]["feature"] == "x"
    with pytest.raises(ValueError, match="Conflicting display_name"):
        MarsBinningReport(pl.DataFrame({"feature": ["x", "x"], "display_name": ["甲", "乙"]}), {}, pl.DataFrame())


def test_legacy_aggregate_dtype_does_not_conflict_with_reliable_summary_type() -> None:
    summary = pl.DataFrame({"feature": ["x"], "dtype": ["String"], "iv": [0.1]})
    trend = pl.DataFrame({"feature": ["x"], "dtype": ["Float64"], "Total": [12.5]})
    report = MarsBinningReport(summary, {"ks": trend}, pl.DataFrame())
    description = report.describe()
    description["parameters"].pop("feature_dtypes")
    legacy = ReportSnapshot({"summary": summary, "trend.ks": trend, "detail": pl.DataFrame()}, description)
    assert legacy.get_table("trend.ks")["dtype"].item() == "String"
    assert legacy.get_table("trend.ks")["Total"].item() == 12.5


@pytest.mark.parametrize("missing", [pd.NA, None, float("nan")])
@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize("dtype_storage", ["string", "object"])
def test_nullable_binning_dtype_normalization_through_report_and_snapshot(
    missing: object, trusted: bool, dtype_storage: str, tmp_path: Path,
) -> None:
    """缺失 dtype 先归一化；仅可靠汇总类型可补齐，保存恢复保持未知与原指标。"""
    summary = pd.DataFrame({"feature": ["x"], "iv": [0.1]})
    if trusted:
        summary["dtype"] = pd.array(["String"], dtype="string")
    trend = pd.DataFrame({
        "feature": ["x"], "dtype": pd.array([missing], dtype=dtype_storage),
        "Total": pd.array([12.5], dtype="Float64"),
    }, index=pd.Index([8], name="row"))
    trend_before = trend.copy(deep=True)
    report = MarsBinningReport(summary, {"ks": trend}, pd.DataFrame())
    result = report.get_table("trend.ks")
    expected_dtype = "String" if trusted else None
    assert result["dtype"].tolist() == [expected_dtype]
    assert result.index.equals(trend.index)
    assert result["Total"].dtype == trend["Total"].dtype
    normalized = normalize_feature_frame(
        result, {}, {"x": "String"} if trusted else {}, discard_aggregate_dtype=True,
    )
    assert_frame_equal(normalized, result)
    assert_frame_equal(trend, trend_before)
    path = tmp_path / "nullable-dtype.marsreport"
    report.save(path)
    for snapshot in (snapshot_report(report), load_report(path)):
        assert_frame_equal(snapshot.get_table("trend.ks"), result)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_missing_binning_dtype_preserves_empty_schema_and_rejects_conflicts(backend: str) -> None:
    """空表保留指标类型，非空原类型矛盾不能被缺失处理放过。"""
    pandas_frame = pd.DataFrame({
        "feature": pd.Series([], dtype="string"), "dtype": pd.Series([], dtype="string"),
        "Total": pd.Series([], dtype="Float64"),
    }, index=pd.Index([], name="row", dtype="int64"))
    frame = pl.from_pandas(pandas_frame) if backend == "polars" else pandas_frame
    result = normalize_feature_frame(frame, {}, {}, discard_aggregate_dtype=True)
    if backend == "pandas":
        assert result.index.equals(frame.index)
        assert result["Total"].dtype == frame["Total"].dtype
        assert_frame_equal(normalize_feature_frame(result, {}, {}, discard_aggregate_dtype=True), result)
    else:
        assert result.schema["Total"] == frame.schema["Total"]
        assert result.equals(normalize_feature_frame(result, {}, {}, discard_aggregate_dtype=True))
    conflicting = pd.DataFrame({"feature": ["x"], "dtype": pd.array(["Int64"], dtype="string")})
    if backend == "polars":
        conflicting = pl.from_pandas(conflicting)
    with pytest.raises(ValueError, match="Conflicting dtype"):
        normalize_feature_frame(conflicting, {}, {"x": "String"}, discard_aggregate_dtype=True)
