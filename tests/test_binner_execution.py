"""分箱计划、schema 校验和列批执行边界的回归。"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.analysis import MarsBinEvaluator
from mars.feature import MarsNativeBinner, MarsStatsSelector
from mars.feature.binning.base import MarsBinnerBase


def _sample() -> tuple[pl.DataFrame, pl.Series]:
    frame = pl.DataFrame({
        "number": [None, -999.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0] * 10,
        "category": [None, "special", "missing", "a", "b", "c", "a", "b"] * 10,
        "other": list(range(80)),
    })
    return frame, pl.Series("y", [0, 1, None, 0, 1, 0, 1, 1] * 10)


def _binner(join_threshold: int) -> MarsNativeBinner:
    return MarsNativeBinner(
        n_bins=4, n_jobs=1, join_threshold=join_threshold,
        special_values=[-999.0, "special"], missing_values=[-1.0, "missing"],
    )


def _guard_sample_eager(
    monkeypatch: pytest.MonkeyPatch, frame: pl.DataFrame,
) -> None:
    """只阻止样本 eager 派生；允许小型规则映射表正常构建。"""
    for method in ("with_columns", "join"):
        original = getattr(pl.DataFrame, method)

        def guard(self: pl.DataFrame, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            assert not (
                self.height == frame.height and set(frame.columns).intersection(self.columns)
            ), "sample eager operation before lazy collect"
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(pl.DataFrame, method, guard)

    original_collect = pl.LazyFrame.collect

    def collect_guard(self: pl.LazyFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        assert not set(frame.columns).intersection(self.collect_schema().names()), (
            "sample lazy plan executed during construction/schema validation"
        )
        return original_collect(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", collect_guard)


@pytest.mark.parametrize("join_threshold", [0, 100])
def test_lazy_schema_never_executes_eager_sample_transforms(
    monkeypatch: pytest.MonkeyPatch, join_threshold: int,
) -> None:
    frame, y = _sample()
    binner = _binner(join_threshold).fit(frame, y)
    original = frame.clone()
    expected = binner.transform(frame)
    mappings = {col: binner.get_bin_mapping(col) for col in frame.columns}
    with monkeypatch.context() as guard:
        _guard_sample_eager(guard, frame)
        planned = binner.transform(frame, lazy=True)
        assert isinstance(planned, pl.LazyFrame)
        assert planned.collect_schema() == expected.schema
    assert_frame_equal(planned.collect(), expected)
    assert_frame_equal(expected.select(frame.columns), original)
    assert {col: binner.get_bin_mapping(col) for col in frame.columns} == mappings
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("join_threshold", [0, 100])
def test_fit_schema_validation_builds_mappings_without_sample_execution(
    monkeypatch: pytest.MonkeyPatch, join_threshold: int,
) -> None:
    frame, y = _sample()
    original = MarsBinnerBase._transform_impl

    def guarded(self: MarsBinnerBase, X: Any, **kwargs: Any) -> Any:
        assert kwargs["lazy"] is True
        with monkeypatch.context() as guard:
            _guard_sample_eager(guard, frame)
            planned = original(self, X, **kwargs)
            planned.collect_schema()
            return planned

    monkeypatch.setattr(MarsBinnerBase, "_transform_impl", guarded)
    binner = _binner(join_threshold).fit(frame, y)
    assert set(binner.bin_mappings_) == set(frame.columns)
    assert binner.get_fit_report()["usable"].to_list() == [True] * frame.width


@pytest.mark.parametrize("join_threshold", [0, 100])
def test_selector_benchmark_schema_does_not_execute_samples(
    monkeypatch: pytest.MonkeyPatch, join_threshold: int,
) -> None:
    frame, y = _sample()
    binner = _binner(join_threshold).fit(frame, y)
    with monkeypatch.context() as guard:
        _guard_sample_eager(guard, frame)
        MarsStatsSelector._validate_benchmark_bins(binner, frame, frame.columns)
    assert set(binner.bin_mappings_) == set(frame.columns)


def test_first_woe_only_prepares_requested_missing_mappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, y = _sample()
    binner = _binner(0).fit(frame, y)
    calls: list[tuple[list[str] | None, int]] = []
    original = binner.profile_bin_performance

    def tracked(*args: Any, **kwargs: Any) -> Any:
        calls.append((kwargs.get("features"), kwargs["batch_size"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(binner, "profile_bin_performance", tracked)
    result = binner.transform(frame, features=["category"], return_type="woe", lazy=True,
                              woe_batch_size=1)
    assert isinstance(result, pl.LazyFrame)
    assert calls == [(["category"], 1)]
    assert set(binner.bin_woes_) == {"category"}
    binner.transform(frame, features=["category"], return_type="woe", lazy=True)
    assert len(calls) == 1
    binner.transform(frame, features=["number"], return_type="woe", lazy=True,
                     woe_batch_size=1)
    assert set(binner.bin_woes_) == {"category", "number"}


@pytest.mark.parametrize("return_type", ["index", "label", "woe"])
def test_join_and_replace_keep_values_order_unknown_and_special_priority(return_type: str) -> None:
    frame, y = _sample()
    frame = pl.concat([frame.head(13), frame.slice(13)], rechunk=False)
    assert frame["category"].n_chunks() > 1
    replacement = _binner(100).fit(frame, y)
    joined = _binner(0).fit(frame, y)
    probe = frame.with_columns(
        pl.when(pl.col("other") == 3).then(pl.lit("unseen")).otherwise(pl.col("category"))
        .alias("category")
    )
    expected = replacement.transform(probe, return_type=return_type)
    assert_frame_equal(joined.transform(probe, return_type=return_type), expected)
    assert_frame_equal(joined.transform(probe, return_type=return_type, lazy=True).collect(), expected)
    assert_frame_equal(expected.select(probe.columns), probe)
    assert joined.bin_mappings_ == replacement.bin_mappings_
    assert joined.bin_woes_ == replacement.bin_woes_
    if return_type == "index":
        assert expected["category_bin"].head(4).to_list() == [-1, -4, -1, -2]


def test_schema_rejects_illegal_rules_and_failed_refit_clears_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, y = _sample()
    binner = _binner(100).fit(frame, y)
    binner.transform(frame, return_type="woe")
    original = binner._fit_impl

    def invalid_rule(X: pl.DataFrame, y: Any = None) -> None:
        original(X, y)
        binner.bin_cuts_["number"] = [float("-inf"), float("nan"), float("inf")]

    monkeypatch.setattr(binner, "_fit_impl", invalid_rule)
    with pytest.raises(ValueError, match="non-finite internal cuts"):
        binner.fit(frame, y)
    assert not binner.__sklearn_is_fitted__()
    assert not binner.bin_cuts_ and not binner.bin_mappings_ and not binner.bin_woes_
    assert binner._cache_X is None and binner._cache_y is None


@pytest.mark.parametrize("batch_size", [1, 16, 100])
def test_performance_projects_before_filter_and_only_transforms_batch(
    monkeypatch: pytest.MonkeyPatch, batch_size: int,
) -> None:
    frame, y = _sample()
    frame = frame.select([pl.col("number").alias(f"f{i}") for i in range(19)])
    binner = _binner(100).fit(frame, y)
    expected = binner.profile_bin_performance(frame, y, batch_size=100,
                                             include_bin_index=True)
    expected_woe = {key: dict(value) for key, value in binner.bin_woes_.items()}
    original_filter = pl.DataFrame.filter
    original_transform = binner.transform
    calls: list[list[str]] = []

    def filter_guard(self: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        assert self.width <= batch_size, "full-width observed population copy"
        return original_filter(self, *args, **kwargs)

    def transform_guard(X: Any, **kwargs: Any) -> Any:
        active = kwargs.get("features")
        assert active is not None and len(active) <= batch_size
        # 当前批只额外携带一列内部标签，标签必须在 Join 前绑定。
        assert len(X.collect_schema()) <= batch_size + 1
        calls.append(active)
        return original_transform(X, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "filter", filter_guard)
    monkeypatch.setattr(binner, "transform", transform_guard)
    actual = binner.profile_bin_performance(frame, y, batch_size=batch_size,
                                           include_bin_index=True)
    assert_frame_equal(actual, expected)
    assert binner.bin_woes_ == expected_woe
    assert [feature for batch in calls for feature in batch] == frame.columns
    assert actual.group_by("feature").agg(pl.col("count").sum())["count"].to_list() == [70] * 19


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_performance_rejects_invalid_batch_width(batch_size: Any) -> None:
    frame, y = _sample()
    binner = _binner(100).fit(frame, y)
    with pytest.raises(ValueError, match="batch_size"):
        binner.profile_bin_performance(frame, y, batch_size=batch_size)


def test_evaluator_batches_current_and_benchmark_without_eager_long_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, y = _sample()
    binner = _binner(0).fit(frame, y)
    current = frame.with_columns(y, pl.Series("group", ["a"] * 40 + ["b"] * 40))
    benchmark = current.head(48)
    evaluator = MarsBinEvaluator()
    options = {
        "features": frame.columns, "target": "y", "group_col": "group",
        "binner": binner, "benchmark_df": benchmark,
    }
    expected = evaluator.evaluate(current, batch_size=3, **options).report
    original_transform = binner.transform
    original_unpivot = pl.DataFrame.unpivot
    calls: list[tuple[int, list[str]]] = []

    def transform_guard(X: pl.DataFrame, **kwargs: Any) -> Any:
        active = kwargs["features"]
        assert len(active) <= 2
        assert len(set(X.columns) & set(frame.columns)) == len(active)
        calls.append((X.height, active))
        return original_transform(X, **kwargs)

    def unpivot_guard(self: pl.DataFrame, *args: Any, **kwargs: Any) -> Any:
        assert not (
            self.height in {current.height, benchmark.height}
            and any(column.endswith("_bin") for column in self.columns)
        ), "eager sample/benchmark long-table allocation"
        return original_unpivot(self, *args, **kwargs)

    monkeypatch.setattr(binner, "transform", transform_guard)
    monkeypatch.setattr(pl.DataFrame, "unpivot", unpivot_guard)
    actual = evaluator.evaluate(current, batch_size=2, **options).report
    assert calls == [(80, ["number", "category"]), (48, ["number", "category"]),
                     (80, ["other"]), (48, ["other"])]
    assert set(actual._query_tables()) == set(expected._query_tables())
    for name in actual._query_tables():
        left, right = actual.get_table(name), expected.get_table(name)
        keys = [column for column in ["feature", "group", "bin_index"] if column in left.columns]
        assert_frame_equal(left.sort(keys), right.sort(keys))
