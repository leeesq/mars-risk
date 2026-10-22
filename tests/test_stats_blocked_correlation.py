"""完整 WOE Pearson 的数值、执行边界和跨批决策回归。"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from mars.compute._correlation import _blocked_pearson
from mars.feature import MarsStatsSelector
from mars.feature.binning import MarsNativeBinner
from mars.feature.binning.base import MarsBinnerBase


@pytest.mark.parametrize("rows", [0, 1, 2, 31, 103])
@pytest.mark.parametrize("block_size", [1, 7, 100])
def test_block_moments_match_dense_degenerate_and_signed(rows: int, block_size: int) -> None:
    rng = np.random.default_rng(127)
    raw = rng.normal(size=(rows, 3))
    values = np.column_stack([raw, -raw[:, 0], raw[:, 0], np.ones(rows)])
    actual, count = _blocked_pearson(
        (values[i:i + block_size] for i in range(0, rows, block_size)), 6,
    )
    assert count == rows
    with np.errstate(invalid="ignore", divide="ignore"):
        if rows > 1:
            expected = np.corrcoef(values, rowvar=False)
        else:
            expected = np.full((6, 6), np.nan)
    np.testing.assert_allclose(actual, expected, atol=2e-15, equal_nan=True)
    np.testing.assert_allclose(actual, actual.T, atol=2e-15, equal_nan=True)
    assert np.isnan(actual[-1, -1])


def test_block_moments_large_offset_and_nonfinite_propagation() -> None:
    rng = np.random.default_rng(15)
    small = rng.integers(-100, 100, size=(91, 3)).astype(float) / 16
    large = small + 2 ** 40
    actual, _ = _blocked_pearson((large[i:i + 7] for i in range(0, 91, 7)), 3)
    np.testing.assert_allclose(actual, np.corrcoef(small, rowvar=False), atol=2e-15)
    invalid = small.copy()
    invalid[4, 0] = np.inf
    invalid[10, 1] = np.nan
    actual, _ = _blocked_pearson([invalid], 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.corrcoef(invalid, rowvar=False)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def _selector(batch: int | None) -> MarsStatsSelector:
    return MarsStatsSelector(
        batch_size=batch, skip_fine_scan=True, psi_thr=None, rc_thr=None,
        rough_iv_thr=-1, rough_lift_thr=0, corr_thr=0.95, n_jobs=1,
        rough_binning_params={"method": "quantile", "n_bins": 4},
    )


@pytest.mark.parametrize("batch", [1, 16, 100, None])
def test_global_woe_cross_batch_dense_oracle_and_all_candidates(
    batch: int | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(141)
    signal = rng.normal(size=4197)
    target = (rng.random(4197) < 1 / (1 + np.exp(-signal))).astype(int).tolist()
    target[-93:] = [None] * 93
    features = [f"feature_{i:02}_long_bureau_source_name" for i in range(19)]
    data = pl.DataFrame({
        **{feature: signal if i in [0, 17] else -signal if i == 18
           else rng.normal(size=4197) for i, feature in enumerate(features)},
        "bad": target,
    })
    blocks: list[np.ndarray] = []
    original = MarsStatsSelector._woe_row_blocks
    calls: list[tuple[int, int]] = []
    transform = MarsBinnerBase.transform

    def track(selector: MarsStatsSelector, frame: pl.DataFrame, names: list[str], _: int) -> Any:
        # 强制更小行块验证边界，不改变候选或人口。
        for values in original(selector, frame, names, 71):
            blocks.append(values.copy())
            yield values

    def bounded(binner: MarsBinnerBase, frame: Any, **kwargs: Any) -> Any:
        if kwargs.get("return_type") == "woe":
            calls.append((frame.height, len(kwargs["features"])))
        return transform(binner, frame, **kwargs)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Full dense DataFrame.corr must not execute")

    monkeypatch.setattr(MarsStatsSelector, "_woe_row_blocks", track)
    monkeypatch.setattr(MarsBinnerBase, "transform", bounded)
    monkeypatch.setattr(pl.DataFrame, "corr", forbidden)
    selector = _selector(batch).fit(
        data, target="bad", features=features,
        feature_data_source={"A": features[:16], "B": features[16:]},
    )
    oracle = np.corrcoef(np.concatenate(blocks), rowvar=False)
    report = selector.get_correlation_report()
    np.testing.assert_allclose(report.get_matrix().to_numpy(), oracle, atol=3e-15)
    assert report.get_table("pairs").height == 19 * 18 // 2
    assert report.describe()["parameters"]["correlation_row_count"] == 4104
    assert max(rows for rows, _ in calls) <= 71
    assert max(cols for _, cols in calls) <= (batch or 100)
    # 同一个旧 dense 矩阵驱动原优先级和决策循环，比较实际事件和最终原序名单。
    dense = _selector(batch)
    dense._stage3_binner = selector._stage3_binner
    dense.target = "bad"
    dense._feature_iv_dict = selector._feature_iv_dict.copy()
    dense._corr_parameters = selector._corr_parameters.copy()
    dense._reset_correlation()
    dense._corr_parameters = selector._corr_parameters.copy()
    monkeypatch.setattr("mars.feature.selection.stats._blocked_pearson", lambda *_: (oracle, 4104))
    expected = dense._filter_corr(data, features)
    assert selector.selected_features_ == expected
    assert len(selector._corr_events) == len(dense._corr_events)
    for actual, expected_event in zip(selector._corr_events, dense._corr_events):
        for key in actual:
            if key in {"correlation", "abs_correlation"} and actual[key] is not None:
                assert actual[key] == pytest.approx(expected_event[key], abs=3e-15)
            else:
                assert actual[key] == expected_event[key]
    assert features[17] not in expected


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_feature_batch_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        MarsStatsSelector(batch_size=value)


def test_threshold_refinement_preserves_strict_dense_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    data = pl.DataFrame({"a": [-1.0, -1.0, 1.0, 1.0] * 9,
                         "b": [-1.0, 1.0, -1.0, 1.0] * 9, "bad": [0, 1, 0, 1] * 9})
    binner = MarsNativeBinner(n_bins=2, n_jobs=1).fit(
        data, data["bad"], features=["a", "b"],
    )
    # 固定已经拟合的映射，验证 WOE 的有符号系数和 null 填零，而非 raw 相关性。
    binner.bin_woes_ = {"a": {0: -2.0, 1: 2.0}, "b": {0: -3.0, 1: 3.0}}
    observed = binner.transform(data, features=["a", "b"], return_type="woe")
    oracle = np.corrcoef(observed.select(["a_woe", "b_woe"]).to_numpy(), rowvar=False)
    selector = _selector(1)
    selector._reset_correlation()
    selector.target = "bad"
    selector.corr_thr = float(abs(oracle[0, 1]))
    selector._stage3_binner = binner
    selector._corr_parameters = {"operator": ">", "priority_metric": "IV", "tie_rule": "ID"}
    perturbed = oracle.copy()
    perturbed[0, 1] = perturbed[1, 0] = np.nextafter(selector.corr_thr, 1.0)
    monkeypatch.setattr("mars.feature.selection.stats._blocked_pearson",
                        lambda *_: (perturbed, data.height))
    assert selector._filter_corr(data, ["a", "b"]) == ["a", "b"]
    assert selector._corr_matrix[0, 1] == oracle[0, 1]
    assert selector._corr_parameters["threshold_refined_pairs"] == 1


def test_threshold_one_skips_quadratic_narrow_pair_rechecks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = _selector(1)
    selector.target = "bad"
    selector._stage3_binner = MarsNativeBinner(n_jobs=1)
    selector.corr_thr = 1.0

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Clipped correlations cannot exceed one; no pair transform needed")

    monkeypatch.setattr(selector._stage3_binner, "transform", forbidden)
    assert selector._refine_threshold_pairs(
        pl.DataFrame(), [f"f{i}" for i in range(16)], np.ones((16, 16)), 100,
    ) == 0


@pytest.mark.parametrize("nonfinite", [None, float("nan"), float("inf")])
def test_woe_null_only_fill_and_signed_negative(
    nonfinite: float | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = pl.DataFrame({"a": [None, -1.0, -1.0, 1.0, 1.0] * 3,
                         "b": [None, -1.0, -1.0, 1.0, 1.0] * 3,
                         "bad": [0, 0, 1, 0, 1] * 3})
    binner = MarsNativeBinner(n_bins=2, n_jobs=1).fit(
        data, data["bad"], features=["a", "b"],
    )
    missing_woe = 0.0 if nonfinite is None else nonfinite
    binner.bin_woes_ = {"a": {-1: missing_woe, 0: -2.0, 1: 2.0},
                        "b": {-1: missing_woe, 0: 2.0, 1: -2.0}}
    if nonfinite is None:
        transform = binner.transform

        def nullable_output(frame: Any, **kwargs: Any) -> Any:
            # 映射本身只允许浮点值；在边界注入合法 nullable WOE 表，检查 selector 的填值策略。
            result = transform(frame, **kwargs)
            return result.with_columns([
                pl.when(pl.col(feature).is_null()).then(None)
                .otherwise(pl.col(f"{feature}_woe")).alias(f"{feature}_woe")
                for feature in kwargs["features"]
            ])

        monkeypatch.setattr(binner, "transform", nullable_output)
    selector = _selector(1)
    selector.target = "bad"
    selector._stage3_binner = binner
    blocks = list(selector._woe_row_blocks(data, ["a", "b"], 4))
    actual, count = _blocked_pearson(blocks, 2)
    dense = binner.transform(data, features=["a", "b"], return_type="woe").select(
        pl.col("a_woe", "b_woe").fill_null(0.0),
    ).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        oracle = np.corrcoef(dense, rowvar=False)
    np.testing.assert_allclose(actual, oracle, atol=2e-15, equal_nan=True)
    assert count == 15
    if nonfinite is None:
        assert actual[0, 1] == pytest.approx(-1)
        assert np.concatenate(blocks)[0, 0] == 0
    else:
        assert np.isnan(actual).all()


@pytest.mark.parametrize("fine", [False, True])
@pytest.mark.parametrize("benchmark", [False, True])
def test_selector_population_and_actual_batches(
    fine: bool, benchmark: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(159)
    signal = rng.normal(size=180)
    data = pl.DataFrame({**{f"x{i}": signal + rng.normal(0, .1, 180) for i in range(5)},
                         "bad": ([0, 1] * 75) + [None] * 30,
                         "g": ["train"] * 120 + ["val"] * 60})
    features = [f"x{i}" for i in range(5)]
    reference = data.head(120) if benchmark else None
    options = dict(skip_fine_scan=not fine, rough_iv_thr=-1, rough_lift_thr=0,
                   iv_thr=-1, lift_thr=0, rc_thr=-1, psi_thr=100,
                   binning_params={"prebinning_method": "quantile", "n_jobs": 1}, n_jobs=1)
    expected = MarsStatsSelector(batch_size=100, **options).fit(
        data, target="bad", features=features, group_col="g", benchmark_df=reference,
    )
    original = MarsBinnerBase._transform_impl
    widths: list[int] = []

    def record(binner: MarsBinnerBase, frame: Any, **kwargs: Any) -> Any:
        # fit 的 schema 计划可含全部规则；真正样本物化由受测列批约束。
        if kwargs.get("return_type") == "woe":
            widths.append(len(kwargs.get("features") or features))
        return original(binner, frame, **kwargs)

    monkeypatch.setattr(MarsBinnerBase, "_transform_impl", record)
    actual = MarsStatsSelector(batch_size=2, **options).fit(
        data, target="bad", features=features, group_col="g", benchmark_df=reference,
    )
    assert actual.selected_features_ == expected.selected_features_
    np.testing.assert_allclose(actual.get_correlation_report().get_matrix(),
                               expected.get_correlation_report().get_matrix(), atol=3e-15)
    assert actual.get_correlation_report().describe()["parameters"]["input_row_count"] == 180
    assert actual.get_correlation_report().describe()["parameters"]["correlation_row_count"] == 150
    assert actual._benchmark_row_count == (120 if benchmark else None)
    assert widths and all(width <= 2 for width in widths)
