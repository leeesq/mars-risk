import inspect

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mars.analysis import MarsBinEvaluator, profile_risk
from mars.feature import MarsLiteOptBinner


def _make_shape_frame(shape: str, *, groups: int = 24, rows_per_group: int = 30) -> tuple[pl.DataFrame, pl.Series]:
    values: list[float] = []
    targets: list[int] = []
    categories: list[str] = []
    center = (groups - 1) / 2

    for group in range(groups):
        if shape == "ascending":
            bad_count = 2 + int(group / (groups - 1) * (rows_per_group - 5))
        elif shape == "descending":
            bad_count = 2 + int((groups - 1 - group) / (groups - 1) * (rows_per_group - 5))
        elif shape == "peak":
            bad_count = 2 + int((1 - abs(group - center) / center) * (rows_per_group - 5))
        elif shape == "valley":
            bad_count = 2 + int(abs(group - center) / center * (rows_per_group - 5))
        else:
            raise ValueError(f"Unknown shape: {shape}")

        values.extend((group + np.linspace(0.0, 0.8, rows_per_group)).tolist())
        targets.extend([1] * bad_count + [0] * (rows_per_group - bad_count))
        categories.extend([f"cat_{group % 5}"] * rows_per_group)

    return pl.DataFrame({"score": values, "segment": categories}), pl.Series("target", targets)


def _normal_stats(binner: MarsLiteOptBinner, X: pl.DataFrame, y: pl.Series) -> pl.DataFrame:
    stats = binner.profile_bin_performance(
        X.select(["score"]),
        y,
        update_woe=False,
        include_bin_index=True,
    )
    if not isinstance(stats, pl.DataFrame):
        stats = pl.from_pandas(stats)
    return stats.filter(pl.col("bin_index") >= 0).sort("bin_index")


def _normal_bad_rates(binner: MarsLiteOptBinner, X: pl.DataFrame, y: pl.Series) -> np.ndarray:
    return _normal_stats(binner, X, y).get_column("bad_rate").to_numpy()


def _is_non_decreasing(values: np.ndarray, tolerance: float = 1e-9) -> bool:
    return bool(np.all(np.diff(values) >= -tolerance))


def _is_non_increasing(values: np.ndarray, tolerance: float = 1e-9) -> bool:
    return bool(np.all(np.diff(values) <= tolerance))


def _is_peak(values: np.ndarray) -> bool:
    pivot = int(np.argmax(values))
    return 0 < pivot < len(values) - 1 and _is_non_decreasing(values[: pivot + 1]) and _is_non_increasing(values[pivot:])


def _is_valley(values: np.ndarray) -> bool:
    pivot = int(np.argmin(values))
    return 0 < pivot < len(values) - 1 and _is_non_increasing(values[: pivot + 1]) and _is_non_decreasing(values[pivot:])


def test_lite_opt_binner_public_signature_exposes_fit_parameters() -> None:
    fit_sig = inspect.signature(MarsLiteOptBinner.fit)

    assert list(fit_sig.parameters) == ["self", "X", "y", "features", "cat_features"]
    assert fit_sig.parameters["y"].default is inspect.Parameter.empty
    assert fit_sig.parameters["features"].kind is inspect.Parameter.KEYWORD_ONLY
    assert fit_sig.parameters["cat_features"].kind is inspect.Parameter.KEYWORD_ONLY


def test_lite_opt_binner_enforces_ascending_trend() -> None:
    X, y = _make_shape_frame("ascending")
    binner = MarsLiteOptBinner(
        n_bins=6,
        n_prebins=24,
        min_bin_size=0.03,
        monotonic_trend="ascending",
    )

    transformed = binner.fit_transform(X.select(["score"]), y, features=["score"])
    bad_rates = _normal_bad_rates(binner, X, y)

    assert "score_bin" in transformed.columns
    assert len(binner.bin_cuts_["score"]) - 1 <= 6
    assert _is_non_decreasing(bad_rates)


def test_lite_opt_binner_enforces_descending_trend() -> None:
    X, y = _make_shape_frame("descending")
    binner = MarsLiteOptBinner(
        n_bins=6,
        n_prebins=24,
        min_bin_size=0.03,
        monotonic_trend="descending",
    )

    binner.fit(X.select(["score"]), y, features=["score"])
    bad_rates = _normal_bad_rates(binner, X, y)

    assert len(binner.bin_cuts_["score"]) - 1 <= 6
    assert _is_non_increasing(bad_rates)


def test_lite_opt_binner_enforces_peak_and_valley_trends() -> None:
    peak_X, peak_y = _make_shape_frame("peak")
    valley_X, valley_y = _make_shape_frame("valley")
    peak_binner = MarsLiteOptBinner(
        n_bins=7,
        n_prebins=24,
        min_bin_size=0.03,
        monotonic_trend="peak",
    )
    valley_binner = MarsLiteOptBinner(
        n_bins=7,
        n_prebins=24,
        min_bin_size=0.03,
        monotonic_trend="valley",
    )

    peak_binner.fit(peak_X.select(["score"]), peak_y, features=["score"])
    valley_binner.fit(valley_X.select(["score"]), valley_y, features=["score"])

    assert _is_peak(_normal_bad_rates(peak_binner, peak_X, peak_y))
    assert _is_valley(_normal_bad_rates(valley_binner, valley_X, valley_y))


def test_lite_opt_binner_auto_selects_valid_candidate() -> None:
    X, y = _make_shape_frame("valley")
    binner = MarsLiteOptBinner(
        n_bins=6,
        n_prebins=24,
        min_bin_size=0.04,
        monotonic_trend="auto",
    )

    binner.fit(X.select(["score"]), y, features=["score"])
    bad_rates = _normal_bad_rates(binner, X, y)

    assert 1 <= len(binner.bin_cuts_["score"]) - 1 <= 6
    assert binner.fitted_trends_["score"] in {"ascending", "descending", "peak", "valley"}
    assert set(binner.candidate_scores_["score"]) == {"ascending", "descending", "peak", "valley"}
    assert (
        _is_non_decreasing(bad_rates)
        or _is_non_increasing(bad_rates)
        or _is_peak(bad_rates)
        or _is_valley(bad_rates)
    )


def test_lite_opt_binner_auto_asc_desc_selects_single_monotonic_direction() -> None:
    ascending_X, ascending_y = _make_shape_frame("ascending")
    descending_X, descending_y = _make_shape_frame("descending")
    ascending_binner = MarsLiteOptBinner(
        n_bins=6,
        n_prebins=24,
        min_bin_size=0.04,
        monotonic_trend="auto_asc_desc",
    )
    descending_binner = MarsLiteOptBinner(
        n_bins=6,
        n_prebins=24,
        min_bin_size=0.04,
        monotonic_trend="auto_asc_desc",
    )

    ascending_binner.fit(ascending_X.select(["score"]), ascending_y, features=["score"])
    descending_binner.fit(descending_X.select(["score"]), descending_y, features=["score"])

    ascending_rates = _normal_bad_rates(ascending_binner, ascending_X, ascending_y)
    descending_rates = _normal_bad_rates(descending_binner, descending_X, descending_y)

    assert ascending_binner.fitted_trends_["score"] == "ascending"
    assert descending_binner.fitted_trends_["score"] == "descending"
    assert set(ascending_binner.candidate_scores_["score"]) == {"ascending", "descending"}
    assert set(descending_binner.candidate_scores_["score"]) == {"ascending", "descending"}
    assert _is_non_decreasing(ascending_rates)
    assert _is_non_increasing(descending_rates)


def test_lite_opt_binner_supports_cart_prebinning_and_profile_bin_index() -> None:
    X, y = _make_shape_frame("ascending")
    binner = MarsLiteOptBinner(
        n_bins=5,
        n_prebins=12,
        min_bin_size=0.03,
        monotonic_trend="ascending",
        prebinning_method="cart",
    )

    binner.fit(X.select(["score"]), y, features=["score"])
    default_stats = binner.profile_bin_performance(X.select(["score"]), y, update_woe=False)
    indexed_stats = binner.profile_bin_performance(
        X.select(["score"]),
        y,
        update_woe=False,
        include_bin_index=True,
    )

    assert "bin_index" not in default_stats.columns
    assert "bin_index" in indexed_stats.columns
    assert binner.fitted_trends_["score"] == "ascending"


def test_lite_opt_binner_supports_pandas_input_and_constant_columns() -> None:
    X, y = _make_shape_frame("ascending", groups=8, rows_per_group=10)
    pandas_X = X.with_columns(pl.lit(1.0).alias("constant")).to_pandas()
    pandas_y = pd.Series(y.to_list(), name="target")
    binner = MarsLiteOptBinner(
        n_bins=4,
        n_prebins=8,
        min_bin_size=0.05,
        monotonic_trend="ascending",
    )

    binner.fit(pandas_X, pandas_y, features=["score", "constant"])
    transformed = binner.transform(pandas_X.head(4), return_type="index")

    assert binner.bin_cuts_["constant"] == [float("-inf"), float("inf")]
    assert "score_bin" in transformed.columns
    assert binner.fitted_trends_["score"] == "ascending"


def test_lite_opt_binner_respects_min_bin_size_after_trend_merge() -> None:
    X, y = _make_shape_frame("ascending", groups=30, rows_per_group=20)
    binner = MarsLiteOptBinner(
        n_bins=10,
        n_prebins=30,
        min_bin_size=0.12,
        monotonic_trend="ascending",
    )

    binner.fit(X.select(["score"]), y, features=["score"])
    stats = _normal_stats(binner, X, y)
    counts = stats.get_column("count").to_numpy()

    assert len(binner.bin_cuts_["score"]) - 1 <= 10
    assert np.all(counts / X.height >= 0.12)


def test_lite_opt_binner_handles_categories_unknowns_missing_and_special_values() -> None:
    X, y = _make_shape_frame("ascending", groups=10, rows_per_group=12)
    X = pl.concat(
        [
            X,
            pl.DataFrame(
                {
                    "score": [-999.0, np.nan, 11.0],
                    "segment": ["SPECIAL", None, "cat_rare"],
                }
            ),
        ],
        how="vertical_relaxed",
    )
    y = pl.Series("target", y.to_list() + [1, 0, 1])
    binner = MarsLiteOptBinner(
        n_bins=4,
        n_prebins=12,
        min_bin_size=0.04,
        monotonic_trend="ascending",
        special_values=[-999.0, "SPECIAL"],
    )

    transformed = binner.fit_transform(
        X,
        y,
        features=["score", "segment"],
        cat_features=["segment"],
    )
    unseen = binner.transform(
        pl.DataFrame({"score": [0.2, -999.0, None], "segment": ["unknown", "SPECIAL", None]}),
        return_type="index",
    )

    assert "segment" in binner.cat_cuts_
    assert len(binner.bin_cuts_["score"]) - 1 <= 4
    assert transformed["score_bin"].to_list()[-2] == -1
    assert transformed["score_bin"].to_list()[-3] == -3
    assert unseen["segment_bin"].to_list() == [-2, -4, -1]


def test_lite_opt_binner_rejects_missing_y_and_invalid_config() -> None:
    X, y = _make_shape_frame("ascending")

    with pytest.raises(ValueError, match="requires y"):
        MarsLiteOptBinner().fit(X.select(["score"]), None)
    with pytest.raises(ValueError, match="auto_asc_desc"):
        MarsLiteOptBinner(monotonic_trend="flat")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="prebinning_method"):
        MarsLiteOptBinner(prebinning_method="rank")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid values"):
        MarsLiteOptBinner().fit(X.select(["score"]), pl.Series("target", [0, 1, 2] * (X.height // 3)))

    assert y.len() == X.height


def test_lite_opt_binner_dict_roundtrip_and_label_transform() -> None:
    X, y = _make_shape_frame("ascending")
    binner = MarsLiteOptBinner(n_bins=5, n_prebins=20, monotonic_trend="auto_asc_desc")
    binner.fit(X.select(["score"]), y, features=["score"])

    restored = MarsLiteOptBinner.from_dict(binner.to_dict())
    labels = restored.transform(X.select(["score"]).head(5), return_type="label")

    assert "score_bin" in labels.columns
    assert restored.monotonic_trend == "auto_asc_desc"
    assert restored.bin_cuts_ == binner.bin_cuts_
    assert restored.fitted_trends_ == binner.fitted_trends_
    assert restored.candidate_scores_ == binner.candidate_scores_


def test_lite_opt_binner_routes_through_evaluator_and_profile_risk() -> None:
    X, y = _make_shape_frame("ascending", groups=12, rows_per_group=12)
    df = X.with_columns(y.alias("target"))

    evaluator_run = MarsBinEvaluator(
        binning_type="lite_opt",
        binner_params={"n_bins": 4, "n_prebins": 12, "monotonic_trend": "auto_asc_desc"},
    ).evaluate(df, target="target", features=["score"])
    profile_run = profile_risk(
        df,
        target="target",
        features=["score"],
        binning_type="lite_opt",
        binner_params={"n_bins": 4, "n_prebins": 12, "monotonic_trend": "auto_asc_desc"},
        plot=False,
    )

    assert isinstance(evaluator_run.binner, MarsLiteOptBinner)
    assert isinstance(profile_run.binner, MarsLiteOptBinner)
    assert "score" in evaluator_run.report.summary_table.get_column("feature").to_list()
