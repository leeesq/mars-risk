import pandas as pd
import polars as pl
import pytest

from mars.modeling import MarsModelingSession
from mars.modeling.slicing import MarsModelDataSplitter


def test_model_data_slicer_strict_split_keeps_day_boundaries_and_other_group():
    df = pd.DataFrame(
        {
            "biz_dt": [
                "2024-01-01", "2024-01-01",
                "2024-01-02", "2024-01-02",
                "2024-01-03", "2024-01-03",
                "2024-01-04", "2024-01-04",
            ],
            "target": [0, 1, 0, 1, 0, -1, 1, 0],
            "x1": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    slicer = MarsModelDataSplitter()
    out = slicer.split_by_time_strictly(
        df,
        time_col="biz_dt",
        target="target",
        split_ratios={"train": 0.5, "val": 0.25, "oot1": 0.25},
    )

    assert isinstance(out, pd.DataFrame)
    invalid_row = out.loc[out["target"] == -1].iloc[0]
    assert invalid_row["dataset_flag"] == "other"

    valid = out[out["dataset_flag"] != "other"].copy()
    date_groups = valid.groupby("biz_dt")["dataset_flag"].nunique()
    assert date_groups.max() == 1


def test_model_data_slicer_marks_invalid_dates_as_other():
    df = pl.DataFrame(
        {
            "biz_dt": ["2024-01-01", "bad-date", "2024-01-02", None],
            "target": [0, 1, 1, 0],
            "x1": [1, 2, 3, 4],
        }
    )

    slicer = MarsModelDataSplitter()
    out = slicer.split_by_time_strictly(
        df,
        time_col="biz_dt",
        target="target",
        split_ratios={"train": 0.5, "val": 0.5},
    )

    assert isinstance(out, pl.DataFrame)
    invalid_flags = out.filter(
        (pl.col("biz_dt") == "bad-date") | pl.col("biz_dt").is_null()
    )["dataset_flag"].to_list()
    assert invalid_flags == ["other", "other"]
    assert "unassigned" not in set(out["dataset_flag"].to_list())


def test_model_data_slicer_hybrid_split_preserves_polars_output_type():
    df = pl.DataFrame(
        {
            "biz_dt": [
                "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02",
                "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04",
            ],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "x1": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    slicer = MarsModelDataSplitter()
    out = slicer.split_hybrid_random_val(
        df,
        time_col="biz_dt",
        target="target",
        split_ratios={"train": 0.5, "val": 0.25, "oot1": 0.25},
        random_seed=9,
    )

    assert isinstance(out, pl.DataFrame)
    assert "dataset_flag" in out.columns
    assert "unassigned" not in set(out["dataset_flag"].to_list())


def test_model_data_slicer_rejects_negative_ratios():
    df = pd.DataFrame(
        {
            "biz_dt": ["2024-01-01", "2024-01-02"],
            "target": [0, 1],
            "x1": [1, 2],
        }
    )
    slicer = MarsModelDataSplitter()

    with pytest.raises(ValueError, match="non-negative"):
        slicer.split_by_time_strictly(
            df,
            time_col="biz_dt",
            target="target",
            split_ratios={"train": 1.1, "val": -0.1},
        )


def test_model_data_slicer_hybrid_requires_train_and_val_keys():
    df = pd.DataFrame(
        {
            "biz_dt": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "target": [0, 1, 0],
            "x1": [1, 2, 3],
        }
    )
    slicer = MarsModelDataSplitter()

    with pytest.raises(ValueError, match="train_key"):
        slicer.split_hybrid_random_val(
            df,
            time_col="biz_dt",
            target="target",
            split_ratios={"val": 0.5, "oot1": 0.5},
        )

    with pytest.raises(ValueError, match="val_key"):
        slicer.split_hybrid_random_val(
            df,
            time_col="biz_dt",
            target="target",
            split_ratios={"train": 0.5, "oot1": 0.5},
        )


def test_model_data_slicer_hybrid_requires_positive_modeling_window():
    df = pd.DataFrame(
        {
            "biz_dt": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "target": [0, 1, 0, 1],
            "x1": [1, 2, 3, 4],
        }
    )
    slicer = MarsModelDataSplitter()

    with pytest.raises(ValueError, match="greater than 0"):
        slicer.split_hybrid_random_val(
            df,
            time_col="biz_dt",
            target="target",
            split_ratios={"train": 0.0, "val": 0.0, "oot1": 1.0},
        )


def test_modeling_session_slice_delegates_to_strict_mode():
    session = MarsModelingSession(
        model_type="xgb",
        features=["x1"],
        target="target",
    )
    df = pd.DataFrame(
        {
            "biz_dt": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "target": [0, 1, 0, 1],
            "x1": [1, 2, 3, 4],
        }
    )

    out = session.slice(
        df,
        time_col="biz_dt",
        split_ratios={"train": 0.5, "val": 0.5},
        mode="strict",
    )

    assert isinstance(out, pd.DataFrame)
    assert "dataset_flag" in out.columns


def test_model_data_slicer_pandas_and_polars_strict_are_consistent():
    df_pd = pd.DataFrame(
        {
            "biz_dt": [
                "2024-01-01", "2024-01-01",
                "2024-01-02", "2024-01-02",
                "2024-01-03", "bad-date",
            ],
            "target": [0, 1, 0, 1, 0, 1],
            "x1": [1, 2, 3, 4, 5, 6],
        }
    )
    ratios = {"train": 0.5, "val": 0.25, "oot1": 0.25}

    out_pd = MarsModelDataSplitter().split_by_time_strictly(
        df_pd,
        time_col="biz_dt",
        target="target",
        split_ratios=ratios,
    )
    out_pl = MarsModelDataSplitter().split_by_time_strictly(
        pl.from_pandas(df_pd),
        time_col="biz_dt",
        target="target",
        split_ratios=ratios,
    )

    assert out_pd["dataset_flag"].tolist() == out_pl["dataset_flag"].to_list()


def test_model_data_slicer_pandas_and_polars_hybrid_are_seed_consistent():
    df_pd = pd.DataFrame(
        {
            "biz_dt": [
                "2024-01-01", "2024-01-01",
                "2024-01-02", "2024-01-02",
                "2024-01-03", "2024-01-03",
                "2024-01-04", "2024-01-04",
            ],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "x1": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    ratios = {"train": 0.5, "val": 0.25, "oot1": 0.25}

    out_pd = MarsModelDataSplitter().split_hybrid_random_val(
        df_pd,
        time_col="biz_dt",
        target="target",
        split_ratios=ratios,
        random_seed=99,
    )
    out_pl = MarsModelDataSplitter().split_hybrid_random_val(
        pl.from_pandas(df_pd),
        time_col="biz_dt",
        target="target",
        split_ratios=ratios,
        random_seed=99,
    )

    assert out_pd["dataset_flag"].tolist() == out_pl["dataset_flag"].to_list()
