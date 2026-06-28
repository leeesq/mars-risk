import polars as pl
import pytest

from mars.analysis.missing_shift import MarsMissingShiftResult, MarsMissingShiftScanner


def _daily_missing_frame(
    *,
    rates: list[float],
    rows_per_day: int = 100,
    feature: str = "x",
) -> pl.DataFrame:
    rows = []
    for day_idx, rate in enumerate(rates, start=1):
        missing_count = int(round(rows_per_day * rate))
        for row_idx in range(rows_per_day):
            rows.append(
                {
                    "apply_dt": f"2026-01-{day_idx:02d}",
                    feature: None if row_idx < missing_count else float(row_idx),
                    "stable": float(row_idx),
                }
            )
    return pl.DataFrame(rows)


def test_missing_shift_scanner_detects_missing_rate_increase() -> None:
    df = _daily_missing_frame(rates=[0.05] * 6 + [0.35] * 6)

    result = MarsMissingShiftScanner().scan(
        df,
        date_col="apply_dt",
        features=["x"],
        feature_data_source={"x": "bureau"},
        min_segment_size=3,
        max_change_points=2,
    )

    assert isinstance(result, MarsMissingShiftResult)
    assert result.detail_table.height >= 1
    row = result.detail_table.sort("anomaly_score", descending=True).row(0, named=True)
    assert row["feature"] == "x"
    assert row["data_source"] == "bureau"
    assert row["direction"] == "increase"
    assert row["abs_delta"] >= 0.25
    assert result.summary_table.get_column("feature").to_list() == ["x"]
    assert result.source_table.get_column("data_source").to_list() == ["bureau"]


def test_missing_shift_scanner_detects_missing_rate_decrease() -> None:
    df = _daily_missing_frame(rates=[0.80] * 6 + [0.20] * 6)

    result = MarsMissingShiftScanner().scan(
        df,
        date_col="apply_dt",
        features=["x"],
        min_segment_size=3,
        max_change_points=2,
    )

    assert result.detail_table.height >= 1
    assert "decrease" in result.detail_table.get_column("direction").to_list()


def test_missing_shift_scanner_detects_low_rate_relative_shift() -> None:
    df = _daily_missing_frame(rates=[0.001] * 6 + [0.02] * 6, rows_per_day=1000)

    result = MarsMissingShiftScanner().scan(
        df,
        date_col="apply_dt",
        features=["x"],
        min_segment_size=3,
        min_abs_delta=0.03,
        min_effect_delta=0.005,
        min_relative_delta=0.30,
        max_change_points=2,
    )

    assert result.detail_table.height >= 1
    row = result.detail_table.sort("anomaly_score", descending=True).row(0, named=True)
    assert row["direction"] == "increase"
    assert row["abs_delta"] < 0.03
    assert "relative_delta" in row["reason"]


def test_missing_shift_scanner_keeps_stable_series_empty() -> None:
    df = _daily_missing_frame(rates=[0.10] * 12)

    result = MarsMissingShiftScanner().scan(
        df,
        date_col="apply_dt",
        features=["x"],
        feature_data_source={"x": "base"},
        min_segment_size=3,
        max_change_points=2,
    )

    assert result.detail_table.is_empty()
    assert result.summary_table.is_empty()
    assert result.source_table.get_column("anomaly_feature_count").to_list() == [0]


def test_missing_shift_scanner_validates_inputs() -> None:
    df = _daily_missing_frame(rates=[0.10] * 4)
    scanner = MarsMissingShiftScanner()

    with pytest.raises(ValueError, match="Date column"):
        scanner.scan(df, date_col="missing_dt", features=["x"])

    with pytest.raises(ValueError, match="Features not found"):
        scanner.scan(df, date_col="apply_dt", features=["missing_feature"])

    with pytest.raises(ValueError, match="min_segment_size"):
        scanner.scan(df, date_col="apply_dt", features=["x"], min_segment_size=1)


def test_missing_shift_result_writes_excel(tmp_path) -> None:
    df = _daily_missing_frame(rates=[0.05] * 6 + [0.35] * 6)
    result = MarsMissingShiftScanner().scan(
        df,
        date_col="apply_dt",
        features=["x"],
        min_segment_size=3,
        max_change_points=2,
    )
    output_path = tmp_path / "missing_shift.xlsx"

    result.write_excel(output_path)

    assert output_path.exists()
