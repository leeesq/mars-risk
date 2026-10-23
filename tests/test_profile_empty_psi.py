"""空当前人口的 PSI 校验、身份、未计算状态与持久展示合同。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.analysis import profile_stats
from mars.compute import to_pandas_frame
from mars.feature import MarsNativeBinner
from mars.reporting import load_report


def _dimensions(reference: str, benchmark: pl.DataFrame) -> dict[str, Any]:
    """让 benchmark 仅包含特征，验证角色列不属于基准结构要求。"""
    options: dict[str, Any] = {}
    if "group" in reference:
        options["group_col"] = "g"
    if "time" in reference:
        options.update(time_col="day", time_grain="day")
    if "benchmark" in reference:
        options["benchmark_df"] = benchmark
    return options


@pytest.mark.parametrize(
    "reference", ["none", "group", "time", "benchmark", "group_benchmark", "time_benchmark"],
)
@pytest.mark.parametrize("mixed_metrics", [False, True])
@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64, pl.String, pl.Categorical, pl.Null])
def test_empty_current_psi_validates_without_fitting_and_keeps_status(
    reference: str, mixed_metrics: bool, dtype: pl.DataType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """合法基准包括全 null；空当前不拟合、不 rechunk，具体 PSI 状态不能漏行。"""
    frame = pl.DataFrame(schema={"x": dtype, "g": pl.String, "day": pl.Datetime})
    benchmark = pl.DataFrame({"x": [None, None]}, schema={"x": dtype})
    before = frame.clone()
    chunks = [frame[column].n_chunks() for column in frame.columns]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Empty current PSI must validate schema without fitting or rechunking")

    monkeypatch.setattr(MarsNativeBinner, "fit", forbidden)
    monkeypatch.setattr(pl.DataFrame, "rechunk", forbidden)
    metadata = {"x": {"display_name": "申请金额", "data_source": "申请表", "unit": "CNY"}}
    report = profile_stats(
        frame, features=["x"], metrics=["psi", "mean", "zeros"] if mixed_metrics else ["psi"],
        feature_metadata=metadata, **_dimensions(reference, benchmark),
    )
    psi = report.get_table("stats.psi")
    assert psi.columns == ["feature", "display_name", "dtype", "data_source", "total"]
    assert psi.schema["total"] == pl.Float64
    assert psi.row(0, named=True) == {
        "feature": "x", "display_name": "申请金额", "dtype": str(frame.schema["x"]),
        "data_source": "申请表", "total": None,
    }
    description = report.describe()
    assert description["feature_metadata"]["x"] == metadata["x"]
    assert description["tables"]["stats.psi"]["fields"]["total"]["unit"] == "dimensionless"
    assert description["source"]["kind"] == "analysis"
    assert description["calculation_state"]["table"] == "calculation_status"
    state = report.get_table("calculation_status", filters={"metric": "psi"})
    reason = "no_group_or_benchmark" if reference == "none" else "empty_current_population"
    assert state.select("feature", "group", "metric", "status", "reason").to_dicts() == [{
        "feature": "x", "group": "Total", "metric": "psi", "status": "not_computed",
        "reason": reason,
    }]
    if mixed_metrics:
        mean_state = report.get_table("calculation_status", filters={"metric": "mean"})
        assert mean_state.height == 1
        assert mean_state["status"][0] == ("undefined" if dtype in {pl.Float32, pl.Float64} else "skipped")
    assert report.report_meta["diagnostics"] == []
    assert_frame_equal(frame, before)
    assert [frame[column].n_chunks() for column in frame.columns] == chunks


@pytest.mark.parametrize("reference", ["benchmark", "group_benchmark", "time_benchmark"])
@pytest.mark.parametrize("metrics", [["psi"], ["psi", "mean", "zeros"]])
@pytest.mark.parametrize(
    ("benchmark", "error"),
    [
        (pl.DataFrame({"other": [1, 2]}), "missing active PSI features"),
        (pl.DataFrame({"x": ["a", "b"]}), "incompatible dtypes"),
        (pl.DataFrame(schema={"x": pl.Float32}), "at least one row"),
    ],
)
def test_empty_and_nonempty_current_reject_same_invalid_benchmark(
    reference: str, metrics: list[str], benchmark: pl.DataFrame, error: str,
) -> None:
    frame = pl.DataFrame({
        "x": pl.Series([1.0, 2.0], dtype=pl.Float32), "g": ["a", "b"],
        "day": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
    })
    messages: list[str] = []
    for current in (frame, frame.head(0)):
        with pytest.raises(ValueError, match=error) as caught:
            profile_stats(current, features=["x"], metrics=metrics, **_dimensions(reference, benchmark))
        messages.append(str(caught.value))
    assert messages[0] == messages[1]


def test_empty_psi_zero_candidates_and_benchmark_scope_keep_existing_contract() -> None:
    current = pl.DataFrame(schema={"g": pl.String, "day": pl.Datetime, "x": pl.Float32})
    report = profile_stats(current.select("g"), features=["g"], metrics=["psi"], group_col="g")
    assert report.get_table("stats.psi").height == 0
    assert "calculation_status" not in report.describe()["tables"]
    with pytest.raises(ValueError, match="at least one active feature"):
        profile_stats(pl.DataFrame({"g": ["a"]}), features=["g"], metrics=["psi"], group_col="g")
    scoped = profile_stats(
        current, features=["x", "day", "g"], metrics=["psi"], group_col="g", time_col="day",
        benchmark_df=pl.DataFrame({"x": [1.0, 2.0]}),
    )
    assert scoped.get_table("stats.psi")["feature"].to_list() == ["x"]
    assert scoped.get_table("calculation_status")["feature"].to_list() == ["x"]


@pytest.mark.parametrize("empty", [False, True])
def test_psi_current_and_reference_chunks_are_preserved(empty: bool) -> None:
    """多 chunk 当前/基准与同值连续输入逐表相同，空人口早返不擦除既有诊断合同。"""
    full = pl.DataFrame({
        "x32": pl.Series([0.0, 1.0, None, 2.0, 0.0, 3.0], dtype=pl.Float32),
        "x64": [1.0, 2.0, 0.0, None, 3.0, 4.0],
        "category": ["a", "b", "a", None, "b", "a"],
        "null_num": pl.Series([None] * 6, dtype=pl.Float64),
        "g": ["a", "a", "b", "b", "a", "b"],
    })
    frame = pl.concat([full.slice(0, 2), full.slice(2, 3), full.slice(5, 1)], rechunk=False)
    current = frame.head(0) if empty else frame
    features = ["x32", "x64", "category", "null_num"]
    benchmark = frame.select(features)
    before_current, before_benchmark = current.clone(), benchmark.clone()
    chunks = {
        name: [[len(part) for part in data[column].get_chunks()] for column in data.columns]
        for name, data in (("current", current), ("benchmark", benchmark))
    }
    assert benchmark["x32"].n_chunks() == 3
    kwargs: dict[str, Any] = {"features": features, "metrics": ["psi", "mean", "zeros"], "group_col": "g"}
    actual = profile_stats(current, benchmark_df=benchmark, **kwargs)
    expected = profile_stats(current.rechunk(), benchmark_df=benchmark.rechunk(), **kwargs)
    assert list(actual.describe()["tables"]) == list(expected.describe()["tables"])
    for name in actual.describe()["tables"]:
        assert_frame_equal(actual.get_table(name), expected.get_table(name))
    if empty:
        assert actual.report_meta["diagnostics"] == []
    else:
        assert actual.report_meta["diagnostics"] == [{
            "component": "psi", "features": ["null_num"],
            "reason": "All values are missing or special.",
        }]
        failed = actual.get_table("calculation_status", features="null_num", filters={"metric": "psi"})
        assert failed["status"].to_list() == ["failed"]
    assert_frame_equal(current, before_current)
    assert_frame_equal(benchmark, before_benchmark)
    assert chunks == {
        name: [[len(part) for part in data[column].get_chunks()] for column in data.columns]
        for name, data in (("current", current), ("benchmark", benchmark))
    }


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_empty_psi_state_is_queryable_displayed_and_loaded_in_fresh_process(
    backend: str, tmp_path: Path,
) -> None:
    frame = pl.DataFrame(schema={"x": pl.Float32, "g": pl.String})
    report = profile_stats(
        frame if backend == "polars" else frame.to_pandas(),
        features=["x"], metrics=["psi", "mean", "zeros"], group_col="g",
        benchmark_df=pl.DataFrame({"x": [0.0, 1.0]}),
        feature_metadata={"x": {"display_name": "额度", "data_source": "申请", "unit": "CNY"}},
    )
    status = to_pandas_frame(report.get_table("calculation_status"))
    page = report.query_page("calculation_status", filters={"metric": "psi"}, limit=1)
    assert page["total_rows"] == page["returned_rows"] == 1
    assert page["next_offset"] is None
    ai = json.loads(report.to_ai_context(tables=["calculation_status"], filters={"metric": "psi"}))
    assert ai["evidence"][0]["rows"][0]["reason"] == "empty_current_population"
    assert "empty_current_population" in report._repr_html_()
    for columns in (None, ["total"]):
        styled = report.show_trend("psi", columns=columns)
        assert "not_computed/empty_current_population" in styled.to_html()
    report.write_html(str(tmp_path / "profile.html"))
    assert "empty_current_population" in (tmp_path / "profile.html").read_text(encoding="utf-8")
    path = tmp_path / "profile.marsreport"
    report.save(path, overwrite=True)
    restored = load_report(path)
    assert restored.describe() == report.describe()
    assert to_pandas_frame(restored.get_table("calculation_status")).equals(status)
    code = """
import sys
from pathlib import Path
import polars as pl
from mars.compute import to_pandas_frame
from mars.reporting import load_report
report = load_report(sys.argv[1])
psi = to_pandas_frame(report.get_table('stats.psi'))
assert psi['total'].isna().all()
assert psi['dtype'].tolist() == ['Float32']
state = to_pandas_frame(report.get_table('calculation_status', filters={'metric': 'psi'}))
assert state['status'].tolist() == ['not_computed']
assert state['reason'].tolist() == ['empty_current_population']
assert report.query_page('calculation_status', filters={'metric': 'psi'})['total_rows'] == 1
assert 'empty_current_population' in report.to_ai_context(tables=['calculation_status'])
report.write_html(sys.argv[2])
assert 'empty_current_population' in Path(sys.argv[2]).read_text(encoding='utf-8')
if isinstance(report.get_table('stats.psi'), pl.DataFrame):
    assert report.get_table('stats.psi').schema['total'] == pl.Float64
print('empty PSI snapshot and status verified')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), str(tmp_path / "restored.html")],
        env={
            **os.environ, "POLARS_MAX_THREADS": "4",
            "PYTHONPATH": os.pathsep.join([
                str(Path(__file__).resolve().parents[1] / "src"), os.environ.get("PYTHONPATH", ""),
            ]),
        },
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "snapshot and status verified" in result.stdout
