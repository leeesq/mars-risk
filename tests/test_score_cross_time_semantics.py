"""验证周期来源元数据与原统计、保存快照及离线证据保持一致。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars.analysis import (
    ScoreCrossReport,
    cross_scores,
    evaluate_score_policy,
    write_score_cross_html,
)
from mars.reporting import ReportSnapshot, load_report


def _report(*, empty: bool = False, **kwargs: Any) -> ScoreCrossReport:
    """显式分段避免拟合差异，只检验真实分组来源。"""
    frame: pl.DataFrame = pl.DataFrame(
        {
            "x": [0.1, 0.8, 0.1, 0.8],
            "y": [0.2, 0.2, 0.9, 0.9],
            "bad": [0, 1, None, 0],
            "sample": ["Total", "Total", "holdout", "holdout"],
            "single_date": ["2026-01-14"] * 4,
            "date": ["2026-01-14"] * 2 + ["2026-02-22"] * 2,
        }
    )
    return cross_scores(
        frame.clear() if empty else frame,
        score_x="x",
        score_y="y",
        score_directions={"x": "higher_risk", "y": "lower_risk"},
        targets=["bad"],
        cutpoints={"x": [0.5], "y": [0.5]},
        **kwargs,
    )


def _payload(path: Path) -> dict[str, Any]:
    """读取实际导出的证据 JSON，浏览器展示另由浏览器用例验收。"""
    match = re.search(
        r'<script type="application/json" id="data">(.*?)</script>',
        path.read_text(encoding="utf-8"),
        re.S,
    )
    assert match is not None
    payload: dict[str, Any] = json.loads(match.group(1))
    return payload


@pytest.mark.parametrize(
    ("options", "group_source", "group_column", "period_column", "grain", "periods"),
    [
        ({}, "none", None, None, None, {"Total"}),
        ({"group_col": "sample"}, "group", "sample", None, None, {"Total"}),
        (
            {"time_col": "single_date"}, "none", None, "single_date", "month",
            {"202601"},
        ),
        (
            {"time_col": "date"}, "none", None, "date", "month",
            {"202601", "202602"},
        ),
        (
            {"group_col": "sample", "time_col": "date", "time_grain": "day"},
            "group", "sample", "date", "day", {date(2026, 1, 14), date(2026, 2, 22)},
        ),
    ],
)
def test_scope_dimensions_record_actual_independent_sources(
    options: dict[str, Any],
    group_source: str,
    group_column: str | None,
    period_column: str | None,
    grain: str | None,
    periods: set[str | date],
) -> None:
    report = _report(**options)
    parameters = report.describe()["parameters"]
    assert parameters["scope_dimensions"] == {
        "group": {"source": group_source, "column": group_column, "grain": None},
        "period": {
            "source": "time" if period_column is not None else "none",
            "column": period_column,
            "grain": grain,
        },
    }
    overall = report.get_table("overall")
    assert set(overall["period"]) == periods
    assert overall["sample_count"].sum() == 4
    assert overall["observed_sample_count"].sum() == 3
    assert overall["bad_sample_count"].sum() == 1
    if group_column is not None:
        assert set(overall["group"]) == {"Total", "holdout"}
    policy = evaluate_score_policy(report, {"type": "expression", "expression": "X <= X1"})
    assert policy.describe()["parameters"]["scope_dimensions"] == parameters["scope_dimensions"]


@pytest.mark.parametrize("period_label", ["Total", None])
@pytest.mark.parametrize("with_time", [False, True])
def test_snapshot_period_labels_do_not_define_time_semantics(
    tmp_path: Path, period_label: str | None, with_time: bool
) -> None:
    report = _report(time_col="single_date") if with_time else _report()
    description = report.describe()
    # 兼容快照可使用任意周期标签；只替换测试夹具标签，不改变生成器的统计契约。
    tables: dict[str, pl.DataFrame] = {
        name: report.get_table(name).with_columns(
            pl.lit(period_label, dtype=pl.String).alias("period")
        )
        if "period" in report.get_table(name).columns
        else report.get_table(name)
        for name in description["tables"]
    }
    snapshot = ReportSnapshot(tables, description)
    archive = tmp_path / "labels.marsreport"
    snapshot.save(archive)
    restored = load_report(archive)
    expected_source = "time" if with_time else "none"
    assert (
        restored.describe()["parameters"]["scope_dimensions"]["period"]["source"]
        == expected_source
    )
    assert restored.get_table("overall")["period"].to_list() == [period_label]
    for name, table in tables.items():
        assert_frame_equal(restored.get_table(name), table)
    html = tmp_path / "labels.html"
    write_score_cross_html(restored, html)
    payload = _payload(html)
    assert (
        payload["description"]["parameters"]["scope_dimensions"]
        == description["parameters"]["scope_dimensions"]
    )
    assert payload["tables"]["overall"] == tables["overall"].to_dicts()


@pytest.mark.parametrize("options", [{}, {"time_col": "date"}])
def test_empty_input_preserves_declared_source_without_inventing_scopes(
    tmp_path: Path, options: dict[str, Any]
) -> None:
    report = _report(empty=True, **options)
    assert report.get_table("cells").height == 0
    assert report.get_table("overall").height == 0
    assert report.describe()["parameters"]["actual_scope_count"] == 0
    expected_source = "time" if options else "none"
    assert (
        report.describe()["parameters"]["scope_dimensions"]["period"]["source"]
        == expected_source
    )
    html = tmp_path / "empty.html"
    write_score_cross_html(report, html)
    assert _payload(html)["tables"]["overall"] == []


@pytest.mark.parametrize(
    "missing_keys",
    [("scope_dimensions",), ("scope_dimensions", "time_col", "time_grain")],
)
def test_legacy_snapshot_keeps_unknown_metadata_and_original_evidence(
    tmp_path: Path, missing_keys: tuple[str, ...]
) -> None:
    report = _report(time_col="date", group_col="sample")
    description = report.describe()
    for key in missing_keys:
        description["parameters"].pop(key, None)
    legacy = ReportSnapshot(
        {name: report.get_table(name) for name in description["tables"]}, description
    )
    archive = tmp_path / "legacy.marsreport"
    legacy.save(archive)
    restored = load_report(archive)
    assert "scope_dimensions" not in restored.describe()["parameters"]
    for name in description["tables"]:
        assert_frame_equal(restored.get_table(name), report.get_table(name))
    assert (
        restored.query_page("cells", filters={"group": "Total"})["reference"]
        == legacy.query_page("cells", filters={"group": "Total"})["reference"]
    )
    html = tmp_path / "legacy.html"
    write_score_cross_html(restored, html)
    payload = _payload(html)
    assert payload["description"]["parameters"] == description["parameters"]
    assert payload["tables"]["overall"] == report.get_table("overall").to_dicts()


def test_scope_dimensions_survive_new_process_load_query_export_and_evidence(tmp_path: Path) -> None:
    report = _report(group_col="sample", time_col="date")
    archive = tmp_path / "cross.marsreport"
    html = tmp_path / "cross.html"
    report.save(archive)
    script = (
        "import json,sys\n"
        "from mars.reporting import load_report\n"
        "from mars.analysis import write_score_cross_html\n"
        "report=load_report(sys.argv[1])\n"
        "write_score_cross_html(report,sys.argv[2])\n"
        "page=report.query_page('overall',filters={'group':'Total'},limit=10)\n"
        "print(json.dumps({'dimensions':report.describe()['parameters']['scope_dimensions'],"
        "'rows':page['data'].to_dicts(),'reference':page['reference']}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(archive), str(html)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    actual = json.loads(completed.stdout)
    page = report.query_page("overall", filters={"group": "Total"}, limit=10)
    assert actual["dimensions"] == report.describe()["parameters"]["scope_dimensions"]
    assert actual["rows"] == page["data"].to_dicts()
    assert actual["reference"] == page["reference"]
    payload = _payload(html)
    assert payload["description"]["report_id"] == report.report_id
    assert payload["tables"]["cells"] == report.get_table("cells").to_dicts()
