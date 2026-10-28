"""客群身份与公共证据查询的独立小型回归。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from mars.feature._scopes import build_scopes, resolve_scopes, scope_filter
from mars.reporting import MarsBinningReport, load_report


def test_scope_identity_missing_and_literal_names() -> None:
    data = pl.DataFrame({"segment": ["All", "Total", "Missing", "null", "", None, "NaN"]})
    scopes = build_scopes(data, "segment")
    assert scopes.height == 8
    assert scopes["scope_id"].n_unique() == 8
    assert build_scopes(data.reverse(), "segment").equals(scopes)
    for value in data["segment"]:
        identity = resolve_scopes(scopes, [value], False)[0]
        assert scope_filter(data, scopes, identity).height == 1
    assert resolve_scopes(scopes) == ["all"]
    assert resolve_scopes(scopes, [], False) == []
    with pytest.raises(ValueError, match="Unknown segment"):
        resolve_scopes(scopes, ["absent"], False)


def test_scope_numeric_and_reference_validation() -> None:
    frame = pl.DataFrame({"s": [1.0, None, float("nan")]})
    scopes = build_scopes(frame, "s")
    assert scopes.height == 3
    assert scope_filter(frame, scopes, resolve_scopes(scopes, [None], False)[0]).height == 2
    with pytest.raises(ValueError, match="infinite"):
        build_scopes(pl.DataFrame({"s": [float("inf")]}), "s")
    with pytest.raises(ValueError, match="dtype"):
        build_scopes(frame, "s", pl.DataFrame({"s": [1]}))
    with pytest.raises(ValueError, match="missing segment_col"):
        build_scopes(frame, "s", pl.DataFrame({"other": [1]}))


def test_scope_report_save_fresh_process_query_and_reference(tmp_path: Path) -> None:
    scopes = build_scopes(pl.DataFrame({"segment": ["seg1", "seg2", "seg3"]}), "segment")
    rows = [
        {"scope_id": identity, "feature": f, "iv": float(i)}
        for identity in scopes["scope_id"] for i, f in enumerate(["a", "b", "c"])
    ]
    report = MarsBinningReport(pl.DataFrame(rows), {}, pl.DataFrame())
    report._scopes = scopes
    page = report.query_page("summary", segments=["seg2"], include_all=False, limit=1)
    assert page["total_rows"] == 3
    assert page["next_offset"] == 1
    assert "segments" not in page["reference"]["query"]
    assert "scope_id" in page["reference"]["query"]["filters"]
    assert report.get_table("summary", **page["reference"]["query"]).equals(page["data"])
    assert report.get_table("summary", segments=["seg2"], include_all=False,
                            filters={"feature": "b"}, columns=["iv"]).columns == ["scope_id", "iv"]
    context = json.loads(report.to_ai_context(tables=["summary"], segments=["seg2"],
                                             include_all=False, max_chars=16000, limit=1))
    assert context["evidence"][0]["rows"][0]["scope_id"] != "all"
    identity = page["data"]["scope_id"][0]
    exact = json.loads(report.to_ai_context(tables=["summary"], filters={"scope_id": identity},
                                           include_all=False, max_chars=16000, limit=1))
    assert exact["evidence"][0]["total_rows"] == 3
    assert exact["evidence"][0]["rows"][0]["scope_id"] == identity
    path = tmp_path / "scoped.marsreport"
    report.save(path)
    restored = load_report(path)
    assert restored.get_table("summary", **page["reference"]["query"]).equals(page["data"])
    code = """
import sys
from mars.reporting import load_report
r = load_report(sys.argv[1])
p = r.query_page('summary', segments=['seg2'], include_all=False, limit=1)
assert p['total_rows'] == 3 and p['next_offset'] == 1
assert p['data'].equals(r.get_table('summary', **p['reference']['query']))
r.write_excel(sys.argv[2])
r.write_html(sys.argv[3])
print('fresh-process evidence replay passed')
"""
    result = subprocess.run([sys.executable, "-c", code, str(path), str(tmp_path / "out.xlsx"),
                             str(tmp_path / "out.html")], capture_output=True, text=True, check=True)
    assert "passed" in result.stdout
