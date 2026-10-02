"""七个任务案例的轻量 API、同源证据、快照消费和公开资源回归。"""

from __future__ import annotations

import json
import math
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mars.reporting import load_report

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/snippets/task_cases.py"
CHECKER = ROOT / "scripts/check_case_assets.py"
PAGES = {
    "data-quality": "profile",
    "binning-stability": "binning",
    "selection-correlation": "selection",
    "score-cross": "score_cross",
    "rule-evidence": "rules",
    "saved-reports": "restore",
    "report-delivery": "delivery",
}


@pytest.fixture(scope="module")
def lightweight_cases(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """900 行只验证接口与语义，不冒充公开 18,000 行数值。"""
    output = tmp_path_factory.mktemp("task-cases-light")
    execution: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--case", "all", "--rows", "900",
            "--output-dir", str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert execution.returncode == 0, execution.stdout + "\n" + execution.stderr
    return output


def test_seven_cases_execute_and_replay_same_batch_evidence(lightweight_cases: Path) -> None:
    checker = runpy.run_path(str(CHECKER))
    snapshots = checker["_validate_evidence"](lightweight_cases)
    assert len(snapshots) >= 5
    summary: dict[str, Any] = json.loads((lightweight_cases / "summary.json").read_text(encoding="utf-8"))
    assert summary["rows"] == 900
    assert summary["seed"] == 20261001
    assert set(summary["cases"]) == {str(number) for number in range(1, 8)}
    assert all(case["report_id"] in snapshots for case in summary["cases"].values())


def test_case_assets_have_real_manifest_and_download_contents(lightweight_cases: Path) -> None:
    checker = runpy.run_path(str(CHECKER))
    manifest = checker["_validate_manifest"](lightweight_cases, None)
    assert manifest["data_config"]["rows"] == 900
    assert manifest["code_fingerprints"]
    assert manifest["dependencies"]


def test_score_cross_denominators_states_and_signed_redundancy(lightweight_cases: Path) -> None:
    report = load_report(lightweight_cases / "score-cross.marsreport")
    overall = report.get_table("overall").to_dicts()
    cells = report.get_table("cells").to_dicts()
    for scope in overall:
        members = [
            row for row in cells
            if all(row[field] == scope[field] for field in ("target", "group", "period"))
        ]
        for field in ("sample_count", "observed_sample_count", "bad_sample_count"):
            assert sum(row[field] for row in members) == scope[field]
        for field in ("weight_sum", "observed_weight_sum", "bad_weight_sum", "tot_amt"):
            assert math.isclose(sum(row[field] for row in members), scope[field], rel_tol=1e-9)
        if scope["group"] == "observation" and scope["target"] == "late60":
            assert scope["observed_sample_count"] == 0
    assert {"empty", "low_sample", "unobserved"}.issubset(row["status"] for row in cells)
    for row in cells:
        if row["observed_weight_sum"]:
            assert math.isclose(row["bad_rate"], row["bad_weight_sum"] / row["observed_weight_sum"])
    raw = load_report(lightweight_cases / "raw-correlation.marsreport")
    woe = load_report(lightweight_cases / "woe-correlation.marsreport")
    assert raw.describe()["parameters"]["representation"] == "raw"
    assert woe.describe()["parameters"]["representation"] == "woe"
    pairs = raw.get_table("pairs").to_dicts()
    negative = next(row for row in pairs if {row["feature_a"], row["feature_b"]} == {"main_score", "score_inverse"})
    positive = next(row for row in pairs if {row["feature_a"], row["feature_b"]} == {"income", "income_copy"})
    assert negative["correlation"] == -1 and positive["correlation"] == 1
    assert negative["abs_correlation"] == positive["abs_correlation"] == 1


def test_consumer_loads_existing_reports_without_wide_table_reanalysis(
    lightweight_cases: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    namespace = runpy.run_path(str(SCRIPT))
    consumer = namespace["_consume"]

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Snapshot consumer must not regenerate wide tables or rerun analysis")

    for name in ("_data", "_sample", "produce", "cross_scores", "profile_risk", "profile_stats"):
        monkeypatch.setitem(consumer.__globals__, name, forbidden)
    for name in ("score-cross.marsreport", "policy.marsreport", "case-4.json"):
        shutil.copyfile(lightweight_cases / name, tmp_path / name)
    consumer(tmp_path)
    result: dict[str, Any] = json.loads((tmp_path / "case-6.json").read_text(encoding="utf-8"))
    assert result["findings"]["snapshot_type"] == "ReportSnapshot"
    assert result["queries"][1]["reference"]["query"]["offset"] == result["queries"][0]["next_offset"]
    assert result["queries"][2]["total_rows"] == 0
    assert result["findings"]["invalid_request"]["exception"] == "ValueError"
    assert result["findings"]["context_chars"] <= result["findings"]["max_chars"]
    assert result["findings"]["budget_omissions"]


def test_current_public_pages_use_shared_code_and_real_evidence() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    index = (ROOT / "docs/demos/index.md").read_text(encoding="utf-8")
    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for number, (page_name, marker) in enumerate(PAGES.items(), start=1):
        page = (ROOT / f"docs/demos/{page_name}.md").read_text(encoding="utf-8")
        assert f"{page_name}.md" in index
        assert f"demos/{page_name}.md" in navigation
        assert f"[start:{marker}]" in source
        assert f"task_cases.py:{marker}" in page
        assert f"case-{number}.json" in page
        assert f"previews.txt:case{number}" in page
        assert f"previews.txt:card{number}" in index
        assert "=== " in page
    for capability in ("数据画像", "分箱评估", "特征筛选", "相关性分析", "模型分交叉", "规则挖掘"):
        assert capability in index
    assert "selection-correlation.md#correlation" in index
    assert (ROOT / "docs/demos/correlation_and_score_cross.ipynb").is_file()
    assert (ROOT / "docs/demos/lgb-modeling-monitoring.ipynb").is_file()


def test_semantic_check_allows_fresh_identity_but_detects_changed_numbers() -> None:
    checker = runpy.run_path(str(CHECKER))
    semantic = checker["_semantic"]
    compare = checker["_assert_equal"]
    published: dict[str, Any] = {
        "report_id": "batch-a", "created_at": "2026-10-01",
        "query": {"rows": [{"count": 36, "bad_rate": 0.25, "status": "ok"}]},
    }
    current: dict[str, Any] = {
        "report_id": "batch-b", "created_at": "2026-10-02",
        "query": {"rows": [{"count": 36, "bad_rate": 0.25, "status": "ok"}]},
    }
    compare(semantic(published), semantic(current))
    current["query"]["rows"][0]["bad_rate"] = 0.3
    with pytest.raises(ValueError, match=r"query.rows\[0\].bad_rate"):
        compare(semantic(published), semantic(current))


def test_docs_ci_recomputes_public_scale_and_checks_built_downloads() -> None:
    workflow = (ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    assert "tests/test_task_cases.py" in workflow
    assert "scripts/check_case_assets.py --site-dir site" in workflow
    assert "--skip-recompute" not in workflow
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "docs/assets/cases/** -text" in attributes
