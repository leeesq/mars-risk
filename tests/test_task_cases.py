"""七个任务案例的 fixture 接口、证据复核、快照消费和可移植下载回归。"""

from __future__ import annotations

import html
import json
import math
import os
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from home_credit_case_fixture import write_case_fixture
from mars.reporting import load_report

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/snippets/task_cases.py"
CHECKER = ROOT / "scripts/check_case_assets.py"
PAGES = {"data-quality": "profile", "binning-stability": "binning", "selection-correlation": "selection",
         "score-cross": "score_cross", "rule-evidence": "rules", "saved-reports": "restore", "report-delivery": "delivery"}


@pytest.fixture(scope="module")
def lightweight_cases(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """显式1800行小fixture只验证接口，不冒充全量公开结果。"""
    root = tmp_path_factory.mktemp("home-credit-case-fixture")
    data = write_case_fixture(root / "raw")
    output = root / "cases"
    execution = subprocess.run([sys.executable, str(SCRIPT), "--case", "all", "--fixture",
        "--data-dir", str(data), "--target-group-rows", "200", "--output-dir", str(output)],
        check=False, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert execution.returncode == 0, execution.stdout + "\n" + execution.stderr
    return output


def test_seven_cases_execute_and_replay_same_batch_evidence(lightweight_cases: Path) -> None:
    summary = json.loads((lightweight_cases / "summary.json").read_text(encoding="utf-8"))
    reports = {report.report_id: report for path in lightweight_cases.glob("*.marsreport") if (report := load_report(path))}
    assert summary["rows"] == 1800 and set(summary["cases"]) == {str(i) for i in range(1, 8)}
    for number in range(1, 8):
        case = json.loads((lightweight_cases / f"case-{number}.json").read_text(encoding="utf-8"))
        assert case["report_id"] in reports
        for query in case["queries"]:
            ref = query["reference"]
            assert reports[ref["report_id"]].get_table(ref["table"], **ref["query"]).to_dicts() == query["rows"]
    preparation = json.loads((lightweight_cases / "preparation-summary.json").read_text(encoding="utf-8"))
    assert preparation["full_dataset"] is False
    assert preparation["partition_stats"][-1]["observed_count"] == 0


@pytest.mark.parametrize("probability", [False, True])
@pytest.mark.parametrize("bin_index", [0, 1, 2])
def test_normal_bin_dsl_matches_public_cross_assignment(
    probability: bool, bin_index: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """数学边界fixture仍用合成值测试缺失、特殊值、非有限与右闭精度。"""
    import polars as pl

    from mars.analysis import cross_scores, get_score_bin_definitions
    from mars.rule import MarsRule, MarsRuleSet

    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    condition = runpy.run_path(str(SCRIPT.parent / "external_agent_rule_case.py"))["_condition"]
    values = [None, float("nan"), -999.0, -777.0, float("-inf"), float("inf"), -0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1]
    data = pl.DataFrame({"sample_id": [str(i) for i in range(len(values))], "main_score": [0.5] * len(values),
                         "aux_score": values, "bad30": [0] * len(values)})
    initial = cross_scores(data, score_x="main_score", score_y="aux_score", targets=["bad30"],
        score_directions={"main_score": "lower_risk", "aux_score": "higher_risk"},
        cutpoints={"main_score": [0.25, 0.75], "aux_score": [0.25, 0.75]},
        special_values={"aux_score": [-999.0, 0.5]}, probability_scores=["aux_score"] if probability else None)
    definitions = get_score_bin_definitions(initial)
    definitions["y"]["missing_values"] = [-777.0]
    cross = cross_scores(data, score_x="main_score", score_y="aux_score", targets=["bad30"],
        score_directions={"main_score": "lower_risk", "aux_score": "higher_risk"},
        bin_definitions=definitions, group_col="sample_id")
    expected = {r["group"] for r in cross.get_table("cells", filters={"y_bin": f"b{bin_index}"}).to_dicts() if r["sample_count"]}
    rule = MarsRule(condition("aux_score", definitions["y"], bin_index))
    actual = MarsRuleSet([rule]).transform(data)
    assert set(actual.filter(pl.col(f"rule__{rule.rule_id}") == 1)["sample_id"]) == expected


def test_downloaded_zip_runs_outside_a_checkout_without_csv(lightweight_cases: Path) -> None:
    """独立下载目录只读快照；提供三文件路径后才能重算。"""
    with TemporaryDirectory(prefix="mars-case-standalone-", dir=ROOT.parent) as directory:
        workspace = Path(directory).resolve()
        assert ROOT not in workspace.parents
        unpacked = workspace / "downloaded"
        with zipfile.ZipFile(lightweight_cases / "cases.zip") as archive:
            assert not any(name.endswith((".csv", ".parquet")) for name in archive.namelist())
            assert "home_credit_data.py" in archive.namelist()
            archive.extractall(unpacked)
        result = subprocess.run([sys.executable, str(unpacked / "task_cases.py"), "--phase", "consume",
            "--output-dir", str(unpacked)], cwd=workspace, capture_output=True, text=True,
            encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}, check=False)
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        output = workspace / "recomputed"
        raw = lightweight_cases.parent / "raw"
        generated = subprocess.run([sys.executable, str(unpacked / "task_cases.py"), "--case", "1", "--fixture",
            "--train-csv", str(raw / "application_train.csv"), "--test-csv", str(raw / "application_test.csv"),
            "--columns-description-csv", str(raw / "HomeCredit_columns_description.csv"),
            "--target-group-rows", "200", "--output-dir", str(output)], cwd=workspace,
            capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, check=False)
        assert generated.returncode == 0, generated.stdout + "\n" + generated.stderr
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["source_commit"] is None and "unknown" in manifest["source_status"]
        assert manifest["data_config"]["rows"] == 1800


def test_automatic_rule_case_has_separate_bounded_generator_evidence(lightweight_cases: Path) -> None:
    automatic = json.loads((lightweight_cases / "rule-generation.json").read_text(encoding="utf-8"))
    assert automatic["report_id"] != load_report(lightweight_cases / "rules.marsreport").report_id
    summary = automatic["queries"][0]["rows"][0]
    assert 0 < summary["candidate_count"] <= automatic["candidate_budget"]
    assert automatic["raw_input_rows"] == automatic["eligible_rows"] + automatic["excluded_rows"]
    assert all("combination" in row["sources"].lower() for row in automatic["queries"][1]["rows"])


def test_binning_has_native_charts_metadata_and_exact_png_dpi(lightweight_cases: Path) -> None:
    content = (lightweight_cases / "binning.html").read_text(encoding="utf-8")
    assert '<img ' in content and 'id="semantics-section"' in content
    metadata = load_report(lightweight_cases / "binning.marsreport").describe()["feature_metadata"]["EXT_SOURCE_2"]
    for name in ("display_name", "data_source", "unit"):
        assert html.escape(str(metadata[name])) in content
    from PIL import Image

    with Image.open(lightweight_cases / "binning-native-home-credit.png") as image:
        assert image.info["dpi"][0] >= 299
    assert (lightweight_cases / "binning-native-home-credit.svg").exists()


def test_score_cross_denominators_and_real_correlation(lightweight_cases: Path) -> None:
    cross = load_report(lightweight_cases / "score-cross.marsreport")
    cells = cross.get_table("cells").to_dicts()
    for scope in cross.get_table("overall").to_dicts():
        members = [row for row in cells if all(row[key] == scope[key] for key in ("target", "group", "period"))]
        for field in ("sample_count", "observed_sample_count", "bad_sample_count"):
            assert sum(row[field] for row in members) == scope[field]
        if scope["group"] == "4_oot_undue":
            assert scope["observed_sample_count"] == 0 and scope["bad_rate"] is None
    for row in cells:
        if row["observed_sample_count"]:
            assert math.isclose(row["bad_rate"], row["bad_sample_count"] / row["observed_sample_count"])
    raw = load_report(lightweight_cases / "raw-correlation.marsreport")
    woe = load_report(lightweight_cases / "woe-correlation.marsreport")
    assert raw.describe()["parameters"]["representation"] == "raw"
    assert woe.describe()["parameters"]["representation"] == "woe"
    assert raw.get_table("pairs").height > 0


def test_consumer_never_prepares_or_recomputes(lightweight_cases: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    consumer = runpy.run_path(str(SCRIPT))["_consume"]
    snapshots: dict[str, bytes] = {path.name: path.read_bytes() for path in lightweight_cases.glob("*.marsreport")}

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("消费阶段禁止访问 CSV 或重新计算")

    for name in ("prepare_home_credit_data", "produce", "cross_scores", "profile_risk", "profile_stats"):
        monkeypatch.setitem(consumer.__globals__, name, forbidden)
    consumer(lightweight_cases)
    result = json.loads((lightweight_cases / "case-6.json").read_text(encoding="utf-8"))
    assert result["findings"]["policy_replay_matches"]
    assert result["findings"]["context_chars"] <= result["findings"]["max_chars"]
    assert all((lightweight_cases / name).read_bytes() == content for name, content in snapshots.items())
    correlation = (lightweight_cases / "correlation.html").read_text(encoding="utf-8")
    assert correlation.startswith("<!doctype html>") and 'name="viewport"' in correlation


def test_current_public_pages_keep_existing_urls_and_shared_snippets() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    index = (ROOT / "docs/demos/index.md").read_text(encoding="utf-8")
    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for number, (page_name, marker) in enumerate(PAGES.items(), start=1):
        page = (ROOT / f"docs/demos/{page_name}.md").read_text(encoding="utf-8")
        assert f"{page_name}.md" in index and f"demos/{page_name}.md" in navigation
        assert f"[start:{marker}]" in source and f"task_cases.py:{marker}" in page
        assert f"case-{number}.json" in page and f"previews.txt:case{number}" in page
    for capability in ("数据画像", "分箱评估", "特征筛选", "相关性分析", "模型分交叉", "规则挖掘"):
        assert capability in index


def test_semantic_check_allows_fresh_identity_but_detects_changed_numbers() -> None:
    checker = runpy.run_path(str(CHECKER))
    semantic, compare = checker["_semantic"], checker["_assert_equal"]
    published: dict[str, Any] = {"report_id": "a", "created_at": "2026-10-01", "query": {"rows": [{"count": 36, "bad_rate": 0.25}]}}
    current: dict[str, Any] = {"report_id": "b", "created_at": "2026-10-02", "query": {"rows": [{"count": 36, "bad_rate": 0.25}]}}
    compare(semantic(published), semantic(current))
    current["query"]["rows"][0]["bad_rate"] = 0.3
    with pytest.raises(ValueError, match=r"query.rows\[0\].bad_rate"):
        compare(semantic(published), semantic(current))


def test_docs_ci_validates_published_snapshots_without_private_csv() -> None:
    workflow = (ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    assert "tests/test_task_cases.py" in workflow
    assert "scripts/check_case_assets.py --mode public --site-dir site" in workflow
    assert "--skip-recompute" not in workflow
    assert "docs/assets/cases/** -text" in (ROOT / ".gitattributes").read_text(encoding="utf-8")
