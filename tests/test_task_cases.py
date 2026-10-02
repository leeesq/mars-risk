"""七个任务案例的轻量 API、同源证据、快照消费和公开资源回归。"""

from __future__ import annotations

import html
import json
import math
import runpy
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from mars.reporting import load_report

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/snippets/task_cases.py"
CHECKER = ROOT / "scripts/check_case_assets.py"
CASE_SEED = 20261002
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
            "--seed", str(CASE_SEED),
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
    assert summary["seed"] == CASE_SEED
    assert set(summary["cases"]) == {str(number) for number in range(1, 8)}
    assert all(case["report_id"] in snapshots for case in summary["cases"].values())
    for report in snapshots.values():
        context = report.describe()["business_context"]
        assert context["dataset_id"] == f"synthetic-consumer-credit-{CASE_SEED}"


def test_normal_cell_rules_keep_actual_member_denominators(lightweight_cases: Path) -> None:
    """规则候选复用正常格时，非法分值不能偷偷增加命中人数或事件人数。"""
    checker = runpy.run_path(str(CHECKER))
    checker["_validate_rule_memberships"](lightweight_cases, {"rows": 900, "seed": CASE_SEED})


@pytest.mark.parametrize("probability", [False, True])
@pytest.mark.parametrize("bin_index", [0, 1, 2])
def test_normal_bin_dsl_matches_public_cross_assignment(
    probability: bool, bin_index: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """逐成员对照公共交叉赋箱，覆盖 infinity、缺失、特殊值、右闭边界与概率域。"""
    import polars as pl

    from mars.analysis import cross_scores, get_score_bin_definitions
    from mars.rule import MarsRule, MarsRuleSet

    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    condition = runpy.run_path(str(SCRIPT.parent / "external_agent_rule_case.py"))["_condition"]
    values = [None, float("nan"), -999.0, -777.0, float("-inf"), float("inf"),
              -0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1]
    data = pl.DataFrame({"sample_id": [str(index) for index in range(len(values))],
                         "main_score": [0.5] * len(values), "aux_score": values,
                         "bad30": [0] * len(values)})
    initial = cross_scores(
        data, score_x="main_score", score_y="aux_score", targets=["bad30"],
        score_directions={"main_score": "lower_risk", "aux_score": "higher_risk"},
        cutpoints={"main_score": [0.25, 0.75], "aux_score": [0.25, 0.75]},
        special_values={"aux_score": [-999.0, 0.5]},
        probability_scores=["aux_score"] if probability else None,
    )
    definitions = get_score_bin_definitions(initial)
    definitions["y"]["missing_values"] = [-777.0]
    cross = cross_scores(
        data, score_x="main_score", score_y="aux_score", targets=["bad30"],
        score_directions={"main_score": "lower_risk", "aux_score": "higher_risk"},
        bin_definitions=definitions, group_col="sample_id",
    )
    cells = cross.get_table("cells", filters={"y_bin": f"b{bin_index}"}).to_dicts()
    expected = {row["group"] for row in cells if row["sample_count"]}
    rule = MarsRule(condition("aux_score", definitions["y"], bin_index))
    applied = MarsRuleSet(rules=[rule]).transform(data)
    actual = set(applied.filter(pl.col(f"rule__{rule.rule_id}") == 1)["sample_id"].to_list())
    assert actual == expected


def test_downloaded_zip_runs_outside_a_checkout(lightweight_cases: Path) -> None:
    """完整下载包在独立目录可生成新案例，再用已有可信快照独立消费。"""
    # Docs CI 的 --basetemp 在仓库内；兄弟临时目录确保真正脱离受测 checkout。
    with TemporaryDirectory(prefix="mars-case-standalone-", dir=ROOT.parent) as directory:
        workspace = Path(directory).resolve()
        assert ROOT not in workspace.parents
        unpacked = workspace / "downloaded"
        with zipfile.ZipFile(lightweight_cases / "cases.zip") as archive:
            archive.extractall(unpacked)
        output = workspace / "new-analysis"
        generated = subprocess.run(
            [sys.executable, str(unpacked / "task_cases.py"), "--case", "1", "--rows", "90",
             "--seed", "20261003", "--output-dir", str(output)],
            cwd=workspace, capture_output=True, text=True, check=False,
        )
        assert generated.returncode == 0, generated.stdout + "\n" + generated.stderr
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["rows"] == 90 and summary["seed"] == 20261003
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["source_commit"] is None
        assert "unknown" in manifest["source_status"]
        assert manifest["code_fingerprints"]
        consumed = subprocess.run(
            [sys.executable, str(unpacked / "task_cases.py"), "--phase", "consume",
             "--output-dir", str(unpacked)],
            cwd=workspace, capture_output=True, text=True, check=False,
        )
        assert consumed.returncode == 0, consumed.stdout + "\n" + consumed.stderr
        restore = json.loads((unpacked / "case-6.json").read_text(encoding="utf-8"))
        assert restore["findings"]["policy_replay_matches"]


def test_automatic_rule_case_has_separate_bounded_generator_evidence(lightweight_cases: Path) -> None:
    """自动发现有独立结果与候选预算，不把手工格子种子包装成生成器。"""
    automatic = json.loads((lightweight_cases / "rule-generation.json").read_text(encoding="utf-8"))
    manual = load_report(lightweight_cases / "rules.marsreport")
    generated = load_report(lightweight_cases / "auto-rules.marsreport")
    assert automatic["report_id"] == generated.report_id != manual.report_id
    summary = automatic["queries"][0]["rows"][0]
    assert 0 < summary["candidate_count"] <= automatic["candidate_budget"]
    assert automatic["raw_input_rows"] == automatic["eligible_rows"] + automatic["excluded_rows"]
    assert automatic["excluded_rows"] > 0
    sources = [row["sources"] for row in automatic["queries"][1]["rows"]]
    assert sources and all("combination" in source.lower() for source in sources)


def test_case_binning_html_keeps_native_charts_and_business_metadata(lightweight_cases: Path) -> None:
    """共享入口实际输出原生图和独立元数据导航，而非只在文档里宣称存在。"""
    content = (lightweight_cases / "binning.html").read_text(encoding="utf-8")
    report = load_report(lightweight_cases / "binning.marsreport")
    assert '<img ' in content
    assert 'id="semantics-section"' in content
    assert 'data-mars-view="semantics"' in content and 'data-page="semantics"' in content
    assert 'mars-semantic-table' in content
    assert 'tabindex="0"' in content and ' evidence table"' in content
    for name in ("display_name", "data_source", "unit"):
        assert name in content
    metadata = report.describe()["feature_metadata"]["main_score"]
    for name in ("display_name", "data_source", "unit"):
        assert html.escape(str(metadata[name])) in content


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
        "parameters": {"polars_version": "1.42.0", "format_version": 1},
        "query": {"rows": [{"count": 36, "bad_rate": 0.25, "status": "ok"}]},
    }
    current: dict[str, Any] = {
        "report_id": "batch-b", "created_at": "2026-10-02",
        "parameters": {"polars_version": "1.44.2", "format_version": 1},
        "query": {"rows": [{"count": 36, "bad_rate": 0.25, "status": "ok"}]},
    }
    compare(semantic(published), semantic(current))
    current["parameters"]["format_version"] = 2
    with pytest.raises(ValueError, match=r"parameters.format_version"):
        compare(semantic(published), semantic(current))
    current["parameters"]["format_version"] = 1
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
