"""容量 harness 的保护、固定维度和独立数值参考，不依赖 mars.agent。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil
import pytest


def _module(name: str) -> Any:
    path = Path(__file__).resolve().parents[1] / "benchmarks" / f"{name}.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("protection", ["timeout", "memory_budget_exceeded"])
def test_scheduler_terminates_worker_and_descendant(tmp_path: Path, protection: str) -> None:
    runner = _module("benchmark_core_capacity")
    child_file = tmp_path / "child.txt"
    # 后代 PID 从实际启动结果取得；保护终止整棵树，并保留已完成阶段诊断。
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import subprocess,sys,time,pathlib,json\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps({'stage':'deliberate_wait','status':'running'}))\n"
        "allocation=bytearray(64*1024**2)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    out = tmp_path / "result.json"
    result = runner._launch(
        [sys.executable, str(worker), str(child_file), str(out)],
        out,
        0.8 if protection == "timeout" else 20,
        10**12 if protection == "timeout" else 40 * 1024**2,
    )
    assert result["status"] == protection
    assert result["exit_status"] == "known"
    assert result["exit_code"] is not None
    assert result["exit_code"] != 0
    if child_file.exists():
        assert not psutil.pid_exists(int(child_file.read_text()))
        assert result["stage"] == "deliberate_wait"


def test_cleanup_reaps_direct_worker_only_with_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _module("benchmark_core_capacity")
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    waited_pids: list[int] = []
    original = psutil.wait_procs

    def capture(processes: list[Any], **kwargs: Any) -> Any:
        waited_pids.extend(item.pid for item in processes)
        return original(processes, **kwargs)

    monkeypatch.setattr(psutil, "wait_procs", capture)
    try:
        runner._stop_tree(process)
        assert process.returncode is not None and process.returncode != 0
        assert process.pid not in waited_pids
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_scheduler_keeps_normal_and_abnormal_real_exit_codes(tmp_path: Path, exit_code: int) -> None:
    runner = _module("benchmark_core_capacity")
    out = tmp_path / "result.json"
    code = (
        "import json,pathlib,sys; "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'stage':'complete','status':'passed'})); "
        "sys.exit(int(sys.argv[2]))"
    )
    result = runner._launch([sys.executable, "-c", code, str(out), str(exit_code)], out, 10, 1024**3)
    assert result["exit_code"] == exit_code
    assert result["exit_status"] == "known"
    assert result["stage"] == "complete"
    assert result["status"] == ("passed" if exit_code == 0 else "failed")


def test_scheduler_cleans_known_descendant_after_parent_naturally_exits(tmp_path: Path) -> None:
    runner = _module("benchmark_core_capacity")
    out = tmp_path / "result.json"
    code = (
        "import json,pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'stage':'complete','status':'passed','child':child.pid})); "
        "time.sleep(0.4)"
    )
    result = runner._launch([sys.executable, "-c", code, str(out)], out, 10, 1024**3)
    assert result["status"] == "passed" and result["exit_code"] == 0
    assert result["cleanup"]["remaining_pids"] == []
    assert result["cleanup"]["descendant_status"] == "stopped"
    child = psutil.Process(result["child"]) if psutil.pid_exists(result["child"]) else None
    assert child is None or not child.is_running() or child.status() == psutil.STATUS_ZOMBIE


def test_cleanup_preserves_unknown_worker_exit_and_wait_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _module("benchmark_core_capacity")

    class UnknownProcess:
        """模拟 OS 回收失败，不伪造正常或固定非零退出码。"""

        pid = 123456789
        returncode = None

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            raise ProcessLookupError("already exited")

        def wait(self, timeout: float) -> None:
            assert timeout <= 5
            raise subprocess.TimeoutExpired("controlled", timeout)

    def missing(pid: int) -> Any:
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", missing)
    process = UnknownProcess()
    cleanup = runner._stop_tree(process)
    assert process.returncode is None
    assert cleanup["worker_exit_status"] == "unknown"
    assert any("worker wait: TimeoutExpired" in reason for reason in cleanup["errors"])


@pytest.mark.parametrize("failure", ["access_denied", "os_error", "suspend_access_denied"])
def test_cleanup_discovery_failure_preserves_unknown_descendants_and_real_exit(
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    runner = _module("benchmark_core_capacity")
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    def denied(parent: psutil.Process, recursive: bool = False) -> list[psutil.Process]:
        assert recursive
        if failure == "access_denied":
            raise psutil.AccessDenied(parent.pid)
        raise OSError("controlled descendant discovery failure")

    def suspend_denied(parent: psutil.Process) -> None:
        raise psutil.AccessDenied(parent.pid)

    if failure == "suspend_access_denied":
        monkeypatch.setattr(psutil.Process, "suspend", suspend_denied)
    else:
        monkeypatch.setattr(psutil.Process, "children", denied)
    try:
        cleanup = runner._stop_tree(process)
        assert process.returncode is not None and process.returncode != 0
        assert cleanup["worker_exit_status"] == "known"
        assert cleanup["descendant_status"] == "unknown"
        assert any(
            reason.startswith(("parent suspend:", "descendant discovery:"))
            for reason in cleanup["errors"]
        )
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_cleanup_discovers_cached_child_descendants_after_parent_exit(tmp_path: Path) -> None:
    runner = _module("benchmark_core_capacity")
    child_file, grandchild_file = tmp_path / "child.txt", tmp_path / "grandchild.txt"
    child_code = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
    )
    parent_code = (
        "import pathlib,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[3],sys.argv[2]])\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
        "while not pathlib.Path(sys.argv[2]).exists(): time.sleep(0.01)\n"
    )
    process = subprocess.Popen([
        sys.executable, "-c", parent_code, str(child_file), str(grandchild_file), child_code,
    ])
    try:
        assert process.wait(timeout=5) == 0
        child_pid, grandchild_pid = int(child_file.read_text()), int(grandchild_file.read_text())
        cleanup = runner._stop_tree(process, [psutil.Process(child_pid)])
        assert process.returncode == 0
        assert {child_pid, grandchild_pid}.issubset(cleanup["descendant_pids"])
        assert cleanup["descendant_status"] == "stopped"
        assert cleanup["remaining_pids"] == []
        for pid in (child_pid, grandchild_pid):
            child = psutil.Process(pid) if psutil.pid_exists(pid) else None
            assert child is None or not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for path in (grandchild_file, child_file):
            if path.exists():
                try:
                    psutil.Process(int(path.read_text())).kill()
                except psutil.NoSuchProcess:
                    pass


def test_cleanup_diagnostic_failure_keeps_completed_worker_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    runner = _module("benchmark_core_capacity")

    def denied(processes: list[Any], timeout: float) -> Any:
        raise psutil.AccessDenied(processes[0].pid)

    monkeypatch.setattr(psutil, "wait_procs", denied)
    out = tmp_path / "result.json"
    code = (
        "import json,pathlib,subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'stage':'deliberate_wait','status':'running'})); "
        "time.sleep(30)"
    )
    result = runner._launch([sys.executable, "-c", code, str(out)], out, 0.8, 1024**3)
    assert result["status"] == "timeout" and result["exit_code"] != 0
    assert result["stage"] == "deliberate_wait"
    assert result["cleanup"]["descendant_status"] == "unknown"
    assert any("AccessDenied" in reason for reason in result["cleanup"]["errors"])


def test_dependency_state_distinguishes_missing_broken_and_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _module("benchmark_core_capacity")
    monkeypatch.delitem(sys.modules, "controlled_optional_module", raising=False)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "0.14")
    assert runner._dependency_state("statsmodels", "controlled_optional_module")["status"] == "not_imported"
    assert runner._dependency_state("statsmodels", "controlled_optional_module", True)["status"] == "import_failed"
    monkeypatch.setitem(sys.modules, "controlled_optional_module", object())
    assert runner._dependency_state("statsmodels", "controlled_optional_module")["status"] == "imported"

    def missing(name: str) -> str:
        raise runner.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runner.importlib.metadata, "version", missing)
    assert runner._dependency_state("statsmodels", "controlled_optional_module", True)["status"] == "not_installed"


def test_linear_diagnostics_record_actual_work_and_reject_optional_branch_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import mars.feature.selection.linear as linear

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    fixtures = _module("benchmark_core_workloads")
    data, features = fixtures._data(400, 8, 42, "pandas", constants=False)
    selector = linear.MarsLinearSelector().fit(data, data["bad"], features=features)
    actual = fixtures._linear_diagnostics(selector)
    if selector._linear_diagnostics_available:
        assert actual["dependency"]["status"] == "imported"
        assert actual["vif"]["status"] == actual["logit"]["status"] == "executed"
        assert actual["vif"]["rows"] > 0 and actual["logit"]["rows"] > 0
    assert actual["stepwise"]["status"] == "skipped"
    monkeypatch.setattr(linear, "optional_import", lambda name: None)
    unavailable = linear.MarsLinearSelector().fit(data, data["bad"], features=features)
    skipped = fixtures._linear_diagnostics(unavailable)
    assert skipped["dependency"]["status"] in {"not_installed", "import_failed"}
    assert skipped["vif"]["status"] == skipped["logit"]["status"] == "skipped"
    assert skipped["vif"]["rows"] == skipped["logit"]["rows"] == 0
    if selector._linear_diagnostics_available:
        runner = _module("benchmark_core_capacity")
        baseline, current = _comparison_fixture(tmp_path), _comparison_fixture(tmp_path)
        for payload, diagnostics in ((baseline, actual), (current, skipped)):
            for round_ in [payload["cases"][0]["warmup"], *payload["cases"][0]["rounds"]]:
                round_["execution_contract"]["diagnostics"] = diagnostics
        runner._compare_baseline(baseline, current)
        assert current["comparison"][0]["status"] == "incomparable"
        assert "time_ratio" not in current["comparison"][0]


def _test_evidence(
    tmp_path: Path, frame: pl.DataFrame | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """使用真实 Report 文件支持容量门单测，证据始终由临时目录清理。"""
    from mars.reporting._result import _result_report

    evidence = _module("_capacity_evidence")
    runner = _module("benchmark_core_capacity")
    if frame is None:
        frame = pl.DataFrame({"feature": ["a", "b"], "correlation": [0.7, -0.3],
                              "woe": [12.0, -13.0], "status": ["computed"] * 2})
    report = _result_report("capacity-test", {"overview": frame},
                            parameters or {"corr_thr": 0.8}, ["a", "b"], None, None)
    saved = evidence._save_evidence(
        report, tmp_path, runner.Measurement(tmp_path / "evidence-meter.json"), "save",
    )
    saved["table_signatures"] = {"overview": runner._signature(report.get_table("overview"))}
    return saved


def _set_test_evidence(round_: dict[str, Any], evidence: dict[str, Any]) -> None:
    """同步真实单轮签名；跨轮仍允许真实浮点舍入差异。"""
    round_["semantic_evidence"] = evidence
    round_["tables"] = evidence["table_signatures"]


def _comparison_fixture(tmp_path: Path) -> dict[str, Any]:
    """构造含真实比较合同的有限轮次，不改动系统可选依赖。"""
    contract = {
        "workload_id": "linear-selection-v1",
        "input_identity": {"kind": "synthetic-fixture", "fixture_version": "test-v1", "seed": 42, "generation": {"rows": 400}, "feature_order": ["a", "b"]},
        "actual_work": {"stage_candidates": [{"stage": "quality", "input": [0, 1], "output": [0, 1]}], "correlation_candidates": [0, 1], "selected": [0, 1], "correlation_status": "computed"},
        "workload": {"rows": 400, "features": 8, "seed": 42},
        "algorithm_parameters": {"corr_method": "spearman", "corr_thr": 0.8},
        "diagnostics": {"vif": {"status": "executed", "rows": 8}},
        "resource_strategy": {"batch_size": 50, "n_jobs": 4},
    }
    evidence = _test_evidence(tmp_path)
    round_ = {
        "status": "passed",
        "stage": "complete",
        "execution_contract": contract,
        "semantic_evidence": evidence,
        "tables": evidence["table_signatures"],
        "environment": {
            "python": "controlled interpreter",
            "platform": "controlled platform",
            "cpu": "controlled cpu",
            "memory_total_bytes": 1024**3,
            "polars_threads": 4,
            "threads_environment": {"POLARS_MAX_THREADS": "4"},
            "participating_dependencies": {
                "statsmodels": {"version": "0.14", "status": "imported"},
            },
            "source_python_sha256": "before",
        },
        "effective_native_threadpools": [],
    }
    return {
        "measurement_contract": {"id": "rss-stages-v1", "interval_seconds": 0.01},
        "parameters": {"timeout": 30, "memory_budget_mib": 1024, "diagnostic_loops": 10},
        "cases": [{
            "case": "linear_selection", "backend": "polars", "status": "passed",
            "workload": {"rows": 400, "features": 8, "batch_size": 50},
            "warmup": deepcopy(round_), "rounds": [round_],
            "summary": {"stages": {"public_compute": {
                "median_seconds": 1.0, "highest_observed_rss_bytes": 1000,
            }}},
        }],
    }


def test_baseline_accepts_same_contract_and_source_provenance_changes(tmp_path: Path) -> None:
    runner = _module("benchmark_core_capacity")
    baseline = _comparison_fixture(tmp_path)
    current = deepcopy(baseline)
    for round_ in [current["cases"][0]["warmup"], *current["cases"][0]["rounds"]]:
        round_["environment"]["source_python_sha256"] = "after"
    runner._compare_baseline(baseline, current)
    assert current["comparison"][0]["status"] == "comparable"
    assert current["comparison"][0]["time_ratio"] == 1.0


@pytest.mark.parametrize("change", [
    "order", "threads", "architecture", "library", "api", "missing_threads",
    "missing_architecture", "added", "removed", "duplicate",
])
def test_native_threadpool_enumeration_order_preserves_resource_contract(
    tmp_path: Path, change: str,
) -> None:
    """重放 Linux 枚举换序；仅顺序可忽略，真实资源及重复库数量继续严格核验。"""
    runner = _module("benchmark_core_capacity")
    baseline = _comparison_fixture(tmp_path)
    pools: list[dict[str, Any]] = [
        {"internal_api": "openblas", "prefix": "libopenblas", "num_threads": 4,
         "architecture": "SkylakeX"},
        {"internal_api": "openmp", "prefix": "libgomp", "num_threads": 4,
         "architecture": None},
        {"internal_api": "openblas", "prefix": "libopenblas", "num_threads": 1,
         "architecture": "Prescott"},
        {"internal_api": "openblas", "prefix": "libscipy_openblas", "num_threads": 4,
         "architecture": "SkylakeX"},
        # 两条相同投影不能由 set/dict 去重，否则会隐藏已加载库的数量变化。
        {"internal_api": "openblas", "prefix": "libopenblas", "num_threads": 4,
         "architecture": "SkylakeX"},
    ]
    for round_ in [baseline["cases"][0]["warmup"], *baseline["cases"][0]["rounds"]]:
        round_["effective_native_threadpools"] = deepcopy(pools)
    current = deepcopy(baseline)
    current["cases"][0]["warmup"]["effective_native_threadpools"] = pools[2:] + pools[:2]
    changed = deepcopy(pools[::-1])
    if change in {"threads", "architecture", "library", "api"}:
        key = {"threads": "num_threads", "architecture": "architecture", "library": "prefix",
               "api": "internal_api"}[change]
        changed[0][key] = 8 if change == "threads" else "different"
    elif change in {"missing_threads", "missing_architecture"}:
        changed[0].pop("num_threads" if change == "missing_threads" else "architecture")
    elif change in {"removed", "duplicate"}:
        changed.pop(0 if change == "duplicate" else 1)
    elif change == "added":
        changed.append({"internal_api": "openmp", "prefix": "libomp", "num_threads": 4})
    current["cases"][0]["rounds"][0]["effective_native_threadpools"] = changed
    before = deepcopy(baseline)
    runner._compare_baseline(baseline, current)
    assert baseline == before
    assert baseline["cases"][0]["rounds"][0]["effective_native_threadpools"] == pools
    entries = current["comparison"]
    assert all(entry["status"] == ("comparable" if change == "order" else "incomparable")
               for entry in entries), entries
    assert all(("time_ratio" in entry) == (change == "order") for entry in entries)
    if change != "order":
        current["parameters"]["comparison_purpose"] = "intentional native resource change"
        runner._compare_baseline(baseline, current)
        assert all(entry["status"] == "resource_strategy_comparison"
                   for entry in current["comparison"])


@pytest.mark.parametrize("change", ["diagnostics", "algorithm", "workload", "measurement", "dependency"])
def test_baseline_rejects_different_effective_work_without_performance_ratios(change: str, tmp_path: Path) -> None:
    runner = _module("benchmark_core_capacity")
    baseline = _comparison_fixture(tmp_path)
    current = deepcopy(baseline)
    for round_ in [current["cases"][0]["warmup"], *current["cases"][0]["rounds"]]:
        contract = round_["execution_contract"]
        if change == "diagnostics":
            contract["diagnostics"]["vif"] = {"status": "skipped", "rows": 0}
        elif change == "algorithm":
            contract["algorithm_parameters"]["corr_method"] = "pearson"
        elif change == "workload":
            contract["workload_id"] = "linear-selection-v2"
        elif change == "dependency":
            round_["environment"]["participating_dependencies"]["statsmodels"]["version"] = "0.15"
    if change == "measurement":
        current["measurement_contract"]["interval_seconds"] = 0.02
    runner._compare_baseline(baseline, current)
    comparison = current["comparison"][0]
    assert comparison["status"] == "incomparable"
    assert comparison["reasons"]
    assert "time_ratio" not in comparison and "review_trigger" not in comparison


def test_baseline_missing_historical_contract_is_explicitly_incomparable(tmp_path: Path) -> None:
    runner = _module("benchmark_core_capacity")
    baseline = _comparison_fixture(tmp_path)
    del baseline["measurement_contract"]
    current = _comparison_fixture(tmp_path)
    runner._compare_baseline(baseline, current)
    assert current["comparison"][0]["status"] == "incomparable"
    assert "measurement" in " ".join(current["comparison"][0]["reasons"])


def test_baseline_refuses_incomplete_cold_consumer_contract_and_changed_dimensions(tmp_path: Path) -> None:
    runner = _module("benchmark_core_capacity")
    baseline, current = _comparison_fixture(tmp_path), _comparison_fixture(tmp_path)
    for payload in (baseline, current):
        for round_ in [payload["cases"][0]["warmup"], *payload["cases"][0]["rounds"]]:
            round_["consumer"] = {"status": "passed"}
    runner._compare_baseline(baseline, current)
    assert current["comparison"][0]["status"] == "incomparable"
    baseline, current = _comparison_fixture(tmp_path), _comparison_fixture(tmp_path)
    current["cases"][0]["workload"]["rows"] = 800
    runner._compare_baseline(baseline, current)
    assert current["comparison"][0]["status"] == "incomparable"
    assert "case.workload.rows" in " ".join(current["comparison"][0]["reasons"])


def test_baseline_distinguishes_execution_failure_and_resource_strategy_comparison(tmp_path: Path) -> None:
    runner = _module("benchmark_core_capacity")
    baseline = _comparison_fixture(tmp_path)
    failed = deepcopy(baseline)
    failed["cases"][0]["status"] = "failed"
    runner._compare_baseline(baseline, failed)
    assert failed["comparison"][0]["status"] == "execution_failed"
    current = deepcopy(baseline)
    for round_ in [current["cases"][0]["warmup"], *current["cases"][0]["rounds"]]:
        round_["execution_contract"]["resource_strategy"]["batch_size"] = 100
    runner._compare_baseline(baseline, current)
    assert current["comparison"][0]["status"] == "incomparable"
    current["parameters"]["comparison_purpose"] = "batch memory strategy acceptance"
    runner._compare_baseline(baseline, current)
    comparison = current["comparison"][0]
    assert comparison["status"] == "resource_strategy_comparison"
    assert comparison["resource_differences"] and comparison["purpose"]


def test_wide_score_fixture_keeps_identical_necessary_columns() -> None:
    fixtures = _module("benchmark_core_workloads")
    narrow, names = fixtures._data(100, 2, 42, "polars", constants=False)
    wide, _ = fixtures._data(100, 52, 42, "polars", constants=False)
    from polars.testing import assert_frame_equal

    assert_frame_equal(narrow, wide.select(narrow.columns))
    assert narrow[names[0]].dtype == pl.Float64
    assert wide["risk_feature_0001"].dtype == pl.Float32
    assert fixtures.workload("profile_columns", "large")["features"] == 3000
    assert fixtures.workload("profile_rows", "large")["rows"] == 1000000


def test_rule_audit_has_real_id_round_grain_and_consistent_counts() -> None:
    fixtures = _module("benchmark_core_workloads")
    result, features, counts = fixtures._rule_fixture(30)
    assert counts["distinct_rule_ids"] == 20
    audit = result.candidate_table
    assert audit.unique(["rule_id", "generation_round"]).height == 30
    assert audit["rule_id"].n_unique() == 20
    assert audit["expression"].n_unique() == 20
    for rule_id in audit["rule_id"].unique():
        rows = result.evaluation.overall_table.filter(
            (pl.col("rule_id") == rule_id)
            & (pl.col("target") == "bad")
            & (pl.col("dataset") == "train")
        )
        hit, miss, total = [
            rows.filter(pl.col("group") == g).row(0, named=True) for g in ("hit", "miss", "total")
        ]
        assert hit["sample_count"] + miss["sample_count"] == total["sample_count"] == 10000
        assert hit["event_count"] + miss["event_count"] == total["event_count"] == 2000
        assert hit["event_rate"] == pytest.approx(hit["event_count"] / hit["sample_count"])
    report = result.to_report(feature_metadata=fixtures._metadata(features))
    assert report.get_table("rule_features").height == counts["expected_bridge_rows"]


def test_signature_checks_schema_order_and_values_without_python_rows() -> None:
    runner = _module("benchmark_core_capacity")
    data = pl.DataFrame({"x": [0.0, np.nan, None, np.inf], "id": [1, 2, 3, 4]})
    signature = runner._signature(data)
    assert signature != runner._signature(data.reverse())
    assert signature != runner._signature(data.with_columns(pl.col("id").cast(pl.Int32)))
    json.dumps(signature, allow_nan=False)


def test_profile_fixture_matches_numpy_reference_and_batches() -> None:
    from mars.analysis import MarsDataProfiler

    fixtures = _module("benchmark_core_workloads")
    data, features = fixtures._data(200, 8, 42, "polars")
    options = {
        "features": features,
        "group_col": "group",
        "enable_sparkline": False,
        "metrics": ["missing", "zeros", "mean", "std"],
    }
    expected = MarsDataProfiler(missing_values=[-999], overview_batch_size=8).generate_profile(
        data, **options
    )
    actual = MarsDataProfiler(missing_values=[-999], overview_batch_size=3).generate_profile(
        data, **options
    )
    from polars.testing import assert_frame_equal

    for name in expected.describe()["tables"]:
        assert_frame_equal(actual.get_table(name), expected.get_table(name))
    values = data[features[0]].to_numpy()
    valid = values[np.isfinite(values) & (values != -999)]
    mean = actual.get_table("stats.mean", features=features[0])["total"][0]
    assert mean == pytest.approx(float(np.mean(valid)), rel=1e-6, abs=1e-6)


def test_context_serializes_only_projected_page(monkeypatch: pytest.MonkeyPatch) -> None:
    import mars.reporting._query as queries
    from mars.reporting import MarsProfileReport

    report = MarsProfileReport(
        pl.DataFrame(
            {
                "feature": [f"x{i}" for i in range(1000)],
                "mean": list(range(1000)),
                "std": [1.0] * 1000,
            }
        ),
        {},
        {},
    )
    original = queries.table_rows
    sizes: list[tuple[int, list[str]]] = []

    def capture(frame: Any) -> Any:
        sizes.append((len(frame), list(frame.columns)))
        return original(frame)

    monkeypatch.setattr(queries, "table_rows", capture)
    report.to_ai_context(
        queries={"overview": {"columns": ["feature", "mean"], "offset": 975, "limit": 20}}
    )
    assert sizes == [(20, ["feature", "mean"])]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_capacity_numeric_oracles_respect_index_and_rollup_rows(backend: str) -> None:
    from mars.analysis import MarsBinEvaluator, MarsDataProfiler, cross_scores

    fixtures = _module("benchmark_core_workloads")
    data, features = fixtures._data(240, 4, 42, backend)
    profile = MarsDataProfiler(missing_values=[-999]).generate_profile(
        data, features=features, metrics=["mean"], enable_sparkline=False
    )
    fixtures._check_statistics("profile", data, profile, features)
    risk = MarsBinEvaluator(binner_params={"n_bins": 4, "special_values": [-999]}).evaluate(
        data,
        features=features,
        target="bad",
        group_col="group",
        weights_col="weight",
        amount_col="amount",
    )
    fixtures._check_statistics("binning", data, risk.report, features)
    cross = cross_scores(
        data,
        score_x=features[0],
        score_y=features[1],
        targets=["bad", "late"],
        score_directions={features[0]: "lower_risk", features[1]: "higher_risk"},
        group_col="group",
        weights_col="weight",
        amount_col="amount",
    )
    fixtures._check_statistics("score_cross", data, cross, features)


def test_rule_explanations_convert_only_selected_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures = _module("benchmark_core_workloads")
    result, _, _ = fixtures._rule_fixture(90)
    captured: list[int] = []
    original = pl.DataFrame.to_dicts

    def record(frame: pl.DataFrame) -> Any:
        if {"dataset", "sample_count", "event_rate"}.issubset(frame.columns):
            captured.append(frame.height)
        return original(frame)

    monkeypatch.setattr(pl.DataFrame, "to_dicts", record)
    report = result.to_report()
    assert captured == [5]
    assert report.get_table("evaluation").height == result.evaluation.overall_table.height


@pytest.mark.parametrize("state", ["no_candidates", "all_rejected", "candidate_unselected"])
def test_empty_rule_states_are_consumable_in_fresh_python(state: str, tmp_path: Path) -> None:
    from mars.rule import MarsRuleSet

    fixtures = _module("benchmark_core_workloads")
    result, features, _ = fixtures._rule_fixture(30)
    candidates = (
        result.candidate_table.clear()
        if state == "no_candidates"
        else (
            result.candidate_table.with_columns(
                pl.lit("rejected" if state == "all_rejected" else "candidate").alias("status")
            )
        )
    )
    evaluation = (
        replace(
            result.evaluation,
            overall_table=result.evaluation.overall_table.clear(),
            slice_table=result.evaluation.slice_table.clear(),
        )
        if state == "no_candidates"
        else result.evaluation
    )
    empty = replace(
        result,
        status="no_rules",
        rule_set=MarsRuleSet([]),
        candidate_table=candidates,
        evaluation=evaluation,
    )
    report = empty.to_report(feature_metadata=fixtures._metadata(features))
    path = tmp_path / "state.marsreport"
    report.save(path)
    code = (
        "import sys,json\nfrom mars.reporting import load_report\n"
        "r=load_report(sys.argv[1])\nassert r.report_id==sys.argv[2]\n"
        "assert r.get_table('summary')['status'][0]=='no_rules'\n"
        "assert r.get_table('candidates').height==int(sys.argv[3])\n"
        "assert r.get_table('rules').is_empty()\n"
        "assert r.describe()['parameters']['analysis_states']['interactions']=='not_computed'\n"
        "p=r.query_page('summary',limit=20)\nassert p['returned_rows']==1\n"
        "assert json.loads(r.to_ai_context(tables=['summary'],max_chars=16000))['evidence'][0]['returned_rows']==1\n"
        "assert 'benchmark_core_workloads' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(path), report.report_id, str(candidates.height)],
        capture_output=True,
        text=True,
        check=True,
    )


def test_incomplete_worker_diagnostics_are_failure_not_normal_empty(tmp_path: Path) -> None:
    runner = _module("benchmark_core_capacity")
    result_path = tmp_path / "partial.json"
    result = runner._launch(
        [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{')",
            str(result_path),
        ],
        result_path,
        10,
        1024**3,
    )
    assert result["status"] == "failed"
    assert result["stage"] == "result_decode"
    assert result["exit_code"] == 0


def test_staged_fixture_preserves_chunks_supervision_and_cross_batch_identity() -> None:
    """容量样本保留真实多块和 Train 标签范围，重复信号跨列批存在。"""
    fixtures = _module("benchmark_core_workloads")
    frame, benchmark, names = fixtures._selection_fixture(801, 35, 42, False)
    assert frame.height == 801 and benchmark.height == 481
    assert all(frame[name].n_chunks() > 1 for name in names)
    assert {frame[name].dtype for name in names} == {pl.Float32, pl.Float64}
    assert frame.filter(pl.col("group") != "Train")["bad"].null_count() == 320
    np.testing.assert_equal(frame[names[3]].to_numpy(), frame[names[-2]].to_numpy())
    np.testing.assert_equal(-frame[names[3]].to_numpy(), frame[names[-1]].to_numpy())


def test_frozen_input_reader_rejects_unmasked_supervision_without_rewriting(tmp_path: Path) -> None:
    """真实复现入口只读冻结文件，错误人口被明确拒绝，不自动补标签掩码。"""
    fixtures = _module("benchmark_core_workloads")
    data = pl.DataFrame({"x": [1.0, 2.0], "bad": [0, 1], "group": ["Train", "Test"]})
    path = tmp_path / "frozen.parquet"
    data.write_parquet(path)
    before = path.read_bytes()
    config = tmp_path / "input.json"
    config.write_text(json.dumps({
        "input_path": path.name, "features": ["x"], "target": "bad", "group_col": "group",
        "supervision_group": "Train",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="non-supervision population"):
        fixtures._read_frozen_inputs(config)
    assert path.read_bytes() == before


def test_measurement_preserves_repeated_stages_and_nested_failure(tmp_path: Path) -> None:
    """重复小批不覆盖峰值，异常继续定位到真实最内层。"""
    runner = _module("benchmark_core_capacity")
    meter = runner.Measurement(tmp_path / "meter.json")
    for _ in range(2):
        meter.run("batch", lambda: 1)
    assert list(meter.result["stages"]) == ["batch", "batch.2"]
    assert meter.result["stages"]["batch"]["process_observation"]["lifetime_peak_rss_bytes"] > 0

    def fail() -> None:
        raise ValueError("controlled")

    with pytest.raises(ValueError, match="controlled"):
        meter.run("public_compute", lambda: meter.run("stability", fail))
    assert meter.result["stage"] == "stability"
    assert meter.result["stages"]["stability"]["status"] == "exception"


@pytest.mark.parametrize("configured", [{}, {"rc_thr": 0.7, "special_values": [-1], "batch_size": 100}])
def test_frozen_selector_does_not_inherit_synthetic_business_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: dict[str, Any],
) -> None:
    """观察真实构造边界，冻结数据省略的业务参数仍使用库默认。"""
    import mars.feature as feature_module

    fixtures = _module("benchmark_core_workloads")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    runner = _module("benchmark_core_capacity")
    frame, benchmark, names = fixtures._selection_fixture(101, 5, 42, False)
    captured: list[dict[str, Any]] = []

    class ConstructionObserved(Exception):
        """在拟合前结束观察，测试不依赖任何数据筛选结果。"""

    def construct(**kwargs: Any) -> None:
        captured.append(kwargs)
        raise ConstructionObserved

    monkeypatch.setattr(feature_module, "MarsStatsSelector", construct)
    monkeypatch.setattr(fixtures, "_read_frozen_inputs", lambda path: (
        frame, benchmark, names, {"selector_parameters": configured}
    ))
    args = argparse.Namespace(
        case="frozen_selection", scale="smoke", seed=42, backend="polars",
        batch_size=16, n_jobs=1, input_config=tmp_path / "unused.json",
    )
    with pytest.raises(ConstructionObserved):
        fixtures._run_staged_analysis(args, runner.Measurement(tmp_path / "frozen.json"))
    assert captured[-1] == {"batch_size": 16, "n_jobs": 1, **configured}
    args.case = "selection_survivors"
    with pytest.raises(ConstructionObserved):
        fixtures._run_staged_analysis(args, runner.Measurement(tmp_path / "synthetic.json"))
    assert captured[-1]["special_values"] == [-999]
    assert captured[-1]["rough_binning_params"] == {"n_bins": 8}
    assert captured[-1]["rc_thr"] is None and captured[-1]["skip_fine_scan"] is True


@pytest.mark.parametrize("case", ["frozen_profile", "frozen_risk", "frozen_selection"])
def test_frozen_cases_execute_one_public_call_in_fresh_worker(tmp_path: Path, case: str) -> None:
    """三个明确入口分别验收统计、风险画像和原生池，不拼接外层分批结果。"""
    fixtures = _module("benchmark_core_workloads")
    frame, benchmark, names = fixtures._selection_fixture(101, 5, 42, False)
    input_path, benchmark_path = tmp_path / "input.parquet", tmp_path / "train.parquet"
    frame.write_parquet(input_path)
    benchmark.write_parquet(benchmark_path)
    input_bytes = input_path.read_bytes()
    config = tmp_path / "input.json"
    config.write_text(json.dumps({
        "input_path": input_path.name, "benchmark_path": benchmark_path.name,
        "features": names, "target": "bad", "group_col": "group", "supervision_group": "Train",
        "profile_call_parameters": {"metrics": ["mean", "psi"], "enable_sparkline": False},
        "risk_parameters": {"binner_params": {"n_bins": 3}},
        "selector_parameters": {"skip_fine_scan": True, "psi_thr": None, "rc_thr": None},
    }), encoding="utf-8")
    runner = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_core_capacity.py"
    output = tmp_path / "result.json"
    completed = subprocess.run([
        sys.executable, str(runner), "--worker", "--case", case,
        "--backend", "polars", "--threads", "1", "--n-jobs", "1", "--batch-size", "2",
        "--input-config", str(config), "--output", str(output),
        "--evidence-dir", str(tmp_path),
    ], capture_output=True, text=True, check=False)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0, result.get("traceback", completed.stderr)
    assert result["status"] == "passed" and result["stage"] == "complete"
    assert result["workload"]["input"]["rows"] == 101
    assert result["workload"]["benchmark"]["rows"] == 61
    assert "public_compute" in result["stages"] and "report_save" in result["stages"]
    assert result["environment"]["polars_threads"] == 1
    assert input_path.read_bytes() == input_bytes


@pytest.mark.parametrize("change", ["input", "benchmark", "feature_order"])
def test_frozen_content_and_full_feature_order_are_comparison_identity(
    tmp_path: Path, change: str,
) -> None:
    """形状/dtype/监督人数不变时，当前数据、基准或完整列序变化仍拒比。"""
    fixtures, runner = _module("benchmark_core_workloads"), _module("benchmark_core_capacity")
    baseline, current = _comparison_fixture(tmp_path), _comparison_fixture(tmp_path)
    frame = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "bad": [0, 1]})
    identities: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for side in ("before", "after"):
        folder = tmp_path / side
        folder.mkdir()
        data = frame.with_columns(pl.col("a") + 1) if side == "after" and change == "input" else frame
        reference = frame.with_columns(pl.col("b") - 1) if side == "after" and change == "benchmark" else frame
        data.write_parquet(folder / "input.parquet")
        reference.write_parquet(folder / "benchmark.parquet")
        config = {"input_path": "input.parquet", "benchmark_path": "benchmark.parquet",
                  "features": ["b", "a"] if side == "after" and change == "feature_order" else ["a", "b"]}
        config_path = folder / "input.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        actual, _, features, read = fixtures._read_frozen_inputs(config_path)
        observations.append(fixtures._frame_observation(actual, features, "bad"))
        identities.append({**read["_input_identity"], "feature_order": features})
    assert observations[0] == observations[1]
    for payload, identity in zip((baseline, current), identities):
        for round_ in [payload["cases"][0]["warmup"], *payload["cases"][0]["rounds"]]:
            round_["execution_contract"]["input_identity"] = identity
    runner._compare_baseline(baseline, current)
    assert current["comparison"][0]["status"] == "incomparable"
    assert "input_identity" in " ".join(current["comparison"][0]["reasons"])
    assert "time_ratio" not in current["comparison"][0]


def test_frozen_reader_rejects_file_change_during_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """读前后重新流式哈希，不复用路径或缓存来掩盖文件真实变化。"""
    fixtures = _module("benchmark_core_workloads")
    path = tmp_path / "input.parquet"
    frame = pl.DataFrame({"a": [1.0, 2.0]})
    frame.write_parquet(path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"input_path": path.name, "features": ["a"]}), encoding="utf-8")
    original = pl.read_parquet

    def changed(*args: Any, **kwargs: Any) -> pl.DataFrame:
        result = original(*args, **kwargs)
        frame.with_columns(pl.col("a") + 1).write_parquet(path)
        return result

    monkeypatch.setattr(pl, "read_parquet", changed)
    with pytest.raises(ValueError, match="changed while being read"):
        fixtures._read_frozen_inputs(config)


def test_identical_frozen_bytes_at_different_paths_remain_comparable(tmp_path: Path) -> None:
    """实际文件路径仅溯源；同一冻结内容的本机副本不被错误拒比。"""
    fixtures, runner = _module("benchmark_core_workloads"), _module("benchmark_core_capacity")
    original, copied = tmp_path / "original", tmp_path / "copied"
    original.mkdir()
    copied.mkdir()
    frame = pl.DataFrame({"a": [1.0, 2.0], "b": [4.0, 3.0]})
    frame.write_parquet(original / "data.parquet")
    shutil.copyfile(original / "data.parquet", copied / "data.parquet")
    baseline, current = _comparison_fixture(tmp_path), _comparison_fixture(tmp_path)
    for payload, folder in ((baseline, original), (current, copied)):
        config = folder / "config.json"
        config.write_text(json.dumps({"input_path": "data.parquet", "features": ["a", "b"]}), encoding="utf-8")
        _, _, features, read = fixtures._read_frozen_inputs(config)
        for round_ in [payload["cases"][0]["warmup"], *payload["cases"][0]["rounds"]]:
            round_["input_files"] = read["_input_files"]
            round_["execution_contract"]["input_identity"] = {
                **read["_input_identity"], "feature_order": features,
            }
    runner._compare_baseline(baseline, current)
    assert current["comparison"][0]["status"] == "comparable"
    assert current["comparison"][0]["time_ratio"] == 1.0


@pytest.mark.parametrize("change", ["attrition", "same_count_identity", "same_count_order", "selected", "status", "warmup", "second_round"])
def test_actual_candidate_routes_and_all_rounds_gate_performance_ratios(
    tmp_path: Path, change: str,
) -> None:
    """80% guard不能替代实际候选去向；预热或任一正式轮漂移都拒比。"""
    runner = _module("benchmark_core_capacity")
    baseline = _comparison_fixture(tmp_path)
    for round_ in [baseline["cases"][0]["warmup"], *baseline["cases"][0]["rounds"]]:
        round_["execution_contract"]["actual_work"] = {
            "stage_candidates": [{"stage": "quality", "input": list(range(200)), "output": list(range(200))}],
            "correlation_candidates": list(range(200)), "selected": list(range(198)), "status": "computed",
        }
    baseline["cases"][0]["rounds"].append(deepcopy(baseline["cases"][0]["rounds"][0]))
    current = deepcopy(baseline)
    item = current["cases"][0]
    target = item["warmup"] if change == "warmup" else item["rounds"][1 if change == "second_round" else 0]
    work = target["execution_contract"]["actual_work"]
    if change == "attrition":
        work.update(correlation_candidates=list(range(161)), selected=list(range(159)))
    elif change == "same_count_identity":
        work["correlation_candidates"][-1] = 201
    elif change == "same_count_order":
        work["correlation_candidates"].reverse()
    elif change == "status":
        work["status"] = "not_computed"
    elif change == "selected":
        work["selected"].reverse()
    else:
        work["stage_candidates"][0]["output"].pop()
    runner._compare_baseline(baseline, current)
    assert current["comparison"][0]["status"] == "incomparable"
    assert "actual_work" in " ".join(current["comparison"][0]["reasons"])
    assert "time_ratio" not in current["comparison"][0]


@pytest.mark.parametrize("change", ["tiny", "woe", "null", "nan", "inf_sign", "identity", "order", "state", "dtype", "column_order", "metadata"])
def test_semantic_sidecar_compares_actual_values_and_special_float_states(
    tmp_path: Path, change: str,
) -> None:
    """允许已知舍入误差，WOE大错、状态及各种特殊值绝不由SHA或容差吞掉。"""
    runner = _module("benchmark_core_capacity")
    baseline = _comparison_fixture(tmp_path)
    data = pl.DataFrame({"feature": ["a", "b", "c", "d", "e", "f"],
                         "coefficient": [0.7, None, float("nan"), float("inf"), -float("inf"), 12.0],
                         "status": ["computed"] * 6})
    evidence = _test_evidence(tmp_path, data)
    for round_ in [baseline["cases"][0]["warmup"], *baseline["cases"][0]["rounds"]]:
        _set_test_evidence(round_, evidence)
    current = deepcopy(baseline)
    values = data["coefficient"].to_list()
    params = None
    if change == "tiny":
        values[0] += 1e-15
    elif change == "woe":
        values[-1] = 0.1
    elif change == "null":
        values[1] = float("nan")
    elif change == "nan":
        values[2] = float("inf")
    elif change == "inf_sign":
        values[3] = -float("inf")
    data = data.with_columns(pl.Series("coefficient", values))
    if change == "identity":
        data = data.with_columns(pl.lit("different").alias("feature"))
    elif change == "order":
        data = data.reverse()
    elif change == "state":
        data = data.with_columns(pl.lit("not_computed").alias("status"))
    elif change == "dtype":
        data = data.with_columns(pl.col("coefficient").cast(pl.Float32))
    elif change == "column_order":
        data = data.select(data.columns[::-1])
    elif change == "metadata":
        params = {"corr_thr": 0.81}
    replacement = _test_evidence(tmp_path, data, params)
    _set_test_evidence(current["cases"][0]["rounds"][0], replacement)
    runner._compare_baseline(baseline, current)
    comparison = current["comparison"][0]
    assert comparison["status"] == ("comparable" if change == "tiny" else "semantic_mismatch")
    assert ("time_ratio" in comparison) == (change == "tiny")


@pytest.mark.parametrize("missing", ["input_identity", "sidecar", "corrupt", "round_evidence", "signature"])
def test_missing_or_corrupt_historical_evidence_never_receives_ratios(
    tmp_path: Path, missing: str,
) -> None:
    runner = _module("benchmark_core_capacity")
    baseline, current = _comparison_fixture(tmp_path), _comparison_fixture(tmp_path)
    round_ = baseline["cases"][0]["warmup"]
    if missing == "input_identity":
        del round_["execution_contract"]["input_identity"]
    elif missing == "round_evidence":
        del baseline["cases"][0]["rounds"][0]["semantic_evidence"]
    elif missing == "signature":
        round_["tables"]["overview"]["sha256"] = "changed-json-only"
    else:
        path = Path(round_["semantic_evidence"]["path"])
        if missing == "sidecar":
            path.unlink()
        else:
            with path.open("ab") as stream:
                stream.write(b"corrupt")
    runner._compare_baseline(baseline, current)
    comparison = current["comparison"][0]
    assert comparison["status"] == "incomparable" and comparison["reasons"]
    assert "time_ratio" not in comparison and "rss_ratio" not in comparison


@pytest.mark.parametrize("side,index", [("baseline", 0), ("baseline", 2), ("current", 0), ("current", 2)])
def test_semantic_validation_checks_each_warmup_and_measured_round(
    tmp_path: Path, side: str, index: int,
) -> None:
    """每一侧的预热和后续正式轮都能触发真实表数值不一致的拒绝。"""
    runner = _module("benchmark_core_capacity")
    baseline = _comparison_fixture(tmp_path)
    baseline["cases"][0]["rounds"].append(deepcopy(baseline["cases"][0]["rounds"][0]))
    current = deepcopy(baseline)
    payload = baseline if side == "baseline" else current
    rounds = [payload["cases"][0]["warmup"], *payload["cases"][0]["rounds"]]
    changed = pl.DataFrame({"feature": ["a", "b"], "correlation": [0.7, -0.3],
                            "woe": [0.1, -13.0], "status": ["computed"] * 2})
    _set_test_evidence(rounds[index], _test_evidence(tmp_path, changed))
    runner._compare_baseline(baseline, current)
    assert current["comparison"][0]["status"] == "semantic_mismatch"
    assert "time_ratio" not in current["comparison"][0]


@pytest.mark.parametrize("change", ["tiny_coefficient", "threshold", "action", "trigger", "order", "protection", "implementation"])
def test_decision_events_are_strict_while_coefficients_and_known_implementation_can_vary(
    tmp_path: Path, change: str,
) -> None:
    """系数容差不重做业务判定；动作/事件顺序/阈值/保护必须精确相同。"""
    from mars.reporting._result import _result_report

    runner, evidence_module = _module("benchmark_core_capacity"), _module("_capacity_evidence")
    baseline, current = _comparison_fixture(tmp_path), _comparison_fixture(tmp_path)
    base = pl.DataFrame({"event_order": [0, 1], "feature": ["a", "b"],
                         "trigger_feature": ["b", "a"], "action": ["keep", "drop"],
                         "correlation": [0.9, 0.9], "threshold": [0.8, 0.8],
                         "white_list_protected": [True, False]})
    modified = base
    parameters: dict[str, Any] = {"corr_thr": 0.8, "status": "computed"}
    if change == "tiny_coefficient":
        modified = base.with_columns(pl.col("correlation") + 1e-15)
    elif change == "threshold":
        modified = base.with_columns(pl.col("threshold") + 1e-15)
    elif change == "action":
        modified = base.with_columns(pl.lit("keep").alias("action"))
    elif change == "trigger":
        modified = base.with_columns(pl.lit("c").alias("trigger_feature"))
    elif change == "order":
        modified = base.reverse()
    elif change == "protection":
        modified = base.with_columns(~pl.col("white_list_protected"))
    elif change == "implementation":
        parameters.update(algorithm="anchored Chan block central moments Float64",
                          row_block_size=4096, feature_batch_size=3, threshold_refined_pairs=0)
    for payload, frame, params in ((baseline, base, {"corr_thr": 0.8, "status": "computed"}),
                                   (current, modified, parameters)):
        report = _result_report("correlation", {"correlation_decisions": frame}, params, ["a", "b"], None, None)
        saved = evidence_module._save_evidence(report, tmp_path, runner.Measurement(tmp_path / "decisions.json"), "save")
        saved["table_signatures"] = {"correlation_decisions": runner._signature(frame)}
        for round_ in [payload["cases"][0]["warmup"], *payload["cases"][0]["rounds"]]:
            _set_test_evidence(round_, saved)
    runner._compare_baseline(baseline, current)
    allowed = change in {"tiny_coefficient", "implementation"}
    assert current["comparison"][0]["status"] == ("comparable" if allowed else "semantic_mismatch")
    assert ("time_ratio" in current["comparison"][0]) == allowed


@pytest.mark.parametrize("case", ["selection_survivors", "frozen_profile"])
def test_fresh_worker_staged_baseline_has_retained_evidence_and_real_roundtrip(
    tmp_path: Path, case: str,
) -> None:
    """实际调度预热和正式轮，保存JSON后由下一次命令读取sidecar完成比较。"""
    runner = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_core_capacity.py"
    baseline_path, current_path = tmp_path / "baseline.json", tmp_path / "current.json"
    command = [sys.executable, str(runner), "--case", case, "--scale", "smoke",
               "--backend", "polars", "--threads", "4", "--repeats", "1", "--n-jobs", "1",
               "--batch-size", "3", "--evidence-dir", str(tmp_path), "--timeout", "90"]
    if case == "frozen_profile":
        frame, _, features = _module("benchmark_core_workloads")._selection_fixture(101, 5, 42, False)
        frame.write_parquet(tmp_path / "input.parquet")
        config = tmp_path / "input.json"
        config.write_text(json.dumps({
            "input_path": "input.parquet", "features": features,
            "profile_call_parameters": {"metrics": ["mean", "zeros"], "enable_sparkline": False},
        }), encoding="utf-8")
        command.extend(["--input-config", str(config)])
    for output, extra in ((baseline_path, []), (current_path, ["--baseline-json", str(baseline_path)])):
        completed = subprocess.run([*command, "--output", str(output), *extra],
                                   capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    assert current["comparison"] and all(
        item["status"] == "comparable" for item in current["comparison"]
    ), json.dumps(current["comparison"], ensure_ascii=False, indent=2)
    paths: set[str] = set()
    for payload in (baseline, current):
        item = payload["cases"][0]
        rounds = [item["warmup"], *item["rounds"]]
        assert len({round_["worker_process_observation"]["pid"] for round_ in rounds}) == len(rounds)
        for round_ in rounds:
            assert round_["exit_code"] == 0 and round_["exit_status"] == "known"
            assert round_["environment"]["polars_threads"] == 4
            if case == "selection_survivors":
                assert round_["execution_contract"]["actual_work"]["correlation_candidates"]
            assert Path(round_["semantic_evidence"]["path"]).is_file()
            assert round_["semantic_evidence"]["path"] not in paths
            paths.add(round_["semantic_evidence"]["path"])
            stages = round_["stages"]
            assert {"public_compute", "report_save", "report_integrity_identity"}.issubset(stages)
