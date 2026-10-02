"""容量 harness 的保护、固定维度和独立数值参考，不依赖 mars.agent。"""

from __future__ import annotations

import importlib.util
import json
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
    monkeypatch: pytest.MonkeyPatch,
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
        baseline, current = _comparison_fixture(), _comparison_fixture()
        for payload, diagnostics in ((baseline, actual), (current, skipped)):
            for round_ in [payload["cases"][0]["warmup"], *payload["cases"][0]["rounds"]]:
                round_["execution_contract"]["diagnostics"] = diagnostics
        runner._compare_baseline(baseline, current)
        assert current["comparison"][0]["status"] == "incomparable"
        assert "time_ratio" not in current["comparison"][0]


def _comparison_fixture() -> dict[str, Any]:
    """构造含真实比较合同的有限轮次，不改动系统可选依赖。"""
    contract = {
        "workload_id": "linear-selection-v1",
        "workload": {"rows": 400, "features": 8, "seed": 42},
        "algorithm_parameters": {"corr_method": "spearman", "corr_thr": 0.8},
        "diagnostics": {"vif": {"status": "executed", "rows": 8}},
        "resource_strategy": {"batch_size": 50, "n_jobs": 4},
    }
    round_ = {
        "status": "passed",
        "stage": "complete",
        "execution_contract": contract,
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


def test_baseline_accepts_same_contract_and_source_provenance_changes() -> None:
    runner = _module("benchmark_core_capacity")
    baseline = _comparison_fixture()
    current = deepcopy(baseline)
    for round_ in [current["cases"][0]["warmup"], *current["cases"][0]["rounds"]]:
        round_["environment"]["source_python_sha256"] = "after"
    runner._compare_baseline(baseline, current)
    assert current["comparison"][0]["status"] == "comparable"
    assert current["comparison"][0]["time_ratio"] == 1.0


@pytest.mark.parametrize("change", ["diagnostics", "algorithm", "workload", "measurement", "dependency"])
def test_baseline_rejects_different_effective_work_without_performance_ratios(change: str) -> None:
    runner = _module("benchmark_core_capacity")
    baseline = _comparison_fixture()
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


def test_baseline_missing_historical_contract_is_explicitly_incomparable() -> None:
    runner = _module("benchmark_core_capacity")
    baseline = _comparison_fixture()
    del baseline["measurement_contract"]
    current = _comparison_fixture()
    runner._compare_baseline(baseline, current)
    assert current["comparison"][0]["status"] == "incomparable"
    assert "measurement" in " ".join(current["comparison"][0]["reasons"])


def test_baseline_refuses_incomplete_cold_consumer_contract_and_changed_dimensions() -> None:
    runner = _module("benchmark_core_capacity")
    baseline, current = _comparison_fixture(), _comparison_fixture()
    for payload in (baseline, current):
        for round_ in [payload["cases"][0]["warmup"], *payload["cases"][0]["rounds"]]:
            round_["consumer"] = {"status": "passed"}
    runner._compare_baseline(baseline, current)
    assert current["comparison"][0]["status"] == "incomparable"
    baseline, current = _comparison_fixture(), _comparison_fixture()
    current["cases"][0]["workload"]["rows"] = 800
    runner._compare_baseline(baseline, current)
    assert current["comparison"][0]["status"] == "incomparable"
    assert "case.workload.rows" in " ".join(current["comparison"][0]["reasons"])


def test_baseline_distinguishes_execution_failure_and_resource_strategy_comparison() -> None:
    runner = _module("benchmark_core_capacity")
    baseline = _comparison_fixture()
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
