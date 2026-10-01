"""容量 harness 的保护、固定维度和独立数值参考，不依赖 mars.agent。"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
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
    assert result["exit_code"] != 0
    if child_file.exists():
        assert not psutil.pid_exists(int(child_file.read_text()))
        assert result["stage"] == "deliberate_wait"


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
