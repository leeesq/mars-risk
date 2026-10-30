"""分客群 benchmark 的源码选择和子进程结果边界。"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "benchmark_segmented_analysis.py"
CASE = "segments-1-balanced-shared"
VALID_RESULT = {"status": "passed", "input": {"identity": {"sha256": "same"}},
                "all_evidence": {"risk_summary": [], "stats_overview": [], "selected_features": []},
                "seconds": 1.0, "retained_report_count": 2}


def _runner() -> Any:
    spec = importlib.util.spec_from_file_location("segmented_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(output: Path) -> list[str]:
    return [str(SCRIPT), "--rows", "200", "--features", "2", "--segments", "1",
            "--distributions", "balanced", "--binning-scopes", "shared",
            "--batch-size", "2", "--output", str(output)]


def _summary(output: Path) -> dict[str, Any]:
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("exit_code", "payload", "expected"),
    [(0, json.dumps(VALID_RESULT), "passed"),
     (0, '{"status":"passed"}', "failed"),
     (0, json.dumps({**VALID_RESULT, "input": {"rows": 1}}), "failed"),
     (0, json.dumps({**VALID_RESULT, "all_evidence": {"risk_summary": [], "stats_overview": []}}), "failed"),
     (7, '{"status":"passed", "exit_code":0}', "failed"),
     (7, None, "failed"), (0, None, "failed"),
     (0, '{broken', "failed"), (0, '[]', "failed"),
     (0, '{"status":"running"}', "failed")],
)
def test_worker_result_must_be_new_valid_and_successful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int,
    payload: str | None, expected: str,
) -> None:
    runner = _runner()
    output = tmp_path / "summary.json"
    old = tmp_path / f"{CASE}.json"
    old.write_text('{"status":"passed"}', encoding="utf-8")

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        case_path = Path(command[command.index("--output") + 1])
        if payload is not None:
            case_path.write_text(payload, encoding="utf-8")
        report_dir = case_path.with_suffix("")
        report_dir.mkdir(exist_ok=True)
        (report_dir / "report-0.marsreport").write_text("report", encoding="utf-8")
        (report_dir / "risk-rules.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, exit_code)

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", _args(output))
    code = runner._main()
    summary = _summary(output)
    case = summary["cases"][0]
    assert code == (0 if expected == "passed" else 1)
    assert summary["status"] == case["status"] == expected
    assert case["exit_code"] == exit_code
    assert Path(case["log"]).exists()
    if expected == "failed":
        assert case["failure_reason"]


def test_multi_case_failure_and_old_report_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    output = tmp_path / "summary.json"
    report_dir = tmp_path / CASE
    report_dir.mkdir()
    (report_dir / "report-9.marsreport").write_text("old", encoding="utf-8")
    (report_dir / "unrelated.txt").write_text("keep", encoding="utf-8")

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        case_path = Path(command[command.index("--output") + 1])
        case_path.write_text(json.dumps(VALID_RESULT), encoding="utf-8")
        artifacts = case_path.with_suffix("")
        artifacts.mkdir(exist_ok=True)
        (artifacts / "report-0.marsreport").write_text("new", encoding="utf-8")
        (artifacts / "risk-rules.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 7 if "segments-2" in case_path.name else 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    args = _args(output)
    args[args.index("--segments") + 1:args.index("--segments") + 2] = ["1", "2"]
    monkeypatch.setattr(sys, "argv", args)
    assert runner._main() == 1
    result = _summary(output)
    assert result["status"] == "failed"
    assert [case["status"] for case in result["cases"]] == ["passed", "failed"]
    assert not (report_dir / "report-9.marsreport").exists()
    assert (report_dir / "unrelated.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("bad_path", ["missing", "empty", "file"])
def test_invalid_source_root_fails_before_worker(tmp_path: Path, bad_path: str) -> None:
    source = tmp_path / bad_path
    if bad_path == "empty":
        source.mkdir()
    if bad_path == "file":
        source.write_text("file", encoding="utf-8")
    completed = subprocess.run([sys.executable, *_args(tmp_path / "summary.json"),
                                "--source-root", str(source)], capture_output=True, text=True)
    assert completed.returncode != 0
    assert not (tmp_path / "summary.json").exists()


def test_real_case_and_import_failure_do_not_reuse_result(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    first = subprocess.run([sys.executable, *_args(output)], env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    success = _summary(output)["cases"][0]
    assert success["status"] == "passed"
    assert Path(success["mars_module"]).resolve().parent == (ROOT / "src" / "mars").resolve()

    source = tmp_path / "broken source"
    package = source / "src" / "mars"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('raise ImportError("controlled benchmark worker import failure")', encoding="utf-8")
    second = subprocess.run([sys.executable, *_args(output), "--source-root", str(source)],
                            env=env, capture_output=True, text=True)
    assert second.returncode != 0
    result = _summary(output)
    case = result["cases"][0]
    assert result["status"] == case["status"] == "failed"
    assert case["exit_code"] != 0
    assert "controlled benchmark worker import failure" in Path(case["log"]).read_text(encoding="utf-8")


def test_external_source_with_spaces(tmp_path: Path) -> None:
    source = tmp_path / "external source"
    shutil.copytree(ROOT / "src" / "mars", source / "src" / "mars")
    output = tmp_path / "summary.json"
    completed = subprocess.run([sys.executable, *_args(output), "--source-root", str(source)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    case = _summary(output)["cases"][0]
    assert Path(case["mars_module"]).resolve().parent == (source / "src" / "mars").resolve()



def test_worker_rejects_import_location_outside_requested_source(tmp_path: Path) -> None:
    source = tmp_path / "source with space"
    package = source / "src" / "mars"
    package.mkdir(parents=True)
    other = tmp_path / "other" / "mars" / "__init__.py"
    (package / "__init__.py").write_text(f"__file__ = {str(other)!r}\n", encoding="utf-8")
    output = tmp_path / "summary.json"
    completed = subprocess.run([sys.executable, *_args(output), "--source-root", str(source)],
                               capture_output=True, text=True)
    assert completed.returncode != 0
    case = _summary(output)["cases"][0]
    assert case["status"] == "failed" and case["exit_code"] != 0
    log = Path(case["log"]).read_text(encoding="utf-8")
    assert str(source) in log and str(other) in log


def test_comparison_baseline_cannot_be_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    evidence = tmp_path / f"{CASE}.json"
    artifacts = evidence.with_suffix("")
    artifacts.mkdir()
    report = artifacts / "report-0.marsreport"
    report.write_text("baseline", encoding="utf-8")
    evidence.write_text("{}", encoding="utf-8")
    original = json.dumps({"status": "passed", "cases": [{"case": CASE, "evidence": str(evidence)}]})
    output.write_text(original, encoding="utf-8")
    completed = subprocess.run([sys.executable, *_args(output), "--compare-to", str(output)],
                               capture_output=True, text=True)
    assert completed.returncode != 0
    assert output.read_text(encoding="utf-8") == original
    assert report.read_text(encoding="utf-8") == "baseline"
    assert evidence.read_text(encoding="utf-8") == "{}"




def test_missing_report_marks_case_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    output = tmp_path / "summary.json"

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        path = Path(command[command.index("--output") + 1])
        path.write_text(json.dumps(VALID_RESULT), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", _args(output))
    assert runner._main() == 1
    case = _summary(output)["cases"][0]
    assert case["status"] == "failed" and case["exit_code"] == 0
    assert case["failure_reason"]


def test_baseline_log_collision_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    baseline = tmp_path / "baseline.json"
    evidence = tmp_path / "baseline-case.json"
    log = tmp_path / f"{CASE}.log"
    log.write_text("baseline log", encoding="utf-8")
    baseline.write_text(json.dumps({"status": "passed", "cases": [{"case": CASE,
                        "evidence": str(evidence), "log": str(log)}]}), encoding="utf-8")
    completed = subprocess.run([sys.executable, *_args(output), "--compare-to", str(baseline)],
                               capture_output=True, text=True)
    assert completed.returncode != 0
    assert log.read_text(encoding="utf-8") == "baseline log"
    assert not output.exists()
