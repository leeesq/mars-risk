"""核心规模验收：隔离轮次、RSS 预算、冷快照消费和同进程重复诊断。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import psutil

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1"
MEASUREMENT_CONTRACT = {
    "id": "independent-process-stages-rss-v1",
    "rss_interval_seconds": 0.01,
    "timers": "perf_counter; nested stages are not additive; imports excluded from stage timers",
    "aggregation": "one excluded warmup; independent-process median seconds and maximum sampled RSS",
    "tree_rss": "sum including descendants; shared memory may be counted more than once",
    "table_signature": "Polars hash_rows(seed=42), 50000-row batches, schema/order/value check",
}
DEPENDENCY_MODULES = {
    "numpy": "numpy", "pandas": "pandas", "polars": "polars", "pyarrow": "pyarrow",
    "scikit-learn": "sklearn", "psutil": "psutil", "scipy": "scipy",
    "statsmodels": "statsmodels", "optbinning": "optbinning", "ortools": "ortools",
    "joblib": "joblib", "ruptures": "ruptures", "xlsxwriter": "xlsxwriter",
    "openpyxl": "openpyxl", "jinja2": "jinja2", "threadpoolctl": "threadpoolctl",
}
THREAD_KEYS = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
REPORT_CASES = (
    "rule_report",
    "rule_mining",
    "rule_states",
    "correlation_short",
    "correlation_long",
    "correlation_1000_short",
    "correlation_1000_long",
)
ANALYSIS_CASES = (
    "profile",
    "profile_batch100",
    "binning",
    "binning_batch100",
    "selection",
    "linear_selection",
    "score_cross",
    "score_cross_wide50",
    "score_cross_wide500",
    "optimal_binning",
)
EXTRA_CASES = ("profile_columns", "profile_rows", "binning_columns", "binning_rows", "rule_bridge")


def _write(path: Path, value: Any) -> None:
    """写有限诊断 JSON；调用方管理输出路径。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


class Measurement:
    """沿用 correlation_score_cross 的阶段 RSS 采样，额外持久化正在执行的阶段。"""

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.result: dict[str, Any] = {"status": "running", "stages": {}, "stage": "import"}

    def run(self, name: str, call: Callable[[], Any]) -> Any:
        """测量一个真实调用，嵌套阶段不会相加成为总时间。"""
        self.result["stage"] = name
        _write(self.destination, self.result)
        process = psutil.Process()
        before = process.memory_info().rss

        def tree_rss(main_rss: int) -> int:
            """单独记录 RSS 和，共享内存可能重复计数。"""
            total = main_rss
            for child in process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total

        before_tree = tree_rss(before)
        peak = [before, before_tree]
        done = threading.Event()

        def sample() -> None:
            """10ms 采样包含 NumPy/Arrow 原生内存；可能遗漏瞬时峰值。"""
            while not done.wait(0.01):
                rss = process.memory_info().rss
                peak[0] = max(peak[0], rss)
                peak[1] = max(peak[1], tree_rss(rss))

        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        start = time.perf_counter()
        try:
            return call()
        finally:
            elapsed = time.perf_counter() - start
            after = process.memory_info().rss
            after_tree = tree_rss(after)
            done.set()
            monitor.join()
            self.result["stages"][name] = {
                "seconds": elapsed,
                "rss_start_bytes": before,
                "rss_end_bytes": after,
                "rss_observed_peak_bytes": max(peak[0], after),
                "rss_observed_increment_bytes": max(peak[0], after) - before,
                "rss_tree_start_bytes": before_tree,
                "rss_tree_end_bytes": after_tree,
                "rss_tree_observed_peak_bytes": max(peak[1], after_tree),
                "rss_tree_observed_increment_bytes": max(peak[1], after_tree) - before_tree,
            }
            _write(self.destination, self.result)


def _signature(frame: Any) -> dict[str, Any]:
    """按有限批次原生行哈希验证快照无损；不展开全表 Python 行。"""
    import polars as pl

    if not isinstance(frame, pl.DataFrame):
        frame = pl.from_pandas(frame, include_index=True)
    digest = hashlib.sha256()
    for offset in range(0, frame.height, 50000):
        digest.update(frame.slice(offset, 50000).hash_rows(seed=42).to_numpy().tobytes())
    return {
        "rows": frame.height,
        "columns": frame.width,
        "schema": {k: str(v) for k, v in frame.schema.items()},
        "sha256": digest.hexdigest(),
    }


def _context(report: Any, queries: dict[str, Any], budget: int) -> dict[str, Any]:
    """区分有效证据、明确拒绝与合法调用却返回空证据。"""
    try:
        context = report.to_ai_context(queries=queries, max_chars=budget)
    except ValueError as exc:
        return {"status": "rejected", "reason": str(exc), "budget": budget}
    payload = json.loads(context)
    rows = sum(e["returned_rows"] for e in payload["evidence"])
    assert len(context) <= budget
    return {
        "status": "passed" if rows else "failed",
        "budget": budget,
        "characters": len(context),
        "evidence_rows": rows,
        "omitted": payload["omitted"],
        "references": [
            {k: e[k] for k in ("reference", "query", "next_offset")} for e in payload["evidence"]
        ],
    }


def _dependency_state(name: str, module: str, unavailable: bool = False) -> dict[str, Any]:
    """区分未安装、未参与导入、成功导入和生产入口已尝试但导入失败。"""
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "status": "not_installed"}
    return {
        "version": version,
        "status": "import_failed" if unavailable else "imported" if module in sys.modules else "not_imported",
    }


def _execution_parameters(parameters: Any) -> tuple[Any, dict[str, Any]]:
    """从实际配置中分出资源策略，避免把 batch/线程策略误当算法变更。"""
    resources: dict[str, Any] = {}

    def split(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                location = f"{path}.{key}" if path else key
                if key in {"batch_size", "overview_batch_size", "n_jobs"}:
                    resources[location] = item
                else:
                    result[key] = split(item, location)
            return result
        if isinstance(value, list):
            return [split(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value

    return split(parameters, ""), resources


def _consume(path: Path, config_path: Path, meter: Measurement, loops: int) -> None:
    """新解释器只读取快照与查询配置，不导入 fixture、拟合或挖掘代码。"""
    from mars.reporting import load_report

    meter.result["import_rss_bytes"] = psutil.Process().memory_info().rss
    assert "benchmark_core_workloads" not in sys.modules
    meter.result["fixture_module_imported"] = False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = meter.run("load", lambda: load_report(path))
    meter.result["execution_contract"] = {
        "workload_id": "core-capacity-snapshot-consumer-v1",
        "workload": {"report_type": report.report_type, "loops": loops},
        "algorithm_parameters": {"query": config["query"], "ai": config["ai"], "export": config["export"]},
        "diagnostics": {"fixture_import": "forbidden", "snapshot_correctness": "full-table-signatures"},
        "resource_strategy": {},
    }
    assert report.report_id == config["report_id"]
    assert report.describe() == config["description"]
    signatures = meter.run(
        "snapshot_correctness",
        lambda: {name: _signature(report.get_table(name)) for name in config["signatures"]},
    )
    assert signatures == config["signatures"], "snapshot table/schema/order changed"
    page = meter.run("query_page", lambda: report.query_page(config["table"], **config["query"]))
    assert _signature(page["data"]) == config["page_signature"]
    assert page["total_rows"] == config["page_total"]
    replay = report.get_table(page["reference"]["table"], **page["reference"]["query"])
    assert _signature(replay) == config["page_signature"]
    meter.result["page"] = {
        k: page[k] for k in ("total_rows", "returned_rows", "next_offset", "truncated")
    }
    contexts = {
        str(b): meter.run(f"context_{b}", partial(_context, report, config["ai"], b))
        for b in (5000, 16000, 512)
    }
    assert contexts["512"]["status"] == "rejected", "minimum description rejection missing"
    meter.result["contexts"] = contexts
    if config.get("feature"):
        meter.run("get_feature", lambda: report.get_feature(config["feature"], limit=20))
        matches = meter.run("search_features", lambda: report.search_features("重复显示名"))
        meter.result["search_matches"] = len(matches)
    if report.report_type == "correlation":
        from mars.reporting import (
            get_correlation_matrix,
            get_related_features,
            show_correlation_matrix,
        )

        peers = meter.run(
            "related20",
            lambda: get_related_features(report, config["matrix_features"][0], limit=20),
        )
        matrix = meter.run(
            "matrix30", lambda: get_correlation_matrix(report, config["matrix_features"])
        )
        meter.result["matrix_shape"] = list(matrix.shape)
        meter.result["related_rows"] = len(peers)
        html = meter.run("html", lambda: show_correlation_matrix(report, max_features=30).to_html())
    elif report.report_type == "score_cross":
        from mars.analysis import evaluate_score_policy, show_score_matrix

        policy = meter.run(
            "policy_replay",
            lambda: evaluate_score_policy(
                report,
                {"type": "and", "x_max_risk_rank": 3, "y_max_risk_rank": 3},
                baseline={"type": "x_only", "x_max_risk_rank": 3},
            ),
        )
        assert _signature(policy.get_table("summary")) == config["policy_signature"]
        html = meter.run(
            "html",
            lambda: show_score_matrix(report, filters={"target": "bad", "group": "OOT"}).to_html(),
        )
    else:
        html = meter.run(
            "html",
            lambda: report.show_table(
                config["table"], **{**config["query"], "offset": 0, "limit": 20}
            ).to_html(),
        )
    meter.result["html_characters"] = len(html)
    if config.get("export"):
        export = path.parent / "small.xlsx"
        meter.run("excel_full_small", lambda: report.write_excel(str(export)))
        meter.result["excel_file_bytes"] = export.stat().st_size
    # 独立诊断进程预热后循环，冷加载和同进程内存池行为分别记录。
    meter.run("diagnostic_warmup", lambda: load_report(path))
    gc.collect()
    iterations: list[dict[str, Any]] = []
    for i in range(loops):
        start_rss = psutil.Process().memory_info().rss
        start = time.perf_counter()
        loaded = load_report(path)
        load_seconds = time.perf_counter() - start
        start = time.perf_counter()
        loaded.query_page(config["table"], **config["query"])
        query_seconds = time.perf_counter() - start
        start = time.perf_counter()
        _context(loaded, config["ai"], 16000)
        context_seconds = time.perf_counter() - start
        del loaded
        gc.collect()
        iterations.append(
            {
                "iteration": i,
                "load_seconds": load_seconds,
                "query_seconds": query_seconds,
                "context_seconds": context_seconds,
                "rss_start_bytes": start_rss,
                "rss_after_release_bytes": psutil.Process().memory_info().rss,
            }
        )
    meter.result["same_process_iterations"] = iterations
    if iterations:
        rss = [v["rss_after_release_bytes"] for v in iterations]
        meter.result["retention"] = {
            "first_bytes": rss[0],
            "last_bytes": rss[-1],
            "second_half_range_bytes": max(rss[len(rss) // 2 :]) - min(rss[len(rss) // 2 :]),
            "note": "observations only; allocator pools can retain memory; no zero-RSS leak assertion",
        }
    valid = all(contexts[str(b)]["status"] == "passed" for b in (5000, 16000))
    meter.result.update(
        status="passed" if valid else "failed",
        stage="complete",
        reason=None if valid else "valid AI budget did not produce evidence",
    )


def _environment(source: Path) -> dict[str, Any]:
    """记录实际依赖、线程、主机和可读取的 cgroup 限制。"""
    import polars as pl

    import mars

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(source.parent), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    limits: dict[str, str] = {}
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        if Path(name).is_file():
            limits[name] = Path(name).read_text().strip()
    cpu = platform.processor()
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    provenance_path = source.parent / "capacity_source.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    )
    source_digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        source_digest.update(str(path.relative_to(source)).replace("\\", "/").encode())
        source_digest.update(path.read_bytes())
    return {
        "python": sys.version,
        "dependencies": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "pandas", "polars", "pyarrow", "scikit-learn", "psutil")
        },
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total,
        "memory_available_bytes": psutil.virtual_memory().available,
        "container_limits": limits or {"status": "not detected; Windows job limit not exposed"},
        "threads_environment": {k: os.environ.get(k) for k in THREAD_KEYS},
        "polars_threads": pl.thread_pool_size(),
        "source": str(source),
        "mars_module_file": mars.__file__,
        "mars_module_version": mars.__version__,
        "installed_mars_distribution_version": importlib.metadata.version("mars-risk"),
        "commit": provenance.get("commit", git("rev-parse", "HEAD")),
        "worktree_status": provenance.get("worktree_status", git("status", "--short")),
        "harness_version": VERSION,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "workloads_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_core_workloads.py").read_bytes()
        ).hexdigest(),
        "source_python_sha256": source_digest.hexdigest(),
        "date_utc": datetime.now(timezone.utc).isoformat(),
    }


def _stop_tree(
    process: subprocess.Popen[Any], descendants: list[psutil.Process] | None = None
) -> dict[str, Any]:
    """有界清理已发现的后代；直属 worker 仅由其 Popen 回收真实退出状态。"""
    cleanup: dict[str, Any] = {"errors": [], "descendant_pids": [], "remaining_pids": []}
    children = {child.pid: child for child in descendants or []}
    parent = None
    try:
        if process.poll() is None:
            parent = psutil.Process(process.pid)
            try:
                parent.suspend()
            except psutil.NoSuchProcess:
                parent = None
            except psutil.Error as exc:
                cleanup["errors"].append(f"parent suspend: {type(exc).__name__}: {exc}")
            if parent is not None:
                children.update({child.pid: child for child in parent.children(recursive=True)})
    except psutil.NoSuchProcess:
        pass
    except (psutil.Error, OSError) as exc:
        cleanup["errors"].append(f"descendant discovery: {type(exc).__name__}: {exc}")
    # 父进程自然退出后，缓存后代可能继续派生；先冻结各存活后代再补查其进程树。
    pending = list(children.values())
    discovered: set[int] = set()
    while pending:
        child = pending.pop()
        if child.pid in discovered:
            continue
        discovered.add(child.pid)
        try:
            child.suspend()
            for descendant in child.children(recursive=True):
                if descendant.pid not in children:
                    children[descendant.pid] = descendant
                    pending.append(descendant)
        except psutil.NoSuchProcess:
            pass
        except (psutil.Error, OSError) as exc:
            cleanup["errors"].append(
                f"descendant discovery: {child.pid} suspend/children: {type(exc).__name__}: {exc}"
            )
    cleanup["descendant_pids"] = sorted(children)
    for child in reversed(list(children.values())):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as exc:
            cleanup["errors"].append(f"descendant {child.pid} kill: {type(exc).__name__}: {exc}")
    # psutil 不等待直属 worker，避免 POSIX waitpid 抢走 Popen 的退出状态。
    try:
        process.kill()
    except ProcessLookupError:
        pass
    except OSError as exc:
        cleanup["errors"].append(f"worker kill: {type(exc).__name__}: {exc}")
    try:
        process.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError) as exc:
        cleanup["errors"].append(f"worker wait: {type(exc).__name__}: {exc}")
    if children:
        try:
            _, alive = psutil.wait_procs(list(children.values()), timeout=5)
            # 僵尸已停止执行；非直属后代的最终回收由其 OS 父进程负责。
            for child in alive:
                try:
                    if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                        cleanup["remaining_pids"].append(child.pid)
                except psutil.NoSuchProcess:
                    pass
        except psutil.Error as exc:
            cleanup["errors"].append(f"descendants wait: {type(exc).__name__}: {exc}")
    cleanup["worker_exit_status"] = "known" if process.returncode is not None else "unknown"
    cleanup["descendant_status"] = "unknown" if any(
        error.startswith(("parent suspend:", "descendant discovery:", "descendants wait:"))
        for error in cleanup["errors"]
    ) else "incomplete" if cleanup["remaining_pids"] else "stopped"
    return cleanup


def _launch(command: list[str], result_path: Path, timeout: float, budget: int) -> dict[str, Any]:
    """串行 worker 保护；同时记录主进程和含子进程的 RSS，日志仅保留尾部。"""
    start = time.perf_counter()
    main_peak = tree_peak = 0
    status = None
    descendants: dict[int, psutil.Process] = {}
    diagnostic_errors: list[str] = []
    cleanup: dict[str, Any] | None = None

    def record_error(message: str) -> None:
        """同一诊断错误最多记录一次，长时间 worker 不积累重复错误。"""
        if message not in diagnostic_errors:
            diagnostic_errors.append(message)

    def running() -> bool:
        """Popen 查询失败也触发有界清理，不能遗留无主 worker。"""
        nonlocal status, cleanup
        try:
            return process.poll() is None
        except OSError as exc:
            record_error(f"worker poll: {type(exc).__name__}: {exc}")
            status = "failed"
            cleanup = _stop_tree(process, list(descendants.values()))
            return False
    log_path = result_path.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        while running():
            try:
                parent = psutil.Process(process.pid)
                main_rss = parent.memory_info().rss
                tree_rss = main_rss
                for child in parent.children(recursive=True):
                    descendants[child.pid] = child
                    try:
                        tree_rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.Error as exc:
                        record_error(f"descendant RSS: {type(exc).__name__}: {exc}")
                main_peak = max(main_peak, main_rss)
                tree_peak = max(tree_peak, tree_rss)
            except psutil.NoSuchProcess:
                pass
            except psutil.Error as exc:
                record_error(f"worker RSS: {type(exc).__name__}: {exc}")
                status = "failed"
            if time.perf_counter() - start > timeout:
                status = "timeout"
            elif tree_peak > budget:
                status = "memory_budget_exceeded"
            if status:
                cleanup = _stop_tree(process, list(descendants.values()))
                break
            time.sleep(0.01)
        if cleanup is None and descendants:
            cleanup = _stop_tree(process, list(descendants.values()))
    try:
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        if not isinstance(result, dict):
            raise ValueError("worker diagnostics must be a JSON object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result = {"status": "failed", "stage": "result_decode", "reason": str(exc)}
        record_error(f"result diagnostics: {type(exc).__name__}: {exc}")
    try:
        with log_path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 10000))
            log_tail = log.read().decode("utf-8", errors="replace")[-2500:]
    except OSError as exc:
        log_tail = ""
        record_error(f"log diagnostics: {type(exc).__name__}: {exc}")
    result.update(
        exit_code=process.returncode,
        exit_status="known" if process.returncode is not None else "unknown",
        process_wall_seconds=time.perf_counter() - start,
        process_main_rss_observed_peak_bytes=main_peak,
        process_tree_rss_observed_peak_bytes=tree_peak,
        log_tail=log_tail,
    )
    if cleanup is not None:
        result["cleanup"] = cleanup
    if diagnostic_errors:
        result["diagnostic_errors"] = diagnostic_errors
    result.setdefault("stage", "startup")
    if status or process.returncode != 0:
        result.update(
            status=status or "failed",
            reason=status or result.get("reason") or "abnormal exit; OOM not confirmed",
        )
    elif result.get("status") == "running":
        result.update(status="failed", reason="worker exited without completion")
    elif "status" not in result:
        result.update(status="failed", reason="worker exited without diagnostics")
    if cleanup and (cleanup["descendant_status"] != "stopped" or cleanup["worker_exit_status"] == "unknown"):
        result.update(status=status or "failed", reason=status or "process tree cleanup incomplete")
    if diagnostic_errors and result["status"] == "passed":
        result.update(status="failed", reason="worker diagnostic collection failed")
    return result


def _parser() -> argparse.ArgumentParser:
    """列明真实 workload；新增诊断维度只由显式 case 选择。"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="smoke: 1000 audits, 40 correlation features, 800x8 analysis, 2000 cross rows\n"
        "standard: 10000 audits, 500/1000 correlation features, 50000x200, 1M cross rows\n"
        "large: 50000 audits, 3000 correlation features, 200000x1000, 1M cross rows\n"
        "*_columns: 50000x3000; *_rows: 1Mx100 (smoke remains small)\n"
        "rule_bridge: same audit fixture with nested bridge/catalog construction diagnostic\n"
        "linear_selection: 400x8 / 3000x40 / 5000x80; optimal_binning: 1000x4 / 5000x10 / 10000x20",
    )
    parser.add_argument("--suite", choices=("reports", "analysis", "all"), default="all")
    parser.add_argument("--scale", choices=("smoke", "standard", "large"), default="smoke")
    parser.add_argument("--case", nargs="+", choices=(*REPORT_CASES, *ANALYSIS_CASES, *EXTRA_CASES))
    parser.add_argument("--backend", choices=("pandas", "polars", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--threads", type=int, default=4,
        help="每进程数值库线程及 selector.n_jobs；独立分箱入口保留并记录默认 n_jobs",
    )
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--memory-budget-mib", type=float, default=8192)
    parser.add_argument("--diagnostic-loops", type=int, default=10)
    parser.add_argument("--source", type=Path, default=ROOT / "src")
    parser.add_argument(
        "--consumer-source", type=Path, help="可选：用另一份源码消费快照，验证旧快照恢复"
    )
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument(
        "--comparison-purpose", help="资源策略刻意改变时说明对照目的；工作量/测量合同仍须相同"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--consume", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--query-config", type=Path, help=argparse.SUPPRESS)
    return parser


def _worker(args: argparse.Namespace) -> int:
    """线程环境在任何计算依赖导入前生效，失败保留阶段和 traceback。"""
    for name in THREAD_KEYS:
        os.environ[name] = str(args.threads)
    sys.path.insert(0, str(args.source.resolve()))
    meter = Measurement(args.output)
    start = time.perf_counter()
    try:
        meter.result["environment"] = _environment(args.source)
        if (
            args.source.resolve()
            not in Path(meter.result["environment"]["mars_module_file"]).resolve().parents
        ):
            raise RuntimeError("imported mars source differs from requested source")
        meter.result["import_rss_bytes"] = psutil.Process().memory_info().rss
        if args.consume:
            _consume(args.consume, args.query_config, meter, args.diagnostic_loops)
        else:
            from benchmark_core_workloads import run_case

            run_case(args, meter)
    except Exception as exc:
        meter.result.update(
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()[-4000:],
        )
    meter.result["worker_seconds"] = time.perf_counter() - start
    if "numpy" in sys.modules:
        from threadpoolctl import threadpool_info

        meter.result["effective_native_threadpools"] = threadpool_info()
    if "environment" in meter.result:
        unavailable = meter.result.get("linear_dependency_state", {}).get("status") in (
            "not_installed", "import_failed"
        )
        states = {
            name: _dependency_state(name, module, unavailable=name == "statsmodels" and unavailable)
            for name, module in DEPENDENCY_MODULES.items()
        }
        meter.result["environment"]["participating_dependencies"] = {
            name: state for name, state in states.items()
            if state["status"] == "imported" or name == "statsmodels" and args.case == "linear_selection"
        }
    _write(args.output, meter.result)
    return 0 if meter.result["status"] == "passed" else 1


def _summary(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总完整执行轮次；消费语义失败仍保留已完成阶段，不估算 P95。"""
    passed = [r for r in rounds if r.get("stage") == "complete"]
    phases = sorted({name for r in passed for name in r.get("stages", {})})
    return {
        "successful_rounds": sum(r["status"] == "passed" for r in rounds),
        "completed_rounds_including_semantic_failures": len(passed),
        "stages": {
            name: {
                "median_seconds": statistics.median(r["stages"][name]["seconds"] for r in passed),
                "highest_observed_rss_bytes": max(
                    r["stages"][name]["rss_observed_peak_bytes"] for r in passed
                ),
            }
            for name in phases
        },
        "median_process_seconds": statistics.median(r["process_wall_seconds"] for r in passed)
        if passed
        else None,
        "highest_observed_tree_rss_bytes": max(
            (r["process_tree_rss_observed_peak_bytes"] for r in rounds), default=0
        ),
    }


def _comparison_state(round_: dict[str, Any]) -> dict[str, Any]:
    """比较生产和冷消费的有效路径，源码与 DLL 路径只作溯源。"""
    contract = round_.get("execution_contract")
    environment = round_.get("environment", {})
    required = {"workload_id", "workload", "algorithm_parameters", "diagnostics", "resource_strategy"}
    consumer = _comparison_state(round_["consumer"]) if "consumer" in round_ else None
    if (
        not isinstance(contract, dict) or not required.issubset(contract)
        or "participating_dependencies" not in environment
        or consumer is not None and consumer.get("missing_contract")
    ):
        return {"missing_contract": True}
    return {
        "execution_contract": contract,
        "environment": {key: environment.get(key) for key in (
            "python", "platform", "cpu", "logical_cpus", "physical_cpus", "memory_total_bytes",
            "container_limits", "participating_dependencies",
        )},
        "resources": {
            "polars_threads": environment.get("polars_threads"),
            "threads_environment": environment.get("threads_environment"),
            "native_threadpools": [{key: pool.get(key) for key in (
                "internal_api", "prefix", "num_threads", "architecture",
            )} for pool in round_.get("effective_native_threadpools", [])],
        },
        "consumer": consumer,
    }


def _compare_execution_state(
    old: Any, new: Any, path: str, reasons: list[str], resources: list[dict[str, Any]]
) -> None:
    """报告具体合同差异路径；资源策略由调用方检查显式对照目的。"""
    if old == new:
        return
    if path.endswith(".resource_strategy") or path.endswith(".resources"):
        resources.append({"path": path, "baseline": old, "current": new})
    elif isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            _compare_execution_state(old.get(key), new.get(key), f"{path}.{key}", reasons, resources)
    else:
        reasons.append(f"effective execution differs: {path} (baseline={old!r}, current={new!r})")


def _compare_baseline(baseline: dict[str, Any], result: dict[str, Any]) -> None:
    """保留历史轮次；只有实际工作、测量和环境可比时才计算普通性能比例。"""
    comparisons: list[dict[str, Any]] = []
    for item in result["cases"]:
        before = next((old for old in baseline["cases"] if (
            old["case"] == item["case"] and old["backend"] == item["backend"]
        )), None)
        entry: dict[str, Any] = {"case": item["case"], "backend": item["backend"]}
        reasons: list[str] = []
        resources: list[dict[str, Any]] = []
        if before is None:
            entry.update(status="incomparable", reasons=["baseline case/backend absent"])
            comparisons.append(entry)
            continue
        entry.update(baseline_case_status=before["status"], current_case_status=item["status"])
        if before["status"] != "passed" or item["status"] != "passed":
            entry.update(status="execution_failed", reasons=["at least one case did not pass execution/semantic checks"])
            comparisons.append(entry)
            continue
        if not baseline.get("measurement_contract", {}).get("id") or not result.get("measurement_contract", {}).get("id"):
            reasons.append("measurement contract absent; historical result has insufficient compatibility information")
        elif baseline["measurement_contract"] != result["measurement_contract"]:
            reasons.append("measurement contract differs")
        for key in ("timeout", "memory_budget_mib", "diagnostic_loops"):
            if baseline["parameters"].get(key) != result["parameters"].get(key):
                reasons.append(f"measurement/protection parameter differs: {key}")
        before_workload, before_resources = _execution_parameters(before["workload"])
        current_workload, current_resources = _execution_parameters(item["workload"])
        _compare_execution_state(before_workload, current_workload, "case.workload", reasons, resources)
        _compare_execution_state(before_resources, current_resources, "case.resource_strategy", reasons, resources)
        old_rounds = [before.get("warmup", {}), *before["rounds"]]
        new_rounds = [item.get("warmup", {}), *item["rounds"]]
        reference = _comparison_state(old_rounds[0])

        if reference.get("missing_contract"):
            reasons.append("baseline execution/dependency contract absent")
        for side, rounds in (("baseline", old_rounds), ("current", new_rounds)):
            for index, round_ in enumerate(rounds):
                state = _comparison_state(round_)
                if state.get("missing_contract"):
                    reasons.append(f"{side} round {index}: execution/dependency contract absent")
                else:
                    _compare_execution_state(reference, state, f"{side}.round[{index}]", reasons, resources)
        old_stages = before.get("summary", {}).get("stages", {})
        new_stages = item.get("summary", {}).get("stages", {})
        if not old_stages or set(old_stages) != set(new_stages):
            reasons.append("measured stage set differs or no completed measurement")
        purpose = result["parameters"].get("comparison_purpose")
        if resources and not purpose:
            reasons.append("resource strategy differs; provide --comparison-purpose for an intentional strategy comparison")
        if reasons:
            entry.update(status="incomparable", reasons=list(dict.fromkeys(reasons)))
            if resources:
                entry["resource_differences"] = resources
            comparisons.append(entry)
            continue
        # 有意义的完整表验收保持原有口径，错误结果不被包装成性能退化。
        digest_check = item["case"].startswith("correlation") or item["case"] in ("rule_report", "rule_bridge")
        if digest_check:
            old_tables = before["rounds"][0]["tables"]
            if any(round_["tables"].keys() != old_tables.keys() or any(
                round_["tables"][name][key] != table[key]
                for name, table in old_tables.items() for key in ("sha256", "schema", "rows", "columns")
            ) for round_ in item["rounds"]):
                entry.update(status="semantic_mismatch", reasons=["baseline table/schema/order/value signatures differ"])
                comparisons.append(entry)
                continue
        item["baseline_semantic_check"] = (
            "full table/schema/order digests" if digest_check else "independent numeric references and focused regressions"
        )
        for name, current in new_stages.items():
            old = old_stages[name]
            if old["median_seconds"] <= 0 or old["highest_observed_rss_bytes"] <= 0:
                comparisons.append({**entry, "stage": name, "status": "incomparable", "reasons": ["baseline metric is nonpositive"]})
                continue
            stage_entry = {
                **entry, "stage": name,
                "status": "resource_strategy_comparison" if resources else "comparable",
                "time_ratio": current["median_seconds"] / old["median_seconds"],
                "rss_ratio": current["highest_observed_rss_bytes"] / old["highest_observed_rss_bytes"],
                "review_trigger": current["median_seconds"] > old["median_seconds"] * 1.2
                or current["highest_observed_rss_bytes"] > old["highest_observed_rss_bytes"] * 1.15,
            }
            if resources:
                stage_entry.update(resource_differences=resources, purpose=purpose)
            comparisons.append(stage_entry)
    result["comparison"] = comparisons


def main() -> int:
    """调度独立预热和测量轮次，历史 JSON 绝不覆盖。"""
    args = _parser().parse_args()
    if args.worker:
        if args.case:
            args.case = args.case[0]
        return _worker(args)
    if min(args.repeats, args.threads, args.timeout, args.memory_budget_mib) <= 0:
        raise ValueError("repeats, threads, timeout and memory budget must be positive")
    if args.diagnostic_loops < 10:
        raise ValueError("diagnostic-loops must be >=10")
    destination = args.output or ROOT / "benchmarks" / "results" / (
        f"core_capacity_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    )
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cases = list(
        REPORT_CASES
        if args.suite == "reports"
        else ANALYSIS_CASES
        if args.suite == "analysis"
        else (*REPORT_CASES, *ANALYSIS_CASES)
    )
    if args.scale == "large":
        cases = [c for c in cases if not c.startswith("correlation_1000")]
    if args.case:
        cases = args.case
    backends = ("pandas", "polars") if args.backend == "both" else (args.backend,)
    budget = int(args.memory_budget_mib * 1024**2)
    result: dict[str, Any] = {
        "harness_version": VERSION,
        "scheduler_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpus": psutil.cpu_count(),
            "memory_total_bytes": psutil.virtual_memory().total,
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "note": "numeric libraries are not imported by the scheduler; effective pools recorded in workers",
        },
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "measurement": {
            "rss_interval_seconds": 0.01,
            "tree_note": "RSS sum may double count shared memory; not exclusive physical memory",
            "limits": "sampling protection may miss brief peaks; nested timers are not additive",
            "statistics": "independent-process medians and maxima; no P95; warmups excluded",
        },
        "measurement_contract": MEASUREMENT_CONTRACT,
        "cases": [],
    }
    with destination.open("x", encoding="utf-8") as reserved:
        json.dump(result, reserved, ensure_ascii=False)
    with TemporaryDirectory(prefix="mars-capacity-") as temporary:
        folder = Path(temporary)
        for case in cases:
            for backend in backends:
                # 提前拒绝明显不可负担目标，不降档；粗估必须与观测分开。
                from benchmark_core_workloads import workload

                dimensions = workload(case, args.scale)
                estimate = (
                    dimensions.get("rows", 0) * dimensions.get("features", 0) * 16 + 512 * 1024**2
                )
                item: dict[str, Any] = {
                    "case": case,
                    "backend": backend,
                    "workload": dimensions,
                    "rounds": [],
                    "warmup": None,
                }
                available = psutil.virtual_memory().available
                item["preflight"] = {
                    "available_ram_bytes": available,
                    "estimated_working_set_bytes": estimate,
                    "rss_tree_budget_bytes": budget,
                    "estimate_note": "rows*features*16 + 512MiB; estimate only, not a capacity guarantee",
                }
                result["cases"].append(item)
                if estimate > budget or estimate > available * 0.8:
                    item.update(
                        status="not_run",
                        reason="conservative working-set estimate exceeds budget/available RAM",
                        estimated_bytes=estimate,
                    )
                    _write(destination, result)
                    continue
                for index in range(-1, args.repeats):
                    out = folder / f"{case}-{backend}-{index}.json"
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--case",
                        case,
                        "--scale",
                        args.scale,
                        "--backend",
                        backend,
                        "--threads",
                        str(args.threads),
                        "--seed",
                        str(args.seed),
                        "--source",
                        str(args.source.resolve()),
                        "--output",
                        str(out),
                        "--timeout",
                        str(args.timeout),
                        "--memory-budget-mib",
                        str(args.memory_budget_mib),
                        "--diagnostic-loops",
                        str(args.diagnostic_loops),
                    ]
                    if args.consumer_source:
                        command.extend(["--consumer-source", str(args.consumer_source.resolve())])
                    raw = _launch(command, out, args.timeout, budget)
                    if index == -1:
                        raw["completed"] = raw.get("stage") == "complete"
                        item["warmup"] = raw
                    else:
                        item["rounds"].append(raw)
                    print(
                        f"{case}/{backend} {'warmup' if index < 0 else index}: {raw['status']} "
                        f"{raw['process_wall_seconds']:.2f}s",
                        flush=True,
                    )
                    _write(destination, result)
                    if raw["status"] in ("timeout", "memory_budget_exceeded"):
                        item["unexecuted_rounds"] = {
                            "status": "not_run",
                            "count": args.repeats - len(item["rounds"]),
                            "reason": "stopped after sampled resource protection; no retry",
                        }
                        break
                item["summary"] = _summary(item["rounds"])
                item["status"] = (
                    "passed"
                    if (
                        len(item["rounds"]) == args.repeats
                        and all(r["status"] == "passed" for r in item["rounds"])
                        and item["warmup"]["status"] == "passed"
                    )
                    else "failed"
                )
                if raw["status"] in ("timeout", "memory_budget_exceeded"):
                    item["status"] = raw["status"]
                _write(destination, result)
    if args.baseline_json:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        _compare_baseline(baseline, result)
        _write(destination, result)
    print(destination)
    return 0 if all(c["status"] == "passed" for c in result["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
