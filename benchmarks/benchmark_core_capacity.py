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


def _consume(path: Path, config_path: Path, meter: Measurement, loops: int) -> None:
    """新解释器只读取快照与查询配置，不导入 fixture、拟合或挖掘代码。"""
    from mars.reporting import load_report

    meter.result["import_rss_bytes"] = psutil.Process().memory_info().rss
    assert "benchmark_core_workloads" not in sys.modules
    meter.result["fixture_module_imported"] = False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = meter.run("load", lambda: load_report(path))
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


def _stop_tree(process: subprocess.Popen[Any]) -> None:
    """终止本轮进程树；先挂起父进程，避免终止期间继续派生。"""
    try:
        parent = psutil.Process(process.pid)
        parent.suspend()
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        psutil.wait_procs([parent, *children], timeout=5)
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=10)


def _launch(command: list[str], result_path: Path, timeout: float, budget: int) -> dict[str, Any]:
    """串行 worker 保护；同时记录主进程和含子进程的 RSS，日志仅保留尾部。"""
    start = time.perf_counter()
    main_peak = tree_peak = 0
    status = None
    log_path = result_path.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        while process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                main_rss = parent.memory_info().rss
                tree_rss = main_rss
                for child in parent.children(recursive=True):
                    try:
                        tree_rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                main_peak = max(main_peak, main_rss)
                tree_peak = max(tree_peak, tree_rss)
            except psutil.NoSuchProcess:
                pass
            if time.perf_counter() - start > timeout:
                status = "timeout"
            elif tree_peak > budget:
                status = "memory_budget_exceeded"
            if status:
                _stop_tree(process)
                break
            time.sleep(0.01)
    try:
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        if not isinstance(result, dict):
            raise ValueError("worker diagnostics must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        result = {"status": "failed", "stage": "result_decode", "reason": str(exc)}
    result.update(
        exit_code=process.returncode,
        process_wall_seconds=time.perf_counter() - start,
        process_main_rss_observed_peak_bytes=main_peak,
        process_tree_rss_observed_peak_bytes=tree_peak,
        log_tail=log_path.read_text(encoding="utf-8", errors="replace")[-2500:],
    )
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
        if any(
            baseline["parameters"][k] != result["parameters"][k]
            for k in (
                "scale",
                "seed",
                "threads",
                "diagnostic_loops",
                "memory_budget_mib",
                "timeout",
            )
        ):
            raise ValueError("baseline workload/measurement parameters differ")
        comparisons = []
        for item in result["cases"]:
            before = next(
                (
                    b
                    for b in baseline["cases"]
                    if b["case"] == item["case"] and b["backend"] == item["backend"]
                ),
                None,
            )
            if (
                before
                and before.get("summary", {}).get("stages")
                and item.get("summary", {}).get("stages")
            ):
                if before["workload"] != item["workload"]:
                    raise ValueError("baseline dimensions differ")
                before_environment = before["warmup"]["environment"]
                after_environment = item["warmup"]["environment"]
                for key in (
                    "python",
                    "dependencies",
                    "platform",
                    "cpu",
                    "memory_total_bytes",
                    "polars_threads",
                ):
                    old_value = before_environment[key]
                    new_value = after_environment[key]
                    if key == "dependencies":
                        # --source 比较的目标包可能遮盖已安装旧 distribution；环境只比较计算依赖。
                        old_value = {k: v for k, v in old_value.items() if k != "mars-risk"}
                        new_value = {k: v for k, v in new_value.items() if k != "mars-risk"}
                    if old_value != new_value:
                        raise ValueError(f"baseline environment differs: {key}")
                if item["case"].startswith("correlation") or item["case"] in (
                    "rule_report",
                    "rule_bridge",
                ):
                    old_tables = before["rounds"][0]["tables"]
                    for current_round in item["rounds"]:
                        for name, old_table in old_tables.items():
                            for key in ("sha256", "schema", "rows", "columns"):
                                if current_round["tables"][name][key] != old_table[key]:
                                    raise AssertionError(f"baseline table differs: {name}/{key}")
                item["baseline_semantic_check"] = (
                    "full table/schema/order digests"
                    if (
                        item["case"].startswith("correlation")
                        or item["case"] in ("rule_report", "rule_bridge")
                    )
                    else "independent numeric references and focused regressions"
                )
                for name, current in item["summary"]["stages"].items():
                    if name not in before["summary"]["stages"]:
                        continue
                    old = before["summary"]["stages"][name]
                    comparisons.append(
                        {
                            "case": item["case"],
                            "backend": item["backend"],
                            "stage": name,
                            "baseline_case_status": before["status"],
                            "current_case_status": item["status"],
                            "time_ratio": current["median_seconds"] / old["median_seconds"],
                            "rss_ratio": current["highest_observed_rss_bytes"]
                            / old["highest_observed_rss_bytes"],
                            "review_trigger": current["median_seconds"]
                            > old["median_seconds"] * 1.2
                            or current["highest_observed_rss_bytes"]
                            > old["highest_observed_rss_bytes"] * 1.15,
                        }
                    )
        result["comparison"] = comparisons
        _write(destination, result)
    print(destination)
    return 0 if all(c["status"] == "passed" for c in result["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
