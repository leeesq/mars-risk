"""隔离进程测量分客群完整公开调用；所有报告和固定规则共存计入 RSS。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from functools import wraps
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
THREAD_KEYS = (
    "POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _check_mars_location(source_root: Path, module_file: str) -> str:
    """核对导入位置属于请求源码树，返回规范化位置。"""
    actual = Path(module_file).resolve()
    expected = (source_root / "src" / "mars").resolve()
    try:
        actual.relative_to(expected)
    except ValueError as exc:
        raise RuntimeError(f"Requested MARS location {expected}; imported {actual}") from exc
    return str(actual)


def _write(path: Path, value: dict[str, Any]) -> None:
    """将有限统计证据写入调用方指定目录，不序列化原始宽表。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _population(rows: int, features: int, segments: int, distribution: str) -> Any:
    """固定 seed 生成同规模完整人口；不为不同模式改变标签、阈值或候选。"""
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(20261029)
    latent = rng.normal(size=rows)
    labels = (latent + rng.normal(scale=0.7, size=rows) > 0).astype(np.int64)
    if distribution == "imbalanced" and segments > 1:
        probabilities = np.array([0.85, *([0.15 / (segments - 1)] * (segments - 1))])
    else:
        probabilities = np.repeat(1.0 / segments, segments)
    customer = rng.choice(segments, size=rows, p=probabilities)
    columns: dict[str, Any] = {
        f"feature_{index:03d}_customer_application_measure": latent * (1 + index / features)
        + rng.normal(scale=0.4 + index / features, size=rows)
        for index in range(features)
    }
    names = list(columns)
    if features > 2:
        columns[names[1]] = columns[names[0]] * 2
        # 客群内方向相反的候选保留在原始声明范围，不先按整体筛掉。
        columns[names[2]] = latent * np.where(customer % 2 == 0, 1, -1) + rng.normal(scale=0.5, size=rows)
    if features > 3:
        columns[names[-1]] = np.where(latent > 0, "high", "low")
    columns.update({
        "customer_segment": [f"seg{value + 1}" for value in customer],
        "dataset": ["Train" if index % 5 < 3 else "Val" if index % 5 == 3 else "Test" for index in range(rows)],
        "target": pl.Series("target", labels).set(pl.Series(rng.random(rows) < 0.08), None),
    })
    frame = pl.DataFrame(columns)
    return pl.concat([frame.slice(0, rows // 3), frame.slice(rows // 3)], rechunk=False)


def _finite(value: Any) -> Any:
    """非有限统计值保留 null，不把缺失统计伪装成零。"""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _signature(frame: Any) -> dict[str, Any]:
    """散列小统计表的 schema、顺序和值，不使用全宽 IPC BytesIO。"""
    digest = hashlib.sha256()
    for offset in range(0, frame.height, 50_000):
        digest.update(frame.slice(offset, 50_000).hash_rows(seed=20261029).to_numpy().tobytes())
    return {"rows": frame.height, "schema": {key: str(value) for key, value in frame.schema.items()},
            "sha256": digest.hexdigest()}


def _worker(args: argparse.Namespace) -> int:
    """在一次连续调用中持有原始输入、画像报告、筛选器及各 scope 规则。"""
    import polars as pl
    from threadpoolctl import threadpool_info

    import mars
    mars_module = _check_mars_location(Path(args.source_root), mars.__file__)
    from mars.analysis import (
        MarsBinEvaluator,
        MarsDataProfiler,
        _raw_ks,
        _risk_profile,
        profile_risk,
        profile_stats,
    )
    from mars.feature.binning import (
        MarsBinnerBase,
        MarsLiteOptBinner,
        MarsNativeBinner,
        MarsOptimalBinner,
    )
    from mars.feature.selection import MarsStatsSelector

    phase = ["setup"]
    observations: list[dict[str, Any]] = []
    diagnostic_seconds: dict[str, dict[str, float]] = {}
    if args.execution_counts:
        from mars.analysis import evaluator as evaluator_module
        from mars.analysis._evaluation import segmented as segmented_module

        def time_calls(owner: Any, name: str, label: str) -> None:
            """按调用阶段累计耗时；不同函数可能相互包含。"""
            original = getattr(owner, name)

            @wraps(original)
            def timed(*values: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    return original(*values, **kwargs)
                finally:
                    timings = diagnostic_seconds.setdefault(phase[0], {})
                    timings[label] = timings.get(label, 0.0) + time.perf_counter() - started

            setattr(owner, name, timed)

        time_calls(segmented_module, "_fit_scope_rules", "fit_rules_including_woe")
        time_calls(MarsBinnerBase, "transform", "bin_transform")
        time_calls(segmented_module, "_collect_shared_statistics", "shared_collection_including_transform")
        time_calls(MarsBinEvaluator, "_collect_batch_statistics", "batch_statistics")
        time_calls(evaluator_module, "calculate_metrics_from_stats", "calculate_metrics")
        time_calls(MarsBinEvaluator, "_format_report", "format_report")
        def observe(owner: Any, name: str, kind: str) -> None:
            """观察公开链调用；LazyFrame 的后续执行次数不能由调用次数推导。"""
            original = getattr(owner, name)

            @wraps(original)
            def wrapper(*values: Any, **kwargs: Any) -> Any:
                caller = sys._getframe(1)
                data = values[1] if isinstance(owner, type) and len(values) > 1 else values[0]
                row_count = data.height if isinstance(data, pl.DataFrame) else (
                    len(data) if hasattr(data, "__len__") and not isinstance(data, pl.LazyFrame) else None
                )
                population = "unknown"
                if kind == "transform":
                    if caller.f_code.co_name == "_collect_shared_statistics":
                        population = "current" if caller.f_locals["group_col"] in data.columns else "benchmark"
                    elif caller.f_code.co_name == "collect_counts":
                        population = "current" if caller.f_locals["with_group"] else "benchmark"
                    elif caller.f_code.co_name in {"evaluate", "_collect_batch_statistics", "_binned_psi_batch"}:
                        population = "current" if data is caller.f_locals.get("batch_df", caller.f_locals.get("df_batch")) else "benchmark"
                    elif caller.f_code.co_name in {"_fit_scope_rules", "profile_bin_performance"}:
                        population = "fit_or_woe"
                selected = kwargs.get("features")
                if selected is None and kind == "raw_ks_calculation":
                    selected = list({feature for columns in kwargs["features_by_target"].values() for feature in columns})
                if selected is None and kind == "correlation" and len(values) > 2:
                    selected = values[2]
                observations.append({"phase": phase[0], "kind": kind, "caller": caller.f_code.co_name,
                                     "population": population, "rows": row_count,
                                     "features": len(selected) if selected is not None else None,
                                     "return_type": kwargs.get("return_type"),
                                     "lazy_requested": kwargs.get("lazy", False)})
                return original(*values, **kwargs)

            setattr(owner, name, wrapper)

        observe(MarsBinnerBase, "transform", "transform")
        observe(MarsBinnerBase, "profile_bin_performance", "woe_preparation")
        for binner_class in (MarsNativeBinner, MarsOptimalBinner, MarsLiteOptBinner):
            observe(binner_class, "fit", "fit")
        observe(MarsStatsSelector, "_filter_corr", "correlation")
        observe(_risk_profile, "_prepare_raw_ks", "raw_ks")
        observe(_raw_ks, "_calculate_raw_ks", "raw_ks_calculation")

    output = Path(args.output)
    frame = _population(args.rows, args.features, args.segment_count, args.distribution)
    features = [name for name in frame.columns if name.startswith("feature_")]
    benchmark = frame.filter(pl.col("dataset") == "Train") if args.reference == "benchmark" else None
    process = psutil.Process()
    start_rss = process.memory_info().rss
    peaks = [start_rss, start_rss]
    done = threading.Event()
    stage_active = threading.Event()
    stage_lock = threading.Lock()
    stage_peaks = [0, 0]

    def memory_sample() -> tuple[int, int]:
        rss = process.memory_info().rss
        tree_rss = rss
        for child in process.children(recursive=True):
            try:
                tree_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return rss, tree_rss

    def sample() -> None:
        """10 ms 采样包含原生内存；共享页在树 RSS 中可能重复计数。"""
        while not done.wait(0.01):
            with stage_lock:
                rss, tree_rss = memory_sample()
                peaks[0] = max(peaks[0], rss)
                peaks[1] = max(peaks[1], tree_rss)
                if stage_active.is_set():
                    stage_peaks[0] = max(stage_peaks[0], rss)
                    stage_peaks[1] = max(stage_peaks[1], tree_rss)

    result: dict[str, Any] = {
        "status": "running", "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "source_root": args.source_root,
        "mars_module": mars_module, "mars_version": mars.__version__,
        "machine": {"logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                    "total_memory_bytes": psutil.virtual_memory().total},
        "versions": {name: importlib.metadata.version(name) for name in (
            "polars", "pandas", "numpy", "pyarrow", "scikit-learn", "scipy", "optbinning", "psutil",
        )},
        "threads": {name: os.environ.get(name) for name in THREAD_KEYS},
        "polars_thread_pool_size": pl.thread_pool_size(),
        "input": {"rows": frame.height, "features": len(features), "requested_segments": args.segment_count,
                  "distribution": args.distribution, "batch_size": args.batch_size,
                  "psi_batch_size": args.psi_batch_size if args.psi_batch_size is not None else 50,
                  "ks_method": args.ks_method,
                  "binning_scope": args.binning_scope, "seed": 20261029,
                  "benchmark_rows": benchmark.height if benchmark is not None else None, "observed_labels": frame["target"].count(),
                  "segment_counts": frame.group_by("customer_segment").len().sort("customer_segment").to_dicts(),
                  "identity": _signature(frame), "benchmark_identity": _signature(benchmark) if benchmark is not None else None},
        "measurement": {"interval_seconds": 0.01, "baseline": "imports and synthetic data creation excluded",
                        "public_chain": "profile_risk + profile_stats + StatsSelector.fit + all available selector reports + saves",
                        "limitations": "sampled RSS may miss transient peaks; public chain excludes subsequent fixed-rule evaluation"},
    }
    _write(output, result)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    start = time.perf_counter()
    try:
        phase[0] = "risk"
        stage_start = time.perf_counter()
        risk = profile_risk(
            frame, target="target", features=features, segment_col="customer_segment",
            group_col="dataset", benchmark_df=benchmark, binning_scope=args.binning_scope,
            n_bins=10, n_jobs=1, batch_size=args.batch_size, ks_method=args.ks_method,
        )
        risk_seconds = time.perf_counter() - stage_start
        phase[0] = "stats"
        stage_start = time.perf_counter()
        metrics = (["missing", "mean", "psi", "schema", "unseen"] if benchmark is not None
                   else ["missing", "mean", "psi"])
        if args.psi_batch_size is None:
            stats = profile_stats(
                frame, features=features, metrics=metrics, segment_col="customer_segment",
                group_col="dataset", benchmark_df=benchmark, binning_scope=args.binning_scope,
            )
        else:
            profiler = MarsDataProfiler(psi_batch_size=args.psi_batch_size, binning_scope=args.binning_scope)
            stats = profiler.generate_profile(
                frame, features=features, metrics=metrics, segment_col="customer_segment",
                group_col="dataset", benchmark_df=benchmark,
            )
        stats_seconds = time.perf_counter() - stage_start
        phase[0] = "selector"
        stage_start = time.perf_counter()
        # 所有业务阈值保留默认值，完整相关性仍使用各 scope 的全部剩余候选。
        selector = MarsStatsSelector(
            selection_scope="per_segment", binning_scope=args.binning_scope,
            batch_size=args.batch_size, n_jobs=1,
        ).fit(frame, target="target", features=features, segment_col="customer_segment",
              group_col="dataset", benchmark_df=benchmark)
        selector_seconds = time.perf_counter() - stage_start
        retained_reports: list[Any] = [risk.report, stats]
        scope_decisions: dict[str, Any] = {}
        for scope_row in selector.scopes_.to_dicts():
            identity = scope_row["scope_id"]
            child = selector.selectors_by_scope_.get(identity)
            if child is None:
                scope_decisions[identity] = scope_row
                continue
            if identity == "all":
                selection: dict[str, Any] = {}
            else:
                from mars.feature.binning._json_codec import decode_json_value

                selection = {"segments": [decode_json_value(json.loads(scope_row["segment_value"]))],
                             "include_all": False}
            decision = selector.get_report(**selection)
            scope_decisions[identity] = {
                "display_name": scope_row["display_name"], "selected": list(child.selected_features_),
                "funnel": list(child._funnel_stats), "decisions": _signature(decision),
            }
            if child.selected_features_:
                retained_reports.append(selector.get_binning_report(frame, benchmark_df=benchmark, **selection))
            if child._corr_report is not None:
                retained_reports.append(selector.get_correlation_report(**selection))
        artifact_dir = output.with_suffix("")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for index, report in enumerate(retained_reports):
            report.save(artifact_dir / f"report-{index}.marsreport", overwrite=True)
        risk.save_binning_rules(artifact_dir / "risk-rules.json")
        end_rss = process.memory_info().rss
        peaks[0] = max(peaks[0], end_rss)
        peaks[1] = max(peaks[1], end_rss)
        elapsed = time.perf_counter() - start
        public_peaks = peaks.copy()
        fixed_seconds = None
        fixed_summary = None
        fixed_detail = None
        fixed_memory = None
        if args.binning_scope == "shared":
            phase[0] = "fixed_rule_risk"
            with stage_lock:
                fixed_start_rss, fixed_start_tree = memory_sample()
                stage_peaks[:] = [fixed_start_rss, fixed_start_tree]
                stage_active.set()
            stage_start = time.perf_counter()
            fixed = MarsBinEvaluator().evaluate(
                frame, target="target", features=features, segment_col="customer_segment",
                group_col="dataset", benchmark_df=benchmark, binner=risk.binner,
                batch_size=args.batch_size,
            )
            fixed_seconds = time.perf_counter() - stage_start
            with stage_lock:
                fixed_end_rss, fixed_end_tree = memory_sample()
                stage_active.clear()
                stage_peaks[0] = max(stage_peaks[0], fixed_end_rss)
                stage_peaks[1] = max(stage_peaks[1], fixed_end_tree)
                fixed_memory = {
                    "start_rss_bytes": fixed_start_rss, "start_tree_rss_bytes": fixed_start_tree,
                    "end_rss_bytes": fixed_end_rss, "end_tree_rss_bytes": fixed_end_tree,
                    "sampled_peak_rss_bytes": stage_peaks[0], "sampled_tree_peak_rss_bytes": stage_peaks[1],
                    "peak_increment_bytes": stage_peaks[0] - fixed_start_rss,
                    "tree_peak_increment_bytes": stage_peaks[1] - fixed_start_tree,
                    "interval_seconds": 0.01,
                }
            fixed_summary = _signature(fixed.report.summary_table)
            fixed_detail = _signature(fixed.report.detail_table)
            retained_reports.append(fixed.report)
        result.update({
            "status": "passed", "seconds": elapsed, "rss_start_bytes": start_rss,
            "risk_seconds": risk_seconds, "stats_seconds": stats_seconds,
            "selector_seconds": selector_seconds, "fixed_rule_risk_seconds": fixed_seconds,
            "fixed_rule_risk_summary": fixed_summary, "fixed_rule_risk_detail": fixed_detail,
            "fixed_rule_memory": fixed_memory,
            "rss_end_bytes": end_rss, "sampled_peak_rss_bytes": public_peaks[0],
            "sampled_peak_increment_bytes": public_peaks[0] - start_rss, "sampled_tree_peak_rss_bytes": public_peaks[1],
            "windows_lifetime_peak_wset_bytes": getattr(process.memory_info(), "peak_wset", None),
            "threadpools": threadpool_info(), "selector_scopes": scope_decisions,
            "retained_report_count": len(retained_reports), "risk_rule_scope_count": len(risk.binners_by_scope),
            "risk_summary": _signature(risk.report.summary_table), "risk_detail": _signature(risk.report.detail_table),
            "stats_overview": _signature(stats.overview_table),
            "all_evidence": {
                "risk_summary": risk.report.summary_table.filter(pl.col("scope_id") == "all").sort("feature").to_dicts(),
                "stats_overview": stats.overview_table.filter(pl.col("scope_id") == "all").sort("feature").to_dicts(),
                "selected_features": selector.selected_features_,
            },
        })
    except Exception as exc:
        result.update(status="failed", exception=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        done.set()
        monitor.join()
        if args.execution_counts:
            result["diagnostic_seconds"] = diagnostic_seconds
            result["diagnostic_timing_limit"] = (
                "Instrumented cumulative wall times by phase; nested timings overlap and must not be summed. "
                "Report format excludes other assembly and saving; public-chain time includes both."
            )
            result["execution_observations"] = observations
            result["execution_observation_limit"] = (
                "Calls and eager input rows only; a LazyFrame can be collected multiple times. "
                "Population unknown where caller does not expose a clear distinction; timings are instrumented."
            )
        _write(output, _finite(result))
    return 0 if result["status"] == "passed" else 1


def _equivalent(left: Any, right: Any) -> bool:
    """同输入整体证据保留身份、顺序和 null，浮点只允许 1e-10 误差。"""
    if isinstance(left, float) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equivalent(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equivalent(a, b) for a, b in zip(left, right))
    return bool(left == right)


def _compare_saved_reports(left: Path, right: Path) -> list[dict[str, Any]]:
    """比较各快照全部表；保留 schema 与 null，浮点使用既有容差。"""
    from polars.testing import assert_frame_equal

    from mars.reporting import load_report

    results: list[dict[str, Any]] = []
    left_files = sorted(left.glob("report-*.marsreport"))
    right_files = sorted(right.glob("report-*.marsreport"))
    if not left_files or not right_files:
        return [{"status": "failed", "reason": "report inventory is empty"}]
    if [p.name for p in left_files] != [p.name for p in right_files]:
        return [{"status": "failed", "reason": "report inventory differs"}]
    for left_path, right_path in zip(left_files, right_files):
        reference = load_report(left_path)
        actual = load_report(right_path)
        reference_description = reference.describe()
        actual_description = actual.describe()
        reference_description.pop("report_id", None)
        actual_description.pop("report_id", None)
        for description in (reference_description, actual_description):
            description.get("parameters", {}).pop("generated_at_utc", None)
        reference_tables = reference_description.pop("tables")
        actual_tables = actual_description.pop("tables")
        results.append({"report": left_path.name, "part": "metadata", "status": "passed" if
                        _equivalent(reference_description, actual_description) else "failed"})
        if reference_tables != actual_tables:
            results.append({"report": left_path.name, "status": "failed", "reason": "table inventory differs"})
            continue
        for name in reference_tables:
            try:
                reference_frame = reference.get_table(name)
                actual_frame = actual.get_table(name)
                keys = [key for key in (
                    "scope_id", "target", "feature", "mars_group", "group", "bin_index", "date",
                    "metric", "feature_1", "feature_2", "stage", "reason",
                ) if key in reference_frame.columns]
                if keys and reference_frame.select(keys).is_duplicated().sum() == 0:
                    reference_frame = reference_frame.sort(keys)
                    actual_frame = actual_frame.sort(keys)
                assert_frame_equal(
                    reference_frame, actual_frame, rel_tol=1e-10, abs_tol=1e-10,
                )
            except (AssertionError, TypeError, ValueError) as exc:
                results.append({"report": left_path.name, "table": name, "status": "failed", "reason": str(exc)})
            else:
                results.append({"report": left_path.name, "table": name, "status": "passed"})
    reference_rules = json.loads((left / "risk-rules.json").read_text(encoding="utf-8"))
    actual_rules = json.loads((right / "risk-rules.json").read_text(encoding="utf-8"))
    results.append({"part": "risk_rules", "status": "passed" if _equivalent(reference_rules, actual_rules)
                    else "failed"})
    return results


def _main() -> int:
    """调度独立子进程；执行前设置线程环境，记录失败而不降低实际工作量。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--segments", type=int, nargs="+", default=[1, 3, 9])
    parser.add_argument("--distributions", nargs="+", choices=["balanced", "imbalanced"], default=["balanced", "imbalanced"])
    parser.add_argument("--binning-scopes", nargs="+", choices=["shared", "per_segment"], default=["shared"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--psi-batch-size", type=int, help="使用公开 MarsDataProfiler 指定画像 PSI 批宽")
    parser.add_argument("--execution-counts", action="store_true", help="记录调用和可见行数，不用于性能计时")
    parser.add_argument("--ks-method", choices=["binned", "raw"], default="binned")
    parser.add_argument("--output", default=".artifacts/segmented/resource/summary.json")
    parser.add_argument("--reference", choices=["benchmark", "first_group"], default="benchmark")
    parser.add_argument("--source-root", default=str(ROOT), help="被测源码根目录")
    parser.add_argument("--compare-to", help="同合同基线 summary.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--segment-count", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--distribution", default="balanced", help=argparse.SUPPRESS)
    parser.add_argument("--binning-scope", default="shared", help=argparse.SUPPRESS)
    args = parser.parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    if not source_root.is_dir() or not (source_root / "src" / "mars" / "__init__.py").is_file():
        parser.error(f"source-root must contain src/mars/__init__.py: {source_root}")
    args.source_root = str(source_root)
    if min(args.rows, args.features, args.threads, args.batch_size, *args.segments) < 1 or (
        args.psi_batch_size is not None and args.psi_batch_size < 1
    ):
        parser.error("rows, features, threads, batch sizes and segments must be positive")
    if args.worker:
        return _worker(args)
    destination = Path(args.output).resolve()
    case_names = [
        f"segments-{segment_count}-{distribution}-{mode}"
        for segment_count in args.segments
        for distribution in args.distributions
        for mode in args.binning_scopes
    ]
    if len(case_names) != len(set(case_names)):
        parser.error("duplicate case names would overwrite results")
    case_paths = [destination.parent / f"{name}.json" for name in case_names]
    if destination in case_paths or destination in [path.with_suffix(".log") for path in case_paths]:
        parser.error(f"summary and case output collide: {destination}")
    for path in case_paths:
        artifact_dir = path.with_suffix("")
        if destination.parent == artifact_dir and (
            destination.name == "risk-rules.json" or destination.match("report-*.marsreport")
        ):
            parser.error(f"summary and case artifact collide: {destination}")
    reference = None
    if args.compare_to:
        reference_path = Path(args.compare_to).resolve()
        try:
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            if not isinstance(reference, dict) or not isinstance(reference.get("cases"), list):
                raise ValueError("expected a summary object with cases")
            protected = {reference_path}
            for previous in reference["cases"]:
                evidence = Path(previous["evidence"]).resolve()
                protected.add(evidence)
                protected.add(Path(previous.get("log", evidence.with_suffix(".log"))).resolve())
                artifact_dir = evidence.with_suffix("")
                protected.update(path.resolve() for path in artifact_dir.glob("report-*.marsreport"))
                protected.add((artifact_dir / "risk-rules.json").resolve())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"invalid comparison summary {reference_path}: {exc}")
        outputs = {destination}
        for segment_count in args.segments:
            for distribution in args.distributions:
                for mode in args.binning_scopes:
                    case_path = destination.parent / f"segments-{segment_count}-{distribution}-{mode}.json"
                    outputs.add(case_path)
                    outputs.add(case_path.with_suffix(".log"))
                    artifact_dir = case_path.with_suffix("")
                    outputs.add(artifact_dir / "risk-rules.json")
                    outputs.update(artifact_dir.glob("report-*.marsreport"))
                    # 新报告名也可能覆盖基线中的报告。
                    for path in protected:
                        if path.parent == artifact_dir and path.match("report-*.marsreport"):
                            outputs.add(path)
        conflict = protected.intersection(path.resolve() for path in outputs)
        if conflict:
            parser.error(f"output would overwrite comparison evidence: {sorted(map(str, conflict))}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({name: str(args.threads) for name in THREAD_KEYS})
    environment["PYTHONPATH"] = str(source_root / "src")
    environment["PYTHONIOENCODING"] = "utf-8"
    cases: list[dict[str, Any]] = []
    for segment_count in args.segments:
        for distribution in args.distributions:
            for mode in args.binning_scopes:
                name = f"segments-{segment_count}-{distribution}-{mode}"
                output = destination.parent / f"{name}.json"
                output.unlink(missing_ok=True)
                artifact_dir = output.with_suffix("")
                for old_report in artifact_dir.glob("report-*.marsreport"):
                    old_report.unlink()
                (artifact_dir / "risk-rules.json").unlink(missing_ok=True)
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--rows", str(args.rows),
                           "--features", str(args.features), "--segment-count", str(segment_count),
                           "--distribution", distribution, "--binning-scope", mode,
                           "--batch-size", str(args.batch_size), "--reference", args.reference,
                           "--output", str(output), "--source-root", str(source_root)]
                if args.psi_batch_size is not None:
                    command.extend(["--psi-batch-size", str(args.psi_batch_size)])
                if args.execution_counts:
                    command.append("--execution-counts")
                command.extend(["--ks-method", args.ks_method])
                with output.with_suffix(".log").open("w", encoding="utf-8") as stream:
                    completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
                case: dict[str, Any] = {}
                reason = None
                if completed.returncode != 0:
                    reason = f"worker exited with code {completed.returncode}"
                else:
                    try:
                        loaded = json.loads(output.read_text(encoding="utf-8"))
                        if not isinstance(loaded, dict):
                            raise ValueError("worker result must be an object")
                        case = loaded
                        if case.get("status") != "passed":
                            raise ValueError(f"worker status is {case.get('status')!r}")
                        for field in ("input", "all_evidence"):
                            if not isinstance(case.get(field), dict) or not case[field]:
                                raise ValueError(f"worker result requires a nonempty {field} object")
                        if not isinstance(case["input"].get("identity"), dict) or not case["input"]["identity"]:
                            raise ValueError("worker result requires input identity")
                        for field in ("risk_summary", "stats_overview", "selected_features"):
                            if not isinstance(case["all_evidence"].get(field), list):
                                raise ValueError(f"worker result requires all_evidence.{field} list")
                        seconds = case.get("seconds")
                        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                                or not math.isfinite(seconds) or seconds < 0):
                            raise ValueError("worker result requires finite nonnegative seconds")
                        count = case.get("retained_report_count")
                        if isinstance(count, bool) or not isinstance(count, int):
                            raise ValueError("worker result requires retained_report_count")
                        saved_count = count - (1 if mode == "shared" else 0)
                        expected_reports = {f"report-{index}.marsreport" for index in range(saved_count)}
                        actual_reports = {path.name for path in artifact_dir.glob("report-*.marsreport")}
                        if (saved_count < 1 or expected_reports != actual_reports
                                or not (artifact_dir / "risk-rules.json").is_file()):
                            raise ValueError("worker report inventory or risk rules are incomplete")
                    except (OSError, ValueError) as exc:
                        reason = f"invalid worker result: {exc}"
                case.update(case=name, exit_code=completed.returncode, evidence=str(output),
                            log=str(output.with_suffix(".log")), status="failed" if reason else "passed")
                if reason:
                    case["failure_reason"] = reason
                cases.append(case)
                print(f"{name}: {case['status']}", flush=True)
    comparisons: list[dict[str, Any]] = []
    for left_index, left in enumerate(cases):
        if left["status"] != "passed":
            continue
        for right in cases[left_index + 1:]:
            if right["status"] != "passed":
                continue
            if left.get("input", {}).get("identity") != right.get("input", {}).get("identity"):
                continue
            if "all_evidence" in left and "all_evidence" in right:
                comparisons.append({"left": left["case"], "right": right["case"],
                                    "same_all_evidence": _equivalent(left["all_evidence"], right["all_evidence"])})
    report_comparisons: list[dict[str, Any]] = []
    comparison_module = None
    if args.compare_to:
        assert reference is not None
        sys.path.insert(0, str(source_root / "src"))
        try:
            import mars

            comparison_module = _check_mars_location(source_root, mars.__file__)
            reference_cases = {case["case"]: case for case in reference["cases"]}
            for case in cases:
                previous = reference_cases.get(case["case"])
                if (reference.get("status") != "passed" or previous is None or previous.get("status") != "passed"
                        or previous.get("exit_code") != 0 or case.get("status") != "passed"
                        or case.get("input") != previous.get("input")):
                    report_comparisons.append({"case": case["case"], "status": "incomparable",
                                               "reason": "input differs or a worker failed"})
                    continue
                tables = _compare_saved_reports(Path(previous["evidence"]).with_suffix(""), Path(case["evidence"]).with_suffix(""))
                report_comparisons.append({"case": case["case"], "status": "passed" if tables and all(
                    table["status"] == "passed" for table in tables
                ) else "failed", "tables": tables})
        except Exception as exc:
            report_comparisons.append({"status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                                       "traceback": traceback.format_exc()})
    passed = (all(case["status"] == "passed" for case in cases)
              and all(item["same_all_evidence"] for item in comparisons)
              and all(item["status"] == "passed" for item in report_comparisons))
    _write(destination, {"status": "passed" if passed else "failed", "cases": cases,
                         "same_input_comparisons": comparisons, "report_comparisons": report_comparisons,
                         "source_root": str(source_root),
                         "comparison_mars_module": comparison_module,
                         "large_data_validated": False})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
