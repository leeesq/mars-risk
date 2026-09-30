"""完整分析链路基准；每次调用在独立进程中测量，产物仅写入指定临时文件。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
parser.add_argument("--rows", type=int, default=20000)
parser.add_argument("--features", type=int, default=201)
parser.add_argument("--batch-size", type=int, default=25)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--stage", choices=["all", "profile"], default="all")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--tables-output", type=Path)
args = parser.parse_args()
os.environ["POLARS_MAX_THREADS"] = str(args.threads)
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(args.source.resolve()))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import psutil  # noqa: E402
from benchmark_binning_speed import MemorySampler, make_matrix  # noqa: E402

import mars.analysis.evaluator as evaluator_module  # noqa: E402
from mars import __version__ as mars_version  # noqa: E402
from mars.analysis import MarsBinEvaluator, MarsDataProfiler  # noqa: E402
from mars.feature.binning.base import MarsBinnerBase  # noqa: E402


def timed(function: Callable[..., Any], name: str, stages: dict[str, float]) -> Callable[..., Any]:
    """只记录已有调用的墙钟耗时，不改变计算及返回值。"""

    def wrapped(*values: Any, **options: Any) -> Any:
        start = time.perf_counter()
        try:
            return function(*values, **options)
        finally:
            stages[name] += time.perf_counter() - start

    return wrapped


def save_tables(tables: dict[str, pl.DataFrame]) -> None:
    """按需保存小结果表，供前后版本做数值比较。"""
    if args.tables_output is None:
        return
    args.tables_output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.write_parquet(args.tables_output / f"{name}.parquet")


def main() -> dict[str, Any]:
    """准备数据后测量完整评估与画像，保留环境、阶段和内存口径。"""
    matrix, target, features = make_matrix(args.rows, args.features, args.seed)
    rng = np.random.default_rng(args.seed)
    frame = pl.DataFrame(matrix, schema=features).with_columns(
        pl.Series("target", target),
        pl.Series("group", np.arange(args.rows) % 4),
        pl.Series("weight", rng.uniform(0.5, 2, args.rows)),
        pl.Series("amount", rng.uniform(100, 1000, args.rows)),
    )
    benchmark = frame.head(args.rows // 2) if args.benchmark else None
    del matrix
    if args.stage == "profile":
        initial_rss = psutil.Process().memory_info().rss / (1024 * 1024)
        start = time.perf_counter()
        with MemorySampler(interval_seconds=0.01) as sampler:
            profile = MarsDataProfiler(overview_batch_size=args.batch_size).generate_profile(
                frame,
                features=features,
                group_col="group",
                enable_sparkline=False,
                metrics=["missing", "zeros", "unique", "mode", "mean", "std", "min", "max"],
            )
        return {
            "profile_seconds": time.perf_counter() - start,
            "input_ready_rss_mb": initial_rss,
            "profile_peak_rss_mb": initial_rss + sampler.stats().peak_delta_mb,
            "profile_peak_increment_mb": sampler.stats().peak_delta_mb,
            "overview_rows": len(profile.overview_table),
        }
    stages: dict[str, float] = defaultdict(float)
    for name, stage in [
        ("build_binner", "fit"),
        ("aggregate_basic_stats", "aggregate"),
        ("get_benchmark_dist", "benchmark"),
        ("calculate_metrics_from_stats", "metrics"),
    ]:
        setattr(evaluator_module, name, timed(getattr(evaluator_module, name), stage, stages))
    MarsBinnerBase.transform = timed(MarsBinnerBase.transform, "transform", stages)
    MarsBinEvaluator._format_report = timed(MarsBinEvaluator._format_report, "report", stages)
    initial_rss = psutil.Process().memory_info().rss / (1024 * 1024)
    start = time.perf_counter()
    with MemorySampler(interval_seconds=0.01) as sampler:
        risk = MarsBinEvaluator(binner_params={"n_bins": 8, "special_values": [-999.0]}).evaluate(
            frame,
            features=features,
            target="target",
            group_col="group",
            weights_col="weight",
            amount_col="amount",
            benchmark_df=benchmark,
            batch_size=args.batch_size,
        )
    evaluation_seconds = time.perf_counter() - start
    evaluation_memory = sampler.stats()
    profiler_start_rss = psutil.Process().memory_info().rss / (1024 * 1024)
    start = time.perf_counter()
    with MemorySampler(interval_seconds=0.01) as profiler_sampler:
        profile = MarsDataProfiler(overview_batch_size=args.batch_size).generate_profile(
            frame,
            features=features,
            group_col="group",
            enable_sparkline=False,
            metrics=["missing", "zeros", "unique", "mode", "mean", "std", "min", "max"],
        )
    result = {
        "environment": {
            "python": sys.version,
            "polars": pl.__version__,
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "threads": pl.thread_pool_size(),
        },
        "workload": {
            "rows": args.rows,
            "features": args.features,
            "seed": args.seed,
            "benchmark": args.benchmark,
            "batch_size": args.batch_size,
        },
        "evaluation_seconds": evaluation_seconds,
        "stages_seconds": dict(stages),
        "input_ready_rss_mb": initial_rss,
        "evaluation_peak_rss_mb": initial_rss + evaluation_memory.peak_delta_mb,
        "evaluation_peak_increment_mb": evaluation_memory.peak_delta_mb,
        "profile_seconds": time.perf_counter() - start,
        "profile_peak_rss_mb": profiler_start_rss + profiler_sampler.stats().peak_delta_mb,
        "profile_peak_increment_mb": profiler_sampler.stats().peak_delta_mb,
        "summary_rows": len(risk.report.summary_table),
        "overview_rows": len(profile.overview_table),
    }
    save_tables(
        {
            "risk.summary": risk.report.summary_table,
            "risk.detail": risk.report.detail_table,
            **{f"risk.trend.{k}": v for k, v in risk.report.trend_tables.items()},
            "profile.overview": profile.overview_table,
            **{f"profile.dq.{k}": v for k, v in profile.dq_tables.items()},
            **{f"profile.stats.{k}": v for k, v in profile.stats_tables.items()},
        }
    )
    return result


if __name__ == "__main__":
    process_start_rss = psutil.Process().memory_info().rss / (1024 * 1024)
    with MemorySampler(interval_seconds=0.01) as process_sampler:
        result = main()
    result["total_peak_rss_mb"] = max(
        process_start_rss + process_sampler.stats().peak_delta_mb,
        result.get("evaluation_peak_rss_mb", 0),
        result.get("profile_peak_rss_mb", 0),
    )
    result["environment"] = {
        "python": sys.version,
        "polars": pl.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "threads": pl.thread_pool_size(),
        "mars": mars_version,
        "source": evaluator_module.__file__,
    }
    result["workload"] = {
        "rows": args.rows,
        "features": args.features,
        "seed": args.seed,
        "benchmark": args.benchmark,
        "batch_size": args.batch_size,
        "stage": args.stage,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
