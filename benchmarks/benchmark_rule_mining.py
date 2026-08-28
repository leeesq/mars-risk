"""比较 Mars 与 deimos-rule 来源快照的组合生成和评估性能。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

import psutil

if TYPE_CHECKING:
    import polars as pl


def build_wide_data(rows: int, features: int, seed: int) -> pl.DataFrame:
    """构造两个引擎共享的确定性宽表样本。"""
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(rows, features)).astype("float32")
    signal_index: int = min(5, features - 1)
    probability = 1 / (
        1
        + np.exp(
            -(
                -2.8
                + 3.0 * ((matrix[:, 0] < -1.1) & (matrix[:, signal_index] > 1.0))
                + 1.6 * (matrix[:, min(3, features - 1)] > 1.4)
            )
        )
    )
    data: Dict[str, Any] = {
        f"feat_{index}": matrix[:, index] for index in range(features)
    }
    data["target"] = rng.binomial(1, probability).astype("int8")
    return pl.DataFrame(data)


def _measure(operation: Callable[[], Tuple[int, int]]) -> Dict[str, Any]:
    """测量操作耗时、进程峰值 RSS 和结果校验值。"""
    process = psutil.Process(os.getpid())
    peak_rss: int = process.memory_info().rss
    stop = threading.Event()

    def sample_memory() -> None:
        """在操作运行期间采样当前进程 RSS。"""
        nonlocal peak_rss
        while not stop.wait(0.01):
            peak_rss = max(peak_rss, process.memory_info().rss)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        candidate_count, checksum = operation()
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        monitor.join()
        peak_rss = max(peak_rss, process.memory_info().rss)
    return {
        "seconds": elapsed,
        "peak_rss_mb": peak_rss / 1024**2,
        "candidate_count": candidate_count,
        "checksum": checksum,
    }


def run_mars(frame: pl.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    """运行 Mars 组合生成和固定长表评估。"""
    import polars as pl

    from mars.rule import MarsCombinationRuleGenerator, MarsRuleEvaluator, MarsRuleSet

    generator = MarsCombinationRuleGenerator(
        n_bins=args.n_bins,
        max_cross_features=2,
        max_candidates=args.max_candidates,
        random_state=args.seed,
        feature_prefilter_top_k=args.prefilter_top_k,
        feature_prefilter_min_features=args.prefilter_min_features,
        feature_prefilter_sample_size=args.prefilter_sample_size,
    )

    phases: Dict[str, float] = {}

    def operation() -> Tuple[int, int]:
        """生成规则并计算命中样本数校验值。"""
        generation_started: float = time.perf_counter()
        rules = generator.generate(frame, target="target")
        phases["generation_seconds"] = time.perf_counter() - generation_started
        evaluation_started: float = time.perf_counter()
        evaluation = MarsRuleEvaluator().evaluate(
            frame,
            MarsRuleSet(rules),
            target="target",
            batch_size=args.batch_size,
        )
        phases["evaluation_seconds"] = time.perf_counter() - evaluation_started
        checksum: int = int(
            evaluation.overall_table.filter(pl.col("group") == "hit")[
                "sample_count"
            ].sum()
            or 0
        )
        return len(rules), checksum

    result: Dict[str, Any] = _measure(operation)
    result["phases"] = phases
    return result


def run_deimos(frame: pl.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    """运行固定来源 checkout 的 deimos-rule 基线。"""
    baseline_root = Path(args.baseline_root).resolve()
    if not (baseline_root / "src" / "deimos").is_dir():
        raise FileNotFoundError(f"deimos-rule 源码目录无效：{baseline_root}")
    sys.path.insert(0, str(baseline_root / "src"))
    from deimos.evaluation.engine import DmRuleEvaluator
    from deimos.generation import combination as deimos_combination
    from deimos.generation.combination import DmCombinationGenerator

    from mars.rule import MarsCombinationRuleGenerator

    def compatible_mars_prefilter(
        *,
        df: pl.DataFrame,
        target: str,
        features: Any,
        top_k: int,
        sample_size: int,
        mars_kwargs: Any,
    ) -> list[str]:
        """适配来源提交所依赖的旧 MarsStatsSelector 构造签名。"""
        del mars_kwargs
        adapter = MarsCombinationRuleGenerator(
            random_state=args.seed,
            feature_prefilter_top_k=top_k,
            feature_prefilter_min_features=1,
            feature_prefilter_sample_size=sample_size,
        )
        return adapter._prefilter_features(df, target, list(features))

    deimos_combination.select_features_with_mars = compatible_mars_prefilter

    generator = DmCombinationGenerator(
        n_bins=args.n_bins,
        max_cross_features=2,
        max_candidates=args.max_candidates,
        random_state=args.seed,
        feature_prefilter_backend="mars",
        feature_prefilter_top_k=args.prefilter_top_k,
        feature_prefilter_min_features=args.prefilter_min_features,
        feature_prefilter_sample_size=args.prefilter_sample_size,
        mars_prefilter_kwargs={"suppress_output": True, "n_jobs": args.n_jobs},
    )

    def operation() -> Tuple[int, int]:
        """生成规则并计算命中样本数校验值。"""
        rules = generator.generate(frame, "target")
        evaluation = DmRuleEvaluator("target").evaluate_batch(
            frame,
            rules,
            batch_size=args.batch_size,
        )
        checksum: int = int(evaluation["total"].sum() or 0)
        return len(rules), checksum

    return _measure(operation)


def _child_command(engine: str, args: argparse.Namespace) -> list[str]:
    """构造隔离引擎进程的确定性命令。"""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--engine",
        engine,
        "--rows",
        str(args.rows),
        "--features",
        str(args.features),
        "--n-bins",
        str(args.n_bins),
        "--max-candidates",
        str(args.max_candidates),
        "--batch-size",
        str(args.batch_size),
        "--prefilter-top-k",
        str(args.prefilter_top_k),
        "--prefilter-min-features",
        str(args.prefilter_min_features),
        "--prefilter-sample-size",
        str(args.prefilter_sample_size),
        "--seed",
        str(args.seed),
        "--n-jobs",
        str(args.n_jobs),
        "--baseline-root",
        str(args.baseline_root),
    ]
    return command


def _run_child(engine: str, args: argparse.Namespace) -> Dict[str, Any]:
    """运行单引擎子进程并读取最后一行 JSON。"""
    completed = subprocess.run(
        _child_command(engine, args),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{engine} benchmark 子进程失败（exit={completed.returncode}）。\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return dict(json.loads(completed.stdout.strip().splitlines()[-1]))


def compare(args: argparse.Namespace) -> Dict[str, Any]:
    """隔离运行两个引擎并执行 15%/20% 退化门禁。"""
    mars_result = _run_child("mars", args)
    time.sleep(args.cooldown_seconds)
    deimos_result = _run_child("deimos", args)
    return gate_results(mars_result, deimos_result)


def gate_results(
    mars_result: Dict[str, Any],
    deimos_result: Dict[str, Any],
) -> Dict[str, Any]:
    """比较两个隔离测量结果并返回发布门禁载荷。"""
    for name, result in (("mars", mars_result), ("deimos", deimos_result)):
        missing = {
            "seconds",
            "peak_rss_mb",
            "candidate_count",
            "rows",
            "features",
        } - set(result)
        if missing:
            raise ValueError(f"{name} benchmark 结果缺少字段：{sorted(missing)}。")
    if (mars_result["rows"], mars_result["features"]) != (
        deimos_result["rows"],
        deimos_result["features"],
    ):
        raise ValueError("Mars 与 deimos benchmark 工作负载不一致。")
    time_ratio: float = mars_result["seconds"] / deimos_result["seconds"]
    memory_ratio: float = mars_result["peak_rss_mb"] / deimos_result["peak_rss_mb"]
    payload: Dict[str, Any] = {
        "workload": {"rows": mars_result["rows"], "features": mars_result["features"]},
        "mars": mars_result,
        "deimos_commit": "e6714c5e795054e44f0c58ad7097668b4117b4a2",
        "deimos": deimos_result,
        "time_ratio": time_ratio,
        "memory_ratio": memory_ratio,
        "passed": time_ratio <= 1.15 and memory_ratio <= 1.20,
    }
    return payload


def _read_result(path: Path) -> Dict[str, Any]:
    """读取单引擎 benchmark JSON 对象。"""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark 结果必须是 JSON 对象：{path}。")
    return dict(payload)


def parse_args() -> argparse.Namespace:
    """解析规则性能 smoke 与发布前对比参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=["mars", "deimos", "compare", "gate"],
        default="mars",
    )
    parser.add_argument("--rows", type=int, default=2_000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--prefilter-top-k", type=int, default=300)
    parser.add_argument("--prefilter-min-features", type=int, default=500)
    parser.add_argument("--prefilter-sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--baseline-root", type=Path, default=Path("../deimos-rule"))
    parser.add_argument("--mars-result", type=Path)
    parser.add_argument("--deimos-result", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    """执行指定规则 benchmark 引擎。"""
    args = parse_args()
    if args.rows < 1 or args.features < 1 or args.max_candidates < 1:
        raise ValueError("rows、features 和 max_candidates 必须至少为 1。")
    if args.engine == "gate":
        if args.mars_result is None or args.deimos_result is None:
            raise ValueError("gate 模式必须同时提供 --mars-result 和 --deimos-result。")
        result = gate_results(
            _read_result(args.mars_result),
            _read_result(args.deimos_result),
        )
    elif args.engine == "compare":
        result = compare(args)
    else:
        frame = build_wide_data(args.rows, args.features, args.seed)
        result = run_mars(frame, args) if args.engine == "mars" else run_deimos(frame, args)
        result.update({"engine": args.engine, "rows": args.rows, "features": args.features})
    serialized: str = json.dumps(result, ensure_ascii=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(serialized)
    if args.engine in {"compare", "gate"} and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
