"""测量规则 evaluator、IoU、analysis 与 cascade 子阶段性能。"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import polars as pl
import psutil

from mars.rule import MarsRule, MarsRuleEvaluator, MarsRuleMiningSpec, MarsRuleSet, mine_rules
from mars.rule.analysis import analyze_rule_set
from mars.rule.workflow import _iou_deduplicate


def _build_workload(rows: int, rule_count: int, seed: int) -> Tuple[pl.DataFrame, List[MarsRule]]:
    """构造确定性规则子阶段宽表和阈值规则。"""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=rows)
    target = ((values > 1.0) | (rng.random(rows) < 0.03)).astype("int8")
    slices = np.asarray([f"2026-{index % 6 + 1:02d}-15" for index in range(rows)])
    frame = pl.DataFrame(
        {
            "x": values,
            "target": target,
            "month": slices,
            "amount": rng.uniform(10.0, 1000.0, size=rows),
            "customer": [f"c{index // 2}" for index in range(rows)],
        }
    )
    thresholds = np.linspace(-2.5, 2.5, num=rule_count)
    rules = [MarsRule(f"x >= {float(value)!r}") for value in thresholds]
    return frame, rules


def _measure(operation: Callable[[], int], repeats: int) -> Dict[str, Any]:
    """返回多轮中位耗时、最大 RSS 和结果校验值。"""
    timings: List[float] = []
    peaks: List[float] = []
    checksums: List[int] = []
    process = psutil.Process()
    for _ in range(repeats):
        peak_rss: int = process.memory_info().rss
        stop = threading.Event()

        def sample_memory(stop_event: threading.Event) -> None:
            """采样单轮 RSS 峰值。"""
            nonlocal peak_rss
            while not stop_event.wait(0.005):
                peak_rss = max(peak_rss, process.memory_info().rss)

        monitor = threading.Thread(target=sample_memory, args=(stop,), daemon=True)
        monitor.start()
        started: float = time.perf_counter()
        try:
            checksums.append(operation())
        finally:
            timings.append(time.perf_counter() - started)
            stop.set()
            monitor.join()
            peaks.append(max(peak_rss, process.memory_info().rss) / 1024**2)
    if len(set(checksums)) != 1:
        raise RuntimeError(f"性能 workload 校验值不稳定：{checksums}。")
    return {
        "median_seconds": statistics.median(timings),
        "peak_rss_mb": max(peaks),
        "repeats": repeats,
        "checksum": checksums[0],
    }


def run_stage(
    stage: str,
    frame: pl.DataFrame,
    rules: List[MarsRule],
    repeats: int,
    batch_size: int,
) -> Dict[str, Any]:
    """运行单个规则子阶段 workload。"""
    rule_set = MarsRuleSet(rules)
    if stage == "evaluator":
        return _measure(
            lambda: MarsRuleEvaluator()
            .evaluate(
                frame,
                rule_set,
                target="target",
                time_col="month",
                time_grain="month",
                batch_size=batch_size,
            )
            .slice_table.height,
            repeats,
        )
    if stage == "iou":
        return _measure(
            lambda: len(
                _iou_deduplicate(
                    frame,
                    rules,
                    0.3,
                    batch_size=batch_size,
                )[0]
            ),
            repeats,
        )
    if stage == "analysis":
        return _measure(
            lambda: analyze_rule_set(
                rule_set,
                frame,
                target="target",
                amount_col="amount",
                customer_col="customer",
                max_pairs=min(5000, len(rules) * (len(rules) - 1) // 2),
            ).interaction_table.height,
            repeats,
        )
    cascade_rules: List[str] = [rule.expression for rule in rules[-min(10, len(rules)) :]]
    return _measure(
        lambda: len(
            mine_rules(
                frame,
                target="target",
                validation_df=frame,
                seed_rules=cascade_rules,
                generators=[],
                spec=MarsRuleMiningSpec(
                    selection_strategy="cascade",
                    max_rounds=3,
                    top_k=3,
                    iou_threshold=1.0,
                    batch_size=batch_size,
                    iou_batch_size=batch_size,
                ),
            ).rule_set.rules
        ),
        repeats,
    )


def gate_results(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    """执行 30% 提速和 10% 内存退化门禁。"""
    stages: Dict[str, Any] = {}
    passed: bool = True
    for stage, current_result in current["stages"].items():
        baseline_result: Dict[str, Any] = baseline["stages"][stage]
        speedup: float = 1.0 - (
            current_result["median_seconds"] / baseline_result["median_seconds"]
        )
        memory_ratio: float = (
            current_result["peak_rss_mb"] / baseline_result["peak_rss_mb"]
        )
        stage_passed: bool = speedup >= 0.30 and memory_ratio <= 1.10
        stages[stage] = {
            "speedup": speedup,
            "memory_ratio": memory_ratio,
            "passed": stage_passed,
        }
        passed = passed and stage_passed
    return {"passed": passed, "stages": stages}


def parse_args() -> argparse.Namespace:
    """解析子阶段性能 benchmark 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["all", "evaluator", "iou", "analysis", "cascade"],
        default="all",
    )
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--rules", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    """运行子阶段 benchmark 和可选发布门禁。"""
    args = parse_args()
    if min(args.rows, args.rules, args.repeats, args.batch_size) < 1:
        raise ValueError("rows、rules、repeats 和 batch-size 必须至少为 1。")
    frame, rules = _build_workload(args.rows, args.rules, args.seed)
    stage_names: List[str] = (
        ["evaluator", "iou", "analysis", "cascade"]
        if args.stage == "all"
        else [args.stage]
    )
    result: Dict[str, Any] = {
        "workload": {"rows": args.rows, "rules": args.rules},
        "stages": {
            stage: run_stage(stage, frame, rules, args.repeats, args.batch_size)
            for stage in stage_names
        },
    }
    if args.baseline_json is not None:
        baseline: Dict[str, Any] = json.loads(
            args.baseline_json.read_text(encoding="utf-8")
        )
        result["gate"] = gate_results(result, baseline)
    serialized: str = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if "gate" in result and not result["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
