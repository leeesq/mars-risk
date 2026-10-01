"""分阶段测量完整相关性结果、宽表投影和保存后回放，不拟合监督分箱。"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import psutil

from mars.analysis import cross_scores, evaluate_score_policy, show_score_matrix
from mars.analysis import score_cross as score_cross_module
from mars.feature import MarsLinearSelector
from mars.reporting import (
    get_correlation_matrix,
    get_related_features,
    load_report,
    show_correlation_matrix,
)


def measure(call: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """单次阶段测量，10ms RSS 采样包含原生分配；不宣称精确分配峰值。"""
    gc.collect()
    process = psutil.Process()
    before = process.memory_info().rss
    peak = [before]
    done = threading.Event()

    def sample() -> None:
        """记录观测 RSS 峰值，阶段结束立即停止。"""
        while not done.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    start = time.perf_counter()
    try:
        result = call()
    finally:
        elapsed = time.perf_counter() - start
        peak[0] = max(peak[0], process.memory_info().rss)
        done.set()
        worker.join()
    return result, {
        "seconds": elapsed,
        "rss_baseline_bytes": before,
        "rss_observed_peak_bytes": peak[0],
        "rss_observed_delta_bytes": peak[0] - before,
    }


def correlation_case(p: int, folder: Path, rng: np.random.Generator) -> dict[str, Any]:
    """分开记录计算、规范存储构造、保存、加载、有限查询阶段。"""
    features = [f"f{i:04d}" for i in range(p)]
    values = rng.normal(size=(1500, p))
    matrix, compute = measure(lambda values=values: np.corrcoef(values, rowvar=False))
    del values
    selector = MarsLinearSelector()
    selector._reset_correlation()
    selector._corr_candidates = features
    selector._corr_input_features = features
    selector._corr_matrix = matrix
    selector._corr_parameters = {
        "representation": "raw",
        "method": "pearson",
        "input_row_count": 1500,
        "correlation_row_count": 1500,
        "benchmark_only": "engine result storage; no selector binning/logit",
    }
    selector._corr_status = "computed"
    _, construction = measure(selector._finish_correlation)
    del matrix
    selector._is_fitted = True
    report = selector.get_correlation_report()
    path = folder / f"correlation-{p}.marsreport"
    _, save = measure(lambda: report.save(path))
    restored, load = measure(lambda: load_report(path))
    _, peers = measure(lambda: get_related_features(restored, features[0], limit=20))
    _, submatrix = measure(lambda: get_correlation_matrix(restored, features[:30]))
    _, display = measure(lambda: show_correlation_matrix(restored, max_features=30).to_html())
    return {
        "rows": 1500,
        "features": p,
        "pairs": p * (p - 1) // 2,
        "compute": compute,
        "construction": construction,
        "save": save,
        "load": load,
        "peers20": peers,
        "submatrix30": submatrix,
        "matrix_display30_html": display,
        "file_bytes": path.stat().st_size,
    }


def cross_case(rows: int, width: int, folder: Path, rng: np.random.Generator) -> dict[str, Any]:
    """宽 Pandas 边界实测投影；删除原表后仅加载快照展示与回放。"""
    x = rng.normal(size=rows)
    y = rng.normal(size=rows)
    data = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "bad": (rng.random(rows) < 0.1).astype(np.int8),
            "late": (rng.random(rows) < 0.15).astype(np.int8),
            "split": np.where(np.arange(rows) % 3, "TEST", "OOT"),
            **{f"unused_{i}": np.zeros(rows, dtype=np.uint8) for i in range(width)},
        }
    )
    columns_seen: list[list[str]] = []
    construction_stages: list[dict[str, Any]] = []
    fit_stages: list[dict[str, Any]] = []
    original = pl.from_pandas
    original_factory = score_cross_module._result_report
    original_fit = score_cross_module._fit_axis

    def factory(*args: Any, **kwargs: Any) -> Any:
        """单独测量公共结果目录与快照构造，区分聚合开销。"""
        result, measurement = measure(lambda: original_factory(*args, **kwargs))
        construction_stages.append(measurement)
        return result

    def fit(*args: Any, **kwargs: Any) -> Any:
        """实测各轴只拟合一次的阶段和调用数。"""
        result, measurement = measure(lambda: original_fit(*args, **kwargs))
        fit_stages.append(measurement)
        return result

    def capture(frame: pd.DataFrame, **kwargs: Any) -> pl.DataFrame:
        """记录实际跨引擎投影字段，而非仅记录预期字段。"""
        columns_seen.append(list(frame.columns))
        return original(frame, **kwargs)

    with patch.object(pl, "from_pandas", capture), patch.object(
        score_cross_module, "_result_report", factory
    ), patch.object(score_cross_module, "_fit_axis", fit):
        report, aggregate = measure(
            lambda data=data: cross_scores(
                data,
                score_x="x",
                score_y="y",
                targets=["bad", "late"],
                score_directions={"x": "lower_risk", "y": "higher_risk"},
                group_col="split",
            )
        )
    del data, x, y
    path = folder / f"cross-{rows}-{width}.marsreport"
    _, save = measure(lambda report=report: report.save(path))
    del report
    restored, load = measure(lambda: load_report(path))
    _, query = measure(
        lambda: restored.get_table("cells", filters={"target": "bad", "group": "OOT"}, limit=20)
    )
    _, replay = measure(
        lambda: evaluate_score_policy(
            restored,
            {"type": "and", "x_max_risk_rank": 3, "y_max_risk_rank": 3},
            baseline={"type": "x_only", "x_max_risk_rank": 3},
        )
    )
    _, display = measure(
        lambda: show_score_matrix(restored, filters={"target": "bad", "group": "OOT"}).to_html()
    )
    return {
        "rows": rows,
        "unused_columns": width,
        "input_columns": width + 5,
        "actual_converted_columns": columns_seen,
        "fit_and_joint_aggregate": aggregate,
        "axis_fits": fit_stages,
        "report_construction": construction_stages,
        "phase_note": "outer total includes nested fit/construction timers and their RSS sampler setup; other processing includes projection, assignment, aggregate and marginals",
        "save": save,
        "load": load,
        "query20": query,
        "replay": replay,
        "notebook_html": display,
        "raw_input_deleted_before_load_and_replay": True,
    }


def main() -> None:
    """运行可复现测量，并记录环境和实际矩阵调用次数。"""
    rng = np.random.default_rng(20261001)
    data = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"f{i}" for i in range(12)])
    calls: list[list[str]] = []
    original = pd.DataFrame.corr
    target = rng.integers(0, 2, size=200)

    def count(frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """只计数特征间 DataFrame 矩阵；不包含 Series 与 target 的关联。"""
        calls.append(list(frame.columns))
        return original(frame, **kwargs)

    with patch.object(pd.DataFrame, "corr", count):
        current = MarsLinearSelector().fit(data, target)
    assert len(calls) == 1
    current_calls = len(calls)
    # 固定改造前版本，提交本次实现后仍能复现同一对照。
    revision = "4b3c88f5af3145619a35028202356d702e83cfba"
    baseline_source = subprocess.run(
        ["git", "show", f"{revision}:src/mars/feature/selection/linear.py"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    namespace: dict[str, Any] = {"__name__": "mars.feature.selection._benchmark_baseline"}
    exec(compile(baseline_source, f"{revision}/linear.py", "exec"), namespace)
    calls.clear()
    with patch.object(pd.DataFrame, "corr", count):
        baseline = namespace["MarsLinearSelector"]().fit(data, target)
    assert len(calls) == 2 and baseline.selected_features_ == current.selected_features_
    output: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "polars": pl.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "pyarrow": importlib.metadata.version("pyarrow"),
            "polars_threads": pl.thread_pool_size(),
            "seed": 20261001,
            "matrix_calls_linear_fit": current_calls,
            "baseline_matrix_calls_linear_fit": len(calls),
            "baseline_revision": revision,
            "baseline_selected_order_matches": baseline.selected_features_
            == current.selected_features_,
            "blas_threads_env": os.getenv("OPENBLAS_NUM_THREADS"),
            "rss_sampling": "10ms process RSS; observed peak, allocator reuse affects deltas; single run, no speedup claim",
            "package_version": "0.0.28",
            "worktree": "uncommitted correlation/score-cross implementation",
        }
    }
    with TemporaryDirectory(prefix="mars-scores-") as temporary:
        folder = Path(temporary)
        output["correlation"] = [correlation_case(p, folder, rng) for p in (500, 1000)]
        output["cross"] = [cross_case(200000, 50, folder, rng), cross_case(1000000, 0, folder, rng)]
    destination = Path("benchmarks/results/correlation_score_cross_20261001.json")
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
