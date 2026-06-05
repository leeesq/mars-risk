"""分箱器速度对比脚本。

该脚本只用于手动复现 README 中的性能表，不作为 pytest 或 CI 的一部分。

默认分为两类 benchmark：

* ``native``：对比 toad 与 ``MarsNativeBinner`` 的等频/等宽分箱，默认
  200,000 行、3,000 个数值特征。每个策略固定先运行 toad，释放内存后再运行 MARS。
* ``optimal``：保持上一版最优分箱对比口径，默认 50,000 行、1,000 个数值特征，
  对比 ``MarsOptimalBinner`` 与 ``optbinning.BinningProcess``。
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
import polars as pl
import psutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mars.feature import MarsNativeBinner, MarsOptimalBinner  # noqa: E402

NativeMethod = Literal["quantile", "uniform"]

BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class PolarsBenchmarkData:
    """保存 MARS benchmark 使用的 Polars 样本。"""

    frame: pl.DataFrame
    target: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class PandasBenchmarkData:
    """保存 toad/optbinning benchmark 使用的 Pandas 样本。"""

    frame: pd.DataFrame
    target: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class BenchmarkResult:
    """保存单个方法的多轮耗时和内存结果。"""

    scenario: str
    name: str
    timings: list[float]
    memory_deltas_mb: list[float]
    peak_memory_deltas_mb: list[float]
    checksum: float
    note: str = ""

    @property
    def avg(self) -> float:
        """返回平均耗时。"""
        return float(np.mean(self.timings))

    @property
    def best(self) -> float:
        """返回最短耗时。"""
        return float(np.min(self.timings))

    @property
    def worst(self) -> float:
        """返回最长耗时。"""
        return float(np.max(self.timings))

    @property
    def avg_memory_delta_mb(self) -> float:
        """返回本轮结束后的平均 RSS 增量。"""
        return float(np.mean(self.memory_deltas_mb))

    @property
    def peak_memory_delta_mb(self) -> float:
        """返回采样期间观察到的最大 RSS 峰值增量。"""
        return float(np.max(self.peak_memory_deltas_mb))


@dataclass(frozen=True)
class MemoryStats:
    """保存单轮 benchmark 的内存增量。"""

    end_delta_mb: float
    peak_delta_mb: float


class MemorySampler:
    """以后台线程采样当前 Python 进程树的 RSS 内存。"""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self._interval_seconds = interval_seconds
        self._process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples_mb: list[float] = []
        self._start_mb = 0.0
        self._end_mb = 0.0

    def __enter__(self) -> MemorySampler:
        """开始采样。"""
        self._start_mb = self._rss_mb()
        self._samples_mb = [self._start_mb]
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._collect,
            name="mars-benchmark-memory-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """停止采样并记录结束 RSS。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._end_mb = self._rss_mb()
        self._samples_mb.append(self._end_mb)

    def stats(self) -> MemoryStats:
        """返回结束增量和峰值增量。"""
        peak_mb = max(self._samples_mb) if self._samples_mb else self._end_mb
        return MemoryStats(
            end_delta_mb=self._end_mb - self._start_mb,
            peak_delta_mb=max(peak_mb - self._start_mb, 0.0),
        )

    def _collect(self) -> None:
        """按固定间隔记录 RSS，直到主线程通知停止。"""
        while not self._stop_event.wait(self._interval_seconds):
            self._samples_mb.append(self._rss_mb())

    def _rss_mb(self) -> float:
        """读取当前进程及其子进程 RSS，单位为 MB。"""
        rss_bytes = 0
        processes = [self._process, *self._process.children(recursive=True)]
        for process in processes:
            try:
                rss_bytes += process.memory_info().rss
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                # 子进程可能在采样瞬间退出，跳过即可，不影响峰值趋势判断。
                continue
        return rss_bytes / BYTES_PER_MB


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="对比 MARS、toad 和 optbinning 的分箱速度。")
    subparsers = parser.add_subparsers(dest="command")

    native = subparsers.add_parser("native", help="对比 toad 与 MarsNativeBinner 的等频/等宽性能。")
    native.add_argument("--rows", type=int, default=200_000, help="合成样本行数。")
    native.add_argument("--features", type=int, default=3_000, help="合成数值特征数。")
    native.add_argument("--repeats", type=int, default=1, help="每个方法重复计时次数。")
    native.add_argument("--seed", type=int, default=2026, help="随机种子。")
    native.add_argument("--bins", type=int, default=8, help="最大分箱数。")

    optimal = subparsers.add_parser("optimal", help="对比 MarsOptimalBinner 与 optbinning。")
    optimal.add_argument("--rows", type=int, default=50_000, help="合成样本行数。")
    optimal.add_argument("--features", type=int, default=1_000, help="合成数值特征数。")
    optimal.add_argument("--repeats", type=int, default=3, help="每个方法重复计时次数。")
    optimal.add_argument("--seed", type=int, default=2026, help="随机种子。")
    optimal.add_argument("--bins", type=int, default=8, help="最大分箱数。")
    optimal.add_argument("--opt-time-limit", type=int, default=1, help="最优分箱单特征求解秒数上限。")

    parser.set_defaults(command="native")
    return parser.parse_args()


def make_matrix(rows: int, features: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """构造包含稳定、漂移、缺失、特殊值和噪声特征的共享宽表矩阵。"""
    rng = np.random.default_rng(seed)
    feature_names = [f"feat_{idx:04d}" for idx in range(features)]
    matrix = rng.normal(loc=0.0, scale=1.0, size=(rows, features)).astype(np.float32)

    block = max(features // 5, 1)
    row_drift = np.linspace(0.0, 0.8, rows, dtype=np.float32).reshape(-1, 1)
    matrix[:, block : 2 * block] += row_drift

    missing_block = matrix[:, 2 * block : 3 * block]
    missing_mask = rng.random(missing_block.shape) < 0.02
    missing_block[missing_mask] = np.nan

    special_block = matrix[:, 3 * block : 4 * block]
    special_mask = rng.random(special_block.shape) < 0.01
    special_block[special_mask] = -999.0

    signal_width = min(20, features)
    weights = np.linspace(1.2, -0.8, signal_width, dtype=np.float32)
    signal_matrix = np.nan_to_num(matrix[:, :signal_width], nan=0.0, posinf=0.0, neginf=0.0)
    raw_score = signal_matrix @ weights + rng.normal(scale=0.7, size=rows)
    target = (raw_score > np.median(raw_score)).astype(np.int32)
    return matrix, target, feature_names


def build_pandas_data(rows: int, features: int, seed: int) -> PandasBenchmarkData:
    """构造 Pandas 样本，供 toad 或 optbinning 使用。"""
    matrix, target, feature_names = make_matrix(rows=rows, features=features, seed=seed)
    frame = pd.DataFrame(matrix, columns=feature_names, copy=False)
    return PandasBenchmarkData(frame=frame, target=target, feature_names=feature_names)


def build_polars_data(rows: int, features: int, seed: int) -> PolarsBenchmarkData:
    """构造 Polars 样本，供 MARS 使用。"""
    matrix, target, feature_names = make_matrix(rows=rows, features=features, seed=seed)
    frame = pl.DataFrame({name: matrix[:, idx] for idx, name in enumerate(feature_names)})
    del matrix
    gc.collect()
    return PolarsBenchmarkData(frame=frame, target=target, feature_names=feature_names)


def release_memory(*objects: object) -> None:
    """释放一次 benchmark 阶段中不再需要的大对象。"""
    del objects
    gc.collect()
    shutdown_parallel_workers()
    gc.collect()


def shutdown_parallel_workers() -> None:
    """关闭 joblib/loky 复用 worker，避免下一轮内存基线被污染。"""
    try:
        from joblib.externals.loky import get_reusable_executor
    except ImportError:
        return

    executor = get_reusable_executor()
    executor.shutdown(wait=True, kill_workers=True)


def consume_frame(frame: object) -> float:
    """轻量消费转换结果，防止被解释器或后端延迟执行绕过。"""
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if isinstance(frame, pl.DataFrame):
        first_col = frame.columns[0]
        return float(frame.select(pl.col(first_col).sum()).item())
    if isinstance(frame, pd.DataFrame):
        return float(frame.iloc[:, 0].sum())
    array = np.asarray(frame)
    return float(np.nansum(array[:, 0]))


def benchmark_toad_native(data: PandasBenchmarkData, *, bins: int, method: NativeMethod) -> float:
    """运行 toad 等频或等宽分箱 + WOE transform。"""
    import toad

    toad_method = "quantile" if method == "quantile" else "step"
    combiner = toad.transform.Combiner()
    combiner.fit(
        data.frame,
        y=data.target,
        method=toad_method,
        n_bins=bins,
    )
    binned = combiner.transform(data.frame)
    transformer = toad.transform.WOETransformer()
    transformer.fit(binned, data.target)
    transformed = transformer.transform(binned)
    checksum = consume_frame(transformed)
    release_memory(combiner, binned, transformer, transformed)
    return checksum


def benchmark_mars_native(data: PolarsBenchmarkData, *, bins: int, method: NativeMethod) -> float:
    """运行 MarsNativeBinner 等频或等宽分箱 + WOE transform。"""
    binner = MarsNativeBinner(
        method=method,
        n_bins=bins,
        special_values=[-999.0],
        min_bin_size=0.01,
        merge_small_bins=True,
        remove_empty_bins=(method == "uniform"),
        n_jobs=-1,
    )
    binner.fit(data.frame, pl.Series("target", data.target), features=data.feature_names)
    transformed = binner.transform(data.frame, return_type="woe", woe_batch_size=200)
    checksum = consume_frame(transformed)
    release_memory(binner, transformed)
    return checksum


def benchmark_mars_optimal(
    data: PolarsBenchmarkData,
    *,
    bins: int,
    opt_time_limit: int,
) -> float:
    """运行 MarsOptimalBinner 的 fit + WOE transform。"""
    binner = MarsOptimalBinner(
        n_bins=bins,
        n_prebins=max(bins * 2, 12),
        min_bin_n_event=1,
        min_bin_size=0.01,
        min_prebin_size=0.01,
        prebinning_method="quantile",
        special_values=[-999.0],
        time_limit=opt_time_limit,
        n_jobs=-1,
    )
    binner.fit(data.frame, pl.Series("target", data.target), features=data.feature_names)
    transformed = binner.transform(data.frame, return_type="woe", woe_batch_size=200)
    checksum = consume_frame(transformed)
    release_memory(binner, transformed)
    return checksum


def benchmark_optbinning(
    data: PandasBenchmarkData,
    *,
    bins: int,
    opt_time_limit: int,
) -> float:
    """运行 optbinning.BinningProcess 的 fit + WOE transform。"""
    from optbinning import BinningProcess

    fit_params = {name: {"time_limit": opt_time_limit} for name in data.feature_names}
    process = BinningProcess(
        variable_names=data.feature_names,
        max_n_prebins=max(bins * 2, 12),
        min_prebin_size=0.01,
        max_n_bins=bins,
        min_bin_size=0.01,
        special_codes=[-999.0],
        binning_fit_params=fit_params,
        n_jobs=-1,
    )
    process.fit(data.frame, data.target)
    transformed = process.transform(data.frame, metric="woe")
    checksum = consume_frame(transformed)
    release_memory(process, transformed)
    return checksum


def run_toad_native_once(
    *,
    rows: int,
    features: int,
    seed: int,
    bins: int,
    method: NativeMethod,
) -> float:
    """构造 Pandas 宽表并完成一轮 toad 原生分箱。"""
    data: PandasBenchmarkData | None = None
    try:
        data = build_pandas_data(rows=rows, features=features, seed=seed)
        return benchmark_toad_native(data, bins=bins, method=method)
    finally:
        release_memory(data)
        data = None


def run_mars_native_once(
    *,
    rows: int,
    features: int,
    seed: int,
    bins: int,
    method: NativeMethod,
) -> float:
    """构造 Polars 宽表并完成一轮 MARS 原生分箱。"""
    data: PolarsBenchmarkData | None = None
    try:
        data = build_polars_data(rows=rows, features=features, seed=seed)
        return benchmark_mars_native(data, bins=bins, method=method)
    finally:
        release_memory(data)
        data = None


def run_mars_optimal_once(
    *,
    rows: int,
    features: int,
    seed: int,
    bins: int,
    opt_time_limit: int,
) -> float:
    """构造 Polars 宽表并完成一轮 MARS 最优分箱。"""
    data: PolarsBenchmarkData | None = None
    try:
        data = build_polars_data(rows=rows, features=features, seed=seed)
        return benchmark_mars_optimal(data, bins=bins, opt_time_limit=opt_time_limit)
    finally:
        release_memory(data)
        data = None


def run_optbinning_once(
    *,
    rows: int,
    features: int,
    seed: int,
    bins: int,
    opt_time_limit: int,
) -> float:
    """构造 Pandas 宽表并完成一轮 optbinning 最优分箱。"""
    data: PandasBenchmarkData | None = None
    try:
        data = build_pandas_data(rows=rows, features=features, seed=seed)
        return benchmark_optbinning(data, bins=bins, opt_time_limit=opt_time_limit)
    finally:
        release_memory(data)
        data = None


def measure(
    scenario: str,
    name: str,
    fn: Callable[[], float],
    repeats: int,
    note: str = "",
) -> BenchmarkResult:
    """重复运行一个 benchmark 方法并记录耗时和 RSS 增量。"""
    timings: list[float] = []
    memory_deltas_mb: list[float] = []
    peak_memory_deltas_mb: list[float] = []
    checksum = 0.0
    for idx in range(repeats):
        gc.collect()
        with MemorySampler() as sampler:
            start = time.perf_counter()
            checksum = fn()
            elapsed = time.perf_counter() - start
        memory_stats = sampler.stats()
        timings.append(elapsed)
        memory_deltas_mb.append(memory_stats.end_delta_mb)
        peak_memory_deltas_mb.append(memory_stats.peak_delta_mb)
        print(
            f"{scenario} | {name} round {idx + 1}/{repeats}: "
            f"{elapsed:.3f}s, 结束增量 {memory_stats.end_delta_mb:.1f} MB, "
            f"峰值增量 {memory_stats.peak_delta_mb:.1f} MB"
        )
    return BenchmarkResult(
        scenario=scenario,
        name=name,
        timings=timings,
        memory_deltas_mb=memory_deltas_mb,
        peak_memory_deltas_mb=peak_memory_deltas_mb,
        checksum=checksum,
        note=note,
    )


def render_markdown(
    results: Sequence[BenchmarkResult],
    *,
    rows: int,
    features: int,
    repeats: int,
    seed: int,
    baseline_name: str,
) -> str:
    """将 benchmark 结果渲染成 README 可直接粘贴的 Markdown。"""
    baseline_by_scenario = {
        result.scenario: result.avg
        for result in results
        if result.name == baseline_name
    }
    lines = [
        f"- 数据规模：`{rows:,}` 行 × `{features:,}` 个数值特征",
        "- 计时范围：数据生成 + fit + WOE transform + 本轮清理",
        "- 内存口径：主进程及其子进程的 RSS；结束增量为本轮结束 RSS - 起始 RSS，峰值增量为采样峰值 RSS - 起始 RSS",
        f"- 重复次数：`{repeats}`；随机种子：`{seed}`",
        f"- Python：`{platform.python_version()}`；系统：`{platform.platform()}`",
        "",
        "| 场景 | 方法 | 平均耗时(s) | 最快(s) | 最慢(s) | 平均结束增量(MB) | 峰值增量(MB) | 相对基准 | 备注 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        baseline = baseline_by_scenario.get(result.scenario)
        relative = result.avg / baseline if baseline else 1.0
        lines.append(
            f"| {result.scenario} | {result.name} | {result.avg:.3f} | "
            f"{result.best:.3f} | {result.worst:.3f} | "
            f"{result.avg_memory_delta_mb:.1f} | {result.peak_memory_delta_mb:.1f} | "
            f"{relative:.2f}x | {result.note} |"
        )
    return "\n".join(lines)


def run_native(args: argparse.Namespace) -> None:
    """运行 toad 与 MARS 原生分箱性能对比。"""
    if args.rows < 200_000:
        raise ValueError("native benchmark requires --rows >= 200000.")
    if args.features < 3_000:
        raise ValueError("native benchmark requires --features >= 3000.")
    if args.repeats < 1:
        raise ValueError("native benchmark requires --repeats >= 1.")
    if importlib.util.find_spec("toad") is None:
        raise ImportError(
            "toad is not installed. Install it with "
            "`conda run -n mars python -m pip install toad` and rerun this benchmark."
        )

    results: list[BenchmarkResult] = []
    native_cases: list[tuple[NativeMethod, str]] = [
        ("quantile", "等频分箱"),
        ("uniform", "等宽分箱"),
    ]
    for method, scenario in native_cases:
        print(f"\n[{scenario}] 先运行 toad，计时含 Pandas 数据生成。")
        results.append(
            measure(
                scenario,
                "toad Combiner + WOETransformer",
                lambda native_method=method: run_toad_native_once(
                    rows=args.rows,
                    features=args.features,
                    seed=args.seed,
                    bins=args.bins,
                    method=native_method,
                ),
                args.repeats,
                note="先运行；外部竞品库，不属于项目依赖",
            )
        )
        release_memory()

        print(f"[{scenario}] toad 阶段已释放，继续运行 MARS，计时含 Polars 数据生成。")
        results.append(
            measure(
                scenario,
                "MarsNativeBinner",
                lambda native_method=method: run_mars_native_once(
                    rows=args.rows,
                    features=args.features,
                    seed=args.seed,
                    bins=args.bins,
                    method=native_method,
                ),
                args.repeats,
                note=f"method={method}",
            )
        )
        release_memory()

    print("")
    print(
        render_markdown(
            results,
            rows=args.rows,
            features=args.features,
            repeats=args.repeats,
            seed=args.seed,
            baseline_name="MarsNativeBinner",
        )
    )


def run_optimal(args: argparse.Namespace) -> None:
    """运行 MARS 最优分箱与 optbinning 对比。"""
    if args.rows < 50_000:
        raise ValueError("optimal benchmark requires --rows >= 50000.")
    if args.features < 1_000:
        raise ValueError("optimal benchmark requires --features >= 1000.")
    if args.repeats < 1:
        raise ValueError("optimal benchmark requires --repeats >= 1.")

    results: list[BenchmarkResult] = []

    print("\n[最优分箱] 运行 MARS，计时含 Polars 数据生成。")
    results.append(
        measure(
            "最优分箱",
            "MarsOptimalBinner",
            lambda: run_mars_optimal_once(
                rows=args.rows,
                features=args.features,
                seed=args.seed,
                bins=args.bins,
                opt_time_limit=args.opt_time_limit,
            ),
            args.repeats,
            note=f"单特征 time_limit={args.opt_time_limit}s",
        )
    )
    release_memory()

    print("[最优分箱] MARS 阶段已释放，运行 optbinning，计时含 Pandas 数据生成。")
    results.append(
        measure(
            "最优分箱",
            "optbinning.BinningProcess",
            lambda: run_optbinning_once(
                rows=args.rows,
                features=args.features,
                seed=args.seed,
                bins=args.bins,
                opt_time_limit=args.opt_time_limit,
            ),
            args.repeats,
            note=f"单特征 time_limit={args.opt_time_limit}s",
        )
    )
    release_memory()

    print("")
    print(
        render_markdown(
            results,
            rows=args.rows,
            features=args.features,
            repeats=args.repeats,
            seed=args.seed,
            baseline_name="MarsOptimalBinner",
        )
    )


def main() -> None:
    """运行命令指定的性能对比流程。"""
    args = parse_args()
    if args.command == "optimal":
        run_optimal(args)
    else:
        run_native(args)


if __name__ == "__main__":
    main()
