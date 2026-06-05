"""分箱器速度对比脚本。

该脚本只用于手动复现 README 中的性能表，不作为 pytest 或 CI 的一部分。
默认数据规模为 50,000 行、1,000 个数值特征，计时范围统一为 fit + WOE transform。
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mars.feature import MarsNativeBinner, MarsOptimalBinner  # noqa: E402


@dataclass(frozen=True)
class BenchmarkData:
    """保存所有对比方法共享的样本数据。"""

    polars_df: pl.DataFrame
    pandas_df: pd.DataFrame
    target: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class BenchmarkResult:
    """保存单个方法的多轮耗时结果。"""

    name: str
    timings: list[float]
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


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="对比 MARS、optbinning 和 toad 的分箱速度。")
    parser.add_argument("--rows", type=int, default=50_000, help="合成样本行数。")
    parser.add_argument("--features", type=int, default=1_000, help="合成数值特征数。")
    parser.add_argument("--repeats", type=int, default=3, help="每个方法重复计时次数。")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子。")
    parser.add_argument("--bins", type=int, default=8, help="最大分箱数。")
    parser.add_argument("--opt-time-limit", type=int, default=1, help="最优分箱单特征求解秒数上限。")
    parser.add_argument("--include-toad", action="store_true", help="同时运行 toad 对比。")
    return parser.parse_args()


def build_dataset(rows: int, features: int, seed: int) -> BenchmarkData:
    """构造包含稳定、漂移、缺失、特殊值和噪声特征的共享宽表。"""
    if rows < 50_000:
        raise ValueError("--rows must be at least 50000 for this benchmark.")
    if features < 1_000:
        raise ValueError("--features must be at least 1000 for this benchmark.")

    rng = np.random.default_rng(seed)
    feature_names = [f"feat_{idx:04d}" for idx in range(features)]
    matrix = rng.normal(loc=0.0, scale=1.0, size=(rows, features)).astype(np.float32)

    block = max(features // 5, 1)
    row_drift = np.linspace(0.0, 0.8, rows, dtype=np.float32).reshape(-1, 1)
    matrix[:, block : 2 * block] += row_drift

    missing_block = matrix[:, 2 * block : 3 * block]
    missing_mask = rng.random(missing_block.shape) < 0.025
    missing_block[missing_mask] = np.nan

    special_block = matrix[:, 3 * block : 4 * block]
    special_mask = rng.random(special_block.shape) < 0.015
    special_block[special_mask] = -999.0

    signal_width = min(20, features)
    weights = np.linspace(1.2, -0.8, signal_width, dtype=np.float32)
    signal_matrix = np.nan_to_num(matrix[:, :signal_width], nan=0.0, posinf=0.0, neginf=0.0)
    raw_score = signal_matrix @ weights + rng.normal(scale=0.7, size=rows)
    target = (raw_score > np.median(raw_score)).astype(np.int32)

    polars_df = pl.DataFrame({name: matrix[:, idx] for idx, name in enumerate(feature_names)})
    pandas_df = pd.DataFrame(matrix, columns=feature_names, copy=False)
    return BenchmarkData(
        polars_df=polars_df,
        pandas_df=pandas_df,
        target=target,
        feature_names=feature_names,
    )


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


def benchmark_mars_native(data: BenchmarkData, bins: int) -> float:
    """运行 MarsNativeBinner 的 fit + WOE transform。"""
    binner = MarsNativeBinner(
        method="quantile",
        n_bins=bins,
        special_values=[-999.0],
        min_bin_size=0.01,
        merge_small_bins=True,
        n_jobs=-1,
    )
    binner.fit(data.polars_df, pl.Series("target", data.target), features=data.feature_names)
    transformed = binner.transform(data.polars_df, return_type="woe", woe_batch_size=200)
    return consume_frame(transformed)


def benchmark_mars_optimal(data: BenchmarkData, bins: int, opt_time_limit: int) -> float:
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
    binner.fit(data.polars_df, pl.Series("target", data.target), features=data.feature_names)
    transformed = binner.transform(data.polars_df, return_type="woe", woe_batch_size=200)
    return consume_frame(transformed)


def benchmark_optbinning(data: BenchmarkData, bins: int, opt_time_limit: int) -> float:
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
    process.fit(data.pandas_df, data.target)
    transformed = process.transform(data.pandas_df, metric="woe")
    return consume_frame(transformed)


def benchmark_toad(data: BenchmarkData, bins: int) -> float:
    """运行 toad Combiner + WOETransformer 的 fit + WOE transform。"""
    import toad

    combiner = toad.transform.Combiner()
    combiner.fit(
        data.pandas_df,
        y=data.target,
        method="quantile",
        n_bins=bins,
    )
    binned = combiner.transform(data.pandas_df)
    transformer = toad.transform.WOETransformer()
    transformer.fit(binned, data.target)
    transformed = transformer.transform(binned)
    return consume_frame(transformed)


def measure(name: str, fn: Callable[[], float], repeats: int, note: str = "") -> BenchmarkResult:
    """重复运行一个 benchmark 方法并记录耗时。"""
    timings: list[float] = []
    checksum = 0.0
    for idx in range(repeats):
        gc.collect()
        start = time.perf_counter()
        checksum = fn()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        print(f"{name} round {idx + 1}/{repeats}: {elapsed:.3f}s")
    return BenchmarkResult(name=name, timings=timings, checksum=checksum, note=note)


def render_markdown(
    results: Sequence[BenchmarkResult],
    *,
    rows: int,
    features: int,
    repeats: int,
    seed: int,
) -> str:
    """将 benchmark 结果渲染成 README 可直接粘贴的 Markdown。"""
    baseline = next((result.avg for result in results if result.name == "MarsNativeBinner"), None)
    lines = [
        f"- 数据规模：`{rows:,}` 行 × `{features:,}` 个数值特征",
        f"- 重复次数：`{repeats}`；随机种子：`{seed}`",
        f"- Python：`{platform.python_version()}`；系统：`{platform.platform()}`",
        "",
        "| 方法 | 平均耗时(s) | 最快(s) | 最慢(s) | 相对 MarsNative | 备注 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        relative = result.avg / baseline if baseline else 1.0
        lines.append(
            f"| {result.name} | {result.avg:.3f} | {result.best:.3f} | "
            f"{result.worst:.3f} | {relative:.2f}x | {result.note} |"
        )
    return "\n".join(lines)


def main() -> None:
    """运行完整性能对比流程。"""
    args = parse_args()
    data = build_dataset(rows=args.rows, features=args.features, seed=args.seed)

    results = [
        measure(
            "MarsNativeBinner",
            lambda: benchmark_mars_native(data, args.bins),
            args.repeats,
            note="Polars 原生等频分箱",
        ),
        measure(
            "MarsOptimalBinner",
            lambda: benchmark_mars_optimal(data, args.bins, args.opt_time_limit),
            args.repeats,
            note=f"单特征 time_limit={args.opt_time_limit}s",
        ),
        measure(
            "optbinning.BinningProcess",
            lambda: benchmark_optbinning(data, args.bins, args.opt_time_limit),
            args.repeats,
            note=f"单特征 time_limit={args.opt_time_limit}s",
        ),
    ]

    if args.include_toad:
        if importlib.util.find_spec("toad") is None:
            raise ImportError(
                "toad is not installed. Install it with "
                "`conda run -n mars python -m pip install toad` and rerun this benchmark."
            )
        results.append(
            measure(
                "toad Combiner + WOETransformer",
                lambda: benchmark_toad(data, args.bins),
                args.repeats,
                note="外部竞品库，不属于项目依赖",
            )
        )

    print("")
    print(
        render_markdown(
            results,
            rows=args.rows,
            features=args.features,
            repeats=args.repeats,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
