"""数据画像 PSI 的局部空人口与有效 Total 示例。"""

from __future__ import annotations

import json
import math
from typing import Any

import polars as pl

from mars.analysis import MarsDataProfiler


def run() -> None:
    """确认客群期间无适用人口时保留 null、局部状态和合并 Total。"""
    original = pl.DataFrame([
        {"s": segment, "g": group, "x": float(index), "z": float(index)}
        for segment in ["seg1", "seg2"]
        for group in ["01", "02"]
        for index in range(20)
    ])
    benchmark = original.filter(pl.col("g") == "01")
    current = original.with_columns(
        pl.when((pl.col("s") == "seg2") & (pl.col("g") == "02"))
        .then(None).otherwise(pl.col("x")).alias("x"),
    )
    results: list[dict[str, Any]] = []
    for mode in ["shared", "per_segment"]:
        for batch_size in [1, 2, 8]:
            report = MarsDataProfiler(
                binning_scope=mode, psi_batch_size=batch_size,
            ).generate_profile(
                current, features=["x", "z"], metrics=["psi"],
                segment_col="s", group_col="g", benchmark_df=benchmark,
                psi_include_missing=False, enable_sparkline=False,
            )
            selection = {"features": "x", "segments": ["seg2"], "include_all": False}
            psi = report.get_table("stats.psi", **selection)
            status = report.get_table("calculation_status", **selection)
            assert psi.schema["02"] == pl.Float64 and psi["02"].item() is None
            assert math.isclose(psi["01"].item(), 0.0, abs_tol=1e-12)
            assert math.isclose(psi["total"].item(), 0.0, abs_tol=1e-12)
            assert status.select("feature", "group", "metric", "status", "reason").to_dicts() == [{
                "feature": "x", "group": "02", "metric": "psi", "status": "not_computed",
                "reason": "no_effective_current_population",
            }]
            results.append({
                "binning_scope": mode, "psi_batch_size": batch_size,
                "psi": psi.select("01", "02", "total").row(0, named=True),
                "status": status.to_dicts(),
            })
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
