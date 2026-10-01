"""小规模查询、上下文与持久化测量，不执行分箱拟合。"""

from __future__ import annotations

import json
import platform
import statistics
import time
import tracemalloc
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import pandas as pd
import polars as pl

from mars.reporting import MarsProfileReport, load_report


def measure(call: Callable[[], Any], repeats: int = 10) -> dict[str, float]:
    elapsed: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        elapsed.append((time.perf_counter() - start) * 1000)
    tracemalloc.start()
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"median_ms": statistics.median(elapsed), "python_peak_bytes": peak}


def main() -> None:
    columns = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(365)]
    overview = pd.DataFrame({"feature": [f"x{i}" for i in range(10000)], "mean": range(10000)})
    trend = pd.DataFrame({"feature": ["x0"], **{day: [i / 365] for i, day in enumerate(columns)}})
    report = MarsProfileReport(
        overview,
        {},
        {"mean": trend},
        feature_metadata={"x0": {"display_name": "示例数值", "description": "测量用模拟数值"}},
    )
    result: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pandas": pd.__version__,
            "polars": pl.__version__,
            "date": date.today().isoformat(),
        },
        "size": {"overview_rows": 10000, "trend_rows": 1, "date_columns": 365},
        "parameters": {
            "repeats": 10,
            "page_offset": 9000,
            "page_limit": 10,
            "budgets": [16000, 5000],
        },
        "page": measure(
            lambda: report.query_page(
                "overview", columns=["feature", "mean"], offset=9000, limit=10
            )
        ),
        "context": {},
        "limitations": "tracemalloc measures Python allocations only, not native memory; single host, no fit or speedup comparison",
    }
    for budget in (16000, 5000):

        def context_call(char_budget: int = budget) -> str:
            return report.to_ai_context(tables=["stats.mean"], features="x0", max_chars=char_budget)

        payload = json.loads(context_call())
        result["context"][str(budget)] = {
            **measure(context_call),
            "characters": len(context_call()),
            "returned_fields": len(payload["evidence"][0]["rows"][0]),
            "omissions": payload["omitted"],
        }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "measurement.marsreport"
        result["save"] = measure(lambda: report.save(path, overwrite=True), repeats=3)
        result["load"] = measure(lambda: load_report(path), repeats=3)
        result["file_bytes"] = path.stat().st_size
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
