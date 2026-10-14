"""从干净内核执行本地 Home Credit Notebook，保留内部验收输出并验证原 CSV 未变化。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def _digest(path: Path) -> str:
    """流式计算输入校验和，不保留原始宽表副本。"""
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


class _AcceptanceClient(NotebookClient):
    """为实际顺序执行记录 cell 进度，源码与计算配置保持原样。"""

    async def async_execute_cell(self, cell: Any, cell_index: int, **kwargs: Any) -> Any:
        """显示每个 cell 的实际执行进度，异常由 nbclient 原样传播。"""
        print(f"Executing cell {cell_index}", flush=True)
        return await super().async_execute_cell(cell, cell_index, **kwargs)


def main() -> None:
    """执行唯一目标 Notebook 并写出可追溯的本地验收记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "examples/home-credit-2018/notebooks/mars_home_credit_2018_use_test.ipynb"
    output = root / ".artifacts/notebook-repair"
    output.mkdir(parents=True, exist_ok=True)
    raw = root / "examples/home-credit-2018/data/raw_data"
    inputs = [raw / name for name in (
        "application_train.csv", "application_test.csv", "HomeCredit_columns_description.csv",
    )]
    before = {path.name: _digest(path) for path in inputs}
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    os.environ["MPLBACKEND"] = "Agg"
    manager = KernelManager(kernel_name=args.kernel)
    manager.kernel_spec.argv[0] = sys.executable
    client = _AcceptanceClient(notebook, timeout=3600, kernel_manager=manager, resources={"metadata": {"path": str(notebook_path.parent)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output / "executed_home_credit.ipynb")
    after = {path.name: _digest(path) for path in inputs}
    if before != after:
        raise RuntimeError("Raw CSV input checksum changed during execution.")
    result = {
        "notebook": str(notebook_path), "cells": len(notebook.cells),
        "kernel": args.kernel, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable, "python_version": sys.version,
        "input_sha256": after,
        "executed_code_cells": [index for index, cell in enumerate(notebook.cells) if cell.cell_type == "code" and cell.execution_count is not None],
    }
    (output / "notebook_execution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
