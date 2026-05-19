"""建模结果 artifact 的轻量读写工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import pandas as pd

if TYPE_CHECKING:
    from mars.modeling.report import MarsModelingReport


def to_json_safe(value: Any) -> Any:
    """
    将嵌套元数据转换为 JSON 可序列化对象。

    Parameters
    ----------
    value : Any
        任意 Python 对象。

    Returns
    -------
    Any
        JSON 可序列化对象。
    """
    if isinstance(value, dict):
        return {str(key): to_json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(inner) for inner in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return value
    return value


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """
    以 UTF-8 写入稳定格式 JSON 文件。

    Parameters
    ----------
    path : pathlib.Path
        输出路径。
    data : dict
        元数据字典。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    """
    读取 artifact JSON 元数据。

    Parameters
    ----------
    path : pathlib.Path
        JSON 文件路径。

    Returns
    -------
    dict
        解析后的元数据。

    Raises
    ------
    FileNotFoundError
        文件不存在时抛出。
    """
    if not path.exists():
        raise FileNotFoundError(f"Artifact metadata file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_report_tables(reports: Dict[str, "MarsModelingReport"], reports_dir: Path) -> Dict[str, str]:
    """
    保存 replay 评估报告表，并返回模型名到文件名的映射。

    Parameters
    ----------
    reports : dict of str to MarsModelingReport
        各 replay 模型的评估报告。
    reports_dir : pathlib.Path
        输出目录。

    Returns
    -------
    dict of str to str
        模型名到 CSV 文件名的映射。
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_files: Dict[str, str] = {}
    for model_name, report in reports.items():
        file_name = f"{model_name}.csv"
        report.summary_table.to_csv(reports_dir / file_name)
        report_files[model_name] = file_name
    return report_files


def load_report_tables(reports_dir: Path, report_files: Dict[str, str]) -> Dict[str, "MarsModelingReport"]:
    """
    从 replay artifact 中恢复评估报告对象。

    Parameters
    ----------
    reports_dir : pathlib.Path
        报告 CSV 目录。
    report_files : dict of str to str
        模型名到 CSV 文件名的映射。

    Returns
    -------
    dict of str to MarsModelingReport
        恢复后的报告对象。
    """
    from mars.modeling.report import MarsModelingReport

    reports: Dict[str, MarsModelingReport] = {}
    for model_name, file_name in report_files.items():
        table_path = reports_dir / file_name
        if not table_path.exists():
            raise FileNotFoundError(f"Artifact report table is missing: {table_path}")
        summary_table = pd.read_csv(table_path, header=[0, 1], index_col=0)
        reports[model_name] = MarsModelingReport(summary_table, caption=f"Model Evaluation by [{summary_table.index.name}]")
    return reports
