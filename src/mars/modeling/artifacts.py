"""建模结果 artifact 的轻量读写工具。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import pandas as pd

if TYPE_CHECKING:
    from mars.modeling.report import MarsModelingReport


def safe_artifact_part(value: Any) -> str:
    """
    将模型类型、target 或指标名转换为稳定的目录片段。

    Parameters
    ----------
    value : Any
        需要写入 artifact 路径的上下文值。

    Returns
    -------
    str
        仅包含字母、数字、下划线和短横线的路径片段。

    Examples
    --------
    >>> safe_artifact_part("Long Target@AUC")
    'long_target_auc'
    """
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z_\\-]+", "_", text)
    return text.strip("_") or "unknown"


def create_artifact_path(
    artifact_dir: str | Path | None,
    *,
    model_type: str,
    target: str,
    optimize_metric: str,
    run_id: str,
) -> Path | None:
    """
    根据运行上下文创建独立 artifact 目录。

    Parameters
    ----------
    artifact_dir : str | Path | None
        artifact 根目录；为 ``None`` 时不落盘。
    model_type : str
        模型类型。
    target : str
        主目标列名。
    optimize_metric : str
        优化指标名。
    run_id : str
        本次运行唯一编号。

    Returns
    -------
    pathlib.Path or None
        新建的运行目录；禁用落盘时返回 ``None``。

    """
    if artifact_dir is None:
        return None
    base_dir = Path(artifact_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "_".join(
        [
            timestamp,
            safe_artifact_part(model_type),
            safe_artifact_part(target),
            safe_artifact_part(optimize_metric),
            safe_artifact_part(run_id),
        ]
    )
    run_path = base_dir / run_name
    run_path.mkdir(parents=True, exist_ok=False)
    return run_path


def step_artifact_dir(base_dir: str | Path, feature_count: int) -> str:
    """
    为 feature growth 的单个 step 生成独立 artifact 根目录。

    Parameters
    ----------
    base_dir : str | Path
        feature growth 根目录。
    feature_count : int
        当前 step 使用的特征数量。

    Returns
    -------
    str
        当前 step 的 artifact 目录字符串。
    """
    return str(Path(base_dir) / f"features_{feature_count}")


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

    Examples
    --------
    >>> to_json_safe({"value": 1})
    {'value': 1}
    """
    if isinstance(value, dict):
        return {str(key): to_json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(inner) for inner in value]
    if hasattr(value, "item") and callable(value.item):
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
    path : Path
        输出路径。
    data : Dict[str, Any]
        元数据字典。

    Returns
    -------
    None
        函数仅产生文件写入副作用。

    Examples
    --------
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "meta.json"
    ...     write_json(path, {"model": "xgb"})
    ...     path.exists()
    True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(to_json_safe(data), ensure_ascii=False, indent=2)
    path.write_text(payload, encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    """
    读取 artifact JSON 元数据。

    Parameters
    ----------
    path : Path
        JSON 文件路径。

    Returns
    -------
    dict
        解析后的元数据。

    Raises
    ------
    FileNotFoundError
        文件不存在时抛出。

    Examples
    --------
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "meta.json"
    ...     write_json(path, {"model": "xgb"})
    ...     read_json(path)["model"]
    'xgb'
    """
    if not path.exists():
        raise FileNotFoundError(f"Artifact metadata file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_report_tables(reports: Dict[str, MarsModelingReport], reports_dir: Path) -> Dict[str, str]:
    """
    保存 replay 评估报告表，并返回模型名到文件名的映射。

    Parameters
    ----------
    reports : Dict[str, MarsModelingReport]
        各 replay 模型的评估报告。
    reports_dir : Path
        输出目录。

    Returns
    -------
    dict of str to str
        模型名到 CSV 文件名的映射。

    Examples
    --------
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from mars.modeling.report import MarsModelingReport
    >>> report = MarsModelingReport(pd.DataFrame({"auc": [80.0]}))
    >>> with TemporaryDirectory() as tmp:
    ...     files = save_report_tables({"champion": report}, Path(tmp))
    ...     files["champion"]
    'champion.csv'
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_files: Dict[str, str] = {}
    for model_name, report in reports.items():
        file_name = f"{model_name}.csv"
        report.summary_table.to_csv(reports_dir / file_name)
        report_files[model_name] = file_name
    return report_files


def load_report_tables(reports_dir: Path, report_files: Dict[str, str]) -> Dict[str, MarsModelingReport]:
    """
    从 replay artifact 中恢复评估报告对象。

    Parameters
    ----------
    reports_dir : Path
        报告 CSV 目录。
    report_files : Dict[str, str]
        模型名到 CSV 文件名的映射。

    Returns
    -------
    dict of str to MarsModelingReport
        恢复后的报告对象。

    Raises
    ------
    FileNotFoundError
        当指定路径不存在时抛出。

    Examples
    --------
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from mars.modeling.report import MarsModelingReport
    >>> report = MarsModelingReport(pd.DataFrame({"auc": [80.0]}))
    >>> with TemporaryDirectory() as tmp:
    ...     reports_dir = Path(tmp)
    ...     files = save_report_tables({"champion": report}, reports_dir)
    ...     reports = load_report_tables(reports_dir, files)
    ...     "champion" in reports
    True
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
