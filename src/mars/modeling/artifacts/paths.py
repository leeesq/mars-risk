"""artifact 路径组织规则。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


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
