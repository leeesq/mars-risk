"""Reporting 层 Matplotlib 初始化工具。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from mars.utils.imports import optional_import


def ensure_matplotlib_environment() -> Path:
    """
    准备 Matplotlib 在受限运行环境中的基础配置目录。

    Returns
    -------
    Path
        Matplotlib 实际使用的配置目录。

    Examples
    --------
    >>> path = ensure_matplotlib_environment()
    >>> path.exists()
    True
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    configured_dir = os.environ.get("MPLCONFIGDIR")
    if configured_dir:
        config_path = Path(configured_dir).expanduser()
    else:
        config_path = Path(tempfile.gettempdir()) / "mars-risk-matplotlib"
        os.environ["MPLCONFIGDIR"] = str(config_path)

    config_path.mkdir(parents=True, exist_ok=True)
    return config_path


def require_pyplot(*, feature_name: str) -> Any:
    """
    加载 Matplotlib pyplot，并在缺失依赖时给出可行动错误。

    Parameters
    ----------
    feature_name : str
        当前绘图能力名称，用于错误信息。

    Returns
    -------
    Any
        ``matplotlib.pyplot`` 模块。

    Raises
    ------
    ImportError
        当 Matplotlib 不可用时抛出。

    Examples
    --------
    >>> require_pyplot(feature_name="MARS reporting").__name__
    'matplotlib.pyplot'
    """
    ensure_matplotlib_environment()
    matplotlib = optional_import("matplotlib")
    if matplotlib is not None:
        matplotlib.use("Agg", force=True)

    pyplot = optional_import("matplotlib.pyplot")
    if pyplot is None:
        raise ImportError(
            f"matplotlib is required for {feature_name}. "
            "It is included in the base mars-risk installation; reinstall mars-risk if missing."
        )
    return pyplot
