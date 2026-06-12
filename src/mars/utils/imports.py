"""可选依赖导入工具。"""

from __future__ import annotations

import importlib
from typing import Any


def optional_import(module_name: str) -> Any:
    """
    尝试导入可选依赖，导入失败时返回 ``None``。

    Parameters
    ----------
    module_name : str
        需要导入的模块名。

    Returns
    -------
    Any
        模块对象；依赖不存在时返回 ``None``。

    Examples
    --------
    >>> optional_import("json").__name__
    'json'
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def require_optional_module(
    module_name: str,
    *,
    feature_name: str = "MARS optional feature",
    extra_hint: str = 'pip install "mars-risk[ml,tuning]"',
) -> Any:
    """
    导入必需的可选依赖，并在缺失时给出统一安装提示。

    Parameters
    ----------
    module_name : str
        需要导入的模块名。
    feature_name : str
        当前依赖服务的功能名称，用于错误信息。
    extra_hint : str
        推荐安装命令或依赖说明。

    Returns
    -------
    Any
        成功导入的模块对象。

    Raises
    ------
    ImportError
        依赖未安装时抛出，并提示用户安装对应 extra。

    Examples
    --------
    >>> require_optional_module("json").__name__
    'json'
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{module_name!r} is required for {feature_name}. "
            f"Install the optional dependency, for example `{extra_hint}`."
        ) from exc
