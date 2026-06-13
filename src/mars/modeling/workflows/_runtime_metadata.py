"""建模工作流运行时元数据辅助函数。"""

from __future__ import annotations

import importlib


def collect_library_versions(*module_names: str) -> dict[str, str | None]:
    """
    收集可选依赖版本，写入可复现实验元数据。

    Parameters
    ----------
    *module_names : str
        需要采集版本的模块名。

    Returns
    -------
    dict of str to str or None
        模块名到版本号的映射；导入失败时值为 ``None``。

    Examples
    --------
    >>> versions = collect_library_versions("json", "module_that_does_not_exist")
    >>> "json" in versions and "module_that_does_not_exist" in versions
    True
    """
    versions: dict[str, str | None] = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", None)
        except Exception:
            versions[module_name] = None
    return versions
