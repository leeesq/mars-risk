"""建模模块内部共享工具。"""

from __future__ import annotations

import importlib
import re
from typing import Any, Dict, Tuple

import pandas as pd
import polars as pl

from mars.utils.frame import (
    FrameLike as FrameLike,
)
from mars.utils.frame import (
    is_polars_dataframe as is_polars_dataframe,
)
from mars.utils.frame import (
    restore_frame_type as restore_frame_type,
)
from mars.utils.frame import (
    to_pandas_frame as to_pandas_frame,
)
from mars.utils.frame import (
    to_polars_frame as to_polars_frame,
)

HISTORY_BASE_COLUMNS = ["trial_num", "trial_state", "is_valid", "val_diff", "max_oot_diff"]
METRIC_NAMES = ("auc", "ks", "f1")


def split_name_sort_key(split_name: str) -> Tuple[int, int, str]:
    """
    生成 train/val/oot 友好的稳定排序键。

    Parameters
    ----------
    split_name : str
        数据集切片名称。

    Returns
    -------
    tuple of int, int, str
        排序键，顺序为 train、val、oot*、其他。

    Examples
    --------
    >>> sorted(["oot2", "train", "val"], key=split_name_sort_key)
    ['train', 'val', 'oot2']
    """
    normalized = str(split_name).strip().lower()
    if "train" in normalized:
        return (0, 0, normalized)
    if "val" in normalized:
        return (1, 0, normalized)
    if "oot" in normalized:
        match = re.search(r"(\d+)", normalized)
        return (2, int(match.group(1)) if match else 10**9, normalized)
    return (3, 0, normalized)


def normalize_dataset_flags(flags: pd.Series | pl.Series) -> pd.Series:
    """
    标准化数据集标识，供 train/val/oot contains 识别使用。

    Parameters
    ----------
    flags : pd.Series | pl.Series
        原始 dataset flag 列。

    Returns
    -------
    pandas.Series
        去空格并转小写后的字符串序列。

    Examples
    --------
    >>> import pandas as pd
    >>> normalize_dataset_flags(pd.Series([" Train ", "OOT1"])).tolist()
    ['train', 'oot1']
    """
    flags_pd = flags.to_pandas() if isinstance(flags, pl.Series) else flags
    return flags_pd.astype(str).str.strip().str.lower()


def validate_dataset_flag_roles(flags: pd.Series | pl.Series) -> None:
    """
    校验 dataset flag 不会同时命中多个保留角色。

    Parameters
    ----------
    flags : pd.Series | pl.Series
        原始或标准化后的数据集标识。

    Returns
    -------
    None
        校验通过时不返回值。

    Raises
    ------
    ValueError
        任一唯一值同时包含 train、val、oot 中多个关键字时抛出。

    Examples
    --------
    >>> import pandas as pd
    >>> validate_dataset_flag_roles(pd.Series(["train", "val", "oot1"]))
    """
    normalized = normalize_dataset_flags(flags)
    unique_flags = sorted(set(normalized.dropna().tolist()))
    conflicts: list[str] = []
    for flag in unique_flags:
        roles = [
            role
            for role, matched in {
                "train": "train" in flag,
                "val": "val" in flag,
                "oot": "oot" in flag,
            }.items()
            if matched
        ]
        if len(roles) > 1:
            conflicts.append(flag)
    if conflicts:
        raise ValueError(
            "Ambiguous dataset_flag values matched multiple split roles: "
            f"{conflicts}. Please rename them so each value contains only one of train/val/oot."
        )


def optional_import(module_name: str) -> Any:
    """
    尝试导入可选依赖，导入失败时返回 ``None``。

    Parameters
    ----------
    module_name : str
        模块名。

    Returns
    -------
    Any
        模块对象或 ``None``。

    Examples
    --------
    >>> optional_import("json").__name__
    'json'
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def require_optional_module(module_name: str) -> Any:
    """
    导入建模后端必需的可选依赖。

    Parameters
    ----------
    module_name : str
        模块名。

    Returns
    -------
    Any
        模块对象。

    Raises
    ------
    ImportError
        依赖未安装时给出 mars-risk extras 安装提示。

    Examples
    --------
    >>> require_optional_module("json").__name__
    'json'
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{module_name!r} is required for mars.modeling. "
            f"Install the optional extras, for example `pip install \"mars-risk[ml,tuning]\"`."
        ) from exc


def collect_library_versions(*module_names: str) -> Dict[str, str | None]:
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
    versions: Dict[str, str | None] = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", None)
        except Exception:
            versions[module_name] = None
    return versions
