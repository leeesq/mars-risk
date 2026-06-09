"""建模模块内部共享工具。"""

from __future__ import annotations

import importlib
import re
from typing import Any, Dict, Tuple, Union

import pandas as pd
import polars as pl

FrameLike = Union[pd.DataFrame, pl.DataFrame]
HISTORY_BASE_COLUMNS = ["trial_num", "trial_state", "is_valid", "val_diff", "max_oot_diff"]
METRIC_NAMES = ("auc", "ks", "f1")


def is_polars_dataframe(df: Any) -> bool:
    """
    判断对象是否为 Polars eager DataFrame。

    Parameters
    ----------
    df : Any
        待检查对象。

    Returns
    -------
    bool
        若对象是 ``polars.DataFrame``，返回 ``True``。

    Examples
    --------
    >>> import polars as pl
    >>> is_polars_dataframe(pl.DataFrame({"x": [1]}))
    True
    """
    return isinstance(df, pl.DataFrame)


def to_pandas_frame(df: FrameLike) -> pd.DataFrame:
    """
    将 Pandas 或 Polars DataFrame 转为 Pandas 副本。

    Parameters
    ----------
    df : FrameLike
        输入数据框。

    Returns
    -------
    pandas.DataFrame
        Pandas 数据框副本。

    Raises
    ------
    TypeError
        输入类型不是 Pandas 或 Polars DataFrame 时抛出。

    Examples
    --------
    >>> import polars as pl
    >>> to_pandas_frame(pl.DataFrame({"x": [1]})).shape
    (1, 1)
    """
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")


def to_polars_frame(df: FrameLike) -> pl.DataFrame:
    """
    将 Pandas 或 Polars DataFrame 转为 Polars 副本。

    Parameters
    ----------
    df : FrameLike
        输入数据框。

    Returns
    -------
    polars.DataFrame
        Polars eager 数据框副本。

    Raises
    ------
    TypeError
        当输入对象类型不受支持时抛出。

    Examples
    --------
    >>> import pandas as pd
    >>> to_polars_frame(pd.DataFrame({"x": [1]})).shape
    (1, 1)
    """
    if isinstance(df, pl.DataFrame):
        return df.clone()
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")


def restore_frame_type(df: FrameLike, prefer_polars: bool) -> FrameLike:
    """
    按调用方偏好的数据引擎恢复输出类型。

    Parameters
    ----------
    df : FrameLike
        内部处理后的数据框。
    prefer_polars : bool
        是否优先返回 Polars DataFrame。

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        与输入链路一致的数据框。

    Examples
    --------
    >>> import pandas as pd
    >>> restore_frame_type(pd.DataFrame({"x": [1]}), prefer_polars=True).shape
    (1, 1)
    """
    if prefer_polars:
        if isinstance(df, pl.DataFrame):
            return df
        return pl.from_pandas(df)
    if isinstance(df, pd.DataFrame):
        return df
    return df.to_pandas()


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
