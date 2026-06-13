"""建模评估相关的稳定排序规则。"""

from __future__ import annotations

import re
from typing import Tuple


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
