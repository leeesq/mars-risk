"""以行块中心矩累计完整 Pearson 矩阵，避免全人口工作数组。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


def _blocked_pearson(
    blocks: Iterable[NDArray[np.float64]], width: int,
) -> tuple[NDArray[np.float64], int]:
    """合并 Float64 行块中心矩；保留非有限值传播和退化列的 NaN 系数。

    调用方提供相同列顺序的独立行块；函数消费后不保留行块。以首行作平移原点，
    再用 Chan 合并中心矩，避免大均值小方差下的平方和相减。空间为 O(P² + RP)，
    时间为 O(NP²)；原点和均值为 O(P)，不生成全 N×P 数组。
    """
    count = 0
    origin: NDArray[np.float64] | None = None
    mean: NDArray[np.float64] = np.zeros(width, dtype=np.float64)
    moment: NDArray[np.float64] = np.zeros((width, width), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for values in blocks:
            if values.ndim != 2 or values.shape[1] != width:
                raise ValueError("Pearson blocks must have the declared feature width.")
            size = values.shape[0]
            if not size:
                continue
            if origin is None:
                origin = values[0].copy()
            # 仅修改本块的私有 Float64 缓冲区，不触碰输入列或其共享 buffer。
            centered = values - origin
            block_mean = centered.mean(axis=0)
            centered -= block_mean
            moment += centered.T @ centered
            total = count + size
            delta = block_mean - mean
            if count:
                moment += np.outer(delta, delta) * (count * size / total)
            mean += delta * (size / total)
            count = total
        scale = np.sqrt(np.diag(moment))
        moment /= scale[:, None]
        moment /= scale[None, :]
    # 与 NumPy corrcoef 的实数边界一致；常量、零/单样本的 NaN 不补为 1。
    np.clip(moment, -1.0, 1.0, out=moment)
    return moment, count
