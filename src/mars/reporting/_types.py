"""Reporting 层公开结果类型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class MarsHtmlRenderResult:
    """
    可嵌入 HTML 片段的渲染结果。

    Attributes
    ----------
    html : str
        可直接插入外部 HTML 报告的片段，不包含完整 ``html`` 或 ``body`` 标签。
    assets : list[Path]
        渲染过程中写出的图片资产路径。内嵌模式下通常为空列表。
    figures : list[Figure] | None
        可选返回的 Matplotlib 图对象，便于 Notebook 调试或调用方继续加工。

    Examples
    --------
    >>> result = MarsHtmlRenderResult(html="<div></div>", assets=[])
    >>> result.assets
    []
    """

    html: str
    assets: list[Path]
    figures: list[Figure] | None = None
