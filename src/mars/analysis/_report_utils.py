"""分析报告导出与展示的内部通用工具。"""

from __future__ import annotations

import pandas as pd
import polars as pl

from mars.compute import to_pandas_frame


def _as_pandas_frame(df: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
    """将展示层输入统一转换为 Pandas DataFrame。"""
    return to_pandas_frame(df)
