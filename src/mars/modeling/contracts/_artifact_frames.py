"""artifact 表对象的共享序列化辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

CSV_FLOAT_FORMAT: str = "%.17g"


def dataframe_schema(df: pd.DataFrame) -> dict[str, str]:
    """记录 DataFrame 列 dtype，供 CSV artifact 读回后恢复类型。"""
    return {str(column): str(dtype) for column, dtype in df.dtypes.items()}


def restore_dataframe_schema(
    df: pd.DataFrame,
    schema: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """按 artifact metadata 中记录的 dtype 恢复 DataFrame。"""
    if not schema:
        return df

    restored = df.copy()
    for column, dtype_value in schema.items():
        if column not in restored.columns:
            continue
        dtype_name = str(dtype_value)
        try:
            if dtype_name.startswith("datetime64"):
                restored[column] = pd.to_datetime(restored[column])
            elif dtype_name == "category":
                restored[column] = restored[column].astype("category")
            else:
                restored[column] = restored[column].astype(dtype_name)
        except (TypeError, ValueError):
            continue
    return restored


def read_artifact_csv(
    path: Path,
    schema: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """读取 artifact CSV，并保持浮点值的 round-trip 精度。"""
    table = pd.read_csv(path, float_precision="round_trip")
    return restore_dataframe_schema(table, schema)
