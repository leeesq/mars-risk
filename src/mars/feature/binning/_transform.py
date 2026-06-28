"""分箱转换内部表达式构造。"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
import polars as pl

from mars.utils.logger import logger


class _TransformBinnerState(Protocol):
    """描述转换表达式所需的最小分箱器状态。"""

    bin_woes_: dict[str, dict[int, float]]
    bin_mappings_: dict[str, dict[int, str]]


def _build_transform_output_expr(
    binner: _TransformBinnerState,
    *,
    col: str,
    final_idx_expr: pl.Expr,
    return_type: Literal["index", "label", "woe"],
) -> pl.Expr:
    """按输出模式把统一分箱索引表达式转换为最终输出列。"""
    if return_type == "index":
        return final_idx_expr.alias(f"{col}_bin")

    if return_type == "woe":
        woe_map = binner.bin_woes_.get(col, {})
        if not woe_map:
            logger.warning("WOE mapping for column '%s' not found. Defaulting to 0.0.", col)
            return pl.lit(0.0).alias(f"{col}_woe")

        clean_woe_map = {
            int(k): float(v)
            for k, v in woe_map.items()
            if k is not None and not (isinstance(k, float) and np.isnan(k))
        }
        expr = final_idx_expr.replace_strict(clean_woe_map, default=0.0).cast(pl.Float32)
        return expr.alias(f"{col}_woe")

    str_map = {str(k): v for k, v in binner.bin_mappings_.get(col, {}).items()}
    return final_idx_expr.cast(pl.Utf8).replace(str_map).alias(f"{col}_bin")
