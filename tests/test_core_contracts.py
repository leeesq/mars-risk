"""Core 层公共契约回归测试。"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from mars.core.base import MarsTransformer
from mars.core.exceptions import DataTypeError, MarsError, NotFittedError


class _IdentityTransformer(MarsTransformer):
    """用于验证基类拟合状态语义的最小转换器。"""

    def _fit_impl(self, X: pl.DataFrame, y: pl.Series | None = None) -> None:
        """拟合阶段不产生额外状态。"""
        return None

    def _transform_impl(self, X: pl.DataFrame) -> pl.DataFrame:
        """原样返回输入数据。"""
        return X


def test_mars_error_preserves_message_and_context() -> None:
    """MarsError 需要同时保留字符串消息和结构化上下文。"""
    error = MarsError("自定义错误", context={"feature": "age"})

    assert str(error) == "自定义错误"
    assert error.message == "自定义错误"
    assert error.context == {"feature": "age"}


def test_transformer_feature_names_are_created_after_fit() -> None:
    """sklearn 拟合状态属性只能在 fit 成功后创建。"""
    transformer = _IdentityTransformer()

    assert not hasattr(transformer, "feature_names_in_")
    with pytest.raises(NotFittedError) as exc_info:
        transformer.get_feature_names_out()

    assert exc_info.value.context["estimator"] == "_IdentityTransformer"

    fitted = transformer.fit(pl.DataFrame({"age": [20], "income": [100.0]}))

    assert fitted.feature_names_in_ == ["age", "income"]
    assert fitted.get_feature_names_out() == ["age", "income"]


def test_core_data_type_error_exposes_context() -> None:
    """核心输入类型错误需要暴露机器可读上下文。"""
    transformer = _IdentityTransformer()

    with pytest.raises(DataTypeError) as exc_info:
        transformer.fit(pd.Series([1, 2, 3]))

    assert exc_info.value.context["input_type"] == "Series"
    assert exc_info.value.context["expected_type"] == "DataFrame"
