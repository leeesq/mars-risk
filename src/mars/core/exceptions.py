"""MARS 公共异常类型。"""

from __future__ import annotations

from typing import Any


class MarsError(Exception):
    """
    MARS 框架的基础异常类。

    所有 MARS 自定义异常都应继承此类。除普通错误消息外，该类允许调用方保存
    机器可读的结构化上下文，便于上层自动化处理或测试断言。

    Attributes
    ----------
    message : str
        用户可读的错误消息。
    context : dict[str, Any]
        机器可读的错误上下文，例如字段名、实际类型或期望类型。

    Examples
    --------
    >>> error = MarsError("自定义 MARS 错误", context={"feature": "age"})
    >>> str(error)
    '自定义 MARS 错误'
    >>> error.context["feature"]
    'age'
    """

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        初始化 MARS 异常。

        Parameters
        ----------
        message : str
            用户可读的错误消息。
        context : dict[str, Any] | None
            机器可读的错误上下文，例如字段名、实际类型或期望类型。
        """
        self.message = message
        self.context = context or {}
        super().__init__(message)


class NotFittedError(MarsError):
    """
    未拟合对象被调用时抛出的异常。

    当转换器、分箱器或模型在未调用 ``fit()`` 前被调用 ``transform``、
    ``predict`` 或其他依赖拟合状态的方法时抛出。

    Examples
    --------
    >>> raise NotFittedError("请先调用 fit()")
    Traceback (most recent call last):
        ...
    mars.core.exceptions.NotFittedError: 请先调用 fit()
    """


class DataTypeError(MarsError):
    """
    输入数据类型不符合 MARS 约束时抛出的异常。

    典型场景包括传入一维 Series、NumPy 数组或无法安全转换为 Polars
    DataFrame 的对象。

    Examples
    --------
    >>> raise DataTypeError("X 必须是 Polars DataFrame")
    Traceback (most recent call last):
        ...
    mars.core.exceptions.DataTypeError: X 必须是 Polars DataFrame
    """
