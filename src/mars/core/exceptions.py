"""MARS 公共异常类型。"""


class MarsError(Exception):
    """
    Mars 框架的基础异常类。

    所有 Mars 自定义异常都应继承此类。

    Parameters
    ----------
    *args : object
        透传给 Python ``Exception`` 的错误消息或上下文对象。

    Attributes
    ----------
    args : tuple
        Python ``Exception`` 保存的原始错误参数。

    Examples
    --------
    >>> raise MarsError("自定义 MARS 错误")
    Traceback (most recent call last):
        ...
    mars.core.exceptions.MarsError: 自定义 MARS 错误
    """


class NotFittedError(MarsError):
    """
    当转换器 (Transformer) 或模型在未调用 fit() 之前被调用 transform/predict 时抛出。

    Parameters
    ----------
    *args : object
        透传给 ``MarsError`` 的错误消息或上下文对象。

    Attributes
    ----------
    args : tuple
        Python ``Exception`` 保存的原始错误参数。

    Examples
    --------
    >>> raise NotFittedError("请先调用 fit()")
    Traceback (most recent call last):
        ...
    mars.core.exceptions.NotFittedError: 请先调用 fit()
    """


class DataTypeError(MarsError):
    """
    当输入数据类型不符合预期 (例如输入了 Numpy Array 而不是 Polars DataFrame) 时抛出。

    Parameters
    ----------
    *args : object
        透传给 ``MarsError`` 的错误消息或上下文对象。

    Attributes
    ----------
    args : tuple
        Python ``Exception`` 保存的原始错误参数。

    Examples
    --------
    >>> raise DataTypeError("X 必须是 Polars DataFrame")
    Traceback (most recent call last):
        ...
    mars.core.exceptions.DataTypeError: X 必须是 Polars DataFrame
    """
