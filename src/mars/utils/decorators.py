"""MARS 通用装饰器工具。"""

import functools
import time
import warnings
from typing import Any, Callable, TypeVar, cast

try:
    from .logger import get_mars_logger
    logger = get_mars_logger()
except ImportError:
    import logging
    logger = logging.getLogger("mars_fallback")

F = TypeVar('F', bound=Callable[..., Any])

def time_it(func: F) -> F:
    """
    [性能监控] 记录函数或方法的执行耗时。

    会自动识别是被装饰的是 "独立函数" 还是 "类方法"，并在日志中打印
    ClassName.MethodName 或 FunctionName。

    Parameters
    ----------
    func : Callable
        需要计时的函数。

    Returns
    -------
    Callable
        带有计时日志的包装函数。

    Examples
    --------
    >>> @time_it
    ... def add_one(value: int) -> int:
    ...     return value + 1
    >>> add_one(1)
    2
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """执行被装饰函数并记录耗时。"""
        start = time.time()

        # 执行业务逻辑
        result = func(*args, **kwargs)

        end = time.time()
        duration = end - start

        # 智能名称解析
        # 如果第一个参数是对象实例且包含 __class__ 属性，通常意味着这是个方法
        if args and hasattr(args[0], '__class__') and not isinstance(args[0], (str, int, float, list, dict)):
             # 格式: ClassName.method_name
            name = f"{args[0].__class__.__name__}.{func.__name__}"
        else:
            # 格式: function_name
            name = func.__name__

        logger.info(f"[TIMER] [{name}] finished in {duration:.4f}s")
        return result

    return cast(F, wrapper)

def deprecated(reason: str) -> Callable[[F], F]:
    """
    [生命周期] 标记函数为“已废弃”的装饰器。

    当调用被装饰的函数时，会触发 FutureWarning, 提示用户该函数即将移除。

    Parameters
    ----------
    reason : str
        废弃原因及替代方案的说明文本。

    Returns
    -------
    Callable
        装饰器函数。

    Examples
    --------
    @deprecated("Use 'new_method' instead.")
    def old_method():
        pass
    """
    def decorator(func: F) -> F:
        """包装目标函数并在调用时发出弃用警告。"""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """执行目标函数前发出弃用警告。"""
            warnings.warn(
                f"Function '{func.__name__}' is deprecated. {reason}",
                category=FutureWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator

def safe_run(default_return: Any = None) -> Callable[[F], F]:
    """
    异常捕获装饰器。

    如果函数执行过程中抛出异常，记录 Error 级别的日志，阻止程序崩溃，
    并返回指定的默认值。常用于非核心路径的辅助功能（如发送通知、绘图）。

    Parameters
    ----------
    default_return : Any, optional
        发生异常时返回的默认值。默认为 None。

    Returns
    -------
    Callable
        装饰器函数。

    Examples
    --------
    >>> @safe_run(default_return="fallback")
    ... def fail() -> str:
    ...     raise RuntimeError("boom")
    >>> fail()
    'fallback'
    """
    def decorator(func: F) -> F:
        """包装目标函数并提供统一异常兜底。"""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """执行目标函数，失败时返回约定的默认值。"""
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                return default_return
        return cast(F, wrapper)
    return decorator
