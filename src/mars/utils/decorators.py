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
    在函数或方法成功返回后记录执行耗时。

    日志记录函数名，按首个位置参数判断是否添加类名。
    耗时单位为秒；函数抛出异常时不记录耗时，异常继续向调用方传播。

    Parameters
    ----------
    func : F
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

        result = func(*args, **kwargs)

        end = time.time()
        duration = end - start

        # 按首个位置参数推测实例方法名称；常见标量和容器按普通函数处理。
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
    在每次调用函数前发出弃用警告。

    警告类型为 ``FutureWarning``，警告位置指向调用方；发出警告后继续执行原函数。

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

    函数抛出 ``Exception`` 时，记录 ERROR 级别日志并返回默认值。
    适用于允许失败的辅助操作，如通知和绘图。

    Parameters
    ----------
    default_return : Any
        捕获 ``Exception`` 后返回的值，默认 ``None``；成功时仍返回原函数结果。

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
