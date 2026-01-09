# mars/utils/decorators.py
import functools
import time
import warnings
from typing import Callable, Any, Union, Optional, Tuple, TypeVar, cast

import pandas as pd
import polars as pl

# 尝试导入 logger，如果失败则使用标准 logging (防止循环依赖或单独使用时的报错)
try:
    from .logger import get_mars_logger
    logger = get_mars_logger()
except ImportError:
    import logging
    logger = logging.getLogger("mars_fallback")

# 定义泛型，用于类型提示，保证装饰器不丢失函数签名信息
F = TypeVar('F', bound=Callable[..., Any])

def auto_polars(func: F) -> F:
    """
    [魔法装饰器] 自动处理 Pandas 到 Polars 的双向转换。

    该装饰器旨在实现 "零代码迁移"。它允许你编写只接受 Polars DataFrame 的核心逻辑，
    但在运行时自动兼容 Pandas DataFrame 输入，并在必要时将结果转回 Pandas。

    Parameters
    ----------
    func : Callable
        需要被装饰的类方法。该方法的第一个参数必须是 `self`，第二个参数必须是数据对象 `X`。

    Returns
    -------
    Callable
        增强后的包装函数。

    Notes
    -----
    1. **零拷贝 (Zero-Copy)**: 使用 `pl.from_pandas` 时会尝试共享内存，尽可能减少开销。
    2. **自动回落**: 如果输入是 Pandas，输出也是 DataFrame 类型，则会自动转换回 Pandas 以保持一致性。
    """
    @functools.wraps(func)
    def wrapper(self: Any, X: Union[pd.DataFrame, pl.DataFrame], *args: Any, **kwargs: Any) -> Any:
        # 1. 类型嗅探：记录原始输入是否为 Pandas
        #    这是决定最后是否需要 "回转" 的依据
        is_pandas_input = isinstance(X, pd.DataFrame)
        
        # 2. 统一入口：强制转换为 Polars
        if is_pandas_input:
            # 利用 Arrow 内存布局进行转换，通常非常快
            X_pl = pl.from_pandas(X) 
        else:
            # 如果已经是 Polars 或其他类型，保持原样
            X_pl = X
            
        # 3. 核心执行：调用被装饰的函数
        #    注意：此时传入的一定是 Polars 对象，因此核心逻辑只需处理 Polars API
        result = func(self, X_pl, *args, **kwargs)
        
        # 4. 统一出口：根据入口类型决定输出类型
        #    条件：(1) 原来是 Pandas (2) 结果是 Polars 表或序列
        if is_pandas_input and isinstance(result, (pl.DataFrame, pl.Series)):
            return result.to_pandas(use_pyarrow_extension_array=True)
            
        return result
    
    return cast(F, wrapper)

def format_output(func: F) -> F:
    """
    [输出控制] 根据实例状态自动转换输出格式的装饰器。

    该装饰器通常用于 DataHealthCheck 等类的方法上。它检查实例属性 `_return_pandas`，
    如果为 True，则将 Polars 结果强制转换为 Pandas。

    Parameters
    ----------
    func : Callable
        需要被装饰的类方法。

    Returns
    -------
    Callable
        增强后的包装函数。

    Notes
    -----
    支持处理以下返回类型：
    1. 单个 `pl.DataFrame`
    2. 包含 `pl.DataFrame` 的 `tuple` (例如 diagnose 方法的返回值)
    """
    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # 1. 执行核心逻辑 (默认返回 Polars)
        result = func(self, *args, **kwargs)
        
        # 2. 检查实例标志位
        #    如果实例没有 _return_pandas 属性或为 False，直接返回原结果 (Polars)
        if not getattr(self, "_return_pandas", False):
            return result
            
        # 3. 执行转换逻辑 (Polars -> Pandas)
        
        # 情况 A: 结果是单个 DataFrame
        if isinstance(result, pl.DataFrame):
            return result.to_pandas()
            
        # 情况 B: 结果是元组 (例如: return missing_df, zeros_df, unique_df)
        #    我们需要遍历元组，只转换其中的 DataFrame 对象，保持其他元素(如 str, int)不变
        elif isinstance(result, tuple):
            return tuple(
                item.to_pandas() if isinstance(item, pl.DataFrame) else item 
                for item in result
            )
            
        # 情况 C: 其他类型 (如 int, str, dict)，不进行转换
        return result
    
    return cast(F, wrapper)

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
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
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
            
        logger.info(f"⏱️ [{name}] finished in {duration:.4f}s")
        return result
    
    return cast(F, wrapper)

def deprecated(reason: str) -> Callable[[F], F]:
    """
    [生命周期] 标记函数为“已废弃”的装饰器。

    当调用被装饰的函数时，会触发 FutureWarning，提示用户该函数即将移除。

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
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 发出警告
            # stacklevel=2 确保警告指向调用该函数的那一行，而不是装饰器内部
            warnings.warn(
                f"⚠️ Function '{func.__name__}' is deprecated. {reason}",
                category=FutureWarning, 
                stacklevel=2
            )
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator

def safe_run(default_return: Any = None) -> Callable[[F], F]:
    """
    [容错保护] 异常捕获装饰器。

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
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 记录详细的错误堆栈，但不中断主进程
                logger.error(f"❌ Error in {func.__name__}: {str(e)}")
                # 在调试模式下，可能希望看到完整的 traceback，可以使用 logger.exception(e)
                return default_return
        return cast(F, wrapper)
    return decorator