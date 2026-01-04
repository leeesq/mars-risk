# mars/utils/decorators.py
import functools
import pandas as pd
import polars as pl
import time

def auto_polars(func):
    """
    [魔法装饰器] 
    1. 自动检测输入 X 是否为 Pandas，如果是，转为 Polars。
    2. 执行核心逻辑（核心逻辑只需写 Polars 版本）。
    3. 如果输入是 Pandas，把结果自动转回 Pandas。
    """
    @functools.wraps(func)
    def wrapper(self, X, *args, **kwargs):
        # 1. 记录原始类型
        is_pandas_input = isinstance(X, pd.DataFrame)
        
        # 2. 转换为 Polars (零拷贝)
        if is_pandas_input:
            X_pl = pl.from_pandas(X)
        else:
            X_pl = X
            
        # 3. 执行核心函数 (传入的是 Polars)
        result = func(self, X_pl, *args, **kwargs)
        
        # 4. 如果原来是 Pandas 且返回了 DataFrame，则转回 Pandas
        if is_pandas_input and isinstance(result, (pl.DataFrame, pl.Series)):
            return result.to_pandas()
            
        return result
    return wrapper

# mars/utils/decorators.py
import time
from .logger import get_mars_logger # 假设你之前写好了 logger

logger = get_mars_logger()

def time_it(func):
    """
    记录函数执行耗时
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        # 自动获取类名（如果是方法）或函数名
        if args and hasattr(args[0], '__class__'):
            name = f"{args[0].__class__.__name__}.{func.__name__}"
        else:
            name = func.__name__
            
        logger.info(f"⏱️ [{name}] finished in {end - start:.4f}s")
        return result
    return wrapper

import warnings

def deprecated(reason):
    """
    标记函数为废弃，打印警告
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"⚠️ Function '{func.__name__}' is deprecated. {reason}",
                Category=FutureWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def safe_run(default_return=None):
    """
    如果函数报错，记录 Error 日志但不中断程序，返回默认值。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 打印错误堆栈，但不抛出
                logger.error(f"❌ Error in {func.__name__}: {str(e)}")
                return default_return
        return wrapper
    return decorator