class MarsError(Exception):
    """
    Mars 框架的基础异常类。
    所有 Mars 自定义异常都应继承此类。
    """
    pass

class NotFittedError(MarsError):
    """
    当转换器 (Transformer) 或模型在未调用 fit() 之前被调用 transform/predict 时抛出。
    """
    pass

class DataTypeError(MarsError):
    """
    当输入数据类型不符合预期 (例如输入了 Numpy Array 而不是 Polars DataFrame) 时抛出。
    """
    pass