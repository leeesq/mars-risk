from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union, Literal

import polars as pl
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from mars.core.exceptions import NotFittedError, DataTypeError
from mars.utils.decorators import time_it


class MarsBaseEstimator(BaseEstimator):
    """
    [MARS 基类] 负责输入数据的类型检测和输出数据的格式化。
    
    集成 Scikit-learn 的 BaseEstimator，支持 set_output API，
    允许用户在管道中灵活控制输出格式（Pandas 或 Polars）。
    """
    
    def __init__(self) -> None:
        # 内部标志位：是否返回 Pandas 格式
        # 默认 False (返回 Polars)，但在 _ensure_polars 中会根据输入自动调整
        self._return_pandas: bool = False

    def set_output(self, transform: Literal["default", "pandas", "polars"] = "default") -> "MarsBaseEstimator":
        """
        兼容 Sklearn 的 set_output API，允许用户强制指定输出格式。

        Parameters
        ----------
        transform : Literal["default", "pandas", "polars"]
            - "pandas": 强制输出 Pandas DataFrame。
            - "polars": 强制输出 Polars DataFrame。
            - "default": 保持默认行为 (通常跟随输入类型)。

        Returns
        -------
        MarsBaseEstimator
            返回实例本身以支持链式调用。
        """
        if transform == "pandas":
            self._return_pandas = True
        elif transform == "polars" or transform == "default":
            self._return_pandas = False
        return self

    def _ensure_polars(self, X: Any) -> pl.DataFrame:
        """
        [类型守卫] 确保输入数据转换为 Polars DataFrame，并记录原始类型。

        Parameters
        ----------
        X : Any
            输入数据，通常为 Pandas 或 Polars 的 DataFrame。

        Returns
        -------
        pl.DataFrame
            转换后的 Polars DataFrame。

        Raises
        ------
        DataTypeError
            当输入不是 DataFrame 类型时抛出。
        """
        # Case 1: 已经是 Polars
        if isinstance(X, pl.DataFrame):
            return X
            
        # Case 2: 是 Pandas
        elif isinstance(X, pd.DataFrame):
            # 自动嗅探：如果输入是 Pandas，输出默认也应该是 Pandas (除非用户手动 set_output 改过)
            self._return_pandas = True
            # Zero-Copy Conversion (尽可能零拷贝)
            return pl.from_pandas(X)
            
        # Case 3: 错误类型
        elif isinstance(X, (pl.LazyFrame, pd.Series, pl.Series)):
            raise DataTypeError(f"Input must be a generic DataFrame, got {type(X)}")
        else:
            raise DataTypeError(f"Mars expects Polars/Pandas DataFrame, got {type(X)}")

    def _format_output(self, data: Any) -> Any:
        """
        [输出格式化] 根据 _return_pandas 标志位，决定是否将结果转回 Pandas。

        支持递归处理字典和列表结构。

        Parameters
        ----------
        data : Any
            待格式化的数据 (DataFrame, Dict, List 等)。

        Returns
        -------
        Any
            格式化后的数据。
        """
        # 如果不需要转 Pandas，或者数据本来就是 Polars，直接返回
        if not self._return_pandas:
            return data

        # 递归处理字典 (常见于 stats_reports)
        if isinstance(data, dict):
            return {k: self._format_output(v) for k, v in data.items()}
        
        # 递归处理列表
        if isinstance(data, list):
            return [self._format_output(v) for v in data]

        # 核心转换逻辑：Polars -> Pandas
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
            
        return data


class MarsTransformer(MarsBaseEstimator, TransformerMixin, ABC):
    """
    [转换器基类]
    集成了自动 Pandas 互操作性。
    """

    def __init__(self):
        super().__init__() # 初始化 _return_pandas
        self.feature_names_in_: List[str] = []
        self._is_fitted: bool = False

    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_in_

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs) -> "MarsTransformer":
        # 嗅探输入类型 + 转 Polars
        X_pl = self._ensure_polars(X)
        
        # 执行核心逻辑
        self._fit_impl(X_pl, y, **kwargs)
        
        # 更新状态
        self.feature_names_in_ = X_pl.columns
        self._is_fitted = True
        return self

    def transform(self, X: Any) -> Union[pl.DataFrame, pd.DataFrame]:
        if not self._is_fitted:
            raise NotFittedError(f"{self.__class__.__name__} is not fitted.")
        
        # 嗅探输入类型 (注意：transform 时输入 Pandas，也会触发输出 Pandas)
        X_pl = self._ensure_polars(X)
        
        # 执行 Polars 逻辑
        X_new = self._transform_impl(X_pl)
        
        # 格式化输出 (Pandas/Polars)
        return self._format_output(X_new)

    @abstractmethod
    def _fit_impl(self, X: pl.DataFrame, y=None, **kwargs): 
        """
        [Abstract Core] 子类必须实现的核心拟合逻辑。
        必须返回 Polars DataFrame。
        """
        pass

    @abstractmethod
    def _transform_impl(self, X: pl.DataFrame) -> pl.DataFrame: 
        """
        [Abstract Core] 子类必须实现的核心转换逻辑。
        """
        pass
