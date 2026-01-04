from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl
import pandas as pd

class MarsPolarsEstimator(BaseEstimator, TransformerMixin):
    """
    所有 MARS 组件的基类。
    自动将 Pandas 输入转换为 Polars 进行计算，计算完视情况转回。
    """
    def _ensure_polars(self, X):
        if isinstance(X, pd.DataFrame):
            # 使用 PyArrow 零拷贝转换 (快!)
            return pl.from_pandas(X) 
        return X

    def _ensure_pandas(self, X):
        if isinstance(X, (pl.DataFrame, pl.Series)):
            return X.to_pandas()
        return X