"""MARS 核心估计器与转换器基类。"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Literal, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

from mars.core.exceptions import DataTypeError, NotFittedError
from mars.utils.decorators import time_it
from mars.utils.logger import logger


class MarsBaseEstimator(BaseEstimator):
    """
    MARS 估计器基类。

    Attributes
    ----------
    _return_pandas : bool
        是否将输出格式化为 Pandas 对象。
    _output_config : {"default", "pandas", "polars"}
        用户通过 ``set_output`` 指定的输出格式偏好。

    Examples
    --------
    >>> estimator = MarsBaseEstimator().set_output("polars")
    >>> estimator._output_config
    'polars'
    """

    _PL_NUMERIC_TYPES = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }

    def __init__(self) -> None:
        self._return_pandas: bool = False
        self._output_config: str = "default"

    def set_output(
        self,
        transform: Literal["default", "pandas", "polars"] = "default",
    ) -> MarsBaseEstimator:
        """
        设置实例输出格式。

        Parameters
        ----------
        transform : Literal['default', 'pandas', 'polars']
            目标输出格式。

        Returns
        -------
        MarsBaseEstimator
            当前实例，支持链式调用。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        Examples
        --------
        >>> estimator = MarsBaseEstimator().set_output("pandas")
        >>> estimator._return_pandas
        True
        """
        if transform not in ["default", "pandas", "polars"]:
            raise ValueError(f"Unknown output format: {transform}")

        self._output_config = transform
        if transform == "pandas":
            self._return_pandas = True
        elif transform == "polars":
            self._return_pandas = False
        return self

    def _determine_output_format(self, input_is_pandas: bool) -> None:
        """根据输出配置和输入类型决定最终输出格式。"""
        if self._output_config == "pandas":
            self._return_pandas = True
        elif self._output_config == "polars":
            self._return_pandas = False
        else:
            self._return_pandas = input_is_pandas

    def _ensure_polars_dataframe(
        self,
        X: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        确保输入被转换为 Polars DataFrame 或 LazyFrame。

        Parameters
        ----------
        X : pl.DataFrame | pl.LazyFrame | pd.DataFrame
            输入数据。

        Returns
        -------
        pl.DataFrame or pl.LazyFrame
            转换后的 Polars 对象。
        """
        if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            self._determine_output_format(input_is_pandas=False)
            return X

        if isinstance(X, pd.DataFrame):
            self._determine_output_format(input_is_pandas=True)
            try:
                X_pl = pl.from_pandas(X)
            except Exception as exc:
                raise DataTypeError(
                    f"Failed to convert Pandas DataFrame to Polars: {exc}",
                    context={
                        "input_type": type(X).__name__,
                        "expected_type": "polars.DataFrame",
                    },
                ) from exc

            self._validate_conversion(X, X_pl)
            return X_pl

        if isinstance(X, (pd.Series, pl.Series)):
            raise DataTypeError(
                f"Input must be a generic DataFrame (2D), got Series (1D): {type(X)}",
                context={
                    "input_type": type(X).__name__,
                    "expected_type": "DataFrame",
                },
            )

        raise DataTypeError(
            f"Mars expects Polars/Pandas DataFrame, got {type(X)}",
            context={
                "input_type": type(X).__name__,
                "expected_type": "polars.DataFrame | pandas.DataFrame",
            },
        )

    def _ensure_polars_series(
        self,
        y: pl.Series | pd.Series | np.ndarray | list | None,
        name: str = "target",
    ) -> pl.Series | None:
        """确保标签 ``y`` 被转换为 Polars Series。"""
        if y is None:
            return None

        if isinstance(y, pl.Series):
            return y

        if isinstance(y, pd.Series):
            return pl.from_pandas(y)

        if isinstance(y, np.ndarray):
            if y.ndim > 1:
                y = y.ravel()
            return pl.Series(name=name, values=y)

        if isinstance(y, list):
            return pl.Series(name=name, values=y)

        return pl.Series(name=name, values=y)

    def _validate_conversion(
        self,
        df_pd: pd.DataFrame,
        df_pl: Union[pl.DataFrame, pl.LazyFrame],
    ) -> None:
        """
        校验 Pandas 转 Polars 后的 schema 是否发生意外劣化。

        Parameters
        ----------
        df_pd : pd.DataFrame
            原始 Pandas 数据。
        df_pl : Union[pl.DataFrame, pl.LazyFrame]
            转换后的 Polars 数据。
        """
        pl_schema = df_pl.schema
        skip_patterns = re.compile(
            r".*(_id|_uuid|_dt|_at|_ts|_date|_time|_on)$|"
            r"^(id_|date_|time_|ts_)|"
            r"^(id|dt|ts|year|month|day|hour|minute|second)$",
            re.IGNORECASE,
        )

        for col in df_pd.columns:
            pd_dtype = df_pd[col].dtype
            pl_dtype = pl_schema.get(col)

            is_pd_numeric = pd.api.types.is_numeric_dtype(pd_dtype)
            is_pl_numeric = pl_dtype in self._PL_NUMERIC_TYPES

            if is_pd_numeric and not is_pl_numeric:
                if pl_dtype == pl.Null:
                    continue
                raise DataTypeError(
                    f"Column '{col}' is numeric in Pandas ({pd_dtype}) "
                    f"but converted to non-numeric in Polars ({pl_dtype}). "
                    "Check for mixed dtypes in your Pandas DataFrame.",
                    context={
                        "feature": str(col),
                        "pandas_dtype": str(pd_dtype),
                        "polars_dtype": str(pl_dtype),
                    },
                )

            if pd_dtype == "object" and pl_dtype == pl.Utf8:
                if skip_patterns.match(str(col)):
                    continue

                if isinstance(df_pl, pl.LazyFrame):
                    sample_series = (
                        df_pl.select(pl.col(col))
                        .drop_nulls()
                        .head(10)
                        .collect()
                        .to_series()
                    )
                else:
                    sample_series = df_pl[col].drop_nulls().head(10)

                if sample_series.len() == 0:
                    continue

                samples = sample_series.to_list()
                looks_like_numeric = True
                try:
                    for sample in samples:
                        sample_str = str(sample).strip()
                        if not sample_str or (sample_str.isdigit() and len(sample_str) > 15):
                            looks_like_numeric = False
                            break
                        float(sample_str)
                except (ValueError, TypeError):
                    looks_like_numeric = False

                if looks_like_numeric:
                    logger.warning(
                        f"\nPotential dirty data detected: column '{col}' looks numeric but is treated as String.\n"
                        f"   - Input (Pandas): object (mixed types)\n"
                        f"   - Output (Polars): Utf8\n"
                        f"   - Sample Values: {samples[:5]}...\n"
                        "   - Impact: This column will be handled as Categorical. "
                        "If it contains dirty strings (e.g. 'null', 'unknown'), "
                        "please clean them upstream or add them to 'missing_values'.",
                        stacklevel=2,
                    )

    def _format_output(self, data: Any) -> Any:
        """
        根据当前输出配置格式化结果对象。

        Parameters
        ----------
        data : Any
            待格式化的数据。

        Returns
        -------
        Any
            格式化后的结果。
        """
        if not self._return_pandas:
            return data

        if isinstance(data, dict):
            return {k: self._format_output(v) for k, v in data.items()}

        if isinstance(data, list):
            return [self._format_output(v) for v in data]

        if isinstance(data, pl.DataFrame):
            return data.to_pandas()

        return data


class MarsTransformer(MarsBaseEstimator, TransformerMixin, ABC):
    """
    MARS 转换器抽象基类。

    Attributes
    ----------
    feature_names_in_ : list of str
        最近一次成功拟合时缓存的输入特征名。
    _is_fitted : bool
        当前实例是否已完成拟合。

    Examples
    --------
    >>> issubclass(MarsTransformer, MarsBaseEstimator)
    True
    """

    def __init__(self) -> None:
        super().__init__()
        self._is_fitted: bool = False

    def __sklearn_is_fitted__(self) -> bool:
        """向 sklearn 暴露当前实例的拟合状态。"""
        return self._is_fitted

    def _check_is_fitted(self) -> None:
        """检查当前转换器是否已完成拟合。"""
        if not self._is_fitted:
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. Call 'fit' first.",
                context={"estimator": self.__class__.__name__},
            )

    def get_feature_names_out(self, input_features: Any | None = None) -> list[str]:
        """
        返回输出特征名列表。

        Parameters
        ----------
        input_features : Any | None
            sklearn 兼容保留参数，当前实现未使用。

        Returns
        -------
        list of str
            拟合阶段缓存的输入特征名。

        Examples
        --------
        >>> class IdentityTransformer(MarsTransformer):
        ...     def _fit_impl(self, X: pl.DataFrame, y: Any | None = None) -> None:
        ...         return None
        ...     def _transform_impl(self, X: pl.DataFrame) -> pl.DataFrame:
        ...         return X
        >>> transformer = IdentityTransformer().fit(pl.DataFrame({"age": [20]}))
        >>> transformer.get_feature_names_out()
        ['age']
        """
        self._check_is_fitted()
        return self.feature_names_in_

    @time_it
    def fit(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: pl.Series | pd.Series | np.ndarray | list[Any] | None = None,
    ) -> MarsTransformer:
        """
        拟合转换器并缓存输入特征信息。

        Parameters
        ----------
        X : pl.DataFrame | pd.DataFrame
            输入特征矩阵。
        y : pl.Series | pd.Series | np.ndarray | list[Any] | None
            目标变量。若为 ``None``，则执行无监督拟合流程。

        Returns
        -------
        MarsTransformer
            拟合完成后的转换器实例。

        Raises
        ------
        ValueError
            当 ``X`` 与 ``y`` 同为 Pandas 对象但索引不一致时抛出。

        Notes
        -----
        该基类仅提供最小公共拟合契约。若子类需要向用户暴露额外参数，
        应直接重写公共 ``fit`` 方法，而不是依赖内部实现钩子的隐式透传。

        Examples
        --------
        >>> class IdentityTransformer(MarsTransformer):
        ...     def _fit_impl(self, X: pl.DataFrame, y: Any | None = None) -> None:
        ...         return None
        ...     def _transform_impl(self, X: pl.DataFrame) -> pl.DataFrame:
        ...         return X
        >>> transformer = IdentityTransformer().fit(pl.DataFrame({"age": [20]}))
        >>> transformer.feature_names_in_
        ['age']
        """
        if isinstance(X, pd.DataFrame) and isinstance(y, (pd.Series, pd.DataFrame)):
            if not X.index.equals(y.index):
                raise ValueError(
                    "X and y have different indices. "
                    "Converting to Polars will lose index information leading to row mismatch. "
                    "Please align indices in Pandas strictly before passing to Mars."
                )

        X_pl = self._ensure_polars_dataframe(X)
        y_pl = self._ensure_polars_series(y)
        self._fit_impl(X_pl, y_pl)

        self.feature_names_in_ = X_pl.columns
        self._is_fitted = True
        return self

    def transform(
        self,
        X: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    ) -> pl.DataFrame | pd.DataFrame | pl.LazyFrame:
        """
        执行转换并返回结果。

        Parameters
        ----------
        X : pl.DataFrame | pl.LazyFrame | pd.DataFrame
            待转换的数据集。

        Returns
        -------
        pl.DataFrame or pd.DataFrame or pl.LazyFrame
            转换后的结果对象，最终类型由 ``set_output`` 配置和输入类型共同决定。

        Notes
        -----
        该基类不再承担子类私有参数的公开入口。若子类需要额外的转换参数，
        应在子类自己的公共 ``transform`` 方法中显式声明。

        Examples
        --------
        >>> class IdentityTransformer(MarsTransformer):
        ...     def _fit_impl(self, X: pl.DataFrame, y: Any | None = None) -> None:
        ...         return None
        ...     def _transform_impl(self, X: pl.DataFrame) -> pl.DataFrame:
        ...         return X
        >>> transformer = IdentityTransformer().fit(pl.DataFrame({"age": [20]}))
        >>> transformer.transform(pl.DataFrame({"age": [30]})).to_dict(as_series=False)
        {'age': [30]}
        """
        self._check_is_fitted()

        X_pl = self._ensure_polars_dataframe(X)
        X_new = self._transform_impl(X_pl)
        return self._format_output(X_new)

    def fit_transform(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Any | None = None,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        先拟合再返回转换结果。

        Parameters
        ----------
        X : Union[pl.DataFrame, pd.DataFrame]
            输入特征矩阵。
        y : Any | None
            目标变量。

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            转换后的结果数据集。

        Notes
        -----
        若子类公开的 ``fit`` 或 ``transform`` 带有额外参数，应同步重写
        ``fit_transform``，避免这些参数再次隐藏在基类实现之外。

        Examples
        --------
        >>> class IdentityTransformer(MarsTransformer):
        ...     def _fit_impl(self, X: pl.DataFrame, y: Any | None = None) -> None:
        ...         return None
        ...     def _transform_impl(self, X: pl.DataFrame) -> pl.DataFrame:
        ...         return X
        >>> IdentityTransformer().fit_transform(pl.DataFrame({"age": [20]})).shape
        (1, 1)
        """
        return self.fit(X, y).transform(X)

    @abstractmethod
    def _fit_impl(self, X: pl.DataFrame, y: Any | None = None) -> None:
        """子类必须实现的核心拟合逻辑。"""

    @abstractmethod
    def _transform_impl(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        子类必须实现的核心转换逻辑。

        Returns
        -------
        pl.DataFrame
            转换后的 Polars 数据集。

        Notes
        -----
        该方法仅作为内部实现钩子使用。若子类存在公开可配置的转换参数，
        应由子类的公共 ``transform`` 方法负责声明与校验。
        """
