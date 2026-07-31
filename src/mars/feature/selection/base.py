"""特征筛选共享基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Union

import pandas as pd
import polars as pl

from mars.core.base import MarsBaseEstimator
from mars.utils.logger import logger


class MarsBaseSelector(MarsBaseEstimator, ABC):
    """
    MARS 特征筛选器抽象基类。

    Attributes
    ----------
    target : str
        当前筛选器绑定的目标变量列名。
    selected_features_ : list of str
        当前筛选流程最终保留的特征列表。
    n_features_in_ : int
        初始输入特征数量。
    report_records_ : list of dict
        筛选过程中的明细决策记录。
    _is_fitted : bool
        当前筛选器是否已完成拟合。

    Examples
    --------
    >>> issubclass(MarsBaseSelector, MarsBaseEstimator)
    True
    """

    def __init__(self) -> None:
        super().__init__()
        self.target: str | None = None
        self.selected_features_: list[str] = []
        self.n_features_in_: int = 0
        self.report_records_: list[dict[str, Any]] = []
        self._is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Any | None = None,
    ) -> MarsBaseSelector:
        """
        执行特征筛选拟合。

        Parameters
        ----------
        X : Union[pl.DataFrame, pd.DataFrame]
            输入特征数据集。
        y : Any | None
            目标变量。若子类无需单独传入标签，可从 ``X`` 中自行解析。

        Returns
        -------
        MarsBaseSelector
            完成拟合后的筛选器实例。

        Examples
        --------
        >>> class KeepAgeSelector(MarsBaseSelector):
        ...     def fit(self, X: pl.DataFrame | pd.DataFrame, y: Any | None = None) -> "MarsBaseSelector":
        ...         self.selected_features_ = ["age"]
        ...         self.n_features_in_ = len(X.columns)
        ...         self._is_fitted = True
        ...         return self
        >>> selector = KeepAgeSelector(target="y").fit(pl.DataFrame({"age": [20], "y": [0]}))
        >>> selector.selected_features_
        ['age']
        """

    def transform(self, X: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
        """
        根据筛选结果裁剪输入数据。

        Parameters
        ----------
        X : Union[pl.DataFrame, pd.DataFrame]
            待裁剪的数据集。

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            仅保留 ``selected_features_`` 以及目标列后的数据集。

        Examples
        --------
        >>> class KeepAgeSelector(MarsBaseSelector):
        ...     def fit(self, X: pl.DataFrame | pd.DataFrame, y: Any | None = None) -> "MarsBaseSelector":
        ...         self.selected_features_ = ["age"]
        ...         self._is_fitted = True
        ...         return self
        >>> selector = KeepAgeSelector(target="y").fit(pl.DataFrame({"age": [20], "y": [0]}))
        >>> selector.transform(pl.DataFrame({"age": [30], "income": [10], "y": [1]})).columns
        ['age', 'y']
        """
        self._check_is_fitted()
        X = self._ensure_polars_dataframe(X)

        cols_to_keep = [c for c in self.selected_features_ if c in X.columns]

        X_out = X.select(cols_to_keep)
        return self._format_output(X_out)

    def fit_transform(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Any | None = None,
        **kwargs: Any,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        先拟合再返回筛选后的结果。

        Parameters
        ----------
        X : Union[pl.DataFrame, pd.DataFrame]
            输入特征数据集。
        y : Any | None
            目标变量。
        **kwargs : Any
            透传给子类 ``fit`` 实现的附加参数。

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            仅保留入选特征及目标列的数据集。

        Examples
        --------
        >>> class KeepAgeSelector(MarsBaseSelector):
        ...     def fit(self, X: pl.DataFrame | pd.DataFrame, y: Any | None = None) -> "MarsBaseSelector":
        ...         self.selected_features_ = ["age"]
        ...         self._is_fitted = True
        ...         return self
        >>> result = KeepAgeSelector(target="y").fit_transform(
        ...     pl.DataFrame({"age": [20], "income": [10], "y": [0]})
        ... )
        >>> result.columns
        ['age', 'y']
        """
        return self.fit(X, y, **kwargs).transform(X)

    def get_report(self) -> pl.DataFrame:
        """
        生成特征筛选报告。

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            包含各特征筛选状态、阶段、原因、指标值与描述的报告表。

        Examples
        --------
        >>> class KeepAgeSelector(MarsBaseSelector):
        ...     def fit(self, X: pl.DataFrame | pd.DataFrame, y: Any | None = None) -> "MarsBaseSelector":
        ...         self.selected_features_ = ["age"]
        ...         self._register_decision("age", status="Selected", stage="demo")
        ...         self._is_fitted = True
        ...         return self
        >>> selector = KeepAgeSelector(target="y").fit(pl.DataFrame({"age": [20], "y": [0]}))
        >>> selector.get_report()["feature"].to_list()
        ['age']
        """
        self._check_is_fitted()

        if not self.report_records_:
            logger.warning("No report records found. Did you forget to call `_register_decision` in a subclass?")
            return pl.DataFrame([])

        report_df = pl.DataFrame(
            self.report_records_,
            schema={
                "feature": pl.Utf8,
                "data_source": pl.Utf8,
                "status": pl.Utf8,
                "stage": pl.Utf8,
                "reason": pl.Utf8,
                "value": pl.Float64,
                "description": pl.Utf8,
            },
        )

        sorted_report = report_df.sort(
            ["status", "stage", "feature"],
            descending=[False, False, False],
        )
        return self._format_output(sorted_report)

    def _register_decision(
        self,
        feature: str,
        status: str,
        stage: str,
        reason: str = "",
        value: float = -1.0,
        desc: str = "",
        data_source: str | None = None,
    ) -> None:
        """
        记录单个特征在筛选过程中的决策结果。

        Parameters
        ----------
        feature : str
            特征名。
        status : str
            决策状态，例如 ``"Selected"`` 或 ``"Dropped"``。
        stage : str
            当前筛选阶段。
        reason : str
            决策依据。
        value : float
            关键指标值。
        desc : str
            详细描述。
        data_source : str | None
            特征所属数据源。
        """
        self.report_records_.append(
            {
                "feature": feature,
                "data_source": data_source,
                "status": status,
                "stage": stage,
                "reason": reason,
                "value": value,
                "description": desc,
            }
        )

    def _get_feature_pool(self, X: pl.DataFrame) -> List[str]:
        """返回初始候选特征池。"""
        return [c for c in X.columns if c != self.target]

    def _check_is_fitted(self) -> None:
        """检查当前筛选器是否已完成拟合。"""
        if not self._is_fitted:
            raise ValueError(f"{self.__class__.__name__} is not fitted yet. Call 'fit' first.")
