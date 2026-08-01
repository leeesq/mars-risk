"""特征筛选共享基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Literal, Sequence, Union

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

    def transform(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        *,
        on_missing: Literal["error", "warn", "ignore"] = "error",
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        根据筛选结果裁剪输入数据。

        Parameters
        ----------
        X : Union[pl.DataFrame, pd.DataFrame]
            待裁剪的数据集。
        on_missing : Literal['error', 'warn', 'ignore']
            输入缺少已选择特征时的处理策略。默认报错；宽松模式仅保留实际存在列。

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            仅保留 ``selected_features_`` 以及目标列后的数据集。

        Raises
        ------
        ValueError
            缺失已选特征或 ``on_missing`` 策略无效时抛出。

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
        if on_missing not in {"error", "warn", "ignore"}:
            raise ValueError("on_missing must be one of {'error', 'warn', 'ignore'}.")
        X = self._ensure_polars_dataframe(X)

        missing_features = [
            feature for feature in self.selected_features_ if feature not in X.columns
        ]
        if missing_features and on_missing == "error":
            raise ValueError(
                f"Selector transform input is missing selected features: {missing_features}."
            )
        if missing_features and on_missing == "warn":
            logger.warning("Selector transform skipped missing features: %s", missing_features)
        cols_to_keep = [feature for feature in self.selected_features_ if feature in X.columns]

        X_out = X.select(cols_to_keep)
        return self._format_output(X_out)

    def get_report(self) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        生成特征筛选报告。

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            包含各特征筛选状态、阶段、原因、指标值与描述的报告表。

        Raises
        ------
        ValueError
            已拟合筛选器没有任何决策记录时抛出。

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
            raise ValueError(
                "Selector decision report is empty; fit must record at least one decision."
            )

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


class _MarsXYSelector(MarsBaseSelector, ABC):
    """需要独立 ``X``/``y`` 输入的内部筛选器契约。"""

    @abstractmethod
    def fit(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Any,
        *,
        features: Sequence[str] | None = None,
    ) -> _MarsXYSelector:
        """拟合使用独立标签输入的筛选器。"""

    def fit_transform(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Any,
        *,
        features: Sequence[str] | None = None,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """拟合筛选器并转换同一特征表。"""
        return self.fit(X, y, features=features).transform(X)
