"""建模评估指标与后端自定义评估函数。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def calculate_ks(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """
    计算百分制 KS 指标。

    Parameters
    ----------
    y_true : numpy.ndarray or pandas.Series
        真实二分类标签。
    y_pred : numpy.ndarray or pandas.Series
        预测为正类的概率或风险分。

    Returns
    -------
    float
        百分制 KS。若标签不足两类，返回 ``0.0``。

    Examples
    --------
    >>> calculate_ks(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    100.0
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if np.unique(y_true_arr).size < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true_arr, y_pred_arr)
    return round(float(np.max(tpr - fpr) * 100), 6)


def calculate_auc(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """
    计算百分制 AUC 指标。

    Parameters
    ----------
    y_true : numpy.ndarray or pandas.Series
        真实二分类标签。
    y_pred : numpy.ndarray or pandas.Series
        预测为正类的概率或风险分。

    Returns
    -------
    float
        百分制 AUC。若标签不足两类，返回随机模型基线 ``50.0``。

    Examples
    --------
    >>> calculate_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    100.0
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if np.unique(y_true_arr).size < 2:
        return 50.0
    return round(float(roc_auc_score(y_true_arr, y_pred_arr) * 100), 6)


def as_probability(preds: Any) -> np.ndarray:
    """
    将后端输出统一为一维概率数组。

    Parameters
    ----------
    preds : Any
        后端预测输出，可能是概率、raw margin 或嵌套数组。

    Returns
    -------
    numpy.ndarray
        一维正类概率数组。

    Examples
    --------
    >>> as_probability(np.array([-1.0, 0.0, 1.0])).round(3).tolist()
    [0.269, 0.5, 0.731]
    """
    arr = np.asarray(preds, dtype=float).reshape(-1)
    if arr.size and (np.nanmin(arr) < 0.0 or np.nanmax(arr) > 1.0):
        arr = 1.0 / (1.0 + np.exp(-arr))
    return arr


def xgb_ks_metric(preds: np.ndarray, dmatrix: Any) -> tuple[str, float]:
    """
    XGBoost 原生训练接口的 KS 自定义评估函数。

    Parameters
    ----------
    preds : numpy.ndarray
        当前迭代的预测输出。
    dmatrix : Any
        XGBoost DMatrix，需提供 ``get_label``。

    Returns
    -------
    tuple of str, float
        指标名与百分制 KS。

    Examples
    --------
    >>> class DummyDMatrix:
    ...     def get_label(self) -> np.ndarray:
    ...         return np.array([0, 1])
    >>> name, value = xgb_ks_metric(np.array([0.1, 0.9]), DummyDMatrix())
    >>> name
    'ks'
    """
    return "ks", calculate_ks(dmatrix.get_label(), as_probability(preds))


def lgb_ks_metric(preds: np.ndarray, dataset: Any) -> tuple[str, float, bool]:
    """
    LightGBM 原生训练接口的 KS 自定义评估函数。

    Parameters
    ----------
    preds : numpy.ndarray
        当前迭代的预测输出。
    dataset : Any
        LightGBM Dataset，需提供 ``get_label``。

    Returns
    -------
    tuple of str, float, bool
        指标名、百分制 KS、是否越大越好。

    Examples
    --------
    >>> class DummyDataset:
    ...     def get_label(self) -> np.ndarray:
    ...         return np.array([0, 1])
    >>> name, value, higher_is_better = lgb_ks_metric(np.array([0.1, 0.9]), DummyDataset())
    >>> higher_is_better
    True
    """
    return "ks", calculate_ks(dataset.get_label(), as_probability(preds)), True


class CatBoostKSMetric:
    """
    CatBoost 自定义 KS 评估指标。

    Parameters
    ----------
    None
        该指标对象不需要初始化参数。

    Attributes
    ----------
    None
        该指标对象不维护可变实例状态。

    Notes
    -----
    CatBoost 的 ``evaluate`` 返回值第一项参与最优迭代选择，第二项为权重。这里不做
    weighted KS，只保证多维或分块输入会被安全一维化。

    Examples
    --------
    >>> metric = CatBoostKSMetric()
    >>> metric.is_max_optimal()
    True
    """

    def is_max_optimal(self) -> bool:
        """
        返回 ``True``，表示 KS 越大越优。

        Returns
        -------
        bool
            固定为 ``True``。

        Examples
        --------
        >>> CatBoostKSMetric().is_max_optimal()
        True
        """
        return True

    def evaluate(self, approxes: Any, target: Any, weight: Any) -> tuple[float, float]:
        """
        计算 CatBoost 当前迭代的 KS。

        Parameters
        ----------
        approxes : Any
            CatBoost 传入的预测近似值。
        target : Any
            真实标签。
        weight : Any
            样本权重，本轮暂不使用。

        Returns
        -------
        tuple of float, float
            0-1 量纲 KS 与固定权重。

        Raises
        ------
        ValueError
            预测长度与标签长度不一致时抛出。

        Examples
        --------
        >>> metric = CatBoostKSMetric()
        >>> value, weight = metric.evaluate([np.array([0.1, 0.9])], np.array([0, 1]), None)
        >>> round(value, 1), weight
        (1.0, 1.0)
        """
        preds = as_probability(approxes[0])
        target_arr = np.asarray(target).reshape(-1)
        if preds.size == 0 and target_arr.size == 0:
            return 0.0, 1.0
        if preds.size != target_arr.size:
            raise ValueError(
                "CatBoost KS metric received mismatched prediction and target lengths: "
                f"{preds.size} vs {target_arr.size}."
            )
        return calculate_ks(target_arr, preds) / 100.0, 1.0

    def get_final_error(self, error: float, weight: float) -> float:
        """
        将 CatBoost 内部 0-1 量纲结果恢复为百分制 KS。

        Parameters
        ----------
        error : float
            CatBoost 内部传入的 0-1 量纲指标值。
        weight : float
            CatBoost 传入的指标权重，本实现不使用。

        Returns
        -------
        float
            百分制 KS。

        Examples
        --------
        >>> CatBoostKSMetric().get_final_error(0.42, 1.0)
        42.0
        """
        return float(error) * 100.0
