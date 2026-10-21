"""建模评估指标与后端自定义评估函数。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Literal, cast

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

MetricDirection = Literal["maximize", "minimize"]
MetricCallable = Callable[[np.ndarray, np.ndarray], float]
BUILTIN_METRIC_NAMES: tuple[str, ...] = ("auc", "ks", "f1")
DEFAULT_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "auc": "maximize",
    "ks": "maximize",
    "f1": "maximize",
}


def calculate_ks(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """
    计算百分制 KS 指标。

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        真实二分类标签。
    y_pred : np.ndarray | pd.Series
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
    y_true : np.ndarray | pd.Series
        真实二分类标签。
    y_pred : np.ndarray | pd.Series
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


def calculate_f1(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    threshold: float = 0.5,
) -> float:
    """
    计算百分制 F1 指标。

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        真实二分类标签。
    y_pred : np.ndarray | pd.Series
        预测为正类的概率或风险分。
    threshold : float
        正类预测阈值，默认 0.5。分数先经 ``as_probability`` 处理，大于等于阈值时预测为正类。

    Returns
    -------
    float
        百分制 F1。若无正例预测或无正例标签，则返回 ``0.0``。

    Examples
    --------
    >>> calculate_f1(np.array([0, 1, 1, 0]), np.array([0.1, 0.8, 0.7, 0.2]))
    100.0
    """
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_pred_arr = as_probability(y_pred)
    predicted_positive = y_pred_arr >= float(threshold)
    true_positive = y_true_arr == 1

    tp = float(np.sum(predicted_positive & true_positive))
    fp = float(np.sum(predicted_positive & ~true_positive))
    fn = float(np.sum(~predicted_positive & true_positive))
    denominator = (2.0 * tp) + fp + fn
    if denominator <= 0.0:
        return 0.0
    return round(float((2.0 * tp / denominator) * 100.0), 6)


def normalize_metric_directions(
    metric_names: Sequence[str],
    metric_directions: Mapping[str, MetricDirection] | None = None,
) -> dict[str, MetricDirection]:
    """
    统一指标名和优化方向的大小写，并补齐未指定的方向。

    Parameters
    ----------
    metric_names : Sequence[str]
        本次调参需要计算的指标名。
    metric_directions : Mapping[str, MetricDirection] | None
        以小写指标名为键的方向覆盖，默认不覆盖。
        未指定的指标使用项目默认方向，未登记的自定义指标默认 ``"maximize"``。

    Returns
    -------
    dict of str to MetricDirection
        每个指标对应的排序方向。

    Raises
    ------
    ValueError
        当指标方向不是 ``"maximize"`` 或 ``"minimize"`` 时抛出。

    Examples
    --------
    >>> normalize_metric_directions(["auc", "logloss"], {"logloss": "minimize"})["logloss"]
    'minimize'
    """
    directions: dict[str, MetricDirection] = {}
    user_directions = dict(metric_directions or {})
    for metric_name in metric_names:
        normalized_name = str(metric_name).lower()
        raw_direction = str(
            user_directions.get(
                normalized_name,
                DEFAULT_METRIC_DIRECTIONS.get(normalized_name, "maximize"),
            )
        ).lower()
        if raw_direction not in {"maximize", "minimize"}:
            raise ValueError(
                f"Unsupported metric direction for {metric_name!r}: {raw_direction!r}. "
                "Expected 'maximize' or 'minimize'."
            )
        directions[normalized_name] = cast(MetricDirection, raw_direction)
    return directions


def resolve_metric_names(
    custom_metrics: Mapping[str, MetricCallable] | None = None,
) -> list[str]:
    """
    生成本次建模需要保存的完整指标名列表。

    Parameters
    ----------
    custom_metrics : Mapping[str, MetricCallable] | None
        用户自定义指标函数字典。

    Returns
    -------
    list of str
        内置指标加自定义指标后的稳定顺序列表。

    Examples
    --------
    >>> resolve_metric_names({"head_tail_lift": lambda y, p: 1.0})[-1]
    'head_tail_lift'
    """
    metric_names = list(BUILTIN_METRIC_NAMES)
    for metric_name in (custom_metrics or {}).keys():
        normalized_name = str(metric_name).lower()
        if normalized_name not in metric_names:
            metric_names.append(normalized_name)
    return metric_names


def evaluate_metric(
    metric_name: str,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    metric_params: Mapping[str, Any] | None = None,
    custom_metrics: Mapping[str, MetricCallable] | None = None,
) -> float:
    """
    按名称计算单个内置或自定义指标。

    Parameters
    ----------
    metric_name : str
        指标名。
    y_true : np.ndarray | pd.Series
        真实二分类标签。
    y_pred : np.ndarray | pd.Series
        预测为正类的概率或风险分。
    metric_params : Mapping[str, Any] | None
        指标参数，默认使用内置配置。当前 F1 读取 ``f1_threshold``，默认 0.5；
        AUC、KS 和自定义指标不读取此字典。
    custom_metrics : Mapping[str, MetricCallable] | None
        用户自定义指标函数字典。

    Returns
    -------
    float
        百分制或用户自定义口径的指标值。

    Raises
    ------
    ValueError
        当指标名无法解析时抛出。

    Examples
    --------
    >>> evaluate_metric("auc", np.array([0, 1]), np.array([0.1, 0.9]))
    100.0
    """
    normalized_name = metric_name.lower()
    params = dict(metric_params or {})
    if normalized_name == "auc":
        return calculate_auc(y_true, y_pred)
    if normalized_name == "ks":
        return calculate_ks(y_true, y_pred)
    if normalized_name == "f1":
        return calculate_f1(
            y_true,
            y_pred,
            threshold=float(params.get("f1_threshold", 0.5)),
        )

    custom_metric = dict(custom_metrics or {}).get(normalized_name)
    if custom_metric is None:
        raise ValueError(f"Unsupported metric: {metric_name!r}.")
    return round(
        float(custom_metric(np.asarray(y_true).reshape(-1), as_probability(y_pred))),
        6,
    )


def evaluate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    metric_names: Sequence[str],
    *,
    metric_params: Mapping[str, Any] | None = None,
    custom_metrics: Mapping[str, MetricCallable] | None = None,
) -> dict[str, float]:
    """
    统一计算一组建模指标。

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        真实二分类标签。
    y_pred : np.ndarray | pd.Series
        预测为正类的概率或风险分。
    metric_names : Sequence[str]
        需要计算的指标名。
    metric_params : Mapping[str, Any] | None
        指标参数。
    custom_metrics : Mapping[str, MetricCallable] | None
        用户自定义指标函数字典。

    Returns
    -------
    dict of str to float
        指标名到指标值的映射。

    Examples
    --------
    >>> sorted(evaluate_metrics(np.array([0, 1]), np.array([0.1, 0.9]), ["auc", "f1"]))
    ['auc', 'f1']
    """
    return {
        str(metric_name).lower(): evaluate_metric(
            str(metric_name),
            y_true,
            y_pred,
            metric_params=metric_params,
            custom_metrics=custom_metrics,
        )
        for metric_name in metric_names
    }


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
        转为 float 并展平的一维数组。任一非缺失值超出 [0, 1] 时，对整组值应用 sigmoid；
        全部值都在 [0, 1] 内时原样返回数值，不重新校准，也不从多列中选择正类列。

    Examples
    --------
    >>> as_probability(np.array([-1.0, 0.0, 1.0])).round(3).tolist()
    [0.269, 0.5, 0.731]
    """
    arr = np.asarray(preds, dtype=float).reshape(-1)
    if arr.size and (np.nanmin(arr) < 0.0 or np.nanmax(arr) > 1.0):
        arr = 1.0 / (1.0 + np.exp(-arr))
    return cast(np.ndarray, arr)


def xgb_ks_metric(preds: np.ndarray, dmatrix: Any) -> tuple[str, float]:
    """
    XGBoost 原生训练接口的 KS 自定义评估函数。

    Parameters
    ----------
    preds : np.ndarray
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
    preds : np.ndarray
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

    Attributes
    ----------
    None
        该指标对象不维护可变实例状态。

    Notes
    -----
    ``evaluate`` 取 ``approxes[0]`` 并展平为一维数组，返回 0～1 范围的 KS 和固定权重 1.0。
    ``get_final_error`` 将 KS 恢复为百分制；样本权重不参与 KS 计算。

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
            CatBoost 传入的样本权重，当前实现不使用，不影响 KS 或返回权重。

        Returns
        -------
        tuple of float, float
            0～1 范围的 KS 与固定权重 1.0。

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
        将 CatBoost 内部 0～1 范围的结果恢复为百分制 KS。

        Parameters
        ----------
        error : float
            CatBoost 内部传入的 0～1 范围指标值。
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
