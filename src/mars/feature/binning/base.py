"""MARS 分箱器共享基类。"""

from __future__ import annotations

import gc
import multiprocessing
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Set, TypeVar, Union, cast

import numpy as np
import pandas as pd
import polars as pl

from mars._compat import collect_streaming
from mars.compute import (
    OrderedMetricSortBy,
    binary_distribution_exprs,
    binary_metric_exprs,
    normalize_ordered_metric_sort_by,
    normalized_auc_expr,
    ordered_binary_metric_exprs,
)
from mars.compute.binning import normalize_binary_target_column
from mars.core.base import MarsTransformer
from mars.feature.binning._json_codec import (
    decode_json_value,
    encode_json_value,
    read_json_artifact,
    write_json_artifact,
)
from mars.utils.decorators import time_it
from mars.utils.logger import logger

_MarsBinnerT = TypeVar("_MarsBinnerT", bound="MarsBinnerBase")


class MarsBinnerBase(MarsTransformer):
    """
    MARS 分箱器抽象基类。

    基类定义公共状态、分箱索引和数值、类别特征的转换方式。
    子类负责学习切点或类别分组规则，基类则负责缓存管理、映射导出、
    WOE 物化、统计报告和 SQL 生成。

    Attributes
    ----------
    bin_cuts_ : dict of str to list of float
        数值特征的切点列表，两端为 ``-inf`` 和 ``inf``；全空特征可为空列表。
    cat_cuts_ : dict of str to list of list
        类别特征的分组规则。每个子列表中的类别映射到同一个正常箱。
    bin_mappings_ : dict of str to dict of int to str
        分箱索引到展示标签的映射。
    bin_woes_ : dict of str to dict of int to float
        各特征的分箱索引到 WOE 值的映射。
    feature_names_in_ : list of str
        拟合时输入的原始特征列名。
    fit_failures_ : dict of str to str
        拟合过程中失败的特征及其诊断信息。

    Notes
    -----
    索引协议如下：
    ``Missing`` 为 ``-1``，
    ``Other`` 为 ``-2``，
    ``Special`` 从 ``-3`` 开始向负方向扩展，
    正常分箱索引从 ``0`` 开始递增。

    Examples
    --------
    >>> issubclass(MarsNativeBinner, MarsBinnerBase)
    True
    """

    # 类型常量: 用于快速判定数值列
    NUMERIC_DTYPES: Set[pl.DataType] = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    }

    # 索引协议常量
    IDX_MISSING: int = -1
    IDX_OTHER: int = -2
    IDX_SPECIAL_START: int = -3

    def __init__(
        self,
        n_bins: int = 10,
        special_values: List[Union[int, float, str]] | None = None,
        missing_values: List[Union[int, float, str]] | None = None,
        join_threshold: int = 100,
        n_jobs: int = -1
   ) -> None:
        """
        配置分箱数量、特殊值、缺失值和并行资源。

        Parameters
        ----------
        n_bins : int
            正常分箱数量上限，不含 Missing、Other 和 Special 箱。
            单调性约束、样本量或重复切点可能使实际箱数减少。
        special_values : List[Union[int, float, str]] | None
            需要单独成箱的业务取值，例如 ``-999``、``-1``。
            这些值不参与正常区间切分，使用从 ``-3`` 开始的负数索引；默认不额外配置。
        missing_values : List[Union[int, float, str]] | None
            除原生 null 和 NaN 外，需要映射到 Missing 箱的取值；默认不额外配置。
        join_threshold : int
            类别转换时选择映射实现的阈值，默认为 100。
            已拟合类别映射的条目数大于该值时使用 Hash Join，否则使用 ``replace``。
        n_jobs : int
            并行任务数。``-1`` 使用 CPU 核心数减 1，至少为 1；
            ``1`` 使用单任务，其余值传给具体分箱算法。

        Notes
        -----
        初始化仅保存配置和空状态；``fit`` 按算法需要创建进程池或线程池。
        """
        super().__init__()
        self.features: list[str] = []
        self.cat_features: list[str] = []
        self.n_bins = n_bins
        self.special_values = special_values if special_values is not None else []
        self.missing_values = missing_values if missing_values is not None else []
        self.join_threshold = join_threshold
        self._requested_n_jobs = n_jobs
        self.n_jobs = max(1, multiprocessing.cpu_count() - 1) if n_jobs == -1 else n_jobs

        # 状态属性初始化
        self.bin_cuts_: Dict[str, List[float]] = {}
        self.cat_cuts_: Dict[str, List[List[Any]]] = {}
        self.bin_mappings_: Dict[str, Dict[int, str]] = {}
        self.bin_woes_: Dict[str, Dict[int, float]] = {}

        # 缓存引用
        self._cache_X: pl.DataFrame | None = None
        self._cache_y: Any | None = None

        self.fit_failures_: Dict[str, str] = {}
        self._fit_report_rows: list[dict[str, Any]] = []
        self._user_bin_rules: dict[str, Any] = {}

    def fit(
        self: _MarsBinnerT,
        X: pl.DataFrame | pd.DataFrame,
        y: pl.Series | pd.Series | np.ndarray | list[Any] | None = None,
        *,
        features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> _MarsBinnerT:
        """
        拟合分箱器并缓存本次分箱特征范围。

        Parameters
        ----------
        X : pl.DataFrame | pd.DataFrame
            输入特征矩阵。
        y : pl.Series | pd.Series | np.ndarray | list[Any] | None
            与 ``X`` 按行对应的目标标签。CART、LiteOpt 和 Optimal 分箱要求提供；
            无监督分箱可省略。提供标签后还会计算 WOE，标签要求由具体算法确定。
        features : list[str] | None
            本次拟合的特征列；不传时由具体分箱器使用全部候选列。
        cat_features : list[str] | None
            强制按类别处理的列；不传时由具体分箱器按数据类型识别。

        Returns
        -------
        MarsBinnerBase
            拟合完成后的当前分箱器实例。

        Raises
        ------
        ValueError
            请求的特征或类别特征不在输入中时抛出。

        Notes
        -----
        重复拟合会重新学习本轮规则、WOE 和诊断，保留构造配置。
        ``update_bins`` 明确设置的规则在本轮仍有同类型特征时继续生效。
        失败拟合后不能转换、评估或保存旧结果；可以读取本轮诊断并再次拟合。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2)
        >>> binner.fit(X, features=["age"]).features
        ['age']
        """
        self._reset_fit_state()
        self.features = list(features or [])
        self.cat_features = list(cat_features or [])
        succeeded = False
        try:
            input_columns = list(X.columns)
            missing_features = [feature for feature in self.features if feature not in input_columns]
            missing_cat_features = [
                feature for feature in self.cat_features if feature not in input_columns
            ]
            if missing_features or missing_cat_features:
                raise ValueError(
                    "Binner fit received feature names that are absent from X: "
                    f"features={missing_features}, cat_features={missing_cat_features}."
                )
            super().fit(X, y)
            # 父类完成算法调度后，仍需验证诊断及映射，才发布本轮可用状态。
            self._is_fitted = False
            X_pl = cast(pl.DataFrame, self._ensure_polars_dataframe(X))
            self._apply_user_bin_rules()
            self._finalize_fit_report(X_pl)
            self._transform_impl(
                X_pl,
                features=self._usable_rule_features(),
                return_type="index",
                woe_batch_size=200,
                lazy=True,
            ).collect_schema()
            succeeded = True
        finally:
            if not succeeded:
                # 仅保留本轮失败诊断，清掉任何局部规则和训练缓存，拒绝读取旧成果。
                self._reset_fit_state(clear_diagnostics=False)
                for row in self._fit_report_rows:
                    row["usable"] = False
                    row["status"] = "failed"
        self._is_fitted = True
        return self

    def _reset_fit_state(self, *, clear_diagnostics: bool = True) -> None:
        """清理每次拟合的已学状态，保留构造参数和用户明确更新的规则。"""
        self._is_fitted = False
        self.feature_names_in_ = []
        self.bin_cuts_ = {}
        self.cat_cuts_ = {}
        self.bin_mappings_ = {}
        self.bin_woes_ = {}
        self._cache_X = None
        self._cache_y = None
        if clear_diagnostics:
            self.fit_failures_ = {}
            self._fit_report_rows = []
        self._reset_extra_fit_state()

    def _reset_extra_fit_state(self) -> None:
        """供子类清理专属已学状态，不改变算法配置。"""

    def _apply_user_bin_rules(self) -> None:
        """仅对本轮已有同类型规则的特征应用用户明确更新的规则。"""
        for feature, splits in self._user_bin_rules.items():
            categorical = bool(splits) and isinstance(splits[0], list)
            if categorical and feature in self.cat_cuts_:
                dtype = self._cache_X.schema[feature] if self._cache_X is not None else pl.Null
                expected_prefix = self._category_key_expr_prefix(dtype)
                if all(
                    self._category_key(value).startswith(expected_prefix)
                    for group in splits for value in group
                ):
                    self.cat_cuts_[feature] = deepcopy(splits)
            elif not categorical and feature in self.bin_cuts_:
                self.bin_cuts_[feature] = deepcopy(splits)

    def _usable_rule_features(self) -> list[str]:
        """按拟合顺序返回具备数值或类别规则的特征。"""
        rule_features = set(self.bin_cuts_) | set(self.cat_cuts_)
        ordered = [feature for feature in self.feature_names_in_ if feature in rule_features]
        remaining = sorted(rule_features - set(ordered))
        return ordered + remaining

    def _finalize_fit_report(self, X: pl.DataFrame) -> None:
        """根据拟合规则、输入分布与失败原因生成统一诊断。"""
        requested = self.features or list(X.columns)
        categorical = set(self.cat_features)
        rows: list[dict[str, Any]] = []
        for feature in requested:
            dtype = X.schema.get(feature)
            is_categorical = feature in categorical or feature in self.cat_cuts_
            has_rule = feature in self.bin_cuts_ or feature in self.cat_cuts_
            reason = self.fit_failures_.get(feature)
            status = "fitted"
            if feature not in X.columns:
                status = "failed"
                reason = reason or "feature is absent from the fit input"
            elif dtype == pl.Null or X.get_column(feature).null_count() == X.height:
                status = "all_missing"
            elif has_rule and X.get_column(feature).n_unique() <= 1:
                status = "constant"
            elif reason and has_rule:
                status = "fallback"
            elif not has_rule:
                status = "failed"
                reason = reason or "no usable binning rule was produced"

            if feature in self.bin_cuts_:
                cuts = self.bin_cuts_[feature]
                n_bins = max(len(cuts) - 1, 1)
            elif feature in self.cat_cuts_:
                n_bins = max(len(self.cat_cuts_[feature]), 1)
            else:
                n_bins = 0
            rows.append(
                {
                    "feature": feature,
                    "dtype": str(dtype) if dtype is not None else None,
                    "feature_type": "categorical" if is_categorical else "numeric",
                    "status": status,
                    "usable": bool(has_rule),
                    "n_bins": int(n_bins),
                    "reason": reason,
                }
            )

        self._fit_report_rows = rows
        if not any(bool(row["usable"]) for row in rows):
            self._is_fitted = False
            reasons = {
                str(row["feature"]): str(row["reason"])
                for row in rows
            }
            raise ValueError(f"Binner fit produced no usable feature rules: {reasons}.")

    def get_fit_report(self) -> pl.DataFrame:
        """
        返回逐特征拟合诊断表。

        Returns
        -------
        pl.DataFrame
            固定包含 ``feature``、``dtype``、``feature_type``、``status``、
            ``usable``、``n_bins`` 与 ``reason``。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> report = MarsNativeBinner(n_bins=2).fit(X).get_fit_report()
        >>> report["status"].to_list()
        ['fitted']
        """
        schema = {
            "feature": pl.Utf8,
            "dtype": pl.Utf8,
            "feature_type": pl.Utf8,
            "status": pl.Utf8,
            "usable": pl.Boolean,
            "n_bins": pl.Int64,
            "reason": pl.Utf8,
        }
        return pl.DataFrame(self._fit_report_rows, schema=schema)

    def transform(
        self,
        X: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame],
        *,
        features: list[str] | None = None,
        on_missing: Literal["error", "warn", "ignore"] = "error",
        return_type: Literal["index", "label", "woe"] = "index",
        woe_batch_size: int = 200,
        lazy: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame]:
        """
        按当前分箱规则将输入特征映射为索引、标签或 WOE 值。

        Parameters
        ----------
        X : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame]
            待转换的数据集。
        features : list[str] | None
            需要转换的已拟合规则子集。不传时要求输入包含全部可用规则特征。
        on_missing : Literal['error', 'warn', 'ignore']
            输入缺少请求特征时的处理策略。默认报错；``warn`` 与 ``ignore``
            只转换实际存在的已知规则列，其中 ``warn`` 还会记录缺列警告。
        return_type : Literal['index', 'label', 'woe']
            输出形式。``"index"`` 返回分箱索引，``"label"`` 返回分箱标签，
            ``"woe"`` 返回对应分箱的 WOE 值。
        woe_batch_size : int
            当 ``return_type="woe"`` 且请求特征尚未物化 WOE 映射时，
            计算必要训练统计的特征列批宽；已有映射复用，不准备未请求特征。
        lazy : bool
            是否强制返回 ``pl.LazyFrame``。默认 ``False`` 沿用输入的执行模式；
            输入本身为 LazyFrame 时，结果仍为 LazyFrame。样本转换保持延迟，
            但首次请求 WOE 时仍须先计算缺失的训练统计。

        Returns
        -------
        pl.DataFrame or pd.DataFrame or pl.LazyFrame
            保留原始列，并追加 ``_bin`` 或 ``_woe`` 转换列。非延迟结果默认返回 Polars 表；
            设置 ``set_output("pandas")`` 后返回 Pandas 表。

        Raises
        ------
        ValueError
            特征缺失、策略无效或 WOE 映射不完整时抛出。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, features=["age"])
        >>> binner.transform(X).columns
        ['age_bin']
        """
        self._check_is_fitted()
        X_pl = self._ensure_polars_dataframe(X)
        schema = X_pl.collect_schema() if isinstance(X_pl, pl.LazyFrame) else X_pl.schema
        active_features = self._resolve_transform_features(schema.names(), features, on_missing)

        if return_type == "woe":
            missing_woe = [feature for feature in active_features if feature not in self.bin_woes_]
            if missing_woe:
                self._materialize_woe(woe_batch_size, features=missing_woe)
                missing_woe = [
                    feature for feature in active_features if feature not in self.bin_woes_
                ]
            if missing_woe:
                raise ValueError(
                    "WOE transform requires fitted WOE mappings for every requested feature; "
                    f"missing mappings: {missing_woe}. Fit with a binary target first."
                )

        X_new = self._transform_impl(
            X_pl,
            features=active_features,
            return_type=return_type,
            woe_batch_size=woe_batch_size,
            lazy=lazy,
        )
        return self._format_output(X_new)

    def _resolve_transform_features(
        self,
        current_columns: list[str],
        features: list[str] | None,
        on_missing: Literal["error", "warn", "ignore"],
    ) -> list[str]:
        """只用列身份校验规则和缺列策略，供转换与逐批评估共用。"""
        if on_missing not in {"error", "warn", "ignore"}:
            raise ValueError("on_missing must be one of {'error', 'warn', 'ignore'}.")
        fitted_features = self._usable_rule_features()
        requested_features = list(features) if features is not None else fitted_features
        unknown_features = [
            feature for feature in requested_features if feature not in fitted_features
        ]
        if unknown_features:
            raise ValueError(
                "Binner transform requested features without usable fitted rules: "
                f"{unknown_features}."
            )

        missing_features = [
            feature for feature in requested_features if feature not in current_columns
        ]
        if missing_features and on_missing == "error":
            raise ValueError(
                "Binner transform input is missing required features: "
                f"{missing_features}. Pass `features` for an intentional subset or set "
                "`on_missing` explicitly."
            )
        if missing_features and on_missing == "warn":
            logger.warning(
                "Binner transform skipped missing requested features: %s",
                missing_features,
            )
        return [
            feature for feature in requested_features if feature in current_columns
        ]

    def fit_transform(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Any | None = None,
        *,
        features: List[str] | None = None,
        cat_features: List[str] | None = None,
        return_type: Literal["index", "label", "woe"] = "index",
        woe_batch_size: int = 200,
        lazy: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame]:
        """
        先拟合分箱器，再返回分箱转换结果。

        Parameters
        ----------
        X : Union[pl.DataFrame, pd.DataFrame]
            输入特征矩阵。
        y : Any | None
            目标变量。无监督分箱场景下可为空。
        features : List[str] | None
            本次拟合和转换的特征列；不传时使用全部候选列。
        cat_features : List[str] | None
            明确指定的类别特征列。
        return_type : Literal['index', 'label', 'woe']
            转换结果形式。
        woe_batch_size : int
            计算 WOE 映射时的批处理特征数。
        lazy : bool
            是否返回 ``pl.LazyFrame``。

        Returns
        -------
        pl.DataFrame or pd.DataFrame or pl.LazyFrame
            分箱转换结果。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2)
        >>> binner.fit_transform(X, return_type="label").columns
        ['age_bin']
        """
        return self.fit(X, y, features=features, cat_features=cat_features).transform(
            X,
            features=features,
            return_type=return_type,
            woe_batch_size=woe_batch_size,
            lazy=lazy,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        将分箱器状态编码为版本化 JSON-compatible 字典。

        Returns
        -------
        dict of str to Any
            包含 artifact 类型、schema 版本、分箱器类型、参数与状态。

        Notes
        -----
        返回结果不包含缓存的训练数据本体。旧版仅含 ``params/state`` 的载荷
        不再受支持。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, features=["age"])
        >>> binner.to_dict()["schema_version"]
        1
        """
        from mars import __version__

        self._check_is_fitted()
        return {
            "artifact_type": "mars_binner",
            "schema_version": 1,
            "binner_type": self._serialization_type(),
            "mars_version": __version__,
            "params": encode_json_value(self._serialization_params()),
            "state": encode_json_value(self._serialization_state()),
        }

    def _serialization_type(self) -> str:
        """返回 artifact 使用的稳定分箱器类型标识。"""
        type_map = {
            "MarsNativeBinner": "native",
            "MarsLiteOptBinner": "lite_opt",
            "MarsOptimalBinner": "optimal",
        }
        try:
            return type_map[type(self).__name__]
        except KeyError as exc:
            raise TypeError(
                f"Unsupported binner class for JSON serialization: {type(self).__name__}."
            ) from exc

    def _serialization_params(self) -> dict[str, Any]:
        """返回基类构造配置，子类在此基础上补充专属参数。"""
        return {
            "n_bins": self.n_bins,
            "special_values": self.special_values,
            "missing_values": self.missing_values,
            "join_threshold": self.join_threshold,
            "n_jobs": self._requested_n_jobs,
        }

    def _serialization_state(self) -> dict[str, Any]:
        """返回不含训练缓存的基类拟合状态。"""
        return {
            "features": self.features,
            "cat_features": self.cat_features,
            "feature_names_in_": self.feature_names_in_,
            "bin_cuts_": self.bin_cuts_,
            "cat_cuts_": self.cat_cuts_,
            "bin_mappings_": self.bin_mappings_,
            "bin_woes_": self.bin_woes_,
            "fit_failures_": self.fit_failures_,
            "fit_report_rows": self._fit_report_rows,
            "user_bin_rules": self._user_bin_rules,
        }

    @staticmethod
    def _build_trend_shape_frame(
        grouped_woe_df: pl.DataFrame,
        trend_col_name: str = "trend_shape"
    ) -> pl.DataFrame:
        """
        基于聚合后的 WOE 列表构建趋势形态结果表。

        Parameters
        ----------
        grouped_woe_df : pl.DataFrame
            至少包含 `feature` 与 `woe` 列的聚合结果。
        trend_col_name : str
            输出趋势列名。

        Returns
        -------
        pl.DataFrame
            包含特征名与趋势标签的结果表。
        """
        rows = []
        for row in grouped_woe_df.iter_rows(named=True):
            rows.append({
                "feature": row["feature"],
                trend_col_name: MarsBinnerBase._detect_trend_scientific(row["woe"]),
            })

        if not rows:
            return pl.DataFrame(schema={"feature": pl.Utf8, trend_col_name: pl.Utf8})

        return pl.DataFrame(rows)

    @classmethod
    def from_dict(cls: type[MarsBinnerBase], data: Dict[str, Any]) -> MarsBinnerBase:
        """
        从当前版本的自描述字典恢复分箱器实例。

        Parameters
        ----------
        data : Dict[str, Any]
            由 ``to_dict`` 生成的状态字典。

        Returns
        -------
        MarsBinnerBase
            恢复后的已拟合分箱器实例。

        Raises
        ------
        ValueError
            载荷类型、schema 版本、分箱器类型或状态结构不受支持时抛出。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> source = MarsNativeBinner(n_bins=2).fit(X)
        >>> restored = MarsBinnerBase.from_dict(source.to_dict())
        >>> type(restored).__name__
        'MarsNativeBinner'
        """
        if data.get("artifact_type") != "mars_binner":
            if set(data) <= {"params", "state"}:
                raise ValueError(
                    "Legacy binner dictionaries are not supported by schema_version=1; "
                    "refit and export the binner with MARS 0.0.26."
                )
            raise ValueError("Invalid binner artifact_type; expected 'mars_binner'.")
        if data.get("schema_version") != 1:
            raise ValueError(
                "Unsupported binner schema_version: "
                f"{data.get('schema_version')!r}; supported version is 1."
            )
        if not isinstance(data.get("mars_version"), str):
            raise ValueError("Binner artifact `mars_version` must be a string.")

        from mars.feature.binning.lite_opt import MarsLiteOptBinner
        from mars.feature.binning.native import MarsNativeBinner
        from mars.feature.binning.optimal import MarsOptimalBinner

        registry: dict[str, type[MarsBinnerBase]] = {
            "native": MarsNativeBinner,
            "lite_opt": MarsLiteOptBinner,
            "optimal": MarsOptimalBinner,
        }
        binner_type = data.get("binner_type")
        if not isinstance(binner_type, str) or binner_type not in registry:
            raise ValueError(f"Unknown binner_type in artifact: {binner_type!r}.")
        target_cls = registry[binner_type]
        if cls is not MarsBinnerBase and cls is not target_cls:
            raise ValueError(
                f"Artifact contains {target_cls.__name__}, but {cls.__name__}.from_dict() "
                "was requested."
            )

        params = decode_json_value(data.get("params"))
        state = decode_json_value(data.get("state"))
        if not isinstance(params, dict) or not isinstance(state, dict):
            raise ValueError("Binner artifact `params` and `state` must decode to mappings.")
        required_state = {
            "features",
            "cat_features",
            "feature_names_in_",
            "bin_cuts_",
            "cat_cuts_",
            "bin_mappings_",
            "bin_woes_",
            "fit_failures_",
            "fit_report_rows",
        }
        if binner_type == "lite_opt":
            required_state.update({"fitted_trends_", "candidate_scores_"})
        missing_state = sorted(required_state - set(state))
        if missing_state:
            raise ValueError(f"Binner artifact state is missing required fields: {missing_state}.")
        user_rules = state.get("user_bin_rules", {})
        if not isinstance(user_rules, dict) or any(
            not isinstance(feature, str) or not isinstance(splits, list)
            for feature, splits in user_rules.items()
        ):
            raise ValueError("Binner artifact `user_bin_rules` must map feature names to lists.")

        instance = target_cls(**params)
        instance._restore_serialization_state(state)
        return instance

    def _restore_serialization_state(self, state: dict[str, Any]) -> None:
        """恢复基类状态，并调用子类专属状态恢复钩子。"""
        self.features = list(state["features"])
        self.cat_features = list(state["cat_features"])
        self.feature_names_in_ = list(state["feature_names_in_"])
        self.bin_cuts_ = dict(state["bin_cuts_"])
        self.cat_cuts_ = dict(state["cat_cuts_"])
        self.bin_mappings_ = dict(state["bin_mappings_"])
        self.bin_woes_ = dict(state["bin_woes_"])
        self.fit_failures_ = dict(state["fit_failures_"])
        self._fit_report_rows = list(state["fit_report_rows"])
        self._user_bin_rules = deepcopy(state.get("user_bin_rules", {}))
        self._restore_legacy_boolean_rules()
        self._cache_X = None
        self._cache_y = None
        self._restore_extra_serialization_state(state)
        self._is_fitted = True

    def _restore_extra_serialization_state(self, state: dict[str, Any]) -> None:
        """供子类恢复额外拟合状态。"""

    def _restore_legacy_boolean_rules(self) -> None:
        """仅依据已保存 Boolean schema 恢复旧求解器的字符串布尔类别。"""
        for row in self._fit_report_rows:
            feature = row.get("feature")
            if row.get("dtype") != "Boolean" or feature not in self.cat_cuts_:
                continue
            self.cat_cuts_[feature] = [
                [
                    value.lower() == "true"
                    if isinstance(value, str) and value.lower() in {"true", "false"}
                    else value
                    for value in group
                ]
                for group in self.cat_cuts_[feature]
            ]

    def save_json(self, path: str | Path) -> None:
        """
        将当前分箱规则原子写入正式 JSON artifact。

        Parameters
        ----------
        path : str | Path
            输出文件路径；父目录必须已经存在。

        Returns
        -------
        None
            方法仅写入 artifact 文件。

        Examples
        --------
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(n_bins=2).fit(X)
        >>> with TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "binner.json"
        ...     binner.save_json(path)
        ...     path.exists()
        True
        """
        self._check_is_fitted()
        write_json_artifact(path, self.to_dict())

    @classmethod
    def load_json(cls: type[MarsBinnerBase], path: str | Path) -> MarsBinnerBase:
        """
        从正式 JSON artifact 恢复具体分箱器。

        Parameters
        ----------
        path : str | Path
            已存在的 artifact 文件路径。

        Returns
        -------
        MarsBinnerBase
            artifact 指定的具体分箱器子类实例。

        Examples
        --------
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> with TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "binner.json"
        ...     MarsNativeBinner(n_bins=2).fit(X).save_json(path)
        ...     restored = MarsBinnerBase.load_json(path)
        ...     type(restored).__name__
        'MarsNativeBinner'
        """
        return cls.from_dict(read_json_artifact(path))

    def __getstate__(self) -> dict[str, Any]:
        """
        Pickle 序列化时的钩子。

        在保存模型时, 自动剔除巨大的训练数据缓存, 只保留配置和计算结果。
        """
        state = self.__dict__.copy()
        # 移除大数据缓存, 防止模型文件变成几百 MB
        state["_cache_X"] = None
        state["_cache_y"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Pickle 反序列化时的钩子。

        恢复模型状态, 并将缓存初始化为 None。
        """
        self.__dict__.update(state)
        # 确保属性存在, 防止 AttributeError
        if "_cache_X" not in self.__dict__:
            self._cache_X = None
        if "_cache_y" not in self.__dict__:
            self._cache_y = None
        if "_fit_report_rows" not in self.__dict__:
            self._fit_report_rows = []
        if "_requested_n_jobs" not in self.__dict__:
            self._requested_n_jobs = self.n_jobs
        if "_user_bin_rules" not in self.__dict__:
            self._user_bin_rules = {}
        self._restore_legacy_boolean_rules()

    def clear_cache(self) -> None:
        """
        清理缓存的训练数据引用。

        Notes
        -----
        该方法会清空 ``_cache_X`` 和 ``_cache_y``，并主动触发一次垃圾回收。
        适合在模型训练完成、无需再次即时重算统计量时调用。

        Returns
        -------
        None
            函数仅清理训练数据缓存。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, features=["age"])
        >>> binner.clear_cache()
        >>> binner._cache_X is None
        True
        """
        self._cache_X = None
        self._cache_y = None
        gc.collect()

    def _get_safe_values(self, dtype: pl.DataType, values: List[Any]) -> List[Any]:
        """
        根据列类型筛选可安全参与比较的配置值。

        按当前列类型筛选缺失值和特殊值配置，避免比较时发生类型不匹配。

        Parameters
        ----------
        dtype : pl.DataType
            当前处理列的原始数据类型。
        values : List[Any]
            用户配置的取值列表，例如 ``[-999, 'unknown', None]``。

        Returns
        -------
        List[Any]
            可与当前列类型比较的配置值列表。

        Notes
        -----
        数值列只接受非布尔的 int、float 配置；Boolean 列只接受布尔配置，
        避免 ``True == 1`` 造成误匹配。其他列的配置值转为字符串。
        ``None`` 和浮点 NaN 由专用缺失判断处理，不加入配置比较列表。
        """
        if not values:
            return []

        is_numeric = dtype in self.NUMERIC_DTYPES
        safe_vals: List[Union[int, float, str]] = []

        for v in values:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue

            if is_numeric:
                # 数值列: 严格保留数值, 剔除 bool (True==1 歧义) 和字符串
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    safe_vals.append(v)
            elif dtype == pl.Boolean:
                # 布尔配置只匹配原生布尔值，避免 True 与数字 1 的隐式等价。
                if isinstance(v, bool):
                    safe_vals.append(v)
            else:
                # 非数值列: 宽容处理, 全部转为字符串以匹配 Categorical/String 列
                safe_vals.append(str(v))

        return safe_vals

    def get_bin_mapping(self, col: str) -> Dict[int, str]:
        """
        获取指定特征的分箱标签映射。

        Parameters
        ----------
        col : str
            特征名称。

        Returns
        -------
        dict of int to str
            分箱索引到标签的映射字典。

        Raises
        ------
        KeyError
            特征没有可用分箱标签映射时抛出。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, features=["age"])
        >>> isinstance(binner.get_bin_mapping("age"), dict)
        True
        """
        self._check_is_fitted()
        if col not in self.bin_mappings_:
            raise KeyError(f"Feature {col!r} has no fitted bin mapping.")
        return dict(self.bin_mappings_[col])

    def _is_numeric(self, series: pl.Series) -> bool:
        """判断输入序列是否为数值类型。"""
        if series.dtype == pl.Null:
            return False
        return series.dtype in self.NUMERIC_DTYPES

    @staticmethod
    def _category_key(value: Any) -> str:
        """编码原生类别键，区分 Boolean，其余沿用既有字符串匹配语义。"""
        if isinstance(value, (bool, np.bool_)):
            return "b:true" if value else "b:false"
        return f"s:{value}"

    @staticmethod
    def _category_key_expr(col: str, dtype: pl.DataType) -> pl.Expr:
        """生成与持久化类别值一致的类型感知匹配表达式，保留空值。"""
        prefix = MarsBinnerBase._category_key_expr_prefix(dtype)
        return pl.lit(prefix) + pl.col(col).cast(pl.Utf8)

    @staticmethod
    def _category_key_expr_prefix(dtype: pl.DataType) -> str:
        """获取类别匹配的物理类型前缀，不改变字符串大小写或编号。"""
        if dtype == pl.Boolean:
            return "b:"
        return "s:"

    def _materialize_woe(
        self, batch_size: int = 200, *, features: list[str] | None = None,
    ) -> None:
        """
        将分箱统计分布转化为 WOE 映射表。

        该方法按特征逐列聚合分箱统计量，避免将宽表物化成超长表，
        从而显著降低大样本、超宽表场景下的峰值内存占用。

        Parameters
        ----------
        batch_size : int
            分批处理的特征数量。
        features : list[str] | None
            仅准备请求且缺失映射的规则特征；不传时评估全部可用规则。
        """
        if self._cache_X is None or self._cache_y is None:
            raise ValueError(
                "WOE mappings are unavailable because the binner has no cached binary target. "
                "Fit with y or load an artifact containing `bin_woes_`."
            )

        y_series = pl.Series(name="_y_tmp", values=self._cache_y)
        self.profile_bin_performance(
            self._cache_X,
            y_series,
            update_woe=True,
            batch_size=batch_size,
            include_bin_index=True,
            features=features,
        )

    def _transform_impl(
        self,
        X: Union[pl.DataFrame, pl.LazyFrame],
        features: list[str] | None = None,
        return_type: Literal["index", "label", "woe"] = "index",
        woe_batch_size: int = 200,
        lazy: bool = False
   ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        按已拟合规则追加分箱转换列。

        支持数值和类别特征，保留输入的 Eager 或 Lazy 执行模式。

        Parameters
        ----------
        X : Union[pl.DataFrame, pl.LazyFrame]
            待转换的特征表。
        features : list[str] | None
            已由公共 ``transform`` 校验的规则特征子集。
        return_type : Literal['index', 'label', 'woe']
            输出格式。``index`` 为 Int16 分箱索引，``label`` 为 Utf8 展示标签，
            ``woe`` 为 Float32 WOE 值。
        woe_batch_size : int
            仅在请求 WOE 且映射尚未计算时生效，控制每批计算的特征数。
        lazy : bool
            为 ``True`` 时强制返回 LazyFrame；默认 ``False`` 沿用输入执行模式。

        Returns
        -------
        Union[pl.DataFrame, pl.LazyFrame]
            保留原始列，并追加 ``_bin`` 或 ``_woe`` 转换列的表。

        Notes
        -----
        Missing 索引为 -1，Other 为 -2，Special 从 -3 向负方向扩展，正常箱从 0 开始。
        数值列用 ``pl.cut`` 切分，缺失值优先于特殊值，特殊值优先于正常区间。
        类别匹配键保留布尔类型标记，文本保持原始大小写和编号。
        类别映射条目数超过 ``join_threshold`` 时使用 Hash Join，否则使用 ``replace``。
        临时列名会避开已有列，转换结束后移除临时列。
        """
        # 执行模式必须在任何样本派生列和类别 join 之前确定；schema 解析不执行样本计划。
        if lazy and isinstance(X, pl.DataFrame):
            X = X.lazy()
        exprs: list[pl.Expr] = []
        temp_join_cols: list[str] = []

        # 索引协议常量: 与下游 Profiler 对齐
        IDX_MISSING = -1
        IDX_OTHER   = -2
        IDX_SPECIAL_START = -3

        # 获取 Schema
        schema_map = X.collect_schema() if isinstance(X, pl.LazyFrame) else X.schema
        current_columns = schema_map.names()
        reserved_columns = set(current_columns)

        def _temporary_column(base_name: str) -> str:
            """为本次转换分配不覆盖用户输入或其他转换结果的临时列。"""
            candidate = base_name
            while candidate in reserved_columns:
                candidate += "_"
            reserved_columns.add(candidate)
            return candidate

        all_train_cols = (
            list(features)
            if features is not None
            else self._usable_rule_features()
        )

        for col in all_train_cols:
            if col not in current_columns:
                continue

            # 计算类型安全值, 防止例如在 Int 列上查询 "unknown" 导致的崩溃
            col_dtype = schema_map[col]
            safe_missing_vals: List[int|float] = self._get_safe_values(col_dtype, self.missing_values)
            safe_special_vals: List[int|float] = self._get_safe_values(col_dtype, self.special_values)
            is_numeric_col = col_dtype in self.NUMERIC_DTYPES

            # Part A: 数值型分箱 (Numeric Binning)
            if col in self.bin_cuts_:
                cuts = self.bin_cuts_[col]

                # 缺失值逻辑: Is Null OR Is Missing Val
                missing_cond = pl.col(col).is_null()
                if is_numeric_col:
                    missing_cond |= pl.col(col).cast(pl.Float64).is_nan()
                for v in safe_missing_vals:
                    missing_cond |= (pl.col(col) == v)
                # 先将缺失值映射到统一索引，再叠加特殊值和常规分箱逻辑。
                layer_missing = pl.when(missing_cond).then(pl.lit(IDX_MISSING, dtype=pl.Int16))

                # 正常分箱逻辑: Cut
                raw_breaks = cuts[1:-1] if len(cuts) > 2 else []
                # cut 的 schema 推导不校验切点取值；显式拒绝非法规则，不能延迟为样本执行错误。
                if any(not np.isfinite(value) for value in raw_breaks):
                    raise ValueError(
                        f"Binner numeric rule for {col!r} contains non-finite internal cuts."
                    )
                # `pl.cut` 要求切点严格递增；高偏态数据可能产生重复分位点，这里统一去重并排序。
                breaks = sorted(list(set(raw_breaks)))

                col_mapping: Dict[int, str] = {IDX_MISSING: "Missing", IDX_OTHER: "Other"} # 分箱标签映射表 IDX -> Label

                # 无切点逻辑
                if not breaks:
                    col_mapping[0] = "00_[-inf, inf)"
                    layer_normal = pl.lit(0, dtype=pl.Int16) # 全部归为 0 号箱
                else:
                    for i in range(len(cuts) - 1):
                        low, high = cuts[i], cuts[i+1]
                        # 调用智能格式化函数
                        low_str = self._format_cut_point(low)
                        high_str = self._format_cut_point(high)
                        col_mapping[i] = f"{i:02d}_[{low_str}, {high_str})"

                    # 显式生成标签，确保后续转换稳定地落回业务索引。
                    bin_labels: List[str] = [str(i) for i in range(len(breaks) + 1)]
                    layer_normal = (
                        pl.col(col)
                        .cut(breaks, labels=bin_labels, left_closed=True)
                        # 先还原逻辑标签，再转回业务索引，避免直接读取底层物理 ID。
                        .cast(pl.Utf8)
                        .cast(pl.Int16)
                   )

                # 特殊值逻辑: 瀑布流覆盖
                current_branch = layer_normal
                if safe_special_vals:
                    for i in range(len(safe_special_vals)-1, -1, -1):
                        v = safe_special_vals[i]
                        idx = IDX_SPECIAL_START - i
                        col_mapping[idx] = f"Special_{v}"
                        # 注意这里的覆盖顺序: 后定义的优先级更高
                        current_branch = pl.when(pl.col(col) == v).then(pl.lit(idx, dtype=pl.Int16)).otherwise(current_branch)

                # 优先级为 Missing -> Special -> Normal。
                final_idx_expr = layer_missing.otherwise(current_branch)
                self.bin_mappings_[col] = col_mapping

            # Part B: 类别型分箱 (Categorical Binning)
            elif hasattr(self, 'cat_cuts_') and col in self.cat_cuts_:
                splits = self.cat_cuts_[col]
                cat_to_idx: Dict[str, int] = {}
                idx_to_label: Dict[int, str] = {IDX_MISSING: "Missing", IDX_OTHER: "Other"}

                # 未命中任何已知类别时，默认落入 Other 箱。
                default_bin_idx = IDX_OTHER

                # 更新映射表
                if safe_special_vals:
                    for i, val in enumerate(safe_special_vals):
                        idx_to_label[IDX_SPECIAL_START - i] = f"Special_{val}"

                for i, group in enumerate(splits):
                    disp_grp = group[:3] if len(group) > 3 else group
                    suffix = ",..." if len(group) > 3 else ""
                    idx_to_label[i] = f"{i:02d}_[{','.join(str(g) for g in disp_grp) + suffix}]"
                    for val in group:
                        val_str = self._category_key(val)
                        cat_to_idx[val_str] = i
                        # 若训练阶段预聚合了 Other 占位类别，则沿用该箱作为默认去向。
                        if isinstance(val, str) and val == "__Mars_Other_Pre__":
                            default_bin_idx = i

                self.bin_mappings_[col] = idx_to_label
                # 强转 String, 确保类别匹配安全
                target_col = self._category_key_expr(col, col_dtype)

                # 缺失值
                missing_cond = pl.col(col).is_null()
                if col_dtype in {pl.Float32, pl.Float64}:
                    missing_cond |= pl.col(col).is_nan()
                for v in safe_missing_vals:
                    missing_cond |= (target_col == self._category_key(v))
                layer_missing = pl.when(missing_cond).then(pl.lit(IDX_MISSING, dtype=pl.Int16))

                # 特殊值
                current_branch = pl.lit(default_bin_idx, dtype=pl.Int16)
                if safe_special_vals:
                    for i in range(len(safe_special_vals)-1, -1, -1):
                        v = safe_special_vals[i]
                        idx = IDX_SPECIAL_START - i
                        # 如果是特殊值则赋予 -3，否则掉入上一层的 current_branch (即 default_bin_idx)
                        current_branch = (
                            pl.when(target_col == self._category_key(v))
                            .then(pl.lit(idx, dtype=pl.Int16))
                            .otherwise(current_branch)
                       )

                # 路由: Join (高基数) vs Replace (低基数)
                if len(cat_to_idx) > self.join_threshold:
                    target_col_name = _temporary_column(f"_{col}_utf8_tmp")
                    index_col_name = _temporary_column(f"_idx_{col}")
                    X = X.with_columns(target_col.alias(target_col_name))
                    map_df = pl.DataFrame({
                        "_k": list(cat_to_idx.keys()),
                        index_col_name: list(cat_to_idx.values())
                    }).cast({"_k": pl.Utf8, index_col_name: pl.Int16})

                    # 跟随输入数据的 eager/lazy 形态，避免额外的执行模式切换。
                    join_tbl = map_df.lazy() if isinstance(X, pl.LazyFrame) else map_df
                    X = X.join(join_tbl, left_on=target_col_name, right_on="_k", how="left")
                    temp_join_cols.extend([index_col_name, target_col_name])

                    layer_normal = pl.col(index_col_name)
                else:
                    # 类别型特征的 Replace 逻辑
                    # 1. 显式转 String 确保匹配安全
                    str_map = {k: str(v) for k, v in cat_to_idx.items()}

                    # 保持 default=None，让特殊值和未知类别都进入后续兜底分支统一处理。
                    layer_normal = (
                        target_col
                        .replace_strict(str_map, default=None)
                        .cast(pl.Int16)
                    )

                # 最终的分箱表达式: Missing -> Normal (Join/Replace Result) -> Special/Other
                final_idx_expr = layer_missing.otherwise(
                    pl.when(layer_normal.is_not_null()).then(layer_normal).otherwise(current_branch)
                )
            else:
                continue

            if return_type == "index":
                exprs.append(final_idx_expr.alias(f"{col}_bin"))
            elif return_type == "woe":
                woe_map = self.bin_woes_.get(col, {})
                if not woe_map:
                    logger.warning(
                        "WOE mapping for column '%s' not found. Defaulting to 0.0.",
                        col,
                    )
                    exprs.append(pl.lit(0.0).alias(f"{col}_woe"))
                else:
                    clean_woe_map = {
                        int(key): float(value)
                        for key, value in woe_map.items()
                        if key is not None and not (
                            isinstance(key, float) and np.isnan(key)
                        )
                    }
                    exprs.append(
                        final_idx_expr
                        .replace_strict(clean_woe_map, default=0.0)
                        .cast(pl.Float32)
                        .alias(f"{col}_woe")
                    )
            else:
                str_map = {
                    str(key): value
                    for key, value in self.bin_mappings_.get(col, {}).items()
                }
                exprs.append(
                    final_idx_expr
                    .cast(pl.Utf8)
                    .replace(str_map)
                    .alias(f"{col}_bin")
                )

        return X.with_columns(exprs).drop(temp_join_cols)

    @staticmethod
    def _detect_trend_scientific(woes: List[float]) -> str:
        """用相邻 WOE 差分判断非递减、非递增和峰谷形态。"""
        y = np.array([w for w in woes if w is not None and not np.isnan(w)])
        n = len(y)

        if n < 2:
            return "scanty"

        # 计算差分
        diff = np.diff(y)

        # 允许相邻值相等，分别判断非递减和非递增。
        if np.all(diff >= 0):
            return "ascending"
        if np.all(diff <= 0):
            return "descending"

        if n < 3:
            return "undefined" # 非单调且点数少于3，无法构成峰谷

        # Peak (倒U型)
        #    Max 必须在中间 (0 < t < n-1)
        t_max = np.argmax(y)
        if 0 < t_max < n - 1:
            # 左侧单调增，右侧单调减
            if np.all(diff[:t_max] >= 0) and np.all(diff[t_max:] <= 0):
                return "peak"

        # Valley (U型)
        #    Min 必须在中间 (0 < t < n-1)
        t_min = np.argmin(y)
        if 0 < t_min < n - 1:
            # 左侧单调减，右侧单调增
            if np.all(diff[:t_min] <= 0) and np.all(diff[t_min:] >= 0):
                return "valley"

        return "undefined"

    @time_it
    def profile_bin_performance(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: pl.Series | pd.Series,
        update_woe: bool = True,
        batch_size: int = 100,
        include_bin_index: bool = False,
        ordered_metric_sort_by: OrderedMetricSortBy = "woe",
        features: list[str] | None = None,
    ) -> pl.DataFrame | pd.DataFrame:
        """
        计算分箱表现统计报告。

        Parameters
        ----------
        X : pl.DataFrame | pd.DataFrame
            原始特征数据集。
        y : pl.Series | pd.Series
            二分类目标标签，仅接受 0/1、Boolean 及对应的合法字符串。
            null、NaN 和空字符串表示未表现样本。``-1`` 视为非法标签，不表示未表现。
        update_woe : bool
            是否用本次已表现样本计算的 WOE 更新 ``bin_woes_``，默认为 ``True``。
            设为 ``False`` 时只返回统计结果，保留已有 WOE 映射。
        batch_size : int
            特征列批宽，必须为正整数。每批先投影必要原始列，再过滤已表现样本、
            转换和聚合；不限制原始表、训练缓存或最终分箱统计表的总内存。
        include_bin_index : bool
            是否在返回明细中保留内部 ``bin_index`` 列。默认不保留，保持既有
            报告展示顺序；轻量最优分箱等内部算法可开启该参数以稳定回溯预分箱顺序。
        ordered_metric_sort_by : OrderedMetricSortBy
            KS/AUC 的排序口径。默认 ``"woe"`` 适合普通特征预测力评估；
            传入 ``"bin_index"`` 时只用正常箱按原始箱序计算 KS/AUC。
        features : list[str] | None
            需要评估的已拟合规则子集；不传时要求 ``X`` 包含全部规则特征。

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            包含各特征各分箱统计量的明细表，通常包括样本数、坏样本数、
            分布占比、WOE、IV、KS、AUC 和 Lift 等指标。

        Raises
        ------
        ValueError
            标签含非法值、``X`` 与 ``y`` 行数不一致，或没有任何已表现样本时抛出。

        Notes
        -----
        该方法依赖当前分箱器已完成拟合，并会复用 ``transform(return_type="index")``
        的输出结果来执行聚合统计。评估前会剔除未表现样本，
        因此本表的样本数、分布和监督指标均使用已表现样本；分箱规则保持不变。合法单类别可生成计数和坏账率，
        缺少两类支持的监督指标沿用缺失状态；校验失败不更新已有 WOE。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> y = pl.Series("target", [0, 0, 1, 1])
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, y, features=["age"])
        >>> stats = binner.profile_bin_performance(X, y)
        >>> "feature" in stats.columns
        True
        """
        self._check_is_fitted()
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        X = cast(pl.DataFrame, self._ensure_polars_dataframe(X))
        active_features = self._resolve_transform_features(X.columns, features, "error")
        effective_ordered_metric_sort_by = normalize_ordered_metric_sort_by(ordered_metric_sort_by)

        raw_name = getattr(y, "name", None)
        if raw_name is None or raw_name == "":
            y_name = "target"
        else:
            y_name = str(raw_name)
        y = cast(pl.Series, self._ensure_polars_series(y, name=y_name))

        if len(y) != X.height:
            raise ValueError(f"Target 'y' length mismatch: X({X.height}) vs y({len(y)})")

        # 复用高层评估的标签契约，非法标签不能进入 bad 聚合或覆盖 WOE。
        y = normalize_binary_target_column(y.to_frame(), y_name).get_column(y_name)

        # 在分箱转换前剔除未表现样本，避免空标签被误计为好样本。
        observed_mask: pl.Series = y.is_not_null()
        if y.dtype in {pl.Float32, pl.Float64}:
            observed_mask = observed_mask & ~y.is_nan()
        observed_count = int(observed_mask.sum())
        if observed_count == 0:
            raise ValueError(
                "Target 'y' must contain at least one observed value after excluding null / NaN."
            )
        if observed_count < X.height:
            y = y.filter(observed_mask)

        agg_results: List[pl.DataFrame] = []
        for i in range(0, len(active_features), batch_size):
            batch_features = active_features[i : i + batch_size]
            batch_cols = [f"{feature}_bin" for feature in batch_features]

            # 原表投影共享现有 buffer；过滤和分箱只分配本批的数组，不复制全宽已表现人口。
            batch_input: pl.LazyFrame = X.select(batch_features).lazy()
            if observed_count < X.height:
                batch_input = batch_input.filter(observed_mask)
            X_bin_lazy = cast(
                pl.LazyFrame,
                self.transform(
                    batch_input,
                    features=batch_features,
                    return_type="index",
                    lazy=True,
                ),
            ).with_columns(pl.lit(y).alias(y_name))

            # 构建仅针对当前批次的查询计划，并在聚合后立即 collect() 物化为极小的表
            batch_query = (
                X_bin_lazy
                .select([y_name] + batch_cols)
                .unpivot(
                    index=[y_name],
                    on=batch_cols,
                    variable_name="feature",
                    value_name="bin_index"
                )
                .with_columns(pl.col("bin_index").cast(pl.Int16))
                .group_by(["feature", "bin_index"])
                .agg([
                    pl.len().alias("count"),
                    pl.col(y_name).sum().alias("bad")
                ])
                .with_columns(pl.col("feature").str.replace(r"_bin$", ""))
            )
            batch_stats = collect_streaming(batch_query)
            agg_results.append(batch_stats)

        if not agg_results:
            return self._format_output(pl.DataFrame())

        stats_df = pl.concat(agg_results)
        del agg_results

        # 基础计算
        stats_df = (
            stats_df
            .with_columns(pl.col("count").alias("observed_count"))
            .with_columns(binary_distribution_exprs(["feature"]))
            .with_columns(
                binary_metric_exprs(
                    actual_dist_col="count_dist",
                    lift_col="Lift",
                    iv_bin_col="bin_iv",
                )
            )
        )
        if effective_ordered_metric_sort_by == "woe":
            stats_df = (
                stats_df
                .with_columns(pl.col("woe").fill_null(-999.0).alias("_woe_sort_key"))
                .sort(["feature", "_woe_sort_key", "bin_index"])
                .with_columns(
                    ordered_binary_metric_exprs(
                        ["feature"],
                        ks_bin_col="bin_ks",
                        auc_bin_col="bin_auc_contrib",
                    )
                )
                .drop("_woe_sort_key")
            )
        else:
            ordered_metrics = (
                stats_df
                .filter(pl.col("bin_index") >= 0)
                .with_columns(binary_distribution_exprs(["feature"]))
                .with_columns(
                    binary_metric_exprs(
                        actual_dist_col="count_dist",
                        lift_col="Lift",
                        iv_bin_col="bin_iv",
                    )
                )
                .sort(["feature", "bin_index"])
                .with_columns(
                    ordered_binary_metric_exprs(
                        ["feature"],
                        ks_bin_col="bin_ks",
                        auc_bin_col="bin_auc_contrib",
                    )
                )
                .select(["feature", "bin_index", "bin_ks", "bin_auc_contrib"])
            )
            stats_df = stats_df.join(ordered_metrics, on=["feature", "bin_index"], how="left")

        stats_df = (
            stats_df
            .with_columns(
                [
                    pl.col("bin_iv").sum().over("feature").alias("IV"),
                    pl.col("bin_ks").max().over("feature").alias("KS"),
                    pl.col("bin_auc_contrib").sum().over("feature").alias("AUC"),
                ]
            )
            .with_columns(
                normalized_auc_expr(auc_col="AUC", output_col="AUC")
            )
            .drop(["bin_auc_contrib"])
        )

        if update_woe:
            woe_data = stats_df.select(["feature", "bin_index", "woe"]).to_dict(
                as_series=False,
            )
            temp_woe_map: defaultdict[str, dict[int, float]] = defaultdict(dict)
            for feature, bin_index, woe in zip(
                woe_data["feature"],
                woe_data["bin_index"],
                woe_data["woe"],
            ):
                valid_bin_index = bin_index is not None and not (
                    isinstance(bin_index, float) and np.isnan(bin_index)
                )
                if valid_bin_index and woe is not None:
                    temp_woe_map[feature][int(bin_index)] = float(woe)
            self.bin_woes_.update(temp_woe_map)

        mapping_rows: list[dict[str, Any]] = []
        for col, map_dict in self.bin_mappings_.items():
            for idx, label in map_dict.items():
                mapping_rows.append(
                    {
                        "feature": col,
                        "bin_index": idx,
                        "bin_label": label,
                    },
                )

        if not mapping_rows:
            return self._format_output(stats_df)

        mapping_df = pl.DataFrame(
            mapping_rows,
            schema={
                "feature": pl.Utf8,
                "bin_index": pl.Int16,
                "bin_label": pl.Utf8,
            },
        )
        final_df = (
            stats_df
            .join(mapping_df, on=["feature", "bin_index"], how="left")
            .with_columns((pl.col("bin_index") < 0).alias("_is_special"))
            .sort(["feature", "_is_special", "bin_index"])
            .drop("_is_special")
            .select(
                [
                    pl.col("feature"),
                    *([pl.col("bin_index")] if include_bin_index else []),
                    pl.col("bin_label").fill_null(pl.col("bin_index").cast(pl.Utf8)),
                    pl.all().exclude(["feature", "bin_index", "bin_label"]),
                ]
            )
        )

        trend_df = self._build_trend_shape_frame(
            stats_df.lazy()
            .filter(pl.col("bin_index") >= 0)
            .sort(["feature", "bin_index"])
            .group_by("feature")
            .agg(pl.col("woe"))
            .collect(),
            trend_col_name="trend_shape",
        )
        final_df = (
            final_df
            .join(trend_df, on="feature", how="left")
            .with_columns(pl.col("trend_shape").fill_null("undefined"))
        )

        base_cols = ["feature"]
        if include_bin_index:
            base_cols.append("bin_index")
        base_cols.extend(["bin_label", "trend_shape"])
        other_cols = [col for col in final_df.columns if col not in base_cols]
        out_df = final_df.select(base_cols + other_cols)
        return self._format_output(out_df)


    def update_bins(
        self,
        bin_rules: Dict[str, Union[List[Union[int, float]], List[List[Any]]]],
        X: Union[pl.DataFrame, pd.DataFrame] | None = None,
        y: Any | None = None,
        *,
        on_unknown: Literal["error", "warn", "ignore"] = "error",
    ) -> pl.DataFrame | pd.DataFrame:
        """
        批量更新分箱规则并即时重算相关统计量。

        用传入的切点或类别分组替换已有规则，再计算已更新特征的 WOE 和分箱统计量。
        更新后的规则会作为用户配置保存，后续拟合时仍有同类型特征则继续应用。

        Parameters
        ----------
        bin_rules : Dict[str, Union[List[Union[int, float]], List[List[Any]]]]
            待修改的特征分箱规则字典。
            - 数值型特征：传入内部切点列表，如 ``{'age': [25, 30, 45]}``，自动补齐 ``-inf`` 和 ``inf``。
            - 类别型特征：传入二维分组列表，如 ``{'city': [['北京', '上海'], ['广州', '深圳'], ['其他']]}``。
        X : Union[pl.DataFrame, pd.DataFrame] | None
            用于重新计算 WOE 的数据。为 ``None`` 时使用 ``fit`` 缓存的 ``_cache_X``。
        y : Any | None
            二分类目标标签，支持 0/1、Boolean 及既有字符串表示；null/NaN 和空字符串
            不参与统计。此接口不将 -1 作为未表现哨兵。若为 None，尝试使用拟合缓存。
        on_unknown : Literal['error', 'warn', 'ignore']
            ``bin_rules`` 包含未知或无规则特征时的处理策略，默认报错。
            ``warn`` 记录警告并跳过，``ignore`` 直接跳过；至少需要更新一个已知特征。

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            返回被修改特征的最新分箱统计分布表。

        Raises
        ------
        ValueError
            缺少重算所需的 ``X``/``y``、标签非法或长度不匹配、没有观测标签、
            ``on_unknown`` 非法或没有已知特征被更新。标签校验失败不会修改规则和 WOE。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> y = pl.Series("target", [0, 0, 1, 1])
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, y, features=["age"])
        >>> updated = binner.update_bins({"age": [35]})
        >>> "feature" in updated.columns
        True
        """
        self._check_is_fitted()
        if on_unknown not in {"error", "warn", "ignore"}:
            raise ValueError("on_unknown must be one of {'error', 'warn', 'ignore'}.")

        known_features = set(self._usable_rule_features())
        unknown_features = [feature for feature in bin_rules if feature not in known_features]
        if unknown_features and on_unknown == "error":
            raise ValueError(f"Cannot update unknown binner features: {unknown_features}.")
        if unknown_features and on_unknown == "warn":
            logger.warning("Binner update skipped unknown features: %s", unknown_features)

        # 提取计算上下文
        calc_X = self._ensure_polars_dataframe(X) if X is not None else self._cache_X
        calc_y = self._ensure_polars_series(y) if y is not None else self._cache_y

        if calc_X is None or calc_y is None:
            raise ValueError(
                "Missing data for WOE recalculation. "
                "Either provide X and y explicitly, or ensure the binner cache is not cleared."
            )

        updated_features = []

        # 在修改规则前校验标签；错误评估不能使原有规则和 WOE 半更新。
        target_name = calc_y.name or "target"
        calc_y = normalize_binary_target_column(
            calc_y.rename(target_name).to_frame(), target_name,
        ).get_column(target_name)
        if len(calc_y) != calc_X.height:
            raise ValueError(f"Target 'y' length mismatch: X({calc_X.height}) vs y({len(calc_y)}).")
        if calc_y.drop_nulls().len() == 0:
            raise ValueError("Target 'y' must contain at least one observed value.")

        # 更新规则，随后重建标签映射和 WOE。
        for feature, splits in bin_rules.items():
            if feature not in known_features:
                continue

            # 智能推断类型：如果列表里的元素还是列表，说明是类别型分组；否则是数值型切点
            is_categorical = len(splits) > 0 and isinstance(splits[0], list)

            if not is_categorical:
                # 数值型特征：补齐边界并去重排序
                numeric_splits = cast(List[Union[int, float]], splits)
                clean_splits = sorted(set(float(value) for value in numeric_splits))
                new_cuts = [float('-inf'), *clean_splits, float('inf')]
                self.bin_cuts_[feature] = new_cuts
                self.cat_cuts_.pop(feature, None)
            else:
                # 类别型特征
                if not hasattr(self, "cat_cuts_"):
                    self.cat_cuts_ = {}
                self.cat_cuts_[feature] = cast(List[List[Any]], splits)
                self.bin_cuts_.pop(feature, None)

            # 清理旧的映射与 WOE 缓存
            if feature in self.bin_mappings_:
                del self.bin_mappings_[feature]
            if feature in self.bin_woes_:
                del self.bin_woes_[feature]

            updated_features.append(feature)

        if not updated_features:
            raise ValueError("No known binner features were updated.")

        # 只评估已更新的特征，避免重算其他特征的统计量。
        stats_df = self.profile_bin_performance(
            X=calc_X.select(updated_features),
            y=calc_y,
            update_woe=True,
            features=updated_features,
        )

        for feature in updated_features:
            self._user_bin_rules[feature] = deepcopy(
                self.cat_cuts_[feature]
                if feature in self.cat_cuts_ else self.bin_cuts_[feature]
            )

        return stats_df

    def prune(self, keep_features: List[str]) -> MarsBinnerBase:
        """
        裁剪分箱器内部状态，仅保留指定特征。

        Parameters
        ----------
        keep_features : List[str]
            需要保留状态的特征列表。

        Returns
        -------
        MarsBinnerBase
            裁剪完成后的当前实例。

        Raises
        ------
        ValueError
            ``keep_features`` 包含未知或无可用规则的特征时抛出。

        Notes
        -----
        该方法会同步裁剪切点、类别分组、用户明确更新的规则、标签映射、WOE 映射以及 ``feature_names_in_``，
        常用于特征筛选完成后缩小序列化模型体积。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50], "income": [5, 6, 7, 8]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, features=["age", "income"])
        >>> binner.prune(["age"]).feature_names_in_
        ['age']
        """
        self._check_is_fitted()
        if not keep_features:
            raise ValueError("`keep_features` must contain at least one fitted feature.")
        known_features = set(self._usable_rule_features())
        unknown_features = [feature for feature in keep_features if feature not in known_features]
        if unknown_features:
            raise ValueError(f"Cannot prune to unknown binner features: {unknown_features}.")
        keep_set = set(keep_features)

        # 过滤字典
        self.bin_cuts_ = {k: v for k, v in self.bin_cuts_.items() if k in keep_set}
        if hasattr(self, "cat_cuts_"):
            self.cat_cuts_ = {k: v for k, v in self.cat_cuts_.items() if k in keep_set}

        self.bin_mappings_ = {k: v for k, v in self.bin_mappings_.items() if k in keep_set}
        self.bin_woes_ = {k: v for k, v in self.bin_woes_.items() if k in keep_set}
        self._user_bin_rules = {
            feature: splits for feature, splits in self._user_bin_rules.items() if feature in keep_set
        }

        # 更新输入特征名单
        self.feature_names_in_ = [f for f in self.feature_names_in_ if f in keep_set]
        self.features = [feature for feature in self.features if feature in keep_set]
        self.cat_features = [feature for feature in self.cat_features if feature in keep_set]
        self.fit_failures_ = {
            feature: reason
            for feature, reason in self.fit_failures_.items()
            if feature in keep_set
        }
        self._fit_report_rows = [
            row for row in self._fit_report_rows if row.get("feature") in keep_set
        ]

        return self

    def generate_sql(
        self,
        features: Union[str, List[str]] | None = None,
        table_prefix: str = "t",
        return_type: Literal["woe", "index", "label"] = "woe",
        map_missing: bool = True,
        map_special: bool = True
    ) -> str:
        """
        将分箱规则导出为 SQL ``CASE WHEN`` 片段。

        Parameters
        ----------
        features : Union[str, List[str]] | None
            特征名称或特征列表。若为 None，则自动导出所有已拟合的特征。
        table_prefix : str
            表别名前缀。例如 "t" 会生成 "t.age"。若为空则直接使用特征名。
        return_type : Literal['woe', 'index', 'label']
            生成 SQL 的目标值类型：
            - 'woe': 输出 WOE 浮点数 (适合 LR 逻辑回归模型部署)
            - 'index': 输出分箱序号 (适合 XGBoost/LightGBM 树模型部署)
            - 'label': 输出分箱的中文/字符标签 (适合 BI 看板、数据分析或规则引擎)
        map_missing : bool
            是否把 SQL NULL 映射到 Missing 箱对应的值，默认为 ``True``；
            设为 ``False`` 时输出 NULL。
        map_special : bool
            是否把特殊值映射到 Special 箱对应的值，默认为 ``True``；
            设为 ``False`` 时保留特殊值原值。

        Returns
        -------
        str
            逗号分隔的 SQL 表达式，可嵌入 ``SELECT`` 子句。

        Raises
        ------
        ValueError
            特征未知、规则或 WOE 映射不完整、SQL 值不可安全表示时抛出。

        Examples
        --------
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, features=["age"])
        >>> sql = binner.generate_sql(features="age", return_type="index")
        >>> "age_index" in sql
        True
        """
        self._check_is_fitted()
        if return_type not in {"woe", "index", "label"}:
            raise ValueError("return_type must be one of {'woe', 'index', 'label'}.")

        # 1. 入参类型归一化
        if features is None:
            target_features = self._usable_rule_features()
        elif isinstance(features, str):
            target_features = [features]
        else:
            target_features = features

        if not target_features:
            raise ValueError("No fitted binner features are available for SQL generation.")
        unknown_features = [
            feature for feature in target_features if feature not in self._usable_rule_features()
        ]
        if unknown_features:
            raise ValueError(f"Cannot generate SQL for unknown binner features: {unknown_features}.")

        def _sql_literal(value: Any) -> str:
            """将类别值格式化为不丢失引号语义的 SQL literal。"""
            if value is None:
                return "NULL"
            if isinstance(value, bool):
                return "TRUE" if value else "FALSE"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_value = float(value)
                if not np.isfinite(numeric_value):
                    raise ValueError(f"Non-finite SQL literal is unsupported: {value!r}.")
                return str(value)
            escaped = str(value).replace("'", "''")
            return f"'{escaped}'"

        # 2. 定义内部核心处理函数：生成单列的 CASE WHEN
        def _generate_single_sql(feature: str) -> str:
            """为单个特征生成 SQL CASE WHEN 片段。"""
            if feature not in self.bin_mappings_:
                raise ValueError(f"Feature {feature!r} has no fitted label mapping.")
            if return_type == "woe" and feature not in self.bin_woes_:
                raise ValueError(
                    f"Feature {feature!r} has no WOE mapping; fit with a binary target first."
                )

            col_name = f"{table_prefix}.{feature}" if table_prefix else feature
            lines = ["CASE"]

            mappings = self.bin_mappings_.get(feature, {})
            woes = self.bin_woes_.get(feature, {})

            def _get_output_val(idx: int) -> str:
                """内部函数：根据输出契约动态格式化 THEN 后置结果"""
                if return_type == "woe":
                    if idx not in woes:
                        raise ValueError(
                            f"Feature {feature!r} has no WOE value for bin index {idx}."
                        )
                    return f"{woes[idx]:.4f}"
                if return_type == "index":
                    return str(idx)
                if idx not in mappings:
                    raise ValueError(
                        f"Feature {feature!r} has no label for bin index {idx}."
                    )
                return _sql_literal(mappings[idx])

            # 处理缺失值
            if map_missing:
                lines.append(f"  WHEN {col_name} IS NULL THEN {_get_output_val(self.IDX_MISSING)}")
            else:
                lines.append(f"  WHEN {col_name} IS NULL THEN NULL")

            # 处理特殊值 (逆序保证优先级)
            special_idx = [k for k in mappings.keys() if k <= self.IDX_SPECIAL_START]
            for idx in sorted(special_idx, reverse=True):
                label = mappings[idx]
                val_str = label.replace("Special_", "")
                if feature in self.cat_cuts_:
                    sql_val = _sql_literal(val_str)
                else:
                    try:
                        numeric_special = float(val_str)
                    except ValueError:
                        sql_val = _sql_literal(val_str)
                    else:
                        sql_val = _sql_literal(numeric_special)

                if map_special:
                    lines.append(f"  WHEN {col_name} = {sql_val} THEN {_get_output_val(idx)}")
                else:
                    lines.append(f"  WHEN {col_name} = {sql_val} THEN {col_name}")

            # 处理数值型特征切点逻辑
            if hasattr(self, "bin_cuts_") and feature in self.bin_cuts_:
                cuts = self.bin_cuts_[feature]
                for i in range(len(cuts) - 1):
                    upper_bound = cuts[i+1]
                    if upper_bound != float('inf'):
                        lines.append(f"  WHEN {col_name} < {upper_bound} THEN {_get_output_val(i)}")
                    else:
                        lines.append(f"  ELSE {_get_output_val(i)}")

            # 处理类别型特征逻辑
            elif hasattr(self, "cat_cuts_") and feature in self.cat_cuts_:
                groups = self.cat_cuts_[feature]
                for i, group in enumerate(groups):
                    if "__Mars_Other_Pre__" in group:
                        continue
                    in_clause = ", ".join(_sql_literal(value) for value in group)
                    lines.append(f"  WHEN {col_name} IN ({in_clause}) THEN {_get_output_val(i)}")

                lines.append(f"  ELSE {_get_output_val(self.IDX_OTHER)}")

            # 兜底逻辑
            if "ELSE" not in "\n".join(lines):
                lines.append(f"  ELSE {_get_output_val(self.IDX_OTHER)}")

            lines.append(f"END AS {feature}_{return_type}")
            return "\n".join(lines)

        # 3. 遍历拼接多个特征的 SQL 代码块
        sql_blocks = [_generate_single_sql(feat) for feat in target_features]

        # 使用逗号加两个换行符拼接，使其满足 SELECT 多列的语法格式
        return ",\n\n".join(sql_blocks)

    @staticmethod
    def _format_cut_point(val: float) -> str:
        """将切点格式化为适合图表和 SQL 展示的字符串。"""
        if val == float('inf'):
            return 'inf'
        if val == float('-inf'):
            return '-inf'
        if val == 0:
            return '0'

        abs_val = abs(val)

        # 超大数字 (>=10000)，使用千分位逗号，如 1,000,000
        if abs_val >= 10000:
            if val == int(val):
                return f"{int(val):,}"
            else:
                # 保留两位小数并剔除末尾多余的 0
                return f"{val:,.2f}".rstrip('0').rstrip('.')

        # 极小数字 (<0.001)，强制使用定点数避免科学计数法, 保留 6 位小数，并动态剔除尾部无效的 0
        elif abs_val < 0.001:
            return f"{val:.6f}".rstrip('0').rstrip('.')

        # 常规数字，最多保留 4 位小数并去掉多余的 0
        else:
            if val == int(val):
                return str(int(val))
            else:
                return f"{val:.4f}".rstrip('0').rstrip('.')
