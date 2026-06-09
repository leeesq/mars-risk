"""MARS 轻量级启发式最优分箱器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, cast

import numpy as np
import pandas as pd
import polars as pl

from mars.utils.logger import logger

from .binner import MarsBinnerBase, MarsNativeBinner

PrebinningMethod = Literal["quantile", "uniform", "cart"]
TrendShape = Literal["ascending", "descending", "peak", "valley", "auto"]
_Direction = Literal["ascending", "descending"]


@dataclass
class _LiteBin:
    """保存一个连续预分箱或合并后分箱的统计量与边界。"""

    start_index: int
    end_index: int
    left_cut: float
    right_cut: float
    count: float
    bad: float

    @property
    def good(self) -> float:
        """返回当前箱内的好样本数。"""
        return max(self.count - self.bad, 0.0)

    @property
    def bad_rate(self) -> float:
        """返回带轻微平滑的坏样本率，降低小样本极端值干扰。"""
        return (self.bad + 0.5) / (self.count + 1.0) if self.count > 0 else 0.5

    def merge(self, other: _LiteBin) -> _LiteBin:
        """合并两个相邻箱并重新计算聚合统计量。"""
        return _LiteBin(
            start_index=self.start_index,
            end_index=other.end_index,
            left_cut=self.left_cut,
            right_cut=other.right_cut,
            count=self.count + other.count,
            bad=self.bad + other.bad,
        )


@dataclass
class _LiteCandidate:
    """记录单个趋势候选的最终分箱与评分。"""

    bins: list[_LiteBin]
    score: float
    shape: str


class MarsLiteOptBinner(MarsBinnerBase):
    """
    轻量级启发式最优分箱器。

    该分箱器面向宽表风控分析场景，先复用 ``MarsNativeBinner`` 生成细粒度预分箱，
    再在预分箱统计表上执行趋势约束合并。它不依赖数学规划求解器，适合作为
    ``MarsOptimalBinner`` 的高速轻量替代方案。

    Attributes
    ----------
    bin_cuts_ : dict of str to list of float
        数值特征最终切点，形态为 ``[-inf, ..., inf]``。
    cat_cuts_ : dict of str to list of list
        类别特征 Top-K 分组规则。
    fit_failures_ : dict of str to str
        拟合失败并回退的特征及原因。
    fitted_trends_ : dict of str to str
        每个数值特征最终采用的趋势形态。
    candidate_scores_ : dict of str to dict of str to float
        每个数值特征在各候选趋势下的惩罚后评分。

    Examples
    --------
    >>> import polars as pl
    >>> X = pl.DataFrame({"score": [0.1, 0.2, 0.8, 0.9]})
    >>> y = pl.Series("target", [0, 0, 1, 1])
    >>> binner = MarsLiteOptBinner(n_bins=2, n_prebins=4)
    >>> binner.fit(X, y).transform(X).columns
    ['score_bin']
    """

    def __init__(
        self,
        *,
        n_bins: int = 10,
        min_bin_size: float = 0.05,
        monotonic_trend: TrendShape = "auto",
        prebinning_method: PrebinningMethod = "quantile",
        n_prebins: int = 50,
        special_values: List[Any] | None = None,
        missing_values: List[Any] | None = None,
        join_threshold: int = 100,
        n_jobs: int = -1,
    ) -> None:
        """
        初始化轻量级最优分箱器。

        Parameters
        ----------
        n_bins : int
            最终正常分箱数量上限，不含缺失值箱和特殊值箱。
        min_bin_size : float
            最终正常箱的最小全量样本占比。
        monotonic_trend : TrendShape
            趋势约束。``"auto"`` 会在递增、递减、峰形和谷形中择优。
        prebinning_method : PrebinningMethod
            预分箱策略，可选 ``"quantile"``、``"uniform"`` 或 ``"cart"``。
        n_prebins : int
            预分箱数量上限。
        special_values : List[Any] | None
            需要独立隔离的业务特殊值。
        missing_values : List[Any] | None
            需要额外识别为缺失的取值。
        join_threshold : int
            高基数类别转换时切换到 Join 映射的阈值。
        n_jobs : int
            预分箱阶段可使用的并行核心数。

        Raises
        ------
        ValueError
            当分箱数量、趋势类型或预分箱策略配置非法时抛出。
        """
        if n_bins < 1:
            raise ValueError("n_bins must be at least 1.")
        if n_prebins < 2:
            raise ValueError("n_prebins must be at least 2.")
        if not 0 <= min_bin_size <= 1:
            raise ValueError("min_bin_size must be in [0, 1].")
        if monotonic_trend not in {"ascending", "descending", "peak", "valley", "auto"}:
            raise ValueError(
                "monotonic_trend must be one of "
                "{'ascending', 'descending', 'peak', 'valley', 'auto'}."
            )
        if prebinning_method not in {"quantile", "uniform", "cart"}:
            raise ValueError("prebinning_method must be one of {'quantile', 'uniform', 'cart'}.")

        super().__init__(
            n_bins=n_bins,
            special_values=special_values,
            missing_values=missing_values,
            join_threshold=join_threshold,
            n_jobs=n_jobs,
        )
        self.min_bin_size = min_bin_size
        self.monotonic_trend = monotonic_trend
        self.prebinning_method = prebinning_method
        self.n_prebins = n_prebins
        self.fitted_trends_: dict[str, str] = {}
        self.candidate_scores_: dict[str, dict[str, float]] = {}

    def fit(  # type: ignore[override]
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: pl.Series | pd.Series | np.ndarray | list[Any],
        *,
        features: List[str] | None = None,
        cat_features: List[str] | None = None,
    ) -> MarsLiteOptBinner:
        """
        拟合轻量级最优分箱规则。

        Parameters
        ----------
        X : pl.DataFrame | pd.DataFrame
            输入特征矩阵。
        y : pl.Series | pd.Series | np.ndarray | list[Any]
            二分类目标变量，必须为 0/1 或布尔值，且不允许为空。
        features : List[str] | None
            本次拟合的特征列；不传时使用全部候选列。
        cat_features : List[str] | None
            明确指定为类别型的特征列。

        Returns
        -------
        MarsLiteOptBinner
            拟合完成后的当前实例。

        Raises
        ------
        ValueError
            当 ``y`` 缺失、标签非法或输入列配置不满足拟合要求时抛出。
        """
        if y is None:
            raise ValueError("MarsLiteOptBinner.fit requires y.")

        self.features = list(features or [])
        self.cat_features = list(cat_features or [])
        super().fit(X, y)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        将轻量级分箱器状态序列化为字典。

        Returns
        -------
        Dict[str, Any]
            包含构造参数与拟合后状态的可序列化字典。
        """
        data = super().to_dict()
        data["params"].update(
            {
                "min_bin_size": self.min_bin_size,
                "monotonic_trend": self.monotonic_trend,
                "prebinning_method": self.prebinning_method,
                "n_prebins": self.n_prebins,
            }
        )
        data["state"].update(
            {
                "fitted_trends_": self.fitted_trends_,
                "candidate_scores_": self.candidate_scores_,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MarsLiteOptBinner:
        """
        从字典恢复轻量级分箱器实例。

        Parameters
        ----------
        data : Dict[str, Any]
            由 ``to_dict`` 生成的状态字典。

        Returns
        -------
        MarsLiteOptBinner
            恢复后的已拟合轻量级分箱器。

        Examples
        --------
        >>> binner = MarsLiteOptBinner.from_dict(
        ...     {
        ...         "params": {"n_bins": 2},
        ...         "state": {"bin_cuts_": {"x": [-float("inf"), float("inf")]}},
        ...     }
        ... )
        >>> binner.fitted_trends_
        {}
        """
        instance = cast("MarsLiteOptBinner", super().from_dict(data))
        state: Dict[str, Any] = data.get("state", {})
        instance.fitted_trends_ = {
            str(feature): str(trend)
            for feature, trend in state.get("fitted_trends_", {}).items()
        }
        instance.candidate_scores_ = {
            str(feature): {
                str(shape): float(score)
                for shape, score in scores.items()
            }
            for feature, scores in state.get("candidate_scores_", {}).items()
        }
        return instance

    def _fit_impl(self, X: pl.DataFrame, y: pl.Series | None = None) -> None:
        """识别列类型并分别拟合数值轻量最优分箱与类别 Top-K 分箱。"""
        if y is None:
            raise ValueError("MarsLiteOptBinner requires target 'y'.")

        y_series = self._validate_binary_y(y, expected_len=X.height)
        self._cache_X = X
        self._cache_y = y_series
        self.fit_failures_ = {}
        self.fitted_trends_ = {}
        self.candidate_scores_ = {}

        y_name = getattr(y_series, "name", None)
        all_target_cols = self.features if self.features else [c for c in X.columns if c != y_name]
        cat_set = set(self.cat_features)

        num_cols: list[str] = []
        cat_cols: list[str] = []
        null_cols: list[str] = []

        for col in all_target_cols:
            if col not in X.columns:
                continue
            if X[col].dtype == pl.Null or X[col].null_count() == X.height:
                null_cols.append(col)
            elif col in cat_set or X[col].dtype in {pl.Utf8, pl.Categorical, pl.Boolean}:
                cat_cols.append(col)
            elif self._is_numeric(X[col]):
                num_cols.append(col)

        for col in null_cols:
            self.bin_cuts_[col] = []

        if num_cols:
            self._fit_numerical_impl(X, y_series, num_cols)
        if cat_cols:
            self._fit_categorical_impl(X, cat_cols)

        if self.fit_failures_:
            logger.warning(
                "MarsLiteOptBinner: %s features fell back during fitting. Sample: %s",
                len(self.fit_failures_),
                list(self.fit_failures_.items())[:3],
            )

    def _validate_binary_y(self, y: pl.Series, expected_len: int) -> pl.Series:
        """校验监督分箱标签长度和取值，返回可聚合的 Int8 标签。"""
        if len(y) != expected_len:
            raise ValueError(f"Target 'y' length mismatch: X({expected_len}) vs y({len(y)}).")

        if y.null_count() > 0:
            raise ValueError("MarsLiteOptBinner requires fully observed binary y without null values.")

        y_checked = y
        if y_checked.dtype in {pl.Float32, pl.Float64}:
            if y_checked.is_nan().any():
                raise ValueError("MarsLiteOptBinner requires fully observed binary y without NaN values.")
            y_checked = y_checked.cast(pl.Float64)

        invalid_values = (
            y_checked
            .filter(~y_checked.is_in([0, 1, False, True]))
            .unique()
            .head(5)
            .to_list()
        )
        if invalid_values:
            raise ValueError(
                f"MarsLiteOptBinner y contains invalid values {invalid_values}. "
                "Please clean it to 0/1/True/False before fitting."
            )
        return y_checked.cast(pl.Int8)

    def _fit_categorical_impl(self, X: pl.DataFrame, cat_cols: list[str]) -> None:
        """复用原生分箱器的类别 Top-K 逻辑，避免重复维护类别映射规则。"""
        native = MarsNativeBinner(
            method="quantile",
            n_bins=self.n_bins,
            special_values=self.special_values,
            missing_values=self.missing_values,
            n_jobs=self.n_jobs,
        )
        native.fit(X, features=cat_cols, cat_features=cat_cols)
        self.cat_cuts_.update(native.cat_cuts_)
        self.bin_cuts_.update({k: v for k, v in native.bin_cuts_.items() if k in cat_cols})
        self.fit_failures_.update(native.fit_failures_)

    def _fit_numerical_impl(self, X: pl.DataFrame, y: pl.Series, num_cols: list[str]) -> None:
        """执行数值特征预分箱、趋势合并和最终切点回写。"""
        valid_cols = self._filter_valid_numeric_columns(X, num_cols)
        if not valid_cols:
            return

        pre_binner = self._fit_pre_binner(X, y, valid_cols)
        profile_raw = pre_binner.profile_bin_performance(
            X.select(valid_cols),
            y,
            update_woe=False,
            include_bin_index=True,
        )
        profile_df = pl.from_pandas(profile_raw) if isinstance(profile_raw, pd.DataFrame) else profile_raw

        for col in valid_cols:
            pre_cuts = pre_binner.bin_cuts_.get(col, [float("-inf"), float("inf")])
            if len(pre_cuts) <= 2:
                self.bin_cuts_[col] = pre_cuts
                continue

            try:
                pre_bins = self._build_prebins_from_profile(profile_df, col, pre_cuts)
                if len(pre_bins) <= 1:
                    self.bin_cuts_[col] = [float("-inf"), float("inf")]
                    continue
                candidate, candidate_scores = self._select_best_candidate(
                    pre_bins,
                    total_count=float(X.height),
                )
                self.bin_cuts_[col] = self._cuts_from_bins(candidate.bins)
                self.fitted_trends_[col] = candidate.shape
                self.candidate_scores_[col] = candidate_scores
            except Exception as exc:
                self.bin_cuts_[col] = pre_cuts
                self.fit_failures_[col] = f"{type(exc).__name__}: {exc}"

    def _filter_valid_numeric_columns(self, X: pl.DataFrame, num_cols: list[str]) -> list[str]:
        """剔除全空和常量数值列，对不可切分列直接写入兜底切点。"""
        valid_cols: list[str] = []
        raw_exclude = self.special_values + self.missing_values

        for col in num_cols:
            series = X.get_column(col)
            safe_exclude = self._get_safe_values(series.dtype, raw_exclude)

            valid_mask = series.is_not_null()
            if series.dtype in {pl.Float32, pl.Float64}:
                valid_mask &= ~series.is_nan()
            if safe_exclude:
                valid_mask &= ~series.is_in(safe_exclude)

            clean_series = series.filter(valid_mask)
            if clean_series.len() == 0 or clean_series.n_unique() <= 1:
                self.bin_cuts_[col] = [float("-inf"), float("inf")]
                continue
            valid_cols.append(col)

        return valid_cols

    def _fit_pre_binner(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        valid_cols: list[str],
    ) -> MarsNativeBinner:
        """拟合轻量最优分箱的第一阶段原生预分箱器。"""
        prebin_min_size = min(self.min_bin_size, 1.0 / max(self.n_prebins, 1))
        pre_binner = MarsNativeBinner(
            method=self.prebinning_method,
            n_bins=self.n_prebins,
            special_values=self.special_values,
            missing_values=self.missing_values,
            min_bin_size=prebin_min_size,
            remove_empty_bins=True,
            n_jobs=self.n_jobs,
        )
        pre_binner.fit(X, y, features=valid_cols)
        return pre_binner

    def _build_prebins_from_profile(
        self,
        profile_df: pl.DataFrame,
        feature: str,
        pre_cuts: list[float],
    ) -> list[_LiteBin]:
        """从预分箱表现表中恢复有序细箱统计，空箱不参与后续趋势拟合。"""
        feature_stats = (
            profile_df
            .filter((pl.col("feature") == feature) & (pl.col("bin_index") >= 0))
            .select(["bin_index", "count", "bad"])
        )
        stat_rows = {
            int(row["bin_index"]): (float(row["count"]), float(row["bad"] or 0.0))
            for row in feature_stats.iter_rows(named=True)
        }

        pre_bins: list[_LiteBin] = []
        for bin_index in range(len(pre_cuts) - 1):
            count, bad = stat_rows.get(bin_index, (0.0, 0.0))
            if count <= 0:
                continue
            pre_bins.append(
                _LiteBin(
                    start_index=bin_index,
                    end_index=bin_index,
                    left_cut=float(pre_cuts[bin_index]),
                    right_cut=float(pre_cuts[bin_index + 1]),
                    count=count,
                    bad=bad,
                )
            )
        return pre_bins

    def _select_best_candidate(
        self,
        pre_bins: list[_LiteBin],
        total_count: float,
    ) -> tuple[_LiteCandidate, dict[str, float]]:
        """基于配置趋势或 auto 策略选择最终候选分箱。"""
        if self.monotonic_trend == "auto":
            shapes: list[str] = ["ascending", "descending", "peak", "valley"]
        else:
            shapes = [self.monotonic_trend]

        candidates: list[_LiteCandidate] = []
        for shape in shapes:
            candidates.append(self._fit_shape_candidate(pre_bins, shape, total_count))

        shape_rank = {"ascending": 0, "descending": 0, "peak": 1, "valley": 1}
        best_candidate = min(
            candidates,
            key=lambda item: (item.score, shape_rank.get(item.shape, 9), len(item.bins)),
        )
        candidate_scores = {candidate.shape: candidate.score for candidate in candidates}
        return best_candidate, candidate_scores

    def _fit_shape_candidate(
        self,
        pre_bins: list[_LiteBin],
        shape: str,
        total_count: float,
    ) -> _LiteCandidate:
        """对指定趋势形态枚举符号序列并返回最优候选。"""
        candidates: list[_LiteCandidate] = []
        for signs in self._sign_patterns(shape, len(pre_bins)):
            blocks, active_signs = self._enforce_signs(pre_bins, signs)
            blocks, active_signs = self._repair_min_bin_size(blocks, active_signs, total_count)
            blocks, _ = self._compress_to_n_bins(blocks, active_signs)
            score = self._candidate_score(blocks, shape, total_count)
            candidates.append(_LiteCandidate(bins=blocks, score=score, shape=shape))

        if not candidates:
            score = self._candidate_score(pre_bins, shape, total_count)
            return _LiteCandidate(bins=list(pre_bins), score=score, shape=shape)

        return min(candidates, key=lambda item: (item.score, len(item.bins)))

    def _sign_patterns(self, shape: str, n_bins: int) -> list[list[_Direction]]:
        """为不同趋势形态生成相邻箱之间的方向约束序列。"""
        if n_bins <= 1:
            return [[]]
        if shape == "ascending":
            return [["ascending"] * (n_bins - 1)]
        if shape == "descending":
            return [["descending"] * (n_bins - 1)]
        if shape == "peak":
            if n_bins <= 2:
                return [
                    ["ascending"] * (n_bins - 1),
                    ["descending"] * (n_bins - 1),
                ]
            return [
                ["ascending" if idx < pivot else "descending" for idx in range(n_bins - 1)]
                for pivot in range(1, n_bins - 1)
            ]
        if shape == "valley":
            if n_bins <= 2:
                return [
                    ["descending"] * (n_bins - 1),
                    ["ascending"] * (n_bins - 1),
                ]
            return [
                ["descending" if idx < pivot else "ascending" for idx in range(n_bins - 1)]
                for pivot in range(1, n_bins - 1)
            ]
        raise ValueError(f"Unsupported trend shape: {shape}")

    def _enforce_signs(
        self,
        bins: list[_LiteBin],
        signs: list[_Direction],
    ) -> tuple[list[_LiteBin], list[_Direction]]:
        """按方向约束执行相邻违反合并，直到序列满足指定趋势。"""
        blocks = list(bins)
        active_signs = list(signs)

        while len(blocks) > 1:
            violation_index = self._first_violation(blocks, active_signs)
            if violation_index is None:
                break
            blocks, active_signs = self._merge_at(blocks, active_signs, violation_index)

        return blocks, active_signs

    def _first_violation(
        self,
        bins: list[_LiteBin],
        signs: list[_Direction],
    ) -> int | None:
        """查找第一个违反相邻方向约束的位置。"""
        tolerance = 1e-12
        for idx, sign in enumerate(signs):
            left_rate = bins[idx].bad_rate
            right_rate = bins[idx + 1].bad_rate
            if sign == "ascending" and left_rate > right_rate + tolerance:
                return idx
            if sign == "descending" and left_rate + tolerance < right_rate:
                return idx
        return None

    def _repair_min_bin_size(
        self,
        bins: list[_LiteBin],
        signs: list[_Direction],
        total_count: float,
    ) -> tuple[list[_LiteBin], list[_Direction]]:
        """后置修复最小箱占比约束，避免提前破坏预分箱趋势信息。"""
        if self.min_bin_size <= 0 or total_count <= 0:
            return bins, signs

        threshold = self.min_bin_size * total_count
        blocks = list(bins)
        active_signs = list(signs)

        while len(blocks) > 1 and any(block.count < threshold for block in blocks):
            merge_index = self._best_repair_merge_index(blocks, threshold)
            blocks, active_signs = self._merge_at(blocks, active_signs, merge_index)
            blocks, active_signs = self._enforce_signs(blocks, active_signs)

        return blocks, active_signs

    def _best_repair_merge_index(self, bins: list[_LiteBin], threshold: float) -> int:
        """选择能修复小箱且带来最小拟合损失的相邻合并位置。"""
        candidate_indices: set[int] = set()
        for idx, block in enumerate(bins):
            if block.count >= threshold:
                continue
            if idx > 0:
                candidate_indices.add(idx - 1)
            if idx < len(bins) - 1:
                candidate_indices.add(idx)

        if not candidate_indices:
            return 0

        return min(candidate_indices, key=lambda idx: self._merge_loss_delta(bins[idx], bins[idx + 1]))

    def _compress_to_n_bins(
        self,
        bins: list[_LiteBin],
        signs: list[_Direction],
    ) -> tuple[list[_LiteBin], list[_Direction]]:
        """在满足趋势后继续压缩箱数到 n_bins 上限以内。"""
        blocks = list(bins)
        active_signs = list(signs)

        while len(blocks) > self.n_bins and len(blocks) > 1:
            merge_index = min(
                range(len(blocks) - 1),
                key=lambda idx: self._merge_loss_delta(blocks[idx], blocks[idx + 1]),
            )
            blocks, active_signs = self._merge_at(blocks, active_signs, merge_index)
            blocks, active_signs = self._enforce_signs(blocks, active_signs)

        return blocks, active_signs

    def _merge_at(
        self,
        bins: list[_LiteBin],
        signs: list[_Direction],
        index: int,
    ) -> tuple[list[_LiteBin], list[_Direction]]:
        """合并指定位置的相邻箱，并同步删除对应方向约束。"""
        merged = bins[index].merge(bins[index + 1])
        new_bins = bins[:index] + [merged] + bins[index + 2 :]
        new_signs = signs[:index] + signs[index + 1 :]
        return new_bins, new_signs

    def _merge_loss_delta(self, left: _LiteBin, right: _LiteBin) -> float:
        """计算合并两个相邻箱带来的二项偏差增量。"""
        return self._block_deviance(left.merge(right)) - (
            self._block_deviance(left) + self._block_deviance(right)
        )

    def _candidate_score(
        self,
        bins: list[_LiteBin],
        shape: str,
        total_count: float,
    ) -> float:
        """计算候选分箱的惩罚后评分，用于 auto 形态择优。"""
        deviance = sum(self._block_deviance(block) for block in bins)
        log_n = float(np.log(max(total_count, 2.0)))
        complexity_penalty = log_n * len(bins)
        shape_penalty = log_n if shape in {"peak", "valley"} else 0.0
        return deviance + complexity_penalty + shape_penalty

    def _block_deviance(self, block: _LiteBin) -> float:
        """计算单个箱在二项分布口径下的负对数似然偏差。"""
        if block.count <= 0:
            return 0.0
        eps = 1e-12
        p = float(np.clip(block.bad_rate, eps, 1.0 - eps))
        return float(-2.0 * (block.bad * np.log(p) + block.good * np.log(1.0 - p)))

    def _cuts_from_bins(self, bins: list[_LiteBin]) -> list[float]:
        """根据最终连续箱恢复基类可识别的物理切点。"""
        if not bins:
            return [float("-inf"), float("inf")]

        inner_cuts = [block.right_cut for block in bins[:-1]]
        clean_inner_cuts: list[float] = []
        for cut in inner_cuts:
            if np.isfinite(cut) and (not clean_inner_cuts or cut > clean_inner_cuts[-1]):
                clean_inner_cuts.append(float(cut))
        return [float("-inf")] + clean_inner_cuts + [float("inf")]
