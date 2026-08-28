"""MARS 规则候选生成器。"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import polars as pl
from joblib import Parallel, delayed
from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, _tree

from mars.compute import FrameLike, to_polars_frame
from mars.feature import MarsStatsSelector
from mars.rule._dsl import expression_to_polars, parse_expression
from mars.rule.contracts import MarsRule
from mars.utils.imports import require_optional_module

NumericLeafSize = Union[int, float]


@dataclass(frozen=True)
class _ModelFeature:
    """描述模型矩阵列到原始特征和缺失语义的映射。"""

    original_name: str
    kind: Literal["value", "missing"]
    fill_value: float


class MarsRuleGenerator(ABC):
    """规则候选生成器抽象接口。

    子类只负责从训练数据生成不含样本指标的 :class:`MarsRule`。最终方向、筛选和去重由
    :func:`mars.rule.mine_rules` 统一完成。
    """

    @abstractmethod
    def generate(
        self,
        df: FrameLike,
        *,
        target: str,
        features: Sequence[str] | None = None,
    ) -> List[MarsRule]:
        """生成规则候选。

        Parameters
        ----------
        df : FrameLike
            训练样本。
        target : str
            主二分类目标。
        features : Sequence[str] | None
            显式候选特征；不传时由生成器推断数值特征。

        Returns
        -------
        list[MarsRule]
            已规范化并精确去重的候选规则。
        """


class MarsCombinationRuleGenerator(MarsRuleGenerator):
    """使用分位点生成单变量和受控交叉规则。

    Parameters
    ----------
    n_bins : int
        每个特征的目标分位区间数。
    max_cross_features : int
        单条组合规则最多包含的特征数。
    max_candidates : int
        生成器候选上限。
    random_state : int | None
        宽表预筛抽样种子。
    prefilter_single_rules : bool
        是否先按样本内 Lift 排序单规则，再生成交叉候选。
    feature_prefilter_top_k : int
        宽表统计预筛最多保留的特征数。
    feature_prefilter_min_features : int
        触发 :class:`MarsStatsSelector` 的数值特征数。
    feature_prefilter_sample_size : int
        统计预筛最大样本数。

    Raises
    ------
    ValueError
        分箱、交叉、候选或特征预筛预算非法时抛出。
    """

    def __init__(
        self,
        n_bins: int = 5,
        max_cross_features: int = 2,
        max_candidates: int = 100_000,
        random_state: int | None = None,
        prefilter_single_rules: bool = True,
        feature_prefilter_top_k: int = 300,
        feature_prefilter_min_features: int = 500,
        feature_prefilter_sample_size: int = 100_000,
    ) -> None:
        if n_bins < 2 or max_cross_features < 1 or max_candidates < 1:
            raise ValueError("n_bins 至少为 2，交叉特征数和候选预算至少为 1。")
        if feature_prefilter_top_k < 1 or feature_prefilter_min_features < 1:
            raise ValueError("特征预筛阈值必须至少为 1。")
        self.n_bins = n_bins
        self.max_cross_features = max_cross_features
        self.max_candidates = max_candidates
        self.random_state = random_state
        self.prefilter_single_rules = prefilter_single_rules
        self.feature_prefilter_top_k = feature_prefilter_top_k
        self.feature_prefilter_min_features = feature_prefilter_min_features
        self.feature_prefilter_sample_size = feature_prefilter_sample_size
        self.selected_features_: List[str] = []

    def generate(
        self,
        df: FrameLike,
        *,
        target: str,
        features: Sequence[str] | None = None,
    ) -> List[MarsRule]:
        """生成分位点与交叉规则。"""
        frame: pl.DataFrame = to_polars_frame(df)
        numeric_features: List[str] = _resolve_numeric_features(frame, target, features)
        numeric_features = self._prefilter_features(frame, target, numeric_features)
        self.selected_features_ = list(numeric_features)
        cut_map: Dict[str, List[float]] = _quantile_cut_map(
            frame,
            numeric_features,
            self.n_bins,
        )
        singles: List[MarsRule] = []
        for feature in numeric_features:
            cuts: List[float] = cut_map[feature]
            identifier: str = _quote_dsl_identifier(feature)
            for cut in cuts:
                singles.extend(
                    [
                        MarsRule(f"{identifier} <= {cut!r}", source="combination"),
                        MarsRule(f"{identifier} > {cut!r}", source="combination"),
                    ]
                )
        singles = _deduplicate_rules(singles)
        if self.prefilter_single_rules:
            singles = _rank_single_rules(frame, target, singles)
            single_budget: int = max(8, math.isqrt(self.max_candidates))
            singles = singles[:single_budget]
        candidates: List[MarsRule] = list(singles)
        if self.max_cross_features > 1 and len(candidates) < self.max_candidates:
            # 复用 deimos 已验证的候选池口径：二阶生成 AND/OR，高阶仅生成 AND。
            for left, right in combinations(singles, 2):
                if set(left.required_features) == set(right.required_features):
                    continue
                for operator in ("AND", "OR"):
                    expression: str = (
                        f"({left.expression}) {operator} ({right.expression})"
                    )
                    candidates.append(MarsRule(expression, source="combination"))
                    if len(candidates) >= self.max_candidates:
                        return _deduplicate_rules(candidates)[: self.max_candidates]
            for size in range(3, self.max_cross_features + 1):
                for selected in combinations(singles, size):
                    feature_sets = [set(rule.required_features) for rule in selected]
                    if len(set.union(*feature_sets)) != size:
                        continue
                    expression = " AND ".join(
                        f"({rule.expression})" for rule in selected
                    )
                    candidates.append(MarsRule(expression, source="combination"))
                    if len(candidates) >= self.max_candidates:
                        return _deduplicate_rules(candidates)[: self.max_candidates]
        return _deduplicate_rules(candidates)[: self.max_candidates]

    def _prefilter_features(
        self,
        frame: pl.DataFrame,
        target: str,
        features: List[str],
    ) -> List[str]:
        """宽表时直接复用 Mars 统计筛选器。"""
        if len(features) < self.feature_prefilter_min_features:
            return features
        sample: pl.DataFrame = frame.select(features + [target])
        if sample.height > self.feature_prefilter_sample_size:
            sample = sample.sample(
                n=self.feature_prefilter_sample_size,
                shuffle=True,
                seed=self.random_state or 42,
            )
        selector = MarsStatsSelector(
            skip_fine_scan=True,
            rough_iv_thr=0.0,
            rough_lift_thr=999.0,
            psi_thr=None,
            rc_thr=None,
            corr_thr=None,
            batch_size=128,
            n_jobs=-1,
        ).set_output("polars")
        selector.fit(sample, target=target, features=features)
        selected: List[str] = [feature for feature in selector.selected_features_ if feature in features]
        iv_values: Mapping[str, Any] = getattr(selector, "_feature_iv_dict", {})
        ranked: List[str] = sorted(
            selected,
            key=lambda feature: (-float(iv_values.get(feature, 0.0) or 0.0), feature),
        )
        return ranked[: self.feature_prefilter_top_k]


class MarsTreeRuleGenerator(MarsRuleGenerator):
    """从多棵随机浅层决策树提取叶子路径。

    每棵树使用确定性中位数填充值和显式缺失指示器训练；提取路径时将缺失分支还原为
    ``IS MISSING``，避免训练矩阵与部署命中语义漂移。
    """

    def __init__(
        self,
        n_trees: int = 15,
        max_depths: Sequence[int] | None = None,
        feature_fraction: float = 0.8,
        n_jobs: int = -1,
        random_state: int | None = None,
        dt_params: Dict[str, Any] | None = None,
        tuning_backend: Literal["none", "optuna"] = "none",
        tuning_trials: int = 10,
    ) -> None:
        if n_trees < 1 or not 0 < feature_fraction <= 1 or tuning_trials < 1:
            raise ValueError("树数量、特征比例或调参次数非法。")
        if tuning_backend not in {"none", "optuna"}:
            raise ValueError("tuning_backend 必须是 'none' 或 'optuna'。")
        self.n_trees = n_trees
        self.max_depths = tuple(max_depths or (2, 3, 4))
        if not self.max_depths or any(depth < 1 for depth in self.max_depths):
            raise ValueError("max_depths 必须包含正整数。")
        self.feature_fraction = feature_fraction
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.dt_params = dict(dt_params or {})
        self.tuning_backend = tuning_backend
        self.tuning_trials = tuning_trials
        self.tree_metadata_: List[Dict[str, Any]] = []

    def generate(
        self,
        df: FrameLike,
        *,
        target: str,
        features: Sequence[str] | None = None,
    ) -> List[MarsRule]:
        """训练浅层树并提取可部署路径规则。"""
        frame: pl.DataFrame = to_polars_frame(df)
        numeric_features: List[str] = _resolve_numeric_features(frame, target, features)
        if not numeric_features:
            self.tree_metadata_ = []
            return []
        model_frame, selected, x_values, y_values, model_features = _model_training_data(
            frame,
            target,
            numeric_features,
        )
        if model_frame.height < 2 or len(np.unique(y_values)) < 2:
            self.tree_metadata_ = []
            return _missing_rules(frame, numeric_features, "tree")
        rng = random.Random(self.random_state)
        sampled_size: int = max(1, int(len(numeric_features) * self.feature_fraction))
        tasks: List[Tuple[int, List[int], List[str], int, int]] = []
        for tree_index in range(self.n_trees):
            sampled: List[str] = sorted(rng.sample(numeric_features, sampled_size))
            depth: int = rng.choice(self.max_depths)
            seed: int = rng.randint(0, 2**31 - 1)
            sampled_indices: List[int] = [
                index
                for index, model_feature in enumerate(model_features)
                if model_feature.original_name in sampled
            ]
            tasks.append((tree_index, sampled_indices, sampled, depth, seed))

        def fit_tree(
            tree_index: int,
            sampled_indices: List[int],
            sampled: List[str],
            depth: int,
            seed: int,
        ) -> Tuple[List[MarsRule], Dict[str, Any]]:
            """拟合一棵树并返回确定性规则与诊断元数据。"""
            local_x: npt.NDArray[Any] = x_values[:, sampled_indices]
            local_features: List[_ModelFeature] = [
                model_features[index] for index in sampled_indices
            ]
            params: Dict[str, Any] = dict(self.dt_params)
            params.setdefault("max_depth", depth)
            params.setdefault("min_samples_leaf", max(1, int(model_frame.height * 0.01)))
            params.setdefault("random_state", seed)
            tuning_metadata: Dict[str, Any] = {
                "status": "disabled",
                "cv_folds": None,
                "best_cv_roc_auc": None,
            }
            if self.tuning_backend == "optuna":
                tuned_params, tuning_metadata = self._tune_params(
                    local_x,
                    y_values,
                    params,
                    seed,
                )
                params.update(tuned_params)
            classifier = DecisionTreeClassifier(**params)
            classifier.fit(local_x, y_values)
            extracted: List[str] = _extract_sklearn_tree_rules(
                classifier,
                local_features,
            )
            rules: List[MarsRule] = [
                MarsRule(expression, source="tree") for expression in extracted
            ]
            metadata: Dict[str, Any] = {
                "tree_index": tree_index,
                "features": sampled,
                "max_depth": params.get("max_depth"),
                "seed": seed,
                "training_sample_count": model_frame.height,
                "rule_count": len(extracted),
                "feature_importances": _aggregate_feature_importances(
                    classifier.feature_importances_,
                    local_features,
                ),
                "tuning": tuning_metadata,
            }
            return rules, metadata

        fitted: List[Tuple[List[MarsRule], Dict[str, Any]]] = Parallel(
            n_jobs=self.n_jobs,
            backend="threading",
        )(delayed(fit_tree)(*task) for task in tasks)
        rules: List[MarsRule] = _missing_rules(frame, selected, "tree")
        metadata: List[Dict[str, Any]] = []
        for tree_rules, tree_metadata in fitted:
            rules.extend(tree_rules)
            metadata.append(tree_metadata)
        self.tree_metadata_ = metadata
        return _deduplicate_rules(rules)

    def _tune_params(
        self,
        x_values: npt.NDArray[Any],
        y_values: npt.NDArray[Any],
        base_params: Mapping[str, Any],
        seed: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """使用确定性分层交叉验证 ROC AUC 执行 Optuna 调参。"""
        optuna: Any = require_optional_module(
            "optuna",
            feature_name="Mars tree rule tuning",
            extra_hint='pip install "mars-risk[tuning]"',
        )
        _, class_counts = np.unique(y_values, return_counts=True)
        minority_count: int = int(class_counts.min()) if len(class_counts) >= 2 else 0
        if minority_count < 2:
            return {}, {
                "status": "skipped_insufficient_minority_class",
                "cv_folds": None,
                "best_cv_roc_auc": None,
                "seed": seed,
            }
        fold_count: int = min(5, minority_count)
        splitter = StratifiedKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=seed,
        )

        def objective(trial: Any) -> float:
            """返回单次浅层树的平均分层 CV ROC AUC。"""
            params: Dict[str, Any] = dict(base_params)
            params["max_depth"] = trial.suggest_int("max_depth", 2, 5)
            params["min_samples_leaf"] = trial.suggest_float(
                "min_samples_leaf",
                0.005,
                0.05,
            )
            model = DecisionTreeClassifier(**params)
            scores: npt.NDArray[Any] = cross_val_score(
                model,
                x_values,
                y_values,
                scoring="roc_auc",
                cv=splitter,
                n_jobs=1,
            )
            return float(scores.mean())

        sampler: Any = optuna.samplers.TPESampler(seed=seed)
        study: Any = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.tuning_trials, show_progress_bar=False)
        return dict(study.best_params), {
            "status": "completed",
            "cv_folds": fold_count,
            "best_cv_roc_auc": float(study.best_value),
            "seed": seed,
            "best_params": dict(study.best_params),
        }


class MarsForestRuleGenerator(MarsRuleGenerator):
    """从随机森林叶子路径提取候选规则。"""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 3,
        feature_fraction: float = 0.8,
        min_samples_leaf: NumericLeafSize = 0.01,
        random_state: int | None = None,
        n_jobs: int = -1,
    ) -> None:
        _validate_tree_ensemble_params(n_estimators, max_depth, feature_fraction, min_samples_leaf)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_fraction = feature_fraction
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.forest_metadata_: Dict[str, Any] = {}

    def generate(
        self,
        df: FrameLike,
        *,
        target: str,
        features: Sequence[str] | None = None,
    ) -> List[MarsRule]:
        """训练随机森林并提取全部叶子路径。"""
        original: pl.DataFrame = to_polars_frame(df)
        frame, selected, x_values, y_values, model_features = _model_training_data(
            original,
            target,
            features,
        )
        if not selected or frame.height < 2 or len(np.unique(y_values)) < 2:
            return []
        classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.feature_fraction,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        classifier.fit(x_values, y_values)
        expressions: List[str] = []
        for estimator in classifier.estimators_:
            expressions.extend(_extract_sklearn_tree_rules(estimator, model_features))
        rules: List[MarsRule] = _missing_rules(original, selected, "forest")
        rules.extend(MarsRule(expression, source="forest") for expression in expressions)
        result: List[MarsRule] = _deduplicate_rules(rules)
        self.forest_metadata_ = {
            "rule_count": len(result),
            "feature_count": len(selected),
            "feature_importances": _aggregate_feature_importances(
                classifier.feature_importances_,
                model_features,
            ),
        }
        return result


class MarsGBDTRuleGenerator(MarsRuleGenerator):
    """从 sklearn 或 LightGBM GBDT 弱学习器提取规则路径。"""

    def __init__(
        self,
        backend: Literal["sklearn", "lightgbm"] = "sklearn",
        n_estimators: int = 80,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        min_samples_leaf: NumericLeafSize = 0.01,
        random_state: int | None = None,
        n_jobs: int = -1,
        model_params: Dict[str, Any] | None = None,
    ) -> None:
        if backend not in {"sklearn", "lightgbm"}:
            raise ValueError("backend 必须是 'sklearn' 或 'lightgbm'。")
        _validate_tree_ensemble_params(n_estimators, max_depth, feature_fraction, min_samples_leaf)
        if learning_rate <= 0:
            raise ValueError("learning_rate 必须为正数。")
        self.backend = backend
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.feature_fraction = feature_fraction
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_params = dict(model_params or {})
        self.gbdt_metadata_: Dict[str, Any] = {}

    def generate(
        self,
        df: FrameLike,
        *,
        target: str,
        features: Sequence[str] | None = None,
    ) -> List[MarsRule]:
        """训练指定 GBDT 后端并提取路径规则。"""
        original: pl.DataFrame = to_polars_frame(df)
        frame, selected, x_values, y_values, model_features = _model_training_data(
            original,
            target,
            features,
        )
        if not selected or frame.height < 2 or len(np.unique(y_values)) < 2:
            return []
        if self.backend == "lightgbm":
            expressions = self._generate_lightgbm(
                x_values,
                y_values,
                model_features,
                frame.height,
            )
        else:
            model_params: Dict[str, Any] = dict(self.model_params)
            model_params.setdefault(
                "max_features",
                max(1, int(len(model_features) * self.feature_fraction)),
            )
            classifier = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                **model_params,
            )
            classifier.fit(x_values, y_values)
            expressions = []
            for estimator in classifier.estimators_.ravel():
                expressions.extend(_extract_sklearn_tree_rules(estimator, model_features))
        rules: List[MarsRule] = _missing_rules(original, selected, "gbdt")
        rules.extend(MarsRule(expression, source="gbdt") for expression in expressions)
        result: List[MarsRule] = _deduplicate_rules(rules)
        self.gbdt_metadata_ = {
            "backend": self.backend,
            "rule_count": len(result),
            "feature_count": len(selected),
        }
        return result

    def _generate_lightgbm(
        self,
        x_values: npt.NDArray[Any],
        y_values: npt.NDArray[Any],
        model_features: Sequence[_ModelFeature],
        sample_count: int,
    ) -> List[str]:
        """延迟导入 LightGBM 并解析 dump_model。"""
        lightgbm: Any = require_optional_module(
            "lightgbm",
            feature_name="Mars GBDT rule generation",
            extra_hint='pip install "mars-risk[ml]"',
        )
        min_child_samples: int = (
            max(1, int(sample_count * self.min_samples_leaf))
            if isinstance(self.min_samples_leaf, float)
            else self.min_samples_leaf
        )
        params: Dict[str, Any] = dict(self.model_params)
        params.update(
            {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "feature_fraction": self.feature_fraction,
                "min_child_samples": min_child_samples,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "verbosity": -1,
            }
        )
        classifier: Any = lightgbm.LGBMClassifier(**params)
        classifier.fit(x_values, y_values)
        model: Dict[str, Any] = classifier.booster_.dump_model()
        expressions: List[str] = []
        for tree in model.get("tree_info", []):
            expressions.extend(
                _extract_lightgbm_rules(
                    tree.get("tree_structure", {}),
                    model_features,
                    [],
                )
            )
        return expressions


class MarsIsolationRuleGenerator(MarsRuleGenerator):
    """从孤立森林异常路径生成无监督候选规则。"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, float, Literal["auto"]] = "auto",
        feature_fraction: float = 0.8,
        contamination: Union[float, Literal["auto"]] = "auto",
        path_depth_limit: int = 4,
        min_leaf_samples: int = 50,
        anomaly_quantile: float = 0.95,
        max_candidates: int = 5000,
        random_state: int | None = None,
        n_jobs: int = -1,
        model_params: Dict[str, Any] | None = None,
    ) -> None:
        if n_estimators < 1 or not 0 < feature_fraction <= 1:
            raise ValueError("n_estimators 或 feature_fraction 非法。")
        if path_depth_limit < 1 or min_leaf_samples < 1 or max_candidates < 1:
            raise ValueError("路径深度、叶子样本数和候选预算必须至少为 1。")
        if not 0 < anomaly_quantile < 1:
            raise ValueError("anomaly_quantile 必须位于 (0, 1)。")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.feature_fraction = feature_fraction
        self.contamination = contamination
        self.path_depth_limit = path_depth_limit
        self.min_leaf_samples = min_leaf_samples
        self.anomaly_quantile = anomaly_quantile
        self.max_candidates = max_candidates
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_params = dict(model_params or {})
        self.iforest_metadata_: Dict[str, Any] = {}

    def generate(
        self,
        df: FrameLike,
        *,
        target: str,
        features: Sequence[str] | None = None,
    ) -> List[MarsRule]:
        """训练孤立森林并提取受深度和预算约束的路径。"""
        original: pl.DataFrame = to_polars_frame(df)
        frame, selected, x_values, _, model_features = _model_training_data(
            original,
            target,
            features,
        )
        if not selected or frame.height < 2:
            return []
        model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.feature_fraction,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.model_params,
        )
        model.fit(x_values)
        anomaly_score: np.ndarray = -model.score_samples(x_values)
        anomaly_threshold: float = float(np.quantile(anomaly_score, self.anomaly_quantile))
        anomaly_mask: np.ndarray = anomaly_score >= anomaly_threshold
        global_anomaly_rate: float = float(anomaly_mask.mean())
        candidates: List[Tuple[float, int, int, str]] = []
        for estimator, feature_indices in zip(model.estimators_, model.estimators_features_):
            local_features: List[_ModelFeature] = [
                model_features[int(index)] for index in feature_indices
            ]
            tree_values: np.ndarray = x_values[:, np.asarray(feature_indices, dtype=int)]
            leaf_ids: np.ndarray = estimator.apply(tree_values)
            leaf_paths: Dict[int, str] = _extract_sklearn_leaf_paths(
                estimator,
                local_features,
                max_depth=self.path_depth_limit,
            )
            for leaf_id in np.unique(leaf_ids):
                expression: str | None = leaf_paths.get(int(leaf_id))
                if expression is None:
                    continue
                hit_mask: np.ndarray = leaf_ids == leaf_id
                sample_count: int = int(hit_mask.sum())
                if sample_count < self.min_leaf_samples:
                    continue
                anomaly_count: int = int(np.logical_and(hit_mask, anomaly_mask).sum())
                anomaly_rate: float = anomaly_count / sample_count
                if anomaly_count and anomaly_rate >= global_anomaly_rate:
                    candidates.append(
                        (anomaly_rate, anomaly_count, sample_count, expression)
                    )
        ranked_expressions: List[str] = [
            item[3]
            for item in sorted(
                candidates,
                key=lambda item: (-item[0], -item[1], -item[2], item[3]),
            )
        ]
        rules: List[MarsRule] = _missing_rules(original, selected, "isolation")
        rules.extend(
            MarsRule(expression, source="isolation") for expression in ranked_expressions
        )
        result: List[MarsRule] = _deduplicate_rules(rules)[: self.max_candidates]
        self.iforest_metadata_ = {
            "rule_count": len(result),
            "feature_count": len(selected),
            "anomaly_quantile": self.anomaly_quantile,
            "anomaly_threshold": anomaly_threshold,
            "global_anomaly_rate": global_anomaly_rate,
            "candidate_leaf_count": len(candidates),
        }
        return result


def _resolve_numeric_features(
    frame: pl.DataFrame,
    target: str,
    features: Sequence[str] | None,
) -> List[str]:
    """解析并校验生成器数值特征。"""
    if target not in frame.columns:
        raise ValueError(f"训练数据缺少目标列：{target!r}。")
    candidates: List[str] = list(features) if features is not None else [
        column for column in frame.columns if column != target
    ]
    missing: List[str] = [column for column in candidates if column not in frame.columns]
    if missing:
        raise ValueError(f"生成器候选特征缺失：{missing}。")
    numeric_types = {
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
    return [column for column in candidates if frame.schema[column] in numeric_types]


def _model_training_data(
    df: FrameLike,
    target: str,
    features: Sequence[str] | None,
) -> Tuple[
    pl.DataFrame,
    List[str],
    npt.NDArray[Any],
    npt.NDArray[Any],
    List[_ModelFeature],
]:
    """构造带数值填充值和显式缺失指示器的模型矩阵。"""
    frame: pl.DataFrame = to_polars_frame(df)
    selected: List[str] = _resolve_numeric_features(frame, target, features)
    target_alias: str = "__mars_model_target"
    work: pl.DataFrame = (
        frame
        .with_columns(pl.col(target).cast(pl.Float64, strict=False).alias(target_alias))
        .filter(pl.col(target_alias).is_not_null() & pl.col(target_alias).is_not_nan())
    )
    expressions: List[pl.Expr] = []
    model_features: List[_ModelFeature] = []
    for index, feature in enumerate(selected):
        numeric: pl.Expr = pl.col(feature).cast(pl.Float64, strict=False)
        valid_values: pl.Series = (
            work
            .select(numeric.alias("value"))["value"]
            .filter(work.select((numeric.is_not_null() & ~numeric.is_nan()).alias("valid"))["valid"])
        )
        median: Any = valid_values.median() if valid_values.len() else None
        fill_value: float = float(median) if median is not None else 0.0
        missing: pl.Expr = numeric.is_null() | numeric.is_nan()
        expressions.extend(
            [
                pl.when(missing)
                .then(pl.lit(fill_value))
                .otherwise(numeric)
                .alias(f"__mars_model_value_{index}"),
                missing.cast(pl.Float64).alias(f"__mars_model_missing_{index}"),
            ]
        )
        model_features.extend(
            [
                _ModelFeature(feature, "value", fill_value),
                _ModelFeature(feature, "missing", fill_value),
            ]
        )
    matrix: pl.DataFrame = work.select(expressions) if expressions else pl.DataFrame()
    x_values: npt.NDArray[Any] = (
        matrix.to_numpy() if expressions else np.empty((work.height, 0))
    )
    y_values: npt.NDArray[Any] = work[target_alias].to_numpy()
    return work, selected, x_values, y_values, model_features


def _quantile_cut_map(
    frame: pl.DataFrame,
    features: Sequence[str],
    n_bins: int,
) -> Dict[str, List[float]]:
    """以单次并行聚合计算全部特征的有限唯一分位切点。"""
    if not features:
        return {}
    aliases: Dict[str, List[str]] = {}
    expressions: List[pl.Expr] = []
    for feature_index, feature in enumerate(features):
        feature_aliases: List[str] = []
        for quantile_index in range(1, n_bins):
            alias: str = f"q_{feature_index}_{quantile_index}"
            feature_aliases.append(alias)
            expressions.append(
                pl.col(feature)
                .drop_nulls()
                .quantile(quantile_index / n_bins)
                .alias(alias)
            )
        aliases[feature] = feature_aliases
    values: Dict[str, Any] = frame.select(expressions).row(0, named=True)
    result: Dict[str, List[float]] = {}
    for feature, feature_aliases in aliases.items():
        cuts: List[float] = []
        for alias in feature_aliases:
            value: Any = values[alias]
            if value is None:
                continue
            number: float = float(value)
            if math.isfinite(number) and number not in cuts:
                cuts.append(number)
        result[feature] = sorted(cuts)
    return result


def _rank_single_rules(
    frame: pl.DataFrame,
    target: str,
    rules: Sequence[MarsRule],
) -> List[MarsRule]:
    """按绝对偏离基准 Lift 的幅度稳定排序单规则。"""
    target_frame: pl.DataFrame = frame.select([*sorted({f for r in rules for f in r.required_features}), target])
    target_frame = target_frame.with_columns(pl.col(target).cast(pl.Float64, strict=False))
    target_frame = target_frame.filter(pl.col(target).is_not_null() & pl.col(target).is_not_nan())
    base_rate: float | None = (
        float(target_frame[target].mean()) if target_frame.height else None
    )
    scores: List[Tuple[float, str, MarsRule]] = []
    for start in range(0, len(rules), 100):
        batch: Sequence[MarsRule] = rules[start : start + 100]
        expressions: List[pl.Expr] = []
        for index, rule in enumerate(batch):
            mask: pl.Expr = expression_to_polars(
                parse_expression(rule.expression),
                target_frame.schema,
            ).fill_null(False)
            expressions.append(
                pl.col(target).filter(mask).mean().alias(f"event_rate_{index}")
            )
        values: Dict[str, Any] = target_frame.select(expressions).row(0, named=True)
        for index, rule in enumerate(batch):
            raw_event_rate: Any = values[f"event_rate_{index}"]
            event_rate: float | None = (
                float(raw_event_rate) if raw_event_rate is not None else None
            )
            lift: float | None = (
                event_rate / base_rate
                if event_rate is not None and base_rate is not None and base_rate != 0.0
                else None
            )
            score: float = abs(lift - 1.0) if lift is not None else -1.0
            scores.append((score, rule.rule_id, rule))
    return [item[2] for item in sorted(scores, key=lambda item: (-item[0], item[1]))]


def _missing_rules(frame: pl.DataFrame, features: Sequence[str], source: str) -> List[MarsRule]:
    """为存在缺失的模型特征生成显式缺失候选。"""
    rules: List[MarsRule] = []
    for feature in features:
        numeric: pl.Expr = pl.col(feature).cast(pl.Float64, strict=False)
        missing_count: int = int(
            frame.select((numeric.is_null() | numeric.is_nan()).sum()).item() or 0
        )
        if missing_count > 0:
            rules.append(
                MarsRule(
                    f"{_quote_dsl_identifier(feature)} IS MISSING",
                    source=source,
                )
            )
    return rules


def _extract_sklearn_tree_rules(
    estimator: Any,
    model_features: Sequence[_ModelFeature],
    *,
    max_depth: int | None = None,
    min_leaf_samples: int = 1,
) -> List[str]:
    """递归提取 sklearn 树的叶子路径。"""
    return list(
        _extract_sklearn_leaf_paths(
            estimator,
            model_features,
            max_depth=max_depth,
            min_leaf_samples=min_leaf_samples,
        ).values()
    )


def _extract_sklearn_leaf_paths(
    estimator: Any,
    model_features: Sequence[_ModelFeature],
    *,
    max_depth: int | None = None,
    min_leaf_samples: int = 1,
) -> Dict[int, str]:
    """递归提取 sklearn 树的叶子节点到可部署缺失路径映射。"""
    tree = estimator.tree_
    rules: Dict[int, str] = {}

    def visit(node: int, path: List[str], depth: int) -> None:
        """遍历节点并记录满足预算的叶子路径。"""
        left: int = int(tree.children_left[node])
        right: int = int(tree.children_right[node])
        is_leaf: bool = left == right
        if is_leaf:
            if path and int(tree.n_node_samples[node]) >= min_leaf_samples:
                rules[node] = " AND ".join(path)
            return
        if max_depth is not None and depth >= max_depth:
            visit(left, path, depth + 1)
            visit(right, path, depth + 1)
            return
        feature_index: int = int(tree.feature[node])
        if feature_index == _tree.TREE_UNDEFINED or feature_index >= len(model_features):
            return
        model_feature: _ModelFeature = model_features[feature_index]
        threshold: float = float(tree.threshold[node])
        visit(
            left,
            path + [_model_split_condition(model_feature, threshold, go_left=True)],
            depth + 1,
        )
        visit(
            right,
            path + [_model_split_condition(model_feature, threshold, go_left=False)],
            depth + 1,
        )

    visit(0, [], 0)
    return rules


def _extract_lightgbm_rules(
    node: Mapping[str, Any],
    model_features: Sequence[_ModelFeature],
    path: List[str],
) -> List[str]:
    """递归提取 LightGBM dump_model 路径。"""
    if "split_feature" not in node:
        return [" AND ".join(path)] if path else []
    feature_index: int = int(node["split_feature"])
    if feature_index >= len(model_features):
        return []
    try:
        threshold: float = float(node["threshold"])
    except (KeyError, TypeError, ValueError):
        return []
    model_feature: _ModelFeature = model_features[feature_index]
    rules: List[str] = []
    rules.extend(
        _extract_lightgbm_rules(
            node.get("left_child", {}),
            model_features,
            path + [_model_split_condition(model_feature, threshold, go_left=True)],
        )
    )
    rules.extend(
        _extract_lightgbm_rules(
            node.get("right_child", {}),
            model_features,
            path + [_model_split_condition(model_feature, threshold, go_left=False)],
        )
    )
    return rules


def _model_split_condition(
    model_feature: _ModelFeature,
    threshold: float,
    *,
    go_left: bool,
) -> str:
    """把模型矩阵分支精确还原为原始特征 DSL。"""
    identifier: str = _quote_dsl_identifier(model_feature.original_name)
    if model_feature.kind == "missing":
        return (
            f"{identifier} IS NOT MISSING"
            if go_left
            else f"{identifier} IS MISSING"
        )
    operator: str = "<=" if go_left else ">"
    comparison: str = f"{identifier} {operator} {threshold!r}"
    missing_takes_branch: bool = (
        model_feature.fill_value <= threshold
        if go_left
        else model_feature.fill_value > threshold
    )
    if missing_takes_branch:
        return f"({identifier} IS MISSING OR {comparison})"
    return f"({identifier} IS NOT MISSING AND {comparison})"


def _aggregate_feature_importances(
    importances: npt.NDArray[Any],
    model_features: Sequence[_ModelFeature],
) -> Dict[str, float]:
    """把数值列和缺失指示器的重要性合并回原始特征。"""
    result: Dict[str, float] = {}
    numeric_importances: List[float] = [float(value) for value in importances]
    for model_feature, importance in zip(model_features, numeric_importances):
        result[model_feature.original_name] = (
            result.get(model_feature.original_name, 0.0) + importance
        )
    return result


def _deduplicate_rules(rules: Iterable[MarsRule]) -> List[MarsRule]:
    """按确定性 ID 保留首次出现规则。"""
    seen: Dict[str, str] = {}
    result: List[MarsRule] = []
    for rule in rules:
        existing: str | None = seen.get(rule.rule_id)
        if existing is not None and existing != rule.expression:
            raise ValueError(f"规则 ID 哈希冲突：{rule.rule_id}。")
        if existing is None:
            seen[rule.rule_id] = rule.expression
            result.append(rule)
    return result


def _quote_dsl_identifier(identifier: str) -> str:
    """安全引用 DSL 标识符。"""
    return '"' + identifier.replace('"', '""') + '"'


def _validate_tree_ensemble_params(
    n_estimators: int,
    max_depth: int,
    feature_fraction: float,
    min_samples_leaf: NumericLeafSize,
) -> None:
    """校验树集成生成器共享参数。"""
    if n_estimators < 1 or max_depth < 1 or not 0 < feature_fraction <= 1:
        raise ValueError("树数量、深度和特征比例非法。")
    if isinstance(min_samples_leaf, float) and not 0 < min_samples_leaf <= 1:
        raise ValueError("浮点 min_samples_leaf 必须位于 (0, 1]。")
    if isinstance(min_samples_leaf, int) and min_samples_leaf < 1:
        raise ValueError("整数 min_samples_leaf 必须至少为 1。")
