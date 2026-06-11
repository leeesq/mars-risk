"""逐步增加特征的建模调参工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from mars.modeling.artifacts import read_json, step_artifact_dir, write_json
from mars.modeling.results import MarsModelTuningResult
from mars.modeling.tuning import MarsModelTuner, _build_spec
from mars.modeling.utils import FrameLike


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    """按输入顺序去重。"""
    seen: set[str] = set()
    output: List[str] = []
    for value in values:
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _json_dumps(value: Any) -> str:
    """生成稳定、可读的 JSON 文本。"""
    return json.dumps(value, ensure_ascii=False)


@dataclass(slots=True)
class MarsFeatureGrowthResult:
    """
    逐步增加特征调参的结构化结果。

    Attributes
    ----------
    summary_table : pandas.DataFrame
        step 级汇总审计表。
    runs : dict of int to MarsModelTuningResult
        每个成功 step 对应的调参结果。
    best_step : int or None
        推荐模型对应的特征数量；若无成功 step 则为 ``None``。
    best_run : MarsModelTuningResult or None
        推荐 step 对应的调参结果。

    Examples
    --------
    >>> run = MarsFeatureGrowthResult(
    ...     model_type="xgb",
    ...     optimize_metric="ks",
    ...     feature_order=["age"],
    ...     steps=[1],
    ...     selection_metric="ks",
    ...     summary_table=pd.DataFrame(),
    ...     runs={},
    ... )
    >>> run.best_features
    []
    """

    model_type: str
    optimize_metric: str
    feature_order: List[str]
    steps: List[int]
    selection_metric: str
    summary_table: pd.DataFrame
    runs: Dict[int, MarsModelTuningResult]
    best_step: int | None = None
    best_run: MarsModelTuningResult | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_model(self) -> Any:
        """
        返回推荐 step 的最佳模型。

        Returns
        -------
        Any
            推荐 step 的最佳模型；若尚无推荐 run，则返回 ``None``。

        Examples
        --------
        >>> run = MarsFeatureGrowthResult("xgb", "ks", ["age"], [1], "ks", pd.DataFrame(), {})
        >>> run.best_model is None
        True
        """
        return None if self.best_run is None else self.best_run.best_model

    @property
    def best_score(self) -> float | None:
        """
        返回推荐 step 的 validation 分数。

        Returns
        -------
        float or None
            推荐 step 的验证集分数；若尚无推荐 run，则返回 ``None``。

        Examples
        --------
        >>> run = MarsFeatureGrowthResult("xgb", "ks", ["age"], [1], "ks", pd.DataFrame(), {})
        >>> run.best_score is None
        True
        """
        return None if self.best_run is None else float(self.best_run.best_score)

    @property
    def best_features(self) -> List[str]:
        """
        返回推荐 step 使用的特征列表。

        Returns
        -------
        list of str
            推荐 step 使用的特征列名；若尚无推荐 run，则返回空列表。

        Examples
        --------
        >>> run = MarsFeatureGrowthResult("xgb", "ks", ["age"], [1], "ks", pd.DataFrame(), {})
        >>> run.best_features
        []
        """
        if self.best_run is None:
            return []
        return list(self.best_run.features)

    def write_artifact(self, path: str) -> Path:
        """
        将特征增长实验结果写入本地 artifact 目录。

        Parameters
        ----------
        path : str
            输出目录。

        Returns
        -------
        pathlib.Path
            artifact 目录路径。

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> run = MarsFeatureGrowthResult("xgb", "ks", ["age"], [1], "ks", pd.DataFrame(), {})
        >>> with TemporaryDirectory() as tmp:
        ...     artifact_dir = run.write_artifact(tmp)
        ...     artifact_dir.exists()
        True
        """
        artifact_dir = Path(path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        runs_dir = artifact_dir / "step_runs"
        runs_dir.mkdir(exist_ok=True)

        summary_path = artifact_dir / "summary.csv"
        self.summary_table.to_csv(summary_path, index=False)

        run_dirs: Dict[str, str] = {}
        for step, run in sorted(self.runs.items()):
            rel_dir = Path("step_runs") / f"step_{step}"
            run.write_artifact(str(artifact_dir / rel_dir))
            run_dirs[str(step)] = str(rel_dir)

        metadata = {
            "artifact_type": "mars_feature_growth_result",
            "model_type": self.model_type,
            "optimize_metric": self.optimize_metric,
            "feature_order": self.feature_order,
            "steps": self.steps,
            "selection_metric": self.selection_metric,
            "best_step": self.best_step,
            "metadata": self.metadata,
            "files": {
                "summary": summary_path.name,
                "runs": run_dirs,
            },
        }
        write_json(artifact_dir / "metadata.json", metadata)
        return artifact_dir

    @classmethod
    def load_artifact(cls: type[MarsFeatureGrowthResult], path: str) -> MarsFeatureGrowthResult:
        """
        从本地 artifact 目录恢复特征增长实验结果。

        Parameters
        ----------
        path : str
            artifact 目录。

        Returns
        -------
        MarsFeatureGrowthResult
            恢复后的实验结果。

        Raises
        ------
        FileNotFoundError
            当指定路径不存在时抛出。
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> run = MarsFeatureGrowthResult("xgb", "ks", ["age"], [1], "ks", pd.DataFrame(), {})
        >>> with TemporaryDirectory() as tmp:
        ...     _ = run.write_artifact(tmp)
        ...     MarsFeatureGrowthResult.load_artifact(tmp).feature_order
        ['age']
        """
        artifact_dir = Path(path)
        metadata = read_json(artifact_dir / "metadata.json")
        if metadata.get("artifact_type") != "mars_feature_growth_result":
            raise ValueError(
                f"Unsupported artifact type for MarsFeatureGrowthResult: {metadata.get('artifact_type')!r}"
            )

        files = metadata.get("files", {})
        summary_path = artifact_dir / files.get("summary", "summary.csv")
        if not summary_path.exists():
            raise FileNotFoundError(f"Feature growth summary file is missing: {summary_path}")

        runs: Dict[int, MarsModelTuningResult] = {}
        for step_text, rel_dir in dict(files.get("runs", {})).items():
            step = int(step_text)
            runs[step] = MarsModelTuningResult.load_artifact(str(artifact_dir / rel_dir))

        best_step = metadata.get("best_step")
        if best_step is not None:
            best_step = int(best_step)
        return cls(
            model_type=str(metadata["model_type"]),
            optimize_metric=str(metadata["optimize_metric"]),
            feature_order=list(metadata.get("feature_order", [])),
            steps=[int(step) for step in metadata.get("steps", [])],
            selection_metric=str(metadata.get("selection_metric", metadata.get("optimize_metric", "ks"))),
            summary_table=pd.read_csv(summary_path),
            runs=runs,
            best_step=best_step,
            best_run=runs.get(best_step) if best_step is not None else None,
            metadata=dict(metadata.get("metadata", {})),
        )


class MarsFeatureIncrementalTuner:
    """
    按特征数量逐步扩展的调参器。

    Attributes
    ----------
    spec : ModelingSpec
        由初始化参数构建的建模规格。
    features : list of str
        去重后的候选特征顺序。

    Examples
    --------
    >>> tuner = MarsFeatureIncrementalTuner(model_type="xgb", features=["age"], target="y")
    >>> tuner.features
    ['age']
    """

    def __init__(
        self,
        *,
        model_type: str,
        features: Sequence[str],
        target: str,
        dataset_flag_col: str = "dataset_flag",
        categorical_features: Sequence[str] | None = None,
        optimize_metric: str = "ks",
        seed: int = 1206,
        lr_feature_mode: str = "numeric",
        lr_binning_type: str = "native",
        lr_binner_kwargs: Mapping[str, Any] | None = None,
        lr_binner: Any | None = None,
    ) -> None:
        """
        初始化特征增量调参器。

        Parameters
        ----------
        model_type : str
            模型后端类型。
        features : Sequence[str]
            候选特征列。
        target : str
            目标列名。
        dataset_flag_col : str
            样本切片标记列名。
        categorical_features : Sequence[str] | None
            类别特征列。
        optimize_metric : str
            调参优化指标。
        seed : int
            随机种子。
        lr_feature_mode : str
            LR 特征模式。
        lr_binning_type : str
            LR WOE 模式使用的分箱器类型，支持 ``native``、``optimal`` 和 ``lite_opt``。
        lr_binner_kwargs : Mapping[str, Any] | None
            构造 LR 分箱器时使用的参数。
        lr_binner : Any | None
            显式复用的 LR 分箱器实例。
        """
        self.spec = _build_spec(
            model_type=model_type,
            features=features,
            target=target,
            dataset_flag_col=dataset_flag_col,
            categorical_features=categorical_features,
            optimize_metric=optimize_metric,
            seed=seed,
            lr_feature_mode=lr_feature_mode,
            lr_binning_type=lr_binning_type,
            lr_binner_kwargs=lr_binner_kwargs,
            lr_binner=lr_binner,
        )

    def _resolve_feature_order(
        self,
        *,
        feature_order: Sequence[str] | None,
        importance_table: pd.DataFrame | None,
    ) -> List[str]:
        """解析增量实验使用的特征顺序。"""
        base_features = list(self.spec.features)
        base_set = set(base_features)

        if feature_order is not None:
            ordered = _dedupe_preserve_order(feature_order)
            unknown = sorted(set(ordered).difference(base_set))
            if unknown:
                raise ValueError(f"feature_order contains unknown features: {unknown}")
        elif importance_table is not None:
            if "feature" not in importance_table.columns:
                raise ValueError("importance_table must contain a 'feature' column.")
            table = importance_table.copy()
            if "rank" in table.columns:
                table = table.sort_values("rank", ascending=True)
            elif "importance" in table.columns:
                table = table.sort_values("importance", ascending=False)
            ordered = _dedupe_preserve_order(table["feature"].astype(str).tolist())
            ordered = [feature for feature in ordered if feature in base_set]
        else:
            ordered = list(base_features)

        ordered_set = set(ordered)
        ordered.extend(feature for feature in base_features if feature not in ordered_set)
        if not ordered:
            raise ValueError("At least one feature is required for incremental tuning.")
        return ordered

    @staticmethod
    def _resolve_steps(
        *,
        total_features: int,
        steps: Sequence[int] | None,
        min_features: int,
        max_features: int | None,
        step_size: int | None,
    ) -> List[int]:
        """解析每轮前缀特征数量。"""
        if total_features <= 0:
            raise ValueError("total_features must be positive.")

        if steps is not None:
            resolved = sorted({min(max(int(step), 1), total_features) for step in steps})
            if not resolved:
                raise ValueError("steps must contain at least one positive integer.")
            return resolved

        max_count = total_features if max_features is None else min(int(max_features), total_features)
        max_count = max(max_count, 1)
        min_count = min(max(int(min_features), 1), max_count)
        if step_size is None:
            step = max(1, min_count)
        else:
            step = max(int(step_size), 1)

        resolved = list(range(min_count, max_count + 1, step))
        if not resolved or resolved[-1] != max_count:
            resolved.append(max_count)
        return sorted(set(resolved))

    @staticmethod
    def _step_history_path(base_path: str | Path, feature_count: int) -> str:
        """为每个 step 生成互不覆盖的 history 路径。"""
        path = Path(base_path)
        suffix = path.suffix or ".csv"
        return str(path.with_name(f"{path.stem}_features_{feature_count}{suffix}"))


    @staticmethod
    def _select_history_row(
        history: pd.DataFrame,
        metric: str,
        *,
        direction: str = "maximize",
    ) -> pd.Series | None:
        """选择某个 step 中用于汇总展示的最佳有效 trial。"""
        if history.empty:
            return None
        score_col = f"val_{metric}"
        if score_col not in history.columns:
            return None
        valid = history.copy()
        if "trial_state" in valid.columns:
            valid = valid[valid["trial_state"].astype(str) == "COMPLETE"]
        if "is_valid" in valid.columns:
            valid = valid[valid["is_valid"].astype(str).str.lower().isin({"true", "1"})]
        scores = pd.to_numeric(valid[score_col], errors="coerce")
        valid = valid.loc[scores.notna()].copy()
        if valid.empty:
            return None
        if direction == "minimize":
            return valid.loc[scores.loc[valid.index].idxmin()]
        return valid.loc[scores.loc[valid.index].idxmax()]

    @staticmethod
    def _success_row(run: MarsModelTuningResult, *, feature_count: int, selection_metric: str) -> Dict[str, Any]:
        """构建成功 step 的轻量审计行。"""
        row: Dict[str, Any] = {
            "feature_count": int(feature_count),
            "status": "complete",
            "features": _json_dumps(list(run.features)),
            "best_score": float(run.best_score),
            "best_iteration": run.best_iteration,
            "backend_data_mode": run.backend_data_mode,
            "history_path": run.history_path,
            "artifact_path": run.artifact_path,
            "model_type": run.model_type,
            "optimize_metric": run.optimize_metric,
            "error": None,
        }
        direction = dict(getattr(run, "metric_directions", {}) or {}).get(selection_metric, "maximize")
        history_row = MarsFeatureIncrementalTuner._select_history_row(
            run.history_table,
            selection_metric,
            direction=direction,
        )
        if history_row is None:
            row["selection_score"] = float(run.best_score)
            return row

        score_col = f"val_{selection_metric}"
        row["selection_score"] = pd.to_numeric(pd.Series([history_row.get(score_col)]), errors="coerce").iloc[0]
        metric_suffixes = tuple(f"_{metric_name}" for metric_name in run.metric_names)
        for col in run.history_table.columns:
            col_name = str(col)
            if col_name in {"trial_num", "is_valid", "val_diff", "max_oot_diff"}:
                row[col_name] = history_row.get(col)
            elif col_name.endswith(metric_suffixes):
                row[col_name] = history_row.get(col)
        return row

    @staticmethod
    def _error_row(feature_count: int, features: Sequence[str], exc: Exception) -> Dict[str, Any]:
        """构建失败 step 的审计行。"""
        return {
            "feature_count": int(feature_count),
            "status": "error",
            "features": _json_dumps(list(features)),
            "best_score": None,
            "selection_score": None,
            "best_iteration": None,
            "backend_data_mode": None,
            "history_path": None,
            "artifact_path": None,
            "model_type": None,
            "optimize_metric": None,
            "error": str(exc)[:300],
        }

    @staticmethod
    def _choose_best(summary_table: pd.DataFrame, runs: Mapping[int, MarsModelTuningResult], metric: str) -> tuple[int | None, MarsModelTuningResult | None]:
        """按 validation 指标选择推荐 step。"""
        if summary_table.empty:
            return None, None
        score_col = f"val_{metric}"
        if score_col not in summary_table.columns:
            score_col = "selection_score"
        complete = summary_table[summary_table["status"] == "complete"].copy()
        if complete.empty or score_col not in complete.columns:
            return None, None
        scores = pd.to_numeric(complete[score_col], errors="coerce")
        complete = complete.loc[scores.notna()].copy()
        if complete.empty:
            return None, None
        sample_run = next(iter(runs.values()), None)
        direction = "maximize"
        if sample_run is not None:
            direction = dict(getattr(sample_run, "metric_directions", {}) or {}).get(metric, "maximize")
        best_idx = (
            scores.loc[complete.index].idxmin()
            if direction == "minimize"
            else scores.loc[complete.index].idxmax()
        )
        best_step = int(summary_table.loc[best_idx, "feature_count"])
        return best_step, runs.get(best_step)

    def tune(
        self,
        df: FrameLike,
        *,
        steps: Sequence[int] | None = None,
        feature_order: Sequence[str] | None = None,
        importance_table: pd.DataFrame | None = None,
        min_features: int = 10,
        max_features: int | None = None,
        step_size: int | None = None,
        mode: str = "prefix",
        selection_metric: str | None = None,
        **tune_kwargs: Any,
    ) -> MarsFeatureGrowthResult:
        """
        执行逐步增加特征的多轮调参。

        Parameters
        ----------
        df : FrameLike
            已带 train/val/OOT 标识的建模样本。
        steps : Sequence[int] | None
            显式指定每轮使用的前 N 个特征数量。
        feature_order : Sequence[str] | None
            人工指定的稳定特征顺序。
        importance_table : pd.DataFrame | None
            包含 ``feature`` 以及可选 ``importance`` / ``rank`` 的特征重要性表。
        min_features : int
            自动生成 step 时的起始特征数。
        max_features : int | None
            自动生成 step 时的最大特征数。
        step_size : int | None
            自动生成 step 时的步长；默认使用 ``min_features``。
        mode : str
            特征增长模式。当前版本只支持前缀扩展。
        selection_metric : str | None
            跨 step 选择推荐模型时使用的 validation 指标。
        **tune_kwargs : Any
            透传给 ``MarsModelTuner.tune`` 的参数。

        Returns
        -------
        MarsFeatureGrowthResult
            包含所有成功 step、汇总表和推荐模型的实验结果。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        Examples
        --------
        >>> tuner = MarsFeatureIncrementalTuner(model_type="xgb", features=["age"], target="y")
        >>> callable(tuner.tune)
        True
        """
        if mode.lower() != "prefix":
            raise ValueError("MarsFeatureIncrementalTuner currently supports only mode='prefix'.")
        metric = (selection_metric or self.spec.optimize_metric).lower()
        ordered_features = self._resolve_feature_order(
            feature_order=feature_order,
            importance_table=importance_table,
        )
        resolved_steps = self._resolve_steps(
            total_features=len(ordered_features),
            steps=steps,
            min_features=min_features,
            max_features=max_features,
            step_size=step_size,
        )

        base_history_path = tune_kwargs.get("history_path")
        base_artifact_dir = tune_kwargs.get("artifact_dir")
        common_tune_kwargs = dict(tune_kwargs)
        common_tune_kwargs.pop("history_path", None)
        common_tune_kwargs.pop("artifact_dir", None)

        rows: List[Dict[str, Any]] = []
        runs: Dict[int, MarsModelTuningResult] = {}
        for feature_count in resolved_steps:
            step_features = ordered_features[:feature_count]
            step_cats = [feature for feature in self.spec.categorical_features if feature in step_features]
            step_tuner = MarsModelTuner(
                model_type=self.spec.model_type,
                features=step_features,
                target=self.spec.target,
                dataset_flag_col=self.spec.dataset_flag_col,
                categorical_features=step_cats,
                optimize_metric=self.spec.optimize_metric,
                seed=self.spec.seed,
                lr_feature_mode=self.spec.lr_feature_mode,
                lr_binning_type=self.spec.lr_binning_type,
                lr_binner_kwargs=self.spec.lr_binner_kwargs,
                lr_binner=self.spec.lr_binner,
            )
            step_kwargs = dict(common_tune_kwargs)
            if base_artifact_dir is not None:
                step_kwargs["artifact_dir"] = step_artifact_dir(
                    base_artifact_dir,
                    feature_count,
                )
            if base_history_path is not None:
                step_kwargs["artifact_dir"] = str(
                    Path(self._step_history_path(base_history_path, feature_count)).with_suffix("")
                )
            try:
                # 每个 step 复用成熟的单次 tuner，避免增量实验绕开后端训练和 artifact 逻辑。
                run = step_tuner.tune(df, **step_kwargs)
            except Exception as exc:
                rows.append(self._error_row(feature_count, step_features, exc))
                continue
            runs[feature_count] = run
            rows.append(self._success_row(run, feature_count=feature_count, selection_metric=metric))

        summary_table = pd.DataFrame(rows)
        best_step, best_run = self._choose_best(summary_table, runs, metric)
        if not summary_table.empty:
            summary_table["is_best"] = False
            if best_step is not None:
                summary_table.loc[summary_table["feature_count"] == best_step, "is_best"] = True

        return MarsFeatureGrowthResult(
            model_type=self.spec.model_type,
            optimize_metric=self.spec.optimize_metric,
            feature_order=ordered_features,
            steps=resolved_steps,
            selection_metric=metric,
            summary_table=summary_table,
            runs=dict(runs),
            best_step=best_step,
            best_run=best_run,
            metadata={
                "mode": mode.lower(),
                "min_features": int(min_features),
                "max_features": max_features,
                "step_size": step_size,
                "selection_rule": f"max validation {metric}",
                # OOT 是稳定性审计口径，默认不参与反选，避免时间外样本被调参污染。
                "oot_used_for_selection": False,
            },
        )
