"""逐步增加特征的建模调参工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from mars.modeling.artifacts import read_json, write_json
from mars.modeling.results import MarsModelingRun
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
class MarsFeatureGrowthRun:
    """
    逐步增加特征调参的结构化结果。

    Parameters
    ----------
    model_type : str
        底层模型后端类型。
    optimize_metric : str
        单次 tuning 使用的优化指标。
    feature_order : list of str
        增量实验使用的稳定特征顺序。
    steps : list of int
        每轮使用的前 N 个特征数量。
    selection_metric : str
        用于从各 step 中挑选推荐模型的 validation 指标。
    summary_table : pandas.DataFrame
        step 级汇总审计表。
    runs : dict of int to MarsModelingRun
        成功完成的 step 对应的单次 tuning run。
    best_step : int, optional
        推荐特征数量。
    best_run : MarsModelingRun, optional
        推荐 step 对应的 tuning run。
    metadata : dict, optional
        额外审计元数据。
    """

    model_type: str
    optimize_metric: str
    feature_order: List[str]
    steps: List[int]
    selection_metric: str
    summary_table: pd.DataFrame
    runs: Dict[int, MarsModelingRun]
    best_step: Optional[int] = None
    best_run: Optional[MarsModelingRun] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_model(self) -> Any:
        """返回推荐 step 的最佳模型。"""
        return None if self.best_run is None else self.best_run.best_model

    @property
    def best_score(self) -> Optional[float]:
        """返回推荐 step 的 validation 分数。"""
        return None if self.best_run is None else float(self.best_run.best_score)

    @property
    def best_features(self) -> List[str]:
        """返回推荐 step 使用的特征列表。"""
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
            "artifact_type": "mars_feature_growth_run",
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
    def load_artifact(cls, path: str) -> "MarsFeatureGrowthRun":
        """
        从本地 artifact 目录恢复特征增长实验结果。

        Parameters
        ----------
        path : str
            artifact 目录。

        Returns
        -------
        MarsFeatureGrowthRun
            恢复后的实验结果。
        """
        artifact_dir = Path(path)
        metadata = read_json(artifact_dir / "metadata.json")
        if metadata.get("artifact_type") != "mars_feature_growth_run":
            raise ValueError(
                f"Unsupported artifact type for MarsFeatureGrowthRun: {metadata.get('artifact_type')!r}"
            )

        files = metadata.get("files", {})
        summary_path = artifact_dir / files.get("summary", "summary.csv")
        if not summary_path.exists():
            raise FileNotFoundError(f"Feature growth summary file is missing: {summary_path}")

        runs: Dict[int, MarsModelingRun] = {}
        for step_text, rel_dir in dict(files.get("runs", {})).items():
            step = int(step_text)
            runs[step] = MarsModelingRun.load_artifact(str(artifact_dir / rel_dir))

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
            best_run=runs.get(best_step),
            metadata=dict(metadata.get("metadata", {})),
        )


class MarsFeatureIncrementalTuner:
    """
    按特征数量逐步扩展的调参器。

    Parameters
    ----------
    model_type : str
        底层模型后端类型。
    features : sequence of str
        候选特征全集。
    target : str
        目标变量列名。
    dataset_flag_col : str, default "dataset_flag"
        数据集切片标识列。
    categorical_features : sequence of str, optional
        类别特征列名。
    optimize_metric : {"auc", "ks"}, default "ks"
        单次 tuning 的优化指标。
    seed : int, default 1206
        随机种子。
    benchmark_col : str, optional
        基准模型分数列。
    time_col : str, optional
        时间列。
    """

    def __init__(
        self,
        *,
        model_type: str,
        features: Sequence[str],
        target: str,
        dataset_flag_col: str = "dataset_flag",
        categorical_features: Optional[Sequence[str]] = None,
        optimize_metric: str = "ks",
        seed: int = 1206,
        benchmark_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> None:
        self.spec = _build_spec(
            model_type=model_type,
            features=features,
            target=target,
            dataset_flag_col=dataset_flag_col,
            categorical_features=categorical_features,
            optimize_metric=optimize_metric,
            seed=seed,
            benchmark_col=benchmark_col,
            time_col=time_col,
        )

    def _resolve_feature_order(
        self,
        *,
        feature_order: Optional[Sequence[str]],
        importance_table: Optional[pd.DataFrame],
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
        steps: Optional[Sequence[int]],
        min_features: int,
        max_features: Optional[int],
        step_size: Optional[int],
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
    def _step_save_path(base_path: str, feature_count: int) -> str:
        """为每个 step 生成互不覆盖的 history 路径。"""
        path = Path(base_path)
        suffix = path.suffix or ".csv"
        return str(path.with_name(f"{path.stem}_features_{feature_count}{suffix}"))

    @staticmethod
    def _select_history_row(history: pd.DataFrame, metric: str) -> Optional[pd.Series]:
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
        return valid.loc[scores.loc[valid.index].idxmax()]

    @staticmethod
    def _success_row(run: MarsModelingRun, *, feature_count: int, selection_metric: str) -> Dict[str, Any]:
        """构建成功 step 的轻量审计行。"""
        row: Dict[str, Any] = {
            "feature_count": int(feature_count),
            "status": "complete",
            "features": _json_dumps(list(run.features)),
            "best_score": float(run.best_score),
            "best_iteration": run.best_iteration,
            "backend_data_mode": run.backend_data_mode,
            "history_path": run.history_path,
            "model_type": run.model_type,
            "optimize_metric": run.optimize_metric,
            "error": None,
        }
        history_row = MarsFeatureIncrementalTuner._select_history_row(run.history_table, selection_metric)
        if history_row is None:
            row["selection_score"] = float(run.best_score)
            return row

        score_col = f"val_{selection_metric}"
        row["selection_score"] = pd.to_numeric(pd.Series([history_row.get(score_col)]), errors="coerce").iloc[0]
        for col in run.history_table.columns:
            col_name = str(col)
            if col_name in {"trial_num", "is_valid", "val_diff", "max_oot_diff"}:
                row[col_name] = history_row.get(col)
            elif col_name.endswith("_ks") or col_name.endswith("_auc"):
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
            "model_type": None,
            "optimize_metric": None,
            "error": str(exc)[:300],
        }

    @staticmethod
    def _choose_best(summary_table: pd.DataFrame, runs: Mapping[int, MarsModelingRun], metric: str) -> tuple[Optional[int], Optional[MarsModelingRun]]:
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
        best_idx = scores.loc[complete.index].idxmax()
        best_step = int(summary_table.loc[best_idx, "feature_count"])
        return best_step, runs.get(best_step)

    def tune(
        self,
        df: FrameLike,
        *,
        steps: Optional[Sequence[int]] = None,
        feature_order: Optional[Sequence[str]] = None,
        importance_table: Optional[pd.DataFrame] = None,
        min_features: int = 10,
        max_features: Optional[int] = None,
        step_size: Optional[int] = None,
        mode: str = "prefix",
        selection_metric: Optional[str] = None,
        **tune_kwargs: Any,
    ) -> MarsFeatureGrowthRun:
        """
        执行逐步增加特征的多轮调参。

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            已带 train/val/OOT 标识的建模样本。
        steps : sequence of int, optional
            显式指定每轮使用的前 N 个特征数量。
        feature_order : sequence of str, optional
            人工指定的稳定特征顺序。
        importance_table : pandas.DataFrame, optional
            包含 ``feature`` 以及可选 ``importance`` / ``rank`` 的特征重要性表。
        min_features : int, default 10
            自动生成 step 时的起始特征数。
        max_features : int, optional
            自动生成 step 时的最大特征数。
        step_size : int, optional
            自动生成 step 时的步长；默认使用 ``min_features``。
        mode : {"prefix"}, default "prefix"
            特征增长模式。当前版本只支持前缀扩展。
        selection_metric : {"auc", "ks"}, optional
            跨 step 选择推荐模型时使用的 validation 指标。
        **tune_kwargs : Any
            透传给 ``MarsModelTuner.tune`` 的参数。

        Returns
        -------
        MarsFeatureGrowthRun
            包含所有成功 step、汇总表和推荐模型的实验结果。
        """
        if mode.lower() != "prefix":
            raise ValueError("MarsFeatureIncrementalTuner currently supports only mode='prefix'.")
        metric = (selection_metric or self.spec.optimize_metric).lower()
        if metric not in {"auc", "ks"}:
            raise ValueError("selection_metric must be one of {'auc', 'ks'}.")

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

        base_save_path = str(tune_kwargs.get("save_path", "feature_growth_history.csv"))
        common_tune_kwargs = dict(tune_kwargs)
        common_tune_kwargs.pop("save_path", None)

        rows: List[Dict[str, Any]] = []
        runs: Dict[int, MarsModelingRun] = {}
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
                benchmark_col=self.spec.benchmark_col,
                time_col=self.spec.time_col,
            )
            step_kwargs = dict(common_tune_kwargs)
            step_kwargs["save_path"] = self._step_save_path(base_save_path, feature_count)
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

        return MarsFeatureGrowthRun(
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
