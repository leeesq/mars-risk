"""逐步增加特征实验的结构化结果对象。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from mars.modeling.artifacts import read_json, write_json
from mars.modeling.contracts.tuning_result import MarsModelTuningResult


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

    def export_artifact(self, path: str) -> Path:
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
        ...     artifact_dir = run.export_artifact(tmp)
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
            run.export_artifact(str(artifact_dir / rel_dir))
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
    def from_artifact(
        cls: type[MarsFeatureGrowthResult],
        path: str,
    ) -> MarsFeatureGrowthResult:
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
        ...     _ = run.export_artifact(tmp)
        ...     MarsFeatureGrowthResult.from_artifact(tmp).feature_order
        ['age']
        """
        artifact_dir = Path(path)
        metadata = read_json(artifact_dir / "metadata.json")
        if metadata.get("artifact_type") != "mars_feature_growth_result":
            raise ValueError(
                "Unsupported artifact type for MarsFeatureGrowthResult: "
                f"{metadata.get('artifact_type')!r}"
            )

        files = metadata.get("files", {})
        summary_path = artifact_dir / files.get("summary", "summary.csv")
        if not summary_path.exists():
            raise FileNotFoundError(f"Feature growth summary file is missing: {summary_path}")

        runs: Dict[int, MarsModelTuningResult] = {}
        for step_text, rel_dir in dict(files.get("runs", {})).items():
            step = int(step_text)
            runs[step] = MarsModelTuningResult.from_artifact(str(artifact_dir / rel_dir))

        best_step = metadata.get("best_step")
        if best_step is not None:
            best_step = int(best_step)
        return cls(
            model_type=str(metadata["model_type"]),
            optimize_metric=str(metadata["optimize_metric"]),
            feature_order=list(metadata.get("feature_order", [])),
            steps=[int(step) for step in metadata.get("steps", [])],
            selection_metric=str(
                metadata.get("selection_metric", metadata.get("optimize_metric", "ks"))
            ),
            summary_table=pd.read_csv(summary_path),
            runs=runs,
            best_step=best_step,
            best_run=runs.get(best_step) if best_step is not None else None,
            metadata=dict(metadata.get("metadata", {})),
        )
