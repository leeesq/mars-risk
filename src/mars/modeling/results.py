"""建模调参和 replay 的可持久化结果对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import joblib
import pandas as pd

from mars.modeling.utils import FrameLike
from mars.modeling.artifacts import load_report_tables, read_json, save_report_tables, write_json

if TYPE_CHECKING:
    from mars.modeling.report import MarsModelingReport



@dataclass(slots=True)
class MarsModelingRun:
    """
    单次调参流程的结构化结果对象。

    Attributes
    ----------
    best_model : Any
        验证集最优模型。
    history_table : pandas.DataFrame
        trial 级训练历史。
    importance_table : pandas.DataFrame
        特征重要性表。
    training_config : dict
        可复现训练配置。
    backend_data_mode : str
        实际后端数据通道。
    category_levels : dict
        类别特征稳定字典。
    """

    model_type: str
    optimize_metric: str
    features: List[str]
    target: str
    dataset_flag_col: str
    categorical_features: List[str]
    best_params: Dict[str, Any]
    best_iteration: Optional[int]
    best_model: Any
    best_score: float
    history_table: pd.DataFrame
    history_path: str
    study: Any
    replay_candidates: List[str]
    importance_table: pd.DataFrame
    training_config: Dict[str, Any] = field(default_factory=dict)
    library_versions: Dict[str, Any] = field(default_factory=dict)
    feature_schema: Dict[str, Any] = field(default_factory=dict)
    backend_data_mode: str = "unknown"
    category_levels: Dict[str, List[Any]] = field(default_factory=dict)

    def write_artifact(self, path: str) -> Path:
        """
        将调参结果写入本地 artifact 目录。

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
        models_dir = artifact_dir / "models"
        models_dir.mkdir(exist_ok=True)

        history_path = artifact_dir / "history.csv"
        importance_path = artifact_dir / "importance.csv"
        model_path = models_dir / "best_model.joblib"

        self.history_table.to_csv(history_path, index=False)
        self.importance_table.to_csv(importance_path, index=False)
        joblib.dump(self.best_model, model_path)

        metadata = {
            "artifact_type": "mars_modeling_run",
            "model_type": self.model_type,
            "optimize_metric": self.optimize_metric,
            "features": self.features,
            "target": self.target,
            "dataset_flag_col": self.dataset_flag_col,
            "categorical_features": self.categorical_features,
            "best_params": self.best_params,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "history_path": self.history_path,
            "replay_candidates": self.replay_candidates,
            "training_config": self.training_config,
            "library_versions": self.library_versions,
            "feature_schema": self.feature_schema,
            "backend_data_mode": self.backend_data_mode,
            "category_levels": self.category_levels,
            "files": {
                "history": history_path.name,
                "importance": importance_path.name,
                "best_model": str(Path("models") / model_path.name),
            },
        }
        write_json(artifact_dir / "metadata.json", metadata)
        return artifact_dir

    @classmethod
    def load_artifact(cls, path: str) -> "MarsModelingRun":
        """从本地 artifact 目录恢复调参结果。"""
        artifact_dir = Path(path)
        metadata = read_json(artifact_dir / "metadata.json")
        if metadata.get("artifact_type") != "mars_modeling_run":
            raise ValueError(f"Unsupported artifact type for MarsModelingRun: {metadata.get('artifact_type')!r}")

        files = metadata.get("files", {})
        history_path = artifact_dir / files.get("history", "history.csv")
        importance_path = artifact_dir / files.get("importance", "importance.csv")
        model_path = artifact_dir / files.get("best_model", str(Path("models") / "best_model.joblib"))

        if not history_path.exists():
            raise FileNotFoundError(f"Artifact history file is missing: {history_path}")
        if not importance_path.exists():
            raise FileNotFoundError(f"Artifact importance file is missing: {importance_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Artifact model file is missing: {model_path}")

        return cls(
            model_type=metadata["model_type"],
            optimize_metric=metadata["optimize_metric"],
            features=list(metadata["features"]),
            target=metadata["target"],
            dataset_flag_col=metadata["dataset_flag_col"],
            categorical_features=list(metadata.get("categorical_features", [])),
            best_params=dict(metadata.get("best_params", {})),
            best_iteration=metadata.get("best_iteration"),
            best_model=joblib.load(model_path),
            best_score=float(metadata["best_score"]),
            history_table=pd.read_csv(history_path),
            history_path=str(metadata.get("history_path", history_path.resolve())),
            study=None,
            replay_candidates=list(metadata.get("replay_candidates", [])),
            importance_table=pd.read_csv(importance_path),
            training_config=dict(metadata.get("training_config", {})),
            library_versions=dict(metadata.get("library_versions", {})),
            feature_schema=dict(metadata.get("feature_schema", {})),
            backend_data_mode=str(metadata.get("backend_data_mode", "unknown")),
            category_levels=dict(metadata.get("category_levels", {})),
        )


@dataclass(slots=True)
class MarsReplayRun:
    """
    Top-K replay 流程的结构化结果对象。

    Attributes
    ----------
    ranking_table : pandas.DataFrame
        用于选取 Top-K trial 的排名表。
    leaderboard_table : pandas.DataFrame
        replay 后的模型排行榜。
    models : dict
        replay 训练得到的模型对象。
    scored_df : pandas.DataFrame or polars.DataFrame, optional
        追加预测列后的数据。
    reports : dict
        每个 replay 模型对应的评估报告。
    """

    model_type: str
    ranking_table: pd.DataFrame
    leaderboard_table: pd.DataFrame
    models: Dict[str, Any]
    scored_df: Optional[FrameLike]
    reports: Dict[str, "MarsModelingReport"]
    importance_tables: Dict[str, pd.DataFrame]

    def write_artifact(self, path: str, include_scored_df: bool = False) -> Path:
        """
        将 replay 结果写入本地 artifact 目录。

        Parameters
        ----------
        path : str
            输出目录。
        include_scored_df : bool, default False
            是否保存评分后的数据框。

        Returns
        -------
        pathlib.Path
            artifact 目录路径。
        """
        artifact_dir = Path(path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        models_dir = artifact_dir / "models"
        models_dir.mkdir(exist_ok=True)
        importance_dir = artifact_dir / "importance_tables"
        importance_dir.mkdir(exist_ok=True)
        reports_dir = artifact_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        ranking_path = artifact_dir / "ranking.csv"
        leaderboard_path = artifact_dir / "leaderboard.csv"
        ranking_path.parent.mkdir(parents=True, exist_ok=True)
        self.ranking_table.to_csv(ranking_path, index=False)
        self.leaderboard_table.to_csv(leaderboard_path, index=False)

        model_files: Dict[str, str] = {}
        for model_name, model in self.models.items():
            file_name = f"{model_name}.joblib"
            joblib.dump(model, models_dir / file_name)
            model_files[model_name] = file_name

        importance_files: Dict[str, str] = {}
        for model_name, table in self.importance_tables.items():
            file_name = f"{model_name}.csv"
            table.to_csv(importance_dir / file_name, index=False)
            importance_files[model_name] = file_name

        report_files = save_report_tables(self.reports, reports_dir)

        scored_df_file: Optional[str] = None
        if include_scored_df and self.scored_df is not None:
            scored_df_file = "scored_df.parquet"
            if isinstance(self.scored_df, pd.DataFrame):
                self.scored_df.to_parquet(artifact_dir / scored_df_file, index=False)
            else:
                self.scored_df.to_pandas().to_parquet(artifact_dir / scored_df_file, index=False)

        metadata = {
            "artifact_type": "mars_replay_run",
            "model_type": self.model_type,
            "include_scored_df": bool(scored_df_file),
            "files": {
                "ranking": ranking_path.name,
                "leaderboard": leaderboard_path.name,
                "scored_df": scored_df_file,
                "models": model_files,
                "importance_tables": importance_files,
                "reports": report_files,
            },
        }
        write_json(artifact_dir / "metadata.json", metadata)
        return artifact_dir

    @classmethod
    def load_artifact(cls, path: str) -> "MarsReplayRun":
        """从本地 artifact 目录恢复 replay 结果。"""
        artifact_dir = Path(path)
        metadata = read_json(artifact_dir / "metadata.json")
        if metadata.get("artifact_type") != "mars_replay_run":
            raise ValueError(f"Unsupported artifact type for MarsReplayRun: {metadata.get('artifact_type')!r}")

        files = metadata.get("files", {})
        ranking_path = artifact_dir / files.get("ranking", "ranking.csv")
        leaderboard_path = artifact_dir / files.get("leaderboard", "leaderboard.csv")
        if not ranking_path.exists():
            raise FileNotFoundError(f"Artifact ranking file is missing: {ranking_path}")
        if not leaderboard_path.exists():
            raise FileNotFoundError(f"Artifact leaderboard file is missing: {leaderboard_path}")

        models: Dict[str, Any] = {}
        for model_name, file_name in files.get("models", {}).items():
            model_path = artifact_dir / "models" / file_name
            if not model_path.exists():
                raise FileNotFoundError(f"Artifact model file is missing: {model_path}")
            models[model_name] = joblib.load(model_path)

        importance_tables: Dict[str, pd.DataFrame] = {}
        for model_name, file_name in files.get("importance_tables", {}).items():
            table_path = artifact_dir / "importance_tables" / file_name
            if not table_path.exists():
                raise FileNotFoundError(f"Artifact importance table is missing: {table_path}")
            importance_tables[model_name] = pd.read_csv(table_path)

        scored_df: Optional[pd.DataFrame] = None
        scored_df_file = files.get("scored_df")
        if scored_df_file:
            scored_path = artifact_dir / scored_df_file
            if not scored_path.exists():
                raise FileNotFoundError(f"Artifact scored dataframe is missing: {scored_path}")
            scored_df = pd.read_parquet(scored_path)

        reports = load_report_tables(artifact_dir / "reports", files.get("reports", {}))

        return cls(
            model_type=metadata["model_type"],
            ranking_table=pd.read_csv(ranking_path),
            leaderboard_table=pd.read_csv(leaderboard_path),
            models=models,
            scored_df=scored_df,
            reports=reports,
            importance_tables=importance_tables,
        )
