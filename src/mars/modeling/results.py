"""Reusable result objects for the MARS modeling workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import joblib
import pandas as pd

from mars.modeling.base import FrameLike

if TYPE_CHECKING:
    from mars.modeling.report import MarsModelingReport


def _to_json_safe(value: Any) -> Any:
    """Convert nested metadata into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(key): _to_json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(inner) for inner in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write metadata JSON using UTF-8 with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    """Read metadata JSON and validate that the file exists."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact metadata file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_report_tables(reports: Dict[str, "MarsModelingReport"], reports_dir: Path) -> Dict[str, str]:
    """Persist report summary tables and return a model-to-file mapping."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_files: Dict[str, str] = {}
    for model_name, report in reports.items():
        file_name = f"{model_name}.csv"
        report.summary_table.to_csv(reports_dir / file_name)
        report_files[model_name] = file_name
    return report_files


def _load_report_tables(reports_dir: Path, report_files: Dict[str, str]) -> Dict[str, "MarsModelingReport"]:
    """Rebuild report objects from persisted summary tables."""
    from mars.modeling.report import MarsModelingReport

    reports: Dict[str, MarsModelingReport] = {}
    for model_name, file_name in report_files.items():
        table_path = reports_dir / file_name
        if not table_path.exists():
            raise FileNotFoundError(f"Artifact report table is missing: {table_path}")
        summary_table = pd.read_csv(table_path, header=[0, 1], index_col=0)
        reports[model_name] = MarsModelingReport(summary_table, caption=f"Model Evaluation by [{summary_table.index.name}]")
    return reports


@dataclass(slots=True)
class MarsModelingRun:
    """Structured result object for one tuning workflow."""

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

    def write_artifact(self, path: str) -> Path:
        """Write the tuning result into a local artifact directory."""
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
            "files": {
                "history": history_path.name,
                "importance": importance_path.name,
                "best_model": str(Path("models") / model_path.name),
            },
        }
        _write_json(artifact_dir / "metadata.json", metadata)
        return artifact_dir

    @classmethod
    def load_artifact(cls, path: str) -> "MarsModelingRun":
        """Load a tuning result from a local artifact directory."""
        artifact_dir = Path(path)
        metadata = _read_json(artifact_dir / "metadata.json")
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
        )


@dataclass(slots=True)
class MarsReplayRun:
    """Structured result object for one Top-K replay workflow."""

    model_type: str
    ranking_table: pd.DataFrame
    leaderboard_table: pd.DataFrame
    models: Dict[str, Any]
    scored_df: Optional[FrameLike]
    reports: Dict[str, "MarsModelingReport"]
    importance_tables: Dict[str, pd.DataFrame]

    def write_artifact(self, path: str, include_scored_df: bool = False) -> Path:
        """Write the replay result into a local artifact directory."""
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

        report_files = _save_report_tables(self.reports, reports_dir)

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
        _write_json(artifact_dir / "metadata.json", metadata)
        return artifact_dir

    @classmethod
    def load_artifact(cls, path: str) -> "MarsReplayRun":
        """Load a replay result from a local artifact directory."""
        artifact_dir = Path(path)
        metadata = _read_json(artifact_dir / "metadata.json")
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

        reports = _load_report_tables(artifact_dir / "reports", files.get("reports", {}))

        return cls(
            model_type=metadata["model_type"],
            ranking_table=pd.read_csv(ranking_path),
            leaderboard_table=pd.read_csv(leaderboard_path),
            models=models,
            scored_df=scored_df,
            reports=reports,
            importance_tables=importance_tables,
        )
