"""单次调参流程的结构化结果对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from mars.modeling.artifacts import read_json, write_json
from mars.modeling.contracts._artifact_frames import (
    CSV_FLOAT_FORMAT,
    dataframe_schema,
    read_artifact_csv,
)


@dataclass(slots=True)
class MarsModelTuningResult:
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
    retained_models : dict
        调参过程中动态保留的 trial 模型。
    retained_model_table : pandas.DataFrame
        已保留模型的 trial 编号、分数和排名。
    artifact_path : str or None
        本次调参产物目录；不落盘时为 ``None``。
    run_id : str or None
        本次调参运行编号。
    metric_names : list of str
        本次调参计算的内置和自定义指标名。
    metric_directions : dict
        各指标的排序方向。
    importance_tables : dict of str to pandas.DataFrame
        native、SHAP 等多来源重要性表。
    metadata : dict
        调参过程元信息和 artifact 元数据。
    training_config : dict
        可复现训练配置。
    backend_data_mode : str
        实际后端数据通道。
    category_levels : dict
        类别特征稳定字典。

    Examples
    --------
    >>> run = MarsModelTuningResult(
    ...     model_type="xgb",
    ...     optimize_metric="ks",
    ...     features=["age"],
    ...     target="y",
    ...     dataset_flag_col="dataset_flag",
    ...     categorical_features=[],
    ...     best_params={},
    ...     best_iteration=None,
    ...     best_model=None,
    ...     best_score=0.0,
    ...     history_table=pd.DataFrame(),
    ...     history_path="history.csv",
    ...     study=None,
    ...     replay_candidates=[],
    ...     importance_table=pd.DataFrame(),
    ... )
    >>> run.features
    ['age']
    """

    model_type: str
    optimize_metric: str
    features: List[str]
    target: str
    dataset_flag_col: str
    categorical_features: List[str]
    best_params: Dict[str, Any]
    best_iteration: int | None
    best_model: Any
    best_score: float
    history_table: pd.DataFrame
    history_path: str | None
    study: Any
    replay_candidates: List[str]
    importance_table: pd.DataFrame
    diagnostic_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    library_versions: Dict[str, Any] = field(default_factory=dict)
    feature_schema: Dict[str, Any] = field(default_factory=dict)
    backend_data_mode: str = "unknown"
    category_levels: Dict[str, List[Any]] = field(default_factory=dict)
    retained_models: Dict[int, Any] = field(default_factory=dict)
    retained_model_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifact_path: str | None = None
    run_id: str | None = None
    metric_names: List[str] = field(default_factory=lambda: ["auc", "ks", "f1"])
    metric_directions: Dict[str, str] = field(default_factory=dict)
    importance_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def export_artifact(self, path: str) -> Path:
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

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> run = MarsModelTuningResult(
        ...     model_type="xgb",
        ...     optimize_metric="ks",
        ...     features=["age"],
        ...     target="y",
        ...     dataset_flag_col="dataset_flag",
        ...     categorical_features=[],
        ...     best_params={},
        ...     best_iteration=None,
        ...     best_model=None,
        ...     best_score=0.0,
        ...     history_table=pd.DataFrame(),
        ...     history_path="history.csv",
        ...     study=None,
        ...     replay_candidates=[],
        ...     importance_table=pd.DataFrame(),
        ... )
        >>> with TemporaryDirectory() as tmp:
        ...     artifact_dir = run.export_artifact(tmp)
        ...     (artifact_dir / "metadata.json").exists()
        True
        """
        artifact_dir = Path(path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        models_dir = artifact_dir / "models"
        models_dir.mkdir(exist_ok=True)
        retained_models_dir = artifact_dir / "retained_models"
        retained_models_dir.mkdir(exist_ok=True)
        importance_tables_dir = artifact_dir / "importance_tables"
        importance_tables_dir.mkdir(exist_ok=True)
        diagnostics_dir = artifact_dir / "diagnostics"
        diagnostics_dir.mkdir(exist_ok=True)

        history_path = artifact_dir / "history.csv"
        importance_path = artifact_dir / "importance.csv"
        retained_model_table_path = artifact_dir / "retained_models.csv"
        model_path = models_dir / "best_model.joblib"

        self.history_table.to_csv(history_path, index=False, float_format=CSV_FLOAT_FORMAT)
        self.importance_table.to_csv(importance_path, index=False, float_format=CSV_FLOAT_FORMAT)
        self.retained_model_table.to_csv(
            retained_model_table_path,
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        joblib.dump(self.best_model, model_path)

        retained_model_files: Dict[str, str] = {}
        for trial_num, model in self.retained_models.items():
            file_name = f"trial_{int(trial_num)}.joblib"
            joblib.dump(model, retained_models_dir / file_name)
            retained_model_files[str(int(trial_num))] = file_name

        importance_table_files: Dict[str, str] = {}
        importance_table_schemas: Dict[str, Dict[str, str]] = {}
        active_importance_tables = dict(self.importance_tables)
        if not active_importance_tables:
            active_importance_tables["primary"] = self.importance_table
        for table_name, table in active_importance_tables.items():
            file_name = f"{table_name}.csv"
            table.to_csv(
                importance_tables_dir / file_name,
                index=False,
                float_format=CSV_FLOAT_FORMAT,
            )
            importance_table_files[table_name] = file_name
            importance_table_schemas[table_name] = dataframe_schema(table)

        diagnostic_files: Dict[str, str] = {}
        diagnostic_schemas: Dict[str, Dict[str, str]] = {}
        for table_name, table in self.diagnostic_tables.items():
            file_name = f"{table_name}.csv"
            table.to_csv(
                diagnostics_dir / file_name,
                index=False,
                float_format=CSV_FLOAT_FORMAT,
            )
            diagnostic_files[table_name] = file_name
            diagnostic_schemas[table_name] = dataframe_schema(table)

        metadata = {
            "artifact_type": "mars_model_tuning_result",
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
            "artifact_path": self.artifact_path,
            "run_id": self.run_id,
            "metric_names": self.metric_names,
            "metric_directions": self.metric_directions,
            "metadata": self.metadata,
            "table_schemas": {
                "history": dataframe_schema(self.history_table),
                "importance": dataframe_schema(self.importance_table),
                "retained_models": dataframe_schema(self.retained_model_table),
                "importance_tables": importance_table_schemas,
                "diagnostics": diagnostic_schemas,
            },
            "files": {
                "history": history_path.name,
                "importance": importance_path.name,
                "retained_model_table": retained_model_table_path.name,
                "best_model": str(Path("models") / model_path.name),
                "retained_models": retained_model_files,
                "importance_tables": importance_table_files,
                "diagnostics": diagnostic_files,
            },
        }
        write_json(artifact_dir / "metadata.json", metadata)
        return artifact_dir

    @classmethod
    def from_artifact(
        cls: type[MarsModelTuningResult],
        path: str,
    ) -> MarsModelTuningResult:
        """
        从本地 artifact 目录恢复调参结果。

        Parameters
        ----------
        path : str
            由 ``export_artifact`` 生成的 artifact 目录。

        Returns
        -------
        MarsModelTuningResult
            恢复后的单次调参结果。

        Raises
        ------
        FileNotFoundError
            当指定路径不存在时抛出。
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> run = MarsModelTuningResult(
        ...     model_type="xgb",
        ...     optimize_metric="ks",
        ...     features=["age"],
        ...     target="y",
        ...     dataset_flag_col="dataset_flag",
        ...     categorical_features=[],
        ...     best_params={},
        ...     best_iteration=None,
        ...     best_model=None,
        ...     best_score=0.0,
        ...     history_table=pd.DataFrame(),
        ...     history_path="history.csv",
        ...     study=None,
        ...     replay_candidates=[],
        ...     importance_table=pd.DataFrame(),
        ... )
        >>> with TemporaryDirectory() as tmp:
        ...     _ = run.export_artifact(tmp)
        ...     MarsModelTuningResult.from_artifact(tmp).features
        ['age']
        """
        artifact_dir = Path(path)
        metadata = read_json(artifact_dir / "metadata.json")
        if metadata.get("artifact_type") != "mars_model_tuning_result":
            raise ValueError(
                "Unsupported artifact type for MarsModelTuningResult: "
                f"{metadata.get('artifact_type')!r}"
            )

        files = metadata.get("files", {})
        history_path = artifact_dir / files.get("history", "history.csv")
        importance_path = artifact_dir / files.get("importance", "importance.csv")
        retained_model_table_path = artifact_dir / files.get("retained_model_table", "retained_models.csv")
        model_path = artifact_dir / files.get("best_model", str(Path("models") / "best_model.joblib"))

        if not history_path.exists():
            raise FileNotFoundError(f"Artifact history file is missing: {history_path}")
        if not importance_path.exists():
            raise FileNotFoundError(f"Artifact importance file is missing: {importance_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Artifact model file is missing: {model_path}")

        diagnostic_tables: Dict[str, pd.DataFrame] = {}
        table_schemas = dict(metadata.get("table_schemas", {}))
        diagnostic_schemas = dict(table_schemas.get("diagnostics", {}))
        for table_name, file_name in dict(files.get("diagnostics", {})).items():
            table_path = artifact_dir / "diagnostics" / file_name
            if not table_path.exists():
                raise FileNotFoundError(f"Artifact diagnostic table is missing: {table_path}")
            diagnostic_tables[table_name] = read_artifact_csv(
                table_path,
                dict(diagnostic_schemas.get(table_name, {})),
            )

        retained_model_table = pd.DataFrame()
        if retained_model_table_path.exists():
            retained_model_table = read_artifact_csv(
                retained_model_table_path,
                dict(table_schemas.get("retained_models", {})),
            )

        retained_models: Dict[int, Any] = {}
        for trial_num, file_name in dict(files.get("retained_models", {})).items():
            retained_model_path = artifact_dir / "retained_models" / file_name
            if not retained_model_path.exists():
                raise FileNotFoundError(
                    f"Artifact retained model file is missing: {retained_model_path}"
                )
            retained_models[int(trial_num)] = joblib.load(retained_model_path)

        importance_tables: Dict[str, pd.DataFrame] = {}
        importance_table_schemas = dict(table_schemas.get("importance_tables", {}))
        for table_name, file_name in dict(files.get("importance_tables", {})).items():
            table_path = artifact_dir / "importance_tables" / file_name
            if not table_path.exists():
                raise FileNotFoundError(f"Artifact importance table is missing: {table_path}")
            importance_tables[table_name] = read_artifact_csv(
                table_path,
                dict(importance_table_schemas.get(table_name, {})),
            )

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
            history_table=read_artifact_csv(
                history_path,
                dict(table_schemas.get("history", {})),
            ),
            history_path=str(metadata.get("history_path", history_path.resolve())),
            study=None,
            replay_candidates=list(metadata.get("replay_candidates", [])),
            importance_table=read_artifact_csv(
                importance_path,
                dict(table_schemas.get("importance", {})),
            ),
            diagnostic_tables=diagnostic_tables,
            training_config=dict(metadata.get("training_config", {})),
            library_versions=dict(metadata.get("library_versions", {})),
            feature_schema=dict(metadata.get("feature_schema", {})),
            backend_data_mode=str(metadata.get("backend_data_mode", "unknown")),
            category_levels=dict(metadata.get("category_levels", {})),
            retained_models=retained_models,
            retained_model_table=retained_model_table,
            artifact_path=metadata.get("artifact_path"),
            run_id=metadata.get("run_id"),
            metric_names=list(metadata.get("metric_names", ["auc", "ks", "f1"])),
            metric_directions=dict(metadata.get("metric_directions", {})),
            importance_tables=importance_tables,
            metadata=dict(metadata.get("metadata", {})),
        )
