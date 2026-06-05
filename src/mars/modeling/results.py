"""建模调参和 replay 的可持久化结果对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping

import joblib
import pandas as pd

from mars.modeling.artifacts import load_report_tables, read_json, save_report_tables, write_json
from mars.modeling.utils import FrameLike

if TYPE_CHECKING:
    from mars.modeling.report import MarsModelingReport


CSV_FLOAT_FORMAT: str = "%.17g"


def _dataframe_schema(df: pd.DataFrame) -> Dict[str, str]:
    """记录 DataFrame 列 dtype，供 CSV artifact 读回后恢复类型。"""
    return {str(column): str(dtype) for column, dtype in df.dtypes.items()}


def _restore_dataframe_schema(df: pd.DataFrame, schema: Mapping[str, Any] | None) -> pd.DataFrame:
    """按 artifact metadata 中记录的 dtype 恢复 DataFrame。"""
    if not schema:
        return df

    restored = df.copy()
    for column, dtype_value in schema.items():
        if column not in restored.columns:
            continue
        dtype_name = str(dtype_value)
        try:
            if dtype_name.startswith("datetime64"):
                restored[column] = pd.to_datetime(restored[column])
            elif dtype_name == "category":
                restored[column] = restored[column].astype("category")
            else:
                restored[column] = restored[column].astype(dtype_name)
        except (TypeError, ValueError):
            continue
    return restored


def _read_artifact_csv(path: Path, schema: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """读取 artifact CSV，并保持浮点值的 round-trip 精度。"""
    table = pd.read_csv(path, float_precision="round_trip")
    return _restore_dataframe_schema(table, schema)


@dataclass(slots=True)
class MarsModelTuningResult:
    """
    单次调参流程的结构化结果对象。

    Parameters
    ----------
    model_type : str
        模型后端类型。
    optimize_metric : str
        调参优化指标。
    features : list of str
        参与训练的特征列名。
    target : str
        目标变量列名。
    dataset_flag_col : str
        数据集切分标识列名。
    categorical_features : list of str
        类别特征列名。
    best_params : dict
        最优 Trial 参数。
    best_iteration : int, optional
        最优迭代轮次。
    best_model : Any
        验证集最优模型。
    best_score : float
        验证集最优分数。
    history_table : pandas.DataFrame
        Trial 级训练历史。
    history_path : str
        训练历史 CSV 路径。
    study : Any
        Optuna study 对象或兼容占位。
    replay_candidates : list of str
        推荐进入 replay 的 trial 标识。
    importance_table : pandas.DataFrame
        特征重要性表。

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
        ...     artifact_dir = run.write_artifact(tmp)
        ...     (artifact_dir / "metadata.json").exists()
        True
        """
        artifact_dir = Path(path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        models_dir = artifact_dir / "models"
        models_dir.mkdir(exist_ok=True)
        diagnostics_dir = artifact_dir / "diagnostics"
        diagnostics_dir.mkdir(exist_ok=True)

        history_path = artifact_dir / "history.csv"
        importance_path = artifact_dir / "importance.csv"
        model_path = models_dir / "best_model.joblib"

        self.history_table.to_csv(history_path, index=False, float_format=CSV_FLOAT_FORMAT)
        self.importance_table.to_csv(importance_path, index=False, float_format=CSV_FLOAT_FORMAT)
        joblib.dump(self.best_model, model_path)

        diagnostic_files: Dict[str, str] = {}
        diagnostic_schemas: Dict[str, Dict[str, str]] = {}
        for table_name, table in self.diagnostic_tables.items():
            file_name = f"{table_name}.csv"
            table.to_csv(diagnostics_dir / file_name, index=False, float_format=CSV_FLOAT_FORMAT)
            diagnostic_files[table_name] = file_name
            diagnostic_schemas[table_name] = _dataframe_schema(table)

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
            "table_schemas": {
                "history": _dataframe_schema(self.history_table),
                "importance": _dataframe_schema(self.importance_table),
                "diagnostics": diagnostic_schemas,
            },
            "files": {
                "history": history_path.name,
                "importance": importance_path.name,
                "best_model": str(Path("models") / model_path.name),
                "diagnostics": diagnostic_files,
            },
        }
        write_json(artifact_dir / "metadata.json", metadata)
        return artifact_dir

    @classmethod
    def load_artifact(cls: type[MarsModelTuningResult], path: str) -> MarsModelTuningResult:
        """
        从本地 artifact 目录恢复调参结果。

        Parameters
        ----------
        path : str
            由 ``write_artifact`` 生成的 artifact 目录。

        Returns
        -------
        MarsModelTuningResult
            恢复后的单次调参结果。

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
        ...     _ = run.write_artifact(tmp)
        ...     MarsModelTuningResult.load_artifact(tmp).features
        ['age']
        """
        artifact_dir = Path(path)
        metadata = read_json(artifact_dir / "metadata.json")
        if metadata.get("artifact_type") != "mars_model_tuning_result":
            raise ValueError(f"Unsupported artifact type for MarsModelTuningResult: {metadata.get('artifact_type')!r}")

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

        diagnostic_tables: Dict[str, pd.DataFrame] = {}
        table_schemas = dict(metadata.get("table_schemas", {}))
        diagnostic_schemas = dict(table_schemas.get("diagnostics", {}))
        for table_name, file_name in dict(files.get("diagnostics", {})).items():
            table_path = artifact_dir / "diagnostics" / file_name
            if not table_path.exists():
                raise FileNotFoundError(f"Artifact diagnostic table is missing: {table_path}")
            diagnostic_tables[table_name] = _read_artifact_csv(
                table_path,
                dict(diagnostic_schemas.get(table_name, {})),
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
            history_table=_read_artifact_csv(
                history_path,
                dict(table_schemas.get("history", {})),
            ),
            history_path=str(metadata.get("history_path", history_path.resolve())),
            study=None,
            replay_candidates=list(metadata.get("replay_candidates", [])),
            importance_table=_read_artifact_csv(
                importance_path,
                dict(table_schemas.get("importance", {})),
            ),
            diagnostic_tables=diagnostic_tables,
            training_config=dict(metadata.get("training_config", {})),
            library_versions=dict(metadata.get("library_versions", {})),
            feature_schema=dict(metadata.get("feature_schema", {})),
            backend_data_mode=str(metadata.get("backend_data_mode", "unknown")),
            category_levels=dict(metadata.get("category_levels", {})),
        )


@dataclass(slots=True)
class MarsModelReplayResult:
    """
    Top-K replay 流程的结构化结果对象。

    Parameters
    ----------
    model_type : str
        模型后端类型。
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
    importance_tables : dict
        每个 replay 模型对应的特征重要性表。

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

    Examples
    --------
    >>> replay = MarsModelReplayResult(
    ...     model_type="xgb",
    ...     ranking_table=pd.DataFrame(),
    ...     leaderboard_table=pd.DataFrame(),
    ...     models={},
    ...     scored_df=None,
    ...     reports={},
    ...     importance_tables={},
    ... )
    >>> replay.models
    {}
    """

    model_type: str
    ranking_table: pd.DataFrame
    leaderboard_table: pd.DataFrame
    models: Dict[str, Any]
    scored_df: FrameLike | None
    reports: Dict[str, MarsModelingReport]
    importance_tables: Dict[str, pd.DataFrame]
    diagnostic_tables: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)

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

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> replay = MarsModelReplayResult(
        ...     model_type="xgb",
        ...     ranking_table=pd.DataFrame(),
        ...     leaderboard_table=pd.DataFrame(),
        ...     models={},
        ...     scored_df=None,
        ...     reports={},
        ...     importance_tables={},
        ... )
        >>> with TemporaryDirectory() as tmp:
        ...     artifact_dir = replay.write_artifact(tmp)
        ...     (artifact_dir / "metadata.json").exists()
        True
        """
        artifact_dir = Path(path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        models_dir = artifact_dir / "models"
        models_dir.mkdir(exist_ok=True)
        importance_dir = artifact_dir / "importance_tables"
        importance_dir.mkdir(exist_ok=True)
        diagnostics_dir = artifact_dir / "diagnostics"
        diagnostics_dir.mkdir(exist_ok=True)
        reports_dir = artifact_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        ranking_path = artifact_dir / "ranking.csv"
        leaderboard_path = artifact_dir / "leaderboard.csv"
        ranking_path.parent.mkdir(parents=True, exist_ok=True)
        self.ranking_table.to_csv(ranking_path, index=False, float_format=CSV_FLOAT_FORMAT)
        self.leaderboard_table.to_csv(leaderboard_path, index=False, float_format=CSV_FLOAT_FORMAT)

        model_files: Dict[str, str] = {}
        for model_name, model in self.models.items():
            file_name = f"{model_name}.joblib"
            joblib.dump(model, models_dir / file_name)
            model_files[model_name] = file_name

        importance_files: Dict[str, str] = {}
        importance_schemas: Dict[str, Dict[str, str]] = {}
        for model_name, table in self.importance_tables.items():
            file_name = f"{model_name}.csv"
            table.to_csv(importance_dir / file_name, index=False, float_format=CSV_FLOAT_FORMAT)
            importance_files[model_name] = file_name
            importance_schemas[model_name] = _dataframe_schema(table)

        diagnostic_files: Dict[str, Dict[str, str]] = {}
        diagnostic_schemas: Dict[str, Dict[str, Dict[str, str]]] = {}
        for model_name, tables in self.diagnostic_tables.items():
            model_dir = diagnostics_dir / model_name
            model_dir.mkdir(exist_ok=True)
            table_files: Dict[str, str] = {}
            table_schemas: Dict[str, Dict[str, str]] = {}
            for table_name, table in tables.items():
                file_name = f"{table_name}.csv"
                table.to_csv(model_dir / file_name, index=False, float_format=CSV_FLOAT_FORMAT)
                table_files[table_name] = file_name
                table_schemas[table_name] = _dataframe_schema(table)
            diagnostic_files[model_name] = table_files
            diagnostic_schemas[model_name] = table_schemas

        report_files = save_report_tables(self.reports, reports_dir)

        scored_df_file: str | None = None
        if include_scored_df and self.scored_df is not None:
            scored_df_file = "scored_df.parquet"
            if isinstance(self.scored_df, pd.DataFrame):
                self.scored_df.to_parquet(artifact_dir / scored_df_file, index=False)
            else:
                self.scored_df.to_pandas().to_parquet(artifact_dir / scored_df_file, index=False)

        metadata = {
            "artifact_type": "mars_model_replay_result",
            "model_type": self.model_type,
            "include_scored_df": bool(scored_df_file),
            "table_schemas": {
                "ranking": _dataframe_schema(self.ranking_table),
                "leaderboard": _dataframe_schema(self.leaderboard_table),
                "importance_tables": importance_schemas,
                "diagnostics": diagnostic_schemas,
            },
            "files": {
                "ranking": ranking_path.name,
                "leaderboard": leaderboard_path.name,
                "scored_df": scored_df_file,
                "models": model_files,
                "importance_tables": importance_files,
                "diagnostics": diagnostic_files,
                "reports": report_files,
            },
        }
        write_json(artifact_dir / "metadata.json", metadata)
        return artifact_dir

    @classmethod
    def load_artifact(cls: type[MarsModelReplayResult], path: str) -> MarsModelReplayResult:
        """
        从本地 artifact 目录恢复 replay 结果。

        Parameters
        ----------
        path : str
            由 ``write_artifact`` 生成的 replay artifact 目录。

        Returns
        -------
        MarsModelReplayResult
            恢复后的 replay 结果。

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> replay = MarsModelReplayResult(
        ...     model_type="xgb",
        ...     ranking_table=pd.DataFrame(),
        ...     leaderboard_table=pd.DataFrame(),
        ...     models={},
        ...     scored_df=None,
        ...     reports={},
        ...     importance_tables={},
        ... )
        >>> with TemporaryDirectory() as tmp:
        ...     _ = replay.write_artifact(tmp)
        ...     MarsModelReplayResult.load_artifact(tmp).model_type
        'xgb'
        """
        artifact_dir = Path(path)
        metadata = read_json(artifact_dir / "metadata.json")
        if metadata.get("artifact_type") != "mars_model_replay_result":
            raise ValueError(f"Unsupported artifact type for MarsModelReplayResult: {metadata.get('artifact_type')!r}")

        files = metadata.get("files", {})
        ranking_path = artifact_dir / files.get("ranking", "ranking.csv")
        leaderboard_path = artifact_dir / files.get("leaderboard", "leaderboard.csv")
        if not ranking_path.exists():
            raise FileNotFoundError(f"Artifact ranking file is missing: {ranking_path}")
        if not leaderboard_path.exists():
            raise FileNotFoundError(f"Artifact leaderboard file is missing: {leaderboard_path}")

        table_schemas = dict(metadata.get("table_schemas", {}))

        models: Dict[str, Any] = {}
        for model_name, file_name in files.get("models", {}).items():
            model_path = artifact_dir / "models" / file_name
            if not model_path.exists():
                raise FileNotFoundError(f"Artifact model file is missing: {model_path}")
            models[model_name] = joblib.load(model_path)

        importance_tables: Dict[str, pd.DataFrame] = {}
        importance_schemas = dict(table_schemas.get("importance_tables", {}))
        for model_name, file_name in files.get("importance_tables", {}).items():
            table_path = artifact_dir / "importance_tables" / file_name
            if not table_path.exists():
                raise FileNotFoundError(f"Artifact importance table is missing: {table_path}")
            importance_tables[model_name] = _read_artifact_csv(
                table_path,
                dict(importance_schemas.get(model_name, {})),
            )

        diagnostic_tables: Dict[str, Dict[str, pd.DataFrame]] = {}
        diagnostic_schemas = dict(table_schemas.get("diagnostics", {}))
        for model_name, table_files in dict(files.get("diagnostics", {})).items():
            model_tables: Dict[str, pd.DataFrame] = {}
            model_schemas = dict(diagnostic_schemas.get(model_name, {}))
            for table_name, file_name in dict(table_files).items():
                table_path = artifact_dir / "diagnostics" / model_name / file_name
                if not table_path.exists():
                    raise FileNotFoundError(f"Artifact diagnostic table is missing: {table_path}")
                model_tables[table_name] = _read_artifact_csv(
                    table_path,
                    dict(model_schemas.get(table_name, {})),
                )
            diagnostic_tables[model_name] = model_tables

        scored_df: pd.DataFrame | None = None
        scored_df_file = files.get("scored_df")
        if scored_df_file:
            scored_path = artifact_dir / scored_df_file
            if not scored_path.exists():
                raise FileNotFoundError(f"Artifact scored dataframe is missing: {scored_path}")
            scored_df = pd.read_parquet(scored_path)

        reports = load_report_tables(artifact_dir / "reports", files.get("reports", {}))

        return cls(
            model_type=metadata["model_type"],
            ranking_table=_read_artifact_csv(
                ranking_path,
                dict(table_schemas.get("ranking", {})),
            ),
            leaderboard_table=_read_artifact_csv(
                leaderboard_path,
                dict(table_schemas.get("leaderboard", {})),
            ),
            models=models,
            scored_df=scored_df,
            reports=reports,
            importance_tables=importance_tables,
            diagnostic_tables=diagnostic_tables,
        )
