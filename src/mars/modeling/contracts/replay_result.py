"""replay 流程的结构化结果对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import joblib
import pandas as pd

from mars.compute import FrameLike
from mars.modeling.artifacts import load_report_tables, read_json, save_report_tables, write_json
from mars.modeling.contracts._artifact_frames import (
    CSV_FLOAT_FORMAT,
    dataframe_schema,
    read_artifact_csv,
)

if TYPE_CHECKING:
    from mars.modeling.contracts.report import MarsModelingReport


@dataclass(slots=True)
class MarsModelReplayResult:
    """
    调参 replay 流程的结构化结果对象。

    Attributes
    ----------
    ranking_table : pandas.DataFrame
        用于选取 Top-K 或指定 trial 的排名表。
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

    def export_artifact(self, path: str, include_scored_df: bool = False) -> Path:
        """
        将 replay 结果写入本地 artifact 目录。

        Parameters
        ----------
        path : str
            输出目录。
        include_scored_df : bool
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
        ...     artifact_dir = replay.export_artifact(tmp)
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
            importance_schemas[model_name] = dataframe_schema(table)

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
                table_schemas[table_name] = dataframe_schema(table)
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
                "ranking": dataframe_schema(self.ranking_table),
                "leaderboard": dataframe_schema(self.leaderboard_table),
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
    def from_artifact(
        cls: type[MarsModelReplayResult],
        path: str,
    ) -> MarsModelReplayResult:
        """
        从本地 artifact 目录恢复 replay 结果。

        Parameters
        ----------
        path : str
            由 ``export_artifact`` 生成的 replay artifact 目录。

        Returns
        -------
        MarsModelReplayResult
            恢复后的 replay 结果。

        Raises
        ------
        FileNotFoundError
            当指定路径不存在时抛出。
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

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
        ...     _ = replay.export_artifact(tmp)
        ...     MarsModelReplayResult.from_artifact(tmp).model_type
        'xgb'
        """
        artifact_dir = Path(path)
        metadata = read_json(artifact_dir / "metadata.json")
        if metadata.get("artifact_type") != "mars_model_replay_result":
            raise ValueError(
                "Unsupported artifact type for MarsModelReplayResult: "
                f"{metadata.get('artifact_type')!r}"
            )

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
            importance_tables[model_name] = read_artifact_csv(
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
                model_tables[table_name] = read_artifact_csv(
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
            ranking_table=read_artifact_csv(
                ranking_path,
                dict(table_schemas.get("ranking", {})),
            ),
            leaderboard_table=read_artifact_csv(
                leaderboard_path,
                dict(table_schemas.get("leaderboard", {})),
            ),
            models=models,
            scored_df=scored_df,
            reports=reports,
            importance_tables=importance_tables,
            diagnostic_tables=diagnostic_tables,
        )
