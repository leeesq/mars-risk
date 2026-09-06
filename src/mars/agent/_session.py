"""保存会话数据快照、完整消息轮次和聚合报告，不自动读写文件。"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import polars as pl

from mars.compute import FrameLike, to_polars_frame

from ._contracts import MarsAgentMessage, MarsAgentReport


@dataclass(frozen=True)
class _Dataset:
    """调用方明确登记的数据快照与字段角色。"""

    id: str
    frame: pl.DataFrame = field(repr=False)
    features: tuple[str, ...]
    target: str | None
    group_columns: tuple[str, ...]
    time_col: str | None
    description: str
    missing_values: tuple[int | float | str, ...]

    def describe(self) -> dict[str, Any]:
        """仅返回字段角色和规模，不提供原始样本。"""
        return {
            "dataset_id": self.id,
            "description": self.description,
            "row_count": self.frame.height,
            "columns": {key: str(value) for key, value in self.frame.schema.items()},
            "features": list(self.features),
            "target": self.target,
            "target_present": (
                self.target in self.frame.columns if self.target else False
            ),
            "group_columns": list(self.group_columns),
            "time_col": self.time_col,
            "missing_values": list(self.missing_values),
        }


class MarsAgentSession:
    """
    显式持有跨轮消息、已登记数据及报告的会话。

    Notes
    -----
    数据登记时只保存允许使用的列。标识不可覆盖，避免旧报告和新数据混淆。
    同一会话不支持并发运行；不同会话彼此隔离。进程退出后状态不保留。
    """

    def __init__(self) -> None:
        self._datasets: dict[str, _Dataset] = {}
        self._reports: dict[str, MarsAgentReport] = {}
        self._messages: list[MarsAgentMessage] = []
        self._lock = Lock()

    def register_dataset(
        self,
        dataset_id: str,
        df: FrameLike,
        *,
        features: list[str],
        target: str | None = None,
        group_columns: list[str] | None = None,
        time_col: str | None = None,
        description: str = "",
        missing_values: list[int | float | str] | None = None,
    ) -> None:
        """
        登记允许 Agent 使用的数据及明确业务角色。

        Parameters
        ----------
        dataset_id : str
            会话内唯一标识，限字母、数字、下划线和连字符，最长 64 字符。
        df : FrameLike
            Pandas 或 Polars 宽表。
        features : list[str]
            允许分析的特征列，不能包含 target 或分组及日期列。
        target : str | None
            二分类标签；允许当前数据缺列，以表示尚无表现数据。
        group_columns : list[str] | None
            允许模型选择的分组列。
        time_col : str | None
            原始日期列。
        description : str
            业务口径说明，最长 1000 字符；将发送给模型。
        missing_values : list[int | float | str] | None
            交由 MARS 处理的额外缺失值。

        Raises
        ------
        RuntimeError
            会话正在运行时抛出。

        Notes
        -----
        登记校验会在标识重复、字段无效或业务角色冲突时抛出 ValueError。
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("session is busy")
        try:
            self._register(
                dataset_id,
                df,
                features,
                target,
                group_columns,
                time_col,
                description,
                missing_values,
            )
        finally:
            self._lock.release()

    def _register(
        self,
        dataset_id: str,
        df: FrameLike,
        features: list[str],
        target: str | None,
        group_columns: list[str] | None,
        time_col: str | None,
        description: str,
        missing_values: list[int | float | str] | None,
    ) -> None:
        """验证字段访问范围并保存投影后的快照。"""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", dataset_id):
            raise ValueError("dataset_id must match [A-Za-z0-9_-]{1,64}")
        if dataset_id in self._datasets:
            raise ValueError(f"dataset_id already registered: {dataset_id}")
        if not features or len(features) != len(set(features)):
            raise ValueError("features must be non-empty and unique")
        if len(description) > 1000:
            raise ValueError("description must contain at most 1000 characters")
        groups = tuple(dict.fromkeys(group_columns or []))
        roles = set(groups) | {
            value for value in (target, time_col) if value is not None
        }
        if set(features) & roles:
            raise ValueError(
                "features must not overlap target, group_columns or time_col"
            )
        frame = to_polars_frame(df)
        columns = list(
            dict.fromkeys([*features, *groups, *([time_col] if time_col else [])])
        )
        if any(column not in frame.columns for column in columns):
            raise ValueError("features, group_columns and time_col must exist in df")
        if target in frame.columns:
            columns.append(str(target))
        self._datasets[dataset_id] = _Dataset(
            dataset_id,
            frame.select(columns),
            tuple(features),
            target,
            groups,
            time_col,
            description,
            tuple(missing_values or []),
        )

    @property
    def messages(self) -> tuple[MarsAgentMessage, ...]:
        """返回完整消息副本，外部修改不会影响后续模型请求。"""
        return tuple(deepcopy(self._messages))

    @property
    def report_ids(self) -> tuple[str, ...]:
        """返回当前会话的报告标识。"""
        return tuple(self._reports)

    def get_report(self, report_id: str) -> MarsAgentReport:
        """
        读取保留完整聚合表的本地报告副本。

        Parameters
        ----------
        report_id : str
            当前会话报告标识。

        Returns
        -------
        MarsAgentReport
            结果快照，可用于本地验证或交给 reporting 层导出。

        Raises
        ------
        KeyError
            报告不存在时抛出。
        """
        if report_id not in self._reports:
            raise KeyError(f"report_id does not exist: {report_id}")
        report = self._reports[report_id]
        return MarsAgentReport(
            report.id,
            report.kind,
            report.dataset_id,
            report.benchmark_id,
            {name: table.clone() for name, table in report.tables.items()},
            deepcopy(report.metadata),
        )

    def clear_history(self) -> None:
        """
        清空完整会话历史，保留数据和报告。

        Raises
        ------
        RuntimeError
            会话正在运行时抛出。
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("session is busy")
        try:
            self._messages.clear()
        finally:
            self._lock.release()

    def _save_report(
        self,
        kind: str,
        dataset_id: str,
        benchmark_id: str | None,
        tables: dict[str, FrameLike],
        metadata: dict[str, Any],
    ) -> MarsAgentReport:
        """保存有确定来源的聚合结果，完整表不直接加入模型消息。"""
        report_id = f"report_{len(self._reports) + 1}"
        report = MarsAgentReport(
            report_id,
            kind,
            dataset_id,
            benchmark_id,
            {name: to_polars_frame(table) for name, table in tables.items()},
            deepcopy(metadata),
        )
        self._reports[report_id] = report
        return report
