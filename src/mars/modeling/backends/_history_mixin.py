"""建模后端 history 与 trial 管理 mixin。"""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from mars.modeling.backends.common import HISTORY_BASE_COLUMNS, METRIC_NAMES


class BackendHistoryMixin:
    """封装后端共享的 history、排序和 best model 管理逻辑。"""

    optimize_metric: str
    NATIVE_TRAINING_METRICS: set[str]
    metric_directions: Mapping[str, Any]
    keep_top_n_models: int
    retained_models: dict[int, Any]
    retained_model_rows: list[dict[str, Any]]
    all_models: dict[int, Any]
    param_space: dict[str, Any]
    history: list[dict[str, Any]]
    metric_names: list[str]
    data_dict: dict[str, Any]

    def get_default_space(self) -> dict[str, Any]:
        """返回默认搜索空间。"""
        raise NotImplementedError

    def _resolve_training_metric(self, training_metric: str | None) -> str:
        """确定训练期监控指标。"""
        candidate = (training_metric or self.optimize_metric).lower()
        if candidate in self.NATIVE_TRAINING_METRICS:
            return candidate
        return "auc"

    def _metric_direction(self, metric_name: str | None = None) -> Any:
        """返回指定指标的优化方向。"""
        return self.metric_directions.get((metric_name or self.optimize_metric).lower(), "maximize")

    def _initial_best_score(self) -> float:
        """生成当前方向下的初始 best score。"""
        return np.inf if self._metric_direction() == "minimize" else -np.inf

    def _is_better_score(self, score: float, baseline: float) -> bool:
        """判断新分数是否优于基线。"""
        return score < baseline if self._metric_direction() == "minimize" else score > baseline

    def _invalid_trial_score(self, penalty_diff: float) -> float:
        """生成泛化失败时的惩罚分。"""
        if self._metric_direction() == "minimize":
            return float(1_000_000.0 + max(0.0, penalty_diff))
        return float(-100.0 - max(0.0, penalty_diff))

    def _generalization_diff(self, train_score: float, compare_score: float) -> float:
        """按当前优化方向计算泛化差值。"""
        if self._metric_direction() == "minimize":
            return round(compare_score - train_score, 6)
        return round(train_score - compare_score, 6)

    def _retain_candidate_model(
        self,
        *,
        trial_num: int,
        model: Any,
        score: float,
        record: Mapping[str, Any],
    ) -> None:
        """动态保留当前最优的 Top-N trial 模型。"""
        if self.keep_top_n_models <= 0:
            return

        self.retained_models[trial_num] = model
        retained_row = {"trial_num": trial_num, "score": float(score), **dict(record)}
        self.retained_model_rows = [
            row
            for row in self.retained_model_rows
            if int(row.get("trial_num", -1)) != trial_num
        ]
        self.retained_model_rows.append(retained_row)

        ascending = self._metric_direction() == "minimize"
        self.retained_model_rows = sorted(
            self.retained_model_rows,
            key=lambda row: float(row.get("score", np.inf if ascending else -np.inf)),
            reverse=not ascending,
        )[: self.keep_top_n_models]
        retained_trial_nums = {int(row["trial_num"]) for row in self.retained_model_rows}
        self.retained_models = {
            retained_trial_num: retained_model
            for retained_trial_num, retained_model in self.retained_models.items()
            if retained_trial_num in retained_trial_nums
        }
        self.all_models = dict(self.retained_models)

    @property
    def replay_param_keys(self) -> list[str]:
        """返回 replay 可用的参数键顺序。"""
        keys = list(self.get_default_space().keys())
        for key in self.param_space.keys():
            if key not in keys:
                keys.append(key)
        return keys

    def _sync_to_disk(self, record: Mapping[str, Any], path: str | Path | None) -> None:
        """将单次 trial 记录追加写入 CSV。"""
        if path is None:
            return
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(record)]).to_csv(
            path_obj,
            mode="a",
            header=not path_obj.exists(),
            index=False,
        )

    def build_history_table(self) -> pd.DataFrame:
        """构造结构化 trial history 表。"""
        history_table = pd.DataFrame(self.history)
        if history_table.empty:
            desired_columns = list(HISTORY_BASE_COLUMNS) + list(self.replay_param_keys)
            metric_names = getattr(self, "metric_names", list(METRIC_NAMES))
            for split_name in self.data_dict.keys():
                for metric_name in metric_names:
                    desired_columns.append(f"{split_name}_{metric_name}")
            return pd.DataFrame(columns=desired_columns)

        param_columns = [col for col in self.replay_param_keys if col in history_table.columns]
        metric_columns: list[str] = []
        for split_name in self.data_dict.keys():
            for metric_name in self.metric_names:
                column_name = f"{split_name}_{metric_name}"
                if column_name in history_table.columns:
                    metric_columns.append(column_name)

        ordered_columns = [*HISTORY_BASE_COLUMNS, *param_columns, *metric_columns]
        extra_columns = [col for col in history_table.columns if col not in ordered_columns]
        return history_table.reindex(columns=ordered_columns + sorted(extra_columns))

    def get_best_iteration(self, model: Any) -> int | None:
        """提取模型的最佳迭代轮次。"""
        best_iteration = getattr(model, "best_iteration", None)
        if isinstance(best_iteration, numbers.Integral):
            return int(best_iteration)
        get_best_iteration = getattr(model, "get_best_iteration", None)
        if callable(get_best_iteration):
            try:
                best_iteration = get_best_iteration()
                if isinstance(best_iteration, numbers.Integral):
                    return int(best_iteration)
            except Exception:
                return None
        return None
