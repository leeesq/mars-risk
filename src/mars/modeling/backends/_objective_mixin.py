"""建模后端 objective 与指标执行 mixin。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mars.compute import FrameLike
from mars.modeling.evaluation.metrics import evaluate_metrics


class BackendObjectiveMixin:
    """封装后端共享的指标计算与 trial 执行逻辑。"""

    metric_names: list[str]
    metric_params: Mapping[str, Any]
    custom_metrics: Mapping[str, Any]
    data_dict: dict[str, FrameLike]
    param_space: dict[str, Any]
    training_metric: str
    optimize_metric: str
    max_diff: float
    use_oot_penalty: bool
    best_score: float
    best_model: Any
    history: list[dict[str, Any]]

    def predict_scores(self, model: Any, split_name: str) -> np.ndarray:
        """返回指定切片的预测分数。"""
        raise NotImplementedError

    def _get_target_array(self, df: FrameLike) -> np.ndarray:
        """返回目标数组。"""
        raise NotImplementedError

    def get_default_space(self) -> dict[str, Any]:
        """返回默认搜索空间。"""
        raise NotImplementedError

    def train_model(
        self,
        trial: Any,
        params: dict[str, Any],
        startup_trials: int,
        training_metric: str,
    ) -> Any:
        """训练单次 trial 模型。"""
        raise NotImplementedError

    def _generalization_diff(self, train_score: float, compare_score: float) -> float:
        """返回泛化差值。"""
        raise NotImplementedError

    def _is_better_score(self, score: float, baseline: float) -> bool:
        """判断分数是否更优。"""
        raise NotImplementedError

    def _retain_candidate_model(
        self,
        *,
        trial_num: int,
        model: Any,
        score: float,
        record: Mapping[str, Any],
    ) -> None:
        """保留候选模型。"""
        raise NotImplementedError

    def _invalid_trial_score(self, penalty_diff: float) -> float:
        """返回无效 trial 的惩罚分。"""
        raise NotImplementedError

    def _sync_to_disk(self, record: Mapping[str, Any], path: str | Path | None) -> None:
        """同步 trial 记录到磁盘。"""
        raise NotImplementedError

    def _evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """统一计算当前建模任务需要的指标。"""
        return evaluate_metrics(
            y_true,
            y_pred,
            self.metric_names,
            metric_params=self.metric_params,
            custom_metrics=self.custom_metrics,
        )

    def evaluate_split(self, model: Any, split_name: str) -> dict[str, float]:
        """评估单个切片上的模型表现。"""
        preds = self.predict_scores(model, split_name)
        y_true = self._get_target_array(self.data_dict[split_name])
        return self._evaluate_predictions(y_true, preds)

    def parse_param_space(self, trial: Any, default_space: Mapping[str, Any]) -> dict[str, Any]:
        """将搜索空间解析为当前 trial 的确定参数。"""
        active_space = dict(default_space)
        active_space.update(self.param_space)

        params: dict[str, Any] = {}
        for name, config in active_space.items():
            if not isinstance(config, (tuple, list)):
                params[name] = config
                continue

            if len(config) == 0:
                raise ValueError(f"Empty config for parameter {name!r}.")

            ptype = config[0]
            if ptype == "int":
                int_low, int_high = int(config[1]), int(config[2])
                int_step = int(config[3]) if len(config) > 3 else 1
                params[name] = trial.suggest_int(
                    name,
                    int_low,
                    int_high,
                    step=int_step,
                )
            elif ptype == "float":
                float_low, float_high = float(config[1]), float(config[2])
                float_step = float(config[3]) if len(config) > 3 else None
                params[name] = (
                    trial.suggest_float(name, float_low, float_high)
                    if float_step is None
                    else trial.suggest_float(name, float_low, float_high, step=float_step)
                )
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, list(config[1]))
            else:
                params[name] = config
        return params

    def objective(
        self,
        trial: Any,
        startup_trials: int,
        history_path: str | Path | None,
    ) -> float:
        """执行一次完整的 optuna trial 生命周期。"""
        record: dict[str, Any] = {
            "trial_num": getattr(trial, "number", -1),
            "trial_state": "INIT_FAIL",
        }

        try:
            trial_num = int(getattr(trial, "number", -1))
            params = self.parse_param_space(trial, self.get_default_space())
            record.update(params)
            model = self.train_model(
                trial=trial,
                params=params,
                startup_trials=startup_trials,
                training_metric=self.training_metric,
            )
            metrics_by_split = {
                split_name: self.evaluate_split(model, split_name)
                for split_name in self.data_dict.keys()
            }
            train_score = metrics_by_split["train"][self.optimize_metric]
            val_score = metrics_by_split["val"][self.optimize_metric]
            oot_scores = [
                split_metrics[self.optimize_metric]
                for split_name, split_metrics in metrics_by_split.items()
                if "oot" in split_name.lower()
            ]

            val_diff = self._generalization_diff(train_score, val_score)
            is_valid = val_diff <= self.max_diff
            max_penalty_diff = val_diff

            max_oot_diff: float | None = None
            if oot_scores:
                oot_diffs = [
                    self._generalization_diff(train_score, oot_score)
                    for oot_score in oot_scores
                ]
                max_oot_diff = max(oot_diffs)
                if self.use_oot_penalty:
                    max_penalty_diff = max(max_penalty_diff, max_oot_diff)
                    if max_oot_diff > self.max_diff:
                        is_valid = False

            record.update(
                {
                    "trial_state": "COMPLETE",
                    "is_valid": is_valid,
                    "val_diff": round(val_diff, 4),
                    "max_oot_diff": round(max_oot_diff, 4) if max_oot_diff is not None else None,
                    **{
                        f"{split_name}_{metric_name}": metric_value
                        for split_name, metrics in metrics_by_split.items()
                        for metric_name, metric_value in metrics.items()
                    },
                }
            )

            if is_valid:
                if self._is_better_score(val_score, self.best_score):
                    self.best_score = val_score
                    self.best_model = model
                self._retain_candidate_model(
                    trial_num=trial_num,
                    model=model,
                    score=val_score,
                    record=record,
                )

            return float(val_score if is_valid else self._invalid_trial_score(max_penalty_diff))

        except Exception as exc:
            optuna_module: Any = None
            try:
                import optuna

                optuna_module = optuna
            except Exception:
                optuna_module = None

            if optuna_module is not None and isinstance(exc, optuna_module.exceptions.TrialPruned):
                record["trial_state"] = "PRUNED"
                raise

            record["trial_state"] = f"ERROR: {str(exc)[:120]}"
            raise
        finally:
            self.history.append(record)
            self._sync_to_disk(record, history_path)
