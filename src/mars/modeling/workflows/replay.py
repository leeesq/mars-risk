"""调参结果 replay 工作流。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import pandas as pd

from mars.compute import FrameLike
from mars.modeling.backends.base import MarsBaseModelStrategy
from mars.modeling.contracts.replay_result import MarsModelReplayResult
from mars.modeling.contracts.specs import ModelingSpec, ReplaySpec
from mars.modeling.contracts.tuning_result import MarsModelTuningResult
from mars.modeling.evaluation import MarsModelEvaluator
from mars.modeling.evaluation.metrics import MetricCallable, MetricDirection
from mars.modeling.inference.predictor import ModelPredictor
from mars.modeling.workflows._backend_factory import build_backend_from_spec
from mars.modeling.workflows._spec_builder import build_modeling_spec

_build_spec = build_modeling_spec


class MarsModelReplayRunner:
    """
    基于 `MarsModelTuningResult` 回放调参结果。

    `MarsModelReplayRunner` 不在构造函数中绑定模型类型、特征列或目标列，而是从
    :meth:`replay` 传入的调优结果中读取建模规格。回放候选既可以按 Top-K 自动选择，
    也可以由调用者传入 trial 编号。benchmark 分数、时间列和辅助验证目标属于本次
    replay 评估上下文，因此保留在方法入参中。

    Examples
    --------
    >>> replay = MarsModelReplayRunner()
    >>> callable(replay.replay)
    True
    """

    def __init__(self) -> None:
        self.spec: ModelingSpec | None = None

    @staticmethod
    def _build_spec_from_result(tuning_result: MarsModelTuningResult) -> ModelingSpec:
        """从调优结果恢复 replay 所需的建模规格。"""
        training_config = dict(getattr(tuning_result, "training_config", {}) or {})
        return _build_spec(
            model_type=tuning_result.model_type,
            features=tuning_result.features,
            target=tuning_result.target,
            dataset_flag_col=tuning_result.dataset_flag_col,
            categorical_features=tuning_result.categorical_features,
            optimize_metric=tuning_result.optimize_metric,
            seed=int(training_config.get("seed", 1206)),
            lr_feature_mode=str(training_config.get("lr_feature_mode", "numeric")),
            lr_binning_type=str(training_config.get("lr_binning_type", "native")),
            lr_binner_kwargs=training_config.get("lr_binner_kwargs"),
        )

    def _build_backend(
        self,
        df: FrameLike,
        *,
        optimize_metric: str | None = None,
        seed: int | None = None,
        metric_params: Mapping[str, Any] | None = None,
        custom_metrics: Mapping[str, MetricCallable] | None = None,
        metric_directions: Mapping[str, MetricDirection] | None = None,
        training_metric: str | None = None,
        backend_metric: Any | None = None,
    ) -> MarsBaseModelStrategy:
        """构建用于 replay 已调优参数集合的后端。"""
        spec = self.spec
        if spec is None:
            raise RuntimeError("Replay spec is unavailable before replay(...) receives a tuning run.")
        return build_backend_from_spec(
            spec,
            df,
            optimize_metric=optimize_metric,
            seed=seed,
            metric_params=metric_params,
            custom_metrics=custom_metrics,
            metric_directions=metric_directions,
            training_metric=training_metric,
            backend_metric=backend_metric,
        )

    def replay(
        self,
        tuning_result: MarsModelTuningResult,
        df: FrameLike,
        *,
        top_k: int = 5,
        sort_metric: str = "ks",
        include_val: bool = True,
        trial_nums: Sequence[int] | None = None,
        retrain: bool = True,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        optimize_metric: str | None = None,
        metric_params: Mapping[str, Any] | None = None,
        custom_metrics: Mapping[str, MetricCallable] | None = None,
        metric_directions: Mapping[str, MetricDirection] | None = None,
        training_metric: str | None = None,
        backend_metric: Any | None = None,
        benchmark_col: str | None = None,
        benchmark_cols: Sequence[str] | None = None,
        time_col: str | None = None,
        val_target: str | None = None,
        aux_targets: Sequence[str] | None = None,
        target_group_cols: Mapping[str, str] | None = None,
        psi_include_missing: bool = False,
    ) -> MarsModelReplayResult:
        """
        回放 Top-K 或指定 trial，并生成模型、打分数据和评估报告。

        Parameters
        ----------
        tuning_result : MarsModelTuningResult
            提供模型类型、特征列、目标列和样本切片配置的调优结果。
        df : FrameLike
            用于重新训练和打分的样本表。
        top_k : int
            要回放的 trial 数量。
        sort_metric : str
            replay 排行表排序指标。
        include_val : bool
            是否将 validation 切片指标纳入平均排序。
        trial_nums : Sequence[int] | None
            指定要 replay 的 trial 编号；传入后按给定顺序回放，``top_k`` 不参与选择。
        retrain : bool
            是否使用 trial 参数重新训练；``False`` 时只使用调参阶段已保留的模型。
        num_boost_round : int
            当调优结果中没有保存该配置时使用的最大 boosting 轮数。
        early_stopping_rounds : int
            当调优结果中没有保存该配置时使用的 early stopping 轮数。
        optimize_metric : str | None
            覆盖 replay 后端使用的优化指标。
        metric_params : Mapping[str, Any] | None
            指标参数，例如 ``f1_threshold``。
        custom_metrics : Mapping[str, MetricCallable] | None
            replay 重训时使用的自定义指标函数字典。
        metric_directions : Mapping[str, MetricDirection] | None
            指标排序方向；会影响 Top-K trial 选择和自定义指标 replay。
        training_metric : str | None
            模型后端训练期监控指标。
        backend_metric : Any | None
            透传给模型后端原生训练接口的自定义 metric。
        benchmark_col : str | None
            benchmark 或 champion 模型分数列名。
        benchmark_cols : Sequence[str] | None
            多个 benchmark 或 champion 模型分数列名。
        time_col : str | None
            原始时间列名，用于补充报告中的时间边界。
        val_target : str | None
            替代验证目标列名。
        aux_targets : Sequence[str] | None
            辅助验证目标列名；不参与训练，只进入 replay 评估报告。
        target_group_cols : Mapping[str, str] | None
            每个目标对应的独立样本切片列名，用于长短 y 表现期不一致的评估。
        psi_include_missing : bool
            replay 评估报告计算 `score_psi` 和 `feature_psi` 时是否纳入缺失值箱。

        Returns
        -------
        MarsModelReplayResult
            包含 replay 排行表、模型、打分数据和评估报告的结果对象。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。
        """
        self.spec = self._build_spec_from_result(tuning_result)
        spec = self.spec

        run_training_config = dict(getattr(tuning_result, "training_config", {}) or {})
        resolved_num_boost_round = (
            int(num_boost_round)
            if num_boost_round != 500 or "num_boost_round" not in run_training_config
            else int(run_training_config.get("num_boost_round", num_boost_round))
        )
        resolved_early_stopping_rounds = (
            int(early_stopping_rounds)
            if early_stopping_rounds != 50 or "early_stopping_rounds" not in run_training_config
            else int(run_training_config.get("early_stopping_rounds", early_stopping_rounds))
        )
        replay_spec = ReplaySpec(
            top_k=top_k,
            sort_metric=sort_metric.lower(),
            include_val=include_val,
            num_boost_round=resolved_num_boost_round,
            early_stopping_rounds=resolved_early_stopping_rounds,
            optimize_metric=(optimize_metric or spec.optimize_metric).lower(),
        )

        history_df = tuning_result.history_table.copy()
        valid_df = history_df[
            (history_df["trial_state"] == "COMPLETE") & history_df["is_valid"]
        ].copy()
        if valid_df.empty:
            raise ValueError("No valid completed trials are available for replay.")

        metric_suffix = f"_{replay_spec.sort_metric}"
        oot_cols = [
            col
            for col in valid_df.columns
            if "oot" in col.lower() and col.endswith(metric_suffix)
        ]
        cols_to_mean = list(oot_cols)
        if replay_spec.include_val:
            val_cols = [
                col
                for col in valid_df.columns
                if col.lower() == f"val_{replay_spec.sort_metric}".lower()
            ]
            cols_to_mean.extend(val_cols)
        if not cols_to_mean:
            raise ValueError(f"No ranking columns were found for sort_metric={replay_spec.sort_metric!r}.")

        metric_direction = dict(getattr(tuning_result, "metric_directions", {}) or {}).get(
            replay_spec.sort_metric,
            "maximize",
        )
        valid_df["custom_mean_score"] = valid_df[cols_to_mean].mean(axis=1)
        if trial_nums is not None:
            requested_trial_nums = [int(trial_num) for trial_num in trial_nums]
            available_trial_nums = set(valid_df["trial_num"].astype(int).tolist())
            missing_trial_nums = [
                trial_num
                for trial_num in requested_trial_nums
                if trial_num not in available_trial_nums
            ]
            if missing_trial_nums:
                raise ValueError(
                    f"Requested trial_nums are not valid completed trials: {missing_trial_nums}."
                )
            trial_order = {
                trial_num: order
                for order, trial_num in enumerate(requested_trial_nums)
            }
            ranking_table = (
                valid_df.loc[valid_df["trial_num"].astype(int).isin(requested_trial_nums)]
                .assign(_trial_order=lambda frame: frame["trial_num"].astype(int).map(trial_order))
                .sort_values("_trial_order")
                .drop(columns=["_trial_order"])
                .copy()
            )
        else:
            ranking_table = (
                valid_df.sort_values(
                    "custom_mean_score",
                    ascending=metric_direction == "minimize",
                )
                .head(replay_spec.top_k)
                .copy()
            )

        restored_metric_directions: dict[str, MetricDirection] = {
            key: cast(MetricDirection, value)
            for key, value in dict(tuning_result.metric_directions).items()
            if value in {"maximize", "minimize"}
        }
        backend = self._build_backend(
            df,
            optimize_metric=replay_spec.optimize_metric,
            seed=spec.seed,
            metric_params=metric_params or tuning_result.training_config.get("metric_params"),
            custom_metrics=custom_metrics,
            metric_directions=metric_directions or restored_metric_directions,
            training_metric=training_metric or tuning_result.training_config.get("training_metric"),
            backend_metric=backend_metric,
        )
        backend.num_boost_round = replay_spec.num_boost_round
        backend.early_stopping_rounds = replay_spec.early_stopping_rounds

        evaluator = MarsModelEvaluator()

        models: dict[str, Any] = {}
        scored_df = df
        reports: dict[str, Any] = {}
        importance_tables: dict[str, pd.DataFrame] = {}
        diagnostic_tables: dict[str, dict[str, pd.DataFrame]] = {}
        leaderboard_rows: list[dict[str, Any]] = []

        for rank, (_, row) in enumerate(ranking_table.iterrows(), start=1):
            trial_num = int(row["trial_num"])
            pure_params = {
                key: row[key]
                for key in tuning_result.replay_candidates
                if key in row.index and pd.notna(row[key])
            }
            if retrain:
                model = backend.train_model(
                    trial=None,
                    params=pure_params,
                    startup_trials=10**9,
                    training_metric=backend.training_metric,
                )
            else:
                if trial_num not in tuning_result.retained_models:
                    raise ValueError(
                        f"trial_num={trial_num} was not retained during tuning. "
                        "Use retrain=True or increase keep_top_n_models."
                    )
                model = tuning_result.retained_models[trial_num]
            model_name = f"top{rank}_trial{trial_num}"
            models[model_name] = model
            importance_tables[model_name] = (
                backend.extract_importance(model)
                if retrain
                else tuning_result.importance_table.copy()
            )
            extract_diagnostics = getattr(backend, "extract_diagnostics", None)
            if callable(extract_diagnostics):
                diagnostic_tables[model_name] = extract_diagnostics(model)

            pred_col = f"prob_{model_name}"
            predictor = ModelPredictor(
                model,
                feature_list=spec.features,
                categorical_features=spec.categorical_features,
                category_levels=getattr(backend, "category_levels", {}),
                model_type=spec.model_type,
            )
            scored_df = predictor.predict(scored_df, pred_col=pred_col, inplace=False)
            reports[model_name] = evaluator.evaluate(
                scored_df,
                pred_col=pred_col,
                group_col=spec.dataset_flag_col,
                target=spec.target,
                benchmark_col=benchmark_col,
                benchmark_cols=benchmark_cols,
                time_col=time_col,
                val_target=val_target,
                aux_targets=aux_targets,
                target_group_cols=target_group_cols,
                feature_cols=spec.features,
                importance_table=importance_tables[model_name],
                psi_include_missing=psi_include_missing,
            )

            leaderboard_row = {
                "rank": rank,
                "model_name": model_name,
                "trial_num": trial_num,
                "custom_mean_score": float(row["custom_mean_score"]),
                "best_iteration": backend.get_best_iteration(model),
                "backend_data_mode": backend.backend_data_mode,
            }
            for column_name, value in row.items():
                if column_name in {"custom_mean_score", "trial_num"}:
                    continue
                if str(column_name).endswith(f"_{replay_spec.sort_metric}") or str(column_name).startswith(
                    "val_"
                ):
                    leaderboard_row[str(column_name)] = value
            leaderboard_rows.append(leaderboard_row)

        leaderboard_table = pd.DataFrame(leaderboard_rows)
        if not leaderboard_table.empty:
            metric_columns = sorted(
                [
                    column
                    for column in leaderboard_table.columns
                    if column
                    not in {
                        "rank",
                        "model_name",
                        "trial_num",
                        "custom_mean_score",
                        "best_iteration",
                        "backend_data_mode",
                    }
                ]
            )
            leaderboard_table = leaderboard_table[
                [
                    "rank",
                    "model_name",
                    "trial_num",
                    "custom_mean_score",
                    "best_iteration",
                    "backend_data_mode",
                    *metric_columns,
                ]
            ]

        return MarsModelReplayResult(
            model_type=spec.model_type,
            ranking_table=ranking_table,
            leaderboard_table=leaderboard_table,
            models=models,
            scored_df=scored_df,
            reports=reports,
            importance_tables=importance_tables,
            diagnostic_tables=diagnostic_tables,
        )
