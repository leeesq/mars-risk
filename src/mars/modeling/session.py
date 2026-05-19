"""High-level AutoML workflow session for ``mars.modeling``."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence
import pandas as pd

from mars.modeling.base import FrameLike
from mars.modeling.data import MarsModelDataSlicer
from mars.modeling.report import MarsModelEvaluator, MarsModelingReport
from mars.modeling.results import MarsModelingRun, MarsReplayRun
from mars.modeling.spec import SplitSpec
from mars.modeling.tuner import MarsModelReplay, MarsModelTuner


class MarsModelingSession:
    """High-level orchestrator that composes the public modeling tools."""

    def __init__(
        self,
        *,
        model_type: str,
        features: Sequence[str],
        target: str,
        dataset_flag_col: str = "dataset_flag",
        categorical_features: Optional[Sequence[str]] = None,
        optimize_metric: str = "ks",
        seed: int = 1206,
        benchmark_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> None:
        self.tuner = MarsModelTuner(
            model_type=model_type,
            features=features,
            target=target,
            dataset_flag_col=dataset_flag_col,
            categorical_features=categorical_features,
            optimize_metric=optimize_metric,
            seed=seed,
            benchmark_col=benchmark_col,
            time_col=time_col,
        )
        self.replay_runner = MarsModelReplay(
            model_type=model_type,
            features=features,
            target=target,
            dataset_flag_col=dataset_flag_col,
            categorical_features=categorical_features,
            optimize_metric=optimize_metric,
            seed=seed,
            benchmark_col=benchmark_col,
            time_col=time_col,
        )

    @property
    def last_run(self) -> Optional[MarsModelingRun]:
        """Return the latest tuning result produced by this session."""
        return self.tuner.last_run

    @property
    def best_model(self) -> Any:
        """Return the best model from the latest tuning run."""
        return self.tuner.best_model

    @property
    def best_score(self) -> Optional[float]:
        """Return the best validation score from the latest tuning run."""
        return self.tuner.best_score

    @property
    def best_params(self) -> Optional[dict[str, Any]]:
        """Return the best parameter set from the latest tuning run."""
        return self.tuner.best_params

    @property
    def history_table(self) -> pd.DataFrame:
        """Return the structured history table from the latest tuning run."""
        return self.tuner.history_table

    def slice(
        self,
        df: FrameLike,
        *,
        time_col: str,
        split_ratios: Mapping[str, float],
        label_col: Optional[str] = None,
        mode: str = "strict",
        train_key: str = "train",
        val_key: str = "val",
        random_seed: int = 42,
    ) -> FrameLike:
        """Split raw modeling data with the public slicer helper."""
        split_spec = SplitSpec(
            time_col=time_col,
            label_col=label_col or self.tuner.spec.target,
            mode=mode.lower(),
            train_key=train_key,
            val_key=val_key,
            random_seed=random_seed,
        )
        slicer = MarsModelDataSlicer(
            df=df,
            time_col=split_spec.time_col,
            label_col=split_spec.label_col,
            dataset_flag_col=self.tuner.spec.dataset_flag_col,
        )
        if split_spec.mode == "strict":
            return slicer.split_by_time_strictly(dict(split_ratios))
        if split_spec.mode == "hybrid":
            return slicer.split_hybrid_random_val(
                dict(split_ratios),
                train_key=split_spec.train_key,
                val_key=split_spec.val_key,
                random_seed=split_spec.random_seed,
            )
        raise ValueError(f"Unsupported slice mode: {mode!r}. Expected 'strict' or 'hybrid'.")

    def tune(self, df: FrameLike, **kwargs: Any) -> MarsModelingRun:
        """Delegate tuning to :class:`MarsModelTuner`."""
        return self.tuner.tune(df, **kwargs)

    def evaluate(
        self,
        df: FrameLike,
        *,
        pred_col: str,
        benchmark_col: Optional[str] = None,
        time_col: Optional[str] = None,
        val_target_col: Optional[str] = None,
        feature_cols: Optional[Sequence[str]] = None,
        importance_table: Optional[pd.DataFrame] = None,
    ) -> MarsModelingReport:
        """Delegate report generation to :class:`MarsModelEvaluator`."""
        run = self.last_run
        resolved_feature_cols = list(feature_cols) if feature_cols is not None else list(self.tuner.spec.features)
        resolved_importance = importance_table
        if resolved_importance is None and run is not None:
            resolved_importance = run.importance_table.copy()
        evaluator = MarsModelEvaluator(
            group_col=self.tuner.spec.dataset_flag_col,
            target_col=self.tuner.spec.target,
            benchmark_col=benchmark_col if benchmark_col is not None else self.tuner.spec.benchmark_col,
            time_col=time_col if time_col is not None else self.tuner.spec.time_col,
            val_target_col=val_target_col,
            feature_cols=resolved_feature_cols,
            importance_table=resolved_importance,
        )
        report = evaluator.evaluate(df, pred_col=pred_col)
        if run is not None:
            report.metadata.update(
                {
                    "history_table": run.history_table.copy(),
                    "importance_table": resolved_importance.copy() if resolved_importance is not None else run.importance_table.copy(),
                    "training_config": dict(run.training_config),
                    "library_versions": dict(run.library_versions),
                    "backend_data_mode": run.backend_data_mode,
                    "model_type": run.model_type,
                    "optimize_metric": run.optimize_metric,
                    "best_score": run.best_score,
                    "best_iteration": run.best_iteration,
                }
            )
        return report

    def replay(
        self,
        run: MarsModelingRun,
        df: FrameLike,
        **kwargs: Any,
    ) -> MarsReplayRun:
        """Delegate replay and rescoring to :class:`MarsModelReplay`."""
        return self.replay_runner.run(run, df, **kwargs)
