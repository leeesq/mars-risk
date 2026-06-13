from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl
import pytest

from mars.core.base import MarsBaseSelector
from mars.feature import MarsImportanceSelector, MarsLinearSelector, MarsLiteOptBinner
from mars.modeling import MarsModelTuningResult
from mars.pipeline import (
    MarsModelingPipeline,
    MarsModelingStep,
    MarsSelectionStep,
    MarsWOEBinningStep,
)


class _KeepSelector(MarsBaseSelector):
    """测试用筛选器：按给定列名保留特征。"""

    def __init__(self, selected: list[str]) -> None:
        super().__init__()
        self._selected = selected

    def fit(
        self,
        X: pl.DataFrame | pd.DataFrame,
        y: Any | None = None,
        *,
        features: list[str] | None = None,
    ) -> _KeepSelector:
        del y
        candidates = list(features or X.columns)
        self.selected_features_ = [feature for feature in candidates if feature in self._selected]
        self.n_features_in_ = len(candidates)
        self._is_fitted = True
        return self


def test_pipeline_rejects_duplicate_step_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        MarsModelingPipeline(
            target="target",
            features=["x1"],
            steps=[
                MarsSelectionStep(name="same", selector=_KeepSelector(["x1"])),
                MarsSelectionStep(name="same", selector=_KeepSelector(["x1"])),
            ],
        )


def test_pipeline_requires_modeling_step_last() -> None:
    with pytest.raises(ValueError, match="last"):
        MarsModelingPipeline(
            target="target",
            features=["x1"],
            steps=[
                MarsModelingStep(name="modeling", model_type="lr"),
                MarsSelectionStep(name="selection", selector=_KeepSelector(["x1"])),
            ],
        )


def test_pipeline_allows_multiple_selection_steps(sample_modeling_pd: pd.DataFrame) -> None:
    importance = pd.DataFrame(
        {
            "feature": ["x1", "x2"],
            "importance": [0.9, 0.1],
        }
    )
    pipeline = MarsModelingPipeline(
        target="target",
        features=["x1", "x2", "x3"],
        steps=[
            MarsSelectionStep(name="keep_two", selector=_KeepSelector(["x1", "x2"])),
            MarsSelectionStep(
                name="importance",
                selector=MarsImportanceSelector(
                    method="importance",
                    selection_mode="top_k",
                    selection_threshold=1,
                ),
                fit_params={"importance_table": importance},
            ),
        ],
    )

    result = pipeline.fit(sample_modeling_pd)
    transformed = pipeline.transform(sample_modeling_pd)

    assert result.active_features == ["x1"]
    assert result.step_results[0].input_features == ["x1", "x2", "x3"]
    assert result.step_results[1].input_features == ["x1", "x2"]
    assert isinstance(transformed, pd.DataFrame)
    with pytest.raises(RuntimeError, match="MarsModelingStep"):
        pipeline.predict(sample_modeling_pd)


def test_pipeline_selection_step_empty_features_raises(sample_modeling_pd: pd.DataFrame) -> None:
    pipeline = MarsModelingPipeline(
        target="target",
        features=["x1", "x2"],
        steps=[MarsSelectionStep(name="empty", selector=_KeepSelector([]))],
    )

    with pytest.raises(ValueError, match="empty"):
        pipeline.fit(sample_modeling_pd)


def test_pipeline_runs_selection_to_internal_woe_lr_modeling(
    sample_modeling_pd: pd.DataFrame,
) -> None:
    pipeline = MarsModelingPipeline(
        target="target",
        features=["x1", "x2", "x3"],
        steps=[
            MarsSelectionStep(name="keep_two", selector=_KeepSelector(["x1", "x2"])),
            MarsModelingStep(
                name="modeling",
                model_type="lr",
                tune_params={
                    "n_trials": 1,
                    "startup_trials": 1,
                    "warmup_steps": 3,
                    "artifact_dir": None,
                    "max_diff": 100.0,
                },
            ),
        ],
    )

    result = pipeline.fit(sample_modeling_pd)
    scored = pipeline.predict(sample_modeling_pd, pred_col="pipeline_score")

    assert result.active_features == ["x1", "x2"]
    assert isinstance(result.modeling_result, MarsModelTuningResult)
    assert result.modeling_result.features == ["x1", "x2"]
    assert result.step_results[-1].metadata["model_type"] == "lr"
    assert result.step_results[-1].metadata["lr_feature_mode"] == "woe"
    assert result.modeling_result.training_config["lr_feature_mode"] == "woe"
    assert result.modeling_result.backend_data_mode == "pandas_native_woe"
    assert "pipeline_score" in scored.columns


def test_pipeline_runs_external_woe_step_without_internal_lr_woe(
    sample_modeling_pd: pd.DataFrame,
) -> None:
    pipeline = MarsModelingPipeline(
        target="target",
        features=["x1", "x2"],
        steps=[
            MarsWOEBinningStep(
                name="woe",
                binner=MarsLiteOptBinner(
                    n_bins=4,
                    n_prebins=10,
                    monotonic_trend="auto_asc_desc",
                    n_jobs=1,
                ),
            ),
            MarsSelectionStep(
                name="linear",
                selector=MarsLinearSelector(
                    corr_thr=0.99,
                    enable_vif_filter=False,
                    enable_stepwise=False,
                ),
            ),
            MarsModelingStep(
                name="modeling",
                model_type="lr",
                tune_params={
                    "n_trials": 1,
                    "startup_trials": 1,
                    "warmup_steps": 3,
                    "artifact_dir": None,
                    "max_diff": 100.0,
                },
            ),
        ],
    )

    result = pipeline.fit(sample_modeling_pd)
    transformed = pipeline.transform(sample_modeling_pd)
    scored = pipeline.predict(sample_modeling_pd, pred_col="pipeline_score")

    assert {"x1_woe", "x2_woe"}.issubset(transformed.columns)
    assert all(feature.endswith("_woe") for feature in result.active_features)
    assert result.feature_map == {"x1": "x1_woe", "x2": "x2_woe"}
    assert result.modeling_result is not None
    assert result.step_results[-1].metadata["lr_feature_mode"] == "numeric"
    assert result.modeling_result.training_config["lr_feature_mode"] == "numeric"
    assert result.modeling_result.backend_data_mode == "pandas_numeric"
    assert "pipeline_score" in scored.columns
