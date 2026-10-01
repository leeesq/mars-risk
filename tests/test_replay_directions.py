from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import polars as pl
import pytest

from mars.modeling import MarsModelReplayResult, MarsModelReplayRunner
from mars.modeling.evaluation.metrics import MetricDirection
from mars.modeling.workflows import replay as replay_module


def _run(
    monkeypatch: pytest.MonkeyPatch,
    historical: dict[str, MetricDirection],
    override: dict[str, MetricDirection] | None,
    trial_nums: list[int] | None = None,
) -> tuple[MarsModelReplayResult, dict[str, Any]]:
    result = SimpleNamespace(
        model_type="lr",
        features=["x"],
        target="y",
        dataset_flag_col="split",
        categorical_features=[],
        optimize_metric="ks",
        training_config={},
        metric_names=["ks", "cost"],
        metric_directions=historical,
        history_table=pd.DataFrame(
            {
                "trial_num": [1, 2],
                "trial_state": ["COMPLETE"] * 2,
                "is_valid": [True] * 2,
                "oot_cost": [5, 20],
                "val_cost": [5, 20],
            }
        ),
        replay_candidates=[],
        importance_table=pd.DataFrame(),
        retained_models={1: object(), 2: object()},
    )
    captured: dict[str, Any] = {}

    def backend(self: Any, df: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(
            training_metric="ks",
            backend_data_mode="test",
            get_best_iteration=lambda model: 1,
        )

    monkeypatch.setattr(MarsModelReplayRunner, "_build_backend", backend)
    monkeypatch.setattr(replay_module.ModelPredictor, "predict", lambda self, df, **kw: df)
    monkeypatch.setattr(replay_module.MarsModelEvaluator, "evaluate", lambda self, df, **kw: None)
    run = MarsModelReplayRunner().replay(
        result,
        pl.DataFrame({"x": [1], "y": [0], "split": ["train"]}),
        sort_metric="CoSt",
        top_k=2,
        retrain=False,
        metric_directions=override,
        trial_nums=trial_nums,
    )
    return run, captured


@pytest.mark.parametrize(
    ("historical", "override", "expected"),
    [
        ({"cost": "maximize"}, {"cost": "minimize"}, [1, 2]),
        ({"cost": "minimize"}, {"cost": "maximize"}, [2, 1]),
        ({"cost": "minimize"}, None, [1, 2]),
        ({"cost": "minimize"}, {}, [1, 2]),
        ({}, None, [2, 1]),
        ({"cost": "maximize", "ks": "minimize"}, {"CoSt": "MINIMIZE"}, [1, 2]),
    ],
)
def test_final_directions_control_ranking_and_backend(
    monkeypatch: pytest.MonkeyPatch,
    historical: dict[str, MetricDirection],
    override: dict[str, MetricDirection] | None,
    expected: list[int],
) -> None:
    run, captured = _run(monkeypatch, historical, override)
    assert run.ranking_table["trial_num"].tolist() == expected
    assert captured["metric_directions"] == run.metric_directions
    assert run.metric_directions["ks"] == historical.get("ks", "maximize")


def test_explicit_trials_keep_order(monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = _run(monkeypatch, {"cost": "maximize"}, {"cost": "minimize"}, [2, 1])
    assert run.leaderboard_table["trial_num"].tolist() == [2, 1]


def test_invalid_direction_rejected_before_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="Unsupported metric direction"):
        _run(monkeypatch, {"cost": "invalid"}, None)


def test_replay_direction_artifact_roundtrip(tmp_path: Any) -> None:
    run = MarsModelReplayResult(
        "lr",
        pd.DataFrame({"trial_num": [1]}),
        pd.DataFrame({"rank": [1]}),
        {},
        None,
        {},
        {},
        metric_directions={"cost": "minimize"},
    )
    path = run.export_artifact(str(tmp_path))
    assert MarsModelReplayResult.from_artifact(str(path)).metric_directions == run.metric_directions
