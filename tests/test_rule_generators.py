"""五类规则生成器测试。"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

import mars.rule.generators as generator_module
from mars.rule import (
    MarsCombinationRuleGenerator,
    MarsForestRuleGenerator,
    MarsGBDTRuleGenerator,
    MarsIsolationRuleGenerator,
    MarsTreeRuleGenerator,
)


def _generator_frame() -> pl.DataFrame:
    """构造小型可复现生成器数据。"""
    rng = np.random.default_rng(42)
    first = rng.normal(size=300)
    second = rng.normal(size=300)
    target = ((first > 0.8) | (second < -1.0)).astype("int8")
    first[0] = np.nan
    return pl.DataFrame({"first": first, "second": second, "target": target}).with_columns(
        pl.when(pl.col("first").is_nan()).then(None).otherwise(pl.col("first")).alias("first")
    )


def test_combination_generator_is_reproducible_and_budgeted() -> None:
    """组合生成器应稳定执行且尊重候选预算。"""
    frame = _generator_frame()
    first = MarsCombinationRuleGenerator(n_bins=4, max_candidates=10, random_state=7)
    second = MarsCombinationRuleGenerator(n_bins=4, max_candidates=10, random_state=7)
    rules_a = first.generate(frame, target="target")
    rules_b = second.generate(frame, target="target")
    assert [rule.rule_id for rule in rules_a] == [rule.rule_id for rule in rules_b]
    assert 0 < len(rules_a) <= 10
    assert any(" OR " in rule.expression for rule in rules_a)


def test_tree_generator_has_explicit_missing_paths_and_is_reproducible() -> None:
    """树生成规则必须把缺失与非缺失路径写入 DSL。"""
    frame = _generator_frame()
    first = MarsTreeRuleGenerator(n_trees=3, max_depths=[2], random_state=7, n_jobs=1)
    second = MarsTreeRuleGenerator(n_trees=3, max_depths=[2], random_state=7, n_jobs=1)
    rules_a = first.generate(frame, target="target")
    rules_b = second.generate(frame, target="target")
    expressions = [rule.expression for rule in rules_a]
    assert [rule.rule_id for rule in rules_a] == [rule.rule_id for rule in rules_b]
    assert any("IS MISSING" in expression for expression in expressions)
    assert any("IS NOT MISSING" in expression for expression in expressions)
    assert all(
        metadata["training_sample_count"] == frame.height
        for metadata in first.tree_metadata_
    )


def test_tree_generator_honors_n_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """浅层树必须把 n_jobs 传给实际并行执行器。"""
    observed: dict[str, Any] = {}

    def fake_parallel(*, n_jobs: int, backend: str) -> Any:
        observed.update({"n_jobs": n_jobs, "backend": backend})

        def run(tasks: Any) -> list[Any]:
            return [function(*args, **kwargs) for function, args, kwargs in tasks]

        return run

    monkeypatch.setattr(generator_module, "Parallel", fake_parallel)
    rules = MarsTreeRuleGenerator(
        n_trees=2,
        max_depths=[2],
        random_state=7,
        n_jobs=3,
    ).generate(_generator_frame(), target="target")

    assert rules
    assert observed == {"n_jobs": 3, "backend": "threading"}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MarsForestRuleGenerator(n_estimators=4, max_depth=2, random_state=11, n_jobs=1),
        lambda: MarsGBDTRuleGenerator(n_estimators=4, max_depth=2, random_state=13, n_jobs=1),
        lambda: MarsIsolationRuleGenerator(
            n_estimators=8,
            max_samples=0.8,
            path_depth_limit=3,
            min_leaf_samples=5,
            max_candidates=30,
            random_state=17,
            n_jobs=1,
        ),
    ],
)
def test_ensemble_generators_are_reproducible(factory: Any) -> None:
    """森林、GBDT 和孤立森林生成结果应由随机种子完全锁定。"""
    frame = _generator_frame()
    rules_a = factory().generate(frame, target="target")
    rules_b = factory().generate(frame, target="target")
    assert rules_a
    assert [rule.rule_id for rule in rules_a] == [rule.rule_id for rule in rules_b]


def test_lightgbm_and_optuna_use_unified_optional_dependency_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """可选后端缺失时应指向 Mars extras 而非 deimos 包。"""
    frame = _generator_frame()

    def missing_module(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(str(kwargs.get("extra_hint")))

    monkeypatch.setattr(generator_module, "require_optional_module", missing_module)
    with pytest.raises(ImportError, match=r"mars-risk\[ml\]"):
        MarsGBDTRuleGenerator(
            backend="lightgbm", n_estimators=2, max_depth=2
        ).generate(frame, target="target")
    with pytest.raises(ImportError, match=r"mars-risk\[tuning\]"):
        MarsTreeRuleGenerator(
            n_trees=1,
            max_depths=[2],
            tuning_backend="optuna",
        ).generate(frame, target="target")


@pytest.mark.optional_ml
def test_lightgbm_optional_backend_generates_rules_when_installed() -> None:
    """安装 ml extra 后 LightGBM dump 路径应进入统一 MarsRule。"""
    pytest.importorskip("lightgbm")
    rules = MarsGBDTRuleGenerator(
        backend="lightgbm",
        n_estimators=3,
        max_depth=2,
        random_state=19,
        n_jobs=1,
    ).generate(_generator_frame(), target="target")
    assert rules
    assert all(rule.source == "gbdt" for rule in rules)


@pytest.mark.optional_ml
def test_optuna_optional_backend_tunes_tree_when_installed() -> None:
    """安装 tuning extra 后浅层树应完成显式 Optuna 调参。"""
    pytest.importorskip("optuna")
    generator = MarsTreeRuleGenerator(
        n_trees=1,
        max_depths=[2],
        tuning_backend="optuna",
        tuning_trials=1,
        random_state=23,
        n_jobs=1,
    )
    rules = generator.generate(_generator_frame(), target="target")
    assert rules
    assert generator.tree_metadata_[0]["tuning"]["status"] == "completed"
    assert generator.tree_metadata_[0]["tuning"]["cv_folds"] >= 2
    assert 0.0 <= generator.tree_metadata_[0]["tuning"]["best_cv_roc_auc"] <= 1.0
