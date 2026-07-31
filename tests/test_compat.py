"""Python 3.8 与旧版 Polars 内部兼容层回归测试。"""

from __future__ import annotations

import pickle
import warnings

import polars as pl

from mars._compat import collect_streaming, polars_is_in, remove_suffix
from mars.pipeline.base import MarsStepResult


class _MembershipCandidates:
    def __init__(self) -> None:
        self.imploded = object()
        self.implode_calls = 0

    def implode(self) -> object:
        self.implode_calls += 1
        return self.imploded


class _MembershipValue:
    def __init__(self) -> None:
        self.received: object | None = None

    def is_in(self, candidates: object) -> object:
        self.received = candidates
        return self


class _LazyFrameProbe:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def collect(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self


def test_polars_is_in_uses_plain_series_on_polars_18(monkeypatch) -> None:
    """Polars 1.8 分支不得 implode 候选 Series。"""
    monkeypatch.setattr(pl, "__version__", "1.8.2")
    value = _MembershipValue()
    candidates = _MembershipCandidates()

    assert polars_is_in(value, candidates) is value
    assert value.received is candidates
    assert candidates.implode_calls == 0


def test_polars_is_in_uses_imploded_series_on_modern_polars(monkeypatch) -> None:
    """现代 Polars 分支应保留无弃用警告的 imploded 候选表达式。"""
    monkeypatch.setattr(pl, "__version__", "1.42.0")
    value = _MembershipValue()
    candidates = _MembershipCandidates()

    assert polars_is_in(value, candidates) is value
    assert value.received is candidates.imploded
    assert candidates.implode_calls == 1


def test_polars_is_in_current_runtime_has_no_deprecation_warning() -> None:
    """当前 Polars 路径应返回正确成员关系且不产生弃用警告。"""
    values = pl.Series("value", [1, 2, None])
    candidates = pl.Series("candidate", [2, 3])

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = polars_is_in(values, candidates)

    assert result.to_list() == [False, True, None]


def test_collect_streaming_selects_version_specific_keyword(monkeypatch) -> None:
    """Streaming collect 应按 Polars 版本选择兼容关键字。"""
    legacy = _LazyFrameProbe()
    monkeypatch.setattr(pl, "__version__", "1.8.2")
    assert collect_streaming(legacy) is legacy
    assert legacy.kwargs == {"streaming": True}

    modern = _LazyFrameProbe()
    monkeypatch.setattr(pl, "__version__", "1.42.0")
    assert collect_streaming(modern) is modern
    assert modern.kwargs == {"engine": "streaming"}


def test_remove_suffix_matches_expected_python_39_behavior() -> None:
    """内部后缀移除函数应只移除真实存在的非空后缀。"""
    assert remove_suffix("income_woe", "_woe") == "income"
    assert remove_suffix("income", "_woe") == "income"
    assert remove_suffix("income", "") == "income"


def test_dataclass_result_remains_pickle_compatible_without_slots() -> None:
    """移除 dataclass slots 后结果对象仍应可 pickle 且字段保持一致。"""
    result = MarsStepResult(
        name="selection",
        step_type="selection",
        input_features=["income", "age"],
        output_features=["income"],
        dropped_features=["age"],
        metadata={"threshold": 0.8},
    )

    restored = pickle.loads(pickle.dumps(result))

    assert restored == result
    assert restored.__dict__ == result.__dict__
