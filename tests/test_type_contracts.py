from __future__ import annotations

import inspect
from pathlib import Path

import polars as pl
import pytest

from mars.feature import (
    MarsImportanceSelector,
    MarsLinearSelector,
    MarsLiteOptBinner,
    MarsOptimalBinner,
    MarsStatsSelector,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_selector_fit_signatures_keep_their_public_contracts() -> None:
    stats_fit = inspect.signature(MarsStatsSelector.fit)
    stats_fit_transform = inspect.signature(MarsStatsSelector.fit_transform)
    linear_fit = inspect.signature(MarsLinearSelector.fit)
    importance_fit = inspect.signature(MarsImportanceSelector.fit)
    importance_fit_transform = inspect.signature(MarsImportanceSelector.fit_transform)

    assert stats_fit.parameters["target"].kind is inspect.Parameter.KEYWORD_ONLY
    assert stats_fit_transform.parameters["target"].kind is inspect.Parameter.KEYWORD_ONLY
    assert linear_fit.parameters["y"].default is inspect.Parameter.empty
    assert importance_fit.parameters["y"].default is None
    assert importance_fit_transform.parameters["y"].default is None
    assert "importance_table" in importance_fit_transform.parameters


@pytest.mark.parametrize("binner", [MarsLiteOptBinner(), MarsOptimalBinner()])
def test_supervised_binners_missing_y_raise_business_error(
    binner: MarsLiteOptBinner | MarsOptimalBinner,
) -> None:
    X = pl.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]})

    with pytest.raises(ValueError, match="requires y"):
        binner.fit(X)


def test_mypy_config_has_no_business_overrides_or_source_ignores() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.8 compatibility
        import tomli as tomllib

    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        config = tomllib.load(file)

    dev_dependencies = config["project"]["optional-dependencies"]["dev"]
    assert "mypy==1.13.0" in dev_dependencies
    overrides = config["tool"]["mypy"].get("overrides", [])
    overridden_modules = [
        module
        for override in overrides
        for module in override.get("module", [])
    ]
    assert not any(module == "mars" or module.startswith("mars.") for module in overridden_modules)

    ignored_lines = []
    for source_path in (PROJECT_ROOT / "src" / "mars").rglob("*.py"):
        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if "type: ignore" in line:
                ignored_lines.append(f"{source_path}:{line_number}")
    assert ignored_lines == []
