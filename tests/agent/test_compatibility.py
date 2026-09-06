from __future__ import annotations

import ast
import builtins
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


def test_agent_sources_parse_as_python310() -> None:
    for path in (ROOT / "src/mars/agent").glob("*.py"):
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=(3, 10),
        )


def test_old_python_guard_runs_before_loading_agent_submodules() -> None:
    source = (ROOT / "src/mars/agent/__init__.py").read_text(encoding="utf-8")

    def guarded_import(name: str, *args, **kwargs):
        if name == "sys":
            return SimpleNamespace(version_info=(3, 9, 0))
        raise AssertionError(f"unexpected import before version guard: {name}")

    namespace = {"__builtins__": {**vars(builtins), "__import__": guarded_import}}
    with pytest.raises(ImportError, match="Python >=3.10"):
        exec(compile(source, "agent/__init__.py", "exec"), namespace)


def test_core_and_agent_import_without_loading_optional_sdk(tmp_path: Path) -> None:
    script = """
import sys
import mars
assert 'mars.agent' not in sys.modules
import mars.agent
assert 'openai' not in sys.modules
assert 'MarsRiskAgent' not in mars.__all__
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_agent_dependency_is_optional_and_python310_gated() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert project["requires-python"] == ">=3.8,<3.13"
    assert not any(
        Requirement(item).name == "openai" for item in project["dependencies"]
    )
    sdk = next(
        Requirement(item)
        for item in project["optional-dependencies"]["agent"]
        if Requirement(item).name == "openai"
    )
    assert not sdk.marker.evaluate({"python_version": "3.9"})
    assert sdk.marker.evaluate({"python_version": "3.10"})


def test_agent_public_exports_have_reference_documentation() -> None:
    import mars.agent as module

    source = (ROOT / "docs/reference/agent.md").read_text(encoding="utf-8")
    assert set(module.__all__) == set(
        re.findall(r"^::: mars\.agent\.(\w+)$", source, re.M)
    )


def test_agent_documented_example_runs_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    values = runpy.run_path(str(ROOT / "docs/snippets/agent_monitoring.py"))
    assert values["monitor_result"].success
    assert values["observation_result"].success
    assert values["observation_table"].height > 0
