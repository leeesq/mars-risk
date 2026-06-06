from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_private_docstrings.py"


def _run_checker(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(path)],
        capture_output=True,
        check=False,
        text=True,
    )


def _write_module(tmp_path: Path, source: str) -> Path:
    module_path = tmp_path / "sample.py"
    module_path.write_text(source, encoding="utf-8")
    return module_path


def test_complex_private_without_docstring_fails(tmp_path: Path) -> None:
    body = "\n".join(f"    value += {index}" for index in range(16))
    module_path = _write_module(
        tmp_path,
        f"""
def _complex_private():
    value = 0
{body}
    return value
""",
    )

    result = _run_checker(module_path)

    assert result.returncode == 1
    assert "缺少中文短 docstring" in result.stderr
    assert "有效代码行数" in result.stderr


def test_complex_private_with_short_docstring_passes(tmp_path: Path) -> None:
    body = "\n".join(f"    value += {index}" for index in range(16))
    module_path = _write_module(
        tmp_path,
        f'''
def _complex_private():
    """汇总多步中间结果，避免调用方重复处理边界。"""
    value = 0
{body}
    return value
''',
    )

    result = _run_checker(module_path)

    assert result.returncode == 0
    assert "检查通过" in result.stdout


def test_simple_private_without_docstring_and_dunder_are_skipped(tmp_path: Path) -> None:
    body = "\n".join(f"        value += {index}" for index in range(20))
    module_path = _write_module(
        tmp_path,
        f"""
def _small_private(value: int) -> int:
    return value + 1


class Demo:
    def __repr__(self) -> str:
        value = 0
{body}
        return str(value)
""",
    )

    result = _run_checker(module_path)

    assert result.returncode == 0


def test_effective_lines_ignore_empty_lines_comments_and_use_strict_threshold(
    tmp_path: Path,
) -> None:
    body = "\n\n".join(f"    # comment {index}\n    value += {index}" for index in range(13))
    module_path = _write_module(
        tmp_path,
        f"""
def _threshold_private():
    value = 0
{body}
    return value
""",
    )

    result = _run_checker(module_path)

    assert result.returncode == 0


def test_full_docstring_on_simple_private_warns_without_failing(tmp_path: Path) -> None:
    module_path = _write_module(
        tmp_path,
        '''
def _simple_private(value: int) -> int:
    """转换输入值。

    Parameters
    ----------
    value : int
        输入值。

    Returns
    -------
    int
        转换后的值。
    """
    return value + 1
''',
    )

    result = _run_checker(module_path)

    assert result.returncode == 0
    assert "WARNING" in result.stdout
    assert "可考虑精简" in result.stdout
