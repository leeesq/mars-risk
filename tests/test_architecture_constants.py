"""内部常量和模块依赖方向的架构回归测试。"""

from pathlib import Path

from mars.analysis import profile_risk as package_profile_risk
from mars.analysis.evaluator import profile_risk as evaluator_profile_risk
from mars.core.constants import (
    DIVISION_EPSILON,
    FLOAT_TOLERANCE,
    METRIC_EPSILON,
    MIN_VARIANCE,
    PROBABILITY_EPSILON,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "mars"


def test_numeric_constants_keep_semantic_values() -> None:
    """内部数值稳定性常量按语义分层，不退化成单一 epsilon。"""
    assert METRIC_EPSILON == 1e-6
    assert DIVISION_EPSILON == 1e-9
    assert FLOAT_TOLERANCE == 1e-12
    assert PROBABILITY_EPSILON == 1e-15
    assert MIN_VARIANCE == 1e-6


def test_hardcoded_epsilon_only_lives_in_constants_module() -> None:
    """源码中的内部 epsilon 数值只能在 constants 模块中定义。"""
    allowed = SRC_ROOT / "core" / "constants.py"
    forbidden_tokens = ("1e-6", "1e-9", "1e-12", "1e-15")
    violations: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        if path == allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            if token in text:
                violations.append(f"{path.relative_to(PROJECT_ROOT)} contains {token}")
    assert not violations


def test_feature_module_does_not_depend_on_modeling_optional_imports() -> None:
    """feature 模块不能为了可选依赖工具反向依赖 modeling。"""
    selector_source = (SRC_ROOT / "feature" / "selector.py").read_text(encoding="utf-8")
    assert "from mars.modeling.utils import require_optional_module" not in selector_source
    assert "from mars.utils.imports import require_optional_module" in selector_source


def test_profile_risk_keeps_original_public_import_path() -> None:
    """profile_risk 拆到内部模块后仍保留原 public 导入路径。"""
    assert package_profile_risk is evaluator_profile_risk
