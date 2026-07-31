"""文档内容、可运行示例和公开 API 覆盖回归测试。"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import runpy
from pathlib import Path
from urllib.parse import unquote

import pytest
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"
SNIPPETS_ROOT = DOCS_ROOT / "snippets"

BASIC_SNIPPETS = [
    "quickstart.py",
    "data_profiling.py",
    "baseline_evaluation.py",
    "feature_selection.py",
    "monitoring.py",
    "reporting_scorecard.py",
]

REFERENCE_MODULES = {
    "mars.analysis": DOCS_ROOT / "reference" / "analysis.md",
    "mars.feature": DOCS_ROOT / "reference" / "feature.md",
    "mars.monitoring": DOCS_ROOT / "reference" / "monitoring.md",
    "mars.reporting": DOCS_ROOT / "reference" / "reporting.md",
    "mars.scoring": DOCS_ROOT / "reference" / "scoring.md",
    "mars.modeling": DOCS_ROOT / "reference" / "modeling.md",
    "mars.pipeline": DOCS_ROOT / "reference" / "modeling.md",
}

PROHIBITED_CONTEXT_PATTERNS = {
    "五月建箱": re.compile(r"五月建箱"),
    "六月评估": re.compile(r"六月评估"),
    "may_df": re.compile(r"\bmay_df\b"),
    "june_df": re.compile(r"\bjune_df\b"),
    "June feature review": re.compile(r"June feature review"),
    "Agent": re.compile(r"\bAgent\b"),
}

MODULE_STABILITY: dict[str, tuple[str, str, str]] = {
    "Analysis": ("mars.analysis", "Stable", "analysis.md"),
    "Feature": ("mars.feature", "Stable", "feature.md"),
    "Reporting": ("mars.reporting", "Stable", "reporting.md"),
    "Monitoring": ("mars.monitoring", "Experimental", "monitoring.md"),
    "Modeling": ("mars.modeling", "Experimental", "modeling.md"),
    "Pipeline": ("mars.pipeline", "Experimental", "modeling.md"),
    "Scoring": ("mars.scoring", "Experimental", "scoring.md"),
}


@pytest.mark.parametrize("snippet_name", BASIC_SNIPPETS)
def test_basic_documentation_snippet_executes(
    snippet_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """基础文档示例应从独立工作目录执行成功。"""
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(SNIPPETS_ROOT / snippet_name))
    assert namespace


@pytest.mark.docs_ml
def test_modeling_pipeline_snippet_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """安装建模依赖后，Experimental Pipeline 示例应执行成功。"""
    pytest.importorskip("lightgbm")
    pytest.importorskip("optuna")
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(SNIPPETS_ROOT / "modeling_pipeline.py"))
    assert namespace["pipeline_result"].active_features
    assert "model_score" in namespace["scored_df"].columns


@pytest.mark.docs_ml
def test_demo_notebook_is_clean_and_executes() -> None:
    """端到端 Notebook 不提交缓存输出，并可在文档环境中从头执行。"""
    nbformat = pytest.importorskip("nbformat")
    notebook_client = pytest.importorskip("nbclient")
    pytest.importorskip("lightgbm")
    pytest.importorskip("optuna")

    notebook_path = DOCS_ROOT / "demos" / "lgb-modeling-monitoring.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(not cell.outputs for cell in code_cells)

    client = notebook_client.NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    executed = client.execute()
    assert all(
        output.get("output_type") != "error"
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )


def test_internal_documentation_links_resolve() -> None:
    """Markdown 和首页任务卡中的内部链接必须指向现有文档。"""
    markdown_link = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    html_link = re.compile(r'href="([^"]+)"')
    failures: list[str] = []

    for source_path in sorted(DOCS_ROOT.rglob("*.md")):
        text = source_path.read_text(encoding="utf-8")
        links = [*markdown_link.findall(text), *html_link.findall(text)]
        for raw_link in links:
            link = unquote(raw_link.split("#", maxsplit=1)[0].strip())
            if not link or "://" in link or link.startswith(("mailto:", "/")):
                continue

            candidate = (source_path.parent / link).resolve()
            alternatives = [candidate]
            if candidate.suffix == "":
                alternatives.extend(
                    [
                        candidate.with_suffix(".md"),
                        candidate / "index.md",
                    ]
                )
            if not any(path.exists() for path in alternatives):
                relative_source = source_path.relative_to(PROJECT_ROOT)
                failures.append(f"{relative_source}: {raw_link}")

    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(("module_name", "reference_path"), REFERENCE_MODULES.items())
def test_public_module_exports_are_in_reference(
    module_name: str,
    reference_path: Path,
) -> None:
    """所有显式 public 导出都必须进入对应 API Reference。"""
    module = importlib.import_module(module_name)
    exports = set(module.__all__)
    reference = reference_path.read_text(encoding="utf-8")
    directives = set(
        re.findall(rf"^::: {re.escape(module_name)}\.([A-Za-z_][A-Za-z0-9_]*)$", reference, re.M)
    )
    assert exports == directives


def test_user_documentation_has_no_conversation_bound_terms() -> None:
    """用户文档不能重新引入依赖历史讨论的业务命名。"""
    paths = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "CONTRIBUTING.md"]
    paths.extend(DOCS_ROOT.rglob("*.md"))
    paths.extend(DOCS_ROOT.rglob("*.svg"))
    paths.append(DOCS_ROOT / "demos" / "lgb-modeling-monitoring.ipynb")

    failures: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for label, pattern in PROHIBITED_CONTEXT_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{path.relative_to(PROJECT_ROOT)} contains {label}")
    assert not failures, "\n".join(failures)


def test_documented_version_matches_package_metadata() -> None:
    """README、网站入口和安装页必须与包版本保持一致。"""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        project_version = tomllib.load(file)["project"]["version"]

    package_source = (PROJECT_ROOT / "src" / "mars" / "__init__.py").read_text(
        encoding="utf-8"
    )
    package_match = re.search(r'^__version__ = "([^"]+)"$', package_source, re.M)
    assert package_match is not None
    assert package_match.group(1) == project_version == "0.0.26"

    required_install_command = "pip install mars-risk==0.0.26"
    for path in [
        PROJECT_ROOT / "README.md",
        DOCS_ROOT / "index.md",
        DOCS_ROOT / "getting-started" / "installation.md",
        DOCS_ROOT / "getting-started" / "quickstart.md",
    ]:
        assert required_install_command in path.read_text(encoding="utf-8")


def test_python_version_dependency_markers_cover_supported_range() -> None:
    """Python 3.8–3.12 应各自解析到唯一的 Polars 与 scikit-learn 约束。"""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    assert project["requires-python"] == ">=3.8,<3.13"
    requirements = [Requirement(value) for value in project["dependencies"]]
    expected = {
        "3.8": {"polars": "==1.8.2", "scikit-learn": "<1.4,>=1.3.2"},
        "3.9": {"polars": ">=1.33.1", "scikit-learn": "<1.7,>=1.6.1"},
        "3.10": {"polars": ">=1.33.1", "scikit-learn": ">=1.7.2"},
        "3.11": {"polars": ">=1.33.1", "scikit-learn": ">=1.7.2"},
        "3.12": {"polars": ">=1.33.1", "scikit-learn": ">=1.7.2"},
    }

    for python_version, expected_specs in expected.items():
        environment = {"python_version": python_version}
        for package_name, expected_spec in expected_specs.items():
            active = [
                requirement
                for requirement in requirements
                if requirement.name == package_name
                and (
                    requirement.marker is None
                    or requirement.marker.evaluate(environment=environment)
                )
            ]
            assert len(active) == 1
            assert str(active[0].specifier) == expected_spec


def test_readme_restores_dynamic_python_and_download_badges() -> None:
    """README badge 应使用动态 PyPI/PePy 数据并保持约定顺序。"""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    badge_fragments = [
        "img.shields.io/pypi/v/mars-risk",
        "img.shields.io/badge/Docs-GitHub%20Pages",
        "img.shields.io/pypi/pyversions/mars-risk",
        "img.shields.io/pepy/dt/mars-risk",
        "img.shields.io/github/actions/workflow/status/leeesq/mars-risk/test.yml",
        "img.shields.io/github/license/leeesq/mars-risk",
    ]
    positions = [readme.index(fragment) for fragment in badge_fragments]
    assert positions == sorted(positions)
    assert 'href="https://pepy.tech/project/mars-risk"' in readme


def test_brand_hero_uses_complete_visual_asset_stack() -> None:
    """README 和网站首页必须保留完整的四层品牌首屏。"""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    homepage = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    for asset_name in [
        "mars-logo.svg",
        "mars-wordmark.svg",
        "mars-tagline.svg",
        "mars-workflow.svg",
    ]:
        assert f'docs/assets/{asset_name}' in readme
        assert f'assets/{asset_name}' in homepage
    assert "mars-home-hero--compact" not in homepage


def test_docs_workflow_deploys_pages_after_main_validation() -> None:
    """Docs 工作流应从 main 验证并部署普通提交或指定 release tag。"""
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "docs.yml").read_text(
        encoding="utf-8"
    )
    assert "release_tag:" in workflow
    assert "ref: ${{ inputs.release_tag || github.ref }}" in workflow
    assert '--release-tag "${{ inputs.release_tag }}"' in workflow
    assert "github.event_name == 'push' || github.event_name == 'workflow_dispatch'" in workflow
    assert "needs: build" in workflow
    for action in [
        "actions/configure-pages@v5",
        "actions/upload-pages-artifact@v3",
        "actions/deploy-pages@v4",
    ]:
        assert action in workflow


def test_release_workflow_dispatches_docs_from_main() -> None:
    """Release workflow 应显式指定仓库，并从 main 派发 Pages 部署。"""
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )
    assert "actions: write" in workflow
    assert "gh workflow run docs.yml" in workflow
    assert '--repo "$GITHUB_REPOSITORY"' in workflow
    assert "--ref main" in workflow
    assert '--raw-field "release_tag=$RELEASE_TAG"' in workflow
    assert "actions/deploy-pages" not in workflow


@pytest.mark.parametrize(
    ("module_label", "module_name", "expected_status", "reference_name"),
    [
        (label, *settings)
        for label, settings in MODULE_STABILITY.items()
    ],
)
def test_module_stability_is_consistent(
    module_label: str,
    module_name: str,
    expected_status: str,
    reference_name: str,
) -> None:
    """模块 docstring、兼容性表、API 索引和 Reference 必须使用同一状态。"""
    module = importlib.import_module(module_name)
    module_docstring = inspect.getdoc(module) or ""
    assert f"MARS {expected_status}" in module_docstring

    stability = (DOCS_ROOT / "project" / "stability.md").read_text(encoding="utf-8")
    stability_match = re.search(
        rf"^\| {re.escape(module_label)} \| (Stable|Experimental) \|",
        stability,
        re.M,
    )
    assert stability_match is not None
    assert stability_match.group(1) == expected_status

    index_label = "Modeling / Pipeline" if module_label in {"Modeling", "Pipeline"} else module_label
    reference_index = (DOCS_ROOT / "reference" / "index.md").read_text(encoding="utf-8")
    index_match = re.search(
        rf"^\| \[{re.escape(index_label)}\]\([^)]+\) \| (Stable|Experimental) \|",
        reference_index,
        re.M,
    )
    assert index_match is not None
    assert index_match.group(1) == expected_status

    reference = (DOCS_ROOT / "reference" / reference_name).read_text(encoding="utf-8")
    expected_marker = (
        "**状态：Stable。**"
        if expected_status == "Stable"
        else '!!! warning "Experimental"'
    )
    assert expected_marker in reference


def test_stability_summaries_and_mixed_guides_are_explicit() -> None:
    """摘要和混合 Guide 应区分 Stable Reporting 与 Experimental 能力。"""
    stable_labels = "、".join(
        label for label, (_, status, _) in MODULE_STABILITY.items() if status == "Stable"
    )
    experimental_labels = "、".join(
        label
        for label, (_, status, _) in MODULE_STABILITY.items()
        if status == "Experimental"
    )
    for path in [PROJECT_ROOT / "README.md", DOCS_ROOT / "index.md"]:
        text = path.read_text(encoding="utf-8")
        assert stable_labels in text
        assert experimental_labels in text

    monitoring_guide = (DOCS_ROOT / "user-guide" / "monitoring.md").read_text(
        encoding="utf-8"
    )
    assert '!!! warning "Experimental"' in monitoring_guide
    assert "report 字段和报警结果增加契约测试" in monitoring_guide

    reporting_guide = (
        DOCS_ROOT / "user-guide" / "reports-and-exports.md"
    ).read_text(encoding="utf-8")
    assert '!!! info "Reporting：Stable"' in reporting_guide
    assert '!!! warning "Scoring：Experimental"' in reporting_guide

    report_objects = (DOCS_ROOT / "reference" / "report-objects.md").read_text(
        encoding="utf-8"
    )
    assert "| `MarsMonitoringReport` | Experimental |" in report_objects
    assert "| `MarsScorecard` | Experimental |" in report_objects


@pytest.mark.parametrize("module_name", ["mars.monitoring", "mars.scoring"])
def test_experimental_public_objects_are_labeled(module_name: str) -> None:
    """Monitoring/Scoring 的全部公开对象必须在 docstring 中声明 Experimental。"""
    module = importlib.import_module(module_name)
    for export_name in module.__all__:
        public_object = getattr(module, export_name)
        docstring = inspect.getdoc(public_object) or ""
        assert "Experimental" in docstring, f"{module_name}.{export_name} lacks status"
