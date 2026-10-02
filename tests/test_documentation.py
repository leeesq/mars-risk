"""文档内容、可运行示例和公开 API 覆盖回归测试。"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import math
import re
import runpy
from pathlib import Path
from typing import Any
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
    "readme_quickstart.py",
    "minimal_report.py",
    "external_candidate_selection.py",
    "quickstart.py",
    "data_profiling.py",
    "baseline_evaluation.py",
    "multitarget_risk.py",
    "feature_selection.py",
    "rule_mining.py",
    "monitoring.py",
    "reporting_scorecard.py",
    "report_queries.py",
    "report_presentations.py",
    "portable_reports.py",
]

REFERENCE_MODULES = {
    "mars.analysis": DOCS_ROOT / "reference" / "analysis.md",
    "mars.feature": DOCS_ROOT / "reference" / "feature.md",
    "mars.rule": DOCS_ROOT / "reference" / "rule.md",
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
}

MODULE_STABILITY: dict[str, tuple[str, str, str]] = {
    "Analysis": ("mars.analysis", "Stable", "analysis.md"),
    "Feature": ("mars.feature", "Stable", "feature.md"),
    "Rule": ("mars.rule", "Experimental", "rule.md"),
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


def test_historical_notebook_is_clean_and_records_current_install_boundary() -> None:
    """即使可选模型依赖缺失，历史 Notebook 的发布内容与安装边界仍需验证。"""
    notebook_path = DOCS_ROOT / "demos" / "lgb-modeling-monitoring.ipynb"
    notebook: dict[str, Any] = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code_cells)
    prose = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    assert "历史／保留案例" in prose and "暂停功能迭代" in prose
    assert 'python -m pip install -e ".[ml,tuning,notebook]"' in prose
    assert "execute: false" in prose


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


def test_deimos_rule_migration_matrix_accounts_for_all_source_regressions() -> None:
    """来源快照的 44 项回归必须逐项标记覆盖、替代或计划内删除。"""
    matrix_path = DOCS_ROOT / "project" / "deimos-rule-migration.md"
    matrix = matrix_path.read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| (\d+) \| `test_[^`]+` \|.+\| "
        r"(Covered|Replaced|Removed by design) \|$",
        matrix,
        re.M,
    )

    assert [int(number) for number, _ in rows] == list(range(1, 45))
    assert "e6714c5e795054e44f0c58ad7097668b4117b4a2" in matrix


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
    assert package_match.group(1) == project_version == "0.0.28"

    # main 的新能力从源码安装，不能用未发布版本的 PyPI 命令充当可用性证明。
    source_install = re.compile(
        r'pip install "git\+https://github\.com/leeesq/mars-risk\.git(?:@[A-Za-z0-9._/-]+)?"'
    )
    for path in [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "README.en.md",
        DOCS_ROOT / "index.md",
        DOCS_ROOT / "getting-started" / "installation.md",
        DOCS_ROOT / "getting-started" / "quickstart.md",
    ]:
        assert source_install.search(path.read_text(encoding="utf-8")), path


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


def test_default_dependencies_include_pandas_styler_runtime() -> None:
    """默认安装必须包含基础报告和 selector 所需的 Pandas Styler 运行时。"""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        requirements = [
            Requirement(value)
            for value in tomllib.load(file)["project"]["dependencies"]
        ]

    jinja_requirements = [
        requirement for requirement in requirements if requirement.name == "jinja2"
    ]
    assert len(jinja_requirements) == 1
    assert str(jinja_requirements[0].specifier) == ">=3.1.2"

    python38_constraints = (PROJECT_ROOT / "constraints" / "python38.txt").read_text(
        encoding="utf-8"
    )
    assert "jinja2==3.1.6" in python38_constraints
    assert "markupsafe==2.1.5" in python38_constraints


@pytest.mark.parametrize("readme_name", ["README.md", "README.en.md"])
def test_readme_restores_dynamic_python_and_download_badges(readme_name: str) -> None:
    """README badge 应使用动态 PyPI/PePy 数据并保持约定顺序。"""
    readme = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
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


def test_brand_hero_preserves_identity_and_complete_badges() -> None:
    """保留 Logo、英文全称与六类徽章，定位用可搜索的文字。"""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    homepage = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    english_readme = (PROJECT_ROOT / "README.en.md").read_text(encoding="utf-8")
    for asset_name in [
        "mars-logo.svg",
        "mars-wordmark.svg",
    ]:
        assert f'docs/assets/{asset_name}' in readme
        assert f'docs/assets/{asset_name}' in english_readme
        assert f'assets/{asset_name}' in homepage
    for text in (readme, homepage):
        assert "面向人和 AI Agent 的高性能风控分析工具箱" in text
        assert "数据画像 · 分箱评估 · 特征筛选 · 相关性分析 · 模型分交叉 · 规则挖掘" in text
        for label in ("PyPI", "Docs", "Python", "Downloads", "CI", "License"):
            assert f'alt="{label}"' in text
    assert "A high-performance risk analysis toolkit for humans and AI agents" in english_readme
    for label in ("PyPI", "Docs", "Python", "Downloads", "CI", "License"):
        assert f'alt="{label}"' in english_readme
    for text in (readme, english_readme, homepage):
        for fragment in (
            "img.shields.io/pypi/v/mars-risk",
            "img.shields.io/badge/Docs-GitHub%20Pages",
            "img.shields.io/pypi/pyversions/mars-risk",
            "img.shields.io/pepy/dt/mars-risk",
            "img.shields.io/github/actions/workflow/status/leeesq/mars-risk/test.yml",
            "img.shields.io/github/license/leeesq/mars-risk",
        ):
            assert fragment in text
        links = dict(
            (label, href) for href, label in re.findall(
                r'<a\b[^>]*href="([^"]+)"[^>]*>\s*<img\b[^>]*alt="(PyPI|Docs|Python|Downloads|CI|License)"',
                text,
            )
        )
        for label, target in {
            "PyPI": "https://pypi.org/project/mars-risk/",
            "Docs": "https://leeesq.github.io/mars-risk/",
            "Python": "https://pypi.org/project/mars-risk/",
            "Downloads": "https://pepy.tech/project/mars-risk",
            "CI": "https://github.com/leeesq/mars-risk/actions/workflows/test.yml",
        }.items():
            assert links[label] == target
        assert links["License"] in {"LICENSE", "https://github.com/leeesq/mars-risk/blob/main/LICENSE"}


def test_readme_hero_uses_same_native_binning_chart_in_both_languages() -> None:
    """两语门面使用可追溯原生分箱图，并提供高清文件；不再复合交叉截图。"""
    for name in ("README.md", "README.en.md"):
        text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        png = "docs/assets/cases/binning-native-main-score.png"
        svg = "docs/assets/cases/binning-native-main-score.svg"
        svg_targets = (svg, "https://leeesq.github.io/mars-risk/assets/cases/binning-native-main-score.svg")
        assert png in text
        assert any(f'href="{target}"' in text or f"]({target})" in text for target in svg_targets)
        assert (PROJECT_ROOT / png).is_file() and (PROJECT_ROOT / svg).is_file()
        assert text.index(png) < text.index("```python")
        assert "readme-preview" not in text
        assert "binning-risk.png" not in text
        assert 'align="center"' in text


def test_rule_case_agent_download_includes_rule_analysis_instructions() -> None:
    """规则案例的外部任务入口应实际指向规则审计，不误链仅交叉格子的任务。"""
    page = DOCS_ROOT / "demos/rule-evidence.md"
    text = page.read_text(encoding="utf-8")
    links = re.findall(r"\[外部 Agent[^\]]*\]\(([^)]+)\)", text)
    assert links
    for link in links:
        target = (page.parent / link.split("#", maxsplit=1)[0]).resolve()
        material = target.read_text(encoding="utf-8")
        assert "rules.marsreport" in material
        assert "candidates" in material and "validation" in material


def test_bilingual_readme_quickstart_uses_one_executable_source() -> None:
    """两语短代码必须逐行同步共享 snippet，语言跳转和本地资源真实存在。"""
    snippet = (SNIPPETS_ROOT / "readme_quickstart.py").read_text(encoding="utf-8")
    match = re.search(
        r"# --8<-- \[start:quickstart\]\n(.*?)\n# --8<-- \[end:quickstart\]",
        snippet,
        re.S,
    )
    assert match is not None
    quickstart = match.group(1).strip()
    readmes = {
        name: (PROJECT_ROOT / name).read_text(encoding="utf-8")
        for name in ("README.md", "README.en.md")
    }
    for name, text in readmes.items():
        blocks = re.findall(r"```python\n(.*?)\n```", text, re.S)
        assert quickstart in [block.strip() for block in blocks], name
        assert "docs/snippets/readme_quickstart.py" in text, name
        other = "README.en.md" if name == "README.md" else "README.md"
        assert re.search(rf"\]\({re.escape(other)}\)", text), name
        for path in re.findall(r'(?:src|srcset|href)="(docs/[^"?#]+)"', text):
            assert (PROJECT_ROOT / path).is_file(), f"{name}: missing {path}"
        for path in re.findall(r"!?\[[^\]]*\]\((docs/[^)#?]+)", text):
            assert (PROJECT_ROOT / path).is_file(), f"{name}: missing {path}"


@pytest.mark.parametrize("readme_name", ["README.md", "README.en.md"])
def test_readme_snapshot_queries_and_static_excel_match_current_report(
    readme_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两语第二代码块独立加载结果，静态 Excel 应包含当前快照的真实表值。"""
    import openpyxl

    from mars.reporting import ReportSnapshot

    monkeypatch.chdir(tmp_path)
    generated: dict[str, Any] = runpy.run_path(str(SNIPPETS_ROOT / "readme_quickstart.py"))
    source_report = generated["report"]
    readme = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", readme, re.S)
    other_name = "README.en.md" if readme_name == "README.md" else "README.md"
    other_blocks = re.findall(
        r"```python\n(.*?)\n```", (PROJECT_ROOT / other_name).read_text(encoding="utf-8"), re.S,
    )
    assert blocks[1].strip() == other_blocks[1].strip()
    # 消费代码使用新的命名空间，仅能从文件恢复，不能复用 quickstart 的宽表或分析器。
    consumed: dict[str, Any] = {}
    exec(compile(blocks[1], readme_name + ":snapshot-consumer", "exec"), consumed)
    snapshot = consumed["report"]
    assert isinstance(snapshot, ReportSnapshot)
    assert snapshot.report_id == source_report.report_id
    page = consumed["page"]
    next_page = consumed["next_page"]
    assert page["returned_rows"] == next_page["returned_rows"] == 1
    assert next_page["reference"]["query"]["offset"] == page["next_offset"]
    assert page["data"].to_dicts() != next_page["data"].to_dicts()
    context_json = consumed["context_json"]
    context: dict[str, Any] = json.loads(context_json)
    json.dumps(context, allow_nan=False)
    assert len(context_json) <= 16000
    assert context["description"]["report_id"] == snapshot.report_id
    assert context["evidence"]
    for evidence in context["evidence"]:
        assert evidence["report_id"] == snapshot.report_id
        replay = snapshot.get_table(evidence["reference"], **evidence["query"])
        assert replay.to_dicts() == evidence["rows"]

    snapshot.write_excel("risk_report.xlsx")
    workbook = openpyxl.load_workbook("risk_report.xlsx", read_only=True, data_only=True)
    try:
        sheet_rows = workbook["000_summary"].iter_rows(values_only=True)
        header = list(next(sheet_rows))
        exported = list(sheet_rows)
        expected = snapshot.get_table("summary").to_dicts()
        assert len(exported) == len(expected) == 2
        for exported_row, expected_row in zip(exported, expected):
            assert exported_row[header.index("feature")] == expected_row["feature"]
            for field in ("iv", "ks"):
                assert math.isclose(exported_row[header.index(field)], expected_row[field], rel_tol=1e-12)
    finally:
        workbook.close()


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


def test_distribution_workflows_reuse_the_verified_artifacts() -> None:
    """PR 与 Release 必须只构建一次，并在 3.8/3.12 验证同一 wheel。"""
    test_workflow = (PROJECT_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    publish_workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "publish.yml"
    ).read_text(encoding="utf-8")

    assert "distribution:" in test_workflow
    assert "needs: distribution" in test_workflow
    assert 'python-version: ["3.8", "3.12"]' in test_workflow
    assert "scripts/verify_distribution.py" in test_workflow
    assert "scripts/smoke_installed_package.py" in test_workflow

    assert "needs: build" in publish_workflow
    assert "needs: verify-wheel" in publish_workflow
    assert "needs: publish" in publish_workflow
    assert publish_workflow.count("python -m build") == 1
    assert publish_workflow.count("id-token: write") == 1
    assert 'python-version: ["3.8", "3.12"]' in publish_workflow
    assert "Publish the exact verified artifacts" in publish_workflow


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
    english = (PROJECT_ROOT / "README.en.md").read_text(encoding="utf-8")
    statements = re.findall(r"([^\n.]+?)\bare (Stable|Experimental)\b", english)
    for status in ("Stable", "Experimental"):
        expected = {label for label, (_, value, _) in MODULE_STABILITY.items() if value == status}
        stated = {
            label for sentence, stated_status in statements if stated_status == status
            for label in MODULE_STABILITY if re.search(rf"\b{label}\b", sentence)
        }
        assert expected == stated
    assert re.search(r"Monitoring[^\n.]+Modeling[^\n.]+Pipeline[^\n.]+Scoring[^\n.]+paused", english)

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
