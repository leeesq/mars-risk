"""校验包元数据、源码版本与可选 release tag 是否一致。"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_project_version(project_root: Path = PROJECT_ROOT) -> str:
    """读取 ``pyproject.toml`` 中的项目版本。"""
    with (project_root / "pyproject.toml").open("rb") as file:
        return str(tomllib.load(file)["project"]["version"])


def read_package_version(project_root: Path = PROJECT_ROOT) -> str:
    """通过 AST 读取包入口的 ``__version__``，避免导入可选依赖。"""
    package_path = project_root / "src" / "mars" / "__init__.py"
    module = ast.parse(package_path.read_text(encoding="utf-8"), filename=str(package_path))
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__version__" for target in statement.targets):
            continue
        if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            return statement.value.value
    raise ValueError(f"Cannot find a string __version__ assignment in {package_path}.")


def normalize_release_tag(release_tag: str) -> str:
    """将 ``v0.0.24`` 或 ``0.0.24`` 规范为包版本文本。"""
    normalized = release_tag.strip()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    if not normalized:
        raise ValueError("Release tag must not be empty.")
    return normalized


def validate_versions(release_tag: str | None = None) -> str:
    """校验项目版本、源码版本和可选 release tag，并返回统一版本。"""
    project_version = read_project_version()
    package_version = read_package_version()
    if project_version != package_version:
        raise ValueError(
            "Version mismatch: "
            f"pyproject.toml={project_version!r}, mars.__version__={package_version!r}."
        )

    if release_tag is not None:
        normalized_tag = normalize_release_tag(release_tag)
        if normalized_tag != project_version:
            raise ValueError(
                "Release tag mismatch: "
                f"tag={normalized_tag!r}, package version={project_version!r}."
            )
    return project_version


def parse_args() -> argparse.Namespace:
    """解析 release 版本校验参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-tag",
        default=None,
        help="可选 GitHub release tag，例如 v0.0.24。",
    )
    return parser.parse_args()


def main() -> None:
    """执行版本一致性校验并打印统一版本。"""
    args = parse_args()
    version = validate_versions(args.release_tag)
    print(f"Version check passed: {version}")


if __name__ == "__main__":
    main()
