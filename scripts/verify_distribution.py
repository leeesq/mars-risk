"""校验 MARS wheel 与 sdist 的元数据、标签和资源完整性。"""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Set, Tuple

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

try:
    from check_release_version import read_package_version
except ModuleNotFoundError:  # pragma: no cover - imported as scripts.verify_distribution
    from scripts.check_release_version import read_package_version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PACKAGE_FILES = {
    "mars/py.typed",
    "mars/reporting/template/mars_bin_report_linux.xlsx",
    "mars/reporting/template/mars_bin_report_win_mac.xlsx",
}
REQUIRED_SDIST_FILES = {
    "LICENSE",
    "NOTICE",
    "MANIFEST.in",
    "README.md",
    "pyproject.toml",
    "src/mars/__init__.py",
    "src/mars/py.typed",
    "src/mars/reporting/template/mars_bin_report_linux.xlsx",
    "src/mars/reporting/template/mars_bin_report_win_mac.xlsx",
}


class DistributionError(ValueError):
    """发布产物未满足 MARS 契约。"""


def _read_project(project_root: Path) -> Mapping[str, Any]:
    """读取 pyproject 中的 project 表。"""
    with (project_root / "pyproject.toml").open("rb") as file:
        payload = tomllib.load(file)
    project = payload.get("project")
    if not isinstance(project, dict):
        raise DistributionError("pyproject.toml must contain a [project] table.")
    return project


def _normalized_requirements(values: Iterable[str]) -> List[str]:
    """将依赖及 marker 规范为可稳定比较的文本。"""
    return sorted(str(Requirement(value)) for value in values)


def _requirement_with_extra(value: str, extra: str) -> str:
    """为 optional dependency 补充对应 extra marker。"""
    requirement = Requirement(value)
    base = str(requirement).split(";", 1)[0].strip()
    markers = []
    if requirement.marker is not None:
        markers.append(str(requirement.marker))
    markers.append(f'extra == "{extra}"')
    return f"{base}; {' and '.join(markers)}"


def _project_requirements(project: Mapping[str, Any]) -> List[str]:
    """展开基础依赖及全部 optional extra 依赖。"""
    requirements = [str(value) for value in project["dependencies"]]
    optional_dependencies = project.get("optional-dependencies", {})
    if not isinstance(optional_dependencies, dict):
        raise DistributionError("project.optional-dependencies must be a table.")
    for extra, values in optional_dependencies.items():
        requirements.extend(
            _requirement_with_extra(str(value), str(extra))
            for value in values
        )
    return requirements


def _single_artifacts(dist_dir: Path) -> Tuple[Path, Path]:
    """返回目录中唯一的 wheel 与 sdist。"""
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise DistributionError(
            "Distribution directory must contain exactly one wheel and one .tar.gz sdist; "
            f"found wheels={len(wheels)}, sdists={len(sdists)}."
        )
    return wheels[0], sdists[0]


def _wheel_metadata(archive: zipfile.ZipFile) -> Tuple[str, bytes, bytes]:
    """读取唯一 dist-info 目录和关键元数据。"""
    names = set(archive.namelist())
    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    if len(metadata_names) != 1:
        raise DistributionError("Wheel must contain exactly one dist-info/METADATA file.")
    metadata_name = metadata_names[0]
    dist_info = metadata_name.rsplit("/", 1)[0]
    required = {
        f"{dist_info}/METADATA",
        f"{dist_info}/WHEEL",
        f"{dist_info}/RECORD",
    }
    missing = sorted(required - names)
    if missing:
        raise DistributionError(f"Wheel dist-info is incomplete; missing {missing}.")
    return dist_info, archive.read(metadata_name), archive.read(f"{dist_info}/WHEEL")


def verify_wheel(wheel_path: Path, project_root: Path = PROJECT_ROOT) -> str:
    """校验 wheel 文件名、元数据、依赖与包资源。"""
    project = _read_project(project_root)
    expected_name = canonicalize_name(str(project["name"]))
    expected_version = str(project["version"])
    parsed_name, parsed_version, _build, tags = parse_wheel_filename(wheel_path.name)
    if canonicalize_name(parsed_name) != expected_name or str(parsed_version) != expected_version:
        raise DistributionError(
            f"Wheel filename identity mismatch: {parsed_name} {parsed_version}, "
            f"expected {expected_name} {expected_version}."
        )
    if {str(tag) for tag in tags} != {"py3-none-any"}:
        raise DistributionError(f"Wheel must use only py3-none-any, got {sorted(map(str, tags))}.")

    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        missing_resources = sorted(REQUIRED_PACKAGE_FILES - names)
        if missing_resources:
            raise DistributionError(f"Wheel is missing package resources: {missing_resources}.")
        _dist_info, metadata_bytes, wheel_bytes = _wheel_metadata(archive)

    metadata = BytesParser(policy=compat32).parsebytes(metadata_bytes)
    if canonicalize_name(str(metadata["Name"])) != expected_name:
        raise DistributionError("Wheel METADATA Name does not match pyproject.toml.")
    if str(metadata["Version"]) != expected_version:
        raise DistributionError("Wheel METADATA Version does not match pyproject.toml.")
    actual_python = str(SpecifierSet(str(metadata["Requires-Python"])))
    expected_python = str(SpecifierSet(str(project["requires-python"])))
    if actual_python != expected_python:
        raise DistributionError("Wheel Requires-Python does not match pyproject.toml.")

    actual_requires = _normalized_requirements(metadata.get_all("Requires-Dist", []))
    expected_requires = _normalized_requirements(_project_requirements(project))
    if actual_requires != expected_requires:
        raise DistributionError(
            "Wheel Requires-Dist entries do not match pyproject.toml dependencies. "
            f"expected={expected_requires}, actual={actual_requires}."
        )

    wheel_text = wheel_bytes.decode("utf-8")
    if "Root-Is-Purelib: true" not in wheel_text or "Tag: py3-none-any" not in wheel_text:
        raise DistributionError("Wheel metadata must declare a purelib py3-none-any artifact.")
    return expected_version


def _strip_sdist_root(names: Iterable[str]) -> Set[str]:
    """移除 sdist 顶层目录并返回相对路径集合。"""
    normalized = {name.rstrip("/") for name in names if name.rstrip("/")}
    roots = {name.split("/", 1)[0] for name in normalized}
    if len(roots) != 1:
        raise DistributionError(f"Sdist must contain exactly one top-level directory, got {roots}.")
    root = next(iter(roots))
    return {
        name[len(root) + 1 :]
        for name in normalized
        if name.startswith(f"{root}/")
    }


def verify_sdist(sdist_path: Path, project_root: Path = PROJECT_ROOT) -> str:
    """校验 sdist 名称以及源码、配置和模板资源。"""
    project = _read_project(project_root)
    expected_version = str(project["version"])
    expected_prefix = f"{canonicalize_name(str(project['name'])).replace('-', '_')}-{expected_version}"
    if sdist_path.name != f"{expected_prefix}.tar.gz":
        raise DistributionError(
            f"Unexpected sdist filename {sdist_path.name!r}; expected {expected_prefix}.tar.gz."
        )
    with tarfile.open(sdist_path, mode="r:gz") as archive:
        relative_names = _strip_sdist_root(member.name for member in archive.getmembers())
    missing = sorted(REQUIRED_SDIST_FILES - relative_names)
    if missing:
        raise DistributionError(f"Sdist is missing required files: {missing}.")
    return expected_version


def verify_distributions(dist_dir: Path, project_root: Path = PROJECT_ROOT) -> Tuple[Path, Path]:
    """校验目录中的唯一 wheel/sdist 并确认源码版本一致。"""
    wheel_path, sdist_path = _single_artifacts(dist_dir)
    wheel_version = verify_wheel(wheel_path, project_root)
    sdist_version = verify_sdist(sdist_path, project_root)
    source_version = read_package_version(project_root)
    if len({wheel_version, sdist_version, source_version}) != 1:
        raise DistributionError(
            "Distribution/source version mismatch: "
            f"wheel={wheel_version}, sdist={sdist_version}, source={source_version}."
        )
    return wheel_path, sdist_path


def parse_args() -> argparse.Namespace:
    """解析发布产物校验参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=PROJECT_ROOT / "dist")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> None:
    """执行静态发布产物门禁。"""
    args = parse_args()
    wheel_path, sdist_path = verify_distributions(
        args.dist_dir.resolve(),
        args.project_root.resolve(),
    )
    print(f"Distribution verification passed: {wheel_path.name}, {sdist_path.name}")


if __name__ == "__main__":
    main()
