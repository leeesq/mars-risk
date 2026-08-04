from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.8 compatibility
    import tomli as tomllib

from scripts.verify_distribution import DistributionError, verify_distributions

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_metadata() -> tuple[str, str, str, list[str]]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]
    dependencies = [str(value) for value in project["dependencies"]]
    for extra, values in project.get("optional-dependencies", {}).items():
        for value in values:
            requirement = str(value)
            if ";" in requirement:
                base, marker = requirement.split(";", 1)
                dependencies.append(f'{base}; {marker.strip()} and extra == "{extra}"')
            else:
                dependencies.append(f'{requirement}; extra == "{extra}"')
    return (
        str(project["name"]),
        str(project["version"]),
        str(project["requires-python"]),
        dependencies,
    )


def _write_fake_wheel(
    dist_dir: Path,
    *,
    missing_template: bool = False,
    metadata_version: str | None = None,
    dependencies: list[str] | None = None,
    tag: str = "py3-none-any",
) -> Path:
    name, version, requires_python, project_dependencies = _project_metadata()
    distribution = name.replace("-", "_")
    wheel_path = dist_dir / f"{distribution}-{version}-{tag}.whl"
    dist_info = f"{distribution}-{version}.dist-info"
    metadata_lines = [
        "Metadata-Version: 2.3",
        f"Name: {name}",
        f"Version: {metadata_version or version}",
        f"Requires-Python: {requires_python}",
    ]
    metadata_lines.extend(
        f"Requires-Dist: {requirement}"
        for requirement in (dependencies if dependencies is not None else project_dependencies)
    )
    wheel_metadata = "Wheel-Version: 1.0\nRoot-Is-Purelib: true\n" f"Tag: {tag}\n"

    with zipfile.ZipFile(wheel_path, mode="w") as archive:
        archive.writestr("mars/py.typed", "")
        archive.writestr(
            "mars/reporting/template/mars_bin_report_linux.xlsx",
            b"linux-template",
        )
        if not missing_template:
            archive.writestr(
                "mars/reporting/template/mars_bin_report_win_mac.xlsx",
                b"win-mac-template",
            )
        archive.writestr(f"{dist_info}/METADATA", "\n".join(metadata_lines) + "\n")
        archive.writestr(f"{dist_info}/WHEEL", wheel_metadata)
        archive.writestr(f"{dist_info}/RECORD", "")
    return wheel_path


def _write_fake_sdist(dist_dir: Path) -> Path:
    name, version, _requires_python, _dependencies = _project_metadata()
    root = f"{name.replace('-', '_')}-{version}"
    sdist_path = dist_dir / f"{root}.tar.gz"
    files = {
        "LICENSE": b"license",
        "MANIFEST.in": b"manifest",
        "README.md": b"readme",
        "pyproject.toml": b"project",
        "src/mars/__init__.py": b"__version__ = '0.0.27'",
        "src/mars/py.typed": b"",
        "src/mars/reporting/template/mars_bin_report_linux.xlsx": b"linux-template",
        "src/mars/reporting/template/mars_bin_report_win_mac.xlsx": b"win-mac-template",
    }
    with tarfile.open(sdist_path, mode="w:gz") as archive:
        for relative_path, payload in files.items():
            info = tarfile.TarInfo(f"{root}/{relative_path}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return sdist_path


def _write_valid_pair(dist_dir: Path) -> None:
    dist_dir.mkdir()
    _write_fake_wheel(dist_dir)
    _write_fake_sdist(dist_dir)


def test_distribution_verifier_accepts_complete_pair(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    _write_valid_pair(dist_dir)

    wheel_path, sdist_path = verify_distributions(dist_dir, PROJECT_ROOT)

    assert wheel_path.suffix == ".whl"
    assert sdist_path.name.endswith(".tar.gz")


def test_distribution_verifier_rejects_missing_template(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_fake_wheel(dist_dir, missing_template=True)
    _write_fake_sdist(dist_dir)

    with pytest.raises(DistributionError, match="missing package resources"):
        verify_distributions(dist_dir, PROJECT_ROOT)


def test_distribution_verifier_rejects_version_mismatch(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_fake_wheel(dist_dir, metadata_version="9.9.9")
    _write_fake_sdist(dist_dir)

    with pytest.raises(DistributionError, match="METADATA Version"):
        verify_distributions(dist_dir, PROJECT_ROOT)


def test_distribution_verifier_rejects_dependency_marker_mismatch(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _name, _version, _requires_python, dependencies = _project_metadata()
    dependencies[0] = "polars==1.8.2; python_version < '3.10'"
    _write_fake_wheel(dist_dir, dependencies=dependencies)
    _write_fake_sdist(dist_dir)

    with pytest.raises(DistributionError, match="Requires-Dist"):
        verify_distributions(dist_dir, PROJECT_ROOT)


def test_distribution_verifier_rejects_platform_wheel(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_fake_wheel(dist_dir, tag="cp38-cp38-win_amd64")
    _write_fake_sdist(dist_dir)

    with pytest.raises(DistributionError, match="py3-none-any"):
        verify_distributions(dist_dir, PROJECT_ROOT)
