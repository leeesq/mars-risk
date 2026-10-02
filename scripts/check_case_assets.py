"""重算公开案例语义，核对证据、快照、下载包和站点资源；不修改公开产物。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/cases"
GENERATOR = ROOT / "docs/snippets/task_cases.py"
CASE_PAGES = (
    "data-quality", "binning-stability", "selection-correlation", "score-cross",
    "rule-evidence", "saved-reports", "report-delivery",
)
VOLATILE_FIELDS = frozenset({
    "report_id", "created_at", "generated_at", "created_at_utc", "generated_at_utc",
    "source_commit", "source_revision", "elapsed_seconds", "polars_version",
})


def _reject_constant(value: str) -> None:
    """拒绝 JSON 非有限常量，避免机器证据依赖非标准解析行为。"""
    raise ValueError(f"Non-finite JSON constant: {value}")


def _read_json(path: Path) -> Any:
    """读取标准有限 JSON；错误保留具体文件路径。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (ValueError, OSError) as error:
        raise ValueError(f"Invalid case JSON {path.name}: {error}") from error


def _safe_path(directory: Path, name: str) -> Path:
    """仅允许产物目录内的相对 POSIX 路径。"""
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe artifact path: {name!r}")
    candidate = directory.joinpath(*relative.parts)
    if not candidate.is_file():
        raise ValueError(f"Missing artifact: {name}")
    return candidate


def _semantic(value: Any) -> Any:
    """排除本来随运行变化的身份和时间，同时保留业务字段、状态和数值。"""
    if isinstance(value, dict):
        return {
            key: _semantic(item) for key, item in value.items()
            if key not in VOLATILE_FIELDS and not key.endswith("_report_id")
        }
    if isinstance(value, list):
        return [_semantic(item) for item in value]
    return value


def _assert_equal(expected: Any, actual: Any, location: str = "summary") -> None:
    """逐字段比较语义，允许浮点最后几位的跨平台差异并报告首个陈旧路径。"""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f"Stale {location}: keys {sorted(expected)} != {sorted(actual)}")
        for key in expected:
            _assert_equal(expected[key], actual[key], f"{location}.{key}")
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f"Stale {location}: row count {len(expected)} != {len(actual)}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_equal(left, right, f"{location}[{index}]")
        return
    if (
        isinstance(expected, (int, float)) and not isinstance(expected, bool)
        and isinstance(actual, (int, float)) and not isinstance(actual, bool)
    ):
        if math.isclose(expected, actual, rel_tol=1e-7, abs_tol=1e-10):
            return
    elif type(expected) is type(actual) and expected == actual:
        return
    raise ValueError(f"Stale {location}: published={expected!r}, generated={actual!r}")


def _table_rows(frame: Any) -> list[dict[str, Any]]:
    """仅物化已经分页的小证据表，支持公共接口的两个后端。"""
    from mars.reporting._serialization import json_safe

    if hasattr(frame, "to_dicts"):
        return json_safe(frame.to_dicts())
    return json_safe(frame.to_dict("records"))


def _code_digest(path: Path) -> str:
    """只比较可执行 AST，说明文字、注释和排版不会强迫重新分析。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


def _validate_evidence(directory: Path) -> dict[str, Any]:
    """只加载已保存快照，重放七例中的实际公共查询与身份关联。"""
    from mars.reporting import load_report

    snapshots: dict[str, Any] = {
        report.report_id: report
        for path in directory.glob("*.marsreport")
        for report in [load_report(path)]
    }
    if not snapshots:
        raise ValueError("Cases contain no loadable .marsreport files")
    manifest: dict[str, Any] = _read_json(directory / "manifest.json")
    summary: dict[str, Any] = _read_json(directory / "summary.json")
    for number in range(1, 8):
        case: dict[str, Any] = _read_json(directory / f"case-{number}.json")
        if str(case["case"]) != str(number) or case["report_id"] not in snapshots:
            raise ValueError(f"case-{number}: report identity is not saved in this batch")
        if not case["queries"] or not case["findings"] or not case["unavailable"]:
            raise ValueError(f"case-{number}: missing evidence, findings, or explicit limitation")
        for key, identity in case["findings"].items():
            if key.endswith("_report_id") and identity not in snapshots:
                raise ValueError(f"case-{number}: related {key} is not saved in this batch")
        for index, query in enumerate(case["queries"]):
            reference = query["reference"]
            if reference["report_id"] not in snapshots:
                raise ValueError(f"case-{number}.queries[{index}]: foreign report identity")
            report = snapshots[reference["report_id"]]
            replay = report.query_page(reference["table"], **reference["query"])
            _assert_equal(query["rows"], _table_rows(replay["data"]), f"case-{number}.query-{index}.rows")
            for field in ("total_rows", "omitted_rows", "next_offset"):
                if query[field] != replay[field]:
                    raise ValueError(f"case-{number}.query-{index}: {field} does not replay")
            if "returned_rows" in query and query["returned_rows"] != len(query["rows"]):
                raise ValueError(f"case-{number}.query-{index}: returned_rows differs from evidence")
        description = case["description"]
        if description["report_id"] != case["report_id"]:
            raise ValueError(f"case-{number}: human description and Agent evidence differ in identity")
        _assert_equal(description, snapshots[case["report_id"]].describe(), f"case-{number}.description")
        if (
            manifest["reports"][str(number)] != case["report_id"]
            or summary["cases"][str(number)]["report_id"] != case["report_id"]
        ):
            raise ValueError(f"case-{number}: manifest, human summary and query identities differ")
        _assert_equal(
            {"report_id": case["report_id"], **case["findings"]},
            summary["cases"][str(number)], f"summary.case-{number}",
        )
        compact: dict[str, Any] = _read_json(directory / f"query-{number}.json")
        _assert_equal(compact["query"], case["queries"][0], f"query-{number}.query")
        for field in ("case", "report_id", "question", "unavailable", "consumer"):
            _assert_equal(compact[field], case[field], f"query-{number}.{field}")
    context_path = directory / "agent-context.json"
    context_text = context_path.read_text(encoding="utf-8").rstrip("\n")
    bounded: dict[str, Any] = _read_json(context_path)
    restore: dict[str, Any] = _read_json(directory / "case-6.json")
    report = snapshots[restore["report_id"]]
    if len(context_text) > restore["findings"]["max_chars"]:
        raise ValueError("Bounded Agent JSON exceeds its declared Unicode character budget")
    if len(context_text) != restore["findings"]["context_chars"]:
        raise ValueError("Bounded Agent JSON character count differs from the saved finding")
    if bounded["description"]["report_id"] != report.report_id:
        raise ValueError("Bounded Agent JSON has a foreign report identity")
    for index, entry in enumerate(bounded["evidence"]):
        if entry["report_id"] != report.report_id:
            raise ValueError(f"Bounded evidence {index} has a foreign report identity")
        rows = _table_rows(report.get_table(entry["reference"], **entry["query"]))
        _assert_equal(entry["rows"], rows, f"bounded.evidence[{index}].rows")
    # 对实际交付的静态工作簿检查可读性，并从同源快照核对代表性交叉首行。
    import openpyxl

    for path in directory.glob("*.xlsx"):
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            if not workbook.sheetnames or not any(sheet.max_row for sheet in workbook):
                raise ValueError(f"Empty Excel delivery: {path.name}")
            if path.name == "score-cross.xlsx":
                sheet_rows = workbook["000_cells"].iter_rows(values_only=True)
                header = list(next(sheet_rows))
                excel_row = next(sheet_rows)
                source_row = _table_rows(report.get_table("cells", limit=1))[0]
                for field in ("sample_count", "observed_sample_count", "bad_sample_count", "bad_rate"):
                    _assert_equal(source_row[field], excel_row[header.index(field)], f"Excel.cells.{field}")
        finally:
            workbook.close()
    preview_path = directory / "preview-evidence.json"
    if preview_path.exists():
        preview: dict[str, Any] = _read_json(preview_path)
        reference = preview["reference"]
        report = snapshots.get(reference["report_id"])
        if report is None:
            raise ValueError("README preview references a foreign report batch")
        rows = _table_rows(report.query_page(reference["table"], **reference["query"])["data"])
        if len(rows) != 1:
            raise ValueError("README preview must cite exactly one queried cell")
        _assert_equal(preview["row"], rows[0], "preview.row")
        excerpt = preview["excerpt"]
        if excerpt["report_id"] != report.report_id or excerpt["table"] != reference["table"]:
            raise ValueError("README preview excerpt identity differs from the actual query")
        for field, value in excerpt.items():
            if field not in {"report_id", "table", "scope"}:
                _assert_equal(value, rows[0][field], f"preview.excerpt.{field}")
        for field, value in excerpt["scope"].items():
            _assert_equal(value, rows[0][field], f"preview.scope.{field}")
    return snapshots


def _validate_manifest(directory: Path, site_dir: Path | None) -> dict[str, Any]:
    """核对 manifest 的实际文件、内容 hash、编码、打包成员和构建资源。"""
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    if not re.search(r"^docs/assets/cases/\*\*\s+-text\s*$", attributes, re.M):
        raise ValueError("Missing .gitattributes byte preservation for docs/assets/cases/** -text")
    manifest: dict[str, Any] = _read_json(directory / "manifest.json")
    if manifest["schema_version"] != 1 or not manifest["artifacts"]:
        raise ValueError("Case manifest has unsupported schema or no artifacts")
    for name in ("task_cases.py", "external_agent_rule_case.py"):
        delivered = _safe_path(directory, name)
        current = ROOT / "docs/snippets" / name
        if _code_digest(delivered) != _code_digest(current):
            raise ValueError(f"Downloadable source is stale: {name}; run --phase finalize")
    seen: set[str] = set()
    for artifact in manifest["artifacts"]:
        name = artifact["path"]
        if name in seen:
            raise ValueError(f"Duplicate artifact in manifest: {name}")
        seen.add(name)
        path = _safe_path(directory, name)
        data = path.read_bytes()
        if len(data) != artifact["size_bytes"]:
            raise ValueError(f"Artifact size differs from manifest: {name}")
        if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Artifact hash differs from manifest: {name}; run --phase finalize")
        if path.suffix in {".json", ".txt", ".md", ".py", ".html", ".svg"}:
            text = data.decode("utf-8")
            if re.search(r"[A-Za-z]:[\\/](?:Users|Desktop|my_download_program)[\\/]", text):
                raise ValueError(f"Private absolute path leaked in public artifact: {name}")
            if path.suffix == ".json":
                _read_json(path)
        if site_dir is not None:
            built = site_dir / "assets/cases" / name
            if not built.is_file() or built.read_bytes() != data:
                raise ValueError(f"Built site does not contain the real case download: {name}")
    archive_path = directory / "cases.zip"
    if not archive_path.is_file():
        raise ValueError("Missing shared download cases.zip")
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_files != seen | {"manifest.json", "cases.zip"}:
        raise ValueError(f"Case files are absent from manifest: {sorted(actual_files - seen - {'manifest.json', 'cases.zip'})}")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("cases.zip contains duplicate members")
        for name in names:
            source = _safe_path(directory, name)
            if archive.read(name) != source.read_bytes():
                raise ValueError(f"cases.zip member differs from its source: {name}")
        required = (seen - {"cases.zip"}) | {"manifest.json"}
        if required != set(names):
            raise ValueError(f"cases.zip members differ from manifest: {sorted(required.symmetric_difference(names))}")
    if site_dir is not None:
        for name in ("manifest.json", "cases.zip"):
            built = site_dir / "assets/cases" / name
            if not built.is_file() or built.read_bytes() != (directory / name).read_bytes():
                raise ValueError(f"Built site download is missing or stale: {name}")
        for page in CASE_PAGES:
            if not (site_dir / "demos" / page / "index.html").is_file():
                raise ValueError(f"Built case page is missing: demos/{page}/")
    return manifest


def _recompute(directory: Path, manifest: dict[str, Any]) -> None:
    """以公开实际 seed/规模重算，比较语义摘要与全部有限查询证据。"""
    config = manifest["data_config"]
    with tempfile.TemporaryDirectory(prefix="mars-case-check-", dir=ROOT) as temporary:
        fresh = Path(temporary)
        execution: subprocess.CompletedProcess[str] = subprocess.run(
            [
                sys.executable, str(GENERATOR), "--case", "all", "--phase", "generate",
                "--rows", str(config["rows"]), "--seed", str(config["seed"]),
                "--output-dir", str(fresh),
            ],
            check=False, capture_output=True, text=True,
        )
        if execution.returncode:
            raise ValueError(f"Public case generation failed (exit {execution.returncode}):\n{execution.stdout}\n{execution.stderr}")
        names = ["summary.json", "agent-context.json"]
        names.extend(f"{prefix}-{number}.json" for prefix in ("case", "query") for number in range(1, 8))
        for name in names:
            _assert_equal(_semantic(_read_json(directory / name)), _semantic(_read_json(fresh / name)), name)
        _assert_equal(
            (directory / "previews.txt").read_text(encoding="utf-8"),
            (fresh / "previews.txt").read_text(encoding="utf-8"), "previews.txt",
        )
    print(f"Published semantic evidence matches current API: rows={config['rows']}, seed={config['seed']}")


def main() -> None:
    """执行检查，任何陈旧或无效证据均以清晰错误退出非零。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=ASSETS)
    parser.add_argument("--site-dir", type=Path, default=None)
    parser.add_argument("--skip-recompute", action="store_true", help="仅检查文件和已有证据；公开 CI 不使用此选项")
    args = parser.parse_args()
    try:
        manifest = _validate_manifest(args.assets_dir, args.site_dir)
        for name in ("preview-evidence.json", "previews.txt", "readme-preview.png", "readme-preview-mobile.png"):
            _safe_path(args.assets_dir, name)
        snapshots = _validate_evidence(args.assets_dir)
        if not args.skip_recompute:
            _recompute(args.assets_dir, manifest)
        print(f"Case assets passed: {len(manifest['artifacts'])} files, {len(snapshots)} snapshots, seven cases")
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Case asset check failed: {error}\n")


if __name__ == "__main__":
    main()
