"""容量对照专用的本机文件身份与 Report 数值证据，不保存原始样本。"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

FLOAT_ATOL = 1e-12
FLOAT_RTOL = 1e-12
EVIDENCE_VERSION = "report-content-v1"
MAX_SIDECAR_BYTES = 512 * 1024**2


def _file_identity(path: Path) -> dict[str, Any]:
    """以 1 MiB 流式 SHA256 读取真实文件；读期间发生变化立即拒绝。"""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError(f"File changed during content identity read: {path}")
    return {"algorithm": "sha256-stream-1MiB", "bytes": after.st_size,
            "sha256": digest.hexdigest()}


def _business_parameters(value: Any) -> Any:
    """只剥离已登记资源参数；未知元数据和业务参数继续严格比较。"""
    if isinstance(value, dict):
        return {key: _business_parameters(item) for key, item in value.items()
                if key not in {"batch_size", "overview_batch_size", "n_jobs"}}
    if isinstance(value, list):
        return [_business_parameters(item) for item in value]
    return value


def _semantic_description(description: dict[str, Any]) -> dict[str, Any]:
    """排除随机报告 ID 和有限已知实现细节，保留完整目录、身份、单位与业务语义。"""
    result = deepcopy(description)
    result.pop("report_id", None)
    parameters = result.get("parameters", {})
    if result.get("report_type") == "MarsProfileReport":
        parameters.pop("generated_at_utc", None)
    if result.get("report_type") == "correlation":
        # 旧 dense 实现没有这些字段；不把任意未知算法名称作为可忽略差异。
        if parameters.get("algorithm") == "anchored Chan block central moments Float64":
            parameters.pop("algorithm")
        for key in ("row_block_size", "feature_batch_size", "threshold_refined_pairs"):
            parameters.pop(key, None)
    result["parameters"] = _business_parameters(parameters)
    return result


def _save_evidence(report: Any, directory: Path, meter: Any, stage: str) -> dict[str, Any]:
    """保存唯一、不覆盖的私有统计快照；失败清理，成功由本机验收记录引用。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"capacity-{uuid4().hex}.marsreport"
    # 独占占位保证永不覆盖旧结果；后续原子替换只替换本次拥有的空文件。
    with path.open("xb"):
        pass
    try:
        meter.run(stage, lambda: report.save(path, overwrite=True))
        if path.stat().st_size > MAX_SIDECAR_BYTES:
            raise ValueError("Capacity semantic sidecar exceeds 512 MiB evidence bound")
        identity = meter.run("report_integrity_identity", lambda: _file_identity(path))
        return {"version": EVIDENCE_VERSION, "path": str(path.resolve()), **identity,
                "report_id": report.report_id,
                "tolerance": {"atol": FLOAT_ATOL, "rtol": FLOAT_RTOL},
                "description": _semantic_description(report.describe())}
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _load_evidence(round_: dict[str, Any]) -> Any:
    """校验实际 sidecar 内容和报告身份，旧 JSON 或损坏文件不补造证据。"""
    from benchmark_core_capacity import _signature

    from mars.reporting import load_report

    evidence = round_.get("semantic_evidence", {})
    if evidence.get("version") != EVIDENCE_VERSION:
        raise ValueError("semantic sidecar evidence absent or unsupported")
    if evidence.get("tolerance") != {"atol": FLOAT_ATOL, "rtol": FLOAT_RTOL}:
        raise ValueError("semantic tolerance contract absent or changed")
    path = Path(evidence.get("path", ""))
    if not path.is_file() or path.stat().st_size > MAX_SIDECAR_BYTES:
        raise ValueError("semantic sidecar missing or exceeds evidence bound")
    identity = _file_identity(path)
    if any(evidence.get(key) != value for key, value in identity.items()):
        raise ValueError("semantic sidecar integrity mismatch")
    report = load_report(path)
    if _file_identity(path) != identity:
        raise ValueError("semantic sidecar changed during load")
    if report.report_id != evidence.get("report_id"):
        raise ValueError("semantic sidecar report identity mismatch")
    if _semantic_description(report.describe()) != evidence.get("description"):
        raise ValueError("semantic sidecar description mismatch")
    # SHA 仅核验本轮记录和本轮真实内容相符；跨实现仍由有限数值容差对照。
    tables = round_.get("tables")
    if not isinstance(tables, dict) or list(tables) != list(report.describe()["tables"]):
        raise ValueError("round table signature catalog absent or differs from semantic sidecar")
    for name, expected in tables.items():
        actual = _signature(report.get_table(name))
        if any(expected.get(key) != actual[key] for key in ("rows", "columns", "schema", "sha256")):
            raise ValueError(f"round table signature disagrees with semantic sidecar: {name}")
    return report


def _values_equal(left: Any, right: Any, *, strict_float: bool = False) -> bool:
    """递归核验有限嵌套值；null、NaN、无穷符号彼此不同，业务阈值严格相等。"""
    if left is None or right is None:
        return left is right
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
        if not math.isfinite(left) or not math.isfinite(right) or strict_float:
            return left == right
        return math.isclose(left, right, rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _values_equal(a, b, strict_float=strict_float) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _values_equal(value, right[key], strict_float=strict_float)
            for key, value in left.items()
        )
    return bool(type(left) is type(right) and left == right)


def _compare_reports(reference: Any, current: Any) -> str | None:
    """按 50000 行小批核验全部表，保持目录/schema/顺序/非浮点值和决策动作严格一致。"""
    import numpy as np
    import polars as pl

    if _semantic_description(reference.describe()) != _semantic_description(current.describe()):
        return "report description/catalog/business metadata differs"
    if list(reference.describe()["tables"]) != list(current.describe()["tables"]):
        return "report table catalog order differs"
    for name in reference.describe()["tables"]:
        left, right = reference.get_table(name), current.get_table(name)
        if not isinstance(left, pl.DataFrame):
            if not left.index.equals(right.index) or not left.dtypes.equals(right.dtypes):
                return f"table {name}: pandas index/schema differs"
            left, right = pl.from_pandas(left), pl.from_pandas(right)
        if left.columns != right.columns or left.schema != right.schema or left.shape != right.shape:
            return f"table {name}: logical schema/shape differs"
        for offset in range(0, left.height, 50000):
            a, b = left.slice(offset, 50000), right.slice(offset, 50000)
            for column, dtype in left.schema.items():
                x, y = a[column], b[column]
                strict = column == "threshold" and name == "correlation_decisions"
                if dtype in (pl.Float32, pl.Float64):
                    if not x.is_null().equals(y.is_null()):
                        return f"table {name}.{column}: null positions differ"
                    xv, yv = x.to_numpy(), y.to_numpy()
                    # isclose 不合并 null 和 NaN；null 掩码已先严格检查。
                    equal = np.equal(xv, yv) | (np.isnan(xv) & np.isnan(yv))
                    if not strict:
                        finite = np.isfinite(xv) & np.isfinite(yv)
                        equal[finite] = np.isclose(xv[finite], yv[finite],
                                                   rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
                    if not bool(np.all(equal)):
                        return f"table {name}.{column}: floating values differ beyond tolerance"
                elif not x.equals(y):
                    # 嵌套 JSON/object 不经字符串近似；只在原生比较失败时逐值检查。
                    if not all(_values_equal(u, v) for u, v in zip(x.to_list(), y.to_list())):
                        return f"table {name}.{column}: identity/order/nonfloating values differ"
    return None


def _verify_rounds(rounds: list[tuple[str, dict[str, Any]]]) -> tuple[str | None, str]:
    """每次预热和正式轮均读取真实完整性证据，再与同一个 baseline 逐表比较。"""
    reference = None
    for label, round_ in rounds:
        try:
            report = _load_evidence(round_)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return f"{label}: insufficient semantic evidence: {exc}", "incomparable"
        if reference is None:
            reference = report
        else:
            reason = _compare_reports(reference, report)
            if reason:
                return f"{label}: {reason}", "semantic_mismatch"
    return None, "verified"
