"""分箱规则 JSON artifact 的无损值编码与原子文件 I/O。"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

_TAG_KEY = "__mars_json_type__"


def encode_json_value(value: Any) -> Any:
    """将受支持的 Python 值编码为严格 JSON 可表示对象。"""
    if isinstance(value, np.generic):
        return {
            _TAG_KEY: "numpy_scalar",
            "dtype": str(value.dtype),
            "value": encode_json_value(value.item()),
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {_TAG_KEY: "float", "value": "nan"}
        if math.isinf(value):
            return {_TAG_KEY: "float", "value": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, Decimal):
        return {_TAG_KEY: "decimal", "value": str(value)}
    if isinstance(value, datetime):
        return {_TAG_KEY: "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {_TAG_KEY: "date", "value": value.isoformat()}
    if isinstance(value, list):
        return [encode_json_value(item) for item in value]
    if isinstance(value, tuple):
        return {
            _TAG_KEY: "tuple",
            "items": [encode_json_value(item) for item in value],
        }
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value) and _TAG_KEY not in value:
            return {
                str(key): encode_json_value(inner)
                for key, inner in sorted(value.items(), key=lambda item: item[0])
            }
        encoded_items = [
            [encode_json_value(key), encode_json_value(inner)]
            for key, inner in value.items()
        ]
        encoded_items.sort(
            key=lambda item: json.dumps(item[0], ensure_ascii=False, sort_keys=True),
        )
        return {_TAG_KEY: "mapping", "items": encoded_items}
    raise TypeError(
        "Binner JSON artifact does not support values of type "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def decode_json_value(value: Any) -> Any:
    """从严格 JSON 对象恢复带类型标记的 Python 值。"""
    if isinstance(value, list):
        return [decode_json_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    tag = value.get(_TAG_KEY)
    if tag is None:
        return {str(key): decode_json_value(inner) for key, inner in value.items()}
    if tag == "float":
        float_value = value.get("value")
        float_map = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}
        if float_value not in float_map:
            raise ValueError(f"Invalid tagged float value: {float_value!r}.")
        return float_map[float_value]
    if tag == "decimal":
        return Decimal(str(value["value"]))
    if tag == "datetime":
        return datetime.fromisoformat(str(value["value"]))
    if tag == "date":
        return date.fromisoformat(str(value["value"]))
    if tag == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("Tagged tuple must contain an `items` list.")
        return tuple(decode_json_value(item) for item in items)
    if tag == "mapping":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("Tagged mapping must contain an `items` list.")
        decoded: dict[Any, Any] = {}
        for item in items:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("Tagged mapping entries must be two-element lists.")
            decoded[decode_json_value(item[0])] = decode_json_value(item[1])
        return decoded
    if tag == "numpy_scalar":
        dtype = value.get("dtype")
        if not isinstance(dtype, str):
            raise ValueError("Tagged NumPy scalar must contain a string `dtype`.")
        decoded_value = decode_json_value(value.get("value"))
        try:
            return np.asarray(decoded_value, dtype=np.dtype(dtype))[()]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid NumPy scalar payload for dtype {dtype!r}.") from exc
    raise ValueError(f"Unknown MARS JSON value tag: {tag!r}.")


def write_json_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    """将 artifact 原子写入已存在的父目录。"""
    output_path = Path(path)
    parent = output_path.parent
    if not parent.exists():
        raise FileNotFoundError(f"Artifact parent directory does not exist: {parent}")

    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=str(parent),
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def read_json_artifact(path: str | Path) -> dict[str, Any]:
    """读取 JSON artifact 并校验其顶层对象类型。"""
    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid binner JSON artifact: {input_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Binner JSON artifact must contain a JSON object.")
    return {str(key): value for key, value in payload.items()}
