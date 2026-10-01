"""把共享画像分箱器的切点适配为固定 Score Cross 定义。"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Literal

import polars as pl

from mars.compute import missing_condition_expr
from mars.feature.binning._json_codec import decode_json_value

from ._evaluation.context import build_binner, normalize_binary_target_column
from ._risk_profile import (
    _ALLOWED_BINNER_PARAM_KEYS,
    _build_effective_binner_params,
    _normalize_profile_risk_binning_type,
    _ProfileRiskMonotonicTrend,
)


def _score_binning_params(
    *,
    binning_type: str,
    method: Literal["quantile", "uniform", "cart"] | None,
    n_bins: int,
    min_bin_size: float | int | None,
    monotonic_trend: _ProfileRiskMonotonicTrend | None,
    missing_values: list[Any] | None,
    special_values: list[float],
    binner_params: dict[str, Any] | None,
    n_jobs: int | None,
) -> dict[str, Any]:
    """复用 profile_risk 参数解析；非法方法和无法生效的配置不能悄悄降级。"""
    kind = _normalize_profile_risk_binning_type(binning_type)
    if method is not None and method not in {"quantile", "uniform", "cart"}:
        raise ValueError("method must be quantile, uniform or cart; use cutpoints for custom bins.")
    if min_bin_size is not None:
        if isinstance(min_bin_size, bool) or not isinstance(min_bin_size, (int, float)):
            raise ValueError("min_bin_size must be a positive integer or a fraction in [0, 1].")
        if isinstance(min_bin_size, int) and min_bin_size < 1:
            raise ValueError("Integer min_bin_size must be positive.")
        if isinstance(min_bin_size, float) and not 0 <= min_bin_size <= 1:
            raise ValueError("Fraction min_bin_size must be in [0, 1].")
    if n_jobs is not None and (type(n_jobs) is not int or n_jobs == 0 or n_jobs < -1):
        raise ValueError("n_jobs must be -1 or a positive integer.")
    params = _build_effective_binner_params(
        binning_type=kind,
        binner_params=binner_params,
        method=method,
        n_bins=n_bins,
        min_bin_size=min_bin_size,
        monotonic_trend=monotonic_trend,
        missing_values=missing_values,
        special_values=special_values,
        n_jobs=n_jobs,
    )
    ignored = set(binner_params or {}) - set(_ALLOWED_BINNER_PARAM_KEYS[kind])
    if ignored:
        warnings.warn(
            f"binner_params ignored for binning_type={kind!r}: {sorted(ignored)}.",
            UserWarning,
            stacklevel=3,
        )
    if "cart_params" in params:
        from sklearn.tree import DecisionTreeClassifier

        cart_params = params["cart_params"]
        if not isinstance(cart_params, dict):
            raise ValueError("cart_params must be a dict.")
        forbidden = {"max_leaf_nodes", "min_samples_leaf"}
        unknown = set(cart_params) - set(inspect.signature(DecisionTreeClassifier).parameters)
        if set(cart_params).intersection(forbidden) or unknown:
            raise ValueError(
                "cart_params contains unknown keys or duplicates n_bins/min_bin_size controls."
            )
    return params


def _fit_score_bins(
    frame: pl.DataFrame,
    score: str,
    *,
    probability: bool,
    binning_type: str,
    binner_params: dict[str, Any],
    target: str | None,
    fit_source: str,
) -> tuple[list[float], dict[str, Any]]:
    """每轴只拟合一次，剔除不可用分数/未表现标签并保存实际来源和诊断。"""
    if target is not None:
        frame = normalize_binary_target_column(frame, target)
    values = pl.col(score).cast(pl.Float64, strict=False)
    missing = missing_condition_expr(
        score, dtype=frame.schema[score], missing_values=binner_params.get("missing_values")
    ) | values.is_nan()
    valid = values.is_finite() & ~missing
    specials = binner_params.get("special_values", [])
    if specials:
        valid &= ~values.is_in(specials)
    if probability:
        valid &= values.is_between(0, 1, closed="both")
    usable_frame: pl.DataFrame = frame.filter(valid)
    valid_score_rows = usable_frame.height
    if not valid_score_rows:
        raise ValueError(f"No valid reference scores for {score!r}; supply explicit cutpoints.")
    if target is not None:
        usable_frame = usable_frame.filter(pl.col(target).is_not_null())
        if usable_frame.height < 2 or usable_frame[target].n_unique() != 2:
            raise ValueError(
                f"Supervised binning for {score!r} needs usable observed 0/1 classes "
                f"in binning_target={target!r} on the selected reference."
            )
    # 用 null 隔离特殊/非法分数而保留参考集人数，沿用共享引擎的全量最小箱分母。
    fit_frame: pl.DataFrame = frame.with_columns(
        pl.when(valid).then(values).otherwise(None).alias(score)
    )
    if target is not None:
        fit_frame = fit_frame.filter(pl.col(target).is_not_null())
    binner = build_binner(
        binning_type=binning_type,
        binner_params=binner_params,
        fit_has_target=target is not None,
        fit_df=fit_frame,
        target=target or "",
        features=[score],
    )
    if score not in binner.bin_cuts_:
        raise ValueError(f"Binner produced no numeric definition for {score!r}.")
    # 仅已启用约束的拟合路径做成员等价适配；显式/已保存定义和无约束历史分段不移动。
    shared_cuts = [float(v) for v in binner.bin_cuts_[score][1:-1]]
    serialized_params: dict[str, Any] = binner.to_dict()["params"]
    fitted_params: dict[str, Any] = decode_json_value(serialized_params)
    minimum = fitted_params.get("min_bin_size", 0)
    minimum_enabled = bool(minimum > 0) and (
        binning_type != "native"
        or fitted_params.get("method") == "cart"
        or fitted_params.get("merge_small_bins", False)
    )
    adapt_endpoints = minimum_enabled or binning_type in {"optimal", "lite_opt"}
    cuts = shared_cuts
    endpoint_diagnostic: dict[str, Any] | None = None
    if adapt_endpoints:
        cuts, counts = _right_closed_fit_cuts(
            usable_frame.select(values.alias(score)), score, shared_cuts
        )
        # 只沿用引擎已有整数人数入口；机械分箱和 LiteOpt 的当前合同为全量比例。
        integer_minimum = isinstance(minimum, int) and (
            binning_type == "optimal" or fitted_params.get("method") == "cart"
        )
        threshold = minimum if integer_minimum else minimum * fit_frame.height
        satisfied = all(count >= threshold for count in counts)
        minimum_status = "not_enabled"
        if minimum_enabled:
            minimum_status = "satisfied" if satisfied else "unsatisfied"
        endpoint_diagnostic = {
            "adaptation": "shared_left_closed_membership",
            "shared_cutpoints": shared_cuts,
            "right_closed_cutpoints": cuts,
            "reference_bin_counts": counts,
            "minimum_enabled": minimum_enabled,
            "minimum_count": threshold if minimum_enabled else None,
            "denominator_row_count": fit_frame.height,
            "minimum_status": minimum_status,
        }
    diagnostic = binner.fit_failures_.get(score)
    if endpoint_diagnostic is not None and endpoint_diagnostic["minimum_status"] == "unsatisfied":
        message = "Right-closed reference bins cannot satisfy the enabled global min_bin_size."
        diagnostic = f"{diagnostic}; {message}" if diagnostic else message
    provenance: dict[str, Any] = {
        "binning_type": binning_type,
        "binner_params": serialized_params,
        "source": fit_source,
        "target": target,
        "reference_row_count": frame.height,
        "valid_score_row_count": valid_score_rows,
        "fit_row_count": fit_frame.height,
        "usable_fit_row_count": usable_frame.height,
        "observed_class_count": usable_frame[target].n_unique() if target else None,
        "requested_n_bins": binner_params["n_bins"],
        "actual_n_bins": len(cuts) + 1,
        "diagnostic": diagnostic,
        "fitted_trend": getattr(binner, "fitted_trends_", {}).get(score),
        "closed": "right",
    }
    if endpoint_diagnostic is not None:
        provenance["endpoint_adaptation"] = endpoint_diagnostic
    return cuts, provenance


def _right_closed_fit_cuts(
    frame: pl.DataFrame, score: str, cuts: list[float]
) -> tuple[list[float], list[int]]:
    """用真实观测前驱适配左闭切点，保持参考集成员、标签趋势及最小箱分母。

    切点命中观测值时取最大严格小于切点的参考值，不使用浮点扰动；空正常箱合并。
    一次 Polars 聚合获取边界信息，Python 只持有与切点数同阶的结果。
    """
    if not cuts:
        return [], [frame.height]
    value = pl.col(score)
    expressions: list[pl.Expr] = []
    for i, cut in enumerate(cuts):
        expressions.extend(
            [
                (value == cut).any().alias(f"hit_{i}"),
                value.filter(value < cut).max().alias(f"previous_{i}"),
                (value < cut).sum().alias(f"count_{i}"),
            ]
        )
    boundaries: dict[str, Any] = frame.select(expressions).row(0, named=True)
    adapted: list[float] = []
    counts: list[int] = []
    previous_count = 0
    for i, cut in enumerate(cuts):
        count = int(boundaries[f"count_{i}"])
        if count <= previous_count or count >= frame.height:
            continue
        previous = boundaries[f"previous_{i}"]
        adapted.append(float(previous) if boundaries[f"hit_{i}"] else cut)
        counts.append(count - previous_count)
        previous_count = count
    counts.append(frame.height - previous_count)
    return adapted, counts
