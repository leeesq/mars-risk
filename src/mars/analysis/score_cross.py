"""固定双分数分段的联合统计及无需明细的历史规则回放。"""

from __future__ import annotations

from copy import deepcopy
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl

from mars._compat import _left_join_nulls
from mars.compute import amount_stats_agg_exprs, binary_stats_agg_exprs, missing_condition_expr
from mars.reporting._artifact import Report, ReportSnapshot
from mars.reporting._metadata import FeatureMetadata
from mars.reporting._result import _result_report

from ._evaluation.context import normalize_binary_target_column, prepare_group_context
from ._risk_profile import _normalize_profile_risk_binning_type, _ProfileRiskMonotonicTrend
from ._score_cross_binning import _fit_score_bins, _score_binning_params
from ._score_cross_expression import (
    _EXPRESSION_LIMITS,
    _evaluate_score_expression,
    _parse_score_expression,
)

_SCOPE = ["target", "group", "period"]
_COUNTS = [
    "sample_count",
    "observed_sample_count",
    "bad_sample_count",
    "weight_sum",
    "observed_weight_sum",
    "bad_weight_sum",
    "tot_amt",
    "good_amt",
    "bad_amt",
]


def _project(df: pl.DataFrame | pd.DataFrame, columns: list[str]) -> pl.DataFrame:
    """跨引擎转换前只投影必需字段，宽表其余列不复制。"""
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}.")
    return (
        df.select(columns) if isinstance(df, pl.DataFrame) else pl.from_pandas(df.loc[:, columns])
    )


def _axis_definition(
    score: str,
    direction: str,
    cuts: list[float],
    specials: list[float],
    probability: bool,
    missing_values: list[Any] | None = None,
) -> dict[str, Any]:
    """纯配置校验，不扫描样本；显式切点和保存定义走同一条路径。"""
    if direction not in {"higher_risk", "lower_risk"}:
        raise ValueError(f"Explicit risk direction required for {score!r}.")
    special_array = np.asarray(specials, dtype=float)
    if not np.isfinite(special_array).all() or len(set(specials)) != len(specials):
        raise ValueError("special_values must be unique finite numbers.")
    if type(probability) is not bool:
        raise ValueError("Saved probability declaration must be a boolean.")
    points = sorted(set(float(c) for c in cuts))
    if not np.isfinite(points).all():
        raise ValueError("cutpoints must be finite numbers; unbounded endpoints are implicit.")
    definition: dict[str, Any] = {
        "score": score,
        "direction": direction,
        "cutpoints": points,
        "special_values": list(specials),
        "probability": probability,
        "closed": "right",
        "actual_n_bins": len(points) + 1,
    }
    if missing_values is not None:
        if not isinstance(missing_values, list):
            raise ValueError("missing_values must be a list or map score IDs to lists.")
        definition["missing_values"] = deepcopy(missing_values)
    return definition


def _fit_axis(
    frame: pl.DataFrame,
    score: str,
    direction: str,
    n_bins: int,
    specials: list[float],
    probability: bool,
    *,
    binning_type: str = "native",
    binner_params: dict[str, Any] | None = None,
    target: str | None = None,
    fit_source: str = "current_input",
) -> dict[str, Any]:
    """共享分箱器拟合一次；固定定义保留 Score Cross 的右闭端点合同。"""
    _axis_definition(score, direction, [], specials, probability)
    effective_params = binner_params or {"n_bins": n_bins, "special_values": specials}
    points, provenance = _fit_score_bins(
        frame,
        score,
        probability=probability,
        binning_type=binning_type,
        binner_params=effective_params,
        target=target,
        fit_source=fit_source,
    )
    definition = _axis_definition(
        score, direction, points, specials, probability, effective_params.get("missing_values")
    )
    definition["fit"] = provenance
    return definition


def _bins(definitions: dict[str, dict[str, Any]]) -> pl.DataFrame:
    """保存原区间顺序和独立风险顺序，无界端点用 null 与显式标记。"""
    rows: list[dict[str, Any]] = []
    for axis, definition in definitions.items():
        points = definition["cutpoints"]
        n = len(points) + 1
        for i in range(n):
            rank = i + 1 if definition["direction"] == "higher_risk" else n - i
            rows.append(
                {
                    "axis": axis,
                    "feature": definition["score"],
                    "bin_id": f"b{i}",
                    "raw_order": i,
                    "risk_rank": rank,
                    "display_label": f"{axis.upper()}{rank}",
                    "kind": "normal",
                    "lower": points[i - 1] if i else None,
                    "upper": points[i] if i < len(points) else None,
                    "lower_unbounded": i == 0,
                    "upper_unbounded": i == len(points),
                    "lower_closed": False,
                    "upper_closed": i < len(points),
                    "special_value": None,
                    "direction": definition["direction"],
                }
            )
        for i, kind in enumerate(
            ["missing", "invalid", *[f"s{j}" for j in range(len(definition["special_values"]))]]
        ):
            rows.append(
                {
                    "axis": axis,
                    "feature": definition["score"],
                    "bin_id": kind,
                    "raw_order": n + i,
                    "risk_rank": None,
                    "display_label": kind,
                    "kind": kind if kind in {"missing", "invalid"} else "special",
                    "lower": None,
                    "upper": None,
                    "lower_unbounded": False,
                    "upper_unbounded": False,
                    "lower_closed": False,
                    "upper_closed": False,
                    "special_value": definition["special_values"][i - 2] if i >= 2 else None,
                    "direction": definition["direction"],
                }
            )
    return pl.DataFrame(
        rows,
        schema_overrides={
            "lower": pl.Float64,
            "upper": pl.Float64,
            "special_value": pl.Float64,
            "risk_rank": pl.Int64,
        },
    ).sort(["axis", "risk_rank", "raw_order"], nulls_last=True)


def _assign(
    definition: dict[str, Any], axis: str, dtype: pl.DataType | None = None
) -> pl.Expr:
    """精确复用右闭固定边界，特殊和非法值不进入正常风险等级。"""
    original = pl.col(definition["score"])
    score = original.cast(pl.Float64, strict=False)
    invalid = ~score.is_finite() | (original.is_not_null() & score.is_null())
    if definition["probability"]:
        invalid = invalid | (score < 0) | (score > 1)
    index = pl.lit(0)
    for point in definition["cutpoints"]:
        index = index + (score > point).cast(pl.Int64)
    # 显式特殊值优先于概率域检查，但缺失和非有限值始终分开。
    missing = missing_condition_expr(
        original, dtype=dtype, missing_values=definition.get("missing_values")
    ) | score.is_nan()
    expression = pl.when(missing).then(pl.lit("missing"))
    for i, value in enumerate(definition["special_values"]):
        expression = expression.when(score == value).then(pl.lit(f"s{i}"))
    return (
        expression.when(invalid)
        .then(pl.lit("invalid"))
        .otherwise(pl.concat_str([pl.lit("b"), index.cast(pl.String)]))
        .alias(f"{axis}_bin")
    )


def _ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """零分母返回 null，不补平滑常数或伪造单位 Lift。"""
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


def _derive(frame: pl.DataFrame, parameters: dict[str, Any]) -> pl.DataFrame:
    """从可加总人数派生同口径风险和未加权 Wilson 区间。"""
    n, bad = pl.col("observed_sample_count"), pl.col("bad_sample_count")
    weighted = parameters["weights_col"] is not None
    denominator = pl.col("observed_weight_sum") if weighted else n
    numerator = pl.col("bad_weight_sum") if weighted else bad
    z = NormalDist().inv_cdf((1 + parameters["confidence_level"]) / 2)
    p = _ratio(bad, n)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) + z * z / (4 * n)) / n).sqrt() / (1 + z * z / n)
    requested = pl.col("target").is_not_null()
    result = frame.with_columns(
        pl.when(pl.col("sample_count") == 0)
        .then(pl.lit("empty"))
        .otherwise(pl.lit("populated"))
        .alias("sample_status"),
        pl.when(requested).then(_ratio(numerator, denominator)).otherwise(None).alias("bad_rate"),
        pl.when(requested)
        .then(_ratio(n, pl.col("sample_count")))
        .otherwise(None)
        .alias("observed_coverage"),
        pl.when(requested & (n > 0))
        .then((center - half).clip(0, 1))
        .otherwise(None)
        .alias("unweighted_ci_lower"),
        pl.when(requested & (n > 0))
        .then((center + half).clip(0, 1))
        .otherwise(None)
        .alias("unweighted_ci_upper"),
        pl.when(~requested)
        .then(pl.lit("not_requested"))
        .when(pl.col("sample_count") == 0)
        .then(pl.lit("empty"))
        .when(n == 0)
        .then(pl.lit("unobserved"))
        .when(denominator == 0)
        .then(pl.lit("invalid_denominator"))
        .when(n < parameters["min_observed"])
        .then(pl.lit("low_sample"))
        .otherwise(pl.lit("valid"))
        .alias("status"),
        pl.when(requested)
        .then(pl.lit("unsupported" if weighted else "not_requested"))
        .otherwise(pl.lit("not_requested"))
        .alias("weighted_ci_status"),
        pl.when(~requested)
        .then(pl.lit("not_requested"))
        .when(n == 0)
        .then(pl.lit("unavailable"))
        .otherwise(pl.lit("valid"))
        .alias("unweighted_ci_status"),
    )
    if "tot_amt" in result.columns:
        result = result.with_columns((pl.col("good_amt") + pl.col("bad_amt")).alias("observed_amt"))
        result = result.with_columns(
            _ratio(pl.col("bad_amt"), pl.col("observed_amt")).alias("amt_bad_rate")
        )
    label_columns = [
        c
        for c in (
            "observed_sample_count",
            "bad_sample_count",
            "observed_weight_sum",
            "bad_weight_sum",
            "good_amt",
            "bad_amt",
            "observed_amt",
            "amt_bad_rate",
        )
        if c in result.columns
    ]
    result = result.with_columns(
        [pl.when(requested).then(pl.col(c)).otherwise(None).alias(c) for c in label_columns]
    )
    return result


def _sum(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """只合并原始可加总统计；不平均风险、区间或重复 TOTAL。"""
    return frame.group_by(keys, maintain_order=True).agg(
        [pl.col(c).sum() for c in _COUNTS if c in frame.columns]
    )


def _lift_status() -> pl.Expr:
    """区分不可用 Lift 的原因；有效零风险在整体率为正时仍为 valid。"""
    return (
        pl.when(pl.col("target").is_null())
        .then(pl.lit("not_requested"))
        .when(pl.col("sample_count") == 0)
        .then(pl.lit("empty"))
        .when(pl.col("bad_rate").is_null())
        .then(pl.col("status"))
        .when(pl.col("overall_bad_rate").is_null())
        .then(pl.col("overall_status"))
        .when(pl.col("overall_bad_rate") <= 0)
        .then(pl.lit("invalid_denominator"))
        .otherwise(pl.lit("valid"))
        .alias("lift_status")
    )


def _marginals(cells: pl.DataFrame, parameters: dict[str, Any]) -> dict[str, pl.DataFrame]:
    """总体与边际来自唯一格子统计，全部包含特殊箱。"""
    overall = _derive(_sum(cells, _SCOPE), parameters)
    rows = _derive(_sum(cells, [*_SCOPE, "x_bin", "x_risk_rank"]), parameters)
    columns = _derive(_sum(cells, [*_SCOPE, "y_bin", "y_risk_rank"]), parameters)
    baseline = overall.select(
        *_SCOPE,
        pl.col("sample_count").alias("scope_sample_count"),
        pl.col("bad_rate").alias("overall_bad_rate"),
        pl.col("status").alias("overall_status"),
    )
    outputs: dict[str, pl.DataFrame] = {}
    for name, frame in (
        ("cells", _derive(cells, parameters)),
        ("row_summary", rows),
        ("column_summary", columns),
        ("overall", overall),
    ):
        frame = _left_join_nulls(frame, baseline, on=_SCOPE).with_columns(
            _ratio(pl.col("sample_count"), pl.col("scope_sample_count")).alias("sample_share"),
            _ratio(pl.col("bad_rate"), pl.col("overall_bad_rate")).alias("lift_vs_overall"),
            _lift_status(),
        )
        if name == "cells":
            frame = _left_join_nulls(
                frame,
                rows.select(*_SCOPE, "x_bin", pl.col("bad_rate").alias("row_bad_rate")),
                on=[*_SCOPE, "x_bin"],
            ).with_columns(
                (pl.col("bad_rate") - pl.col("row_bad_rate")).alias("delta_vs_row"),
                _ratio(pl.col("bad_rate"), pl.col("row_bad_rate")).alias("lift_vs_row"),
            )
        outputs[name] = frame
    return outputs


def cross_scores(
    df: pl.DataFrame | pd.DataFrame,
    *,
    score_x: str,
    score_y: str,
    score_directions: dict[str, str],
    targets: list[str] | None = None,
    cutpoints: dict[str, list[float]] | None = None,
    binning_reference: pl.DataFrame | pd.DataFrame | None = None,
    n_bins: int | dict[str, int] = 5,
    bin_definitions: dict[str, dict[str, Any]] | None = None,
    binning_type: Literal["native", "optimal", "lite_opt"] = "native",
    method: Literal["quantile", "uniform", "cart"] | None = None,
    min_bin_size: float | int | None = None,
    monotonic_trend: _ProfileRiskMonotonicTrend | None = None,
    binner_params: dict[str, Any] | None = None,
    binning_target: str | None = None,
    n_jobs: int | None = None,
    group_col: str | None = None,
    time_col: str | None = None,
    time_grain: str | None = None,
    weights_col: str | None = None,
    amount_col: str | None = None,
    special_values: dict[str, list[float]] | None = None,
    missing_values: list[Any] | dict[str, list[Any]] | None = None,
    probability_scores: list[str] | None = None,
    min_observed: int = 30,
    confidence_level: float = 0.95,
    max_scopes: int = 10000,
    feature_metadata: FeatureMetadata | None = None,
    business_context: dict[str, Any] | None = None,
) -> ScoreCrossReport:
    """用一次固定分段和联合聚合比较同一行上已对齐的两个模型分。

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        同一行已对齐的样本；跨引擎转换前投影必要字段，不合并独立数据集。
    score_x : str
        主模型分原始 ID。
    score_y : str
        辅助模型分原始 ID，必须不同于 score_x。
    score_directions : dict[str, str]
        两个 ID 均须声明 higher_risk（高分高风险）或 lower_risk（高分低风险）。
    targets : list[str] | None
        二元标签列表；缺失标签为未表现，可全好或全坏。None/空列表只分析分布。
    cutpoints : dict[str, list[float]] | None
        两个原始 ID 的有限切点；区间右闭，无界覆盖 reference 范围外的有限分。
    binning_reference : pl.DataFrame | pd.DataFrame | None
        用户指定的拟合参考集；不传时当前输入各轴拟合一次，绝不猜训练分组。
        监督分箱须在此参考集中提供 binning_target；跨组、周期和评估目标复用定义。
    n_bins : int | dict[str, int]
        请求正常段数，默认 5；也可按两个原始 score ID 指定不同箱数。实际箱数可能减少。
    bin_definitions : dict[str, dict[str, Any]] | None
        get_score_bin_definitions 的保存定义，使用时不拟合；与 reference/cutpoints 互斥。
    binning_type : Literal["native", "optimal", "lite_opt"]
        复用 profile_risk 的共享分箱引擎；默认 native。显式切点/保存定义不接受拟合配置。
    method : Literal["quantile", "uniform", "cart"] | None
        native 为等频、等宽或树；optimal/lite_opt 为预分箱方法。None 使用引擎默认值，
        native 默认 quantile。自定义区间使用 cutpoints，不支持 method="custom"。
    min_bin_size : float | int | None
        透传共享分箱器的最小箱大小约束；None 使用引擎默认值。native CART 支持整数人数。
    monotonic_trend : _ProfileRiskMonotonicTrend | None
        复用 profile_risk 趋势配置；监督最优分箱默认 auto_asc_desc，native 发警告并忽略。
    binner_params : dict[str, Any] | None
        复用 profile_risk 高级参数解析，如 native 的 cart_params、merge_small_bins、
        remove_empty_bins。公开参数和 prebinning_method 不可放入此映射。
    binning_target : str | None
        CART/optimal/lite_opt 的唯一拟合标签；None 默认首个 targets，并保存实际标签。
        可显式选择独立标签，参考集上校验 0/1/缺失及两个有效类别；不随评估目标重拟合。
    n_jobs : int | None
        共享分箱器并行参数；None 使用引擎默认值，-1 或正整数。
    group_col : str | None
        既有分组字段；与时间同时提供时只保留实际 group/period 组合。
    time_col : str | None
        日期字段，复用既有日期解析及时间粒度。
    time_grain : str | None
        month/day/week 等既有语义；须配 time_col。
    weights_col : str | None
        有限非负权重，允许零；bad_rate 使用 bad_weight_sum / observed_weight_sum。
    amount_col : str | None
        复用金额 helper：null/NaN/负值不贡献金额；非有限非缺失金额报错。
    special_values : dict[str, list[float]] | None
        每个 score 的显式有限特殊值，单独保存为特殊箱，不混入风险等级。
    missing_values : list[Any] | dict[str, list[Any]] | None
        共享缺失值语义；一个列表应用于两轴，或按原始 score ID 指定。随定义保存、加载复用。
    probability_scores : list[str] | None
        明确声明为 [0,1] 概率的 score；域外值进入 invalid，普通分不限制域。
    min_observed : int
        低样本提示门槛，仅影响状态，保留实际风险。
    confidence_level : float
        未加权整数 bad/n 的 Wilson 置信水平；加权区间不支持。
    max_scopes : int
        实际 group/period 组合上限；输出规模为 scopes × targets × 完整箱对。
    feature_metadata : FeatureMetadata | None
        复用原始 ID、展示名、来源、业务描述和单位格式。
    business_context : dict[str, Any] | None
        样本及标签业务语义；未声明的单位、训练含义等记录 unknown。

    Returns
    -------
    ScoreCrossReport
        独立 bins/cells/边际/overall 快照；加载后仍可用公共展示与规则回放函数。

    Raises
    ------
    ValueError
        列、方向、分段、标签、权重或规模配置无效。

    Notes
    -----
    计数和占比包含所有特殊箱；多个 target 的人数不可相加。比例为小数，delta
    为小数差。规则只精确回放完整保存分段，不支持箱内连续阈值的插值。

    Examples
    --------
    >>> report = cross_scores(df, score_x="main", score_y="aux", targets=["bad"],
    ...     score_directions={"main": "lower_risk", "aux": "higher_risk"})  # doctest: +SKIP
    """
    labels = list(targets or [])
    scores = [score_x, score_y]
    if score_x == score_y or len(set(labels)) != len(labels) or set(scores).intersection(labels):
        raise ValueError("Scores must differ and targets must be unique non-score fields.")
    if any(score_directions.get(s) not in {"higher_risk", "lower_risk"} for s in scores):
        raise ValueError("Explicit higher_risk/lower_risk direction required for both scores.")
    requested_bins: dict[str, int]
    if isinstance(n_bins, dict):
        if set(n_bins) != set(scores):
            raise ValueError("n_bins mapping must define both score IDs.")
        requested_bins = dict(n_bins)
    else:
        requested_bins = {s: n_bins for s in scores}
    if any(type(v) is not int or not 1 <= v <= 100 for v in requested_bins.values()):
        raise ValueError("n_bins must contain integers between 1 and 100.")
    if type(min_observed) is not int or min_observed < 0 or not 0 < confidence_level < 1:
        raise ValueError("Invalid min_observed or confidence_level.")
    if type(max_scopes) is not int or max_scopes < 1:
        raise ValueError("max_scopes must be positive.")
    if sum(v is not None for v in (cutpoints, binning_reference, bin_definitions)) > 1:
        raise ValueError("cutpoints, reference and saved definitions are mutually exclusive.")
    if cutpoints is not None and set(cutpoints) != set(scores):
        raise ValueError("cutpoints must define both score IDs.")
    if set(probability_scores or []) - set(scores) or set(special_values or {}) - set(scores):
        raise ValueError("Probability/special configurations must name the two scores.")
    if isinstance(missing_values, dict) and set(missing_values) - set(scores):
        raise ValueError("missing_values mapping must name the two scores.")
    missing_by_score: dict[str, list[Any] | None] = {
        s: missing_values.get(s) if isinstance(missing_values, dict) else missing_values
        for s in scores
    }
    if any(v is not None and not isinstance(v, list) for v in missing_by_score.values()):
        raise ValueError("missing_values must be a list or map score IDs to lists.")
    kind = _normalize_profile_risk_binning_type(binning_type)
    automatic = cutpoints is None and bin_definitions is None
    fitting_options = (
        kind != "native"
        or method is not None
        or min_bin_size is not None
        or monotonic_trend is not None
        or binner_params is not None
        or binning_target is not None
        or n_jobs is not None
    )
    if not automatic and fitting_options:
        raise ValueError("Fitting options cannot accompany explicit cutpoints or saved definitions.")
    supervised = automatic and (kind in {"optimal", "lite_opt"} or method == "cart")
    fit_target = binning_target or (labels[0] if supervised and labels else None)
    if supervised and fit_target is None:
        raise ValueError("Supervised binning requires binning_target or a first targets label.")
    if binning_target is not None and (not supervised or binning_target in scores):
        raise ValueError("binning_target must be a non-score label for supervised binning.")
    effective_params: dict[str, dict[str, Any]] = {}
    if automatic:
        for score in scores:
            effective_params[score] = _score_binning_params(
                binning_type=kind,
                method=method,
                n_bins=requested_bins[score],
                min_bin_size=min_bin_size,
                monotonic_trend=monotonic_trend,
                missing_values=missing_by_score[score],
                special_values=(special_values or {}).get(score, []),
                binner_params=binner_params,
                n_jobs=n_jobs,
            )
    columns = list(
        dict.fromkeys(
            [
                *scores,
                *labels,
                *([fit_target] if fit_target and binning_reference is None else []),
                *[c for c in (group_col, time_col, weights_col, amount_col) if c],
            ]
        )
    )
    reserved = {"__cross_group", "__cross_period", "x_bin", "y_bin"}
    if reserved.intersection(columns):
        raise ValueError(
            f"Input fields conflict with internal names: {reserved.intersection(columns)}."
        )
    frame = _project(df, columns)
    for target in labels:
        frame = normalize_binary_target_column(frame, target)
    if weights_col:
        w = pl.col(weights_col).cast(pl.Float64, strict=False)
        if frame.select((w.is_null() | ~w.is_finite() | (w < 0)).any()).item():
            raise ValueError("Weights must be finite, non-null and non-negative.")
        frame = frame.with_columns(w.alias(weights_col))
    if amount_col:
        a = pl.col(amount_col).cast(pl.Float64, strict=False)
        if frame.select(
            (pl.col(amount_col).is_not_null() & a.is_null() | a.is_infinite()).any()
        ).item():
            raise ValueError("Amount values must be numeric and cannot be infinite.")
        frame = frame.with_columns(a.fill_nan(None).alias(amount_col))
    # 分组和时间独立复用原上下文；网格只从真实组合生成。
    frame, _ = prepare_group_context(
        frame, group_col=group_col, time_col=None, time_grain=None, mars_group_col="__cross_group"
    )
    frame, _ = prepare_group_context(
        frame,
        group_col=None,
        time_col=time_col,
        time_grain=time_grain,
        mars_group_col="__cross_period",
    )
    scopes = frame.select("__cross_group", "__cross_period").unique(maintain_order=True)
    if scopes.height > max_scopes:
        raise ValueError(f"Actual scopes {scopes.height} exceed max_scopes={max_scopes}.")
    if bin_definitions is None:
        fit_source = (
            "explicit_cutpoints"
            if cutpoints is not None
            else "reference"
            if binning_reference is not None
            else "current_input"
        )
        reference_columns = [*scores, *([fit_target] if fit_target else [])]
        reference = (
            _project(binning_reference, reference_columns)
            if binning_reference is not None
            else frame.select(reference_columns)
        )
        definitions = {
            axis: _fit_axis(
                reference,
                score,
                score_directions[score],
                requested_bins[score],
                (special_values or {}).get(score, []),
                score in (probability_scores or []),
                binning_type=kind,
                binner_params=effective_params[score],
                target=fit_target,
                fit_source=fit_source,
            )
            if cutpoints is None
            else _axis_definition(
                score,
                score_directions[score],
                cutpoints[score],
                (special_values or {}).get(score, []),
                score in (probability_scores or []),
                missing_by_score[score],
            )
            for axis, score in zip(("x", "y"), scores)
        }
    else:
        definitions = deepcopy(bin_definitions)
        if set(definitions) != {"x", "y"}:
            raise ValueError("Saved definitions must contain x and y.")
        for axis, score in zip(("x", "y"), scores):
            d = definitions[axis]
            if (
                d.get("score") != score
                or d.get("direction") != score_directions[score]
                or d.get("closed") != "right"
            ):
                raise ValueError("Saved axis score, direction or endpoint semantics differ.")
            validated = _axis_definition(
                score,
                d["direction"],
                d["cutpoints"],
                d["special_values"],
                d["probability"],
                d.get("missing_values"),
            )
            if {k: v for k, v in d.items() if k != "fit"} != validated:
                raise ValueError("Invalid or inconsistent saved bin definition.")
            if "fit" in d and (
                not isinstance(d["fit"], dict)
                or d["fit"].get("actual_n_bins") != d["actual_n_bins"]
                or d["fit"].get("closed") != "right"
            ):
                raise ValueError("Invalid saved bin fitting provenance.")
            if special_values is not None and special_values.get(score, []) != d["special_values"]:
                raise ValueError("special_values differ from saved definitions.")
            if probability_scores is not None and (score in probability_scores) != d["probability"]:
                raise ValueError("probability_scores differ from saved definitions.")
            if missing_values is not None and (missing_by_score[score] or []) != d.get("missing_values", []):
                raise ValueError("missing_values differ from saved definitions.")
        fit_source = "saved_definition"
    bins = _bins(definitions)
    frame = frame.with_columns(
        [_assign(definitions[axis], axis, frame.schema[score]) for axis, score in zip(("x", "y"), scores)]
    )
    keys = ["__cross_group", "__cross_period", "x_bin", "y_bin"]
    expressions = [pl.len().cast(pl.Int64).alias("sample_count")]
    if weights_col:
        expressions.append(pl.col(weights_col).sum().alias("weight_sum"))
    for i, target in enumerate(labels):
        stats = binary_stats_agg_exprs(
            target,
            count_col=f"t{i}_count",
            observed_count_col=f"t{i}_observed",
            bad_col=f"t{i}_bad",
        )
        expressions.extend(stats[1:])
        if weights_col:
            expressions.extend(
                binary_stats_agg_exprs(
                    target,
                    weight_col=weights_col,
                    count_col=f"t{i}_w",
                    observed_count_col=f"t{i}_wo",
                    bad_col=f"t{i}_wb",
                )[1:]
            )
        if amount_col:
            expressions.extend(
                amount_stats_agg_exprs(
                    target,
                    amount_col,
                    total_amount_col=f"t{i}_amount",
                    good_amount_col=f"t{i}_good_amt",
                    bad_amount_col=f"t{i}_bad_amt",
                )
            )
    if not labels and amount_col:
        from mars.compute import total_amount_expr

        expressions.append(total_amount_expr(amount_col))
    aggregated = frame.group_by(keys, maintain_order=True).agg(expressions)
    axis_tables = {
        axis: bins.filter(pl.col("axis") == axis).select(
            pl.col("bin_id").alias(f"{axis}_bin"), pl.col("risk_rank").alias(f"{axis}_risk_rank")
        )
        for axis in ("x", "y")
    }
    grid = scopes.join(axis_tables["x"], how="cross").join(axis_tables["y"], how="cross")
    # 只填聚合数值，不能把特殊箱 rank 的 null 填成正常风险等级。
    numeric = aggregated.columns[len(keys) :]
    full = _left_join_nulls(grid, aggregated, on=keys).with_columns(pl.col(numeric).fill_null(0))
    chunks: list[pl.DataFrame] = []
    requested_targets: list[str | None] = list(labels) if labels else [None]
    for i, target in enumerate(requested_targets):
        selected = [
            pl.col("__cross_group").alias("group"),
            pl.col("__cross_period").alias("period"),
            pl.lit(target, dtype=pl.String).alias("target"),
            pl.col("x_bin"),
            pl.col("y_bin"),
            pl.col("x_risk_rank"),
            pl.col("y_risk_rank"),
            pl.col("sample_count"),
        ]
        if target is not None:
            selected.extend(
                [
                    pl.col(f"t{i}_observed").cast(pl.Int64).alias("observed_sample_count"),
                    pl.col(f"t{i}_bad").cast(pl.Int64).alias("bad_sample_count"),
                ]
            )
        else:
            selected.extend(
                [
                    pl.lit(None, dtype=pl.Int64).alias(c)
                    for c in ("observed_sample_count", "bad_sample_count")
                ]
            )
        if weights_col:
            selected.append(pl.col("weight_sum"))
            selected.extend(
                [
                    pl.col(f"t{i}_{source}").alias(dest)
                    if target is not None
                    else pl.lit(None, dtype=pl.Float64).alias(dest)
                    for source, dest in (("wo", "observed_weight_sum"), ("wb", "bad_weight_sum"))
                ]
            )
        if amount_col:
            selected.extend(
                [
                    pl.col(f"t{i}_{source}").alias(dest)
                    if target is not None
                    else pl.col("tot_amt")
                    if dest == "tot_amt"
                    else pl.lit(None, dtype=pl.Float64).alias(dest)
                    for source, dest in (
                        ("amount", "tot_amt"),
                        ("good_amt", "good_amt"),
                        ("bad_amt", "bad_amt"),
                    )
                ]
            )
        chunks.append(full.select(selected))
    cells = pl.concat(chunks).sort(
        [*_SCOPE, "x_risk_rank", "y_risk_rank", "x_bin", "y_bin"], nulls_last=True
    )
    fitted_targets = {axis: d.get("fit", {}).get("target") for axis, d in definitions.items()}
    common_target = fitted_targets["x"] if fitted_targets["x"] == fitted_targets["y"] else None
    parameters = {
        "score_x": score_x,
        "score_y": score_y,
        "score_directions": score_directions,
        "targets": labels,
        "requested_n_bins": n_bins,
        "actual_n_bins": {d["score"]: d["actual_n_bins"] for d in definitions.values()},
        "bin_definitions": definitions,
        "binning_type": kind if automatic else None,
        "binning_target": common_target,
        "fitted_targets": fitted_targets,
        "fit_performed": automatic,
        "binning_parameters": {axis: d.get("fit") for axis, d in definitions.items()},
        "binning_reference_policy": "explicit reference or current input; fitted once per axis across all scopes/targets",
        "fit_source": fit_source,
        "input_row_count": len(df),
        "projected_columns": columns,
        "group_col": group_col,
        "time_col": time_col,
        "time_grain": time_grain,
        "actual_scope_count": scopes.height,
        "max_scopes": max_scopes,
        "weights_col": weights_col,
        "amount_col": amount_col,
        "min_observed": min_observed,
        "confidence_level": confidence_level,
        "sample_share_denominator": "all samples including special bins, per target/group/period",
        "bad_rate_denominator": "observed_weight_sum" if weights_col else "observed_sample_count",
        "confidence_interval": "unweighted integer Wilson; weighted CI unsupported",
        "amount_policy": "existing helper: nonnegative amounts contribute; negative/null/NaN contribute zero; infinity rejected",
        "score_missing_policy": "null/NaN/custom missing_values -> missing; nonfinite/nonnumeric/probability domain violation -> invalid",
        "ratio_unit": "fraction; delta fraction difference",
        "sample_semantics": "historical sample retention; population extrapolation unknown",
    }
    tables = _marginals(cells, parameters)
    tables["bins"] = bins.with_columns(pl.lit(fit_source).alias("fit_source"))
    snapshot = _result_report(
        "score_cross",
        tables,
        parameters,
        scores,
        feature_metadata,
        business_context,
        scopes={name: scores for name in tables if name != "bins"},
        scope_roles={"score_x": score_x, "score_y": score_y},
        definitions={
            "sample_count": "all samples in cell/scope including score specials",
            "observed_sample_count": "non-null normalized binary label count",
            "bad_sample_count": "label=1 count among observed",
            "delta_vs_row": "bad_rate - X marginal bad_rate; fraction difference",
            "unweighted_ci_lower": "integer bad/n Wilson lower; never weighted rate CI",
            "unweighted_ci_upper": "integer bad/n Wilson upper; never weighted rate CI",
        },
        grains={
            "cells": "target/group/period/x_bin/y_bin",
            "row_summary": "target/group/period/x_bin",
            "column_summary": "target/group/period/y_bin",
            "overall": "target/group/period",
            "bins": "axis/bin_id",
        },
    )
    return ScoreCrossReport(snapshot._tables, snapshot.describe())


def get_score_bin_definitions(report: Report) -> dict[str, dict[str, Any]]:
    """提取保存的固定定义供下一报告复用，不拟合或读取原始样本。

    Parameters
    ----------
    report : Report
        score_cross 报告或加载后的快照。

    Returns
    -------
    dict[str, dict[str, Any]]
        方向、切点、端点和特殊值定义的独立副本。

    Raises
    ------
    ValueError
        报告类型无效。

    Examples
    --------
    >>> definitions = get_score_bin_definitions(restored)  # doctest: +SKIP
    """
    if report.report_type != "score_cross":
        raise ValueError("Expected a score_cross report.")
    return deepcopy(report.describe()["parameters"]["bin_definitions"])


def _policy_mask(
    cells: pl.DataFrame, rule: dict[str, Any], definitions: dict[str, Any]
) -> pl.Series:
    """校验声明式完整分段规则；只依赖相应轴，禁止连续箱内阈值。"""
    kind = rule.get("type")
    common = {"type", "missing_score", "accepted_special_bins"}
    fields = {
        "x_only": {"x_max_risk_rank"},
        "y_only": {"y_max_risk_rank"},
        "and": {"x_max_risk_rank", "y_max_risk_rank"},
        "or": {"x_max_risk_rank", "y_max_risk_rank"},
        "staircase": {"steps"},
        "expression": {"expression"},
    }
    if kind not in fields or set(rule) - (common | fields[kind]) or not fields[kind].issubset(rule):
        raise ValueError(
            "Use supported bin/risk_rank policy fields; continuous thresholds require raw data."
        )
    if rule.get("missing_score", "reject") != "reject":
        raise ValueError(
            "Only missing_score='reject' is supported; explicit specials may be accepted."
        )
    if kind == "expression":
        if "accepted_special_bins" in rule and rule["accepted_special_bins"] != {}:
            raise ValueError("分箱表达式只覆盖正常箱；特殊箱请使用已有显式 policy 策略。")
        ast = _parse_score_expression(
            rule["expression"], definitions["x"]["actual_n_bins"], definitions["y"]["actual_n_bins"]
        )
        return pl.Series(
            "pass",
            [
                _evaluate_score_expression(ast, x, y)
                for x, y in cells.select("x_risk_rank", "y_risk_rank").iter_rows()
            ],
            dtype=pl.Boolean,
        )
    specials = rule.get("accepted_special_bins", {})
    if not isinstance(specials, dict) or set(specials) - {"x", "y"}:
        raise ValueError("accepted_special_bins must map x/y to explicit special bin IDs.")
    for axis, ids in specials.items():
        known = {
            "missing",
            "invalid",
            *[f"s{i}" for i in range(len(definitions[axis]["special_values"]))],
        }
        if not isinstance(ids, list) or set(ids) - known:
            raise ValueError("Unknown accepted special bin IDs.")

    def axis_pass(axis: str, rank: Any) -> pl.Expr:
        """轴的完整正常段门槛及显式特殊箱接受条件。"""
        n = definitions[axis]["actual_n_bins"]
        if type(rank) is not int or not 0 <= rank <= n:
            raise ValueError(f"{axis}_max_risk_rank must be an integer within 0..{n}.")
        return (pl.col(f"{axis}_risk_rank") <= rank).fill_null(False) | pl.col(f"{axis}_bin").is_in(
            specials.get(axis, [])
        )

    if kind == "staircase":
        steps = rule["steps"]
        if not isinstance(steps, dict):
            raise ValueError("staircase steps must map actual X bin IDs to actions.")
        known_x = {f"b{i}" for i in range(definitions["x"]["actual_n_bins"])}
        known_x.update(specials.get("x", []))
        if set(steps) - known_x:
            raise ValueError("Unknown or unaccepted special X staircase bin.")
        mask = pl.lit(False)
        for bin_id, step in steps.items():
            if step == {"action": "reject"}:
                continue
            if (
                not isinstance(step, dict)
                or set(step) != {"action", "y_max_risk_rank"}
                or step["action"] != "accept"
            ):
                raise ValueError(
                    "Each step is reject, or accept with y_max_risk_rank; omitted X bins reject."
                )
            mask = mask | ((pl.col("x_bin") == bin_id) & axis_pass("y", step["y_max_risk_rank"]))
    elif kind == "x_only":
        mask = axis_pass("x", rule["x_max_risk_rank"])
    elif kind == "y_only":
        mask = axis_pass("y", rule["y_max_risk_rank"])
    else:
        x = axis_pass("x", rule["x_max_risk_rank"])
        y = axis_pass("y", rule["y_max_risk_rank"])
        mask = x & y if kind == "and" else x | y
    return cells.select(mask.alias("pass")).to_series()


def _partition_summary(
    cells: pl.DataFrame,
    scope: pl.DataFrame,
    keys: list[str],
    parameters: dict[str, Any],
) -> pl.DataFrame:
    """每区域重算整数统计和风险，保留原样本好坏分母。"""
    # 所有规则区域均存在，即使某个区域没有格子或全量拒绝，也保存整数零和 empty。
    grid = scope.select(_SCOPE)
    for key in keys:
        grid = grid.join(pl.DataFrame({key: [False, True]}), how="cross")
    raw = _sum(cells, [*_SCOPE, *keys])
    raw = _left_join_nulls(grid, raw, on=[*_SCOPE, *keys]).with_columns(
        pl.col([c for c in _COUNTS if c in raw.columns]).fill_null(0)
    )
    summary = _derive(raw, parameters)
    totals = scope.select(
        *_SCOPE,
        pl.col("sample_count").alias("original_sample_count"),
        pl.col("observed_sample_count").alias("original_observed_count"),
        pl.col("bad_sample_count").alias("original_bad_count"),
        pl.col("bad_rate").alias("overall_bad_rate"),
        pl.col("status").alias("overall_status"),
    )
    summary = _left_join_nulls(summary, totals, on=_SCOPE)
    return summary.with_columns(
        _ratio(pl.col("sample_count"), pl.col("original_sample_count")).alias("sample_share"),
        _ratio(pl.col("bad_rate"), pl.col("overall_bad_rate")).alias("lift_vs_overall"),
        _lift_status(),
        (pl.col("observed_sample_count") - pl.col("bad_sample_count")).alias("good_sample_count"),
        _ratio(pl.col("bad_sample_count"), pl.col("original_bad_count")).alias("bad_sample_share"),
        _ratio(
            pl.col("observed_sample_count") - pl.col("bad_sample_count"),
            pl.col("original_observed_count") - pl.col("original_bad_count"),
        ).alias("good_sample_share"),
    )


def evaluate_score_policy(
    report: Report,
    candidate: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
) -> ReportSnapshot:
    """仅从已保存格子回放完整分段组合，返回绑定父报告的独立派生快照。

    Parameters
    ----------
    report : Report
        score_cross 报告，支持 load_report 的通用快照；不需要原样本。
    candidate : dict[str, Any]
        type 为 x_only/y_only/and/or/staircase/expression。统一规则使用 x/y_max_risk_rank；
        staircase 使用 steps={实际 X bin_id: {action: accept, y_max_risk_rank: 整数}}，
        整段拒绝为 {action: reject}，未列出的 X 段拒绝。missing_score 默认 reject，
        accepted_special_bins 可显式接受特殊箱。仅依赖的轴影响通过条件。
        expression 使用 expression="X <= X2 AND Y <= Y3"；编号为保存的正常箱风险
        序位，支持同轴标签或整数、比较符及大小写不敏感的 AND/OR，AND 优先于 OR。
        括号覆盖优先级；最多 240 字符、96 词元、12 层括号。只覆盖完整正常箱集合，
        不接受连续阈值或隐含特殊箱；不产生自动审批语义。
    baseline : dict[str, Any] | None
        显式基准规则；None 为全原样本基准，记录为 all_samples。

    Returns
    -------
    ReportSnapshot
        summary（候选/基准的留存和剔除）、regions（四布尔区域）、changes（实际覆盖
        和风险差）、axis_regions（统一 X/Y 门槛时）、cell_decisions 和 bins。保存不会
        改父报告；不自动写文件，不输出上线结论或外推拒绝样本标签。

    Raises
    ------
    ValueError
        报告类型、规则或分段门槛无效；连续箱内阈值无法精确回放。

    Examples
    --------
    >>> replay = evaluate_score_policy(restored,
    ...     {"type": "and", "x_max_risk_rank": 3, "y_max_risk_rank": 3})  # doctest: +SKIP
    >>> expression = evaluate_score_policy(restored,
    ...     {"type": "expression", "expression": "X <= X2 AND Y >= Y3"})  # doctest: +SKIP
    """
    if report.report_type != "score_cross":
        raise ValueError("Expected a score_cross report.")
    definitions = get_score_bin_definitions(report)
    description = report.describe()
    parameters = deepcopy(description["parameters"])
    cells = report.get_table("cells")
    candidate_pass = _policy_mask(cells, candidate, definitions)
    baseline_pass = (
        _policy_mask(cells, baseline, definitions)
        if baseline is not None
        else pl.Series("baseline", [True] * len(cells), dtype=pl.Boolean)
    )
    decisions = cells.with_columns(
        candidate_pass.alias("candidate_pass"), baseline_pass.alias("baseline_pass")
    )
    overall = report.get_table("overall")
    regions = _partition_summary(
        decisions, overall, ["baseline_pass", "candidate_pass"], parameters
    )
    chunks: list[pl.DataFrame] = []
    for name, column in (("candidate", "candidate_pass"), ("baseline", "baseline_pass")):
        chunk = decisions.with_columns(pl.col(column).alias("retained"))
        chunks.append(
            _partition_summary(chunk, overall, ["retained"], parameters).with_columns(
                pl.lit(name).alias("rule")
            )
        )
    summary = pl.concat(chunks)
    retained = summary.filter(pl.col("retained"))
    change = retained.filter(pl.col("rule") == "candidate").select(
        *_SCOPE,
        pl.col("sample_count").alias("candidate_retained_count"),
        pl.col("sample_share").alias("candidate_retained_share"),
        pl.col("bad_rate").alias("candidate_bad_rate"),
    )
    change = _left_join_nulls(
        change,
        retained.filter(pl.col("rule") == "baseline").select(
            *_SCOPE,
            pl.col("sample_count").alias("baseline_retained_count"),
            pl.col("sample_share").alias("baseline_retained_share"),
            pl.col("bad_rate").alias("baseline_bad_rate"),
        ),
        on=_SCOPE,
    ).with_columns(
        (pl.col("candidate_retained_count") - pl.col("baseline_retained_count")).alias(
            "retained_count_delta"
        ),
        (pl.col("candidate_retained_share") - pl.col("baseline_retained_share")).alias(
            "retained_share_delta"
        ),
        (pl.col("candidate_bad_rate") - pl.col("baseline_bad_rate")).alias("bad_rate_delta"),
    )
    tables = {
        "summary": summary,
        "regions": regions,
        "changes": change,
        "cell_decisions": decisions.select(
            *_SCOPE, "x_bin", "y_bin", "baseline_pass", "candidate_pass", "sample_count"
        ),
        "bins": report.get_table("bins"),
    }
    if candidate["type"] in {"and", "or"}:
        x_rule = {**candidate, "type": "x_only"}
        y_rule = {**candidate, "type": "y_only"}
        x_rule.pop("y_max_risk_rank")
        y_rule.pop("x_max_risk_rank")
        axis_cells = cells.with_columns(
            _policy_mask(cells, x_rule, definitions).alias("x_pass"),
            _policy_mask(cells, y_rule, definitions).alias("y_pass"),
        )
        tables["axis_regions"] = _partition_summary(
            axis_cells, overall, ["x_pass", "y_pass"], parameters
        )
    parameters.update(
        parent_report_id=report.report_id,
        candidate={"missing_score": "reject", **candidate},
        baseline={"missing_score": "reject", **baseline}
        if baseline is not None
        else {"type": "all_samples"},
        replay_precision="exact whole saved bins only",
        comparison="historical sample retention; coverage differences explicit",
    )
    # 保存限定 AST 供机器核验；可回放的 candidate 本身保留原输入契约，不要求原始数据。
    for name, rule in (("candidate", candidate), ("baseline", baseline)):
        if rule is not None and rule["type"] == "expression":
            parameters[f"{name}_expression_ast"] = _parse_score_expression(
                rule["expression"], definitions["x"]["actual_n_bins"], definitions["y"]["actual_n_bins"]
            )
            parameters["expression_syntax"] = {
                "version": 1,
                "limits": dict(_EXPRESSION_LIMITS),
                "normal_bins_only": True,
                "ordering": "saved risk_rank ascending, stable across target/group/period",
            }
    return _result_report(
        "score_policy",
        tables,
        parameters,
        [parameters["score_x"], parameters["score_y"]],
        description["feature_metadata"],
        description["business_context"],
        scopes={
            name: [parameters["score_x"], parameters["score_y"]]
            for name in tables
            if name != "bins"
        },
        scope_roles={"score_x": parameters["score_x"], "score_y": parameters["score_y"]},
        definitions={
            "lift_status": "Lift 状态：valid 包括有效零；empty、unobserved、not_requested、invalid_denominator 表示不可用原因",
            "overall_status": "当前 target/group/period 整体坏账率状态；含全部特殊箱",
        },
        grains={
            "summary": "target/group/period/rule/retained",
            "regions": "target/group/period/baseline_pass/candidate_pass",
            "changes": "target/group/period",
            "cell_decisions": "target/group/period/x_bin/y_bin",
            "axis_regions": "target/group/period/x_pass/y_pass",
            "bins": "axis/bin_id",
        },
    )


class ScoreCrossReport(ReportSnapshot):
    """固定分段的独立双分数报告；专用方法委托公共函数供加载后的快照复用。

    Parameters
    ----------
    tables : dict[str, Any]
        已聚合的列式小统计表，不包含个体明细。
    description : dict[str, Any]
        身份、单位、分段定义及真实计算参数。

    Examples
    --------
    >>> report = cross_scores(df, score_x="main", score_y="aux",
    ...     score_directions={"main": "lower_risk", "aux": "higher_risk"})  # doctest: +SKIP
    """

    def evaluate_policy(
        self, candidate: dict[str, Any], *, baseline: dict[str, Any] | None = None
    ) -> ReportSnapshot:
        """从保存格子回放完整分段规则，保持父报告不变。

        Parameters
        ----------
        candidate : dict[str, Any]
            声明式 x_only/y_only/and/or/staircase/expression 规则，格式见 evaluate_score_policy。
        baseline : dict[str, Any] | None
            显式基准；None 为 all_samples。缺失默认拒绝，仅依赖轴影响通过条件。

        Returns
        -------
        ReportSnapshot
            可保存的规则、留存/剔除、四区域和实际覆盖差；绑定父 report_id。

        Raises
        ------
        ValueError
            规则或门槛无效；箱内连续阈值无法精确回放。

        Examples
        --------
        >>> report.evaluate_policy({"type": "x_only", "x_max_risk_rank": 3})  # doctest: +SKIP
        """  # 异常由共用回放函数传播；pydoclint noqa: DOC502
        return evaluate_score_policy(self, candidate, baseline=baseline)
