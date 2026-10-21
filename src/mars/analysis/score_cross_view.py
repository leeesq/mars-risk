"""聚合双分数报告的 Notebook 与自包含离线 HTML 视图。"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from mars.reporting._artifact import Report
from mars.reporting._serialization import encode

from ._score_cross_expression import (
    _EXPRESSION_JAVASCRIPT,
    _EXPRESSION_LIMITS,
    _score_rule_examples,
)
from ._score_cross_html import SCORE_CROSS_HTML
from .score_cross import _COUNTS, get_score_bin_definitions


def get_score_cell(
    report: Report,
    x_bin: str,
    y_bin: str,
    *,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """查询指定交叉格子的统计证据和固定区间。

    Parameters
    ----------
    report : Report
        原始 score_cross 报告或加载后的快照。
    x_bin : str
        已保存的 X 轴 bin_id，例如 ``"b0"``；使用原始箱标识，不使用展示编号 X1。
    y_bin : str
        已保存的 Y 轴 bin_id，例如 ``"b1"``；使用原始箱标识，不使用展示编号 Y1。
    filters : dict[str, Any] | None
        按 target、group 和 period 等字段筛选。默认返回此格子在全部样本范围下的记录，
        每条记录保留目标、分组和周期。

    Returns
    -------
    dict[str, Any]
        ``page`` 为格子统计查询页，包含 report_id、table 和 query；
        ``intervals`` 为按 x、y 轴组织的原生区间表。

    Raises
    ------
    ValueError
        报告或 bin_id 无效。

    Examples
    --------
    >>> get_score_cell(restored, "b0", "b1", filters={"group": "OOT"})  # doctest: +SKIP
    """
    get_score_bin_definitions(report)
    intervals = {
        axis: report.get_table("bins", filters={"axis": axis, "bin_id": bin_id})
        for axis, bin_id in (("x", x_bin), ("y", y_bin))
    }
    if any(not len(frame) for frame in intervals.values()):
        raise ValueError("Unknown bin_id.")
    return {
        "page": report.query_page(
            "cells", filters={**(filters or {}), "x_bin": x_bin, "y_bin": y_bin}, limit=None
        ),
        "intervals": intervals,
    }


def show_score_matrix(
    report: Report,
    *,
    filters: dict[str, Any] | None = None,
    metric: str = "bad_rate",
    include_special: bool = False,
    color_range: tuple[float, float] | None = None,
) -> Any:
    """显示单个样本范围的双分数矩阵，按低风险到高风险排列。

    Parameters
    ----------
    report : Report
        score_cross 报告或加载后的快照。
    filters : dict[str, Any] | None
        按 target、group 和 period 筛选。默认使用首个匹配范围，并在 caption 中注明。
    metric : str
        格子展示指标，可选 bad_rate、delta_vs_row 和 sample_share，默认 bad_rate。
        坏率和样本占比显示为百分比，行坏率差显示为百分点。
    include_special : bool
        是否展示特殊箱，默认折叠并注明省略的样本数。折叠后仍使用包含特殊箱的边际和总体。
    color_range : tuple[float, float] | None
        用小数指定色标上下界。默认坏率和样本占比使用 [0, 1]，行坏率差使用 [-1, 1]。
        上下界须递增，行坏率差的色标须关于 0 对称。

    Returns
    -------
    Any
        可在 Notebook 显示、用 to_html/to_excel 导出的 Pandas Styler。
        格子标注有表现人数；TOTAL 展示重算后的边际指标，行坏率差视图的 TOTAL 展示坏率。

    Raises
    ------
    ValueError
        报告、指标、色标或样本范围无效。

    Examples
    --------
    >>> show_score_matrix(restored, filters={"target": "bad", "group": "OOT"})  # doctest: +SKIP
    """
    definitions = get_score_bin_definitions(report)
    if metric not in {"bad_rate", "delta_vs_row", "sample_share"}:
        raise ValueError("metric must be bad_rate, delta_vs_row or sample_share.")
    scopes = report.get_table("overall", filters=filters, limit=1)
    if not len(scopes):
        raise ValueError("No matching sample scope.")
    scope = {c: scopes[c][0] for c in ("target", "group", "period")}
    all_cells = report.get_table("cells", filters=scope)
    omitted = 0
    if not include_special:
        normal = pl.col("x_risk_rank").is_not_null() & pl.col("y_risk_rank").is_not_null()
        omitted = all_cells.filter(~normal)["sample_count"].sum()
        all_cells = all_cells.filter(normal)
    columns = ["x_bin", "y_bin", metric, "observed_sample_count", "status"]
    # 只有选定范围和展示列转换 Pandas；颜色与文本共享同一页证据。
    frame = all_cells.select(columns).to_pandas()
    x_bins = report.get_table("bins", filters={"axis": "x"}).to_dicts()
    y_bins = report.get_table("bins", filters={"axis": "y"}).to_dicts()
    xs = [r["bin_id"] for r in x_bins if include_special or r["kind"] == "normal"]
    ys = [r["bin_id"] for r in y_bins if include_special or r["kind"] == "normal"]
    values = frame.pivot(index="x_bin", columns="y_bin", values=metric).reindex(
        index=xs, columns=ys
    )
    text = pd.DataFrame("", index=xs, columns=ys)
    statuses = pd.DataFrame("", index=xs, columns=ys)
    for row in frame.to_dict("records"):
        value = row[metric]
        risk = (
            "—"
            if pd.isna(value)
            else f"{value * 100:.2f}" + ("pp" if metric == "delta_vs_row" else "%")
        )
        count = row["observed_sample_count"]
        text.loc[row["x_bin"], row["y_bin"]] = (
            f"{risk} | n={'—' if pd.isna(count) else int(count)} | {row['status']}"
        )
        statuses.loc[row["x_bin"], row["y_bin"]] = row["status"]
    # TOTAL 直接读取边际重算后的指标；行坏率差视图的 TOTAL 展示边际坏率。
    margin_metric = "sample_share" if metric == "sample_share" else "bad_rate"
    for axis, ids, table in (("x", xs, "row_summary"), ("y", ys, "column_summary")):
        marginal = report.get_table(table, filters=scope)
        for row in marginal.iter_rows(named=True):
            if row[f"{axis}_bin"] not in ids:
                continue
            v = row[margin_metric]
            label = "—" if v is None else f"{100 * v:.2f}%"
            if axis == "x":
                text.loc[row["x_bin"], "TOTAL"] = label
            else:
                text.loc["TOTAL", row["y_bin"]] = label
    v = scopes[margin_metric][0]
    text.loc["TOTAL", "TOTAL"] = "—" if v is None else f"{100 * v:.2f}%"
    bounds = color_range or ((-1.0, 1.0) if metric == "delta_vs_row" else (0.0, 1.0))
    if not bounds[0] < bounds[1] or metric == "delta_vs_row" and bounds[0] != -bounds[1]:
        raise ValueError("Color range must increase; delta range must be centered on zero.")
    styles = pd.DataFrame("", index=text.index, columns=text.columns)
    for x in xs:
        for y in ys:
            value = values.loc[x, y]
            if pd.isna(value):
                color = "#eee"
            else:
                strength = min(
                    1.0,
                    max(
                        0.0,
                        abs(value) / bounds[1]
                        if metric == "delta_vs_row"
                        else (value - bounds[0]) / (bounds[1] - bounds[0]),
                    ),
                )
                color = f"rgba({'45,100,200' if metric == 'delta_vs_row' and value < 0 else '210,60,45'},{0.08 + 0.65 * strength})"
            styles.loc[x, y] = f"background-color:{color};" + (
                "border:2px dashed #777;" if statuses.loc[x, y] == "low_sample" else ""
            )
    metadata = report.describe()["feature_metadata"]
    names = [
        f"{metadata.get(d['score'], {}).get('display_name') or d['score']} [{d['score']}]"
        for d in definitions.values()
    ]
    caption = f"X={names[0]}; Y={names[1]}; {scope}; omitted special samples={omitted}; TOTAL={margin_metric}; low_sample dashed border"
    return (
        text.style.apply(lambda _: styles, axis=None)
        .format(escape="html")
        .format_index(escape="html", axis=0)
        .format_index(escape="html", axis=1)
        .set_caption(escape(caption))
    )


def write_score_cross_html(
    report: Report,
    path: str | Path,
    *,
    report_name: str = "MARS Score Cross",
    policy_reports: list[Report] | None = None,
) -> None:
    """导出自包含 HTML，展示交叉矩阵、沿两轴的风险变化和分箱规则回放。

    Parameters
    ----------
    report : Report
        score_cross 报告，支持加载后的快照。
    path : str | Path
        目标路径；父目录须存在。
    report_name : str
        页面标题，默认 ``"MARS Score Cross"``；写入 HTML 前会转义。
    policy_reports : list[Report] | None
        已回放的派生规则报告，须来自当前 report_id。提供后可离线查看留存、
        四区域、覆盖差和格子差异；默认不附加派生报告。

    Returns
    -------
    None
        写入自包含 HTML；使用保存分箱、行列边际及包含特殊箱的 overall。
        点击或键盘选择不改变规则；规则只匹配常规完整分箱，独立应用或清除。

    Raises
    ------
    ValueError
        报告或规则父 ID 不匹配。

    Examples
    --------
    >>> write_score_cross_html(restored, "cross.html", policy_reports=[replay])  # doctest: +SKIP
    """
    definitions: dict[str, dict[str, Any]] = get_score_bin_definitions(report)
    bins: pl.DataFrame = report.get_table("bins")
    normal_counts: dict[str, int] = {
        axis: bins.filter((pl.col("axis") == axis) & (pl.col("kind") == "normal")).height
        for axis in ("x", "y")
    }
    parameters: dict[str, Any] = report.describe()["parameters"]
    cell_columns: list[str] = report.get_table("cells").columns
    policies: list[dict[str, Any]] = []
    for policy in policy_reports or []:
        if (
            policy.report_type != "score_policy"
            or policy.describe()["parameters"]["parent_report_id"] != report.report_id
        ):
            raise ValueError("Policy report must derive from this report_id.")
        policies.append(
            {
                "description": policy.describe(),
                "tables": {n: policy.get_table(n).to_dicts() for n in policy.describe()["tables"]},
            }
        )
    payload: dict[str, Any] = {
        "description": report.describe(),
        "tables": {n: report.get_table(n).to_dicts() for n in report.describe()["tables"]},
        "policies": policies,
        "bin_definitions": definitions,
        "expression_limits": _EXPRESSION_LIMITS,
        "rule_examples": _score_rule_examples(normal_counts["x"], normal_counts["y"]),
        "rule_contract": {
            "count_fields": [field for field in _COUNTS if field in cell_columns],
            "weighted": parameters["weights_col"] is not None,
            "sample_share_denominator": "overall.sample_count including special bins",
            "lift_denominator": "overall.bad_rate including special bins",
        },
    }
    serialized = encode(payload).replace("<", "\\u003c").replace("&", "\\u0026")
    replacements: dict[str, str] = {
        "TITLE": escape(report_name),
        "DATA": serialized,
        "EXPRESSION_JS": _EXPRESSION_JAVASCRIPT,
    }
    # 单次替换固定模板标记；用户文本中的同名字符串不能被再次展开。
    html = re.sub(
        r"__(TITLE|DATA|EXPRESSION_JS)__",
        lambda match: replacements[match[1]],
        SCORE_CROSS_HTML,
    )
    Path(path).write_text(html, encoding="utf-8")
