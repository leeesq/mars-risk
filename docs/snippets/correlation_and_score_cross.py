"""合成数据的相关性、交叉矩阵与外部 Agent 消费闭环。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from mars.analysis import (
    cross_scores,
    evaluate_score_policy,
    get_score_bin_definitions,
    get_score_cell,
    show_score_matrix,
    write_score_cross_html,
)
from mars.feature import MarsLinearSelector, MarsStatsSelector
from mars.reporting import (
    get_correlation_matrix,
    get_related_features,
    load_report,
    show_correlation_matrix,
)


def run(output: Path) -> None:
    """生成合成证据、离线 HTML/Excel，并在不传原始数据的情况下查询与回放。"""
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261001)
    n = 600
    latent = rng.normal(size=n)
    another = rng.normal(size=n)
    bad = (rng.random(n) < 1 / (1 + np.exp(-latent))).astype(float)
    late = (rng.random(n) < 1 / (1 + np.exp(-0.7 * latent - 0.4 * another))).astype(float)
    bad[::9] = np.nan
    late[::4] = np.nan
    data = pd.DataFrame(
        {"loan_cnt_3m": latent, "inq_cnt_1m": -latent, "income": another, "bad": bad}
    )
    metadata = {
        "loan_cnt_3m": {"display_name": "借贷次数", "data_source": "credit"},
        "inq_cnt_1m": {"display_name": "查询次数", "data_source": "credit"},
        "income": {"display_name": "收入", "data_source": "application"},
    }
    selector = MarsLinearSelector(corr_thr=0.95).fit(
        data.drop(columns="bad"),
        data["bad"],
        feature_metadata=metadata,
        business_context={"dataset_id": "synthetic-example", "sample_unit": "synthetic row"},
    )
    corr = selector.get_correlation_report()
    print("Synthetic Linear candidates:", corr.describe()["parameters"]["candidate_scope"])
    print(corr.get_related("loan_cnt_3m"))
    print(corr.get_table("correlation_decisions"))
    corr.save(output / "selector-correlation.marsreport", overwrite=True)
    restored_corr = load_report(output / "selector-correlation.marsreport")
    print(get_correlation_matrix(restored_corr, ["loan_cnt_3m", "inq_cnt_1m", "income"]))
    print(get_related_features(restored_corr, "inq_cnt_1m", sources="application"))
    matrix = show_correlation_matrix(restored_corr)
    (output / "correlation-matrix.html").write_text(matrix.to_html(), encoding="utf-8")
    matrix.to_excel(output / "correlation-matrix.xlsx")

    # WOE 与 raw 有不同业务含义，分别保存，不能把两份矩阵直接当作同一口径比较。
    stats = MarsStatsSelector(
        skip_fine_scan=True,
        psi_thr=None,
        rc_thr=None,
        corr_thr=0.8,
        rough_iv_thr=-1,
        rough_lift_thr=0,
        rough_binning_params={"method": "quantile", "n_bins": 4},
        n_jobs=1,
    )
    stats.fit(
        pl.from_pandas(data), target="bad", features=list(metadata), feature_metadata=metadata
    )
    stats.get_correlation_report().save(output / "woe-correlation.marsreport", overwrite=True)

    main = -latent + rng.normal(scale=0.4, size=n)
    auxiliary = 1 / (1 + np.exp(-0.6 * latent - 0.7 * another))
    main[::47] = np.nan
    auxiliary[::37] = np.nan
    cross_data = pd.DataFrame(
        {
            "score_main": main,
            "prob_aux": auxiliary,
            "bad": bad,
            "late": late,
            "split": ["TEST"] * 400 + ["OOT"] * 200,
        }
    )
    reference = pd.DataFrame(
        {"score_main": -rng.normal(size=1000), "prob_aux": rng.uniform(size=1000)}
    )
    directions = {"score_main": "lower_risk", "prob_aux": "higher_risk"}
    cross = cross_scores(
        cross_data,
        score_x="score_main",
        score_y="prob_aux",
        targets=["bad", "late"],
        score_directions=directions,
        binning_reference=reference,
        n_bins=5,
        group_col="split",
        probability_scores=["prob_aux"],
        min_observed=20,
        feature_metadata={
            "score_main": {"display_name": "主模型分", "data_source": "champion"},
            "prob_aux": {"display_name": "辅助概率", "data_source": "challenger"},
        },
        business_context={
            "dataset_id": "synthetic-example",
            "sample_unit": "synthetic row",
            "label_definition": "synthetic Bernoulli labels; no real performance claim",
        },
    )
    cross.save(output / "score-cross.marsreport", overwrite=True)
    restored = load_report(output / "score-cross.marsreport")
    print(get_score_cell(restored, "b2", "b2", filters={"target": "bad", "group": "OOT"}))
    show_score_matrix(
        restored, filters={"target": "bad", "group": "OOT"}, metric="delta_vs_row"
    ).to_excel(output / "row-delta.xlsx")
    baseline = {"type": "x_only", "x_max_risk_rank": 3, "missing_score": "reject"}
    rules = [
        baseline,
        {"type": "and", "x_max_risk_rank": 3, "y_max_risk_rank": 3},
        {"type": "or", "x_max_risk_rank": 3, "y_max_risk_rank": 3},
        {
            "type": "staircase",
            "steps": {
                "b4": {"action": "accept", "y_max_risk_rank": 4},
                "b3": {"action": "accept", "y_max_risk_rank": 3},
                "b2": {"action": "accept", "y_max_risk_rank": 2},
                "b1": {"action": "reject"},
            },
        },
    ]
    policies = [evaluate_score_policy(restored, rule, baseline=baseline) for rule in rules]
    for i, policy in enumerate(policies):
        policy.save(output / f"policy-{i}.marsreport", overwrite=True)
        policy.write_excel(str(output / f"policy-{i}.xlsx"))
        print(policy.get_table("changes"))
    write_score_cross_html(
        restored,
        output / "score-cross.html",
        policy_reports=policies,
        report_name="合成双模型分示例：历史样本回放",
    )
    restored.write_excel(str(output / "score-cross.xlsx"))

    # 外部 Agent 只读取公开 JSON 与分页证据，无 session、原样本、模型或 LLM 调用。
    context = restored.to_ai_context(
        queries={
            "cells": {
                "filters": {"target": "bad", "group": "OOT", "x_bin": "b2"},
                "sort_by": "y_risk_rank",
                "limit": 8,
            }
        },
        max_chars=16000,
    )
    json.loads(context)
    (output / "external-agent-context.json").write_text(context, encoding="utf-8")
    page = restored.query_page("cells", filters={"target": "bad", "group": "OOT"}, limit=10)
    if page["next_offset"] is not None:
        print(
            restored.query_page(
                "cells",
                filters={"target": "bad", "group": "OOT"},
                offset=page["next_offset"],
                limit=10,
            )["reference"]
        )
    definitions = get_score_bin_definitions(restored)
    reused = cross_scores(
        cross_data,
        score_x="score_main",
        score_y="prob_aux",
        score_directions=directions,
        bin_definitions=definitions,
    )
    assert get_score_bin_definitions(reused) == definitions


def main() -> None:
    """运行合成示例，默认产物保存到 examples/output/score-reports。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("examples/output/score-reports"))
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
