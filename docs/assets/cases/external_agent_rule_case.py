"""模拟消费信贷：模型分交叉、候选验证与快照独立消费；无需 LLM/API Key。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mars.analysis import cross_scores, get_score_bin_definitions
from mars.reporting import Report, load_report
from mars.rule import MarsRule, MarsRuleMiningSpec, mine_rules

QUESTION = "同一主模型分等级内，辅助分能否进一步区分风险？哪些组合规则值得进入独立验证？"
METADATA = {
    "main_score": {
        "display_name": "主模型分",
        "data_source": "champion",
        "description": "模拟固定分，高分低风险",
        "unit": "score_points",
    },
    "aux_score": {
        "display_name": "辅助模型分",
        "data_source": "challenger",
        "description": "模拟固定分，高分高风险",
        "unit": "score_points",
    },
}
DIRECTIONS = {"main_score": "lower_risk", "aux_score": "higher_risk"}


def encode(value: Any) -> str:
    """案例仅编码有限统计和公开目录；通用非有限编码交给 Report.to_ai_context。"""
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def table_rows(frame: Any) -> list[dict[str, Any]]:
    """仅物化已分页的小证据表，兼容公共接口的两种表后端。"""
    return frame.to_dicts() if isinstance(frame, pl.DataFrame) else frame.to_dict("records")


def _sample(n: int = 18000, seed: int = 20261001) -> pl.DataFrame:
    """用固定种子生成独立时间样本；标签缺失不填好样本，模型分无需训练。"""
    if n < 90 or n % 9:
        raise ValueError("n 必须不少于 90 且可被 9 整除，以保留九个月和三个独立分区。")
    rng = np.random.default_rng(seed)
    main, auxiliary = rng.normal(size=(2, n))
    probability = 1 / (1 + np.exp(3.2 + main - 2.4 * auxiliary))
    bad = (rng.random(n) < probability).astype(float)
    late = (rng.random(n) < 1 / (1 + np.exp(2.8 + 0.8 * main - 1.9 * auxiliary))).astype(float)
    bad[rng.random(n) < 0.08] = np.nan
    late[rng.random(n) < 0.20] = np.nan
    main_score, aux_score = 600 + 80 * main, 500 + 100 * auxiliary
    main_score[::211] = np.nan
    aux_score[::157] = -999
    aux_score[::199] = np.nan
    return pl.DataFrame(
        {
            "sample_id": [f"S{i:05d}" for i in range(n)],
            "application_date": [f"2026-{i // (n // 9) + 1:02d}-15" for i in range(n)],
            "segment": rng.choice(["new", "returning"], n),
            "dataset": [
                "discovery" if i < n // 3 else "validation" if i < 2 * n // 3 else "observation"
                for i in range(n)
            ],
            "main_score": main_score,
            "aux_score": aux_score,
            "bad30": bad,
            "late60": late,
            "amount": rng.uniform(2000, 30000, n),
        }
    ).with_columns(pl.col("bad30", "late60", "main_score", "aux_score").fill_nan(None))


def _condition(feature: str, definition: dict[str, Any], index: int) -> str:
    """将现有右闭正常分段显式写为现有 DSL，不新增语法。"""
    cuts = definition["cutpoints"]
    parts = [f"{feature} IS NOT MISSING", f"{feature} != -999"]
    if index:
        parts.append(f"{feature} > {cuts[index - 1]!r}")
    if index < len(cuts):
        parts.append(f"{feature} <= {cuts[index]!r}")
    return " AND ".join(parts)


def _page(report: Report, table: str, query: dict[str, Any], question: str) -> dict[str, Any]:
    """仅对按需分页后的小统计表编码，引用可在同一快照重放。"""
    page = report.query_page(table, **query)
    return {
        "question": question,
        "reference": page["reference"],
        "total_rows": page["total_rows"],
        "next_offset": page["next_offset"],
        "omitted_rows": page["omitted_rows"],
        "rows": table_rows(page["data"]),
    }


def produce(
    output_dir: Path,
    *,
    data: pl.DataFrame | None = None,
    cross_report: Report | None = None,
) -> None:
    """生成发现和独立验证报告，或复用同批已有交叉报告。

    Parameters
    ----------
    output_dir : Path
        产物目录，不保存原始宽表；缺失目录会自动创建。
    data : pl.DataFrame | None
        共享合成宽表。为 None 时使用默认 18,000 行、固定 seed 的样本。
    cross_report : Report | None
        同一 data 的已计算交叉报告，包含固定发现分箱与 discovery 分区。
        为 None 时在本函数内计算；传入时复用其真实定义和格子证据。

    Returns
    -------
    None
        写入报告快照、适用导出、说明及提示材料。

    Examples
    --------
    >>> produce(Path("output/agent-rule-case"))  # doctest: +SKIP
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _sample() if data is None else data
    train = data.filter(pl.col("dataset") == "discovery")
    validation = data.filter(pl.col("dataset") == "validation")
    assert not set(train["sample_id"]).intersection(validation["sample_id"])
    context: dict[str, Any] = {
        "dataset_id": "synthetic-consumer-credit-20261001",
        "sample_unit": "模拟申请，每行独立 sample_id",
        "simulated": True,
        "question": QUESTION,
        "currency": "CNY",
        "labels": {
            "bad30": {"definition": "模拟30日违约", "performance_window": "30天"},
            "late60": {"definition": "模拟60日逾期", "performance_window": "60天"},
        },
        "splits": {
            "discovery": f"2026-01至03，{train.height}独立行；确定分段及候选",
            "validation": f"2026-04至06，{validation.height}独立行；一次独立门禁检验",
            "observation": f"2026-07至09，{data.height - train.height - validation.height}独立行；仅交叉观察，不参与选择",
        },
        "score_directions": DIRECTIONS,
        "amount_definition": "模拟申请金额，不代表损失或利润",
        "independence": "不同 sample_id 和时间，不从验证或观察重新挑选候选；并非真实业务效果",
    }
    if cross_report is None:
        discovery = cross_scores(
            train,
            score_x="main_score",
            score_y="aux_score",
            targets=["bad30", "late60"],
            score_directions=DIRECTIONS,
            n_bins=4,
            special_values={"aux_score": [-999]},
            feature_metadata=METADATA,
            business_context=context,
        )
        definitions = get_score_bin_definitions(discovery)
    else:
        definitions = get_score_bin_definitions(cross_report)
    # 同一份开发分段复用于全部时期；group 标识数据集，未聚合客群维度。
    cross = cross_report if cross_report is not None else cross_scores(
        data,
        score_x="main_score",
        score_y="aux_score",
        targets=["bad30", "late60"],
        score_directions=DIRECTIONS,
        bin_definitions=definitions,
        group_col="dataset",
        amount_col="amount",
        feature_metadata=METADATA,
        business_context={**context, "binning_origin": "only discovery Jan-Mar"},
    )
    definitions = get_score_bin_definitions(cross)
    discovery_period = cross.get_table(
        "cells", filters={"group": "discovery"}, columns=["period"], limit=1
    )["period"][0]
    cross.save(output_dir / "score-cross.marsreport", overwrite=True)
    cells = cross.get_table(
        "cells",
        filters={
            "group": "discovery",
            "period": discovery_period,
            "target": "bad30",
            "x_bin": {"op": "in", "value": ["b1", "b2"]},
            "y_bin": {"op": "in", "value": ["b0", "b1", "b2", "b3"]},
        },
        sort_by="delta_vs_row",
        descending=True,
    )
    # 两个高差异格子、一个反证及开发 Lift 介于候选/验证门槛的格子；选择仅用开发证据。
    picks = [*cells.head(2).to_dicts(), cells.tail(1).row(0, named=True)]
    moderate = cross.get_table(
        "cells",
        filters={
            "group": "discovery",
            "period": discovery_period,
            "target": "bad30",
            "x_bin": {"op": "in", "value": ["b0", "b1", "b2", "b3"]},
            "y_bin": {"op": "in", "value": ["b0", "b1", "b2", "b3"]},
            "lift_vs_overall": {"op": "ge", "value": 1.2},
        },
        sort_by="lift_vs_overall",
        limit=1,
    )
    if moderate.height:
        picks.append(moderate.row(0, named=True))
    seeds: list[MarsRule] = []
    origins: dict[str, Any] = {}
    for cell in picks:
        expression = f"({_condition('main_score', definitions['x'], int(cell['x_bin'][1:]))}) AND ({_condition('aux_score', definitions['y'], int(cell['y_bin'][1:]))})"
        rule = MarsRule(expression, source="cross_discovery")
        seeds.append(rule)
        origins[rule.rule_id] = {
            "kind": "development_cell",
            "reference": cross.query_page(
                "cells",
                filters={
                    "group": "discovery",
                    "period": cell["period"],
                    "target": "bad30",
                    "x_bin": cell["x_bin"],
                    "y_bin": cell["y_bin"],
                },
                limit=1,
            )["reference"],
        }
    rare = MarsRule("main_score > 900 AND aux_score > 900", source="manual_stress_check")
    seeds.append(rare)
    origins[rare.rule_id] = {
        "kind": "sample_insufficiency_check",
        "cross_reference": None,
        "limitation": "手工覆盖压力候选，无交叉格子对应，不制造证据链",
    }
    result = mine_rules(
        train,
        target="bad30",
        validation_df=validation,
        aux_targets=["late60"],
        features=list(METADATA),
        time_col="application_date",
        time_grain="month",
        amount_col="amount",
        customer_col="sample_id",
        seed_rules=seeds,
        generators=[],
        spec=MarsRuleMiningSpec.production(),
    )
    report = result.to_report(
        feature_metadata=METADATA,
        business_context={
            **context,
            "candidate_origins": origins,
            "slice_scope": "只计算 validation 时间切片；没有客群或规则 observation 评估",
            "analysis_scope": "高级分析未执行；不得根据空白推断没有交互或稳定性问题",
        },
    )
    report.save(output_dir / "rules.marsreport", overwrite=True)
    if result.rule_set.rules and result.rule_set.qualification in {
        "validated",
        "temporally_validated",
    }:
        result.rule_set.save_json(output_dir / "ruleset.json")
    report.write_html(output_dir / "rules.html")
    report.write_excel(output_dir / "rules.xlsx")
    (output_dir / "case-notes.json").write_text(
        encode(
            {
                "question": QUESTION,
                "simulated": True,
                "splits": context["splits"],
                "independent_rows": True,
                "raw_data_saved": False,
                "snapshots": ["score-cross.marsreport", "rules.marsreport"],
                "agent_execution": "deterministic acceptance; not autonomous model analysis",
            }
        ),
        encoding="utf-8",
    )
    prompt = Path(__file__).with_name("external_agent_rule_prompt.md").read_text(encoding="utf-8")
    (output_dir / "external-agent-task.md").write_text(
        prompt.replace("{{OUTPUT_DIR}}", "."), encoding="utf-8"
    )


def consume(output_dir: Path) -> dict[str, Any]:
    """新进程只加载两份快照与公开说明，按问题查询并形成可审阅的确定性验收输出。"""
    cross, rules = (
        load_report(output_dir / "score-cross.marsreport"),
        load_report(output_dir / "rules.marsreport"),
    )
    descriptions = {r.report_id: r.describe() for r in (cross, rules)}
    trace: list[dict[str, Any]] = []
    trace.append(
        _page(
            cross,
            "cells",
            {
                "filters": {"group": "discovery", "target": "bad30", "x_bin": "b1"},
                "columns": [
                    "group",
                    "period",
                    "target",
                    "x_bin",
                    "y_bin",
                    "sample_count",
                    "observed_sample_count",
                    "bad_sample_count",
                    "bad_rate",
                    "row_bad_rate",
                    "delta_vs_row",
                    "status",
                ],
                "sort_by": "y_bin",
                "limit": 10,
            },
            "主模型固定 b1 等级内辅助分风险和表现分母有何差异？",
        )
    )
    trace.append(
        _page(
            rules,
            "candidates",
            {
                "features": "main_score",
                "sources": "challenger",
                "columns": [
                    "rule_id",
                    "sources",
                    "status",
                    "rejection_stage",
                    "reason",
                    "candidate_filter_passed",
                    "validation_filter_passed",
                    "q_value",
                    "lift_ci_lower",
                ],
                "limit": 2,
            },
            "哪些候选入选或淘汰，实际门禁是什么？",
        )
    )
    while trace[-1]["next_offset"] is not None:
        trace.append(
            _page(
                rules,
                "candidates",
                {**trace[-1]["reference"]["query"], "offset": trace[-1]["next_offset"]},
                "继续候选审计下一页，包括覆盖不足或反证候选",
            )
        )
    selected = rules.get_table("rules", columns=["rule_id"], limit=1)
    if selected.height:
        rule_id = selected["rule_id"][0]
    else:
        candidate = rules.get_table("candidates", columns=["rule_id"], limit=1)
        rule_id = candidate["rule_id"][0] if candidate.height else None
    if rule_id is not None:
        trace.append(
            _page(
                rules,
                "evaluation",
                {
                    "filters": {
                        "rule_id": rule_id,
                        "dataset": "validation",
                        "target": "bad30",
                        "group": "hit",
                    },
                    "columns": [
                        "dataset",
                        "target",
                        "slice",
                        "rule_id",
                        "group",
                        "sample_count",
                        "event_count",
                        "coverage",
                        "event_rate",
                        "lift",
                        "lift_ci_lower",
                        "q_value",
                    ],
                    "limit": 2,
                },
                "所选规则在独立验证主目标中表现怎样？",
            )
        )
        trace.append(
            _page(
                rules,
                "slices",
                {
                    "filters": {
                        "rule_id": rule_id,
                        "dataset": "validation",
                        "target": "late60",
                        "group": "hit",
                        "slice": "2026-04",
                    },
                    "columns": [
                        "dataset",
                        "target",
                        "slice",
                        "rule_id",
                        "sample_count",
                        "event_count",
                        "event_rate",
                        "lift",
                    ],
                    "limit": 2,
                },
                "该规则在验证期4月辅助目标表现怎样？",
            )
        )
    trace.append(
        _page(
            cross,
            "cells",
            {
                "filters": {
                    "group": "observation",
                    "target": "bad30",
                    "x_bin": "b1",
                    "y_bin": "b3",
                },
                "columns": [
                    "group",
                    "period",
                    "target",
                    "x_bin",
                    "y_bin",
                    "sample_count",
                    "observed_sample_count",
                    "bad_rate",
                    "row_bad_rate",
                    "delta_vs_row",
                ],
                "limit": 1,
            },
            "后续观察同格子是否有一致的风险差异？不以此重新选择规则",
        )
    )
    # 每个真实查询都可重放；缺失维度根据目录明确回答，禁止重算或编造客群值。
    for entry in trace:
        reference = entry["reference"]
        report = cross if reference["report_id"] == cross.report_id else rules
        assert (
            table_rows(report.get_table(reference["table"], **reference["query"])) == entry["rows"]
        )
    unavailable = {
        "question": "该规则在 returning 客群及 observation 期的客户风险表现、交互和 bootstrap 怎样？",
        "answer": "当前报告无法回答",
        "reason": "规则只评估 discovery/train、validation 整体及时间切片；未计算客群、observation 规则评估和高级分析",
        "required": "追加指定独立样本的规则客群/observation 评估及显式高级分析，另存新报告，不在查询时重算",
        "reference": {
            "report_id": rules.report_id,
            "operation": "describe",
            "fields": ["parameters", "business_context.slice_scope"],
        },
    }
    context = rules.to_ai_context(
        queries={
            "summary": {"limit": 1},
            "evaluation": {
                "features": "aux_score",
                "sources": "champion",
                "filters": {"dataset": "validation", "target": "bad30", "group": "hit"},
                "columns": ["rule_id", "dataset", "target", "sample_count", "event_rate", "lift"],
                "limit": 3,
            },
        },
        max_chars=12000,
    )
    assert len(context) <= 12000
    json.loads(context)
    (output_dir / "agent-context.json").write_text(context, encoding="utf-8")
    (output_dir / "query-trace.json").write_text(
        encode(
            {
                "consumer": "deterministic scripted questions; no LLM",
                "catalogs": descriptions,
                "queries": trace,
                "unavailable": unavailable,
            }
        ),
        encoding="utf-8",
    )
    normal_rows = [
        r for r in trace[0]["rows"] if r["y_bin"].startswith("b") and r["bad_rate"] is not None
    ]
    findings: list[dict[str, Any]] = []
    if normal_rows:
        lower = min(normal_rows, key=lambda r: r["bad_rate"])
        upper = max(normal_rows, key=lambda r: r["bad_rate"])
        findings.append(
            {
                "claim": "同一发现期主模型 b1 内存在辅助等级风险差异；只是关联证据",
                "low_y_bin": lower["y_bin"],
                "low_event_rate": lower["bad_rate"],
                "low_observed_count": lower["observed_sample_count"],
                "high_y_bin": upper["y_bin"],
                "high_event_rate": upper["bad_rate"],
                "high_observed_count": upper["observed_sample_count"],
                "reference": trace[0]["reference"],
            }
        )
    findings.append(
        {
            "claim": "生产门禁的实际候选及入选数；无入选也是合法结果，不强制部署",
            "rows": rules.get_table("summary").to_dicts(),
            "reference": rules.query_page("summary", limit=1)["reference"],
        }
    )
    review = {
        "kind": "deterministic_evidence_review",
        "question": QUESTION,
        "simulated": True,
        "findings": findings,
        "discovery": trace[0],
        "selection": [item for item in trace if item["reference"]["table"] == "candidates"],
        "independent_validation": [
            item for item in trace if item["reference"]["table"] in {"evaluation", "slices"}
        ],
        "observation": trace[-1],
        "missing_information": unavailable,
        "limitations": [
            "模拟统计不代表真实效果；相关差异不是因果增量",
            "候选仅依据发现集选择；RuleSet JSON 和报告分别保存；不部署",
            "这是确定性公共接口验收输出，不声称模型自主完成分析",
        ],
    }
    (output_dir / "evidence-review.json").write_text(encode(review), encoding="utf-8")
    lines = [
        "# 确定性证据复核（模拟数据，无 LLM）",
        QUESTION,
        "发现证据见 query-trace.json 的第一条 cells 引用；风险比率仅以已表现人数为分母。",
        "候选审计：",
    ]
    lines.extend(encode(finding) for finding in findings)
    for entry in review["selection"]:
        lines.extend(
            f"- {row['rule_id']}: {row['status']}；阶段={row['rejection_stage']}；原因={row['reason']}；引用={encode(entry['reference'])}"
            for row in entry["rows"]
        )
    for entry in review["independent_validation"]:
        lines.extend(
            f"- 独立验证证据 {encode(row)}；引用={encode(entry['reference'])}"
            for row in entry["rows"]
        )
    lines.extend(
        [
            unavailable["answer"] + "：" + unavailable["reason"],
            "后续观察只用于交叉证据复核；未重新筛选候选。模拟发现与验证不得外推真实业务或自动部署。",
        ]
    )
    (output_dir / "review.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return review


def main() -> None:
    """产物写到浅目录；默认完成计算后启动只读快照的新 Python 消费进程。"""
    parser = argparse.ArgumentParser(description=QUESTION)
    parser.add_argument("--output-dir", type=Path, default=Path("output/agent-rule-case"))
    parser.add_argument(
        "--consume-only", action="store_true", help="只加载既有快照；不生成数据或重新计算"
    )
    args = parser.parse_args()
    if args.consume_only:
        consume(args.output_dir)
    else:
        produce(args.output_dir)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--consume-only",
                "--output-dir",
                str(args.output_dir.resolve()),
            ],
            check=True,
        )
    print(f"确定性验收产物：{args.output_dir.resolve()}；真实模型行为未由此脚本验证。")


if __name__ == "__main__":
    main()
