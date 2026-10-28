"""分客群的确定性示例与新进程证据验收，不需要模型服务。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from mars.analysis import MarsRiskProfile, profile_risk, profile_stats
from mars.feature import MarsStatsSelector
from mars.reporting import load_report


def make_data() -> pl.DataFrame:
    """构造方向相反的三个模拟客群；日期与客群均为演示定义。"""
    rows = []
    for segment_index in range(3):
        for group_index, group in enumerate(["Train", "Val", "Test"]):
            for index in range(40):
                x = index % 10
                rows.append({
                    "segment": f"seg{segment_index + 1}", "split": group,
                    "date": date(2026, group_index + 1, 1) + timedelta(days=index % 20),
                    "x": float(x), "category": ["A", "B", "Missing"][index % 3],
                    "y": int(x >= 5) if segment_index != 1 else int(x < 5),
                })
    return pl.DataFrame(rows)


def restore_only(output: Path) -> None:
    """只读快照执行查询、预算、静态导出和原生图；不读取原样本。"""
    from mars.agent import MarsAgentSession, MarsAgentToolCall
    from mars.agent._tools import _MarsTools

    report = load_report(output / "risk.marsreport")
    group_col = report.describe()["parameters"]["group_col"]
    options = {"segments": ["seg2"], "include_all": False, "features": ["x"],
               "filters": {group_col: "Val"}, "limit": 2}
    page = report.query_page("detail", **options)
    assert page["total_rows"] > 2 and page["next_offset"] == 2
    assert page["data"].equals(report.get_table("detail", **page["reference"]["query"]))
    next_page = report.query_page("detail", **{**options, "offset": page["next_offset"]})
    assert next_page["returned_rows"] > 0
    context = report.to_ai_context(queries={"detail": options}, max_chars=24000)
    assert len(context) <= 24000
    session = MarsAgentSession()
    handle = session.register_report(report)
    result = _MarsTools(session, max_result_chars=24000).execute(MarsAgentToolCall(
        "seg2-evidence", "get_report_context", {"report_id": handle, "queries": {"detail": options}},
    ))
    assert result.success, result.error_message
    evidence = result.data["evidence"][0]
    assert evidence["rows"] and all(row[group_col] == "Val" for row in evidence["rows"])
    assert len({row["scope_id"] for row in evidence["rows"]}) == 1
    report.write_html(str(output / "restored.html"), segments=["seg2"], include_all=False)
    report.write_excel(str(output / "snapshot.xlsx"))
    figures = report.build_risk_trend_figures("x", segments=["seg2"], include_all=False, max_plots=1)
    assert len(figures) == 1
    figures[0].savefig(output / "restored.png")
    import matplotlib.pyplot as plt

    plt.close("all")
    (output / "context.json").write_text(context, encoding="utf-8")
    (output / "reference.json").write_text(json.dumps(page["reference"], ensure_ascii=False), encoding="utf-8")
    (output / "agent-evidence.json").write_text(json.dumps(result.data, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"fresh_process": True, "total_rows": page["total_rows"],
                      "page_rows": page["returned_rows"], "next_offset": page["next_offset"],
                      "context_chars": len(context), "agent_success": result.success}))


def run(output: Path) -> None:
    """实施两种画像、客群名单、规则复用与报告独立恢复的完整示例。"""
    output.mkdir(parents=True, exist_ok=True)
    frame = make_data()
    benchmark = frame.filter(pl.col("split") == "Train")
    parameters = {"target": "y", "features": ["x", "category"], "group_col": "split",
                  "time_col": "date", "benchmark_df": benchmark, "method": "quantile", "n_bins": 3}
    overall = profile_risk(frame, **parameters)
    shared = profile_risk(frame, segment_col="segment", binning_scope="shared", **parameters)
    independent = profile_risk(frame, segment_col="segment", binning_scope="per_segment", **parameters)
    assert overall.report.summary_table.height == 2
    assert shared.report.summary_table.height == 8
    independent.save_binning_rules(output / "rules.json")
    fixed = MarsRiskProfile.load_binning_rules(output / "rules.json")
    assert fixed["all"].transform(frame, features=["x"]).equals(independent.binner.transform(frame, features=["x"]))
    missing = profile_risk(frame, segment_col="segment", binning_scope="shared", **{
        **parameters, "benchmark_df": benchmark.filter(pl.col("segment") != "seg2"),
    })
    status = missing.report.get_table("calculation_status", segments=["seg2"], include_all=False,
                                      filters={"metric": "psi", "status": "not_computed"})
    assert status.height > 0
    assert missing.report.get_table("summary", segments=["seg2"], include_all=False)["psi_max"].null_count() == 2
    stats = profile_stats(frame, segment_col="segment", features=["x", "category"],
                          group_col="split", benchmark_df=benchmark, metrics=["missing", "mean", "psi"])
    stats.write_html(str(output / "profile.html"), segments=["seg1", "seg2", "seg3"])
    selector = MarsStatsSelector(selection_scope="per_segment", binning_scope="shared",
                                 binning_params={"prebinning_method": "quantile", "n_bins": 3},
                                 batch_size=1, n_jobs=1)
    selector.fit(frame, target="y", features=["x", "category"], segment_col="segment",
                 group_col="split", benchmark_df=benchmark)
    print("selected_features_by_segment_", selector.selected_features_by_segment_)
    report = shared.report
    report.write_excel(str(output / "seg2.xlsx"), segments=["seg2"], include_all=False)
    report.write_html(str(output / "risk.html"), segments=["seg1", "seg2", "seg3"], max_plots=4)
    report.save_risk_trend_images(output, "x", segments=["seg2"], include_all=False,
                                  image_format="png", max_plots=1)
    report.save(output / "risk.marsreport", overwrite=True)
    result = subprocess.run([sys.executable, __file__, "--restore-only", "--output-dir", str(output)],
                            capture_output=True, text=True, check=True)
    print(result.stdout)


def main() -> None:
    """支持保留验收产物或在临时目录执行。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args()
    if args.restore_only:
        restore_only(args.output_dir)
    elif args.output_dir:
        run(args.output_dir)
    else:
        with TemporaryDirectory() as temporary:
            run(Path(temporary))


if __name__ == "__main__":
    main()
