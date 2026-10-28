"""复用正式 Home Credit 输入准备，验证完整人口上的代表性分客群分析。"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from benchmark_core_capacity import Measurement
from benchmark_segmented_analysis import ROOT, THREAD_KEYS, _finite, _signature, _write


def _worker(args: argparse.Namespace) -> int:
    """保持正式 seed、五分区、原标签和全部样本，仅增加明确模拟客群。"""
    import polars as pl
    import psutil
    from polars.testing import assert_frame_equal

    import mars
    from mars.analysis import MarsBinEvaluator, profile_risk, profile_stats
    from mars.feature.selection import MarsStatsSelector

    sys.path.insert(0, str(ROOT / "docs" / "snippets"))
    from home_credit_data import DEFAULT_DISPLAY_FEATURES, prepare_home_credit_data

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    measurement = Measurement(output.with_suffix(".measurement.json"))
    evidence: dict[str, Any] = {
        "status": "running", "python": sys.version, "executable": sys.executable,
        "mars_module": mars.__file__, "mars_version": mars.__version__,
        "machine": {"logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                    "total_memory_bytes": psutil.virtual_memory().total},
        "versions": {name: importlib.metadata.version(name) for name in (
            "polars", "pandas", "numpy", "pyarrow", "scikit-learn", "optbinning", "psutil",
        )},
        "threads": {name: os.environ.get(name) for name in THREAD_KEYS},
        "polars_thread_pool_size": pl.thread_pool_size(), "binning_scope": args.binning_scope,
        "limitations": ["仅既有 DEFAULT_DISPLAY_FEATURES 代表候选，不是完整宽特征容量验收",
                        "demo_segment = SK_ID_CURR % 3，仅演示客群，不是真实业务客群",
                        "APPLY_DATE 继续为正式生成器的模拟日期，不是真实申请日期",
                        "未修改原始 CSV、seed、五分区或 TARGET 清理方式"],
    }
    _write(output, evidence)
    try:
        prepared = measurement.run("formal_input_preparation", lambda: prepare_home_credit_data(args.data_dir))
        candidates = [name for name in DEFAULT_DISPLAY_FEATURES if name in prepared.fcols]
        metadata = {feature: prepared.feature_metadata[feature] for feature in candidates}
        context = {**prepared.business_context,
                   "demonstration_segment": {"field": "demo_segment", "simulated": True,
                                             "definition": "SK_ID_CURR modulo 3"}}
        evidence["formal_input_summary"] = prepared.summary
        evidence["candidate_features"] = candidates
        evidence["full_candidate_count"] = len(prepared.fcols)
        frame = prepared.df.select([
            *candidates, "SK_ID_CURR", "TARGET", "dataset_flag", "APPLY_DATE", "AMT_CREDIT",
        ]).with_columns((pl.col("SK_ID_CURR") % 3).alias("demo_segment"))
        del prepared
        gc.collect()
        benchmark = frame.filter(pl.col("dataset_flag") == "0train")
        evidence["actual_population"] = {
            "rows": frame.height, "observed_labels": frame["TARGET"].count(),
            "features": len(candidates), "benchmark_rows": benchmark.height,
            "identity": _signature(frame), "benchmark_identity": _signature(benchmark),
            "group_scope_counts": frame.group_by(["demo_segment", "dataset_flag"]).agg(
                pl.len().alias("rows"), pl.col("TARGET").count().alias("observed_labels"),
                pl.col("TARGET").sum().alias("bad"),
            ).sort(["demo_segment", "dataset_flag"]).to_dicts(),
        }

        def compute() -> tuple[Any, Any, Any, list[Any]]:
            """完整公开调用及报告保存连续执行，所有产物保持共存。"""
            risk = profile_risk(
                frame, target="TARGET", features=candidates, segment_col="demo_segment",
                group_col="dataset_flag", time_col="APPLY_DATE", amount_col="AMT_CREDIT",
                benchmark_df=benchmark, binning_scope=args.binning_scope,
                feature_metadata=metadata, business_context=context, n_jobs=1, batch_size=16,
            )
            stats = profile_stats(
                frame, features=candidates, metrics=["missing", "mean", "psi", "schema", "unseen"],
                segment_col="demo_segment", group_col="dataset_flag", time_col="APPLY_DATE",
                benchmark_df=benchmark, binning_scope=args.binning_scope,
                feature_metadata=metadata, business_context=context,
            )
            selector = MarsStatsSelector(
                selection_scope="per_segment", binning_scope=args.binning_scope,
                batch_size=16, n_jobs=1,
            ).fit(frame, target="TARGET", features=candidates, segment_col="demo_segment",
                  group_col="dataset_flag", time_col="APPLY_DATE", benchmark_df=benchmark,
                  feature_metadata=metadata, business_context=context)
            reports: list[Any] = [risk.report, stats]
            for row in selector.scopes_.to_dicts():
                identity = row["scope_id"]
                child = selector.selectors_by_scope_.get(identity)
                if child is None:
                    continue
                selection: dict[str, Any] = {} if identity == "all" else {
                    "segments": [int(row["segment_value"])], "include_all": False,
                }
                if child.selected_features_:
                    reports.append(selector.get_binning_report(frame, benchmark_df=benchmark, **selection))
                if child._corr_report is not None:
                    reports.append(selector.get_correlation_report(**selection))
            directory = output.with_suffix("")
            directory.mkdir(parents=True, exist_ok=True)
            for index, report in enumerate(reports):
                report.save(directory / f"report-{index}.marsreport", overwrite=True)
            risk.save_binning_rules(directory / "risk-rules.json")
            return risk, stats, selector, reports

        risk, stats, selector, retained_reports = measurement.run("public_compute", compute)

        def verify() -> dict[str, Any]:
            """固定整体规则核对原入口，并逐 scope 手工检查完整与表现人口。"""
            risk_reference = MarsBinEvaluator().evaluate(
                frame, target="TARGET", features=candidates, group_col="dataset_flag",
                time_col="APPLY_DATE", amount_col="AMT_CREDIT", benchmark_df=benchmark,
                binner=risk.binner, feature_metadata=metadata, business_context=context, batch_size=16,
            ).report
            actual = risk.report.detail_table.filter(pl.col("scope_id") == "all").drop("scope_id")
            assert_frame_equal(actual.sort(["feature", "mars_group", "bin_index"]),
                               risk_reference.detail_table.sort(["feature", "mars_group", "bin_index"]))
            stats_reference = profile_stats(
                frame, features=candidates, metrics=["missing", "mean", "psi", "schema", "unseen"],
                group_col="dataset_flag", time_col="APPLY_DATE", benchmark_df=benchmark,
                feature_metadata=metadata, business_context=context,
            )
            for metric in stats.stats_tables:
                actual_stats = stats.stats_tables[metric].filter(pl.col("scope_id") == "all").drop("scope_id")
                assert_frame_equal(actual_stats, stats_reference.stats_tables[metric])
            checks: list[dict[str, Any]] = []
            for row in risk.scopes.to_dicts():
                identity = row["scope_id"]
                population = frame if identity == "all" else frame.filter(
                    pl.col("demo_segment") == int(row["segment_value"])
                )
                detail = risk.report.detail_table.filter(
                    (pl.col("scope_id") == identity) & (pl.col("feature") == candidates[0])
                    & (pl.col("mars_group") == "Total") & (pl.col("bin_index") != 9999)
                )
                assert detail["count"].sum() == population.height
                assert detail["observed_count"].sum() == population["TARGET"].count()
                assert detail["bad"].sum() == population["TARGET"].sum()
                checks.append({"scope_id": identity, "rows": population.height,
                               "observed_labels": population["TARGET"].count(), "status": "passed"})
            return {"all_fixed_rule_detail": "passed", "all_stats": "passed", "manual_scope_population": checks}

        evidence["verification"] = measurement.run("independent_reference", verify)
        evidence["selector_scopes"] = {
            identity: {"selected_features": child.selected_features_, "funnel": child._funnel_stats,
                       "decisions": _signature(child.get_report())}
            for identity, child in selector.selectors_by_scope_.items()
        }
        evidence.update(status="passed", retained_report_count=len(retained_reports),
                        risk_rule_scope_count=len(risk.binners_by_scope), measurement=measurement.result)
    except Exception as exc:
        evidence.update(status="failed", exception=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(), measurement=measurement.result)
    _write(output, _finite(evidence))
    return 0 if evidence["status"] == "passed" else 1


def _main() -> int:
    """每种模式用独立进程验收，避免前一轮对象或线程池污染资源记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="examples/home-credit-2018/data/raw_data")
    parser.add_argument("--binning-scopes", nargs="+", choices=["shared", "per_segment"], default=["shared"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default=".artifacts/segmented/home-credit.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--binning-scope", default="shared", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.worker:
        return _worker(args)
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({name: str(args.threads) for name in THREAD_KEYS})
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONIOENCODING"] = "utf-8"
    cases: list[dict[str, Any]] = []
    for mode in args.binning_scopes:
        output = destination.with_name(f"{destination.stem}-{mode}.json")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--data-dir", args.data_dir,
                   "--binning-scope", mode, "--output", str(output)]
        with output.with_suffix(".log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
        case = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"status": "failed"}
        cases.append({"mode": mode, "exit_code": completed.returncode, "evidence": str(output), **case})
        print(f"Home Credit {mode}: {case['status']}", flush=True)
    passed = all(case["status"] == "passed" for case in cases)
    _write(destination, {"status": "passed" if passed else "failed", "cases": cases,
                         "full_feature_capacity_validated": False})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
