"""统一验收的固定业务 fixture 和真实入口；导入本模块不会导入计算依赖。"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

WORKLOAD_CONTRACT = "core-capacity-workloads-v1"


def workload(case: str, scale: str) -> dict[str, Any]:
    """按档位声明实际规模，扩行和扩列诊断显式选择。"""
    index = ("smoke", "standard", "large").index(scale)
    rows, features = ((800, 8), (50000, 200), (200000, 1000))[index]
    dimensions: dict[str, Any] = {"rows": rows, "features": features, "batch_size": 50}
    if case in ("rule_report", "rule_bridge"):
        dimensions = {
            "audit_rows": (1000, 10000, 50000)[index],
            "features": 32,
            "fixture": "analytical synthetic candidate audit; not mining",
        }
    elif case.startswith("rule_"):
        dimensions = {
            "rows": (480, 2000, 4000)[index],
            "features": 3,
            "max_candidates": 50,
            "fixture": "real independent train/validation mining",
        }
    elif case.startswith("correlation"):
        p = (40, 1000 if "1000" in case else 500, 3000)[index]
        dimensions = {
            "rows": 1500,
            "features": p,
            "pairs": p * (p - 1) // 2,
            "name_style": "long" if "long" in case else "short",
            "fixture": "NumPy signed Pearson matrix injection; not selector.fit",
        }
    elif case.startswith("score_cross"):
        width = 500 if "500" in case else 50 if "50" in case else 0
        dimensions = {
            "rows": 2000 if index == 0 else 1000000,
            "features": width + 2,
            "unused_columns": width,
            "fixture": "real cross_scores",
        }
    elif case == "linear_selection":
        rows, features = ((400, 8), (3000, 40), (5000, 80))[index]
        dimensions.update(rows=rows, features=features)
    elif case == "optimal_binning":
        rows, features = ((1000, 4), (5000, 10), (10000, 20))[index]
        dimensions.update(rows=rows, features=features, algorithm="optimal")
    elif case.endswith("_columns") and index:
        dimensions.update(rows=50000, features=3000)
    elif case.endswith("_rows") and index:
        dimensions.update(rows=1000000, features=100)
    if "batch100" in case:
        dimensions["batch_size"] = 100
    return dimensions


def _linear_diagnostics(selector: Any) -> dict[str, Any]:
    """记录生产入口实际导入与诊断执行状态；不补跑诊断改变测量工作量。"""
    from benchmark_core_capacity import _dependency_state

    available = selector._linear_diagnostics_available
    dependency = _dependency_state("statsmodels", "statsmodels", unavailable=not available)
    has_features = bool(selector.selected_features_)
    diagnostics: dict[str, Any] = {"dependency": dependency}
    for name, table in (("vif", selector.vif_table_), ("logit", selector.coef_table_)):
        rows = len(table)
        if rows:
            status, reason = "executed", None
        elif not available:
            status, reason = "skipped", "statsmodels unavailable"
        elif name == "logit" and has_features:
            status, reason = "failed", "fit produced no coefficients"
        else:
            status, reason = "skipped", "no eligible features"
        diagnostics[name] = {"status": status, "rows": rows, "reason": reason}
    diagnostics["stepwise"] = {
        "status": "executed" if selector.enable_stepwise else "skipped",
        "reason": "enabled" if selector.enable_stepwise else "explicitly disabled",
    }
    return diagnostics


def _metadata(features: list[str]) -> dict[str, Any]:
    """重复中文显示名与业务来源不替代英文身份。"""
    return {
        f: {
            "display_name": "重复显示名" if i < 2 else f"指标{i}",
            "data_source": ("application", "bank", "bureau", "behavior")[i % 4],
            "description": "固定种子模拟风控数值，单位及表现范围由上下文明确。" * 2,
            "unit": "dimensionless",
        }
        for i, f in enumerate(features)
    }


def _context_metadata() -> dict[str, Any]:
    """保留必要标签和来源说明，业务说明不伪装真实客户数据。"""
    return {
        "dataset_id": "synthetic-capacity",
        "population": "synthetic credit applicants",
        "currency": "CNY",
        "labels": {
            "bad": {"definition": "synthetic primary default", "performance_window": "90 days"},
            "late": {
                "definition": "synthetic auxiliary delinquency",
                "performance_window": "30 days",
            },
        },
    }


def _check_statistics(case: str, data: Any, report: Any, features: list[str]) -> dict[str, Any]:
    """用少量原始列的 NumPy 参考验证数值，宽表绝不完整转 Python 行。"""
    import numpy as np
    import polars as pl
    from numpy.testing import assert_allclose

    if case.startswith("profile"):
        references = []
        for feature in features[:3]:
            values = data[feature].to_numpy()
            valid = values[np.isfinite(values) & (values != -999)]
            actual = report.get_table("stats.mean", features=feature)["total"].to_numpy()[0]
            expected_mean = float(valid.mean())
            assert_allclose(actual, expected_mean, rtol=1e-6, atol=1e-6)
            references.append({"feature": feature, "mean": expected_mean})
        return {
            "oracle": "NumPy finite values excluding configured -999",
            "references": references,
            "rtol": 1e-6,
            "atol": 1e-6,
        }
    if case.startswith("score_cross") or case.startswith("binning") or case == "optimal_binning":
        weights, labels, amount = (data[name].to_numpy() for name in ("weight", "bad", "amount"))
        observed = np.isfinite(labels)
        bad = observed & (labels == 1)
        expected = {
            "count": float(weights.sum()),
            "observed_count": float(weights[observed].sum()),
            "bad": float(weights[bad].sum()),
            "tot_amt": float(amount.sum()),
            "bad_amt": float(amount[bad].sum()),
        }
        if case.startswith("score_cross"):
            rows = report.get_table("overall", filters={"target": "bad"})
            rename = {
                "count": "weight_sum",
                "observed_count": "observed_weight_sum",
                "bad": "bad_weight_sum",
            }
        else:
            rows = report.get_table(
                "detail", features=features[0], filters={"mars_group": "Total", "bin_index": 9999}
            )
            rename = {}
        if not isinstance(rows, pl.DataFrame):
            rows = pl.from_pandas(rows)
        for key, value in expected.items():
            assert_allclose(rows[rename.get(key, key)].sum(), value, rtol=1e-6, atol=1e-6)
        return {
            "oracle": "independent NumPy observed labels, weights and amounts",
            "references": expected,
            "rtol": 1e-6,
            "atol": 1e-6,
        }
    return {"oracle": "existing selector numeric, ordering and matrix-reuse regression suite"}


def _data(
    rows: int, p: int, seed: int, backend: str, constants: bool = True
) -> tuple[Any, list[str]]:
    """有相关结构的混合 dtype，少量常量/低基数，固定缺失、零与特殊码。"""
    import numpy as np
    import pandas as pd
    import polars as pl

    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(rows, 8)).astype("float32")
    features = [f"risk_feature_{i:04d}" for i in range(p)]
    columns: dict[str, Any] = {}
    for i, f in enumerate(features):
        feature_rng = np.random.default_rng(seed + i + 100)
        values = (0.8 * latent[:, i % 8] + feature_rng.normal(scale=0.6, size=rows)).astype(
            "float64" if i % 5 == 0 else "float32"
        )
        if constants and i == p - 1:
            values[:] = 2.0
        elif constants and i == p - 2 and p > 4:
            values[:] = np.digitize(values, [-1, 0, 1])
        values[::101] = np.nan
        values[17::211] = -999
        values[13::157] = 0
        columns[f] = values
    rng = np.random.default_rng(seed + 100000)
    bad = (latent[:, 0] + 0.3 * latent[:, 1] + rng.normal(size=rows) * 0.8 > 0.8).astype(float)
    late = (latent[:, 1] + rng.normal(size=rows) * 0.7 > 0.7).astype(float)
    bad[::19] = np.nan
    late[::7] = np.nan
    columns.update(
        bad=bad,
        late=late,
        group=np.array(["TEST", "OOT", "NEW", "REPEAT"])[np.arange(rows) % 4],
        weight=rng.uniform(0.5, 2, rows),
        amount=rng.uniform(100, 5000, rows),
        customer=np.arange(rows) // 2,
    )
    data = (
        pd.DataFrame(columns)
        if backend == "pandas"
        else pl.DataFrame(columns).with_columns(pl.col("bad", "late").fill_nan(None))
    )
    return data, features


def _rule_fixture(count: int) -> tuple[Any, list[str], dict[str, Any]]:
    """不同 DSL、统计和轮次的合成审计，指标粒度与真实结果一致。"""
    import polars as pl

    from mars.rule import MarsRule, MarsRuleMiningResult, MarsRuleMiningSpec, MarsRuleSet
    from mars.rule.evaluator import MarsRuleEvaluation

    features = [
        f"credit_application_behavior_monthly_rolling_risk_measurement_{i:03d}" for i in range(32)
    ]
    unique = count * 2 // 3
    rules = [
        MarsRule(
            f"{features[i % 32]} >= {i / unique:.8f}"
            + (f" AND {features[(i + 1) % 32]} >= 0" if i % 3 == 0 else "")
        )
        for i in range(unique)
    ]
    audit: list[dict[str, Any]] = []
    for i in range(count):
        j = i % unique
        round_ = 1 + i // unique
        selected = j < 5 and round_ == 1
        audit.append(
            {
                "rule_id": rules[j].rule_id,
                "expression": rules[j].expression,
                "sources": ["seed" if j % 2 else "combination"],
                "status": "selected" if selected else "rejected" if j % 4 == 0 else "candidate",
                "generation_round": round_,
                "selection_round": round_ if selected else None,
                "rejection_stage": "validation_filter" if j % 4 == 0 and not selected else None,
                "reason": "synthetic validation gate" if j % 4 == 0 and not selected else "",
                "budget_position": j + 1,
            }
        )
    overall: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    for dataset, n in (("train", 10000), ("validation", 8000)):
        for target, fraction in (("bad", 1.0), ("late", 0.8)):
            total = int(n * fraction)
            base_bad = total // 5
            for i, rule in enumerate(rules):
                hit = max(20, int(total * (0.02 + 0.38 * (unique - i) / unique)))
                events = min(base_bad, int(hit * (0.22 + 0.2 * (i % 17) / 17)))
                for group, samples, bads in (
                    ("hit", hit, events),
                    ("miss", total - hit, base_bad - events),
                    ("total", total, base_bad),
                ):
                    rate = bads / samples
                    row = {
                        "rule_id": rule.rule_id,
                        "dataset": dataset,
                        "target": target,
                        "slice": "__overall__",
                        "group": group,
                        "sample_count": samples,
                        "event_count": bads,
                        "event_rate": rate,
                        "coverage": samples / total,
                        "lift": rate / (base_bad / total),
                        "amount_total": samples * 1000.0,
                        "event_amount": bads * 1000.0,
                        "customer_count": samples // 2,
                        "fixture_statistics": "synthetic analytical counts, not evaluated raw samples",
                    }
                    overall.append(row)
                    for scope in ("NEW", "REPEAT"):
                        ns = samples // 2 if scope == "NEW" else samples - samples // 2
                        nb = bads // 2 if scope == "NEW" else bads - bads // 2
                        slices.append(
                            {
                                **row,
                                "slice": scope,
                                "sample_count": ns,
                                "event_count": nb,
                                "event_rate": nb / ns,
                                "coverage": ns / (total // 2),
                                "lift": (nb / ns) / (base_bad / total),
                                "amount_total": ns * 1000.0,
                                "event_amount": nb * 1000.0,
                                "customer_count": ns // 2,
                            }
                        )
    candidate = pl.DataFrame(audit)
    evaluation = MarsRuleEvaluation(pl.DataFrame(overall), pl.DataFrame(slices))
    result = MarsRuleMiningResult(
        "success",
        MarsRuleSet(rules[:5]),
        candidate,
        evaluation,
        MarsRuleMiningSpec(selection_strategy="cascade"),
        {
            "target": "bad",
            "aux_targets": ["late"],
            "features": features,
            "validation_status": "independent",
            "group_col": "segment",
            "fixture": "synthetic analytical audit; repeated IDs only in distinct rounds",
        },
    )
    return (
        result,
        features,
        {
            "audit_rows": count,
            "distinct_rule_ids": unique,
            "slice_rows": len(slices),
            "evaluation_rows": len(overall),
            "expected_bridge_rows": sum(2 if i % 3 == 0 else 1 for i in range(unique)),
        },
    )


def _real_rules(
    args: argparse.Namespace, meter: Any, dimensions: dict[str, Any]
) -> tuple[Any, list[str]]:
    """真实 mine_rules 与三种合法零入选状态；整数样本可人工复核。"""
    import numpy as np
    import pandas as pd
    import polars as pl

    from mars.rule import MarsRuleMiningSpec, MarsRuleSet, mine_rules

    def sample(start: int) -> Any:
        values = np.arange(start, start + dimensions["rows"])
        frame = pl.DataFrame(
            {
                "income": values % 120,
                "salary": values % 2,
                "unused": values,
                "bad": (values % 120 >= 85).astype(int),
                "late": [None if v % 7 == 0 else int(v % 120 >= 80) for v in values],
                "segment": ["NEW" if v % 2 else "REPEAT" for v in values],
                "amount": (values % 100 + 1) * 100.0,
                "customer": values // 2,
            }
        )
        return frame.to_pandas() if args.backend == "pandas" else frame

    train, validation = meter.run("input_prepare", lambda: (sample(0), sample(10000)))
    features = ["income", "salary", "unused"]
    seeds = ["income >= 90 AND salary >= 0", "income >= 100", "income < 10"]
    result = meter.run(
        "public_compute",
        lambda: mine_rules(
            train,
            target="bad",
            validation_df=validation,
            aux_targets=["late"],
            features=features,
            group_col="segment",
            amount_col="amount",
            customer_col="customer",
            seed_rules=seeds,
            generators=[],
            spec=MarsRuleMiningSpec(max_candidates=50, iou_threshold=1.0),
        ),
    )
    if args.case == "rule_states":
        states = []
        for state in ("no_candidates", "all_rejected", "candidate_unselected"):
            if isinstance(validation, pd.DataFrame):
                inverted = validation.assign(bad=1 - validation["bad"])
            else:
                inverted = validation.with_columns((1 - pl.col("bad")).alias("bad"))
            current = mine_rules(
                train,
                target="bad",
                features=features,
                validation_df=inverted if state != "no_candidates" else validation,
                seed_rules=[] if state == "no_candidates" else ["income >= 100"],
                generators=[],
                spec=MarsRuleMiningSpec(max_candidates=50),
            )
            if state == "candidate_unselected":
                current = replace(
                    result,
                    status="no_rules",
                    rule_set=MarsRuleSet([]),
                    candidate_table=result.candidate_table.with_columns(
                        pl.lit("candidate").alias("status")
                    ),
                )
            report = current.to_report()
            assert report.get_table("rules").is_empty()
            states.append(
                {
                    "state": state,
                    "status": current.status,
                    "candidate_rows": report.get_table("candidates").height,
                    "fixture": "empty final selection on real audit"
                    if state == "candidate_unselected"
                    else "real mining",
                }
            )
        meter.result["empty_states"] = states
    row = result.evaluation.overall_table.filter(
        (pl.col("dataset") == "train") & (pl.col("target") == "bad") & (pl.col("group") == "hit")
    )
    assert row.height > 0
    meter.result["correctness"] = {"real_mining": True, "evaluation_rows": row.height}
    return result, features


def run_case(args: argparse.Namespace, meter: Any) -> None:
    """先完整公共调用，再只消费已有报告；临时宽表和快照自动清理。"""
    import numpy as np
    import pandas as pd
    import polars as pl
    from benchmark_core_capacity import (
        _execution_parameters,
        _launch,
        _signature,
        _write,
    )

    from mars.analysis import (
        MarsBinEvaluator,
        MarsDataProfiler,
        cross_scores,
        evaluate_score_policy,
    )
    from mars.feature import MarsLinearSelector, MarsStatsSelector
    from mars.reporting import get_correlation_matrix

    meter.result["import_rss_bytes"] = __import__("psutil").Process().memory_info().rss
    dimensions = workload(args.case, args.scale)
    meter.result["workload"] = {
        **dimensions,
        "seed": args.seed,
        "input_backend": args.backend,
        "distribution": "8 latent factors + Gaussian noise; mixed float32/64; constant/low-cardinality",
        "missing": "NaN 1/101, special -999 1/211, legal zero 1/157; bad unobserved 1/19, late 1/7",
    }
    case = args.case
    estimator_parameters: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    if case.startswith("correlation"):
        meter.result["workload"].update(
            distribution="float64 Gaussian; feature1 = -feature0 + noise; last feature constant",
            missing="one NaN in penultimate feature; NumPy propagation; no labels or special codes",
        )
    elif case in ("rule_report", "rule_bridge"):
        meter.result["workload"].update(
            input_backend="synthetic_polars_statistics",
            distribution="varied valid audit statistics; repeated IDs only across actual rounds",
            missing="auxiliary label observed on 80% of samples; no raw-table missing simulation",
        )
    elif case.startswith("rule_"):
        meter.result["workload"].update(
            distribution="small real mining fixture with independently sampled train/validation",
            missing="see recorded mine_rules parameters; not the wide-analysis fixture",
        )
    with TemporaryDirectory(prefix="mars-report-", dir=args.output.parent) as temporary:
        folder = Path(temporary)
        if case in ("rule_report", "rule_bridge"):
            result, features, counts = meter.run(
                "input_prepare", lambda: _rule_fixture(dimensions["audit_rows"])
            )
            meter.result["fixture_counts"] = counts

            def construct_rule_report() -> Any:
                """普通入口与桥接子阶段诊断共享完整 to_report 调用。"""
                return result.to_report(
                    feature_metadata=_metadata(features), business_context=_context_metadata()
                )

            if case == "rule_bridge":
                import mars.rule.report as rule_report_module

                original_snapshot = rule_report_module._rule_snapshot

                def snapshot(*values: Any, **kwargs: Any) -> Any:
                    """包含桥接和语义目录装配；嵌套计时不相加，不重复计算。"""
                    return meter.run(
                        "bridge_and_semantics_construct",
                        lambda: original_snapshot(*values, **kwargs),
                    )

                with patch.object(rule_report_module, "_rule_snapshot", snapshot):
                    report = meter.run("report_construct", construct_rule_report)
            else:
                report = meter.run("report_construct", construct_rule_report)
            assert report.get_table("rule_features").height == counts["expected_bridge_rows"]
            from polars.testing import assert_frame_equal

            reference = report.get_table("evaluation", features=features[0]).filter(
                pl.col("rule_id").is_in(
                    report.get_table("rule_features", features=features[1])["rule_id"].implode()
                )
            )
            actual = report.get_table("evaluation", features=features[0], sources="bank")
            assert_frame_equal(reference, actual)
            assert report.get_table("candidates").height == dimensions["audit_rows"]
            query = {
                "features": features[0],
                "sources": "bank",
                "filters": {"dataset": "validation"},
                "columns": ["rule_id", "target", "sample_count", "event_rate", "lift"],
                "sort_by": "lift",
                "descending": True,
                "offset": 20,
                "limit": 20,
            }
            table = "evaluation"
        elif case.startswith("rule_"):
            result, features = _real_rules(args, meter, dimensions)
            report = meter.run(
                "report_construct",
                lambda: result.to_report(
                    feature_metadata=_metadata(features), business_context=_context_metadata()
                ),
            )
            table = "slices"
            query = {
                "filters": {"dataset": "validation"},
                "columns": ["rule_id", "target", "sample_count"],
                "sort_by": "sample_count",
                "descending": True,
                "offset": 20,
                "limit": 20,
            }
        elif case.startswith("correlation"):
            features = [
                f"f{i:04d}"
                if dimensions["name_style"] == "short"
                else f"credit_bureau_application_behavior_monthly_rolling_risk_measurement_{i:04d}"
                for i in range(dimensions["features"])
            ]
            rng = np.random.default_rng(args.seed)

            def correlation_data() -> Any:
                values = rng.normal(size=(1500, len(features)))
                values[:, 1] = -values[:, 0] + rng.normal(scale=0.05, size=1500)
                values[:, -1] = 1.0
                values[0, -2] = np.nan
                return (
                    pd.DataFrame(values, columns=features)
                    if args.backend == "pandas"
                    else pl.DataFrame(values, schema=features)
                )

            data = meter.run("input_prepare", correlation_data)
            meter.result["input_bytes"] = (
                int(data.memory_usage(deep=True).sum())
                if isinstance(data, pd.DataFrame)
                else data.estimated_size()
            )
            matrix = meter.run("matrix_compute", lambda: np.corrcoef(data.to_numpy(), rowvar=False))
            assert matrix[0, 1] < -0.99 and np.isnan(matrix[-1, -1])
            selector = MarsLinearSelector()
            selector._reset_correlation()
            selector._corr_candidates = selector._corr_input_features = features
            selector._corr_matrix = matrix
            selector._corr_parameters = {
                "representation": "raw",
                "method": "pearson",
                "input_row_count": 1500,
                "correlation_row_count": 1500,
                "missing_policy": "NumPy diagnostic propagation, not selector complete-case",
                "fixture": dimensions["fixture"],
            }
            selector._corr_status = "computed"
            meter.run("report_construct", selector._finish_correlation)
            selector._is_fitted = True
            report = selector.get_correlation_report()
            np.testing.assert_allclose(
                get_correlation_matrix(report, features[:4]), matrix[:4, :4], rtol=1e-12, atol=1e-12
            )
            assert report.get_table("pairs").height == dimensions["pairs"]
            assert (
                report.get_table("features", features=features[-1])["diagonal_status"][0]
                == "unavailable"
            )
            del data, matrix, selector
            table = "pairs"
            query = {
                "features": features[0],
                "columns": ["feature_a", "feature_b", "correlation"],
                "sort_by": "abs_correlation",
                "descending": True,
                "offset": 20,
                "limit": 20,
            }
        else:
            data, features = meter.run(
                "input_prepare",
                lambda: _data(
                    dimensions["rows"],
                    dimensions["features"],
                    args.seed,
                    args.backend,
                    constants=not case.startswith("score_cross"),
                ),
            )
            meter.result["input_bytes"] = (
                int(data.memory_usage(deep=True).sum())
                if isinstance(data, pd.DataFrame)
                else data.estimated_size()
            )
            meter.result["input_ready_rss_bytes"] = __import__("psutil").Process().memory_info().rss
            if case.startswith("score_cross"):
                necessary = [*features[:2], "bad", "late", "group", "weight", "amount"]
                data = (
                    data.drop(columns=["customer"])
                    if isinstance(data, pd.DataFrame)
                    else data.drop("customer")
                )
                meter.result["input_columns"] = len(data.columns)
                meter.result["input_bytes"] = (
                    int(data.memory_usage(deep=True).sum())
                    if isinstance(data, pd.DataFrame)
                    else data.estimated_size()
                )
                # 仅增加真实 float32/64 数值；这些无关列计入输入准备和全进程峰值。
                calls: list[list[str]] = []
                fits: list[str] = []
                aggregations: list[dict[str, Any]] = []
                import mars.analysis.score_cross as cross_module

                original = pl.from_pandas
                original_fit = cross_module._fit_axis
                original_group_by = pl.DataFrame.group_by

                def capture(frame: Any, **kwargs: Any) -> Any:
                    calls.append(list(frame.columns))
                    return original(frame, **kwargs)

                def fit(*values: Any, **kwargs: Any) -> Any:
                    fits.append("axis")
                    return original_fit(*values, **kwargs)

                def group_by(frame: Any, *keys: Any, **kwargs: Any) -> Any:
                    """观察同一个原始样本聚合是否同时包含两个标签，不改变表达式。"""
                    expected = ["__cross_group", "__cross_period", "x_bin", "y_bin"]
                    if (
                        keys
                        and isinstance(keys[0], list)
                        and all(isinstance(key, str) for key in keys[0])
                        and keys[0] == expected
                    ):
                        aggregations.append(
                            {
                                "rows": frame.height,
                                "keys": expected,
                                "targets": [
                                    name for name in ("bad", "late") if name in frame.columns
                                ],
                            }
                        )
                    return original_group_by(frame, *keys, **kwargs)

                # 极少量 inf、Null/NaN 与特殊值落入专用箱，合法 0 保留。
                if isinstance(data, pd.DataFrame):
                    data.loc[1, features[0]] = np.inf
                    data.loc[2, features[1]] = -np.inf
                    reference = data.loc[:, necessary].head(dimensions["rows"] // 2).copy()
                else:
                    data = data.with_columns(
                        pl.when(pl.int_range(pl.len()) == 1)
                        .then(pl.lit(float("inf"), dtype=data.schema[features[0]]))
                        .otherwise(pl.col(features[0]))
                        .alias(features[0]),
                        pl.when(pl.int_range(pl.len()) == 2)
                        .then(pl.lit(float("-inf"), dtype=data.schema[features[1]]))
                        .otherwise(pl.col(features[1]))
                        .alias(features[1]),
                    )
                    reference = data.select(necessary).head(dimensions["rows"] // 2)
                with patch.object(pl, "from_pandas", capture), patch.object(
                    cross_module, "_fit_axis", fit
                ), patch.object(pl.DataFrame, "group_by", group_by):
                    report = meter.run(
                        "public_compute",
                        lambda: cross_scores(
                            data,
                            score_x=features[0],
                            score_y=features[1],
                            targets=["bad", "late"],
                            score_directions={
                                features[0]: "lower_risk",
                                features[1]: "higher_risk",
                            },
                            group_col="group",
                            weights_col="weight",
                            amount_col="amount",
                            binning_reference=reference,
                            special_values={features[0]: [-999], features[1]: [-999]},
                            feature_metadata=_metadata(features[:2]),
                            business_context=_context_metadata(),
                        ),
                    )
                assert len(fits) == 2
                assert len(aggregations) == 1 and aggregations[0]["targets"] == ["bad", "late"]
                assert all(set(cols) <= set(necessary) for cols in calls)
                meter.result.update(
                    converted_columns=calls,
                    axis_fit_calls=len(fits),
                    joint_aggregations=aggregations,
                )
                overall = report.get_table("overall", filters={"target": "bad"})
                assert overall["sample_count"].sum() == dimensions["rows"]
                observed = (
                    data["bad"].notna().sum()
                    if isinstance(data, pd.DataFrame)
                    else data["bad"].is_not_null().sum()
                )
                assert overall["observed_sample_count"].sum() == observed
                table = "cells"
                query = {
                    "filters": {"target": "bad", "group": "OOT"},
                    "columns": [
                        "x_bin",
                        "y_bin",
                        "sample_count",
                        "observed_sample_count",
                        "bad_rate",
                    ],
                    "sort_by": "sample_count",
                    "descending": True,
                    "offset": 20,
                    "limit": 20,
                }
            elif case.startswith("profile"):
                profiler = MarsDataProfiler(
                    overview_batch_size=dimensions["batch_size"], missing_values=[-999]
                )
                estimator_parameters = profiler.get_params(deep=False)
                report = meter.run(
                    "public_compute",
                    lambda: profiler.generate_profile(
                        data,
                        features=features,
                        group_col="group",
                        benchmark_df=data.head(dimensions["rows"] // 2),
                        enable_sparkline=False,
                        metrics=["missing", "zeros", "unique", "mode", "mean", "std", "min", "max"],
                        feature_metadata=_metadata(features),
                        business_context=_context_metadata(),
                    ),
                )
                table = "overview"
                query = {
                    "columns": ["feature", "mean"],
                    "sort_by": "feature",
                    "offset": max(0, len(features) - 20),
                    "limit": 20,
                }
            elif case == "selection":
                import mars.analysis.evaluator as evaluator_module
                import mars.analysis.profiler as profiler_module

                evidence_calls: dict[str, list[Any]] = {
                    "profile": [],
                    "evaluation": [],
                    "matrix": [],
                }
                original_profile = profiler_module.MarsDataProfiler.generate_profile
                original_evaluate = evaluator_module.MarsBinEvaluator.evaluate
                original_corr = pl.DataFrame.corr

                def capture_profile(tool: Any, frame: Any, **kwargs: Any) -> Any:
                    evidence_calls["profile"].append({"features": kwargs.get("features")})
                    return original_profile(tool, frame, **kwargs)

                def capture_evaluate(tool: Any, *values: Any, **kwargs: Any) -> Any:
                    evidence_calls["evaluation"].append(
                        {
                            "features": kwargs.get("features"),
                            "reused_binner": kwargs.get("binner") is not None,
                        }
                    )
                    return original_evaluate(tool, *values, **kwargs)

                def capture_matrix(frame: Any, **kwargs: Any) -> Any:
                    evidence_calls["matrix"].append(list(frame.columns))
                    return original_corr(frame, **kwargs)

                selector = MarsStatsSelector(
                    binning_params={"n_bins": 8},
                    rough_binning_params={"n_bins": 8},
                    missing_values=[],
                    special_values=[-999],
                    batch_size=dimensions["batch_size"],
                    n_jobs=args.threads,
                )
                estimator_parameters = selector.get_params(deep=False)
                with patch.object(
                    profiler_module.MarsDataProfiler, "generate_profile", capture_profile
                ), patch.object(
                    evaluator_module.MarsBinEvaluator, "evaluate", capture_evaluate
                ), patch.object(pl.DataFrame, "corr", capture_matrix):
                    meter.run(
                        "public_compute",
                        lambda: selector.fit(
                            data,
                            target="bad",
                            features=features,
                            group_col="group",
                            feature_metadata=_metadata(features),
                            business_context=_context_metadata(),
                        ),
                    )
                report = selector.get_correlation_report()
                assert selector._stage3_binner is not None
                meter.result["selection"] = {
                    "requested_n_jobs": selector.n_jobs,
                    "fine_binner_n_jobs": selector._stage3_binner.n_jobs,
                    "input_count": len(features),
                    "candidates": len(selector._corr_candidates),
                    "selected_count": len(selector.selected_features_),
                    "selected_order": selector.selected_features_,
                    "parameters": report.describe()["parameters"],
                    "evidence_calls": evidence_calls,
                    "funnel": selector._funnel_stats,
                }
                assert len(evidence_calls["matrix"]) == 1
                assert (
                    report.get_table("features").filter(pl.col("selected"))["feature"].to_list()
                    == selector.selected_features_
                )
                table = "features"
                query = {
                    "columns": ["feature", "selected", "participated"],
                    "sort_by": "feature",
                    "limit": 20,
                    "offset": 0,
                }
            elif case == "linear_selection":
                calls = []
                corr = pd.DataFrame.corr

                def capture_corr(frame: Any, **kwargs: Any) -> Any:
                    calls.append(list(frame.columns))
                    return corr(frame, **kwargs)

                selector = MarsLinearSelector(corr_thr=0.8, n_jobs=args.threads)
                estimator_parameters = selector.get_params(deep=False)
                # 标准公共入口包括可用的默认 VIF/Logit 诊断；不以禁用诊断提速。
                with patch.object(pd.DataFrame, "corr", capture_corr):
                    meter.run(
                        "public_compute",
                        lambda: selector.fit(
                            data,
                            data["bad"],
                            features=features,
                            feature_metadata=_metadata(features),
                            business_context=_context_metadata(),
                        ),
                    )
                assert len(calls) == 1
                report = selector.get_correlation_report()
                diagnostics = _linear_diagnostics(selector)
                meter.result["linear_dependency_state"] = diagnostics["dependency"]
                meter.result["selection"] = {
                    "matrix_calls": len(calls),
                    "selected_order": selector.selected_features_,
                    "candidate_count": len(selector._corr_candidates),
                    "parameters": report.describe()["parameters"],
                }
                table = "pairs"
                query = {
                    "columns": ["feature_a", "feature_b", "correlation"],
                    "limit": 20,
                    "offset": 0,
                }
            else:
                parameters = {"n_bins": 8, "special_values": [-999]}
                evaluator = MarsBinEvaluator(
                    binning_type="optimal" if case == "optimal_binning" else "native",
                    binner_params=parameters,
                )
                run = meter.run(
                    "public_compute",
                    lambda: evaluator.evaluate(
                        data,
                        target="bad",
                        features=features,
                        group_col="group",
                        weights_col="weight",
                        amount_col="amount",
                        benchmark_df=data.head(dimensions["rows"] // 2),
                        batch_size=dimensions["batch_size"],
                        feature_metadata=_metadata(features),
                        business_context=_context_metadata(),
                    ),
                )
                report = run.report
                estimator_parameters = run.binner.get_params(deep=False)
                meter.result["binner_n_jobs"] = run.binner.n_jobs
                table = "summary"
                query = {
                    "columns": ["feature", "iv", "ks"],
                    "sort_by": "feature",
                    "limit": 20,
                    "offset": max(0, len(features) - 20),
                }
            meter.result["correctness"] = meter.run(
                "statistics_correctness", lambda: _check_statistics(case, data, report, features)
            )
            del data
        meter.result["input_ready_rss_bytes"] = meter.result.get(
            "input_ready_rss_bytes", meter.result["stages"]["input_prepare"]["rss_end_bytes"]
        )
        # 快照契约逐表全量哈希；有限证据单独校验，均不导出宽表或全表 Python 行。
        description = report.describe()
        from mars.reporting._serialization import json_safe

        algorithm, resources = _execution_parameters(json_safe({
            "report": description["parameters"], "estimator": estimator_parameters,
        }))
        effective_workload, workload_resources = _execution_parameters(meter.result["workload"])
        meter.result["execution_contract"] = {
            "workload_id": WORKLOAD_CONTRACT,
            "workload": {"case": case, **effective_workload},
            "algorithm_parameters": algorithm,
            "diagnostics": diagnostics,
            "resource_strategy": {**resources, **workload_resources},
        }
        signatures = meter.run(
            "table_correctness",
            lambda: {name: _signature(report.get_table(name)) for name in description["tables"]},
        )
        meter.result["tables"] = {
            name: {
                **sig,
                "backend": type(report.get_table(name)).__module__.split(".")[0],
                "storage_bytes": report.get_table(name).estimated_size()
                if isinstance(report.get_table(name), pl.DataFrame)
                else int(report.get_table(name).memory_usage(deep=True).sum()),
            }
            for name, sig in signatures.items()
        }
        page = meter.run("query_page", lambda: report.query_page(table, **query))
        assert page["returned_rows"] > 0
        path = folder / "report.marsreport"
        meter.run("save", lambda: report.save(path))
        meter.result["file_bytes"] = path.stat().st_size
        ai_query = {**query, "offset": 0, "limit": 20}
        config: dict[str, Any] = {
            "report_id": report.report_id,
            "description": description,
            "signatures": signatures,
            "page_signature": _signature(page["data"]),
            "page_total": page["total_rows"],
            "table": table,
            "query": query,
            "ai": {table: ai_query},
            "feature": features[0],
            "matrix_features": description["parameters"].get("candidate_scope", features)[:30],
            "export": args.scale == "smoke" and case in ("rule_mining", "score_cross"),
        }
        if report.report_type == "score_cross":
            policy = evaluate_score_policy(
                report,
                {"type": "and", "x_max_risk_rank": 3, "y_max_risk_rank": 3},
                baseline={"type": "x_only", "x_max_risk_rank": 3},
            )
            config["policy_signature"] = _signature(policy.get_table("summary"))
        config_path = folder / "queries.json"
        _write(config_path, config)
        report = None
        gc.collect()
        out = folder / "consumer.json"
        consumer = meter.run(
            "fresh_process_consume",
            lambda: _launch(
                [
                    sys.executable,
                    str(Path(__file__).with_name("benchmark_core_capacity.py")),
                    "--worker",
                    "--consume",
                    str(path),
                    "--query-config",
                    str(config_path),
                    "--source",
                    str((getattr(args, "consumer_source", None) or args.source).resolve()),
                    "--threads",
                    str(args.threads),
                    "--diagnostic-loops",
                    str(args.diagnostic_loops),
                    "--output",
                    str(out),
                ],
                out,
                args.timeout,
                int(args.memory_budget_mib * 1024**2),
            ),
        )
        meter.result["consumer"] = consumer
        meter.result.update(
            status=consumer["status"], stage="complete", reason=consumer.get("reason")
        )
