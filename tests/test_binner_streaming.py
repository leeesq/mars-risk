"""独立线程进程中的类别 Join、监督人口和跨列批 WOE 对齐回归。"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mars._compat import collect_streaming
from mars.feature import MarsNativeBinner, MarsStatsSelector
from mars.feature.binning.base import MarsBinnerBase
from mars.reporting import load_report


def _sample() -> tuple[pl.DataFrame, pl.Series, list[str]]:
    """同时放入两个 Join、跨批数值列、未表现标签和潜在内部列名。"""
    count = 83
    numeric = [None, -999.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    category = [None, "special", "missing", "a", "b", "c", "a", "b"]
    features = [f"number_{i:02}" for i in range(19)]
    features[1], features[17] = "category_a", "category_b"
    values: dict[str, Any] = {
        feature: [numeric[(row + i) % 8] for row in range(count)]
        for i, feature in enumerate(features)
    }
    values["category_a"] = [category[row % 8] for row in range(count)]
    values["category_b"] = [category[(row + 3) % 8] for row in range(count)]
    # 用户字段与内部标签、Join key、行身份及其候选后缀重名，均须原样保留。
    for name in (
        "target", "_mars_target", "_mars_target_1", "_mars_join_order",
        "_mars_join_order_1", "_idx_category_a", "_category_a_utf8_tmp",
        "_idx_category_b", "_category_b_utf8_tmp",
    ):
        values[name] = list(range(count))
    values["row_identity"] = list(range(count))
    frame = pl.DataFrame(values)
    frame = pl.concat([frame.head(13), frame.slice(13, 22), frame.slice(35)], rechunk=False)
    labels = [float(value) if value is not None else None
              for value in ([0, 1, None, 0, 1, 0, 1, 1] * 11)[:count]]
    labels[12] = float("nan")
    return frame, pl.Series("target", labels), features


def _binner(threshold: int) -> MarsNativeBinner:
    return MarsNativeBinner(
        n_bins=4, n_jobs=1, join_threshold=threshold,
        special_values=[-999.0, "special"], missing_values=[-1.0, "missing"],
    )


def _selector(
    batch: int, binner: MarsNativeBinner, features: list[str], ivs: dict[str, float],
) -> MarsStatsSelector:
    """只隔离相关性阶段，其参考矩阵由 Replace 全人口转换独立生成。"""
    selector = MarsStatsSelector(batch_size=batch, corr_thr=0.9, n_jobs=1)
    selector._reset_correlation()
    selector.target = "bad"
    selector.white_list = ["category_b"]
    selector._stage3_binner = binner
    selector._corr_input_features = features.copy()
    selector._feature_iv_dict = ivs.copy()
    selector._corr_parameters = {"operator": ">", "priority_metric": "IV", "tie_rule": "ID"}
    return selector


def _assert_dense_tables(actual: pl.DataFrame, expected: pl.DataFrame) -> None:
    """离散字段和 null 掩码严格一致，只有真实浮点值使用原数值容差。"""
    assert actual.schema == expected.schema
    assert actual.shape == expected.shape
    floating = [name for name, dtype in actual.schema.items()
                if dtype in {pl.Float32, pl.Float64}]
    assert_frame_equal(actual.drop(floating), expected.drop(floating))
    for name in floating:
        assert actual[name].is_null().to_list() == expected[name].is_null().to_list()
        np.testing.assert_allclose(actual[name].to_numpy(), expected[name].to_numpy(),
                                   atol=3e-15, rtol=0, equal_nan=True)


def _restore(directory: Path) -> None:
    """新解释器加载正式规则和快照，查询无需训练缓存或重新估计 WOE。"""
    binner = MarsBinnerBase.load_json(directory / "rules.json")
    assert binner._cache_X is None and binner._cache_y is None
    probe = pl.read_ipc(directory / "probe.arrow", memory_map=False)
    features = json.loads((directory / "features.json").read_text(encoding="utf-8"))
    with patch.object(binner, "profile_bin_performance", side_effect=AssertionError("WOE refit")):
        for kind in ("index", "label", "woe"):
            expected = pl.read_ipc(directory / f"{kind}.arrow", memory_map=False)
            planned = binner.transform(probe, features=features, return_type=kind, lazy=True)
            assert_frame_equal(collect_streaming(planned), expected)
            assert_frame_equal(planned.collect(), expected)
    report = load_report(directory / "correlation.marsreport")
    for name in ("features", "pairs", "correlation_decisions", "selection"):
        expected = pl.read_ipc(directory / f"report_{name}.arrow", memory_map=False)
        assert_frame_equal(report.get_table(name), expected)
        assert_frame_equal(report.query_page(name, limit=3)["data"], expected.head(3))
    assert report.get_feature("category_a")["tables"]["pairs"].height == len(features) - 1
    context = json.loads(report.to_ai_context(tables=["pairs"], limit=3, max_chars=16000))
    assert context["evidence"][0]["returned_rows"] == 3


def _worker(directory: Path) -> None:
    """在 Polars 首次导入前配置好的线程池中执行真实普通与 streaming 引擎。"""
    assert pl.thread_pool_size() == int(os.environ["POLARS_MAX_THREADS"])
    frame, labels, features = _sample()
    original = frame.clone()
    chunks = [frame.get_column(name).n_chunks() for name in frame.columns]
    assert min(chunks) > 1
    observed = labels.is_not_null() & ~labels.is_nan()
    normalized = labels.filter(observed).cast(pl.Int64)
    replacement = _binner(100).fit(frame, labels, features=features)
    expected_stats = replacement.profile_bin_performance(
        frame, labels, features=features, batch_size=100, include_bin_index=True,
    )
    indices = replacement.transform(frame, features=features).filter(observed)
    # 逐行 Zip 的标签是独立计数参考，不能从受测 Join 后的行位置派生。
    for feature in features:
        bins: dict[int, list[int]] = {}
        for index, target in zip(indices[f"{feature}_bin"].to_list(), normalized.to_list()):
            bins.setdefault(index, []).append(target)
        actual = expected_stats.filter(pl.col("feature") == feature)
        assert {row["bin_index"]: (row["count"], row["bad"]) for row in actual.to_dicts()} == {
            index: (len(targets), sum(targets)) for index, targets in bins.items()
        }
    probe = frame.with_columns(
        pl.when(pl.col("row_identity") == 3).then(pl.lit("unseen_a"))
        .otherwise(pl.col("category_a")).alias("category_a"),
        pl.when(pl.col("row_identity") == 8).then(pl.lit("unseen_b"))
        .otherwise(pl.col("category_b")).alias("category_b"),
    )
    expected = {
        kind: replacement.transform(probe, features=features, return_type=kind)
        for kind in ("index", "label", "woe")
    }
    assert expected["index"]["category_a_bin"].head(4).to_list() == [-1, -4, -1, -2]
    assert expected["index"]["category_b_bin"][8] == -2
    correlation_input = frame.with_columns(
        pl.when(pl.lit(observed)).then(pl.lit(labels)).otherwise(None).alias("bad"),
    )
    dense_values = replacement.transform(frame.filter(observed), features=features,
                                         return_type="woe").select([
        pl.col(f"{feature}_woe").fill_null(0).cast(pl.Float64) for feature in features
    ]).to_numpy()
    dense_matrix = np.corrcoef(dense_values, rowvar=False)
    ivs = dict(expected_stats.select("feature", "IV").unique().iter_rows())
    for batch in (1, 16, 100):
        joined = _binner(0).fit(frame, labels, features=features)
        assert set(joined.cat_cuts_) == {"category_a", "category_b"}
        actual_stats = joined.profile_bin_performance(
            frame, labels, features=features, batch_size=batch, include_bin_index=True,
        )
        assert_frame_equal(actual_stats, expected_stats)
        assert joined.bin_woes_ == replacement.bin_woes_
        for kind, reference in expected.items():
            assert_frame_equal(joined.transform(probe, features=features, return_type=kind), reference)
            for source in (probe, probe.lazy()):
                planned = joined.transform(source, features=features, return_type=kind, lazy=True)
                assert_frame_equal(planned.collect(), reference)
                assert_frame_equal(collect_streaming(planned), reference)
                assert_frame_equal(collect_streaming(planned).select(probe.columns), probe)
                assert set(planned.collect_schema().names()) == set(reference.columns)
        # 不提供唯一业务字段；完全重复的行也必须保留其位置与数量。
        duplicates = probe.select(features)
        assert duplicates.is_duplicated().any()
        assert_frame_equal(
            collect_streaming(joined.transform(duplicates, features=features, lazy=True)),
            replacement.transform(duplicates, features=features),
        )
        selector = _selector(batch, joined, features, ivs)
        np.testing.assert_array_equal(
            np.concatenate(list(selector._woe_row_blocks(correlation_input, features, 17))),
            dense_values,
        )
        selector.selected_features_ = selector._filter_corr(correlation_input, features)
        np.testing.assert_allclose(selector._corr_matrix, dense_matrix, atol=3e-15, rtol=0)
        dense = _selector(batch, replacement, features, ivs)
        with patch("mars.feature.selection.stats._blocked_pearson",
                   return_value=(dense_matrix.copy(), len(normalized))):
            dense.selected_features_ = dense._filter_corr(correlation_input, features)
        assert selector.selected_features_ == dense.selected_features_
        assert selector.selected_features_ == [f for f in features if f in selector.selected_features_]
        for candidate in (selector, dense):
            candidate._finish_correlation()
            candidate._is_fitted = True
        report, dense_report = selector.get_correlation_report(), dense.get_correlation_report()
        for name in ("features", "pairs", "correlation_decisions", "selection"):
            _assert_dense_tables(report.get_table(name), dense_report.get_table(name))
        protected = report.get_table("correlation_decisions", filters={"action": "protect"})
        assert protected["feature"].to_list() == ["category_b"]
        assert report.get_table("pairs").height == len(features) * (len(features) - 1) // 2
        # 已计算 WOE 只能复用映射；失败 refit 必须清空缓存、规则及历史映射。
        mappings, woes = copy.deepcopy(joined.bin_mappings_), copy.deepcopy(joined.bin_woes_)
        with patch.object(joined, "profile_bin_performance", side_effect=AssertionError("WOE refit")):
            assert_frame_equal(collect_streaming(joined.transform(
                probe, features=features, return_type="woe", lazy=True,
            )), expected["woe"])
        assert joined.bin_mappings_ == mappings and joined.bin_woes_ == woes
        if batch == 100:
            joined.save_json(directory / "rules.json")
            probe.write_ipc(directory / "probe.arrow")
            (directory / "features.json").write_text(json.dumps(features), encoding="utf-8")
            for kind, reference in expected.items():
                reference.write_ipc(directory / f"{kind}.arrow")
            report.save(directory / "correlation.marsreport", overwrite=True)
            for name in ("features", "pairs", "correlation_decisions", "selection"):
                report.get_table(name).write_ipc(directory / f"report_{name}.arrow")
        with pytest.raises(ValueError, match="absent"):
            joined.fit(frame, labels, features=["absent"])
        assert not joined.__sklearn_is_fitted__()
        assert not joined.bin_cuts_ and not joined.cat_cuts_
        assert not joined.bin_mappings_ and not joined.bin_woes_
        assert joined._cache_X is None and joined._cache_y is None
    assert_frame_equal(frame, original)
    assert [frame.get_column(name).n_chunks() for name in frame.columns] == chunks
    restored = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--restore", str(directory)],
        env=os.environ.copy(), capture_output=True, text=True, timeout=120,
    )
    assert restored.returncode == 0, restored.stdout + restored.stderr
    print(json.dumps({"polars": pl.__version__, "threads": pl.thread_pool_size(),
                      "rows": frame.height, "observed": len(normalized), "batches": [1, 16, 100],
                      "engines": ["eager", "collect", "collect_streaming"], "restore_exit": 0}))


@pytest.mark.parametrize("threads", [1, 4])
def test_streaming_join_population_and_dense_reference_in_fresh_process(
    threads: int, tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env["POLARS_MAX_THREADS"] = str(threads)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence = json.loads(completed.stdout.strip().splitlines()[-1])
    assert evidence["threads"] == threads
    assert evidence["batches"] == [1, 16, 100] and evidence["restore_exit"] == 0


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]))
    else:
        assert sys.argv[1] == "--restore"
        _restore(Path(sys.argv[2]))
