"""Home Credit 共享准备的输入、稳定散列、整日分区和标签边界。"""

from __future__ import annotations

import csv
import json
import runpy
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from docs.snippets.home_credit_data import (
    GROUPS,
    ROLE_COLUMNS,
    HomeCreditConfig,
    HomeCreditInputs,
    _partition_labeled_days,
    prepare_home_credit_data,
)

ROOT = Path(__file__).resolve().parents[1]


def _fixture_inputs(directory: Path) -> dict[str, Path]:
    """创建明确的小型 application fixture；不读取私人或官方数据。"""
    directory.mkdir(parents=True, exist_ok=True)
    train = pl.DataFrame({
        "SK_ID_CURR": range(1, 241), "TARGET": [index % 2 for index in range(240)],
        "AMT_CREDIT": [1000.5] * 240, "EXT_SOURCE_2": [index / 240 for index in range(240)],
        "DAYS_EMPLOYED": [-200] * 239 + [365243], "UNKNOWN_FEATURE": [1] * 240,
    })
    test = pl.DataFrame({
        "SK_ID_CURR": range(10001, 10021), "AMT_CREDIT": [2000] * 20,
        "EXT_SOURCE_2": [0.5] * 20, "DAYS_EMPLOYED": [-100] * 20,
        "UNKNOWN_FEATURE": [2] * 20,
    })
    paths = {"train": directory / "application_train.csv", "test": directory / "application_test.csv",
             "description": directory / "HomeCredit_columns_description.csv"}
    train.write_csv(paths["train"])
    test.write_csv(paths["test"])
    with paths["description"].open("w", encoding="cp1252", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["", "Table", "Row", "Description", "Special"])
        writer.writerows([
            [1, "application_{train|test}.csv", "SK_ID_CURR", "ID of loan in our sample", ""],
            [2, "application_train.csv", "TARGET", "Original binary payment difficulty target", ""],
            [3, "application_train.csv", "AMT_CREDIT", "Credit amount of the loan", ""],
            [4, "application_test.csv", "AMT_CREDIT", "Credit amount of the loan", ""],
            [5, "application_{train|test}.csv", "EXT_SOURCE_2", "Normalized score from external data source", "normalized"],
            [6, "application_{train|test}.csv", "DAYS_EMPLOYED", "How many days before application current employment started…", "time only relative to the application"],
            [7, "bureau.csv", "SK_ID_CURR", "This bureau definition must not override application", ""],
            [8, "previous_application.csv", "PRIVATE_OTHER_TABLE_FIELD", "Other table only", ""],
        ])
    return paths


def _prepare(directory: Path) -> HomeCreditInputs:
    return prepare_home_credit_data(
        directory, config=HomeCreditConfig(target_group_rows=30), require_full_dataset=False,
    )


def _assignment(df: pl.DataFrame) -> list[dict[str, Any]]:
    return df.select("SK_ID_CURR", "APPLY_DATE", "dataset_flag").sort("SK_ID_CURR").to_dicts()


def test_preserves_every_row_and_cleans_only_unavailable_labels(tmp_path: Path) -> None:
    paths = _fixture_inputs(tmp_path)
    train_original = pl.read_csv(paths["train"]).select("SK_ID_CURR", "TARGET").sort("SK_ID_CURR")
    prepared = _prepare(tmp_path)
    assert prepared.df.shape == (260, 8)
    assert prepared.df.schema["TARGET"] == pl.Int8
    assert prepared.df.schema["APPLY_DATE"] == pl.Date
    assert prepared.df["SK_ID_CURR"].n_unique() == 260
    assert set(prepared.df["dataset_flag"].unique()) == set(GROUPS)
    labeled = prepared.df.filter(pl.col("dataset_flag") != "4_oot_undue")
    assert labeled.select("SK_ID_CURR", "TARGET").sort("SK_ID_CURR").equals(train_original)
    overdue = prepared.df.filter(pl.col("dataset_flag") == "4_oot_undue")
    assert overdue.height == overdue["TARGET"].null_count() == 20
    assert set(prepared.df["TARGET"].drop_nulls().unique()) == {0, 1}
    assert labeled["APPLY_DATE"].min() >= date(2026, 1, 1)
    assert labeled["APPLY_DATE"].max() <= date(2026, 6, 30)
    assert overdue["APPLY_DATE"].min() >= date(2026, 7, 1)
    assert overdue["APPLY_DATE"].max() <= date(2026, 7, 31)
    assert prepared.benchmark_df["dataset_flag"].unique().to_list() == ["0train"]
    assert prepared.benchmark_df is prepared.benchmark_df
    assert labeled.group_by("APPLY_DATE").agg(pl.col("dataset_flag").n_unique())["dataset_flag"].max() == 1
    bounds = [(row["date_min"], row["date_max"]) for row in prepared.summary["partition_stats"]]
    assert all(left[1] < right[0] for left, right in zip(bounds, bounds[1:]))
    undue = prepared.summary["partition_stats"][-1]
    assert undue["observed_count"] == 0 and undue["null_count"] == 20
    assert undue["observed_bad_rate"] is None and undue["label_status"] == "unobserved"


def test_csv_shuffle_and_binary_label_flip_cannot_change_assignment(tmp_path: Path) -> None:
    paths = _fixture_inputs(tmp_path)
    before = _assignment(_prepare(tmp_path).df)
    pl.read_csv(paths["train"]).sample(fraction=1, shuffle=True, seed=932).write_csv(paths["train"])
    pl.read_csv(paths["test"]).sample(fraction=1, shuffle=True, seed=176).write_csv(paths["test"])
    assert _assignment(_prepare(tmp_path).df) == before
    pl.read_csv(paths["train"]).with_columns((1 - pl.col("TARGET")).alias("TARGET")).write_csv(paths["train"])
    assert _assignment(_prepare(tmp_path).df) == before


def test_new_process_replays_the_same_fixture_dates_and_partitions(tmp_path: Path) -> None:
    _fixture_inputs(tmp_path)
    expected = json.dumps(_assignment(_prepare(tmp_path).df), default=str, sort_keys=True)
    code = (
        "import json,sys; from pathlib import Path; "
        f"sys.path.insert(0,{str(ROOT / 'docs/snippets')!r}); "
        "from home_credit_data import prepare_home_credit_data,HomeCreditConfig; "
        f"p=prepare_home_credit_data({str(tmp_path)!r},config=HomeCreditConfig(target_group_rows=30),require_full_dataset=False); "
        "print(json.dumps(p.df.select('SK_ID_CURR','APPLY_DATE','dataset_flag').sort('SK_ID_CURR').to_dicts(),default=str,sort_keys=True))"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == expected


def test_whole_days_choose_nearest_counts_and_earlier_boundary_on_tie() -> None:
    days = [date(2026, 1, 1) + timedelta(days=offset) for offset in range(7)]
    # 最末日 2 人与末两日 6 人距目标 4 均为 2；确定选更早的末两日。
    daily = list(zip(days, [11, 2, 5, 3, 7, 4, 2]))
    groups = _partition_labeled_days(daily, target_rows=4)
    assert groups[days[5]] == groups[days[6]] == "3oot"
    assert groups[days[4]] == "2test"
    assert groups[days[3]] == "1val"
    assert all(groups[day] == "0train" for day in days[:3])


@pytest.mark.parametrize("daily,target_rows", [
    ([(date(2026, 1, 1), 1)] * 3, 1),
    ([(date(2026, 1, 1) + timedelta(days=i), 1) for i in range(4)], 0),
    ([(date(2026, 1, 1) + timedelta(days=i), 0) for i in range(4)], 1),
])
def test_inadequate_natural_days_or_invalid_target_fails(daily: list[tuple[date, int]], target_rows: int) -> None:
    with pytest.raises(ValueError):
        _partition_labeled_days(daily, target_rows)


def test_metadata_is_application_only_and_roles_are_excluded(tmp_path: Path) -> None:
    _fixture_inputs(tmp_path)
    prepared = _prepare(tmp_path)
    assert not ROLE_COLUMNS.intersection(prepared.fcols)
    assert prepared.fcols == ["EXT_SOURCE_2", "DAYS_EMPLOYED", "UNKNOWN_FEATURE"]
    metadata = prepared.feature_metadata
    assert metadata["SK_ID_CURR"]["description"] == "ID of loan in our sample"
    assert "PRIVATE_OTHER_TABLE_FIELD" not in metadata
    assert metadata["EXT_SOURCE_2"]["display_name"] == "外部来源分 2"
    assert metadata["EXT_SOURCE_2"]["unit"] == "归一化值（无量纲）"
    assert metadata["AMT_CREDIT"]["unit"] == "原始金额单位（币种未公开）"
    assert prepared.summary["metadata"]["field_definitions"]["UNKNOWN_FEATURE"]["definition_status"] == "unknown"
    assert prepared.summary["metadata"]["duplicate_application_rows_merged"] == 1
    assert prepared.summary["description_encoding"] == "cp1252"
    assert prepared.summary["columns_description"]["rows"] == 8
    assert prepared.summary["columns_description"]["application_rows"] == 6
    assert prepared.summary["special_values"]["DAYS_EMPLOYED"]["count"] == 1
    serialized = json.dumps(prepared.summary, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert "members" not in serialized


def test_conflicting_official_definitions_are_unknown_instead_of_invented(tmp_path: Path) -> None:
    paths = _fixture_inputs(tmp_path)
    with paths["description"].open("a", encoding="cp1252", newline="") as stream:
        csv.writer(stream).writerow([9, "application_test.csv", "EXT_SOURCE_2", "Conflicting external source definition", ""])
    prepared = _prepare(tmp_path)
    definition = prepared.summary["metadata"]["field_definitions"]["EXT_SOURCE_2"]
    assert definition["definition_status"] == "conflict"
    assert len(definition["official_definitions"]) == 2
    assert prepared.summary["metadata"]["conflicting_definitions"] == ["EXT_SOURCE_2"]


@pytest.mark.parametrize("corruption,match", [
    ("duplicate_train_id", "唯一"), ("null_test_id", "非空"),
    ("overlapping_id", "交集"), ("illegal_target", "0/1"),
    ("null_target", "0/1"), ("missing_business_field", "字段集合"),
    ("test_has_target", "不应提供 TARGET"), ("reserved_role", "保留字段"),
])
def test_unexpected_schema_labels_or_ids_fail_explicitly(tmp_path: Path, corruption: str, match: str) -> None:
    paths = _fixture_inputs(tmp_path)
    train, test = pl.read_csv(paths["train"]), pl.read_csv(paths["test"])
    if corruption == "duplicate_train_id":
        train = train.with_columns(pl.lit(1).alias("SK_ID_CURR"))
    elif corruption == "null_test_id":
        test = test.with_columns(pl.lit(None).alias("SK_ID_CURR"))
    elif corruption == "overlapping_id":
        test = test.with_columns((pl.col("SK_ID_CURR") - 10000).alias("SK_ID_CURR"))
    elif corruption == "illegal_target":
        train = train.with_columns(pl.lit(-1).alias("TARGET"))
    elif corruption == "null_target":
        train = train.with_columns(pl.lit(None).alias("TARGET"))
    elif corruption == "missing_business_field":
        test = test.drop("EXT_SOURCE_2")
    elif corruption == "test_has_target":
        test = test.with_columns(pl.lit(0).alias("TARGET"))
    elif corruption == "reserved_role":
        train = train.with_columns(pl.lit("0train").alias("dataset_flag"))
    train.write_csv(paths["train"])
    test.write_csv(paths["test"])
    with pytest.raises(ValueError, match=match):
        _prepare(tmp_path)


def test_duplicate_csv_headers_cannot_be_silently_renamed(tmp_path: Path) -> None:
    paths = _fixture_inputs(tmp_path)
    content = paths["test"].read_text(encoding="utf-8")
    paths["test"].write_text(content.replace("UNKNOWN_FEATURE", "EXT_SOURCE_2", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="重复列名"):
        _prepare(tmp_path)


def test_missing_input_and_non_official_scale_do_not_fallback(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="application_train.csv"):
        prepare_home_credit_data(tmp_path)
    _fixture_inputs(tmp_path)
    with pytest.raises(ValueError, match="正式输入规模不符"):
        prepare_home_credit_data(tmp_path)


def test_three_explicit_csv_overrides_are_supported(tmp_path: Path) -> None:
    paths = _fixture_inputs(tmp_path / "inputs")
    prepared = prepare_home_credit_data(
        tmp_path / "nonexistent", train_csv=paths["train"], test_csv=paths["test"],
        columns_description_csv=paths["description"],
        config=HomeCreditConfig(target_group_rows=30), require_full_dataset=False,
    )
    assert prepared.df.height == 260


def test_shared_metadata_passes_mars_contract_and_real_risk_entry(tmp_path: Path) -> None:
    from mars.analysis import profile_risk
    from mars.reporting._metadata import normalize_metadata

    _fixture_inputs(tmp_path)
    prepared = _prepare(tmp_path)
    normalized = normalize_metadata(prepared.feature_metadata, prepared.fcols)
    assert all(set(entry) <= {"display_name", "description", "data_source", "unit"} for entry in normalized.values())
    report = profile_risk(
        prepared.df, target="TARGET", features=["EXT_SOURCE_2"], group_col="dataset_flag",
        benchmark_df=prepared.benchmark_df, amount_col="AMT_CREDIT",
        feature_metadata=prepared.feature_metadata, business_context=prepared.business_context,
        n_bins=3, n_jobs=1,
    )
    description = report.report.describe()
    assert description["feature_metadata"]["EXT_SOURCE_2"]["display_name"] == "外部来源分 2"
    assert description["business_context"]["labels"]["TARGET"]["available_in"] == "application_train.csv"


@pytest.mark.parametrize("corruption", ["invented_unlabeled_risk", "invented_unlabeled_observation", "amount_in_features"])
def test_public_aggregate_validation_rejects_invented_labels_and_role_leak(tmp_path: Path, corruption: str) -> None:
    _fixture_inputs(tmp_path / "inputs")
    prepared = _prepare(tmp_path / "inputs")
    preparation = prepared.summary
    if corruption == "invented_unlabeled_risk":
        preparation["partition_stats"][-1]["observed_bad_rate"] = 0.0
    elif corruption == "invented_unlabeled_observation":
        undue = preparation["partition_stats"][-1]
        undue.update({"observed_count": 1, "null_count": 19, "target_0_count": 1, "label_coverage": 0.05})
    else:
        preparation["fcols"].append("AMT_CREDIT")
    manifest = {"data_config": {"preparation": preparation, "rows": preparation["rows"], "seed": preparation["seed"]}}
    for name, value in (
        ("preparation-summary.json", preparation), ("feature-metadata.json", prepared.feature_metadata),
        ("summary.json", {"rows": preparation["rows"], "splits": {row["dataset_flag"]: row["total_count"] for row in preparation["partition_stats"]}}),
    ):
        (tmp_path / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    checker = runpy.run_path(str(ROOT / "scripts/check_case_assets.py"))
    with pytest.raises(ValueError, match="Original test labels|role/source"):
        checker["_validate_preparation"](tmp_path, manifest)


def test_public_excel_reassembles_long_unicode_metadata_and_detects_missing_piece() -> None:
    import openpyxl

    from mars.reporting._metadata import export_semantics

    checker = runpy.run_path(str(ROOT / "scripts/check_case_assets.py"))
    context = {"description": "边界有空格  🙂" * 9000, "source": "application_train.csv"}
    frame = export_semantics({"business_context": context})["BusinessContext"]
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(list(frame.columns))
    for row in frame.itertuples(index=False, name=None):
        sheet.append(row)
    assert checker["_excel_json_mapping"](sheet, "fixture") == context
    sheet.delete_rows(2)
    with pytest.raises(ValueError, match="Missing Excel metadata chunk"):
        checker["_excel_json_mapping"](sheet, "fixture")


@pytest.mark.parametrize("malformed_bytes", [False, True])
def test_recompute_captures_utf8_failure_and_rejects_invalid_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed_bytes: bool,
) -> None:
    _fixture_inputs(tmp_path / "inputs")
    prepared = _prepare(tmp_path / "inputs")
    checker = runpy.run_path(str(ROOT / "scripts/check_case_assets.py"))
    real_run = subprocess.run
    code = (
        "import sys; sys.stdout.buffer.write(bytes([255])); sys.exit(9)"
        if malformed_bytes else
        "import sys; print('公开案例生成失败'); print('未观测标签保留为空', file=sys.stderr); sys.exit(9)"
    )

    def run_failure(_command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        """真实子进程输出中文或坏字节，保留 checker 的编码与严格解码配置。"""
        assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
        return real_run([sys.executable, "-c", code], **kwargs)

    monkeypatch.setattr(checker["subprocess"], "run", run_failure)
    manifest = {"data_config": {
        "seed": prepared.summary["seed"], "target_group_rows": 30,
        "preparation": prepared.summary,
    }}
    if malformed_bytes:
        with pytest.raises(UnicodeDecodeError):
            checker["_recompute"](tmp_path, manifest, ["--data-dir", str(tmp_path / "inputs")])
    else:
        with pytest.raises(ValueError, match="公开案例生成失败\\r?\\n\\n未观测标签保留为空"):
            checker["_recompute"](tmp_path, manifest, ["--data-dir", str(tmp_path / "inputs")])
