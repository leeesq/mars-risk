"""Home Credit 2018 两表的确定性输入准备；只公开聚合摘要，不导出原始样本。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any

import polars as pl

PREPARATION_VERSION = "home-credit-2018-v1"
DEFAULT_DATA_DIR = Path("examples/home-credit-2018/data/raw_data")
TRAIN_START = date(2026, 1, 1)
TRAIN_END = date(2026, 6, 30)
TEST_START = date(2026, 7, 1)
TEST_END = date(2026, 7, 31)
GROUPS = ("0train", "1val", "2test", "3oot", "4_oot_undue")
ROLE_COLUMNS = {"SK_ID_CURR", "TARGET", "APPLY_DATE", "dataset_flag", "AMT_CREDIT"}
DEFAULT_DISPLAY_FEATURES = [
    "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1", "AMT_INCOME_TOTAL",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_CONTRACT_TYPE",
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "OCCUPATION_TYPE", "REGION_POPULATION_RELATIVE", "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH", "CNT_FAM_MEMBERS",
]

# 翻译只登记原字段含义；不猜测币种、外部分提供机构、表现期或衍生变量。
DISPLAY_NAMES: dict[str, str] = {
    "SK_ID_CURR": "申请记录标识", "TARGET": "比赛定义的还款困难事件",
    "NAME_CONTRACT_TYPE": "贷款合同类型", "CODE_GENDER": "客户性别",
    "FLAG_OWN_CAR": "是否拥有汽车", "FLAG_OWN_REALTY": "是否拥有住宅",
    "CNT_CHILDREN": "子女人数", "AMT_INCOME_TOTAL": "客户收入",
    "AMT_CREDIT": "贷款金额", "AMT_ANNUITY": "贷款年金",
    "AMT_GOODS_PRICE": "贷款对应商品价格", "NAME_TYPE_SUITE": "申请时陪同人员类型",
    "NAME_INCOME_TYPE": "收入类型", "NAME_EDUCATION_TYPE": "最高教育程度",
    "NAME_FAMILY_STATUS": "家庭状态", "NAME_HOUSING_TYPE": "居住情况",
    "REGION_POPULATION_RELATIVE": "居住地区归一化人口规模",
    "DAYS_BIRTH": "申请时年龄天数（原始值）",
    "DAYS_EMPLOYED": "开始当前工作相对申请的天数",
    "DAYS_REGISTRATION": "变更登记相对申请的天数",
    "DAYS_ID_PUBLISH": "变更身份证件相对申请的天数",
    "OWN_CAR_AGE": "汽车年龄", "FLAG_MOBIL": "是否提供手机号码",
    "FLAG_EMP_PHONE": "是否提供工作电话",
    "FLAG_WORK_PHONE": "是否提供家庭电话（官方字段说明）",
    "FLAG_CONT_MOBILE": "手机是否可联系", "FLAG_PHONE": "是否提供家庭电话",
    "FLAG_EMAIL": "是否提供电子邮箱", "OCCUPATION_TYPE": "职业类型",
    "CNT_FAM_MEMBERS": "家庭成员人数", "REGION_RATING_CLIENT": "居住地区评级",
    "REGION_RATING_CLIENT_W_CITY": "考虑城市的居住地区评级",
    "WEEKDAY_APPR_PROCESS_START": "申请星期", "HOUR_APPR_PROCESS_START": "申请大致小时",
    "REG_REGION_NOT_LIVE_REGION": "永久与联系地址地区是否不同",
    "REG_REGION_NOT_WORK_REGION": "永久与工作地址地区是否不同",
    "LIVE_REGION_NOT_WORK_REGION": "联系与工作地址地区是否不同",
    "REG_CITY_NOT_LIVE_CITY": "永久与联系地址城市是否不同",
    "REG_CITY_NOT_WORK_CITY": "永久与工作地址城市是否不同",
    "LIVE_CITY_NOT_WORK_CITY": "联系与工作地址城市是否不同",
    "ORGANIZATION_TYPE": "工作单位类型",
    "EXT_SOURCE_1": "外部来源分 1", "EXT_SOURCE_2": "外部来源分 2",
    "EXT_SOURCE_3": "外部来源分 3", "FONDKAPREMONT_MODE": "建筑信息 FONDKAPREMONT 众数（细项未公开）",
    "HOUSETYPE_MODE": "住房类型众数", "TOTALAREA_MODE": "建筑总面积归一化众数",
    "WALLSMATERIAL_MODE": "墙体材料众数", "EMERGENCYSTATE_MODE": "建筑状态信息众数（细项未公开）",
    "OBS_30_CNT_SOCIAL_CIRCLE": "社交圈可观察的 30 天逾期记录数",
    "DEF_30_CNT_SOCIAL_CIRCLE": "社交圈发生 30 天逾期的记录数",
    "OBS_60_CNT_SOCIAL_CIRCLE": "社交圈可观察的 60 天逾期记录数",
    "DEF_60_CNT_SOCIAL_CIRCLE": "社交圈发生 60 天逾期的记录数",
    "DAYS_LAST_PHONE_CHANGE": "更换电话相对申请的天数",
    "AMT_REQ_CREDIT_BUREAU_HOUR": "申请前一小时征信查询次数",
    "AMT_REQ_CREDIT_BUREAU_DAY": "申请前一天征信查询次数（不含前一小时）",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "申请前一周征信查询次数（不含前一天）",
    "AMT_REQ_CREDIT_BUREAU_MON": "申请前一个月征信查询次数（不含前一周）",
    "AMT_REQ_CREDIT_BUREAU_QRT": "申请前三个月征信查询次数（不含前一个月）",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "申请前一年征信查询次数（官方说明原文有歧义）",
}
_BUILDING_NAMES = {
    "APARTMENTS": "建筑公寓信息", "BASEMENTAREA": "地下室面积",
    "YEARS_BEGINEXPLUATATION": "建筑开始使用年限", "YEARS_BUILD": "建筑建成年限",
    "COMMONAREA": "公共区域面积", "ELEVATORS": "电梯数量", "ENTRANCES": "入口数量",
    "FLOORSMAX": "最高楼层", "FLOORSMIN": "最低楼层", "LANDAREA": "土地面积",
    "LIVINGAPARTMENTS": "居住公寓信息", "LIVINGAREA": "居住面积",
    "NONLIVINGAPARTMENTS": "非居住公寓信息", "NONLIVINGAREA": "非居住面积",
}
for _base, _label in _BUILDING_NAMES.items():
    for _suffix, _statistic in (("AVG", "均值"), ("MODE", "众数"), ("MEDI", "中位数")):
        DISPLAY_NAMES[f"{_base}_{_suffix}"] = f"{_label}归一化{_statistic}"
for _number in range(2, 22):
    DISPLAY_NAMES[f"FLAG_DOCUMENT_{_number}"] = f"是否提供文件 {_number}"


@dataclass(frozen=True)
class HomeCreditConfig:
    """控制公开案例的稳定散列与完整自然日分区目标。

    Parameters
    ----------
    seed : int
        固定散列 seed，默认 20261001；不是随机数发生器状态。
    target_group_rows : int
        1val、2test、3oot 各组的目标人数，默认 30000；测试可显式缩小。

    Examples
    --------
    >>> HomeCreditConfig().target_group_rows
    30000
    """

    seed: int = 20261001
    target_group_rows: int = 30000


@dataclass
class HomeCreditInputs:
    """供多个案例复用的准备结果，不保存或发布真实逐行数据。

    Parameters
    ----------
    df : pl.DataFrame
        原特征、原标签以及 APPLY_DATE、dataset_flag；未公开标签已转 null。
    fcols : list[str]
        除 ID、目标、日期、分区和贷款金额角色以外的完整候选字段。
    feature_metadata : dict[str, dict[str, Any]]
        application 字段的官方定义、中文显示名、来源与有依据的单位。
    business_context : dict[str, Any]
        真实数据、标签可用性、模拟日期、分区用途与金额含义。
    summary : dict[str, Any]
        输入指纹和聚合核验；不含样本 ID、成员清单或本机绝对路径。

    Examples
    --------
    >>> prepared = prepare_home_credit_data()  # doctest: +SKIP
    >>> prepared.benchmark_df  # doctest: +SKIP
    """

    df: pl.DataFrame
    fcols: list[str]
    feature_metadata: dict[str, dict[str, Any]]
    business_context: dict[str, Any]
    summary: dict[str, Any]

    @cached_property
    def benchmark_df(self) -> pl.DataFrame:
        """返回唯一参考分布与拟合分区 0train。

        Returns
        -------
        pl.DataFrame
            dataset_flag 为 0train 的样本；每个准备对象仅物化一次，调用方可按任务投影字段。

        Examples
        --------
        >>> prepared.benchmark_df.height  # doctest: +SKIP
        """
        return self.df.filter(pl.col("dataset_flag") == "0train")


def _sha256_file(path: Path) -> str:
    """分块计算输入指纹，不把原始文件读入额外宽表。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_application(path: Path, *, labeled: bool) -> pl.DataFrame:
    """先核对原始 CSV 头部，再读取明确整数 ID/标签；不补造业务列。"""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f"{path.name}: 空 CSV 或重复列名。")
    required = {"SK_ID_CURR", "AMT_CREDIT"}
    if labeled:
        required.add("TARGET")
    if not required.issubset(header):
        raise ValueError(f"{path.name}: 缺少必要角色字段 {sorted(required - set(header))}。")
    if not labeled and "TARGET" in header:
        raise ValueError(f"{path.name}: application_test 不应提供 TARGET。")
    reserved = {"APPLY_DATE", "dataset_flag", "__home_credit_source"}.intersection(header)
    if reserved:
        raise ValueError(f"{path.name}: 原始输入包含保留字段 {sorted(reserved)}。")
    overrides: dict[str, pl.DataType] = {"SK_ID_CURR": pl.Int64}
    if labeled:
        overrides["TARGET"] = pl.Int8
    try:
        result = pl.read_csv(path, schema_overrides=overrides, infer_schema_length=None)
    except pl.exceptions.PolarsError as exc:
        raise ValueError(f"{path.name}: CSV 字段类型或格式不符合原始输入契约。") from exc
    if result.height == 0:
        raise ValueError(f"{path.name}: 输入没有样本。")
    ids = result.get_column("SK_ID_CURR")
    if ids.null_count() or ids.n_unique() != result.height:
        raise ValueError(f"{path.name}: SK_ID_CURR 必须非空且唯一。")
    if labeled:
        labels = result.get_column("TARGET")
        if labels.null_count() or not set(labels.unique().to_list()).issubset({0, 1}):
            raise ValueError(f"{path.name}: TARGET 必须全部为原始合法 0/1。")
    return result


def _stable_dates(ids: pl.Series, *, source: str, seed: int) -> pl.Series:
    """仅用来源决定区间，用 UTF-8 的 seed/整数 ID 的 SHA256 全摘要决定偏移。"""
    start, end = (TRAIN_START, TRAIN_END) if source == "train" else (TEST_START, TEST_END)
    days = (end - start).days + 1
    values: list[date] = [
        start + timedelta(days=int.from_bytes(
            hashlib.sha256(f"{seed}:{int(identifier)}".encode("utf-8")).digest(), "big"  # noqa: UP012
        ) % days)
        for identifier in ids
    ]
    return pl.Series("APPLY_DATE", values, dtype=pl.Date)


def _partition_labeled_days(daily: list[tuple[date, int]], target_rows: int) -> dict[date, str]:
    """倒序选最接近目标人数的完整日期段；误差相同选更早起日，保留前组非空。"""
    if target_rows <= 0:
        raise ValueError("target_group_rows 必须为正整数。")
    if len(daily) < 4 or any(count <= 0 for _, count in daily):
        raise ValueError("有标签样本不足四个非空自然日，不能划分四个非空分区。")
    dates = [day for day, _ in daily]
    if dates != sorted(set(dates)):
        raise ValueError("每日人数输入必须按不重复日期升序排列。")
    end = len(daily)
    assignments: dict[date, str] = {}
    for group, remaining_groups in (("3oot", 3), ("2test", 2), ("1val", 1)):
        suffix = 0
        candidates: list[tuple[int, int]] = []
        for start in range(end - 1, remaining_groups - 1, -1):
            suffix += daily[start][1]
            candidates.append((abs(suffix - target_rows), start))
        _, start = min(candidates)
        assignments.update({day: group for day, _ in daily[start:end]})
        end = start
    assignments.update({day: "0train" for day, _ in daily[:end]})
    return assignments


def _description_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """读取实际字段说明；原文件包含 cp1252 的省略号，记录真实解码方式。"""
    raw = path.read_bytes()
    try:
        decoded = raw.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        decoded = raw.decode("cp1252")
        encoding = "cp1252"
    reader = csv.DictReader(io.StringIO(decoded))
    if not {"Table", "Row", "Description", "Special"}.issubset(reader.fieldnames or []):
        raise ValueError(f"{path.name}: 字段说明必须含 Table/Row/Description/Special。")
    accepted = {"application_{train|test}.csv", "application_train.csv", "application_test.csv"}
    all_rows: list[dict[str, str]] = [dict(row) for row in reader]
    rows = [row for row in all_rows if row.get("Table", "").strip() in accepted]
    if not rows:
        raise ValueError(f"{path.name}: 未找到 application_train/application_test 字段说明。")
    return rows, {
        "encoding": encoding,
        "rows": len(all_rows),
        "columns": len(reader.fieldnames or []),
        "column_names": reader.fieldnames,
        "tables": sorted({row["Table"] for row in all_rows}),
        "application_rows": len(rows),
    }


def _unit(field: str, description: str, special: str) -> str:
    """只根据官方定义提供单位；金额不推断币种，未明确单位保持未公开。"""
    if field.startswith("AMT_") and not field.startswith("AMT_REQ_"):
        return "原始金额单位（币种未公开）"
    if special.strip() == "normalized":
        return "归一化值（无量纲）"
    if "days" in description.lower() and field.startswith("DAYS_"):
        return "天（相对申请日，保留原始符号）"
    if "number" in description.lower() or field.startswith(("OBS_", "DEF_", "CNT_")):
        return "次数或人数（按官方定义）"
    return "未公开"


def _build_metadata(
    columns: list[str], description_rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """合并相同官方说明，冲突保留原定义并标记未知，不混入其他子表字段。"""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in description_rows:
        grouped.setdefault(row["Row"].strip(), []).append(row)
    metadata: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    conflicts: list[str] = []
    duplicate_rows = 0
    field_definitions: dict[str, dict[str, Any]] = {}
    for field in columns:
        rows = grouped.get(field, [])
        definitions = sorted({row["Description"].strip() for row in rows if row["Description"].strip()})
        duplicate_rows += max(0, len(rows) - 1)
        if not definitions:
            missing.append(field)
            description, status = "未知：未找到官方 application 字段说明。", "unknown"
        elif len(definitions) > 1:
            conflicts.append(field)
            description, status = "未知：官方 application 字段说明存在冲突。", "conflict"
        else:
            description, status = definitions[0], "documented"
        specials = sorted({row["Special"].strip() for row in rows if row["Special"].strip()})
        metadata[field] = {
            "display_name": DISPLAY_NAMES.get(field, f"未提供中文释义（{field}）"),
            "description": description,
            "data_source": "Home Credit 2018 application_{train|test}.csv",
            "unit": _unit(field, description, specials[0] if len(specials) == 1 else ""),
        }
        field_definitions[field] = {
            "definition_status": status, "official_definitions": definitions, "official_special": specials,
        }
    metadata["APPLY_DATE"] = {
        "display_name": "模拟申请日期", "description": "功能演示日期，按固定 SHA256 配置分配；非真实申请或表现日期。",
        "data_source": PREPARATION_VERSION, "unit": "日期",
    }
    metadata["dataset_flag"] = {
        "display_name": "演示分区", "description": "有标签样本按完整模拟自然日切四组；原始 application_test 全部为 4_oot_undue。",
        "data_source": PREPARATION_VERSION, "unit": "组别",
    }
    return metadata, {
        "missing_definitions": missing, "conflicting_definitions": conflicts,
        "duplicate_application_rows_merged": duplicate_rows,
        "application_fields_documented": len(columns) - len(missing) - len(conflicts),
        "field_definitions": field_definitions,
    }


def _partition_stats(df: pl.DataFrame) -> list[dict[str, Any]]:
    """按真实数据记录分母、观测标签与组别日期；无标签坏率保持 null。"""
    grouped = (
        df.group_by("dataset_flag")
        .agg(
            pl.len().alias("total_count"),
            pl.col("APPLY_DATE").min().alias("date_min"),
            pl.col("APPLY_DATE").max().alias("date_max"),
            pl.col("TARGET").count().alias("observed_count"),
            pl.col("TARGET").null_count().alias("null_count"),
            (pl.col("TARGET") == 0).sum().alias("target_0_count"),
            (pl.col("TARGET") == 1).sum().alias("target_1_count"),
            pl.col("TARGET").mean().alias("observed_bad_rate"),
        )
        .sort("dataset_flag")
    )
    rows: list[dict[str, Any]] = grouped.to_dicts()
    for row in rows:
        row["date_min"] = row["date_min"].isoformat()
        row["date_max"] = row["date_max"].isoformat()
        row["label_coverage"] = row["observed_count"] / row["total_count"]
        row["label_status"] = "observed" if row["observed_count"] else "unobserved"
    return rows


def prepare_home_credit_data(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    train_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
    columns_description_csv: str | Path | None = None,
    config: HomeCreditConfig | None = None,
    require_full_dataset: bool = True,
) -> HomeCreditInputs:
    """读取两个 application 文件，生成稳定模拟日期和五分区，保留全部原始样本。

    Parameters
    ----------
    data_dir : str | Path
        三个原始 CSV 所在目录；默认仓库相对路径，缺输入明确失败。
    train_csv : str | Path | None
        覆盖 application_train.csv 路径，不依赖固定本机路径。
    test_csv : str | Path | None
        覆盖 application_test.csv 路径；该文件必须没有 TARGET。
    columns_description_csv : str | Path | None
        覆盖 HomeCredit_columns_description.csv 字段说明路径。
    config : HomeCreditConfig | None
        固定 seed 与完整自然日目标人数，None 使用默认配置。
    require_full_dataset : bool
        默认核对官方 307511/48744 行和 122/121 列；仅显式测试 fixture 可关闭。

    Returns
    -------
    HomeCreditInputs
        已清理 null 标签的共享输入、完整候选字段、元数据与聚合核验。

    Raises
    ------
    FileNotFoundError
        必需 CSV 缺失；不回退合成数据或下载原始文件。
    ValueError
        schema、标签、ID、完整规模或完整自然日分区无法满足输入契约。

    Examples
    --------
    >>> prepared = prepare_home_credit_data("my-local-home-credit")  # doctest: +SKIP
    >>> prepared.df.select("TARGET", "APPLY_DATE", "dataset_flag")  # doctest: +SKIP

    Notes
    -----
    散列文本固定为 UTF-8 的 ``{seed}:{十进制整数 SK_ID_CURR}``，使用 SHA256 全部
    32 字节按 big-endian 转整数，再对区间天数取模。0/1 标签值不参与日期和分区。
    """
    settings = config if config is not None else HomeCreditConfig()
    if settings.target_group_rows <= 0:
        raise ValueError("target_group_rows 必须为正整数。")
    base = Path(data_dir)
    paths: dict[str, Path] = {
        "train": Path(train_csv) if train_csv is not None else base / "application_train.csv",
        "test": Path(test_csv) if test_csv is not None else base / "application_test.csv",
        "description": Path(columns_description_csv) if columns_description_csv is not None else base / "HomeCredit_columns_description.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"必需 Home Credit 输入缺失：{path}")
    fingerprints = {kind: _sha256_file(path) for kind, path in paths.items()}
    train = _read_application(paths["train"], labeled=True)
    test = _read_application(paths["test"], labeled=False)
    original_columns = train.columns
    original_shapes = {"train": train.shape, "test": test.shape}
    expected_features = set(original_columns) - {"TARGET"}
    if set(test.columns) != expected_features:
        raise ValueError(
            "train/test 除 TARGET 外字段集合不一致："
            f"test 缺少 {sorted(expected_features - set(test.columns))}；"
            f"test 多出 {sorted(set(test.columns) - expected_features)}。"
        )
    if require_full_dataset:
        if train.shape != (307511, 122) or test.shape != (48744, 121):
            raise ValueError(f"正式输入规模不符：train={train.shape}，test={test.shape}。")
        if set(original_columns) != set(DISPLAY_NAMES):
            raise ValueError("正式 application 字段与 Home Credit 2018 schema 不一致。")
    overlap = train.select("SK_ID_CURR").join(test.select("SK_ID_CURR"), on="SK_ID_CURR", how="inner")
    if overlap.height:
        raise ValueError("application_train/application_test 的 SK_ID_CURR 交集必须为空。")

    # 先对齐原始列序与必要角色类型；vertical_relaxed 仅统一对应数值类型。
    train = train.with_columns(_stable_dates(train["SK_ID_CURR"], source="train", seed=settings.seed))
    test = (
        test.with_columns(pl.lit(-1, dtype=pl.Int8).alias("TARGET"))
        .select(original_columns)
        .with_columns(_stable_dates(test["SK_ID_CURR"], source="test", seed=settings.seed))
    )
    incompatible: list[str] = [
        field for field in original_columns
        if (train.schema[field] == pl.String) != (test.schema[field] == pl.String)
        and train.schema[field] != pl.Null and test.schema[field] != pl.Null
    ]
    if incompatible:
        raise ValueError(f"train/test 字段类型冲突，不能将全部数值转字符串：{incompatible}。")
    daily_rows = (
        train.select("SK_ID_CURR", "APPLY_DATE")
        .sort(["APPLY_DATE", "SK_ID_CURR"])
        .group_by("APPLY_DATE").len().sort("APPLY_DATE")
        .iter_rows()
    )
    daily: list[tuple[date, int]] = [(day, count) for day, count in daily_rows]
    assignments = _partition_labeled_days(daily, settings.target_group_rows)
    day_groups = pl.DataFrame({"APPLY_DATE": list(assignments), "dataset_flag": list(assignments.values())})
    train = train.join(day_groups, on="APPLY_DATE", how="left")
    test = test.with_columns(pl.lit("4_oot_undue").alias("dataset_flag"))
    df = (
        pl.concat([train, test], how="vertical_relaxed")
        .with_columns(pl.when(pl.col("TARGET") == -1).then(None).otherwise(pl.col("TARGET")).cast(pl.Int8).alias("TARGET"))
        .sort(["APPLY_DATE", "SK_ID_CURR"])
    )
    fcols = [field for field in original_columns if field not in ROLE_COLUMNS]
    description_rows, description_summary = _description_rows(paths["description"])
    metadata, metadata_summary = _build_metadata(original_columns, description_rows)
    partition_stats = _partition_stats(df)

    # 元数据分别登记真实特征、原始标签、演示来源分组和模拟日期，避免单一 simulated 标志。
    context: dict[str, Any] = {
        "dataset_id": "home-credit-default-risk-2018",
        "source": "https://www.kaggle.com/competitions/home-credit-default-risk/data",
        "real_features": {"source": "application_train.csv/application_test.csv", "simulated": False, "derived_features_added": False},
        "target": {"field": "TARGET", "definition": metadata["TARGET"]["description"], "simulated": False,
                   "available_in": "application_train.csv", "unavailable_in": "application_test.csv",
                   "unobserved_value": None, "performance_window": "X/Y 未公开；不推断 bad30/bad60"},
        "date": {"field": "APPLY_DATE", "simulated": True, "train_range": [TRAIN_START.isoformat(), TRAIN_END.isoformat()],
                 "test_range": [TEST_START.isoformat(), TEST_END.isoformat()],
                 "limitation": "功能演示日期，不支持真实跨期泛化、季节性或线上漂移结论。"},
        "partitions": {"field": "dataset_flag", "simulated": True,
                       "0train": "拟合、候选发现、开发筛选与 PSI 参考分布",
                       "1val": "验证或筛选；参与选择后不是最终独立评估",
                       "2test": "有标签独立评估，不等同于原始 application_test",
                       "3oot": "模拟时间外独立评估，不是真实业务 OOT",
                       "4_oot_undue": "原始比赛测试表，标签未公开；名称不代表真实贷款未到期"},
        "amount": {"field": "AMT_CREDIT", "unit": "原始金额单位", "currency": "未公开",
                   "meaning": "贷款金额，不等于损失或利润", "included_as_feature": False},
        "weights": None,
        "sample_scope": "两个 application 文件的全部样本" if require_full_dataset else "显式小型测试 fixture",
        "preparation_version": PREPARATION_VERSION,
    }
    special_values: dict[str, Any] = {}
    if "DAYS_EMPLOYED" in df.columns:
        special_values["DAYS_EMPLOYED"] = {
            "value": 365243,
            "count": df.filter(pl.col("DAYS_EMPLOYED") == 365243).height,
            "handling": "保留原值；该异常正值作为本字段的特殊值配置，不应用于其他字段。",
            "meaning": "官方字段说明未解释此值，不推断实际就业时长。",
        }
    context["special_values"] = special_values
    context["labels"] = {"TARGET": dict(context["target"])}
    file_summary: list[dict[str, Any]] = []
    for kind, path in paths.items():
        if _sha256_file(path) != fingerprints[kind]:
            raise ValueError(f"{path.name}: 准备期间输入指纹变化，停止以避免混批。")
        original_shape = original_shapes.get(kind)
        file_summary.append({
            "role": kind, "name": path.name, "sha256": fingerprints[kind], "bytes": path.stat().st_size,
            "rows": original_shape[0] if original_shape else description_summary["rows"],
            "columns": original_shape[1] if original_shape else description_summary["columns"],
        })
    summary: dict[str, Any] = {
        "preparation_version": PREPARATION_VERSION,
        "input_files": file_summary,
        "description_encoding": description_summary["encoding"],
        "columns_description": description_summary,
        "full_dataset": require_full_dataset,
        "seed": settings.seed,
        "hash_config": {"algorithm": "SHA256", "encoding": "UTF-8", "id_format": "十进制整数，无前导零",
                        "text_format": "{seed}:{SK_ID_CURR}", "integer_conversion": "全部 32 字节 big-endian",
                        "offset_rule": "散列整数 modulo 区间天数（train 181/test 31）",
                        "uses_binary_target_value": False},
        "partition_config": {"target_group_rows": settings.target_group_rows,
                             "algorithm": "从末日向前选完整自然日人数最接近目标的日期段，依次 3oot/2test/1val",
                             "tie_break": "人数误差相同选择更早起始日；为剩余分区保留非空自然日"},
        "merged_shape_before_roles": [df.height, len(original_columns)],
        "final_shape": list(df.shape), "rows": df.height,
        "partition_stats": partition_stats,
        "fcols": fcols, "candidate_feature_count": len(fcols),
        "display_features": [field for field in DEFAULT_DISPLAY_FEATURES if field in fcols],
        "metadata": metadata_summary, "special_values": special_values,
        "invariants": {"source_ids_unique": True, "source_ids_disjoint": True, "all_rows_preserved": True,
                       "test_target_all_null": df.filter(pl.col("dataset_flag") == "4_oot_undue")["TARGET"].null_count() == test.height,
                       "whole_natural_days": True, "raw_files_unchanged": True},
    }
    return HomeCreditInputs(df, fcols, metadata, context, summary)


def main() -> None:
    """从显式本地 CSV 准备共享输入并打印无真实逐行数据的聚合摘要。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--columns-description-csv", type=Path)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--target-group-rows", type=int, default=30000)
    parser.add_argument("--fixture-inputs", action="store_true", help="只用于显式小型测试 CSV，不能生成正式公开资产")
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()
    prepared = prepare_home_credit_data(
        args.data_dir, train_csv=args.train_csv, test_csv=args.test_csv,
        columns_description_csv=args.columns_description_csv,
        config=HomeCreditConfig(args.seed, args.target_group_rows),
        require_full_dataset=not args.fixture_inputs,
    )
    content = json.dumps(prepared.summary, ensure_ascii=False, indent=2) + "\n"
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
