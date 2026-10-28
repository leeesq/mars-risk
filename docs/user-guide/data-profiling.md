---
description: 使用 MarsDataProfiler 检查宽表的数据质量、分布和分组漂移。
---

# 数据画像

先看[案例 1：这份数据能直接用吗？](../demos/data-quality.md)，再回本页核对参数。

## 适用场景

分箱或建模前，先用数据画像检查字段是否可用、缺失和特殊值是否异常。
再比较不同时间或业务分组的分布，判断是否发生漂移。

## 前置条件

- 输入是 Pandas 或 Polars DataFrame。
- `features` 只包含需要画像的业务特征，不包含 target、分组和日期列。
- 业务缺失码和特殊值需要显式配置，例如 `-999`。

## 完整调用

```python
--8<-- "docs/snippets/data_profiling.py"
```

`MarsDataProfiler` 的构造函数保存缺失值处理等可复用策略。每次调用 `generate_profile()` 时，
传入本次数据、特征、分组、指标，以及可选的 PSI 基准样本。
画像器不会随机抽样；数据量较大、需要抽样时，由调用方在传入前完成。

多 chunk Polars 输入可直接调用，无需预先全表 `rechunk()`。不适用的字符串 mean/std 等指标
返回 typed null，并通过 `calculation_status` 说明跳过原因；非数值 zeros 保留原有零值语义。
PSI 使用独立的分布计算流程。既无分组/时间参考、也无 benchmark 时，仍保留 PSI 表和特征身份，
值为 Float64 null，状态为 `not_computed/no_group_or_benchmark`，不会填为零。
真实空分组与文字 `null` 使用不同显示身份。纯 PSI 请求的 overview 仍包含输入特征身份。

## 输出

| 字段 | 用途 |
| --- | --- |
| `overview_table` | 特征级质量和统计汇总 |
| `dq_tables` | 缺失、零值等数据质量趋势 |
| `stats_tables` | 均值、分位数等统计趋势 |
| `comparison_tables` | 显式请求的 schema drift 与 unseen rate |
| `report_meta` | 版本、UTC 生成时间、数据规模、运行配置和诊断 |
| `get_profile_data()` | 返回 overview、DQ、统计趋势和 comparisons 四个字段 |

`ProfileData` 包含四个字段；从旧版三元素位置解包迁移时，需要增加 `comparisons`。
调用 `write_excel()` 可导出报告，其中包含 Metadata 和 comparison 工作表；
调用 `write_html()` 可生成自包含的单个 HTML 文件，支持页面切换、全局搜索和表格排序。

## 分组与日期

已有月份、渠道或客群列时传 `group_col`。只有原始日期时，传 `time_col` 与 `time_grain` 生成分组。
同时存在时，`group_col` 决定面板分组，`time_col` 保留日期语义。

## 分客群画像

同时比较多个客群时，在运行方法传入 `segment_col`，保留原有 `group_col` 或时间分组。
`profile_stats()` 和 `MarsDataProfiler.generate_profile()` 都支持此参数。
客群列自动从默认特征中排除。显式声明它为特征、分组、日期或类别特征时会报角色冲突。

```python
import polars as pl
from mars.analysis import MarsDataProfiler, profile_stats

df = pl.DataFrame({
    "customer_segment": ["seg1"] * 4 + ["seg2"] * 4 + ["seg3"] * 4,
    "month": ["2026-01", "2026-01", "2026-02", "2026-02"] * 3,
    "income": [100.0, 110.0, 120.0, 130.0, 200.0, 220.0, 210.0, 230.0,
               300.0, 310.0, 340.0, 350.0],
})
benchmark = df.filter(pl.col("month") == "2026-01")
shared = profile_stats(
    df,
    metrics=["missing", "mean", "psi"],
    features=["income"],
    segment_col="customer_segment",
    group_col="month",
    benchmark_df=benchmark,
    binning_scope="shared",
)
independent = MarsDataProfiler(binning_scope="per_segment").generate_profile(
    df,
    metrics=["missing", "mean", "psi"],
    features=["income"],
    segment_col="customer_segment",
    group_col="month",
    benchmark_df=benchmark,
    enable_sparkline=False,
)
seg2_mean = shared.get_table(
    "stats.mean", segments=["seg2"], include_all=False, features=["income"]
)
```

`binning_scope` 是画像器的构造策略；便捷函数显式转交同名参数。
`shared` 在既定整体拟合人口学习一套 PSI 规则；`per_segment` 为 All 和各客群分别拟合。
切点独立时，同一箱号可能表示不同区间。均值、缺失率等统计不随这项策略变化。
`per_segment` 必须同时指定 `segment_col` 并请求 `psi`，否则入口报错。

开启客群后，各事实表增加 `scope_id`，`get_table("scopes")` 返回身份目录。
All 使用完整当前人口；每个 scope 内的 `total` 使用该人口跨原分组重新聚合，
不平均月份或客群的统计。缺失客群保留，真实字符串 `All`、`Total`、`Missing`、
`null`、空字符串和缺失值各有独立身份。查询使用业务原值，跨 scope 查询保留 `scope_id`。
未传 `segment_col` 时，原有表结构和默认调用保持不变。

## PSI 口径

画像 PSI 会先对数值或类别特征分箱，再比较各分组的分布。缺失率通常单独观察，因此默认不把缺失
箱和特殊值箱纳入 PSI。需要复现其他口径时显式设置 `psi_include_missing` 和
`psi_include_special`。

未传 `benchmark_df` 时，画像器在当前数据上拟合分箱，并以最小分组作为 expected distribution。
传入 `benchmark_df` 后，画像器会用全部基准样本拟合分箱，并以基准样本的整体分布作为当前
各分组和 `total` 的参考分布。基准表无需包含分组列或日期列；基准样本不会计入
overview、缺失率、均值或其他当前数据统计。

分客群时，显式 benchmark 必须包含同名客群列，并保留可无歧义匹配的类型。
All 的 expected 来自整体 benchmark；seg2 的 expected 只来自 benchmark 中的 seg2。
共享切点不会使各客群共用整体 expected。PSI 的分布人口包含当前 scope 的适用完整样本，
缺失箱和特殊值箱仍由原有开关决定。

无 benchmark 时，先按原分组类型选出整体最小组，再把该组固定应用到各客群。
客群缺少该组时记录 `not_computed/missing_first_group`，不会改用它自己的下一期。
显式 benchmark 缺少某个客群时，PSI 为 Float64 null，并记录
`not_computed/missing_segment_benchmark`；其他统计继续计算，其他客群正常执行。
`report_meta["scope_references"]` 记录各 scope 的规则拟合行数、expected 行数和实际参考组。

```python
missing_reference = profile_stats(
    df,
    metrics=["mean", "psi"],
    features=["income"],
    segment_col="customer_segment",
    group_col="month",
    benchmark_df=benchmark.filter(pl.col("customer_segment") != "seg2"),
)
seg2_status = missing_reference.get_table(
    "calculation_status", segments=["seg2"], include_all=False,
    filters={"metric": "psi"},
)
assert seg2_status["reason"].to_list() == ["missing_segment_benchmark"]
```

```python
benchmark_df = df.filter(pl.col("month") == "2026-01").drop("month")
current_df = df.filter(pl.col("month") == "2026-02")

trend_report = profiler.generate_profile(
    current_df,
    benchmark_df=benchmark_df,
    features=["income", "utilization", "segment"],
    group_col="month",
    metrics=["missing", "mean", "psi"],
)
```

如果当前数据没有分组列，仍可直接比较两张表的整体分布；此时 PSI 表只包含 `feature`、`dtype`
和 `total`，不会生成分组稳定性统计列。

```python
total_report = profiler.generate_profile(
    current_df.drop("month"),
    benchmark_df=benchmark_df,
    features=["income", "utilization", "segment"],
    metrics=["psi"],
)
total_psi = total_report.stats_tables["psi"]
```

数值特征可通过 `psi_merge_small_bins` 和 `psi_min_bin_size` 控制小箱合并；类别特征通过
`psi_n_bins` 控制保留的 Top-K 类别，其余类别进入 Other 箱。

## Schema drift 与 unseen rate

`schema` 和 `unseen` 不在默认指标中，使用时必须显式请求并传入 `benchmark_df`。
省略 `features` 时，schema 表会检查当前表和基准表的全部业务列，并区分 `matched`、
`compatible_change`、`incompatible_change`、`current_only` 和 `benchmark_only`。

```python
comparison_report = profiler.generate_profile(
    current_df,
    benchmark_df=benchmark_df,
    metrics=["schema", "unseen"],
    categorical_features=["integer_encoded_segment"],
    group_col="month",
)
schema_drift = comparison_report.comparison_tables["schema"]
unseen_rate = comparison_report.comparison_tables["unseen"]
schema_view = comparison_report.show_trend("schema")
unseen_view = comparison_report.show_trend("unseen", sort_by="feature", sort_ascending=True)
schema_html = schema_view.to_html()
unseen_html = unseen_view.to_html()
```

默认按表内 `total` 排序；没有 `total` 的 schema 表按 `feature` 稳定排序。
显式传入合法的 `sort_by` 时，使用指定排序；字段不存在时会报错。
展示时，dtype、status 和 reason 保持文本，只对数值列设置渐变和数值格式。
可空数值列、全空数值列和空查询结果也可以渲染。
完整输入示例见[报告展示与静态导出](reports-and-exports.md)。

`unseen` 自动适用于字符串、Categorical、Enum 和 Boolean；整数编码类别必须放入
`categorical_features`。缺失值、NaN、自定义缺失码和特殊值不进入分子或分母。
没有参考值、没有当前有效值、缺列或 dtype 不兼容时，报告会保留该行并记录状态。

客群画像的 schema 描述两侧原始字段类型，过滤后人口为空也不改成 Null dtype。
`unseen` 只比较对应客群的 benchmark 类别；缺该客群参考时记录 `no_reference_values`，
不会借用整体类别集合。`overview`、DQ、连续统计、PSI、schema、unseen 和计算状态
都带同一 `scope_id`，可保存为 `.marsreport` 后在新进程查询。
报告快照保存统计证据，不恢复画像器、PSI 分箱器或原始宽表。

## 常见失败

- `features` 包含不存在的列：先固定本次特征清单，不要依赖全表自动推断。
- 业务缺失码仍进入均值或分布：在构造 `MarsDataProfiler` 时传入 `missing_values`。
- 期望时间趋势但只提供月份字符串：需要原始日期语义时同时传入 `time_col`。
- benchmark 缺少业务特征或 dtype 不兼容：显式基准采用严格校验，按报错列清单检查并统一两侧的列名和类型。
- 请求 `schema` / `unseen` 却没有 benchmark：这两个指标不会自动选取当前期内部基准。

## 下一步

- 对可用特征计算 IV、KS 和风险趋势：[分箱与风险评估](binning-risk-evaluation.md)。
- 用质量、区分度和稳定性规则压缩特征集合：[特征筛选](feature-selection.md)。
- 查询全部参数：[Analysis API](../reference/analysis.md)。
