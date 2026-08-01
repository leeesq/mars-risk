---
description: 使用 MarsDataProfiler 检查宽表的数据质量、分布和分组漂移。
---

# 数据画像

## 适用场景

在分箱或建模前使用数据画像回答以下问题：字段是否可用，缺失和特殊值是否异常，分布是否稳定，
不同时间或业务分组之间是否发生漂移。

## 前置条件

- 输入是 Pandas 或 Polars DataFrame。
- `features` 只包含需要画像的业务特征，不包含 target、分组和日期列。
- 业务缺失码和特殊值需要显式配置，例如 `-999`。

## 完整调用

```python
--8<-- "docs/snippets/data_profiling.py"
```

`MarsDataProfiler` 的构造函数保存可复用的缺失值等策略；`generate_profile()` 接收本次数据、特征、
分组、指标和可选的 PSI 基准样本。数据量较大时，由调用方在传入前显式抽样，画像器不会在内部
随机抽样。

## 输出

| 字段 | 用途 |
| --- | --- |
| `overview_table` | 特征级质量和统计汇总 |
| `dq_tables` | 缺失、零值等数据质量趋势 |
| `stats_tables` | 均值、分位数等统计趋势 |
| `comparison_tables` | 显式请求的 schema drift 与 unseen rate |
| `report_meta` | 版本、UTC 生成时间、数据规模、运行配置和诊断 |
| `get_profile_data()` | 返回 overview、DQ、统计趋势和 comparisons 四个字段 |

`ProfileData` 已由三字段扩展为四字段；旧的三元素位置解包需要增加
`comparisons`。报告可直接使用 `write_excel()` 导出 Metadata 和 comparison 工作表，
或使用 `write_html()` 生成带页面切换、全局搜索和表格排序的自包含单文件。

## 分组与日期

已有月份、渠道或客群列时传 `group_col`。只有原始日期时，传 `time_col` 与 `time_grain` 生成分组。
同时存在时，`group_col` 决定面板分组，`time_col` 保留日期语义。

## PSI 口径

画像 PSI 会先对数值或类别特征分箱，再比较各分组的分布。缺失率通常单独观察，因此默认不把缺失
箱和特殊值箱纳入 PSI。需要复现其他口径时显式设置 `psi_include_missing` 和
`psi_include_special`。

未传 `benchmark_df` 时，画像器在当前数据上拟合分箱，并以最小分组作为 expected distribution。
传入 `benchmark_df` 后，画像器改为在全量 benchmark 上拟合分箱，并用其全量分布作为所有当前
分组和 `total` 的 expected distribution。benchmark 不需要包含分组列或日期列，也不会进入
overview、缺失率、均值或其他当前数据统计。

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

`schema` 和 `unseen` 不在默认指标中，必须显式请求并传入 `benchmark_df`。未传
`features` 时，schema 表覆盖 current 与 benchmark 业务列并集，并区分 `matched`、
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
```

`unseen` 自动适用于字符串、Categorical、Enum 和 Boolean；整数编码类别必须放入
`categorical_features`。缺失值、NaN、自定义缺失码和特殊值不进入分子或分母。
无参考值、无当前有效值、缺列或 dtype 不兼容会以明确状态留在表中，不会被静默删除。

## 常见失败

- `features` 包含不存在的列：先固定本次特征清单，不要依赖全表自动推断。
- 业务缺失码仍进入均值或分布：在构造 `MarsDataProfiler` 时传入 `missing_values`。
- 期望时间趋势但只提供月份字符串：需要原始日期语义时同时传入 `time_col`。
- benchmark 缺少业务特征或 dtype 不兼容：显式基准采用严格校验，按报错列清单对齐两侧 schema。
- 请求 `schema` / `unseen` 却没有 benchmark：这两个指标不会自动选取当前期内部基准。

## 下一步

- 对可用特征计算 IV、KS 和风险趋势：[分箱与风险评估](binning-risk-evaluation.md)。
- 用质量、区分度和稳定性规则压缩特征集合：[特征筛选](feature-selection.md)。
- 查询全部参数：[Analysis API](../reference/analysis.md)。
