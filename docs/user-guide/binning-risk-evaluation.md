---
description: 自动建箱、复用基准期规则并读取 IV、KS、AUC、Lift、PSI 和趋势结果。
---

# 分箱与风险评估

先看[案例 2：区分度与稳定性](../demos/binning-stability.md)，再回本页核对分箱与评估参数。

## 适用场景

本指南说明如何评估单个特征与二分类 target 的关系，以及如何用固定的基准期分箱规则评估当前数据。
你可以读取 IV、KS、AUC、Lift、坏账率、PSI、缺失率和分箱趋势。

## 入口选择

| 目标 | 入口 |
| --- | --- |
| 按高层参数自动构建分箱器 | `profile_risk()` |
| 传入或复用已拟合分箱器 | `MarsBinEvaluator.evaluate()` |
| 只需要分箱和转换 | `MarsNativeBinner`、`MarsLiteOptBinner`、`MarsOptimalBinner` |

`profile_risk()` 不接受显式 `binner`。需要固定规则时使用 evaluator。

## 基准期规则评估当前期

下面的完整示例使用带标签的 `baseline_df` 拟合 CART 分箱，并评估 target 尚未表现的
`current_df`：

```python
--8<-- "docs/snippets/baseline_evaluation.py"
```

评估器优先使用显式传入的 `binner`，其次用 `benchmark_df` 拟合规则，最后才用当前 `df`。
基准样本不计入当前期 Total，但会提供 PSI 的参考分布（expected distribution）。

## 分箱器选择

| 类型 | 适用情况 | 标签要求 |
| --- | --- | --- |
| `native` + `quantile` | 快速等频分箱、宽表初筛 | 不要求 |
| `native` + `uniform` | 需要固定宽度区间 | 不要求 |
| `native` + `cart` | 使用 target 的轻量监督分箱 | 要求 |
| `lite_opt` | 轻量单调监督分箱 | 要求 |
| `optimal` | 数学规划最优分箱与类别合并 | 要求 |

三个分箱器都继承 `MarsBinnerBase`，共享 `transform()`、`profile_bin_performance()`、
`get_fit_report()`、JSON artifact 和 `prune()`。

`transform()` 默认要求输入包含全部可用规则列。只转换子集时显式传
`features=[...]`；确实需要宽松处理时才使用 `on_missing="warn"` 或 `"ignore"`。
`update_bins()` 对未知特征默认报错，`prune()` 和 `get_bin_mapping()` 遇到未知名称时也会报错。

### 重复拟合与失败恢复

每次 `fit()` 都重新学习本次特征的规则、WOE、映射和诊断。连续拟合相同数据与新实例等价；
改变 target、特征集合或特征类型时不会复用上次学到的结果。构造参数保持不变，
本次传入的 `features` 和 `cat_features` 决定本次范围。

通过 `update_bins()` 设置的规则属于用户配置。后续拟合仍包含该特征、且类型兼容时，分箱器会继续
应用规则，并重新计算本次 WOE；数值规则不会用于新变成类别类型的特征。
`prune()` 会同时移除所裁剪特征的用户规则。JSON 保存和加载会保留这些规则；
旧 schema 1 artifact 未包含这一可选字段时，仍可加载。

拟合失败后，转换、分箱表现评估、规则查询、SQL 导出和保存都会按未拟合状态报错，不能读取上次成功结果。
调用 `get_fit_report()` 可以查看本次失败诊断；若在参数校验阶段失败，诊断可能为空。
再次成功执行 `fit()` 后，可以恢复使用。
宽表中仍有可用规则时，分箱器会保留局部失败的诊断和成功的回退规则（fallback），允许部分特征成功。

### Boolean、类别与部分标签

Pandas `bool` / nullable `boolean` 与 Polars `Boolean` 都按类别特征处理。原生分箱示例：

```python
import polars as pl
from mars.feature import MarsNativeBinner

features = pl.DataFrame({"flag": [False, False, True, True, None]})
target = pl.Series("target", [0, 0, 1, 1, None])
binner = MarsNativeBinner(n_bins=2).fit(features, target, cat_features=["flag"])
bins = binner.transform(features)
stats = binner.profile_bin_performance(features, target)
assert set(bins["flag_bin"].head(4)) == {0, 1}
assert bins["flag_bin"][-1] == -1
```

Boolean 类别的匹配键保留类型。推理时，整数 `0/1` 或文本 `"True"`、`"false"` 不会转换为布尔值，
未见类别进入 Other。普通文本仍区分大小写，`"True"`、`"true"`、
`"1"`、`"001"` 和字面 `"nan"` 是独立类别。原有非 Boolean 类别的字符串匹配保持不变，
例如明确设为类别的整数 `1` 可以匹配文本 `"1"`，但不会匹配 `"001"` 或浮点表示 `1.0`。
同一列混合多种物理类型时，应在输入前清理；Pandas 转换无法保持类型时会报错。
Null 和数值 NaN 进入 Missing，转换结果保留用户原列，不暴露内部临时列，也不会因临时列覆盖原列。
Native/LiteOpt 会为全空 Boolean 列保留空类别规则，后续非空值进入 Other，null 进入 Missing。
Optimal 在全空输入没有可用规则时会明确失败。

`profile_bin_performance()` 与高层风险评估共享二分类标签校验：观测值允许 `0/1`、
Boolean，及文本 `"0"`、`"1"`、`"true"`、`"false"`、`"True"`、`"False"`。
Null、NaN 和空字符串表示未表现。`-1`、`2`、`0.5` 等非法值会在聚合前报错，
不会参与坏样本求和或改写 WOE。直接调用表现评估接口时，合法单类别可以返回计数和坏账率，
没有任何观测标签则报错。
高层监督分箱仍须满足所选算法的标签要求；标签部分缺失时，高层评估会先筛选拟合样本。
直接调用分箱器的监督拟合接口时，应单独遵守该分箱器的输入要求。

## 分箱诊断与 JSON artifact

`get_fit_report()` 固定返回 Polars 表，包含 `feature`、`dtype`、`feature_type`、
`status`、`usable`、`n_bins` 和 `reason`。宽表拟合可以保留成功 fallback，无规则特征
则标记为 `failed`；全部失败会抛出 `ValueError`。

```python
binner.fit(baseline_df, target, features=features)
fit_report = binner.get_fit_report()
binner.save_json("artifacts/binner.json")
restored = MarsBinnerBase.load_json("artifacts/binner.json")
```

JSON 顶层包含 `artifact_type`、`schema_version`、`binner_type`、`mars_version`、`params`
和 `state`，当前 `schema_version=1`。这是正式跨版本规则格式；旧 `{params, state}` 载荷不兼容，
必须用 0.0.26 重新拟合或导出。Pickle/joblib 只用于 Python 进程级便利存储。

## 输出

`MarsRiskProfile` 保存本次 `report`、`binner`、`targets` 和 `metadata`。常用 report 字段：

| 字段 | 用途 |
| --- | --- |
| `summary_table` | 特征级指标汇总和排序 |
| `detail_table` | 分箱样本数、坏账率、WOE 和 IV 明细 |
| `trend_tables` | PSI、缺失率和坏账率等分组趋势 |
| `missing_by_day_table` | 使用 `time_col` 计算的按日缺失趋势 |

## 多目标与默认特征

`target=["bad30", "bad60"]` 使用首个目标作为分箱参考，只拟合一次；后续目标复用同一组
边界，各自按本目标的有效标签计算风险指标。提供 `benchmark_df` 时，监督分箱使用基准表中的
首目标。没有独立的参考目标选择参数，调整目标列表顺序会改变参考。

省略 `features` 时，高层入口只推断一次，排除全部目标列及声明的 `group_col`、`time_col`、
`weights_col`、`amount_col`；raw KS 和所有目标评估使用同一特征集。显式 `features` 保留现有
选择优先级和校验。目标不能同时声明为分组、时间、权重或金额角色；排除后没有特征会明确报错。

```python
--8<-- "docs/snippets/multitarget_risk.py"
```

`summary_table` 的 `target` 和 `detail_table` 的 `y` 区分各目标；`trend_tables` 仍只展示
首目标。null/NaN 标签各自排除，不用另一个目标的观测状态补全。

## 原始数值 KS

仅 `profile_risk()` 支持以下选项，evaluator、monitor、pipeline 和 Agent 工具不增加同名参数：

```python
profile = profile_risk(
    df,
    target="target",
    features=["income", "utilization"],
    ks_method="raw",           # 默认 "binned"
    max_raw_ks_features=50,
)
```

raw 模式按原始数值排序，将相同取值合并，计算好坏两类经验累计分布的最大绝对差，
再乘以 100。它直接替换数值特征的最终 `ks`；类别特征保留分箱 KS。
默认分箱 KS 按 WOE 排序，原始值 KS 按数值顺序计算。排序依据不同，两者可能明显不同；
选择时应先确定需要回答的业务问题。
`ordered_metric_sort_by` 在 raw 模式仍控制分箱 AUC 等指标，但不影响数值特征的最终 KS。

原始值计算排除未观测标签、Null、NaN、无穷值及 `missing_values`、`special_values`。
这只改变 KS 的样本范围，其他指标和分箱输入要求保持原样；例如无穷值使分箱器无法拟合时，
仍需清理分箱输入或提供可拟合的 `benchmark_df`。基准样本不会混入当前数据的 KS。
传入 `weights_col` 时使用加权累计分布，有效样本的权重必须有限且非负；零权重不贡献分布。
过滤后缺少任一类别的正权重总量时返回空 KS，常数特征且两类有效时返回 0。
原有全局单类别标签校验仍生效。

默认最多允许 50 个数值特征，类别特征不计入。超过限制在分箱和排序前报错，可通过
`features` 缩小范围或显式提高上限。上限必须是正整数；关闭 raw 模式时不执行数量限制。
该限制不保证运行时间，耗时仍受行数、分组和标签数量影响。

Python 汇总表、趋势表、图表和 KS 排序都读取最终值。公共查询、`describe()`、AI 上下文、
保存／加载和 HTML 也遵循源报告的 `ks_method`。
raw 模式会在 `detail_table` 的每个分箱行重复保存最终 `ks`；`ks_bin` 仍保存逐箱分箱 KS，
不能据此恢复原始值 KS。
多标签分别计算，`trend_tables` 沿用仅展示主标签的契约。
`report_meta["ks_source_by_feature"]` 记录各特征使用的方法，
`report_meta["raw_ks_diagnostics"]` 记录数值特征各标签、各组的有效样本数和空值原因。
`valid_count` 包含符合样本条件的零权重行；`no_valid_samples` 表示没有有效样本，
`insufficient_class_weight` 表示至少一类没有正权重。

`MarsBinningReport.write_excel()` 的原生 Excel 无论源报告为 raw 还是 binned，都使用同一
22 列模板并统一展示分箱 KS（0–100 点）。导出明细的 `ks` 按特征、目标和实际分组取
`max(detail.ks_bin)`，广播到同组全部箱行；逐箱 `ks_bin` 保留原值。KS 页和综合指标页
读取该导出列，综合指标仍保留模板的 average 聚合方式。最终汇总取对应 Total 的分箱 KS，
最终 KS 趋势沿用首目标、原分组顺序及身份元数据。全部逐箱 KS 未定义的组保持空值。

导出只读取已计算结果，不重新拟合或评估，也不改写源报告。
Excel 口径说明分别记录源报告的 `ks_method`、导出使用的 binned 口径、`max(detail.ks_bin)` 来源和单位；
其中保留的 raw 诊断属于源报告。
openpyxl 只写入结构和数据，需要在原生 Excel 中刷新透视表；xlwings 可以调用 Excel 刷新和重算。
通用 `ReportSnapshot.write_excel()` 则静态写出保存的原报告数值。

### 本地性能参考

2026-09-10 在 macOS、本地 `mars_env`（Python 3.11）实测。数据为随机种子 20260910 的
10 万行标准正态数值、4 个分组、单个二分类标签，native 等频分箱、`n_jobs=1`，
Polars 使用默认线程数。
每种配置使用三个独立进程，下表取中位数；计时不含导入和造数，RSS 每 5 ms 采样。

| 数值特征数 | 模式 | 耗时（秒） | 峰值 RSS（MiB） | 相对调用前增加的 RSS（MiB） |
| --- | --- | --- | --- | --- |
| 10 | binned | 0.059 | 420.3 | 134.9 |
| 10 | raw | 0.279 | 467.8 | 184.5 |
| 50 | binned | 0.191 | 731.9 | 414.3 |
| 50 | raw | 1.264 | 762.4 | 453.1 |

这些数值用于观察开启功能的增量，不构成性能承诺；机器、数据分布、标签数与分组数都会影响结果。

## 时间与 PSI

风险趋势图必须有有效 `time_col`。`group_col` 决定面板分组，但不能替代真实日期范围。只有未传
`group_col` 时，`time_grain` 才根据 `time_col` 生成分组。

`psi_include_missing` 和 `psi_include_special` 控制对应分箱是否进入 PSI；缺失率会单独报告，
监控场景通常保持两者为 `False`。

## 常见失败

- 监督分箱数据只有一个有效 target 类别：改用带完整标签的基准样本，或选择无监督分箱。
- `benchmark_df` 缺少 active feature 或权重列：基准数据必须包含拟合规则所需的全部列。
- 复用规则时仍调用 `profile_risk()`：改用 `MarsBinEvaluator.evaluate(..., binner=...)`。
- 生成图表时没有 `time_col`：重新评估并提供原始日期列。
- WOE 转换或 SQL 报缺少映射：使用有两个有效类别的 target 拟合并完成 WOE 统计。

## 下一步

- 将筛选规则应用到宽表：[特征筛选](feature-selection.md)。
- 周期性监控固定规则：[特征与模型监控](monitoring.md)。
- 查询分箱器和 evaluator 的精确签名：[Feature API](../reference/feature.md) 与
  [Analysis API](../reference/analysis.md)。
