---
description: 自动建箱、复用基准期规则并读取 IV、KS、AUC、Lift、PSI 和趋势结果。
---

# 分箱与风险评估

## 适用场景

使用本指南评估单个特征与二分类 target 的关系，或使用稳定的基准期分箱规则评估当前数据。常见
输出包括 IV、KS、AUC、Lift、坏账率、PSI、缺失率和分箱趋势。

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

规则来源优先级为显式 `binner`、`benchmark_df`、当前 `df`。基准样本不会进入当前期 Total，
但会提供 PSI expected distribution。

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
`update_bins()` 对未知特征默认报错，`prune()` 和 `get_bin_mapping()` 也不再静默忽略未知名称。

### 重复拟合与失败恢复

每次 `fit()` 都重新学习本次特征的规则、WOE、映射和诊断。连续拟合相同数据与新实例等价；
改变 target、特征集合或特征类型时不会复用上次学到的结果。构造参数保持不变，
本次传入的 `features` 和 `cat_features` 决定本次范围。

通过 `update_bins()` 明确设置的规则属于用户配置：后续拟合仍包含兼容类型的该特征时继续
应用，并重新计算本轮 WOE；数值规则不会套用到新的类别特征。`prune()` 会同时移除被裁剪
特征的明确规则。JSON 保存和加载保留这些规则；旧 schema 1 artifact 没有该可选字段时仍可加载。

拟合失败后，转换、分箱表现评估、规则查询、SQL 导出和保存都不能读取上次成功结果，
会按未拟合状态报错。`get_fit_report()` 可以查看本次生成的失败诊断（参数校验阶段失败
可能为空），再次有效 `fit()` 可以恢复。宽表中有可用规则的局部失败和成功 fallback
仍按原有契约保留，不要求每个特征都成功。

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

Boolean 类别使用有类型的匹配键；推理时整数 `0/1` 或文本 `"True"`、`"false"` 不会
被悄悄解释为布尔值，未见类别沿用 Other。普通文本仍区分大小写，`"True"`、`"true"`、
`"1"`、`"001"` 和字面 `"nan"` 是独立类别。原有非 Boolean 类别的字符串匹配保持不变，
例如明确设为类别的整数 `1` 可以匹配文本 `"1"`，但不会匹配 `"001"` 或浮点表示 `1.0`。
混合物理类型应在输入前清理；Pandas 转换无法保持类型时明确报错。Null 和数值 NaN 进入
Missing，转换结果保留用户原列，不泄漏或覆盖内部临时列。Native/LiteOpt 的全空 Boolean
列保留空类别规则，后续非空值进入 Other，null 进入 Missing；Optimal 的全空输入仍沿用
没有可用规则时明确失败的契约，不将其伪装成求解成功。

`profile_bin_performance()` 与高层风险评估共享二分类标签校验：观测值允许 `0/1`、
Boolean，及文本 `"0"`、`"1"`、`"true"`、`"false"`、`"True"`、`"False"`。
Null、NaN 和空字符串表示未表现；`-1` 在此接口不是未表现哨兵，和 `2`、`0.5` 等一样
在聚合前报错，不会参与坏样本求和或覆盖 WOE。直接表现接口保留合法单类别的计数和坏账率，
没有任何观测标签则报错；高层监督分箱仍要求满足所选算法的标签条件。
直接分箱器的监督拟合要求保持原样，部分标签的高层评估会按既有规则筛选拟合样本。

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
默认分箱 KS 按 WOE 排序，而原始值 KS 按数值顺序计算，两者可能明显不同，不能简单理解为精度升级。
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

汇总表、趋势表、图表与 KS 排序统一读取最终值。raw 模式在 `detail_table` 增加重复到每个
分箱行的最终 `ks`；`ks_bin` 仍是分箱中间值，不能用于恢复原始值 KS。
多标签分别计算，`trend_tables` 沿用仅展示主标签的契约。
`report_meta["ks_source_by_feature"]` 记录各特征使用的方法，
`report_meta["raw_ks_diagnostics"]` 记录数值特征各标签、各组的有效样本数和空值原因。
`valid_count` 包含符合样本条件的零权重行；`no_valid_samples` 表示没有有效样本，
`insufficient_class_weight` 表示至少一类没有正权重。

raw 模式的 Excel 导出使用已计算的汇总、趋势、完整明细和诊断工作表，打开后即可读取结果，
不依赖模板透视缓存刷新。默认分箱模式仍使用原模板。

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
