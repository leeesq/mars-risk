# 特征/模型监控

调用 `MarsMonitor.monitor()` 可得到特征/模型的 PSI、缺失率、分箱占比、表现覆盖率和默认报警摘要。
调用方按监控周期传入样本与基准期，保存返回的 report，并在自身运行环境中配置触发频率和报警处置。

## 支持场景

- **前端监控**：例如申请入口、渠道、产品、地区等在进入模型前的特征分布、缺失率和分箱占比变化。
- **后端监控**：例如模型输出分布、表现期 target 覆盖率和已表现样本风险指标。

## 基本用法

```python
from mars.monitoring import MarsMonitor, generate_monitoring_alert

report = MarsMonitor(
    binner_params={"method": "quantile", "n_bins": 5},
    psi_include_missing=False,
    psi_include_special=False,
).monitor(
    df,
    features=["model_score", "income", "utilization"],
    target="target",
    group_col="month",
    psi_include_missing=False,
    psi_include_special=False,
    trend_column_order="desc",
)

alert_text = generate_monitoring_alert(
    report,
    score_key="model_score",
    model_features=["income", "utilization"],
)
```

## target 规则

`target` 只接受 `0`、`1`、`True`、`False` 和空值：

- `0` / `False`：好样本。
- `1` / `True`：坏样本。
- `null` / `NaN`：尚未到表现期。

`"0"`、`"1"`、`"true"`、`"false"`、`-1`、`2`、`"pending"` 等非空异常值会直接抛出 `ValueError`。用户需要在进入监控模块前完成 target 清洗。

## 有标签与无标签监控

```python
distribution_report = MarsMonitor().monitor(
    df,
    features=["income", "utilization"],
    target=None,
    group_col="month",
)
```

`target=None` 时只输出无标签分布监控。传入 target 时：

- PSI、缺失率和分箱占比使用全量样本。
- 坏账率、IV、KS、AUC、Lift 等标签指标只使用已表现样本。
- 某个时间段 target 全为空时，该时间段仍保留 PSI、缺失率和分箱占比，标签类指标输出空值。

有 target 的监控可以设置 `binning_type="lite_opt"` 使用轻量监督式最优分箱；无标签分布监控建议保持默认 `native`。

当当前期尚未到表现期、没有 target 列或 target 全为空时，仍可传入 target 名称和带标签的
`benchmark_df`。MARS 会用 benchmark 拟合监督分箱规则，并把当前期报告作为无标签监控输出：

```python
report = MarsMonitor(
    binner_params={"method": "cart", "n_bins": 5},
).monitor(
    june_df,
    features=["model_score", "income"],
    target="target",
    benchmark_df=may_labeled_df,
    group_col="month",
)
```

benchmark 必须包含全部监控特征；指定 `weights_col` 时也必须包含对应权重列。传入已拟合
`binner` 时，该 binner 优先，benchmark 只提供 PSI 基准和可选的 RC 基准。

## PSI 与缺失值

监控默认不把缺失箱和特殊值箱纳入 PSI：

```python
monitor = MarsMonitor(
    psi_include_missing=False,
    psi_include_special=False,
)
```

缺失率会单独进入趋势监控，因此默认不混入 PSI。需要复现某些历史口径时，可以在 `MarsMonitor(...)` 构造函数中设置默认值，也可以在单次 `monitor(...)` 调用中显式覆盖这两个参数。

## 趋势列顺序

`trend_column_order` 控制趋势宽表的时间或分组列顺序：

- `"asc"`：从早到晚，默认行为。
- `"desc"`：从晚到早，适合让最新月份靠前展示。

`Total` 列如果存在，会固定在最后。报警器会读取 `MarsMonitoringReport.metadata` 中记录的趋势列顺序，识别基准期和最新期。

## report 数据结构

| 字段 | 含义 |
| --- | --- |
| `summary_table` | 特征级监控汇总 |
| `detail_table` | 分箱明细 |
| `trend_tables` | PSI、缺失率、坏账率等趋势表 |
| `missing_by_day_table` | 按日缺失率趋势 |
| `bin_stat_table` | 每个特征、每个分箱的统计量 |
| `bin_stat_trend_tables` | 分箱统计量随时间或分组变化的趋势表 |
| `target_observation_table` | target 表现覆盖率、未表现样本数和已表现样本坏账率 |
| `metadata` | 运行上下文、趋势列顺序和 PSI 口径 |

## 报警摘要

```python
from mars.monitoring import MarsMonitoringAlertConfig, generate_monitoring_alert

alert_text = generate_monitoring_alert(
    report,
    score_key="model_score",
    model_features=["income", "utilization"],
    config=MarsMonitoringAlertConfig(
        psi_warn=0.10,
        psi_critical=0.25,
    ),
)
```

报警摘要读取已有 report 表；缺少某类表或字段时会跳过对应检查。将摘要作为复核输入时，请结合
自己的阈值、审批规则和处置流程决定后续动作。
