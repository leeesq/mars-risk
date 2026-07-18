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
`to_dict()` / `from_dict()` 和 `prune()`。

## 输出

`MarsRiskProfile` 保存本次 `report`、`binner`、`targets` 和 `metadata`。常用 report 字段：

| 字段 | 用途 |
| --- | --- |
| `summary_table` | 特征级指标汇总和排序 |
| `detail_table` | 分箱样本数、坏账率、WOE 和 IV 明细 |
| `trend_tables` | PSI、缺失率和坏账率等分组趋势 |
| `missing_by_day_table` | 使用 `time_col` 计算的按日缺失趋势 |

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

## 下一步

- 将筛选规则应用到宽表：[特征筛选](feature-selection.md)。
- 周期性监控固定规则：[特征与模型监控](monitoring.md)。
- 查询分箱器和 evaluator 的精确签名：[Feature API](../reference/feature.md) 与
  [Analysis API](../reference/analysis.md)。
