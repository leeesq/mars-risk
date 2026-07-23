---
description: 从安装后的空环境完成一次可运行的风险评估，并读取结构化结果。
---

# 10 分钟 Quickstart

本页从一个小型 Polars 数据集开始，完成风险评估并读取返回对象。运行前只需要安装 MARS：

```bash
pip install mars-risk==0.0.25
```

## 1. 运行完整示例

下面的代码块是受测试的完整示例，可以直接保存并运行。样本包含原始申请日期、已有月份分组、
两个数值特征、一个类别特征和二分类 target。

```python
--8<-- "docs/snippets/quickstart.py"
```

## 2. 读取返回对象

`profile_risk()` 返回 `MarsRiskProfile`，而不是直接返回文件路径：

| 字段 | 含义 |
| --- | --- |
| `report` | `MarsBinningReport`，包含汇总、明细、趋势和导出方法 |
| `binner` | 本次拟合的分箱器，可复用相同规则转换其他数据 |
| `targets` | 本次评估的 target 列表 |
| `metadata` | 特征、分组、日期和分箱配置等运行元数据 |

`group_col="month"` 决定报告按哪个分组展开；`time_col="apply_dt"` 保存真实日期范围并支持按日
缺失趋势。两者职责不同，详见[数据角色与运行边界](../concepts/data-and-runs.md)。

## 3. 导出报告

在示例末尾的 `report` 对象上继续调用：

```python
report.write_html(
    "risk_report.html",
    report_name="Current-period risk review",
    max_plots=20,
    chart_embed_mode="auto",
)
```

HTML 图表需要评估时已经提供有效 `time_col`。需要控制基准期、复用分箱规则或评估无标签当前期时，
进入[分箱与风险评估](../user-guide/binning-risk-evaluation.md)。

## 下一步

- 检查缺失、分布和 PSI：[数据画像](../user-guide/data-profiling.md)。
- 使用基准期规则评估当前期：[分箱与风险评估](../user-guide/binning-risk-evaluation.md)。
- 监控尚未充分表现的数据：[特征与模型监控](../user-guide/monitoring.md)。
- 查询精确签名、默认值和异常：[API Reference](../reference/index.md)。
