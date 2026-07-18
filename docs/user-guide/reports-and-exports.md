---
description: 读取结构化 report，导出 Excel/HTML，并从分箱规则构建评分卡和部署 SQL。
---

# 报告与评分卡

!!! info "Reporting：Stable"

    本页的结构化 report 读取和 Excel/HTML 导出属于 Stable Reporting 能力。评分卡能力的状态
    单独标注在对应章节。

## 适用场景

Report 用于继续筛选、复盘和组合计算；Excel/HTML 用于归档或人工交付；Scorecard 将已拟合分箱规则
与逻辑回归系数转换为评分映射和 SQL。

## 1. 获得 Report

下面的受测试示例定义了 `report`，后续导出调用均基于该对象：

```python
--8<-- "docs/snippets/quickstart.py"
```

常见对象与字段：

| Report | 状态 | 高价值字段 |
| --- | --- | --- |
| `MarsProfileReport` | Stable | `overview_table`、`dq_tables`、`stats_tables` |
| `MarsBinningReport` | Stable | `summary_table`、`detail_table`、`trend_tables` |
| `MarsMonitoringReport` | Experimental | 监控汇总、分箱统计、表现覆盖率和元数据 |
| `MarsModelingReport` | Experimental | 多样本切片的汇总、明细、趋势和元数据 |

## 2. 导出 Excel 或 HTML

以下代码继续使用上一步定义的 `report`：

```python
report.write_excel("risk_report.xlsx", engine="openpyxl")
report.write_html(
    "risk_report.html",
    report_name="Current-period risk review",
    max_plots=100,
    chart_embed_mode="auto",
)
```

| HTML 模式 | 行为 |
| --- | --- |
| `auto` | 小报告内嵌图片，大报告生成同级资产目录并懒加载 |
| `inline` | 所有图片内嵌，适合必须单文件离线转发的报告 |
| `asset` | 强制使用相对路径图片目录，适合大报告归档 |

风险趋势图和 HTML Charts 需要评估阶段已经提供有效 `time_col`。`group_col` 不能替代日期范围。

## 3. 单独复用趋势图

继续使用上一步定义的 `report`：

```python
figures = report.build_risk_trend_figures(features=["income"])
fragment = report.render_risk_trends_html(
    features=["income"],
    image_format="svg",
    embed_mode="inline",
)
```

`fragment.html` 是可嵌入现有模板的 HTML 片段；资产模式同时返回已写入的图片路径。

## 4. 构建评分卡

!!! warning "Scoring：Experimental"

    评分映射、刻度参数和 SQL 输出仍可能调整。受控生产使用应固定 `mars-risk==0.0.24`，
    并为 `points_table` 和生成 SQL 增加契约测试。

```python
--8<-- "docs/snippets/reporting_scorecard.py"
```

评分卡要求分箱器已经使用 target 拟合并具备 WOE 映射。系数字典的特征必须与分箱器规则一致。

## 常见失败

- HTML 图表为空：确认生成 report 时传入了有效 `time_col`。
- 大报告单文件打开缓慢：使用 `auto` 或 `asset`，不要强制内嵌数百张图片。
- 评分卡提示缺少映射：确认 binner 已拟合、特征名一致且包含 WOE 统计。

## 下一步

- 理解 report 与 artifact 的边界：[Report 与 Artifact](../concepts/reports-and-artifacts.md)。
- 查询导出对象：[Reporting API](../reference/reporting.md)。
- 查询评分卡签名：[Scoring API](../reference/scoring.md)。
