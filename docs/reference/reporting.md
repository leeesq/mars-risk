---
description: Reporting stable API：画像、分箱报告和 HTML 渲染结果。
---

# Reporting

**状态：Stable。** 导出流程见[报告与评分卡](../user-guide/reports-and-exports.md)。

::: mars.reporting.MarsProfileReport
    options:
      inherited_members: true

::: mars.reporting.MarsBinningReport
    options:
      inherited_members: true

::: mars.reporting.MarsHtmlRenderResult

::: mars.reporting.ProfileData

::: mars.reporting.Report

::: mars.reporting.ReportSnapshot
    options:
      inherited_members: true

::: mars.reporting.load_report

::: mars.reporting.snapshot_report

## 可携带相关性证据

使用方式与保存后的专用操作见[相关性与模型分交叉](../user-guide/correlation-and-score-cross.md)。

::: mars.reporting.CorrelationReport
    options:
      inherited_members: true

::: mars.reporting.get_correlation_matrix

::: mars.reporting.get_related_features

::: mars.reporting.show_correlation_matrix

## 模型分交叉报告

`ScoreCrossReport` 的公开导入位于 `mars.analysis`，签名见[Analysis API](analysis.md)，
导出和保存后的规则回放见[模型分交叉指南](../user-guide/correlation-and-score-cross.md#固定分段交叉)。
