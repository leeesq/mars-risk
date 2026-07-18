---
description: MARS 0.0.23 文档入口：安装、最小风险评估、任务指南和 API Reference。
---

# MARS

<div class="mars-home-hero mars-home-hero--compact">
  <img class="mars-home-logo" src="assets/mars-logo.svg" alt="MARS">
  <img class="mars-home-wordmark" src="assets/mars-wordmark.svg" alt="MODELING ANALYSIS RISK SCORE">
</div>

MARS 是面向信贷风控宽表分析、分箱评估、特征筛选、建模和监控的
Polars-first Python 工具库。它接受 Pandas 或 Polars 数据，并返回可继续读取和加工的结构化
report 对象。

=== "安装"

    ```bash
    pip install mars-risk==0.0.23
    ```

=== "最小风险评估"

    ```python
    --8<-- "docs/snippets/quickstart.py"
    ```

`risk_profile.report` 提供汇总、分箱明细和趋势表；`risk_profile.binner` 保存本次拟合的分箱规则。
完整说明见[10 分钟 Quickstart](getting-started/quickstart.md)。

## 从任务开始

<div class="mars-task-grid" markdown>

<a class="mars-task-card" href="user-guide/data-profiling/">
  <strong>检查数据质量</strong>
  <span>缺失、特殊值、分布、PSI 与分组趋势</span>
  <em>MarsDataProfiler</em>
</a>

<a class="mars-task-card" href="user-guide/binning-risk-evaluation/">
  <strong>评估特征风险</strong>
  <span>分箱、IV、KS、AUC、Lift、PSI 与风险趋势</span>
  <em>profile_risk / MarsBinEvaluator</em>
</a>

<a class="mars-task-card" href="user-guide/feature-selection/">
  <strong>筛选候选特征</strong>
  <span>质量、区分度、稳定性、相关性与重要性</span>
  <em>MarsStatsSelector</em>
</a>

<a class="mars-task-card" href="user-guide/modeling-pipeline/">
  <strong>训练与复现实验</strong>
  <span>样本切分、调参、replay、WOE 与 Pipeline</span>
  <em>Experimental</em>
</a>

<a class="mars-task-card" href="user-guide/monitoring/">
  <strong>监控特征或模型</strong>
  <span>分布漂移、缺失趋势、表现覆盖率与报警摘要</span>
  <em>Experimental</em>
</a>

<a class="mars-task-card" href="user-guide/reports-and-exports/">
  <strong>交付报告或评分卡</strong>
  <span>Excel、HTML、结构化表、评分映射与 SQL</span>
  <em>Reporting Stable / Scoring Experimental</em>
</a>

</div>

## 入口怎么选

| 目标 | 推荐入口 | 主要输出 |
| --- | --- | --- |
| 数据质量与分布检查 | `MarsDataProfiler.generate_profile()` | `MarsProfileReport` |
| 一次性风险评估 | `profile_risk()` | `MarsRiskProfile` |
| 固定分箱规则评估 | `MarsBinEvaluator.evaluate()` | `MarsRiskProfile` |
| 统计、线性或重要性筛选 | `MarsStatsSelector` 等 | 特征列表与筛选报告 |
| 周期分布与表现监控 | `MarsMonitor.monitor()` | `MarsMonitoringReport` |
| 建模与实验复现 | `MarsModelingSession` / `MarsModelingPipeline` | 建模结果与 artifact |

## 稳定性

Analysis、Feature、Reporting 是当前 **Stable** 模块。Monitoring、Modeling、Pipeline、Scoring
为 **Experimental**：受控生产流程应固定精确版本，并为关键结果和导出产物增加契约回归。
完整规则见[稳定性与兼容性](project/stability.md)。

!!! info "版本"

    本站面向 MARS `0.0.23`。该版本正式发布前，安装命令仅用于文档预览验收，不能作为已发布
    PyPI 包的可用性证明。
