# MARS

<div align="center">
  <img src="assets/mars-logo.svg" alt="MARS" width="560">
  <br>
  <img src="assets/mars-wordmark.svg" alt="MODELING ANALYSIS RISK SCORE" width="720">
  <h2>面向信贷风控分析与建模的 Polars-first 高性能工具库</h2>
  <img src="assets/mars-pipeline.svg" alt="Profile -> Bin/Evaluate -> Analyze -> Select -> Modeling Pipeline -> Monitor -> Report" width="920">
</div>

MARS 覆盖数据画像、分箱评估、特征分析、特征筛选、Modeling Pipeline、特征/模型监控指标计算和 Excel/HTML 报表导出。它以宽表特征为主线，串联训练前分析、建模期调参与评估、特征与模型分稳定性观察和报表导出，让日常风控建模流程更容易复用、审计和交付。

## 适用场景

- 信贷风控宽表数据探索、特征质量检查和稳定性分析。
- 连续特征、类别特征和业务特殊值的分箱评估。
- 基于 IV、KS、AUC、PSI、缺失率、相关性和模型重要性的特征筛选。
- XGBoost、LightGBM、CatBoost 和 Logistic Regression 的建模调参、replay 和建模评估。
- 将模型分、prob、score 当作特殊特征，计算特征/模型监控指标。
- 将画像、分箱评估和建模评估结果导出为 Excel/HTML，或读取 report 对象做二次加工。

## 推荐阅读路径

1. [安装](getting-started/installation.md)
2. [快速开始](getting-started/quickstart.md)
3. [核心 API 约定](getting-started/api-conventions.md)
4. [数据画像与特征分析](user-guide/data-profiling.md)
5. [分箱与风险评估](user-guide/binning-risk-evaluation.md)
6. [特征筛选](user-guide/feature-selection.md)
7. [Modeling Pipeline](user-guide/modeling-pipeline.md)
8. [特征/模型监控](user-guide/monitoring.md)
9. [报表导出与二次加工](user-guide/reports-and-exports.md)
10. [性能对比](performance/benchmark.md)
11. [FAQ](faq.md)

## 模块地图

| 模块 | 主要 API | 主要产出 |
| --- | --- | --- |
| 数据画像 | `MarsDataProfiler`、`profile_stats` | `MarsProfileReport` |
| 分箱评估 | `MarsNativeBinner`、`MarsOptimalBinner`、`MarsBinEvaluator`、`profile_risk` | `MarsRiskProfile`、`MarsEvaluationReport` |
| 特征筛选 | `MarsStatsSelector`、`MarsLinearSelector`、`MarsImportanceSelector` | `selected_features_`、筛选报告 |
| Modeling Pipeline | `MarsModelingSession`、`MarsModelTuner`、`MarsModelReplayRunner`、`MarsModelEvaluator` | `MarsModelTuningResult`、`MarsModelReplayResult`、`MarsModelingReport` |
| 特征/模型监控 | `MarsMonitor`、`generate_monitoring_alert` | `MarsMonitoringReport`、报警摘要 |
| 报表与评分卡 | `write_excel`、`write_html`、`build_scorecard` | Excel、HTML、`MarsScorecard` |

## 设计取向

- **性能优先**：核心计算优先使用 Polars，面向宽表、大样本、多特征风控场景优化。
- **sklearn 风格**：底层算法对象保持 `fit` / `transform` / `evaluate` 等熟悉范式。
- **风控全链路**：围绕数据画像、分箱评估、特征筛选、建模、监控指标计算和报表导出组织能力。
- **Pandas/Polars 兼容**：支持 Pandas 和 Polars 输入输出，核心计算尽量走 Polars。
- **report 可复盘**：各模块 report 保留汇总、明细、趋势、元数据等多粒度结构化数据。

## 项目边界

MARS 不是完整线上监控平台，也不是模型注册、调度、审批和看板系统。MARS 提供可复用的风控计算、结构化 report 和默认摘要工具，监控窗口、模型版本、调度方式、业务阈值和处置流程由使用者结合内部系统定义。

