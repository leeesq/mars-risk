# MARS

<div align="center">
<img src="docs/assets/mars-logo.svg" alt="MARS" width="800">
<img src="docs/assets/mars-wordmark.svg" alt="MODELING ANALYSIS RISK SCORE" width="720">

<p align="center">
  <a href="https://pypi.org/project/mars-risk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mars-risk?style=flat-square&label=PyPI&color=2f6f8f"></a>
  <a href="https://leeesq.github.io/mars-risk/"><img alt="Docs" src="https://img.shields.io/badge/Docs-GitHub%20Pages-7c3aed?style=flat-square"></a>
  <a href="https://pypi.org/project/mars-risk/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/mars-risk?style=flat-square&label=Python&color=364f6b"></a>
  <a href="https://pepy.tech/project/mars-risk"><img alt="Downloads" src="https://img.shields.io/pepy/dt/mars-risk?style=flat-square&label=Downloads&color=0f766e"></a>
  <a href="https://github.com/leeesq/mars-risk/actions/workflows/test.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/leeesq/mars-risk/test.yml?branch=main&style=flat-square&label=CI&color=1f7a5a"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/leeesq/mars-risk?style=flat-square&label=License&color=6c5ce7"></a>
</p>
</div>

## 面向人和 AI Agent 的风控分析工具箱

**数据画像 · 分箱评估 · 特征筛选 · 相关性分析 · 模型分交叉 · 规则挖掘**

MARS 以 Polars 为计算基础，接收 Pandas 或 Polars 宽表，输出可查询、可交付、可复用的结构化分析报告。
人工分析可以读取统计表、查看图表、导出 Excel／HTML；AI Agent 可以读取报告目录、业务元数据、
指标口径、实际参数、计算状态与证据，保存并继续使用已有结果。

外部 Agent 可以直接消费公共分析与报告接口；可选的 `mars.agent` 是一种使用方式。

[开始人工分析](https://leeesq.github.io/mars-risk/getting-started/quickstart/) ·
[将报告交给 AI Agent](https://leeesq.github.io/mars-risk/user-guide/external-agents/) ·
[完整文档](https://leeesq.github.io/mars-risk/)

目前优先改进分析性能、内存效率、人工使用体验和 AI 可用性。
**监控、建模（含 Pipeline）、评分卡暂时停止功能迭代**，现有功能与使用文档继续保留。
可以借助编程型 AI 和 MARS 的分析、报告能力，构建更贴合业务的监控、建模与评分流程。
这些下游模块直接适配核心接口变化；上游不为其保留兼容层。
[稳定性与适配原则](https://leeesq.github.io/mars-risk/project/stability/)说明完整边界。

## 能解决的问题

| 问题 | 能力 | 主要输出 |
| --- | --- | --- |
| 数据能直接用吗？ | 数据画像 | 缺失、分布、样本概况、分组与时间变化 |
| 哪些特征有区分度？ | 分箱评估 | IV、KS、坏率、稳定性与分箱明细 |
| 哪些特征值得保留？ | 特征筛选 | 入选特征、筛选记录与取舍证据 |
| 特征是否重复表达信息？ | 相关性分析 | 邻居、矩阵、方法与筛选决策证据 |
| 两个模型分如何组合？ | 模型分交叉 | 固定分段、交叉客群表现与保存后策略回放 |
| 哪些组合条件需要关注？ | 规则挖掘（Experimental） | 候选规则、覆盖率、风险表现与验证结果 |

## 安装

源码版本为 **0.0.28**。本文公共报告与新分析能力面向当前源码，优先从 GitHub 安装：

```bash
pip install "git+https://github.com/leeesq/mars-risk.git"
```

截至 2026-10-01，[PyPI 已发布版本](https://pypi.org/project/mars-risk/)为 **0.0.27**，不包含本文全部新能力：

```bash
pip install mars-risk==0.0.27
```

基础包支持 Python 3.8–3.12；Python 3.8 使用冻结依赖栈。可选模型、调参、Notebook、
文档开发及内置 Agent 以 Python 3.10+ 为支持环境。
[安装指南](https://leeesq.github.io/mars-risk/getting-started/installation/)列出各 extras 与兼容约束。

## 最小分析示例

```python
import polars as pl
from mars.analysis import profile_risk

df = pl.DataFrame({
    "income": [3200, 3600, 5200, 6100, 3400, 4300, 5800, 6800],
    "utilization": [0.72, 0.61, 0.29, 0.18, 0.66, 0.48, 0.24, 0.12],
    "target": [1, 1, 0, 0, 1, 1, 0, 0],
})
report = profile_risk(
    df, target="target", features=["income", "utilization"],
    method="quantile", n_bins=4,
).report
top_features = report.get_table("summary", sort_by="iv", descending=True, limit=10)
report.show_summary(limit=10)
report.write_html("risk_report.html", include_charts=False)
report.save("risk_report.marsreport")
```

也可用 `write_excel()` 交付统计表；日期趋势和图表需要计算时提供有效日期上下文。
[Quickstart](https://leeesq.github.io/mars-risk/getting-started/quickstart/)给出完整示例。

## 将结果交给 AI Agent

在新进程中直接加载公共快照，无需原始宽表、原分析器或内置 Agent 会话：

```python
from mars.reporting import load_report

report = load_report("risk_report.marsreport")
description = report.describe()
page = report.query_page("summary", limit=10)
context_json = report.to_ai_context(tables=["summary"], max_chars=16000)
```

JSON 摘要可粘贴到其他对话框；完整 `.marsreport` 供具备 Python／工具调用能力的外部 Agent 加载查询。
普通对话模型收到快照文件后仍需要执行环境。`max_chars` 是最终 JSON 的字符预算，**不是 token 数**，
完整结果保存在快照中。恢复类型为 `ReportSnapshot`，可 `show_table`、导出与查询；
原报告的 `show_summary` 等展示方法不能一概套用到快照。
[外部 Agent 指南](https://leeesq.github.io/mars-risk/user-guide/external-agents/)说明目录、证据、元数据与恢复边界。

## 按任务查文档

| 任务 | 文档 |
| --- | --- |
| 数据质量与分布变化 | [数据画像](https://leeesq.github.io/mars-risk/user-guide/data-profiling/) |
| 分箱、区分度与稳定性 | [分箱评估](https://leeesq.github.io/mars-risk/user-guide/binning-risk-evaluation/) |
| 筛选与取舍 | [特征筛选](https://leeesq.github.io/mars-risk/user-guide/feature-selection/) |
| 相关性邻居和矩阵 | [相关性分析](https://leeesq.github.io/mars-risk/user-guide/correlation-and-score-cross/#相关性报告) |
| 双模型分客群与策略 | [模型分交叉](https://leeesq.github.io/mars-risk/user-guide/correlation-and-score-cross/#固定分段交叉) |
| 规则生成与验证 | [规则挖掘（Experimental）](https://leeesq.github.io/mars-risk/user-guide/rule-mining/) |
| 查询、导出、保存与恢复 | [公共报告](https://leeesq.github.io/mars-risk/user-guide/reports-and-exports/) |
| 交给外部 Agent 继续使用 | [外部 Agent](https://leeesq.github.io/mars-risk/user-guide/external-agents/) |
| 可选自然语言工具调用 | [内置 Agent（Experimental）](https://leeesq.github.io/mars-risk/user-guide/agent/) |

保留功能，暂缓迭代：[建模／Pipeline](https://leeesq.github.io/mars-risk/user-guide/modeling-pipeline/) ·
[监控](https://leeesq.github.io/mars-risk/user-guide/monitoring/) ·
[评分卡](https://leeesq.github.io/mars-risk/user-guide/scorecard/)。
精确签名见 [API Reference](https://leeesq.github.io/mars-risk/reference/)。

## 稳定性与贡献

Analysis、Feature、Reporting 为 Stable；Rule、Monitoring、Modeling、Pipeline、Scoring 和 Agent 为
Experimental。接口成熟度与开发投入分别说明：暂停迭代不改变模块成熟度。
核心公共契约和保存文件的变化仍需记录影响与迁移。
详见[稳定性规则](https://leeesq.github.io/mars-risk/project/stability/)、
[CONTRIBUTING.md](CONTRIBUTING.md)及 [MIT License](LICENSE)。
