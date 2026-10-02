**中文** / [English](README.en.md)

<p align="center">
<picture>
<source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/mars-logo.svg">
<source media="(prefers-color-scheme: dark)" srcset="docs/assets/mars-logo-dark.gif">
<img src="docs/assets/mars-logo-light.gif" alt="MARS" width="480">
</picture>
<br>
<img src="docs/assets/mars-wordmark.svg" alt="MODELING ANALYSIS RISK SCORE" width="480">
</p>
<p align="center">
  <a href="https://pypi.org/project/mars-risk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mars-risk?style=flat-square&label=PyPI&color=2f6f8f"></a>
  <a href="https://leeesq.github.io/mars-risk/"><img alt="Docs" src="https://img.shields.io/badge/Docs-GitHub%20Pages-7c3aed?style=flat-square"></a>
  <a href="https://pypi.org/project/mars-risk/"><img alt="Python" src="https://img.shields.io/badge/Python-3.8--3.12-364f6b?style=flat-square"></a>
  <a href="https://pepy.tech/project/mars-risk"><img alt="Downloads" src="https://img.shields.io/pepy/dt/mars-risk?style=flat-square&label=Downloads&color=0f766e"></a>
  <a href="https://github.com/leeesq/mars-risk/actions/workflows/test.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/leeesq/mars-risk/test.yml?branch=main&style=flat-square&label=CI&color=1f7a5a"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/leeesq/mars-risk?style=flat-square&label=License&color=6c5ce7"></a>
</p>

<h1 align="center">面向人和 AI Agent 的高性能信贷风控工具箱</h1>
<p align="center"><strong>数据画像 · 分箱评估 · 特征筛选 · 相关性分析 · 模型分交叉 · 规则挖掘</strong></p>
<p align="center">从 Pandas 或 Polars 宽表，得到可查看、可查询、可交付、可复用的风控分析结果。<br>用原生图表支持人工判断，用结构化报告连接你的建模、监控归因与分析 Agent。</p>
<p align="center"><a href="https://leeesq.github.io/mars-risk/getting-started/quickstart/">开始分析</a> · <a href="https://leeesq.github.io/mars-risk/demos/">实战案例</a> · <a href="https://leeesq.github.io/mars-risk/user-guide/external-agents/">面向 AI Agent</a> · <a href="https://leeesq.github.io/mars-risk/">完整文档</a></p>

<p align="center">
<a href="https://leeesq.github.io/mars-risk/assets/cases/binning-native-main-score.svg">
<img src="docs/assets/cases/binning-native-main-score.png" alt="MARS 原生分箱趋势图：固定分箱、样本分布、件数与金额 bad rate、稳定性" width="1040">
</a>
</p>

<p align="center"><sub>MARS 原生分箱趋势图，直接导出自分析报告；图表与统计表共享同一计算结果。点击查看高清 SVG。</sub><br>
<a href="https://leeesq.github.io/mars-risk/demos/binning-stability/">分箱与稳定性实战</a> · <a href="https://leeesq.github.io/mars-risk/assets/cases/binning.html">打开交互式报告</a></p>

## MARS 能帮你做什么

MARS 以 Polars 为计算基础，接受 Pandas 或 Polars 宽表，将数据画像、分箱风险评估、特征筛选、相关性和规则分析组织为统一的分析报告。你可以查看分布与风险趋势，比较客群和时间，追溯特征保留与删除的原因，再将结果交付为图表、HTML 或 Excel。

同一份结果也能交给 **外部 AI Agent**。公共报告接口保留表目录、指标口径、特征业务元数据、计算状态和可重放证据，保存后仍能按需查询。Agent 可以把 MARS 作为分析计算层，结合你选择的模型库、实验工具和业务流程，持续提出问题、读取证据和完成分析。

| 核心能力 | 可以看到什么 | 实战入口 |
| --- | --- | --- |
| 数据画像 | 缺失与特殊值、样本与标签覆盖、分布变化 | [判断数据是否可以使用](https://leeesq.github.io/mars-risk/demos/data-quality/) |
| 分箱与风险评估 | 原生趋势图、固定分箱、多目标 IV / KS / PSI、件数与金额风险 | [比较区分度与稳定性](https://leeesq.github.io/mars-risk/demos/binning-stability/) |
| 特征筛选 | 候选特征、筛选步骤、保留与删除理由 | [理解特征取舍](https://leeesq.github.io/mars-risk/demos/selection-correlation/) |
| 相关性分析 | 带符号的冗余关系，分别查看 raw / WOE 证据 | [检查特征之间的关系](https://leeesq.github.io/mars-risk/demos/selection-correlation/#correlation) |
| 模型分交叉 | 同一主等级内的风险分离与组合证据 | [比较辅助模型分](https://leeesq.github.io/mars-risk/demos/score-cross/) |
| 规则挖掘 · Experimental | 候选来源、筛选、覆盖与独立验证 | [用证据审查规则](https://leeesq.github.io/mars-risk/demos/rule-evidence/) |

[七个实战案例](https://leeesq.github.io/mars-risk/demos/)共用 18,000 行合成申请，发现、验证、观察分区独立。人工图表和 Agent 查询读取实际报告；合成统计用于展示能力，不代表真实信贷业务效果。

## 基于 MARS，构建你的 Agent

MARS 提供计算与证据能力，你可以自由组合 Agent 编排、模型训练工具和业务决策流程。

| 外部 Agent 场景 | 可复用的 MARS 能力 | 可以形成的结果 |
| --- | --- | --- |
| 建模 Agent | 数据质量、分箱评估、特征筛选、相关性、风险评估 | 特征候选与淘汰理由，交给自选训练工具的评估证据 |
| 监控与归因 Agent | 分组画像、缺失和分布变化、固定分箱风险趋势、稳定性 | 定位变化人群与特征，汇总证据，提出待验证的归因假设 |
| 规则分析 Agent | 候选规则、命中评估、发现审计与独立验证 | 带风险、覆盖与验证结果的可审查候选 |
| 报告分析 Agent | 表目录、业务元数据、有限查询、快照与证据引用 | 跨会话继续分析，交付可追溯的回答 |

这些是可以基于 MARS 构建的工作流，完整 Agent 由使用者组合实现。可选 <code>mars.agent</code> 也复用公共报告接口。[外部 Agent 使用指南](https://leeesq.github.io/mars-risk/user-guide/external-agents/) · [规则证据与 Agent](https://leeesq.github.io/mars-risk/user-guide/rule-reports-and-agents/)。

## 安装

本页示例对应 **main / 源码版本 0.0.28**，使用当前源码安装：

```bash
pip install "git+https://github.com/leeesq/mars-risk.git"
```

PyPI 当前发布 **0.0.27**，尚不包含这里展示的全部新能力。基础包支持 Python **3.8–3.12**，3.8 使用冻结依赖栈；可选建模、调参、Notebook、文档开发与内置 Agent SDK 使用 **Python 3.10+**。[安装与可选依赖](https://leeesq.github.io/mars-risk/getting-started/installation/)。

## 先得到一份报告

在空工作目录运行这个独立示例：

```python
import polars as pl

from mars.analysis import profile_risk

df = pl.DataFrame({
    "date": ["2026-01-01"] * 4 + ["2026-02-01"] * 4,
    "income": [3200, 3600, 5200, 6100, 3400, 4300, 5800, 6800],
    "utilization": [0.72, 0.61, 0.29, 0.18, 0.66, 0.48, 0.24, 0.12],
    "target": [1, 1, 0, 0, 1, 1, 0, 0],
}).with_columns(pl.col("date").str.to_date())
report = profile_risk(
    df, target="target", features=["income", "utilization"],
    time_col="date", method="quantile", n_bins=4,
).report
print(report.get_table("summary", sort_by="iv", descending=True, limit=2))
report.write_html("risk_report.html", chart_embed_mode="inline")
report.save("risk_report.marsreport")
```

[共享可运行源码](docs/snippets/readme_quickstart.py) · [完整 Quickstart](https://leeesq.github.io/mars-risk/getting-started/quickstart/)。八行小样本用于说明接口；完整趋势图和可下载结果见[分箱实战](https://leeesq.github.io/mars-risk/demos/binning-stability/)。原生图可以直接通过 <code>report.save_risk_trend_images()</code> 保存为 SVG 或高清 PNG。

## 将同一份结果交给 AI Agent

在新进程加载已保存的报告，按需读取证据：

```python
from mars.reporting import load_report

report = load_report("risk_report.marsreport")
description = report.describe()
page = report.query_page("summary", limit=1)
next_page = report.query_page("summary", offset=page["next_offset"], limit=1)
context_json = report.to_ai_context(tables=["summary"], max_chars=16000)
```

<code>describe()</code> 提供真实表目录、口径、参数与状态；<code>query_page()</code> 返回结果、分页和可重放证据引用；<code>to_ai_context()</code> 生成有限 JSON，**<code>max_chars</code> 是字符预算，不是 token 数**。查询已有结果无需 LLM、API Key、<code>MarsAgentSession</code>、原始宽表或分析器。

加载得到 **ReportSnapshot**，包含已保存统计和元数据。<code>report.write_excel("risk_report.xlsx")</code> 导出静态表，<code>report.show_table("summary", limit=10)</code> 用于展示。需要新维度或原始记录的问题，应发起新的计算。[保存后继续查询](https://leeesq.github.io/mars-risk/demos/saved-reports/) · [交付 HTML、Excel、快照与 Agent 材料](https://leeesq.github.io/mars-risk/demos/report-delivery/)。

## 当前迭代方向

**Analysis、Feature、Reporting 为 Stable；Rule、Monitoring、Modeling、Pipeline、Scoring、Agent 为 Experimental。** 这些标记说明接口成熟度。

当前重点改进核心分析性能、内存效率、人工体验与外部 Agent 消费能力。**监控、建模（含 Pipeline）、评分卡暂时停止功能迭代**，现有功能和文档保留，处理必要修复与上游适配。你可以借助 AI、MARS 分析能力和自选模型工具构建更定制的流程。三个下游直接适配核心代码，上游不为它们保留兼容别名或冗余计算。

[稳定性与适配原则](https://leeesq.github.io/mars-risk/project/stability/) · [保留的 LightGBM 示例](https://leeesq.github.io/mars-risk/demos/history/)。

## 复现与贡献

```bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
pip install -e ".[docs]"
python docs/snippets/task_cases.py --case all --output-dir output/task-cases
```

[案例下载与复现说明](https://leeesq.github.io/mars-risk/demos/#run) · [贡献指南](CONTRIBUTING.md) · [工程 Skill](.codex/skills/mars-risk-engineering/SKILL.md) · [MIT License](LICENSE)。
