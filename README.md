**中文** / [English](https://github.com/leeesq/mars-risk/blob/main/README.en.md)

<p align="center">
<picture>
<source media="(prefers-reduced-motion: reduce)" srcset="https://leeesq.github.io/mars-risk/assets/mars-logo.svg">
<source media="(prefers-color-scheme: dark)" srcset="https://leeesq.github.io/mars-risk/assets/mars-logo-dark.gif">
<img src="https://leeesq.github.io/mars-risk/assets/mars-logo-light.gif" alt="MARS" width="480">
</picture>
<br>
<img src="https://leeesq.github.io/mars-risk/assets/mars-wordmark.svg" alt="MODELING ANALYSIS RISK SCORE" width="480">
</p>
<p align="center">
  <a href="https://pypi.org/project/mars-risk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mars-risk?style=flat-square&label=PyPI&color=2f6f8f"></a>
  <a href="https://leeesq.github.io/mars-risk/"><img alt="Docs" src="https://img.shields.io/badge/Docs-GitHub%20Pages-7c3aed?style=flat-square"></a>
  <a href="https://pypi.org/project/mars-risk/"><img alt="Python" src="https://img.shields.io/badge/Python-3.8--3.12-364f6b?style=flat-square"></a>
  <a href="https://pepy.tech/project/mars-risk"><img alt="Downloads" src="https://img.shields.io/pepy/dt/mars-risk?style=flat-square&label=Downloads&color=0f766e"></a>
  <a href="https://github.com/leeesq/mars-risk/actions/workflows/test.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/leeesq/mars-risk/test.yml?branch=main&style=flat-square&label=CI&color=1f7a5a"></a>
  <a href="https://github.com/leeesq/mars-risk/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/leeesq/mars-risk?style=flat-square&label=License&color=6c5ce7"></a>
</p>

<h1 align="center">面向人和 AI Agent 的高性能信贷风控工具箱</h1>
<p align="center"><strong>数据画像 · 分箱评估 · 特征筛选 · 相关性分析 · 模型分交叉 · 规则挖掘</strong></p>
<p align="center">用 Pandas 或 Polars 宽表分析信贷风险，查看图表、查询统计并导出报告。<br>分析人员用原生图表判断风险，AI Agent 从同一份报告中查询证据。</p>
<p align="center"><a href="https://leeesq.github.io/mars-risk/getting-started/quickstart/">开始分析</a> · <a href="https://leeesq.github.io/mars-risk/demos/">实战案例</a> · <a href="https://leeesq.github.io/mars-risk/user-guide/external-agents/">面向 AI Agent</a> · <a href="https://leeesq.github.io/mars-risk/">完整文档</a></p>

<p align="center">
<a href="https://leeesq.github.io/mars-risk/assets/cases/binning-native-home-credit.svg">
<img src="https://leeesq.github.io/mars-risk/assets/cases/binning-native-home-credit.png" alt="MARS 原生分箱趋势图：固定分箱、样本分布、件数与金额 bad rate、稳定性" width="1040">
</a>
</p>

<p align="center"><sub>Home Credit 外部来源分 EXT_SOURCE_2，按 Train 固定分箱原生导出；APPLY_DATE 为模拟日期。点击查看高清 SVG。</sub><br>
<a href="https://leeesq.github.io/mars-risk/demos/binning-stability/">分箱与稳定性实战</a> · <a href="https://leeesq.github.io/mars-risk/assets/cases/binning.html">打开交互式报告</a></p>

## MARS 能帮你做什么

MARS 使用 Polars 计算，接受 Pandas 或 Polars 宽表。你可以检查数据质量，评估分箱后的风险，筛选特征，分析相关性并挖掘规则。分析结果保存在报告中，供你比较不同客群和时间的分布与风险，查看特征保留和删除的原因，并导出图表、HTML 或 Excel。

外部 AI Agent 通过公共报告接口查询同一份结果。报告会保存表目录、指标口径、特征业务元数据和计算状态，查询会返回可重放的证据引用。Agent 可以据此继续分析，也可以调用你选择的模型库和实验工具。

| 核心能力 | 可以看到什么 | 实战入口 |
| --- | --- | --- |
| 数据画像 | 缺失与特殊值、样本与标签覆盖、分布变化 | [判断数据是否可以使用](https://leeesq.github.io/mars-risk/demos/data-quality/) |
| 分箱与风险评估 | 原生趋势图、固定分箱、多目标 IV / KS / PSI、件数与金额风险 | [比较区分度与稳定性](https://leeesq.github.io/mars-risk/demos/binning-stability/) |
| 特征筛选 | 候选特征、筛选步骤、保留与删除理由 | [理解特征取舍](https://leeesq.github.io/mars-risk/demos/selection-correlation/) |
| 相关性分析 | 查看相关系数的正负，分别比较原始值（raw）和 WOE 结果 | [检查特征之间的关系](https://leeesq.github.io/mars-risk/demos/selection-correlation/#correlation) |
| 模型分交叉 | 将两项外部来源分交叉分组，比较各组风险和组合条件 | [比较外部来源分](https://leeesq.github.io/mars-risk/demos/score-cross/) |
| 规则挖掘 · Experimental | 候选来源、筛选、覆盖与独立验证 | [用证据审查规则](https://leeesq.github.io/mars-risk/demos/rule-evidence/) |

[七个实战案例](https://leeesq.github.io/mars-risk/demos/)共用 Home Credit 的全部 356,255 条申请。原始特征与 TARGET 来自比赛，APPLY_DATE 为模拟日期；四个有标签分区之后，保留原比赛 test 的 48,744 条无标签记录。[输入与五分区说明](https://leeesq.github.io/mars-risk/user-guide/home-credit-data/)解释样本用途与日期边界。

## 基于 MARS，构建你的 Agent

MARS 负责计算并提供分析证据。你负责组织 Agent 的操作步骤，选择训练工具，并制定业务决策流程。

| 外部 Agent 场景 | 可复用的 MARS 能力 | 可以形成的结果 |
| --- | --- | --- |
| 建模 Agent | 数据质量、分箱评估、特征筛选、相关性、风险评估 | 选出候选特征，查看淘汰理由，将评估结果交给自选训练工具 |
| 监控与归因 Agent | 分组画像、缺失和分布变化、固定分箱风险趋势、稳定性 | 定位变化人群与特征，汇总证据，提出待验证的归因假设 |
| 规则分析 Agent | 候选规则、命中评估、发现审计与独立验证 | 审查候选规则的风险、覆盖范围和独立验证结果 |
| 报告分析 Agent | 表目录、业务元数据、分页查询、快照与证据引用 | 在新会话继续查询，通过证据引用核对回答 |

你需要自行实现这些 Agent 工作流。MARS 也提供可选的 <code>mars.agent</code>，它同样使用公共报告接口。[外部 Agent 使用指南](https://leeesq.github.io/mars-risk/user-guide/external-agents/) · [规则证据与 Agent](https://leeesq.github.io/mars-risk/user-guide/rule-reports-and-agents/)。

## 安装

**MARS 0.0.28 已发布到 PyPI**，可安装固定版本：

```bash
pip install mars-risk==0.0.28
```

需要当前开发分支时，从源码安装：

```bash
pip install "git+https://github.com/leeesq/mars-risk.git"
```

基础包支持 Python **3.8～3.12**，其中 Python 3.8 使用固定版本的依赖。可选建模、调参、Notebook、文档开发与内置 Agent SDK 使用 **Python 3.10+**。[安装与可选依赖](https://leeesq.github.io/mars-risk/getting-started/installation/)。

## 先得到一份报告

先按[数据准备指南](https://leeesq.github.io/mars-risk/user-guide/home-credit-data/)下载三个官方 CSV，放到仓库的 `examples/home-credit-2018/data/raw_data/`。在仓库根目录运行：

```bash
python docs/snippets/readme_quickstart.py
```

脚本执行以下代码。单独复制代码时，将 [home_credit_data.py](https://github.com/leeesq/mars-risk/blob/main/docs/snippets/home_credit_data.py) 放在脚本旁，并按指南准备数据路径。

```python
from home_credit_data import prepare_home_credit_data

from mars.analysis import profile_risk

inputs = prepare_home_credit_data()
report = profile_risk(
    inputs.df, target="TARGET", features=["EXT_SOURCE_2", "EXT_SOURCE_3"],
    benchmark_df=inputs.benchmark_df, group_col="dataset_flag", time_col="APPLY_DATE",
    amount_col="AMT_CREDIT", method="quantile", n_bins=4,
    feature_metadata=inputs.feature_metadata, business_context=inputs.business_context,
).report
report.save_risk_trend_images(
    "risk_charts", features="EXT_SOURCE_2", target="TARGET", image_format="png", dpi=300,
)
report.write_html("risk_report.html", chart_embed_mode="inline")
report.save("risk_report.marsreport")
print(report.query_page("summary", features="EXT_SOURCE_2", limit=1)["data"])
```

运行后，当前目录会生成 `risk_charts/`、`risk_report.html` 和 `risk_report.marsreport`，终端会打印一行汇总数据。准备函数统一读取两个 application 文件，并补充字段说明、模拟日期和分区。你也可以用 <code>report.save_risk_trend_images()</code> 将原生图保存为 SVG 或高清 PNG。

[共享可运行源码](https://github.com/leeesq/mars-risk/blob/main/docs/snippets/readme_quickstart.py) · [完整 Quickstart](https://leeesq.github.io/mars-risk/getting-started/quickstart/) · [完整趋势图与下载结果](https://leeesq.github.io/mars-risk/demos/binning-stability/)。

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

先用 <code>describe()</code> 查看表目录、指标口径、参数和状态，再用 <code>query_page()</code> 分页查询。每页包含数据和可重放的证据引用。<code>to_ai_context()</code> 将指定表整理为 JSON，并限制输出长度；**<code>max_chars</code> 按字符计数，不按 token 计数**。查询已有结果无需 LLM、API Key、<code>MarsAgentSession</code>、原始宽表或分析器。

加载后得到 ReportSnapshot，其中包含已保存的统计结果和元数据。调用 <code>report.write_excel("risk_report.xlsx")</code> 可导出静态表，调用 <code>report.show_table("summary", limit=10)</code> 可查看汇总表。快照只能回答已有统计支持的问题；需要新增分析维度或查看原始记录时，必须读取原数据重新计算。[保存后继续查询](https://leeesq.github.io/mars-risk/demos/saved-reports/) · [交付 HTML、Excel、快照与 Agent 材料](https://leeesq.github.io/mars-risk/demos/report-delivery/)。

## 当前迭代方向

**Analysis、Feature、Reporting 为 Stable；Rule、Monitoring、Modeling、Pipeline、Scoring、Agent 为 Experimental。** 这些标记说明接口成熟度。

当前重点是提高核心分析的计算速度、减少内存占用，并改进人工查看和外部 Agent 查询报告的体验。**监控、建模（含 Pipeline）、评分卡暂时停止功能迭代**。我们保留现有功能和文档，处理必要修复，并在核心接口变化时直接修改这三个下游模块的调用代码。核心代码不为它们保留兼容别名或冗余计算。

你可以结合 AI、MARS 分析报告和自选模型工具，构建适合业务的流程。

[稳定性与适配原则](https://leeesq.github.io/mars-risk/project/stability/) · [保留的 LightGBM 示例](https://leeesq.github.io/mars-risk/demos/history/)。

## 复现与贡献

```bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
pip install -e ".[docs]"
python docs/snippets/task_cases.py --case all --data-dir examples/home-credit-2018/data/raw_data --output-dir output/task-cases
```

[案例下载与复现说明](https://leeesq.github.io/mars-risk/demos/#run) · [贡献指南](https://github.com/leeesq/mars-risk/blob/main/CONTRIBUTING.md) · [工程 Skill](https://github.com/leeesq/mars-risk/blob/main/.codex/skills/mars-risk-engineering/SKILL.md) · [MIT License](https://github.com/leeesq/mars-risk/blob/main/LICENSE)。
