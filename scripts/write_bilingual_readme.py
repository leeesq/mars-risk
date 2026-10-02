"""从共同品牌、原生图和可运行代码生成完整中英文 README。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://leeesq.github.io/mars-risk/"
FENCE = chr(96) * 3


def _quickstart() -> str:
    """读取共享代码，避免两种语言维护不同 API 示例。"""
    source = (ROOT / "docs/snippets/readme_quickstart.py").read_text(encoding="utf-8")
    return source.split("# --8<-- [start:quickstart]\n", 1)[1].split(
        "# --8<-- [end:quickstart]", 1
    )[0].rstrip()


def _brand() -> str:
    """保留原字形、英文全称和六类徽章的真实链接。"""
    original = (ROOT / "docs/index.md").read_text(encoding="utf-8")
    badges = original.split('<p class="mars-home-badges">', 1)[1].split("</p>", 1)[0]
    badges = badges.replace(
        'href="https://github.com/leeesq/mars-risk/blob/main/LICENSE"', 'href="LICENSE"'
    )
    return (
        '<p align="center">\n'
        '<picture>\n'
        '<source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/mars-logo.svg">\n'
        '<source media="(prefers-color-scheme: dark)" srcset="docs/assets/mars-logo-dark.gif">\n'
        '<img src="docs/assets/mars-logo-light.gif" alt="MARS" width="480">\n'
        '</picture>\n<br>\n'
        '<img src="docs/assets/mars-wordmark.svg" '
        'alt="MODELING ANALYSIS RISK SCORE" width="480">\n</p>\n'
        '<p align="center">' + badges + '</p>'
    )


def _hero(english: bool) -> str:
    """为 GitHub 使用显式 HTML 居中，不依赖文档站 CSS。"""
    title = (
        "A high-performance credit risk toolkit for humans and AI agents"
        if english else "面向人和 AI Agent 的高性能信贷风控工具箱"
    )
    capabilities = (
        "Data profiling · Binning evaluation · Feature selection · Correlation analysis · "
        "Score cross analysis · Rule mining"
        if english else
        "数据画像 · 分箱评估 · 特征筛选 · 相关性分析 · 模型分交叉 · 规则挖掘"
    )
    description = (
        "Turn Pandas or Polars data into risk evidence you can inspect, query, deliver and reuse.<br>"
        "Native charts for analysts. Structured reports for your modeling, monitoring and analysis agents."
        if english else
        "从 Pandas 或 Polars 宽表，得到可查看、可查询、可交付、可复用的风控分析结果。<br>"
        "用原生图表支持人工判断，用结构化报告连接你的建模、监控归因与分析 Agent。"
    )
    links = (
        [("Start analyzing", "getting-started/quickstart/"), ("Practical cases", "demos/"),
         ("Build with AI agents", "user-guide/external-agents/"), ("Documentation · 中文", "")]
        if english else
        [("开始分析", "getting-started/quickstart/"), ("实战案例", "demos/"),
         ("面向 AI Agent", "user-guide/external-agents/"), ("完整文档", "")]
    )
    navigation = " · ".join(f'<a href="{SITE}{path}">{label}</a>' for label, path in links)
    return (
        _brand() + f'\n\n<h1 align="center">{title}</h1>\n'
        f'<p align="center"><strong>{capabilities}</strong></p>\n'
        f'<p align="center">{description}</p>\n'
        f'<p align="center">{navigation}</p>'
    )


def _preview(english: bool) -> str:
    """直接展示原生分箱图，并链接未缩放的矢量图。"""
    alt = (
        "MARS native binning trends: fixed bins, distributions, bad rate, amount risk and stability"
        if english else "MARS 原生分箱趋势图：固定分箱、样本分布、件数与金额 bad rate、稳定性"
    )
    caption = (
        "Native MARS binning chart, exported from the same analysis as its tables. Click for the full-resolution SVG."
        if english else "MARS 原生分箱趋势图，直接导出自分析报告；图表与统计表共享同一计算结果。点击查看高清 SVG。"
    )
    case_label = "Binning and stability case" if english else "分箱与稳定性实战"
    html_label = "Open the interactive report" if english else "打开交互式报告"
    return (
        '<p align="center">\n'
        f'<a href="{SITE}assets/cases/binning-native-main-score.svg">\n'
        '<img src="docs/assets/cases/binning-native-main-score.png" '
        f'alt="{alt}" width="1040">\n</a>\n</p>\n\n'
        f'<p align="center"><sub>{caption}</sub><br>\n'
        f'<a href="{SITE}demos/binning-stability/">{case_label}</a> · '
        f'<a href="{SITE}assets/cases/binning.html">{html_label}</a></p>'
    )


def _readme(english: bool) -> str:
    """生成完整使用者介绍，接口代码与两个语言版本保持同源。"""
    hero = _hero(english)
    preview = _preview(english)
    code = _quickstart()
    agent = """from mars.reporting import load_report

report = load_report("risk_report.marsreport")
description = report.describe()
page = report.query_page("summary", limit=1)
next_page = report.query_page("summary", offset=page["next_offset"], limit=1)
context_json = report.to_ai_context(tables=["summary"], max_chars=16000)"""
    if english:
        return f"""[中文](README.md) / **English**

{hero}

{preview}

## What MARS brings to your workflow

MARS uses Polars for calculation and accepts Pandas or Polars wide tables. It brings data profiling, binning, risk evaluation, feature selection, correlation and rule analysis into a shared set of reports. Inspect distributions, compare populations and periods, understand selection decisions, then deliver the results as charts, HTML or Excel.

The same results are available to **external AI agents** through public report APIs. Tables, metric semantics, feature metadata, calculation states and replayable evidence remain queryable after saving and loading. Your agent can use MARS as its analysis layer and combine it with the model libraries, experiment tools and business workflow you choose.

| Capability | What you can inspect | Practical example |
| --- | --- | --- |
| Data profiling | Missing and special values, sample and label coverage, distributions | [Check whether the data is ready]({SITE}demos/data-quality/) |
| Binning and risk evaluation | Native trend charts, fixed bins, multi-target IV / KS / PSI, count and amount risk | [Compare discrimination and stability]({SITE}demos/binning-stability/) |
| Feature selection | Candidate decisions and reasons across screening stages | [Understand retained and removed features]({SITE}demos/selection-correlation/) |
| Correlation analysis | Signed redundancy and separate raw / WOE evidence | [Inspect related features]({SITE}demos/selection-correlation/#correlation) |
| Score cross analysis | Risk separation within a main-score tier and combination evidence | [Compare auxiliary scores]({SITE}demos/score-cross/) |
| Rule mining · Experimental | Candidate origins, screening, coverage and independent validation | [Review rules with evidence]({SITE}demos/rule-evidence/) |

[All seven cases]({SITE}demos/) use 18,000 synthetic applications with separate discovery, validation and observation partitions. Charts and Agent queries consume the actual reports. The statistics demonstrate the interfaces and do not establish real lending outcomes.

## Build your agents on MARS

MARS provides the calculation and evidence layer. You supply orchestration, model training and business decisions.

| Your external agent | MARS building blocks | Useful output |
| --- | --- | --- |
| Modeling agent | Data quality, binning, feature selection, correlation and risk evaluation | Feature candidates, removal reasons and evaluation evidence for your chosen training tools |
| Monitoring and diagnosis agent | Grouped profiles, missing and distribution changes, fixed-bin risk trends and stability | Locate changing populations or features, gather evidence and propose hypotheses to validate |
| Rule analysis agent | Candidate rules, hit evaluation, discovery audit and independent validation | Reviewable candidates with risk, coverage and validation results |
| Report analysis agent | Report catalog, business metadata, bounded queries, snapshots and evidence references | Continue an analysis across sessions and deliver traceable answers |

These are workflows you can build with MARS; complete agents are composed by the user. The optional <code>mars.agent</code> integration also consumes the public report interface. [External Agent guide · 中文]({SITE}user-guide/external-agents/) · [Rule evidence guide · 中文]({SITE}user-guide/rule-reports-and-agents/).

## Install

The examples follow **main / source version 0.0.28**. Install the current source:

{FENCE}bash
pip install "git+https://github.com/leeesq/mars-risk.git"
{FENCE}

PyPI currently publishes **0.0.27**, which does not contain all capabilities shown here. The base package supports Python **3.8–3.12**; Python 3.8 uses a frozen dependency stack. Use **Python 3.10+** for optional modeling, tuning, notebooks, documentation development and the built-in Agent SDK. [Installation and extras · 中文]({SITE}getting-started/installation/).

## Start with one report

Run this self-contained example in an empty working directory:

{FENCE}python
{code}
{FENCE}

[Shared runnable source](docs/snippets/readme_quickstart.py) · [Full quickstart · 中文]({SITE}getting-started/quickstart/). The eight-row sample explains the API. For a full trend chart and downloadable results, use the [binning case]({SITE}demos/binning-stability/). Native figures can be saved directly as SVG or high-resolution PNG with <code>report.save_risk_trend_images()</code>.

## Give the same result to an AI agent

In a new process, load the saved report and query only the evidence you need:

{FENCE}python
{agent}
{FENCE}

<code>describe()</code> exposes the real table catalog, semantics, parameters and states. <code>query_page()</code> returns rows, pagination and a replayable evidence reference. <code>to_ai_context()</code> produces bounded JSON; **<code>max_chars</code> counts characters, not tokens**. The consumer does not need an LLM, API key, <code>MarsAgentSession</code>, original table or analyzer merely to query the saved results.

Loading returns a **ReportSnapshot** with saved statistics and metadata. <code>report.write_excel("risk_report.xlsx")</code> exports static tables, and <code>report.show_table("summary", limit=10)</code> displays them. Questions requiring new dimensions or original records need a new calculation. [Save and query results]({SITE}demos/saved-reports/) · [Deliver HTML, Excel, snapshots and Agent materials]({SITE}demos/report-delivery/).

## Development focus

**Analysis, Feature and Reporting are Stable. Rule, Monitoring, Modeling, Pipeline, Scoring and Agent are Experimental.** These labels describe interface maturity.

Current development focuses on core analysis performance, memory efficiency, human use and external Agent consumption. **Monitoring, Modeling (including Pipeline) and Scoring are paused for feature development.** Existing functionality and documentation remain, with necessary fixes and direct adaptation to upstream changes. Combine AI, MARS analysis and your own model tools to build tailored workflows. These downstream modules adapt to the core; upstream code does not retain compatibility aliases or redundant computation for them.

[Stability policy · 中文]({SITE}project/stability/) · [Retained LightGBM example · 中文]({SITE}demos/history/).

## Reproduce and contribute

{FENCE}bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
pip install -e ".[docs]"
python docs/snippets/task_cases.py --case all --output-dir output/task-cases
{FENCE}

[Case downloads and reproducibility]({SITE}demos/#run) · [Contribution guide](CONTRIBUTING.md) · [Engineering Skill](.codex/skills/mars-risk-engineering/SKILL.md) · [MIT License](LICENSE).
"""
    return f"""**中文** / [English](README.en.md)

{hero}

{preview}

## MARS 能帮你做什么

MARS 以 Polars 为计算基础，接受 Pandas 或 Polars 宽表，将数据画像、分箱风险评估、特征筛选、相关性和规则分析组织为统一的分析报告。你可以查看分布与风险趋势，比较客群和时间，追溯特征保留与删除的原因，再将结果交付为图表、HTML 或 Excel。

同一份结果也能交给 **外部 AI Agent**。公共报告接口保留表目录、指标口径、特征业务元数据、计算状态和可重放证据，保存后仍能按需查询。Agent 可以把 MARS 作为分析计算层，结合你选择的模型库、实验工具和业务流程，持续提出问题、读取证据和完成分析。

| 核心能力 | 可以看到什么 | 实战入口 |
| --- | --- | --- |
| 数据画像 | 缺失与特殊值、样本与标签覆盖、分布变化 | [判断数据是否可以使用]({SITE}demos/data-quality/) |
| 分箱与风险评估 | 原生趋势图、固定分箱、多目标 IV / KS / PSI、件数与金额风险 | [比较区分度与稳定性]({SITE}demos/binning-stability/) |
| 特征筛选 | 候选特征、筛选步骤、保留与删除理由 | [理解特征取舍]({SITE}demos/selection-correlation/) |
| 相关性分析 | 带符号的冗余关系，分别查看 raw / WOE 证据 | [检查特征之间的关系]({SITE}demos/selection-correlation/#correlation) |
| 模型分交叉 | 同一主等级内的风险分离与组合证据 | [比较辅助模型分]({SITE}demos/score-cross/) |
| 规则挖掘 · Experimental | 候选来源、筛选、覆盖与独立验证 | [用证据审查规则]({SITE}demos/rule-evidence/) |

[七个实战案例]({SITE}demos/)共用 18,000 行合成申请，发现、验证、观察分区独立。人工图表和 Agent 查询读取实际报告；合成统计用于展示能力，不代表真实信贷业务效果。

## 基于 MARS，构建你的 Agent

MARS 提供计算与证据能力，你可以自由组合 Agent 编排、模型训练工具和业务决策流程。

| 外部 Agent 场景 | 可复用的 MARS 能力 | 可以形成的结果 |
| --- | --- | --- |
| 建模 Agent | 数据质量、分箱评估、特征筛选、相关性、风险评估 | 特征候选与淘汰理由，交给自选训练工具的评估证据 |
| 监控与归因 Agent | 分组画像、缺失和分布变化、固定分箱风险趋势、稳定性 | 定位变化人群与特征，汇总证据，提出待验证的归因假设 |
| 规则分析 Agent | 候选规则、命中评估、发现审计与独立验证 | 带风险、覆盖与验证结果的可审查候选 |
| 报告分析 Agent | 表目录、业务元数据、有限查询、快照与证据引用 | 跨会话继续分析，交付可追溯的回答 |

这些是可以基于 MARS 构建的工作流，完整 Agent 由使用者组合实现。可选 <code>mars.agent</code> 也复用公共报告接口。[外部 Agent 使用指南]({SITE}user-guide/external-agents/) · [规则证据与 Agent]({SITE}user-guide/rule-reports-and-agents/)。

## 安装

本页示例对应 **main / 源码版本 0.0.28**，使用当前源码安装：

{FENCE}bash
pip install "git+https://github.com/leeesq/mars-risk.git"
{FENCE}

PyPI 当前发布 **0.0.27**，尚不包含这里展示的全部新能力。基础包支持 Python **3.8–3.12**，3.8 使用冻结依赖栈；可选建模、调参、Notebook、文档开发与内置 Agent SDK 使用 **Python 3.10+**。[安装与可选依赖]({SITE}getting-started/installation/)。

## 先得到一份报告

在空工作目录运行这个独立示例：

{FENCE}python
{code}
{FENCE}

[共享可运行源码](docs/snippets/readme_quickstart.py) · [完整 Quickstart]({SITE}getting-started/quickstart/)。八行小样本用于说明接口；完整趋势图和可下载结果见[分箱实战]({SITE}demos/binning-stability/)。原生图可以直接通过 <code>report.save_risk_trend_images()</code> 保存为 SVG 或高清 PNG。

## 将同一份结果交给 AI Agent

在新进程加载已保存的报告，按需读取证据：

{FENCE}python
{agent}
{FENCE}

<code>describe()</code> 提供真实表目录、口径、参数与状态；<code>query_page()</code> 返回结果、分页和可重放证据引用；<code>to_ai_context()</code> 生成有限 JSON，**<code>max_chars</code> 是字符预算，不是 token 数**。查询已有结果无需 LLM、API Key、<code>MarsAgentSession</code>、原始宽表或分析器。

加载得到 **ReportSnapshot**，包含已保存统计和元数据。<code>report.write_excel("risk_report.xlsx")</code> 导出静态表，<code>report.show_table("summary", limit=10)</code> 用于展示。需要新维度或原始记录的问题，应发起新的计算。[保存后继续查询]({SITE}demos/saved-reports/) · [交付 HTML、Excel、快照与 Agent 材料]({SITE}demos/report-delivery/)。

## 当前迭代方向

**Analysis、Feature、Reporting 为 Stable；Rule、Monitoring、Modeling、Pipeline、Scoring、Agent 为 Experimental。** 这些标记说明接口成熟度。

当前重点改进核心分析性能、内存效率、人工体验与外部 Agent 消费能力。**监控、建模（含 Pipeline）、评分卡暂时停止功能迭代**，现有功能和文档保留，处理必要修复与上游适配。你可以借助 AI、MARS 分析能力和自选模型工具构建更定制的流程。三个下游直接适配核心代码，上游不为它们保留兼容别名或冗余计算。

[稳定性与适配原则]({SITE}project/stability/) · [保留的 LightGBM 示例]({SITE}demos/history/)。

## 复现与贡献

{FENCE}bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
pip install -e ".[docs]"
python docs/snippets/task_cases.py --case all --output-dir output/task-cases
{FENCE}

[案例下载与复现说明]({SITE}demos/#run) · [贡献指南](CONTRIBUTING.md) · [工程 Skill](.codex/skills/mars-risk-engineering/SKILL.md) · [MIT License](LICENSE)。
"""


def main() -> None:
    """同步两个 README，图片由原生报告导出流程生成。"""
    for name, english in (("README.md", False), ("README.en.md", True)):
        (ROOT / name).write_text(_readme(english), encoding="utf-8")


if __name__ == "__main__":
    main()
