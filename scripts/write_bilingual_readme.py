"""从共同品牌、原生图和可运行代码生成完整中英文 README。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://leeesq.github.io/mars-risk/"
SOURCE = "https://github.com/leeesq/mars-risk/blob/main/"
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
    return (
        '<p align="center">\n'
        '<picture>\n'
        f'<source media="(prefers-reduced-motion: reduce)" srcset="{SITE}assets/mars-logo.svg">\n'
        f'<source media="(prefers-color-scheme: dark)" srcset="{SITE}assets/mars-logo-dark.gif">\n'
        f'<img src="{SITE}assets/mars-logo-light.gif" alt="MARS" width="480">\n'
        '</picture>\n<br>\n'
        f'<img src="{SITE}assets/mars-wordmark.svg" '
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
        "Analyze credit risk from Pandas or Polars tables, inspect charts, query statistics and export reports.<br>"
        "Analysts use native charts to assess risk. AI agents query evidence from the same reports."
        if english else
        "用 Pandas 或 Polars 宽表分析信贷风险，查看图表、查询统计并导出报告。<br>"
        "分析人员用原生图表判断风险，AI Agent 从同一份报告中查询证据。"
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
        "Home Credit EXT_SOURCE_2, exported by MARS from fixed Train bins. APPLY_DATE is simulated. Click for the full-resolution SVG."
        if english else "Home Credit 外部来源分 EXT_SOURCE_2，按 Train 固定分箱原生导出；APPLY_DATE 为模拟日期。点击查看高清 SVG。"
    )
    case_label = "Binning and stability case" if english else "分箱与稳定性实战"
    html_label = "Open the interactive report" if english else "打开交互式报告"
    return (
        '<p align="center">\n'
        f'<a href="{SITE}assets/cases/binning-native-home-credit.svg">\n'
        f'<img src="{SITE}assets/cases/binning-native-home-credit.png" '
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
        return f"""[中文]({SOURCE}README.md) / **English**

{hero}

{preview}

## What MARS brings to your workflow

MARS uses Polars for computation and accepts Pandas or Polars wide tables. Check data quality, evaluate risk across bins, select features, analyze correlations and mine rules. Reports retain the results so you can compare distributions and risk across populations and periods, inspect selection decisions, and export charts, HTML or Excel.

External AI agents query the same results through public report APIs. Reports preserve the table catalog, metric definitions, feature metadata and calculation states. Queries return replayable evidence references. Your agent can use these results to continue an analysis and call the model libraries and experiment tools you choose.

| Capability | What you can inspect | Practical example |
| --- | --- | --- |
| Data profiling | Missing and special values, sample and label coverage, distributions | [Check whether the data is ready]({SITE}demos/data-quality/) |
| Binning and risk evaluation | Native trend charts, fixed bins, multi-target IV / KS / PSI, count and amount risk | [Compare discrimination and stability]({SITE}demos/binning-stability/) |
| Feature selection | Candidate decisions and reasons across screening stages | [Understand retained and removed features]({SITE}demos/selection-correlation/) |
| Correlation analysis | Positive and negative correlations, with separate raw-value and WOE results | [Inspect related features]({SITE}demos/selection-correlation/#correlation) |
| Score cross analysis | Risk across joint segments of two external-source scores and their combination conditions | [Compare external-source scores]({SITE}demos/score-cross/) |
| Rule mining · Experimental | Candidate origins, screening, coverage and independent validation | [Review rules with evidence]({SITE}demos/rule-evidence/) |

[All seven cases]({SITE}demos/) share all 356,255 Home Credit application records. Original features and TARGET come from the competition; APPLY_DATE is simulated. Four labeled partitions precede the 48,744 unlabeled competition-test records. [Inputs and five partitions]({SITE}user-guide/home-credit-data/) explain their uses.

## Build your agents on MARS

MARS performs calculations and provides analysis evidence. You define the agent's steps, choose training tools and set the business decision process.

| Your external agent | MARS building blocks | Useful output |
| --- | --- | --- |
| Modeling agent | Data quality, binning, feature selection, correlation and risk evaluation | Select candidate features, inspect removal reasons and pass evaluation results to your chosen training tools |
| Monitoring and diagnosis agent | Grouped profiles, missing and distribution changes, fixed-bin risk trends and stability | Locate changing populations or features, gather evidence and propose hypotheses to validate |
| Rule analysis agent | Candidate rules, hit evaluation, discovery audit and independent validation | Review each candidate's risk, coverage and independent validation results |
| Report analysis agent | Report catalog, business metadata, paginated queries, snapshots and evidence references | Continue querying in a new session and check answers against evidence references |

You implement these agent workflows. MARS also offers an optional <code>mars.agent</code> integration that uses the same public report interface. [External Agent guide · 中文]({SITE}user-guide/external-agents/) · [Rule evidence guide · 中文]({SITE}user-guide/rule-reports-and-agents/).

## Install

**MARS 0.0.28 is published on PyPI**. Install the fixed version:

{FENCE}bash
pip install mars-risk==0.0.28
{FENCE}

For the current development branch, install from source:

{FENCE}bash
pip install "git+https://github.com/leeesq/mars-risk.git"
{FENCE}

The base package supports Python **3.8-3.12**, with pinned dependencies for Python 3.8. Use **Python 3.10+** for optional modeling, tuning, notebooks, documentation development and the built-in Agent SDK. [Installation and extras · 中文]({SITE}getting-started/installation/).

## Start with one report

Follow the [data preparation guide]({SITE}user-guide/home-credit-data/) to download the three official CSV files into `examples/home-credit-2018/data/raw_data/`. From the repository root, run:

{FENCE}bash
python docs/snippets/readme_quickstart.py
{FENCE}

The script runs the code below. If copying it elsewhere, place [home_credit_data.py]({SOURCE}docs/snippets/home_credit_data.py) beside your script and prepare the data paths as described in the guide.

{FENCE}python
{code}
{FENCE}

The script creates `risk_charts/`, `risk_report.html` and `risk_report.marsreport` in the current directory, then prints one summary row. The preparation function reads both application files and adds field descriptions, simulated dates and partitions. You can also use <code>report.save_risk_trend_images()</code> to save native figures as SVG or high-resolution PNG.

[Shared runnable source]({SOURCE}docs/snippets/readme_quickstart.py) · [Full quickstart · 中文]({SITE}getting-started/quickstart/) · [Full trend charts and downloads]({SITE}demos/binning-stability/).

## Give the same result to an AI agent

In a new process, load the saved report and query only the evidence you need:

{FENCE}python
{agent}
{FENCE}

Start with <code>describe()</code> to inspect the table catalog, metric definitions, parameters and states, then use <code>query_page()</code> to read pages of results. Each page includes data and a replayable evidence reference. <code>to_ai_context()</code> produces JSON from the selected tables with a length limit; **<code>max_chars</code> counts characters, not tokens**. Querying saved results requires no LLM, API key, <code>MarsAgentSession</code>, original table or analyzer.

Loading returns a ReportSnapshot containing saved statistics and metadata. Use <code>report.write_excel("risk_report.xlsx")</code> to export static tables or <code>report.show_table("summary", limit=10)</code> to display the summary. A snapshot can answer only questions supported by its saved statistics. New analysis dimensions or questions about original records require the source data and a new calculation. [Save and query results]({SITE}demos/saved-reports/) · [Deliver HTML, Excel, snapshots and Agent materials]({SITE}demos/report-delivery/).

## Development focus

**Analysis, Feature and Reporting are Stable. Rule, Monitoring, Modeling, Pipeline, Scoring and Agent are Experimental.** These labels describe interface maturity.

Current development focuses on faster core analysis, lower memory use, and better report viewing and querying for people and external agents. **Monitoring, Modeling (including Pipeline) and Scoring are paused for feature development.** We retain existing functionality and documentation, make necessary fixes, and update these downstream modules directly when core interfaces change. Core code does not retain compatibility aliases or redundant computation for them.

Combine AI, MARS analysis reports and your chosen model tools to build workflows for your business.

[Stability policy · 中文]({SITE}project/stability/) · [Retained LightGBM example · 中文]({SITE}demos/history/).

## Reproduce and contribute

{FENCE}bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
pip install -e ".[docs]"
python docs/snippets/task_cases.py --case all --data-dir examples/home-credit-2018/data/raw_data --output-dir output/task-cases
{FENCE}

[Case downloads and reproducibility]({SITE}demos/#run) · [Contribution guide]({SOURCE}CONTRIBUTING.md) · [Engineering Skill]({SOURCE}.codex/skills/mars-risk-engineering/SKILL.md) · [MIT License]({SOURCE}LICENSE).
"""
    return f"""**中文** / [English]({SOURCE}README.en.md)

{hero}

{preview}

## MARS 能帮你做什么

MARS 使用 Polars 计算，接受 Pandas 或 Polars 宽表。你可以检查数据质量，评估分箱后的风险，筛选特征，分析相关性并挖掘规则。分析结果保存在报告中，供你比较不同客群和时间的分布与风险，查看特征保留和删除的原因，并导出图表、HTML 或 Excel。

外部 AI Agent 通过公共报告接口查询同一份结果。报告会保存表目录、指标口径、特征业务元数据和计算状态，查询会返回可重放的证据引用。Agent 可以据此继续分析，也可以调用你选择的模型库和实验工具。

| 核心能力 | 可以看到什么 | 实战入口 |
| --- | --- | --- |
| 数据画像 | 缺失与特殊值、样本与标签覆盖、分布变化 | [判断数据是否可以使用]({SITE}demos/data-quality/) |
| 分箱与风险评估 | 原生趋势图、固定分箱、多目标 IV / KS / PSI、件数与金额风险 | [比较区分度与稳定性]({SITE}demos/binning-stability/) |
| 特征筛选 | 候选特征、筛选步骤、保留与删除理由 | [理解特征取舍]({SITE}demos/selection-correlation/) |
| 相关性分析 | 查看相关系数的正负，分别比较原始值（raw）和 WOE 结果 | [检查特征之间的关系]({SITE}demos/selection-correlation/#correlation) |
| 模型分交叉 | 将两项外部来源分交叉分组，比较各组风险和组合条件 | [比较外部来源分]({SITE}demos/score-cross/) |
| 规则挖掘 · Experimental | 候选来源、筛选、覆盖与独立验证 | [用证据审查规则]({SITE}demos/rule-evidence/) |

[七个实战案例]({SITE}demos/)共用 Home Credit 的全部 356,255 条申请。原始特征与 TARGET 来自比赛，APPLY_DATE 为模拟日期；四个有标签分区之后，保留原比赛 test 的 48,744 条无标签记录。[输入与五分区说明]({SITE}user-guide/home-credit-data/)解释样本用途与日期边界。

## 基于 MARS，构建你的 Agent

MARS 负责计算并提供分析证据。你负责组织 Agent 的操作步骤，选择训练工具，并制定业务决策流程。

| 外部 Agent 场景 | 可复用的 MARS 能力 | 可以形成的结果 |
| --- | --- | --- |
| 建模 Agent | 数据质量、分箱评估、特征筛选、相关性、风险评估 | 选出候选特征，查看淘汰理由，将评估结果交给自选训练工具 |
| 监控与归因 Agent | 分组画像、缺失和分布变化、固定分箱风险趋势、稳定性 | 定位变化人群与特征，汇总证据，提出待验证的归因假设 |
| 规则分析 Agent | 候选规则、命中评估、发现审计与独立验证 | 审查候选规则的风险、覆盖范围和独立验证结果 |
| 报告分析 Agent | 表目录、业务元数据、分页查询、快照与证据引用 | 在新会话继续查询，通过证据引用核对回答 |

你需要自行实现这些 Agent 工作流。MARS 也提供可选的 <code>mars.agent</code>，它同样使用公共报告接口。[外部 Agent 使用指南]({SITE}user-guide/external-agents/) · [规则证据与 Agent]({SITE}user-guide/rule-reports-and-agents/)。

## 安装

**MARS 0.0.28 已发布到 PyPI**，可安装固定版本：

{FENCE}bash
pip install mars-risk==0.0.28
{FENCE}

需要当前开发分支时，从源码安装：

{FENCE}bash
pip install "git+https://github.com/leeesq/mars-risk.git"
{FENCE}

基础包支持 Python **3.8～3.12**，其中 Python 3.8 使用固定版本的依赖。可选建模、调参、Notebook、文档开发与内置 Agent SDK 使用 **Python 3.10+**。[安装与可选依赖]({SITE}getting-started/installation/)。

## 先得到一份报告

先按[数据准备指南]({SITE}user-guide/home-credit-data/)下载三个官方 CSV，放到仓库的 `examples/home-credit-2018/data/raw_data/`。在仓库根目录运行：

{FENCE}bash
python docs/snippets/readme_quickstart.py
{FENCE}

脚本执行以下代码。单独复制代码时，将 [home_credit_data.py]({SOURCE}docs/snippets/home_credit_data.py) 放在脚本旁，并按指南准备数据路径。

{FENCE}python
{code}
{FENCE}

运行后，当前目录会生成 `risk_charts/`、`risk_report.html` 和 `risk_report.marsreport`，终端会打印一行汇总数据。准备函数统一读取两个 application 文件，并补充字段说明、模拟日期和分区。你也可以用 <code>report.save_risk_trend_images()</code> 将原生图保存为 SVG 或高清 PNG。

[共享可运行源码]({SOURCE}docs/snippets/readme_quickstart.py) · [完整 Quickstart]({SITE}getting-started/quickstart/) · [完整趋势图与下载结果]({SITE}demos/binning-stability/)。

## 将同一份结果交给 AI Agent

在新进程加载已保存的报告，按需读取证据：

{FENCE}python
{agent}
{FENCE}

先用 <code>describe()</code> 查看表目录、指标口径、参数和状态，再用 <code>query_page()</code> 分页查询。每页包含数据和可重放的证据引用。<code>to_ai_context()</code> 将指定表整理为 JSON，并限制输出长度；**<code>max_chars</code> 按字符计数，不按 token 计数**。查询已有结果无需 LLM、API Key、<code>MarsAgentSession</code>、原始宽表或分析器。

加载后得到 ReportSnapshot，其中包含已保存的统计结果和元数据。调用 <code>report.write_excel("risk_report.xlsx")</code> 可导出静态表，调用 <code>report.show_table("summary", limit=10)</code> 可查看汇总表。快照只能回答已有统计支持的问题；需要新增分析维度或查看原始记录时，必须读取原数据重新计算。[保存后继续查询]({SITE}demos/saved-reports/) · [交付 HTML、Excel、快照与 Agent 材料]({SITE}demos/report-delivery/)。

## 当前迭代方向

**Analysis、Feature、Reporting 为 Stable；Rule、Monitoring、Modeling、Pipeline、Scoring、Agent 为 Experimental。** 这些标记说明接口成熟度。

当前重点是提高核心分析的计算速度、减少内存占用，并改进人工查看和外部 Agent 查询报告的体验。**监控、建模（含 Pipeline）、评分卡暂时停止功能迭代**。我们保留现有功能和文档，处理必要修复，并在核心接口变化时直接修改这三个下游模块的调用代码。核心代码不为它们保留兼容别名或冗余计算。

你可以结合 AI、MARS 分析报告和自选模型工具，构建适合业务的流程。

[稳定性与适配原则]({SITE}project/stability/) · [保留的 LightGBM 示例]({SITE}demos/history/)。

## 复现与贡献

{FENCE}bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
pip install -e ".[docs]"
python docs/snippets/task_cases.py --case all --data-dir examples/home-credit-2018/data/raw_data --output-dir output/task-cases
{FENCE}

[案例下载与复现说明]({SITE}demos/#run) · [贡献指南]({SOURCE}CONTRIBUTING.md) · [工程 Skill]({SOURCE}.codex/skills/mars-risk-engineering/SKILL.md) · [MIT License]({SOURCE}LICENSE)。
"""


def main() -> None:
    """同步两个 README，图片由原生报告导出流程生成。"""
    for name, english in (("README.md", False), ("README.en.md", True)):
        (ROOT / name).write_text(_readme(english), encoding="utf-8")


if __name__ == "__main__":
    main()
