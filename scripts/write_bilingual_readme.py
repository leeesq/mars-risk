"""用共同品牌、安装和代码源维护两个完整语言的 README。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "746b8fa76439e6841b466bd590c35249a975ab56"
SITE = "https://leeesq.github.io/mars-risk/"


def _quickstart() -> str:
    """读取已测试区域；两种语言不得维护各自算法。"""
    source = (ROOT / "docs/snippets/readme_quickstart.py").read_text(encoding="utf-8")
    return source.split("# --8<-- [start:quickstart]\n", 1)[1].split(
        "# --8<-- [end:quickstart]", 1
    )[0].rstrip()


def _brand() -> str:
    """复用原品牌与完整动态徽章，不更换素材身份。"""
    original = (ROOT / "docs/index.md").read_text(encoding="utf-8")
    badges = original.split('<p class="mars-home-badges">', 1)[1].split("</p>", 1)[0]
    badges = badges.replace(
        'href="https://github.com/leeesq/mars-risk/blob/main/LICENSE"', 'href="LICENSE"'
    )
    return (
        '<div align="center">\n'
        '<img src="docs/assets/mars-logo.svg" alt="MARS" width="480">\n'
        '<br>\n'
        '<img src="docs/assets/mars-wordmark.svg" '
        'alt="MODELING ANALYSIS RISK SCORE" width="480">\n'
        '<p align="center">' + badges + '</p>\n</div>'
    )


def _preview(english: bool) -> str:
    """真实截图使用窄屏裁切回退，仍连接同一份报告证据。"""
    alt = (
        "Actual Score Cross matrix and public Agent query from the same synthetic report"
        if english else "同一合成报告的真实模型分交叉矩阵与公共 Agent 查询证据"
    )
    return (
        '<picture>\n'
        '  <source media="(max-width: 600px)" '
        'srcset="docs/assets/cases/readme-preview-mobile.png">\n'
        '  <img src="docs/assets/cases/readme-preview.png" '
        f'alt="{alt}" width="1040">\n'
        '</picture>'
    )


def _readme(english: bool) -> str:
    """组成自然语言完整版本，API 与安装共用精确源。"""
    code = _quickstart()
    brand = _brand()
    preview = _preview(english)
    install = f'pip install "git+https://github.com/leeesq/mars-risk.git@{SOURCE_COMMIT}"'
    agent = '''from mars.reporting import load_report

report = load_report("risk_report.marsreport")
description = report.describe()
page = report.query_page("summary", limit=1)
next_page = report.query_page("summary", offset=page["next_offset"], limit=1)
context_json = report.to_ai_context(tables=["summary"], max_chars=16000)'''
    if english:
        return f'''[中文](README.md) / **English**

{brand}

# A risk analysis toolkit for humans and AI agents

**Data profiling · Binning evaluation · Feature selection · Correlation analysis · Score cross analysis · Rule mining**

Turn a Pandas or Polars wide table into analysis reports that people can read, query, deliver and reuse. Human views and external Agent evidence come from the same Report.

{preview}

Actual browser capture of a synthetic Score Cross report; the current report UI is in Chinese. The matrix and the JSON excerpt share the same report identity and query scope. [Full evidence and provenance](docs/assets/cases/case-4.json) · [Runnable case](docs/demos/score-cross.md).

[Start analyzing]({SITE}getting-started/quickstart/) · [Seven practical cases](docs/demos/index.md) · [External Agent guide]({SITE}user-guide/external-agents/) · [Full documentation (Chinese)]({SITE})

## Choose a task

| Your question | Capability | Result and runnable case |
| --- | --- | --- |
| Can this data be used as it is? | Data profiling | [Quality, labels, schema and distribution evidence](docs/demos/data-quality.md) |
| Which features discriminate risk and remain stable? | Binning evaluation | [Fixed reference bins, multi-target IV / KS / PSI](docs/demos/binning-stability.md) |
| Why keep one feature and remove another? | Feature selection + correlation analysis | [Decisions, signed redundancy, separate raw / WOE representations](docs/demos/selection-correlation.md) |
| Does an auxiliary score separate risk within a main-score tier? | Score cross analysis | [Interactive matrix, row-relative Δ, cell evidence and policy replay](docs/demos/score-cross.md) |
| Which candidate rules warrant review? | Rule mining — Experimental | [Discovery audit, independent validation and coverage](docs/demos/rule-evidence.md) |
| How can I save and query this analysis later? | Public reports | [Fresh-process loading, pagination and character budgets](docs/demos/saved-reports.md) |
| How do I deliver one analysis to different users? | Report exports | [HTML / Excel / .marsreport / bounded JSON / Agent prompt materials](docs/demos/report-delivery.md) |

The seven cases share 18,000 synthetic rows with a fixed seed. Discovery and validation are separate; observation has no `late60` labels, while `bad30` remains observed where available. Synthetic statistics do not establish business profit, causal effects, approval decisions or production readiness. Detailed parameters have one authority: [API Reference (Chinese)]({SITE}reference/).

## Installation and version boundaries

The source version is **0.0.28**. The examples use current public report capabilities. Install the verified source commit:

```bash
{install}
```

On **2026-10-02**, the live [PyPI package](https://pypi.org/project/mars-risk/) was **0.0.27**. It does not contain all the capabilities shown here. Source 0.0.28 has not been published by this task. The six badges retain their actual dynamic sources.

```bash
pip install mars-risk==0.0.27
```

The base package supports Python **3.8–3.12**, with a frozen dependency stack for 3.8. Public cases were generated with Python 3.11 and are checked with Python 3.12; use **Python 3.10+** for optional modeling, tuning, notebooks, documentation development and the built-in Agent SDK. [Installation guide (Chinese)]({SITE}getting-started/installation/) explains extras and constraints.

To reproduce the new case scripts, use this task branch until it is merged:

```bash
git clone --branch codex/task-cases-bilingual-readme https://github.com/leeesq/mars-risk.git
cd mars-risk
pip install -e ".[docs]"
python docs/snippets/task_cases.py --case all --output-dir docs/assets/cases
```

Case sources and committed downloads are available on this branch. New interactive URLs on the documentation site become available when the branch is merged and the existing Pages workflow deploys it. This task does not merge or request an additional deployment.

## A useful first report

Run in an empty working directory. This self-contained example creates a risk summary, an HTML report and a snapshot:

```python
{code}
```

[Full tested script](docs/snippets/readme_quickstart.py) · [Quickstart (Chinese)]({SITE}getting-started/quickstart/). The eight-row fixture demonstrates the interface; its IV values are not evidence of real model quality. HTML here omits charts; provide valid date context when calculating time trends.

## One Report for people and external Agents

People can read the HTML. In a **new process**, an external consumer can load the snapshot without the original wide table, analyzer, LLM, API key or `MarsAgentSession`:

```python
{agent}
```

The loaded `report` is a ReportSnapshot; `report.write_excel("risk_report.xlsx")` exports current static tables. The original binning report's Excel method uses the legacy pivot template and may require native Excel refresh; use the loaded snapshot for this static delivery.

`describe()` discovers the actual tables, schema, semantics and supported capabilities. `query_page()` returns a bounded page, counts, a continuation offset and a replayable evidence reference. `to_ai_context()` serializes bounded JSON: **`max_chars` measures final JSON characters, not tokens**. Check clipping and status before answering; an empty result, an unavailable value and a valid zero mean different things.

Loading returns **ReportSnapshot**. It restores report identity, metadata and saved tables; use `show_table()` for generic display. Dedicated score-policy replay uses saved aggregates. Loading does not restore the analyzer, original records, arbitrary methods, a trained model or a deployable RuleSet. New dimensions and individual-record questions require new analysis.

[Persistence and query case](docs/demos/saved-reports.md) demonstrates next pages, empty results, invalid requests and actual budget clipping. [Delivery case](docs/demos/report-delivery.md) lists supported formats and limitations. Excel is a static delivery; HTML interactivity and offline behavior are verified for each actual export. TXT files are prompt materials, and the local deterministic consumer is explicitly identified as a script. Example answers are manually written and checked against the evidence; no LLM experiment is claimed.

## Maturity and development focus

**Analysis, Feature and Reporting are Stable. Rule, Monitoring, Modeling, Pipeline, Scoring and Agent are Experimental.** Maturity and investment are separate.

Current work prioritizes core analysis performance, memory efficiency, human use and public report consumption by external Agents. **Monitoring, Modeling (including Pipeline) and Scoring are paused for feature development.** Existing functionality, documentation and historical cases remain, with correctness fixes, run fixes and direct adaptation to upstream changes. Those downstream modules do not cause upstream compatibility aliases or duplicate computation. This exception does not permit arbitrary changes to core public APIs or saved reports.

[Stability policy (Chinese)]({SITE}project/stability/) · [Retained LightGBM case](docs/demos/history.md) · [Detailed rule and Agent guide](docs/user-guide/rule-reports-and-agents.md).

## Reproduce, verify and contribute

Generation, evidence verification and site building are explicit, separate steps. MkDocs renders preverified resources; `mkdocs-jupyter` uses `execute: false`, which does not execute notebooks. See the [case index](docs/demos/index.md) for the artifact capability matrix, shared downloads and reproducibility details.

```bash
python scripts/check_case_assets.py
python -m pytest -q tests/test_task_cases.py tests/test_documentation.py -m "not docs_ml"
python -m mkdocs build --strict
```

Development checks, dependencies and actual browser acceptance are documented in [CONTRIBUTING.md](CONTRIBUTING.md). [Case provenance](docs/assets/cases/manifest.json) records seed, scale, source commit, dependency versions and file hashes. [MIT License](LICENSE).
'''
    return f'''**中文** / [English](README.en.md)

{brand}

# 面向人和 AI Agent 的风控分析工具箱

**数据画像 · 分箱评估 · 特征筛选 · 相关性分析 · 模型分交叉 · 规则挖掘**

从 Pandas 或 Polars 宽表得到可查询、可交付、可复用的分析报告。人工图表与外部 Agent 查询共享同一 Report。

{preview}

本轮真实浏览器截图：合成模型分交叉矩阵与同一报告的公共查询 JSON。身份、范围与数值可在[完整证据](docs/assets/cases/case-4.json)中核对；[进入可运行案例](docs/demos/score-cross.md)。

[开始分析]({SITE}getting-started/quickstart/) · [七个实战案例](docs/demos/index.md) · [外部 Agent 使用]({SITE}user-guide/external-agents/) · [完整文档]({SITE})

## 从要解决的问题开始

| 问题 | 对应能力 | 结果与可运行案例 |
| --- | --- | --- |
| 这份数据能直接用吗？ | 数据画像 | [质量、标签、schema 与分布证据](docs/demos/data-quality.md) |
| 哪些特征有区分度，而且足够稳定？ | 分箱评估 | [固定参考分箱、多目标 IV / KS / PSI](docs/demos/binning-stability.md) |
| 为什么保留这个特征、删除另一个？ | 特征筛选 + 相关性分析 | [真实决策、带符号冗余、raw / WOE 分别解释](docs/demos/selection-correlation.md) |
| 主模型同等级内，辅助分还能区分风险吗？ | 模型分交叉 | [交互矩阵、相对行基线 Δ、格子证据与策略回放](docs/demos/score-cross.md) |
| 从候选规则走到可以审查的证据？ | 规则挖掘 · Experimental | [发现审计、独立验证、表现与覆盖](docs/demos/rule-evidence.md) |
| 分析完成后，如何保存并继续查询？ | 公共报告 | [新进程加载、分页与字符预算](docs/demos/saved-reports.md) |
| 同一次分析，如何交付给不同使用者？ | 报告导出 | [HTML / Excel / .marsreport / 有边界 JSON / 提示材料](docs/demos/report-delivery.md) |

七例共用固定 seed 的 18,000 行合成数据，发现与验证分离。观察期 `late60` 无标签，`bad30` 保留实际有效标签。合成统计不外推业务收益、因果关系、策略批准或生产结论。精确参数以 [API Reference]({SITE}reference/) 为单一权威来源。

## 安装与版本边界

源码版本为 **0.0.28**。本页公共报告与新分析能力使用已核验源码，固定安装 commit：

```bash
{install}
```

截至 **2026-10-02**，实时核验 [PyPI 已发布版本](https://pypi.org/project/mars-risk/)为 **0.0.27**，不包含本页全部新能力。本轮没有发布 0.0.28；六个徽章保留真实动态来源。

```bash
pip install mars-risk==0.0.27
```

基础包支持 Python **3.8–3.12**，3.8 使用冻结依赖栈。本轮公开案例以 Python 3.11 生成、Python 3.12 检查；可选模型、调参、Notebook、文档开发与内置 Agent SDK 使用 **Python 3.10+**。[安装指南]({SITE}getting-started/installation/)列出 extras 与兼容约束。

复现新案例脚本，在合并前使用本任务分支：

```bash
git clone --branch codex/task-cases-bilingual-readme https://github.com/leeesq/mars-risk.git
cd mars-risk
pip install -e ".[docs]"
python docs/snippets/task_cases.py --case all --output-dir docs/assets/cases
```

新案例源码和已提交下载资源在此分支可取得；文档站新增交互路径在合并并由既有 Pages 流程部署后可用。本轮不自动合并或额外部署。

## 先得到一份有用的报告

在空工作目录执行。代码独立产生风险摘要、HTML 与快照：

```python
{code}
```

[完整已测试脚本](docs/snippets/readme_quickstart.py) · [Quickstart]({SITE}getting-started/quickstart/)。八行小夹具用于说明接口，其 IV 不证明真实模型质量。这里的 HTML 不含图表；计算时间趋势需提供有效日期上下文。

## 同一 Report，人工阅读与 Agent 查询

人工可读 HTML。在**新进程**中加载快照，不需要原始宽表、原分析器、LLM、API Key 或 `MarsAgentSession`：

```python
{agent}
```

此时 `report` 是 ReportSnapshot，`report.write_excel("risk_report.xlsx")` 导出当前静态表。原分箱报告的 Excel 方法使用旧透视模板，可能需要原生 Excel 刷新；本例静态交付使用加载后的快照。

`describe()` 发现真实表目录、schema、口径与支持能力；`query_page()` 返回有限页、表行数、下一页 offset 与可重放证据引用。`to_ai_context()` 生成有限 JSON，**`max_chars` 是最终 JSON 字符预算，不是 token 数**。回答前检查裁剪与状态；空结果、不可用值与有效零值分别表达。

恢复类型为 **ReportSnapshot**，保留身份、元数据和已保存统计表，用 `show_table()` 通用展示。专用模型分策略回放使用保存的聚合证据。恢复不还原分析器、原始记录、任意方法、训练模型或可部署 RuleSet；新维度与个体记录问题需要重新分析。

[保存查询案例](docs/demos/saved-reports.md)展示下一页、空结果、无效请求与真实预算裁剪。[交付案例](docs/demos/report-delivery.md)列出各格式能力与限制：Excel 为静态交付，HTML 交互和离线性逐份导出验证。TXT 是提示材料，本地确定性消费者明确标为脚本；示例回答是人工整理并用证据核验，不冒充 LLM 实验。

## 成熟度与投入方向

**Analysis、Feature、Reporting 为 Stable；Rule、Monitoring、Modeling、Pipeline、Scoring、Agent 为 Experimental。** 成熟度与开发投入分别表达。

优先改进核心分析性能、内存效率、人工体验与外部 Agent 公共报告消费。**监控、建模（含 Pipeline）、评分卡暂时停止功能迭代**；现有功能、文档与历史案例保留，处理必要正确性修复、运行修复和上游直接适配。上游不为三个下游增加旧别名、兼容壳或冗余计算；该豁免不扩展到核心公共 API 和已保存报告。

[稳定性与适配原则]({SITE}project/stability/) · [历史 LightGBM 案例](docs/demos/history.md) · [规则与 Agent 详细指南](docs/user-guide/rule-reports-and-agents.md)。

## 复现、验证与贡献

生成、证据验证、站点构建分别执行。MkDocs 消费已核验资源；`mkdocs-jupyter` 的 `execute: false` 仅渲染，不代替运行。[案例索引](docs/demos/index.md)提供产物能力矩阵、共享下载与复现说明。

```bash
python scripts/check_case_assets.py
python -m pytest -q tests/test_task_cases.py tests/test_documentation.py -m "not docs_ml"
python -m mkdocs build --strict
```

开发检查、依赖和真实浏览器验收见 [CONTRIBUTING.md](CONTRIBUTING.md)。[产物来源记录](docs/assets/cases/manifest.json)保留 seed、规模、源码 commit、依赖版本与文件 hash。[MIT License](LICENSE)。
'''


def main() -> None:
    """写入两个 README；图和数字由真实案例流程提供。"""
    for name, english in (("README.md", False), ("README.en.md", True)):
        (ROOT / name).write_text(_readme(english), encoding="utf-8")


if __name__ == "__main__":
    main()
