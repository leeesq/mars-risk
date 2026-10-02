**中文** / [English](README.en.md)

<div align="center">
<img src="docs/assets/mars-logo.svg" alt="MARS" width="480">
<br>
<img src="docs/assets/mars-wordmark.svg" alt="MODELING ANALYSIS RISK SCORE" width="480">
<p align="center">
  <a href="https://pypi.org/project/mars-risk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mars-risk?style=flat-square&label=PyPI&color=2f6f8f"></a>
  <a href="https://leeesq.github.io/mars-risk/"><img alt="Docs" src="https://img.shields.io/badge/Docs-GitHub%20Pages-7c3aed?style=flat-square"></a>
  <a href="https://pypi.org/project/mars-risk/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/mars-risk?style=flat-square&label=Python&color=364f6b"></a>
  <a href="https://pepy.tech/project/mars-risk"><img alt="Downloads" src="https://img.shields.io/pepy/dt/mars-risk?style=flat-square&label=Downloads&color=0f766e"></a>
  <a href="https://github.com/leeesq/mars-risk/actions/workflows/test.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/leeesq/mars-risk/test.yml?branch=main&style=flat-square&label=CI&color=1f7a5a"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/leeesq/mars-risk?style=flat-square&label=License&color=6c5ce7"></a>
</p>
</div>

# 面向人和 AI Agent 的风控分析工具箱

**数据画像 · 分箱评估 · 特征筛选 · 相关性分析 · 模型分交叉 · 规则挖掘**

从 Pandas 或 Polars 宽表得到可查询、可交付、可复用的分析报告。人工图表与外部 Agent 查询共享同一 Report。

<picture>
  <source media="(max-width: 600px)" srcset="docs/assets/cases/readme-preview-mobile.png">
  <img src="docs/assets/cases/readme-preview.png" alt="同一合成报告的真实模型分交叉矩阵与公共 Agent 查询证据" width="1040">
</picture>

本轮真实浏览器截图：合成模型分交叉矩阵与同一报告的公共查询 JSON。身份、范围与数值可在[完整证据](docs/assets/cases/case-4.json)中核对；[进入可运行案例](docs/demos/score-cross.md)。

[开始分析](https://leeesq.github.io/mars-risk/getting-started/quickstart/) · [七个实战案例](docs/demos/index.md) · [外部 Agent 使用](https://leeesq.github.io/mars-risk/user-guide/external-agents/) · [完整文档](https://leeesq.github.io/mars-risk/)

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

七例共用固定 seed 的 18,000 行合成数据，发现与验证分离。观察期 `late60` 无标签，`bad30` 保留实际有效标签。合成统计不外推业务收益、因果关系、策略批准或生产结论。精确参数以 [API Reference](https://leeesq.github.io/mars-risk/reference/) 为单一权威来源。

## 安装与版本边界

源码版本为 **0.0.28**。本页公共报告与新分析能力使用已核验源码，固定安装 commit：

```bash
pip install "git+https://github.com/leeesq/mars-risk.git@746b8fa76439e6841b466bd590c35249a975ab56"
```

截至 **2026-10-02**，实时核验 [PyPI 已发布版本](https://pypi.org/project/mars-risk/)为 **0.0.27**，不包含本页全部新能力。本轮没有发布 0.0.28；六个徽章保留真实动态来源。

```bash
pip install mars-risk==0.0.27
```

基础包支持 Python **3.8–3.12**，3.8 使用冻结依赖栈。本轮公开案例以 Python 3.11 生成、Python 3.12 检查；可选模型、调参、Notebook、文档开发与内置 Agent SDK 使用 **Python 3.10+**。[安装指南](https://leeesq.github.io/mars-risk/getting-started/installation/)列出 extras 与兼容约束。

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
print(report.get_table("summary", sort_by="iv", descending=True, limit=2))
report.write_html("risk_report.html", include_charts=False)
report.save("risk_report.marsreport")
```

[完整已测试脚本](docs/snippets/readme_quickstart.py) · [Quickstart](https://leeesq.github.io/mars-risk/getting-started/quickstart/)。八行小夹具用于说明接口，其 IV 不证明真实模型质量。这里的 HTML 不含图表；计算时间趋势需提供有效日期上下文。

## 同一 Report，人工阅读与 Agent 查询

人工可读 HTML。在**新进程**中加载快照，不需要原始宽表、原分析器、LLM、API Key 或 `MarsAgentSession`：

```python
from mars.reporting import load_report

report = load_report("risk_report.marsreport")
description = report.describe()
page = report.query_page("summary", limit=1)
next_page = report.query_page("summary", offset=page["next_offset"], limit=1)
context_json = report.to_ai_context(tables=["summary"], max_chars=16000)
```

此时 `report` 是 ReportSnapshot，`report.write_excel("risk_report.xlsx")` 导出当前静态表。原分箱报告的 Excel 方法使用旧透视模板，可能需要原生 Excel 刷新；本例静态交付使用加载后的快照。

`describe()` 发现真实表目录、schema、口径与支持能力；`query_page()` 返回有限页、表行数、下一页 offset 与可重放证据引用。`to_ai_context()` 生成有限 JSON，**`max_chars` 是最终 JSON 字符预算，不是 token 数**。回答前检查裁剪与状态；空结果、不可用值与有效零值分别表达。

恢复类型为 **ReportSnapshot**，保留身份、元数据和已保存统计表，用 `show_table()` 通用展示。专用模型分策略回放使用保存的聚合证据。恢复不还原分析器、原始记录、任意方法、训练模型或可部署 RuleSet；新维度与个体记录问题需要重新分析。

[保存查询案例](docs/demos/saved-reports.md)展示下一页、空结果、无效请求与真实预算裁剪。[交付案例](docs/demos/report-delivery.md)列出各格式能力与限制：Excel 为静态交付，HTML 交互和离线性逐份导出验证。TXT 是提示材料，本地确定性消费者明确标为脚本；示例回答是人工整理并用证据核验，不冒充 LLM 实验。

## 成熟度与投入方向

**Analysis、Feature、Reporting 为 Stable；Rule、Monitoring、Modeling、Pipeline、Scoring、Agent 为 Experimental。** 成熟度与开发投入分别表达。

优先改进核心分析性能、内存效率、人工体验与外部 Agent 公共报告消费。**监控、建模（含 Pipeline）、评分卡暂时停止功能迭代**；现有功能、文档与历史案例保留，处理必要正确性修复、运行修复和上游直接适配。上游不为三个下游增加旧别名、兼容壳或冗余计算；该豁免不扩展到核心公共 API 和已保存报告。

[稳定性与适配原则](https://leeesq.github.io/mars-risk/project/stability/) · [历史 LightGBM 案例](docs/demos/history.md) · [规则与 Agent 详细指南](docs/user-guide/rule-reports-and-agents.md)。

## 复现、验证与贡献

生成、证据验证、站点构建分别执行。MkDocs 消费已核验资源；`mkdocs-jupyter` 的 `execute: false` 仅渲染，不代替运行。[案例索引](docs/demos/index.md)提供产物能力矩阵、共享下载与复现说明。

```bash
python scripts/check_case_assets.py
python -m pytest -q tests/test_task_cases.py tests/test_documentation.py -m "not docs_ml"
python -m mkdocs build --strict
```

开发检查、依赖和真实浏览器验收见 [CONTRIBUTING.md](CONTRIBUTING.md)。[产物来源记录](docs/assets/cases/manifest.json)保留 seed、规模、源码 commit、依赖版本与文件 hash。[MIT License](LICENSE)。
