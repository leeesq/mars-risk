---
title: MARS
description: 面向人和 AI Agent 的高性能风控分析工具箱。原生分箱图、数据画像、特征筛选、相关性、模型分交叉、规则与可复用报告。
hide:
  - toc
---

<div class="mars-home" markdown="1">

<div class="mars-home-hero" markdown="1">

<picture class="mars-home-logo-wrap">
<source media="(prefers-reduced-motion: reduce)" srcset="assets/mars-logo.svg">
<img class="mars-home-logo" src="assets/mars-logo-animated.svg" alt="MARS">
</picture>
<img class="mars-home-wordmark" src="assets/mars-wordmark.svg" alt="MODELING ANALYSIS RISK SCORE">

<p class="mars-home-badges">
  <a href="https://pypi.org/project/mars-risk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mars-risk?style=flat-square&label=PyPI&color=2f6f8f"></a>
  <a href="https://leeesq.github.io/mars-risk/"><img alt="Docs" src="https://img.shields.io/badge/Docs-GitHub%20Pages-7c3aed?style=flat-square"></a>
  <a href="https://pypi.org/project/mars-risk/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/mars-risk?style=flat-square&label=Python&color=364f6b"></a>
  <a href="https://pepy.tech/project/mars-risk"><img alt="Downloads" src="https://img.shields.io/pepy/dt/mars-risk?style=flat-square&label=Downloads&color=0f766e"></a>
  <a href="https://github.com/leeesq/mars-risk/actions/workflows/test.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/leeesq/mars-risk/test.yml?branch=main&style=flat-square&label=CI&color=1f7a5a"></a>
  <a href="https://github.com/leeesq/mars-risk/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/leeesq/mars-risk?style=flat-square&label=License&color=6c5ce7"></a>
</p>

# 面向人和 AI Agent 的高性能风控分析工具箱

**数据画像 · 分箱评估 · 特征筛选 · 相关性分析 · 模型分交叉 · 规则挖掘**

从 Pandas 或 Polars 宽表，得到可查看、可查询、可交付、可复用的风控分析结果。

用原生图表支持人工判断，用结构化报告连接你的建模、监控归因与分析 Agent。

<div class="mars-home-actions">
<a class="mars-home-button mars-home-button-primary" href="getting-started/quickstart/">开始分析 →</a>
<a class="mars-home-button" href="demos/">查看实战案例 →</a>
<a class="mars-home-button" href="user-guide/external-agents/">面向 AI Agent →</a>
</div>

</div>

<figure class="mars-home-preview mars-native-preview">
<a href="assets/cases/binning-native-main-score.svg" aria-label="查看原生分箱趋势图高清 SVG">
<img src="assets/cases/binning-native-main-score.png" alt="MARS 原生分箱趋势图：固定分箱、样本分布、件数与金额 bad rate、多分区稳定性" width="1600">
</a>
<figcaption>直接导出自 MARS 分箱报告，保留原生布局与完整统计细节。点击查看高清图。<br>
<a href="demos/binning-stability/">分箱与稳定性实战 →</a> · <a href="assets/cases/binning.html">打开交互式报告 →</a></figcaption>
</figure>

## 把分析结果变成可以继续使用的证据

MARS 以 Polars 为计算基础，接受 Pandas 或 Polars 宽表。数据质量、分箱风险、
特征取舍和规则表现都能留下统计表、图表与报告；按特征、标签、客群和时间比较，
再将结果交付为 HTML、Excel 或高清图片。

同一份报告也可以交给外部 AI Agent。表目录、指标口径、特征业务元数据、
计算状态与证据引用随结果保存，让 Agent 按需查询，跨会话继续分析。
你可以将这些能力与自选模型库、实验工具和业务流程组合。

## 从工作任务开始

<div class="mars-task-grid">
<a class="mars-task-card" href="demos/data-quality/"><em>PROFILE</em><strong>数据能直接用吗？</strong><span>先检查缺失与特殊值、标签覆盖、schema 和分布变化。</span><span class="mars-card-link">数据画像 →</span></a>

<a class="mars-task-card" href="demos/binning-stability/"><em>BIN / EVALUATE</em><strong>特征有区分度，而且稳定吗？</strong><span>用原生分箱图比较分布与风险，保留固定边界和多目标口径。</span><span class="mars-card-link">分箱与风险评估 →</span></a>

<a class="mars-task-card" href="demos/selection-correlation/"><em>SELECT</em><strong>为什么保留这个，删除另一个？</strong><span>查看候选、筛选步骤和保留／删除的实际理由。</span><span class="mars-card-link">特征筛选 →</span></a>

<a class="mars-task-card" href="demos/selection-correlation/#correlation"><em>CORRELATION</em><strong>哪些特征提供了重复信息？</strong><span>分别审查 raw / WOE 的带符号相关性证据。</span><span class="mars-card-link">相关性分析 →</span></a>

<a class="mars-task-card" href="demos/score-cross/"><em>SCORE CROSS</em><strong>辅助分还能进一步区分风险吗？</strong><span>在主模型同等级内比较风险梯度、格子证据和组合效果。</span><span class="mars-card-link">模型分交叉 →</span></a>

<a class="mars-task-card" href="demos/rule-evidence/"><em>RULE MINING · EXPERIMENTAL</em><strong>规则能走到独立验证吗？</strong><span>从候选发现到筛选、覆盖和独立验证，查看完整证据。</span><span class="mars-card-link">规则挖掘 →</span></a>
</div>

[七个实战案例与完整下载](demos/index.md) · [保存后继续查询](demos/saved-reports.md) ·
[交付给人和 Agent](demos/report-delivery.md)

案例采用 18,000 行合成申请，人工图表与 Agent 查询读取同一批报告；展示结果不代表真实业务收益。

## 基于 MARS，构建你的 Agent

MARS 提供计算与证据层。Agent 的编排、训练工具和业务决策可以由你自由组合。

| 场景 | 可复用能力 | 可以形成的结果 |
| --- | --- | --- |
| 建模 Agent | 数据质量、分箱评估、特征筛选、相关性、风险评估 | 特征候选与取舍理由，供自选训练工具继续使用的评估证据 |
| 监控与归因 Agent | 分组画像、缺失和分布变化、固定分箱风险趋势、稳定性 | 定位变化人群与特征，提出待验证的归因假设 |
| 规则分析 Agent | 候选规则、命中评估、发现审计、独立验证 | 带风险、覆盖和验证结果的可审查候选 |
| 报告分析 Agent | 表目录、业务元数据、分页、快照、证据引用 | 跨会话继续分析，输出可追溯的回答 |

这些是基于底层能力组合的工作流，完整 Agent 由使用者实现。可选的内部
Agent 也消费公共报告接口。[查看外部 Agent 指南](user-guide/external-agents.md) ·
[规则证据与 Agent](user-guide/rule-reports-and-agents.md)

## 计算一次，持续使用

先生成图表、HTML 和快照，再在另一个进程加载同一份结果。

=== "人：查看与交付"

    ```python
    --8<-- "docs/snippets/readme_quickstart.py:quickstart"
    ```

=== "AI Agent：查询与复用"

    ```python
    from mars.reporting import load_report

    --8<-- "docs/snippets/minimal_report.py:agent"
    ```

    加载返回 ReportSnapshot；查询已保存统计无需原始宽表或内置 Agent 会话。
    用 restored.show_table("summary", limit=10) 查看结果。

<div class="mars-home-values">
<div><strong>业务信息随结果保留</strong><span>标签定义、单位、来源与实际参数，未知信息如实标记。</span></div>
<div><strong>按需读取分析证据</strong><span>先筛选、投影和分页，再生成有字符预算的 JSON。</span></div>
<div><strong>跨会话继续分析</strong><span>保存完整快照，恢复报告身份、表目录与可查询证据。</span></div>
</div>

JSON 摘要可以粘贴到对话框；max_chars 是字符预算，不是 token 数。
完整 .marsreport 供具备 Python／工具能力的 Agent 继续查询；
新维度与原始记录问题需要发起新的计算。

<div class="mars-callout" markdown="1">

## 当前迭代方向

**Analysis、Feature、Reporting 为 Stable；Rule、Monitoring、Modeling、Pipeline、Scoring、Agent 为 Experimental。**
这些标记说明接口成熟度。

优先改进分析性能、内存效率、人工体验与外部 Agent 使用。
**监控、建模（包含 Pipeline）、评分卡暂时停止功能迭代**；现有功能与文档保留，
处理必要修复及上游适配。三个下游直接适配核心代码，上游不为它们增加兼容别名或冗余计算。
你可以结合编程型 AI、MARS 分析报告与自选模型工具构建定制流程。

[稳定性与适配原则](project/stability.md) · [建模／Pipeline](user-guide/modeling-pipeline.md) ·
[监控](user-guide/monitoring.md) · [评分卡](user-guide/scorecard.md)

</div>

本站对应 **main / 源码 0.0.28**；PyPI 当前发布 0.0.27，新能力使用源码安装：

```bash
pip install "git+https://github.com/leeesq/mars-risk.git"
```

[安装与可选依赖](getting-started/installation.md) · [API Reference](reference/index.md) ·
[历史 LightGBM 示例](demos/history.md) · [GitHub](https://github.com/leeesq/mars-risk) ·
[MIT License](https://github.com/leeesq/mars-risk/blob/main/LICENSE)

</div>
