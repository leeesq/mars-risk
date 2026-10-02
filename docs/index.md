---
title: MARS
description: 面向人和 AI Agent 的风控分析工具箱。数据画像、分箱评估、特征筛选、相关性分析、模型分交叉与规则挖掘。
hide:
  - toc
---

<div class="mars-home" markdown="1">

<div class="mars-home-hero" markdown="1">
<img class="mars-home-logo" src="assets/mars-logo.svg" alt="MARS">
<img class="mars-home-wordmark" src="assets/mars-wordmark.svg" alt="MODELING ANALYSIS RISK SCORE">

<p class="mars-home-badges">
  <a href="https://pypi.org/project/mars-risk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mars-risk?style=flat-square&label=PyPI&color=2f6f8f"></a>
  <a href="https://leeesq.github.io/mars-risk/"><img alt="Docs" src="https://img.shields.io/badge/Docs-GitHub%20Pages-7c3aed?style=flat-square"></a>
  <a href="https://pypi.org/project/mars-risk/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/mars-risk?style=flat-square&label=Python&color=364f6b"></a>
  <a href="https://pepy.tech/project/mars-risk"><img alt="Downloads" src="https://img.shields.io/pepy/dt/mars-risk?style=flat-square&label=Downloads&color=0f766e"></a>
  <a href="https://github.com/leeesq/mars-risk/actions/workflows/test.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/leeesq/mars-risk/test.yml?branch=main&style=flat-square&label=CI&color=1f7a5a"></a>
  <a href="https://github.com/leeesq/mars-risk/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/leeesq/mars-risk?style=flat-square&label=License&color=6c5ce7"></a>
</p>

# 面向人和 AI Agent 的风控分析工具箱

以 Polars 为计算基础，接收 Pandas 或 Polars 宽表，输出可查询、可交付、可复用的分析报告。

用表格和图表支持人工分析，用结构化结果支持 AI Agent 继续工作。

<div class="mars-home-actions">
<a class="mars-home-button mars-home-button-primary" href="getting-started/quickstart/">开始人工分析 →</a>
<a class="mars-home-button" href="user-guide/external-agents/">将报告交给 AI Agent →</a>
</div>

**数据画像 · 分箱评估 · 特征筛选 · 相关性分析 · 模型分交叉 · 规则挖掘**

</div>

## 从工作任务开始

<div class="mars-task-grid">
<a class="mars-task-card" href="user-guide/data-profiling/"><em>PROFILE</em><strong>数据能直接用吗？</strong><span>缺失、分布、样本概况、分组和时间变化</span><span class="mars-card-link">查看指南 →</span></a>

<a class="mars-task-card" href="user-guide/binning-risk-evaluation/"><em>BIN / EVALUATE</em><strong>哪些特征有区分度？</strong><span>分箱、IV、KS、坏率与稳定性</span><span class="mars-card-link">查看指南 →</span></a>

<a class="mars-task-card" href="user-guide/feature-selection/"><em>SELECT / CORRELATION</em><strong>哪些特征值得保留？</strong><span>筛选记录、相关性和取舍证据</span><span class="mars-card-link">查看指南 →</span></a>

<a class="mars-task-card" href="user-guide/correlation-and-score-cross/#固定分段交叉"><em>SCORE CROSS</em><strong>两个模型分如何组合？</strong><span>固定分段、交叉表现、客群定位、策略回放</span><span class="mars-card-link">查看指南 →</span></a>

<a class="mars-task-card" href="user-guide/rule-mining/"><em>RULE MINING</em><strong>哪些组合条件需要关注？</strong><span>候选规则、覆盖率、风险表现和验证 · Experimental</span><span class="mars-card-link">查看指南 →</span></a>

<a class="mars-task-card" href="user-guide/external-agents/"><em>REPORT</em><strong>如何把分析结果继续用？</strong><span>查询、展示、导出、保存与外部 Agent 使用</span><span class="mars-card-link">查看指南 →</span></a>
</div>

## 计算一次，持续使用

同一份报告，衔接人工分析与 Agent 工作。下面用小数据生成 `report`；
第二个页签加载它保存的文件，在新进程也可继续查询。

=== "人：查看与交付"

    ```python
    import polars as pl
    from mars.analysis import profile_risk

    df = pl.DataFrame({
        "income": [3200, 3600, 5200, 6100, 3400, 4300, 5800, 6800],
        "utilization": [0.72, 0.61, 0.29, 0.18, 0.66, 0.48, 0.24, 0.12],
        "target": [1, 1, 0, 0, 1, 1, 0, 0],
    })
    --8<-- "docs/snippets/minimal_report.py:analysis"
    ```

=== "AI Agent：查询与复用"

    ```python
    from mars.reporting import load_report

    --8<-- "docs/snippets/minimal_report.py:agent"
    ```

    上个页签已保存 `risk_report.marsreport`。查询无需原始宽表或内置 Agent 会话。
    恢复后用 `restored.show_table("summary", limit=10)` 查看表格。

<div class="mars-home-values">
<div><strong>业务信息随结果保留</strong><span>标签定义、单位、来源与实际参数，未知信息如实标记。</span></div>
<div><strong>按需读取分析证据</strong><span>先筛选、投影和分页，再生成有字符预算的 JSON。</span></div>
<div><strong>跨会话继续分析</strong><span>保存完整快照，恢复报告身份、表目录与可查询证据。</span></div>
</div>

JSON 摘要适合粘贴到对话框，`max_chars` 是字符预算，不是 token 数。
完整 `.marsreport` 适合具备 Python／工具能力的外部 Agent；
普通对话模型接收文件后仍需要执行环境。
恢复返回 `ReportSnapshot`，能力见[外部 Agent 指南](user-guide/external-agents.md)。

## 具体分析场景

- [同一主模型等级内，辅助分能否进一步区分风险？](user-guide/correlation-and-score-cross.md#固定分段交叉)
- [这个特征为什么在筛选中被剔除？](user-guide/correlation-and-score-cross.md#相关性报告)
- [两个时期的分布变化体现在哪些特征？](user-guide/data-profiling.md)
- [怎样将报告交给另一个 Agent，继续读取证据？](user-guide/external-agents.md#分页与证据回放)

<div class="mars-callout" markdown="1">

## 建模、监控、评分卡：保留功能，暂缓迭代

Analysis、Feature、Reporting 为 Stable；Rule、Monitoring、Modeling、Pipeline、Scoring 和 Agent
为 Experimental。成熟度与开发投入分别表达。

当前优先改进分析性能、内存效率、人工体验与 AI 可用性。
建模（包含 Pipeline）、监控、评分卡暂时停止功能迭代；现有功能与文档保留。
用户可借助编程型 AI、MARS 分析与报告能力及自己的数据和模型工具构建定制流程。
三个模块直接适配核心接口变化，完整规则见[稳定性与兼容性](project/stability.md)。

[建模／Pipeline](user-guide/modeling-pipeline.md) · [监控](user-guide/monitoring.md) ·
[评分卡](user-guide/scorecard.md)。它们继续标记 Experimental；暂停说明开发投入。

</div>

[安装](getting-started/installation.md) · [Quickstart](getting-started/quickstart.md) ·
[API Reference](reference/index.md) · [可运行示例](user-guide/external-agents.md#完整示例) ·
[GitHub](https://github.com/leeesq/mars-risk) · [MIT License](https://github.com/leeesq/mars-risk/blob/main/LICENSE)

本站对应当前源码 **0.0.28**，包含尚未发布的新能力。按本文运行，请安装源码：

```bash
pip install "git+https://github.com/leeesq/mars-risk.git"
```

截至 2026-10-02，PyPI 已发布版本为 [0.0.27](https://pypi.org/project/mars-risk/)；
动态徽章展示发布来源，不代表 main 的能力已经发布。

</div>
