---
description: 七个任务型风控实战案例；同一批 Home Credit 2018 全量数据、公共报告和可下载证据。
hide:
  - toc
---

# 从问题选择实战案例

**数据画像 · 分箱评估 · 特征筛选 · 相关性分析 · 模型分交叉 · 规则挖掘**。
从 Home Credit 两张 application 表开始，先检查质量，再评估与筛选特征、发现并验证规则，最后保存和交付分析结果。
每个案例提供人工可读的结果、可运行代码和 Agent 查询证据，方便替换成自己的数据继续分析。

<div class="mars-cases" markdown="1">

<figure class="mars-native-preview">
<a href="../assets/cases/binning-native-home-credit.svg" aria-label="查看分箱趋势图高清原图">
<img src="../assets/cases/binning-native-home-credit.png" alt="MARS 原生分箱趋势图：分区分布、件数与金额 bad rate、区分度和稳定性">
</a>
<figcaption>原生分箱图直接导出，图表与查询结果共享同一份报告。<a href="binning-stability/">进入分箱实战 →</a></figcaption>
</figure>

<div class="mars-case-grid" markdown="1">

<div class="mars-case-card" markdown="1">

**01 · 数据画像**

### [这份数据能直接用吗？](data-quality.md)

样本与有效标签、missing/special、schema 差异和未见类别。

--8<-- "docs/assets/cases/previews.txt:card1"

[质量摘要与前置检查 →](data-quality.md)

</div>

<div class="mars-case-card" markdown="1">

**02 · 分箱评估**

### [哪些特征有区分度，而且稳定？](binning-stability.md)

固定参考分箱，比较不同分区与目标的 IV、KS、PSI 和箱风险。

--8<-- "docs/assets/cases/previews.txt:card2"

[趋势图、指标与分箱证据 →](binning-stability.md)

</div>

<div class="mars-case-card" markdown="1">

**03 · 特征筛选 · 相关性分析**

### [为什么保留这个，删除另一个？](selection-correlation.md)

候选、步骤与取舍审计；raw 与目标感知 WOE 分开读取。

--8<-- "docs/assets/cases/previews.txt:card3"

[筛选决策 →](selection-correlation.md) · [相关性证据 →](selection-correlation.md#correlation)

</div>

<div class="mars-case-card" markdown="1">

**04 · 模型分交叉**

### [两项外部来源分如何共同区分风险？](score-cross.md)

矩阵、Δ pp、Lift、双向梯度、选格与聚合 policy 回放。

--8<-- "docs/assets/cases/previews.txt:card4"

[交互矩阵与格子查询 →](score-cross.md)

</div>

<div class="mars-case-card" markdown="1">

**05 · 规则挖掘 · Experimental**

### [候选规则如何变成可审查证据？](rule-evidence.md)

发现来源、候选审计、独立验证和未通过原因。

--8<-- "docs/assets/cases/previews.txt:card5"

[候选发现与验证结果 →](rule-evidence.md)

</div>

<div class="mars-case-card" markdown="1">

**06 · 公共报告查询与保存**

### [分析完成后，如何保存并继续查询？](saved-reports.md)

新进程只加载快照；分页、预算、空结果与无效请求。

--8<-- "docs/assets/cases/previews.txt:card6"

[独立加载与查询轨迹 →](saved-reports.md)

</div>

<div class="mars-case-card" markdown="1">

**07 · 公共报告交付**

### [同一次分析，如何交付给不同使用者？](report-delivery.md)

HTML、静态 Excel、快照、有限 JSON 和外部 Agent TXT 的真实边界。

--8<-- "docs/assets/cases/previews.txt:card7"

[格式、下载与值对齐 →](report-delivery.md)

</div>

</div>

</div>

## 先看产物能力，再选择格式 { #formats }

按接收方选择格式：人工阅读用 HTML／Excel，高清图用于汇报，完整快照用于跨进程查询。
案例 6/7 复用已有报告和共享包。
文件目录、大小、哈希和同批报告身份见[manifest.json](../assets/cases/manifest.json)。

| 案例 | 报告／真实结果 | HTML | 静态 Excel | .marsreport | 有限 JSON / TXT |
| --- | --- | --- | --- | --- | --- |
| 1 数据能否使用 | MarsProfileReport；案例侧 schema/未见类别前置检查 | profile.html | profile.xlsx | profile.marsreport | case-1.json |
| 2 区分度与稳定性 | MarsBinningReport | binning.html | binning.xlsx | binning.marsreport | case-2.json |
| 3 筛选与相关性 | 真实 selector 决策；raw / WOE CorrelationReport | correlation.html | selection.xlsx / correlation.xlsx | raw-correlation / woe-correlation | case-3.json / selection.json |
| 4 辅助分交叉 | ScoreCrossReport；已有 policy 聚合回放 | score-cross.html | score-cross.xlsx / policy.xlsx | score-cross / policy | case-4.json |
| 5 候选规则证据 | MarsRuleReport（Experimental） | rules.html | rules.xlsx | rules.marsreport | case-5.json；RuleSet 仅在真实资格允许时 |
| 6 保存后查询 | 复用分箱 ReportSnapshot | 复用 | 复用 | 复用 | case-6.json / agent-context.json |
| 7 多种交付 | 复用同批报告与导出 | 复用 | 复用 | 复用 | case-7.json / external-agent-task.txt |

## 输入、五分区与指标 { #data-dictionary }

本批保留 Home Credit 两张 application 表的全部 356,255 条记录。TARGET 来自原比赛训练表，比赛测试表的 48,744 条记录保持 null。APPLY_DATE 是按 ID 稳定散列生成的演示日期。0train 用于拟合和 PSI 基准，1val 用于验证或筛选，2test 与 3oot 用于固定结果的独立复核，4_oot_undue 仅表示比赛未公开标签。日期、元数据和角色映射见[输入方案与分区说明](../user-guide/home-credit-data.md)。

所有字段仍使用官方英文 ID，中文显示名与官方定义随报告保存。金额角色 AMT_CREDIT 默认从候选 fcols 排除，币种未公开。完整候选与展示子集、输入 SHA256、准备版本和实际分区见 manifest。
Bad Rate 为比例，KS 为 0–100；IV、PSI 和 Lift 无量纲。全样本分布与已观测标签指标分别取适用分母，未观测不补零。

## 运行与下载 { #run }

MARS 0.0.28 已发布到 PyPI。核心案例无需 API Key、LLM 或模型训练 extra。
先登录 Kaggle 官方页面，自行确认并接受适用条款，再下载指南列出的三个 CSV。数据仅保留本地，不在项目 MIT 许可下再分发。

```bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
python -m pip install -e ".[docs]"
python docs/snippets/task_cases.py --case all --data-dir examples/home-credit-2018/data/raw_data --output-dir output/task-cases
```

[完整共享源码](../snippets/task_cases.py) · [共享数据 helper](../snippets/home_credit_data.py) · [当前案例下载](../assets/cases/cases.zip) · [真实摘要](../assets/cases/summary.json) · [生成环境](../assets/cases/generation-environment.json)

下载包在仓库外解压可直接加载快照查询；重新计算需要三个原文件。两类验收分别运行，公开 CI 不访问 Kaggle、不下载原数据。

```bash
python scripts/check_case_assets.py --mode public --site-dir site
python scripts/check_case_assets.py --mode recompute --data-dir examples/home-credit-2018/data/raw_data
```

公开模式验证哈希、快照、状态、引用、导出与显示一致，不能证明原始成员全部复算。完整复算核对输入指纹、准备、规则成员与业务统计。报告身份和时间戳允许变化，字段、阈值、实际依赖和数值差异不忽略。
轻量临时 CSV fixture 只验证行为，不能代替全量结果。

[本次迁移验收](../project/home-credit-migration-validation.md) · [历史 LightGBM 案例](history.md)
