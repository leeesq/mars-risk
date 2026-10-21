---
description: 七个任务型风控实战案例；同一批 Home Credit 2018 全量数据、公共报告和可下载证据。
hide:
  - toc
---

# 从问题选择实战案例

**数据画像 · 分箱评估 · 特征筛选 · 相关性分析 · 模型分交叉 · 规则挖掘**。
本组案例使用 Home Credit 的两张 application 表。先检查数据质量，再评估和筛选特征、发现并验证规则，最后保存报告并交付分析结果。
每篇都提供结果页面、可运行代码和 Agent 查询证据。你可以先查看结果，再运行代码，或替换成自己的数据继续分析。

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

检查样本数、有效标签、缺失值与特殊值，比较字段结构和未见类别。

--8<-- "docs/assets/cases/previews.txt:card1"

[质量摘要与前置检查 →](data-quality.md)

</div>

<div class="mars-case-card" markdown="1">

**02 · 分箱评估**

### [哪些特征有区分度，而且稳定？](binning-stability.md)

固定分箱边界，比较不同分区和目标的 IV、KS、PSI，以及各箱风险。

--8<-- "docs/assets/cases/previews.txt:card2"

[趋势图、指标与分箱证据 →](binning-stability.md)

</div>

<div class="mars-case-card" markdown="1">

**03 · 特征筛选 · 相关性分析**

### [为什么保留这个，删除另一个？](selection-correlation.md)

查看候选特征、筛选步骤和取舍理由，分别读取原始值（raw）与目标感知 WOE 的结果。

--8<-- "docs/assets/cases/previews.txt:card3"

[筛选决策 →](selection-correlation.md) · [相关性证据 →](selection-correlation.md#correlation)

</div>

<div class="mars-case-card" markdown="1">

**04 · 模型分交叉**

### [两项外部来源分如何共同区分风险？](score-cross.md)

查看交叉矩阵、Δ pp 和 Lift，选中格子比较双向风险梯度，再用已聚合的统计回放 policy。

--8<-- "docs/assets/cases/previews.txt:card4"

[交互矩阵与格子查询 →](score-cross.md)

</div>

<div class="mars-case-card" markdown="1">

**05 · 规则挖掘 · Experimental**

### [候选规则如何变成可审查证据？](rule-evidence.md)

查看候选规则的来源、审计记录、独立验证结果和未通过原因。

--8<-- "docs/assets/cases/previews.txt:card5"

[候选发现与验证结果 →](rule-evidence.md)

</div>

<div class="mars-case-card" markdown="1">

**06 · 公共报告查询与保存**

### [分析完成后，如何保存并继续查询？](saved-reports.md)

在新进程加载快照，分页查询结果，检查字符预算、空结果和无效请求。

--8<-- "docs/assets/cases/previews.txt:card6"

[独立加载与查询轨迹 →](saved-reports.md)

</div>

<div class="mars-case-card" markdown="1">

**07 · 公共报告交付**

### [同一次分析，如何交付给不同使用者？](report-delivery.md)

比较 HTML、静态 Excel、快照、限定范围的 JSON 和外部 Agent TXT 各能交付什么。

--8<-- "docs/assets/cases/previews.txt:card7"

[格式、下载与数值核对 →](report-delivery.md)

</div>

</div>

</div>

## 先看产物能力，再选择格式 { #formats }

按接收方选择格式：人工阅读用 HTML 或 Excel，汇报用高清图，需要在新进程继续查询时使用完整快照。
案例 6 和案例 7 复用前面已经生成的报告和共享包。
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

本组案例使用 Home Credit 两张 application 表的全部 356,255 条记录。`TARGET` 来自原比赛训练表；比赛测试表的 48,744 条记录没有公开标签，因此保持 null。`APPLY_DATE` 是按 ID 稳定散列生成的演示日期。

`0train` 用于拟合，也作为 PSI 的基准样本。`1val` 用于验证或筛选，`2test` 和 `3oot` 用于独立复核固定结果；`4_oot_undue` 对应比赛未公开标签的记录。日期、元数据和字段角色见[输入方案与分区说明](../user-guide/home-credit-data.md)。

所有字段保留官方英文 ID，中文显示名和官方定义随报告保存。`AMT_CREDIT` 用作金额字段，默认从候选字段 `fcols` 中排除，币种未公开。manifest 记录全部候选字段、展示子集、输入 SHA256、准备版本和实际分区。

Bad Rate 为比例，KS 的范围为 0～100；IV、PSI 和 Lift 无量纲。分布统计使用全样本，监督指标使用已观测标签等适用分母；未观测指标不补零。

## 运行与下载 { #run }

MARS 0.0.28 已发布到 PyPI。核心案例无需 API Key、LLM 或模型训练 extra。
先登录 Kaggle 官方页面，确认并接受适用条款，再下载指南列出的三个 CSV。数据只保留在本地，不随项目按 MIT 许可再分发。

```bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
python -m pip install -e ".[docs]"
python docs/snippets/task_cases.py --case all --data-dir examples/home-credit-2018/data/raw_data --output-dir output/task-cases
```

[完整共享源码](../snippets/task_cases.py) · [共享数据 helper](../snippets/home_credit_data.py) · [当前案例下载](../assets/cases/cases.zip) · [真实摘要](../assets/cases/summary.json) · [生成环境](../assets/cases/generation-environment.json)

将下载包解压到仓库外，也可以直接加载快照查询。重新计算需要三个原始文件。以下两类验收分别运行；公开 CI 不访问 Kaggle，也不下载原数据。

```bash
python scripts/check_case_assets.py --mode public --site-dir site
python scripts/check_case_assets.py --mode recompute --data-dir examples/home-credit-2018/data/raw_data
```

公开模式核对文件哈希、快照、状态和引用，并检查导出与显示结果是否一致，但不重新计算原始样本的规则命中成员。
完整复算会核对输入指纹、数据准备结果、规则成员和业务统计。报告身份和时间戳允许变化；字段、阈值、实际依赖和数值差异仍须核对。
轻量临时 CSV 测试夹具只验证行为，不能代替全量数据的结果。

[本次迁移验收](../project/home-credit-migration-validation.md) · [历史 LightGBM 案例](history.md)
