---
description: 人工对话与外部 AI Agent 通过公共报告读取元数据、查询证据、保存并跨会话继续分析。
---

# 将报告交给外部 AI Agent

从[案例 6：只加载快照继续查询](../demos/saved-reports.md)和
[案例 7：不同使用者的交付](../demos/report-delivery.md)查看实际有限 JSON 与下载文件。

MARS 是面向人和 AI Agent 的风控分析工具箱。外部 Agent 直接消费公共分析与
`mars.reporting` 接口，无需 `MarsAgentSession` 或模型 SDK。先计算报告，再按需读取证据；
查询已有报告不会重新分箱、重算指标或访问原始样本。
本文使用[当前源码安装](../getting-started/installation.md)，新能力不代表已发布 PyPI 包的能力。

## 最小链路

取得报告、查询 Top-K、展示与导出，再保存完整结果：

```python
--8<-- "docs/snippets/minimal_report.py"
```

`report` 是原始分箱报告；`restored` 是 `ReportSnapshot`。上述示例的输出写到工作目录，
保存默认拒绝覆盖。跨进程时只需文件和安装好的 MARS，无需原始宽表、分箱器或会话。
要导出 Excel，可用 `report.write_excel("risk_report.xlsx")`。

## 先读取目录、参数与状态

`describe()` 返回持久 `report_id`、报告类型、表目录、粒度、行数、字段类型、单位、
实际 `parameters`、状态、限制及来源。画像通常有 `overview`、`dq.<metric>`、
`stats.<metric>`、`comparison.<metric>`；分箱报告有 `summary`、`detail`、`trend.<metric>`，
其他报告以真实目录为准。没有计算的表不会凭空出现。

先核对标签、样本范围、基准、分箱拟合来源、缺失处理及指标单位，再比较数值。
KS 使用百分制，坏率与缺失率用比例，IV／PSI 无量纲；金额与币种独立记录。
`calculation_status`、诊断和原始空值一起解释未表现、未计算、失败、样本不足和未定义值。
没有状态行也不能证明所有指标均可用；不要将缺失值解释为 0 或“没有风险”。
字段语义按报告和表选择：画像／分箱的 `calculation_status` 使用
`computed/not_computed/unobserved/undefined/insufficient_samples/failed/skipped`；
`no_target` 是未提供目标，`no_observed_labels` 是目标没有已观测标签。
Score Cross 的格子、边际和整体表另用 `valid/low_sample/empty/unobserved/not_requested/invalid_denominator`，
不能将这些状态套到通用计算状态或 schema/unseen 的比较状态上。可计算的零与未计算、失败、空值分别解释。

## 业务元数据与稳定英文身份

计算时可传 `feature_metadata`，键为原始英文字段名，字段包括 `display_name`（中文名）、
`data_source`、`description`、`unit`。也支持含 `feature` 列的 Pandas／Polars 字典表。
报告不会重命名原宽表，也不在每个分箱／日期行重复长描述。

```python
metadata = {
    "income": {"display_name": "月收入", "data_source": "application",
               "description": "申请人申报月收入", "unit": "CNY/month"},
    "salary": {"display_name": "月收入", "data_source": "bank",
               "description": "授权流水入账月收入", "unit": "CNY/month"},
}
```

中文名可以重名。`report.search_features("月收入")` 返回候选及英文键；结合来源和定义确认后，
使用 `get_feature("income")` 或 `get_table(..., features="income")` 查询。
中文展示标签不能替代稳定 ID。

`business_context` 是调用方登记的 JSON 业务解释，可以包含：

| 内容 | 示例键及用途 |
| --- | --- |
| 标签定义与表现期 | `labels={target: {definition, positive_class, negative_class, performance_window}}` |
| 样本范围 | `sample={scope, filter, time_range}` |
| 切片说明 | `splits={train, val, test, oot}`，按实际存在的切片登记 |
| 金额与币种 | `currency` 及明确的金额口径 |
| 模型分方向 | `score_direction`，交叉分析还须显式提供计算方向参数 |

未登记的解释保持 `unknown`。`describe().context_source` 区分用户提供的信息与实际执行参数；
用户说明不能覆盖权重、分箱、筛选或样本处理事实。完整可运行字典例子见下方 portable_reports。

## 分页与证据回放 { #分页与证据回放 }

继续使用上面加载的 `restored`：

```python
page = restored.query_page(
    "summary", features="income", sort_by="iv", descending=True,
    columns=["feature", "iv", "ks"], offset=0, limit=10,
)
evidence = page["reference"]
if page["next_offset"] is not None:
    next_page = restored.query_page(
        "summary", features="income", sort_by="iv", descending=True,
        columns=["feature", "iv", "ks"], offset=page["next_offset"], limit=10,
    )
```

每页包含 `total_rows`、`returned_rows`、`omitted_rows`、`next_offset` 与 `reference`。
引用保留报告身份、表名和查询参数；保存后可在同一快照重放。
先筛选／排序，再列投影和分页，避免先把完整表转换为 Pandas／JSON。
筛选只支持已有字段及限定操作符，不执行 SQL 或任意表达式。
仅投影数值时，`to_ai_context()` 的 `evidence.identities` 按相同顺序补充每行未输出的特征身份列；
与 `rows` 按行对应。相关性等关系表保留两端身份，规则关系继续采用已保存的桥接身份。
`description.feature_metadata` 只关联返回页涉及的实体和必要 scope，不含未返回的请求特征。

## JSON 摘要与字符预算

```python
context_json = restored.to_ai_context(
    max_chars=16000,
    queries={
        "summary": {"features": "income", "sort_by": "iv", "descending": True,
                    "columns": ["feature", "iv", "ks"], "limit": 5},
        "detail": {"features": "income", "columns": ["feature", "count", "bad_rate"],
                   "limit": 4},
    },
)
```

预算覆盖最终序列化 JSON 的 Unicode 字符，包括目录、元数据与省略说明，**不是 token 数**。
可以选择表、特征、来源、筛选条件及列；不同表使用 `queries` 指定不同参数。
超预算会省略完整字段、行或说明块，记录数量、原因与继续查询引用。
`evidence.query` 始终表示最终生效的查询；裁列后写入实际 `columns`，裁行后写入实际 `limit`。
直接使用这份查询可重建展示的行、列、顺序和值，无需自行补裁剪参数：

```python
import json

context = json.loads(context_json)
for item in context["evidence"]:
    replayed = restored.get_table(item["reference"], **item["query"])
```

`identities` 与业务元数据也计入预算；裁行时同步裁减，元数据不会夹带后续页特征。
非空查询至少保留一条完整证据；必要说明和这条证据无法一起放入预算时抛 `ValueError`。
可提高预算，或用 `columns` 投影需要的字段。计算为空与预算拒绝保持区别，完整数据仍在报告和快照中。
多个字段相同的空值口径可合并到表级 `field_defaults.null`，字段继承该默认定义；指标单位和含义仍逐列说明。
Null 使用 JSON null；非有限浮点使用 `$mars` 类型标记；数值 0 保持 0。
筛选标量及 `eq`／`in` 的值接受同一受限标签，因此 JSON 往返、保存加载及新进程可直接重放。
只接受 `{"$mars":"float","value":"nan"}`、`"inf"`、`"-inf"` 三种值且不允许多余字段；
未知或畸形标签明确报错。普通 `"inf"`、`"-inf"`、`"NaN"` 字符串保持原类型，
NaN 沿用查询后端的既有谓词语义，不承诺与自身相等，也不新增过滤操作符。

人工对话：将 `context_json` 粘贴到另一个对话框，并要求回答引用表与样本口径；摘要范围外的
问题需补充查询结果。具备工具能力的外部 Agent：在 Python 环境加载完整快照，按目录读取
所需证据。普通对话模型不会因收到 `.marsreport` 就自动获得 Python 执行环境。

## 保存与恢复的能力边界

`.marsreport` 是既有 ZIP + manifest + Parquet 格式（版本 1），保留所有公共统计表、
表／列顺序、报告身份、业务元数据、实际参数、状态、诊断及来源。
它不是 pickle，不保存任意可执行对象，不用于恢复原分析器或任意重算。
目录与类型必须匹配；当前没有跨格式版本迁移器，跨 Pandas／Polars 版本的 dtype 迁移也不保证。

| 恢复对象／原报告类型 | 可继续使用 | 边界 |
| --- | --- | --- |
| 所有支持公共契约的 `ReportSnapshot` | describe、查询、证据、show_table、HTML／Excel、再次保存 | 不普遍恢复原对象 show_summary、分箱器、模型或拟合状态 |
| 画像／分箱 | 已有统计表、业务字典、状态与趋势 | 不能重算新样本或新分箱 |
| 相关性 | 专用邻居、子矩阵、筛选证据与矩阵展示 | 不重新计算另一方法／另一缺失口径 |
| 模型分交叉 | 已保存格子、边际、箱定义及支持的规则回放 | 无个体行；不能区分格子内个体或改变原始聚合维度 |
| 规则报告（Experimental） | 候选审计、验证、解释、按规则/特征/业务来源关联查询及显式高级分析 | 不重建 RuleSet 部署资格，不 transform 新样本；规则指标不按特征展开 |
| 其他领域对象 | 仅其已实现的公共契约或专用 artifact 能力 | RuleSet／模型 artifact 不等于通用报告快照 |

恢复后的展示使用 `restored.show_table("summary", limit=10)`，导出使用
`restored.write_html(...)`／`write_excel(...)`；不能假定原对象所有方法都恢复。
详细格式、原子保存与非有限数约定见[报告保存与恢复](reports-and-exports.md#portable-analysis-reports)。

## 相关性与模型分交叉

相关性快照的 `pairs`、`correlation_decisions`、`selection` 记录候选池、方法与实际筛选证据。
`mars.reporting.get_related_features`、`get_correlation_matrix`、`show_correlation_matrix`
消费公共查询结果，保存后仍可使用；默认矩阵展示限制范围，完整对仍可分页查询。

交叉报告的 `cells`、`row_summary`、`column_summary`、`overall`、`bins` 记录固定分段，
方向、缺失／invalid 箱、基准、已表现分母、权重、金额与币种均须保留。
恢复后可使用 `mars.analysis.evaluate_score_policy` 回放支持的规则，
以及 `get_score_cell`／`show_score_matrix`／`write_score_cross_html`。
回放使用已保存聚合证据，不访问原始样本，也不保证新政策的未来收益。
具体查询与规则配置见[相关性与模型分交叉](correlation-and-score-cross.md)。

## 完整示例 { #完整示例 }

[规则报告与外部 Agent 完整案例](rule-reports-and-agents.md)演示模型分交叉 → 开发候选 →
独立 production 验证 → 两类快照 → 新进程分页追问与证据复核，
提供无 API Key 的确定性验收及可直接交给外部编程 Agent 的任务 Prompt。

- [报告查询源码](https://github.com/leeesq/mars-risk/blob/main/docs/snippets/report_queries.py)：筛选、投影与上下文。
- [跨进程报告源码](https://github.com/leeesq/mars-risk/blob/main/docs/snippets/portable_reports.py)：业务字典、重名搜索、新进程恢复与人工导出。
- [相关性与交叉源码](https://github.com/leeesq/mars-risk/blob/main/docs/snippets/correlation_and_score_cross.py)：合成数据、保存后的专用消费和策略回放。

跨进程示例不需要 LLM 或重训练：

```python
--8<-- "docs/snippets/portable_reports.py"
```

## 可选的内置 Agent

`mars.agent` 是公共报告契约的消费者。`session.register_report(restored)` 可登记已有快照，
`get_report_table`、`describe_report`、`get_report_context` 继续查询，不必重新登记原数据。
运行器的计算预算只约束新计算，读取已有大型报告继续遵守分页与输出预算。
工具、安装和会话示例见[内置 Agent](agent.md)。开发 MARS 本身的规则在仓库工程 Skill，
它不是普通报告使用入口。
