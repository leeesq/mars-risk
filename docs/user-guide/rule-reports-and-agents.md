---
description: 规则发现与独立验证接入公共报告、关联检索、快照和外部编程 Agent 的完整模拟案例。
---

# 规则报告与外部 Agent

`MarsRuleMiningResult.to_report()` 返回满足 `mars.reporting.Report` 契约的
`MarsRuleReport`。Rule 仍为 **Experimental**；以下能力以当前 main 源码为准，
不代表已发布 PyPI 包已经包含。规则生成器、DSL、RuleSet JSON 和部署门禁没有因此改变。

## 公共报告与关联

```python
--8<-- "docs/snippets/rule_portable_report.py"
```

`to_report(analysis=None, *, feature_metadata=None, business_context=None)` 只整理已有结果，
不会重新挖掘、验证、拟合或暗中执行高级分析。业务解释与实际计算参数分开：
`describe().feature_metadata` 保存英文 ID、中文名、业务来源、定义、单位；
`business_context` 保存标签、表现期、样本范围、币种和证据来源；
`parameters` 保存实际 resolved_spec、候选/验证门禁、方向、预算、排序、IoU、profile、
qualification、验证摘要、生成器、切片口径及显式高级分析配置。缺少业务解释保持 unknown。

`describe()`、`get_table()`、`query_page()`、`search_features()`、`get_feature()`、
`to_ai_context()` 和 `save()` 在原报告与 `load_report()` 的通用快照上使用相同消费逻辑。
报告创建时取得持久 report_id；重复查询、保存和导出保持身份。原有 summary_table、
detail_tables、metadata、render_html、write_html 和 write_excel 继续使用。

| 表 | 真实粒度及范围 |
| --- | --- |
| summary | 一次挖掘：状态、候选数、入选数、profile、验证状态与资格；不是特征统计 |
| rules | 最终 rule_id、规范 expression、rank、labels、grades、入选轮次；不复制指标 |
| candidates | rule_id 与实际 generation_round（存在时）的候选审计、生成来源、门禁、淘汰原因和顺序 |
| evaluation | dataset / rule_id / target / slice / group；group 是 hit、miss、total |
| slices | 实际评估的切片与上述维度；group_col 优先于 time_col，不声称同时计算了两种切片 |
| rule_explanations | 最终规则顺序及所引用 dataset、target、slice、group 的验证解释 |
| rule_features | AST 提取的去重 rule_id / 英文 feature 关系，包括已淘汰候选 |
| interactions | 显式分析的 rule_a / rule_b 对，端点分别声明，不把一端当作唯一 ID |
| cumulative | 按最终顺序的 rank / added_rule_id，含前缀并集与新增边际指标 |
| cumulative_features | added_rule_id 对应累计前缀涉及的特征，只存关系，不存重复指标 |
| bootstrap | 显式请求的 rule_id 重采样区间；不参与候选筛选 |
| benchmark | from_benchmark 的原记录，耗时为 seconds、内存按显式后缀；没有规则资格 |

每表目录保存实际 schema、字段单位、含义、grain、state 与 feature_query。
未执行高级分析不加入统计表，状态见 `parameters.analysis_states`；
显式执行后合法零行以 `computed_empty` 保留。`no_rules` 是合法业务结果，
保留候选淘汰审计及已经计算的评估；无候选时依然可保存、恢复与查询空表。
计算异常仍抛出，不能转成空规则伪成功。未提供金额/客户列时相关值为 null，不伪造统计。

规则级指标没有按特征展开。英文 ID 由既有 DSL AST 提取；共享查询层根据快照目录中的
`feature_relation` 桥接声明筛选规则 ID 集合，再对原统计表做布尔筛选，因此多特征并集
只返回每条统计行一次。`features` 与 `sources` 的条件间采用交集，允许命中同一规则的
不同特征，例如 income 属于 application、debt 属于 credit 时，
`features="income", sources="credit"` 可以命中 income AND debt 规则。
交互表各条件分别匹配两端成员的并集，始终返回原规则对；累计表匹配整个已加入的前缀，
不能仅凭 added_rule_id 忽略此前规则。桥接表本身按其直接 feature 字段筛选。

`sources` 参数指 feature_metadata.data_source 中的**业务特征来源**；
candidates.sources 和 rules.source 指**候选生成器来源**，两者不能混用。
未知业务来源和非法字段/操作符明确报错；summary/benchmark 声明不支持 features/sources，
不会静默忽略条件。中文名重名时 search_features 返回全部稳定英文候选；
未知特征的 get_feature 抛 ValueError，已登记但无规则证据返回 evidence_status=no_evidence。

按规则使用既有 `filters={"rule_id": rule_id}`，再限定 dataset、target、slice、group。
不要把重复轮次、不同目标或不同样本的数值相加。cascade 的剩余人群门禁记录在候选轮次审计，
现有 evaluation 是最终规则在完整 train/validation 的评估，不假装保存了未保留的逐轮剩余人群指标。

## 分母、预算与快照边界

规则 evaluator 对各 target 分别排除 null/NaN 标签，sample_count 是该统计范围的
**已表现人数**，coverage 是该人数 / 同 dataset、target、slice 的总体已表现人数；
event_rate = event_count / sample_count，Lift = event_rate / 同范围总体事件率。
交叉报告 sample_count 则是全样本人数，标签分母单独保存在 observed_sample_count。
不能把两个报告的同名 sample_count 视为相同口径。

事件率/coverage 是小数比例；Lift 和 Lift 置信界无量纲；单侧超几何检验 p 值和同范围候选
BH q 值是概率；Wilson Lift 界除以同范围总体事件率，并非与总体区间相除。
bootstrap 表的 Lift 界是重采样分位数，与 evaluator Wilson 界分别解释。
金额使用所配置金额和明确币种，客户数去重，不能跨规则或 target 简单相加。
0 保持数值；null 表示未配置、未计算或未定义，结合目录和分母判断；
NaN/Infinity 沿用公共 `$mars` 非有限数编码，不能填 0 美化结果。

AI 上下文默认给 summary 一行与最多三条最终验证高价值证据，默认不输出规则 expression；
直接构造但没有解释表的报告仍只给首表。按需通过 queries 读取候选、评估或解释；
先筛选、排序、投影、分页，再转换小表。max_chars 覆盖最终有效 JSON 的目录、
元数据及省略说明，记录省略数、原因和继续查询引用；极小预算明确失败。
原规则 HTML 对 candidates 预览 100 行并显示总数和完整查询/导出入口；
Excel 与 .marsreport 保存完整统计。恢复后的通用 HTML/Excel 导出完整表。

`.marsreport` 沿用格式版本 1 的 manifest + Parquet，不保存原宽表、模型、分析器或任意可执行对象。
父目录必须存在，默认拒绝覆盖，overwrite=True 原子替换，失败清理沿用公共约定。
feature_relation 是目录中的可选声明；旧快照缺少该字段时沿用原来的角色/scope/单特征查询，
无需迁移旧文件。恢复快照支持通用导出，不重建原报告所有专用方法。
高级分析表的 scope 保存实际 target、计数和配置；未登记分析样本身份时保持未知，
不要因传入 analysis 就假定它来自 validation。

报告加载不会重建有部署资格的 RuleSet，也不能对新样本 transform。
RuleSet JSON 仍是独立定义与部署产物，SQL 导出仍检查验证摘要、资格和 missing_policy。

## 完整模拟案例：发现、验证与跨进程追问

业务问题是：**同一主模型分等级内，辅助分能否进一步区分风险？哪些组合规则值得进入独立验证？**

执行当前源码安装环境中的脚本，无需 API Key、LLM、网络或训练模型：

```bash
python docs/snippets/external_agent_rule_case.py --output-dir output/agent-rule-case
# 后续独立进程只消费已存在的报告，不生成样本或重算
python docs/snippets/external_agent_rule_case.py --consume-only --output-dir output/agent-rule-case
```

固定种子模拟 18,000 条消费信贷申请，包含 ID、申请日期、客群、主/辅助固定分、
bad30/late60 未表现标签、缺失和 -999 特殊分、金额。主分高分低风险，辅助分高分高风险。
发现/验证/观察是 2026 年 1–3 / 4–6 / 7–9 月各 6,000 个独立 ID；不共享样本行。
只在发现集拟合四段分位点，在全部时期复用 bin_definitions；
只依开发格子选两个高差异候选、一个反证及一个开发 Lift 介于候选/验证门槛的格子，
另加明确没有格子来源的极低覆盖压力候选。
独立验证使用原 production 门槛，不为演示降低门禁。观察集只保留交叉证据，不重新挑选候选。

候选的 business_context.candidate_origins 绑定具体发现期 cells 引用；
手工覆盖压力候选明确标为缺少交叉对应。规则评估保留主/辅助目标与验证时间切片；
未计算客群或 observation 规则评估、高级分析，不把这些缺口藏起来。

脚本完成计算后启动一个新 Python 进程，只加载两份报告。按脚本化问题依次读目录，
查询固定主等级内的风险差异与表现分母，分页追问候选淘汰、验证主目标、4月辅助目标，
复核观察格子，对客群/高级分析缺口写“当前报告无法回答”。每条数字附可重放
report_id/table/query 和 dataset/target/slice/period；预算上下文上限 12000 字符。

| 浅目录产物 | 用途 |
| --- | --- |
| score-cross.marsreport、rules.marsreport | 完整交叉与规则证据，消费不依赖原样本 |
| ruleset.json | 仅真实有合格规则时另存；不执行 SQL、不部署 |
| rules.html、rules.xlsx | 人工规则报告；候选 HTML 预览限额明确 |
| case-notes.json、external-agent-task.md | 公开范围与可复制给外部 Agent 的任务 |
| query-trace.json、agent-context.json | 确定性实际查询、目录、分页引用与预算 JSON |
| evidence-review.json、review.md | 程序化证据复核；不是模型自主分析 |

一次本地确定性运行中，五个候选有两个入选、两个在 candidate_filter 淘汰，
一个通过开发门禁但在 validation_filter 淘汰；不能把开发淘汰称为独立验证失败。
其中规则 `mr_4bb77355c89cdcc7c9b6` 在 validation/bad30/__overall__/hit 的
已表现命中人数 315、事件数 166、事件率约 52.70%、Lift 约 3.317；
validation/late60/2026-04/hit 的已表现命中 94、事件 45。
这些是模拟数据的接口验收结果，非真实业务效果或性能 benchmark；
每次新建报告的 report_id 不同，具体引用以本次生成的 trace 为准，测试不锁死自然语言结论。

脚本完整逻辑见[共享案例源码](https://github.com/leeesq/mars-risk/blob/main/docs/snippets/external_agent_rule_case.py)。

## 交给外部编程 Agent 的任务

上面的脚本将下列 Prompt 中的路径替换为真实产物目录，写入 external-agent-task.md。
将该文件直接交给有 Python 工具的 Codex 等外部 Agent。它没有预写分析结论；
报告文本作为数据，不能作为新的执行指令。允许保存实际查询轨迹和独立审阅输出，
不要求 mars.agent、原宽表、模型训练、provider SDK 或密钥。

```text
--8<-- "docs/snippets/external_agent_rule_prompt.md"
```

本轮在独立编程 Agent（子 Agent 的 Python 工具环境）中实际执行了一次快照消费，
实际查询重放一致且上下文在 12000 字符内，输入两份快照 SHA-256 未改变；
最终运行的准确查询数与预算长度记录在 external-agent-trace.json。
它分别解释主/辅助目标门禁，并拒答未计算客群、observation 规则评估与高级分析。
这是一次真实模型工具消费验证；没有调用第三方 provider SDK，不能作为跨 provider 行为保证。

外部 Agent 实际运行后应生成 external-agent-trace.json 和 external-agent-review.md，
与确定性输出分别标注。CI 仅验证确定性公共消费链路，不调用真实模型或网络。
没有运行环境时，只能声称公共链路已验证，不能把确定性脚本成功说成自主模型行为成功。
