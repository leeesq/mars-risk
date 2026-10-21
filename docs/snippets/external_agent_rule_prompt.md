你具备 Python 工具，需要从已保存的报告中分析证据。先用 `load_report` 加载 `score-cross.marsreport`、`rules.marsreport` 和 `rules-independent.marsreport`，不访问原始 CSV。
调用 `describe()`，核对 `describe()['tables']` 中的实际表名、字段、粒度和状态，再读取 `parameters` 中的生效配置。随后用 `query_page` 分页取证。

`EXT_SOURCE_2/3` 是公开的外部来源分。Train 决定切点和候选，Val 参与规则选择。
先在 `rules.marsreport` 的 `candidates` 表查询 `rule_id`、`status`、`rejection_stage` 和 `reason`，逐条解释候选在哪个阶段保留或被拒绝，再查询 `rules` 表，核对最终入选规则及表达式。

本批验证指标保存在 `evaluation` 表，没有名为 `validation` 的独立表。
用 `query_page('evaluation', filters={'dataset': 'validation', 'target': 'TARGET', 'group': 'hit'}, limit=8)` 读取 `1val` 的 `sample_count`、`event_count`、`coverage`、`event_rate`、`lift`、置信区间和 `q_value`。
其中 `dataset='validation'` 表示参与选择的 Val，不能作为最终独立 Test/OOT 的证据。

读取 `summary`，以及 `parameters.resolved_spec` 中的 `candidate_filter`、`validation_filter` 和 `max_fdr` 等实际门槛，结合 `candidates` 记录的拒绝原因判断结果。
未执行的门禁不能写成通过。`temporal_assessed` 为 false，表示本批没有真实跨期验证。

在 `score-cross.marsreport` 的 `cells` 表查询 Train 格子的人数、风险和切点定义，解释候选来源。
规则筛选后，再查询 `rules-independent.marsreport` 的 `slices` 表，以 `slice='2test'/'3oot'`、`group='hit'` 获取固定规则的独立证据。
在同一张 `slices` 表中，查询 `slice='4_oot_undue'` 的分布，以及 `total_count`、`observed_count`、`unobserved_count` 和 `status`。监督指标为 null 表示未观测，不能改写成零坏率。

`query_page` 若返回 `next_offset`，就用这个值作为下一页的 `offset` 继续查询。每条结论都要引用对应的实际 `reference`。
区分 `total_count` 与 `sample_count`（已观测人数）；报告表缺失或指标未计算时，明确说明。

`APPLY_DATE` 是模拟日期，不能证明真实跨期泛化。报告未提供币种、损失、利润和因果效果。
`to_ai_context` 的 `max_chars` 按 Unicode 字符计数，不按 token 计数。完成这些查询无需内置 Agent 或 SDK。
