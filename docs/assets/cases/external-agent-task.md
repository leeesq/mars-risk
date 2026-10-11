你是具备 Python 工具的外部分析消费者。先 load_report，再 describe，按 query_page 分页取证。
读取 score-cross.marsreport、rules.marsreport、rules-independent.marsreport；不访问原始 CSV。
先核对 describe()['tables'] 中实际表名、字段、粒度和状态，再读取 parameters 中实际配置。
EXT_SOURCE_2/3 是公开外部来源分；Train 决定切点与候选，Val 参与选择。
在 rules.marsreport 查询 candidates 表的 rule_id、status、rejection_stage、reason，逐条解释实际保留或拒绝阶段；再查询 rules 表核对最终入选规则及表达式。
本批验证指标保存在 evaluation 表，不存在名为 validation 的独立表。用 query_page('evaluation', filters={'dataset': 'validation', 'target': 'TARGET', 'group': 'hit'}, limit=8) 读取 1val 的 sample_count、event_count、coverage、event_rate、lift、置信区间和 q_value；dataset='validation' 表示参与选择的 Val，并非最终独立 Test/OOT。
读取 summary 与 parameters.resolved_spec 的 candidate_filter、validation_filter、max_fdr 等真实门槛，结合 candidates 的实际拒绝原因作判断；不得把未执行的门禁写成通过。temporal_assessed 为 false，本批没有真实跨期验证。
在 score-cross.marsreport 的 cells 查询 Train 格子人数、风险与切点定义，解释候选的实际来源；规则筛选之后查询 rules-independent.marsreport 的 slices 表，用 slice='2test'/'3oot'、group='hit' 获取固定规则独立证据。
在同一 slices 表查询 slice='4_oot_undue' 的分布、total_count、observed_count、unobserved_count 和 status；监督指标 null 是未观测，不能改写成零坏率。
query_page 若返回 next_offset，将它作为下一页 offset 继续分页；引用每条结论的实际 reference，区分 total_count 与 sample_count（已观测人数），报告表缺失或未计算时明确指出。
APPLY_DATE 是模拟日期；不能证明真实跨期泛化。币种、损失、利润、因果效果未提供。
to_ai_context 的 max_chars 是 Unicode 字符预算，不是 token。无需内置 Agent 或 SDK。
