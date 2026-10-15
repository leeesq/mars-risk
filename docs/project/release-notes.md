---
description: MARS 0.0.28 的用户可见变化、兼容性说明和升级检查项。
---

# Release Notes

## Unreleased

- Notebook 分箱摘要在原后端按少量身份和 PSI 列统计，可信 summary 足够时不读取完整 detail；
  仅在身份不足时回退明细的 feature 列计数，仍与完整 HTML 使用同一口径。
- 修复 Pandas 3 下 Grouped Pivot 空组排序与取值异常，保留真实空组及其数值。
  宽趋势在真实空组与文字 `null` 同时存在时使用不冲突的空组展示列，raw KS 与原生 Excel
  沿用同一映射，明细组键保持原值。公开案例按正式流程同步报告身份列、查询证据与下载产物。
- 修复画像 Notebook 宽趋势按业务组名猜测比例的问题，Notebook/HTML/Excel 统一按字段和
  指标语义格式化；保留比例端点、整数与加权小数。nullable `dtype` 的 `pd.NA`、None、NaN
  先安全处理，再校验可信类型；未知原始类型不从聚合 Float64 推断，非空矛盾仍拒绝。
- 分箱 HTML 图表搜索纳入登记的中文名，保持英文身份、来源及搜索交集；`include_detail`
  真正控制 Grouped Pivot 区域及数据，与 `include_charts` 独立。Notebook 与完整 HTML
  统一按唯一英文特征计数，PSI > 0.25 的任一标签触发仅计该特征一次。
- 原生分箱 Excel 两引擎共用附表单位与精度，有内容的“最终汇总”作为默认打开页；
  原生分箱、画像及静态快照写入的业务字符串均按字面文字保存，保留数值、日期及模板公式。
- `query_page()` 默认 `limit=10`，显式页长只接受正整数或 None，拒绝 0、负数、bool 和浮点数；
  续页偏移严格推进。`get_table/get_feature/to_ai_context` 及每表 queries 的零行能力保留，
  空证据仍以 `limit=0` 经 `get_table` 回放，继续取证另给正页长查询。
- 画像与分箱的原生特征表默认增加 `display_name`、原输入 `dtype`、`data_source`，
  身份列顺序为 `feature → display_name → dtype → data_source`。显式 `columns` 仍是精确投影。
  旧 `.marsreport` 先按磁盘 schema 严格校验，再在内存补列；旧分箱趋势的聚合类型不作为原始类型。
- 画像 HTML 在 DQ/Stats 内增加指标页签，复用原值进行排序与交集搜索；率展示百分比，
  PSI/IV 保持无量纲数值，KS 保持 0–100 点，格式化不改变底层统计。
- 新分箱报告移除按特征重复的总体 `trend.bad_rate`。箱坏率、总体箱统计、金额坏率和
  原生风险曲线继续保留；历史快照已保存的表仍可查询。
- 原生风险图增加中文名、显式分组的真实样本日期范围，以及统一 `include_total=True`
  参数。隐藏 Total 面板不会改变全局统计、RC 参考、明细或计算分母。
- 原生分箱 Excel 的 raw/binned 源报告复用同一 22 列模板，主展示统一为分箱 KS（0–100 点）。
  导出 `ks` 按特征、目标和实际分组取 `max(detail.ks_bin)`；最终汇总取对应 Total，
  最终 KS 趋势保留首目标契约，综合指标保留模板的 average 聚合。导出说明区分源报告与
  Excel 口径，raw 诊断仍属于源报告。Python 的 raw `ks` 与逐箱 `ks_bin`、查询、AI 上下文、
  保存/加载、HTML 和绘图保持原样；通用快照 Excel 仍忠实导出保存的原报告数值。
  同时修复 openpyxl 明细空值残留及附加表筛选区域重叠导致的原生 Excel 拒开。

## 0.0.28

!!! note "版本与发布日期"

    本页记录 0.0.28 的版本变更。实际发布日期及安装可用性以
    [GitHub Release](https://github.com/leeesq/mars-risk/releases) 和
    [PyPI](https://pypi.org/project/mars-risk/) 为准。

MARS 的定位更新为**面向人和 AI Agent 的高性能信贷风控工具箱**。本版围绕数据画像、
分箱评估、特征筛选、相关性分析、模型分交叉和规则挖掘，将计算结果整理为可查询、保存、
交付和继续复用的结构化证据。

### 本版重点

- 统一公共报告、业务元数据、分页证据和 `.marsreport` 快照；外部 Agent 可以在新进程中
  消费已有统计，不依赖内置会话或原始宽表。
- 新增完整相关性证据与固定分段模型分交叉；保存结果后仍能查询、展示子矩阵或回放组合规则。
- 新增 Experimental `mars.rule`，覆盖候选生成、独立验证、淘汰审计、规则报告和部署资格检查。
- 改进批量计算、宽表投影和报告内存使用，补充可复现的规模验收、原始记录及测量边界。
- 监控、建模（含 Pipeline）、评分卡暂停功能迭代，保留现有能力、必要修复和直接上游适配。

### 公共报告、业务元数据与外部 Agent

- Stable `mars.reporting.Report` 契约与 `ReportSnapshot` 提供 `describe()`、`get_table()`、
  `query_page()`、`search_features()`、`get_feature()`、`to_ai_context()`、保存与加载。
  画像、分箱、相关性、模型分交叉和规则报告共用消费入口；具体分析模块的成熟度分别标注。
- 画像、分箱分析和 StatsSelector 接受统一 `feature_metadata` 与 `business_context`，记录
  中文名、业务定义、来源、单位、标签窗口及样本范围。原统计表的英文特征标识保持不变；
  核心分析的旧来源参数保留薄适配，冲突明确报错。
- `.marsreport` 保存持久身份、表目录、实际参数、元数据及 Parquet 统计表，不保存原始宽表、
  模型或任意可执行对象。外部 Agent 不需要 `MarsAgentSession` 即可加载、查询和引用证据。
- `queries` 支持异构表查询；趋势上下文将日期作为维度，只定义一次指标。筛选、排序、投影
  和分页先于展示或 JSON 转换；最终有效 JSON 的目录、身份、元数据和省略说明均计入字符预算。
- `evidence.query` 记录裁剪后的有效列和页长，可重建展示子集；`identities` 按行补充数值
  投影省略的特征身份，关系表保留两端。身份和元数据随实际页缩小，不升级快照格式。
- 非有限数 JSON 统一为 `{"$mars":"float","value":"nan|inf|-inf"}`，取代报告字符串标记或
  Agent null。普通 `"NaN"` 字符串不转换；未知、畸形或带多余字段的数值标签明确拒绝。
  查询筛选、JSON 往返、保存加载和新进程证据重放复用同一受限编码。
- 报告状态定义按实际报告及表选择，通用计算状态与 Score Cross 状态分离。格式版本一旧快照
  在语义读取边界修正文案，已有统计数值、状态编码和文件格式不变；合法的同列 key/feature
  桥接关系可以继续查询和生成 AI 上下文。

使用方式见[报告与交付](../user-guide/reports-and-exports.md)、
[外部 Agent 指南](../user-guide/external-agents.md)及
[规则报告与外部 Agent](../user-guide/rule-reports-and-agents.md)。

### 数据画像、分箱评估与特征筛选

- 分箱评估的同一 `batch_size` 覆盖当前/基准转换、聚合、按日缺失和特征起点参考；跨批次只
  保留小统计表。画像趋势在 `overview_batch_size` 内联合聚合指标，并复用口径一致的统计。
- 分箱器每次 fit 重建已学规则、WOE、映射和诊断；失败重拟合使成果入口失效，再次成功拟合
  可恢复。构造配置和显式更新的适用规则保留，避免读取上一次拟合的陈旧结果。
- 修复 Pandas bool/nullable Boolean 与 Polars Boolean 的类别映射；保留 Missing、Other、
  原始字符串大小写及编号，转换临时列不覆盖或泄漏用户列。
- 多目标省略 `features` 时一次排除全部目标和声明角色列；首目标参考拟合与首目标趋势语义
  不变。目标兼任角色或没有候选特征时明确报错。
- 公共分箱表现统计先验证二分类标签；null/NaN 保持未观测，非法标签（含 -1）明确拒绝。
  合法单类别标签仍可评估，不会产生负坏样本数或超过 1 的坏率。
- `mono` 在单箱或坏率恒定时返回未定义值，不再使用历史占位 `1.0`；原因见
  `calculation_status`。无标签与全未表现标签分别标为 `not_computed` 和 `unobserved`。
- WOE 并列箱按 `bin_index` 确定累计指标顺序；补齐特征起点参考与金额统计同时使用时的字段。
- 画像 Notebook 按实际 schema 选择默认排序；数值渐变和格式只作用于数值列，文本状态、
  nullable、全空列和空查询结果可正常展示。

### 相关性证据

- Stats/Linear Selector 新增 `get_correlation_report()`，保存完整候选关系、特征状态、
  筛选决策及元数据；原 `get_report()` 的决策 DataFrame 返回类型保持不变。
- Linear 的 Checked/max_corr 与筛选复用同一矩阵，避免重复计算；Stats 继续使用目标感知
  WOE Pearson，Linear 继续使用原始数值 Spearman 及现有缺失处理、阈值和优先级。
- 有符号关系按规范上三角存储，保留实际对角信息；报告快照固定后释放密集缓存。非有限
  相关值为 null + unavailable，不伪造 0 或对角 1；完整关系规模仍为 O(p²)。
- 查询支持两个端点的特征/来源关联、真实淘汰事件、有限展示和诱导子矩阵恢复。
  保存后的相关性报告继续支持这些操作，无需重新计算原始特征矩阵。

口径与示例见[相关性证据与双模型分交叉](../user-guide/correlation-and-score-cross.md)。

### 固定分段模型分交叉

- 新增 `cross_scores()` 及独立报告，先投影必要列，再联合聚合多个标签及实际 group/period。
  同一报告的目标、客群和时期共用固定分段，正常、missing、invalid 与特殊箱分别保留。
- 自动拟合复用 `profile_risk` 的 native/optimal/lite_opt 引擎和配置解析，支持
  quantile/uniform/cart、每轴箱数、参考集及明确监督目标。显式切点和保存定义无拟合复用。
- 保持正常箱右闭区间合同；已启用最小箱约束按最终右闭赋箱核对拟合参考集。
  固定定义用于 OOT 或独立验证集时不重新拟合，也不强制新数据保持参考集的箱占比。
- 自定义非有限缺失码优先于 invalid 判定；提取定义及保存加载后复用恢复合法浮点标签，
  保持 missing/invalid 分类，不修改调用方的定义。
- 保存分箱方向、真实 bin_id、风险顺序、显示标签和完整区间证据；边际和整体从整数计数
  重算，坏率、Lift、行内差异、权重、金额及未加权 Wilson 区间保留各自实际分母与状态。
- `evaluate_score_policy()` 支持可保存的 expression 类型，沿用组合规则聚合与特殊箱策略。
  正常格子转换规则不额外命中非有限分值；报告恢复后可继续查询、重导出及规则回放。
- 无时间维度时不把内部 `Total` 当作周期；实际范围来源随报告保存，单周期或缺少时间声明
  的旧快照明确提示验证边界。显示标签和 Lift 状态为加法字段，既有快照无需格式升级。
- 离线 HTML 提供固定风险编号矩阵、双向梯度、正常分箱规则及格子证据；风险指标统一显示
  Bad Rate，邻近切点提高显示精度。固定色阶、低样本纹理、选中与规则命中框保持独立。

### 规则生成、验证与部署

- 将 `deimos-rule` 来源快照 `e6714c5e795054e44f0c58ad7097668b4117b4a2` 重设计为
  Experimental `mars.rule`，纳入同一个 `mars-risk` wheel；没有 `deimos` namespace、
  `Dm*` 兼容别名或旧 RuleSet JSON 读取分支。
- 新增 `mine_rules()`、类型化筛选策略、五类生成器、固定长表评估、训练/验证隔离、切片稳定性、
  精确与 IoU 去重、`ranked` / `cascade` 选择和完整候选淘汰审计。
- 新增受限 DSL v2，明确区分 `NULL` 与浮点 `MISSING(null/NaN)`，并增加 schema 与资源预算
  fail-closed 校验。表达式经过 AST 解析、规范化、重复条件简化和明显矛盾检测。
- 新增严格 `schema_version=1` RuleSet artifact、同类型 `transform()`、等级命中计数、按需交互与
  累计分析，以及 HTML/Excel 报告。
- `cascade` 每轮在剩余训练与验证人群重新生成、筛选和审计候选；模型生成器使用显式缺失指示器，
  浅层树的 `n_jobs` 真实控制并行训练，Optuna 改用确定性分层 CV ROC AUC。
- 新增 explore/production profile、`exploratory`/`validated`/`temporally_validated` 资格、Wilson
  Lift 保守界、单侧精确检验与 BH-FDR 生产硬门禁；探索 RuleSet 默认禁止部署 SQL。
- 候选预算改为 seed 优先和生成器确定性轮询；IoU 使用批量压缩位图，切片评估使用单次分组聚合，
  高级分析复用一次命中矩阵并支持可选 top-k bootstrap。
- 高级分析恢复金额和客户维度的交互、累计与边际指标；报告新增结构化规则解释和 benchmark
  HTML 构造能力。44 项来源回归的迁移状态见[Deimos Rule 迁移矩阵](deimos-rule-migration.md)。
- 组合生成器在 500 个数值特征时直接复用 `MarsStatsSelector`；LightGBM 和 Optuna 分别复用
  `[ml]` 与 `[tuning]`，并保持延迟导入。

- `MarsRuleMiningResult.to_report()` 接入公共报告契约。多特征关系由 AST 提取，并通过桥接表
  筛选规则 ID；规则级统计不按特征展开重复。业务特征来源与候选生成器来源分别解释。
- 规则报告保存实际门禁、参数、资格、验证摘要和高级分析状态。`no_rules` 是合法结果，保留
  已计算证据及淘汰审计；未执行分析与合法零行分别表达，异常不转换成空规则伪成功。
- 规则报告可在新进程保存恢复、查询和生成有预算的 AI 上下文；`.marsreport` 是证据快照，
  不重建部署 RuleSet，也不能替代 RuleSet JSON 对新样本 transform 或导出 SQL。

### 兼容性与来源

- 公开入口仅为 `mars.rule`，不从根 `mars` 重复导出；模块状态为 Experimental。
- Python 支持范围仍为 3.8–3.12；核心规则测试进入全部版本矩阵，可选后端在 Python 3.10 验证。
- 来源和许可证记录见[规则模块来源](rule-origin.md)。旧 `deimos-rule` artifact 必须重新挖掘或
  重新导出，不能直接载入。

### 可选内置 Agent

- 新增 Experimental `mars.agent`，最低 Python 3.10；公共报告不依赖该模块。
- 提供登记数据、画像、风险评估、监控、报告目录/说明、分页查询、特征检索和上下文工具。
  `register_report` 复用公共快照及持久标识，无原始数据的报告明确记录来源。
- 支持跨轮会话、参数和证据引用、调用与上下文预算，以及结构化错误反馈。计算复用既有
  公开 API，不新增独立计算引擎。
- `MarsAgentComputeBudget` 对默认与显式 `features` 同等检查；超预算返回
  `COMPUTE_BUDGET_EXCEEDED`。原来省略 features 的调用需要缩小范围或显式调整预算；
  公共分析 API、报告查询和快照格式不变。
- OpenAI 兼容 provider 使用可选 `[agent]` extra，普通导入不加载 SDK，不自动读取数据文件
  或执行代码。`tests/agent` 验证真实 MARS 计算及模拟模型协议，不等同线上模型验证。

### 展示、交付与案例

- README 与首页使用原生高清分箱趋势图，保留完整分箱和分组信息；同步双语介绍、任务导航、
  外部 Agent 入口，交叉分析保留独立案例。
- 分箱 HTML 的业务元数据导航可独立使用，概况按特征去重，直接展示分组透视；未生成按日
  缺失统计时不显示空栏目。Profile/通用快照 HTML 的全局与局部搜索按交集生效，清空、
  排序和导航保留当前条件。
- HTML/Excel 附带业务元信息；补充 `snapshot_report(...).write_excel(...)` 静态当前值
  交付及独立进程示例。原分箱透视模板保留，原生 Excel 刷新需求单独说明。
- 实战案例包含实际筛选门禁、自动候选、独立验证、保存恢复和外部 Agent 任务。
  可在独立目录复现，下载证据记录实际 seed 与业务上下文；确定性验收与真实模型任务分别说明。

### 性能验收与工程

- 补充规则报告、短/长特征名相关性报告、画像、分箱、筛选和模型分交叉的分档规模验收；
  原始记录保留环境、依赖、线程、参数、耗时、进程 RSS、超时/内存预算及未运行场景。
- 保存后的消费在独立进程执行，检查查询、AI 字符预算、有限展示和同进程重复消费；
  原始结果和测量限制见[核心规模验收](../performance/core-capacity.md)，不将合成基准
  扩展为任意业务规模或所有操作系统的性能承诺。
- 容量 benchmark 的直属 worker 只由 `Popen` 回收，预算状态与真实退出码分别记录；
  基线核对工作量、实际诊断分支、参与依赖和测量合同，不可比较时不输出常规性能结论。

### 维护范围与 Python 支持

- 监控（Monitoring）、建模（Modeling，含 Pipeline）、评分卡（Scoring）暂停功能迭代。
  保留现有入口和文档，处理必要正确性、运行修复及上游适配；可借助 AI、核心分析报告和
  自有模型工具构建更贴合业务的流程。
- 这三个下游直接适配上游方法和结构；上游不为它们保留旧接口、旧字段、兼容分支或冗余
  计算。不承诺三个模块的历史签名或产物兼容，但保留当前仓库可运行。
  此边界不扩展到核心公共 API、报告字段或已保存分析文件的全部兼容要求。
- replay 在候选选择前合并指标方向，本次逐项覆盖优先于历史；None/空映射保留历史。
  `MarsModelReplayResult.metric_directions` 和 artifact 记录实际方向；旧 artifact 未记录时
  读为 `{}`，表示未知，不伪造历史方向；其他表和模型格式不变。
- Monitoring 的 Agent 查询实际执行登记来源筛选，`sources` 与 `features` 取交集；未知
  来源或缺少可靠来源信息的旧路径明确失败，避免返回虚假的全量证据。
- 基础包继续支持 Python >=3.8,<3.13；3.8 使用仓库冻结栈，Polars 1.8.2 与
  scikit-learn 1.3.x；3.9–3.12 限定 `polars>=1.33.1,<2`，避免自动安装尚未适配的
  Polars 2.0。3.9 使用依赖边界栈。可选建模、调参、Notebook、文档/开发工具和
  内置 Agent 以 Python 3.10+ 为支持环境，并非所有 extra 都通过元数据禁止旧解释器安装。
  Python 3.8 的运行兼容不延长其官方安全维护周期。

### 从 0.0.27 升级检查

本版发布收尾的依赖、文档、安装产物与报告交互结果见[0.0.28 验收记录](release-028-validation.md)。

- 对读取报告 JSON 的调用，采用 `$mars` 非有限浮点标签并保留状态语义；不要把 null、
  未表现、未计算或未定义指标一律填 0，也不要将普通 `"NaN"` 字符串解码为浮点数。
- 对依赖 `mono == 1.0` 占位或直接比较 NaN 的调用，改为同时检查 `calculation_status`。
- 清理标签中的非法编码；-1 不再被公共分箱表现统计当作合法 0/1 标签，null/NaN 仍表示
  未表现。多目标且省略 features 的调用要核对自动排除全部目标后的实际候选。
- 将原始数据交给分析入口，将保存报告交给查询/导出入口；核心快照与 RuleSet 部署产物
  分开管理。旧 Deimos RuleSet 必须重新挖掘或导出；格式版本一 `.marsreport` 不因本轮
  加法字段和关系声明而需要整体迁移。
- 模型分交叉跨期比较复用固定定义；不要在 OOT 重新拟合切点或把最小箱比例当作 OOT 约束。
  规则与交叉报告的同名计数字段可能有不同分母，按目录及已表现计数解释。
- 内置 Agent 使用 Python 3.10+ 和相应 extra，核对 `MarsAgentComputeBudget`；
  保留 `evidence.query` 和报告身份，使查询结果能够重放。
- 使用暂停模块的调用需随上游变化直接更新；replay 需要覆盖方向时提供明确逐项映射。
  依赖与升级路径见[安装指南](../getting-started/installation.md)。

## 0.0.27

本版本修正数据画像 PSI、nullable target 统计和统计筛选器空结果口径，并升级实验性的
缺失率异常扫描器。`MarsMissingShiftScanner` 仍不从 `mars.analysis` 顶层导出，本次配置与
结果 schema 调整属于有意的实验 API breaking change。

### 数据画像与筛选修正

- 显式 `benchmark_df` 中的退化特征不再导致整份 PSI 画像失败；不可计算特征保留为
  `null` PSI，并写入 diagnostics，其他特征继续计算。
- `profile_bin_performance()` 排除 target 为 null/NaN 的未表现样本，再计算 count、
  good/bad、WOE、IV、KS、AUC 和 Lift；全空标签给出明确错误。
- `MarsStatsSelector.fit()` 允许零特征存活，保留漏斗与决策报告，不再调用 `prune([])`。
- 普通趋势、PSI 和 unseen 画像删除 `group_mean`、`group_var`、`group_cv`，保留逐组值和
  `total`。

### 缺失率异常扫描

- 新增 `MarsMissingShiftConfig`，阈值和检测器配置不再作为 `scan()` 的扁平参数传递。
- 同时支持 `segment_shift`、`boundary`、`point`、`high_level`，可准确定位首日或末日异常、
  内部单日尖峰、持续分段变化和长期高缺失。
- 统计候选统一经过最小效果门槛与 Benjamini-Hochberg 全局 FDR；小期望频数使用 Fisher
  精确检验，其余使用两比例检验。
- 低样本日期保留在长版 `trend_table`，但不参与检测，也不会跨日期桥接检测窗口。
- 重叠检测证据合并为一个业务事件，并通过 `detected_by` 保留来源。
- `MarsMissingShiftResult` 新增 Notebook 格式化表、趋势图和四表格式化 Excel 导出；
  `trend_table` 取代旧的宽版 `missing_rate_table`。

### 升级检查

- 将缺失率扫描调用迁移到 `MarsMissingShiftConfig`，并改用 `trend_table`。
- 如果业务需要识别扫描区间第一天的相对异常，优先提供扫描期之前的 `benchmark_df`；
  没有 benchmark 时只能使用后续有效日期或绝对高缺失红线。
- 持续高缺失默认红线为 90%，自然稀疏特征应通过
  `feature_high_missing_rate_thresholds` 显式覆盖。

## 0.0.26

发布依赖补充 `Jinja2>=3.1.2`，确保默认安装即可使用基础报告、特征筛选器和
`Pandas Styler` 展示接口；Python 3.8 冻结栈使用 Jinja2 3.1.6 与 MarkupSafe 2.1.5。

该版本将基础包的运行范围扩展到 Python 3.8–3.12，并对 Analysis、Feature 和
Reporting Stable API 执行 fail-closed 收口。本版本包含明确的 API、报告和序列化 breaking changes，
升级前必须按本页的迁移清单核对。

### Python 与依赖兼容

- Python 3.8 固定使用 Polars 1.8.2，并将 scikit-learn 限制在 1.3.x；仓库通过
  `constraints/python38.txt` 固定验证栈。Windows 环境固定 OSQP 1.0.4，避免旧 0.6.x
  在 Polars 已加载后导入时的原生库崩溃。
- Python 3.9 使用现代 Polars 与 scikit-learn 1.6.x；Python 3.10–3.12 延续当前现代依赖。
- Python 3.8 已停止官方安全维护。MARS 的兼容承诺仅表示冻结栈可以运行，不延长解释器的
  安全支持周期。
- `ml`、`tuning`、`notebook`、`docs` 和 `dev` extras 要求 Python 3.10+；Python 3.6、3.7
  以及 3.13+ 不在本版本支持范围内。

### 实现与结果口径

- 新增内部 Polars 兼容层，统一 membership 与 streaming collect 的跨版本差异。
- KS/AUC 的前一累计分布改为“当前累计值减当前箱分布”，以兼容 Polars 1.8 的窗口表达式
  限制；指标结果与现代 Polars 保持一致。
- Python 3.8 语言兼容改造本身只涉及注解求值、dataclass、zip 和字符串后缀处理，
  不改变业务算法；本版本的 Stable API 与 artifact 变更单独列在下文。
- 特征筛选的监督指标和 WOE 相关性只使用 target 非空的已表现样本；质量、分布和 PSI 仍
  使用全量样本。
- Optimal Binner 的失败特征统一批量回退到 Native Binner，避免随失败特征数增长的重复拟合。
- `profile_stats()` 与 `MarsDataProfiler.generate_profile()` 新增 `benchmark_df`：基准样本只负责
  PSI 分箱和 expected distribution，不进入当前数据的质量与统计指标；未分组时可直接输出
  当前全量相对 benchmark 的 `total` PSI。
- 数据画像删除 `sample_frac` 参数。抽样改由调用方在传入前显式完成；仍传该参数的旧调用会
  收到 Python 标准 `TypeError`，升级时应删除参数并在外部准备抽样 DataFrame。

### Stable API 与报告加固

- Binner `transform()` 新增 `features` 和 `on_missing`，默认要求全部规则列齐全；Selector
  `transform()` 使用相同的严格缺列策略。`update_bins()`、`prune()` 和 `get_bin_mapping()`
  不再静默忽略未知特征。
- 三种 Binner 新增固定 schema 的 `get_fit_report()`。合法 fallback 可继续，真正无规则的
  特征标为 `failed`，全部失败终止。
- `to_dict()` / `from_dict()` 改为 `schema_version=1` 的自描述 artifact，新增 `save_json()`
  和 `MarsBinnerBase.load_json()`。旧 `{params, state}` 载荷不兼容，必须重新拟合或导出。
- WOE transform/SQL 必须具备完整 WOE 映射；SQL 类别值使用安全引号转义。
- 报告级指标列缺失会终止；单特征空值、NaN 或 Inf 指标以 `metric_unavailable`
  淘汰并记录。`MarsImportanceSelector` 删除未实现的 `rfe` / `sfm` 公开选项。
- Excel、HTML、JSON 写入、资源读取和空报告导出失败现在会显式抛异常；Binning
  HTML 如需图表，任一图表构建失败会使导出失败，可显式使用 `include_charts=False`。

### 画像对比能力

- `profile_stats()` 和 `generate_profile()` 新增 `categorical_features`，使整数编码类别同时进入
  unseen 与类别 PSI 口径。
- 新增显式 `schema` 和 `unseen` metrics，不加入默认指标。Schema 表区分两侧列存在性、
  兼容和不兼容 dtype 变化；unseen 排除缺失与特殊值，输出 total 与分组趋势。
- `MarsProfileReport` 新增 `comparison_tables`、`report_meta` 和自包含交互式 `write_html()`。
  Profile Excel 新增 Metadata 和 comparison 工作表。
- `ProfileData` 从三字段扩展为四字段，位置解包调用需增加 `comparisons`。

### 发布与工程门禁

- 普通 CI 和 Release 均只构建一次 wheel/sdist，静态核对版本、Python 范围、依赖 marker、
  `py.typed`、Excel 模板和 dist-info，再运行 `twine check`。
- 同一 wheel 会在 Python 3.8 与 3.12 全新环境中独立安装，并验证递归导入、`profile_risk`、
  selector、Pandas Styler、Excel/HTML 报告和安装后的模板资源；发布 job 不再重新构建。
- Mypy 固定为 1.13.0，并以 Python 3.8 为统一目标检查全部 `src/mars`；业务模块 override 与
  源码 `type: ignore` 已清零，第三方动态边界通过显式类型收窄处理。

### 升级检查

- Python 3.8 环境按约束文件重建，不要在已有环境中强制覆盖整套依赖。
- 使用可选建模或文档依赖时升级到 Python 3.10+。
- 使用数据画像内部抽样的调用，改为先对 DataFrame 显式抽样，再调用 `profile_stats()` 或
  `generate_profile()`。
- 将 Binner 旧 dict/JSON 载荷全部用 0.0.26 重新拟合或 `save_json()` 导出。
- 将依赖缺列静默忽略的 Binner/Selector 调用改为显式 `features` 或 `on_missing`。
- 将 `ProfileData` 的三元素解包改为四元素，并移除 Importance Selector 的 `rfe` / `sfm` 配置。
- 发布前同时验证 Python 3.8 冻结栈、Python 3.9 依赖边界和 Python 3.10–3.12 现代栈。

## 0.0.24

从当前公开版本 `0.0.21` 升级到 `0.0.24` 时，重点核对分析报告链路、基准样本语义、
趋势图时间范围、HTML 大报告和 Modeling/Pipeline 契约。

### 用户可见变化

- `profile_risk()` 返回 `MarsRiskProfile`，同时提供 `report`、`binner`、`targets` 和 `metadata`。
- `benchmark_df` 统一用于基准期分箱和 PSI expected distribution，不进入当前期 Total。
- `MarsStatsSelector.fit()` 支持 `benchmark_df`，筛选指标仍在当前 `df` 上计算。
- 风险趋势图的时间范围只来自有效 `time_col`；`group_col` 只负责面板分组。
- HTML 报告支持可检索视图、图表数量控制、图片资产模式和懒加载。
- Modeling/Pipeline 增加结果对象、replay、artifact 和多 target 评估能力，状态仍为 Experimental。

### 升级检查

- 将旧代码中直接假定 `profile_risk()` 返回 report 的访问改为 `risk_profile.report`。
- 使用固定分箱规则时改用 `MarsBinEvaluator.evaluate(..., binner=...)`。
- 生成趋势图或 Charts HTML 前显式提供有效 `time_col`。
- 使用 `MarsStatsSelector` 默认 PSI/RC 阈值时提供 `group_col` 或 `time_col`；静态筛选应显式关闭
  对应阈值。
- 升级 Modeling/Pipeline 后重新核对结果字段和 artifact 路径。
