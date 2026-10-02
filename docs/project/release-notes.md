---
description: MARS 0.0.28 的用户可见变化、兼容性说明和升级检查项。
---

# Release Notes

## Unreleased

- README 与首页改用 MARS 原生高清分箱趋势图，保留完整分箱与分组信息；
  双语入口、项目介绍及外部 Agent 场景同步更新，交叉分析保留为独立案例。
- 分箱 HTML 的业务元数据导航独立可用，数据概况按特征去重，分组透视直接展示；
  未生成按日缺失统计时不提供空栏目。最小示例接入真实日期并恢复原生图表。
- Score Cross HTML 的边界显示使用适当精度，邻近切点自动提高精度；
  原始定义、完整边界证据和规则回放不变，图表风险指标统一使用 Bad Rate。
- 修正案例中正常交叉箱转换为规则时额外命中非有限分值的问题；
  候选条件与来源正常箱验证成员一致，重新生成对应独立验证结果与下载资源。
- 实战案例补充实际筛选门禁、自动候选生成和对应外部 Agent 任务；
  复现材料可在独立目录运行，报告来源采用本次真实 seed 与上下文。

- 画像 Notebook 比较表按实际 schema 选择默认排序；数值渐变和格式只作用于数值列，
  文本状态、nullable、全空列和空查询结果可实际渲染。
- Profile 与通用快照 HTML 的全局和局部搜索按交集生效，清空、排序和页面导航保留当前条件。
- 补齐已有 `snapshot_report(...).write_excel(...)` 静态当前值交付示例与独立进程回归，
  原分箱透视模板保留，原生 Excel 刷新需求单独说明；报告索引补齐相关性与模型分交叉。
  安装提示区分未发布源码0.0.28与已发布0.0.27，不提升版本。

- 分箱器每次 fit 重建已学规则、WOE、映射与诊断；失败重拟合使成果入口失效，
  再次成功拟合可恢复。构造配置和显式更新的适用规则保留。
- 修复 Pandas bool/nullable Boolean 与 Polars Boolean 的类别映射，保留 Missing、Other、
  原始字符串大小写和编号；转换临时列不覆盖或泄漏用户列。
- 多目标省略 features 时一次性排除全部目标和声明角色列；首目标参考拟合与首目标趋势语义
  不变。目标兼任角色或没有候选特征时明确报错。
- 公共分箱表现统计在聚合前复用二分类标签校验；null/NaN 保持未观测，非法标签（含 -1）
  明确拒绝，避免负坏样本数或坏率超过 1。合法单类别分箱表现统计保持可用。
- Monitoring 的 Agent 兼容查询真实执行登记来源筛选；`sources` 与 `features` 取交集，
  未知来源或缺少可靠来源信息的旧路径明确失败，不再返回虚假的全量成功证据。
- 报告状态定义按真实报告与表选择，通用计算状态与 Score Cross 状态分离；版本一旧报告
  在语义读取边界修正文案，已有统计数值、状态编码和文件格式不变。
- 查询筛选复用既有受限非有限浮点标签，JSON 往返、保存加载与新进程可重放；
  未知、畸形或带多余字段的标签明确拒绝，普通字符串及现有 NaN 谓词语义不变。
- 预算上下文的 `evidence.query` 记录裁剪后的有效列和页长，可直接重建展示子集；
  `identities` 按行补充数值投影省略的既有特征身份，关系表保留两端。
  身份和业务元数据随实际页及裁行缩小，全部开销计入完整 JSON 预算，无格式升级。

- 修复 Score Cross 无时间维度时把内部 `Total` 当作周期的问题；补充实际范围来源元数据，
  单周期与缺少时间声明的旧快照明确提示验证边界，保存统计键和证据契约保持兼容。
- 提高离线交叉矩阵的小字、热力背景文字、SVG 标签及禁用控件对比度，保留固定色阶、
  低样本纹理、选中外框与独立规则命中内框。

- 修复 Score Cross 自动拟合的左闭切点转右闭定义后最小箱约束失效，以及定义提取、保存加载
  后自定义非有限缺失码变成 invalid 的问题；显式切点与固定定义不重新拟合。
- 修复合法的同列 key/feature 桥接关系生成 AI 上下文失败，保留通用快照查询与统计行粒度。
- 容量 benchmark 的直属 worker 退出状态仅由 Popen 回收，预算状态与真实退出码分别保存；
  基线对照核对工作量、有效诊断分支、参与依赖和测量合同，不可比较时不输出常规性能结论。

- Score Cross 自动分箱复用 `profile_risk` 的 native/optimal/lite_opt 引擎与配置解析，
  支持 quantile/uniform/cart、每轴箱数及明确监督拟合目标；旧切点与保存定义继续无拟合复用。
- Score Cross 离线 HTML 接入固定风险编号矩阵、双向梯度、完整区间证据及受限正常分箱规则。
  `evaluate_score_policy` 增加可保存的 expression 类型，沿用既有规则聚合与特殊箱策略。
  新增显示标签和 Lift 状态是加法字段；旧 `.marsreport` 仍可查询、重导出和回放，无格式升级。

- 定位统一为“面向人和 AI Agent 的风控分析工具箱”；README、首页与导航增加外部 Agent 入口。
- Monitoring、Modeling／Pipeline、Scoring 暂停功能迭代，保留功能、必要修复与直接上游适配。
- replay 在候选选择前统一合并方向，本次逐项覆盖优先于历史；None／空映射都保留历史。
  `MarsModelReplayResult.metric_directions` 及 artifact 增加实际方向记录。旧 artifact 未记录方向时
  读为 `{}`（未知，不伪造历史方向）；其他表、模型格式不变。
- 内置 Agent 新增 `MarsAgentComputeBudget`，默认与显式 features 同等检查；超预算返回
  `COMPUTE_BUDGET_EXCEEDED`。原来省略 features 触发无限制计算的调用需缩小范围或调整预算。
  公共分析 API、报告查询和 `.marsreport` 格式不变。

- 画像、分箱分析和 StatsSelector 接受统一 `feature_metadata` 与 `business_context`；支持中文名、业务定义、来源、单位及多标签窗口。旧来源参数保留薄适配，冲突明确报错。
- Stable `mars.reporting.Report` 公共契约与 `ReportSnapshot` 支持检索、分页证据和单文件 `.marsreport` 保存恢复；外部 Agent 无需内部会话即可查询。
- 趋势上下文将日期作为维度，只定义一次指标；`queries` 支持异构表查询，最终 JSON 按字符预算裁剪完整字段、行或说明块。
- 非有限数 JSON 从报告字符串标记/Agent null 统一为 `{"$mars":"float","value":"nan|inf|-inf"}`。消费者需迁移解码规则，普通 `"NaN"` 字符串不转换。
- `mono` 单箱或坏率恒定时返回未定义值，不再返回历史占位 `1.0`；状态和原因见 `calculation_status`。无标签与全未表现标签分别标为 `not_computed` 和 `unobserved`。
- Agent 登记公共报告快照并保留持久标识；新增特征检索与上下文工具。HTML/Excel 集中附带业务元信息；原统计表的英文特征标识保持不变。

- 分箱评估的同一 batch_size 覆盖当前/基准转换、聚合、按日缺失与特征起点参考；跨批次只保留小统计表。
- 画像趋势在 overview_batch_size 内联合聚合多个指标，复用口径一致的 overview 统计。
- 两类报告新增 describe/get_table/to_ai_context/get_feature，展示支持列选择、来源筛选和 Top-K。
- Agent 新增 register_report 与已有报告目录，无原始数据的外部报告使用明确来源和独立快照。
- WOE 并列箱按 bin_index 确定累计指标顺序；补齐特征起点参考与金额统计同时使用时的字段。

- 新增 Experimental `mars.agent`，最低 Python 3.10，核心 MARS 的 Python 版本范围不变。
- 新增登记数据、画像、风险评估、监控、报告目录/说明和分页报告查询工具，计算复用现有公开 API。
- 新增跨轮会话、报告参数及证据引用、调用与上下文预算、结构化错误反馈。
- OpenAI 兼容 provider 使用可选 `[agent]` extra，普通导入不加载 SDK，不自动读取数据文件或执行代码。
- 定向回归位于 `tests/agent`，包括真实 MARS 计算对比与模拟模型协议测试，不调用线上模型。

## 0.0.28

本版本将 `deimos-rule` 来源快照 `e6714c5e795054e44f0c58ad7097668b4117b4a2` 完整重设计为
Experimental `mars.rule` 模块，并随同一个 `mars-risk` wheel 发布。没有 `deimos` namespace、
`Dm*` 兼容别名或旧 RuleSet JSON 读取分支。

### 规则生成、验证与部署

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

### 兼容性与来源

- 公开入口仅为 `mars.rule`，不从根 `mars` 重复导出；模块状态为 Experimental。
- Python 支持范围仍为 3.8–3.12；核心规则测试进入全部版本矩阵，可选后端在 Python 3.10 验证。
- 来源和许可证记录见[规则模块来源](rule-origin.md)。旧 `deimos-rule` artifact 必须重新挖掘或
  重新导出，不能直接载入。

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

!!! note "发布状态"

    本页随 `0.0.28` 源码维护。版本发布前，`main` 站点内容仅作为预览；PyPI 发布和 release tag
    通过版本一致性检查后，安装命令才表示正式可用。
