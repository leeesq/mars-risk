# M12 原生宽表修复与容量收口规划

[English](PLAN_ENG.md) · [执行进度](PROGRESS.md) · [Codex 首轮任务](CODEX_START.md) · [已有上游验收](../m12-upstream.md)

## 1. 目标、范围和当前状态

本轮目标仓库是 **leeesq/mars-risk**。规划基于已核对的 `main` 提交 `05b212e9b720e9ec6e9af0bec050e2be2685dc36`。本次规划提交只增加文档，没有实施新修复，也没有运行真实数据或测试。

目标不是再次实现已经合入的修复，而是回答：**固定 Python 3.8 环境下，当前 MARS 能否不依赖调用方的外层拆分，在本机完成原 M12 规模的原生画像和统计筛选，并保持数值、人口和报告合同？** 如果仍有真实缺口，复现后在正确的上游位置做最小修复。

成功必须分层表达：源码修复存在、定向回归通过、某个规模的原生计算完成、报告冷恢复通过、下游集成完成，是不同结论。MARS 通过不等于整个 Agent 已经高效，也不意味着模型 KS 会提高。

本轮不修改建模 Agent 的 Runtime、Registry、Snapshot、审批、预算、冻结或恢复规则；不训练 XGBoost，不接 LLM，不扩展 `mars.agent`，不新增调参器或选择引擎。Monitoring、Modeling/Pipeline、Scoring 的暂停迭代边界不变。根 README、版本号、依赖约束和发布流程默认不改。

## 2. 已修复事实与尚未验证事项

以下“已修复”指当前源码及仓库回归已有对应实现，不是本轮重新运行的结论。依据为 [已有上游验收](../m12-upstream.md) 和所列源码；历史验收记录不得覆盖或改写。

| ID | 当前事实 | 本轮动作 |
| --- | --- | --- |
| U01 | profiler 的 null/常量采用标量聚合，PSI 已与普通占位 pivot 分离 | 复跑旧 Polars、多 chunk、空人口边界，不重新设计 PSI |
| U02 | Binner 在样本表达式和类别 Join 前选择 lazy；fit/benchmark schema 不再依赖全宽 eager 转换 | 验证执行边界及样本顺序，不仅比较返回类型 |
| U03 | selector 的 batch_size 已贯通 Quality、Rough Performance、稳定性和 WOE 转换 | 观察真实批宽和对象生命周期；参数不等于总内存上限 |
| U04 | 完整 WOE Pearson 使用 Float64 行块中心矩；保留全局 P² 矩阵、完整关系和原决策顺序 | 核查数值、阈值附近复核、跨批关系与实际容量 |
| U05 | streaming 类别 Join 已保序，表现标签在转换前随行绑定；空 PSI 状态已补齐 | 回归真实 streaming、多线程、碰撞列名及保存恢复 |
| U06 | benchmark 已有 frozen_profile/risk/selection、完整语义 sidecar 和可比性检查；线程池枚举换序误拒已修复 | 复用现有 harness，不另造容量框架 |

相关实现：`src/mars/analysis/_profiling/`、`analysis/evaluator.py`、`analysis/_evaluation/aggregation.py`、`feature/binning/base.py`、`feature/selection/stats.py`、`compute/_correlation.py`、`benchmarks/benchmark_core_capacity.py`、`benchmark_core_workloads.py`、`_capacity_evidence.py`。

已有文档记录 Windows 上一次 200,000×1,000 的高存活率合成 selector 成功；它没有验证用户 Mac/16 GiB 上的冻结 M12 输入。当前公开文档明确记载真实 257,997 行风险画像及 297,440 行 selector 尚未在该上游验收中执行。这是本轮最重要的证据缺口，不应直接写成“源码仍有已知相同 bug”。

## 3. 不可改变的边界

- 原始文件只读；输入表、特征顺序、dtype、null/sentinel、监督人口、Train benchmark、缺失/特殊值和生效业务参数均有明确身份。不得为了跑通减少样本、删除来源、关闭 PSI/corr、调整阈值或把 WOE 相关性改为 raw 相关性。
- 只能使用用户已经提供并授权的 M12 工作区；从明确 manifest 跟随引用，不扫描整个主目录，不重新拉数据库数据。真实输入、完整业务报告、原始特征名、真实路径、manifest 内容与数据指纹仅留本地忽略目录，不上传这个公开仓库或 CI。
- selector 的非 Train 标签在输入中物理置 null；无监督画像不携带 target；development risk 只包含 Train/Val/Test，分箱和 benchmark 口径来自冻结配置。不能用 `supervision_group` 将合法 development risk 的 Val/Test 标签意外清空。
- 生产计算可以在库内部按列批/行块执行，但验收对象是一次完整公共调用。禁止调用方把特征拆三份、分别筛相关性后取并集，或用外层画像拼接来冒充当前原生能力。
- 历史失败与中止保留。旧已冻结 Run 不换依赖续跑，不改旧 identity、Receipt 或模型产物；必要的上游计算验证使用独立本地目录。
- 不设置新的 available-only 硬门，不将低 available、压缩或 swap 单独判为不能运行。真实任务串行、资源只观测；现有 timeout/退出保护必须显式记录，中止、异常、内核终止分别报告，不自动无界重试，不终止用户无关进程。
- 修复前后业务离散结果严格一致；浮点差异按已有语义比较规则检查。不能用扩大容差、修改预期名单或忽略排序来消除真实变化。

## 4. 分阶段执行

### N0 — 固定基线与可复现实验入口

读取实际 `AGENTS.md`、[工程 Skill](../../../.codex/skills/mars-risk-engineering/SKILL.md)、性能与质量参考、pyproject、当前 CI 和本目录文件。继续 `fix/m12-native-capacity-closeout` 分支；如有用户修改，不 stash/reset/覆盖。

本机真实入口使用已有 `modeling_xgb082` 环境：Python 3.8.20、XGBoost 0.82 保持不变；核验 Polars、NumPy、Pandas、PyArrow 实际版本，不升级。用显式 `PYTHONPATH=<当前MARS仓库>/src` 的子进程验证新代码，记录 `mars.__file__`，不要为验收改动其他已冻结 Run 使用的 editable 指针。现代版本测试只使用已经可用的隔离环境或既有 CI。

复用现有 benchmark，先执行 `--help` 核实参数。记录源码 commit/content identity、完整命令、线程环境、机器/系统、measurement 方法和退出码。先做一条小型 fresh-worker smoke，验证 JSON、sidecar、保存及下一进程消费，不先跑全量或重型矩阵。

**完成条件 N0：** 当前源码确实被导入；工作区和输入保护建立；小型公开调用与证据消费成功。文档中的历史通过数不能算本轮结果。

### N1 — 固定真实分析输入，不重新建模

先在已授权工作区查找冻结的 normalized 分析输入及对应 manifest。不能将最终少量入模列的交付数据当作原 1,111 候选，也不能从表的当前顺序猜测候选名单。

若已有正确的 frozen Parquet/IPC，直接复用并核验。若只有 raw Snapshot，只允许增加薄的、显式的一次性分析导出：依据历史冻结的 missing/label/split 合同准备新分析表，保留原 dtype/特征顺序、记录每个转换及 null 计数，并验证 raw 未变。它不是重新预筛或重新获取数据；不能在 benchmark 的 reader 中隐藏预处理。不能证明与历史分析人口等价时，该项标记 `input_contract_blocked`，继续不依赖它的合成验证。

| case | 当前人口 | benchmark | target 可见范围 | 特征 |
| --- | ---: | ---: | --- | ---: |
| 分区 profile_stats | 297,440 | 180,893 Train | 输入不带 target | 原 1,111 |
| 日 profile_stats | 297,440 | 180,893 Train | 输入不带 target | 同一 1,111 |
| development profile_risk / evaluator | 257,997 | 180,893 Train | 仅 development 合法标签；规则拟合仍按冻结 Train 配置 | 同一 1,111 |
| MarsStatsSelector | 297,440 | 180,893 Train | 非 Train 物理 null；监督只使用 Train | 同一 1,111 |

这些规模来自已有公开上游验收说明；本轮必须核验实数，不能为匹配文档删除记录。每个 case 独立配置，并完整复制其历史实际分箱、分组、日期、PSI、RC、名单、missing/special 等参数。不能让 frozen case 继承合成 fixture 的默认值；先把 merge 后的 effective parameters 输出到本地再计算。

**完成条件 N1：** 四份本地配置与完整有序候选都可核验；值、schema、人口、监督范围和 benchmark 一致；没有真实数据进入 Git。数据不可用时记 `pending_local_input`，不是库功能失败或通过。

### N2 — 原生连续计算与报告验收

按顺序串行执行：分区画像、日画像、development risk、完整 selector。每项先做一次实际单 worker 验证；正式 harness 的预热和 repeats 是额外调用，必须明确计数，不能把 `repeats=1` 误写为总共一次。没有必要重跑已知会 OOM 的旧版真实全宽路径。

每个 worker 连续覆盖输入读取、一次公共计算、必要报告构建、保存。不得在 Quality/Rough/Stability/Corr 之间退出重启来隐藏对象共存峰值。单独阶段分析可以用于定位，但不能代替完整调用通过。

现有 `frozen_profile`、`frozen_risk`、`frozen_selection` 优先复用。`frozen_risk` 当前走 `MarsBinEvaluator.evaluate()`；另做小型 `profile_risk()` wrapper 对照，真实下游依赖的参数转发也要覆盖。计时观察 wrapper 可以保留；不能替换生产算法或返回值。不要调用未支持的 API。

每项必须记录：输入/基准人数、有效监督数、dtype/chunks、特征顺序、实际阶段候选进出、实际列批和行块、阶段 wall、public_compute wall、整个 worker wall、主/树采样 RSS、lifetime peak、Darwin 可取得的 footprint/compressed/swap、退出状态。缺测明确说明；树 RSS 可重复计共享页，lifetime peak 不是阶段增量，不能把 footprint 当物理 RSS。

逐项成功后立即保留可验证报告及完成记录；后项失败只修复和重跑失败项，前项符合相同源码/参数/输入身份时可以复用。源码影响前项时才重验；不能将跨源码资产无条件复用。

新进程加载保存的报告，验证 `describe/get_table/query_page/to_ai_context` 及必要 `get_feature` 查询；不重新计算，不加载原宽表。原生表目录、schema、null、顺序、计算状态与原对象一致。

**完成条件 N2：** 四项原生调用分别真实成功，完整 selector 不是外层拆分结果；全部报告可独立消费。只通过两项就分别标两项，不用统一 Passed 遮盖遗漏。

### N3 — 仅对真实失败做定点修复

N2 已通过的路径默认不改实现。失败先区分输入合同、环境/导入、测量工具、存储、数值错误和实际容量问题。保留失败 PID、原始输出、退出码、能取得的内核记录和最内层阶段；没有分配证据时，不把某一条 `unpivot` 写成唯一根因。

优先审查真实热点：当前/benchmark 的批次投影，lazy/schema 的执行边界，long 表生命周期，监督过滤是否先缩列，WOE 当前块及 NumPy 临时数组，完整关系报告构造/保存。全局 O(P²) 输出和 O(NP²) 计算不能宣传为已消失。

先写小型复现，再做最小源代码修复，保持已有公共 API。优先复用 compute、Binner 和 reporting 实现；不添加通用资源调度器、第二套低内存 selector、任意永久缓存、循环 GC 或业务专属判断。不要通过清除公共 update_bins/首次 WOE 必需训练状态来伪造降峰。

每次修复：记录源码与参数变化、定向测试，再独立重跑原失败真实 case。不得同一进程边改源码边继续。只有有实质变化才重试；同配置反复跑到一次成功不构成可靠修复。

**完成条件 N3：** 每个已修复缺陷有 before/after 复现与相同真实 case 成功证据；或明确记录仍存在的边界。若无新缺陷，用 `not_required`，不要为了“有代码提交”制造重构。

### N4 — 结果等价与下游交接

使用已修正的同一 harness 比较两侧。完整输入内容、顺序、角色、有效参数、人口、实际候选路径、资源和报告证据符合比较合同后才输出性能比例。仅环境/线程改变的结果按既有 resource_strategy_comparison 解释；缺 sidecar 或历史证据不足时为 incomparable。

可以用同口径合成 dense oracle 验证矩阵/边/决策，并把旧的已保存 M12 分批报告作为 **策略参考**。旧分批路径与新原生路径不同，不能未经等价证明直接称普通性能 A/B；也不强制为了公平比例重跑危险旧路径。不可解释的选择变化是失败，不是可忽略的小浮点误差。

记录当前代码的兼容边界：旧公开报告按已支持格式加载，新报告保存恢复，Binner/WOE 规则及非计算查询不退化。核心 API 或报告结构若确需改变，单独说明迁移并覆盖消费者；默认不改变。

形成下游交接表：调用前整表 rechunk、外层每批 profile_risk、三份 selector 适配，分别是“可在下游独立对照后移除”还是“仍需保留”。本轮不直接删除另一个仓库的适配，不变更旧 Run 的 MARS pin；原生 MARS 成功后，仍需要一个新下游验收任务决定采用哪条路径。输入导出与 benchmark 进程不等于 Agent 端到端验收。

**完成条件 N4：** 有数值/报告合同核对、明确的可比性结论、按适配项逐一列出的迁移建议；不声称已修复 Agent 的 View/hash/Registry/停止策略。

### N5 — 规模边界、文档和交付

核心通过目标是上述 M12 1,111 列原生路径。用高存活率合成 fixture 补一个 2,000 列的分级规模观察：先 50,000 行；更大行数是否执行根据已记录资源与当前任务范围决定，未跑必须 Deferred。不得将初期淘汰到很少列的结果称为完整宽候选相关性容量；记录进入 corr 的实际列数，沿用至少 80% 的规模 guard。

不以一次 1,111 列或合成 2,000 列结果保证 5,887 列全量输入。能力表必须注明场景、行/列/基准规模、单/多线程、机器、调用方式和边界，不写普遍可用的内存上限或固定提速百分比。

按实际影响执行质量门。开发中定向测试；稳定后核心变更在 Python 3.8 跑适用基础套件，现代覆盖使用已有 CI；不每修一行都重跑全部容量任务，不降低断言或屏蔽 warnings。静态工具缺失不升级建模环境；报告未执行，或使用已有隔离质量环境。

更新本目录进度和独立脱敏验收页，在原 `m12-upstream.md` 追加索引而不替换历史。根 README 默认不改。必要时添加 MkDocs 导航，链接/严格构建按现有可用文档环境核验；不发布包或部署。代码与测试、验收文档分职责提交，普通 push 当前分支，不自动 merge。

**完成条件 N5：** 实际限制、通过/失败/延期、命令与证据映射完整；所有公开文件经过隐私检查；CN/ENG 保持同一任务、参数和验收 ID。

## 5. 最小验收矩阵

| ID | 必须证明什么 | 主要层级 |
| --- | --- | --- |
| T01 | 旧 Polars、多 chunk、PSI-only/mixed、非适用常量和空人口保持形状、dtype、状态，不要求调用方 rechunk | 定向回归 |
| T02 | lazy 建计划和 collect_schema 不执行样本 eager 转换；实际 collect 的值和映射正确 | 执行路径回归 |
| T03 | streaming Join 的行序、标签与跨列批 WOE 对齐；线程 1/4、重复行、未知类和临时列名碰撞 | 独立进程回归 |
| T04 | Quality/Rough/Stability/WOE 的真实批宽和监督人口符合配置；benchmark 不先全宽物化 | 路径回归 + 真实观测 |
| T05 | 全局 signed WOE Pearson、退化列、阈值附近严格分支、IV/ID 优先级、跨批边和最终顺序一致 | 数值 oracle |
| T06 | 分区/日画像、development risk、selector 四项在原完整人口/候选上单公共调用成功 | 真实规模 |
| T07 | 完整候选、表结构、null/非浮点值、数值和决策语义匹配，未知差异拒绝 | 结果对照 |
| T08 | 保存后新进程查询不依赖原始数据或重算，状态和元数据可消费 | 冷恢复 |
| T09 | 输入内容或顺序、工作量、资源、sidecar 变化被正确分类；线程池仅换序不误拒 | harness 回归 |
| T10 | 输入未改、业务数据未上传、当前实现与历史记录分开、下游边界明确 | 交付检查 |

优先扩展已有 `test_analysis_batching.py`、`test_binner_execution.py`、`test_binner_streaming.py`、`test_stats_blocked_correlation.py`、`test_profile_empty_psi.py`、`test_core_capacity_benchmark.py`。不要重复建一套几乎相同的测试框架。

## 6. 产物和判定

公开：本规划、进度、通用修复、合成测试、命令模板和脱敏结论。私有：输入/配置完整指纹、完整特征与报告、原始日志、真实 source/manifest 路径、测量 JSON 和 sidecar，放用户指定的忽略目录，独占命名且不覆盖旧证据。

每项最少记录 `case/status/source/environment/input_contract/effective_parameters/actual_workload/measurements/output_contract/command/exit/failure_or_limit`。这是一份证据字段清单，不要求新建 Pydantic schema 或另一套 Registry；复用 harness 已有字段，缺项才补。

状态使用 `not_started / in_progress / passed / failed / pending_local_input / blocked / deferred / not_required`，不把环境缺项当 passed。最终至少分开声明：

1. 已修复源码与定向回归。
2. 原生 M12 四项真实容量与报告冷消费。
3. 更宽候选的已测边界。
4. 下游 Agent 集成：本仓库不代验收。
5. 模型效果和研究质量：本轮未评估。

## 7. 首轮执行顺序

首轮按 N0 → N1 → N2 推进；遇到已复现的上游缺陷，进入 N3 定点闭环，再完成 N4/N5 的实际可完成部分。没有真实输入时完成合成路径和明确的本地配置说明，不伪造真实容量成功；有输入且无阻断时，不在每个阶段重复请求许可。

第一优先级是**取得原生真实调用的可信结论**，不是增加模块数、测试数量或再次训练模型。完整 Codex 入口见 [CODEX_START.md](CODEX_START.md)。
