---
name: mars-risk-engineering
description: 用于 leeesq/mars-risk 本身的实现、审查、重构、测试、性能与内存优化、文档、CI、打包及发布任务，覆盖画像、分箱评估、筛选、相关性、模型分交叉、规则挖掘和公共报告／外部 Agent 契约。不是通用消费信贷建模 Skill，也不是普通用户报告接入指南。
---

# MARS Risk Engineering

开发、维护和审查 MARS 本身；普通分析使用方法见 docs/user-guide。
项目定位：**面向人和 AI Agent 的高性能风控分析工具箱**。
核心能力：**数据画像 · 分箱评估 · 特征筛选 · 相关性分析 · 模型分交叉 · 规则挖掘**。

## 开始工作

1. 读取实际 AGENTS.md（如有），运行 git status --short，保护已有修改。
2. 使用 rg 查找相关 public 入口、调用方、测试和文档，不凭旧提交设计。
3. 读取 pyproject.toml、相关 CI、源码与测试，再决定实现。
4. 列简短计划，先复现缺陷，再修改，最后按实际影响验证。
5. 发布、push、merge、部署只在用户授权范围内执行，不因本 Skill 自动授权。

## 按任务加载参考

只加载本次需要的参考；不要每次把所有文档全文读入上下文。

| 任务 | 参考 | 要解决的问题 |
| --- | --- | --- |
| 模块边界、上游改造、暂停模块适配、重构 | [architecture.md](references/architecture.md) | 谁调用谁，状态归属与改造边界 |
| 指标、分箱、缺失、筛选、相关性／交叉口径 | [api-and-metrics.md](references/api-and-metrics.md) | 单位、分母、参数及数值语义 |
| 新报告字段、查询、元数据、快照、外部 Agent | [reports-and-agents.md](references/reports-and-agents.md) | 公共结果契约与消费边界 |
| 性能、内存、规模预算、展示与保存 | [performance-and-memory.md](references/performance-and-memory.md) | 投影、物化、复用与规模控制 |
| 测试、文档、CI、打包、交付 | [quality-and-delivery.md](references/quality-and-delivery.md) | 适用检查与交付证据 |

报告字段变更通常加载报告、API 与质量参考；
相关性内存优化加载架构、性能、API 及质量参考。
纯首页文案调整通常只加载质量参考，涉及承诺时再核对报告／API。
运行版本、完整签名、状态及命令以 pyproject.toml、源码、CI 与稳定性文档为准。
不要将这些易漂移信息复制到多套速查表。

## 开发投入与兼容范围

- 优先改进核心分析性能、内存效率、人工体验、结构化报告与外部 Agent 使用。
- Monitoring、Modeling（包括 Pipeline）、Scoring 暂停功能迭代。
- 保留其现有功能、入口与文档，处理必要正确性、运行修复及上游适配。
- 用户可以借助编程型 AI、核心分析报告与自己的模型工具构建业务流程。
- 三个下游直接适配核心方法／结构变化，修改调用代码及受影响行为测试。
- 上游不为三者保留旧接口、旧字段、兼容分支、转发壳、双轨实现或冗余计算。
- 不要求保留三者历史签名／产物，但不能使当前仓库的三者无法运行。
- 此豁免不扩展到所有核心公共 API、报告字段或已保存分析文件。
- 核心公共契约变化仍须记录影响与迁移方式，按影响同步文档与示例。
- 不采用全局“默认 breaking changes”或“所有文档同波修改”。
- Stable／Experimental 说明成熟度；暂停说明投入，分别表达。
- 不新增候选选择引擎或通用 Agent 编排平台来绕过暂停边界。

## API 与状态

- 构造函数保存稳定策略、阈值及规格；运行方法接收数据与本次上下文。
- 底层 estimator 使用 X, y；高层工作流使用 df, target。
- 不在同一 public method 同时暴露 y 和 target。
- fit／transform／predict／evaluate 的行为保持清晰，不把本次数据隐式留在复用工具中。
- 明确拟合状态归属；Result 保存运行产物，Report 保存可消费的分析证据。
- 覆盖默认配置时区分 None 与空映射，并记录最终生效参数。
- 新入口明确 Public、Internal 或 Experimental，不默认一切新能力都 Stable。
- 公共返回结构先核对真实消费者，再做合理改造。
- 回归验证行为与业务不变量，不锁定无意义实现细节。

## 共享实现

- DataFrame 转换、投影与物化复用 mars.compute。
- 核心计算优先 Polars／pl.Expr 与共享统计算子。
- 缺失处理复用共享 Null／NaN／missing_values 语义。
- profiler／evaluator／monitoring／scanner／detector 不各自维护缺失算法。
- 高层 profile_stats 不作为底层缺失计算依赖。
- 数值常量复用 mars.core.constants，不散落稳定性 epsilon。
- 可选依赖复用 mars.utils.imports，重 SDK 延迟加载。
- 分箱共享能力放 MarsBinnerBase，避免重复 WOE、transform、规则导出与序列化。
- 建模评估复用已有评估表与指标，不重写 PSI／ROC／KS／Lift。
- 报告契约／查询／快照独立于具体 Modeling／Monitoring；底层 compute 不依赖报告对象。
- 结构化结果是主要接口，HTML／Excel／plot 是消费与导出能力。
- 模型产物、报告快照、分箱规则、规则集各由所属契约负责保存。
- 公共报告不能依赖 Modeling 才能保存，不统一塞进 mars.modeling.artifacts。
- mars.agent 是公共契约消费者，不独占报告查询／保存。
- 不把重导出或核心计算的转发层放入 utils。

## 修改纪律

- 新建／修改函数具完整参数和返回类型注解，复杂中间值使用明确类型。
- public API 使用与签名、返回及异常一致的 NumPy docstring。
- 自然语言注释与 docstring 使用中文；标识符和 NumPy section 保持英文。
- 复杂私有 helper 写中文短说明，解释输入、约束与失败条件。
- 注释解释意图、业务口径和取舍，不逐句翻译实现。
- 不用 type: ignore 掩盖可修复错误，不删关键验证来通过检查。
- 不引入无必要依赖，不增加虚构 API、性能数字或可用性承诺。
- 手工编辑优先 apply_patch；修改范围贴合任务。
- 不提交临时数据、模型、benchmark 产物或站点构建目录，除非用户明确要求。
- 修改源码时同步受影响类型、docstring、调用方、测试、文档及示例。
- 查询先筛选／排序／投影／分页，再转展示或 JSON。
- 预算在解析默认参数之后检查，显式与默认参数同等受限。
- 优化复用已有结果，不能改变指标公式、缺失或样本口径。

## 验证与完成

1. 先跑小型定向回归及受影响调用方测试。
2. 根据质量参考与 CI 选择适用静态门、文档示例、严格构建及完整测试。
3. 性能任务先证明结果等价，再按需要运行独立 benchmark。
4. 文档任务执行共享 snippets，检查链接／锚点；可用环境下检查响应式视觉。
5. Skill 修改验证 frontmatter、相对引用、UI 配置与实际场景决策。
6. 区分代码失败、已有失败与环境限制；不把未运行写成通过。
7. 结束前运行 git diff --check、查看状态，排除意外产物。
8. 交付说明变化、理由、验证、实际限制及必要迁移；无授权不发布。

环境不限 Windows 或 conda；命令按项目配置与当前可用解释器选择。
依赖或 DLL 错误先诊断，不把历史沙箱问题当作自动提权规则。
