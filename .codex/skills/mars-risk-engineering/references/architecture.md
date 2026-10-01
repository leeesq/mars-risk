# 架构与依赖方向

涉及模块边界、核心接口改造、暂停模块适配或大规模重构时读取。
权威入口见源码一级领域包；模块状态见 docs/project/stability.md。

## 职责与调用

以下箭头表示“调用方 → 被调用方”，不表示计算必须经由报告。

- core：基础协议、异常与数值常量，不调用高层领域模块。
- utils：轻量日期、格式化、日志与可选导入，不承载核心计算或重展示。
- compute → core／utils：共享表达式、缺失、统计、分箱算子与物化。
- feature → compute／core／utils：分箱与筛选，不反向调用 Modeling。
- analysis → feature／compute：画像、分箱评估、固定分段交叉与分析工作流。
- 领域 Result／Report → reporting 的共享契约、查询、元数据和快照基础能力。
- reporting 的共享协议／查询／快照 → 基础类型与表工具，不依赖具体 Modeling／Monitoring。
- 展示／导出 adapter 消费领域结果；不把底层计算改为接收高层报告。
- modeling → 核心分析／feature／compute：保留切分、后端、tune、replay 和模型 artifact。
- pipeline → feature／modeling：保留串联，不另定义指标和报告规则。
- monitoring → 核心分析／feature／compute：保留监控指标与报警摘要，不是调度平台。
- scoring → 既有规则／模型参数及展示工具：保留评分映射与部署转换。
- agent → 公开 analysis／monitoring／reporting：受预算约束的消费者，无指标双轨实现。
- rule：规则发现、验证与部署契约，按实际复用共享计算，不新增通用建模平台。

共享报告可以被领域报告调用，也可以被外部消费者直接查询；这不要求底层 compute
依赖高层 Report，不要求所有领域通过具体建模报告或监控报告保存。

## 暂停模块直接适配上游

Monitoring、Modeling／Pipeline、Scoring 暂停功能迭代，保留功能和必要修复。
上游方法、返回结构或计算接口改变时，直接修改三个下游的调用代码和对应行为测试。
不为三者保留旧字段、转发壳、兼容分支、双轨实现或冗余计算。
当前仓库适配后仍须运行，不要求维持历史调用签名／旧产物格式。
豁免仅针对三者；核心公共 API、报告字段和持久文件仍需评估影响并说明迁移。
Stable／Experimental 与开发投入分别表达。

## 对象与产物归属

- 构造函数保存策略，运行方法接收数据及本次上下文，记录最终配置。
- 拟合状态属于分析器／estimator；Result 属于运行，Report 属于证据。
- 公共报告按既有 .marsreport 契约保存，不依赖模型模块。
- 模型 artifact、分箱规则、RuleSet 各由所属契约管理；按需复用已有通用 I/O。
- 不把所有产物读写集中到 mars.modeling.artifacts。
- 特征筛选与相关性报告复用同一计算表示、缺失策略、方法、阈值和决策事件。
- 交叉报告与保存后策略回放复用固定分段和已聚合证据。

## 重构边界

先拆纯 helper、查询、adapter、表构造、展示和 I/O，再考虑工作流。
不为文件变短制造循环依赖、过深目录或难追踪碎片。
Public 路径以真实公开入口为准；Internal helper 使用明确私有命名。
合理核心改造按实际影响同步源码、消费者、测试、文档和迁移。
Pandas 留在明确后端／渲染边界；核心新增链路优先 Polars，转换先投影。
架构原则在本 references 维护，不引用不存在的历史规划文件。
