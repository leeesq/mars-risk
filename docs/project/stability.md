---
description: MARS 0.0.x 的模块稳定性、兼容性承诺和升级建议。
---

# 稳定性与兼容性

MARS 仍处于 `0.0.x` 阶段。Stable 表示模块已有推荐入口，并明确约定了结构化返回结果。
这一标记不代表 `1.x` 级别的长期兼容承诺。

| 模块 | 状态 | 升级预期 |
| --- | --- | --- |
| Analysis | Stable | 优先保持入口、核心参数和 report 字段兼容 |
| Feature | Stable | 优先保持分箱器、选择器的调用和规则序列化兼容 |
| Monitoring | Experimental | report 字段、target 校验和报警结果仍可能调整 |
| Agent | Experimental | Python 3.10+；工具、会话和结果契约仍可能调整；模型依赖可选 |
| Reporting | Stable | 优先保持结构化字段和导出入口兼容 |
| Scoring | Experimental | 评分映射、刻度参数和 SQL 输出仍可能调整 |
| Modeling | Experimental | 参数、结果对象和 artifact 结构仍可能调整 |
| Pipeline | Experimental | step 的输入输出约定、结果字段和编排限制仍可能调整 |
| Rule | Experimental | DSL、筛选策略、结果对象和 artifact 仍可能调整 |

## 开发投入与接口成熟度 { #development-focus }

当前优先改进分析计算性能、内存效率、人工分析体验、结构化报告和外部 Agent 的使用体验。
监控、建模（包括 Modeling 和 Pipeline）、评分卡暂时停止功能迭代。
保留现有功能、入口和使用文档，只处理必要的正确性修复、运行修复和上游适配。
你可以借助编程型 AI，结合 MARS 核心分析、报告和自己的数据、模型工具构建业务流程。
Stable 和 Experimental 表示接口成熟度；暂停迭代说明开发投入，两者分别描述。

## 暂停模块的下游适配 { #暂停模块的下游适配 }

核心分析、计算或报告的方法、返回结构改变时，三个下游模块 Monitoring、Modeling／Pipeline 和
Scoring 直接修改调用代码，并更新对应的行为测试。
上游不为这三个模块保留旧接口、旧字段、兼容分支、转发包装、双轨实现或冗余计算。

适配后，三个模块必须能在当前仓库中运行，并通过受影响的行为测试；无需维持历史签名和旧产物格式。
这一兼容性例外仅适用于这三个下游模块。核心公共 API、报告字段或已保存分析文件发生变化时，
仍须明确记录影响和迁移方式。

## 升级规则

分客群分析是新增的显式能力，当前只覆盖风险画像、数据画像和 StatsSelector。
未提供 `segment_col` 时沿用原始表结构和默认结果。开启后读取事实表时必须保留 `scope_id`，
不能仅凭 feature 连接不同人口。报告文件继续使用格式 1；新增目录不要求迁移历史无客群文件。
原生 Excel 的单 scope 限制、独立规则产物和局部失败合同见[分客群分析](../user-guide/segmented-analysis.md)。

- 安装时固定精确版本，例如 `pip install mars-risk==0.0.28`。
  使用 main 源码时，按[安装指南](../getting-started/installation.md)安装，
  并记录实际核验的提交。仓库版本号不代表已经上传 PyPI。
- 升级前阅读[Release Notes](release-notes.md)，并在测试数据上验证依赖的字段和文件路径。
- Experimental 模块的调用方应为关键结果对象增加契约测试。
- `main` 文档可以作为预览部署；只有对应版本发布到 PyPI 后，安装命令才表示正式可用。
