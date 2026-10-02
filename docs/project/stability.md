---
description: MARS 0.0.x 的模块稳定性、兼容性承诺和升级建议。
---

# 稳定性与兼容性

MARS 仍处于 `0.0.x` 阶段。稳定标记表示该模块已经形成推荐入口和结构化返回契约，不代表遵循
`1.x` 级别的长期兼容承诺。

| 模块 | 状态 | 升级预期 |
| --- | --- | --- |
| Analysis | Stable | 优先保持入口、核心参数和 report 字段兼容 |
| Feature | Stable | 优先保持 binner/selector 调用和规则序列化兼容 |
| Monitoring | Experimental | report 字段、target 校验和报警结果仍可能调整 |
| Agent | Experimental | Python 3.10+；工具、会话和结果契约仍可能调整；模型依赖可选 |
| Reporting | Stable | 优先保持结构化字段和导出入口兼容 |
| Scoring | Experimental | 评分映射、刻度参数和 SQL 输出仍可能调整 |
| Modeling | Experimental | 参数、结果对象和 artifact 结构仍可能调整 |
| Pipeline | Experimental | step 契约、结果字段和编排限制仍可能调整 |
| Rule | Experimental | DSL、筛选策略、结果对象和 artifact 仍可能调整 |

## 开发投入与接口成熟度 { #development-focus }

当前优先改进分析计算性能、内存效率、人工分析体验、结构化报告及外部 Agent 使用能力。
监控、建模（包括 Modeling 和 Pipeline）、评分卡暂时停止功能迭代。
保留现有功能、入口和使用文档，只处理必要的正确性修复、运行修复和上游适配。
用户可以借助编程型 AI、MARS 核心分析与报告及自己的数据／模型工具构建业务流程。
Stable／Experimental 表示接口成熟度；暂停说明开发投入，两者独立。

## 暂停模块的下游适配 { #暂停模块的下游适配 }

核心分析、计算或报告方法／返回结构改变时，直接修改 Monitoring、Modeling／Pipeline、
Scoring 的调用代码及对应行为测试。上游不为三个下游模块保留旧接口、旧字段、兼容分支、
转发壳、双轨实现或冗余计算，不因历史调用方式阻碍核心合理改造。

不保留兼容层不等于允许当前仓库的三个模块无法运行；适配后须通过受影响行为测试，
但不要求维持历史签名和旧产物格式。此豁免仅针对这三个下游模块；核心公共 API、
报告字段及已保存分析文件的变化仍需明确记录影响和迁移方式。

## 升级规则

- 已发布包固定精确版本；当前 0.0.28 源码按[安装指南](../getting-started/installation.md)
  安装并固定核验过的提交，不使用尚未发布版本的 PyPI 固定安装命令。
- 升级前阅读[Release Notes](release-notes.md)，并在测试数据上验证依赖的字段和文件路径。
- Experimental 模块的调用方应为关键结果对象增加契约测试。
- `main` 文档可以作为预览部署；只有对应版本发布到 PyPI 后，安装命令才表示正式可用。
