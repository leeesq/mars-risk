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
| Reporting | Stable | 优先保持结构化字段和导出入口兼容 |
| Scoring | Experimental | 评分映射、刻度参数和 SQL 输出仍可能调整 |
| Modeling | Experimental | 参数、结果对象和 artifact 结构仍可能调整 |
| Pipeline | Experimental | step 契约、结果字段和编排限制仍可能调整 |

## 升级规则

- 生产流程固定精确版本，例如 `mars-risk==0.0.26`。
- 升级前阅读[Release Notes](release-notes.md)，并在测试数据上验证依赖的字段和文件路径。
- Experimental 模块的调用方应为关键结果对象增加契约测试。
- `main` 文档可以作为预览部署；只有对应版本发布到 PyPI 后，安装命令才表示正式可用。
