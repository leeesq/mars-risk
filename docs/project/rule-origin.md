---
description: mars.rule 的 deimos-rule 来源提交与许可证记录。
---

# 规则模块来源

`mars.rule` 的设计与实现以 `deimos-rule` 提交
`e6714c5e795054e44f0c58ad7097668b4117b4a2` 为来源快照。迁移时按 MARS 的公共接口约定重新设计代码，
使用受限 AST DSL、严格校验的 artifact 和统一的 Polars 计算流程。
仓库没有导入来源项目的独立 Git 历史，也不提供旧命名空间或 artifact 的兼容层。

来源项目使用 MIT License，版权声明为：

> Copyright (c) 2026 deimos-rule contributors

MIT 许可全文与仓库根目录的 `LICENSE` 相同。本记录随源码和发布文档保留，用于满足来源
追溯与许可证通知要求。
