---
description: 为 MARS 补充公共能力和文档时需要满足的内容与验证要求。
---

# 贡献文档

请按变更的实际影响选择检查项。纯文档修改无需重新训练模型或运行大规模 benchmark。
暂停模块的维护范围和适配规则见[稳定性](stability.md#暂停模块的下游适配)。

完整开发规范见仓库根目录的
[CONTRIBUTING.md](https://github.com/leeesq/mars-risk/blob/main/CONTRIBUTING.md)。

公共能力变更至少需要同步：

- 对应任务指南，说明输入、完整调用、输出和影响调用方式的限制。
- API Reference 中的索引与引用，以及准确的公开 API NumPy 风格 docstring。
- 与文档共享源码的可运行示例。
- Stable 或 Experimental 状态，以及必要的 Release Notes。
- 严格文档构建（`mkdocs build --strict`）、文档示例、内部链接和公开 API 覆盖检查。
- 打包或发布时，静态校验 wheel/sdist，并将同一个 wheel 分别安装到全新 Python 3.8、3.12 环境中运行 smoke。
- 用 Mypy 1.13.0 按 Python 3.8 目标检查全部业务源码，结果须为零错误，且不得用业务模块 override 或 `type: ignore` 绕过检查。

文档只描述仓库中已经存在并可验证的能力。性能结论必须附带可复现脚本、数据规模、参数、版本、
硬件环境和测量限制。
