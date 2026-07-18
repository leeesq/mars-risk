---
description: 为 MARS 补充公共能力和文档时需要满足的内容与验证要求。
---

# 贡献文档

完整开发规范见仓库根目录的
[CONTRIBUTING.md](https://github.com/leeesq/mars-risk/blob/main/CONTRIBUTING.md)。

公共能力变更至少需要同步：

- 对应任务 Guide，说明输入、完整调用、输出和影响调用方式的限制。
- API Reference 可发现性和准确的 NumPy 风格公开 docstring。
- 与文档共享源码的可运行示例。
- Stable 或 Experimental 状态，以及必要的 Release Notes。
- `mkdocs build --strict`、文档示例、内部链接和公开 API 覆盖检查。

文档只描述仓库中已经存在并可验证的能力。性能结论必须附带可复现脚本、数据规模、参数、版本、
硬件环境和测量限制。
