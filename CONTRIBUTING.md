# 贡献指南

## 文档职责

文档面向第一次接触 MARS、具备 Python 和基础信贷风控知识的外部用户。每个任务 Guide 必须说明：

1. 适用场景和前置条件。
2. 完整、可运行的输入和调用。
3. 返回对象、结构化表或文件。
4. 会改变调用方式的边界和常见失败。
5. 对应 API Reference 与下一步。

文档只描述已实现且可验证的能力。不要将某次讨论中的月份、变量或隐含前置步骤写成产品概念；
基准数据、当前数据和开发数据使用语义化名称。

公共能力变更必须同步更新：

- 对应 Guide 和公开 NumPy 风格 docstring。
- `docs/reference/` 中的公开 API 可发现性。
- `docs/snippets/` 中与文档共享源码的可执行示例。
- Stable 或 Experimental 标记。
- 用户可见变化对应的 Release Notes。

性能结论必须附带可复现脚本、版本、数据规模、参数、硬件、运行日期和测量限制。缺少环境信息的
历史数字不能作为当前版本结论。

## 验证

提交前运行：

```bash
python -m ruff check src tests scripts docs/snippets
python -m mypy src/mars
pydoclint src/mars
python scripts/check_private_docstrings.py src/mars
python -m pytest -q tests/test_documentation.py
python -m mkdocs build --strict
```

Modeling、Pipeline 或 Notebook 示例还需要安装 `ml,tuning` extra 并执行文档集成测试。
