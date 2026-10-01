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

## 变更范围与适配

按实际影响同步调用方、测试、文档与示例，不要求无关文档同波修改。
监控、建模（含 Pipeline）、评分卡暂停功能迭代，仍处理必要修复并直接适配核心变化。
上游不为三个模块保留兼容层；适配后必须保持当前仓库可运行。
此原则不扩大为任意破坏核心公共 API 或已保存报告；影响及迁移见
[稳定性规则](https://leeesq.github.io/mars-risk/project/stability/)。
纯文档变更无需重训练或大规模 benchmark；打包与发布检查在对应任务执行。

## 验证

提交前运行：

```bash
python -m ruff check src tests scripts docs/snippets
python -m mypy src/mars
pydoclint src/mars
python scripts/check_private_docstrings.py src/mars
python -m pytest -q tests/test_documentation.py
python -m mkdocs build --strict
python -m build
python scripts/verify_distribution.py --dist-dir dist
python -m twine check dist/*
```

Modeling、Pipeline 或 Notebook 示例还需要安装 `ml,tuning` extra 并执行文档集成测试。

### 离线报告的真实浏览器验收

使用 Python >=3.10 的独立开发环境安装 `.[dev,browser]`，复用本机 Chrome（`--channel chrome`），
或运行 `python -m playwright install chromium` 后使用 `--channel chromium`。
浏览器不是 MARS 核心安装依赖。下面的 `<tmp>` 是独立可写目录，产物不提交到仓库：

```bash
python tests/browser/fixtures.py --phase generate --output <tmp>
python tests/browser/fixtures.py --phase export --output <tmp>
python tests/browser/score_cross.py --manifest <tmp>/manifest.json --output <tmp>/browser --channel chrome
python tests/browser/representatives.py --output <tmp> --channel chrome
python -m mkdocs build --strict --site-dir <tmp>/docs/site
python tests/browser/docs.py --site-dir <tmp>/docs/site --output <tmp>/docs --channel chrome
```

两次夹具命令必须分别运行；第二个进程只加载 `.marsreport`，不读取原始宽表或重新拟合。
Score Cross 通过 `file://` 操作真实页面，数字与公共 Python API 对照；文档入口自行启动并关闭
本地 HTTP 预览。脚本保存截图和 JSON 日志，失败以非零退出码返回。
浏览器/依赖缺失是未完成验收，不能由 Node 或静态测试代替。
本轮实际结果与限制见[浏览器验收记录](docs/performance/correlation-score-cross.md#2026-10-02-真实浏览器验收)。

发布产物必须只构建一次。普通 CI 和 Release 都会把同一份 wheel 分别安装到全新 Python 3.8
与 3.12 环境，从仓库外运行 `scripts/smoke_installed_package.py`；PyPI job 只能上传两端均已
验证的 artifact，不得重新构建。Mypy 固定为 1.13.0，并统一按 Python 3.8 语法目标检查；
`mars.*` 业务模块不得通过 override 或 `type: ignore` 绕过类型错误。
