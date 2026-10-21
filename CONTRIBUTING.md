# 贡献指南

## 文档职责

文档面向第一次使用 MARS、具备 Python 和基础信贷风控知识的用户。每篇任务指南必须说明：

1. 适用场景和前置条件。
2. 如何准备输入，以及完整、可运行的调用。
3. 返回对象、结构化表或文件。
4. 影响调用方式的限制，以及常见错误。
5. 对应 API Reference 与下一步。

文档只描述已实现且可验证的能力。示例应明确写出前置步骤，并按用途命名基准数据、当前数据和开发数据。
不要把某次讨论中的月份或临时变量写成产品概念。

公共能力变更必须同步更新：

- 对应任务指南和公开 API 的 NumPy 风格 docstring。
- `docs/reference/` 中的公开 API 索引与引用。
- `docs/snippets/` 中与文档共享源码的可执行示例。
- Stable 或 Experimental 标记。
- 用户可见变化对应的 Release Notes。

性能结论必须附带可复现脚本、版本、数据规模、参数、硬件、运行日期和测量限制。缺少环境信息的
历史数字不能作为当前版本结论。

## 变更范围与适配

按变更的实际影响更新调用方、测试、文档和示例，无需同时修改无关文档。
监控、建模（含 Pipeline）、评分卡暂停功能迭代，仍处理必要修复并直接适配核心变化。
上游不为三个模块保留兼容层；适配后必须保持当前仓库可运行。
核心公共 API 和已保存报告仍须遵守兼容性要求，变更影响及迁移方式见
[稳定性规则](https://leeesq.github.io/mars-risk/project/stability/)。
纯文档变更无需重训练或大规模 benchmark；打包与发布检查在对应任务执行。

## 验证

按变更范围运行相关检查。下面列出文档、源码和打包的检查命令，打包检查在打包或发布任务中执行：

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

生成和导出夹具的两条命令必须分别运行。导出进程只加载 `.marsreport`，不读取原始宽表或重新拟合。
Score Cross 脚本通过 `file://` 操作真实页面，并将页面数值与公共 Python API 的结果对照。
文档浏览器脚本会自行启动、关闭本地 HTTP 预览。脚本保存截图和 JSON 日志，失败时返回非零退出码。
缺少浏览器或依赖时，应记录为未完成浏览器验收；Node 或静态测试不能代替浏览器交互。
2026-10-02 的实际结果与限制见[浏览器验收记录](docs/performance/correlation-score-cross.md#2026-10-02-真实浏览器验收)。

发布产物必须只构建一次。普通 CI 和 Release 都会把同一份 wheel 分别安装到全新 Python 3.8
与 3.12 环境，从仓库外运行 `scripts/smoke_installed_package.py`；PyPI job 只能上传两端均已
验证的 artifact，不得重新构建。Mypy 固定为 1.13.0，并统一按 Python 3.8 语法目标检查；
`mars.*` 业务模块不得通过 override 或 `type: ignore` 绕过类型错误。
