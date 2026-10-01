# 质量与交付

涉及测试、CI、文档、打包、发布或 Skill 修改时读取。
命令和解释器范围以 pyproject.toml、.github/workflows 和 constraints 为准，
不固定 Windows／conda，不重复维护完整版本和签名目录。

## 按变更选择验证

| 变更 | 必要验证 |
| --- | --- |
| 指标／业务计算 | 定向数值、边界与业务不变量；受影响调用方测试 |
| 报告／查询／快照 | 类型、表结构、元数据、状态、证据、分页、最终字符预算、保存恢复 |
| 性能／内存 | 结果等价、小型规模回归；必要时独立 benchmark |
| README／指南／首页 | 共享示例执行、链接和锚点、严格构建；可用环境下响应式视觉检查 |
| Skill | frontmatter、相对引用、UI 配置、规则一致性与实际场景走查 |
| 打包／发布 | 实际 CI 的单次构建、静态校验与同一 wheel 安装 smoke；授权范围内交付 |

先复现 bug，再修复并跑定向测试；随后选择适用仓库质量门。
只改文案不重训练、不跑大 benchmark；改变核心返回结构应测试真实下游。
不修改测试迎合错误结果，不删除重要断言；允许修正与新授权契约矛盾的旧测试。

## 通用质量门

按影响范围与 CI 选择，不表示每次改一行文字都执行所有命令：

```bash
python -m ruff check src tests benchmarks scripts docs/snippets
python -m mypy src/mars
pydoclint src/mars
python scripts/check_private_docstrings.py src/mars
python -m pytest -q
python -m mkdocs build --strict
git diff --check
```

public NumPy docstring 的参数顺序、类型、返回和异常与签名同步。
复用现有可运行 snippets，输出写临时目录，避免覆盖用户文件。
不把未运行说成通过；区分代码失败、已有失败、依赖缺失和环境限制。
DLL／依赖问题先定位，不自动提权、盲目重装或削弱验证。

## 报告持久化与导出

新增／修改持久契约时验证：保存 → 新进程加载 → 查询 → 展示／导出 → 证据引用。
无需原始宽表、分析器或原会话。检查 schema、身份、参数、元数据、状态及省略记录。
Excel／HTML 检查实际表头和数值，不只断言文件存在。
快照按报告类型验证专用能力与明确不支持的能力。

## 文档站与视觉

保留既有 MkDocs Material、搜索、导航、主题切换、代码复制与页签。
尽量保留 URL／锚点，必要时旧入口链接到新指南。
首页专用 class 限定样式，不污染指南表格、Notebook 或 API。
可用浏览器检查 1440／768／390px、深浅色、焦点、整体溢出和一个指南／API 页面。
保存可审阅截图；浏览器缺失如实记录，构建通过不等于截图验收完成。
site/ 不提交，未经授权不部署。

## 性能证据

独立测量时记录环境、版本、规模、参数、耗时、峰值／增量内存和测量限制。
大规模比较分进程／分阶段，避免竞品污染内存基线。
只交付实测数字，不用估算冒充精准 CPU／内存限制。

## Skill 场景走查

- 新增报告字段：加载 reports-and-agents、api-and-metrics、quality；验证目录、单位、快照与证据。
- 相关性内存优化：加载 architecture、performance、api、quality；验证同一矩阵／决策与结果等价。
- 上游改造适配暂停模块：加载 architecture、api、quality；直接改调用方，跑三个模块受影响行为测试。
- 纯首页文案：加载 quality；核对承诺、链接、示例、严格构建和可用的视觉检查，无重训练。

UI 默认提示包含 $mars-risk-engineering，与主文件一致，不加虚构依赖。
只修改仓库 Skill，不安装到个人目录。

## 交付

先查看 git diff 和 status，保护用户已有工作，不回滚无关文件。
说明做了什么、为什么、测试结果、实际限制及公共契约迁移。
打包／发布以 CONTRIBUTING.md 与实际发布 CI 为准；无授权不 commit／push／merge／部署。
