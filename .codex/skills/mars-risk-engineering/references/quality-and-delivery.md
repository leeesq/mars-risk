# 质量与交付流程

## 代码与文档

- Python 3.10+ 原生类型注解。
- public API 使用完整 NumPy docstring，参数、返回值和异常与签名一致。
- 自然语言注释/docstring 使用中文。
- Ruff `D` 负责 docstring 格式和缺失检查；pydoclint 负责签名一致性。
- 有效代码行数大于 15 或命中复杂规则的私有方法，必须有中文 docstring。
- API、类名、参数名变化时同步：
  - `src`
  - `tests`
  - `README.md`
  - `docs`
  - notebook/demo（若受影响）
- 当前结构收口阶段允许 breaking changes，但不允许留下“代码已改、文档和测试未收口”的中间态。
- 涉及架构重构或新增功能时，同步检查是否违反
  `迭代进度/MARS_架构基准与新增功能准入原则.md`。

## 测试策略

- 先写回归测试复现问题，再修改实现。
- 测试覆盖 public 行为和重要内部不变量，不锁死无意义实现细节。
- Pandas/Polars 输入输出都要考虑。
- 报告导出测试应读取生成文件验证 sheet、表头和数据，不只检查文件存在。
- 重 benchmark 继续独立于 pytest 主流程和默认 CI 质量门。
- 轻量 benchmark smoke / 性能回归基准可以进入本地质量门或 CI。
- README 只记录真实跑出的 benchmark 结果。
- 大规模 benchmark 要分进程/分阶段运行竞品和 MARS，记录时间、增量内存和峰值内存。

## 文档站

- README 负责定位、快速入口和主链路。
- MkDocs 负责完整用户指南、API Reference、Demo 和性能说明。
- User Guide 可以逐步使用带 outputs 的 notebook 展示，但 CI 只渲染已有输出，不重新训练重模型。
- 文档构建必须通过 `mkdocs build --strict`。
- `site/` 不提交。

## Git

- 修改前检查 dirty worktree。
- 不使用 `git reset --hard` 或 `git checkout --` 回滚用户修改。
- push 前运行：

```powershell
git diff --check
git status --short
git fetch origin
```

- 远程前进时先 rebase 并重跑质量门；冲突时停止并报告。
- 只有用户明确要求才 commit/push。

## Windows 环境

- 为避免 conda 输出 GBK 编码失败：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
```

- pytest 绘图环境：

```powershell
$env:MPLBACKEND='Agg'
```

- `_ctypes: 拒绝访问` 常由沙箱权限引起。先确认命令本身合理，再请求审批用同一命令非沙箱运行。
