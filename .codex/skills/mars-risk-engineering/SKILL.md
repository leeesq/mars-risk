---
name: mars-risk-engineering
description: Project-specific internal engineering workflow for the leeesq/mars-risk Python library. Use when Codex reviews, refactors, implements, tests, documents, benchmarks, packages, releases, or maintains MARS code for data profiling, binning, risk evaluation, feature selection, modeling, pipelines, monitoring, reports, API design, module naming, docstrings, CI, MkDocs, README, or internal/PyPI delivery.
---

# MARS Risk Engineering

## 工作原则

1. 先读代码、测试、`pyproject.toml`、README 和相关 docs，再决定实现方式。
2. 优先遵守仓库内的架构基准与新增功能准入原则；新增能力先判断是否属于主线和底座。
3. 当前结构收口阶段默认允许 breaking changes；但必须同波收口 `src`、`tests`、`README.md`、`docs`、skill 与规划文档，不保留旧深层路径兼容壳。
4. 优先复用 MARS 已有基类、共享 `compute` 能力、共享缺失语义、DataFrame、可选依赖、reporting 和 artifact 工具。
5. 自然语言注释和 docstring 使用中文；API 名、参数名、NumPy section 标题保留英文。
6. 修改源码时同步更新类型注解、docstring、测试和用户文档。
7. 不回滚用户已有改动，不清理与当前任务无关的 dirty worktree。

## 开始工作

1. 运行 `git status --short`，识别已有修改。
2. 使用 `rg` 搜索 public 入口、内部调用、测试、README 和 docs。
3. 优先阅读 [架构基准与新增功能准入原则](../../../迭代进度/MARS_架构基准与新增功能准入原则.md)。
4. 阅读 [架构边界](references/architecture.md)；涉及 API、指标或口径时再读
   [API 与业务口径](references/api-and-metrics.md)。
5. 涉及测试、文档、CI、benchmark、打包或发布时读取
   [质量与交付流程](references/quality-and-delivery.md)。
6. 如果任务涉及 V2 规划、新功能准入或架构演进，再补读：
   - [MARS_架构与能力演进任务清单_V2](../../../迭代进度/MARS_架构与能力演进任务清单_V2.md)
   - [MARS_架构与能力演进任务清单_V2_详细实施步骤](../../../迭代进度/MARS_架构与能力演进任务清单_V2_详细实施步骤.md)

## API 设计

- 构造函数只保存稳定策略、模型规格和阈值。
- `fit/evaluate/generate/split/tune/monitor` 接收数据、列名、样本范围和本次运行选项。
- 底层算法使用 `X, y`；高层风控工作流使用 `df, target`。
- 同一个 public method 不同时暴露 `y` 和 `target`。
- sklearn 风格对象优先保持 `fit/transform/predict/evaluate` 语义。
- 有状态对象必须明确拟合状态归属；一次运行结果优先放入 `Result`、`Report` 或
  `Profile` 对象，不把本次数据状态隐式留在可复用工具实例中。
- 避免同一配置在 init 和 method 重复出现；必须支持覆盖时使用清晰的
  `None -> 实例默认值` 语义并记录最终生效配置。
- 新增入口先判断它属于 Public API、Internal API 还是 Experimental API；不要默认把一切新能力直接暴露到稳定 public surface。

## 实现边界

- DataFrame 转换与物化策略统一复用 `mars.compute`；不要再新增 `mars.utils.frame` 一类转发壳。
- 可选依赖使用 `mars.utils.imports`。
- 数值稳定性常量使用 `mars.core.constants`，禁止在源码重新散落
  `1e-6/1e-9/1e-12/1e-15`。
- 共享统计与表达式优先复用 `mars.compute`；新增底层算子先判断是否应进入共享 `compute` 层。
- 缺失语义与缺失统计必须复用共享底座；`Null / NaN / missing_values` 不得在
  `profiler / evaluator / monitoring / scanner / detector` 中各自维护一套逻辑。
- `profile_stats` / `MarsProfileReport` 只作为高层画像入口，不作为 scanner / detector 的底层计算依赖。
- PSI 公式和缺失/特殊箱口径优先复用 `mars.compute`、`MarsBinEvaluator`
  或共享 stability helper。
- 分箱器共享能力放在 `MarsBinnerBase`；不要在 Native、Optimal、LiteOpt 重复实现
  transform、WOE、规则导出、SQL 或序列化。
- 建模评估优先复用已有风险评估和 `modeling.evaluation_tables`，不要再手写一套 PSI、
  ROC、KS、Lift 或 Calibration。
- `reporting` 是独立层；重 HTML / Excel / plot 职责优先落在 `mars.reporting`，
  不要继续把重展现逻辑沉到 `mars.utils`。
- `mars.utils` 只保留轻量格式化、日期、日志、可选依赖等基础工具；不要再把核心计算或重展现转发层放回 `utils`。
- artifact 路径、JSON、表格和模型读写集中在 `mars.modeling.artifacts`。
- Pandas 只能留在明确白名单区域，例如 logistic backend、scorecard export、
  report / plot rendering；新增核心链路默认优先保持在 Polars / `pl.Expr` 内完成。

## 修改纪律

- 手工编辑使用 `apply_patch`。
- 新增或修改函数必须有完整类型注解。
- public API 使用完整 NumPy docstring；复杂私有 helper 使用真实、简洁的中文 docstring。
- 注释说明业务意图、约束和取舍，不逐句翻译代码。
- 不使用 `type: ignore` 掩盖可修复问题。
- 不新增第三方依赖，除非确有必要并同步 `pyproject.toml`、README 和测试。
- 不把 notebook、benchmark 产物、模型 artifact、站点构建目录提交到主仓库，除非任务明确要求。

## 验证顺序

先跑定向测试，再跑完整质量门：

```powershell
conda run -n mars python -m ruff check src tests benchmarks scripts
conda run -n mars python -m mypy src\mars
conda run -n mars pydoclint src\mars
conda run -n mars python scripts\check_private_docstrings.py src\mars
$env:MPLBACKEND='Agg'; conda run -n mars python -m pytest -q --basetemp .pytest-tmp-codex
conda run -n mars python -m mkdocs build --strict
```

Windows 沙箱若因 `_ctypes`、pandas、click 或 MkDocs 导入报“拒绝访问”，确认不是代码错误后，
请求审批并在非沙箱环境重跑同一命令。

## 完成标准

- 行为和指标口径有回归测试。
- public API 的代码、docstring、README、docs 和示例一致。
- `ruff`、`mypy`、`pydoclint`、私有 docstring 检查、pytest 通过。
- 文档变更通过 `mkdocs build --strict`。
- 若任务涉及新功能准入或架构演进，结果需与 `迭代进度/MARS_架构基准与新增功能准入原则.md` 一致。
- `git diff --check` 通过，工作树没有意外产物。
- 只有用户明确要求时才 commit、merge、push 或发布。
