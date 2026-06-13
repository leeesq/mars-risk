# MARS 架构与能力演进任务清单 V2

## 1. 使用说明

这份文档是对当前 V2 架构与能力演进建议的可执行拆解版。

组织方式统一为：

- `Issue 标题`
- `目标`
- `涉及文件`
- `验收标准`

默认按优先级排序，可直接用于：

- GitHub Issues
- 项目 roadmap
- 里程碑拆分
- 重构任务排期

---

## 2. 明确排除项

以下事项当前不做，未来也不做，不再进入后续规划或任务拆解：

- Reject Inference / 拒绝推断
- 监控阈值触发 Webhook
- 二维特征交互探索
- 通过率-收益模拟器 / ROI
- “强化单调性约束参数”这一新增方向

说明：

- 当前分箱单调性能力不再作为新增任务推进，因为源码里已经有 `monotonic_trend`，并且已支持 `auto_asc_desc`。
- 相关实现可见：
  - `src/mars/feature/optimal_binner.py`
  - `src/mars/feature/lite_opt_binner.py`

---

## 3. P0 任务

`P0` 的目标是一次性收口 `compute / reporting / registry / public surface`，形成稳定的 `P1` 可开工基线。当前执行口径是不保留旧 facade、不保留旧深层导入兼容，稳定入口只收口到 `mars` 与一级领域包。

### P0 状态总览

- `P0-01` 已彻底完成（2026-06-13）
- `P0-02` 已彻底完成（2026-06-13）
- `P0-03` 已彻底完成（2026-06-13）
- `P0-04` 已彻底完成（2026-06-13）

## P0-01 建立共享 `compute` 层并明确 Pandas 白名单边界（已完成，2026-06-13）

- 当前状态：
  已开始，已彻底完成（2026-06-13）。
- 完成说明：
  已落地 `src/mars/compute/`，共享缺失语义、共享缺失统计和统一物化策略已收口；仓库内部已删除 `src/mars/utils/frame.py` 与 `src/mars/analysis/stability.py` 旧壳，调用方已直连 `mars.compute`。
- Issue 标题：
  `refactor: introduce shared compute layer and define pandas boundary`
- 目标：
  把现有分散的 `pl.Expr` 实践上提为共享计算层，并在这一层同时立住共享缺失语义、缺失统计 helper 与物化策略边界；`compute` 是整个 `P0` 的第一锚点。
- 涉及文件：
  - `src/mars/analysis/profiler.py`
  - `src/mars/analysis/evaluator.py`
  - `src/mars/utils/date.py`
  - `src/mars/modeling/evaluation.py`
  - `src/mars/modeling/prediction.py`
  - `src/mars/scoring/scorecard.py`
  - `src/mars/feature/selector.py`
  - 新增 `src/mars/compute/`
  - 新增 `src/mars/compute/materialization.py` 或等价模块
  - 新增 `src/mars/compute/missing.py` 或等价模块
- 验收标准：
  - 新建 `src/mars/compute/exprs` 或等价结构。
  - KS / PSI / WOE / Bad Rate 中至少 2 类算子完成共享表达式抽象。
  - 至少抽出一层共享缺失语义 / 缺失统计 helper，例如：
    - `missing_condition_expr`
    - `missing_rate_expr`
    - `build_missing_by_period_stats`
  - `MarsDataProfiler` 与 `analysis/evaluator.py` 中现有缺失率统计逻辑开始收敛到共享 helper。
  - `profile_stats` / `MarsProfileReport` 仍是高层画像入口，不被 scanner / detector 直接当作底层计算接口。
  - 文档明确 Pandas 白名单与物化策略边界：
    - logistic backend
    - scorecard export
    - report / plot rendering
  - `modeling/evaluation.py` 中最早阶段的全表 `to_pandas()` 被延后或移除。
  - `src/mars/utils/frame.py` 与 `src/mars/analysis/stability.py` 这类过渡壳被删除，内部调用直连 `mars.compute`。

## P0-02 抽离 `reporting` 层并纯化 `utils`（已完成，2026-06-13）

- 当前状态：
  已开始，已彻底完成（2026-06-13）。
- 完成说明：
  已落地 `src/mars/reporting/`，重展现能力已从 `analysis / modeling / utils` 收口到独立层；`src/mars/utils/plotter.py`、`src/mars/analysis/_html_assets.py`、`src/mars/modeling/html_report.py` 等旧壳已删除。
- Issue 标题：
  `refactor: extract reporting layer and purify utils`
- 目标：
  在 `compute` 锚点确立后，将 HTML/Excel/图表导出逻辑从 `analysis`、`modeling`、`utils` 中抽离，建立统一的 `reporting` 模块；让 `utils` 回归无状态、轻依赖的基础工具包。
- 涉及文件：
  - `src/mars/analysis/report.py`
  - `src/mars/analysis/_html_assets.py`
  - `src/mars/modeling/html_report.py`
  - `src/mars/utils/plotter.py`
  - `src/mars/utils/html.py`
  - 新增 `src/mars/reporting/`
- 验收标准：
  - 新建 `src/mars/reporting/`，至少有渲染和导出两个明确子层。
  - `utils` 中不再保留重展现职责模块。
  - `analysis` / `modeling` 只持有结果对象，不直接拼长 HTML 模板。
  - `reporting` 只消费结构化结果对象，不反向承载底层统计计算。
  - `src/mars/utils/plotter.py`、`src/mars/analysis/_html_assets.py`、`src/mars/modeling/html_report.py` 等过渡壳被删除。
  - 现有主要 HTML / Excel 输出能力行为不回退。

## P0-03 将 `BACKEND_MAP` 演进为 registry，并统一训练/预测后端适配（已完成，2026-06-13）

- 当前状态：
  已开始，已彻底完成（2026-06-13）。
- 完成说明：
  已落地 `src/mars/modeling/backends/registry.py` 与 `adapters.py`，训练 / replay / prediction 已统一走 registry 主路径；`ModelPredictor` 主路径改为显式 `model_type` 驱动，不再保留按模型对象猜 backend 的 fallback。
- Issue 标题：
  `refactor: replace backend map with registry and shared model adapter`
- 目标：
  用正式注册表替代静态 `BACKEND_MAP`，并引入统一的训练/预测能力适配层，优先解决训练/预测双轨分叉，不把外部生态扩展作为当前必做目标。
- 涉及文件：
  - `src/mars/modeling/tuning.py`
  - `src/mars/modeling/backends/__init__.py`
  - `src/mars/modeling/backends/base.py`
  - `src/mars/modeling/backends/xgboost.py`
  - `src/mars/modeling/backends/lightgbm.py`
  - `src/mars/modeling/backends/catboost.py`
  - `src/mars/modeling/backends/logistic.py`
  - `src/mars/modeling/prediction.py`
  - 新增 `src/mars/modeling/backends/registry.py`
  - 新增 `src/mars/modeling/backends/adapters.py`
- 验收标准：
  - `BACKEND_MAP` 不再是唯一派发入口。
  - 提供 `register_backend()` 与 `get_backend()`。
  - 预测逻辑不再直接堆叠多段 `isinstance` 分支，而是通过 adapter 能力层分发。
  - 内置四类后端全部通过 registry 注册。
  - 训练与预测两条主路径都通过同一套 adapter / registry 分发。
  - `ModelPredictor` 主路径必须显式接收 `model_type`，不再保留“只给模型对象自动猜 backend”的 fallback。
  - 当前目标是消除双轨，不要求一步做到完整插件生态。

## P0-04 为 `compute / reporting / registry` 服务的最小拆包（已完成，2026-06-13）

- 当前状态：
  已开始，已彻底完成（2026-06-13，按当前停止兼容版 `P0` 收口口径）。
- 完成说明：
  `analysis/report.py`、`analysis/evaluator.py`、`analysis/profiler.py`、`modeling/tuning.py` 等已完成职责型深拆；一级领域包导出已补齐，README / docs / tests 已切换到稳定入口，旧 facade 与旧深层兼容路径已清场。
- Issue 标题：
  `refactor: complete structural split and remove legacy public surface`
- 目标：
  围绕 `compute / reporting / registry` 完成系统性深拆与 public surface 清场；稳定入口收口到 `mars` 与一级领域包，旧深层路径不再承诺兼容。
- 涉及文件：
  - `src/mars/analysis/report.py`
  - `src/mars/analysis/evaluator.py`
  - `src/mars/analysis/profiler.py`
  - `src/mars/feature/selector.py`
  - `src/mars/modeling/tuning.py`
  - `src/mars/modeling/prediction.py`
- 验收标准：
  - `analysis/report.py`、`analysis/evaluator.py`、`analysis/profiler.py`、`modeling/tuning.py` 至少完成一轮职责型拆分。
  - 一级领域包 `mars.analysis`、`mars.modeling`、`mars.feature`、`mars.monitoring`、`mars.pipeline` 成为可直接使用的稳定入口。
  - README / docs / tests 不再直接依赖旧深层路径。
  - 旧 facade 与兼容转发层被删除，不保留旧 public surface。

## 3.1 P0 旧新映射

- 旧 `P0-03` -> 新 `P0-01`
- 旧 `P0-01` -> 新 `P0-02`
- 旧 `P0-04` -> 新 `P0-03`
- 旧 `P0-02` -> 新 `P0-04`，且范围缩小为“只为 `compute / reporting / registry` 服务的最小拆包”
- 旧 `P0-05` -> 新 `P1-18`

---

## 4. P1 任务

## P1-01 补齐 sklearn 协议并新增协议合规测试

- Issue 标题：
  `test/refactor: enforce estimator protocol consistency`
- 目标：
  把现有 `MarsBaseEstimator` 路线贯彻到主要 estimator / transformer / selector / binner 对象，并用测试固定行为。
- 涉及文件：
  - `src/mars/core/base.py`
  - `src/mars/feature/base.py`
  - `src/mars/feature/native_binner.py`
  - `src/mars/feature/optimal_binner.py`
  - `src/mars/feature/lite_opt_binner.py`
  - `src/mars/feature/selector.py`
  - 新增 `tests/core/`
  - 新增 `tests/contracts/`
- 验收标准：
  - 主要 `fit()` 返回 `self`。
  - 未拟合调用 `transform()` / `predict()` 时错误行为统一。
  - `set_output()` 行为在关键对象上一致。
  - 新增协议合规测试并通过。

## P1-02 统一配置对象生命周期与声明式校验

- Issue 标题：
  `refactor: centralize config/spec validation lifecycle`
- 目标：
  让 config/spec 从“只承载字段”升级为“解析、标准化、校验、冻结”的统一入口。
- 涉及文件：
  - `src/mars/analysis/config.py`
  - `src/mars/modeling/spec.py`
  - `src/mars/modeling/tuning.py`
  - `src/mars/monitoring/alerting.py`
- 验收标准：
  - 至少 `MarsProfileConfig` 和 `ModelingSpec` 有集中校验逻辑。
  - 关键参数校验不再散落在多个业务方法里重复实现。
  - 用户收到的报错能说明：
    - 哪个参数错了
    - 合法取值是什么
    - 如何修复

## P1-03 DRY 重构与胖类瘦身

- Issue 标题：
  `refactor: remove duplicated preprocessing logic and split god methods`
- 目标：
  消除重复特征类型探测、公共预处理和超长 `_fit_impl()` / 评估组装方法，并承接原 `P0-02` 中不再属于 `P0` 的系统性深拆范围。
- 涉及文件：
  - `src/mars/feature/base.py`
  - `src/mars/feature/native_binner.py`
  - `src/mars/feature/optimal_binner.py`
  - `src/mars/feature/lite_opt_binner.py`
  - `src/mars/analysis/evaluator.py`
- 验收标准：
  - 特征类型探测逻辑收口到基类或共享 helper。
  - 至少 2 个超长主流程方法被拆为更细私有方法。
  - 新增对应单元测试覆盖拆分后的 helper。
  - 原 `P0-02` 中未纳入 `P0-04` 的深拆项有明确迁移清单，并按模块族继续落地。

## P1-04 升级测试体系：镜像目录、快照测试、契约测试

- Issue 标题：
  `test: restructure test suite and add snapshot/contract coverage`
- 目标：
  让测试结构和项目结构对齐，并覆盖报告输出这类最容易“静悄悄坏掉”的区域。
- 涉及文件：
  - `tests/`
  - `src/mars/analysis/report.py`
  - `src/mars/modeling/html_report.py`
  - `src/mars/analysis/_html_assets.py`
  - `src/mars/scoring/scorecard.py`
- 验收标准：
  - `tests/` 至少按 `analysis/`、`feature/`、`modeling/` 分层。
  - 新增 HTML/summary table/scorecard output 的快照测试。
  - 新增 Public API 契约测试。
  - 报表关键字段顺序、列名变更能被测试发现。

## P1-05 收口工程质量细节

- Issue 标题：
  `chore: tighten engineering quality and runtime safety`
- 目标：
  收口 Ruff、类型现代化、异常记录、日志性能和 Joblib 共享内存细节。
- 涉及文件：
  - `pyproject.toml`
  - `src/mars/core/base.py`
  - `src/mars/modeling/evaluation.py`
  - `src/mars/modeling/backends/base.py`
  - `src/mars/utils/logger.py`
  - `src/mars/analysis/evaluator.py`
- 验收标准：
  - 至少恢复一部分 `UP` 相关现代化规则，不继续长期豁免。
  - 关键 Worker 错误记录保留完整堆栈。
  - 大日志调用改用惰性占位符。
  - Joblib 并行路径明确配置共享内存阈值或等价策略。

## P1-06 补齐 Vintage、Swap、金额纵向分析

- Issue 标题：
  `feat: add vintage swap and amount longitudinal analysis`
- 目标：
  补齐最贴近风控日常分析的第一批业务能力。
- 涉及文件：
  - `src/mars/modeling/evaluation.py`
  - `src/mars/analysis/report.py`
  - 新增 `src/mars/modeling/vintage.py`
  - 新增 `src/mars/modeling/swap.py`
- 验收标准：
  - 提供可复用的 Vintage 分析入口。
  - 提供新旧模型 Swap 分析入口。
  - 支持金额口径的 Vintage / Roll Rate 输出。
  - 输出结果可进入现有 report 渲染链路。

## P1-07 建立全局配置中心

- Issue 标题：
  `feat: introduce global runtime config center`
- 目标：
  为运行时默认行为提供统一入口，而不是在各模块分散管理。
- 涉及文件：
  - 新增 `src/mars/config.py`
  - `src/mars/__init__.py`
  - `src/mars/core/base.py`
  - `src/mars/modeling/evaluation.py`
  - `src/mars/analysis/config.py`
- 验收标准：
  - 至少支持统一设置：
    - 随机种子
    - 默认输出格式
    - 默认日志级别
    - 默认 streaming 行为
  - 有清晰的读取与覆盖规则。

---

## P1-08 为产物与导出结构建立协议版本化

- Issue 标题：
  `chore: version artifact and export schemas`
- 目标：
  为落盘产物和对外导出结构建立稳定协议，避免后续字段演进时打碎兼容性。
- 涉及文件：
  - `src/mars/modeling/results.py`
  - `src/mars/modeling/artifacts.py`
  - `src/mars/scoring/scorecard.py`
- 验收标准：
  - 关键产物结构包含 `schema_version`。
  - 产物中能追踪 `mars_version`、关键配置摘要或特征签名。
  - 导出协议升级时有明确兼容策略。

## P1-09 建立统一的数据物化策略层

- Issue 标题：
  `refactor: centralize materialization policy across polars pandas and arrow`
- 目标：
  收口 `Polars -> Pandas / Arrow` 转换策略，避免不同模块各自决定物化方式。
- 涉及文件：
  - `src/mars/utils/frame.py`
  - `src/mars/modeling/prediction.py`
  - `src/mars/modeling/backends/base.py`
  - `src/mars/modeling/backends/logistic.py`
  - 新增 `src/mars/compute/materialization.py` 或等价模块
- 验收标准：
  - 关键路径不再直接散落调用 `to_pandas()` / `to_arrow()`。
  - 至少形成统一 helper 或 policy 层。
  - 文档明确三类目标：
    - 保持在 Polars
    - 转 Arrow
    - 转 Pandas

## P1-10 为预测链路增加 schema 契约保护

- Issue 标题：
  `feat: enforce prediction schema contract checks`
- 目标：
  在训练和预测之间建立显式 schema 契约，降低线上列漂移、dtype 漂移、类别字典漂移的风险。
- 涉及文件：
  - `src/mars/modeling/prediction.py`
  - `src/mars/pipeline/pipeline.py`
  - `src/mars/modeling/spec.py`
  - `src/mars/modeling/results.py`
- 验收标准：
  - 训练结果中保存特征签名、列顺序、dtype 或等价 schema 信息。
  - 预测时能识别缺列、额外列、dtype 不匹配、类别水平漂移。
  - 错误提示明确指出不匹配项和修复建议。

## P1-11 建立公共 API 兼容层与废弃策略

- Issue 标题：
  `chore: add public api compatibility and deprecation policy`
- 目标：
  在未来拆包和重构过程中，保护顶层 Facade 与包级公共出口的稳定性。
- 涉及文件：
  - `src/mars/__init__.py`
  - `src/mars/feature/__init__.py`
  - `src/mars/modeling/__init__.py`
  - `src/mars/pipeline/__init__.py`
  - `src/mars/core/__init__.py`
- 验收标准：
  - 对外公开入口有清晰边界。
  - 旧入口迁移时有 deprecation 提示或兼容转发。
  - 文档说明哪些入口属于稳定 Public API。

## P1-12 为核心链路补性能回归基准

- Issue 标题：
  `test: add benchmark smoke coverage for performance regression`
- 目标：
  把性能退化从“事后感觉变慢”变成“基准能提前报警”。
- 涉及文件：
  - `benchmarks/`
  - `src/mars/analysis/profiler.py`
  - `src/mars/modeling/evaluation.py`
  - `src/mars/feature/native_binner.py`
  - `src/mars/feature/optimal_binner.py`
  - `src/mars/feature/lite_opt_binner.py`
- 验收标准：
  - 至少为 profiler、evaluation、1 个 binner 建 benchmark smoke case。
  - 能对明显的 Pandas 回退或表达式退化给出基准差异。
  - 基准任务可在本地和 CI 低成本运行。

## P1-13 收口领域异常层级

- Issue 标题：
  `refactor: consolidate domain-specific exception hierarchy`
- 目标：
  基于现有异常基类，逐步把散落的 `ValueError` / `TypeError` 收口到更清晰的领域异常。
- 涉及文件：
  - `src/mars/core/exceptions.py`
  - `src/mars/core/base.py`
  - `src/mars/modeling/tuning.py`
  - `src/mars/modeling/prediction.py`
  - `src/mars/analysis/evaluator.py`
  - `src/mars/feature/base.py`
- 验收标准：
  - 至少补齐 3 类高频领域异常。
  - 关键入口错误不再大量直接抛裸 `ValueError`。
  - 错误类型与错误上下文能支持更快定位问题。

---

## P1-14 定义稳定 API 层级与 experimental 命名空间

- Issue 标题：
  `architecture: define stable api compute and experimental layers`
- 目标：
  从架构层面明确哪些能力属于稳定承诺，哪些属于内核实现，哪些属于实验能力。
- 涉及文件：
  - `src/mars/__init__.py`
  - `src/mars/core/__init__.py`
  - `src/mars/feature/__init__.py`
  - `src/mars/modeling/__init__.py`
  - 新增 `src/mars/experimental/`（如需要）
- 验收标准：
  - 文档明确 Public API、Internal API、Experimental API 的边界。
  - 至少有一层稳定承诺入口与一层实验能力入口。
  - 包级导出不再无限扩张。

## P1-15 为建模与评估链路落 run manifest

- Issue 标题：
  `feat: persist run manifest for modeling and evaluation workflows`
- 目标：
  为训练、评估、导出统一生成可回放的运行清单，增强可复现性和审计性。
- 涉及文件：
  - `src/mars/modeling/spec.py`
  - `src/mars/modeling/session.py`
  - `src/mars/modeling/results.py`
  - `src/mars/modeling/artifacts.py`
- 验收标准：
  - 每次关键运行能输出标准 manifest。
  - manifest 至少包含版本、特征、配置、随机种子、切分信息、产物路径。
  - manifest 可被导出并重新读取。

## P1-16 在 pipeline / session 增加泄漏防护检查

- Issue 标题：
  `feat: add leakage guardrails to pipeline and modeling session`
- 目标：
  在正式训练前尽早识别时间穿越、目标泄漏、样本交叉污染等高风险问题。
- 涉及文件：
  - `src/mars/pipeline/pipeline.py`
  - `src/mars/modeling/session.py`
  - `src/mars/modeling/spec.py`
- 验收标准：
  - 至少覆盖时间字段、目标字段误入特征、训练/验证样本污染三类检查。
  - 错误提示能说明触发规则与可能修复方式。
  - 检查逻辑可单独开关或配置。

## P1-17 为特征筛选与监控补齐缺失率异常检测

- Issue 标题：
  `feat: add missing-rate anomaly detection for feature screening and monitoring`
- 目标：
  让 MARS 能自动识别按天或按期缺失率的异常跳变，覆盖建模前特征筛查和上线后监控两类场景，替代人工逐日巡检。
- 涉及文件：
  - `src/mars/monitoring/monitor.py`
  - `src/mars/monitoring/alerting.py`
  - `src/mars/analysis/evaluator.py`
  - `src/mars/feature/selector.py`
  - `docs/user-guide/monitoring.md`
  - `docs/user-guide/feature-selection.md`
  - 新增 `src/mars/monitoring/anomaly.py` 或等价模块
- 验收标准：
  - 至少提供两类入口：
    - 建模前 / 离线：对特征缺失率时间序列做异常扫描，输出异常特征和异常时间窗口。
    - 监控 / 在线：对最新分组或最新日期缺失率做自动异常判别。
  - 实现依赖共享缺失语义和按期缺失统计底座，不单独重复定义“什么算缺失”。
  - 至少支持识别缺失率异常上升和异常下降。
  - 在线检测应考虑每日样本量，优先采用 `p-chart` / 双侧 `CUSUM` 或等价二项比例方法；假设检验可作为确认而不是唯一规则。
  - 输出结构化异常结果，至少包含 `feature`、`date/group`、`missing_rate`、`baseline/reference`、`anomaly_direction`、`anomaly_score`。
  - 不强依赖 `profile_stats` / `MarsProfileReport` 这类高层画像入口；如已有 `missing_by_day_table`，可直接复用或提供等价入口。
  - 如提供 `feature_data_source`，可聚合到数据源级别，识别多特征同步异常。
  - 只输出 report / table / summary，不包含 Webhook、调度或外部处置流程。

## P1-18 金额加权评估 / 件数与金额双视角评估

- Issue 标题：
  `feat: add amount-weighted evaluation and dual-perspective binning`
- 目标：
  在模型评估和分箱评估中补齐金额 / 敞口口径，形成件数与金额双视角；这项能力从原 `P0-05` 下放到 `P1`，作为进入 `P1` 后优先执行的第一张业务增强单。
- 涉及文件：
  - `src/mars/modeling/evaluation.py`
  - `src/mars/modeling/metrics.py`
  - `src/mars/analysis/evaluator.py`
  - `src/mars/analysis/report.py`
  - `src/mars/reporting/`
- 验收标准：
  - `evaluate()` 支持 `amount_col` 或 `weight_col`。
  - summary table 增加：
    - `Total Amount`
    - `Bad Amount`
    - `Amount Bad Rate`
  - AUC / KS 支持 `sample_weight`。
  - bin report 中可同时展示件数坏账率和金额坏账率。
  - 输出进入统一结果对象和 `reporting` 链路，不只是在模板层临时拼字段。
  - 范围只到金额加权评估与双视角输出，不扩展到 ROI、通过率-收益模拟器或其他业务模拟。

---

## 5. P2 任务

## P2-01 为 SHAP 报告集成预留统一入口

- Issue 标题：
  `feat: integrate shap explanation into reporting pipeline`
- 目标：
  让 SHAP 输出能进入统一报告链路，而不是散落在单独分析脚本里。
- 涉及文件：
  - `src/mars/modeling/tuning.py`
  - `src/mars/modeling/report.py`
  - `src/mars/modeling/results.py`
  - 新增 `src/mars/modeling/explain.py`
- 验收标准：
  - SHAP 输出有稳定的数据结构封装。
  - 报告层可消费该结构。
  - 无 SHAP 依赖时错误提示明确。

## P2-02 评分卡与模型结果标准化导出

- Issue 标题：
  `feat: standardize model export formats`
- 目标：
  为评分卡和建模结果提供更标准的交付格式。
- 涉及文件：
  - `src/mars/scoring/scorecard.py`
  - `src/mars/modeling/results.py`
  - `src/mars/modeling/artifacts.py`
- 验收标准：
  - 至少支持 JSON 结构化导出。
  - 导出字段含义稳定且文档化。
  - 导出结果可被外部系统直接消费或解析。

## P2-03 为 streaming / out-of-core 建立正式入口

- Issue 标题：
  `feat: add streaming execution switches for large-scale workloads`
- 目标：
  让超大样本场景具备更正式的 streaming 开关和边界说明。
- 涉及文件：
  - `src/mars/analysis/profiler.py`
  - `src/mars/feature/base.py`
  - `src/mars/modeling/evaluation.py`
  - `src/mars/config.py`
- 验收标准：
  - 关键路径支持显式启用 streaming。
  - 文档说明哪些链路支持 streaming，哪些不支持。
  - 至少一条大数据路径有验证样例。

## P2-04 做依赖分层与安装瘦身

- Issue 标题：
  `chore: separate core dependencies from reporting extras`
- 目标：
  将核心计算依赖和重可视化/导出依赖进一步分层。
- 涉及文件：
  - `pyproject.toml`
  - `README.md`
  - `docs/getting-started/installation.md`
- 验收标准：
  - 安装说明区分核心安装与可选安装。
  - reporting / notebook / shap / model backends 的依赖边界更清晰。
  - 核心安装不强绑定不必要的重型展示依赖。

## P2-05 升级交互式文档展示

- Issue 标题：
  `docs: add notebook-driven interactive documentation showcases`
- 目标：
  用交互式文档展示核心报告和分析能力，提升可理解性与产品感知。
- 涉及文件：
  - `mkdocs.yml`
  - `docs/`
  - `docs/demos/`
- 验收标准：
  - 至少有 1 到 2 个 Notebook 示例被无缝渲染进文档。
  - 能展示 Vintage / Swap / 报表输出示例。
  - 文档构建流程稳定。

---

## P2-06 为评分卡补充 reason code 输出

- Issue 标题：
  `feat: add reason code output for scorecard scenarios`
- 目标：
  为评分卡输出可解释的原因码，增强业务沟通、审批解释和客诉场景支持。
- 涉及文件：
  - `src/mars/scoring/scorecard.py`
  - `src/mars/modeling/report.py`
  - `src/mars/modeling/results.py`
- 验收标准：
  - 至少支持输出每个样本的主要负向贡献项或规则级原因码。
  - 原因码结构可序列化并进入报告链路。
  - 文档包含最小示例。

## P2-07 为报告增加业务语义层与口径说明卡片

- Issue 标题：
  `feat: add business semantic layer and metric definition cards to reports`
- 目标：
  让报告不仅对模型同学可读，也能直接面向业务评审与复盘会议。
- 涉及文件：
  - `src/mars/analysis/report.py`
  - `src/mars/modeling/report.py`
  - 新增 `src/mars/reporting/semantics.py`（如需要）
- 验收标准：
  - 报告支持字段业务别名或指标释义。
  - 关键指标可展示口径说明卡片。
  - 语义信息可通过配置注入。

## P2-08 增加标准业务切片模板

- Issue 标题：
  `feat: provide reusable business segmentation templates`
- 目标：
  为常见业务分析切片提供标准模板，减少分析师重复拼接 group 逻辑。
- 涉及文件：
  - `src/mars/modeling/evaluation.py`
  - `src/mars/analysis/report.py`
  - 新增 `src/mars/analysis/segments.py`（如需要）
- 验收标准：
  - 至少提供 3 到 5 类常用业务切片模板。
  - 模板可被评估或报告模块直接消费。
  - 使用方式文档化。

## P2-09 收敛最小可用路径并新增 dry-run / preflight 模式

- Issue 标题：
  `feat/docs: add preflight mode and minimum user journeys`
- 目标：
  降低第一次使用 MARS 的门槛，在正式跑大任务前先完成前置校验。
- 涉及文件：
  - `README.md`
  - `docs/getting-started/quickstart.md`
  - `docs/user-guide/modeling-pipeline.md`
  - `src/mars/pipeline/pipeline.py`
  - `src/mars/modeling/session.py`
- 验收标准：
  - 文档明确 3 到 4 条最小可用路径。
  - 提供 dry-run / preflight 检查入口或等价模式。
  - 可在正式执行前校验字段、dtype、target、split、backend、依赖。

---

## 6. 建议的建单顺序

如果按真正落地执行来排，我建议 issue 创建顺序如下。`P0` 的退出标准是形成 `P1` 可开工基线；进入 `P1` 后，第一张业务增强单优先开 `P1-18`。

1. `P0-01` 建立共享 `compute` 层并明确 Pandas 白名单边界（已完成）
2. `P0-02` 抽离 `reporting` 层并纯化 `utils`（已完成）
3. `P0-03` 将 `BACKEND_MAP` 演进为 registry，并统一训练/预测后端适配（已完成）
4. `P0-04` 为 `compute / reporting / registry` 服务的最小拆包（已完成）
5. `P1-18` 金额加权评估 / 件数与金额双视角评估
6. `P1-01` 补齐 sklearn 协议并新增协议合规测试
7. `P1-02` 统一配置对象生命周期与声明式校验
8. `P1-14` 定义稳定 API 层级与 experimental 命名空间
9. `P1-04` 升级测试体系：镜像目录、快照测试、契约测试
10. `P1-03` DRY 重构与胖类瘦身
11. `P1-07` 建立全局配置中心
12. `P1-08` 为产物与导出结构建立协议版本化
13. `P1-09` 建立统一的数据物化策略层
14. `P1-10` 为预测链路增加 schema 契约保护
15. `P1-15` 为建模与评估链路落 run manifest
16. `P1-16` 在 pipeline / session 增加泄漏防护检查
17. `P1-17` 为特征筛选与监控补齐缺失率异常检测
18. `P1-05` 收口工程质量细节
19. `P1-11` 建立公共 API 兼容层与废弃策略
20. `P1-13` 收口领域异常层级
21. `P1-12` 为核心链路补性能回归基准
22. `P1-06` 补齐 Vintage、Swap、金额纵向分析
23. `P2-09` 收敛最小可用路径并新增 dry-run / preflight 模式
24. `P2-06` 为评分卡补充 reason code 输出
25. `P2-07` 为报告增加业务语义层与口径说明卡片
26. `P2-08` 增加标准业务切片模板

---

## 7. 备注

这份任务清单已经和当前 V2 规划对齐，并且已经移除了以下方向：

- Reject Inference
- Webhook 告警
- 二维特征交互探索
- ROI 模拟器
- 单调性约束增强任务

其中单调性相关项之所以移除，是因为当前源码已经存在：

- `monotonic_trend`
- `auto_asc_desc`

所以这里不再把它列为后续新增功能。
