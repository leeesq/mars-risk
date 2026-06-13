# MARS 架构与能力演进任务清单 V2 详细实施步骤

## 1. 文档定位

这份文档是对 [MARS_架构与能力演进任务清单_V2.md](./MARS_架构与能力演进任务清单_V2.md) 的进一步展开版。

原任务清单回答的是：

- 要做什么
- 目标是什么
- 涉及哪些文件
- 如何验收

这份“详细实施步骤版”回答的是：

- 先做哪一步
- 中间如何拆层
- 哪些任务有依赖关系
- 每项落地时更合理的推进顺序是什么

默认目标不是“一次性把全部任务做完”，而是帮助团队以更低返工成本推进每个 issue。

---

## 2. 总体实施原则

### 2.1 先抽底座，再补功能

对 MARS 这类内部框架来说，最怕的是：

- 功能做得快
- 口径越来越多
- 模块间重复实现越来越重

因此整体顺序必须坚持：

1. 先做共享底座和边界收口。
2. 再做业务能力增强。
3. 最后再做体验和展示增强。

### 2.2 直接收口稳定入口，不保留旧兼容壳

当前 `P0` 执行口径是：

1. 先确定稳定入口只收口到顶层 `mars` 与一级领域包。
2. 代码、测试、README、docs 同波切到新入口。
3. 旧 facade、旧深层导入路径和兼容转发层直接删除。
4. 不留下“新旧路径并存”的长期中间态。

### 2.3 先统一缺失语义，再做缺失异常检测

缺失率异常检测相关能力必须建立在共享缺失语义底座上。

具体来说：

- `Null`
- `NaN`
- `missing_values`
- 类型兼容判断

都应复用统一 helper，而不是在 `profiler`、`evaluator`、`monitoring`、`scanner`、`detector` 中各写一套。

### 2.4 先结构化输出，再考虑外部集成

监控、异常检测、报告、导出相关能力，优先输出：

- 表
- 结果对象
- 结构化 summary

不在本阶段内耦合：

- Webhook
- 调度系统
- 监控看板
- 自动处置流程

---

## 3. 推荐实施波次

### Wave 0：准备与基线

建议先完成以下准备工作：

1. 固化当前测试基线，至少保证核心测试可跑。
2. 明确顶层公共 API 清单，记录当前对外入口。
3. 记录当前典型大文件、`to_pandas()` 热点、HTML 输出路径。
4. 为后续重构准备一个“迁移日志”文档，记录旧路径到新路径的映射。

### Wave 1：形成 `P1` 可开工基线

这一波只包含 4 个纯结构地基事项，并按下面顺序推进：

- `P0-01` 建立共享 `compute` 层并明确 Pandas 白名单边界（已完成，2026-06-13）
- `P0-02` 抽离 `reporting` 层并纯化 `utils`（已完成，2026-06-13）
- `P0-03` 将 `BACKEND_MAP` 演进为 registry，并统一训练/预测后端适配（已完成，2026-06-13）
- `P0-04` 完成系统性深拆与最终 public surface 清场（已完成，2026-06-13）

当前状态：截至 2026-06-13，Wave 1 四项已全部彻底完成，已形成 `P1` 可开工基线。

`P0` 退出条件统一定义为：

- 共享缺失语义、共享缺失统计和 Pandas 白名单已立
- `reporting` 与 `utils` 的边界已建立
- 训练 / 预测后端分发不再双轨
- 直接阻塞 `P1` 的大文件 / 大模块耦合已拆到足够程度
- 允许进入 `P1`，且旧 facade / 旧深层入口已完成清场

### Wave 2：`P1` 首批能力与规范收口

这一波对应：

- `P1-18`
- `P1-01` 到 `P1-05`
- `P1-07` 到 `P1-17`

### Wave 3：业务增强与中期能力

这一波对应：

- `P1-06`
- `P2-01` 到 `P2-09`

---

## 4. P0 任务详细实施步骤

## P0-01 建立共享 `compute` 层并明确 Pandas 白名单边界（已完成，2026-06-13）

- 当前状态：
  已开始，已彻底完成（2026-06-13）。
- 完成说明：
  已新增 `src/mars/compute/`，共享缺失语义、共享缺失统计、统一物化策略和稳定性低层算子已收口；旧 `utils.frame` / `analysis.stability` 壳已删除，内部调用已切到新底座。

### 前置依赖

- 无强依赖。
- 这是整个 `P0` 的第一锚点，应优先启动。

### 实施步骤

1. 识别现有共享算子候选。
   先从以下位置抽共性：
   - `analysis/profiler.py`
   - `analysis/stability.py`
   - `analysis/evaluator.py`
   - `utils/date.py`
2. 新建 `src/mars/compute/`。
   建议第一版至少包含：
   - `exprs/`
   - `stats/`
   - `materialization.py`
   - `missing.py`
3. 先抽缺失语义底座。
   建议优先形成：
   - `missing_condition_expr`
   - `missing_rate_expr`
   - `build_missing_by_period_stats`
4. 再抽公共统计算子。
   先选 2 到 3 类高频算子，例如：
   - PSI
   - Bad Rate
   - WOE
5. 收口 Pandas 白名单与物化边界。
   建统一 helper 或 policy，显式标注哪里允许转 Pandas：
   - logistic backend
   - scorecard export
   - report / plot rendering
6. 改造已有调用方。
   先让 `MarsDataProfiler` 与 `analysis/evaluator.py` 复用共享缺失 helper。
7. 为后续 scanner / detector 留标准入口。
   明确规定它们不直接依赖 `profile_stats` / `MarsProfileReport`。

### 实施注意点

- 这是后续缺失率异常检测、统一物化策略和性能回归基准的前置地基。
- 先抽“语义”和“算子”，不要一开始就搞复杂的 query planner。

## P0-02 抽离 `reporting` 层并纯化 `utils`（已完成，2026-06-13）

- 当前状态：
  已开始，已彻底完成（2026-06-13）。
- 完成说明：
  已新增 `src/mars/reporting/`，报告渲染、导出和重绘图逻辑已迁移；`utils` 不再承载重展现职责，相关旧壳已清理。

### 前置依赖

- 建议建立在 `P0-01` 的共享 `compute` 锚点之上推进。

### 实施步骤

1. 盘点现有展现职责。
   重点梳理 `analysis/report.py`、`analysis/_html_assets.py`、`modeling/html_report.py`、`utils/plotter.py`、`utils/html.py` 中哪些是：
   - 结果对象
   - HTML 片段
   - 静态资源
   - 导出逻辑
2. 新建 `src/mars/reporting/` 目录骨架。
   建议先拆出：
   - `renderers/`
   - `exports/`
   - `assets/`
   - `templates/`
3. 迁移纯展现逻辑。
   先移动不依赖大量业务状态的模块，例如：
   - HTML 片段拼接
   - Excel 导出
   - plot helper
4. 删除旧入口兼容壳。
   旧模块不再保留轻量转发；同波完成 README、docs、tests 的入口切换。
5. 清理 `utils`。
   把重展现职责迁出后，明确 `utils` 仅保留无状态、轻依赖 helper。
6. 收口结果边界。
   确保 `reporting` 只消费结构化结果对象或 `compute` 输出，不反向承载底层统计计算。
7. 回归验证。
   验证现有 HTML / Excel / plot 结果行为不回退。

### 实施注意点

- 不要一开始就重写所有模板。
- 先搬家，再收口接口，再统一模板规范。

## P0-03 将 `BACKEND_MAP` 演进为 registry，并统一训练/预测后端适配（已完成，2026-06-13）

- 当前状态：
  已开始，已彻底完成（2026-06-13）。
- 完成说明：
  backend registry 与 adapter 已成为训练 / replay / prediction 的唯一主路径；预测侧不再保留按模型对象猜 backend 的兼容 fallback。

### 前置依赖

- 建议在 `P0-01` 明确公共底座后推进。
- 如需拆 `modeling/tuning.py` 的阻塞块，可与 `P0-04` 配合实施。

### 实施步骤

1. 盘点现有后端能力。
   先列出每个 backend 需要暴露哪些动作：
   - fit
   - predict
   - predict_proba
   - feature importance
   - artifact save/load
2. 定义 backend adapter 接口。
   明确统一契约，而不是直接让 tuning 和 prediction 自己猜模型类型。
3. 引入 registry。
   提供：
   - `register_backend`
   - `get_backend`
   - `list_backends`
4. 先迁移内置 backend。
   例如先迁 logistic / lightgbm / xgboost 之类现有主路径。
5. 改造训练与预测入口。
   `tuning.py` 与 `prediction.py` 都走统一 adapter。
6. 把当前完成标准收口到“解决训练 / 预测双轨”。
   外部插件生态只保留最小扩展点，不作为本轮完成条件。

### 实施注意点

- 先解决“训练和预测两套识别逻辑”的问题。
- 插件化是结果，不是第一目标。

## P0-04 完成系统性深拆与最终 public surface 清场（已完成，2026-06-13）

- 当前状态：
  已开始，已彻底完成（2026-06-13，按当前停止兼容版 `P0` 收口口径）。
- 完成说明：
  已完成系统性职责拆分并收口 public surface，稳定入口收口到 `mars` 与一级领域包；README、docs、tests 和项目 skill 已同步切到新基线。

### 前置依赖

- 建议在前三项边界已经基本明确后推进。

### 实施步骤

1. 先列“阻塞拆分地图”。
   只标注那些直接阻塞 `compute`、`reporting`、`registry` 收口的大文件和耦合块，例如：
   - `analysis/report.py`
   - `analysis/evaluator.py`
   - `analysis/profiler.py`
   - `feature/selector.py`
   - `modeling/tuning.py`
   - `modeling/prediction.py`
2. 优先抽“低耦合且直接解阻塞”的块。
   重点是：
   - 结果对象
   - rendering glue
   - backend dispatch
   - 共享 helper
   - import cycle breaker
3. 只在必要时创建最小子包结构。
   目标是支撑前三项收口，而不是把 `analysis / feature / modeling` 一次性深拆完成。
4. 保持顶层入口稳定。
   对外继续从原有包级入口导出，内部再切新结构。
5. 记录剩余深拆清单。
   将未纳入本轮的系统性拆包范围明确移交给 `P1-03` 承接。
6. 回归验证。
   确认前三项能力收口后，不再被原大文件里的耦合反复拖回去。

### 实施注意点

- 每次拆分最好只动一个大模块族。
- 不要把“目录拆开”误当成“职责已清晰”，命名和导出边界必须一起收口。
- 这一步的目标是“足够进入 `P1`”，不是“把历史结构债一次性清空”。

---

## 5. P1 任务详细实施步骤

## P1-01 补齐 sklearn 协议并新增协议合规测试

### 前置依赖

- 建议在 `P0-02` 后推进。

### 实施步骤

1. 盘点现有 estimator / transformer。
2. 统一 `__init__` 只绑定超参。
3. 统一拟合后状态命名为 `_` 后缀。
4. 补 `get_params / set_params`、必要时补 `fit_transform`。
5. 为关键类加协议合规测试。

## P1-02 统一配置对象生命周期与声明式校验

### 前置依赖

- 可与 `P1-07` 一起设计。

### 实施步骤

1. 盘点当前 config/spec/dataclass 分布。
2. 区分：
   - 用户输入配置
   - 运行时解析配置
   - 拟合后状态配置
3. 统一校验入口。
4. 先在高频配置对象落地。
5. 补配置错误测试。

## P1-03 DRY 重构与胖类瘦身（进行中）

### 前置依赖

- 强依赖 `P0-04` 已完成的 modeling 一级分层与稳定入口收口成果。
- 同时承接原 `P0-02` 中不再放在 `P0` 的系统性深拆范围，但本阶段重点已经从“先分类目录”转为“继续完成 modeling 边界收口”。

### 当前状态

- `src/mars/modeling/` 已完成 `backends / workflows / inference / evaluation / contracts / artifacts` 分层。
- `mars.modeling` 已成为稳定入口，当前不再需要为“是否建立内部分类文件夹”做架构决策。
- 剩余工作主要集中在第二阶段收口：`contracts` 纯化、`ModelingSpec` 收口、`inference / evaluation` 解耦、胖文件继续拆薄、重复 helper 与依赖方向清理。

### 实施步骤

1. 先纯化 `contracts`。
   预期出口：`contracts` 只保留结构化对象与协议；结果对象不再继续承担 artifact I/O，高层流程也不再反向渗入 `contracts`。
2. 再收口 `ModelingSpec`。
   预期出口：session / tuner / replay / feature growth 统一围绕 `ModelingSpec` / `ReplaySpec` 传递规格，不再平行维护一套大参数与构造逻辑。
3. 再拆 `inference / evaluation` 职责。
   预期出口：`predictor` 聚焦推理与薄包装便利能力；评估主逻辑回到 `evaluation`，不再把评分后评估装配写死在推理层。
4. 再继续拆剩余胖文件。
   预期出口：重点处理 `workflows/tuner.py`、`workflows/feature_growth.py`、`evaluation/metrics.py`、`contracts/tuning_result.py` 等热点文件，把职责拆到更清晰的内部 helper 或同层模块中。
5. 最后清理重复 helper 和内部依赖方向。
   预期出口：`split_name_sort_key`、`_json_dumps` 等重复实现收口为单一来源；`contracts` 不再反向依赖高层，`inference` 与 `evaluation` 的内部依赖方向清晰稳定。
6. 补回归测试并同步更新文档。
   预期出口：涉及 `mars.modeling` 主链路的 tests、README 和 docs 与新边界保持一致，不再出现“目录已分层但文档仍按旧平铺结构理解”的失真。

## P1-04 升级测试体系：镜像目录、快照测试、契约测试

### 前置依赖

- 建议在 `P0-01`、`P0-02` 后推进。

### 实施步骤

1. 整理测试目录与源码目录对应关系。
2. 为 report / export / summary 结果加 snapshot tests。
3. 为 Public API、artifact schema、backend adapter 加 contract tests。
4. 为大重构路径补 smoke tests。
5. 建立测试命名与 fixture 规范。

## P1-05 收口工程质量细节

### 前置依赖

- 无强依赖。

### 实施步骤

1. 收敛 Ruff 忽略项。
2. 恢复一部分现代化规则。
3. 收口日志调用风格。
4. 检查 joblib / 并行路径的内存策略。
5. 补运行期错误上下文。

## P1-06 补齐 Vintage、Swap、金额纵向分析

### 前置依赖

- 建议晚于 `P1-18`。

### 实施步骤

1. 先定义结果对象和输入口径。
2. 实现 `VintageAnalyzer`。
3. 实现 `SwapAnalyzer`。
4. 在金额维度补 Vintage / Roll Rate。
5. 接到现有 report 渲染链路。

## P1-07 建立全局配置中心

### 前置依赖

- 与 `P1-02` 协同设计更合适。

### 实施步骤

1. 先定义“全局默认”和“局部覆盖”的边界。
2. 新建 `mars.config` 或 `mars.options`。
3. 接入高频默认项：
   - 随机种子
   - 默认输出格式
   - 默认日志级别
   - 默认 streaming
4. 补读取优先级文档。
5. 验证局部覆盖不破坏旧行为。

## P1-08 为产物与导出结构建立协议版本化

### 前置依赖

- 建议晚于 `P1-15` 一起落。

### 实施步骤

1. 盘点当前 artifacts / results / scorecard 导出结构。
2. 为每类结构增加 `schema_version`。
3. 补 `mars_version`、配置摘要、特征签名。
4. 设计兼容读取策略。
5. 为旧产物准备至少一条兼容回读测试。

## P1-09 建立统一的数据物化策略层

### 前置依赖

- 与 `P0-03` 强相关。

### 实施步骤

1. 全局搜索 `to_pandas()` / `to_arrow()`。
2. 定义物化目标枚举或 policy。
3. 新建统一物化 helper。
4. 优先替换最早期、最隐式的转换点。
5. 记录保留 Pandas 岛的原因。

## P1-10 为预测链路增加 schema 契约保护

### 前置依赖

- 建议晚于 `P1-15`。

### 实施步骤

1. 定义训练产物中的 schema 签名格式。
2. 保存：
   - 列顺序
   - dtype
   - 类别水平
   - 特征签名
3. 在预测入口做契约检查。
4. 为缺列、多列、dtype 漂移分别报清晰错误。
5. 补线上风格 smoke case。

## P1-11 建立公共 API 兼容层与废弃策略

### 前置依赖

- 建议在 `P0-02` 后推进。

### 实施步骤

1. 列出当前包级公共入口。
2. 标记稳定 / 内部 / 实验级别。
3. 为迁移入口加兼容转发。
4. 为旧路径补 deprecation 提示。
5. 文档化稳定 API 范围。

## P1-12 为核心链路补性能回归基准

### 前置依赖

- 最好晚于 `P0-03`。

### 实施步骤

1. 选 3 条最关键链路。
2. 为每条链路建 benchmark smoke case。
3. 固定输入规模与输出指标。
4. 比较表达式路径与 Pandas 回退路径。
5. 把 benchmark 纳入本地可重复运行脚本。

## P1-13 收口领域异常层级

### 前置依赖

- 无强依赖。

### 实施步骤

1. 盘点高频裸 `ValueError` / `TypeError`。
2. 定义领域异常分类。
3. 从关键入口先替换。
4. 保证错误上下文更清晰。
5. 为高频错误补测试。

## P1-14 定义稳定 API 层级与 experimental 命名空间

### 前置依赖

- 与 `P1-11` 协同推进。

### 实施步骤

1. 明确稳定 / internal / experimental 三层。
2. 盘点当前顶层导出。
3. 对实验能力集中到 `experimental/` 或等价命名空间。
4. 对稳定入口建立文档承诺。
5. 防止包级导出无限扩张。

## P1-15 为建模与评估链路落 run manifest

### 前置依赖

- 与 `P1-08`、`P1-10` 强相关。

### 实施步骤

1. 定义 manifest schema。
2. 在训练、评估、导出流程接入 manifest 生成。
3. 记录版本、配置、随机种子、特征、切分、产物路径。
4. 增加 manifest 读取能力。
5. 补回放 / 复现 smoke tests。

## P1-16 在 pipeline / session 增加泄漏防护检查

### 前置依赖

- 无强依赖。

### 实施步骤

1. 先定义泄漏规则清单。
2. 在 `pipeline / session` 入口做 pre-check。
3. 支持开关与配置。
4. 错误信息给出修复建议。
5. 为时间穿越、目标入特征、样本污染加测试。

## P1-17 为特征筛选与监控补齐缺失率异常检测

### 前置依赖

- 强依赖 `P0-03` 中共享缺失语义底座。

### 实施步骤

1. 先抽底座。
   确保已有：
   - `missing_condition_expr`
   - `missing_rate_expr`
   - `build_missing_by_period_stats`
2. 先做建模前 scanner。
   建议实现离线扫描器，例如 `MarsMissingShiftScanner`，职责是：
   - 输入原始明细表或按期缺失统计表
   - 输出异常特征和异常时间窗口
3. 再做上线后 detector。
   建议实现监控检测器，例如 `MarsMissingShiftDetector`，职责是：
   - 消费最新日期或最新分组
   - 给出异常方向与严重度
4. 方法论落地顺序。
   建议优先：
   - scanner：变点检测或二段分割式比例差异扫描
   - detector：`p-chart` + 双侧 `CUSUM`
   - 假设检验：只做确认器
5. 输出统一结构化结果。
   至少包含：
   - `feature`
   - `date/group`
   - `missing_rate`
   - `baseline/reference`
   - `direction`
   - `severity` 或 `anomaly_score`
   - `data_source`
6. 对接数据源级聚合。
   若配置了 `feature_data_source`，支持源级异常摘要。
7. 不耦合外部系统。
   本阶段只输出 report / table / summary，不接 Webhook、调度、外部告警平台。

### 实施注意点

- 不要直接依赖 `profile_stats` / `MarsProfileReport` 作为底层计算入口。
- 要复用其缺失语义，但底层应走共享 helper。

---

## P1-18 金额加权评估 / 件数与金额双视角评估

### 前置依赖

- 建议晚于 `P0` 完整退出后推进。
- 最好建立在 `P0-01` 的共享 `compute` 底座和 `P0-02` 的 `reporting` 边界之上。

### 实施步骤

1. 盘点当前评估指标中哪些适合加金额权重。
2. 统一 `amount_col` / `weight_col` 入口与传递链路。
3. 在 `evaluation` 主流程中补权重版聚合，并优先复用共享 `compute` / metrics helper。
4. 在 summary / detail / trend / bin report 中补齐件数与金额双口径字段。
5. 验证金额权重为空、全 1、真实 exposure 三类场景。
6. 确保输出进入统一结果对象和 `reporting` 链路，而不是只在模板层拼字段。

### 实施注意点

- 这项是进入 `P1` 后优先执行的第一张业务增强单。
- 先做“金额加权评估与双视角输出”，不扩展到 ROI、通过率-收益模拟器或其他业务模拟。

---

## 6. P2 任务详细实施步骤

## P2-01 为 SHAP 报告集成预留统一入口

### 实施步骤

1. 定义 SHAP 结果结构。
2. 抽 explain 子模块。
3. 把 report 消费口标准化。
4. 区分“有 SHAP 依赖”和“无 SHAP 依赖”的行为。

## P2-02 评分卡与模型结果标准化导出

### 实施步骤

1. 统一导出 schema。
2. 优先补 JSON 结构化导出。
3. 明确字段语义和版本。
4. 为外部消费准备最小示例。

## P2-03 为 streaming / out-of-core 建立正式入口

### 实施步骤

1. 明确支持 streaming 的链路白名单。
2. 在关键入口增加显式开关。
3. 记录不支持场景。
4. 补一条大样本验证路径。

## P2-04 做依赖分层与安装瘦身

### 实施步骤

1. 盘点当前依赖分组。
2. 把核心依赖与可选依赖拆开。
3. 更新安装文档。
4. 验证最小安装路径可用。

## P2-05 升级交互式文档展示

### 实施步骤

1. 选择 1 到 2 个高价值 demo。
2. 接入 notebook 渲染。
3. 把报告、监控、Vintage 等示例嵌进文档。
4. 稳定文档构建流程。

## P2-06 为评分卡补充 reason code 输出

### 实施步骤

1. 定义 reason code 结果结构。
2. 从评分卡贡献或规则映射生成原因码。
3. 接入结果对象和报告层。
4. 补最小样例。

## P2-07 为报告增加业务语义层与口径说明卡片

### 实施步骤

1. 定义语义配置结构。
2. 支持字段别名、指标释义、口径注释。
3. 在 report 渲染层接语义信息。
4. 为关键指标增加说明卡片。

## P2-08 增加标准业务切片模板

### 实施步骤

1. 选定首批标准切片模板。
2. 抽出模板定义层。
3. 让 evaluation / report 直接消费模板。
4. 在文档中给出使用示例。

## P2-09 收敛最小可用路径并新增 dry-run / preflight 模式

### 实施步骤

1. 梳理 3 到 4 条最小路径。
2. 在 README 与 docs 中收口示例。
3. 在 pipeline / session 增加 preflight 检查入口。
4. 校验字段、dtype、target、split、backend、依赖。

---

## 7. 建议的逐步执行顺序

如果按真实落地来拆，我建议按下面这个更细的施工顺序推进：

1. `P0-01` 建共享 `compute`，优先抽缺失语义底座并立 Pandas 白名单（已完成）
2. `P0-02` 抽 `reporting`，同时纯化 `utils`（已完成）
3. `P0-03` 收口 backend registry，统一训练 / 预测后端适配（已完成）
4. `P0-04` 做只为前三项服务的最小拆包（已完成）
5. `P1-18` 先补金额加权评估 / 件数与金额双视角
6. `P1-04` 先补测试骨架，保护后续重构
7. `P1-01` sklearn 协议补齐
8. `P1-02` + `P1-07` 做配置治理和全局配置中心
9. `P1-15` + `P1-08` + `P1-10` 一起收口 manifest / schema / versioning
10. `P1-16` 泄漏防护
11. `P1-17` 缺失率异常检测
12. `P1-05` + `P1-12` + `P1-13` 做工程质量、性能基准、异常收口
13. `P1-06` Vintage / Swap / 金额纵向分析
14. `P2` 体验与中期增强项

---

## 8. 备注

### 8.1 关于缺失率异常检测

这个方向在 MARS 中的产品定义应保持克制：

- 它是“数据质量异常检测器”
- 不是“自动裁决特征生死的引擎”

因此它的职责应是：

- 自动发现异常
- 输出结构化结果
- 辅助人工复核

而不是：

- 自动删特征
- 自动阻断建模
- 自动联动外部处置流程

### 8.2 关于 `profile_stats` 与 `MarsDataProfiler`

后续 scanner / detector 不应直接依赖 `profile_stats` 这一高层画像入口。
但它们必须复用从 `MarsDataProfiler` 中抽出的共享缺失语义和缺失统计底座。

这是为了同时满足两件事：

- 避免重复造轮子
- 避免高层画像 API 反向成为所有下游能力的硬依赖
