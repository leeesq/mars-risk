---
description: MARS 0.0.26 的用户可见变化、兼容性说明和升级检查项。
---

# Release Notes

## 0.0.26

发布依赖补充 `Jinja2>=3.1.2`，确保默认安装即可使用基础报告、特征筛选器和
`Pandas Styler` 展示接口；Python 3.8 冻结栈使用 Jinja2 3.1.6 与 MarkupSafe 2.1.5。

该版本将基础包的运行范围扩展到 Python 3.8–3.12，并对 Analysis、Feature 和
Reporting Stable API 执行 fail-closed 收口。本版本包含明确的 API、报告和序列化 breaking changes，
升级前必须按本页的迁移清单核对。

### Python 与依赖兼容

- Python 3.8 固定使用 Polars 1.8.2，并将 scikit-learn 限制在 1.3.x；仓库通过
  `constraints/python38.txt` 固定验证栈。Windows 环境固定 OSQP 1.0.4，避免旧 0.6.x
  在 Polars 已加载后导入时的原生库崩溃。
- Python 3.9 使用现代 Polars 与 scikit-learn 1.6.x；Python 3.10–3.12 延续当前现代依赖。
- Python 3.8 已停止官方安全维护。MARS 的兼容承诺仅表示冻结栈可以运行，不延长解释器的
  安全支持周期。
- `ml`、`tuning`、`notebook`、`docs` 和 `dev` extras 要求 Python 3.10+；Python 3.6、3.7
  以及 3.13+ 不在本版本支持范围内。

### 实现与结果口径

- 新增内部 Polars 兼容层，统一 membership 与 streaming collect 的跨版本差异。
- KS/AUC 的前一累计分布改为“当前累计值减当前箱分布”，以兼容 Polars 1.8 的窗口表达式
  限制；指标结果与现代 Polars 保持一致。
- Python 3.8 语言兼容改造本身只涉及注解求值、dataclass、zip 和字符串后缀处理，
  不改变业务算法；本版本的 Stable API 与 artifact 变更单独列在下文。
- 特征筛选的监督指标和 WOE 相关性只使用 target 非空的已表现样本；质量、分布和 PSI 仍
  使用全量样本。
- Optimal Binner 的失败特征统一批量回退到 Native Binner，避免随失败特征数增长的重复拟合。
- `profile_stats()` 与 `MarsDataProfiler.generate_profile()` 新增 `benchmark_df`：基准样本只负责
  PSI 分箱和 expected distribution，不进入当前数据的质量与统计指标；未分组时可直接输出
  当前全量相对 benchmark 的 `total` PSI。
- 数据画像删除 `sample_frac` 参数。抽样改由调用方在传入前显式完成；仍传该参数的旧调用会
  收到 Python 标准 `TypeError`，升级时应删除参数并在外部准备抽样 DataFrame。

### Stable API 与报告加固

- Binner `transform()` 新增 `features` 和 `on_missing`，默认要求全部规则列齐全；Selector
  `transform()` 使用相同的严格缺列策略。`update_bins()`、`prune()` 和 `get_bin_mapping()`
  不再静默忽略未知特征。
- 三种 Binner 新增固定 schema 的 `get_fit_report()`。合法 fallback 可继续，真正无规则的
  特征标为 `failed`，全部失败终止。
- `to_dict()` / `from_dict()` 改为 `schema_version=1` 的自描述 artifact，新增 `save_json()`
  和 `MarsBinnerBase.load_json()`。旧 `{params, state}` 载荷不兼容，必须重新拟合或导出。
- WOE transform/SQL 必须具备完整 WOE 映射；SQL 类别值使用安全引号转义。
- 报告级指标列缺失会终止；单特征空值、NaN 或 Inf 指标以 `metric_unavailable`
  淘汰并记录。`MarsImportanceSelector` 删除未实现的 `rfe` / `sfm` 公开选项。
- Excel、HTML、JSON 写入、资源读取和空报告导出失败现在会显式抛异常；Binning
  HTML 如需图表，任一图表构建失败会使导出失败，可显式使用 `include_charts=False`。

### 画像对比能力

- `profile_stats()` 和 `generate_profile()` 新增 `categorical_features`，使整数编码类别同时进入
  unseen 与类别 PSI 口径。
- 新增显式 `schema` 和 `unseen` metrics，不加入默认指标。Schema 表区分两侧列存在性、
  兼容和不兼容 dtype 变化；unseen 排除缺失与特殊值，输出 total 与分组趋势。
- `MarsProfileReport` 新增 `comparison_tables`、`report_meta` 和自包含交互式 `write_html()`。
  Profile Excel 新增 Metadata 和 comparison 工作表。
- `ProfileData` 从三字段扩展为四字段，位置解包调用需增加 `comparisons`。

### 发布与工程门禁

- 普通 CI 和 Release 均只构建一次 wheel/sdist，静态核对版本、Python 范围、依赖 marker、
  `py.typed`、Excel 模板和 dist-info，再运行 `twine check`。
- 同一 wheel 会在 Python 3.8 与 3.12 全新环境中独立安装，并验证递归导入、`profile_risk`、
  selector、Pandas Styler、Excel/HTML 报告和安装后的模板资源；发布 job 不再重新构建。
- Mypy 固定为 1.13.0，并以 Python 3.8 为统一目标检查全部 `src/mars`；业务模块 override 与
  源码 `type: ignore` 已清零，第三方动态边界通过显式类型收窄处理。

### 升级检查

- Python 3.8 环境按约束文件重建，不要在已有环境中强制覆盖整套依赖。
- 使用可选建模或文档依赖时升级到 Python 3.10+。
- 使用数据画像内部抽样的调用，改为先对 DataFrame 显式抽样，再调用 `profile_stats()` 或
  `generate_profile()`。
- 将 Binner 旧 dict/JSON 载荷全部用 0.0.26 重新拟合或 `save_json()` 导出。
- 将依赖缺列静默忽略的 Binner/Selector 调用改为显式 `features` 或 `on_missing`。
- 将 `ProfileData` 的三元素解包改为四元素，并移除 Importance Selector 的 `rfe` / `sfm` 配置。
- 发布前同时验证 Python 3.8 冻结栈、Python 3.9 依赖边界和 Python 3.10–3.12 现代栈。

## 0.0.24

从当前公开版本 `0.0.21` 升级到 `0.0.24` 时，重点核对分析报告链路、基准样本语义、
趋势图时间范围、HTML 大报告和 Modeling/Pipeline 契约。

### 用户可见变化

- `profile_risk()` 返回 `MarsRiskProfile`，同时提供 `report`、`binner`、`targets` 和 `metadata`。
- `benchmark_df` 统一用于基准期分箱和 PSI expected distribution，不进入当前期 Total。
- `MarsStatsSelector.fit()` 支持 `benchmark_df`，筛选指标仍在当前 `df` 上计算。
- 风险趋势图的时间范围只来自有效 `time_col`；`group_col` 只负责面板分组。
- HTML 报告支持可检索视图、图表数量控制、图片资产模式和懒加载。
- Modeling/Pipeline 增加结果对象、replay、artifact 和多 target 评估能力，状态仍为 Experimental。

### 升级检查

- 将旧代码中直接假定 `profile_risk()` 返回 report 的访问改为 `risk_profile.report`。
- 使用固定分箱规则时改用 `MarsBinEvaluator.evaluate(..., binner=...)`。
- 生成趋势图或 Charts HTML 前显式提供有效 `time_col`。
- 使用 `MarsStatsSelector` 默认 PSI/RC 阈值时提供 `group_col` 或 `time_col`；静态筛选应显式关闭
  对应阈值。
- 升级 Modeling/Pipeline 后重新核对结果字段和 artifact 路径。

!!! note "发布状态"

    本页随 `0.0.26` 源码维护。版本发布前，`main` 站点内容仅作为预览；PyPI 发布和 release tag
    通过版本一致性检查后，安装命令才表示正式可用。
