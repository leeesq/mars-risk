---
description: 查询、展示、导出与保存公共分析报告；保留评分卡旧章节入口。
---

# 报告查询与导出

任务型入口：[案例 6：保存后继续查询](../demos/saved-reports.md) ·
[案例 7：同次分析的多种交付](../demos/report-delivery.md)。

外部 Agent、业务元数据、证据与跨会话使用见[外部 Agent 指南](external-agents.md)。
评分卡另见[评分卡](scorecard.md)，本页旧章节及锚点继续保留。

!!! info "Reporting：Stable"

    本页的结构化 report 读取和 Excel/HTML 导出属于 Stable Reporting 能力。评分卡能力的状态
    单独标注在对应章节。

## 适用场景

Report 用于继续筛选、复盘和组合计算；Excel/HTML 用于归档或人工交付；Scorecard 将已拟合分箱规则
与逻辑回归系数转换为评分映射和 SQL。

## 1. 获得 Report

下面的受测试示例定义了 `report`，后续导出调用均基于该对象：

```python
--8<-- "docs/snippets/quickstart.py"
```

常见对象与字段：

| Report | 状态 | 高价值字段 |
| --- | --- | --- |
| `MarsProfileReport` | Stable | `overview_table`、`dq_tables`、`stats_tables`、`comparison_tables`、`report_meta` |
| `MarsBinningReport` | Stable | `summary_table`、`detail_table`、`trend_tables` |
| `CorrelationReport` | Stable | `features`、`pairs`、`correlation_decisions` 等公共表；[相关性指南](correlation-and-score-cross.md#相关性报告) |
| `ScoreCrossReport`（`mars.analysis`） | Stable | `cells`、边际、`overall`、`bins`；[模型分交叉指南](correlation-and-score-cross.md#固定分段交叉) |
| `MarsRuleReport` | Experimental | `summary_table`、`detail_tables`、`metadata`；公共查询、桥接关联、保存恢复 |
| `MarsMonitoringReport` | Experimental | 监控汇总、分箱统计、表现覆盖率和元数据 |
| `MarsModelingReport` | Experimental | 多样本切片的汇总、明细、趋势和元数据 |

## 查询已有报告和交给 AI

画像、分箱、相关性与模型分交叉报告提供 Stable `describe()`、`get_table()`、`to_ai_context()` 和 `get_feature()`。
这些方法只查询已计算结果，不调用 LLM、不重新计算统计，也不修改原报告。

规则报告也满足公共 Report 契约，summary 为挖掘级汇总；成员特征查询通过轻量桥接，
不复制规则指标。候选、切片、显式高级分析与 no_rules 状态说明见
[规则报告与外部 Agent](rule-reports-and-agents.md)。

```python
--8<-- "docs/snippets/report_queries.py"
```

`describe()` 返回表粒度、行数、字段类型、指标单位、实际参数和限制。画像表名为 `overview`、
`dq.<metric>`、`stats.<metric>`、`comparison.<metric>`；分箱表名为 `summary`、`detail`、
`trend.<metric>`，可选附表为 `missing_by_day` 和 `risk_corr_reference`。没有计算的表不在目录中。

`get_table()` 保持所选表的 Pandas/Polars 类型。支持 `features`、`columns`、`filters`、
`sort_by`、`descending`、`offset`、`limit` 和 `sources`。筛选是字段到标量相等条件，或
`{"op": "gt", "value": 0.1}`；操作符只允许 eq/ne/lt/le/gt/ge/in/not_in/is_null/is_not_null。
没有 SQL 或表达式执行入口。排序在列投影之前，排序配合 `limit` 就是 Top-K。
`sources` 使用统一 `feature_metadata` 中的来源，旧来源参数通过薄适配合并；未知来源会报错。

画像/分箱 AI JSON 默认只含 overview/summary 前 10 行；规则报告默认 summary 一行和最多三条
最终验证证据，不默认输出 expression。预算为 16000 个 Unicode 字符，不是 token 数。
可以按表、特征和列缩小范围。省略的参数和行有明确引用及原因；预算连说明都容不下时抛
`ValueError`，不会输出无效 JSON。日期使用 ISO-8601；浮点非有限值使用带类型的 `$mars` 标记；
Null 使用 JSON null，0 保留数值。Null 的业务原因需要结合标签状态、诊断与计算参数解释。
KS 是 0–100 百分制指标，缺失率和坏率是 0–1 比例，IV/PSI 是无量纲数值；币种、标签定义和观察
窗口未登记时标为 unknown。比较前核对拟合来源、参考、权重、箱范围和指标排序；JSON 不生成归因结论。

`get_feature("income", limit=100)` 返回关联表、每表省略行数和 unavailable 类别。
`show_overview`、`show_summary`、`show_trend` 也支持 `columns`、`limit`、`sources`，复用同一查询。
这些旧展示方法保持不传 limit 时的历史默认；新增 get_feature 与 AI 上下文有默认规模上限。

## 2. 导出 Excel 或 HTML

以下代码继续使用上一步定义的 `report`：

```python
report.write_excel("risk_report.xlsx", engine="openpyxl")
report.write_html(
    "risk_report.html",
    report_name="Current-period risk review",
    max_plots=100,
    chart_embed_mode="auto",
)
```

| HTML 模式 | 行为 |
| --- | --- |
| `auto` | 小报告内嵌图片，大报告生成同级资产目录并懒加载 |
| `inline` | 所有图片内嵌，适合必须单文件离线转发的报告 |
| `asset` | 强制使用相对路径图片目录，适合大报告归档 |

风险趋势图和 HTML Charts 需要评估阶段已经提供有效 `time_col`。`group_col` 不能替代日期范围。

`MarsProfileReport.write_html()` 始终生成无外部资源的单文件，包含 Metadata、Overview、
DQ、Stats 和 Comparisons 页面；它不生成图表。所有 Stable 报告导出已改为严格失败：
资源缺失、请求内容未生成或写入失败都会抛异常，不再只记日志后返回成功。
全局搜索与每张表的局部搜索按交集生效；清空其中一个仍保留另一个条件。
局部搜索只影响所属表，排序和页面切换保留条件。当前是页面导航，没有行级分页。

分箱报告的原始 Excel 入口保留透视模板；缓存可能仍是模板占位值，需要在原生 Excel 中
刷新透视表并保存。`openpyxl.load_workbook(..., data_only=True)` 不会执行这个刷新。
向人工用户交付本次当前数值时，使用已有静态快照入口：

```python
from mars.reporting import snapshot_report, load_report

snapshot_report(report).write_excel("risk_static.xlsx")
report.save("risk.marsreport")
# 另一进程只持有保存文件时，也能直接静态导出。
restored = load_report("risk.marsreport")
restored.write_excel("risk_restored_static.xlsx")
```

静态工作簿逐公共表写出当前值，不依赖透视缓存；它不提供原模板的可刷新透视交互。
最短完整示例同时包含保存、加载和实际 Notebook 比较表渲染：

```python
--8<-- "docs/snippets/report_presentations.py"
```

## 3. 单独复用趋势图

继续使用上一步定义的 `report`：

```python
figures = report.build_risk_trend_figures(features=["income"])
fragment = report.render_risk_trends_html(
    features=["income"],
    image_format="svg",
    embed_mode="inline",
)
```

`fragment.html` 是可嵌入现有模板的 HTML 片段；资产模式同时返回已写入的图片路径。

## 4. 构建评分卡

评分卡为 Experimental，暂时停止功能迭代；现有转换与部署能力继续保留。
必要的正确性、运行修复与上游适配继续处理，规则见[稳定性](../project/stability.md)。

!!! warning "Scoring：Experimental"

    评分映射、刻度参数和 SQL 输出仍可能调整。当前 0.0.28 是源码版本，按
    [安装指南](../getting-started/installation.md)安装并固定核验过的源码提交；正式发布后再固定
    对应 PyPI 版本，并为 `points_table` 和生成 SQL 增加契约测试。

```python
--8<-- "docs/snippets/reporting_scorecard.py"
```

评分卡要求分箱器已经使用 target 拟合并具备 WOE 映射。系数字典的特征必须与分箱器规则一致。

## 常见失败

- HTML 图表为空：确认生成 report 时传入了有效 `time_col`。
- 大报告单文件打开缓慢：使用 `auto` 或 `asset`，不要强制内嵌数百张图片。
- 评分卡提示缺少映射：确认 binner 已拟合、特征名一致且包含 WOE 统计。

## 下一步

- 理解 report 与 artifact 的边界：[Report 与 Artifact](../concepts/reports-and-artifacts.md)。
- 查询导出对象：[Reporting API](../reference/reporting.md)。
- 查询评分卡签名：[Scoring API](../reference/scoring.md)。

## 可携带的公共分析报告 { #portable-analysis-reports }

**状态：Stable。** 画像与分箱报告实现 `mars.reporting.Report` 契约。外部建模 Agent、诊断归因
Agent 可以直接调用 `describe()`、`search_features()`、`get_table()`、`query_page()`、
`get_feature()`、`to_ai_context()` 和 `save()`，无需 `MarsAgentSession`、Notebook 或原始宽表。

完整可运行示例（包括新 Python 进程恢复、两种 Agent 消费方式及 HTML/Excel 导出）：

```bash
python docs/snippets/portable_reports.py
```

```python
--8<-- "docs/snippets/portable_reports.py"
```

### 特征元数据与业务上下文

`profile_stats`、`MarsDataProfiler.generate_profile`、`profile_risk`、`MarsBinEvaluator.evaluate`、
`MarsStatsSelector.fit/fit_transform` 和两个报告构造器都接受 `feature_metadata`、
`business_context`。内部 Agent 的 `register_dataset` 也可登记这些信息。

特征字典以**原始英文字段名**为稳定键，支持 `display_name/description/data_source/unit` 四个
可选字符串字段，缺失值允许 null。Pandas/Polars DataFrame 必须有 `feature` 列，其余列对应上述
字段；重复记录、非字符串值和未知字段报错。字典可覆盖整个项目，报告按实际分析特征裁剪。
重复中文名合法，检索返回全部候选及英文标识；`features=` 和 `get_feature` 仍只查询原始标识。

缺少显示名时，人类展示回退英文名，元数据中仍保留“未提供”的事实。中文名称按展示查询后的
小表附加，不重命名原始输入和统计表，也不把业务定义重复到每个日期或箱。来源旧参数在分析/
selector 中为“来源 → 特征列表”，在分箱报告构造器中为“特征 → 来源”。两者归一到元数据；
与新字典来源相同则合并，不同则报错。旧参数仍要求仅包含 active features；全项目字典请改用
`feature_metadata`。`UNMAPPED` 仅为旧统计表展示标记，不会作为已知业务来源写入元数据。

业务上下文是字符串键的 JSON 对象，支持字符串、布尔、数值、null、嵌套字典、列表和 ISO 可编码
的 date/datetime；不支持 tuple、集合、自定义对象或可执行 Python 对象。`$mars` 是保留键。
多标签定义使用 `labels={target: {definition, positive_class, negative_class, performance_window}}`；
`sample` 描述范围/筛选/时间区间，`splits` 描述 train/val/test/oot，`currency` 和
`score_direction` 保存调用方解释。其他 JSON 业务字段也可保存。不提供的信息保持 unknown；
不根据字段名推测业务定义。`describe().context_source` 区分用户信息与实际计算参数，
`parameters` 记录真正执行的权重、排序、分箱、基准和诊断；用户上下文不改变计算公式。
`unit` 是特征值单位，不是 KS/IV/PSI 的指标单位。

### 分页查询、证据与上下文预算

`get_table` 保留原生 DataFrame 类型；筛选、排序、分页后返回独立容器。支持原标识、
来源、列投影、eq/ne/lt/le/gt/ge/in/not_in/is_null/is_not_null 筛选、稳定排序、offset、limit。
日期/分组宽表范围通过 columns 选择，长明细通过 filters 选择。真实日期类型列支持 ISO-8601
字符串筛选，使 JSON 证据查询可直接重放；普通字符串列不进行日期推断。Pandas 索引仍保留，但不作为
查询列；需要筛选索引时先在自己的消费代码中处理。

`query_page` 使用相同参数，额外返回 `total_rows/returned_rows/omitted_rows/truncated`、
省略原因、`next_offset` 与 `reference={report_id, table, query}`。排序、投影和分页条件
可在保存恢复后重放。持久 report_id 是分析产物身份；会话登记返回的 report_id 参数实际是
会话句柄，Agent 工具另外提供 `persistent_report_id/evidence_reference`。外部报告如实记录
原 source，dataset_id 为 None；登记后修改原报告数据或元信息不影响已登记快照。

`to_ai_context(queries={table: get_table_options}, max_chars=16000)` 支持不同表使用不同条件；
旧的 features/columns/filters/limit 等共同参数会适配到相同查询路径。摘要不能替代完整文件。
桥接关系的 key 与 feature 可以指向同一列；原报告和保存后的快照均可筛选、分页和生成
AI 证据，统计行不会按成员展开。有限上下文关联保留实际返回页涉及的成员和关系两端；
报告声明的 `feature_scope` 用于解释实际返回的汇总行，不附带未返回特征的字典。
窄投影省略身份列时，`evidence.identities` 与 `rows` 一一对应，保留既有身份字段；裁行同步裁身份。
宽趋势的指标只定义一次，日期/分组以原始列标识为维度，证据保持与原表对应的紧凑宽行。
`evidence.query` 是实际展示子集的有效查询；预算裁剪后 `columns/limit` 同步缩减，
可直接传回 `get_table` 重建展示行和列。原筛选后的 `total_rows` 与省略说明保留。
非有限值筛选使用既有 `$mars` float 标记，标量与 `in` 列表均可 JSON 回放；畸形标签明确报错，
普通字符串不转换，NaN 相等和空判断沿用各后端现有规则。
预算涵盖整个最终 JSON 的 Unicode 字符，包含查询与身份，**不是 token 数**。超预算依次裁剪完整时间列、行及
说明块，省略记录包含数量、原因和 describe/get_table 定位。极小预算容不下必要身份及引用时
报 ValueError。未选择特征的项目字典不会默认注入上下文。

### 单文件格式与恢复 { #snapshot-format }

`.marsreport` 是 ZIP，格式版本 **1**：`manifest.json` 保存持久身份、报告类型、全部表目录、
粒度、字段类型、指标定义、参数、上下文、元数据、状态/诊断及来源；`tables/0000.parquet` 等
文件保存每张完整统计表，清单把编号路径映射回公共表名。它不是 pickle，不反序列化代码，
外部程序可用 ZIP、JSON、Parquet 标准库读取。表顺序和列顺序明确保留。

保存按表处理，不把整个报告转换为 Pandas 或大 JSON。父目录需存在；默认拒绝覆盖，
`overwrite=True` 原子替换。写入在同目录临时文件完成后安装；失败清理临时文件，
不改变已有完整文件。无效 ZIP、清单、缺失表、schema/顺序/行数不一致和非 1 版本明确报错；
当前没有历史版本迁移器。

清单记录各表后端及版本。跨后端版本若不能保留声明的 dtype，会明确拒绝恢复；
尚未承诺任意 Pandas/Polars 版本组合之间的 schema 迁移。格式仍可由标准 Parquet 工具读取。

恢复返回 `ReportSnapshot`，保留原报告类型、身份、目录语义和每张表的后端，无需原始宽表、
分箱器或会话，也不会重新计算。`show_table` 可查看中文名与英文标识；
`write_html/write_excel` 导出全部公共表。原报告的 HTML/Excel 继续使用现有展示方式，并集中新增
`FeatureMetadata/BusinessContext/ReportSemantics` 信息。Excel 是人类展示产物，
需要无损复用统计值时使用 marsreport 文件。

Pandas 索引（包括 MultiIndex）、类别、日期和时区依赖 Arrow pandas schema 元数据保留；
Polars 使用原生 Parquet 类型。清单记录规则并验证恢复 schema。表列必须是唯一字符串名，
无法编码的对象报错。Parquet 区分 null、浮点 NaN、±Infinity 和零；若原 Pandas 浮点列已经将
缺失折叠为 NaN，恢复不会伪造二者之间的区分。日期 JSON 使用 ISO-8601，date 与 datetime
依其实际 DataFrame 类型分别输出日期或完整时间戳。

### 状态与迁移

JSON 中的浮点非有限值统一为 `{"$mars":"float","value":"nan"}`、
`{"$mars":"float","value":"inf"}` 或 `{"$mars":"float","value":"-inf"}`；null 表示缺失值，
零仍是数值零，普通字符串 `"NaN"/"Infinity"` 不转换。描述、AI 上下文、Agent 工具及文件清单
共用此规则。原消费者需停止把报告字符串标记或 Agent null 当作统一非有限数编码。

`calculation_status` 按目标/特征/分组/指标族保存计算状态与原因；无标签为 not_computed，
标签存在但全未表现为 unobserved。单箱、正常箱样本不足或坏率恒定时 mono 未定义，
不再把 NaN/null 填为 1.0。使用旧占位筛选单调性的代码应检查状态后处理缺失。
IV、PSI 和原始/分箱 KS 公式保持现有口径；KS 为 0–100 点数，坏率为 0–1 比例，
count 字段根据实际 weights_col 标明原始样本数或权重和。类别原始 KS 回退和失败仍见
`ks_source_by_feature/raw_ks_diagnostics`；分箱失败/跳过继续见已有 fit/diagnostics 信息。

画像的状态表只记录全量统计例外（样本不足、非数值字段跳过、未定义值及已有 PSI 失败诊断），
不生成每个趋势单元格的状态对象；缺少状态行不应代替核对实际值和诊断。
HTML/Excel 也集中导出该状态表；原报告 Excel 工作表名为 `CalculationStatus`。
