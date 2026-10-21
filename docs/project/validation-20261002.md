---
description: 2026-10-02 MARS 计算、报告消费与人工展示修复的实际验收及环境边界。
---

# 计算、报告消费与展示验收（2026-10-02）

本记录保存 2026-10-02 任务分支 `codex/mars-contract-fixes-20261002` 的验收结果。
分支从 `e990813fbc4148f138f7cd2025f47637d385c623` 开始，已包含当时评审的 main 提交
`80d93fcdeb8ea17f85740e773c188ff025ee2428`；开工 fetch 后，远端仍为这两个提交。
任务直接在原 HTML 修复提交之后创建分支，保留原独立提交，没有 cherry-pick、merge 或 rebase。

当时使用的仓库为 `D:/Desktop/credit-risk/mars-risk`，父目录中的其他项目和未跟踪文件未纳入变更。
这次任务没有修改 README、版本号、依赖范围和工作流触发条件，也没有创建 PR、发布、tag、合并或部署。

## 逐项行为与回归位置

| 编号 | 原行为与最终契约 | 共享修复与证据 |
| --- | --- | --- |
| A1 | 原来重拟合时会复用旧 WOE 和特征。修复后，每次 fit 重建已学状态；拟合失败后，成果入口拒绝返回旧结果，再次成功拟合后恢复可用。构造配置和用户显式更新的适用规则仍保留。 | `feature/binning/base.py` 的公共 fit 边界；`test_binner_refit_boolean_labels.py` 覆盖三种分箱器、两个后端、标签反转、x→z、类型切换、失败与恢复、配置、规则和诊断。 |
| A2 | 原来 Boolean 全部进入 Other，且 Pandas bool 校验错误。修复后，类别键统一按类型匹配，nullable/null 进入 Missing，未知值进入 Other。字符串大小写、数字字符串和用户同名列仍保留。 | `core/base.py` 与共享类别映射；同一回归文件覆盖 replace/join、JSON、独立进程、三种分箱器、全空 Boolean、普通字符串和临时列。 |
| A3 | 原来多目标分析默认把其他 target 当作特征。修复后，只推断一次特征，并排除全部目标和实际角色列。仍只用首个目标拟合一次，所有目标共用分箱边界，趋势也只针对首个目标。 | `analysis/_risk_profile.py`；`test_profile_risk_multitarget.py` 覆盖 2/3 个目标、benchmark、权重、分组、时间、部分标签、raw KS、显式特征和拟合次数；`multitarget_risk.py` 是受测试覆盖的示例。 |
| A4 | 原来非法标签会产生负 bad 或大于 1 的坏率。修复后，聚合和更新前复用已有二分类规范化逻辑，遇到非法值先报错，避免只更新部分状态。 | 原规范化实现移至已有 `compute/binning.py`，由高层入口和公共分箱表现统计共用。null/NaN/空字符串表示未观测，Boolean 和合法字符串可用。这个接口不支持 -1 哨兵值；合法单类别可计算，全部未观测时明确报错。 |
| B1 | 原来 Monitoring 忽略 sources 筛选，却将条件写入证据。修复后，查询按来源和特征条件的交集筛选；未知来源报错，旧路径缺少可靠来源元数据时明确拒绝查询。 | 适配 Agent 查询旧报告的路径；`agent/test_monitor_sources.py`。没有新增监控能力或上游包装层。 |
| B2 | 通用 not_computed/no_target 被解释为 Score Cross 状态；现在按报告类型、表、字段限定语义。 | `_semantics.py` 和快照读取边界；`test_report_evidence_contract.py` 覆盖原报告、快照、保存/加载、Score Cross/Policy、comparison 和自定义规则定义。 |
| B3 | 原来合法 JSON 中带类型的非有限值被当作操作符对象。修复后，只在筛选入口严格解码已有的 `$mars` float 标签。 | `_query.py` 复用 `_serialization.py`；测试标量/in、±Infinity、NaN/Null 的已有语义，验证普通字符串保持原样、非法标签被拒绝，覆盖两个后端及新进程。 |
| B4 | 原来裁剪列或行后，evidence 仍记录原查询。修复后，columns/limit 与实际展示子集一致，JSON 中的查询和报告标识也计入总预算。 | 现有上下文裁剪路径；回归包含 365 列宽表、非零 offset、多表、严格 JSON、重复调用和准确查询回放。 |
| B5 | 原来只投影数值时会丢失特征字典，关系查询也缺少另一端的身份。修复后，元数据按实际返回页中的特征裁剪；投影未包含身份时补充等长的 identities，裁剪行时同步裁剪身份。 | 同一上下文路径；回归包括多行数值投影、关系两端、空页、排序、来源交集和成员关联查询，不携带全项目字典。 |
| C1 | 原透视工作簿需要 Excel 刷新缓存。已有 snapshot 静态导出可以直接写出当前值，本次补充测试和入口文档。 | `test_static_excel.py` 在两个后端逐表关闭、重开文件，核对数值、中文名、NaN、权重和 KS/IV，并在新进程导出保存的报告。原透视 XML 和 refreshOnLoad 保留；实际用 Excel 刷新、另存并关闭后，20 项缓存数值核对通过。 |
| C2 | 原来默认排序列可能不在实际 schema 中，文本列也可能在应用渐变时被转换为 float。修复后，按实际 schema 选择默认排序列，只给数值列应用数值样式。 | `profile_report.py`、`_profile_excel.py`；`test_profile_presentation.py` 实际执行 Styler.to_html，覆盖分组与不分组、默认与显式排序、不可比较、空值和 nullable，并核对查询、HTML、Excel 的值一致。 |
| C3 | 原来全局搜索与局部搜索的事件互相覆盖 hidden。修复后，按当前两个条件的交集重新筛选。 | `_profile_html.py` 同时服务原报告和快照；DOM 测试与真实 Chrome 脚本分开记录。当时页面只有导航，没有行级分页；验收确认切页和排序后保留筛选。 |
| C4 | 报告索引遗漏 Correlation/ScoreCross，固定安装命令指向尚未发布的版本，容易误导用户。 | 更新已有指南、索引、受测试覆盖的展示示例和 Unreleased 变更记录。安装说明链接当时 PyPI/GitHub 已发布的 0.0.27；源码仍为 0.0.28，未升级版本号。 |

## 环境与命令

当时使用 Windows 11 build 26200，Intel64 Family 6 Model 183。
主环境为 Python 3.12.13、Pandas 3.0.3、Polars 1.42.0、NumPy 2.4.6、PyArrow 24.0.0、optbinning 0.21.0 和 statsmodels 0.14.6。
检查工具为 Ruff 0.16.9、Mypy 1.13.0、pydoclint 0.10.1 和 pytest 9.1.1。
Python 3.8 使用 Pandas 2.0.3/Polars 1.8.2；3.9 使用 Pandas 2.3.3/Polars 1.36.1；
3.10 使用 Pandas 2.3.3/Polars 1.37.1；3.11 使用 Pandas 3.0.3/Polars 1.42.0。

本地统一设置 `PYTHONPATH=src`，日志使用 UTF-8。已有 pytest cache 不可写，因此这次检查
加 `-p no:cacheprovider` 和新的 `--basetemp .pytest-tmp-<batch>`，没有清理用户缓存。
Python 3.10 首次 import `_ctypes` 被沙箱拒绝，补充进程内 PATH 后仍失败。
经授权，在沙箱外运行相同回归测试后通过，没有修改系统 PATH、DLL 或权限。

A 类修复的结束检查批次覆盖新增回归及分箱、风险、评估、batching、selection、correlation、Score Cross、
rule generators、计算、Monitoring、Scoring、replay 和文档，共 437 passed、5 deselected。
Python 3.9/3.10/3.11 只运行新增的两个 A 类测试文件，各 112 passed；这些结果与结束检查批次有重叠，不能累加为独立测试总数。

真实 Optimal 测试中，记录类调用真实 `super().fit()`，确认状态为 OPTIMAL/FEASIBLE，且没有 fallback。
六行小例子和已有测试覆盖 fallback；模拟混合局部失败的 stub 只验证对应分支。
旧 Boolean JSON/pickle 测试通过人工替换旧字段构造载荷，属于模拟历史状态。
B 类修复使用的 `tests/fixtures/legacy_e990813_no_target.marsreport` 则由真实 e990813 源码在独立进程中生成。

最终验证实际执行以下命令；Windows 使用上列环境的绝对解释器路径。

```bash
python -m ruff check src tests scripts docs/snippets --no-cache
python -m mypy src/mars
pydoclint src/mars
python scripts/check_private_docstrings.py src/mars
python -m pytest -q -ra -m "not docs_ml and not optional_ml" -p no:cacheprovider --basetemp .pytest-tmp-final-core312-stable
python -m pytest -q tests/test_documentation.py -m "not docs_ml" -p no:cacheprovider --basetemp .pytest-tmp-C-docs
python -m mkdocs build --strict --site-dir <validation>/docs/site
python scripts/check_release_version.py
python -m build --no-isolation --outdir <validation>/dist
python scripts/verify_distribution.py --dist-dir <validation>/dist
python -m twine check <validation>/dist/*
```

生产源码稳定后，全仓默认依赖测试的结果为 **1018 passed、3 skipped、7 deselected**，耗时 141.94 秒。
Ruff、覆盖 157 个源码文件的 Mypy、pydoclint 和私有 docstring 检查通过，保留 6 条已有提示；版本一致性检查仍为 0.0.28。
文档测试为 43 passed、2 deselected，strict build 通过。

wheel/sdist 各构建一次，资源、元数据和 Twine 检查通过。本地复用已有构建依赖；
远端工作流仍在隔离环境中构建，并将同一个 wheel 交给两个 Python 版本安装测试。

最终版本矩阵的小批次运行以下 7 个测试文件，同样禁用缓存并使用新的 basetemp：
`test_binner_refit_boolean_labels.py test_profile_risk_multitarget.py test_report_evidence_contract.py
test_profile_presentation.py test_profile_html.py test_static_excel.py test_report_query.py`。
Python 3.8/3.9 的整批结果各为 230 passed，最终 HTML 修改后，两版本各补验 4 项，均通过。
Python 3.10/3.11 的原整批结果各为 226 passed、4 failed。新增测试错误地预期 Pandas 一定用 NBSP 表示字面空格；
改为从实际导出单元格读取后，各补验 4 项通过。原失败命令不能记为整批通过。
Python 3.12 的最终完整批次覆盖当时全部核心测试。各批次有重叠，不能相加成独立测试总数。

本地安装了 statsmodels 和真实 Optimal 求解器，缺少 XGBoost/LightGBM/CatBoost/Optuna。
3 个 skip 来自 modeling 模块缺少 XGBoost，7 个 deselected 由 docs_ml/optional_ml 标记排除。
本地未执行完整可选 ML 检查和两个 docs_ml 集成示例，远端 modeling 结果另行核验。
当时的任务分支不触发 Docs 工作流，因此不能宣称 docs_ml 已在远端执行。

首次全仓检查中，3 个 private-docstrings 测试将子进程的 UTF-8 输出按 Windows GBK 解码，导致失败。
设置进程级 `PYTHONUTF8=1` 后，5 项定向测试通过，最终全仓检查也通过。
一次附加的全仓收集遇到任务临时目录的 ACL 错误；将临时产物移至验证目录后，最终默认命令通过，没有修改 ACL 或增加排除标记。
这些失败及补跑日志保留在工作区外的任务验证目录，没有修改无关测试来消除失败。

## 持久化与人工交付边界

`.marsreport` 格式仍为 1，没有引入新的序列化系统。独立进程只读取已保存文件，不读取原始宽表或重新拟合。
回归核对报告标识、describe 语义、sources、投影后的特征字典、预算和合法 JSON 的准确回放。
原统计、权重和 KS 定义不变；不可比较或未计算的状态不会被转换为有效零值。

`from mars.reporting import snapshot_report; snapshot_report(report).write_excel(path)`
是已有的静态 Excel 导出入口，恢复后的 `load_report(path).write_excel(...)` 同样导出静态值。
使用 openpyxl `data_only=True` 关闭、重开文件后，数值核对通过，确认无需 Excel 刷新即可读取本次数值。
静态读取与原生刷新分别验收。当时会话最初的 Cua 入口禁用 native 操作；随后发现已安装的
Computer Use `node_repl + @oai/sky` Windows 入口可用。首次启动返回
`Computer Use app approval timed out`，用户明确允许重试并通过应用授权后成功启动。
第一次文件对话框曾出现索引失效和 foreground process id 错误。恢复窗口后，通过键盘
打开专用的确定性透视工作簿，按 Ctrl+Alt+F5 全部刷新，按 F12 另存，然后关闭工作簿。
没有使用 COM 或修改 Office 安全设置；第三方 OfficePLUS 推广浮窗没有参与验收。

原生 Excel 版本为 16.0.20430.20092。8 行数据包含中文类别、Missing 和权重，
通过公共 Native fit/update_bins 和 MarsBinEvaluator 生成报告；刷新前的缓存验证按预期失败。
用 Excel 刷新保存后，独立 Python 进程以 openpyxl `data_only=True` 读取
`<validation>/native-excel/mars_native_refreshed.xlsx`，按 feature 逐项核对综合指标及 KS/IV/AUC/PSI/Missing 页。
20 项全部通过，关闭工作簿后重读，仍有 20 项通过。

其中 income/category 的 KS 分别为 99.999975 和 85.714263，Missing 均为 9.090909%。
生成脚本、expected.json、原始及刷新后的工作簿、verification.json 保留在任务验证目录，
没有将 Excel 文件或截图提交到源码。实际命令为：

```bash
python <validation>/native-excel/generate_native_excel.py
python <validation>/native-excel/verify_native_excel.py --workbook <validation>/native-excel/mars_native_pivot.xlsx
python <validation>/native-excel/verify_native_excel.py --workbook <validation>/native-excel/mars_native_refreshed.xlsx
```

真实浏览器验收使用已有的 Playwright 1.63.0 与本机 Chrome 154.0.8037.95，采用 headless 模式和 file://。
`tests/browser/profile_search.py --output <validation>/c-profile-browser-final-v2 --channel chrome`
实际导出当时的原始报告和从文件恢复的快照，在 1440×1000、390×1000 两种视口中共验收 4 个页面。
每页执行 36 次操作，涵盖搜索、清空、重复输入、大小写、中文、空格、局部隔离、排序和导航，并核对 16 个数值字段。
pageerror、console_error、CSP 和失败请求均为 0，没有外部请求，页面根宽不超出视口。
验收时实际查看了本次 empty、local-cleared 和 final 截图，确认空交集没有数据行，清除局部条件后仍保留全局条件；
窄屏表格在自身容器中横向滚动，搜索和导航控件可见。当时的 HTML 模板 SHA256 为
`ab7de5a5996a33b32340f6680fb7efa03a046039854e27c915c1c76fe3c6ab69`。
DOM 测试另在 Node VM 中运行，不计为视觉验收。检查源码时发现，hidden 会改变 innerText 中空格的表现；
修复后以单元格 textContent 和 tab 稳定拼接，保留字面空格，不 trim，也不新增模糊匹配。
新脚本最初名为 profile.py，曾遮蔽 stdlib 并导致相邻浏览器脚本失败，改名后已重新验证。

`tests/browser/score_cross_time.py --output <validation>/score-time-final --channel chrome`
全部 16 个夹具通过；原 e990813 的 Score Cross HTML 源码字节未改。
`tests/browser/docs.py --site-dir <validation>/docs/site --output <validation>/docs-browser --channel chrome`
12 种视口和主题状态通过，验收时已查看移动端指南截图。截图和 JSON 日志未纳入源码提交。

报告加载与查询的性能验收复用 `benchmarks/benchmark_report_consumption.py`，分别通过 PYTHONPATH 导入
固定的 e990813 归档源码和修复后的源码。两次测量使用同一台机器、相同依赖和无随机种子的确定性数据。
overview 为 10,000 行，趋势为 1 行 × 365 个日期列，offset 为 9000、limit 为 10；查询取 10 次中位数，save/load 各测 3 次。

| 操作 | e990813 中位耗时（ms） | 修复后中位耗时（ms） | Python 分配峰值（前→后） |
| --- | ---: | ---: | --- |
| 分页 | 0.310 | 0.321 | 11811→11811 bytes |
| 16,000 字符上下文 | 25.611 | 27.093 | 1,900,371→1,900,799 bytes |
| 5000 字符上下文 | 185.756 | 34.824 | 1,900,077→1,909,165 bytes |
| 保存 | 88.869 | 88.084 | 2,519,533→2,520,758 bytes |
| 加载 | 63.235 | 60.590 | 4,437,879→4,437,722 bytes |

5000 字符预算下，实际输出由 4985 变为 4975 字符，字段由 93 个变为 66 个。
少展示 27 个字段，是因为有效 query.columns 也计入预算，属于兼容性变化。
初版逐列裁剪曾耗时 241.69 ms；后来沿用已有裁剪逻辑，按稳定前缀二分搜索，复测得到上表。

文件均为 587,098 bytes。tracemalloc 只测量 Python 分配，不包含原生 RSS。
这次单机微测量不能证明拟合、整体容量或普遍加速；原始 JSON 位于任务验证目录 report-contract-before/after.json。

## 远端验证边界

首推 `027942f844380f2b3f32734e38b0f74c87717eb0` 的 CI
[36987554951](https://github.com/leeesq/mars-risk/actions/runs/36987554951)
终态为 failure。其余 10 项全部成功，quality 只在 Ruff I001 失败，后续静态检查未执行。
新增的历史文件目录 `tests/fixtures` 与浏览器脚本同级的 `fixtures.py` 重名，CI 上的 Ruff 0.16.10
将该 import 归为 first-party。本地 Windows Ruff 0.16.9 禁用缓存后也复现了失败；先前使用缓存的检查
没有发现新增目录导致的归类变化，因此不能将差异归因于平台或工具版本。

修复在已有 isort 配置中，将两个自有浏览器模块 `fixtures`、`score_cross` 明确归为 first-party，
并重新排列 time/contrast 两个报告查询脚本的 imports。包版本、依赖、运行逻辑和检查项均未修改。
修正后，禁用缓存的全仓 Ruff、Mypy、pydoclint、私有 docstring 与 strict docs build 均通过，
16 个真实浏览器周期夹具也通过。对比度脚本使用已有 standard 快照，并由当时修复后的代码重新导出，
参数为 `--narrow --copy-feedback --hover-controls --channel chrome`，共 23 个状态、0 失败，
普通文字的最小对比度为 4.5047。HTML 源码没有改变，验收时已查看 hover 截图。原始结果保存至
任务验证目录 `score-time-ci-fix`、`contrast-ci-fix/contrast.json`。
当时随后重新推送，并核验新 SHA 的全部 CI；最终状态在该任务的交付回复中提供。

修正提交 `b451781726f8e09827065c73f252738ea91037f7` 的 CI
[36988991823](https://github.com/leeesq/mars-risk/actions/runs/36988991823)
已核验终态为 success，全部 11 项成功。记录写到这里时，补充原生 Excel 实测结果的最后一次文档提交
尚待核验完整 SHA；当时要求最终回复只引用最终 SHA 的检查结果。

当时的推送范围仅限已核实的任务分支和 origin，没有 force push；远端完整 SHA 及 CI 终态随该任务的最终回复核验。
CI push 覆盖 quality、core 3.10～3.12、legacy 3.8/3.9、modeling、单次 distribution、
同一 wheel 的 installed smoke 3.8/3.12 和 rule performance smoke。
Docs 只在 main push、面向 main 的 PR 或手动 dispatch 时触发；这次任务没有创建 PR，
也没有 dispatch 带 deploy 的工作流。文档在本地通过 strict build 和真实浏览器验收，
历史 Docs 成功结果不计为这次提交的检查结果。
