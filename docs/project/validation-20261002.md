---
description: 2026-10-02 MARS 计算、报告消费与人工展示修复的实际验收及环境边界。
---

# 计算、报告消费与展示验收（2026-10-02）

本记录对应任务分支 `codex/mars-contract-fixes-20261002`。起点是
`e990813fbc4148f138f7cd2025f47637d385c623`，已包含评审 main
`80d93fcdeb8ea17f85740e773c188ff025ee2428`。开工 fetch 后远端仍为这两个提交。
任务在原 HTML 修复提交之后直接建分支，保留其独立提交，没有 cherry-pick、merge 或 rebase。
实际仓库为 `D:/Desktop/credit-risk/mars-risk`；父目录其他项目和未跟踪文件没有纳入变更。
README、版本号、依赖范围、工作流触发条件未改；没有 PR、发布、tag、合并或部署。

## 逐项行为与回归位置

| 编号 | 原行为与最终契约 | 共享修复与证据 |
| --- | --- | --- |
| A1 | 重拟合复用旧 WOE/特征；现在每次重建已学状态，失败后成果入口拒绝旧结果，恢复拟合可用。保留构造配置与用户显式更新的适用规则。 | `feature/binning/base.py` 的公共 fit 边界；`test_binner_refit_boolean_labels.py` 覆盖三种分箱器、两后端、标签反转、x→z、类型切换、失败/恢复、配置/规则/诊断。 |
| A2 | Boolean 全进入 Other，Pandas bool 校验错误；现在类型意识的类别键一致，nullable/null 进 Missing，未知进 Other。字符串大小写、数字字符串和用户同名列保留。 | `core/base.py` 与共享类别映射；同一回归文件覆盖 replace/join、JSON、独立进程、三分箱器、全空 Boolean、普通字符串、临时列。 |
| A3 | 多目标默认把其他 target 当特征；现在只推断一次并排除全部目标和实际角色列；仍由首目标拟合一次，各目标共用边界，趋势仍仅首目标。 | `analysis/_risk_profile.py`；`test_profile_risk_multitarget.py` 覆盖2/3目标、benchmark、权重/分组/时间、部分标签、raw KS、显式特征与拟合次数；`multitarget_risk.py` 是受测试示例。 |
| A4 | 非法标签生成负 bad 或坏率>1；现在聚合/更新前复用已有二分类规范化，非法值先报错且不半更新。 | 原规范化实现移至已有 `compute/binning.py`，高层和公共分箱表现共享；null/NaN/空字符串未观察，Boolean/合法字符串可用。该具体接口不支持 -1 哨兵；合法单类别可计算，全未观察明确报错。 |
| B1 | Monitoring 忽略 sources 却写入证据；现在复用公共查询的来源/特征交集，未知来源报错；没有可靠来源元数据的旧路径明确拒绝。 | Agent 旧报告消费适配；`agent/test_monitor_sources.py`。没有新增监控能力或上游包装层。 |
| B2 | 通用 not_computed/no_target 被解释为 Score Cross 状态；现在按报告类型、表、字段限定语义。 | `_semantics.py` 和快照读取边界；`test_report_evidence_contract.py` 覆盖原报告、快照、保存/加载、Score Cross/Policy、comparison 和自定义规则定义。 |
| B3 | 合法 JSON 的带类型非有限值被当作操作符对象；现在只在筛选边界严格解码既有 `$mars` float。 | `_query.py` 复用 `_serialization.py`；测试标量/in、±Infinity、NaN/Null既有语义、普通字符串和坏标签拒绝，两后端及新进程。 |
| B4 | 裁剪列/行后 evidence 仍声称原查询；现在有效 columns/limit 同展示子集，整体 JSON 连查询/身份开销一起计预算。 | 现有上下文裁剪路径；回归含365列宽表、非零offset、多表、严格JSON、重复调用和精确查询回放。 |
| B5 | 只投影数值丢失特征字典，关系查询缺另一端；现在基于真实页身份裁剪元数据，投影缺身份时补等长 identities，裁行同步裁身份。 | 同一上下文路径；回归包括多行数字投影、关系两端、空页、排序、来源交集和成员桥接，不携带全项目字典。 |
| C1 | 原透视工作簿的缓存可能需要 Excel 刷新；已有 snapshot 静态路径直接写出当前值，本次补足测试和入口文档。 | `test_static_excel.py` 两后端逐表关闭/重开核对值、中文名、NaN、权重、KS/IV、保存文件新进程导出；原透视 XML 和 refreshOnLoad 保留。 |
| C2 | schema 默认排序列不存在，文本列渐变时转float；现在根据实际schema选择默认排序，仅数值列使用数值样式。 | `profile_report.py`、`_profile_excel.py`；`test_profile_presentation.py` 实际执行 Styler.to_html，覆盖分组/不分组、默认/显式、不可比较/空/nullable 与查询/HTML/Excel值一致。 |
| C3 | 全局/局部事件互相覆盖 hidden；现在按当前两条件的交集重算。 | `_profile_html.py` 同时服务原报告/快照；DOM测试与真实 Chrome脚本分开记录。当前只有页面导航，没有行级分页；验证切页前后与排序保留筛选。 |
| C4 | 报告索引遗漏Correlation/ScoreCross，未发布固定安装提示容易误导。 | 现有指南/索引、受测试展示示例、Unreleased变更记录；安装说明链接当前PyPI/GitHub已发布0.0.27，源码仍0.0.28，不升版本。 |

## 环境与命令

Windows 11 build 26200，Intel64 Family 6 Model 183。主环境 Python3.12.13、
Pandas3.0.3、Polars1.42.0、NumPy2.4.6、PyArrow24.0.0、optbinning0.21.0、
statsmodels0.14.6。Ruff0.16.9、Mypy1.13.0、pydoclint0.10.1、pytest9.1.1。
3.8使用Pandas2.0.3/Polars1.8.2；3.9使用Pandas2.3.3/Polars1.36.1；
3.10使用Pandas2.3.3/Polars1.37.1；3.11使用Pandas3.0.3/Polars1.42.0。

本地统一 `PYTHONPATH=src`，日志使用UTF-8。已有 pytest cache 不可写，故本轮
加 `-p no:cacheprovider` 和新的 `--basetemp .pytest-tmp-<batch>`，没有清理用户缓存。
3.10首次 import `_ctypes` 被沙箱拒绝；进程内PATH补足后仍失败，授权仅在沙箱外运行
相同回归测试后通过，未更改系统PATH、DLL或权限。

A退出批次覆盖新增回归及分箱、风险、评估、batching、selection、correlation、Score Cross、
rule generators、计算、Monitoring、Scoring、replay、文档，共437 passed、5 deselected。
3.9/3.10/3.11仅新增A两文件各112 passed，不能与退出批次累加为唯一测试数。
真实 Optimal 测试的记录类调用真实 `super().fit()`，确认状态OPTIMAL/FEASIBLE且无fallback；
六行小例子及既有测试覆盖fallback。混合局部失败的stub只证明该分支。
旧Boolean JSON/pickle测试通过人工置换旧字段构造，属于模拟历史状态；
B的 `tests/fixtures/legacy_e990813_no_target.marsreport` 则由真实e990813源码独立进程生成。

最终验证实际执行以下命令；Windows 使用上列环境的绝对解释器路径。

```bash
python -m ruff check src tests scripts docs/snippets
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

全仓默认依赖测试在生产源码稳定后 **1018 passed、3 skipped、7 deselected**，141.94秒。
Ruff通过，Mypy157个源码文件通过，pydoclint通过，私有docstring检查通过（6条已有提示），
版本一致性仍为0.0.28。文档43 passed、2 deselected；strict build通过。
wheel/sdist各构建一次，资源与元数据检查、Twine通过；本地使用已有构建依赖，
远端工作流仍在隔离构建环境产出同一wheel供两个版本安装测试。

最终版本矩阵小批次命令为以下7个测试文件加同样nocache/fresh basetemp：
`test_binner_refit_boolean_labels.py test_profile_risk_multitarget.py test_report_evidence_contract.py
test_profile_presentation.py test_profile_html.py test_static_excel.py test_report_query.py`。
3.8/3.9整批各230 passed；最终HTML改动后两版本各4 passed受影响补验。
3.10/3.11原整批各226 passed、4 failed；失败来自新增测试对Pandas字面空格表示
固定为NBSP的错误预期，改为从真实导出单元格读取后各4 passed。不能将原失败命令写为整批通过。
3.12最终完整批次覆盖全部当前核心测试。各批有重叠，不相加成唯一总数。

本地有statsmodels与真实Optimal求解器，缺少XGBoost/LightGBM/CatBoost/Optuna；
3个skip是modeling模块缺XGBoost，7个deselected由真实docs_ml/optional_ml标记排除。
本地未执行完整可选ML与两个docs_ml集成例子；远端modeling结果单独核验，
任务分支不触发Docs工作流，不能宣称其docs_ml已经远端执行。

首次全仓批次3个private-docstrings测试因子进程UTF-8输出按Windows GBK捕获而失败；
设置进程级 `PYTHONUTF8=1` 后5项定向通过，最终全仓通过。一次附加全仓收集遇到本任务
临时目录ACL错误；临时产物移至验证目录后最终默认命令通过，未修改ACL或扩大排除标记。
这些失败及补跑日志保留在工作区外的任务验证目录，没有修改无关测试来消除失败。

## 持久化与人工交付边界

`.marsreport`格式仍为1，不引入新序列化系统。独立消费进程只读保存文件，不读原宽表、
不拟合；回归核对身份、describe语义、sources、投影字典、预算与合法JSON精确回放。
原统计/权重/KS定义不改变。不可比较或未计算状态不伪装为有效零。

`from mars.reporting import snapshot_report; snapshot_report(report).write_excel(path)`
是已有静态Excel入口，恢复后的 `load_report(path).write_excel(...)` 同样静态。
openpyxl `data_only=True` 的关闭/重开验收通过后，说明无需Excel刷新即可读取本次数值。
这不等于原生Excel刷新验收。本会话的原生桌面控制接口未开放，无法操作Excel；
透视模式仅验证结构/XML，原生打开、刷新、保存后缓存仍未验证。

真实浏览器使用已有Playwright1.63.0与本机Chrome154.0.8037.95，headless、file://。
`tests/browser/profile_search.py --output <validation>/c-profile-browser-final-v2 --channel chrome`
实际导出当前报告及文件恢复的快照；1440×1000、390×1000共4页，每页36次搜索/
清空/重复输入/大小写/中文/空格/局部隔离/排序/导航操作，核对16个数值字段。
pageerror、console_error、CSP、失败请求均0，无外部请求，页面根宽不超视口。
已实际查看本次empty、local-cleared与final截图，空交集无数据行，清局部仍保留全局条件；
窄屏表格在本地容器横向滚动，搜索和导航控件可见。当前HTML模板SHA256为
`ab7de5a5996a33b32340f6680fb7efa03a046039854e27c915c1c76fe3c6ab69`。
DOM测试另在Node VM运行，不当作视觉验收。源码曾发现hidden改变innerText空格表现，
现以单元格textContent和tab稳定拼接；保留字面空格，不trim、不新增模糊匹配。
新脚本原名profile.py曾遮蔽stdlib并使相邻浏览器脚本失败，已改名且复验。

`tests/browser/score_cross_time.py --output <validation>/score-time-final --channel chrome`
全部16个夹具通过；原e990813的Score Cross HTML源码字节未改。
`tests/browser/docs.py --site-dir <validation>/docs/site --output <validation>/docs-browser --channel chrome`
12种视口/主题状态通过，已查看移动指南截图。截图/JSON日志不进入源码提交。

报告消费性能复用 `benchmarks/benchmark_report_consumption.py`，分别通过PYTHONPATH导入
不可变e990813归档源码和修复源码；同机、相同依赖、确定性数据无随机种子，
overview10000行、趋势1行×365日期列、offset9000/limit10、10次中位数，save/load各3次。

| 操作 | e990813中位ms | 修复后中位ms | Python分配峰值（前→后） |
| --- | ---: | ---: | --- |
| 分页 | 0.310 | 0.321 | 11811→11811 bytes |
| 16000字符上下文 | 25.611 | 27.093 | 1900371→1900799 bytes |
| 5000字符上下文 | 185.756 | 34.824 | 1900077→1909165 bytes |
| 保存 | 88.869 | 88.084 | 2519533→2520758 bytes |
| 加载 | 63.235 | 60.590 | 4437879→4437722 bytes |

5000预算实际输出4985→4975字符、93→66字段；少展示27字段是有效query.columns也计入
预算的兼容变化。逐列初版曾241.69ms，沿现有裁剪逻辑改稳定前缀二分后复测得到上表。
文件均587098 bytes。tracemalloc仅测Python分配、不含原生RSS；单机微测量不外推
拟合、整体容量或普遍加速。原始JSON在任务验证目录report-contract-before/after.json。

## 远端验证边界

仅推送核实的任务分支和origin，不force push。远端精确SHA及CI终态随最终回复核验。
CI push覆盖quality、core3.10—3.12、legacy3.8/3.9、modeling、单次distribution、
同一wheel的installed smoke3.8/3.12和rule performance smoke。
Docs只对main push、面向main的PR或手动dispatch触发；本任务不建PR，
也不dispatch带deploy的工作流。文档在本地strict build和真实浏览器验收，
不能引用历史Docs成功作为本次提交结果。
