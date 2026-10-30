# 分客群分析本地实施记录

## 起点与授权

2026-10-29 在 `D:/Desktop/credit-risk/mars-risk` 核对并 fetch。
远端 main 为 `edf7259f6b3a1bd785668a11af7736d2d09b0fb8`。
实施分支为 `feat/segmented-analysis-core`，起点及 HEAD 为
`2c54a2fe4d3cf57c110f33f8fdad74ea88412c28`。切换前工作树干净。
实施阶段未在 main 修改功能，未 commit、push、PR、merge、发布或修改版本。
2026-10-29，用户在本地对话中随后明确授权提交并普通 push 到
`feat/segmented-analysis-core`，保留 main。本次提交包含源码、回归、脚本与文档，
本地数据、环境、验收产物和站点构建目录继续留在忽略目录中。

## 实施合同 { #scope-contract }

- `profile_risk(..., segment_col=None, binning_scope="shared", segment_binners=None)`；
  `MarsBinEvaluator(binning_scope="shared").evaluate(..., segment_col=None, segment_binners=None)`。
- `profile_stats(..., segment_col=None, binning_scope="shared")`；
  `MarsDataProfiler(binning_scope="shared").generate_profile(..., segment_col=None)`。
- `MarsStatsSelector(selection_scope="global", binning_scope="shared")`，运行时 `fit(..., segment_col=None)`。
  `global + per_segment` 拒绝；未提供客群列时要求默认策略。
- 整体规则属于 `MarsRiskProfile.binner`。`binners_by_scope` 按稳定身份保存规则；
  `save_binning_rules/load_binning_rules` 保存正式规则，不保存原表。
- `scopes` 的字段为 `scope_id/scope_kind/segment_col/segment_dtype/segment_value/`
  `display_name/scope_order/status/reason`。整体 ID 为 `all`，客群 ID 为列名、dtype 与受限编码值的 SHA256。
  缺失客群归一 Null/NaN，字符串不改写。不支持复合对象和无穷值。
- 启用客群后，各事实表在原有粒度前加 `scope_id`；趋势仍只覆盖原有目标范围。
  未启用客群时不加列。All 独立聚合完整人口，Total 汇总当前人口的原分组。
- 完整 `get_table` 默认返回全部 scope。显式 `segments` 与 `include_all`、
  feature/source/target/group 条件取交集。跨人口投影保留 `scope_id`。
  人工摘要、绘图和 AI context 默认 All，分页引用固化实际 scope 条件。
- 缺基准、缺首组、单类别拟合、算法失败分别记录局部原因。
  标签、权重、角色和基准结构错误在编排之前拒绝。
- `.marsreport` 继续使用格式 1 的 manifest + Parquet。新增客群语义不构成文件格式升级。
  历史无 scope 文件沿用既有恢复；快照只保存统计证据，绘图 adapter 只读统计。

## 实施与验收状态

P00–P05 的核心源码、回归与消费已实现，并完成下列本地验收。
权威计划按实际证据勾选，未做的联合转换优化继续保留未完成。

资源边界：当前 shared 规则只拟合一次，各 scope 顺序评估时仍会重复转换。
StatsSelector shared 模式先预评估粗筛，确定精筛候选并集，因此有额外预评估成本。
不能据此宣称已完成计划中的单次转换优化或大规模容量验收。

### 数值与固定规则

- 未分客群路径的 9 张公共表与保存的参考 schema、数值一致。
  原生 PNG 前后 SHA256 相同，为 `f1c17168096cd5565c6beabcd4abaaa506bf5bbcf11f0128733a948c2d5b75178`。
- 风险定向现代环境 184 passed，旧环境新增 20 passed；数据画像两环境各 138 passed。
  最后补充的筛选与消费窗口为 30 passed。
- 正式规则在现代环境拟合后，独立 Python 3.8 进程只加载 JSON；禁止所有 fit，
  All 与 3 个客群固定转换一致，未持有训练缓存。见 `.artifacts/segmented/risk-fixed-newprocess.json`。
- 完整示例为 `docs/snippets/segmented_analysis.py`。同一快照的新进程 seg2/x/Val 查询匹配 4 行，
  每页 2 行，下一页偏移 2；AI context 为 7,781 字符，24,000 字符预算内实际 Agent 工具查询成功。
  HTML、静态 Excel、PNG 与引用重放均完成。证据在 `.artifacts/segmented/acceptance/`。

### 浏览器与 Excel

- 本地真实 Chrome 经 Playwright 打开离线文件，1440px 与 390px 截图已人工查看。
  客群切换、已登记中文特征检索、来源交集、空结果、scope/view 锚点重载和离线懒加载图片通过。
  不是用 Node 或 HTML 文件存在替代浏览器验收。截图与机器记录在 acceptance 目录。
- Excel 16 可用。openpyxl 与 xlwings 两引擎的 seg2 文件均在隐藏 Excel 实例中打开，
  刷新透视、全量重算并保存；22 列、活动页、最终汇总和分箱 KS 对应公共表。
  三客群另核权重、金额、raw 源报告与 binned 最终 KS，以及未表现客群的 null。
  单客群独立参考的纯 null IV Float32 无损提升至合并事实表的 Float64 后比较，原始 dtype 身份不变。
  记录为 `.artifacts/segmented/excel-evidence.json`、`weighted-excel-evidence.json`。

### 真实输入与资源 { #real-data-resources }

Home Credit 正式输入生成器确认 307,511 行 train、48,744 行 test，总计 356,255 行。
seed 为 20261001，保留五分区，0train 为 215,374 行。原 CSV 和预处理文件未修改。
复用既有 20 个代表候选，用 `SK_ID_CURR % 3` 明确构造演示客群；这些不是业务真实客群或真实申请时间。

两模式均执行完整公开风险画像、数据画像、StatsSelector、各 scope 报告与固定规则共存及保存。
独立 All 固定规则参考、统计参考和逐 scope 手工人数、已表现人数、坏人数通过。
All 名单均 14 个；shared 客群名单为 13/13/14，per_segment 为 13/13/13。

| 测量 | 行与候选 | 实际公开调用耗时 | 采样峰值 RSS |
| --- | --- | --- | --- |
| 小型 12 组 | 1,000 行、12 候选、1/3/9 客群、均衡/不均衡、两模式 | 4.744–9.236 秒 | 301.38–315.09 MiB |
| Home Credit shared | 完整 356,255 行、20 代表候选、3 客群 | 36.400 秒 | 986.30 MiB |
| Home Credit per_segment | 同上 | 55.136 秒 | 1,488.33 MiB |

小型矩阵 12/12 通过，同输入 All 证据比较 10/10 通过，容差为 1e-10，保留 null、身份和顺序。
依赖实际为 Python 3.12.13、Polars 1.42.0、Pandas 3.0.3、NumPy 2.4.6、Arrow 24.0.0、
scikit-learn 1.9.0、optbinning 0.21.0，Polars 线程为 1。
来源为 `.artifacts/segmented/resource/summary.json`、`home-credit.json` 和各子进程记录。

RSS 每 10ms 采样，包含三入口报告、规则及保存，不代表操作系统硬内存上限。
准备器全宽加载后的分配器基线不同；包含准备/导入的生命周期峰值分别为 1,593.77/1,514.82 MiB。
存在本地并行验证竞争，不作提速或容量外推，也没有完成 Home Credit 全宽候选的容量验收。

### 环境与质量门

全包 Ruff、Mypy（164 个源文件）、public/private docstring、严格 MkDocs 和案例资产检查通过。
案例检查为 72 文件、11 快照、7 案例；未要求重建既有资产，未手改 manifest/hash。
旧图和已有下游接口通过原有基础测试消费；没有为暂停模块加兼容壳。

完整基础命令使用 CI 标记 `not docs_ml and not optional_ml`，仅忽略本地已有
`benchmarks/_artifacts` 测量输出目录，保留全部源测试。设置 Agg 和线程 1，输出实际 mars 路径。
首轮 Windows 父进程 GBK 与子输出 UTF-8 不一致引发三项 docstring 测试失败；
统一 `PYTHONUTF8=1` 后相关复验通过，未修改检查器或断言。
Python 3.8 独立下载测试曾读到外部旧安装包，现使用仓库内隔离 venv editable 安装。
Python 3.9 曾在 OSQP 原生扩展导入崩溃，隔离环境只固定 OSQP 1.0.4 后定向 6 项通过。
原业务环境与全局环境均未修改。最终完整结果如下，不把首轮失败算通过。

| 环境 | 实际 Python / Polars / Pandas | 完整基础结果与最终复验 |
| --- | --- | --- |
| 3.8 隔离冻结栈 | 3.8.20 / 1.8.2 / 2.0.3 | clean 全包 1441 passed、13 skipped、7 deselected，exit 0 |
| 3.9 隔离依赖边界 | 3.9.20 / 1.36.1 / 2.3.3 | clean 全包 1441 passed、13 skipped、7 deselected，exit 0；OSQP 1.0.4 |
| 3.10 现有环境 | 3.10.19 / 1.37.1 / 2.3.3 | 首轮全包 1574 passed、3 failed、1 skipped、7 deselected；编码相关文件 UTF-8 复验 5 passed，最终消费 7 passed |
| 3.11 现有隔离环境 | 3.11.15 / 1.44.2 / 2.3.3 | 首轮全包 1536 passed、3 failed、3 skipped、7 deselected；UTF-8 与最终消费增量 15 passed |
| 3.12 现代栈 | 3.12.13 / 1.42.0 / 3.0.3 | clean 全包 1540 passed、3 skipped、7 deselected，exit 0 |

这些完整运行发生在最后的视图证据身份与早期 fit 状态清理之前。
最后补丁仅影响失败时清理、导出视图的 report_id/source/有效引用及 Protocol 类型签名，
不改变成功分析的指标或筛选策略；按影响另跑客群筛选、消费、查询与快照回归及全包静态门。
各轮完整结果、首次失败、复验和实际解释器路径均保留在 `.artifacts/segmented/`，
不把多个命令拼成一次未实际发生的 clean 全包。

最后状态与报告身份补丁的客群筛选/消费/身份三文件，在 3.8、3.9 与 3.12 各 34 passed，退出码均 0。
公共查询、快照与身份的扩大定向为 112 passed。最终全包 Ruff、Mypy、两类 docstring 检查通过。
导出视图沿用原报告 report_id/source，并保存逐表 `scope_table_references`，引用可在原报告重放；
失败的 fit 在角色验证之前清空旧规则、决策、漏斗和相关性状态。

| 阶段 | 核心源码与回归入口 |
| --- | --- |
| P00 | `feature/_scopes.py`、`tests/test_segment_scopes.py`，受控参考位于 `.artifacts/segmented/baseline/` |
| P01 | `reporting/_query.py`、`_artifact.py`、`_metadata.py`、`_scope_view.py` |
| P02 | `analysis/_risk_profile.py`、`evaluator.py`、`_evaluation/segmented.py`、`_raw_ks.py` |
| P03 | `analysis/profiler.py`、`_profiling/segmented.py`、`psi.py` |
| P04 | `feature/selection/stats.py`、`_segmented.py`、`tests/test_segmented_selector.py` |
| P05 | `reporting/binning_report.py`、`profile_report.py`、`_binning_plot.py`、`html_assets.py`、`tests/test_segmented_consumption.py` |

最终适用命令如下，使用记录中的实际解释器；3.8/3.9 只在仓库隔离 venv 安装。

```powershell
$env:PYTHONUTF8 = "1"
$env:MPLBACKEND = "Agg"
$env:POLARS_MAX_THREADS = "1"
python -m pytest -q -m "not docs_ml and not optional_ml" --ignore=benchmarks/_artifacts --basetemp=.artifacts/segmented/base-clean
python -m ruff check src tests scripts benchmarks docs/snippets
python -m mypy src/mars
pydoclint src/mars
python scripts/check_private_docstrings.py src/mars
python docs/snippets/segmented_analysis.py --output-dir .artifacts/segmented/acceptance
python -m mkdocs build --strict
python scripts/check_case_assets.py --mode public --site-dir site
git diff --check
```

原生绘图的新进程记录另覆盖现代线程 4、多 chunk，以及旧 Polars 的固定规则恢复。
现代/旧版同时执行结果明确容差和 null 掩码；没有改成全宽 rechunk 或删断言。
本轮没有在 Linux/macOS 做交互验收，没有验收 Home Credit 全宽候选容量或无限高基数客群。

### 迁移与后续

无 segment 的原调用和 schema 保持。新增固定 binner 便捷入口；开启 segment 后消费者需按
`scope_id` 连接，避免 feature 同名扩行。完整表和默认人工视图的范围不同，查询可重放。
报告文件格式仍为 1，不恢复 estimator；规则产物独立 JSON，不能互相替代。
P06 模型分交叉、P07 规则评估、P08 缺失异常诊断和 Linear/Importance 独立客群筛选未实施。

## F01–F08 审查修复验收 { #review-fixes-f01-f08 }

本节记录 2026-10-29 从 `d725908ad316ee6e2f36d50c7e3f945c67e27979` 开始的独立修复。
上文是此前实施的历史记录，不能当作本轮通过证据。
本次起点分支为 `feat/segmented-analysis-core`，remote 为 `git@github.com:leeesq/mars-risk.git`，
起始工作树干净。修复验收阶段未切换 main，未 commit、push、创建 PR、merge、发布或改版本。
未改筛选阈值、候选定义、正式数据生成器或既有案例 manifest/hash。

修复验收完成后，用户在本地对话中明确授权 push。本次交付据此提交并普通推送到
`feat/segmented-analysis-core`，包含 F01–F08 的源码、回归、示例和文档。
本地环境、测试日志、验收产物和站点构建目录继续留在忽略目录，不纳入提交。
该授权不扩展到创建 PR、合并、发布或修改版本。

### 逐项结果与回归入口

| 问题 | 根因与核心修复 | 修复前实际表现 | 修复后数值与状态 | 回归入口 |
| --- | --- | --- | --- | --- |
| F01 | 客群单类被允许评估，但贡献与最终指标仍按两类计算。共享统计按好坏有效质量判定，保留全空聚合；图和 HTML 同步保护全空贡献 | seg2 全 0 时 IV=12.7177352905、KS=99.9999991667、AUC=1，同时状态称单类别 | 三指标及 iv_bin/ks_bin/auc_bin 为 typed null，按期记录 single_observed_class；合法坏率、金额与 PSI 保留 | `tests/test_segmented_metric_fixes.py`，既有 evaluator/raw KS/plot/Excel 回归 |
| F02 | 预拟合 binner 绕过显式 benchmark 退化校验，且 shared 传入局部 fit 人口 | 常量基准 A/B PSI=0；全缺失基准 B≈26.93785、total≈6.56117 | 实际整体或独立拟合人口退化时 PSI 全 null，状态 failed；整体有效、局部常量时 B=26.9378479946 合法保留 | `tests/test_segmented_fit_fixes.py` |
| F03 | All 子筛选器的参考组复制回公开实例，下一次 global fit 未清理 | reused PSI≈0.135154、名单为空；fresh PSI=0、保留 x | 公开入口先重建运行状态；reused/fresh 均保留 x，PSI=0、RC=1，参考、漏斗和相关决策相同 | `tests/test_segmented_fit_fixes.py` |
| F04 | 一个缺参考标志同时擦除 PSI 与其他来源 RC | benchmark 缺 seg2 时 PSI 和 first_group RC 都 null | PSI null；有效 first_group/total RC=1；反向缺首组仍保留有效 benchmark PSI；无标签、零参考权重和常量坏率分别说明 | `tests/test_segmented_metric_fixes.py` |
| F05 | 次目标无标签评估完成后，元数据再次硬投影声明但不存在的目标 | Polars ColumnNotFoundError；Pandas 缺列 KeyError | 只转换存在的目标窄列；保留声明顺序、缺失目标身份及 null 事件率；y/absent 正常返回两行整体 summary | `tests/test_profile_risk_multitarget.py` |
| F06 | 状态查询遗漏 scope 条件，且固定两列解包包含 scope_id 的返回表 | seg1/seg2 show_trend 均 ValueError: too many values to unpack | 按字段名读状态，按实际 scope/feature 配对和显示期间筛选；seg1 无 seg2 原因，seg2 显示自身 null 与状态 | `tests/test_segmented_query_fixes.py` |
| F07 | PSI 空人口在 scope 总量检查中遗漏具体 group；底层全空贡献 sum 再生成 0 | seg2/02 PSI=0，缺不可计算状态 | 02=null 且 no_effective_current_population，01/Total=0；开启适用缺失／特殊箱时保留有效结果，零权重另记原因 | `tests/test_segmented_metric_fixes.py` |
| F08 | 显式 scope filter 覆盖已解析范围，忽略 include_all=False；pop 默认值混淆 null 与无键 | include_all=False + scope_id=all 返回整体一行 | 返回零行；直接指定 seg2 仍有效。None/[]、null、未知身份、业务值 All、单 scope 默认与引用重放一致 | `tests/test_segmented_query_fixes.py` |

F01 覆盖 shared/per_segment、raw/binned、全 0/全 1/无表现/仅一类正权重、多目标、
单期与有效 Total、开发两类但评估单类、正常箱与缺失箱的标签质量、监督拟合失败。
当前统计 WOE 与正式转换映射分开；正式 JSON 恢复固定转换一致。
Lift 使用自身分母，合法全 1 的 Lift 保留；全 0 的基准坏率为 0 时不伪造 Lift。
未分客群评估对非法输入和单类别输入的原有入口约束继续保留。

F02 诊断只在实际拟合人口按特征批次执行。shared 复用一次整体诊断，独立规则用各自人口。
显式空诊断与未提供诊断分开，内部首组 reference 不冒充显式 benchmark。
覆盖 mixed metrics、Null/Float32/Float64、Pandas/Polars、缺失／特殊值四种开关及无 benchmark 旧行为。
F03 覆盖三种合法组合、策略与分组变化、benchmark 增删、成功后失败及失败后再成功。

F04 的参考表和 metadata 分别保留 PSI expected 与 RC total/first_group/benchmark 来源。
F07 从已聚合分布判断 feature/group/Total 的有效质量，不新增全宽 mask 或独立 PSI 算法。
已用 `git archive d725908` 的本地源码副本证明：旧未分客群路径在全期无适用人口时也会输出 PSI=0。
该数值遗漏修在共享计算层，不能归为本次客群提交首次引入。

F05 新增 24 个案例，覆盖缺失／存在／全空辅助目标、缺主目标和两目标均无标签，
同时覆盖 raw/binned、两种后端及分客群与整体。保存恢复后的 summary/detail/status 和事件率一致。
F06/F08 新增 70 个案例，覆盖空查询、空状态、显式投影省略 feature、分页后身份、
多客群、原始表／query_page／AI context 及保存恢复后的有效 reference。

### 复现与专项证据

本地忽略目录保留实际命令输出；这些产物不是包内容，也没有提交到 Git。

| 证据 | 结果与位置 |
| --- | --- |
| F01/F04/F07 原始数值与修复数值 | `.artifacts/segmented-fixes/metrics-before.log`、`metrics-after.log`；旧底层证据 `f07-plain-before.log` |
| F02/F03 原始失败 | `.tmp/f02-f03-before.txt`，5 failed；实际前后数值 `.tmp/f02-f03-evidence.json` |
| F05 原始失败 | `.artifacts/segmented-fixes/f05-before.log`，1 failed、34 deselected（maxfail=1） |
| F06/F08 原始失败 | `.artifacts/query-fixes-before-native.log`，28 failed、42 passed，exit 1 |
| F02/F03 最终专项 | `.tmp/f02-f03-complete312.txt`、`complete38.txt`，各 59 passed，exit 0 |
| F05 最终专项 | `.artifacts/segmented-fixes/f05-expanded.log`，58 passed，exit 0 |
| F06/F08 最终专项与邻近 | `.artifacts/query-fixes-after-native.log`、`after38.log` 各 70 passed；`adjacent312.log` 306 passed，exit 0 |
| 风险与消费专项 | `.artifacts/segmented-fixes/metrics-consumption312.log`、`metrics-consumption38.log` 各 65 passed；`metrics-adjacent312.log` 150 passed，exit 0 |

专项完成后，交叉审查补齐正常箱仅一类时的排序区分度状态、全 Missing/Special 期间 RC 状态、
参考箱无交集的配对不足原因，以及金额 Lift 独立分母。切换本次 fit 的 RC 策略时，前置校验读取新策略。
另修正本轮参考检查误清空合法 feature-start RC 的回归，保留其来源，人口行数明确为 None。
完整基础运行发生在这些末轮小修之前；其后五版本各运行 105 项最终专项，结果分开记录。
首轮全包在修改过程中运行，结果为 1705 passed、5 failed、3 skipped、7 deselected，exit 1，
保留在 `.artifacts/segmented-fixes/full312.log`。这些失败未计为通过。
最初新增快照回归的默认硬链接保存在沙箱内收到 WinError 5，后续授权测试在正常权限下复验。
未修改产品保存算法或删除既有断言。个别新增快照测试使用正式支持的 overwrite=True 保存路径。
本次 fit 策略切换的补充复现为 `.tmp/f03-policy-before.txt`（2 failed、59 deselected），
修复后两方向通过。新 feature-start 测试在旧版 Polars 得到 RC=0.9999999999999998，
新增断言采用 rel=0、abs=1e-12 检查数学上的 1；未修改既有断言、业务公式或原测试容差。
末轮扩大回归曾误写不存在的 `test_report_artifact.py`，no tests ran、exit 4，记录在 `reviewed312.log`，
纠正后以 `reviewed-final*.log` 另行记录。示例扩展也曾因配置冲突和传入不支持参数两次 exit 1，
保留 `snippet-first.log`、`snippet-after.log`；最终正确调用单独运行，未计失败命令为通过。
首轮 `static-ruff.json` 因新增查询测试中的多余 Pandas 导入 exit 1；移除该导入后，
最终全包 Ruff 通过，失败记录保留。

### 完整示例与恢复消费

`docs/snippets/segmented_analysis.py` 最终运行 exit 0，记录为
`.artifacts/segmented-fixes/snippet-reviewed.log` 和同名 JSON，产物在该目录的 `acceptance/`。
同一份 `risk.marsreport` 的新进程 seg2/x/Val 查询共 4 行，每页 2 行，next_offset=2。
AI context 为 7,911 字符，实际 Agent 工具成功，query reference 在同一持久身份下重放一致。

新增 `undefined.marsreport` 保存有效开发规则下单类别 seg2 和 Val 期 x 全缺失的统计。
新进程仅加载快照，断言 IV/KS/AUC 及逐箱贡献全 null，Val PSI null、Train PSI 有效，
并重放状态 reference、输出 HTML、静态 Excel、PNG 和 AI context。恢复过程不访问原表或 fit。
`undefined-native.xlsx` 仍是原生 22 列模板；恢复后静态长表为 `undefined-snapshot.xlsx`，两者职责不同。
正式规则为独立 `rules.json`，保存恢复后的固定转换相同，不由统计快照恢复 estimator。
新进程还只加载正式规则和十行模拟探针，四个 scope 的固定转换均与保存前一致。

PNG 已实际查看，单类期和 Total 的 IV/KS/AUC 均显示 n.a.，Val 缺失率为 100%，
总体缺失率为 33.33%。Excel 内容回归读取实际表头、数据单元格、最终汇总及分箱 KS，
HTML 回归检查实际数据与聚合逻辑，没有用文件存在代替数值验收。
本次 Office 原生启动的应用审批超时；浏览器 Computer Use 因无法可靠识别当前 URL 而终止。
Office 打开／重算和浏览器交互视觉验收本轮未完成，上文历史验收不替代本轮验收。

### 最终环境、命令与边界

线程在导入前设置为 POLARS_MAX_THREADS=4、OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1，
同时设置 PYTHONUTF8=1 和 MPLBACKEND=Agg。现代多线程和多 chunk 的行序／标签绑定保留原回归。
各命令实际解释器、源码路径、依赖、线程、耗时及返回码保存在
`.artifacts/segmented-fixes/final*.json` 和同名日志，不修改现有业务／全局环境。
完整依赖版本另存为 `environments.json`。本轮 3.11 使用 optbinning 1.0.0，
3.12 使用 0.21.0；3.8 的 0.20.1 与 sklearn 1.3.2 配套，未强行混用现代 sklearn。

完整基础命令与 CI 一致，排除 docs_ml/optional_ml，仅额外忽略本地 benchmark 产物目录。
下表分开记录完整基础运行与末轮源码的专项，不能将两者相加视为一次完整测试。

| 环境 | 解释器与实际依赖 | 完整基础结果／末轮专项 |
| --- | --- | --- |
| 3.8 | `.artifacts/segmented/env38-final/Scripts/python.exe`；3.8.20 / Polars 1.8.2 / Pandas 2.0.3 / NumPy 1.24.4 | 1635 passed、13 skipped、7 deselected；末轮 105 passed，均 exit 0 |
| 3.9 | `.artifacts/segmented/env39-final/Scripts/python.exe`；3.9.20 / 1.36.1 / 2.3.3 / 2.0.2 | 1635 passed、13 skipped、7 deselected；末轮 105 passed，均 exit 0 |
| 3.10 | `D:/my_download_program/anaconda3/envs/mars/python.exe`；3.10.19 / 1.37.1 / 2.3.3 / 2.2.6 | 1772 passed、1 skipped、7 deselected；末轮 105 passed，均 exit 0 |
| 3.11 | `D:/Desktop/credit-risk/.artifacts/closure-20261015/envs/py311/Scripts/python.exe`；3.11.15 / 1.44.2 / 2.3.3 / 2.4.6 | 1734 passed、3 skipped、7 deselected；末轮 105 passed，均 exit 0 |
| 3.12 | `D:/my_download_program/anaconda3/envs/mars312/python.exe`；3.12.13 / 1.42.0 / 3.0.3 / 2.4.6 | 1734 passed、3 skipped、7 deselected；末轮 105 passed，均 exit 0 |

```powershell
python -m pytest -q -m "not docs_ml and not optional_ml" --ignore=benchmarks/_artifacts --basetemp=.artifacts/segmented-fixes/final312-tmp
python -m ruff check src tests benchmarks scripts docs/snippets
python -m mypy src/mars
pydoclint src/mars
python scripts/check_private_docstrings.py src/mars
python docs/snippets/segmented_analysis.py --output-dir .artifacts/segmented-fixes/acceptance
python -m mkdocs build --strict
python scripts/check_case_assets.py --mode public --site-dir site
git diff --check
```

Ruff、全包 Mypy（164 个文件）、public/private docstring 已通过，exit 0。
扩大消费回归为 3.8/3.9 各 610 passed、5 skipped，3.10/3.11/3.12 各 615 passed，
记录在 `reviewed-final*.log`，均 exit 0。该轮在最后 feature-start 补修前收集测试；
末轮 105 项含其新增回归，五版本全部通过。未把两轮拼成一次更大的通过数字。
严格 MkDocs 的实际子进程构建先通过 exit 0，但本地记录器在打印日志时遇到 GBK 编码错误，
该外层命令 exit 1，未当作全命令通过。记录器统一 UTF-8 后另行复验。
复验 `mkdocs-reviewed.log` 对应完整命令 exit 0。案例资产检查为 72 文件、11 快照、7 案例，
exit 0，记录在 `case-assets-final.log`；`git diff --check` 通过。
没有运行远端 Docs workflow；本轮没有 PR，不将缺少 Docs run 解释为失败或通过。
Home Credit 输入、seed、模拟日期、五分区和候选定义均未修改；本次未重跑正式数据代表性复验或全宽容量测试，
不沿用历史资源数据宣称提速或容量。没有执行 Linux/macOS 或可选 ML 专项。
shared 每批一次联合转换仍未完成，P06–P08 未启动；三个暂停模块未新增兼容壳或功能。

## F09 数据画像 PSI 适用人口修复 { #review-fix-f09 }

本节记录 2026-10-30 的独立修复。起始分支为 `feat/segmented-analysis-core`，
HEAD 为 `63c29be76a82c92c3714fe83e7a47ebb165c4a04`，工作树干净。
修复验收阶段保留该分支，未 commit、push、创建 PR、合并、发布或修改版本号。
此前 F01–F08 的 push 授权不适用于本轮。

验收完成后，用户在本地对话中明确授权提交并普通 push 到 `feat/segmented-analysis-core`。
本次据此提交 F09 的源码、回归、示例与文档；本地验收产物继续留在忽略目录。
该授权不包含创建 PR、合并、发布或修改版本号。

### 根因与计算合同

该问题在 `d725908ad316ee6e2f36d50c7e3f945c67e27979` 和 `63c29be` 均可复现，
属于既有数据画像路径的覆盖缺口。F01–F08 的回归没有覆盖这条路径。
本地只向 `.artifacts/f09/` 提取两个提交的历史 `src/`，通过独立进程导入历史源码复现，
没有切换或回退当前工作树。远端 CI 成功不替代本缺口的业务回归。

`_profiling/psi.py::calc_psi_from_stats` 原先从同批所有特征取得 group 集合，
再与分箱骨架做笛卡尔积。z 在 02 有观测时，会替没有有效观测的 x/02 生成骨架；
概率空值随后填为 epsilon，得到看似有效的 PSI。
原 feature 级检查和诊断去重也无法解释一个特征的多个不可计算期间。
无显式 benchmark 时，首组某特征为空参考的情况同样会误算。

修复从每批已聚合频数表保留过滤前业务身份，再按箱策略分别计算当前期间、固定参考与 Total 的适用人口。
只有两侧分布适用时，才将该键与箱骨架关联，并沿用原公式、平滑常量和分母。
每批只物化小频数表，不复制每组原始宽表，不增加拟合次数；shared 仍只拟合一次。
有效 Total 按全部当前期的合并计数计算，不平均各期 PSI。

当前期间无适用人口时，仅该 scope/feature/group 返回 Float64 null，
状态为 `not_computed / no_effective_current_population`。
固定参考无适用人口时，该特征各比较期间及 Total 均为 null，
状态为 `not_computed / no_effective_reference_population`。
原因优先解释不可用参考；局部状态使用原 group 标签，Total 沿用 `Total` 状态到 `total` 列的映射。
`metrics.py::build_profile_status` 接收局部 group/status，并保留原有全局拟合失败的 `* / failed` 语义。
合法常量局部参考、数值 0、纳入缺失／特殊箱后有效的人口继续计算。

Pandas 输出在 PSI 缺失数值列使用 nullable Float64，保证 JSON 为正式 null。
查询、分页、AI、趋势和静态导出读取保存的同一份统计与状态，不另行计算。
旧快照按磁盘已保存统计读取；加载时不重算、不改写历史 PSI。

源码修改位置如下。

| 文件 | 本轮职责 |
| --- | --- |
| `analysis/_profiling/psi.py` | 从批内频数判定三侧人口，保留完整业务键和内部原因列，按特征／期间收集局部诊断 |
| `analysis/_profiling/metrics.py`、`types.py` | 装配局部 group/status，保留全局失败形状，按业务身份去重 |
| `analysis/profiler.py`、`_profiling/segmented.py` | 原生及客群合并 PSI 的 Pandas nullable 输出，诊断与状态沿原报告装配 |
| `reporting/profile_report.py`、`_query.py` | 趋势与 AI 的局部原因只解释最终可见 null 单元格，保留 scope／特征／期间身份 |
| `reporting/_semantics.py` | 在既有 describe 语义中解释两个正式原因码 |

### 修复前后证据

以下值来自同一 80 行确定性输入、默认固定规则和 `psi_include_missing=False`。
两个历史提交的 shared／per_segment 均表现相同。

| seg2/x | 历史 batch=1 | 历史 batch=2、8 | 修复后 batch=1、2、8 |
| --- | --- | --- | --- |
| 01 | 0 | 0 | 0 |
| 02 | null，无原因 | ≈11.538983029798928，无原因 | Float64 null，局部 `not_computed` |
| total | 约 0 | 约 0 | 约 0，仍为有效合并人口 |

未分客群、无显式 benchmark 的首组全缺失示例，排除缺失箱时，历史结果为
01=0、02≈10.8458327809、total≈10.8458327809；修复后全部为 null，并明确参考无适用人口。
纳入缺失箱时，首组缺失箱是合法参考。01=0、02=24.66133052261688、
total=5.422869866992135，沿用原公式保留。

最小回归先在基线运行，`population-before.log` 为 6 failed、exit 1，
同时暴露 batch=1 缺状态与 batch=2／16 错误数值。该失败输出先于源码修改保留。
历史源码复现为 `baseline-d725908.log` 和 `baseline-63c29be.log`，两命令均 exit 0；
这里的 exit 0 表示复现脚本执行完成，不表示历史错误通过新合同。
新示例 `docs/snippets/profile_psi_population.py` 在 3.8／3.12 均 exit 0，
逐项断言局部 null、状态和有效 Total，原始输出为 `snippet38.log`、`snippet312.log`。
最终源码另在 Polars 1.44.2 执行同一复现脚本，`repro-after.log`、exit 0，
确认两模式 batch=1／2／8 的 02 均为 null，Total=0，状态只指向 seg2/x/02。

### 专项与持久消费

`tests/test_profile_psi_population.py` 用 42 项业务回归覆盖批次、特征顺序、同批邻居、
未分客群与两种规则策略、首组空参考、整体空人口、多个空期间、缺失／特殊箱开关，
以及数值 0、类别、null／数值／日期分组、多 chunk 与拟合次数。
有效 PSI 的 oracle 使用固定箱小频数，先判断人口，再独立按正式公式计算；
Total 使用合并计数核对，另验证它与期间 PSI 平均值不同。

`tests/test_profile_psi_population_consumption.py` 的 35 项回归覆盖两后端的原生与恢复报告。
逐表比较 `stats.psi`、`calculation_status` 的数值、dtype、元数据和稳定 scope 身份，
验证分页计数、字符预算、reference 重放和 AI 的正式 JSON null。
01 投影、z 和其他客群不带 02 的局部原因；显式省略 feature 的投影继续有效。
AI description 中的局部 diagnostics 也收窄到最终证据，不影响旧全局拟合诊断。
真实字符串 group=`Total` 的 null 与有效汇总 `total=0` 分开解释。

导出检查读取实际 HTML 表头、单元格和 `data-sort-*` 属性：02 的类型为 missing，
排序值为空，显示为缺失符号；01 和 total 仍保存数值 0。
openpyxl 读取原生和快照 Excel，确认 02 单元格为 None，01／total 为零，
CalculationStatus 中只有准确的局部状态行。
新进程只加载 `.marsreport`，随后执行查询、AI、reference 重放、趋势和静态 HTML／Excel，
不访问原表或分箱器。另构造历史错误数值快照，加载后原值仍保留，不自动套用新结果合同。
可审阅产物保存在 `.artifacts/f09/acceptance/`，包含 `standalone.marsreport`、
`standalone.html` 和 `standalone.xlsx`。它们复制自实际通过测试的新进程输出目录
`consumption312-verified-tmp/test_population_snapshot_new_p0/`，逐文件 SHA256 与源文件相同。
另在匹配现代隔离环境执行 `python .artifacts/f09/acceptance/consume.py`，
只读该快照生成 `ai-context.json` 和 `describe.json`；`acceptance-json.log` 为 exit 0，
再次断言 JSON 的局部 null、有效 01／total、准确状态及 reference 重放。

| 实际命令窗口 | 结果与日志 |
| --- | --- |
| 3.8／3.12 业务环境各执行 `python -m pytest -q tests/test_profile_psi_population.py`，使用 pytest 默认临时目录 | 各 42 passed，无失败／跳过，exit 0；`population-legacy-final.log`、`population-modern-final.log`，返回码另存同名 `.exit` |
| 3.8 `python -m pytest -q tests/test_profile_psi_population_consumption.py --basetemp=.artifacts/f09/consumption38-verified-tmp` | 35 passed、31 warnings，43.93 秒，exit 0；`consumption38-verified.log` |
| 3.12 业务环境 `python -m pytest -q tests/test_profile_psi_population_consumption.py --basetemp=.artifacts/f09/consumption312-verified-tmp` | 35 passed、47 warnings，64.19 秒，exit 0；`consumption312-verified.log` |

42 项专项只显式设置 `POLARS_MAX_THREADS=4` 和 `PYTHONUTF8=1`。
两端解释器分别为下表的 3.8 与 3.12 业务环境，运行早于最后的 Total 消费修复。
35 项消费专项另设置 Agg 和 BLAS／OMP 单线程，运行于最终消费源码。
这些专项分开运行，不相加声称为一次完整通过。最终完整基础运行另外覆盖最终文件。
警告来自既有 Arrow／Pandas／依赖弃用提示，未通过跳过断言规避结果。

### 本地环境与实际命令

所有证据放在忽略目录 `.artifacts/f09/`。`run.py` 只在子进程真正退出后，
才将 JSON 的 `state` 改为 finished，并写入实际 `exit_code`、命令、耗时及日志路径。
通过 `run.py` 执行的命令在导入前固定 `PYTHONUTF8=1`、`MPLBACKEND=Agg`、`POLARS_MAX_THREADS=4`、
`OMP_NUM_THREADS=1` 和 `OPENBLAS_NUM_THREADS=1`。

| 环境 | 实际解释器与依赖 | 导入来源 |
| --- | --- | --- |
| 3.8 | `.artifacts/segmented/env38-final/Scripts/python.exe`；Python 3.8.20、Polars 1.8.2、Pandas 2.0.3、NumPy 1.24.4、Arrow 17.0.0、sklearn 1.3.2、optbinning 0.20.1、pytest 7.4.4 | mars 从当前仓库 `src/mars`；依赖从既有 mars38 环境 |
| 3.12 | `D:/my_download_program/anaconda3/envs/mars312/python.exe`；Python 3.12.13、Polars 1.42.0、Pandas 3.0.3、NumPy 2.4.6、Arrow 24.0.0、sklearn 1.9.0、optbinning 0.21.0、pytest 9.1.1 | mars 从当前仓库 `src/mars`；依赖从既有 mars312 环境 |
| 3.12 现代隔离栈 | `D:/Desktop/credit-risk/.artifacts/closure-20261015/envs/py312/Scripts/python.exe`；Python 3.12.13、Polars 1.44.2、Pandas 3.0.6、NumPy 2.4.6、Arrow 26.0.0、sklearn 1.9.1、optbinning 1.0.0、pytest 9.1.1 | mars 从当前仓库 `src/mars`；依赖从既有 py312 隔离环境 |

实际导入位置见 `environment38.log`、`environment312.log` 和 `environment312-isolated.log`，均 exit 0。
本轮使用现成环境，没有重建五版本环境，没有重跑全宽容量或正式案例生成。

```powershell
python .artifacts/f09/environment.py
python .artifacts/f09/baseline_repro.py .artifacts/f09/d725908/src
python .artifacts/f09/baseline_repro.py .artifacts/f09/63c29be/src
python -m pytest -q -m "not docs_ml and not optional_ml" --ignore=benchmarks/_artifacts --basetemp=.artifacts/f09/full-final312-tmp
python -m ruff check src tests benchmarks scripts docs/snippets
python -m mypy src/mars
pydoclint src/mars
python scripts/check_private_docstrings.py src/mars
python docs/snippets/profile_psi_population.py
python -m mkdocs build --strict --site-dir .artifacts/f09/site
git diff --check
```

首轮 full312／full38 在最后 Pandas null 输出修复前启动，不能作为最终源码的完整通过证据。
同一轮中，下一 worker 对已记录 pytest PID 的 `Get-Process`／`Stop-Process` 查询报告不存在，
exec session 却仍为 running。未将这个可见性限制写成测试断言失败或已终止。
随后向这两个 session 发送 Ctrl+C，得到两个外层 exit 1；子进程退出码未知，
另存 `full312-interrupted.json` 和 `full38-interrupted.json`，未计为完整通过。
首次 public docstring 子进程 exit 0，但外层记录器打印 emoji 时遇到 GBK 编码异常，外层 exit 1；
修正记录器 UTF-8 输出后，`public-docstring-final` 整条命令 exit 0。
这些初轮记录保留，不与最后通过数拼接。

首轮完整运行出现快照保存失败。单独探针 `snapshot-sandbox-probe.log` 为
1 failed、2 passed、exit 1，栈定位到既有 `os.link` 原子保存受到沙箱 WinError 5 拒绝。
未更改产品保存算法或正式测试。按同一路径在沙箱外复验，
`snapshot-outside-sandbox.log` 为 4 passed、11 deselected、exit 0；
匹配现代隔离栈的 `snapshot-elevated-probe.log` 也为 4 passed、exit 0。
最终完整基础测试据此在沙箱外运行，仅将验收产物写入本仓库专属临时目录。

最终 Ruff、public docstring 和 private docstring 均 exit 0，分别见
`ruff-final.log`、`public-docstring-last.log` 和 `private-docstring-final.log`。
private 检查保留 6 条既有简单 helper 的建议警告，复杂私有函数检查通过。
全包 Mypy 为 164 个源文件通过、exit 0，见 `mypy-final.log`。
严格 MkDocs 首轮 `mkdocs-first.log` 为 exit 0；补齐最终完整验收数字后，
末次 `mkdocs-last.log`／`.json` 再次 exit 0，整命令 12.172 秒。
构建输出位于 `.artifacts/f09/site/`，未重建案例资产或正式网站数据。

跳过原因另用 `-rs` 小范围复核，不重复完整套件。
`skips38.log` 为 1 passed、13 skipped、exit 0：3 个建模测试模块缺少 xgboost，
规则属性模块缺少 hypothesis，SHAP 案例缺少 shap，另 8 个 Agent 案例要求 Python 3.10+。
`skips312.log` 为 1 passed、3 skipped、exit 0，3 项均为建模模块缺少 xgboost。
确切 node ID、命令和原因保存在同名 JSON／日志；这些结果不与完整套件相加。

最终源码的两套完整基础命令均真正退出，没有中断、挂起或只打印 passed 的情况。
每套包含全部源测试，并按 CI 排除 `docs_ml`／`optional_ml`，仅额外忽略本地已有 benchmark 产物目录。
关联回归包括 `test_profiler.py`、`test_profile_empty_psi.py`、`test_analysis_batching.py`，
既有客群 profiler／fit／query／consumption 文件，以及 F01–F08 的 metric、fit、query 和多目标回归。
新增 42 项人口与 35 项消费测试也均由最终完整运行收集，覆盖最后一次源码修改。

| 最终完整环境 | 结果 | 实际退出与日志 |
| --- | --- | --- |
| Python 3.8.20／Polars 1.8.2／Pandas 2.0.3 | 1718 passed、13 skipped、7 deselected、435 warnings；无失败 | pytest 716.21 秒，整命令 718.093 秒；exit 0；`full-final38.log`／`.json` |
| Python 3.12.13／Polars 1.44.2／Pandas 3.0.6，匹配隔离栈 | 1817 passed、3 skipped、7 deselected、1051 warnings；无失败 | pytest 740.51 秒，整命令 742.766 秒；exit 0；`full-final312.log`／`.json` |

两个完整命令的实际解释器与 basetemp 分别如下；完整参数、环境和耗时另存同名 JSON。

```powershell
& .artifacts/segmented/env38-final/Scripts/python.exe -m pytest -q -m "not docs_ml and not optional_ml" --ignore=benchmarks/_artifacts --basetemp=.artifacts/f09/full-final38-tmp
& D:/Desktop/credit-risk/.artifacts/closure-20261015/envs/py312/Scripts/python.exe -m pytest -q -m "not docs_ml and not optional_ml" --ignore=benchmarks/_artifacts --basetemp=.artifacts/f09/full-final312-tmp
```

完整运行中的跳过数与上述 `-rs` 原因审计一致。7 项 deselected 来自 CI 的明确 marker 排除。
警告主要为 Pandas／Arrow 及分箱依赖的弃用提示、openpyxl 对模板扩展的提示、
图形回归中备用字体缺少部分中文字形，以及旧断言参数名称的弃用提示。
没有将环境失败误写为业务回归，也未静默跳过本轮专项。
两套最终源码完整运行独立记录，不与早期专项或中断运行拼接。

验收完成时的工作树为 12 个已跟踪文件修改和 3 个新文件，当时保持未提交、未推送。
本地日志、基线副本、快照、导出和站点产物留在忽略目录，不纳入源码交付。

### 边界与后续

本轮不修改 Home Credit 原始 CSV、预处理、模拟日期、五分区、候选、正式案例配置或分箱参数。
HTML／Excel 以静态内容及新进程恢复验证，浏览器和 Office 原生交互未作为本轮已完成验收。
独立审查另发现真实字符串 group=`*` 的既有入口边界：该值在旧版的 PSI 列投影中也会触发
Polars `select("*")` 展开导致的 DuplicateError。F09 未扩大修复这个分组入口问题，
也未改变既有标签；真实 group=`Total` 的局部状态与汇总列投影冲突已在本轮覆盖。
F09 验收时 shared 每批一次联合转换仍未完成。P06–P08 未启动，
监控、建模／Pipeline 和评分卡保持暂停功能迭代；本轮完成后停止。

## Shared 分箱批内复用（进行中） { #shared-batch-reuse }

本轮从 `feat/segmented-analysis-core` 的 `19670dc528debb9d5d330dada300ce664234c5a2`
开始，起始工作树干净。独立基线由 `git archive` 建立在
`.artifacts/segmented/shared-reuse-baseline/`，没有切换或回退开发工作树。
默认 Python 3.12.4 无法导入 mars。实际使用的 3.8.20／Polars 1.8.2／Pandas 2.0.3
和 3.12.13／Polars 1.42.0／Pandas 3.0.3 环境均从本仓库 `src/mars` 导入。

风险画像先准备 shared 规则，再按批转换完整当前人口及显式参考人口。
各 scope 从当前批的转换结果收集基础统计、参考和上线窗口，随后沿用原指标与报告装配。
基础聚合目前仍逐 scope 执行。数据画像 PSI 按批聚合客群频数，继续使用 F09 的
`calc_psi_from_stats`。联合批转换失败时，该批回到局部路径以保留失败隔离。
非 PSI 画像及 selector 的阶段调度仍逐 scope 执行。

固定规则计数测试覆盖 1、3、9 客群。每个特征批当前人口转换一次，显式参考人口转换一次。
数据画像两个单特征批在 90 行当前人口上实际执行 180 行；显式参考为 45 行时增加 90 行。
无显式参考时从当前频数取首组。拟合及 WOE 准备、raw KS 和相关性不计入上述转换次数。

独立基线的画像 PSI 对照覆盖 benchmark 独有客群、缺首组、混合类型和缺失／特殊箱开关：
48 张表和 8 份元数据一致，浮点容差为 1e-10。完整公开链小样本对照中，3／9 客群分别有
103／223 项快照表、元数据和规则比较通过。证据位于 `.artifacts/segmented/` 的
`profile-oracle-*`、`reuse-isolated-baseline/` 和 `reuse-isolated-current/`。

独占顺序测量使用 1,000 行、12 特征、均衡客群、显式参考、batch 16 和线程 1。
3／9 客群完整调用时间分别为基线 5.273／7.468 秒、当前 5.311／6.925 秒；
固定规则风险评估分别为 0.159／0.547 秒和 0.144／0.423 秒；
进程树采样峰值分别为 310.8／311.1 MiB 和 307.0／309.4 MiB。
各组仅运行一次，RSS 每 10 ms 采样，不能推断稳定吞吐或全宽容量。

中型对照使用 100,000 行、100 特征、3 个均衡客群、显式参考、batch 16 和线程 1。
基线及当前调用均 exit 0，103 项快照表、元数据和规则对照通过。完整调用分别为
138.408 秒和 113.894 秒，固定规则风险评估为 4.755 秒和 4.014 秒，进程树采样峰值
为 2,526.3 MiB 和 2,487.9 MiB。基线前段与测试和 Mypy 并发，当前侧条件不同，
不能计算普通提速比。固定规则评估未单独记录峰值 RSS；日志与结果在
`reuse-medium-baseline/` 和 `reuse-medium-current/`。

专项在 3.8 与 3.12 各有 205 项通过、1 项失败，失败发生于沙箱内 `os.link` 的
Windows WinError 5。两环境的风险回归各为 132 passed、1 deselected；最后边界修改后
各补测 9 项通过。这些窗口不能合并成一次完整通过。

两次启动完整基础测试后发现画像 PSI 的内部列名碰撞，源码仍需修复。
为避免混用运行中的源码，两套运行均主动中断，外层退出码为 1；
分别运行至约 16%／19% 和 8%／3%，不计为完整通过。
随后第三次完整运行期间审查发现风险画像客群列与分箱输出同名的回归，
两套运行分别约到 16% 和 19% 时主动中断，外层退出码均为 1。
第四次运行期间发现时间列同名回归，分别约到 8% 和 11% 时中断，外层退出码均为 1。
独占资源测量也早于内部列名修复，测量输入没有触发该碰撞，不能作为最终源码的完整资源验收。
合法客群列名与 PSI 内部字段冲突的修复后，3.8 和 3.12 的画像专项各为 76 passed、exit 0；
独立基线的 48 张表及 8 份元数据再次相符。一次 3.8 测试曾因运行时类型别名不兼容而失败，
修正后重跑通过。
风险画像的客群列名可能与分箱输出同名，现用独立内部身份保留客群。
两环境各 17 项专项通过，覆盖同名客群、多 chunk、权重金额和参考独有客群。
进一步独立审查发现时间列与分箱输出同名时，新路径会覆盖原始日期，现已从原始批次读取日期。
显式指定 `PYTHONPATH` 并核实 `mars.__file__` 后，确认目标、权重或金额列与分箱输出
同名时，基线的普通 evaluator 也抛出 DuplicateError；本轮未扩大修复该既有入口限制。
数字特征名与画像 PSI 内部索引冲突也已修复，两环境各 21 项专项通过。

```powershell
& .artifacts/segmented/env38-final/Scripts/python.exe -m pytest -q -m "not docs_ml and not optional_ml" --ignore=benchmarks/_artifacts --basetemp=.artifacts/segmented/reuse-final4-38
& D:/my_download_program/anaconda3/envs/mars312/python.exe -m pytest -q -m "not docs_ml and not optional_ml" --ignore=benchmarks/_artifacts --basetemp=.artifacts/segmented/reuse-final4-312
& D:/my_download_program/anaconda3/envs/mars312/python.exe -m ruff check src tests scripts benchmarks docs/snippets
& D:/my_download_program/anaconda3/envs/mars312/python.exe -m mypy src/mars
& D:/my_download_program/anaconda3/envs/mars312/Scripts/pydoclint.exe src/mars
& D:/my_download_program/anaconda3/envs/mars312/python.exe scripts/check_private_docstrings.py src/mars
```

最终源码的完整基础命令使用上述环境变量，分别指定 `reuse-final4-38` 和
`reuse-final4-312` 为 basetemp。3.8 为 1,757 passed、13 skipped、7 deselected，
556.54 秒，exit 0；3.12 为 1,856 passed、3 skipped、7 deselected，623.75 秒，exit 0。
日志为 `.artifacts/segmented/reuse-final4-38.log` 和 `reuse-final4-312.log`。
测试在沙箱外运行，以允许报告保存使用 Windows 硬链接。
最终源码的 Ruff、全包 Mypy 和 public/private docstring 均 exit 0；private 检查有 6 条既有建议警告。
最终小样本资源矩阵使用 1,000 行、12 特征，两侧使用相同风险与画像 PSI 批宽。
每侧调用与固定规则评估连续运行，报告和规则共存。每组只运行一次。

| 客群与参考 | batch／线程 | 完整调用基线 → 当前（秒） | 固定规则风险基线 → 当前（秒） | 进程树峰值基线 → 当前（MiB） |
| --- | --- | --- | --- | --- |
| 1 客群，首组 | 1／4 | 6.586 → 6.732 | 0.124 → 0.150 | 324.1 → 330.2 |
| 3 客群，显式 | 16／1 | 5.072 → 5.502 | 0.180 → 0.184 | 307.5 → 309.6 |
| 9 客群，显式 | 16／1 | 7.053 → 6.770 | 0.426 → 0.366 | 310.5 → 313.1 |
| 9 客群不均衡，首组 | 100／4 | 8.993 → 8.754 | 0.393 → 0.450 | 329.8 → 340.4 |

各侧 worker 均 exit 0，前 3 组完整对照 exit 0。最后一组首次比较器遗漏状态表的
`group` 业务键，整条比较命令 exit 1；修正后对原快照重比 156 项通过、exit 0。
原始失败和重比结果均保留在 `.artifacts/segmented/reuse-final-b100-current/`。
这些结果不能推断普遍提速，固定规则阶段未单独采样 RSS。
最终源码的中型顺序复测使用 100,000 行、100 特征、3 个均衡客群、显式参考、
风险和画像 PSI 批宽 16、线程 1。两侧进程均 exit 0，103 项快照表、元数据和规则对照通过。
基线 → 当前的完整调用为 114.304 → 111.661 秒，风险画像为 6.890 → 6.228 秒，
数据画像为 4.250 → 3.345 秒，selector 为 98.323 → 97.048 秒，
固定规则风险评估为 4.509 → 4.219 秒，10 ms 进程树采样峰值为 2,565.3 → 2,484.6 MiB。
每侧仅一次，固定规则阶段未单独采样峰值，系统背景负载仍可能影响结果。
原始日志和 JSON 保存在 `reuse-final-medium-baseline/` 和 `reuse-final-medium-current/`。
P02 的优化项仍待补齐剩余验收范围，暂不勾选。

执行观察另使用 300 行当前人口、180 行参考、4 特征、3 客群、batch 2 和 raw KS。
两批合计，风险画像当前转换从 8 次／1,200 行变为 2 次／600 行，
参考转换从 8 次／720 行变为 2 次／360 行；画像 PSI 和固定规则风险评估相同。
风险及画像的规则拟合各为 1 次／180 行，风险规则 WOE 准备转换为 2 次／360 行，
两侧一致。raw KS 公开入口为 1 次／300 行，底层 scope 计算为 4 次／600 行；
selector 相关性入口为 4 次／600 输入行，两侧一致。selector 阶段转换未池化。
观察模式不能将 LazyFrame 构造次数等同于实际执行次数；行级执行合同另由专项验证。
原始记录位于 `reuse-counts-baseline-v3/` 和 `reuse-counts-current-v3/`。
尚未逐项测量基础聚合及报告装配时间，固定规则评估没有独立峰值 RSS。
没有重新运行 Home Credit 代表候选或验收全宽容量；既有数据与正式案例配置未改动。
完整基础测试覆盖报告保存恢复、查询、静态 HTML／Excel／PNG 和规则恢复等既有用例，
另新增同一 shared 报告的集中消费回归，两环境各 1 passed、exit 0：
新进程验证分页、reference 重放、预算拒绝及 AI context、detail／status、静态 Excel、
HTML 和 PNG；另起进程恢复规则 JSON 并禁止 fit 后评估。原生报告另验证单 scope 22 列。
首次测试曾因预算不足及对快照 Excel 使用错误 API 失败；按既有合同修正测试后通过。
本轮未验收浏览器或 Office 原生交互。
selector 仍独立执行各 scope 的阶段编排，本轮没有实现筛选阶段池化。
P06–P08、Linear／Importance 独立客群和暂停模块的功能迭代均未启动。
