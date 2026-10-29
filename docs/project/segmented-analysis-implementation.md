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
