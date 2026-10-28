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
