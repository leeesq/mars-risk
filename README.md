# MARS

<div align="center">

**Modeling Analysis Risk Score**

从 `Data Profile`、`Binning`、`Evaluation`、`Selection` 到 `Excel Export`  
一套围绕 `polars`、评分卡与信贷风控工作流打造的 Python 工具链

[![PyPI version](https://img.shields.io/pypi/v/mars-risk?style=flat-square)](https://pypi.org/project/mars-risk/)
[![Python Versions](https://img.shields.io/pypi/pyversions/mars-risk?style=flat-square)](https://pypi.org/project/mars-risk/)
[![CI](https://github.com/leeesq/mars-risk/actions/workflows/test.yml/badge.svg)](https://github.com/leeesq/mars-risk/actions/workflows/test.yml)
[![License](https://img.shields.io/github/license/leeesq/mars-risk?style=flat-square)](LICENSE)

</div>

```text
 __________________________________________________________________________
    __  ___ ___    ____  _____
   /  |/  //   |  / __ \/ ___/
  / /|_/ // /| | / /_/ /\__ \
 / /  / // ___ |/ _, _/___/ /
/_/  /_//_/  |_/_/ |_|/____/

 MODELING ANALYSIS RISK SCORE
 __________________________________________________________________________
```

> `Profile -> Bin -> Evaluate -> Select -> Export`
>
> MARS 不是一个泛化的 AutoML 框架。  
> 它更像一套为风控分析师、评分卡建模工程师和特征治理场景准备的“日常工作台”。

## MARS 是什么

MARS 是一个面向信贷风控、评分卡建模和特征稳定性监控的 Python 工具库，目标很明确：

1. 用 `polars` 做更快、更整洁的数据与特征处理。
2. 用 sklearn 风格 API 组织分箱、评估、筛选等核心流程。
3. 把风控工作中高频但分散的动作，收敛成一套能复用、能导出、能落地的工具链。

如果你的日常工作里经常出现这些动作：

- 看数据质量和画像趋势
- 做数值/类别特征分箱
- 算 IV、KS、AUC、PSI、单调性、风险一致性
- 评估一批候选特征并导出 Excel 报表
- 做一轮可解释、可回放的特征筛选

那 MARS 想解决的就是这一整段链路。

## 为什么是 MARS

### 1. Polars-first，但不强迫你放弃 Pandas

- 核心计算优先走 `polars`
- 支持直接传入 Pandas DataFrame
- 对于核心报告对象，遵循“输入是什么，输出就尽量保持什么”的约定
- 展示层和 Excel 导出层内部按需回退到 Pandas，不把这种回退扩散到计算层

### 2. 更贴近风控工作流

MARS 不只做“一个分箱器”或“一个 PSI 函数”，而是把常见风控流程拆成四类组件：

| 工作阶段 | 主 API | 解决什么问题 | 典型产出 |
| --- | --- | --- | --- |
| 数据画像 | `MarsDataProfiler` | 缺失、零值、众数、分布、趋势、PSI | `MarsProfileReport` |
| 分箱转换 | `MarsNativeBinner` / `MarsOptimalBinner` | 连续/类别分箱、WOE 映射、SQL 生成 | 分箱结果、映射表、SQL |
| 特征评估 | `MarsBinEvaluator` / `profile_risk` | IV、KS、AUC、PSI、单调性、趋势报表 | `MarsEvaluationReport` |
| 特征筛选 | `MarsStatsSelector` | 质量筛选、粗筛、精筛、稳定性、相关性漏斗 | `selected_features_`、筛选报表 |

### 3. 不只算指标，还重视“拿去用”

- 报告对象支持 `show_*` 交互式查看
- 支持 Excel 导出
- 分箱器支持导出 SQL `CASE WHEN`
- 结果对象保留后续复用能力，而不是“一次性函数输出”

## 适合什么场景

MARS 目前尤其适合这些任务：

- 信贷风控特征初筛
- 评分卡开发中的分箱、WOE、特征监控
- 月度/周度的特征稳定性巡检
- 风控分析结果导出给业务、策略、模型管理或审计团队
- 在 Pandas 项目中逐步迁移到 Polars 的中间阶段

## 当前项目状态

MARS 当前仍处于 `0.x` 阶段。

这意味着：

- API 已经可以用于实际分析和日常工作
- 项目已经补上了 pytest、CI、教程、README、extras 和一批开箱问题修复
- 但它还在持续做“开源成熟化”工作，尤其是：
  - 深化测试覆盖
  - 继续清理带有 Pandas 风格的 Polars 实现
  - 统一文档、注释和输出体验

如果你想找的是一个“已经有成熟社区和稳定大版本承诺”的项目，MARS 还没走到那个阶段。  
如果你想找的是一个“风控向、能用、可改、可继续打磨”的工具库，它已经进入一个挺值得继续投入的区间。

## 安装

### 基础安装

```bash
pip install mars-risk
```

### 安装带 Excel 导出的能力

```bash
pip install "mars-risk[excel]"
```

### 安装带绘图能力

```bash
pip install "mars-risk[plot]"
```

### 安装 Notebook 支持

```bash
pip install "mars-risk[notebook]"
```

### 安装完整开发环境

```bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
pip install -e ".[dev]"
```

### Python 版本要求

- `Python >= 3.10`

## 快速开始

下面这组示例尽量覆盖 MARS 的主线工作流。

### 1. 准备一份小型样例数据

```python
import polars as pl

df = pl.DataFrame(
    {
        "month": [
            "2024-01", "2024-01", "2024-01", "2024-01",
            "2024-02", "2024-02", "2024-02", "2024-02",
            "2024-03", "2024-03", "2024-03", "2024-03",
        ],
        "income": [3200, 3600, -999, None, 3300, 4200, -999, 5800, 3400, 4300, None, 6100],
        "utilization": [0.12, 0.18, 0.52, 0.61, 0.14, 0.29, 0.54, 0.58, 0.16, 0.31, 0.56, 0.63],
        "segment": ["new", "repeat", "vip", "vip", "new", "repeat", "vip", "vip", "new", "repeat", "vip", "vip"],
        "target": [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    }
)
```

### 2. 做一次数据画像

```python
from mars.analysis import MarsDataProfiler

profiler = MarsDataProfiler(
    df,
    missing_values=[-999],
)

profile_report = profiler.generate_profile(
    profile_by="month",
    config_overrides={
        "enable_sparkline": False,
        "dq_metrics": ["missing", "zeros"],
        "stat_metrics": ["mean", "psi"],
    },
)

overview = profile_report.overview_table
missing_trend = profile_report.dq_tables["missing"]
psi_trend = profile_report.stats_tables["psi"]
```

你通常会从这里开始：

- `overview` 看整张表的全局画像
- `dq_tables["missing"]` 看缺失率趋势
- `stats_tables["psi"]` 看稳定性变化

导出画像报表：

```python
profile_report.write_excel("mars_profile.xlsx")
```

```python
from mars.analysis import profile_stats

quick_profile = profile_stats(
    df,
    metrics=["missing", "mean"],
    features=["income", "utilization"],
    profile_by="month",
    missing_values=[-999],
)

quick_profile.show_overview()
quick_profile.show_trend("missing")
```

### 3. 一键做特征评估

`profile_risk()` 的返回值是：

```python
(report, evaluator)
```

其中：

- `report` 负责承载汇总表、趋势表、明细表和报表导出
- `evaluator` 保留拟合后的分箱器和评估上下文，便于继续复用

```python
from mars.analysis import profile_risk

eval_report, evaluator = profile_risk(
    df,
    target="target",
    features=["income", "utilization", "segment"],
    profile_by="month",
    binning_type="native",
    n_bins=4,
    binner_kwargs={"method": "quantile"},
    plot=False,
)

summary = eval_report.summary_table
detail = eval_report.detail_table
trend_psi = eval_report.trend_tables["psi"]
```

导出评估报表：

```python
eval_report.write_excel("mars_evaluation.xlsx", engine="openpyxl")
eval_report.write_html("mars_evaluation.html")
```

### 4. 直接使用分箱器

```python
from mars.feature import MarsNativeBinner

X = df.select(["income", "utilization", "segment"])
y = df.get_column("target")

binner = MarsNativeBinner(
    method="quantile",
    n_bins=4,
    cat_features=["segment"],
    special_values=[-999],
)

binner.fit(X, y)
X_binned = binner.transform(X, return_type="index")
X_woe = binner.transform(X, return_type="woe")
income_mapping = binner.get_bin_mapping("income")
```

如果你希望把分箱逻辑拿到 SQL 里部署：

```python
sql = binner.generate_sql(
    features=["income", "utilization"],
    table_prefix="t",
    return_type="woe",
)
```

```python
from mars.scoring import build_scorecard

scorecard = build_scorecard(
    binner=binner,
    coefficients={"income": -0.35, "utilization": 0.82},
    intercept=-1.1,
    pdo=50,
    base_score=600,
    base_odds=20,
)

scorecard.write_excel("mars_scorecard.xlsx")
sql_score = scorecard.generate_sql(table_prefix="t", score_name="credit_score")
```

### 6. 做一轮特征筛选

```python
from mars.feature import MarsStatsSelector

selector = MarsStatsSelector(
    target="target",
    profile_by="month",
    rough_iv_thr=0.01,
    iv_thr=0.02,
    psi_thr=0.25,
    rc_thr=0.5,
)

selector.fit(df)

selected_features = selector.selected_features_
selector.export_selector_report("mars_selector_report.xlsx")
selector.save_selector_lists("mars_lists.json")
```

`MarsStatsSelector` 内部是一个漏斗式筛选流程，典型阶段包括：

- 数据质量校验
- 原生分箱粗筛
- 最优分箱精筛
- PSI 稳定性过滤
- 风险一致性过滤
- 相关性去重

## 输入输出约定

这是 README 里最值得提前讲清楚的一部分。

### Polars / Pandas 约定

- 传入 `Polars DataFrame`，核心报告对象会优先保持 `Polars`
- 传入 `Pandas DataFrame`，核心报告对象会优先保持 `Pandas`
- 展示层（`Styler` / HTML / Excel）内部会按需转为 Pandas，这是为了兼容样式系统，不代表计算层退回 Pandas

### `profile_risk()` 的返回值

始终返回：

```python
(MarsEvaluationReport, MarsBinEvaluator)
```

不要把它当成只返回一张表的函数来用。

### 无标签模式

如果你把 `target=None` 传给 `profile_risk()`，MARS 会进入无标签模式：

- 仍然可以做分布和稳定性分析
- 仍然可以产出 PSI 等分布类指标
- 但不会再产生依赖真实标签的区分度指标

这对于“监控一批线上样本分布是否漂移”很有用。

## 如何选择分箱器

### `MarsNativeBinner`

推荐作为默认起点。

适合：

- 日常宽表批量评估
- 希望先把流程跑通
- 更看重速度和工程稳定性
- 需要快速得到 WOE / 分箱映射 / SQL

典型配置：

```python
MarsNativeBinner(method="quantile", n_bins=10)
```

### `MarsOptimalBinner`

适合更严肃的评分卡建模场景。

适合：

- 对单调性和切点质量要求更高
- 愿意接受更高的计算成本
- 需要更“评分卡味”的分箱约束

在一键评估里使用：

```python
report, evaluator = profile_risk(
    df,
    target="target",
    profile_by="month",
    binning_type="opt",
    plot=False,
)
```

## 报告对象能做什么

### `MarsProfileReport`

主要入口：

- `overview_table`
- `dq_tables`
- `stats_tables`
- `show_overview()`
- `show_trend(metric)`
- `write_excel(...)`

适合回答这些问题：

- 哪些列缺失高、零值高、众数高
- 哪些特征在不同月份分布波动很大
- 哪些数值统计量在跨期上不稳定

### `MarsEvaluationReport`

主要入口：

- `summary_table`
- `trend_tables`
- `detail_table`
- `show_summary()`
- `show_trend(metric)`
- `write_excel(path, engine="openpyxl")`

适合回答这些问题：

- 哪些特征 IV / KS / AUC 表现更好
- 哪些特征最大 PSI 偏高
- 哪些特征单调性或风险一致性不理想
- 每个分箱的样本占比、坏率、Lift、WOE 是什么

## 公开 API 概览

### `mars.analysis`

| API | 说明 |
| --- | --- |
| `MarsDataProfiler` | 数据画像与趋势分析 |
| `MarsProfileConfig` | 画像配置对象 |
| `MarsProfileReport` | 画像报告对象 |
| `MarsBinEvaluator` | 特征评估器 |
| `MarsEvaluationReport` | 评估报告对象 |
| `profile_stats` | 轻量统计画像入口，适合快速看缺失率/均值等指标 |
| `profile_risk` | 一键评估入口 |

### `mars.feature`

| API | 说明 |
| --- | --- |
| `MarsNativeBinner` | 原生高性能分箱器 |
| `MarsOptimalBinner` | 带约束的最优分箱器 |
| `MarsStatsSelector` | 漏斗式特征筛选器 |

### `mars.scoring`

| API | 说明 |
| --- | --- |
| `build_scorecard` | 基于已拟合分箱器和 LR 系数生成评分卡 |
| `MarsScorecard` | 评分卡结果对象，支持表格导出与 SQL 生成 |

### 参数兼容提醒

`MarsBinEvaluator` 现在推荐使用：

```python
MarsBinEvaluator(..., binning_type="native")
```

旧参数：

```python
MarsBinEvaluator(..., bining_type="native")
```

仍然兼容，但已经是弃用入口，不建议在新代码里继续使用。

## 教程与仓库资源

推荐阅读顺序：

1. [tutorial/quickstart.md](tutorial/quickstart.md)
2. [tutorial/performance_audit.md](tutorial/performance_audit.md)
3. [tutorial/benchmark_synthetic.py](tutorial/benchmark_synthetic.py)

仓库里同时还保留了一些历史 notebook、样例文件和导出产物，用于开发过程中的验证与参考。  
如果你是第一次接触这个项目，优先看上面的三份内容就够了。

## 可选依赖说明

| Extra | 用途 |
| --- | --- |
| `excel` | `openpyxl`、`xlsxwriter`、`xlwings`，用于 Excel 导出 |
| `plot` | `matplotlib`、`seaborn`，用于风险趋势图绘制 |
| `notebook` | `jupyterlab`，用于 Notebook 交互体验 |
| `ml` | `xgboost`、`lightgbm`、`catboost` |
| `dev` | pytest、格式化、导出、绘图与基准测试相关依赖 |

## 测试与开发

当前仓库已经包含基于合成数据的 pytest 覆盖，重点保护：

- `MarsNativeBinner` 的分箱与映射行为
- `MarsDataProfiler` 的画像流程和报告输出
- `MarsBinEvaluator` / `profile_risk` 的核心评估路径
- 多目标场景和 Pandas/Polars 返回类型约定
- Excel 模板资源与导出烟测

本地运行测试：

```bash
python -m pytest -q
```

运行轻量 benchmark：

```bash
python tutorial/benchmark_synthetic.py --rows 1000 --repeats 1
```

## 常见问题

### 1. `profile_risk()` 为什么返回两个对象

因为 `report` 和 `evaluator` 承担的职责不同：

- `report` 负责结果承载、展示和导出
- `evaluator` 保留分箱器和后续分析能力

这让“看结果”和“继续复用规则”可以同时成立。

### 2. 不安装 `plot` extra 可以用吗

可以。

核心分箱、评估、画像、筛选都可以工作。  
只有在调用绘图相关方法时才需要安装 `plot` extra。

### 3. Excel 导出一定要本地装 Excel 吗

不一定。

- `openpyxl` 路径不要求本地安装 Excel，跨平台更稳
- `xlwings` 路径更适合本机有 Excel 的环境，格式保留能力更强

### 4. 我现在是 Pandas 项目，能直接接吗

可以。

MARS 支持直接传入 Pandas DataFrame。  
如果以后你希望把更多计算迁到 Polars，MARS 也比较适合作为过渡层。

### 5. 这是一个已经稳定的大版本项目吗

还不是。

MARS 现在更像一个已经进入“可持续打磨期”的 `0.x` 项目：  
能用、能测、能导出、能继续优化，但还在不断补齐开源成熟度。

## 接下来会继续做什么

当前比较明确的后续方向有：

- 继续补测试覆盖和回归护栏
- 继续清理带 Pandas 风格的 Polars 实现
- 继续统一注释、文档字符串和输出文案
- 优化 README、教程和示例的一致性
- 提升 `binner / evaluator / selector` 这条主链路的可维护性

## 参与方式

欢迎：

- 提 issue
- 提 PR
- 提出真实业务中的使用反馈
- 提出你希望优先补的示例、教程或导出能力

如果你在使用中踩到了 API 不一致、报表体验不顺、Polars 性能问题，或者只是觉得 README 还不够清楚，这类反馈都很有价值。

## License

See [LICENSE](LICENSE).
