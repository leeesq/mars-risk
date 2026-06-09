# 分箱与风险评估

分箱评估用于观察单个特征与 target 的关系，包括 IV、KS、AUC、Lift、坏账率、PSI、缺失率和分箱趋势等指标。MARS 支持原生分箱、轻量最优分箱和数学规划最优分箱，也支持将分箱规则继续复用到转换、监控和报表链路。

## 高层入口：`profile_risk`

```python
from mars.analysis import profile_risk

risk_profile = profile_risk(
    df,
    target="target",
    features=["income", "utilization", "segment"],
    group_col="month",
    binning_type="native",
    binner_params={"method": "quantile", "n_bins": 5},
    plot=False,
)
```

返回值是 `MarsRiskProfile(report, binner, targets, metadata)`：

- `report`：`MarsEvaluationReport`，保存汇总表、明细表、趋势表和导出能力。
- `binner`：本次拟合或复用的分箱器。
- `targets`：本次评估的 target 列表。
- `metadata`：运行上下文。

## 评估器入口：`MarsBinEvaluator`

```python
from mars.analysis import MarsBinEvaluator

evaluator = MarsBinEvaluator(
    binning_type="native",
    binner_params={"method": "quantile", "n_bins": 5},
)

risk_profile = evaluator.evaluate(
    df,
    target="target",
    features=["income", "utilization"],
    group_col="month",
)
```

`MarsBinEvaluator` 不把 target 和 features 放进构造函数。每次 `evaluate` 都可以传入新的数据、目标列和特征范围，避免复用对象时状态串扰。

## 分箱器共享能力

`MarsNativeBinner`、`MarsLiteOptBinner` 和 `MarsOptimalBinner` 都继承 `MarsBinnerBase`。
子类只负责拟合各自的分箱规则，规则转换、WOE 转换、分箱效果评估、规则裁剪和序列化由基类统一提供。

| 共享方法 | 用途 |
| --- | --- |
| `fit_transform` / `transform` | 生成 index、label 或 WOE 形式的分箱结果 |
| `profile_bin_performance` | 基于已拟合规则计算分箱明细、IV、WOE 和坏账率 |
| `to_dict` / `from_dict` | 保存和恢复分箱规则 |
| `prune` | 只保留指定特征的分箱规则 |

用户通常只需要实例化具体子类，例如 `MarsNativeBinner` 或 `MarsLiteOptBinner`；
这些共享能力会自动继承，无需单独操作 `MarsBinnerBase`。

## 原生分箱：`MarsNativeBinner`

```python
from mars.feature import MarsNativeBinner

binner = MarsNativeBinner(
    method="quantile",
    n_bins=5,
    special_values=[-999],
    merge_small_bins=True,
    remove_empty_bins=True,
)

binner.fit(X, y, cat_features=["segment"])
X_bin = binner.transform(X, return_type="index")
X_woe = binner.transform(X, return_type="woe")
```

常用 `method`：

- `quantile`：等频分箱，默认方法，可无标签运行。
- `uniform`：等宽分箱，可无标签运行。
- `cart`：基于 target 的监督分箱，必须传入 `y`。

原生分箱支持处理类别特征，缺失值、`NaN`、自定义 `missing_values` 和业务特殊值 `special_values` 会独立隔离，不挤占正常分箱数量。

## 轻量最优分箱：`MarsLiteOptBinner`

```python
from mars.feature import MarsLiteOptBinner

lite_binner = MarsLiteOptBinner(
    n_bins=6,
    n_prebins=50,
    monotonic_trend="auto",
    prebinning_method="quantile",
)

lite_binner.fit(X, y, cat_features=["segment"])
```

`MarsLiteOptBinner.fit(X, y, ...)` 要求 `y` 必填。它先使用原生预分箱生成细箱，再在预分箱统计表上做趋势约束合并，支持 `ascending`、`descending`、`peak`、`valley` 和 `auto`。`fitted_trends_` 会记录每个数值特征最终采用的趋势，`candidate_scores_` 会记录 auto 候选评分。

## 最优分箱：`MarsOptimalBinner`

```python
from mars.feature import MarsOptimalBinner

optimal_binner = MarsOptimalBinner(
    max_n_bins=6,
    min_bin_size=0.05,
    time_limit=1,
)

optimal_binner.fit(X, y, cat_features=["segment"])
X_opt_woe = optimal_binner.transform(X, return_type="woe")
```

`MarsOptimalBinner.fit(X, y, ...)` 要求 `y` 必填。最优分箱支持类别合并，适合降低高基数类别带来的不稳定性。

## 分箱评估报告

```python
summary = risk_profile.report.summary_table
detail = risk_profile.report.detail_table
trends = risk_profile.report.trend_tables

risk_profile.report.write_excel("risk_report.xlsx", engine="openpyxl")
risk_profile.report.write_html("risk_report.html")
```

| 表 | 用途 |
| --- | --- |
| `summary_table` | 特征级指标汇总，用于排序、筛选和报告首页 |
| `detail_table` | 分箱明细，包含箱内样本数、坏账率、WOE、IV 等 |
| `trend_tables` | 按时间或分组展开的 PSI、缺失率、坏账率等趋势 |
| `missing_by_day_table` | 按日缺失率趋势，适合排查数据链路异常 |

## PSI 口径

分箱评估可以通过 `psi_include_missing` 和 `psi_include_special` 控制 PSI 是否纳入缺失箱和特殊值箱。监控场景通常建议默认不纳入缺失箱，因为缺失率会单独监控。

```python
risk_profile = profile_risk(
    df,
    target="target",
    features=["income"],
    group_col="month",
    psi_include_missing=False,
    psi_include_special=False,
)
```
