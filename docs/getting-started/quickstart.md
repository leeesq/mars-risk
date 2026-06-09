# 快速开始

本页用一个小样本串起数据画像、分箱评估、分箱器、特征筛选、特征/模型监控、Modeling 建模和 Pipeline 编排。

## 准备样本

```python
import polars as pl

df = pl.DataFrame(
    {
        "apply_dt": [
            "2024-01-03", "2024-01-10", "2024-01-17", "2024-01-24",
            "2024-02-03", "2024-02-10", "2024-02-17", "2024-02-24",
            "2024-03-03", "2024-03-10", "2024-03-17", "2024-03-24",
        ],
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

## 数据画像

```python
from mars.analysis import MarsDataProfiler, profile_stats

profile_report = MarsDataProfiler(missing_values=[-999]).generate_profile(
    df,
    group_col="month",
    config_overrides={
        "enable_sparkline": False,
        "dq_metrics": ["missing", "zeros"],
        "stat_metrics": ["mean", "psi"],
    },
)

quick_report = profile_stats(
    df,
    metrics=["missing", "mean"],
    features=["income", "utilization"],
    group_col="month",
    missing_values=[-999],
)
```

常用结果：

```python
profile_report.overview_table
profile_report.dq_tables
profile_report.stats_tables
```

## 分箱评估

```python
from mars.analysis import profile_risk

risk_profile = profile_risk(
    df,
    target="target",
    features=["income", "utilization", "segment"],
    group_col="month",
    binning_type="native",
    binner_params={"method": "quantile", "n_bins": 4},
    plot=False,
)

eval_report = risk_profile.report
binner = risk_profile.binner
summary = eval_report.summary_table
```

`profile_risk()` 返回 `MarsRiskProfile(report, binner, targets, metadata)`。`report` 用于查看指标和导出报表，`binner` 用于复用分箱规则。

## 分箱器

```python
from mars.feature import MarsLiteOptBinner, MarsNativeBinner

X = df.select(["income", "utilization", "segment"])
y = df.get_column("target")

binner = MarsNativeBinner(method="quantile", n_bins=4, special_values=[-999])
binner.fit(X, y, cat_features=["segment"])

lite_binner = MarsLiteOptBinner(n_bins=4, n_prebins=30, monotonic_trend="auto")
lite_binner.fit(X, y, cat_features=["segment"])

X_bin = binner.transform(X, return_type="index")
X_woe = binner.transform(X, return_type="woe")
income_mapping = binner.get_bin_mapping("income")
```

底层分箱器采用 `X, y` 风格。`method="cart"` 和 `MarsLiteOptBinner` 需要传入 `y`，`method="quantile"` 和 `method="uniform"` 可以无标签运行。

## 特征筛选

```python
from mars.feature import MarsStatsSelector

selector = MarsStatsSelector(
    missing_thr=0.9,
    iv_thr=0.01,
    psi_thr=0.25,
    skip_fine_scan=True,
)

selector.fit(
    df,
    target="target",
    features=["income", "utilization", "segment"],
    group_col="month",
    feature_data_source={
        "user_profile": ["income"],
        "credit_usage": ["utilization"],
        "application": ["segment"],
    },
)

selected_features = selector.selected_features_
selection_report = selector.get_eval_report(df)
```

## 特征/模型监控

```python
from mars.monitoring import MarsMonitor, generate_monitoring_alert

monitor_df = df.with_columns((pl.col("utilization") * 100).alias("model_score"))

monitor_report = MarsMonitor(
    binner_params={"method": "quantile", "n_bins": 4},
).monitor(
    monitor_df,
    features=["model_score", "income", "utilization"],
    target="target",
    group_col="month",
    trend_column_order="desc",
)

alert_text = generate_monitoring_alert(
    monitor_report,
    score_key="model_score",
    model_features=["income", "utilization"],
)
```

`MarsMonitor` 是特征/模型监控的通用指标计算层。`trend_column_order="desc"` 可以让最新时间列展示在趋势宽表最前面，报警摘要会按 report 记录的顺序识别基准期和最新期。

## Modeling / Pipeline

Modeling 建模和 Pipeline 编排仍在快速迭代中，接口约定、结果对象和调参参数后续可能发生较大变化。当前 Modeling 支持 XGBoost、LightGBM、CatBoost 和 Logistic Regression。

如果希望把筛选、可选 WOE 和建模串成一条链路，可以使用 `mars.pipeline`。树模型通常走“筛选 -> 建模”：

```python
from mars.feature import MarsStatsSelector
from mars.pipeline import MarsModelingPipeline, MarsModelingStep, MarsSelectionStep

pipeline = MarsModelingPipeline(
    target="target",
    features=["income", "utilization", "segment"],
    steps=[
        MarsSelectionStep(
            name="stats_filter",
            selector=MarsStatsSelector(iv_thr=0.02),
        ),
        MarsModelingStep(
            name="modeling",
            model_type="lgb",
            time_col="apply_dt",
            split_ratios={"train": 0.6, "val": 0.2, "oot": 0.2},
            tune_params={"n_trials": 20, "artifact_dir": None},
        ),
    ],
)

pipeline_result = pipeline.fit(df)
```

LR / 评分卡链路可以在中间显式加入 `MarsWOEBinningStep`，让后续步骤消费 `*_woe` 特征。也可以直接使用 `mars.modeling` 做单次建模会话：

```python
from mars.modeling import MarsModelingSession
from mars.modeling.tuning import MarsModelReplayRunner

session = MarsModelingSession(
    model_type="lgb",
    features=["income", "utilization", "segment"],
    target="target",
    categorical_features=["segment"],
    optimize_metric="ks",
    seed=1206,
)

modeling_df = session.slice(
    df,
    time_col="apply_dt",
    split_ratios={"train": 0.6, "val": 0.2, "oot": 0.2},
)

tuning_result = session.tune(
    modeling_df,
    n_trials=20,
    artifact_dir=None,
)

replay_result = MarsModelReplayRunner().run(
    tuning_result,
    modeling_df,
    top_k=3,
    sort_metric="ks",
)
```

## 报表导出

```python
profile_report.write_excel("mars_profile.xlsx")
eval_report.write_excel("mars_evaluation.xlsx", engine="openpyxl")
eval_report.write_html("mars_evaluation.html")
```

各模块 report 也可以直接读取结构化表，接入内部看板或交给 Agent 做二次加工。
