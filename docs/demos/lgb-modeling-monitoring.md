# LightGBM 建模与监控 Demo

本页展示一个静态化的 notebook 结果样例：用合成信贷风控宽表训练 LightGBM，再对模型结果和核心特征做特征监控和模型监控。页面中的表格是演示输出节选，后续可以替换为完整 notebook 跑出的真实结果。

## 数据集设定

| 项目 | 设定 |
| --- | --- |
| 样本量 | 30,000 |
| 特征数 | 100 |
| 时间跨度 | 2024-01 到 2025-06，按月切片 |
| 目标变量 | `target`，最新月份部分样本为空，模拟尚未到表现期 |
| 模型 | LightGBM，`model_type="lgb"` |
| 监控对象 | 模型输出、核心模型特征、漂移特征、高缺失特征 |

示例数据包含收入、负债率、额度使用率、查询次数、历史逾期、渠道、地区、职业等业务字段，也包含匿名宽表特征。部分字段被注入分布漂移、缺失异常和特殊值，用来展示监控结果。

## 建模代码片段

```python
from mars.modeling import MarsModelingSession
from mars.modeling.tuning import MarsModelReplayRunner

session = MarsModelingSession(
    model_type="lgb",
    features=model_features,
    target="target",
    categorical_features=["channel", "region", "occupation"],
    optimize_metric="ks",
    seed=2026,
)

modeling_df = session.slice(
    df,
    time_col="apply_dt",
    split_ratios={"train": 0.6, "val": 0.2, "oot": 0.2},
)

tuning_result = session.tune(
    modeling_df,
    n_trials=10,
    history_path=None,
)

replay_result = MarsModelReplayRunner().run(
    tuning_result,
    modeling_df,
    top_k=3,
    sort_metric="ks",
)
```

## Modeling Pipeline 输出节选

| trial | dataset | auc | ks | lift_top10 | logloss |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | train | 0.781 | 0.425 | 2.36 | 0.364 |
| 7 | val | 0.753 | 0.384 | 2.12 | 0.381 |
| 7 | oot | 0.741 | 0.361 | 1.96 | 0.392 |
| 3 | val | 0.747 | 0.371 | 2.04 | 0.386 |
| 9 | val | 0.744 | 0.366 | 2.01 | 0.389 |

| feature | importance | data_source |
| --- | ---: | --- |
| utilization | 0.184 | credit_behavior |
| overdue_cnt_12m | 0.151 | credit_history |
| query_cnt_30d | 0.119 | application |
| debt_ratio | 0.096 | asset_liability |
| anon_risk_017 | 0.083 | anonymous |
| channel | 0.051 | application |

## 监控代码片段

```python
from mars.monitoring import MarsMonitor, generate_monitoring_alert

monitor_report = MarsMonitor(
    binner_params={"method": "quantile", "n_bins": 5},
    psi_include_missing=False,
    psi_include_special=False,
).monitor(
    scored_df,
    features=["model_score", *model_features],
    target="target",
    group_col="month",
    trend_column_order="desc",
)

alert_text = generate_monitoring_alert(
    monitor_report,
    score_key="model_score",
    model_features=model_features,
)
```

## 监控汇总节选

| feature | data_source | psi_max | missing_max | observed_rate_min | bad_rate_max |
| --- | --- | ---: | ---: | ---: | ---: |
| model_score | model_output | 0.286 | 0.000 | 0.42 | 0.231 |
| utilization | credit_behavior | 0.192 | 0.018 | 0.42 | 0.219 |
| query_cnt_30d | application | 0.146 | 0.011 | 0.42 | 0.205 |
| anon_drift_004 | anonymous | 0.318 | 0.025 | 0.42 | 0.188 |
| income | application | 0.058 | 0.071 | 0.42 | 0.176 |
| high_missing_002 | anonymous | 0.074 | 0.214 | 0.42 | 0.161 |

## target 表现覆盖率

| month | count | observed_count | pending_count | observed_rate | observed_bad_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025-06 | 1,654 | 694 | 960 | 0.420 | 0.231 |
| 2025-05 | 1,671 | 1,671 | 0 | 1.000 | 0.214 |
| 2025-04 | 1,681 | 1,681 | 0 | 1.000 | 0.206 |
| 2025-03 | 1,659 | 1,659 | 0 | 1.000 | 0.198 |

## 分箱趋势节选

| feature | bin_label | 2025-06 | 2025-05 | 2025-04 | 2025-03 |
| --- | --- | ---: | ---: | ---: | ---: |
| model_score | (-inf, 0.18] | 0.112 | 0.168 | 0.174 | 0.181 |
| model_score | (0.18, 0.31] | 0.174 | 0.203 | 0.211 | 0.208 |
| model_score | (0.31, 0.47] | 0.221 | 0.224 | 0.219 | 0.216 |
| model_score | (0.47, 0.66] | 0.238 | 0.207 | 0.205 | 0.201 |
| model_score | (0.66, inf) | 0.255 | 0.198 | 0.191 | 0.196 |

## 报警摘要示例

```text
MARS 监控报警摘要

总体结论：发现 2 个严重项、3 个警告项、2 个关注项。

严重：
1. model_score PSI=0.286，超过严重阈值 0.25，模型输出分布出现明显漂移。
2. anon_drift_004 PSI=0.318，超过严重阈值 0.25，建议复核该特征近期分布变化。

警告：
1. utilization PSI=0.192，超过警告阈值 0.10。
2. high_missing_002 缺失率最高 21.4%，较基准期明显上升。
3. 2025-06 target 表现覆盖率为 42.0%，标签尚未充分成熟，风险指标解释需谨慎。

关注：
1. query_cnt_30d 分箱占比在最新月份有一定变化。
2. income 缺失率出现轻微上升，可结合数据源稳定性复核。
```

## 解读要点

- Modeling Pipeline 输出用于回答模型效果、重要性和 replay 稳定性问题。
- `MarsMonitoringReport` 输出用于回答分布漂移、缺失趋势、分箱占比、target 表现覆盖率和风险指标趋势问题。
- 报警摘要是基于 report 的默认复核入口，不替代业务方自己的阈值策略、审批流程和处置动作。
