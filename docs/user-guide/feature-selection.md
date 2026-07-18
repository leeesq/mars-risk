# 特征筛选

特征筛选用于把宽表候选特征压缩到更适合建模和监控的特征集合。MARS 目前提供统计筛选、线性模型筛选和重要性筛选三类入口。

## 统计筛选：`MarsStatsSelector`

```python
from mars.feature import MarsStatsSelector

selector = MarsStatsSelector(
    missing_thr=0.9,
    iv_thr=0.01,
    psi_thr=0.25,
    corr_thr=0.85,
    skip_fine_scan=True,
)

selector.fit(
    df,
    target="target",
    benchmark_df=benchmark_df,
    features=["income", "utilization", "segment"],
    group_col="month",
    feature_data_source={
        "user_profile": ["income"],
        "credit_usage": ["utilization"],
        "application": ["segment"],
    },
)
```

常用结果：

```python
selected_features = selector.selected_features_
decision_table = selector.decision_table_
report = selector.get_binning_report(df, benchmark_df=benchmark_df)
report.plot_risk_trends(max_plots=5)
```

`feature_data_source` 表示本次候选特征全集的数据源配置。如果部分特征在筛选过程中被过滤，传给评估报告的数据源映射会自动裁剪到当前 `active features`；筛选决策表仍会保留被过滤特征的来源信息，方便复盘。

`benchmark_df` 用于拟合粗筛和精筛分箱规则，并提供 PSI expected distribution；它不会进入
`df` 的 Total 或趋势分组。缺失率、IV、Lift、RC 和 WOE 相关性仍在 `df` 上计算，因此 `df`
必须包含至少两个有效 target 类别。selector 不保存原始 benchmark，调用
`get_binning_report()` 时需要重新传入。

默认 `psi_thr` 和 `rc_thr` 均已启用，必须提供 `group_col` 或 `time_col`。只做静态筛选时显式
关闭两项稳定性检查：

```python
selector = MarsStatsSelector(
    psi_thr=None,
    rc_thr=None,
)
selector.fit(df, target="target", features=features)
```

同时提供 `group_col`、`time_col` 和 `time_grain` 时，`group_col` 决定趋势分组，`time_col`
保留日期上下文；只有未传 `group_col` 时才由 `time_col/time_grain` 生成时间分组。

## 线性模型筛选：`MarsLinearSelector`

`MarsLinearSelector` 使用 sklearn 风格，`fit(X, y, ...)` 中 `y` 必填。

```python
from mars.feature import MarsLinearSelector

selector = MarsLinearSelector(
    penalty="l1",
    C=0.1,
)

selector.fit(X, y, features=["income", "utilization"])
selected = selector.selected_features_
```

## 重要性筛选：`MarsImportanceSelector`

重要性筛选可以直接消费已有重要性表，也可以训练 estimator 或使用 SHAP。

```python
from mars.feature import MarsImportanceSelector

selector = MarsImportanceSelector(selection_threshold=50)

selector.fit(
    X,
    importance_table=importance_table,
    features=["income", "utilization"],
)
```

如果直接传入 `importance_table`，可以不传 `y`。如果需要训练 estimator 或计算 SHAP，则必须传入 `y`。

## 白名单与黑名单

```python
selector.fit(
    df,
    target="target",
    features=features,
    group_col="month",
    white_list=["age"],
    black_list=["debug_feature"],
)
```

- `white_list` 用于保护必须保留的特征。
- `black_list` 用于剔除不可用、合规受限或调试字段。

## 使用建议

- 先用质量、稳定性和单变量风险筛掉明显不可用特征。
- 再用相关性或模型重要性减少重复信息。
- 对业务强约束特征使用 `white_list`，对泄漏字段或上线不可用字段使用 `black_list`。
- 对数据源分组做监控和复盘，避免某个来源特征被大规模过滤后仍在后续评估中误用。
