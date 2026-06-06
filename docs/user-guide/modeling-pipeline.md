# Modeling Pipeline

Modeling Pipeline 用于组织样本切分、模型调参、Top-K replay、建模评估和特征重要性输出。

!!! warning "快速迭代模块"
    Modeling Pipeline 仍在快速迭代中，可能不稳定；后续接口约定、结果对象和调参参数都可能发生较大变动。建议在生产流程中固定版本，并在升级前检查返回对象和字段名称。

## 支持模型

| `model_type` | 模型 |
| --- | --- |
| `xgb` | XGBoost |
| `lgb` | LightGBM |
| `cbt` / `cat` / `catboost` | CatBoost |
| `lr` / `logistic` | Logistic Regression |

Logistic Regression 支持 numeric 与 WOE 两种特征模式。

## 建模会话

```python
from mars.modeling import MarsModelingSession

session = MarsModelingSession(
    model_type="lgb",
    features=["income", "utilization", "segment"],
    target="target",
    categorical_features=["segment"],
    optimize_metric="ks",
    seed=1206,
)
```

## 样本切分

```python
modeling_df = session.slice(
    df,
    time_col="apply_dt",
    split_ratios={"train": 0.6, "val": 0.2, "oot": 0.2},
)
```

`time_col` 是切分任务参数，不放在 session 构造函数里。`target` 默认使用 session 中的建模目标，也可以在 `slice` 时显式覆盖。

## 调参

```python
tuning_result = session.tune(
    modeling_df,
    n_trials=20,
    history_path=None,
)
```

`history_path=None` 表示不落盘。传入路径时才写调参历史；如果路径已存在且 `overwrite=False`，会抛出 `FileExistsError`。

## Top-K replay

```python
from mars.modeling.tuning import MarsModelReplayRunner

replay_result = MarsModelReplayRunner().run(
    tuning_result,
    modeling_df,
    top_k=3,
    sort_metric="ks",
)
```

`MarsModelReplayRunner` 从 `MarsModelTuningResult` 读取模型规格，不需要在构造函数重复传入 `model_type`、`features` 或 `target`。

## 建模评估器

`MarsModelEvaluator` 是建模评估器，不是完整模型监控平台。它用于对已打分样本构建 train/val/oot 或业务切片上的评估报告。

模型输出稳定性可以通过 `Score PSI` 和 `score_psi` 观察。如果需要按自定义时间或业务切片进行特征/模型监控，可以使用 `MarsMonitor` 输出结构化监控指标。
