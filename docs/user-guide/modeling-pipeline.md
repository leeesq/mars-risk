# Modeling Pipeline

Modeling Pipeline 用于组织样本切分、模型调参、按 trial 回放、建模评估和特征重要性输出。

!!! warning "快速迭代模块"
    Modeling Pipeline 仍在快速迭代中，可能不稳定；后续接口约定、结果对象和调参参数都可能发生较大变动。生产流程建议固定版本，并在升级前检查返回对象和字段名称。

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

构造函数保存稳定建模规格；样本表、时间列、切分比例和输出路径都放在方法参数中。

## 样本切分

```python
modeling_df = session.slice(
    df,
    time_col="apply_dt",
    split_ratios={"train": 0.6, "val": 0.2, "oot": 0.2},
)
```

`time_col` 是切分任务参数，不放在 session 构造函数里。`target` 默认使用 session 中的建模目标，也可以在 `slice` 时显式覆盖。

如果主目标和辅助目标的表现期不一致，可以使用 `MarsModelDataSplitter.split_by_target_observation(...)` 为每个 target 生成独立切片列：

```python
from mars.modeling.slicing import MarsModelDataSplitter

modeling_df = MarsModelDataSplitter().split_by_target_observation(
    df,
    time_col="apply_dt",
    target="long_y",
    aux_targets=["short_y_1", "short_y_2"],
    split_ratios={"train": 0.6, "val": 0.2, "oot": 0.2},
)
```

主 target 的切片用于训练；辅助 target 只用于评估，不反向污染训练切片。

## 调参

```python
tuning_result = session.tune(
    modeling_df,
    n_trials=20,
    artifact_dir=None,
)
```

`artifact_dir=None` 表示完全不落盘。默认 `artifact_dir="modeling_artifacts"` 会为每次调参创建独立运行目录，写入 `history.csv`、`run_config.json`、`metadata.json`、特征重要性、最优模型和动态保留的 Top-N 模型。

### 指标

内置优化指标包括 `auc`、`ks` 和 `f1`。`f1` 默认使用 `f1_threshold=0.5`，可通过 `metric_params` 覆盖。

```python
tuning_result = session.tune(
    modeling_df,
    optimize_metric="f1",
    metric_params={"f1_threshold": 0.4},
)
```

自定义 metric 使用 `custom_metrics` 注册，标准签名是 `func(y_true, y_pred) -> float`：

```python
def head_tail_lift(y_true, y_pred):
    ...

tuning_result = session.tune(
    modeling_df,
    optimize_metric="head_tail_lift",
    custom_metrics={"head_tail_lift": head_tail_lift},
    training_metric="auc",
)
```

`training_metric` 用于模型后端训练期 early stopping 或 pruning；如果优化指标无法被模型原生训练接口识别，可以让训练期使用 `auc`，最终 trial objective 仍按自定义 metric 计算。需要向 XGBoost、LightGBM 或 CatBoost 透传原生 metric 时，可以使用 `backend_metric`。

### 特征重要性

默认输出模型原生重要性。显式请求 SHAP 时，会对 best model 计算 SHAP importance：

```python
tuning_result = session.tune(
    modeling_df,
    importance_methods=("native", "shap"),
    shap_sample_size=5000,
    shap_background_size=1000,
)
```

SHAP 是可选依赖；未安装 `shap` 但显式请求时会抛出清晰的 `ImportError`。

## Replay

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

如果用户复盘 `history.csv` 后想指定 trial 编号，可以传 `trial_nums`：

```python
replay_result = MarsModelReplayRunner().run(
    tuning_result,
    modeling_df,
    trial_nums=[7, 2],
    retrain=True,
)
```

`retrain=False` 会直接使用调参阶段保留的模型打分；如果指定 trial 没有被保留，会抛出 `ValueError` 并提示改用 `retrain=True`。

## 多 benchmark 与多 target 评估

建模训练只使用主 target。辅助 target 不参与训练，只进入评估报告：

```python
report = session.evaluate(
    scored_df,
    pred_col="score",
    benchmark_cols=["old_score", "rule_score"],
    aux_targets=["short_y_1", "short_y_2"],
    target_group_cols={
        "short_y_1": "short_y_1__dataset_flag",
        "short_y_2": "short_y_2__dataset_flag",
    },
)
```

汇总表会按主 target 和辅助 target 生成评估区块，并为每个 benchmark 输出 AUC、KS、F1 与差异指标。

## 建模评估器

`MarsModelEvaluator` 是建模评估器，不是完整模型监控平台。它用于对已打分样本构建 train/val/oot 或业务切片上的评估报告。

模型输出稳定性可以通过 `Score PSI` 和 `score_psi` 观察。如果需要按自定义时间或业务切片进行特征/模型监控，可以使用 `MarsMonitor` 输出结构化监控指标。
