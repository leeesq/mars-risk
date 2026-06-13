# Modeling / Pipeline

`mars.modeling` 负责样本切分、模型调参、按 trial 回放、建模评估和特征重要性输出。`mars.pipeline` 负责把多层特征筛选、可选 WOE 分箱和最终建模串成一条可复用链路。

!!! warning "快速迭代模块"
    Modeling 建模和 Pipeline 编排仍在快速迭代中，可能不稳定；后续接口约定、结果对象和调参参数都可能发生较大变动。生产流程建议固定版本，并在升级前检查返回对象和字段名称。

## 支持模型

| `model_type` | 模型 |
| --- | --- |
| `xgb` | XGBoost |
| `lgb` | LightGBM |
| `cbt` / `cat` / `catboost` | CatBoost |
| `lr` / `logistic` | Logistic Regression |

Logistic Regression 支持 numeric 与 WOE 两种特征模式。

## Pipeline 编排

`mars.pipeline` 提供 MARS 自己的轻量编排层，用于把多个特征筛选步骤、可选 WOE 分箱步骤和最终建模步骤串起来。它不是 sklearn `Pipeline` 的严格子类，但保留 `fit`、`transform` 和 `predict` 的调用习惯。

树模型通常不需要先做 WOE 分箱，推荐直接走“筛选 -> 建模”：

```python
from mars.feature import MarsImportanceSelector, MarsStatsSelector
from mars.pipeline import MarsModelingPipeline, MarsModelingStep, MarsSelectionStep

pipeline = MarsModelingPipeline(
    target="target",
    features=["income", "utilization", "segment"],
    steps=[
        MarsSelectionStep(
            name="stats_filter",
            selector=MarsStatsSelector(iv_thr=0.02),
        ),
        MarsSelectionStep(
            name="importance_filter",
            selector=MarsImportanceSelector(selection_mode="top_k", selection_threshold=80),
            fit_params={"importance_table": importance_table},
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
scored_df = pipeline.predict(df, pred_col="model_score")
```

LR / 评分卡链路可以显式加入 `MarsWOEBinningStep`。该模式会先生成 `*_woe` 特征，再让后续筛选和 LR numeric 后端消费这些 WOE 列，避免重复触发 LR 后端内部 WOE：

```python
from mars.feature import MarsLinearSelector, MarsLiteOptBinner, MarsStatsSelector
from mars.pipeline import (
    MarsModelingPipeline,
    MarsModelingStep,
    MarsSelectionStep,
    MarsWOEBinningStep,
)

pipeline = MarsModelingPipeline(
    target="target",
    features=["income", "utilization"],
    steps=[
        MarsSelectionStep(
            name="stats_filter",
            selector=MarsStatsSelector(iv_thr=0.02),
        ),
        MarsWOEBinningStep(
            name="woe",
            binner=MarsLiteOptBinner(n_bins=6, monotonic_trend="auto_asc_desc"),
        ),
        MarsSelectionStep(
            name="linear_filter",
            selector=MarsLinearSelector(corr_thr=0.8),
        ),
        MarsModelingStep(
            name="modeling",
            model_type="lr",
            tune_params={"n_trials": 10, "artifact_dir": None},
        ),
    ],
)
```

`MarsSelectionStep` 可以出现多次，每一步都只消费上一阶段的 active features。`MarsModelingStep` 最多出现一次且必须放在最后；如果任意筛选步骤筛空特征，Pipeline 会直接抛出 `ValueError`。

## Modeling 建模会话

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
from mars.modeling import MarsModelDataSplitter

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
from mars.modeling import MarsModelReplayRunner

replay_result = MarsModelReplayRunner().replay(
    tuning_result,
    modeling_df,
    top_k=3,
    sort_metric="ks",
)
```

`MarsModelReplayRunner` 从 `MarsModelTuningResult` 读取模型规格，不需要在构造函数重复传入 `model_type`、`features` 或 `target`。

如果用户复盘 `history.csv` 后想指定 trial 编号，可以传 `trial_nums`：

```python
replay_result = MarsModelReplayRunner().replay(
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
    psi_include_missing=False,
    target_group_cols={
        "short_y_1": "short_y_1__dataset_flag",
        "short_y_2": "short_y_2__dataset_flag",
    },
)
```

汇总表会按主 target 和辅助 target 生成评估区块，并为每个 benchmark 输出 AUC、KS、F1 与差异指标。

## 建模评估器

`MarsModelEvaluator` 是建模评估器，不是完整模型监控平台。它用于对已打分样本构建 train/val/oot 或业务切片上的评估报告。

模型输出稳定性可以通过 `Score PSI` 和 `score_psi` 观察，特征漂移可以通过 `feature_psi` 观察。建模评估复用分箱评估器计算 PSI 明细，因此 `score_psi` / `feature_psi` 会保留分箱标签、分箱 PSI 和趋势明细；该链路只提供 `psi_include_missing` 控制缺失箱是否纳入 PSI。业务特殊值需要在分箱评估或监控模块中显式配置，建模评估不会额外猜测特殊值。

如果需要按自定义时间或业务切片进行特征/模型监控，可以使用 `MarsMonitor` 输出结构化监控指标。
