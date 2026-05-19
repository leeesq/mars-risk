"""MARS 建模调参与基础评估工具。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numbers
import importlib
import re

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score, roc_curve


FrameLike = Union[pd.DataFrame, pl.DataFrame]
HISTORY_BASE_COLUMNS = ["trial_num", "trial_state", "is_valid", "val_diff", "max_oot_diff"]
METRIC_NAMES = ("auc", "ks")


def is_polars_dataframe(df: Any) -> bool:
    """
    判断输入对象是否为 Polars eager DataFrame。

    Parameters
    ----------
    df : Any
        待检查对象。

    Returns
    -------
    bool
        若对象是 `polars.DataFrame`，返回 ``True``；否则返回 ``False``。
    """
    return isinstance(df, pl.DataFrame)


def to_pandas_frame(df: FrameLike) -> pd.DataFrame:
    """
    将 Pandas 或 Polars 数据框转换为 Pandas 数据框。

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame
        输入数据框。

    Returns
    -------
    pandas.DataFrame
        转换后的 Pandas 数据框副本。

    Raises
    ------
    TypeError
        当输入既不是 Pandas 也不是 Polars 数据框时抛出。
    """
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")


def to_polars_frame(df: FrameLike) -> pl.DataFrame:
    """Convert a pandas or polars DataFrame into a Polars eager DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df.clone()
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")


def restore_frame_type(df: Union[pd.DataFrame, pl.DataFrame], prefer_polars: bool) -> FrameLike:
    """
    按调用方期望的公开类型恢复数据框。

    Parameters
    ----------
    df : pandas.DataFrame
        内部处理后的 Pandas 数据框。
    prefer_polars : bool
        是否需要将结果恢复为 Polars 数据框。

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        与调用方输入风格一致的数据框。
    """
    if prefer_polars:
        if isinstance(df, pl.DataFrame):
            return df
        return pl.from_pandas(df)
    if isinstance(df, pd.DataFrame):
        return df
    return df.to_pandas()


def calculate_ks(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    计算百分制 KS 指标。

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        真实二分类标签。
    y_pred : np.ndarray or pd.Series
        预测正类分数。

    Returns
    -------
    float
        百分制 KS 值。当标签中不同时包含正负两类时返回 ``0.0``。
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if np.unique(y_true_arr).size < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(y_true_arr, y_pred_arr)
    return round(float(np.max(tpr - fpr) * 100), 6)


def calculate_auc(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    计算百分制 AUC 指标。

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        真实二分类标签。
    y_pred : np.ndarray or pd.Series
        预测正类分数。

    Returns
    -------
    float
        百分制 AUC 值。当标签中不同时包含正负两类时返回 ``50.0``。
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if np.unique(y_true_arr).size < 2:
        return 50.0

    return round(float(roc_auc_score(y_true_arr, y_pred_arr) * 100), 6)


def split_name_sort_key(split_name: str) -> Tuple[int, int, str]:
    """
    生成数据切片名称的稳定排序键。

    Parameters
    ----------
    split_name : str
        切片名称。

    Returns
    -------
    tuple of int, int, str
        用于排序的稳定键，顺序为 ``train``、``val``、``oot*``、其他。
    """
    normalized = str(split_name).strip().lower()
    if "train" in normalized:
        return (0, 0, normalized)
    if "val" in normalized:
        return (1, 0, normalized)
    if "oot" in normalized:
        match = re.search(r"(\d+)", normalized)
        return (2, int(match.group(1)) if match else 10**9, normalized)
    return (3, 0, normalized)


def normalize_dataset_flags(flags: Union[pd.Series, pl.Series]) -> pd.Series:
    """Normalize dataset flag labels before train/val/oot role detection."""
    if isinstance(flags, pl.Series):
        flags_pd = flags.to_pandas()
    else:
        flags_pd = flags
    return flags_pd.astype(str).str.strip().str.lower()


def validate_dataset_flag_roles(flags: Union[pd.Series, pl.Series]) -> None:
    """Reject ambiguous split labels that match multiple reserved roles."""
    normalized = normalize_dataset_flags(flags)
    unique_flags = sorted(set(normalized.dropna().tolist()))
    conflicts: List[str] = []
    for flag in unique_flags:
        roles = [
            role
            for role, matched in {
                "train": "train" in flag,
                "val": "val" in flag,
                "oot": "oot" in flag,
            }.items()
            if matched
        ]
        if len(roles) > 1:
            conflicts.append(flag)
    if conflicts:
        raise ValueError(
            "Ambiguous dataset_flag values matched multiple split roles: "
            f"{conflicts}. Please rename them so each value contains only one of train/val/oot."
        )


def collect_library_versions(*module_names: str) -> Dict[str, Optional[str]]:
    """Collect optional dependency versions for reproducible modeling artifacts."""
    versions: Dict[str, Optional[str]] = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", None)
        except Exception:
            versions[module_name] = None
    return versions


class MarsBaseModelTuner(ABC):
    """
    MARS 二分类模型调参基类。

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame
        输入数据集，需包含特征列、目标列和数据集标识列。
    features : sequence of str
        参与训练的特征列名。
    target : str
        目标变量列名。
    optimize_metric : {"auc", "ks"}, default "ks"
        Trial 优化目标。
    param_space : dict, optional
        用户自定义搜索空间，会覆盖或补充默认搜索空间。
    max_diff : float, default 3.0
        允许的训练集与验证集指标衰减阈值，单位为百分点。
    seed : int, default 1206
        随机种子。
    use_oot_penalty : bool, default False
        是否额外使用最差 OOT 衰减对 Trial 进行惩罚。
    dataset_flag_col : str, default "dataset_flag"
        数据集切分标识列名。
    categorical_features : sequence of str, optional
        需要按类别特征处理的字段名，仅对支持原生类别特征的后端生效。

    Attributes
    ----------
    data_dict : dict of str to pandas.DataFrame
        按 ``train``、``val`` 和 ``oot*`` 组织好的数据切片。
    history : list of dict
        调参历史记录，每个元素对应一次 Trial 的落盘信息。
    all_models : dict of int to Any
        训练完成的 Trial 模型缓存，键为 Trial 编号。
    best_model : Any
        当前验证集最佳模型。
    best_score : float
        当前验证集最佳分数。

    Notes
    -----
    该基类只定义调参与评估骨架。具体训练、后端缓存构建与预测逻辑由子类实现。
    """

    SUPPORTED_OPTIMIZE_METRICS = {"auc", "ks"}

    def __init__(
        self,
        df: FrameLike,
        features: Sequence[str],
        target: str,
        *,
        optimize_metric: str = "ks",
        param_space: Optional[Mapping[str, Any]] = None,
        max_diff: float = 3.0,
        seed: int = 1206,
        use_oot_penalty: bool = False,
        dataset_flag_col: str = "dataset_flag",
        categorical_features: Optional[Sequence[str]] = None,
    ) -> None:
        self._input_is_polars: bool = is_polars_dataframe(df)
        if isinstance(df, pl.DataFrame):
            self.df_pl: Optional[pl.DataFrame] = df.clone()
            self.df_pd: Optional[pd.DataFrame] = None
            self.df_native: FrameLike = self.df_pl
            native_columns = list(self.df_pl.columns)
        elif isinstance(df, pd.DataFrame):
            self.df_pl = None
            self.df_pd = df.copy()
            self.df_native = self.df_pd
            native_columns = list(self.df_pd.columns)
        else:
            raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")
        self.features: List[str] = list(features)
        self.target: str = target
        self.optimize_metric: str = optimize_metric.lower()
        self.param_space: Dict[str, Any] = dict(param_space or {})
        self.max_diff: float = float(max_diff)
        self.seed: int = int(seed)
        self.use_oot_penalty: bool = use_oot_penalty
        self.dataset_flag_col: str = dataset_flag_col
        self.categorical_features: List[str] = list(categorical_features or [])

        if self.optimize_metric not in self.SUPPORTED_OPTIMIZE_METRICS:
            raise ValueError(
                f"Unsupported optimize_metric: {optimize_metric!r}. "
                f"Expected one of {sorted(self.SUPPORTED_OPTIMIZE_METRICS)}."
            )

        required_cols = set(self.features + [self.target, self.dataset_flag_col])
        missing_cols = required_cols.difference(native_columns)
        if missing_cols:
            raise ValueError(f"Input data is missing required columns: {sorted(missing_cols)}")

        cat_missing = set(self.categorical_features).difference(self.features)
        if cat_missing:
            raise ValueError(
                f"Categorical features must be included in features. Missing from features: {sorted(cat_missing)}"
            )

        self.history: List[Dict[str, Any]] = []
        self.all_models: Dict[int, Any] = {}
        self.best_model: Any = None
        self.best_score: float = -np.inf

        self.num_boost_round: int = 500
        self.early_stopping_rounds: int = 50
        self.training_metric: str = "auc"
        self.backend_data_mode: str = "unset"
        if self._input_is_polars:
            assert self.df_pl is not None
            self.feature_schema = {
                feature: str(self.df_pl.schema.get(feature))
                for feature in self.features
            }
        else:
            assert self.df_pd is not None
            self.feature_schema = {
                feature: str(self.df_pd.dtypes.get(feature))
                for feature in self.features
            }

        self._prepare_data()
        self._build_backend_data()

    @property
    def split_names(self) -> List[str]:
        """
        返回当前可用的数据切片名称列表。

        Returns
        -------
        list of str
            训练顺序下的切片名称，至少包含 ``train`` 与 ``val``。
        """
        return list(self.data_dict.keys())

    @property
    def replay_param_keys(self) -> List[str]:
        """
        返回可用于重训回放的参数键名列表。

        Returns
        -------
        list of str
            按定义顺序去重后的参数键名列表。
        """
        keys = list(self.get_default_space().keys())
        for key in self.param_space.keys():
            if key not in keys:
                keys.append(key)
        return keys

    def _prepare_data(self) -> None:
        """
        从 `dataset_flag_col` 中解析训练、验证与 OOT 数据集。

        Raises
        ------
        ValueError
            当缺少训练集或验证集切片时抛出。
        """
        if self._input_is_polars:
            assert self.df_pl is not None
            flags_pd = normalize_dataset_flags(self.df_pl.get_column(self.dataset_flag_col))
            validate_dataset_flag_roles(flags_pd)
            train_mask_pd = flags_pd.str.contains("train", na=False)
            val_mask_pd = flags_pd.str.contains("val", na=False)

            train_mask = pl.Series("__mask__", train_mask_pd.to_numpy())
            val_mask = pl.Series("__mask__", val_mask_pd.to_numpy())

            train_df = self.df_pl.filter(train_mask)
            val_df = self.df_pl.filter(val_mask)

            if train_df.is_empty():
                raise ValueError("No training rows were found from dataset_flag contains 'train'.")
            if val_df.is_empty():
                raise ValueError("No validation rows were found from dataset_flag contains 'val'.")

            self.data_dict: Dict[str, FrameLike] = {
                "train": train_df,
                "val": val_df,
            }

            original_flags = self.df_pl.get_column(self.dataset_flag_col).cast(pl.Utf8).to_list()
            oot_flags = sorted(
                {
                    original_flag
                    for original_flag in original_flags
                    if "oot" in str(original_flag).lower()
                },
                key=split_name_sort_key,
            )
            for flag in oot_flags:
                self.data_dict[str(flag)] = self.df_pl.filter(
                    pl.col(self.dataset_flag_col).cast(pl.Utf8) == str(flag)
                )
            return

        assert self.df_pd is not None
        flags_pd = normalize_dataset_flags(self.df_pd[self.dataset_flag_col])
        validate_dataset_flag_roles(flags_pd)
        train_mask = flags_pd.str.contains("train", na=False)
        val_mask = flags_pd.str.contains("val", na=False)

        train_df = self.df_pd.loc[train_mask].copy()
        val_df = self.df_pd.loc[val_mask].copy()

        if train_df.empty:
            raise ValueError("No training rows were found from dataset_flag contains 'train'.")
        if val_df.empty:
            raise ValueError("No validation rows were found from dataset_flag contains 'val'.")

        self.data_dict = {
            "train": train_df,
            "val": val_df,
        }

        original_flags = self.df_pd[self.dataset_flag_col].astype(str).tolist()
        oot_flags = sorted(
            {
                original_flag
                for original_flag in original_flags
                if "oot" in str(original_flag).lower()
            },
            key=split_name_sort_key,
        )
        for flag in oot_flags:
            self.data_dict[str(flag)] = self.df_pd.loc[
                self.df_pd[self.dataset_flag_col].astype(str) == str(flag)
            ].copy()

    def _get_feature_frame(self, df: FrameLike, *, for_categorical_backend: bool) -> pd.DataFrame:
        """
        生成后端可直接消费的特征数据框。

        Parameters
        ----------
        df : pandas.DataFrame
            单个切片的数据集。
        for_categorical_backend : bool
            是否需要为支持原生类别特征的后端转换类别 dtype。

        Returns
        -------
        pandas.DataFrame
            后端可直接使用的特征数据框。
        """
        if isinstance(df, pd.DataFrame):
            X = df.loc[:, self.features].copy()
        elif isinstance(df, pl.DataFrame):
            X = df.select(self.features).to_pandas()
        else:
            raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")
        if for_categorical_backend:
            for feature in self.categorical_features:
                if feature in X.columns:
                    X[feature] = X[feature].astype("category")
        return X

    def _get_feature_polars(self, df: FrameLike) -> pl.DataFrame:
        """Return selected features as a Polars DataFrame."""
        if isinstance(df, pl.DataFrame):
            return df.select(self.features)
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df.loc[:, self.features])
        raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")

    def _get_feature_arrow(self, df: FrameLike) -> Any:
        """Return selected features as a PyArrow table for reduced-copy backends."""
        return self._get_feature_polars(df).to_arrow()

    def _has_categorical_backend_features(self) -> bool:
        return bool(self.categorical_features)

    def _get_target_array(self, df: FrameLike) -> np.ndarray:
        """
        取出单个切片的目标数组。

        Parameters
        ----------
        df : pandas.DataFrame
            单个切片的数据集。

        Returns
        -------
        numpy.ndarray
            目标变量数组。
        """
        if isinstance(df, pd.DataFrame):
            return df[self.target].to_numpy()
        if isinstance(df, pl.DataFrame):
            return df.get_column(self.target).to_numpy()
        raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}.")

    def _evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算建模模块统一使用的二分类评估指标。

        Parameters
        ----------
        y_true : numpy.ndarray
            真实标签。
        y_pred : numpy.ndarray
            预测分数。

        Returns
        -------
        dict of str to float
            包含 ``auc`` 与 ``ks`` 的百分制指标字典。
        """
        return {
            "auc": calculate_auc(y_true, y_pred),
            "ks": calculate_ks(y_true, y_pred),
        }

    def evaluate_split(self, model: Any, split_name: str) -> Dict[str, float]:
        """
        评估指定切片上的模型表现。

        Parameters
        ----------
        model : Any
            已训练模型。
        split_name : str
            切片名称，例如 ``train``、``val`` 或某个 ``oot*``。

        Returns
        -------
        dict of str to float
            指定切片上的 ``auc`` 与 ``ks`` 百分制指标。
        """
        preds = self.predict_scores(model, split_name)
        y_true = self._get_target_array(self.data_dict[split_name])
        return self._evaluate_predictions(y_true, preds)

    def parse_param_space(self, trial: Any, default_space: Mapping[str, Any]) -> Dict[str, Any]:
        """
        将活动搜索空间解析为具体参数值。

        Parameters
        ----------
        trial : Any
            当前 Optuna Trial 对象。
        default_space : mapping of str to Any
            当前后端的默认搜索空间。

        Returns
        -------
        dict of str to Any
            可直接传给模型训练接口的确定性参数字典。

        Notes
        -----
        支持的元组约定包括：

        - ``("int", low, high[, step])``
        - ``("float", low, high[, step])``
        - ``("categorical", values)``
        """
        active_space = dict(default_space)
        active_space.update(self.param_space)

        params: Dict[str, Any] = {}
        for name, config in active_space.items():
            if not isinstance(config, (tuple, list)):
                params[name] = config
                continue

            if len(config) == 0:
                raise ValueError(f"Empty config for parameter {name!r}.")

            ptype = config[0]
            if ptype == "int":
                low, high = int(config[1]), int(config[2])
                step = int(config[3]) if len(config) > 3 else 1
                params[name] = trial.suggest_int(name, low, high, step=step)
            elif ptype == "float":
                low, high = float(config[1]), float(config[2])
                step = float(config[3]) if len(config) > 3 else None
                if step is None:
                    params[name] = trial.suggest_float(name, low, high)
                else:
                    params[name] = trial.suggest_float(name, low, high, step=step)
            elif ptype == "categorical":
                values = list(config[1])
                params[name] = trial.suggest_categorical(name, values)
            else:
                params[name] = config

        return params

    def _sync_to_disk(self, record: Mapping[str, Any], path: str) -> None:
        """
        将单次 Trial 记录追加写入 CSV。

        Parameters
        ----------
        record : mapping of str to Any
            单次 Trial 的记录内容。
        path : str
            CSV 输出路径。
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(record)]).to_csv(
            path_obj,
            mode="a",
            header=not path_obj.exists(),
            index=False,
        )

    def build_history_table(self) -> pd.DataFrame:
        """
        构建结构化 Trial 历史表。

        Returns
        -------
        pandas.DataFrame
            列顺序稳定、便于分析与回放的历史表。
        """
        history_table = pd.DataFrame(self.history)
        if history_table.empty:
            desired_columns = list(HISTORY_BASE_COLUMNS) + list(self.replay_param_keys)
            for split_name in self.split_names:
                for metric_name in METRIC_NAMES:
                    desired_columns.append(f"{split_name}_{metric_name}")
            return pd.DataFrame(columns=desired_columns)

        param_columns = [col for col in self.replay_param_keys if col in history_table.columns]
        metric_columns: List[str] = []
        for split_name in self.split_names:
            for metric_name in METRIC_NAMES:
                column_name = f"{split_name}_{metric_name}"
                if column_name in history_table.columns:
                    metric_columns.append(column_name)

        ordered_columns = [
            *HISTORY_BASE_COLUMNS,
            *param_columns,
            *metric_columns,
        ]
        extra_columns = [col for col in history_table.columns if col not in ordered_columns]
        return history_table.reindex(columns=ordered_columns + sorted(extra_columns))

    def objective(self, trial: Any, startup_trials: int, save_path: str) -> float:
        """
        执行单次 Trial 的完整生命周期。

        Parameters
        ----------
        trial : Any
            当前 Optuna Trial 对象。
        startup_trials : int
            启用剪枝前的预热 Trial 数量。
        save_path : str
            调参历史 CSV 输出路径。

        Returns
        -------
        float
            当前 Trial 的目标分数；若未通过泛化约束，则返回惩罚分。
        """
        record: Dict[str, Any] = {
            "trial_num": getattr(trial, "number", -1),
            "trial_state": "INIT_FAIL",
        }

        try:
            params = self.parse_param_space(trial, self.get_default_space())
            record.update(params)

            # 训练期可以使用 AUC 作为 early stopping / pruning 代理指标，
            # 但最终 Trial 得分仍由 optimize_metric 决定。
            model = self.train_model(
                trial=trial,
                params=params,
                startup_trials=startup_trials,
                training_metric=self.training_metric,
            )
            self.all_models[getattr(trial, "number", len(self.all_models))] = model

            # 所有切片统一评估后，再决定是否触发验证集 / OOT 泛化惩罚。
            metrics_by_split: Dict[str, Dict[str, float]] = {
                split_name: self.evaluate_split(model, split_name)
                for split_name in self.split_names
            }

            train_score = metrics_by_split["train"][self.optimize_metric]
            val_score = metrics_by_split["val"][self.optimize_metric]
            oot_scores: List[float] = [
                split_metrics[self.optimize_metric]
                for split_name, split_metrics in metrics_by_split.items()
                if "oot" in split_name.lower()
            ]

            val_diff = round(train_score - val_score, 6)
            is_valid = val_diff <= self.max_diff
            max_penalty_diff = val_diff

            max_oot_diff: Optional[float] = None
            if oot_scores:
                max_oot_diff = round(train_score - min(oot_scores), 6)
                if self.use_oot_penalty:
                    # 开启 OOT 惩罚后，以最差的时序外样本衰减作为额外约束，
                    # 逼迫搜索过程偏向更稳健的参数组合。
                    max_penalty_diff = max(max_penalty_diff, max_oot_diff)
                    if max_oot_diff > self.max_diff:
                        is_valid = False

            record.update(
                {
                    "trial_state": "COMPLETE",
                    "is_valid": is_valid,
                    "val_diff": round(val_diff, 4),
                    "max_oot_diff": round(max_oot_diff, 4) if max_oot_diff is not None else None,
                    **{
                        f"{split_name}_{metric_name}": metric_value
                        for split_name, metrics in metrics_by_split.items()
                        for metric_name, metric_value in metrics.items()
                    },
                }
            )

            # 仅当 Trial 通过泛化校验时，才允许刷新全局 best model。
            if is_valid and val_score > self.best_score:
                self.best_score = val_score
                self.best_model = model

            return float(val_score if is_valid else -100.0 - max_penalty_diff)

        except Exception as exc:
            optuna_module = None
            try:
                import optuna as optuna_module  # type: ignore
            except Exception:
                optuna_module = None

            if optuna_module is not None and isinstance(exc, optuna_module.exceptions.TrialPruned):
                record["trial_state"] = "PRUNED"
                raise

            record["trial_state"] = f"ERROR: {str(exc)[:120]}"
            raise
        finally:
            # 无论 Trial 成功、剪枝还是异常，都要保留 history 并立即落盘。
            self.history.append(record)
            self._sync_to_disk(record, save_path)

    def get_best_iteration(self, model: Any) -> Optional[int]:
        """
        返回模型的最佳迭代轮次。

        Parameters
        ----------
        model : Any
            已训练模型。

        Returns
        -------
        int or None
            若模型暴露 `best_iteration`，则返回其整数值；否则返回 ``None``。
        """
        best_iteration = getattr(model, "best_iteration", None)
        if isinstance(best_iteration, numbers.Integral):
            return int(best_iteration)
        get_best_iteration = getattr(model, "get_best_iteration", None)
        if callable(get_best_iteration):
            try:
                best_iteration = get_best_iteration()
                if isinstance(best_iteration, numbers.Integral):
                    return int(best_iteration)
            except Exception:
                return None
        return None

    @abstractmethod
    def _build_backend_data(self) -> None:
        """构建后端专用的缓存数据结构。"""

    @abstractmethod
    def get_default_space(self) -> Dict[str, Any]:
        """返回当前后端的默认搜索空间。"""

    @abstractmethod
    def train_model(
        self,
        trial: Any,
        params: Dict[str, Any],
        startup_trials: int,
        training_metric: str,
    ) -> Any:
        """
        训练单次 Trial 模型。

        Parameters
        ----------
        trial : Any
            当前 Trial 对象。
        params : dict of str to Any
            当前 Trial 的确定性超参数。
        startup_trials : int
            启用剪枝前的预热 Trial 数量。
        training_metric : str
            训练期监控指标。

        Returns
        -------
        Any
            训练完成的模型对象。
        """

    @abstractmethod
    def predict_scores(self, model: Any, split_name: str) -> np.ndarray:
        """
        对已缓存切片执行分数预测。

        Parameters
        ----------
        model : Any
            已训练模型。
        split_name : str
            切片名称。

        Returns
        -------
        numpy.ndarray
            预测分数数组。
        """
