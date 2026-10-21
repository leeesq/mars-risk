"""与后端注册表对齐的训练后评分工具。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import polars as pl

from mars.compute import FrameLike, is_polars_dataframe, restore_frame_type, to_pandas_frame
from mars.modeling.backends.adapters import get_prediction_adapter
from mars.modeling.backends.registry import registered_backend_names, resolve_backend_name
from mars.modeling.contracts.report import MarsModelingReport
from mars.modeling.evaluation import MarsModelEvaluator


class ModelPredictor:
    """按训练时的特征顺序和类别取值调用模型预测后端。

    本类不拟合模型。输入为 Polars、``inplace=False`` 且未声明类别特征时，
    优先尝试后端的 Polars 路径；其他情况使用 Pandas 路径，再恢复输出类型。
    """

    def __init__(
        self,
        model: Any,
        feature_list: Sequence[str],
        *,
        categorical_features: Sequence[str] | None = None,
        model_type: str,
        category_levels: Dict[str, Sequence[Any]] | None = None,
    ) -> None:
        """保存模型和预测输入约定。

        Parameters
        ----------
        model : Any
            已训练模型，由 ``model_type`` 对应的预测适配器调用。
        feature_list : Sequence[str]
            训练时使用的特征列及顺序。预测时只将这些列交给模型，缺列会报错。
        categorical_features : Sequence[str] | None
            需要恢复为 Pandas category 的特征，默认无类别特征。
        model_type : str
            显式指定的模型后端名称，预测时按注册表解析，不从模型对象推测。
        category_levels : Dict[str, Sequence[Any]] | None
            训练时的类别取值，默认 ``None``。提供后按这些类别转换，未知类别变为缺失；
            未登记取值的类别列按本次输入创建 category。
        """
        self.model: Any = model
        self.features: List[str] = list(feature_list)
        self.categorical_features: List[str] = list(categorical_features or [])
        self.category_levels: Dict[str, List[Any]] = {
            str(feature): list(levels)
            for feature, levels in dict(category_levels or {}).items()
        }
        self.model_type = str(model_type).lower()

    def _resolve_backend_name(self) -> str:
        """将显式后端标识解析为规范注册表键。"""
        try:
            return resolve_backend_name(self.model_type)
        except KeyError as exc:
            raise ValueError(
                f"Unsupported model_type: {self.model_type!r}. "
                f"Expected one of {registered_backend_names()}."
            ) from exc

    def _prepare_pandas_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """选择特征列并稳定 Pandas 预测路径中的类别 dtype。"""
        missing = sorted(set(self.features).difference(df.columns))
        if missing:
            raise ValueError(f"Input data is missing required features: {missing}")

        X = df.loc[:, self.features].copy()
        for feature in self.categorical_features:
            if feature not in X.columns:
                continue
            categories = self.category_levels.get(feature)
            if categories is not None:
                X[feature] = X[feature].astype(pd.CategoricalDtype(categories=categories))
            else:
                X[feature] = X[feature].astype("category")
        return X

    def _predict_pandas_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """基于已准备好的 Pandas 特征矩阵执行预测。"""
        adapter = get_prediction_adapter(self._resolve_backend_name())
        return cast(np.ndarray, np.asarray(adapter.predict_pandas(self.model, X)))

    def _safe_predict_logic(self, df: pd.DataFrame) -> np.ndarray:
        """通过已注册适配器执行 Pandas 预测路径。"""
        return self._predict_pandas_matrix(self._prepare_pandas_features(df))

    def _safe_predict_logic_polars(self, df: pl.DataFrame) -> np.ndarray:
        """优先走适配器的 Polars 路径，必要时回退到 Pandas。"""
        if self.categorical_features:
            return self._safe_predict_logic(to_pandas_frame(df))

        missing = sorted(set(self.features).difference(df.columns))
        if missing:
            raise ValueError(f"Input data is missing required features: {missing}")

        adapter = get_prediction_adapter(self._resolve_backend_name())
        preds = adapter.predict_polars(self.model, df.select(self.features))
        if preds is not None:
            return cast(np.ndarray, np.asarray(preds))

        return self._safe_predict_logic(to_pandas_frame(df))

    def predict(
        self,
        df: FrameLike,
        pred_col: str = "pred_score",
        inplace: bool = False,
    ) -> FrameLike:
        """在样本表中追加或覆盖预测分列。

        Parameters
        ----------
        df : FrameLike
            包含全部训练特征的样本表，其他列原样保留。
        pred_col : str
            预测分列名，默认 ``"pred_score"``；同名列会覆盖。
        inplace : bool
            默认 ``False``，返回新表。为 True 且输入为 Pandas 时直接写入原表；
            Polars 输入始终返回新表。

        Returns
        -------
        FrameLike
            含预测分的样本表，保持输入的 Pandas 或 Polars 类型。

        Notes
        -----
        预测分含义由模型及后端决定。缺少训练特征或后端名称无效时抛出 ValueError。
        """
        prefer_polars = is_polars_dataframe(df)
        if prefer_polars and not inplace and isinstance(df, pl.DataFrame):
            preds = self._safe_predict_logic_polars(df)
            return df.with_columns(pl.Series(pred_col, preds))
        df_pd = df if isinstance(df, pd.DataFrame) and inplace else to_pandas_frame(df)
        df_pd[pred_col] = self._safe_predict_logic(df_pd)
        return restore_frame_type(df_pd, prefer_polars)

    def evaluate(
        self,
        df: FrameLike,
        group_col: str,
        target: str,
        *,
        time_col: str | None = None,
        val_target: str | None = None,
        benchmark_col: str | None = None,
        benchmark_cols: Sequence[str] | None = None,
        aux_targets: Sequence[str] | None = None,
        target_group_cols: Mapping[str, str] | None = None,
        pred_col: str = "pred_score",
        psi_include_missing: bool = False,
    ) -> MarsModelingReport:
        """对输入样本打分，并立即生成建模评估报告。"""
        scored = self.predict(df, pred_col=pred_col, inplace=False)
        return MarsModelEvaluator().evaluate(
            scored,
            pred_col=pred_col,
            group_col=group_col,
            target=target,
            time_col=time_col,
            benchmark_col=benchmark_col,
            benchmark_cols=benchmark_cols,
            val_target=val_target,
            aux_targets=aux_targets,
            target_group_cols=target_group_cols,
            psi_include_missing=psi_include_missing,
        )
