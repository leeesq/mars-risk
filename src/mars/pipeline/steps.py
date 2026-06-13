"""MARS Pipeline 具体 step 实现。"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence, cast

import polars as pl

from mars.compute import FrameLike, to_polars_frame
from mars.core.base import MarsBaseSelector
from mars.feature import MarsBinnerBase, MarsStatsSelector
from mars.modeling import MarsModelingSession
from mars.pipeline.base import MarsPipelineStep, MarsStepResult
from mars.utils.logger import logger


class MarsSelectionStep(MarsPipelineStep):
    """
    Pipeline 中的特征筛选 step。

    该 step 可以在同一个 Pipeline 中出现多次，每一步都只消费上一阶段输出的
    active features。
    """

    def __init__(
        self,
        name: str,
        selector: MarsBaseSelector,
        fit_params: Mapping[str, Any] | None = None,
    ) -> None:
        """
        初始化筛选 step。

        Parameters
        ----------
        name : str
            step 唯一名称。
        selector : MarsBaseSelector
            已配置好筛选策略的 MARS selector。
        fit_params : Mapping[str, Any] | None
            传给 selector ``fit`` 的本次任务参数。对于 ``MarsStatsSelector``，会额外
            自动传入 ``target`` 和当前 active features；对于 sklearn 风格 selector，
            会自动传入 ``X``、``y`` 和当前 active features。
        """
        super().__init__(name)
        self.selector = selector
        self.fit_params = dict(fit_params or {})

    def fit_transform(
        self,
        df: pl.DataFrame,
        *,
        target: str,
        active_features: Sequence[str],
        pipeline_state: MutableMapping[str, Any],
    ) -> tuple[pl.DataFrame, list[str], MarsStepResult]:
        """
        拟合 selector 并输出新的 active features。

        Parameters
        ----------
        df : pl.DataFrame
            当前 Pipeline 工作表。
        target : str
            建模主目标列。
        active_features : Sequence[str]
            当前候选特征列。
        pipeline_state : MutableMapping[str, Any]
            Pipeline 级状态；筛选 step 当前只读该状态。

        Returns
        -------
        tuple of polars.DataFrame, list of str, MarsStepResult
            原工作表、筛选后的 active features 和 step 结果。

        Raises
        ------
        ValueError
            当前 step 筛空所有特征时抛出。
        """
        del pipeline_state
        logger.info("Running selection step %s with %s.", self.name, type(self.selector).__name__)
        input_features = list(active_features)
        params = dict(self.fit_params)
        if isinstance(self.selector, MarsStatsSelector):
            self.selector.fit(
                df,
                target=target,
                features=input_features,
                **params,
            )
        else:
            y = df.get_column(target)
            feature_frame = df.select(input_features)
            selector_fit = cast(Any, self.selector.fit)
            selector_fit(feature_frame, y, features=input_features, **params)

        output_features = list(self.selector.selected_features_)
        if not output_features:
            raise ValueError(f"Selection step {self.name!r} dropped all active features.")

        dropped_features = [
            feature for feature in input_features if feature not in set(output_features)
        ]
        return df, output_features, MarsStepResult(
            name=self.name,
            step_type="selection",
            input_features=input_features,
            output_features=output_features,
            dropped_features=dropped_features,
            report=self._selector_report(),
            metadata={"selector": type(self.selector).__name__},
        )

    def transform(
        self,
        df: pl.DataFrame,
        *,
        active_features: Sequence[str],
        pipeline_state: MutableMapping[str, Any],
    ) -> tuple[pl.DataFrame, list[str]]:
        """
        转换阶段不裁剪工作表，只更新 active features。

        Parameters
        ----------
        df : pl.DataFrame
            当前 Pipeline 工作表。
        active_features : Sequence[str]
            当前 active features。
        pipeline_state : MutableMapping[str, Any]
            Pipeline 级状态；筛选 step 当前只读该状态。

        Returns
        -------
        tuple of polars.DataFrame and list of str
            原工作表和拟合阶段保存的 selected features。
        """
        del active_features, pipeline_state
        return df, list(self.selector.selected_features_)

    def _selector_report(self) -> Any:
        """尽量读取 selector 的结构化报告，缺失报告能力时返回 ``None``。"""
        get_report = getattr(self.selector, "get_report", None)
        if callable(get_report):
            return get_report()
        return None


class MarsWOEBinningStep(MarsPipelineStep):
    """
    Pipeline 中显式生成 WOE 特征的分箱 step。

    该 step 主要服务 LR 和评分卡链路；树模型可以显式使用，但不作为默认推荐路径。
    """

    def __init__(
        self,
        name: str,
        binner: MarsBinnerBase,
        cat_features: Sequence[str] | None = None,
        woe_batch_size: int = 200,
    ) -> None:
        """
        初始化 WOE 分箱 step。

        Parameters
        ----------
        name : str
            step 唯一名称。
        binner : MarsBinnerBase
            已配置好分箱策略的 MARS binner。
        cat_features : Sequence[str] | None
            当前 active features 中需要按类别特征处理的列。
        woe_batch_size : int
            WOE 映射物化时的批大小。
        """
        super().__init__(name)
        self.binner = binner
        self.cat_features = list(cat_features or [])
        self.woe_batch_size = woe_batch_size

    def fit_transform(
        self,
        df: pl.DataFrame,
        *,
        target: str,
        active_features: Sequence[str],
        pipeline_state: MutableMapping[str, Any],
    ) -> tuple[pl.DataFrame, list[str], MarsStepResult]:
        """
        拟合分箱器并追加 WOE 特征列。

        Parameters
        ----------
        df : pl.DataFrame
            当前 Pipeline 工作表。
        target : str
            建模主目标列。
        active_features : Sequence[str]
            当前需要分箱的特征列。
        pipeline_state : MutableMapping[str, Any]
            Pipeline 级状态，会在本步骤更新 ``feature_map`` 和 ``has_woe_step``。

        Returns
        -------
        tuple of polars.DataFrame, list of str, MarsStepResult
            追加 WOE 列后的工作表、WOE 特征列和 step 结果。
        """
        logger.info("Running WOE binning step %s.", self.name)
        input_features = list(active_features)
        y = df.get_column(target)
        binned = self.binner.fit_transform(
            df.select(input_features),
            y,
            features=input_features,
            cat_features=self.cat_features,
            return_type="woe",
            woe_batch_size=self.woe_batch_size,
        )
        binned_pl = to_polars_frame(binned)
        output_features = [f"{feature}_woe" for feature in input_features]
        _validate_woe_columns(self.name, binned_pl, output_features)

        updated_df = _append_or_replace_columns(df, binned_pl.select(output_features))
        feature_map = cast(dict[str, str], pipeline_state.setdefault("feature_map", {}))
        for source, derived in zip(input_features, output_features, strict=False):
            feature_map[source] = derived
        pipeline_state["has_woe_step"] = True

        return updated_df, output_features, MarsStepResult(
            name=self.name,
            step_type="woe_binning",
            input_features=input_features,
            output_features=output_features,
            dropped_features=[],
            report=None,
            metadata={
                "binner": type(self.binner).__name__,
                "return_type": "woe",
            },
        )

    def transform(
        self,
        df: pl.DataFrame,
        *,
        active_features: Sequence[str],
        pipeline_state: MutableMapping[str, Any],
    ) -> tuple[pl.DataFrame, list[str]]:
        """
        使用已拟合分箱器追加 WOE 特征列。

        Parameters
        ----------
        df : pl.DataFrame
            当前 Pipeline 工作表。
        active_features : Sequence[str]
            当前需要转换的特征列。
        pipeline_state : MutableMapping[str, Any]
            Pipeline 级状态，会在本步骤更新 ``has_woe_step``。

        Returns
        -------
        tuple of polars.DataFrame and list of str
            追加 WOE 列后的工作表和 WOE 特征列。
        """
        input_features = list(active_features)
        transformed = self.binner.transform(
            df.select(input_features),
            return_type="woe",
            woe_batch_size=self.woe_batch_size,
        )
        transformed_pl = to_polars_frame(transformed)
        output_features = [f"{feature}_woe" for feature in input_features]
        _validate_woe_columns(self.name, transformed_pl, output_features)
        pipeline_state["has_woe_step"] = True
        return _append_or_replace_columns(df, transformed_pl.select(output_features)), output_features


class MarsModelingStep(MarsPipelineStep):
    """
    Pipeline 中的最终建模 step。

    该 step 最多出现一次，且必须位于 Pipeline 最后。
    """

    def __init__(
        self,
        name: str,
        model_type: str,
        *,
        time_col: str | None = None,
        split_ratios: Mapping[str, float] | None = None,
        dataset_flag_col: str = "dataset_flag",
        categorical_features: Sequence[str] | None = None,
        optimize_metric: str = "ks",
        seed: int = 1206,
        tune_params: Mapping[str, Any] | None = None,
        slice_params: Mapping[str, Any] | None = None,
    ) -> None:
        """
        初始化建模 step。

        Parameters
        ----------
        name : str
            step 唯一名称。
        model_type : str
            建模后端类型，例如 ``"lgb"``、``"xgb"``、``"cbt"`` 或 ``"lr"``。
        time_col : str | None
            需要自动切分样本时使用的时间列。
        split_ratios : Mapping[str, float] | None
            训练、验证和 OOT 的切分比例；与 ``time_col`` 同时提供时自动调用
            ``session.slice``。
        dataset_flag_col : str
            建模样本切片列名。
        categorical_features : Sequence[str] | None
            进入建模后端的类别特征列。若上游已生成 WOE 特征，通常应保持为空。
        optimize_metric : str
            调参优化指标。
        seed : int
            随机种子。
        tune_params : Mapping[str, Any] | None
            传给 ``MarsModelingSession.tune`` 的参数。
        slice_params : Mapping[str, Any] | None
            传给 ``MarsModelingSession.slice`` 的额外参数。
        """
        super().__init__(name)
        self.model_type = model_type
        self.time_col = time_col
        self.split_ratios = dict(split_ratios) if split_ratios is not None else None
        self.dataset_flag_col = dataset_flag_col
        self.categorical_features = list(categorical_features or [])
        self.optimize_metric = optimize_metric
        self.seed = seed
        self.tune_params = dict(tune_params or {})
        self.slice_params = dict(slice_params or {})
        self.session: MarsModelingSession | None = None
        self.modeling_df_: pl.DataFrame | None = None

    def fit_transform(
        self,
        df: pl.DataFrame,
        *,
        target: str,
        active_features: Sequence[str],
        pipeline_state: MutableMapping[str, Any],
    ) -> tuple[pl.DataFrame, list[str], MarsStepResult]:
        """
        基于最终 active features 创建 session 并执行调参。

        Parameters
        ----------
        df : pl.DataFrame
            当前 Pipeline 工作表。
        target : str
            建模主目标列。
        active_features : Sequence[str]
            最终进入建模的特征列。
        pipeline_state : MutableMapping[str, Any]
            Pipeline 级状态；如果已经执行 WOE step，LR 会按 numeric 模式消费外部 WOE
            列；否则 LR 会启用后端自身的 WOE 转换。

        Returns
        -------
        tuple of polars.DataFrame, list of str, MarsStepResult
            建模工作表、建模特征列和 step 结果。

        Raises
        ------
        ValueError
            建模特征为空、切分参数不完整或缺少 dataset flag 时抛出。
        """
        input_features = list(active_features)
        if not input_features:
            raise ValueError("MarsModelingStep requires at least one active feature.")

        categorical_features = [
            feature for feature in self.categorical_features if feature in set(input_features)
        ]
        has_woe_step = bool(pipeline_state.get("has_woe_step", False))
        lr_feature_mode = "numeric" if has_woe_step else "woe"
        self.session = MarsModelingSession(
            model_type=self.model_type,
            features=input_features,
            target=target,
            dataset_flag_col=self.dataset_flag_col,
            categorical_features=categorical_features,
            optimize_metric=self.optimize_metric,
            seed=self.seed,
            lr_feature_mode=lr_feature_mode,
        )

        modeling_df: FrameLike = df
        working_df = df
        if self.split_ratios is not None:
            if self.time_col is None:
                raise ValueError(
                    "MarsModelingStep requires time_col when split_ratios is provided."
                )
            modeling_df = self.session.slice(
                modeling_df,
                time_col=self.time_col,
                split_ratios=self.split_ratios,
                **self.slice_params,
            )
            working_df = to_polars_frame(modeling_df)
        elif self.dataset_flag_col not in working_df.columns:
            raise ValueError(
                "MarsModelingStep requires either split_ratios with time_col or an existing "
                f"{self.dataset_flag_col!r} column."
            )

        logger.info("Running modeling step %s with model_type=%s.", self.name, self.model_type)
        tuning_result = self.session.tune(modeling_df, **self.tune_params)
        self.modeling_df_ = working_df

        return working_df, list(tuning_result.features), MarsStepResult(
            name=self.name,
            step_type="modeling",
            input_features=input_features,
            output_features=list(tuning_result.features),
            dropped_features=[],
            report=tuning_result,
            metadata={
                "model_type": self.model_type,
                "dataset_flag_col": self.dataset_flag_col,
                "backend_data_mode": tuning_result.backend_data_mode,
                "has_prior_woe_step": has_woe_step,
                "lr_feature_mode": lr_feature_mode,
            },
        )

    def transform(
        self,
        df: pl.DataFrame,
        *,
        active_features: Sequence[str],
        pipeline_state: MutableMapping[str, Any],
    ) -> tuple[pl.DataFrame, list[str]]:
        """
        建模 step 在普通 transform 中不打分，只透传样本和 active features。

        Parameters
        ----------
        df : pl.DataFrame
            当前 Pipeline 工作表。
        active_features : Sequence[str]
            最终建模特征列。
        pipeline_state : MutableMapping[str, Any]
            Pipeline 级状态；当前只读该状态。

        Returns
        -------
        tuple of polars.DataFrame and list of str
            原工作表和原 active features。
        """
        del pipeline_state
        return df, list(active_features)



def _append_or_replace_columns(base_df: pl.DataFrame, new_columns: pl.DataFrame) -> pl.DataFrame:
    """追加派生列；若列已存在则先删除旧列以避免重复列名。"""
    existing = [column for column in new_columns.columns if column in base_df.columns]
    if existing:
        base_df = base_df.drop(existing)
    return base_df.hstack(new_columns)


def _validate_woe_columns(step_name: str, df: pl.DataFrame, columns: Sequence[str]) -> None:
    """校验 WOE 转换是否生成所有预期列。"""
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"WOE step {step_name!r} did not produce columns: {missing}.")
