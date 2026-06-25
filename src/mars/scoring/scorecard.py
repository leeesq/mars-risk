"""MARS 评分卡构建、取整、导出与 SQL 生成工具。"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any, Dict, List, Union

import pandas as pd
import polars as pl

from mars.feature.binning.base import MarsBinnerBase


def _ensure_binner_scorecard_artifacts(
    binner: MarsBinnerBase,
    features: List[str],
) -> None:
    """确保分箱器已具备评分卡所需的映射表和 WOE 字典。"""
    missing_mappings = [feature for feature in features if feature not in binner.bin_mappings_]
    missing_woes = [feature for feature in features if not binner.bin_woes_.get(feature)]

    if not missing_mappings and not missing_woes:
        return

    if binner._cache_X is None or binner._cache_y is None:
        missing_bits: List[str] = []
        if missing_mappings:
            missing_bits.append("bin mappings")
        if missing_woes:
            missing_bits.append("WOE values")
        raise ValueError(
            "Scorecard generation requires cached fit data to materialize "
            f"{', '.join(missing_bits)}. Refit the binner on the target features first."
        )

    target_features = [feature for feature in features if feature in binner._cache_X.columns]
    if not target_features:
        raise ValueError("None of the requested scorecard features are available in cached binner data.")

    # 复用分箱器的画像路径，确保映射表和 WOE 使用同一套业务规则生成。
    binner.profile_bin_performance(
        binner._cache_X.select(target_features),
        binner._cache_y,
        update_woe=True,
    )


@dataclass
class MarsScorecard:
    """
    评分卡结果对象。

    该对象封装了由已拟合分箱器和逻辑回归系数推导出的分值明细表，
    同时保留评分卡刻度参数，便于导出 CSV、Excel 或生成部署 SQL。

    Attributes
    ----------
    points_table : pl.DataFrame or pd.DataFrame
        评分卡分值明细表，包含特征、分箱、WOE、系数与最终分值。
    base_points : float
        基础分。
    factor : float
        评分卡缩放因子。
    offset : float
        评分卡偏移量。
    pdo : float
        Points to Double the Odds 参数。
    base_score : float
        基准分数。
    base_odds : float
        基准赔率。
    intercept : float
        逻辑回归截距项。
    coefficients : dict of str to float
        特征系数字典。

    Examples
    --------
    >>> card = MarsScorecard(
    ...     points_table=pl.DataFrame({"feature": ["age"], "bin_index": [0], "points": [12.0]}),
    ...     base_points=600.0,
    ...     factor=28.85,
    ...     offset=600.0,
    ...     pdo=20.0,
    ...     base_score=600.0,
    ...     base_odds=50.0,
    ...     intercept=0.0,
    ...     coefficients={"age": 0.3},
    ...     _binner=None,
    ... )
    >>> card.base_points
    600.0
    """

    points_table: Union[pl.DataFrame, pd.DataFrame]
    base_points: float
    factor: float
    offset: float
    pdo: float
    base_score: float
    base_odds: float
    intercept: float
    coefficients: Dict[str, float]
    _binner: MarsBinnerBase

    @staticmethod
    def _format_score_value(value: float) -> str:
        """将 SQL 分值格式化为稳定的整数或小数字符串。"""
        value_float = float(value)
        if value_float.is_integer():
            return str(int(value_float))
        return f"{value_float:.6f}"

    def to_integer(self, round_decimals: int = 0, rebalance: bool = True) -> MarsScorecard:
        """
        返回分值取整后的评分卡副本。

        Parameters
        ----------
        round_decimals : int
            分值保留的小数位数。为 ``0`` 时，分箱分值列会转为整数。
        rebalance : bool
            是否把四舍五入产生的总分漂移回补到基础分。

        Returns
        -------
        MarsScorecard
            取整后的新评分卡对象，原对象不被修改。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        Examples
        --------
        >>> card = MarsScorecard(
        ...     points_table=pl.DataFrame({"feature": ["age"], "bin_index": [0], "points": [12.4]}),
        ...     base_points=600.0,
        ...     factor=28.85,
        ...     offset=600.0,
        ...     pdo=20.0,
        ...     base_score=600.0,
        ...     base_odds=50.0,
        ...     intercept=0.0,
        ...     coefficients={"age": 0.3},
        ...     _binner=None,
        ... )
        >>> rounded = card.to_integer()
        >>> isinstance(rounded, MarsScorecard)
        True
        """
        decimals = int(round_decimals)
        table_is_polars = isinstance(self.points_table, pl.DataFrame)
        table_pd = self.points_table.to_pandas() if table_is_polars else self.points_table.copy()
        if "points" not in table_pd.columns:
            raise ValueError("points_table must contain a 'points' column.")

        original_points = table_pd["points"].astype(float)
        rounded_points = original_points.round(decimals)
        rounded_base = round(float(self.base_points), decimals)
        if rebalance:
            original_total = round(float(self.base_points) + float(original_points.sum()), decimals)
            rounded_base = round(original_total - float(rounded_points.sum()), decimals)

        table_pd["points_original"] = original_points
        table_pd["points"] = rounded_points
        table_pd["points_round_error"] = rounded_points - original_points
        if decimals == 0:
            table_pd["points"] = table_pd["points"].astype("int64")

        if table_is_polars:
            points_table: Union[pl.DataFrame, pd.DataFrame] = pl.from_pandas(table_pd)
        else:
            points_table = table_pd

        return MarsScorecard(
            points_table=points_table,
            base_points=int(rounded_base) if decimals == 0 else float(rounded_base),
            factor=self.factor,
            offset=self.offset,
            pdo=self.pdo,
            base_score=self.base_score,
            base_odds=self.base_odds,
            intercept=self.intercept,
            coefficients=dict(self.coefficients),
            _binner=self._binner,
        )

    def write_csv(self, path: str = "mars_scorecard.csv") -> None:
        """
        导出评分卡分值表为 CSV 文件。

        Parameters
        ----------
        path : str
            输出文件路径。

        Returns
        -------
        None
            函数仅产生 CSV 文件写入副作用。

        Examples
        --------
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> card = MarsScorecard(
        ...     points_table=pl.DataFrame({"feature": ["age"], "bin_index": [0], "points": [12.0]}),
        ...     base_points=600.0,
        ...     factor=28.85,
        ...     offset=600.0,
        ...     pdo=20.0,
        ...     base_score=600.0,
        ...     base_odds=50.0,
        ...     intercept=0.0,
        ...     coefficients={"age": 0.3},
        ...     _binner=None,
        ... )
        >>> with TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "scorecard.csv"
        ...     card.write_csv(str(path))
        ...     path.exists()
        True
        """
        df = self.points_table.to_pandas() if isinstance(self.points_table, pl.DataFrame) else self.points_table
        df.to_csv(path, index=False)

    def write_excel(self, path: str = "mars_scorecard.xlsx") -> None:
        """
        导出评分卡为 Excel 文件。

        Parameters
        ----------
        path : str
            输出文件路径。

        Returns
        -------
        None
            函数仅产生 Excel 文件写入副作用。

        Notes
        -----
        导出结果包含 ``Config`` 与 ``Points`` 两个工作表，分别记录评分卡参数
        与分值明细。若环境缺少 ``xlsxwriter``，会自动回退到 ``openpyxl``。

        Examples
        --------
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> card = MarsScorecard(
        ...     points_table=pl.DataFrame({"feature": ["age"], "bin_index": [0], "points": [12.0]}),
        ...     base_points=600.0,
        ...     factor=28.85,
        ...     offset=600.0,
        ...     pdo=20.0,
        ...     base_score=600.0,
        ...     base_odds=50.0,
        ...     intercept=0.0,
        ...     coefficients={"age": 0.3},
        ...     _binner=None,
        ... )
        >>> with TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "scorecard.xlsx"
        ...     card.write_excel(str(path))
        ...     path.exists()
        True
        """
        df = self.points_table.to_pandas() if isinstance(self.points_table, pl.DataFrame) else self.points_table
        config_df = pd.DataFrame(
            [
                ("base_points", self.base_points),
                ("factor", self.factor),
                ("offset", self.offset),
                ("pdo", self.pdo),
                ("base_score", self.base_score),
                ("base_odds", self.base_odds),
                ("intercept", self.intercept),
            ],
            columns=["item", "value"],
        )

        engine = "xlsxwriter"
        try:
            import xlsxwriter  # noqa: F401
        except ImportError:
            engine = "openpyxl"

        with pd.ExcelWriter(path, engine=engine) as writer:
            config_df.to_excel(writer, sheet_name="Config", index=False)
            df.to_excel(writer, sheet_name="Points", index=False)

    def _get_points_map(self, feature: str) -> Dict[int, float]:
        """提取单个特征的 bin_index 到 points 映射。"""
        table_pd = self.points_table.to_pandas() if isinstance(self.points_table, pl.DataFrame) else self.points_table
        feat_df = table_pd[table_pd["feature"] == feature]
        return {
            int(row["bin_index"]): float(row["points"])
            for _, row in feat_df.iterrows()
        }

    def _generate_feature_points_case(self, feature: str, table_prefix: str) -> str:
        """
        为单个特征生成分箱得分的 SQL ``CASE WHEN`` 表达式。

        方法会复用分箱器中的数值切点或类别分组规则，并将缺失、特殊值和
        Other 箱映射到对应分值。
        """
        point_map = self._get_points_map(feature)
        mappings = self._binner.bin_mappings_.get(feature, {})
        col_name = f"{table_prefix}.{feature}" if table_prefix else feature
        lines = ["CASE"]

        lines.append(
            f"  WHEN {col_name} IS NULL THEN {self._format_score_value(point_map.get(MarsBinnerBase.IDX_MISSING, 0.0))}"
        )

        special_idx = [k for k in mappings.keys() if int(k) <= MarsBinnerBase.IDX_SPECIAL_START]
        for idx in sorted((int(i) for i in special_idx), reverse=True):
            label = str(mappings[idx])
            val_str = label.replace("Special_", "")
            try:
                float(val_str)
                sql_val = val_str
            except ValueError:
                sql_val = f"'{val_str}'"
            lines.append(f"  WHEN {col_name} = {sql_val} THEN {self._format_score_value(point_map.get(idx, 0.0))}")

        if hasattr(self._binner, "bin_cuts_") and feature in self._binner.bin_cuts_:
            cuts = self._binner.bin_cuts_[feature]
            for i in range(len(cuts) - 1):
                upper_bound = cuts[i + 1]
                points_val = point_map.get(i, 0.0)
                if upper_bound != float("inf"):
                    lines.append(f"  WHEN {col_name} < {upper_bound} THEN {self._format_score_value(points_val)}")
                else:
                    lines.append(f"  ELSE {self._format_score_value(points_val)}")
        elif hasattr(self._binner, "cat_cuts_") and feature in self._binner.cat_cuts_:
            groups = self._binner.cat_cuts_[feature]
            for i, group in enumerate(groups):
                if "__Mars_Other_Pre__" in group:
                    continue
                in_clause = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in group])
                lines.append(f"  WHEN {col_name} IN ({in_clause}) THEN {self._format_score_value(point_map.get(i, 0.0))}")
            lines.append(f"  ELSE {self._format_score_value(point_map.get(MarsBinnerBase.IDX_OTHER, 0.0))}")
        elif "ELSE" not in "\n".join(lines):
            lines.append(f"  ELSE {self._format_score_value(point_map.get(MarsBinnerBase.IDX_OTHER, 0.0))}")

        lines.append(f"END AS {feature}_points")
        return "\n".join(lines)

    def generate_sql(
        self,
        *,
        features: List[str] | None = None,
        table_prefix: str = "t",
        score_name: str = "score",
        include_base_points: bool = True,
    ) -> str:
        """
        生成评分卡 SQL 片段。

        Parameters
        ----------
        features : List[str] | None
            需要生成 SQL 的特征列表。默认为全部系数特征。
        table_prefix : str
            SQL 中引用特征列时使用的表别名前缀。传空字符串表示不加前缀。
        score_name : str
            最终总分字段名称。
        include_base_points : bool
            是否在总分表达式中包含基础分。

        Returns
        -------
        str
            可直接嵌入 ``SELECT`` 语句的 SQL 片段。若没有任何有效特征，则返回空字符串。

        Examples
        --------
        >>> from mars.feature import MarsNativeBinner
        >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
        >>> y = pl.Series("target", [0, 0, 1, 1])
        >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, y, features=["age"])
        >>> scorecard = build_scorecard(
        ...     binner,
        ...     {"age": 0.3},
        ...     intercept=-1.2,
        ...     pdo=20,
        ...     base_score=600,
        ...     base_odds=50,
        ... )
        >>> sql = scorecard.generate_sql(features=["age"], table_prefix="t")
        >>> "age_points" in sql
        True
        """
        target_features = features or list(self.coefficients.keys())
        valid_features = [f for f in target_features if f in self.coefficients]
        if not valid_features:
            return ""

        terms: List[str] = []
        feature_blocks: List[str] = []

        for feature in valid_features:
            feature_case = self._generate_feature_points_case(feature, table_prefix)
            feature_blocks.append(feature_case)
            terms.append(feature_case.rsplit(" AS ", 1)[0])

        expr = " +\n".join(terms)
        if include_base_points:
            expr = f"{self._format_score_value(self.base_points)} +\n" + expr

        return ",\n\n".join(feature_blocks + [f"({expr}) AS {score_name}"])


def build_scorecard(
    binner: MarsBinnerBase,
    coefficients: Dict[str, float],
    intercept: float,
    pdo: float,
    base_score: float,
    base_odds: float,
) -> MarsScorecard:
    """
    基于分箱器和逻辑回归系数构建评分卡。

    Parameters
    ----------
    binner : MarsBinnerBase
        已拟合的分箱器，且能够提供分箱映射与 WOE 信息。
    coefficients : Dict[str, float]
        特征系数字典，键为特征名，值为对应逻辑回归系数。
    intercept : float
        逻辑回归截距项。
    pdo : float
        Points to Double the Odds 参数，必须为正数。
    base_score : float
        评分卡基准分数。
    base_odds : float
        基准赔率，必须为正数。

    Returns
    -------
    MarsScorecard
        构建完成的评分卡对象。

    Raises
    ------
    ValueError
        当 ``pdo`` 或 ``base_odds`` 非正，``coefficients`` 为空，
        或分箱器缺少构建评分卡所需的映射信息时抛出。

    Examples
    --------
    >>> from mars.feature import MarsNativeBinner
    >>> X = pl.DataFrame({"age": [20, 30, 40, 50]})
    >>> y = pl.Series("target", [0, 0, 1, 1])
    >>> binner = MarsNativeBinner(method="quantile", n_bins=2).fit(X, y, features=["age"])
    >>> card = build_scorecard(binner, {"age": 0.3}, intercept=-1.2, pdo=20, base_score=600, base_odds=50)
    >>> isinstance(card, MarsScorecard)
    True
    """
    binner._check_is_fitted()

    if pdo <= 0:
        raise ValueError("`pdo` must be positive.")
    if base_odds <= 0:
        raise ValueError("`base_odds` must be positive.")
    if not coefficients:
        raise ValueError("`coefficients` must not be empty.")

    coefficient_features = list(coefficients.keys())
    _ensure_binner_scorecard_artifacts(binner, coefficient_features)

    input_is_pandas = bool(getattr(binner, "_return_pandas", False))
    factor = float(pdo / log(2))
    offset = float(base_score - factor * log(base_odds))
    base_points = float(offset - factor * intercept)

    def _sort_key(bin_index: int) -> tuple[int, int]:
        """确保普通箱在前，缺失和兜底箱在后稳定排序。"""
        if bin_index >= 0:
            return (0, bin_index)
        if bin_index == MarsBinnerBase.IDX_MISSING:
            return (1, 0)
        if bin_index == MarsBinnerBase.IDX_OTHER:
            return (1, 1)
        return (2, abs(bin_index))

    rows: List[Dict[str, Any]] = []
    for feature, coefficient in coefficients.items():
        if feature not in binner.bin_mappings_:
            raise ValueError(f"Feature '{feature}' not found in fitted binner mappings.")

        mappings = binner.bin_mappings_[feature]
        woe_map = binner.bin_woes_.get(feature, {})
        for bin_index in sorted((int(idx) for idx in mappings.keys()), key=_sort_key):
            woe = float(woe_map.get(bin_index, 0.0))
            points = float(-factor * float(coefficient) * woe)
            rows.append(
                {
                    "feature": feature,
                    "bin_index": int(bin_index),
                    "bin_label": str(mappings[bin_index]),
                    "coefficient": float(coefficient),
                    "woe": woe,
                    "points": points,
                }
            )

    points_df = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "feature": pl.String,
            "bin_index": pl.Int64,
            "bin_label": pl.String,
            "coefficient": pl.Float64,
            "woe": pl.Float64,
            "points": pl.Float64,
        }
    )

    points_output = binner._format_output(points_df)
    if input_is_pandas and isinstance(points_output, pl.DataFrame):
        points_output = points_output.to_pandas()

    return MarsScorecard(
        points_table=points_output,
        base_points=base_points,
        factor=factor,
        offset=offset,
        pdo=float(pdo),
        base_score=float(base_score),
        base_odds=float(base_odds),
        intercept=float(intercept),
        coefficients={k: float(v) for k, v in coefficients.items()},
        _binner=binner,
    )
