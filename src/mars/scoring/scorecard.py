from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import polars as pl

from mars.feature.binner import MarsBinnerBase


def _ensure_binner_scorecard_artifacts(
    binner: MarsBinnerBase,
    features: List[str],
) -> None:
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

    # Reuse the binner's profiling path so mappings and WOE values are generated
    # with the same business rules as the rest of the package.
    binner.profile_bin_performance(
        binner._cache_X.select(target_features),
        binner._cache_y,
        update_woe=True,
    )


@dataclass
class MarsScorecard:
    """
    Scorecard artifact built from a fitted binner and LR coefficients.
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

    def write_csv(self, path: str = "mars_scorecard.csv") -> None:
        df = self.points_table.to_pandas() if isinstance(self.points_table, pl.DataFrame) else self.points_table
        df.to_csv(path, index=False)

    def write_excel(self, path: str = "mars_scorecard.xlsx") -> None:
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
        table_pd = self.points_table.to_pandas() if isinstance(self.points_table, pl.DataFrame) else self.points_table
        feat_df = table_pd[table_pd["feature"] == feature]
        return {
            int(row["bin_index"]): float(row["points"])
            for _, row in feat_df.iterrows()
        }

    def _generate_feature_points_case(self, feature: str, table_prefix: str) -> str:
        point_map = self._get_points_map(feature)
        mappings = self._binner.bin_mappings_.get(feature, {})
        col_name = f"{table_prefix}.{feature}" if table_prefix else feature
        lines = ["CASE"]

        lines.append(
            f"  WHEN {col_name} IS NULL THEN {point_map.get(MarsBinnerBase.IDX_MISSING, 0.0):.6f}"
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
            lines.append(f"  WHEN {col_name} = {sql_val} THEN {point_map.get(idx, 0.0):.6f}")

        if hasattr(self._binner, "bin_cuts_") and feature in self._binner.bin_cuts_:
            cuts = self._binner.bin_cuts_[feature]
            for i in range(len(cuts) - 1):
                upper_bound = cuts[i + 1]
                points_val = point_map.get(i, 0.0)
                if upper_bound != float("inf"):
                    lines.append(f"  WHEN {col_name} < {upper_bound} THEN {points_val:.6f}")
                else:
                    lines.append(f"  ELSE {points_val:.6f}")
        elif hasattr(self._binner, "cat_cuts_") and feature in self._binner.cat_cuts_:
            groups = self._binner.cat_cuts_[feature]
            for i, group in enumerate(groups):
                if "__Mars_Other_Pre__" in group:
                    continue
                in_clause = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in group])
                lines.append(f"  WHEN {col_name} IN ({in_clause}) THEN {point_map.get(i, 0.0):.6f}")
            lines.append(f"  ELSE {point_map.get(MarsBinnerBase.IDX_OTHER, 0.0):.6f}")
        elif "ELSE" not in "\n".join(lines):
            lines.append(f"  ELSE {point_map.get(MarsBinnerBase.IDX_OTHER, 0.0):.6f}")

        lines.append(f"END AS {feature}_points")
        return "\n".join(lines)

    def generate_sql(
        self,
        *,
        features: Optional[List[str]] = None,
        table_prefix: str = "t",
        score_name: str = "score",
        include_base_points: bool = True,
    ) -> str:
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
            expr = f"{self.base_points:.6f} +\n" + expr

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
    Build a scorecard from a fitted binner and logistic regression coefficients.
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
