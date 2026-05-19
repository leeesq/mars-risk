"""Evaluation reports and prediction helpers for ``mars.modeling``."""

from __future__ import annotations

import base64
from io import BytesIO
import html
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

from mars.modeling.base import (
    FrameLike,
    calculate_auc,
    calculate_ks,
    is_polars_dataframe,
    restore_frame_type,
    split_name_sort_key,
    to_pandas_frame,
)


def _optional_import(module_name: str) -> Any:
    """Import an optional dependency and return ``None`` when unavailable."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


class MarsModelingReport:
    """Container for grouped model evaluation summaries."""

    def __init__(
        self,
        summary_table: pd.DataFrame,
        caption: str = "MARS Model Evaluation",
        detail_tables: Optional[Dict[str, pd.DataFrame]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.summary_table: pd.DataFrame = summary_table
        self.caption: str = caption
        self.detail_tables: Dict[str, pd.DataFrame] = dict(detail_tables or {})
        self.metadata: Dict[str, Any] = dict(metadata or {})

    @property
    def styled_summary(self) -> Any:
        """Return a styled summary suitable for notebook rendering."""
        all_cols = list(self.summary_table.columns)
        numeric_cols = [
            col
            for col in all_cols
            if isinstance(col, tuple)
            and col[1]
            in {
                "New AUC",
                "New KS",
                "Bench AUC",
                "Bench KS",
                "AUC Diff",
                "KS Diff",
                "LogLoss",
                "Brier",
                "Score PSI",
                "Top 10% Capture",
                "Top 20% Capture",
            }
        ]
        rate_cols = [
            col
            for col in all_cols
            if isinstance(col, tuple) and col[1] == "Bad Rate"
        ]
        count_cols = [
            col
            for col in all_cols
            if isinstance(col, tuple) and col[1] in {"Total Count", "Good", "Bad"}
        ]
        time_cols = [
            col
            for col in all_cols
            if isinstance(col, tuple) and col[1] in {"Start Time", "End Time"}
        ]

        styler = self.summary_table.style
        if numeric_cols:
            styler = styler.format("{:.2f}", subset=numeric_cols, na_rep="-")
        if rate_cols:
            styler = styler.format("{:.2%}", subset=rate_cols, na_rep="-")
        if count_cols:
            styler = styler.format("{:,.0f}", subset=count_cols, na_rep="-")
        if time_cols:
            styler = styler.format("{:%Y-%m-%d}", subset=time_cols, na_rep="-")

        return styler.set_caption(self.caption)

    def show_summary(self) -> Any:
        """Return the styled summary view."""
        return self.styled_summary

    def to_pandas(self) -> pd.DataFrame:
        """Return a copy of the underlying summary table."""
        return self.summary_table.copy()

    def write_excel(self, path: str = "mars_model_evaluation.xlsx", engine: Optional[str] = None) -> None:
        """Write the summary table to an Excel workbook."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if not self.detail_tables:
            self.summary_table.to_excel(path_obj, engine=engine)
            return
        with pd.ExcelWriter(path_obj, engine=engine) as writer:
            self.summary_table.to_excel(writer, sheet_name="summary")
            for name, table in self.detail_tables.items():
                table.to_excel(writer, sheet_name=str(name)[:31], index=False)

    def to_html(
        self,
        path: str = "mars_model_report.html",
        *,
        title: Optional[str] = None,
        run: Optional[Any] = None,
        scorecard: Optional[Any] = None,
        importance_table: Optional[pd.DataFrame] = None,
        history_table: Optional[pd.DataFrame] = None,
        top_features: int = 20,
        dpi: int = 150,
    ) -> Path:
        """Write a single-file HTML model report and return the output path."""
        renderer = _ModelReportHtmlRenderer(
            report=self,
            title=title or self.caption,
            run=run,
            scorecard=scorecard,
            importance_table=importance_table,
            history_table=history_table,
            top_features=top_features,
            dpi=dpi,
        )
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(renderer.render(), encoding="utf-8")
        return path_obj


class _ModelReportHtmlRenderer:
    """Private single-file HTML renderer for modeling reports."""

    def __init__(
        self,
        *,
        report: MarsModelingReport,
        title: str,
        run: Optional[Any],
        scorecard: Optional[Any],
        importance_table: Optional[pd.DataFrame],
        history_table: Optional[pd.DataFrame],
        top_features: int,
        dpi: int,
    ) -> None:
        self.report = report
        self.title = title
        self.scorecard = scorecard
        self.top_features = max(int(top_features), 1)
        self.dpi = int(dpi)
        self.metadata: Dict[str, Any] = dict(report.metadata)
        if run is not None:
            self.metadata.setdefault("training_config", getattr(run, "training_config", {}))
            self.metadata.setdefault("library_versions", getattr(run, "library_versions", {}))
            self.metadata.setdefault("backend_data_mode", getattr(run, "backend_data_mode", None))
            self.metadata.setdefault("model_type", getattr(run, "model_type", None))
            self.metadata.setdefault("optimize_metric", getattr(run, "optimize_metric", None))
            self.metadata.setdefault("best_score", getattr(run, "best_score", None))
            self.metadata.setdefault("best_iteration", getattr(run, "best_iteration", None))
            if importance_table is None:
                importance_table = getattr(run, "importance_table", None)
            if history_table is None:
                history_table = getattr(run, "history_table", None)
        self.importance_table = self._coerce_optional_frame(
            importance_table if importance_table is not None else self.metadata.get("importance_table")
        )
        self.history_table = self._coerce_optional_frame(
            history_table if history_table is not None else self.metadata.get("history_table")
        )

    @staticmethod
    def _coerce_optional_frame(value: Any) -> Optional[pd.DataFrame]:
        if value is None:
            return None
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, pl.DataFrame):
            return value.to_pandas()
        return None

    @staticmethod
    def _escape(value: Any) -> str:
        return html.escape("" if value is None else str(value), quote=True)

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    @classmethod
    def _format_value(cls, value: Any, *, percent: bool = False) -> str:
        if cls._is_missing(value):
            return "-"
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return pd.to_datetime(value).strftime("%Y-%m-%d")
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            val = float(value)
            if not np.isfinite(val):
                return "-"
            if percent:
                return f"{val:.2%}"
            if abs(val) >= 1000:
                return f"{val:,.2f}"
            return f"{val:.4f}"
        return str(value)

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        flat = df.copy()
        flat.columns = [
            " | ".join(str(part) for part in col if str(part) not in {"", "nan"})
            if isinstance(col, tuple)
            else str(col)
            for col in flat.columns
        ]
        return flat

    @classmethod
    def _table_html(
        cls,
        df: Optional[pd.DataFrame],
        *,
        table_id: str,
        empty_text: str = "No data available.",
        max_rows: Optional[int] = None,
    ) -> str:
        if df is None or df.empty:
            return f'<div class="mars-empty">{cls._escape(empty_text)}</div>'
        view = cls._flatten_columns(df)
        if max_rows is not None and len(view) > max_rows:
            view = view.head(max_rows).copy()
            note = f'<div class="mars-note">Showing first {max_rows:,} rows.</div>'
        else:
            note = ""

        headers = "".join(
            f'<th><button type="button" onclick="marsSortTable(\'{table_id}\',{idx})">{cls._escape(col)}</button></th>'
            for idx, col in enumerate(view.columns)
        )
        body_rows: List[str] = []
        for _, row in view.iterrows():
            row_class = ""
            psi_values = [
                pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                for col in view.columns
                if "psi" in str(col).lower()
            ]
            if any(pd.notna(value) and float(value) >= 0.25 for value in psi_values):
                row_class = ' class="mars-risk-high"'
            elif any(pd.notna(value) and float(value) >= 0.10 for value in psi_values):
                row_class = ' class="mars-risk-mid"'
            cells = []
            for col in view.columns:
                lower_col = str(col).lower()
                percent = any(token in lower_col for token in ("rate", "pct", "capture"))
                cells.append(f"<td>{cls._escape(cls._format_value(row[col], percent=percent))}</td>")
            body_rows.append(f"<tr{row_class}>{''.join(cells)}</tr>")

        return f"""
        <div class="mars-table-tools">
            <input type="search" placeholder="Search table..." oninput="marsFilterTable('{table_id}', this.value)" />
        </div>
        <div class="mars-table-wrap">
            <table id="{table_id}" class="mars-table">
                <thead><tr>{headers}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
            </table>
        </div>
        {note}
        """

    @staticmethod
    def _figure_to_img(fig: Any, *, dpi: int) -> str:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=dpi)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("ascii")
        matplotlib = _optional_import("matplotlib.pyplot")
        if matplotlib is not None:
            matplotlib.close(fig)
        return f'<img class="mars-chart-img" src="data:image/png;base64,{encoded}" alt="MARS model chart" />'

    @staticmethod
    def _require_pyplot() -> Any:
        plt = _optional_import("matplotlib.pyplot")
        if plt is None:
            raise ImportError(
                "matplotlib is required for MarsModelingReport.to_html(). "
                "Install it with `pip install \"mars-risk[plot]\"`."
            )
        return plt

    def _line_chart(
        self,
        df: pd.DataFrame,
        *,
        x_col: str,
        y_col: str,
        title: str,
        xlabel: str,
        ylabel: str,
        diagonal: bool = False,
    ) -> str:
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            return '<div class="mars-empty">Chart data is unavailable.</div>'
        plt = self._require_pyplot()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        group_col = str(self.metadata.get("group_col", "dataset_flag"))
        if group_col not in df.columns:
            group_col = df.columns[0]
        for group, part in df.groupby(group_col, sort=False):
            ax.plot(part[x_col], part[y_col], linewidth=1.8, label=str(group))
        if diagonal:
            ax.plot([0, 1], [0, 1], color="#8b9aaa", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        return self._figure_to_img(fig, dpi=self.dpi)

    def _score_distribution_chart(self) -> str:
        df = self.report.detail_tables.get("score_distribution", pd.DataFrame())
        required = {"bin_center", "pct", "target_value"}
        if df.empty or not required.issubset(df.columns):
            return '<div class="mars-empty">Score distribution data is unavailable.</div>'
        plt = self._require_pyplot()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        group_col = str(self.metadata.get("group_col", "dataset_flag"))
        for (group, target), part in df.groupby([group_col, "target_value"], sort=False):
            label = f"{group} | y={target}"
            ax.plot(part["bin_center"], part["pct"], linewidth=1.5, label=label)
        ax.set_title("Score Distribution")
        ax.set_xlabel("Predicted risk score")
        ax.set_ylabel("Distribution")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7)
        return self._figure_to_img(fig, dpi=self.dpi)

    def _rank_ordering_chart(self) -> str:
        df = self.report.detail_tables.get("decile_lift", pd.DataFrame())
        if df.empty or "decile" not in df.columns or "bad_rate" not in df.columns:
            return '<div class="mars-empty">Risk rank-ordering data is unavailable.</div>'
        plt = self._require_pyplot()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        group_col = str(self.metadata.get("group_col", "dataset_flag"))
        first_group = next(iter(df[group_col].astype(str).unique()), None) if group_col in df.columns else None
        if first_group is not None and "count" in df.columns:
            counts = df[df[group_col].astype(str) == first_group].sort_values("decile")
            ax.bar(counts["decile"], counts["count"], color="#dbeafe", alpha=0.65, label=f"{first_group} count")
        ax2 = ax.twinx()
        for group, part in df.groupby(group_col, sort=False):
            part = part.sort_values("decile")
            ax2.plot(part["decile"], part["bad_rate"], marker="o", linewidth=1.8, label=f"{group} bad rate")
        ax.set_title("Risk Rank-ordering")
        ax.set_xlabel("Score decile, high risk first")
        ax.set_ylabel("Count")
        ax2.set_ylabel("Bad rate")
        ax.grid(True, alpha=0.20)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best", fontsize=7)
        return self._figure_to_img(fig, dpi=self.dpi)

    def _importance_chart(self, importance: Optional[pd.DataFrame]) -> str:
        if importance is None or importance.empty or not {"feature", "importance"}.issubset(importance.columns):
            return '<div class="mars-empty">Feature importance data is unavailable.</div>'
        plot_df = importance.sort_values("importance", ascending=False).head(self.top_features).iloc[::-1]
        plt = self._require_pyplot()
        fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.28 * len(plot_df) + 1.2)))
        ax.barh(plot_df["feature"].astype(str), pd.to_numeric(plot_df["importance"], errors="coerce"), color="#2563eb")
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.25)
        return self._figure_to_img(fig, dpi=self.dpi)

    def _tuning_chart(self, history: Optional[pd.DataFrame]) -> str:
        if history is None or history.empty or "trial_num" not in history.columns:
            return '<div class="mars-empty">Tuning history data is unavailable.</div>'
        metric_cols = [
            col for col in history.columns
            if str(col).lower().endswith(("_ks", "_auc")) or str(col) in {"custom_mean_score", "best_score"}
        ][:8]
        if not metric_cols:
            return '<div class="mars-empty">No plottable tuning metrics were found.</div>'
        plt = self._require_pyplot()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        x = pd.to_numeric(history["trial_num"], errors="coerce")
        for col in metric_cols:
            y = pd.to_numeric(history[col], errors="coerce")
            if y.notna().any():
                ax.plot(x, y, marker="o", linewidth=1.4, markersize=3, label=str(col))
        ax.set_title("Optimization History")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Metric")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7)
        return self._figure_to_img(fig, dpi=self.dpi)

    def _summary_cards_html(self) -> str:
        summary = self.report.summary_table
        cards: List[Tuple[str, str]] = []
        if not summary.empty and isinstance(summary.columns, pd.MultiIndex):
            total_col = next((col for col in summary.columns if col[1] == "Total Count"), None)
            bad_col = next((col for col in summary.columns if col[1] == "Bad"), None)
            rate_col = next((col for col in summary.columns if col[1] == "Bad Rate"), None)
            ks_col = next((col for col in summary.columns if col[1] == "New KS"), None)
            auc_col = next((col for col in summary.columns if col[1] == "New AUC"), None)
            psi_col = next((col for col in summary.columns if col[1] == "Score PSI"), None)
            if total_col is not None:
                cards.append(("Samples", self._format_value(pd.to_numeric(summary[total_col], errors="coerce").sum())))
            if bad_col is not None:
                cards.append(("Bads", self._format_value(pd.to_numeric(summary[bad_col], errors="coerce").sum())))
            if rate_col is not None:
                cards.append(("Avg Bad Rate", self._format_value(pd.to_numeric(summary[rate_col], errors="coerce").mean(), percent=True)))
            if ks_col is not None:
                cards.append(("Best KS", self._format_value(pd.to_numeric(summary[ks_col], errors="coerce").max())))
            if auc_col is not None:
                cards.append(("Best AUC", self._format_value(pd.to_numeric(summary[auc_col], errors="coerce").max())))
            if psi_col is not None:
                cards.append(("Max Score PSI", self._format_value(pd.to_numeric(summary[psi_col], errors="coerce").max())))
        for key in ["model_type", "optimize_metric", "backend_data_mode", "best_iteration"]:
            value = self.metadata.get(key)
            if value is not None:
                cards.append((key.replace("_", " ").title(), self._format_value(value)))
        return "".join(f'<div class="mars-card"><span>{self._escape(label)}</span><strong>{self._escape(value)}</strong></div>' for label, value in cards)

    def _metadata_table(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for group_name, payload in [
            ("training_config", self.metadata.get("training_config")),
            ("library_versions", self.metadata.get("library_versions")),
        ]:
            if isinstance(payload, dict):
                for key, value in payload.items():
                    rows.append({"section": group_name, "item": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value})
        for key in ["model_type", "optimize_metric", "backend_data_mode", "best_score", "best_iteration"]:
            if key in self.metadata:
                rows.append({"section": "run", "item": key, "value": self.metadata.get(key)})
        return pd.DataFrame(rows)

    def _scorecard_section(self) -> str:
        if self.scorecard is None:
            return ""
        config_rows = []
        for item in ["pdo", "base_score", "base_odds", "base_points", "factor", "offset", "intercept"]:
            if hasattr(self.scorecard, item):
                config_rows.append({"item": item, "value": getattr(self.scorecard, item)})
        points = getattr(self.scorecard, "points_table", None)
        points_df = self._coerce_optional_frame(points)
        body = self._table_html(pd.DataFrame(config_rows), table_id="scorecard-config")
        body += self._table_html(points_df, table_id="scorecard-points", max_rows=200)
        return self._section("Scorecard", body, "scorecard")

    @staticmethod
    def _section(title: str, body: str, section_id: str, *, open_by_default: bool = True) -> str:
        open_attr = " open" if open_by_default else ""
        return f'<details id="{section_id}" class="mars-section"{open_attr}><summary>{html.escape(title)}</summary>{body}</details>'

    def _build_document(self, body: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{self._escape(self.title)}</title>
<style>{self._styles()}</style>
</head>
<body>
<main class="mars-page">
<header class="mars-hero">
  <div>
    <p class="mars-kicker">MARS Modeling Report</p>
    <h1>{self._escape(self.title)}</h1>
    <p>Single-file model audit report for discrimination, calibration, stability, explainability, and tuning review.</p>
  </div>
</header>
{body}
</main>
<script>{self._script()}</script>
</body>
</html>"""

    @staticmethod
    def _styles() -> str:
        return """
        :root{--ink:#172033;--muted:#64748b;--line:#dbe3ec;--soft:#f6f8fb;--accent:#2563eb;--warn:#fff7ed;--danger:#fff1f2}
        body{margin:0;background:#eef3f8;color:var(--ink);font-family:Inter,Segoe UI,Arial,sans-serif}
        .mars-page{max-width:1240px;margin:0 auto;padding:26px}
        .mars-hero{background:#fff;border:1px solid var(--line);border-radius:10px;padding:24px;margin-bottom:16px;box-shadow:0 12px 30px rgba(35,55,80,.08)}
        .mars-kicker{margin:0 0 8px;color:var(--accent);font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
        h1{margin:0 0 8px;font-size:30px;letter-spacing:0}.mars-hero p{color:var(--muted)}
        .mars-section{background:#fff;border:1px solid var(--line);border-radius:10px;margin:14px 0;padding:0;box-shadow:0 8px 22px rgba(35,55,80,.06);overflow:hidden}
        .mars-section>summary{cursor:pointer;list-style:none;padding:16px 18px;font-weight:750;border-bottom:1px solid var(--line);background:#fbfdff}
        .mars-section>summary::-webkit-details-marker{display:none}.mars-section>*:not(summary){margin:16px 18px}
        .mars-card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
        .mars-card{border:1px solid var(--line);border-radius:8px;padding:13px;background:var(--soft)}
        .mars-card span{display:block;color:var(--muted);font-size:12px}.mars-card strong{display:block;margin-top:5px;font-size:20px}
        .mars-chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}.mars-chart{border:1px solid var(--line);border-radius:8px;padding:10px;background:#fff}
        .mars-chart h3{margin:0 0 8px;font-size:15px}.mars-chart-img{width:100%;height:auto;display:block}
        .mars-table-tools{margin:10px 0}.mars-table-tools input{width:min(360px,100%);padding:8px 10px;border:1px solid var(--line);border-radius:7px}
        .mars-table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px}.mars-table{border-collapse:collapse;width:100%;font-size:12px;background:#fff}
        .mars-table th{position:sticky;top:0;background:#edf4fb;z-index:1}.mars-table th button{border:0;background:transparent;font:inherit;font-weight:700;cursor:pointer;width:100%;text-align:left;padding:9px}
        .mars-table td{border-top:1px solid #edf1f5;padding:8px;white-space:nowrap}.mars-table tr:nth-child(even) td{background:#fbfdff}
        .mars-risk-mid td{background:var(--warn)!important}.mars-risk-high td{background:var(--danger)!important;color:#9f1239}
        .mars-empty{border:1px dashed var(--line);border-radius:8px;padding:14px;background:#fbfdff;color:var(--muted)}
        .mars-note{color:var(--muted);font-size:12px;margin-top:6px}
        @media(max-width:760px){.mars-page{padding:12px}.mars-chart-grid{grid-template-columns:1fr}h1{font-size:24px}}
        """

    @staticmethod
    def _script() -> str:
        return """
        function marsFilterTable(tableId, query){
          const table=document.getElementById(tableId); if(!table) return;
          const q=(query||'').toLowerCase();
          table.querySelectorAll('tbody tr').forEach(row=>{row.style.display=row.textContent.toLowerCase().includes(q)?'':'none';});
        }
        function marsSortTable(tableId, colIndex){
          const table=document.getElementById(tableId); if(!table) return;
          const tbody=table.querySelector('tbody');
          const rows=Array.from(tbody.querySelectorAll('tr'));
          const dir=table.getAttribute('data-sort-dir')==='asc'?'desc':'asc';
          table.setAttribute('data-sort-dir',dir);
          rows.sort((a,b)=>{
            const av=(a.children[colIndex]?.textContent||'').replace(/,/g,'');
            const bv=(b.children[colIndex]?.textContent||'').replace(/,/g,'');
            const an=Number(av), bn=Number(bv);
            const cmp=Number.isFinite(an)&&Number.isFinite(bn)?an-bn:av.localeCompare(bv);
            return dir==='asc'?cmp:-cmp;
          });
          rows.forEach(row=>tbody.appendChild(row));
        }
        """

    def render(self) -> str:
        summary_df = self.report.summary_table.reset_index()
        body_parts: List[str] = []
        executive = f'<div class="mars-card-grid">{self._summary_cards_html()}</div>'
        executive += self._table_html(summary_df, table_id="summary-table")
        body_parts.append(self._section("Executive Summary", executive, "executive-summary"))

        roc = self.report.detail_tables.get("roc_curve", pd.DataFrame())
        ks = self.report.detail_tables.get("ks_curve", pd.DataFrame())
        calibration = self.report.detail_tables.get("calibration_curve", pd.DataFrame())
        charts = [
            ("ROC Curve", self._line_chart(roc, x_col="fpr", y_col="tpr", title="ROC Curve", xlabel="False positive rate", ylabel="True positive rate", diagonal=True)),
            ("KS Curve", self._line_chart(ks, x_col="sample_pct", y_col="ks", title="KS Curve", xlabel="Sample percentile", ylabel="KS")),
            ("Score Distribution", self._score_distribution_chart()),
            ("Risk Rank-ordering", self._rank_ordering_chart()),
            ("Calibration Curve", self._line_chart(calibration, x_col="pred_mean", y_col="bad_rate", title="Reliability Diagram", xlabel="Mean predicted risk", ylabel="Observed bad rate", diagonal=True)),
        ]
        chart_html = '<div class="mars-chart-grid">' + "".join(
            f'<article class="mars-chart"><h3>{self._escape(name)}</h3>{chart}</article>' for name, chart in charts
        ) + "</div>"
        body_parts.append(self._section("Discrimination & Calibration", chart_html, "discrimination-calibration"))

        stability = self._table_html(self.report.detail_tables.get("score_psi"), table_id="score-psi-table")
        feature_psi = self.report.detail_tables.get("feature_psi")
        if feature_psi is not None and not feature_psi.empty:
            feature_view = feature_psi.sort_values("feature_psi", ascending=False).head(max(self.top_features * 3, self.top_features))
            stability += "<h3>Feature PSI Top Drift</h3>" + self._table_html(feature_view, table_id="feature-psi-table")
        body_parts.append(self._section("Stability", stability, "stability"))

        explain = self._importance_chart(self.importance_table)
        explain += self._table_html(self.importance_table, table_id="importance-table", max_rows=max(self.top_features * 3, 50))
        shap_table = self.report.detail_tables.get("shap_summary")
        if shap_table is not None and not shap_table.empty:
            explain += "<h3>SHAP Summary</h3>" + self._table_html(shap_table, table_id="shap-table", max_rows=200)
        body_parts.append(self._section("Explainability", explain, "explainability", open_by_default=False))

        audit = self._tuning_chart(self.history_table)
        audit += self._table_html(self.history_table, table_id="history-table", max_rows=300)
        metadata_df = self._metadata_table()
        if not metadata_df.empty:
            audit += "<h3>Training Metadata</h3>" + self._table_html(metadata_df, table_id="metadata-table")
        body_parts.append(self._section("Tuning Audit", audit, "tuning-audit", open_by_default=False))

        scorecard_section = self._scorecard_section()
        if scorecard_section:
            body_parts.append(scorecard_section)

        return self._build_document("".join(body_parts))


class MarsModelEvaluator:
    """Reusable evaluation tool for scored binary risk datasets."""

    COLUMN_ORDER = [
        "Total Count",
        "Good",
        "Bad",
        "Bad Rate",
        "New AUC",
        "New KS",
        "LogLoss",
        "Brier",
        "Score PSI",
        "Top 10% Capture",
        "Top 20% Capture",
        "Bench AUC",
        "Bench KS",
        "AUC Diff",
        "KS Diff",
    ]

    def __init__(
        self,
        *,
        group_col: str,
        target_col: str,
        benchmark_col: Optional[str] = None,
        time_col: Optional[str] = None,
        val_target_col: Optional[str] = None,
        feature_cols: Optional[Sequence[str]] = None,
        importance_table: Optional[pd.DataFrame] = None,
    ) -> None:
        self.group_col: str = group_col
        self.target_col: str = target_col
        self.benchmark_col: Optional[str] = benchmark_col
        self.time_col: Optional[str] = time_col
        self.val_target_col: Optional[str] = val_target_col
        self.feature_cols: List[str] = list(feature_cols or [])
        self.importance_table: Optional[pd.DataFrame] = None if importance_table is None else importance_table.copy()

    def _validate_frame(self, df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
        """Validate required columns and normalize time columns when configured."""
        required = {self.group_col, pred_col, self.target_col}
        if self.time_col:
            required.add(self.time_col)
        if self.benchmark_col:
            required.add(self.benchmark_col)
        if self.val_target_col:
            required.add(self.val_target_col)

        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Evaluation data is missing required columns: {sorted(missing)}")

        normalized = df.copy()
        if self.time_col is not None:
            normalized[self.time_col] = pd.to_datetime(normalized[self.time_col], errors="coerce")
        return normalized

    def _calc_metric_block(
        self,
        sub_df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        section_label: Optional[str] = None,
        score_psi: Optional[float] = None,
    ) -> Dict[Tuple[str, str], Any]:
        """Build one target block for the grouped evaluation report."""
        y_true = pd.to_numeric(sub_df[target_col], errors="coerce")
        y_pred = pd.to_numeric(sub_df[pred_col], errors="coerce")
        valid_mask = y_true.notna() & y_pred.notna() & (y_true >= 0)

        valid_y = y_true[valid_mask]
        valid_pred = y_pred[valid_mask]

        total_count = int(valid_y.shape[0])
        if total_count > 0:
            bad_count = float(valid_y.sum())
            good_count = float(total_count - bad_count)
            bad_rate = float(bad_count / total_count)
        else:
            bad_count = np.nan
            good_count = np.nan
            bad_rate = np.nan

        block: Dict[Tuple[str, str], Any] = {}
        section = section_label or f"Target: {target_col}"
        block[(section, "Total Count")] = total_count
        block[(section, "Good")] = good_count
        block[(section, "Bad")] = bad_count
        block[(section, "Bad Rate")] = bad_rate

        if total_count > 0 and valid_y.nunique() >= 2:
            block[(section, "New AUC")] = calculate_auc(valid_y.to_numpy(), valid_pred.to_numpy())
            block[(section, "New KS")] = calculate_ks(valid_y.to_numpy(), valid_pred.to_numpy())
            clipped_pred = np.clip(valid_pred.to_numpy(dtype=float), 1e-15, 1 - 1e-15)
            y_arr = valid_y.to_numpy(dtype=float)
            block[(section, "LogLoss")] = float(
                -np.mean(y_arr * np.log(clipped_pred) + (1.0 - y_arr) * np.log(1.0 - clipped_pred))
            )
            block[(section, "Brier")] = float(np.mean((clipped_pred - y_arr) ** 2))
            total_bad = float(y_arr.sum())
            order = np.argsort(-clipped_pred)
            for pct, label in [(0.10, "Top 10% Capture"), (0.20, "Top 20% Capture")]:
                top_n = max(int(np.ceil(len(order) * pct)), 1)
                block[(section, label)] = (
                    float(y_arr[order[:top_n]].sum() / total_bad)
                    if total_bad > 0
                    else np.nan
                )
        else:
            block[(section, "New AUC")] = np.nan
            block[(section, "New KS")] = np.nan
            block[(section, "LogLoss")] = np.nan
            block[(section, "Brier")] = np.nan
            block[(section, "Top 10% Capture")] = np.nan
            block[(section, "Top 20% Capture")] = np.nan

        block[(section, "Score PSI")] = score_psi if score_psi is not None else np.nan

        if self.benchmark_col:
            bench_pred = pd.to_numeric(sub_df[self.benchmark_col], errors="coerce")
            bench_mask = valid_mask & bench_pred.notna()
            bench_y = y_true[bench_mask]
            bench_scores = bench_pred[bench_mask]
            if bench_y.shape[0] > 0 and bench_y.nunique() >= 2:
                bench_auc = calculate_auc(bench_y.to_numpy(), bench_scores.to_numpy())
                bench_ks = calculate_ks(bench_y.to_numpy(), bench_scores.to_numpy())
            else:
                bench_auc = np.nan
                bench_ks = np.nan

            block[(section, "Bench AUC")] = bench_auc
            block[(section, "Bench KS")] = bench_ks
            block[(section, "AUC Diff")] = (
                block[(section, "New AUC")] - bench_auc
                if pd.notna(block[(section, "New AUC")]) and pd.notna(bench_auc)
                else np.nan
            )
            block[(section, "KS Diff")] = (
                block[(section, "New KS")] - bench_ks
                if pd.notna(block[(section, "New KS")]) and pd.notna(bench_ks)
                else np.nan
            )

        return block

    def _get_ordered_groups(self, df: pd.DataFrame) -> List[str]:
        """Return grouped split names in stable MARS order."""
        groups = df[self.group_col].astype(str).unique().tolist()
        return sorted(groups, key=split_name_sort_key)

    def _get_ordered_columns(self, available_columns: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Return a stable column layout for the final report."""
        ordered_columns: List[Tuple[str, str]] = []
        if self.time_col:
            for time_name in ("Start Time", "End Time"):
                candidate = ("Time Period", time_name)
                if candidate in available_columns:
                    ordered_columns.append(candidate)

        sections = [f"Target: {self.target_col}"]
        if self.val_target_col:
            sections.append(f"Val Target: {self.val_target_col}")

        for section in sections:
            for column_name in self.COLUMN_ORDER:
                candidate = (section, column_name)
                if candidate in available_columns:
                    ordered_columns.append(candidate)

        remaining_columns = [col for col in available_columns if col not in ordered_columns]
        return ordered_columns + remaining_columns

    def _build_score_bins(self, baseline_scores: pd.Series) -> Optional[np.ndarray]:
        """Build stable decile cut points from the first available group."""
        clean_scores = pd.to_numeric(baseline_scores, errors="coerce").dropna()
        if clean_scores.nunique() < 2:
            return None
        quantiles = np.linspace(0.0, 1.0, 11)
        bins = np.unique(np.quantile(clean_scores.to_numpy(dtype=float), quantiles))
        if bins.size < 2:
            return None
        bins[0] = -np.inf
        bins[-1] = np.inf
        return bins

    def _build_score_psi_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        ordered_groups: Sequence[str],
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Calculate score distribution PSI for each group against the first group."""
        if not ordered_groups:
            return pd.DataFrame(), {}
        baseline_group = ordered_groups[0]
        baseline_scores = df.loc[df[self.group_col].astype(str) == str(baseline_group), pred_col]
        bins = self._build_score_bins(baseline_scores)
        if bins is None:
            return pd.DataFrame(), {str(group): np.nan for group in ordered_groups}

        rows: List[Dict[str, Any]] = []
        expected_counts = pd.cut(
            pd.to_numeric(baseline_scores, errors="coerce"),
            bins=bins,
            include_lowest=True,
            duplicates="drop",
        ).value_counts(sort=False)
        expected_dist = expected_counts / max(float(expected_counts.sum()), 1.0)
        psi_map: Dict[str, float] = {}

        for group in ordered_groups:
            group_scores = pd.to_numeric(
                df.loc[df[self.group_col].astype(str) == str(group), pred_col],
                errors="coerce",
            )
            actual_counts = pd.cut(
                group_scores,
                bins=bins,
                include_lowest=True,
                duplicates="drop",
            ).value_counts(sort=False)
            actual_dist = actual_counts / max(float(actual_counts.sum()), 1.0)
            psi_values = (actual_dist - expected_dist) * np.log(
                (actual_dist + 1e-6) / (expected_dist + 1e-6)
            )
            psi_map[str(group)] = float(psi_values.sum())
            for idx, interval in enumerate(expected_dist.index):
                rows.append(
                    {
                        self.group_col: group,
                        "bin": idx + 1,
                        "score_range": str(interval),
                        "expected_pct": float(expected_dist.iloc[idx]),
                        "actual_pct": float(actual_dist.iloc[idx]),
                        "psi": float(psi_values.iloc[idx]),
                    }
                )

        return pd.DataFrame(rows), psi_map

    def _build_decile_lift_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        ordered_groups: Sequence[str],
    ) -> pd.DataFrame:
        """Build grouped decile lift details ordered by descending model score."""
        rows: List[Dict[str, Any]] = []
        for group in ordered_groups:
            sub_df = df[df[self.group_col].astype(str) == str(group)].copy()
            y_true = pd.to_numeric(sub_df[target_col], errors="coerce")
            y_pred = pd.to_numeric(sub_df[pred_col], errors="coerce")
            valid = sub_df.loc[y_true.notna() & y_pred.notna() & (y_true >= 0)].copy()
            if valid.empty:
                continue
            valid["_target"] = pd.to_numeric(valid[target_col], errors="coerce")
            valid["_score"] = pd.to_numeric(valid[pred_col], errors="coerce")
            valid = valid.sort_values("_score", ascending=False).reset_index(drop=True)
            decile_count = min(10, max(int(valid.shape[0]), 1))
            valid["_decile"] = np.floor(np.arange(valid.shape[0]) * decile_count / valid.shape[0]).astype(int) + 1
            base_bad_rate = float(valid["_target"].mean()) if valid.shape[0] else np.nan
            total_bad = float(valid["_target"].sum())
            for decile, part in valid.groupby("_decile", sort=True):
                bad = float(part["_target"].sum())
                count = int(part.shape[0])
                bad_rate = float(bad / count) if count else np.nan
                rows.append(
                    {
                        self.group_col: group,
                        "decile": int(decile),
                        "count": count,
                        "bad": bad,
                        "bad_rate": bad_rate,
                        "lift": bad_rate / base_bad_rate if base_bad_rate and pd.notna(base_bad_rate) else np.nan,
                        "capture_rate": bad / total_bad if total_bad > 0 else np.nan,
                        "min_score": float(part["_score"].min()),
                        "max_score": float(part["_score"].max()),
                    }
                )
        return pd.DataFrame(rows)

    def _valid_score_arrays(
        self,
        sub_df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return clean binary target and score arrays for chart details."""
        y_true = pd.to_numeric(sub_df[target_col], errors="coerce")
        y_pred = pd.to_numeric(sub_df[pred_col], errors="coerce")
        mask = y_true.notna() & y_pred.notna() & (y_true >= 0)
        return y_true[mask].to_numpy(dtype=float), y_pred[mask].to_numpy(dtype=float)

    @staticmethod
    def _thin_arrays(max_points: int, **arrays: np.ndarray) -> Dict[str, np.ndarray]:
        """Downsample aligned arrays to keep report detail tables lightweight."""
        if not arrays:
            return {}
        size = len(next(iter(arrays.values())))
        if size <= max_points:
            return arrays
        idx = np.unique(np.linspace(0, size - 1, max_points).astype(int))
        return {name: values[idx] for name, values in arrays.items()}

    def _build_roc_curve_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        ordered_groups: Sequence[str],
    ) -> pd.DataFrame:
        """Build ROC curve detail rows for each split."""
        rows: List[Dict[str, Any]] = []
        for group in ordered_groups:
            sub_df = df[df[self.group_col].astype(str) == str(group)]
            y, score = self._valid_score_arrays(sub_df, pred_col=pred_col, target_col=target_col)
            pos = float(y.sum())
            neg = float(len(y) - pos)
            if len(y) == 0 or pos <= 0 or neg <= 0:
                continue
            order = np.argsort(-score)
            y_sorted = y[order]
            score_sorted = score[order]
            tpr = np.r_[0.0, np.cumsum(y_sorted) / pos, 1.0]
            fpr = np.r_[0.0, np.cumsum(1.0 - y_sorted) / neg, 1.0]
            threshold = np.r_[np.inf, score_sorted, -np.inf]
            thinned = self._thin_arrays(500, fpr=fpr, tpr=tpr, threshold=threshold)
            for fpr_val, tpr_val, threshold_val in zip(thinned["fpr"], thinned["tpr"], thinned["threshold"]):
                rows.append(
                    {
                        self.group_col: group,
                        "fpr": float(fpr_val),
                        "tpr": float(tpr_val),
                        "threshold": float(threshold_val) if np.isfinite(threshold_val) else threshold_val,
                    }
                )
        return pd.DataFrame(rows)

    def _build_ks_curve_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        ordered_groups: Sequence[str],
    ) -> pd.DataFrame:
        """Build KS curve detail rows for each split."""
        rows: List[Dict[str, Any]] = []
        for group in ordered_groups:
            sub_df = df[df[self.group_col].astype(str) == str(group)]
            y, score = self._valid_score_arrays(sub_df, pred_col=pred_col, target_col=target_col)
            pos = float(y.sum())
            neg = float(len(y) - pos)
            if len(y) == 0 or pos <= 0 or neg <= 0:
                continue
            order = np.argsort(-score)
            y_sorted = y[order]
            bad_cum = np.cumsum(y_sorted) / pos
            good_cum = np.cumsum(1.0 - y_sorted) / neg
            sample_pct = np.arange(1, len(y_sorted) + 1, dtype=float) / len(y_sorted)
            ks = np.abs(bad_cum - good_cum)
            thinned = self._thin_arrays(
                500,
                sample_pct=sample_pct,
                bad_cum_rate=bad_cum,
                good_cum_rate=good_cum,
                ks=ks,
            )
            for idx in range(len(thinned["sample_pct"])):
                rows.append(
                    {
                        self.group_col: group,
                        "sample_pct": float(thinned["sample_pct"][idx]),
                        "bad_cum_rate": float(thinned["bad_cum_rate"][idx]),
                        "good_cum_rate": float(thinned["good_cum_rate"][idx]),
                        "ks": float(thinned["ks"][idx]),
                    }
                )
        return pd.DataFrame(rows)

    def _build_calibration_curve_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        ordered_groups: Sequence[str],
    ) -> pd.DataFrame:
        """Build reliability diagram detail rows by quantile bin."""
        rows: List[Dict[str, Any]] = []
        for group in ordered_groups:
            sub_df = df[df[self.group_col].astype(str) == str(group)]
            y, score = self._valid_score_arrays(sub_df, pred_col=pred_col, target_col=target_col)
            if len(y) == 0:
                continue
            valid = pd.DataFrame({"target": y, "score": score})
            bin_count = min(10, max(int(valid["score"].nunique()), 1))
            if bin_count <= 1:
                valid["_bin"] = 1
            else:
                try:
                    valid["_bin"] = pd.qcut(valid["score"], q=bin_count, duplicates="drop", labels=False) + 1
                except ValueError:
                    valid["_bin"] = pd.cut(valid["score"], bins=bin_count, duplicates="drop", labels=False) + 1
            for bin_idx, part in valid.groupby("_bin", sort=True):
                rows.append(
                    {
                        self.group_col: group,
                        "bin": int(bin_idx) if pd.notna(bin_idx) else np.nan,
                        "count": int(part.shape[0]),
                        "pred_mean": float(part["score"].mean()),
                        "bad_rate": float(part["target"].mean()),
                    }
                )
        return pd.DataFrame(rows)

    def _build_score_distribution_detail(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        ordered_groups: Sequence[str],
    ) -> pd.DataFrame:
        """Build binned score distribution rows split by target value."""
        scores = pd.to_numeric(df[pred_col], errors="coerce").dropna()
        if scores.empty:
            return pd.DataFrame()
        min_score = float(scores.min())
        max_score = float(scores.max())
        if min_score == max_score:
            min_score -= 1e-6
            max_score += 1e-6
        bins = np.linspace(min_score, max_score, 31)
        rows: List[Dict[str, Any]] = []
        for group in ordered_groups:
            sub_df = df[df[self.group_col].astype(str) == str(group)].copy()
            sub_df["_score"] = pd.to_numeric(sub_df[pred_col], errors="coerce")
            sub_df["_target"] = pd.to_numeric(sub_df[target_col], errors="coerce")
            sub_df = sub_df[sub_df["_score"].notna() & sub_df["_target"].notna() & (sub_df["_target"] >= 0)]
            for target_value, target_part in sub_df.groupby("_target", sort=True):
                counts = pd.cut(target_part["_score"], bins=bins, include_lowest=True).value_counts(sort=False)
                denom = max(float(counts.sum()), 1.0)
                for idx, interval in enumerate(counts.index):
                    rows.append(
                        {
                            self.group_col: group,
                            "target_value": int(target_value),
                            "bin": idx + 1,
                            "score_min": float(interval.left),
                            "score_max": float(interval.right),
                            "bin_center": float((interval.left + interval.right) / 2.0),
                            "count": int(counts.iloc[idx]),
                            "pct": float(counts.iloc[idx] / denom),
                        }
                    )
        return pd.DataFrame(rows)

    @staticmethod
    def _feature_distribution(series: pd.Series, baseline: pd.Series) -> Tuple[pd.Series, str]:
        """Return aligned feature distribution for PSI using numeric bins or categories."""
        baseline_clean = baseline.copy()
        series_clean = series.copy()
        if pd.api.types.is_numeric_dtype(baseline_clean):
            clean = pd.to_numeric(baseline_clean, errors="coerce").dropna()
            if clean.nunique() >= 2:
                bins = np.unique(np.quantile(clean.to_numpy(dtype=float), np.linspace(0, 1, 11)))
                if bins.size >= 2:
                    bins[0] = -np.inf
                    bins[-1] = np.inf
                    dist = pd.cut(pd.to_numeric(series_clean, errors="coerce"), bins=bins, include_lowest=True)
                    dist = dist.astype("object").where(pd.notna(dist), "__MISSING__").astype(str)
                    return dist, "numeric"
        base_str = baseline_clean.astype("object").where(baseline_clean.notna(), "__MISSING__").astype(str)
        top_levels = base_str.value_counts().head(20).index.tolist()
        if "__OTHER__" not in top_levels:
            top_levels.append("__OTHER__")
        values = series_clean.astype("object").where(series_clean.notna(), "__MISSING__").astype(str)
        values = values.where(values.isin(top_levels), "__OTHER__")
        return values, "categorical"

    def _build_feature_psi_detail(
        self,
        df: pd.DataFrame,
        *,
        ordered_groups: Sequence[str],
    ) -> pd.DataFrame:
        """Build feature-level PSI detail rows against the first split."""
        feature_cols = [col for col in self.feature_cols if col in df.columns]
        if not ordered_groups or not feature_cols:
            return pd.DataFrame()
        baseline_group = ordered_groups[0]
        baseline_df = df[df[self.group_col].astype(str) == str(baseline_group)]
        rows: List[Dict[str, Any]] = []
        for feature in feature_cols:
            baseline_bins, bin_type = self._feature_distribution(baseline_df[feature], baseline_df[feature])
            expected_counts = baseline_bins.value_counts(sort=False)
            expected_dist = expected_counts / max(float(expected_counts.sum()), 1.0)
            for group in ordered_groups:
                group_df = df[df[self.group_col].astype(str) == str(group)]
                actual_bins, _ = self._feature_distribution(group_df[feature], baseline_df[feature])
                actual_counts = actual_bins.value_counts(sort=False)
                aligned = pd.concat(
                    [expected_dist.rename("expected_pct"), (actual_counts / max(float(actual_counts.sum()), 1.0)).rename("actual_pct")],
                    axis=1,
                ).fillna(0.0)
                psi_values = (aligned["actual_pct"] - aligned["expected_pct"]) * np.log(
                    (aligned["actual_pct"] + 1e-6) / (aligned["expected_pct"] + 1e-6)
                )
                feature_psi = float(psi_values.sum())
                for bin_label, psi_value in psi_values.items():
                    rows.append(
                        {
                            "feature": feature,
                            self.group_col: group,
                            "bin": str(bin_label),
                            "bin_type": bin_type,
                            "expected_pct": float(aligned.loc[bin_label, "expected_pct"]),
                            "actual_pct": float(aligned.loc[bin_label, "actual_pct"]),
                            "psi": float(psi_value),
                            "feature_psi": feature_psi,
                        }
                    )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["feature_psi", "feature", self.group_col], ascending=[False, True, True])

    def evaluate(self, df: FrameLike, *, pred_col: str) -> MarsModelingReport:
        """Evaluate a scored dataset and return a structured report object."""
        df_pd = self._validate_frame(to_pandas_frame(df), pred_col)
        rows: List[Dict[Any, Any]] = []
        ordered_groups = self._get_ordered_groups(df_pd)
        score_psi_detail, score_psi_map = self._build_score_psi_detail(
            df_pd,
            pred_col=pred_col,
            ordered_groups=ordered_groups,
        )

        for group in ordered_groups:
            sub_df = df_pd[df_pd[self.group_col].astype(str) == str(group)].copy()
            row: Dict[Any, Any] = {self.group_col: group}

            if self.time_col:
                row[("Time Period", "Start Time")] = sub_df[self.time_col].min()
                row[("Time Period", "End Time")] = sub_df[self.time_col].max()

            row.update(
                self._calc_metric_block(
                    sub_df,
                    pred_col=pred_col,
                    target_col=self.target_col,
                    score_psi=score_psi_map.get(str(group)),
                )
            )
            if self.val_target_col:
                row.update(
                    self._calc_metric_block(
                        sub_df,
                        pred_col=pred_col,
                        target_col=self.val_target_col,
                        section_label=f"Val Target: {self.val_target_col}",
                    )
                )
            rows.append(row)

        summary = pd.DataFrame(rows).set_index(self.group_col)
        tuple_cols = [col for col in summary.columns if isinstance(col, tuple)]
        ordered_tuple_cols = self._get_ordered_columns(tuple_cols)
        summary = summary.reindex(columns=ordered_tuple_cols)
        summary.columns = pd.MultiIndex.from_tuples(ordered_tuple_cols)
        detail_tables = {
            "decile_lift": self._build_decile_lift_detail(
                df_pd,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=ordered_groups,
            ),
            "score_psi": score_psi_detail,
            "roc_curve": self._build_roc_curve_detail(
                df_pd,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=ordered_groups,
            ),
            "ks_curve": self._build_ks_curve_detail(
                df_pd,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=ordered_groups,
            ),
            "calibration_curve": self._build_calibration_curve_detail(
                df_pd,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=ordered_groups,
            ),
            "score_distribution": self._build_score_distribution_detail(
                df_pd,
                pred_col=pred_col,
                target_col=self.target_col,
                ordered_groups=ordered_groups,
            ),
        }
        feature_psi = self._build_feature_psi_detail(df_pd, ordered_groups=ordered_groups)
        if not feature_psi.empty:
            detail_tables["feature_psi"] = feature_psi
        metadata: Dict[str, Any] = {
            "group_col": self.group_col,
            "target_col": self.target_col,
            "pred_col": pred_col,
            "benchmark_col": self.benchmark_col,
            "time_col": self.time_col,
            "val_target_col": self.val_target_col,
            "feature_cols": [col for col in self.feature_cols if col in df_pd.columns],
        }
        if self.importance_table is not None:
            metadata["importance_table"] = self.importance_table.copy()
        return MarsModelingReport(
            summary,
            caption=f"Model Evaluation by [{self.group_col}]",
            detail_tables=detail_tables,
            metadata=metadata,
        )

class _ModelPredictor:
    """Internal prediction helper for trained tree models."""

    def __init__(
        self,
        model: Any,
        feature_list: Sequence[str],
        categorical_features: Optional[Sequence[str]] = None,
        category_levels: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> None:
        self.model: Any = model
        self.features: List[str] = list(feature_list)
        self.categorical_features: List[str] = list(categorical_features or [])
        self.category_levels: Dict[str, List[Any]] = {
            str(feature): list(levels)
            for feature, levels in dict(category_levels or {}).items()
        }

    def _safe_predict_logic(self, df: pd.DataFrame) -> np.ndarray:
        """Dispatch prediction logic to the correct backend implementation."""
        missing = sorted(set(self.features).difference(df.columns))
        if missing:
            raise ValueError(f"Input data is missing required features: {missing}")

        X = df.loc[:, self.features].copy()
        for feature in self.categorical_features:
            if feature in X.columns:
                categories = self.category_levels.get(feature)
                if categories is not None:
                    X[feature] = X[feature].astype(pd.CategoricalDtype(categories=categories))
                else:
                    X[feature] = X[feature].astype("category")

        xgb = _optional_import("xgboost")
        lgb = _optional_import("lightgbm")
        catboost = _optional_import("catboost")

        if xgb is not None and isinstance(self.model, getattr(xgb, "Booster", tuple())):
            dtest = xgb.DMatrix(X, enable_categorical=bool(self.categorical_features))
            best_iteration = getattr(self.model, "best_iteration", None)
            if best_iteration is None:
                return np.asarray(self.model.predict(dtest))
            return np.asarray(self.model.predict(dtest, iteration_range=(0, best_iteration + 1)))

        if xgb is not None and isinstance(self.model, getattr(xgb, "XGBModel", tuple())):
            preds = self.model.predict_proba(X)
            return np.asarray(preds[:, 1])

        if lgb is not None and isinstance(self.model, getattr(lgb, "Booster", tuple())):
            best_iteration = getattr(self.model, "best_iteration", None)
            return np.asarray(self.model.predict(X, num_iteration=best_iteration or None))

        if lgb is not None and isinstance(self.model, getattr(lgb, "LGBMModel", tuple())):
            preds = self.model.predict_proba(X)
            return np.asarray(preds[:, 1])

        if catboost is not None and isinstance(self.model, getattr(catboost, "CatBoost", tuple())):
            preds = self.model.predict_proba(X)
            return np.asarray(preds[:, 1])

        raise TypeError(f"Unsupported model type: {type(self.model)!r}")

    def _safe_predict_logic_polars(self, df: pl.DataFrame) -> np.ndarray:
        """Predict directly from Polars/Arrow for numeric-only booster paths."""
        if self.categorical_features:
            return self._safe_predict_logic(df.to_pandas())

        missing = sorted(set(self.features).difference(df.columns))
        if missing:
            raise ValueError(f"Input data is missing required features: {missing}")

        X_arrow = df.select(self.features).to_arrow()
        xgb = _optional_import("xgboost")
        lgb = _optional_import("lightgbm")

        if xgb is not None and isinstance(self.model, getattr(xgb, "Booster", tuple())):
            dtest = xgb.DMatrix(X_arrow)
            best_iteration = getattr(self.model, "best_iteration", None)
            if best_iteration is None:
                return np.asarray(self.model.predict(dtest))
            return np.asarray(self.model.predict(dtest, iteration_range=(0, best_iteration + 1)))

        if lgb is not None and isinstance(self.model, getattr(lgb, "Booster", tuple())):
            best_iteration = getattr(self.model, "best_iteration", None)
            return np.asarray(self.model.predict(X_arrow, num_iteration=best_iteration or None))

        return self._safe_predict_logic(df.to_pandas())

    def predict(
        self,
        df: FrameLike,
        pred_col_name: str = "pred_score",
        inplace: bool = False,
    ) -> FrameLike:
        """Score a dataset and append the prediction column."""
        prefer_polars = is_polars_dataframe(df)
        if prefer_polars and not inplace and isinstance(df, pl.DataFrame):
            preds = self._safe_predict_logic_polars(df)
            return df.with_columns(pl.Series(pred_col_name, preds))
        df_pd = df if isinstance(df, pd.DataFrame) and inplace else to_pandas_frame(df)
        df_pd[pred_col_name] = self._safe_predict_logic(df_pd)
        return restore_frame_type(df_pd, prefer_polars)

    def evaluate(
        self,
        df: FrameLike,
        group_col: str,
        target_col: str,
        *,
        time_col: Optional[str] = None,
        val_target_col: Optional[str] = None,
        benchmark_col: Optional[str] = None,
        pred_col_name: str = "pred_score",
    ) -> MarsModelingReport:
        """Score a dataset and immediately return an evaluation report."""
        scored = self.predict(df, pred_col_name=pred_col_name, inplace=False)
        evaluator = MarsModelEvaluator(
            group_col=group_col,
            target_col=target_col,
            time_col=time_col,
            benchmark_col=benchmark_col,
            val_target_col=val_target_col,
        )
        return evaluator.evaluate(scored, pred_col=pred_col_name)
