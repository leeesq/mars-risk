"""建模报告的单文件 HTML 渲染器。"""

from __future__ import annotations

import base64
from io import BytesIO
import html
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from mars.modeling.report import MarsModelingReport
from mars.modeling.utils import optional_import as _optional_import

class _ModelReportHtmlRenderer:
    """
    建模报告单文件 HTML 渲染器。

    Notes
    -----
    图表使用 Matplotlib 渲染为 PNG 后以 Base64 内嵌，避免报告依赖外部 CDN 或
    Python 运行环境。
    """

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
        self.feature_growth_summary = self._coerce_optional_frame(self.metadata.get("feature_growth_summary"))

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

    def _feature_growth_chart(self, summary: Optional[pd.DataFrame]) -> str:
        if summary is None or summary.empty or "feature_count" not in summary.columns:
            return '<div class="mars-empty">Feature growth data is unavailable.</div>'
        metric = str(
            self.metadata.get(
                "feature_growth_selection_metric",
                self.metadata.get("optimize_metric", "ks"),
            )
        ).lower()
        y_cols = [col for col in [f"train_{metric}", f"val_{metric}", "selection_score"] if col in summary.columns]
        oot_cols = [
            col for col in summary.columns
            if str(col).lower().startswith("oot") and str(col).lower().endswith(f"_{metric}")
        ]
        y_cols.extend(oot_cols[:3])
        if not y_cols:
            return '<div class="mars-empty">No plottable feature growth metrics were found.</div>'

        if "status" in summary.columns:
            plot_df = summary[summary["status"].astype(str) == "complete"].copy()
        else:
            plot_df = summary.copy()
        if plot_df.empty:
            return '<div class="mars-empty">No successful feature growth step was found.</div>'
        plt = self._require_pyplot()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        x = pd.to_numeric(plot_df["feature_count"], errors="coerce")
        for col in y_cols:
            y = pd.to_numeric(plot_df[col], errors="coerce")
            if y.notna().any():
                ax.plot(x, y, marker="o", linewidth=1.6, markersize=4, label=str(col))
        best_step = self.metadata.get("feature_growth_best_step")
        if best_step is not None:
            ax.axvline(float(best_step), color="#ef4444", linestyle="--", linewidth=1.2, label="best step")
        ax.set_title("Feature Growth Audit")
        ax.set_xlabel("Feature count")
        ax.set_ylabel(metric.upper())
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

        if self.feature_growth_summary is not None and not self.feature_growth_summary.empty:
            growth = self._feature_growth_chart(self.feature_growth_summary)
            growth += self._table_html(self.feature_growth_summary, table_id="feature-growth-table", max_rows=300)
            body_parts.append(self._section("Feature Growth Audit", growth, "feature-growth-audit", open_by_default=False))

        scorecard_section = self._scorecard_section()
        if scorecard_section:
            body_parts.append(scorecard_section)

        return self._build_document("".join(body_parts))
