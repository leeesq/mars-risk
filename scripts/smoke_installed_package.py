"""在仅安装 wheel 的全新环境中验证 MARS 默认依赖全链路。"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from html.parser import HTMLParser
from pathlib import Path

import polars as pl


class _SmokeHTMLParser(HTMLParser):
    """记录 HTML 是否包含至少一个开始标签。"""

    def __init__(self) -> None:
        super().__init__()
        self.has_start_tag = False

    def handle_starttag(self, tag: str, attrs: object) -> None:
        """记录解析到的开始标签。"""
        del tag, attrs
        self.has_start_tag = True


def _assert_installed_import(repo_root: Path) -> Path:
    """确认 mars 来自当前环境 site-packages，而非仓库源码。"""
    import mars

    package_file = Path(mars.__file__).resolve()
    source_root = (repo_root / "src").resolve()
    if source_root == package_file or source_root in package_file.parents:
        raise AssertionError(f"mars was imported from repository source: {package_file}")
    prefix = Path(sys.prefix).resolve()
    if prefix not in package_file.parents:
        raise AssertionError(f"mars is outside the active environment {prefix}: {package_file}")
    return package_file.parent


def _import_all_modules() -> None:
    """递归导入默认依赖支持的全部 mars 模块。"""
    import mars

    failures = []
    for module_info in pkgutil.walk_packages(mars.__path__, prefix="mars."):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:
            failures.append(f"{module_info.name}: {type(exc).__name__}: {exc}")
    if failures:
        raise AssertionError("Recursive imports failed:\n" + "\n".join(failures))


def _sample_frame() -> pl.DataFrame:
    """构造同时覆盖报告、分箱和筛选的小型二分类样本。"""
    return pl.DataFrame(
        {
            "apply_date": [
                "2026-01-03", "2026-01-10", "2026-01-17", "2026-01-24",
                "2026-02-03", "2026-02-10", "2026-02-17", "2026-02-24",
                "2026-03-03", "2026-03-10", "2026-03-17", "2026-03-24",
            ],
            "period": ["2026-01"] * 4 + ["2026-02"] * 4 + ["2026-03"] * 4,
            "income": [2600, 3100, 4100, 5200, 2800, 3500, 4600, 5900, 3000, 3900, 5000, 6500],
            "utilization": [0.82, 0.70, 0.42, 0.18, 0.78, 0.61, 0.35, 0.15, 0.75, 0.55, 0.28, 0.10],
            "target": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
        }
    )


def _assert_nonempty_html(path: Path) -> None:
    """确认 HTML 文件非空且可被标准解析器解析。"""
    content = path.read_text(encoding="utf-8")
    parser = _SmokeHTMLParser()
    parser.feed(content)
    if not content.strip() or not parser.has_start_tag:
        raise AssertionError(f"HTML report is empty or invalid: {path}")


def run_smoke(repo_root: Path, output_dir: Path) -> None:
    """执行安装后运行链路和资源校验。"""
    package_dir = _assert_installed_import(repo_root)
    _import_all_modules()

    from mars.analysis import profile_risk
    from mars.feature import MarsStatsSelector

    templates = package_dir / "reporting" / "template"
    for file_name in ("mars_bin_report_linux.xlsx", "mars_bin_report_win_mac.xlsx"):
        template = templates / file_name
        if not template.is_file() or template.stat().st_size == 0:
            raise AssertionError(f"Installed template is missing or empty: {template}")

    df = _sample_frame()
    profile = profile_risk(
        df,
        target="target",
        features=["income", "utilization"],
        group_col="period",
        time_col="apply_date",
        binning_type="native",
        method="quantile",
        n_bins=3,
        n_jobs=1,
    )
    styler_html = profile.report.show_summary().to_html()
    if "<table" not in styler_html:
        raise AssertionError("Pandas Styler did not render an HTML table.")

    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / "risk_report.xlsx"
    html_path = output_dir / "risk_report.html"
    profile.report.write_excel(str(excel_path))
    profile.report.write_html(str(html_path))
    if not excel_path.is_file() or excel_path.stat().st_size == 0:
        raise AssertionError("Excel report was not written.")
    _assert_nonempty_html(html_path)

    selector = MarsStatsSelector(
        skip_fine_scan=True,
        rough_iv_thr=-1.0,
        rough_lift_thr=0.0,
        psi_thr=None,
        rc_thr=None,
        corr_thr=None,
        n_jobs=1,
    )
    selector.fit(df, target="target", features=["income", "utilization"])
    if not selector.selected_features_:
        raise AssertionError("MarsStatsSelector did not retain any smoke-test feature.")


def parse_args() -> argparse.Namespace:
    """解析安装后 smoke 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """执行 installed-wheel smoke。"""
    args = parse_args()
    run_smoke(args.repo_root.resolve(), args.output_dir.resolve())
    print("Installed-wheel smoke passed.")


if __name__ == "__main__":
    main()
