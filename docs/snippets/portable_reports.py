"""完整报告跨进程使用示例；业务定义仅为模拟数据定义。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import polars as pl

from mars.analysis import profile_risk, profile_stats
from mars.reporting import load_report


def main() -> None:
    """生成、展示、导出、保存和恢复报告，并演示两种 Agent 消费方式。"""
    df = pl.DataFrame(
        {
            "income": [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
            "salary": [2100, 2800, 4200, 4700, 5900, 7100, 8100, 8900],
            "y30": [0, 1, 0, 1, None, None, None, None],
            "y60": [None] * 8,
            "split": ["train"] * 4 + ["oot"] * 4,
        }
    )
    dictionary = {
        "income": {
            "display_name": "月收入",
            "description": "模拟申请的申报税后月收入",
            "data_source": "application",
            "unit": "CNY/month",
        },
        "salary": {
            "display_name": "月收入",
            "description": "模拟授权流水的入账月收入",
            "data_source": "bank",
            "unit": "CNY/month",
        },
        "other_project_feature": {"display_name": "其他项目字段"},
    }
    business_context = {
        "labels": {
            "y30": {
                "definition": "模拟首期还款违约",
                "positive_class": "违约",
                "negative_class": "履约",
                "performance_window": "放款后30天",
            },
            "y60": {"definition": "模拟第二期还款违约", "performance_window": "放款后60天"},
        },
        "sample": {
            "scope": "示例模拟申请",
            "filter": "全部8条模拟样本",
            "time_range": ["2026-01-01", "2026-02-28"],
        },
        "splits": {"train": "模拟开发样本", "oot": "模拟未来样本"},
        "currency": "CNY",
        "score_direction": "unknown",
    }
    rows = [{"feature": feature, **metadata} for feature, metadata in dictionary.items()]
    # DataFrame 必须含明确的 feature 标识列；两种后端使用同一套校验。
    pandas_dictionary = pd.DataFrame(rows)
    polars_dictionary = pl.DataFrame(rows)
    profile = profile_stats(
        df,
        features=["income", "salary"],
        metrics=["missing", "mean"],
        group_col="split",
        feature_metadata=pandas_dictionary,
        business_context=business_context,
    )
    polars_profile = profile_stats(
        df, features=["income", "salary"], metrics=["mean"], feature_metadata=polars_dictionary
    )
    assert polars_profile.get_feature("income")["metadata"]["display_name"] == "月收入"
    displayed = profile.show_overview(limit=2).data
    print(displayed.loc[:, ["feature", "display_name", "mean"]])
    run = profile_risk(
        df,
        target=["y30", "y60"],
        features=["income", "salary"],
        group_col="split",
        n_bins=2,
        feature_metadata=dictionary,
        business_context=business_context,
    )
    report = run.report
    print(report.show_summary(limit=4).data)
    candidates = report.search_features("月收入")
    assert {item["feature"] for item in candidates} == {"income", "salary"}
    identifier = report.search_features("授权流水", sources="bank")[0]["feature"]
    print(report.get_feature(identifier)["metadata"])
    print(report.get_table("calculation_status", filters={"target": "y60"}))
    persistent_id = report.report_id
    with TemporaryDirectory() as directory:
        root = Path(directory)
        report.write_html(str(root / "analysis.html"), include_charts=False)
        report.write_excel(str(root / "analysis.xlsx"))
        path = root / "analysis.marsreport"
        report.save(path)
        # 新 Python 进程只读取文件；没有原宽表、分箱器或会话。
        code = (
            "from mars.reporting import load_report; import sys; "
            "r=load_report(sys.argv[1]); "
            "assert r.report_id == sys.argv[2]; "
            "print(r.get_table('summary', features='income', limit=2))"
        )
        subprocess.run([sys.executable, "-c", code, str(path), persistent_id], check=True)
        del report, run, df
        restored = load_report(path)
        assert restored.report_id == persistent_id
        # 外部建模/诊断 Agent 直接使用公共契约，不需要 MarsAgentSession。
        catalog = restored.describe()
        print(list(catalog["tables"]))
        page = restored.query_page(
            "detail",
            features=identifier,
            filters={"y": "y30"},
            sort_by="bin_index",
            offset=0,
            limit=3,
        )
        print(page["reference"], page["total_rows"], page["next_offset"])
        context = restored.to_ai_context(
            features=identifier,
            max_chars=8000,
            queries={
                "summary": {"sort_by": "iv", "descending": True, "limit": 2},
                "detail": {
                    "filters": {"y": "y30"},
                    "columns": ["feature", "y", "bin_index", "count", "bad_rate"],
                    "limit": 3,
                },
                "calculation_status": {"filters": {"target": "y60"}, "limit": 3},
            },
        )
        assert len(context) <= 8000
        assert json.loads(context)["evidence"]
        # 内部 Agent 也是该契约的消费者；句柄与持久标识分别记录。
        if sys.version_info >= (3, 10):
            from mars.agent import MarsAgentSession, MarsRiskAgent

            original_session = MarsAgentSession()
            original_session.register_report(restored)
            del original_session
            session = MarsAgentSession()
            handle = session.register_report(restored)
            result = MarsRiskAgent().execute_tool(
                "get_report_table",
                {"report_id": handle, "table": "summary", "features": [identifier], "limit": 2},
                session=session,
            )
            assert result.success
            assert result.data["persistent_report_id"] == persistent_id
            print(handle, persistent_id)
        restored.write_html(str(root / "restored.html"))
        restored.write_excel(str(root / "restored.xlsx"))


if __name__ == "__main__":
    main()
