"""无需模型凭据的 Agent 工具示例，演示登记、监控和证据读取。"""

import polars as pl

from mars.agent import MarsAgentSession, MarsRiskAgent

baseline_df = pl.DataFrame(
    {
        "score": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        "target": [0, 0, 0, 0, 1, 1, 1, 1],
    }
)
current_df = pl.DataFrame(
    {
        "score": [0.15, 0.3, 0.65, 0.85, 0.2, 0.4, 0.7, 0.95],
        "target": [0, 0, 1, 1, None, None, None, None],
        "month": ["2026-01"] * 4 + ["2026-02"] * 4,
    }
)

session = MarsAgentSession()
session.register_dataset(
    "baseline",
    baseline_df,
    features=["score"],
    target="target",
    description="基准样本；target=1 表示坏样本，0 表示好样本。",
)
session.register_dataset(
    "current",
    current_df,
    features=["score"],
    target="target",
    group_columns=["month"],
    description="当前样本；target 空值表示尚未表现。观察窗口由业务调用方确认。",
)

agent = MarsRiskAgent()
monitor_result = agent.execute_tool(
    "monitor_data",
    {
        "dataset_id": "current",
        "benchmark_id": "baseline",
        "group_col": "month",
        "n_bins": 2,
    },
    session=session,
)
if not monitor_result.success:
    raise RuntimeError(monitor_result.error_message)

report_id = monitor_result.data["report_id"]
observation_result = agent.execute_tool(
    "get_report_table",
    {"report_id": report_id, "table": "target_observation"},
    session=session,
)
report = session.get_report(report_id)
observation_table = report.tables["target_observation"]
