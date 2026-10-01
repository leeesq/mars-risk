---
description: 用可选的 Python 3.10+ Agent 模块登记数据、调用监控与分析工具并保留报告证据。
---

# 风控分析 Agent

!!! warning "Experimental · Python 3.10–3.12"

    `mars.agent` 是可选的上层编排模块，首版支持画像、分箱风险评估、监控与报告查询。
    参数、会话及结果契约仍可能调整。核心 MARS 保持 Python 3.8–3.12；在旧解释器上导入
    `mars.agent` 会给出明确错误，不影响普通 `import mars`。

外部 Agent 可独立使用公共报告，无需本模块；见[外部 Agent 指南](external-agents.md)。

## 适用场景与安装

适用于围绕已准备好的 Pandas/Polars 宽表进行交互式风险分析。调用方必须明确数据集、允许分析的
特征、标签、分组列及业务口径；Agent 不按列名猜测好坏定义或观察窗口。

在包含此功能的本地源码目录执行：

```bash
python -m pip install -e ".[agent]"
```

`agent` extra 仅提供 OpenAI 兼容 SDK，依赖带有 Python >=3.10 条件。
使用自定义 provider 或直接执行确定性工具不需要该 SDK。普通 `import mars` 和 `import mars.agent`
不会读取凭据或创建网络连接；创建 `MarsOpenAIProvider` 时才导入 SDK。

## 直接登记已有报告

已完成画像或分箱评估时可以直接登记结果，不必登记原始宽表或重新计算：

```python
import polars as pl
from mars.analysis import profile_stats
from mars.agent import MarsAgentSession, MarsRiskAgent

report = profile_stats(pl.DataFrame({"income": [1000., None, 3000.]}), metrics=["missing", "mean"])
session = MarsAgentSession()
report_id = session.register_report(report)
agent = MarsRiskAgent()
catalog = agent.execute_tool("list_reports", {}, session=session)
page = agent.execute_tool(
    "get_report_table",
    {"report_id": report_id, "table": "overview", "columns": ["feature", "missing_rate"], "limit": 10},
    session=session,
)
assert page.success
```

`register_report()` 接受满足 `mars.reporting.Report` 的对象（包括外部实现及
`load_report()` 恢复的快照），也接受持有 report 的 `MarsRiskProfile`。
返回会话内唯一 ID，可以通过 `report_id=` 指定但不可覆盖。元数据和表保留快照；修改原报告或
`get_report()` 返回的副本不影响会话。外部报告 `dataset_id=None`、`benchmark_id=None`，
`metadata.source` 明确标明 external_report，原报告的参数和 describe 说明保存在 metadata 中。
这些 ID 不代表已经登记原始数据集；Agent 的计算权限没有扩大。

`persistent_report_id` 标识同一分析产物，独立于上述会话句柄。`get_report_table` 的
`evidence_reference` 可在文件恢复后重放；`search_report_features` 按业务名/定义/来源返回
明确英文标识，`get_report_context` 复用公共 `to_ai_context` 的异构查询与字符预算。
`register_dataset` 可传入 `feature_metadata/business_context`，分析工具会向报告传递。
外部消费者无需创建内部会话，完整示例见
[可携带的公共分析报告](reports-and-exports.md#portable-analysis-reports)。

## 先验证确定性工具

下面的完整示例不调用模型、不需要 API Key。它与模型调用使用相同的工具参数验证和 MARS 适配链路。

```python
--8<-- "docs/snippets/agent_monitoring.py"
```

`monitor_result.data` 返回报告 ID、实际参数及可查询表目录。完整表保存在本地 `session`，
模型通过 `get_report_table` 分页获取聚合结果。示例当前期末组未充分表现，
`target_observation` 保留其覆盖率为 0、观察坏率为 null 的语义。

## 接入自然语言分析

在上述示例创建的 `session` 上配置支持 Chat Completions 工具调用的模型：

```python
import os

from mars.agent import MarsOpenAIProvider, MarsRiskAgent

provider = MarsOpenAIProvider(
    model=os.environ["LLM_MODEL"],
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL"),
)
try:
    agent = MarsRiskAgent(provider, max_iterations=8, max_tool_calls=24)
    result = agent.run(
        "比较 current 与 baseline，按 month 监控 score。先检查表现覆盖情况，再解释变化并引用报告。",
        session=session,
    )
    print(result.text)
    print(result.status, result.report_ids)
    followup = agent.run("哪些结论还需要补充观察窗口信息？", session=session)
    print(followup.text)
finally:
    provider.close()
```

`base_url=None` 使用 SDK 默认地址，兼容服务必须同时指定自己的地址、模型和凭据。
模型服务未返回有效 JSON 工具参数时抛出 `ValueError`；网络异常由 SDK 的超时与重试配置控制。
测试使用模拟模型响应，不证明任何具体线上模型的诊断准确率。

## 工具边界

| 工具 | 主要参数 | 返回内容 |
| --- | --- | --- |
| `list_datasets` | 无 | 已登记 ID 和说明 |
| `list_reports` | 可选 offset/limit | 已有报告来源和表目录 |
| `describe_report` | report_id | 单位、实际参数和限制；可按 table/columns/parameter_keys 缩小 |
| `describe_dataset` | `dataset_id` | 允许列的类型、角色、行数和缺失定义 |
| `profile_data` | `dataset_id`, `metrics` | 画像报告；PSI 必须提供 `benchmark_id` |
| `evaluate_risk` | `dataset_id`, 可选 `benchmark_id` | native/quantile 分箱评估报告 |
| `monitor_data` | `dataset_id`, `benchmark_id` | 分布、风险与表现覆盖率报告 |
| `get_report_table` | `report_id`, `table` | 可筛选、排序、投影和分页的聚合结果 |

计算工具可通过 `features` 选择登记特征子集，通过 `group_col` 选择已登记分组。
不传 `group_col` 时按 MARS 默认整体/日期规则处理。`evaluate_risk` 和 `monitor_data`
支持 `n_bins`（默认 5，默认预算范围 2–50，可通过 compute_budget 调整），首版固定无监督 native/quantile 分箱。
画像指标支持 `missing`、`mean`、`std`、`min`、`max`、`psi`。
缺失定义通过登记数据的 `missing_values` 传入；当前与基准必须一致。
PSI 默认不包括缺失及特殊箱，可显式设置 `psi_include_missing`、`psi_include_special`。
标签统计及缺失处理均复用 MARS 原实现。

报告查询支持：

```python
page = agent.execute_tool(
    "get_report_table",
    {
        "report_id": report_id,
        "table": "detail",
        "filters": {"feature": "score"},
        "sort_by": "count",
        "descending": True,
        "columns": ["feature", "count", "bad_rate"],
        "offset": 0,
        "limit": 10,
    },
    session=session,
)
```

筛选为最多四个列的精确值匹配，随后排序、选择列、分页。`next_offset` 非空表示还有结果。
输出超过字符预算时自动减小页长；单行过大时需通过 `columns` 缩小范围。
浮点非有限值使用公共带类型标记 `{"$mars":"float","value":"nan/inf/-inf"}`（value 为三者之一），
JSON null 表示缺失，普通字符串标记保持字符串；完整本地表保留 MARS 原值。Agent 不提供原始样本读取、Shell、
任意 Python/SQL 执行、模型训练或业务决策修改工具。

登记列名、描述、类别分箱和聚合结果可能发送给配置的模型服务。登记前由调用方筛除身份字段，
确认服务可以接收相关内容；聚合输出本身不等同于自动匿名化。

## 计算预算与输出预算

`MarsAgentComputeBudget` 约束 `profile_data`、`evaluate_risk` 和现有 `monitor_data`。
登记可以包含大量授权特征；计算统一解析默认参数后校验实际特征数。
省略 features 和显式传入 201 个特征都在领域计算前拒绝；显式选择少量授权列可继续分析。
普通用户直接调用公共分析 API 不受此预算限制。

| 配置 | 默认上限 | 依据与用途 |
| --- | --- | --- |
| max_features | 200 | 沿用原 schema 的交互特征边界 |
| max_current_rows / max_benchmark_rows | 各 2,000,000 | 单侧防线；还需通过总单元格检查 |
| max_input_cells | 40,000,000 | (当前 + 基准行数) × 特征数，避免两侧宽表一起放大 |
| max_groups | 120 | 两侧分组基数之和，整体行另计；月度／有限客群场景 |
| max_time_windows | 366 | 两侧有效日期基数之和，覆盖按日缺失表；双方全年需显式提高 |
| max_bins | 50 | 数值 n_bins 或估算类别基数；默认沿用原分箱 schema 上限 |
| max_estimated_rows | 500,000 | 汇总、趋势、分箱和按日明细合计的保守行数 |
| max_estimated_cells | 10,000,000 | 同时限制宽趋势和明细结果单元格 |

这些是可调整的执行政策，不是测得的 CPU／内存硬限额。
仓库 `benchmarks/benchmark_analysis.py` 的默认工作负载为 20,000 行、201 特征；
输入单元格政策留出约一个数量级余量，Agent 默认仍要求拆分超过 200 的特征请求。
2,000,000 行允许少量特征的大样本请求，但并不保证任意机器可完成；
分组、日数及结果预算分别约束展开规模。需要更大任务时由调用方评估并提高预算。

预检查顺序：先 O(1) 读取行数／特征数；再投影角色列，使用共享日期表达式统计月度或显式分组、
有效日期数量；有分箱时统计非数值特征基数，两侧相加作为保守正常箱上界。
不扫描数值列的唯一值，不复制宽表，不静默抽样、裁掉特征或截断样本。
profile_data 的 PSI 实际默认 10 箱；其他两工具默认 5 箱。
正常箱之外另估缺失／特殊箱，按日缺失规模也独立计入。

估算使用 S=1+两侧组数+基准存在标记、P=特征数、B=各特征估算正常箱数+2 的总和：
汇总行为 P×S×M（画像 M=请求指标数+2，其他工具 M=16），分箱明细为 B×S
（monitor_data 再乘 2），非画像的按日行为 P×有效日期数×2。
单元格估算为 汇总行×(S+12)+分箱明细行×32+按日行×8。
这覆盖当前工具的主要展开和宽趋势，允许保守高估，不伪装成精确结果大小或运行时限额。

```python
from mars.agent import MarsAgentComputeBudget, MarsRiskAgent

agent = MarsRiskAgent(
    compute_budget=MarsAgentComputeBudget(max_features=40, max_input_cells=8_000_000),
    max_result_chars=16000,
)
```

工具 schema 随该运行器的 feature／n_bins 上限更新；省略参数也受同一运行时校验。
超限返回 `success=False`、`error_code="COMPUTE_BUDGET_EXCEEDED"`，
data 含 `dimension`、`actual`、`limit`、`estimated`、`suggestion`，不包含原始样本，
不会创建伪成功报告。调用方可以缩小范围、登记更短时间／样本切片，或显式调整预算。

成功报告 metadata 分开记录 `agent_compute_budget`（规模、上限、估算版本）和
`agent_output_budget`（max_result_chars）；`agent_parameters` 记录解析后的实际参数。
`max_result_chars` 限制最终 JSON，分页控制输出，不能代替计算预算。

已有 report、get_report_table、query_page、to_ai_context 和 get_report_context 复用已计算结果，
不受单次分析特征数限制；大型报告仍可按输出预算分页读取。
本轮不新增计算缓存；优先使用会话已有报告。会话隔离与数据登记快照语义保持不变。
monitor_data 遵循相同预算，监控模块继续暂停功能迭代。

## 会话、结果与证据

- `MarsAgentSession` 保存登记时投影的数据快照，不允许复用 ID 覆盖数据。
- 同一 session 连续调用 `run()` 会带上前序消息；不同 session 的数据和报告相互隔离。
- 工具顺序执行，允许先创建报告再查询同一报告。一个 session 同时只能执行一个操作。
- `get_report()` 返回完整聚合表及元数据副本，包含 `agent_parameters` 实际参数；不自动落盘。
- `MarsAgentResult.tool_results` 保留本轮工具结果；`report_ids` 只收录实际创建或读取的报告。
- 模型被要求引用 `report_id/table`。报告 ID 可追溯不代表模型每条文字结论已经自动核实。
- `max_iterations`、`max_tool_calls` 限制单轮执行；超预算的最后一批工具整体不执行。
- `max_context_chars` 是字符预算，不是 tokenizer 测量值；预算包含系统提示、工具定义和消息。
  超限会返回 `context_limit`，不自动压缩或拆散工具调用与结果。
- `clear_history()` 清空消息但保留数据和报告，之后可在新请求中指定已有报告 ID。
- `completed` 仅表示模型正常停止；`incomplete` 包括输出截断，其他状态说明具体资源上限。
- Provider 异常不提交未完成轮次；已经成功计算的报告仍可通过 `session.report_ids` 找到。

## 验证

仅运行 Agent 定向测试：

```bash
python -m pytest -q tests/agent
```

覆盖真实 MARS 小数据计算的结果一致性、会话、错误反馈、预算、协议与可选依赖边界。
Python 3.8/3.9 测试收集跳过该目录。公开接口见 [Agent API](../reference/agent.md)。
