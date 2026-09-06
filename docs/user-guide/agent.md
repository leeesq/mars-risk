---
description: 用可选的 Python 3.10+ Agent 模块登记数据、调用监控与分析工具并保留报告证据。
---

# 风控分析 Agent

!!! warning "Experimental · Python 3.10–3.12"

    `mars.agent` 是可选的上层编排模块，首版支持画像、分箱风险评估、监控与报告查询。
    参数、会话及结果契约仍可能调整。核心 MARS 保持 Python 3.8–3.12；在旧解释器上导入
    `mars.agent` 会给出明确错误，不影响普通 `import mars`。

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
| `describe_dataset` | `dataset_id` | 允许列的类型、角色、行数和缺失定义 |
| `profile_data` | `dataset_id`, `metrics` | 画像报告；PSI 必须提供 `benchmark_id` |
| `evaluate_risk` | `dataset_id`, 可选 `benchmark_id` | native/quantile 分箱评估报告 |
| `monitor_data` | `dataset_id`, `benchmark_id` | 分布、风险与表现覆盖率报告 |
| `get_report_table` | `report_id`, `table` | 可筛选、排序、投影和分页的聚合结果 |

计算工具可通过 `features` 选择登记特征子集，通过 `group_col` 选择已登记分组。
不传 `group_col` 时按 MARS 默认整体/日期规则处理。`evaluate_risk` 和 `monitor_data`
支持 `n_bins`（默认 5，范围 2–50），首版固定无监督 native/quantile 分箱。
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
非有限数值转为 JSON null，完整本地表保留 MARS 原值。Agent 不提供原始样本读取、Shell、
任意 Python/SQL 执行、模型训练或业务决策修改工具。

登记列名、描述、类别分箱和聚合结果可能发送给配置的模型服务。登记前由调用方筛除身份字段，
确认服务可以接收相关内容；聚合输出本身不等同于自动匿名化。

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
