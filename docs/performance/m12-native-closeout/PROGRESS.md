# M12 Native Closeout — 执行进度 / Progress

[中文规划](PLAN_CN.md) · [English plan](PLAN_ENG.md) · [Codex task](CODEX_START.md)

## 当前状态 / Current status

**Planning only / 仅规划已落库。** 此文件初始化不代表开始计算或完成修复。No implementation, test, real-data execution, or capacity result is claimed by this planning commit.

- Repository: `leeesq/mars-risk`
- Inspected baseline: `05b212e9b720e9ec6e9af0bec050e2be2685dc36`
- Work branch: `fix/m12-native-capacity-closeout`
- Historical upstream evidence: [m12-upstream.md](../m12-upstream.md)
- Scope: native upstream analysis/selection acceptance and reproduced fixes; not Agent Runtime changes or predictive-model training.

## 工作包 / Work packages

| ID | 工作包 / Work package | Status | 本轮证据 / Current-round evidence |
| --- | --- | --- | --- |
| N0 | 基线、环境、小型 fresh worker / Baseline, environment, smoke | not_started | — |
| N1 | 冻结分析输入 / Frozen analysis inputs | not_started | — |
| N2 | 四项完整原生调用 / Four complete native calls | not_started | — |
| N3 | 复现后的最小修复 / Reproduced minimal repairs | not_started | 失败后触发；无缺陷则 not_required / Conditional |
| N4 | 数值、报告与下游交接 / Parity, reports, downstream handoff | not_started | — |
| N5 | 容量边界与交付 / Capacity boundaries and delivery | not_started | — |

## 真实 case / Real cases

| Case | Status | 人口/特征核验 / Population-feature verification | 计算与保存 / Compute-save | 冷消费 / Cold consume |
| --- | --- | --- | --- | --- |
| Partition profile | not_started | — | — | — |
| Daily profile | not_started | — | — | — |
| Development risk | not_started | — | — | — |
| Native selector | not_started | — | — | — |
| 50k×2000 high-survival synthetic | not_started | — | — | — |
| Larger optional scale | deferred | 未指定执行 / Not scheduled | — | — |

## 验收映射 / Acceptance mapping

T01–T10 的定义见双语规划；逐项补实际源码、命令、退出码和本轮证据。Use the definitions in both plans and add actual revision, command, exit status, and evidence for each item. Do not copy historical passing counts into current results.

| IDs | Status | Evidence |
| --- | --- | --- |
| T01, T02, T03 | not_started | — |
| T04, T05 | not_started | — |
| T06, T07, T08 | not_started | — |
| T09, T10 | not_started | — |

## 记录纪律 / Recording rules

每个完成项追加：实际时间与时区、源码版本、影响范围、精确命令、passed/failed/skipped/deselected、退出码、测量口径、限制、下一步。Append actual timestamp/timezone, source, changed scope, exact command, test counts, exit code, measurement definition, limitation, and next action.

真实输入路径、原始特征名、业务结果、完整数据指纹及 sidecar 留本地；公开页只放脱敏索引与结果。Private paths, original features, business outputs, complete data fingerprints, and sidecars stay local. Public evidence references are sanitized, not invented accessible artifacts.

首次实际执行前登记本地输入可用性；没有输入用 pending_local_input。原生成功与下游集成、模型效果分开。确认无新缺陷时允许 N3=not_required。Do not create unnecessary code changes to mark a repair task complete.

## 当前限制 / Current limitations

本规划尚未确认本机输入路径、尚未测 Mac/16 GiB 原生四项调用、未运行下游 Agent，未评估模型效果。The planning task has not established local input availability, measured the four native Mac/16 GiB calls, run the downstream Agent, or assessed model quality.
