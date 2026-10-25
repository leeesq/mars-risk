# Codex 首轮执行任务：MARS 原生宽表容量收口

本文件是可直接执行的任务说明，不是已完成记录。先读 [中文规划](PLAN_CN.md)、[英文镜像](PLAN_ENG.md) 和 [进度](PROGRESS.md)。固有标识符保持英文。

## 任务目标

继续 `leeesq/mars-risk` 的 `fix/m12-native-capacity-closeout` 分支。基线计算源码为 `05b212e9b720e9ec6e9af0bec050e2be2685dc36`，其后规划提交只增加四份文档。先证明最新原生 MARS 能否处理原 M12 宽表，再修复实际复现的上游缺口，不重复开发已合入修复。

本轮不是重启建模 Agent，不训练模型、不访问数据库、不接 LLM、不修改另一个仓库、不放松任何样本/标签/PSI/相关性口径。用户已经授权本仓库修复与此前 M12 本地数据的受限使用，不要重新索取已确认的业务决议。

## 1. 安全接续

在 MARS 项目目录执行 git status、branch、remote 检查。fetch 后核对远端分支；本地没有该分支且工作区安全时，从 origin/fix/m12-native-capacity-closeout 建立 tracking 分支。已存在则安全续接。不 reset、stash、rebase、force push、自动 merge；不覆盖用户修改。实际基线有新提交时先阅读差异并记录，不回退到旧代码。

读取适用 AGENTS.md、.codex/skills/mars-risk-engineering/SKILL.md、性能与质量参考、pyproject、当前 CI、docs/performance/m12-upstream.md 和现有 benchmark/test 实现。

真实本机优先使用已有 `$HOME/miniforge3/envs/modeling_xgb082/bin/python`；验证实际 Python 3.8.20、Polars 1.8.2 及其余已固定依赖，XGBoost 0.82 不升级且不训练。验证新 MARS 用显式当前仓库 src 的 PYTHONPATH，记录 mars.__file__；不改变旧冻结 Run 的 editable 安装或环境。依赖/平台不同要如实记录，不凭版本字符串推断使用了新源码。

## 2. 首先复用，不先写框架

复用 benchmark_core_capacity.py、benchmark_core_workloads.py、_capacity_evidence.py 和当前 frozen_* 入口。先读 --help、做小型 fresh-worker 闭环并保存精确命令。定向执行现有多chunk、lazy、streaming对齐、blocked correlation、empty PSI、capacity comparison 测试。只补实际缺少的测试。

不要开工先跑整个历史 M12 或全部重型回归。不要添加通用资源调度器、第二套 selector、永久宽表缓存或新的Agent协议。

## 3. 从已授权工作区取得冻结输入

只检查此前用户指定的 M12 目录及其 manifest 引用，不扫描整个 HOME。真实数据不复制到 Git 跟踪目录。优先复用已保存的 normalized 分析输入；只有raw时按规划N1做显式一次性导出并核验，benchmark reader不能隐式替换缺失或标签。

真实 case 为原1,111候选，不是交付模型的少量入模列：

- 分区画像：297440行、独立Train benchmark180893行，Feature-only。
- 日画像：相同完整人口和候选，Feature-only，日期/粒度来自冻结计划。
- development risk：257997行Train/Val/Test，Train benchmark180893行；不含OOT。
- selector：297440行、Train benchmark180893行，非Train target物理null。

复制完整实际业务配置，检查合并后的effective parameters，不使用合成fixture默认值。保留所有候选及其顺序，不抽样、不改阈值、不新增来源过滤。不能用supervision_group误伤risk的合法development标签。

输入缺失或无法证明等价时记录pending_local_input/input_contract_blocked，完成不依赖真实输入的工作，集中说明缺什么；不能伪造数据或把合成成功当真实成功。有输入就继续，不逐项重复问批准。

## 4. 串行验证四项原生公共调用

按分区profile、日profile、risk、selector顺序执行。每项先一次单worker；harness预热与重复另计。先确认timeout行为，不将人为超时当OOM，不添加基于available的提前拒绝门。资源只观测，不kill用户无关进程，不无界重试。

每项必须在一个worker内完成公共计算、必要报告和保存，不在selector内部各阶段重启进程来藏峰值。禁止外层三份selector/外层risk合并冒充原生通过。风险公共wrapper转发用小型对照覆盖；实际规模复用原harness。

记录阶段与整体时间、实际批宽、人数/候选去向、主/树RSS与lifetime高水位，能获取的Darwin footprint/swap独立列出；测不到写unknown。保存每份完成报告，再新进程读取并用公共查询验证，不重算、不读原始宽表。

已成功阶段不因后续失败而无条件重跑；只在输入/源码/参数影响其语义时重验。历史旧版全宽OOM无需为A/B再次复现。

## 5. 失败后才修代码

分类问题并保留完整失败记录。先小型复现，再最小修复共享上游逻辑，跑定向回归，再复测相同真实失败case。保持WOE、人口、缺失、strict threshold、IV/ID顺序、白名单、报告合同；不放宽断言/容差，不以删除完整相关关系报告换取通过。

观测wrapper可以计时但不能改生产结果。没有分配证据不要把某一语句写成唯一根因。若本轮原生路径全部通过，N3标not_required，不为制造代码提交而重构。

## 6. 完成数值和交付核对

使用同一修正harness的输入、工作量、资源和sidecar比较门。策略不同的旧分批结果仅作明确标注的参考，不直接计算普通性能比例。完整离散结果/名单/顺序必须一致，未知差异拒绝。

按规划补50k×2000高存活率合成观察，记录真正进入全局相关性的列数。更大规模不默认展开，未跑标Deferred，不将1111列通过推广到5887列。

形成下游适配移除建议，但不实际修改Agent或旧Run：整表rechunk、外层risk分批、三份selector各自写清证据和保留条件。MARS通过不代表Agent重复View/hash/Registry恢复或过早收敛已修好，也不承诺KS提高。

## 7. 测试、文档、提交

开发中定向回归；核心源码稳定后执行适用Python3.8基础套件及现有静态门。其他版本由已有隔离环境/CI覆盖，不升级建模环境、不删测试、不屏蔽warnings。缺少工具如实记not_run。精确命令、计数、退出码及本轮源码均保存。

更新PROGRESS.md与双语规划受影响内容，新增脱敏验收说明并在m12-upstream追加索引，保留全部历史失败。普通push当前分支，不merge、发布、部署或改包版本。真实数据、完整报告、配置、特征名、绝对路径、指纹与sidecar保持本地，提交前逐文件检查，不用git add .。

最终分开汇报：源码/定向回归；四项原生真实case；保存/冷消费；实际资源和规模边界；下游尚未验收事项；模型效果未评估。通过项给证据，未完成项给具体原因。任务应推进到上述合法终态，不再停在重复讨论已授权的路线。
