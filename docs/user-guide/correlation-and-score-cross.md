# 相关性证据与双模型分交叉

两个功能均输出独立报告，不需要 MarsAgentSession。`.marsreport` 沿用 manifest +
Parquet 格式，保留身份、原生表、稳定顺序和业务语义；没有 pickle、个体数据或模型。
所有机器比例用小数，`delta_vs_row` 为小数差，展示时转为百分点。

<span id="相关性报告"></span>

## 相关性报告

`MarsLinearSelector.fit(X, y, features=..., feature_metadata=..., business_context=...)`
和 `MarsStatsSelector.fit(df, target=..., ...)` 后调用 `get_correlation_report()`。
`get_report()` 的决策 DataFrame 返回类型保持不变。

| 筛选器 | 计算输入及缺失处理 | 阈值/优先级 |
| --- | --- | --- |
| Stats | Stage 3 binner 的目标感知 WOE；规范化 target 非空行；WOE null 填 0；Pearson | 严格 `>`；IV 降序，同值按 ID 升序，白名单保护 |
| Linear | 请求池中可数值化的字段与 target 一起删除不完整行；inf 转 NaN；默认 raw Spearman | `>=`；绝对 target Spearman，强度同值保留左侧候选 |

Linear 的清洗池包含随后因 target strength 无效而排除的常量列，未改为 pairwise deletion。
Stats 在 `skip_fine_scan=True` 时继承 rough binner 作为 Stage 3 的 WOE 来源；参数会记录
实际来源和拟合配置。两种表示/方法/样本范围不应直接当成同一口径比较。

`features` 是输入特征目录，包含未参与状态、实际对角值及最终入选状态；`pairs` 保存
完整相关阶段候选池的规范上三角（含被剔除者）；`correlation_decisions` 是执行时的
结构化事件；`selection` 和 Stats 的 `funnel` 沿用已有记录。
signed 值只有一份规范存储；`abs_correlation=abs(correlation)` 物化用于原生排序。
非有限相关值存为 null + unavailable，不伪造 0 或对角 1；引擎无法确认的原因不细分。
元数据样本数为全局计算输入行数，绝不冒充原始逐对非缺失重叠数。
Linear 的 Checked/max_corr 复用同一矩阵，统计原候选池其他特征（包含已剔除者）的
最大绝对相关；它不表示最终入选特征之间的最大值，说明文本已澄清此范围。

```python
from mars.reporting import (
    load_report, get_related_features, get_correlation_matrix, show_correlation_matrix,
)

corr = selector.get_correlation_report()
corr.get_table("pairs", features=["loan_cnt_3m"],
               sort_by="abs_correlation", descending=True, limit=20)
corr.get_table("correlation_decisions", features="loan_cnt_3m")
corr.save("selector-correlation.marsreport")
restored = load_report("selector-correlation.marsreport")
peers = get_related_features(restored, "loan_cnt_3m", sources="credit", limit=20)
matrix = get_correlation_matrix(restored, ["loan_cnt_3m", "inq_cnt_1m", "income"])
show_correlation_matrix(restored, max_features=30)  # Notebook Styler，可 to_html/to_excel
```

`pairs.features` 匹配任一端点，多 ID 采用并集；`pairs.sources` 匹配任一端点来源。
两种公共条件采用交集，可以由不同端点分别匹配。`get_related_features.sources` 只匹配
peer 来源，另一端统一命名为 `peer_feature`，返回最终状态及真实触发事件引用。
同绝对相关值保持规范候选顺序；分页稳定。普通表查询未知 ID 返回空表；专用矩阵/peer
操作未知候选 ID 报 ValueError。矩阵是所选集合的诱导子矩阵，不重新计算。
矩阵默认最多显示 30 个特征，固定 [-1,1] 发散色标，提示省略数；完整数据仍可查询。

每次 fit 先重置缓存，失败后不能读上次报告。相关关闭、无候选、候选不足分别记录
skipped 状态；未计算的对角为空值。仅在 fit 成功后固定快照并释放 dense 缓存。

<span id="固定分段交叉"></span>

## 固定分段交叉

```python
from mars.analysis import cross_scores, get_score_bin_definitions, get_score_cell, show_score_matrix

report = cross_scores(
    df, score_x="score_main", score_y="prob_aux", targets=["fpd7", "mob1_dpd7"],
    score_directions={"score_main": "lower_risk", "prob_aux": "higher_risk"},
    binning_reference=df_reference, n_bins=5, group_col="dataset_flag",
    probability_scores=["prob_aux"], min_observed=30,
    feature_metadata=metadata, business_context=context,
)
show_score_matrix(report, filters={"target": "fpd7", "group": "OOT"})
show_score_matrix(report, filters={"target": "fpd7", "group": "OOT"},
                  metric="delta_vs_row", color_range=(-0.1, 0.1))
get_score_cell(report, "b2", "b1", filters={"target": "fpd7", "group": "OOT"})
report.get_table("cells", filters={"target": "fpd7", "group": "OOT", "x_bin": "b2"},
                 sort_by="y_risk_rank")  # 同一 X 分段内 Y 的有序梯度
```

两个分数已经在同一行对齐；不做独立表合并。输入先投影必要列，再跨引擎转换。
标签批量联合聚合到一份格子分布，小统计表再展开不同 target；不同标签人数不能相加。
`group_col` 与 `time_col` 可同时提供，只输出实际 group/period 组合；时间解析及粒度
复用既有语义。`max_scopes` 默认 10000，超过时报错，避免无界输出。

分段配置互斥：显式 `cutpoints={两个原始ID: 切点列表}`、`binning_reference` 或
`bin_definitions=get_score_bin_definitions(previous_report)`。无配置时全量当前输入各轴
拟合一次，`fit_source=current_input`，不会猜哪个分组是训练集。所有 target/分组/月共用
同一份 bins。默认请求 5 段，也可用 `n_bins={"score_main": 4, "prob_aux": 6}`；
实际正常箱数记录在定义及 `parameters.actual_n_bins`，重复值、常量、小样本或合并可减少箱数。
不同报告不能假定分位箱可比，应复用保存定义。没有有效 reference 分数时须给显式切点。

### 与 profile_risk 共用分箱

自动路径使用 `profile_risk` 的配置解析和现有分箱器，不在 Score Cross 重写分位点、等宽或树。

| 参数 | 行为 |
| --- | --- |
| `binning_type` | `native`（默认）、`optimal`、`lite_opt`；不支持历史 `opt` 别名 |
| `method` | `quantile`、`uniform`、`cart`；None 使用对应引擎的实际默认值，最优引擎按现有规则映射预分箱方法 |
| `n_bins` | 整数 1–100，或同时覆盖两个原始 score ID 的整数映射；特殊箱不占正常箱数 |
| `min_bin_size` | 按共享引擎的有效样本及分母约束拟合参考集；native CART 整数为人数、浮点为比例，最优引擎使用比例；实际配置随定义保存 |
| `monotonic_trend` | 共用引擎的趋势约束；native 不执行趋势约束，沿用现有警告 |
| `missing_values` | 共享缺失语义；可用全轴列表或按 score ID 提供列表，保存后继续复用 |
| `special_values` | 按 score ID 指定有限特殊值，优先于概率域检查，特殊箱无 risk_rank |
| `binner_params` | native 可透传 `cart_params`、`merge_small_bins`、`remove_empty_bins`；已有但不适用的参数警告忽略，未知或重复公开控制项报错 |
| `binning_target` | 监督拟合目标；默认首个 `targets`，也可明确指定参考集中的另一列。保存实际目标，不随页面切换重拟合 |
| `n_jobs` | 共用分箱器的并行参数；不更改报告的 scope 定义 |

`method="custom"` 不是公开方法。自定义走 `cutpoints`，保存复用走 `bin_definitions`。
公开 method、n_bins 等控制项不能重复塞进 `binner_params`，`prebinning_method` 也不能用来
绕过高层 method 契约；冲突项按 `profile_risk` 的原规则报错。
这两条无拟合路径不能附带 method、非默认引擎、监督目标或其他拟合配置；明确报错，不静默猜优先级。
`n_bins` 的旧默认和显式切点调用兼容，实际箱数以切点/定义为准。

监督路径（native cart、optimal、lite_opt）在所选参考集中验证 0/1 标签，剔除未表现标签并检查
各轴可用分数样本是否具备两个类别；全好/全坏仍可用于报告评估，但不满足监督拟合要求。
拟合过程不使用评估 `weights_col` 重新定义树目标。每轴定义的 `fit` 保存参考/可用样本数、
目标、实际引擎参数、实际箱数及诊断；`get_score_bin_definitions` 将这些来源随定义返回。
保存定义复用时 `fit_performed=False`，历史 `binning_target`/`fitted_targets` 仍明确记录。
`binning_reference` 可来自任何用户指定的参考集，TRAIN/VAL/TEST/OOT 只是普通组名。

已启用约束的自动拟合路径会将共享引擎的左闭切点适配为右闭定义，保留拟合参考样本的箱成员；切点恰好命中
观测值时使用其前一个真实观测值作为右闭上界，不添加 epsilon。已启用的最小箱约束按最终
右闭赋箱独立检查，实际箱数减少或无法满足的情况随 `fit` 记录。约束只针对拟合参考集，
固定定义用于 OOT、独立验证集或其他数据时不重新拟合，也不强制新数据保持原箱占比。
未启用约束的 native 自动路径沿用原来的右闭切点行为。

默认概率示例需要 `application_bad_prob`、`behavior_bad_prob` 两个 0–1 字段；评估输入有
`bad`、`later`、`cohort`，参考集还需有监督标签 `bad`：

```python
report = cross_scores(
    df, score_x="application_bad_prob", score_y="behavior_bad_prob",
    score_directions={"application_bad_prob": "higher_risk", "behavior_bad_prob": "higher_risk"},
    probability_scores=["application_bad_prob", "behavior_bad_prob"],
    targets=["bad", "later"], group_col="cohort",
    binning_reference=df_reference, binning_type="native", method="cart", binning_target="bad",
    n_bins={"application_bad_prob": 4, "behavior_bad_prob": 6}, min_bin_size=0.05,
    binner_params={"cart_params": {"max_depth": 3}, "merge_small_bins": True},
)
saved_bins = get_score_bin_definitions(report)
```

方法路径、参数透传与一次拟合由 `tests/test_score_cross_binning.py` 覆盖。普通 score 和混合方向
仍支持；风险方向及概率身份必须显式声明，不能由字段名推断。

正常箱 `b0...` 按原数值顺序定义为右闭区间 `(lower,upper]`，外端无界，覆盖 reference
范围外有限分数。`bins` 另存 risk_rank，展示与规则均低风险→高风险，高分低风险的轴
顺序反转，但 bin_id 和切点不反转。`bins.display_label` 是稳定的 X1/Y1 等展示编号，与
真实 bin_id、risk_rank 和端点同表保存；空组不删箱、不重新编号。旧快照按已保存 risk_rank
确定同样的标签。无界端点用 null + unbounded 标记，不写 JSON Infinity。
null/NaN 为 missing；非有限/无法数值化分为 invalid。概率域外值进入 invalid；普通分无
[0,1] 限制。`special_values={ID: [有限值]}` 单独成箱，显式特殊值优先于概率域检查。
显式 `missing_values` 优先于非法值判断，因此声明的 `inf`/`-inf` 进入 missing；未声明的
无穷值仍进入 invalid。提取定义及 `.marsreport` 加载后复用会恢复合法的受限浮点标签，
重新传入语义相同的缺失配置可继续复用，真正冲突的配置报错；不会修改调用方的定义。
`describe`、AI 上下文及文件清单仍使用标准 JSON 的 `$mars` 浮点标签。

| 表 | 粒度/含义 |
| --- | --- |
| cells | target/group/period/x_bin/y_bin；包含空正常与特殊格子 |
| row_summary | X 边际，也是 row_bad_rate 基准 |
| column_summary | Y 边际 |
| overall | 同标签、同样本范围总体 |
| bins | axis/bin_id；区间、开闭、方向、风险顺序、拟合来源 |

边际和总体从原始整数计数相加后重算，绝不平均格子坏率/Lift/区间。
sample_share 包含全部特殊箱；折叠特殊箱不会重新归一化正常格占比。
行/列边际也包括另一轴的隐藏特殊箱，页面基线直接读边际表。
observed_coverage=有表现人数/全部人数；bad_rate=坏人数/有表现人数。
Δ 始终为格子坏率减同 X 行坏率，内部小数差只在显示时乘 100 成 pp。
Lift 的基准是同 target/group/period 的真实 overall，零或未观测基准返回 null；
有效零分子且基准正数时为 0.00×。`lift_status` 与 `overall_status` 可独立查询，Lift 表示
相对整体风险倍数，不是模型提升幅度。min_observed 只标注 low_sample，
不删除数值。空格子为 empty，非空无表现为 unobserved，有表现无坏样本是有效 0，
无 target 为 not_requested（风险与标签计数空值），全好/全坏标签均支持。

`weights_col` 沿用有限非负权重，零权重允许；bad_rate 全程为 bad_weight_sum/
observed_weight_sum，格子/边际/总体/规则均一致。仍保留真实整数人数，区间明确命名
unweighted_ci_lower/upper，为默认 95% Wilson（`confidence_level` 可配置）；加权页面明确写
“未加权 Wilson”，加权区间标记 unsupported，不使用权重
总和冒充二项样本量。`amount_col` 复用既有金额 helper：非负金额贡献 tot/good/bad_amt，
null/NaN/负值不贡献金额；无限或非数值金额报错，口径写入参数。金额坏率分母为
good_amt+bad_amt，独立于权重风险口径。

## 保存后的规则回放

```python
from mars.reporting import load_report
from mars.analysis import evaluate_score_policy, write_score_cross_html

report.save("score-cross.marsreport")
restored = load_report("score-cross.marsreport")
baseline = {"type": "x_only", "x_max_risk_rank": 3, "missing_score": "reject"}
candidate = {"type": "and", "x_max_risk_rank": 3, "y_max_risk_rank": 3}
policy = evaluate_score_policy(restored, candidate, baseline=baseline)
policy.get_table("summary")
policy.get_table("regions")
policy.get_table("changes")
policy.save("score-policy.marsreport")
write_score_cross_html(restored, "cross.html", policy_reports=[policy])
restored.write_excel("cross.xlsx")
```

支持 x_only/y_only/and/or；门槛是整数 risk_rank（0 整轴拒绝，最大值为实际段数）。
缺失/非法轴默认拒绝，x_only 不依赖 Y，y_only 不依赖 X。OR 允许任一依赖轴通过；每格
计数一次。`accepted_special_bins={"x": ["missing"], "y": ["s0"]}` 可显式接受特殊箱。

staircase 示例：

```python
candidate = {"type": "staircase", "steps": {
    "b4": {"action": "accept", "y_max_risk_rank": 4},
    "b3": {"action": "accept", "y_max_risk_rank": 2},
    "b2": {"action": "reject"},
}}
policy = evaluate_score_policy(restored, candidate, baseline=baseline)
```

未列出的 X 段确定拒绝。特殊 X 段需先显式列入 accepted_special_bins，再配置该段 step。
这是 conditioned/staircase 规则，不能描述成全局独立 Y 门槛。箱内连续新阈值报错，
粗交叉不能插值或按比例估计精确结果。规则随报告保存固定原区间与父 report_id。
回放不修改父报告、不自动写文件、不选择最优策略或部署。

### 正常分箱表达式与离线规则

```python
rule = {"type": "expression", "expression": "X <= X2 AND (Y <= Y3 OR Y = Y5)"}
policy = evaluate_score_policy(restored, rule)
policy.get_table("summary", filters={"rule": "candidate", "retained": True})
policy.get_table("cell_decisions")
policy.save("expression-policy.marsreport")
```

X/Y、AND/OR 和同轴标签大小写不敏感。支持 `< <= = == != >= >`；AND 优先于 OR，
括号覆盖优先级。标签右侧允许整数简写；`X <= 2` 等于 `X <= X2`，表示固定风险顺序的
前两行，不能解释成原始概率阈值 2。跨轴标签、非整数、越界、未知词元或源码均报中文错误。
上限为 240 字符、96 词元、12 层括号，严格解析成限定 AST 后显式解释，不执行用户代码。
正常箱表达式排除任一轴的特殊箱，OR 重叠只计一次；底层旧 policy 的显式特殊箱策略仍保留。
非单调命中集合是分箱诊断集合，不自动映射为低风险通过门槛或业务审批。

表达式、解析 AST、限制、固定定义及父 report_id 随派生报告保存，机器可查询同一 summary，
包含人数、表现覆盖、bad_rate、真实总体 Lift 和状态。页面结果也只加总已有格子证据，使用
同一个 overall 分母；命中空箱与无命中显示 empty，未观测、零权重分母及有效零分别保留语义。

离线页面只提供两个镜像阶梯复制示例；5×5 时分别为 8 格，实际 NxM 会生成合法缩减表达式
及准确格数。复制只写剪贴板，不填入或应用。新输入无效时上一条已应用规则保留并明示；
目标、组、真实周期切换保留表达式和当前格子，恢复正常视图只清规则边框与汇总。

summary 保存候选/基准的 retained/rejected 人数、占比、表现覆盖、风险、好坏数量及其
原样本好坏分母比例；regions 是基准通过×候选通过四区域（包括空区域），changes 明示
实际留存人数/覆盖/风险差。AND/OR 另有 axis_regions，区分 X通过/Y拒绝的精筛候选区
与 X拒绝/Y通过的扩量候选区。不能只比较更严格规则的风险而隐藏覆盖差，也不会声称
粗分段达到完全等覆盖。基准未给定时为 all_samples，包含特殊箱。

默认业务名称为历史“样本留存率”，不是审批通过率。调用者应提供样本限制：例如只有
存量通过客户标签，不能外推拒绝客户或未来完整申请客群。Wilson 描述单格不确定性，
没有完成多重检验，也不证明辅助分具有稳定提升。

## 人与外部 Agent 共用证据

`describe()` 记录表角色：pairs 有两个逐行端点；cross 的统计表声明报告级 score_x/
score_y scope，无意义的单特征行不复制。标签、group、period 作为独立维度。中文名重复
不会改变原始 ID 关联键。get_feature/search_features 能找到两端或两模型分。
原生筛选、裁列、排序和分页后才转换展示数据。`to_ai_context(queries=..., max_chars=...)`
输出完整 JSON，省略完整记录并保留截断数/继续位置；不默认塞整矩阵或个体明细。
证据引用由 report_id、表名、可重放 query 及记录中的 feature pair/target/group/period/
bin_id 定位。未知业务字段保持 unknown。

Notebook `show_score_matrix` 默认显示正常箱，标注特殊样本省略数，提供风险/row delta/
全体样本占比着色及重算边际；匹配多范围时选择首个并在 caption 说明。Excel 包含所有
公共表与元数据，比例保持小数。离线 HTML 自包含无 CDN，切换范围、指标、特殊箱、
色标。紧凑矩阵逐格显示人数、全 scope 占比、坏率及 Lift；键盘箭头或点击只选格。
侧栏显示真实区间、X 行基线、Δ、覆盖、Wilson 和证据；两图分别固定 X 沿 Y、固定 Y 沿 X，
采用真实行/列边际参考线，空/未观测断线，有效零保留在 0。
规则输入和结果区独立，内侧命中边框保留热力颜色和选中外框。显式预回放 policy 的原统计
仍在折叠区可查。页面没有月份数据时不会制造月份控件，也不声称已经验证跨月或 OOT 稳定性。
旧保存报告可重导出新页面，无需原始明细、浏览器状态或重新拟合；已有字段/格式保持可读。

从已有报告运行重导出与可选规则，而不生成教程合成数据：

```bash
python docs/snippets/correlation_and_score_cross.py --report path/to/existing.marsreport --output output/score-cross-review --rule "X <= X2 AND Y <= Y3"
```

输出自包含 HTML、既有 Excel、独立 policy 快照和分页证据 JSON；该路径由
`tests/test_score_cross_portable_ui.py` 验证。HTML 可直接离线打开，无 CDN/字体/遥测/服务器请求。

完整可运行示例：[correlation_and_score_cross.py](../snippets/correlation_and_score_cross.py)。
执行：`python docs/snippets/correlation_and_score_cross.py --output examples/output/score-reports`。
数据明确为合成，不代表真实模型表现；生成 HTML、Excel、JSON、相关/交叉/派生快照。
Notebook：[correlation_and_score_cross.ipynb](../demos/correlation_and_score_cross.ipynb)。

兼容性调整：基础 Linear 相关筛选不再因未安装可选 statsmodels 而触发关闭的 VIF 和
Logit 诊断导入失败。未安装时这些可选诊断表为空并记录状态，筛选结果不变；显式启用
VIF/stepwise 仍要求安装 ml extra。安装 statsmodels 的既有用户行为保持，公开 API 无需迁移。

实际测试、环境、分阶段耗时与内存、未验证范围见[验收与基准记录](../performance/correlation-score-cross.md)。
