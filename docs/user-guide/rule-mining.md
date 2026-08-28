---
description: 使用 mars.rule 生成、验证、筛选、部署和报告策略规则。
---

# 规则生成与部署

!!! warning "Experimental"

    `mars.rule` 首版随 MARS 0.0.28 发布。它不兼容旧 `deimos.*` 导入或旧 RuleSet JSON；生产流程
    应固定 `mars-risk==0.0.28` 并保存候选审计、解析后的 spec 和验证状态。

## 最小工作流

下面的示例只使用 seed DSL，便于明确展示训练筛选、验证筛选和部署输出。省略 `generators` 时，
MARS 会默认同时运行组合规则和浅层树。

```python
--8<-- "docs/snippets/rule_mining.py"
```

`MarsRuleMiningResult` 不保存原始 DataFrame。核心字段是 `status`、`rule_set`、
`candidate_table`、`evaluation`、`spec` 和 `metadata`。没有规则合法入选时，`status="no_rules"`，
同时返回空 RuleSet 和淘汰审计；数据、DSL 或计算错误不会伪装成空结果。

## DSL v2 与缺失语义

DSL 支持普通或双引号标识符、数字/字符串/布尔值、`< <= = == != >= >`、`IS NULL`、
`IS NOT NULL`、`IS MISSING`、`IS NOT MISSING`、`AND`、`OR`、`NOT` 和括号。字符串使用
单引号并以 SQL 双写方式转义：

```python
from mars.rule import MarsRule

rule = MarsRule('"customer type" = \'New\' AND income IS MISSING')
```

`IS NULL` 只匹配真正的 null；浮点列的 `IS MISSING` 同时匹配 null 和 NaN。普通比较不会命中
null/NaN。DSL 在执行前校验缺列、字面量类型、表达式长度、token、AST 节点数和嵌套深度。
函数、算术、`IN`、子查询和任意 SQL 都会被拒绝。规则先解析为 AST，再规范化、简化重复条件、
检测明显矛盾，最后生成确定性的 `mr_<sha256 前 20 位>` ID。`MarsRule` 不保存样本指标。

## 默认筛选与方向

高风险默认候选阈值为 Lift ≥ 1.2、事件数 ≥ 3、覆盖率 1%–50%；验证阈值为 Lift ≥ 2.0、
事件数 ≥ 3、覆盖率 1%–50%。提供时间或业务切片时至少 80% 切片通过。

```python
from mars.rule import MarsRuleMiningSpec

low_risk_spec = MarsRuleMiningSpec.low_risk(top_k=5)
```

低风险候选/验证 Lift 上限分别为 0.9/0.8，不要求最少事件数。两种方向默认 `top_k=10`、
`max_candidates=100_000`、IoU 0.3、随机种子 42 和 `ranked` 排序。需要逐轮挖掘时，显式设置
`selection_strategy="cascade"`；每轮都会在剩余训练样本重新生成候选，在剩余验证样本重新筛选，
选出一条后同步剔除命中样本，最多运行 `max_rounds=10`。候选审计通过 `generation_round` 和
`selection_round` 记录逐轮决策。

筛选只接受 `MarsRuleFilter` 与 `MarsRuleMetricCondition`，不接受 tuple、mapping 或指标 SQL：

```python
from mars.rule import MarsRuleFilter, MarsRuleMetricCondition

strict = MarsRuleFilter(
    conditions=(
        MarsRuleMetricCondition("lift", ">=", 3.0),
        MarsRuleMetricCondition("event_count", ">=", 10),
    ),
    targets="all",
)
```

## Explore 与 Production

`spec=None` 等价于 explore：允许没有独立验证集，并返回 `qualification="exploratory"`。
explore 适合 notebook 和候选诊断，但默认禁止生成部署 SQL。

```python
from mars.rule import MarsRuleMiningSpec, mine_rules

production_result = mine_rules(
    train_df,
    target="target",
    validation_df=validation_df,
    spec=MarsRuleMiningSpec.production(),
)
```

production 必须提供独立 `validation_df`。验证阶段使用单侧 95% Wilson Lift 保守界、单侧
超几何精确检验和 Benjamini-Hochberg `q <= 0.05` 硬门禁；自定义 filter 只能叠加，不能关闭
这些约束。提供至少 3 个有效时间切片后，至少 80% 切片通过的规则集升级为
`temporally_validated`，否则保持 `validated`；已有足够切片但稳定性失败的规则会被淘汰。

## 生成器与可选依赖

- 默认：`MarsCombinationRuleGenerator`、`MarsTreeRuleGenerator`。
- 显式启用：`MarsForestRuleGenerator`、`MarsGBDTRuleGenerator`、
  `MarsIsolationRuleGenerator`。
- LightGBM：安装 `mars-risk[ml]` 后设置 `backend="lightgbm"`。
- Optuna：安装 `mars-risk[tuning]` 后设置树生成器 `tuning_backend="optuna"`。

自动生成只使用主目标和数值特征；`aux_targets` 只参与评估与筛选。类别规则可作为 seed DSL。
数值特征达到 500 个时，组合生成器最多抽样 100,000 行，并直接复用 `MarsStatsSelector` 预筛到
最多 300 个特征。

树、森林、GBDT 和孤立森林不会再依赖不可部署的缺失 sentinel。训练矩阵为每个数值特征构造
稳定填充值和显式缺失指示器，并把模型分支还原为 DSL 的 `IS MISSING`、`IS NOT MISSING`、比较和受控
`OR` 条件。`MarsTreeRuleGenerator.n_jobs` 控制多棵浅层树的 Joblib 并行度。
Optuna 使用固定种子的分层交叉验证 ROC AUC，不再优化训练集准确率。

## 评估、部署和报告

评估固定输出 `dataset/rule_id/target/slice/group` 长表，包含样本数、事件数、覆盖率、事件率、
Lift、事件率/Lift 置信界、p/q 值，以及显式配置后的金额和客户指标。每个目标分别排除自身
null/NaN；零分母为 null，筛选时视为不通过。

```python
from mars.rule import MarsRuleReport

rule_set = production_result.rule_set
rule_set.save_json("rules.json")
restored = type(rule_set).load_json("rules.json")
sql_columns = restored.generate_sql(table_alias="applications")

analysis = result.analyze(
    validation_df,
    amount_col="loan_amount",
    customer_col="customer_id",
    max_pairs=5000,
    bootstrap_repeats=500,
)
report = result.to_report(analysis)
report.write_html("rule-report.html")
report.write_excel("rule-report.xlsx")

benchmark_report = MarsRuleReport.from_benchmark(
    [{"engine": "mars", "seconds": 12.1, "peak_mb": 2730.0}]
)
benchmark_html = benchmark_report.render_html()
```

`analysis.interaction_table` 和 `analysis.cumulative_table` 同时包含样本、金额和客户维度的组合、累计
及边际指标。`result.to_report()` 总是为最终规则生成 `rule_explanations` 结构化解释表；未显式传入
高级分析时仍省略 interactions/cumulative/bootstrap section。bootstrap 默认关闭，只对最终
top-k 规则执行，不参与候选筛选。

RuleSet JSON 固定 `artifact_type="mars_rule_set"`、`schema_version=1` 和
`expression_version=2`，并保存 qualification 和验证摘要。未知字段/版本、重复 ID、篡改 ID、
未知等级引用均 fail closed。`generate_sql()` 只承诺 ANSI SQL `CASE WHEN` 命中列、等级计数和
总命中数；默认要求至少 `validated`。含 `IS MISSING` 的规则只有在调用方声明
`missing_policy="normalized_to_null"` 时才允许导出，表示 SQL 上游已完成 NaN→NULL 规范化。
