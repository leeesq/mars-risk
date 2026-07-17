# Scoring

评分卡构建、分数映射和部署导出。

`build_scorecard()` 消费已拟合的 binner 与模型系数，生成分数映射和部署 SQL。若还没有稳定分箱规则，
先完成[分箱与风险评估](../user-guide/binning-risk-evaluation.md)；若需要 LR/WOE 全流程，阅读
[Modeling / Pipeline](../user-guide/modeling-pipeline.md)。

::: mars.scoring.MarsScorecard

::: mars.scoring.build_scorecard
