---
description: deimos-rule e6714c5 到 mars.rule 的逐项功能与回归迁移矩阵。
---

# Deimos Rule 迁移矩阵

本页固定对照来源提交 `e6714c5e795054e44f0c58ad7097668b4117b4a2`。`Covered` 表示能力已由
Mars 新契约覆盖；`Replaced` 表示能力保留但调用方式重设计；`Removed by design` 表示发布计划明确
删除的旧兼容面，不应重新引入 `deimos.*`、旧 artifact 或双后端适配层。

| # | Deimos 回归 | Mars 落点 | 状态 |
|---:|---|---|---|
| 1 | `test_simplifier_contracts` | `test_dsl_normalization_is_deterministic_and_simplifies_duplicates`、`test_dsl_rejects_static_contradictions` | Covered |
| 2 | `test_translator_quotes_identifiers` | `test_ruleset_sql_matches_reference_database` | Covered |
| 3 | `test_invalid_rule_raises` | `test_dsl_rejects_non_v1_syntax` | Covered |
| 4 | `test_config_validation` | `test_filters_and_specs_reject_untyped_legacy_values` | Replaced |
| 5 | `test_evaluator_math_accuracy` | `test_evaluator_fixed_long_table_metrics_and_null_targets` | Covered |
| 6 | `test_evaluator_zero_denominator_metrics` | `test_evaluator_zero_denominators_are_null_and_fail_filter` | Covered |
| 7 | `test_evaluate_by_slice_all_group_metrics` | `test_evaluator_fixed_long_table_metrics_and_null_targets` | Covered |
| 8 | `test_evaluate_by_slice_dt_and_multi_target` | `test_evaluator_fixed_long_table_metrics_and_null_targets` | Covered |
| 9 | `test_deduplicator` | `test_mine_rules_uses_validation_and_returns_auditable_result` | Covered |
| 10 | `test_oot_stability` | `test_time_slice_pass_rate_rejects_unstable_rule` | Covered |
| 11 | `test_whitelist_pipeline_and_transform` | `test_low_risk_spec_selects_low_lift_rule`、`test_ruleset_transform_preserves_input_type_and_counts` | Replaced |
| 12 | `test_pipeline_maps_amount_and_customer_metrics` | `test_evaluator_fixed_long_table_metrics_and_null_targets` | Covered |
| 13 | `test_candidate_budget_is_enforced` | `test_combination_generator_is_reproducible_and_budgeted` | Covered |
| 14 | `test_feature_prefilter_skips_small_width` | `test_combination_generator_is_reproducible_and_budgeted` | Covered |
| 15 | `test_native_feature_prefilter_is_stable` | 500+ 特征统一复用 `MarsStatsSelector`，不保留 native 后端 | Removed by design |
| 16 | `test_mars_feature_selector_returns_top_k` | 组合生成器 `_prefilter_features` 与 100k×1000 性能门禁 | Replaced |
| 17 | `test_mars_feature_selector_missing_dependency` | Mars 内部直接依赖自身 selector，不再有可选 Mars adapter | Removed by design |
| 18 | `test_auto_prefilter_falls_back_to_native` | 不允许静默回退第二套实现 | Removed by design |
| 19 | `test_mars_backend_is_strict_when_missing` | Mars selector 是同 wheel 基础能力 | Removed by design |
| 20 | `test_evaluator_fills_only_referenced_rule_columns` | `MarsRuleEvaluator` 批量 AST mask 与缺列 fail closed 回归 | Covered |
| 21 | `test_tree_generator_is_reproducible` | `test_tree_generator_has_explicit_missing_paths_and_is_reproducible` | Covered |
| 22 | `test_rule_filter_sort_and_grade_helpers` | typed filter、稳定排序、RuleSet grade 回归 | Replaced |
| 23 | `test_rule_query_config_mapping_aliases` | tuple、mapping 和指标 SQL 不再接受 | Removed by design |
| 24 | `test_rule_set_transform_counts` | `test_ruleset_transform_preserves_input_type_and_counts` | Covered |
| 25 | `test_rule_mining_session_rule_view_mode` | `test_mine_rules_uses_validation_and_returns_auditable_result` 的 seed-only 模式 | Replaced |
| 26 | `test_rule_mining_session_combination_mode` | 默认 `mine_rules` 与组合生成器回归 | Replaced |
| 27 | `test_tree_generator_params_and_metadata` | 缺失路径、`n_jobs`、训练样本数与 feature importance 回归 | Covered |
| 28 | `test_rule_interactions_math_accuracy` | `test_on_demand_analysis_has_interaction_and_cumulative_tables` | Covered |
| 29 | `test_rule_interactions_limits_and_zero_denominators` | `max_pairs` 和 null 比率回归 | Covered |
| 30 | `test_cumulative_rules_or_and_marginal_metrics` | 样本、金额、客户累计与边际回归 | Covered |
| 31 | `test_build_rule_report_from_rule_set_and_html` | `test_report_omits_analysis_until_explicitly_supplied` | Replaced |
| 32 | `test_build_rule_report_from_session_result` | `MarsRuleMiningResult.to_report()` 回归 | Replaced |
| 33 | `test_empty_rule_report_html` | 空分析 section 省略和安全 HTML 导出回归 | Covered |
| 34 | `test_benchmark_results_to_html` | `test_benchmark_report_renders_without_writing` | Replaced |
| 35 | `test_forest_generator_is_reproducible_and_evaluable` | `test_ensemble_generators_are_reproducible` | Covered |
| 36 | `test_gbdt_generator_sklearn_is_reproducible_and_evaluable` | `test_ensemble_generators_are_reproducible` | Covered |
| 37 | `test_gbdt_lightgbm_missing_dependency` | `test_lightgbm_and_optuna_use_unified_optional_dependency_message` | Covered |
| 38 | `test_gbdt_lightgbm_optional_backend_if_installed` | `test_lightgbm_optional_backend_generates_rules_when_installed` | Covered |
| 39 | `test_isolation_forest_generator_is_reproducible_and_evaluable` | `test_ensemble_generators_are_reproducible` | Covered |
| 40 | `test_model_generators_validation_and_exports` | 五生成器 public API、参数校验和 root 非导出回归 | Covered |
| 41 | `test_rule_mining_session_model_modes` | 显式 `generators=` 组合与默认生成器契约 | Replaced |
| 42 | `test_rule_set_serialization_and_polars` | `test_ruleset_json_round_trip_and_strict_validation`、同类型 transform 回归 | Replaced |
| 43 | `test_mars_adapter_missing_dependency` | `integrations.mars` 不再存在 | Removed by design |
| 44 | `test_mars_adapter_builds_hit_features` | `MarsRuleSet.transform()` 后直接调用 Mars 分析 API | Replaced |

## 关键语义差异

- `cascade` 每轮在剩余训练样本重新生成候选，并在剩余验证样本重新筛选；审计记录
  `generation_round` 与 `selection_round`。
- 树、森林、GBDT 和孤立森林使用“稳定数值填充列 + 显式缺失指示列”训练，模型分支会精确还原为
  `IS NULL`、`IS NOT NULL` 及受控的 `OR` 条件，不保存不可部署的 sentinel。
- 高级分析继续覆盖样本口径，并恢复交互、累计和边际的金额与客户口径。
- 报告中的 `rule_explanations` 是结构化表；benchmark 通过
  `MarsRuleReport.from_benchmark(...).render_html()` 或 `write_html()` 导出。
