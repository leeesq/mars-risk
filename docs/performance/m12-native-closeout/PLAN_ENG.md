# M12 Native Wide-Table Repair and Capacity Closeout Plan

[中文](PLAN_CN.md) · [Progress](PROGRESS.md) · [First Codex task](CODEX_START.md) · [Existing upstream acceptance](../m12-upstream.md)

## 1. Goal, scope, and current status

The target repository is **leeesq/mars-risk**. This plan is based on the inspected `main` commit `05b212e9b720e9ec6e9af0bec050e2be2685dc36`. The planning commit adds documentation only. It implements no new fix and runs neither real data nor tests.

The goal is not to reimplement merged fixes. It is to answer: **under the fixed Python 3.8 environment, can current MARS complete native profiling and statistical selection at the original M12 scale on the local machine, without caller-side feature splitting, while preserving numerical, population, and report contracts?** Reproduce any remaining gap before making the smallest appropriate upstream repair.

Implementation present, targeted regression passed, a specific native workload completed, report cold restoration passed, and downstream integration completed are separate conclusions. MARS acceptance does not establish Agent efficiency or improved model KS.

Do not modify the modeling Agent's Runtime, Registry, Snapshot, approval, budgets, freeze, or recovery. Do not train XGBoost, connect an LLM, extend `mars.agent`, or introduce another tuner or selection engine. The iteration pause for Monitoring, Modeling/Pipeline, and Scoring remains. Leave the root README, package version, dependency constraints, and release process unchanged by default.

## 2. Implemented fixes versus unverified claims

“Implemented” below means corresponding code and repository regressions exist at the inspected revision, not that this planning task reran them. Sources are [existing upstream acceptance](../m12-upstream.md) and the listed modules. Preserve historical acceptance records.

| ID | Current evidence | Action in this round |
| --- | --- | --- |
| U01 | Profiler null/constants use scalar aggregation; PSI is separate from generic placeholder pivots | Rerun old-Polars, multi-chunk, and empty-population boundaries; do not redesign PSI |
| U02 | Binner selects lazy execution before sample expressions/category joins; fit/benchmark schema checks no longer require full-width eager transforms | Verify execution boundaries and row order, not only return types |
| U03 | Selector batch_size reaches Quality, Rough Performance, stability, and WOE conversion | Observe actual batch widths and object lifetimes; this is not a total-memory limit |
| U04 | Full WOE Pearson uses Float64 row-block central moments and retains the global P² matrix, complete relationships, and original decisions | Check numerics, threshold refinement, cross-batch relationships, and actual capacity |
| U05 | Streaming categorical joins preserve order; performance labels travel with rows before transformation; empty-PSI statuses are retained | Regress actual streaming, multiple threads, name collisions, and save/load |
| U06 | The benchmark already supports frozen_profile/risk/selection, semantic sidecars, comparability gates, and thread-pool enumeration-order normalization | Reuse the existing harness, not a new capacity framework |

Relevant code: `src/mars/analysis/_profiling/`, `analysis/evaluator.py`, `analysis/_evaluation/aggregation.py`, `feature/binning/base.py`, `feature/selection/stats.py`, `compute/_correlation.py`, `benchmarks/benchmark_core_capacity.py`, `benchmark_core_workloads.py`, and `_capacity_evidence.py`.

Existing documentation records one successful Windows 200,000×1,000 high-survival synthetic selector workload. It does not validate the frozen M12 inputs on the user's Mac/16 GiB machine. The public upstream record explicitly says its real 257,997-row risk analysis and 297,440-row selector were not executed. That is the principal evidence gap, not proof that the same old source defects remain.

## 3. Invariants

- Keep original files read-only. Bind input values, feature order, dtypes, null/sentinel representation, supervised population, Train benchmark, missing/special rules, and effective business parameters. Do not reduce rows, remove sources, disable PSI/correlation, change thresholds, or substitute raw correlation for WOE to make a run pass.
- Access only the user's already provided and authorized M12 workspace. Follow explicit manifest references; do not scan the home directory or fetch database data. Real inputs, complete business reports, original feature names, actual paths, manifest contents, and data fingerprints stay in local ignored storage, never this public repository or CI.
- Selector inputs physically null non-Train targets. Unsupervised profiling receives no target. Development risk includes only Train/Val/Test, with fitting and benchmark semantics copied from frozen configuration. Do not accidentally null valid Val/Test development-risk labels through supervision_group.
- Library-internal column/row batching is permitted; acceptance concerns one complete public call. Caller-side three-way selection followed by a union of correlations, or merged outer risk-profile batches, cannot establish native capability.
- Preserve failed/cancelled attempts. Do not resume old frozen Runs under new dependencies or alter their identities, Receipts, or models. Use separate local directories for upstream validation.
- Add no available-only hard gate. Low available memory, compression, or swap alone is not a failure. Execute real tasks serially with observation-only resource telemetry. Record existing timeout/exit protection explicitly, distinguish cancellation, exceptions, and kernel termination, avoid unbounded retries, and never terminate unrelated user processes.
- Discrete business outcomes must match exactly. Check floating differences under the existing semantic comparison contract; do not enlarge tolerances, change expected feature lists, or discard ordering to hide a difference.

## 4. Work packages

### N0 — Pin the baseline and reproducible entry point

Read applicable `AGENTS.md`, the [engineering skill](../../../.codex/skills/mars-risk-engineering/SKILL.md), performance/quality references, pyproject, current CI, and this directory. Continue on `fix/m12-native-capacity-closeout`; do not stash, reset, or overwrite user changes.

Use the existing local `modeling_xgb082` environment for real execution: keep Python 3.8.20 and XGBoost 0.82; record actual Polars, NumPy, Pandas, and PyArrow versions without upgrading. Validate new code in child processes using explicit `PYTHONPATH=<current-MARS-repository>/src`, record `mars.__file__`, and do not repoint editable dependencies used by other frozen Runs. Modern-version coverage uses already available isolated environments or existing CI.

Reuse the benchmark and inspect its current `--help`. Record source commit/content identity, exact commands, thread environment, machine/OS, measurement method, and exits. Begin with one small fresh-worker smoke covering JSON, sidecars, saving, and consumption by another process, not the complete suite or a heavy matrix.

**N0 acceptance:** the current source is actually imported; workspace/input protection is established; the small public call and evidence consumer succeed. Historical counts are not current test results.

### N1 — Freeze real analysis inputs without rebuilding models

Locate frozen normalized analysis inputs and their manifests inside the authorized workspace. Do not substitute the final small-feature delivery dataset for the original 1,111 candidates or infer candidate order from incidental table order.

Reuse correct frozen Parquet/IPC when available. With only raw Snapshots, permit a thin, explicit one-time analysis export based on the historical missing/label/split contract: preserve dtypes/order, record transformations and null counts, and prove raw files unchanged. This is not another selection stage or data acquisition. Do not hide preparation inside the benchmark reader. When equivalence to the historical analysis population cannot be established, mark input_contract_blocked and continue independent synthetic checks.

| Case | Current population | Benchmark | Target visibility | Features |
| --- | ---: | ---: | --- | ---: |
| Partition profile_stats | 297,440 | 180,893 Train | No target in input | Original 1,111 |
| Daily profile_stats | 297,440 | 180,893 Train | No target in input | Same 1,111 |
| Development profile_risk / evaluator | 257,997 | 180,893 Train | Valid development labels only; fitting follows frozen Train configuration | Same 1,111 |
| MarsStatsSelector | 297,440 | 180,893 Train | Non-Train targets physically null; supervision is Train-only | Same 1,111 |

These dimensions come from the existing public upstream record. Measure actual counts; do not delete records to match them. Give each case its own configuration and copy its complete historical binning, grouping, dates, PSI, RC, lists, and missing/special parameters. Frozen cases must not inherit synthetic-fixture defaults. Persist merged effective parameters locally before computation.

**N1 acceptance:** four local configurations and full ordered candidate sets are verifiable; values, schema, population, supervision, and benchmark match; no real data enters Git. Unavailable data is pending_local_input, not a library failure or success.

### N2 — Native continuous computation and report acceptance

Run serially: partition profile, daily profile, development risk, and complete selector. First perform one actual single-worker attempt per case. Harness warmups and repeats are additional calls and must be counted; repeats=1 does not necessarily mean one total invocation. Do not rerun a known dangerous old real full-width path merely to produce an A/B number.

Each worker continuously covers reading, one public computation, required report construction, and saving. Do not restart between Quality/Rough/Stability/Corr to conceal coexistence peaks. Isolated stage analysis may diagnose failures but cannot establish complete-call capacity.

Reuse frozen_profile, frozen_risk, and frozen_selection. frozen_risk currently calls MarsBinEvaluator.evaluate(); add a small profile_risk wrapper parity check to cover downstream parameter forwarding. Observational timing wrappers may remain; do not replace production algorithms or results. Do not invent unsupported APIs.

Record input/benchmark rows, valid supervised rows, dtypes/chunks, feature order, actual stage candidate movement, real column/row batches, stage wall times, public_compute time, entire-worker time, sampled main/tree RSS, lifetime peak, obtainable Darwin footprint/compression/swap, and exit status. Disclose missing measurements. Tree RSS can double-count shared pages; lifetime high-water is not a stage increment; footprint is not physical RSS.

Persist a verified report and completion record after each successful case. When a later case fails, repair and rerun that case; reuse earlier artifacts only when source/parameter/input identities remain applicable. Revalidate earlier cases affected by a source change, rather than blindly reusing cross-version assets.

Load reports in a new process and check describe/get_table/query_page/to_ai_context plus relevant get_feature queries. Do not recompute or load original wide inputs. Catalogs, schemas, nulls, order, calculation states, and metadata must match the original report.

**N2 acceptance:** all four native calls actually succeed; complete selector is not a merged caller-side workflow; every report is independently consumable. Report partial success per case rather than hiding omissions behind one Passed label.

### N3 — Repair only reproduced remaining failures

Do not change already passing N2 paths by default. Classify failures as input contracts, environment/imports, measurement tooling, storage, numerics, or actual capacity. Preserve PID, raw output, exit code, obtainable kernel evidence, and the deepest stage. Without allocation evidence, do not call one unpivot statement the unique root cause.

Inspect measured hotspots first: current/benchmark batch projections, lazy/schema boundaries, long-table lifetimes, projection before supervised filtering, current WOE block and NumPy temporaries, and complete relationship report construction/saving. Global O(P²) output and O(NP²) computation have not disappeared.

Create a small reproduction before the smallest source repair, retaining public APIs. Reuse compute, Binner, and reporting. Do not introduce a generic resource scheduler, a second low-memory selector, unbounded permanent caches, GC loops, or business-specific branches. Do not discard training state required by public update_bins/first WOE to manufacture a lower peak.

For each fix, record source/parameter changes, run focused regressions, and rerun the original failed real case separately. Never edit source while continuing the same process. Retry only after a material change; repeatedly executing an identical failure until one pass is not evidence of repair.

**N3 acceptance:** each repaired defect has before/after reproduction and success for the same real case, or a clearly retained limitation. If no new defect is found, use not_required rather than refactoring to manufacture a code commit.

### N4 — Semantic parity and downstream handoff

Compare both sides with the same corrected harness. Only emit performance ratios after full input content/order, roles, effective parameters, population, actual candidate path, resources, and report evidence meet the comparison contract. Classify intentional environment/thread differences as existing resource_strategy_comparison; missing sidecars or inadequate historical evidence means incomparable.

Use equivalent synthetic dense oracles for matrices, edges, and decisions. Previously saved M12 batched reports may serve as a **strategy reference**. Old caller-side batches and new native calls are not automatically a normal implementation A/B; do not require a dangerous old-path rerun for a fair ratio. Unexplained selection changes are failures, not dismissible floating noise.

Record compatibility boundaries: supported historical public report loading, new report roundtrips, Binner/WOE rules, and computation-free queries. If a core API or report change is truly necessary, document migration and test consumers separately; changes are not the default.

Produce a downstream handoff table for whole-frame rechunk, outer profile_risk batches, and three-part selector adapters: “removable after independent downstream comparison” or “still required.” Do not remove another repository's adapters or change an old Run's MARS pin. A new downstream acceptance task decides adoption after native MARS succeeds. Input export and benchmark execution are not Agent end-to-end acceptance.

**N4 acceptance:** numerical/report comparisons, explicit comparability classification, and adapter-specific migration guidance exist. Do not claim to have repaired Agent View/hash/Registry/stop-policy issues.

### N5 — Capacity boundaries, documentation, and delivery

The primary acceptance target is the native 1,111-feature M12 workload above. Add a staged high-survival 2,000-feature synthetic observation, starting with 50,000 rows. Larger row counts depend on recorded resources and the task scope; unexecuted cases remain Deferred. Do not call a workload full-width correlation capacity after early stages discard most candidates. Record actual correlation width and retain the existing at-least-80% scale guard.

A successful 1,111-feature case or synthetic 2,000-feature case does not guarantee a full 5,887-feature input. The capability table must state scenario, input/feature/benchmark size, threads, machine, invocation path, and limitations; do not publish a universal memory ceiling or fixed speedup percentage.

Use impact-based quality checks. Run targeted tests during development; after core changes stabilize, run the applicable Python 3.8 base suite, with modern coverage from existing CI. Do not rerun every heavy capacity test after each edit, weaken assertions, or silence warnings. Missing static tools do not justify upgrading the modeling environment; report not_run or use an existing isolated quality environment.

Update progress and a separate sanitized acceptance page. Append a navigation pointer to the original m12-upstream.md without rewriting history. Leave the root README unchanged by default. Add MkDocs navigation only if needed and check links/strict builds with an available docs environment. Do not release a package or deploy. Separate implementation/tests from acceptance documentation in commits; normal-push this branch without automatic merge.

**N5 acceptance:** limitations, passes/failures/deferrals, commands, and evidence mappings are complete; public files pass privacy review; CN/ENG retain matching tasks, parameters, and acceptance IDs.

## 5. Minimum acceptance matrix

| ID | Required evidence | Main level |
| --- | --- | --- |
| T01 | Old Polars, multi-chunk, PSI-only/mixed, non-applicable constants and empty populations retain shapes, dtypes, and statuses without caller rechunk | Targeted regression |
| T02 | Lazy planning and collect_schema avoid eager sample transforms; actual collect produces correct values/mappings | Execution-path regression |
| T03 | Streaming joins align rows, labels, and cross-batch WOE under threads 1/4, duplicate rows, unknown categories, and temporary-name collisions | Fresh-process regression |
| T04 | Real Quality/Rough/Stability/WOE batch widths and supervised populations match configuration; benchmark avoids premature full-width materialization | Path regression plus real observation |
| T05 | Global signed WOE Pearson, degenerate columns, strict threshold refinement, IV/ID priority, cross-batch edges, and final order match | Numerical oracle |
| T06 | Partition/daily profile, development risk, and selector each succeed as one public call on complete original populations/candidates | Real scale |
| T07 | Full candidates, table schemas, null/non-floating values, numerical values, and decisions match; unknown differences are rejected | Semantic comparison |
| T08 | New-process consumption needs neither raw data nor recomputation and preserves statuses/metadata | Cold restoration |
| T09 | Input/order, workload, resource, and sidecar changes are correctly classified; thread-pool reordering alone does not cause false rejection | Harness regression |
| T10 | Inputs unchanged, business data not uploaded, current/historical evidence separated, downstream boundaries explicit | Delivery inspection |

Prefer existing test_analysis_batching.py, test_binner_execution.py, test_binner_streaming.py, test_stats_blocked_correlation.py, test_profile_empty_psi.py, and test_core_capacity_benchmark.py. Do not create another near-duplicate testing framework.

## 6. Artifacts and verdicts

Public artifacts: plans, progress, generic repairs, synthetic tests, command templates, and sanitized conclusions. Private artifacts: complete input/config fingerprints, original features/reports, raw logs, actual source/manifest locations, measurement JSON, and sidecars, stored in user-approved ignored directories with unique names and no overwriting.

Record at least case/status/source/environment/input_contract/effective_parameters/actual_workload/measurements/output_contract/command/exit/failure_or_limit. This is an evidence field checklist, not a request for another Pydantic schema or Registry. Reuse harness fields and add only missing evidence.

Use not_started / in_progress / passed / failed / pending_local_input / blocked / deferred / not_required. Missing environment coverage is not passed. Final conclusions separately state:

1. Implemented fixes and targeted regressions.
2. Four native real-scale M12 cases and cold report consumption.
3. Measured limits for wider candidate pools.
4. Downstream Agent integration: not certified by this repository.
5. Model performance and research quality: not evaluated here.

## 7. First execution order

Proceed N0 → N1 → N2. Enter N3 for a reproduced upstream defect, then finish the applicable N4/N5 work. Without real inputs, finish synthetic checks and explicit local configuration guidance without fabricating real capacity acceptance. With inputs and no blocker, do not ask for repeated approval at every stage.

The first priority is **a trustworthy outcome for the native real public call**, not more modules, more test counts, or another model-training run. The complete Codex entry is [CODEX_START.md](CODEX_START.md).
