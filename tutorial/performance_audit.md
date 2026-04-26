# Performance Audit Checklist

这份清单记录了当前版本已经完成的低风险性能整理，以及下一轮值得继续推进的热点。

## 已完成

- `MarsBinEvaluator` 改为惰性导入 `MarsPlotter`，避免把绘图依赖变成基础硬依赖。
- 控制台日志增加编码安全处理，避免 Windows GBK 终端因为 emoji 触发异常。
- `MarsStatsSelector.export_selector_report` 去掉一次重复的 `to_pandas()` 转换。
- 增加基于合成数据的 pytest 护栏，后续优化可以更放心地动核心路径。

## 下一轮优先看什么

- `feature/binner.py`
  关注循环内 `collect()`、`map_elements()` 和可批量表达式化的片段。
- `analysis/evaluator.py`
  关注多目标模式下的 DataFrame 拼接和 Pandas/Polars 回退成本。
- `analysis/report.py`
  关注仅为展示用途触发的 `to_pandas()`，避免影响主计算路径。
- `feature/selector.py`
  关注粗筛和精筛阶段之间的数据复用，减少重复评估。

## 审计原则

1. 先保行为一致，再动性能。
2. 优先删掉不必要转换，再考虑算法重写。
3. 优先让 Polars 处理整列，再减少 Python 循环。
4. 每次优化前后都跑 benchmark 和 pytest。

## 推荐流程

```bash
python tutorial/benchmark_synthetic.py
python -m pytest -q
```

## 观察指标

- `MarsDataProfiler.generate_profile` 总耗时
- `MarsNativeBinner.fit` / `transform` 总耗时
- `profile_risk` 端到端耗时
- 峰值内存占用
- Pandas 输入与 Polars 输入的差异
