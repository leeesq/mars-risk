# 版本一旧报告 fixture

`legacy_e990813_no_target.marsreport` 由 `e990813fbc4148f138f7cd2025f47637d385c623`
的源码在独立 Python 3.12.13 进程生成，使用 Pandas 3.0.3、Polars 1.42.0。
文件保留该提交真实输出的 `not_computed/no_target` 统计及历史错误的 Score Cross 状态文案。
测试直接读取固定文件；CI 不依赖 Git 历史、不读取原始数据，也不重新拟合。

复现时将该提交的 `src` 放在独立目录，让 `PYTHONPATH` 指向该目录，然后执行：

```python
import polars as pl
from mars.analysis import profile_risk

report = profile_risk(
    pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}), features=["x"], n_bins=2,
).report
report.save("legacy_e990813_no_target.marsreport")
```

新版本只在已知报告类型与表的语义解析边界纠正文案；报告格式仍为 1，已有数据和状态编码不变。
