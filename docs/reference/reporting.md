# Reporting

报告对象、结构化结果容器和导出入口。

先读[报告导出与二次加工](../user-guide/reports-and-exports.md)选择 Excel、单文件 HTML 或图片资产
模式；各 report 可用字段见[Report 对象](report-objects.md)。

!!! note "大规模 HTML"

    `MarsBinningReport.write_html()` 默认每个 target 最多生成 500 张图。`auto` 在图表超过 50 张时
    使用同级资产目录与懒加载；需要严格单文件时传 `chart_embed_mode="inline"`。

::: mars.reporting.MarsProfileReport

::: mars.reporting.MarsBinningReport

::: mars.reporting.ProfileData
