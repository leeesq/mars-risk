# Monitoring

特征监控、模型监控和报警摘要。

阅读[监控用户指南](../user-guide/monitoring.md)获取有标签、无标签、未表现期和 benchmark 建箱的
完整示例。这里保留自动生成的精确签名。

!!! tip "未表现期"

    当前期 target 缺列或全空时，可保留 target 名称并传入带有效标签的 `benchmark_df`，以基准期
    完成监督建箱、以当前期输出无标签分布监控。显式 `binner` 的优先级最高。

::: mars.monitoring.MarsMonitor

::: mars.monitoring.MarsMonitoringReport

::: mars.monitoring.MarsMonitoringData

::: mars.monitoring.MarsMonitoringAlertConfig

::: mars.monitoring.MarsMonitoringAlerter

::: mars.monitoring.generate_monitoring_alert
