"""独立展现层命名空间。

按需从 ``mars.reporting.html_assets``、``mars.reporting.plotter`` 等子模块
导入具体能力，避免形成跨层提前加载依赖。
"""

__all__: list[str] = []
