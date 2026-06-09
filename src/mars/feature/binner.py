"""MARS 分箱器兼容导入层。

新代码建议从 ``mars.feature.base``、``mars.feature.native_binner``、
``mars.feature.optimal_binner`` 或 ``mars.feature.lite_opt_binner`` 导入具体实现。
"""

from mars.feature.base import MarsBinnerBase
from mars.feature.native_binner import MarsNativeBinner
from mars.feature.optimal_binner import MarsOptimalBinner

__all__ = ["MarsBinnerBase", "MarsNativeBinner", "MarsOptimalBinner"]
