import logging
import sys

# 尝试导入 colorlog，如果没有安装则回退到标准 logging
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

# 定义单例 Logger 名称
LOGGER_NAME = "mars"

def get_mars_logger(level=logging.INFO):
    """
    获取 MARS 专属的全局 Logger。
    
    特性：
    1. 单例模式：防止 Jupyter 中重复打印日志。
    2. 彩色输出：Info/Warn/Error 颜色区分。
    3. 标准格式：[MARS] 时间 - 级别 - 消息
    """
    logger = logging.getLogger(LOGGER_NAME)
    
    # 防止重复添加 Handler (Jupyter Notebook 常见痛点)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)

    if HAS_COLORLOG:
        # 定义颜色方案 (火星风格：Info用绿色或白色，警告黄色，错误红色)
        log_colors = {
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
        
        # 格式：[颜色][MARS] 时间 - 级别 - 消息 [重置]
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s[MARS] %(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors=log_colors,
            reset=True,
            style='%'
        )
    else:
        # 回退格式 (无颜色)
        formatter = logging.Formatter(
            '[MARS] %(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 阻止日志向上传播到 root logger (防止被其他库的配置干扰)
    logger.propagate = False

    return logger

def set_log_level(level):
    """
    动态修改日志级别。
    
    Usage:
        from mars.utils.logger import set_log_level
        set_log_level('DEBUG')
    """
    logger = logging.getLogger(LOGGER_NAME)
    
    if isinstance(level, str):
        level = level.upper()
        # 映射字符串到 logging 常量
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        level = levels.get(level, logging.INFO)
        
    logger.setLevel(level)
    # 同时修改 handler 的级别，确保生效
    for handler in logger.handlers:
        handler.setLevel(level)

# 初始化一个默认 logger 实例，方便直接导入使用
logger = get_mars_logger()