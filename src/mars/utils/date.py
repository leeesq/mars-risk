# mars/utils/date.py

from typing import Union

import polars as pl


class MarsDate:
    """
    [MarsDate] 日期处理核心组件 (Pure Polars Edition).

    专为 Polars DataFrame 操作设计。
    所有方法均返回 ``pl.Expr`` 对象，可直接用于 Polars 的表达式系统中。

    Notes
    -----
    该类不直接处理数据，而是构建 Polars 表达式树。
    这意味着它的开销极低，且能完美融入 ``lazy()`` 执行计划中。
    """

    @staticmethod
    def _to_expr(col: Union[str, pl.Expr]) -> pl.Expr:
        """
        [Internal] 将输入归一化为 Polars 表达式。

        Parameters
        ----------
        col : Union[str, pl.Expr]
            如果是字符串，视为列名并转换为 ``pl.col(col)``。
            如果是表达式，原样返回。

        Returns
        -------
        pl.Expr
            Polars 表达式对象。
        """
        if isinstance(col, str):
            return pl.col(col)
        return col

    @staticmethod
    def smart_parse_expr(col: Union[str, pl.Expr]) -> pl.Expr:
        """
        [智能解析] 生成多路尝试的日期解析表达式。

        采用 "Coalesce" (多路合并) 策略，能够自动处理混合格式的脏数据。

        优化策略
        --------
        1. **类型优先保护**: 优先尝试直接 Cast。如果输入已经是 Date/Datetime，
           则跳过后续字符串解析，大幅提升处理规整数据时的性能。
        2. **强制转 String**: 对于无法直接 Cast 的类型，转换为 ``pl.Utf8`` 统一处理。
           这解决了整数日期 (如 20250101) 被误读为天数偏移的 bug。
        3. **多格式尝试**: 依次尝试解析常用的 ISO 格式、紧凑格式、斜杠和点号格式。

        Parameters
        ----------
        col : Union[str, pl.Expr]
            待解析的列名或表达式。支持 String, Int (如 20230101), Date, Datetime 类型。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Date`` 的表达式。无法解析的值将变为 Null。
        """
        expr = MarsDate._to_expr(col)

        # 预生成 String 表达式用于多格式解析尝试
        str_expr = expr.cast(pl.Utf8)

        # Coalesce: 从上到下尝试，返回第一个非 Null 的结果
        return pl.coalesce([
            # 1. 尝试直接 Cast
            # 如果是原生 Date/Datetime 或标准 "YYYY-MM-DD" 字符串，此步最高效
            expr.cast(pl.Date, strict=False),

            # 2. 标准 ISO 格式 (2025-01-01)
            # 强化匹配：部分特殊 Object 转 Str 后可能符合此格式
            str_expr.str.to_date("%Y-%m-%d", strict=False),

            # 3. 紧凑格式 (20250101)
            # 解决 Int 类型转为 Str 后的情况
            str_expr.str.to_date("%Y%m%d", strict=False),

            # 4. 斜杠格式 (2025/01/01)
            str_expr.str.to_date("%Y/%m/%d", strict=False),

            # 5. 点号格式 (2025.01.01)
            str_expr.str.to_date("%Y.%m.%d", strict=False),
        ])

    @staticmethod
    def dt2day(dt: Union[str, pl.Expr], interval: str = "1d") -> pl.Expr:
        """
        将日期转换为指定天数粒度 (如 '1d', '3d', '14d')。
        如果是多天 (>1d)，则以该列的最小日期 (min) 作为锚点计算区间，
        并返回类似周粒度的字符串区间表现形式 (如 '20260101-0103')。

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。
        interval : str, default "1d"
            时间间隔，支持 "day", "1d", "3d", "14d", "30d" 等格式。

        Returns
        -------
        pl.Expr
            当 interval 为 1d 时，返回 pl.Date 类型。
            当 interval > 1d 时，返回 pl.Utf8 (String) 区间格式。
        """
        parsed_dt = MarsDate.smart_parse_expr(dt)

        # 解析传入的 interval 参数
        interval = interval.lower().strip()
        if interval == "day":
            n_days = 1
        elif interval.endswith("d") and interval[:-1].isdigit():
            n_days = int(interval[:-1])
        else:
            raise ValueError(f"Invalid interval format '{interval}'. Expected 'day' or 'Nd' (e.g., '3d', '14d').")

        # 如果是 1 天，保持原样返回 pl.Date
        if n_days == 1:
            return parsed_dt

        # 多天逻辑 (>1d)
        # 获取该列的全局最小日期 (锚点)
        min_dt = parsed_dt.min()

        # 计算每一行日期与全局锚点相差的天数
        diff_days = (parsed_dt - min_dt).dt.total_days()

        # 计算该行所属区间的起始偏移天数
        # 数学逻辑：例如 n=3，相差 4 天 -> (4 // 3) * 3 = 3，即落在第 3 天开始的区间
        offset_days_expr = (diff_days // n_days) * n_days

        # 动态推算区间的起止日期
        start_of_period = min_dt + pl.duration(days=offset_days_expr)
        end_of_period = start_of_period + pl.duration(days=n_days - 1)

        # 拼接为 "YYYYMMDD-MMDD" 的字符串格式，保持与 week 一致的视觉体验
        return pl.concat_str([
            start_of_period.dt.strftime("%Y%m%d"),
            pl.lit("-"),
            end_of_period.dt.strftime("%m%d")
        ])

    @staticmethod
    def dt2week(dt: Union[str, pl.Expr]) -> pl.Expr:
        """
        将日期转换为 'Week' 粒度的字符串区间 (如 '20260126-0201').

        逻辑：向下取整到周一作为起点，加上 6 天作为周末终点，最后拼接字符串。

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Utf8`` (String) 的表达式。
        """
        # 解析并截断到周一 (起点)
        start_of_week = MarsDate.smart_parse_expr(dt).dt.truncate("1w")
        # 加上 6 天得到周日 (终点)
        end_of_week = start_of_week + pl.duration(days=6)

        # 拼接为 "YYYYMMDD-MMDD" 格式
        return pl.concat_str([
            start_of_week.dt.strftime("%Y%m%d"),
            pl.lit("-"),
            end_of_week.dt.strftime("%m%d")
        ])

    @staticmethod
    def dt2month(dt: Union[str, pl.Expr]) -> pl.Expr:
        """
        将日期转换为 'Month' 粒度的字符串 (如 '202601').

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Utf8`` (String) 的表达式。
        """
        # 直接使用 strftime 提取年月即可，无需 truncate
        return MarsDate.smart_parse_expr(dt).dt.strftime("%Y%m")

    @staticmethod
    def format_dt(dt: Union[str, pl.Expr], fmt: str = "%Y-%m-%d") -> pl.Expr:
        """
        将日期解析并格式化为指定字符串。

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。
        fmt : str, optional
            输出的格式化字符串，默认 "%Y-%m-%d"。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Utf8`` (String) 的表达式。
        """
        return MarsDate.smart_parse_expr(dt).dt.strftime(fmt)
