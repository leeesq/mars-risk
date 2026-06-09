"""MARS 日期解析与粒度转换工具。"""

import re
from typing import Union

import polars as pl


class MarsDate:
    """
    [MarsDate] 日期处理核心组件 (Pure Polars Edition).

    专为 Polars DataFrame 操作设计。
    所有方法均返回 ``pl.Expr`` 对象，可直接用于 Polars 的表达式系统中。

    Attributes
    ----------
    None
        该工具类不维护实例状态。

    Notes
    -----
    该类不直接处理数据，而是构建 Polars 表达式树。
    这意味着它的开销极低，且能完美融入 ``lazy()`` 执行计划中。

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"dt": ["2026-01-02"]})
    >>> df.select(MarsDate.dt2month("dt").alias("month")).item()
    '202601'
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

        Notes
        -----
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

        Examples
        --------
        >>> df = pl.DataFrame({"dt": ["2026-01-02", "20260103"]})
        >>> df.select(MarsDate.smart_parse_expr("dt").dt.strftime("%Y%m%d")).to_series().to_list()
        ['20260102', '20260103']
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
        interval : str
            时间间隔，支持 "day", "1d", "3d", "14d", "30d" 等格式。

        Returns
        -------
        pl.Expr
            当 interval 为 1d 时，返回 pl.Date 类型。
            当 interval > 1d 时，返回 pl.Utf8 (String) 区间格式。

        Raises
        ------
        ValueError
            当输入参数、列配置或数据状态不满足当前方法要求时抛出。

        Examples
        --------
        >>> df = pl.DataFrame({"dt": ["2026-01-01", "2026-01-03"]})
        >>> df.select(MarsDate.dt2day("dt", "3d").alias("bucket")).to_series().to_list()
        ['20260101-0103', '20260101-0103']
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
    def dt2week(dt: Union[str, pl.Expr], interval: str = "1w") -> pl.Expr:
        """
        将日期转换为指定周数粒度的字符串区间 (如 '20260126-0201').

        逻辑：向下取整到周一作为起点，加上 6 天作为周末终点，最后拼接字符串。

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。
        interval : str
            周聚合间隔，支持 ``"week"``、``"1w"``、``"2w"`` 等格式。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Utf8`` (String) 的表达式。

        Examples
        --------
        >>> df = pl.DataFrame({"dt": ["2026-01-28"]})
        >>> df.select(MarsDate.dt2week("dt").alias("week")).item()
        '20260126-0201'
        """
        unit, n_units = MarsDate._parse_time_grain(interval)
        if unit != "w":
            raise ValueError(f"Invalid week interval '{interval}'. Expected 'week' or 'Nw'.")

        # 先截断到周一，再按全局最早周一做多周区间锚点。
        start_of_week = MarsDate.smart_parse_expr(dt).dt.truncate("1w")
        min_week = start_of_week.min()
        diff_days = (start_of_week - min_week).dt.total_days()
        bucket_days = (diff_days // (n_units * 7)) * (n_units * 7)
        start_of_period = min_week + pl.duration(days=bucket_days)
        end_of_period = start_of_period + pl.duration(days=n_units * 7 - 1)

        return pl.concat_str([
            start_of_period.dt.strftime("%Y%m%d"),
            pl.lit("-"),
            end_of_period.dt.strftime("%m%d")
        ])

    @staticmethod
    def dt2month(dt: Union[str, pl.Expr], interval: str = "1m") -> pl.Expr:
        """
        将日期转换为指定月数粒度的字符串。

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。
        interval : str
            月聚合间隔，支持 ``"month"``、``"1m"``、``"2m"`` 等格式。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Utf8`` (String) 的表达式。

        Examples
        --------
        >>> df = pl.DataFrame({"dt": ["2026-01-28"]})
        >>> df.select(MarsDate.dt2month("dt").alias("month")).item()
        '202601'
        """
        unit, n_units = MarsDate._parse_time_grain(interval)
        if unit != "m":
            raise ValueError(f"Invalid month interval '{interval}'. Expected 'month' or 'Nm'.")

        parsed_dt = MarsDate.smart_parse_expr(dt)
        if n_units == 1:
            return parsed_dt.dt.strftime("%Y%m")

        # MARS 语义中 m 明确表示自然月，避免把 1m 交给 Polars 解释成分钟。
        month_index = parsed_dt.dt.year() * 12 + parsed_dt.dt.month() - 1
        start_month_index = (month_index // n_units) * n_units
        end_month_index = start_month_index + n_units - 1

        start_year = (start_month_index // 12).cast(pl.Int32)
        start_month = (start_month_index % 12 + 1).cast(pl.Int32)
        end_year = (end_month_index // 12).cast(pl.Int32)
        end_month = (end_month_index % 12 + 1).cast(pl.Int32)

        start_month_date = pl.date(start_year, start_month, 1)
        end_month_date = pl.date(end_year, end_month, 1)
        return pl.concat_str([
            start_month_date.dt.strftime("%Y%m"),
            pl.lit("-"),
            end_month_date.dt.strftime("%Y%m"),
        ])

    @staticmethod
    def from_grain(dt: Union[str, pl.Expr], grain: str) -> pl.Expr:
        """
        根据统一时间粒度生成日期分组表达式。

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。
        grain : str
            时间粒度，支持 ``"day"``、``"week"``、``"month"``、
            ``"1d"``、``"1w"``、``"2w"``、``"1m"``、``"2m"`` 等格式。

        Returns
        -------
        pl.Expr
            可直接用于 ``with_columns`` 的 Polars 表达式。

        Raises
        ------
        ValueError
            当粒度格式不是 MARS 支持的日期粒度时抛出。
        """
        unit, n_units = MarsDate._parse_time_grain(grain)
        if unit == "d":
            return MarsDate.dt2day(dt, interval=f"{n_units}d")
        if unit == "w":
            return MarsDate.dt2week(dt, interval=f"{n_units}w")
        return MarsDate.dt2month(dt, interval=f"{n_units}m")

    @staticmethod
    def is_time_grain(grain: str | None) -> bool:
        """判断字符串是否为 MARS 支持的日期聚合粒度。"""
        if not isinstance(grain, str):
            return False
        try:
            MarsDate._parse_time_grain(grain)
        except ValueError:
            return False
        return True

    @staticmethod
    def _parse_time_grain(grain: str) -> tuple[str, int]:
        """解析 MARS 日期粒度，返回单位和正整数间隔。"""
        normalized = grain.strip().lower()
        aliases = {
            "day": ("d", 1),
            "week": ("w", 1),
            "month": ("m", 1),
        }
        if normalized in aliases:
            return aliases[normalized]

        match = re.fullmatch(r"([1-9]\d*)([dwm])", normalized)
        if match is None:
            raise ValueError(
                f"Invalid time grain '{grain}'. Expected day/week/month or Nd/Nw/Nm."
            )
        return match.group(2), int(match.group(1))

    @staticmethod
    def format_dt(dt: Union[str, pl.Expr], fmt: str = "%Y-%m-%d") -> pl.Expr:
        """
        将日期解析并格式化为指定字符串。

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。
        fmt : str
            输出的格式化字符串，默认 "%Y-%m-%d"。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Utf8`` (String) 的表达式。

        Examples
        --------
        >>> df = pl.DataFrame({"dt": ["2026-01-28"]})
        >>> df.select(MarsDate.format_dt("dt", "%Y/%m/%d").alias("dt")).item()
        '2026/01/28'
        """
        return MarsDate.smart_parse_expr(dt).dt.strftime(fmt)
