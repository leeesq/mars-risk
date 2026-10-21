"""MARS 日期解析与粒度转换工具。"""

from __future__ import annotations

import re
from typing import Union

import polars as pl


class MarsDate:
    """
    为 Polars 数据框构造日期解析和分组表达式。

    专为 Polars DataFrame 操作设计。
    日期解析、分组和格式化方法返回 ``pl.Expr``，可用于 ``select`` 或 ``with_columns``。

    Attributes
    ----------
    None
        该工具类不维护实例状态。

    Notes
    -----
    构造表达式时不读取数据。表达式在数据框执行查询时求值，也可用于 LazyFrame。

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
        将列名转为 Polars 表达式，已有表达式原样返回。

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
        按多种日期格式依次尝试解析，取首个成功结果。

        同一列可以包含多种支持的日期格式。所有格式均解析失败时，结果为 Null。

        Notes
        -----
        输入先转为 ``pl.Utf8``，再按显式格式解析，避免直接将整数日期解释为天数偏移。
        依次尝试 ISO 日期、紧凑日期、斜杠日期、点号日期和带时间的 ISO 字符串。
        Datetime 解析结果只保留日期部分。

        Parameters
        ----------
        col : Union[str, pl.Expr]
            待解析的列名或表达式。支持 String、Int（如 20230101）、Date 和 Datetime 类型。

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

        # 先转字符串，避免把 20250101 这样的整数日期解释为天数偏移。
        str_expr = expr.cast(pl.Utf8)

        # coalesce 逐项取首个非 Null 结果；显式解析格式也避免直接 cast 日期的弃用警告。
        return pl.coalesce([
            # 标准 ISO 日期也覆盖原生 Date 列转成字符串后的值。
            str_expr.str.to_date("%Y-%m-%d", strict=False),

            # 紧凑日期覆盖整数转成字符串后的值。
            str_expr.str.to_date("%Y%m%d", strict=False),

            # 兼容斜杠和点号分隔的日期。
            str_expr.str.to_date("%Y/%m/%d", strict=False),

            str_expr.str.to_date("%Y.%m.%d", strict=False),

            # 原生 Datetime 列转成字符串后，解析时间部分并取日期。
            str_expr.str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False).dt.date(),
            str_expr.str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).dt.date(),
        ])

    @staticmethod
    def dt2day(dt: Union[str, pl.Expr], interval: str = "1d") -> pl.Expr:
        """
        按指定天数生成日期或日期区间。

        多天区间以当前表达式作用范围内的最小有效日期为起点。
        区间标签形如 ``"20260101-0103"``，包含起止两天。

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。
        interval : str
            每个区间的天数，默认 ``"1d"``。支持 ``"day"``、``"1d"``、
            ``"3d"``、``"14d"`` 和 ``"30d"`` 等格式。

        Returns
        -------
        pl.Expr
            间隔为 1 天时返回 ``pl.Date``；多天时返回 ``pl.Utf8``（String）区间标签。
            无法解析的日期保留为 Null。

        Raises
        ------
        ValueError
            ``interval`` 既不是 ``"day"``，也不是数字加 ``"d"`` 的格式时抛出。

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

        # 多天区间从本次数据的最早有效日期开始，同一日期在不同数据切片中可能分到不同区间。
        min_dt = parsed_dt.min()

        # 计算每一行日期与全局锚点相差的天数
        diff_days = (parsed_dt - min_dt).dt.total_days()

        # 计算该行所属区间的起始偏移天数
        # 数学逻辑：例如 n=3，相差 4 天 -> (4 // 3) * 3 = 3，即落在第 3 天开始的区间
        offset_days_expr = (diff_days // n_days) * n_days

        # 动态推算区间的起止日期
        start_of_period = min_dt + pl.duration(days=offset_days_expr)
        end_of_period = start_of_period + pl.duration(days=n_days - 1)

        # 区间末日包含在内，标签与周分组统一使用 YYYYMMDD-MMDD 格式。
        return pl.concat_str([
            start_of_period.dt.strftime("%Y%m%d"),
            pl.lit("-"),
            end_of_period.dt.strftime("%m%d")
        ])

    @staticmethod
    def dt2week(dt: Union[str, pl.Expr], interval: str = "1w") -> pl.Expr:
        """
        按指定周数生成日期区间标签，如 ``"20260126-0201"``。

        每周从周一开始。多周区间以当前数据中最早日期所在周的周一为起点。

        Parameters
        ----------
        dt : Union[str, pl.Expr]
            日期列名或表达式。
        interval : str
            每个区间的周数，默认 ``"1w"``。支持 ``"week"``、``"1w"``、
            ``"2w"`` 等格式，周数必须为正整数。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Utf8``（String）的区间标签表达式，包含起止两天。

        Raises
        ------
        ValueError
            ``interval`` 不是 ``"week"`` 或正整数加 ``"w"`` 的格式时抛出。

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
            每个区间的自然月数，默认 ``"1m"``。支持 ``"month"``、``"1m"``、
            ``"2m"`` 等格式，月数必须为正整数。``m`` 表示月。

        Returns
        -------
        pl.Expr
            类型为 ``pl.Utf8``（String）的表达式。单月标签为 ``YYYYMM``，
            多月标签为 ``YYYYMM-YYYYMM``；多月区间按连续月份序号整除划分。

        Raises
        ------
        ValueError
            ``interval`` 不是 ``"month"`` 或正整数加 ``"m"`` 的格式时抛出。

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
        try:
            unit, n_units = MarsDate._parse_time_grain(grain)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
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
            控制输出字符串的日期格式，默认 ``"%Y-%m-%d"``。

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
