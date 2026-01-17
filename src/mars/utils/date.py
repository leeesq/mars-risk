import polars as pl
from typing import Union

class MarsDate:
    """
    [Mars 工具箱] 日期处理核心组件 (Pure Polars Edition).
    
    专为 Polars DataFrame 操作设计。
    所有方法均返回 pl.Expr，利用 Rust 引擎进行惰性求值和并行计算。
    """

    @staticmethod
    def _to_expr(col: Union[str, pl.Expr]) -> pl.Expr:
        """内部转换：字符串默认视为列名"""
        if isinstance(col, str):
            return pl.col(col)
        return col

    @staticmethod
    def smart_parse_expr(col: Union[str, pl.Expr]) -> pl.Expr:
        """
        [智能解析] 生成多路尝试的解析表达式。
        
        优化策略: 
        1. 优先将所有输入视为 String 进行格式解析 (最安全，防止 Int 被误读为天数)。
        2. 将直接 Cast 放在最后作为兜底 (Handling valid Date/Datetime types)。
        
        支持: ISO(YYYY-MM-DD), YYYYMMDD, Slash(/), Dot(.), Int(20250101).
        """
        expr = MarsDate._to_expr(col)
        
        # 1. 强制转为 String 以统一处理 
        #    (解决了 Int 20250101 被误读为 "5万年以后" 的 Bug)
        str_expr = expr.cast(pl.Utf8)

        # 2. Coalesce: 从上到下尝试，返回第一个非 Null 的结果
        return pl.coalesce([
            # A. 紧凑格式 (20250101) - 包含 Int 类型转为 Str 后的情况
            str_expr.str.to_date("%Y%m%d", strict=False),
            
            # B. 标准 ISO 格式 (2025-01-01) - 包含原生的 Date/Datetime 转为 Str 后的情况
            str_expr.str.to_date("%Y-%m-%d", strict=False),
            
            # C. 斜杠格式 (2025/01/01)
            str_expr.str.to_date("%Y/%m/%d", strict=False),
            
            # D. 点号格式 (2025.01.01)
            str_expr.str.to_date("%Y.%m.%d", strict=False),
            
            # E. [兜底] 尝试直接 Cast
            #    如果输入本身已经是 pl.Date 或 pl.Datetime，上面的转 Str 解析也能成功，
            #    但为了保险起见（或者处理某些带时区的特殊 Datetime），保留这个作为最后手段。
            #    注意：Int 类型在步骤 A 就会命中返回，不会走到这一步，所以安全了。
            expr.cast(pl.Date, strict=False),
        ])

    @staticmethod
    def dt2day(dt: Union[str, pl.Expr]) -> pl.Expr:
        """转换为 'Day' 粒度 (pl.Date 类型)"""
        return MarsDate.smart_parse_expr(dt)

    @staticmethod
    def dt2week(dt: Union[str, pl.Expr]) -> pl.Expr:
        """转换为 'Week' 粒度 (所在周的周一)"""
        return (
            MarsDate.smart_parse_expr(dt)
            .dt.truncate("1w")
            .cast(pl.Date) 
        )

    @staticmethod
    def dt2month(dt: Union[str, pl.Expr]) -> pl.Expr:
        """转换为 'Month' 粒度 (所在月的1号)"""
        return (
            MarsDate.smart_parse_expr(dt)
            .dt.truncate("1mo")
            .cast(pl.Date)
        )
    
    @staticmethod
    def format_dt(dt: Union[str, pl.Expr], fmt: str = "%Y-%m-%d") -> pl.Expr:
        """[展示用] 将日期转换为字符串"""
        return MarsDate.smart_parse_expr(dt).dt.strftime(fmt)