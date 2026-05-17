"""MARS 核心估计器、转换器与筛选器基类。"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union, Literal, Dict
import re

import polars as pl
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from mars.core.exceptions import NotFittedError, DataTypeError
from mars.utils.decorators import time_it
from mars.utils.logger import logger


class MarsBaseEstimator(BaseEstimator):
    """
    [MARS 基类] 负责输入数据的类型检测和输出数据的格式化。
    
    集成 Scikit-learn 的 BaseEstimator，支持 set_output API，
    允许用户在管道中灵活控制输出格式（Pandas 或 Polars）。
    """
    
    _PL_NUMERIC_TYPES = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, 
        pl.Float32, pl.Float64
    }

    def __init__(self) -> None:
        """
        初始化基础估计器状态。

        Notes
        -----
        初始化阶段仅负责记录输出格式偏好，不执行任何数据转换。
        """
        self._return_pandas: bool = False
        self._output_config: str = "default"

    def set_output(self, transform: Literal["default", "pandas", "polars"] = "default") -> "MarsBaseEstimator":
        """
        兼容 Sklearn 的 set_output API, 允许用户强制指定输出格式。

        Parameters
        ----------
        transform : Literal["default", "pandas", "polars"]
            - "pandas": 强制输出 Pandas DataFrame。
            - "polars": 强制输出 Polars DataFrame。
            - "default": 保持默认行为, 跟随输入类型。

        Returns
        -------
        MarsBaseEstimator
            返回实例本身以支持链式调用。
        """
        if transform not in ["default", "pandas", "polars"]:
            raise ValueError(f"Unknown output format: {transform}")
            
        self._output_config = transform
        
        if transform == "pandas":
            self._return_pandas = True
        elif transform == "polars":
            self._return_pandas = False
            
        return self

    def _determine_output_format(self, input_is_pandas: bool) -> None:
        """
        根据用户配置(_output_config)和输入类型，决定最终输出格式。
        
        决策优先级: set_output > 输入类型
        """
        if self._output_config == "pandas":
            self._return_pandas = True
        elif self._output_config == "polars":
            self._return_pandas = False
        else: # default
            self._return_pandas = input_is_pandas

    def _ensure_polars_dataframe(
        self, 
        X: pl.DataFrame | pl.LazyFrame | pd.DataFrame
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        确保输入数据转换为 Polars DataFrame/LazyFrame, 并执行严格校验。
        
        同时根据输入类型设置默认的输出格式。
        """
        # Case 1: 已经是 Polars (Eager or Lazy)
        if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            self._determine_output_format(input_is_pandas=False)
            return X

        # Case 2: 是 Pandas
        elif isinstance(X, pd.DataFrame):
            # 决定输出格式
            self._determine_output_format(input_is_pandas=True)
            
            # 转换
            try:
                X_pl = pl.from_pandas(X)
            except Exception as e:
                raise DataTypeError(f"Failed to convert Pandas DataFrame to Polars: {e}")
            
            # 转换后的类型一致性检查
            self._validate_conversion(X, X_pl)
            
            return X_pl
            
        elif isinstance(X, (pd.Series, pl.Series)):
            raise DataTypeError(f"Input must be a generic DataFrame (2D), got Series (1D): {type(X)}")
        else:
            raise DataTypeError(f"Mars expects Polars/Pandas DataFrame, got {type(X)}")
        
    def _ensure_polars_series(
        self, 
        y: pl.Series | pd.Series | np.ndarray | list | None, 
        name: str = "target"
    ) -> Optional[pl.Series]:
        """确保标签 y 转换为 Polars Series。"""
        if y is None:
            return None
            
        if isinstance(y, pl.Series):
            return y
            
        if isinstance(y, pd.Series):
            return pl.from_pandas(y)
            
        if isinstance(y, np.ndarray):
            # 确保 1D 数组
            if y.ndim > 1:
                y = y.ravel()
            return pl.Series(name=name, values=y)
            
        if isinstance(y, list):
            return pl.Series(name=name, values=y)
            
        return pl.Series(name=name, values=y)

    def _validate_conversion(self, df_pd: pd.DataFrame, df_pl: Union[pl.DataFrame, pl.LazyFrame]):
        """
        对比 Pandas 和 Polars 的 Schema, 防止数值类型意外崩坏为字符串。
        """
        # 预处理：直接获取 Schema (LazyFrame 支持直接读取元数据，无需 collect)
        pl_schema = df_pl.schema

        # 定义需要跳过启发式检查的列名模式
        SKIP_PATTERNS = re.compile(
            r".*(_id|_uuid|_dt|_at|_ts|_date|_time|_on)$|"  # 后缀: user_id, created_at
            r"^(id_|date_|time_|ts_)|"                # 前缀: id_code
            r"^(id|dt|ts|year|month|day|hour|minute|second)$", # 精确匹配
            re.IGNORECASE
        )

        for col in df_pd.columns:
            pd_dtype = df_pd[col].dtype
            pl_dtype = pl_schema.get(col)
            
            # 检查 1: 严格拦截 (Pandas 明确是数值 -> Polars 变成了非数值)
            is_pd_numeric = pd.api.types.is_numeric_dtype(pd_dtype)
            is_pl_numeric = pl_dtype in self._PL_NUMERIC_TYPES
            
            if is_pd_numeric and not is_pl_numeric:
                # 容忍全空列 (Polars 默认推断为 Null 类型)
                if pl_dtype == pl.Null:
                    continue
                raise DataTypeError(
                    f"Column '{col}' is numeric in Pandas ({pd_dtype}) "
                    f"but converted to non-numeric in Polars ({pl_dtype}). "
                    f"Check for mixed dtypes in your Pandas DataFrame."
                )

            # 检查 2: 脏数据陷阱预警 (针对 Object -> Utf8 的启发式嗅探)
            if pd_dtype == "object" and pl_dtype == pl.Utf8:
                
                # 如果命中了 ID 或 时间类的命名规则，直接跳过警告逻辑
                if SKIP_PATTERNS.match(str(col)):
                    continue

                # 仅抽样当前列的少量非空值，用于识别“看起来像数值”的脏字符串列。
                if isinstance(df_pl, pl.LazyFrame):
                    # 对 LazyFrame 只物化当前列的前几个非空样本，避免整表 collect。
                    sample_series = (
                        df_pl.select(pl.col(col))
                        .drop_nulls()
                        .head(10)
                        .collect()
                        .to_series()
                    )
                else:
                    sample_series = df_pl[col].drop_nulls().head(10)

                if sample_series.len() == 0:
                    continue
                
                samples = sample_series.to_list()
                
                looks_like_numeric = True
                try:
                    for s in samples:
                        # 排除掉纯空格、空字符串或长 ID 类数字的干扰, 如果长度超过 15 位且是纯数字，不需要预警
                        s_str = str(s).strip()
                        if not s_str or (s_str.isdigit() and len(s_str) > 15):
                            looks_like_numeric = False
                            break
                        float(s_str)
                except (ValueError, TypeError):
                    looks_like_numeric = False
                
                if looks_like_numeric:
                    logger.warning(
                        f"\nPotential dirty data detected: column '{col}' looks numeric but is treated as String.\n"
                        f"   - Input (Pandas): object (mixed types)\n"
                        f"   - Output (Polars): Utf8\n"
                        f"   - Sample Values: {samples[:5]}...\n"
                        f"   - Impact: This column will be handled as Categorical. If it contains dirty strings "
                        f"(e.g. 'null', 'unknown'), please clean them upstream or add them to 'missing_values'.",
                        stacklevel=2
                    )

    def _format_output(self, data: Any) -> Any:
        """
        根据 _return_pandas 标志位，决定是否将结果转回 Pandas。

        支持递归处理字典和列表结构。

        Parameters
        ----------
        data : Any
            待格式化的数据 (DataFrame, Dict, List 等)。

        Returns
        -------
        Any
            格式化后的数据。
        """
        # 如果不需要转 Pandas，或者数据本来就是 Polars，直接返回
        if not self._return_pandas:
            return data

        # 递归处理字典
        if isinstance(data, dict):
            return {k: self._format_output(v) for k, v in data.items()}
        
        # 递归处理列表
        if isinstance(data, list):
            return [self._format_output(v) for v in data]

        # Polars -> Pandas
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
            
        return data

class MarsTransformer(MarsBaseEstimator, TransformerMixin, ABC):
    """
    [Mars 转换器基类] 集成了自动 Pandas 互操作性。
    """

    def __init__(self):
        """初始化转换器运行状态。"""
        super().__init__() 
        self.feature_names_in_: List[str] = []
        self._is_fitted: bool = False

    def __sklearn_is_fitted__(self) -> bool:
        """向 sklearn 暴露当前实例的拟合状态。"""
        return self._is_fitted
    
    def _check_is_fitted(self):
        """检查当前转换器实例是否已完成拟合。"""
        if not self._is_fitted:
            from mars.core.exceptions import NotFittedError
            raise NotFittedError(f"{self.__class__.__name__} is not fitted yet. Call 'fit' first.")

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        返回转换器输出特征名列表。

        Parameters
        ----------
        input_features : Any, optional
            sklearn 兼容保留参数，当前实现未使用。

        Returns
        -------
        list of str
            拟合阶段缓存的输入特征名。
        """
        return self.feature_names_in_

    @time_it
    def fit(
        self, 
        X: pl.DataFrame | pd.DataFrame, 
        y: Optional[pl.Series | pd.Series | np.ndarray | list[Any]] = None, 
        **kwargs
    ) -> "MarsTransformer":
        """
        拟合转换器并缓存输入特征信息。

        Parameters
        ----------
        X : DataFrame
            输入特征矩阵。
        y : Series or ndarray or list, optional
            目标变量。
        **kwargs
            透传给底层 `_fit_impl` 的附加参数。

        Returns
        -------
        MarsTransformer
            拟合完成后的转换器实例。
        """
        if isinstance(X, pd.DataFrame) and isinstance(y, (pd.Series, pd.DataFrame)):
            if not X.index.equals(y.index):
                raise ValueError(
                    "X and y have different indices. "
                    "Converting to Polars will lose index information leading to row mismatch. "
                    "Please align indices in Pandas strictly before passing to Mars."
                )
                
        X_pl = self._ensure_polars_dataframe(X)
        y_pl = self._ensure_polars_series(y)
        self._fit_impl(X_pl, y_pl, **kwargs)
        
        # 更新状态
        self.feature_names_in_ = X_pl.columns
        self._is_fitted = True
        return self

    def transform(
        self, 
        X: pl.DataFrame | pl.LazyFrame | pd.DataFrame, 
        **kwargs
    ) -> pl.DataFrame | pd.DataFrame | pl.LazyFrame:
        """
        模板方法：Transform。
        
        支持通过 **kwargs 向 _transform_impl 传递额外参数 (如 return_type)。
        """
        self._check_is_fitted()
        
        X_pl = self._ensure_polars_dataframe(X)
        X_new = self._transform_impl(X_pl, **kwargs)
        
        # 输出格式化 (Polars -> Pandas/List/Dict)
        return self._format_output(X_new)
    
    def fit_transform(
        self, 
        X: Union[pl.DataFrame, pd.DataFrame], 
        y: Optional[Any] = None, **kwargs
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        执行拟合并返回转换结果。
        """
        return self.fit(X, y, **kwargs).transform(X)

    @abstractmethod
    def _fit_impl(self, X: pl.DataFrame, y=None, **kwargs): 
        """
        子类必须实现的核心拟合逻辑。
        """
        pass

    @abstractmethod
    def _transform_impl(self, X: pl.DataFrame) -> pl.DataFrame: 
        """
        子类必须实现的核心转换逻辑。
        必须返回 Polars DataFrame。
        """
        pass
    
class MarsBaseSelector(MarsBaseEstimator, ABC):
    """
    [Mars 特征筛选器基类] MarsBaseSelector
    
    所有特征筛选组件 (Stats, Linear, Importance) 的父类。
    它封装了特征管理的通用逻辑，并提供了一套标准的“尸检报告”生成机制。
    
    Attributes
    ----------
    selected_features_ : List[str]
        最终幸存的特征列表。
    n_features_in_ : int
        初始输入的特征数量。
    report_records_ : List[Dict[str, Any]]
        筛选过程的详细日志记录。
        格式: [{'feature': 'age', 'status': 'Dropped', 'stage': 'Quality', 'reason': 'Missing>0.95', 'value': 0.98}, ...]
    """

    def __init__(self, target: str):
        """
        初始化筛选器公共状态。

        Parameters
        ----------
        target : str
            目标变量列名。
        """
        super().__init__()
        self.target = target
        
        # 状态属性
        self.selected_features_: List[str] = []
        self.n_features_in_: int = 0
        self.report_records_: List[Dict[str, Any]] = []
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, X: Union[pl.DataFrame, pd.DataFrame], y: Optional[Any] = None) -> "MarsBaseSelector":
        """
        [抽象方法] 执行筛选逻辑。必须由子类实现。
        """
        pass

    def transform(self, X: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
        """
        [通用方法] 根据筛选结果裁剪特征。
        
        只保留 selected_features_ 中的列，以及 target (如果存在)。
        """
        self._check_is_fitted()
        X = self._ensure_polars_dataframe(X)
        
        # 构造保留列表：选中特征 + 目标列(若存在)
        # 注意：我们通常不删 Target，方便 Pipeline 后续步骤使用
        cols_to_keep = [c for c in self.selected_features_ if c in X.columns]
        
        # 如果原始数据里有 Target，也带上（除非它已经在 selected_features_ 里）
        if self.target in X.columns and self.target not in cols_to_keep:
            cols_to_keep.append(self.target)
            
        X_out = X.select(cols_to_keep)
        return self._format_output(X_out)
    
    def fit_transform(
        self, 
        X: Union[pl.DataFrame, pd.DataFrame], 
        y: Optional[Any] = None, 
        **kwargs
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        执行拟合并返回筛选后的数据。
        """
        return self.fit(X, y, **kwargs).transform(X)

    def get_report(self) -> pl.DataFrame:
        """
        [核心功能] 生成特征筛选“尸检报告”。
        
        Returns
        -------
        pl.DataFrame
            包含每个特征的筛选状态、淘汰阶段、淘汰原因及具体指标值。
            Schema: [feature, status, stage, reason, value, description]
        """
        self._check_is_fitted()
        
        if not self.report_records_:
            logger.warning("No report records found. Did you forget to call `_register_decision` in a subclass?")
            return pl.DataFrame([])

        # 将记录列表转换为 DataFrame
        report_df = pl.DataFrame(self.report_records_, schema={
            "feature": pl.Utf8,
            "data_source": pl.Utf8,
            "status": pl.Utf8,       # 'Selected' / 'Dropped'
            "stage": pl.Utf8,        # 'Quality', 'IV_Rough', 'PSI'
            "reason": pl.Utf8,       # 'Low IV', 'High Corr'
            "value": pl.Float64,     # 具体的指标值 (如 IV=0.001)
            "description": pl.Utf8   # 详细描述
        })
        
        # 按照 状态(Dropped优先) -> 阶段 -> 特征名 排序
        sorted_report = report_df.sort(["status", "stage", "feature"], descending=[False, False, False])
        # 【修复】通过格式化输出保证双引擎兼容
        return self._format_output(sorted_report)

    def _register_decision(
        self, 
        feature: str, 
        status: str, 
        stage: str, 
        reason: str = "", 
        value: float = -1.0,
        desc: str = "",
        data_source: Optional[str] = None
    ):
        """
        记录单个特征在筛选过程中的决策结果。
        
        Parameters
        ----------
        feature : str
            特征名。
        status : str
            'Selected' 或 'Dropped'。
        stage : str
            当前筛选阶段 (如 'Rough_Scan', 'VIF_Check')。
        reason : str
            决策依据 (如 'IV < 0.02')。
        value : float
            关键指标值 (如 0.015)。
        """
        self.report_records_.append({
            "feature": feature,
            "data_source": data_source,
            "status": status,
            "stage": stage,
            "reason": reason,
            "value": value,
            "description": desc
        })

    def _get_feature_pool(self, X: pl.DataFrame) -> List[str]:
        """
        获取初始特征池。
        """
        # 这里可以加入通用的排除逻辑，比如日期列、ID列等
        # 暂时只排除 Target
        return [c for c in X.columns if c != self.target]

    def _check_is_fitted(self):
        """校验筛选器是否已经完成拟合。"""
        if not self._is_fitted:
            raise ValueError(f"{self.__class__.__name__} is not fitted yet. Call 'fit' first.")
