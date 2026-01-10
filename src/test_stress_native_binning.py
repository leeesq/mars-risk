import polars as pl
import numpy as np
import time
import psutil
import os
import gc
from typing import Tuple

# 调整导入路径以匹配你的项目结构
try:
    # 注意：这里引用的是最新的 binning 模块
    from mars.feature.binning import MarsNativeBinner
    from mars.utils.logger import set_log_level, logger
except ImportError:
    import sys
    sys.path.append("./src")
    from mars.feature.binning import MarsNativeBinner
    from mars.utils.logger import set_log_level, logger

# ==========================================
# ⚙️ 压测配置
# ==========================================
set_log_level("INFO")

N_ROWS = 200_000      # 20万行
N_COLS = 2_000         # 2000特征 (大宽表)
SPECIAL_VAL = -999.0   # 特殊值
MISSING_VAL = -1.0     # 业务缺失值

def get_memory_usage() -> float:
    """获取当前进程内存占用 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def generate_massive_data() -> Tuple[pl.DataFrame, np.ndarray]:
    """
    极速生成 20万 x 2000列 数据 (直接操作 Numpy 内存，避免循环)
    """
    logger.info(f"🚀 [DataGen] Allocating {N_ROWS:,} rows x {N_COLS} columns...")
    t0 = time.time()
    
    # 1. 直接生成大矩阵 (Float32 节省内存)
    data_matrix = np.random.randn(N_ROWS, N_COLS).astype(np.float32)
    
    # 2. 注入 Special Values (-999) - 随机 5%
    flat_view = data_matrix.ravel()
    n_elements = flat_view.size
    n_special = int(n_elements * 0.05)
    indices_spec = np.random.choice(n_elements, n_special, replace=False)
    flat_view[indices_spec] = SPECIAL_VAL
    
    # 3. 注入 Missing Values (None/NaN) - 随机 5%
    n_missing = int(n_elements * 0.05)
    indices_miss = np.random.choice(n_elements, n_missing, replace=False)
    flat_view[indices_miss] = np.nan
    
    # 4. 生成 Target
    prob = 1 / (1 + np.exp(-np.clip(data_matrix[:, 0], -10, 10)))
    y = (np.random.rand(N_ROWS) < prob).astype(int)
    
    # 5. 转 Polars (Zero-Copy 转换)
    logger.info("📦 Wrapping into Polars DataFrame...")
    col_names = [f"f_{i}" for i in range(N_COLS)]
    df = pl.from_numpy(data_matrix, schema=col_names)
    
    logger.info(f"✅ Data Ready in {time.time() - t0:.2f}s | Memory: {get_memory_usage():.2f} MB")
    return df, y

def test_method(df: pl.DataFrame, y: np.ndarray, method: str, desc: str):
    print("\n" + "-"*60)
    print(f"🧪 Testing Method: [{method.upper()}] - {desc}")
    print("-"*60)
    
    gc.collect()
    mem_start = get_memory_usage()
    
    # 初始化
    binner = MarsNativeBinner(
        method=method,
        n_bins=5,
        special_values=[SPECIAL_VAL],
        missing_values=[MISSING_VAL], 
        n_jobs=-1 
    )
    
    # --- Fit 测试 ---
    t0 = time.time()
    binner.fit(df, y)
    t_fit = time.time() - t0
    mem_peak = get_memory_usage() - mem_start
    
    print(f"   ⏱️  Fit Time:       {t_fit:.4f} s")
    print(f"   💾  Mem Overhead:   {mem_peak:.2f} MB")
    
    # 打印一些切点信息用于验证
    if method in ["cart", "uniform"]:
        cut_0 = binner.bin_cuts_.get("f_0", [])
        print(f"   🔎  Cuts (f_0):     {cut_0}")

    # --- Transform 测试 ---
    t1 = time.time()
    df_res = binner.transform(df)
    # 强制计算
    _ = df_res[f"f_0_bin"].value_counts()
    t_trans = time.time() - t1
    
    print(f"   🚀  Transform Time: {t_trans:.4f} s")
    
    # --- 验证 ---
    counts = df_res["f_0_bin"].value_counts().sort("f_0_bin")
    print(f"\n   🧐 Sample Distribution (f_0_bin):")
    print(counts.head(7))
    
    return t_fit, t_trans

def run_stress_test():
    # 1. 数据生成
    df_train, y_train = generate_massive_data()
    
    print("\n" + "="*80)
    print(f"🔥 MARS NATIVE BINNER STRESS TEST")
    print(f"🔥 Dimensions: {N_ROWS:,} rows x {N_COLS} columns")
    print("="*80)
    
    # 2. 测试 Quantile (基准线)
    t_fit_q, t_trans_q = test_method(
        df_train, y_train, 
        "quantile", 
        "Pure Polars (Zero-Copy)"
    )

    # 3. 测试 Uniform (等宽分箱 - 新增)
    # 预期：比 Quantile 更快，因为 min/max 计算比 quantile 排序要快得多
    t_fit_u, t_trans_u = test_method(
        df_train, y_train, 
        "uniform", 
        "Pure Polars (Min/Max)"
    )
    
    # 4. 测试 Decision Tree (并行)
    t_fit_cart, t_trans_cart = test_method(
        df_train, y_train, 
        "cart", 
        "Parallel Sklearn (n_jobs=-1)"
    )
    
    # 5. 总结
    print("\n" + "="*80)
    print("🏆 FINAL SCOREBOARD")
    print("="*80)
    print(f"{'Method':<15} | {'Fit Time':<15} | {'Transform Time':<15} | {'Note':<20}")
    print("-" * 75)
    print(f"{'Quantile':<15} | {t_fit_q:<15.4f} | {t_trans_q:<15.4f} | {'Sorting based'}")
    print(f"{'Uniform':<15} | {t_fit_u:<15.4f} | {t_trans_u:<15.4f} | {'Min/Max based'}")
    print(f"{'DT (Parallel)':<15} | {t_fit_cart:<15.4f} | {t_trans_cart:<15.4f} | {'Tree based'}")
    print("-" * 75)
    
    # 显式清理
    del df_train, y_train
    gc.collect()

if __name__ == "__main__":
    run_stress_test()