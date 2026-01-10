import polars as pl
import pandas as pd
import numpy as np
import time
import psutil
import os
import gc
import sys
from typing import Tuple, Union

# ==========================================
# 🛠️ Attempt to import Mars module
# ==========================================
try:
    from mars.analysis.profiler import MarsDataProfiler  # Updated import path
    from mars.utils.logger import set_log_level, logger
except ImportError as e:
    print(f"❌ Failed to import Mars module: {e}")
    print("Please check your PYTHONPATH.")
    sys.exit(1)

# ==========================================
# ⚙️ Stress Test Configuration
# ==========================================
set_log_level("WARNING")  # Reduce IO interference

# Standard medium-sized risk dataset configuration (200k rows x 2000 cols)
N_ROWS: int = 200000
N_COLS: int = 2000
N_CATS: int = 50
N_GROUPS: int = 12

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def get_memory_usage() -> float:
    """Get current process memory (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def generate_data() -> pl.DataFrame:
    """Fast large-scale test data generation"""
    print(f"{Colors.CYAN}🚀 [DataGen] Generating {N_ROWS:,} rows x {N_COLS} cols...{Colors.RESET}")
    start = time.time()
    
    # 1. Numerical columns (Matrix generation is faster)
    n_num = N_COLS - N_CATS
    # Using float32 to save memory for the demo, can be float64
    data = (np.random.randn(N_ROWS, n_num).astype(np.float32) * 10) + 100
    
    # Inject missing values (-999)
    mask = np.random.random(data.shape) < 0.1
    data[mask] = -999
    
    data_dict = {f"num_{i}": data[:, i] for i in range(n_num)}
    
    # 2. Categorical columns
    cats = ["A", "B", "C", "D", "E", "unknown", None]
    for i in range(N_CATS):
        data_dict[f"cat_{i}"] = np.random.choice(cats, size=N_ROWS).tolist()
        
    # 3. Group column
    groups = [f"2023{m:02d}" for m in range(1, N_GROUPS + 1)]
    data_dict["month"] = np.random.choice(groups, size=N_ROWS).tolist()
    
    df = pl.DataFrame(data_dict)
    size_mb = df.estimated_size('mb')
    print(f"✅ Data Ready! Size: {size_mb:.2f} MB | Time: {time.time()-start:.2f}s")
    
    # Check if memory is sufficient for conversion later
    if size_mb * 3 > psutil.virtual_memory().available / 1024 / 1024:
        print(f"{Colors.RED}⚠️ Warning: Dataset might be too large for Pandas conversion on this machine.{Colors.RESET}")
        
    return df

def run_benchmark_round(df: Union[pl.DataFrame, pd.DataFrame], backend: str) -> Tuple[float, float, float]:
    """Execute a single round of stress testing"""
    print(f"\n🔹 Testing Backend: {Colors.BOLD}{backend}{Colors.RESET}")
    print("-" * 60)
    
    # Configuration overrides for benchmark (disable sparklines for pure calc speed)
    # If you want to test sparkline performance, remove "enable_sparkline": False
    bench_config = {"enable_sparkline": False} 

    # 1. Initialization
    gc.collect()
    t0 = time.time()
    # Initialize Engine
    profiler = MarsDataProfiler(df, custom_missing_values=[-999, "unknown"])
    t_init = time.time() - t0
    print(f"   1. Init Engine       : {t_init:.4f} s")
    
    # 2. Overview Only (Simulates old get_report)
    # Using profile_by=None to get global stats
    t1 = time.time()
    _ = profiler.generate_profile(profile_by=None, config_overrides=bench_config)
    t_report = time.time() - t1
    print(f"   2. Full Overview     : {t_report:.4f} s")
    
    # 3. Group Profile (Stability Analysis)
    # Using profile_by="month"
    t2 = time.time()
    _ = profiler.generate_profile(
        profile_by="month", 
        config_overrides=bench_config # Reuse config to keep sparklines off
    )
    t_profile = time.time() - t2
    print(f"   3. Group Profile (by): {t_profile:.4f} s")
    
    return t_init, t_report, t_profile

def print_final_report(pl_times, pd_times):
    """Print comparison report"""
    stages = ["Initialization", "Get Full Overview", "Generate Group Profile"]
    
    print(f"\n{Colors.BOLD}{'🏆 BENCHMARK RESULTS (Time in Seconds)':^65}{Colors.RESET}")
    print("=" * 65)
    print(f"| {'Stage':<24} | {'Polars':<10} | {'Pandas':<10} | {'Speedup':<10} |")
    print("|" + "-"*26 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "|")
    
    for stage, t_pl, t_pd in zip(stages, pl_times, pd_times):
        # Calculate speedup
        speedup = t_pd / t_pl if t_pl > 0 else 0
        
        # Color code the winner
        if t_pl < t_pd:
            pl_str = f"{Colors.GREEN}{t_pl:.4f}{Colors.RESET}"
            pd_str = f"{t_pd:.4f}"
        else:
            pl_str = f"{t_pl:.4f}"
            pd_str = f"{Colors.GREEN}{t_pd:.4f}{Colors.RESET}"
            
        print(f"| {stage:<24} | {pl_str:<19} | {pd_str:<19} | {speedup:>9.1f}x |")
    print("=" * 65)
    print(f"{Colors.CYAN}* Speedup > 1.0x means Polars is faster.{Colors.RESET}")
    print(f"{Colors.CYAN}* Note: Sparklines were disabled for pure calculation benchmark.{Colors.RESET}\n")

if __name__ == "__main__":
    # 1. Generate Data (Polars Native)
    df_pl = generate_data()
    
    # 2. Run Polars Benchmark
    pl_results = run_benchmark_round(df_pl, "Polars (Native)")
    
    # 3. Convert to Pandas for Comparison
    print(f"\n{Colors.CYAN}🔄 Converting to Pandas for compatibility test...{Colors.RESET}")
    try:
        t_conv = time.time()
        df_pd = df_pl.to_pandas()
        print(f"   Conversion time: {time.time() - t_conv:.2f}s")
        
        # Explicitly delete Polars DF to free memory if tight
        # del df_pl 
        gc.collect()
    except MemoryError:
        print(f"{Colors.RED}❌ OOM during conversion! Skipping Pandas test.{Colors.RESET}")
        sys.exit(1)
        
    # 4. Run Pandas Benchmark
    pd_results = run_benchmark_round(df_pd, "Pandas (Compat)")
    
    # 5. Show Summary
    print_final_report(pl_results, pd_results)