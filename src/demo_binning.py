import polars as pl
import numpy as np
import time
from mars.feature.binning import BinningProcess
from mars.utils.logger import set_log_level, logger

set_log_level("INFO")

def run_binning_test():
    N_ROWS = 1_000_000
    N_COLS = 200
    
    logger.info(f"生成模拟数据: {N_ROWS:,} 行, {N_COLS} 特征...")
    
    # 1. 造数据
    # 特征 X
    data = np.random.randn(N_ROWS, N_COLS)
    # 目标 Y (造一个简单的线性关系，让 IV 有意义)
    # y = 1 if (3*x_0 - 2*x_1 + noise) > 0 else 0
    logits = 3 * data[:, 0] - 2 * data[:, 1] + np.random.randn(N_ROWS)
    y = (logits > 0).astype(int)
    
    data_dict = {f"feat_{i}": data[:, i] for i in range(N_COLS)}
    data_dict["target"] = y
    
    # 注入一些 Null 值测试鲁棒性
    # 把 feat_0 的前 1000 个设为 None
    # 注意: Polars 构造时 None 会转为 null
    
    df = pl.DataFrame(data_dict)
    
    # 模拟 feat_0 缺失
    df = df.with_columns(
        pl.when(pl.col("feat_0") > 2).then(None).otherwise(pl.col("feat_0")).alias("feat_0")
    )
    
    print("-" * 60)
    logger.info(">>> 开始自动分箱 (Decision Tree Binning) ...")
    
    # 2. 初始化分箱器
    binner = BinningProcess(
        target="target",
        max_bins=5,
        min_samples_leaf=0.05,
        method="decision_tree" # 最优分箱
    )
    
    # 3. 训练 (Fit)
    start_time = time.time()
    feature_cols = [c for c in df.columns if c != "target"]
    binner.fit(df, feature_cols)
    end_time = time.time()
    
    print("-" * 60)
    logger.info(f"✅ 分箱完成! 耗时: {end_time - start_time:.4f} 秒")
    
    # 4. 查看 IV 结果
    print("\n>>> IV (Information Value) 汇总表:")
    iv_df = binner.get_summary()
    print(iv_df)
    
    print("\n>>> 验证逻辑:")
    print("feat_0 和 feat_1 应该是强特征 (IV 高)，其他是噪声 (IV 低)。")
    
    # 5. 查看某个特征的分箱切点
    print(f"\n>>> feat_0 的切分点 (Splits):")
    print(binner.bin_cuts["feat_0"])

if __name__ == "__main__":
    run_binning_test()