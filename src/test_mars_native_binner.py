import unittest
import numpy as np
import polars as pl
import shutil
import tempfile
import sys
import os

# ------------------------------------------------------
# 动态调整路径以导入你的模块 (假设你的代码在 ./src 下)
# ------------------------------------------------------
try:
    from mars.feature.binning import MarsNativeBinner
except ImportError:
    # 如果你在当前目录下直接运行，尝试直接导入
    sys.path.append("./src")
    try:
        from mars.feature.binning import MarsNativeBinner
    except ImportError:
        print("❌ 无法导入 MarsNativeBinner，请确保文件路径正确。")
        sys.exit(1)

class TestMarsNativeBinner(unittest.TestCase):
    
    def setUp(self):
        """每个测试用例运行前的初始化"""
        print(f"\n{'='*10} Running: {self._testMethodName} {'='*10}")

    def test_low_cardinality_fix(self):
        """
        场景 1: 【低基数测试】
        数据只有 [1, 2, 3]，但要求分 10 箱。
        期望：不报错，不产生空箱，自动切换为按值分箱。
        """
        print("测试数据: 只有 1, 2, 3 三种值，要求 n_bins=10")
        df = pl.DataFrame({"feature": [1]*50 + [2]*30 + [3]*20})
        
        # 测试 Uniform
        binner = MarsNativeBinner(method="uniform", n_bins=10)
        binner.fit(df)
        cuts = binner.bin_cuts_["feature"]
        
        print(f"Uniform Cuts: {cuts}")
        
        # 验证：
        # 1. 切点数量应该很少 (3个值的中间点只有2个，加上inf共4个)
        # 理想切点: [-inf, 1.5, 2.5, inf]
        self.assertTrue(len(cuts) <= 5, "切点数量应该自动缩减")
        self.assertIn(1.5, cuts, "应该包含 1 和 2 的中间点")
        self.assertIn(2.5, cuts, "应该包含 2 和 3 的中间点")
        
        # Transform 验证
        res = binner.transform(df)
        counts = res["feature_bin"].value_counts()
        print("分箱结果分布:\n", counts)
        # 应该只有 3 个箱子 (00, 01, 02)
        self.assertEqual(len(counts), 3)

    def test_empty_bin_pruning(self):
        """
        场景 2: 【空箱合并测试】
        数据分布两极分化：0 和 100，中间断层。
        Uniform 分箱会在 20, 40, 60, 80 切分，导致中间全是空箱。
        期望：自动移除中间的无效切点。
        """
        print("测试数据: 0(100个) ...断层... 100(100个)，要求 Uniform 分 5 箱")
        df = pl.DataFrame({"feature": [0.0]*100 + [100.0]*100})
        
        binner = MarsNativeBinner(method="uniform", n_bins=5)
        binner.fit(df)
        cuts = binner.bin_cuts_["feature"]
        print(f"Optimized Cuts: {cuts}")
        
        # Transform 验证
        res = binner.transform(df)
        bin_counts = res["feature_bin"].value_counts()
        print("分箱结果分布:\n", bin_counts)
        
        # 验证没有 Count=0 的箱子被生成
        # 如果逻辑正确，应该只会有两个主要的箱子 (包含0的和包含100的)
        # 可能会有 inf 箱子，但中间不应有空箱
        # 注意：value_counts 默认不显示 count=0 的行，所以我们通过 cuts 长度判断
        # 原始 cuts 应该是 20, 40, 60, 80。优化后应该移除大部分。
        self.assertLess(len(cuts), 7, "中间的空箱切点应该被移除")

    def test_all_nulls(self):
        """
        场景 3: 【全空值测试】
        数据全为 Null 或 NaN。
        期望：不崩溃，所有结果标记为 Missing。
        """
        print("测试数据: 全为 None 或 NaN")
        df = pl.DataFrame({"feature": [None, np.nan, None]})
        
        binner = MarsNativeBinner(method="quantile", n_bins=5)
        binner.fit(df)
        cuts = binner.bin_cuts_["feature"]
        print(f"Cuts: {cuts}")
        
        # 应该只有 [-inf, inf]
        self.assertEqual(cuts, [float('-inf'), float('inf')])
        
        res = binner.transform(df)
        print("结果预览:\n", res)
        # 验证全为 Missing
        self.assertTrue(all(x == "Missing" for x in res["feature_bin"].to_list()))

    def test_single_value(self):
        """
        场景 4: 【单一值测试】
        数据全为 5.0。
        期望：不崩溃，分箱结果合理 (可能全在1个箱子)。
        """
        print("测试数据: 全为 5.0")
        df = pl.DataFrame({"feature": [5.0] * 100})
        
        binner = MarsNativeBinner(method="uniform", n_bins=5)
        binner.fit(df)
        cuts = binner.bin_cuts_["feature"]
        print(f"Cuts: {cuts}")
        
        res = binner.transform(df)
        print("结果分布:\n", res["feature_bin"].value_counts())
        self.assertEqual(cuts, [float('-inf'), float('inf')])

    def test_special_values_isolation(self):
        """
        场景 5: 【特殊值/缺失值隔离】
        混合数据：正常值 + 特殊值(-999) + 缺失值(None)。
        期望：特殊值和缺失值被正确隔离。
        """
        print("测试数据: 混合正常值、-999、-1、None")
        # -1 是用户定义的 missing, -999 是 special
        data = [10, 20, 30] * 10 + [-999]*5 + [-1]*5 + [None]*5
        df = pl.DataFrame({"feature": data})
        
        binner = MarsNativeBinner(
            method="quantile", 
            n_bins=3,
            special_values=[-999],
            missing_values=[-1]
        )
        binner.fit(df)
        res = binner.transform(df)
        
        print("结果分布:\n", res["feature_bin"].value_counts())
        
        # 验证
        vals = res["feature_bin"].to_list()
        self.assertIn("Missing", vals)       # 对应 -1 和 None
        self.assertIn("Special_-999", vals)  # 对应 -999
        # 检查是否还有其他箱子 (正常数值箱)
        normal_bins = [v for v in vals if "0" in v and "[" in v]
        self.assertTrue(len(normal_bins) > 0)

    def test_cart_parallel_correctness(self):
        """
        场景 6: 【决策树并行测试】
        验证 joblib 并行是否能在有 Target 的情况下正常工作。
        """
        print("测试数据: DT 分箱 (带 y)")
        # 构造一个简单的线性关系
        X = np.random.rand(1000, 2)
        # y = 1 if x0 > 0.5 else 0
        y = (X[:, 0] > 0.5).astype(int)
        
        df = pl.DataFrame(X, schema=["f0", "f1"])
        
        # 强制使用 2 个核心
        binner = MarsNativeBinner(method="cart", n_bins=3, n_jobs=2)
        binner.fit(df, y)
        
        cuts_f0 = binner.bin_cuts_["f0"]
        print(f"DT Cuts f0 (Target相关): {cuts_f0}")
        print(f"DT Cuts f1 (噪音): {binner.bin_cuts_['f1']}")
        
        # 验证 f0 的切点应该在 0.5 附近
        # cuts 格式: [-inf, 0.5xxx, inf]
        has_cut_near_05 = any(0.4 < c < 0.6 for c in cuts_f0)
        self.assertTrue(has_cut_near_05, "DT 应该能找到 0.5 这个重要切点")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)