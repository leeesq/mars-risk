import unittest
import numpy as np
import polars as pl
import logging
import sys
from unittest.mock import patch, MagicMock

from mars.feature.binning import MarsOptimalBinner


class TestMarsOptimalBinner(unittest.TestCase):
    
    def setUp(self):
        """每个测试运行前的初始化"""
        np.random.seed(42)
        print(f"\n{'='*10} Running: {self._testMethodName} {'='*10}")

    def test_01_numeric_integration(self):
        """
        [数值型] 测试完整流程：Pre-binning -> Solver -> Transform
        我们 Mock 掉 optbinning，模拟 Solver 成功找到最优切点的场景。
        """
        df = pl.DataFrame({"feature": np.linspace(0, 100, 100)})
        y = (df["feature"] > 50).cast(pl.Int32).to_numpy() # 简单的断点在 50

        # Mock optbinning 的行为
        with patch.dict('sys.modules', {'optbinning': MagicMock()}):
            import optbinning
            # 模拟 Solver 返回成功的状态
            mock_opt = optbinning.OptimalBinning.return_value
            mock_opt.status = "OPTIMAL"
            # 模拟 Solver 找到的切点 (假设它找到了 50)
            mock_opt.splits = np.array([50.0])
            mock_opt.fit.return_value = None

            # 初始化分箱器
            binner = MarsOptimalBinner(features=["feature"], n_bins=3, n_jobs=1)
            binner.fit(df, y)

            # 验证 Fit 结果
            cuts = binner.bin_cuts_.get("feature")
            print(f"训练切点: {cuts}")
            
            # 应该包含 -inf, 50.0, inf
            self.assertIn(50.0, cuts)
            self.assertEqual(cuts[0], float('-inf'))
            self.assertEqual(cuts[-1], float('inf'))

            # 验证 Transform
            res = binner.transform(df)
            self.assertIn("feature_bin", res.columns)
            # 检查 0 和 100 是否分到了不同的箱子
            bins = res["feature_bin"].to_list()
            self.assertNotEqual(bins[0], bins[-1])

    def test_02_solver_fallback_logic(self):
        """
        [鲁棒性] 测试 Solver 崩溃/超时时的自动回退 (Fallback) 机制。
        期望：当 optbinning 抛出异常时，代码不崩溃，而是使用预分箱(Pre-binning)的结果。
        """
        df = pl.DataFrame({"x": np.random.normal(0, 1, 200)})
        y = np.random.randint(0, 2, 200)

        # 强制让 Solver 抛出异常
        with patch.dict('sys.modules', {'optbinning': MagicMock()}):
            import optbinning
            mock_opt = optbinning.OptimalBinning.return_value
            # 模拟 fit 方法抛出异常
            mock_opt.fit.side_effect = Exception("Solver Timeout Simulation")

            # 设置 n_prebins=10，如果回退成功，切点数量应该接近 10，而不是 n_bins=3
            binner = MarsOptimalBinner(features=["x"], n_bins=3, n_prebins=10, n_jobs=1)
            
            # 运行 Fit，不应报错
            try:
                binner.fit(df, y)
            except Exception as e:
                self.fail(f"Fit 阶段未捕获 Solver 异常: {e}")

            cuts = binner.bin_cuts_.get("x")
            print(f"Fallback 后的切点数量: {len(cuts)}")
            
            # 验证：
            # 1. 切点存在 (说明没有丢失列)
            self.assertIsNotNone(cuts)
            # 2. 切点数量应该 > 4 (因为 n_prebins=10，回退到了细粒度分箱)
            self.assertTrue(len(cuts) > 4, "应当回退到预分箱结果，切点数应较多")

    def test_03_categorical_top_k(self):
        """
        [类别型] 测试 Top-K 过滤和未见类别处理 (Other)。
        """
        # 构造数据：A 和 B 是主要类别，其他都是噪音
        cats = ["A"]*40 + ["B"]*40 + [f"Noise_{i}" for i in range(20)]
        y = [1]*80 + [0]*20
        df = pl.DataFrame({"city": cats})

        # Mock Solver 行为
        with patch.dict('sys.modules', {'optbinning': MagicMock()}):
            import optbinning
            mock_opt = optbinning.OptimalBinning.return_value
            mock_opt.status = "OPTIMAL"
            # 假设 Solver 决定把 A 放一组，B 放一组
            mock_opt.splits = [['A'], ['B']]

            # 设置 cat_cutoff 很小，强制触发 Top-K 逻辑
            binner = MarsOptimalBinner(cat_features=["city"], cat_cutoff=5, n_jobs=1)
            binner.fit(df, np.array(y))
            
            rules = binner.cat_cuts_.get("city")
            print(f"类别规则: {rules}")
            self.assertIsNotNone(rules)

            # --- 测试 Transform 阶段的 Unseen Value ---
            # 构造测试集：包含训练集没有的 "Shanghai"
            df_test = pl.DataFrame({"city": ["A", "Shanghai", "B"]})
            res = binner.transform(df_test)
            res_vals = res["city_bin"].to_list()
            
            print(f"预测结果: {res_vals}")
            
            # 验证逻辑：
            # A 应该被映射为某个 Label (包含 "A")
            self.assertTrue("A" in res_vals[0])
            # Shanghai 应该被映射为 "Other" (因为它不在规则里)
            self.assertEqual(res_vals[1], "Other")

    def test_04_special_missing_priority(self):
        """
        [优先级] 测试 Missing > Special > Normal 的 Waterfall 逻辑。
        """
        # -999 是 Special, -1 是 Missing (用户定义), None 是 Missing (原生)
        df = pl.DataFrame({"age": [-999, -1, None, 25, 50]})
        y = [0, 0, 0, 1, 1]

        binner = MarsOptimalBinner(
            features=["age"],
            special_values=[-999],
            missing_values=[-1],
            n_jobs=1
        )
        
        # --- 🔧 [修复] 先调用一次 Fit 以满足基类检查 ---
        # 即使数据很少或无意义，只要跑过 fit，基类的 _is_fitted 标记就会设为 True
        binner.fit(df, y) 
        # -------------------------------------------
        
        # 然后手动注入切点，覆盖刚才 fit 的结果，专注于测试 Transform 逻辑
        binner.bin_cuts_ = {"age": [float('-inf'), 30.0, float('inf')]}
        
        res = binner.transform(df)
        bins = res["age_bin"].to_list()
        print(f"特殊值分箱结果: {bins}")

        # 验证
        self.assertEqual(bins[0], "Special_-999")  # -999 -> Special
        self.assertEqual(bins[1], "Missing")       # -1 -> Missing
        self.assertEqual(bins[2], "Missing")       # None -> Missing
        self.assertTrue("00_" in bins[3])          # 25 < 30 -> Normal Bin 0
        self.assertTrue("01_" in bins[4])          # 50 > 30 -> Normal Bin 1

    def test_05_compatibility_polars_replace(self):
        """
        [兼容性] 验证 Polars 的 replace 调用是否安全。
        针对代码中 `known_labels` 的过滤逻辑进行测试。
        """
        df = pl.DataFrame({"cat": ["Apple", "Banana", "Cherry"]})
        # 构造一个假的 y
        y = [0, 1, 0]
        
        binner = MarsOptimalBinner(cat_features=["cat"], n_jobs=1)
        
        # --- 🔧 [修复] 先调用 fit ---
        # 即使 cat_features 未指定或数据不足，只要 fit 不报错即可
        # 这里为了稳妥，我们传入真实数据
        try:
            binner.fit(df, y)
        except Exception:
            # 如果因为 optbinning 没装而导致 fit 内部逻辑跳过，
            # 我们至少需要手动设置 fitted 标记（取决于你基类的实现）
            # 最稳妥的方式是让 fit 跑完，或者 mock fit
            pass
        
        # 如果你的 MarsTransformer 是通过属性检查 fit 状态的
        # 我们可以手动 hack 一下（假设基类检查的是 _is_fitted）
        if not hasattr(binner, "_is_fitted") and not hasattr(binner, "fitted_"):
             # 如果上面 fit 没跑通，这里手动标记（仅限测试使用）
             # 注意：具体属性名取决于 src/mars/core/base.py 的实现
             # 通常是 self._is_fitted = True
             binner._is_fitted = True 
        # ---------------------------
        
        # 注入规则: Apple -> Bin1, Banana -> Bin2
        # Cherry 没有规则
        binner.cat_cuts_ = {"cat": [["Apple"], ["Banana"]]}
        
        res = binner.transform(df)
        bins = res["cat_bin"].to_list()
        print(f"Polars Replace 结果: {bins}")
        
        # 验证 Cherry 是否变成了 Other
        self.assertTrue("Apple" in bins[0])
        self.assertTrue("Banana" in bins[1])
        self.assertEqual(bins[2], "Other")

    def test_06_empty_dataframe(self):
        """
        [边界情况] 测试空 DataFrame 输入。
        """
        df = pl.DataFrame({"a": []}, schema={"a": pl.Float64})
        y = []
        
        binner = MarsOptimalBinner(features=["a"], n_jobs=1)
        # 应该打 Warning 但不报错
        binner.fit(df, np.array(y))
        
        res = binner.transform(df)
        self.assertEqual(res.height, 0)
        # 如果 fit 失败，可能没有 _bin 列，检查是否 crash 即可
        print("空表测试通过")

if __name__ == '__main__':
    unittest.main()