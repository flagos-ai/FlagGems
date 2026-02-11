import torch
import pandas as pd
import numpy as np
from flag_gems.ops.log10 import log10, log10_, log10_out
import triton

def run_test():
    results = []
    # 1. 定义维度覆盖: 1D 到 5D
    # 2. 定义规模覆盖: 小、常规、大
    test_cases = [
        {"name": "极小尺寸", "shape": (1, 1), "dim": 2},
        {"name": "小尺寸", "shape": (8, 8), "dim": 2},
        {"name": "常规尺寸", "shape": (256, 256), "dim": 2},
        {"name": "大尺寸", "shape": (1024, 1024), "dim": 2},
        {"name": "超大尺寸", "shape": (4096, 4096), "dim": 2},
        {"name": "1D张量", "shape": (10000,), "dim": 1},
        {"name": "3D批量", "shape": (32, 64, 64), "dim": 3},
        {"name": "4D深度学习常用", "shape": (8, 3, 224, 224), "dim": 4},
        {"name": "5D高维", "shape": (2, 2, 16, 32, 32), "dim": 5},
    ]

    for case in test_cases:
        shape = case["shape"]
        # 准备数据 (避开负数和零，log10 合法定义域为正数)
        x = torch.rand(shape, device="cuda", dtype=torch.float32) * 100 + 0.1
        
        # --- 功能完整性测试: 非原位 ---
        ref = torch.log10(x)
        ms_pt = triton.testing.do_bench(lambda: torch.log10(x))
        
        res_gems = log10(x)
        ms_gems = triton.testing.do_bench(lambda: log10(x))
        
        correct = torch.allclose(res_gems, ref, atol=1.3e-6, rtol=1e-5)
        
        # --- 功能完整性测试: 原位 ---
        x_inplace = x.clone()
        log10_(x_inplace)
        inplace_correct = torch.allclose(x_inplace, ref, atol=1.3e-6, rtol=1e-5)
        
        # --- 功能完整性测试: Out 参数 ---
        out_buffer = torch.empty_like(x)
        log10_out(x, out_buffer)
        out_correct = torch.allclose(out_buffer, ref, atol=1.3e-6, rtol=1e-5)

        results.append({
            "测试场景": case["name"],
            "Shape": str(shape),
            "维数": case["dim"],
            "正确性(Normal/Inplace/Out)": f"{correct}/{inplace_correct}/{out_correct}",
            "PT耗时(ms)": round(ms_pt, 4),
            "Gems耗时(ms)": round(ms_gems, 4),
            "加速比": round(ms_pt / ms_gems, 2)
        })

    # 生成 DataFrame 并打印表格
    df = pd.DataFrame(results)
    print("\n=== FlagGems log10 测例覆盖清单及加速比对比表 ===")
    print(df.to_markdown(index=False))
    df.to_csv("log10_test_report.csv", index=False)

if __name__ == "__main__":
    run_test()