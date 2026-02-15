import time

import torch
from tabulate import tabulate

import flag_gems

def benchmark_asinh(shape, dtype, iters=100):
    # 初始化数据
    x = torch.randn(shape, device="cuda", dtype=dtype)
    
    # Warm up: 必须预热，排除 Triton 首次编译耗时
    for _ in range(10):
        with flag_gems.use_gems():
            _ = torch.asinh(x)
    torch.cuda.synchronize()

    # 测试 PyTorch 原生性能
    start_ptr = time.time()
    for _ in range(iters):
        _ = torch.asinh(x)
    torch.cuda.synchronize()
    end_ptr = time.time()
    t_ptr = (end_ptr - start_ptr) / iters * 1000  # 毫秒

    # 测试 FlagGems 性能
    start_gems = time.time()
    with flag_gems.use_gems():
        for _ in range(iters):
            _ = torch.asinh(x)
    torch.cuda.synchronize()
    end_gems = time.time()
    t_gems = (end_gems - start_gems) / iters * 1000  # 毫秒

    return t_ptr, t_gems

def run_performance_test():
    # 严格按照 4.1.4 要求的规模
    scales = [
        ("Small", (1, 1)),
        ("Small", (8, 8)),
        ("Normal", (64, 64)),
        ("Normal", (256, 256)),
        ("Large", (1024, 1024)),
        ("Large", (4096, 4096)),
    ]
    
    # 按照 4.1.3 要求的数据类型
    dtypes = [torch.float16, torch.float32, torch.bfloat16]
    
    results = []
    print("🚀 Starting Performance Benchmark for 'asinh'...")
    
    for tag, shape in scales:
        for dtype in dtypes:
            t_ptr, t_gems = benchmark_asinh(shape, dtype)
            speedup = t_ptr / t_gems
            results.append([
                tag, 
                str(shape), 
                str(dtype).replace("torch.", ""), 
                f"{t_ptr:.4f}", 
                f"{t_gems:.4f}", 
                f"{speedup:.2f}x"
            ])
            
    headers = ["Scale", "Shape", "Dtype", "PyTorch (ms)", "FlagGems (ms)", "Speedup"]
    print("\n" + tabulate(results, headers=headers, tablefmt="github"))
    
    # 额外提醒：4.1.4 要求提供测例覆盖清单
    print("\n✅ Benchmark completed. Please copy the table above into your PR description.")

if __name__ == "__main__":
    run_performance_test()