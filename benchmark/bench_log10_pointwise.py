"""
Performance benchmark for log10 (FlagGems · 赛道一 · 初级算子)

计时方法: torch.cuda.Event（GPU 异步安全）
加速比要求: ≥ 0.9（赛事硬性门槛）
"""

from __future__ import annotations

import os
import sys

import flag_gems
import torch

from log10_pointwise_submit import log10

# ---------------------------------------------------------------------------
# 尝试使用 FlagGems 官方 benchmark 框架
# submit 目录在 ~/workspace/submit，FlagGems 根在 ~/workspace/FlagGems
# ---------------------------------------------------------------------------
_FLAGGEMS_ROOT = os.path.join(os.path.dirname(__file__), "..", "FlagGems")
_FLAGGEMS_BENCH = os.path.join(_FLAGGEMS_ROOT, "benchmark")
if os.path.isdir(_FLAGGEMS_BENCH) and _FLAGGEMS_ROOT not in sys.path:
    sys.path.insert(0, _FLAGGEMS_ROOT)

try:
    from benchmark.performance_utils import GenericBenchmark, unary_input_fn

    _HAS_OFFICIAL_BENCH = True
except ImportError:
    _HAS_OFFICIAL_BENCH = False

# ---------------------------------------------------------------------------
# 测试配置
# ---------------------------------------------------------------------------
BENCHMARK_SHAPES = [
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024 * 1024,),
    (64, 64, 64),
    (2, 1024, 1024),
]

FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


# ===========================================================================
# CUDA Event 精确计时（median latency）
# ===========================================================================
def _bench(op, x: torch.Tensor, warmup: int = 50, rep: int = 200) -> float:
    """返回中位延迟（ms）。必须用 CUDA Event，time.time() 对 GPU 异步操作不准确。"""
    for _ in range(warmup):
        op(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    lats = []
    for _ in range(rep):
        start.record()
        op(x)
        end.record()
        torch.cuda.synchronize()
        lats.append(start.elapsed_time(end))

    lats.sort()
    return lats[len(lats) // 2]


# ===========================================================================
# 官方框架路径
# ===========================================================================
def run_official():
    bench = GenericBenchmark(
        op_name="log10",
        torch_op=torch.log10,
        input_fn=unary_input_fn,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(log10)
    bench.run()


# ===========================================================================
# 独立 benchmark（官方框架不可用时的 fallback）
# ===========================================================================
def run_standalone():
    W = 72
    print("=" * W)
    print("  log10 Benchmark — Iluvatar BI-V150 / corex-4.4.0")
    print("  计时: torch.cuda.Event  |  统计: 200 次中位延迟")
    print("=" * W)
    print(
        f"  {'Shape':<20} {'Dtype':<12} {'PyTorch':>10}  {'FlagGems':>10}  {'Speedup':>8}  Status"
    )
    print("-" * W)

    speedups = []
    for dtype in FLOAT_DTYPES:
        for shape in BENCHMARK_SHAPES:
            x = (torch.rand(shape, dtype=torch.float32, device="cuda").abs() + 1e-3).to(
                dtype
            )

            t_ref = _bench(torch.log10, x)
            t_gems = _bench(log10, x)
            sp = t_ref / t_gems
            speedups.append(sp)
            status = "✅" if sp >= 0.9 else "❌"

            shape_s = "×".join(map(str, shape))
            print(
                f"  {shape_s:<20} {str(dtype):<12} "
                f"{t_ref:>9.4f}ms  {t_gems:>9.4f}ms  {sp:>7.3f}x  {status}"
            )
        print()

    avg = sum(speedups) / len(speedups)
    mn = min(speedups)
    print("=" * W)
    print(f"  平均加速比: {avg:.3f}x")
    print(f"  最低加速比: {mn:.3f}x")
    print(f"  整体结果  : {'✅ PASS (≥0.9)' if mn >= 0.9 else '❌ FAIL (<0.9)'}")
    print("=" * W)

    # 内存带宽利用率（float32 1024×1024）
    x32 = torch.rand(1024, 1024, dtype=torch.float32, device="cuda").abs() + 1e-3
    t_ms = _bench(log10, x32)
    # 读 N 元素 + 写 N 元素
    gbps = 2 * x32.numel() * x32.element_size() / (t_ms * 1e-3) / 1e9
    print(f"\n  内存带宽利用率 (fp32, 1024×1024): {gbps:.1f} GB/s  [{t_ms:.4f} ms]")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，请检查 GPU 环境。")
        sys.exit(1)

    print(f"Device   : {torch.cuda.get_device_name(0)}")
    print(f"FlagGems : {flag_gems.__version__}")
    print(
        f"模式     : {'官方 GenericBenchmark' if _HAS_OFFICIAL_BENCH else '独立 CUDA-Event benchmark'}\n"
    )

    if _HAS_OFFICIAL_BENCH:
        run_official()
    else:
        run_standalone()
