import sys

import torch
import triton

from flag_gems.ops.rms_norm import (
    _DW_TARGET_LAYOUT,
    rms_norm_grad_dw_kernel,
    rms_norm_grad_dw_kernel_tle,
)

# (name, M, N, ROW_BLOCK_SIZE, COL_BLOCK_SIZE, num_warps)
CANDIDATES = [
    ("case1", 1024, 4096, 16, 256, 4),
    ("case2", 2048, 4096, 128, 512, 8),
    ("case3", 4096, 4096, 16, 256, 8),
    ("case4", 4096, 4096, 16, 512, 16),
    ("case5", 4096, 4096, 32, 512, 16),
    ("case6", 4096, 4096, 32, 256, 8),
    ("case7", 4096, 4096, 16, 128, 4),
]

_CASE_CFG = {name: (M, N, ROW, COL, nw) for name, M, N, ROW, COL, nw in CANDIDATES}
_CASE_NAMES = [c[0] for c in CANDIDATES]

_INTEGRATED_ROW = 16
_INTEGRATED_COL = 256
_INTEGRATED_NUM_WARPS = 4


def build_layout(row_block_size, col_block_size, num_warps):
    import triton.experimental.tle.language as tle

    denom = 32 * num_warps
    assert (
        col_block_size % denom == 0
    ), f"col_block_size={col_block_size} not divisible by 32*num_warps={denom}"
    size_per_thread_col = col_block_size // denom
    return tle.gpu.BlockEncoding(
        size_per_thread=[row_block_size, size_per_thread_col],
        threads_per_warp=[1, 32],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )


def make_inputs(M, N, dtype):
    X = torch.randn(M, N, dtype=dtype, device="cuda")
    DY = torch.randn(M, N, dtype=dtype, device="cuda")
    INV_RMS = torch.rand(M, dtype=torch.float32, device="cuda") + 0.5
    return X, DY, INV_RMS


def run_one(
    kernel_fn,
    M,
    N,
    ROW_BLOCK_SIZE,
    COL_BLOCK_SIZE,
    num_warps,
    dtype,
    extra_kwargs=None,
    warmup=25,
    rep=100,
):
    extra_kwargs = extra_kwargs or {}
    torch.manual_seed(0)
    X, DY, INV_RMS = make_inputs(M, N, dtype)
    row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
    DW = torch.empty(row_block_num, N, dtype=torch.float32, device="cuda")
    grid = (row_block_num, triton.cdiv(N, COL_BLOCK_SIZE))

    def launch():
        kernel_fn[grid](
            X,
            DY,
            INV_RMS,
            DW,
            N,
            1,
            N,
            1,
            M,
            N,
            ROW_BLOCK_SIZE,
            COL_BLOCK_SIZE,
            num_warps=num_warps,
            **extra_kwargs,
        )

    launch()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(launch, warmup=warmup, rep=rep)
    return ms


def benchmark_case(name, dtype):
    M, N, ROW_BLOCK_SIZE, COL_BLOCK_SIZE, num_warps = _CASE_CFG[name]

    ms_base = run_one(
        rms_norm_grad_dw_kernel,
        M,
        N,
        ROW_BLOCK_SIZE,
        COL_BLOCK_SIZE,
        num_warps,
        dtype,
    )

    is_integrated_config = (
        ROW_BLOCK_SIZE == _INTEGRATED_ROW
        and COL_BLOCK_SIZE == _INTEGRATED_COL
        and num_warps == _INTEGRATED_NUM_WARPS
    )
    layout = (
        _DW_TARGET_LAYOUT
        if is_integrated_config
        else build_layout(ROW_BLOCK_SIZE, COL_BLOCK_SIZE, num_warps)
    )

    ms_set_layout = run_one(
        rms_norm_grad_dw_kernel_tle,
        M,
        N,
        ROW_BLOCK_SIZE,
        COL_BLOCK_SIZE,
        num_warps,
        dtype,
        extra_kwargs={"TARGET_LAYOUT": layout},
    )

    return ms_base, ms_set_layout, ms_base / ms_set_layout


def main():
    for dtype_name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        print(f"\n=== dtype={dtype_name} ===")
        header = (
            f"{'case':6s} {'shape(M,N)':>15} {'ROW':>5} {'COL':>5} {'nw':>3}  "
            f"{'baseline(ms)':>13} {'set_layout(ms)':>15} {'speedup':>9}"
        )
        print(header)
        print("-" * len(header))
        for name in _CASE_NAMES:
            M, N, ROW, COL, nw = _CASE_CFG[name]
            ms_base, ms_set_layout, speedup = benchmark_case(name, dtype)
            print(
                f"{name:6s} {f'{M}x{N}':>15} {ROW:>5} {COL:>5} {nw:>3}  "
                f"{ms_base:>13.4f} {ms_set_layout:>15.4f} {speedup:>8.3f}x"
            )

    print("\n=== end-to-end rms_norm forward+backward ===")
    bench_end_to_end()


def bench_end_to_end():
    rn_mod = sys.modules["flag_gems.ops.rms_norm"]

    shapes = [(1024, 4096), (4096, 4096)]
    header = f"{'shape(M,N)':>15} {'dtype':>8} {'baseline(ms)':>13} {'tle(ms)':>10} {'speedup':>9}"
    print(header)
    print("-" * len(header))

    for M, N in shapes:
        for dtype_name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
            torch.manual_seed(0)
            x = torch.randn(M, N, dtype=dtype, device="cuda", requires_grad=True)
            weight = torch.randn(N, dtype=dtype, device="cuda", requires_grad=True)
            dy = torch.randn(M, N, dtype=dtype, device="cuda")

            y, inv_rms = rn_mod.rms_norm_forward(x, [N], weight)

            def run_backward(force_tle):
                orig = rn_mod._dw_tle_available
                rn_mod._dw_tle_available = lambda _x: force_tle
                try:
                    rn_mod.rms_norm_backward(dy, x, inv_rms, [N], weight)
                    torch.cuda.synchronize()
                finally:
                    rn_mod._dw_tle_available = orig

            ms_base = triton.testing.do_bench(
                lambda: run_backward(False), warmup=25, rep=100
            )
            ms_tle = triton.testing.do_bench(
                lambda: run_backward(True), warmup=25, rep=100
            )
            speedup = ms_base / ms_tle if ms_tle > 0 else float("nan")
            print(
                f"{f'{M}x{N}':>15} {dtype_name:>8} {ms_base:>13.4f} "
                f"{ms_tle:>10.4f} {speedup:>8.3f}x"
            )


if __name__ == "__main__":
    main()
