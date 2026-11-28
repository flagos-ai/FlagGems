# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SM90 short-K FP8 GEMM with ordinary copies and ordered K32 products.

One warpgroup loads the complete short-K tile into shared memory. Every K32
product completes before its numeric FP32 scales and ordered accumulation.
This path has no TMA, warp specialization, pipe or process-wide allocator.
"""

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit
def _block_fp8_tle_short(
    A,
    B,
    C,
    As,
    Bs,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ASM: tl.constexpr,
    ASK: tl.constexpr,
    BSN: tl.constexpr,
    BSK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    N_MAJOR: tl.constexpr,
):
    pid = tl.program_id(0)
    if N_MAJOR:
        mi = pid % tl.cdiv(M, BM)
        ni = pid // tl.cdiv(M, BM)
    else:
        mi = pid // tl.cdiv(N, BN)
        ni = pid % tl.cdiv(N, BN)
    rows = mi * BM + tl.arange(0, BM)
    cols = ni * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_smem = tle.gpu.alloc(
        [BM, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem, layout=None
    )
    b_smem = tle.gpu.alloc(
        [BN, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem, layout=None
    )
    tle.gpu.copy(
        A + rows[:, None] * K + rk[None, :],
        a_smem,
        [BM, BK],
        mask=(rows[:, None] < M) & (rk[None, :] < K),
    )
    tle.gpu.copy(
        B + cols[:, None] * K + rk[None, :],
        b_smem,
        [BN, BK],
        mask=(cols[:, None] < N) & (rk[None, :] < K),
    )
    acc = tl.zeros((BN, BM), tl.float32)
    for group in tl.static_range(K // 32):
        av = a_smem.subslice(group * 32, 32, 1)
        bv = b_smem.subslice(group * 32, 32, 1)
        dot = tle.gpu.wgmma(bv, av, trans_b=True)
        dot = tle.gpu.wgmma_wait(0, dot)
        sa = tl.load(As + rows * ASM + group * ASK, rows < M, 0.0)
        sb = tl.load(Bs + cols // 32 * BSN + group * BSK, cols < N, 0.0)
        acc += dot * sb[:, None] * sa[None, :]
    value = tl.trans(acc)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        value,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def block_fp8_matmul_tle_short(
    A,
    B,
    As,
    Bs,
    output_dtype=torch.bfloat16,
    *,
    tile_m=16,
    tile_n=64,
    n_major=False,
    max_registers=128,
    return_kernel=False,
):
    """Generic E4M3FN A[M,K] @ B[N,K].T with numeric FP32 K32 scales.

    K is a positive multiple of32, at most512. The public caller qualifies a
    finite measured schedule; direct calls have no implicit fallback.
    """
    if not (A.is_cuda and A.device == B.device == As.device == Bs.device):
        raise ValueError("all inputs must share a CUDA device")
    if torch.cuda.get_device_capability(A.device) != (9, 0):
        raise ValueError("TLE short-K GEMM is validated on SM90 only")
    if A.dtype != torch.float8_e4m3fn or B.dtype != A.dtype:
        raise TypeError("A and B must use E4M3FN storage")
    if As.dtype != torch.float32 or Bs.dtype != torch.float32:
        raise TypeError("scales must be numeric FP32")
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("output must be BF16 or FP32")
    if A.ndim != 2 or B.ndim != 2 or not A.is_contiguous() or not B.is_contiguous():
        raise ValueError("A and B must be contiguous matrices")
    if A.data_ptr() % 16 or B.data_ptr() % 16:
        raise ValueError("short-K A and B pointers must be 16-byte aligned")
    m, k = A.shape
    n, bk = B.shape
    if k <= 0 or k > 512 or k % 32 or bk != k or n <= 0:
        raise ValueError("K must match B and be a positive multiple of32 at most512")
    if As.shape != (m, k // 32) or Bs.shape != (triton.cdiv(n, 32), k // 32):
        raise ValueError("invalid block32 scale shapes")
    if tile_m < 8 or tile_m & (tile_m - 1) or tile_n < 64 or tile_n & (tile_n - 1):
        raise ValueError("tile_m/tile_n must be powers of two, at least8/64")
    out = torch.empty((m, n), device=A.device, dtype=output_dtype)
    if m == 0:
        return (out, None) if return_kernel else out
    launch = {} if max_registers is None else {"maxnreg": max_registers}
    kernel = _block_fp8_tle_short[(triton.cdiv(m, tile_m) * triton.cdiv(n, tile_n),)](
        A,
        B,
        out,
        As,
        Bs,
        m,
        n,
        k,
        As.stride(0),
        As.stride(1),
        Bs.stride(0),
        Bs.stride(1),
        tile_m,
        tile_n,
        triton.next_power_of_2(k),
        n_major,
        num_warps=4,
        num_stages=1,
        **launch,
    )
    return (out, kernel) if return_kernel else out
