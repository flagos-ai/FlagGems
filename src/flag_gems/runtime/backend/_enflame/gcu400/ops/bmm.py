# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# No native-kernel delegation: every bmm / bmm_out call runs the Triton kernel
# below, at every workload size.
#
# The tile table that drives ``_select_gemm_config`` was measured on L600
# (bench kernel mode, do_bench median w1000/r100, cold L2). It reads as a ratio
# of the Triton kernel to a native vendor GEMM, and (128,256,256) is the tile
# sweet spot (matching the matrix_power job's 50+ combo sweep):
#   (2,384^3)   Triton 1.08-1.19x   (2,512^3)  Triton 1.12-1.17x
#   (2,640^3)   Triton 0.96x        (2,768^3)  Triton 0.93x
#   (4,512^3)   Triton 0.64x        (16,384^3) Triton 0.57x
#   (16,1024^3) Triton 0.51x        (16,4096^3) Triton 0.55x
# The right column is where a saturated native vendor GEMM pipeline would win.
# This backend does not take that shortcut: the Triton kernel runs those shapes
# too, and the table only explains the tile choice.
# ---------------------------------------------------------------------------


# Hardware DSM (distributed shared memory) budget of one GCU400 cluster. The
# staged A/B tiles of every software-pipeline stage live there, so a tile table
# that ignores this limit fails to compile ("out of resource: dsm memory").
#
# The compiler-level budget is 512KiB and every tier below is sized against it,
# which keeps the legacy tile table byte-for-byte unchanged.
_DSM_LIMIT_BYTES = 512 * 1024

# Real hardware DSM budget of one cluster as reported by the compile error
# ("Hardware limit: 917504"). The fp32 accumulator of the epilogue is counted in
# that pool as well, which is what makes a 512x256 output tile the largest
# affordable 16-bit tile (512*256*4 = 512KiB of accumulator plus the staged A/B
# tiles). Only the 16-bit wide-tile tier below uses the real budget.
_DSM_HARD_LIMIT_BYTES = 917504


def _pow2_ceil(value):
    return 1 << max(0, int(value) - 1).bit_length()


def _clamp_tile(dim, tile, floor=64):
    # Never request a tile far larger than the problem dimension. Oversized
    # tiles both waste DSM and overflow it: (B, 64, 1, 4096) used to request a
    # 256-wide N tile for a single output column and failed to compile.
    #
    # The floor of 64 is a correctness guard, not just padding: on GCU400 the
    # unmasked (EVEN) store/download path silently computes wrong values for
    # BLOCK_M in {16, 32} combined with BLOCK_N >= 256, e.g. (2,32,512,4096)
    # with (32,256,256) returns relerr ~2.7 while (32,128,256) and
    # (64,256,256) are exact. Keeping BLOCK_M >= 64 avoids that region.
    if dim >= tile:
        return tile
    return max(floor, min(tile, _pow2_ceil(dim)))


def _fit_num_stages(
    block_m, block_n, block_k, elem_bytes, num_stages, limit=_DSM_LIMIT_BYTES
):
    # Shrink the software pipeline until the staged tiles fit the DSM budget.
    while num_stages > 1:
        staged = (block_m * block_k + block_k * block_n) * elem_bytes * num_stages
        if staged <= limit:
            break
        num_stages -= 1
    return num_stages


def _fit_num_stages_with_acc(
    block_m, block_n, block_k, elem_bytes, num_stages, max_stages=4
):
    # Same as _fit_num_stages but for the wide-tile tier, where the fp32
    # accumulator of the epilogue occupies the same pool as the staged A/B
    # tiles. Sizing only the staged tiles overflows: (512,256,64,ns=2) reports
    # "Required: 1703936" once the 512KiB accumulator is added in.
    acc_bytes = block_m * block_n * 4
    for stages in range(min(max_stages, num_stages), 0, -1):
        staged = (block_m * block_k + block_k * block_n) * elem_bytes * stages
        if acc_bytes + staged <= _DSM_HARD_LIMIT_BYTES:
            return stages
    return 1


# Candidate 16-bit tiles for the band above the tiny tier, ordered by
# descending output area. Throughput of the GCU400 GEMM core scales with the
# tile area (measured on (2,4096^2): 64x64 -> 5.6, 128x128 -> 11.7,
# 256x128/128x256 -> 15.5, 256x256 -> 22.7, 512x256 -> 30.0 TFLOPS), so the
# largest tile that still keeps the grid busy wins.
_16BIT_TILE_LADDER = (
    (512, 256, 64, 2),
    (256, 256, 64, 4),
    (256, 128, 128, 4),
    (128, 256, 64, 4),
    (128, 128, 64, 4),
    (64, 64, 32, 4),
)

# One GCU400 "wave" of output tiles. Below this the wide tile is dropped one
# ladder step at a time until the grid can fill the clusters again.
_MIN_PROGRAMS_PER_TILE = 24


def _select_16bit_tiles(Batch, M, N, K):
    selected = None
    for block_m, block_n, block_k, num_warps in _16BIT_TILE_LADDER:
        block_m = _clamp_tile(M, block_m)
        block_n = _clamp_tile(N, block_n)
        block_k = _clamp_tile(K, block_k, floor=16)
        selected = (block_m, block_n, block_k, num_warps)
        programs = Batch * triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
        if programs >= _MIN_PROGRAMS_PER_TILE:
            break
    return selected


def _select_gemm_config(Batch, M, N, K, dtype):
    # Single source of truth for the GEMM tile table, shared by bmm / bmm_out /
    # baddbmm / baddbmm_out / baddbmm_ (all of them compute the same batched
    # GEMM core; baddbmm only adds a stride-based bias epilogue).
    #
    # Tiles were swept per shape tier on GCU400 (4-1, 2 clusters x 12 SIPs):
    #   * (2,384^3)  fp16/bf16: (128,128,512,w4,ns2) 13.4us vs (128,128,256) 14.7us
    #     vs the previous (256,128,128) 16.9us and (64,128,128,w1) 25.3us.
    #     A 512-deep K block keeps the inner loop at one masked iteration.
    #   * the whole <= 512^3 tier (the Triton-only band) was re-swept later:
    #     a 256-wide M tile with a 128-deep K block beats the old 512-deep K
    #     block, because the deep K block leaves too few output tiles to fill
    #     the clusters while the shallow one actually gets double buffered.
    #     (2,384^3) fp16/bf16 1.08x -> 1.24x, fp32 0.97x -> 1.13x of native;
    #     (2,128^3) clamps down to (128,128,128), still the sweep winner.
    #   * saturated shapes (>= 4096^2, batch >= 8) keep the wide fp16 tile.
    #     Re-measured on 2026-11-21 on board A084P50610002 with an interleaved
    #     same-process A/B (min of 3 rounds, real torch.bmm baseline): the wide
    #     (128,256,*) tile is 0.52-0.55x of native at (2,4096^2), (16,1024^2),
    #     (16,2048^2) and (16,4096^2), while the square (256,256,128) tile that
    #     was tried in between lands at 0.33-0.35x at every one of those shapes
    #     (256x128x128 0.39-0.42x, 128x128x128 0.25-0.27x).  The wide tile is
    #     therefore kept for the 16-bit large band on this board.
    #
    # DSM note: the staged A/B tiles must fit _DSM_LIMIT_BYTES for two pipeline
    # stages, otherwise _fit_num_stages collapses num_stages to 1 and the K loop
    # loses its double buffer entirely.  (128,128,512) costs 128KB per stage in
    # fp32, so fp32 never uses a 512-deep K block; the tier below keeps
    # (256,128,128) = 48KB per stage, which double buffers comfortably.
    #   * superseded on 2026-11-22: the (128,256,*) tiles above were limited by
    #     the 512KiB "compiler" DSM budget, but a cluster really owns 917504 B.
    #     Re-sweeping with the real budget shows the 16-bit core scales with
    #     tile area up to (512,256,64), which is ~2x the old wide tile at every
    #     saturated shape, so the two former 16-bit branches collapse into
    #     _select_16bit_tiles below.
    num_stages = 3
    if K < 128:
        block_m, block_n, block_k, num_warps, num_stages = 64, 64, 32, 4, 2
    elif M <= 512 and N <= 512 and K <= 512:
        if dtype == torch.float32:
            block_m, block_n, block_k, num_warps, num_stages = 256, 128, 128, 4, 2
        else:
            block_m, block_n, block_k, num_warps, num_stages = 256, 256, 128, 4, 2
    elif dtype == torch.float32:
        block_m, block_n, block_k, num_warps = 128, 256, 128, 2
    else:
        # 16-bit band above the tiny tier. The wide 512x256 output tile is the
        # largest one the DSM pool can hold together with its fp32 accumulator
        # and it is worth ~2x over the previous (128,256,*) tile at every
        # saturated shape, so it replaces both former 16-bit tiers (the
        # "saturated" branch and the generic branch).
        block_m, block_n, block_k, num_warps = _select_16bit_tiles(Batch, M, N, K)
        elem_bytes = 2
        num_stages = _fit_num_stages_with_acc(
            block_m, block_n, block_k, elem_bytes, num_stages
        )
        even = M % block_m == 0 and N % block_n == 0 and K % block_k == 0
        return block_m, block_n, block_k, num_warps, num_stages, even

    block_m = _clamp_tile(M, block_m)
    block_n = _clamp_tile(N, block_n)
    block_k = _clamp_tile(K, block_k, floor=16)
    if block_n > 256 and block_m < 128:
        # Keep the wide-N tiles on tall enough blocks; see _clamp_tile notes.
        block_n = 256
    elem_bytes = 4 if dtype in (torch.float32, torch.float64) else 2
    num_stages = _fit_num_stages(block_m, block_n, block_k, elem_bytes, num_stages)

    # Exact decision: only when all three dimensions are divisible by the actually selected tiles are load/store
    # free of out-of-bounds access, in which case an unmasked fast path is compiled; otherwise use the masked generic
    # path.
    even = M % block_m == 0 and N % block_n == 0 and K % block_k == 0
    return block_m, block_n, block_k, num_warps, num_stages, even


# Conservative last-resort tile used when the primary config exceeds the DSM
# budget at compile time (keeps the operator usable instead of raising).
_SAFE_FALLBACK_CONFIG = (64, 64, 64, 4, 2)


def _select_bmm_config(Batch, M, N, K, dtype):
    return _select_gemm_config(Batch, M, N, K, dtype)


@libentry()
@triton.jit
def bmm_kernel(
    A,
    B,
    C,
    Batch,
    M,
    N,
    K,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = grid_m * grid_n

    pid_b = pid // tiles_per_batch
    pid_mn = pid % tiles_per_batch

    width = GROUP_M * grid_n
    group_id = pid_mn // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_n = (pid_mn % width) // group_size

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        A
        + pid_b * stride_ab
        + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    )
    b_ptrs = (
        B
        + pid_b * stride_bb
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    # EVEN is a compile-time constant, so the compiler statically trims it to a single loop,
    # avoiding a runtime branch interrupting the tl.dot pipeline (a runtime branch degrades dot to scalar compute).
    if EVEN:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = (
        C
        + pid_b * stride_cb
        + offs_cm[:, None] * stride_cm
        + offs_cn[None, :] * stride_cn
    )
    c = acc.to(C.dtype.element_ty)
    if EVEN:
        tl.store(c_ptrs, c)
    else:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def _launch_bmm(A, B, out, Batch, M, N, K):
    # Centralize config dispatch and kernel launch, shared by bmm / bmm_out.
    block_m, block_n, block_k, num_warps, num_stages, even = _select_bmm_config(
        Batch, M, N, K, A.dtype
    )
    _run_bmm_kernel(
        A,
        B,
        out,
        Batch,
        M,
        N,
        K,
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        even,
    )


def _run_bmm_kernel(
    A, B, out, Batch, M, N, K, block_m, block_n, block_k, num_warps, num_stages, even
):
    grid = (Batch * triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    launch = dict(
        dot_out_dtype=tl.float32,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        EVEN=even,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    args = (
        A,
        B,
        out,
        Batch,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    try:
        bmm_kernel[grid](*args, **launch)
    except triton.runtime.errors.OutOfResources:
        # The selected tiles do not fit the hardware DSM budget for this shape;
        # retry once with the conservative tile instead of failing the operator.
        block_m, block_n, block_k, num_warps, num_stages = _SAFE_FALLBACK_CONFIG
        block_m = _clamp_tile(M, block_m)
        block_n = _clamp_tile(N, block_n)
        block_k = _clamp_tile(K, block_k, floor=16)
        even = M % block_m == 0 and N % block_n == 0 and K % block_k == 0
        grid = (Batch * triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
        bmm_kernel[grid](
            *args,
            dot_out_dtype=tl.float32,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            EVEN=even,
            num_warps=num_warps,
            num_stages=num_stages,
        )


def _as_gemm_friendly(A, B):
    # The kernel indexes A as [m, k] and B as [k, n] using explicit strides, so
    # any layout with a unit stride along the *inner* dot axis is fed directly.
    # Forcing .contiguous() on a transposed B used to materialize a copy through
    # an extra kernel: (2,512^3) measured 24.1us copy+kernel vs 19.5us strided
    # (native 25.1us). Only layouts with no unit stride on either axis (e.g. a
    # sliced/broadcast view) still get a copy.
    if A.stride(2) != 1 and not A.is_contiguous():
        A = A.contiguous()
    if B.stride(2) != 1 and B.stride(1) != 1 and not B.is_contiguous():
        B = B.contiguous()
    return A, B


def bmm(A, B):
    logger.debug("GEMS_ENFLAME BMM")
    assert A.shape[0] == B.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    Batch, M, K = A.shape
    _, _, N = B.shape
    A, B = _as_gemm_friendly(A, B)
    out = torch.empty((Batch, M, N), dtype=A.dtype, device=A.device)
    _launch_bmm(A, B, out, Batch, M, N, K)
    return out


def bmm_out(A, B, out):
    logger.debug("GEMS_ENFLAME BMM_OUT")
    assert A.shape[0] == B.shape[0] == out.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    Batch, M, K = A.shape
    _, _, N = B.shape
    A, B = _as_gemm_friendly(A, B)
    _launch_bmm(A, B, out, Batch, M, N, K)
    return out
