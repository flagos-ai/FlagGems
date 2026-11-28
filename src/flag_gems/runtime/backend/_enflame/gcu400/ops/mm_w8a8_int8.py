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

"""Enflame GCU400 prequantized W8A8 INT8 GEMM.

``out = (A @ B) * scale_a * scale_b (+ bias)`` with INT8 codes on both sides.
A is ``[M, K]``, B is ``[K, N]``; scales are FP32 and either scalar or
per-row A / per-column B; the optional bias is a contiguous ``[N]`` vector.
The INT8 product is accumulated in INT32 (INT64 when K is long enough to
overflow INT32) and the epilogue is applied once in FP32 before the single
conversion to ``out_dtype``, so the whole chain stays fused in one kernel.
"""

import logging
import os

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_FLOATS = (torch.float16, torch.bfloat16, torch.float32)

# 127 * 127 * K must stay below INT32_MAX; below this bound an INT32
# accumulator cannot overflow, above it the INT64 accumulator is required.
_MAX_K_WITHOUT_INT32_OVERFLOW = 131071
_GROUP_M = 8
# GEMV (reduction-shaped) only wins on tiny problems of "single row + extremely narrow N + small K": at that point the
# matrix path
# has only single-digit CTAs, and fixed launch cost accounts for more than half. Measured on GCU400 (batch10):
#   1x1024x128 : gemv 0.0102ms vs tile 0.0111ms  -> keep gemv
#   1x2048x256 : gemv 0.0126ms vs tile 0.0104ms  -> tile
#   1x4096x64  : gemv 0.0155ms vs tile 0.0105ms  -> tile
#   8x4096x128 : gemv 0.0755ms vs tile 0.0125ms  -> tile(6x)
# Therefore the threshold is tightened to M==1, N<=1024, K<=256.
_MAX_GEMV_M = 1
_GEMV_MAX_N = 1024
_MAX_GEMV_K = 256

# GCU400 single-dimension grid limit; exceeding it causes OutOfResources at launch.
_MAX_GRID_X = 65535


@libentry()
@triton.jit
def _w8a8_int8_gemm_kernel(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    A_SCALAR: tl.constexpr,
    B_SCALAR: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACC_INT64: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # Grouped tile order: consecutive program ids stay inside one M band so the
    # B tiles they touch are reused by the L2 while the A rows stay in cache.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    first_m = group_id * GROUP_M
    group_size = min(grid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Strides are passed in, so row-major, column-major and broadcast (stride 0)
    # inputs are all handled without a host-side repack.
    a_ptrs = (
        a_ptr
        + offs_m[:, None].to(tl.int64) * stride_am
        + offs_k[None, :].to(tl.int64) * stride_ak
    )
    b_ptrs = (
        b_ptr
        + offs_k[:, None].to(tl.int64) * stride_bk
        + offs_n[None, :].to(tl.int64) * stride_bn
    )

    if ACC_INT64:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int64)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        if EVEN_M and EVEN_K:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0,
            )
        if EVEN_N and EVEN_K:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(
                b_ptrs,
                mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_remaining),
                other=0,
            )
        if ACC_INT64:
            acc += tl.dot(a, b, out_dtype=tl.int32).to(tl.int64)
        else:
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Dequantize once, in FP32: dot * scale_a[row] * scale_b[col] + bias[col].
    if A_SCALAR:
        sa = tl.load(sa_ptr) + tl.zeros((BLOCK_M,), dtype=tl.float32)
    else:
        sa = tl.load(sa_ptr + offs_m, mask=offs_m < M, other=0.0)
    if B_SCALAR:
        sb = tl.load(sb_ptr) + tl.zeros((BLOCK_N,), dtype=tl.float32)
    else:
        sb = tl.load(sb_ptr + offs_n, mask=offs_n < N, other=0.0)

    if HAS_BIAS:
        # If the two multiplies and one add are written as `scaled * sb + bias`, the GCU backend fuses the latter
        # multiply-add into
        # a single-rounding FMA (t*sb+bias rounds only once), which differs from the FP32
        # reference convention of "multiply then add each rounded" by 1 ULP (measured 22/91 elements mismatched;
        # fp16/bf16 outputs occasionally amplified to a code-value difference).
        # Use fma(..., 0) to force one rounding of the product first, then add the bias, preserving double-rounding
        # semantics.
        scaled = acc.to(tl.float32) * sa[:, None]
        product = tl.math.fma(
            scaled, sb[None, :], tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        )
        value = (
            product
            + tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[
                None, :
            ]
        )
    else:
        value = acc.to(tl.float32) * sa[:, None] * sb[None, :]

    # The output row stride is given by the caller: when chunking along N, out is a column slice of the original matrix,
    # so the row stride is still the full N, not the kernel-local N.
    out_ptrs = (
        out_ptr
        + offs_m[:, None].to(tl.int64) * stride_on
        + offs_n[None, :].to(tl.int64)
    )
    if EVEN_M and EVEN_N:
        tl.store(out_ptrs, value.to(out_ptr.dtype.element_ty))
    else:
        tl.store(
            out_ptrs,
            value.to(out_ptr.dtype.element_ty),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


@libentry()
@triton.jit
def _w8a8_int8_bias_fill_kernel(
    bias_ptr,
    out_ptr,
    M,
    N,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """K == 0: the INT8 product is zero, so out is just the broadcast bias."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N
    if HAS_BIAS:
        value = tl.load(bias_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
    else:
        value = tl.zeros((BLOCK_N,), dtype=tl.float32)
    out_ptrs = out_ptr + pid_m.to(tl.int64) * N + offs_n.to(tl.int64)
    tl.store(out_ptrs, value.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _w8a8_int8_gemv_kernel(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_SCALAR: tl.constexpr,
    B_SCALAR: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Thin-M path: one program per (row, N tile), reduction along K.

    With M <= 8 the MMA tile wastes >= 87% of its rows, so the tile-shaped
    kernel is bound by wasted matrix work instead of HBM. This variant streams
    B once, multiplies by the broadcast A row and reduces along K, which turns
    the same problem into a pure HBM-bound scan (B is the whole working set).
    """
    # 2D launch (N tiles, M rows): an in-kernel `pid % grid_n` division by the
    # runtime grid width produced wrong N tiles on GCU400, so the mapping is
    # left to the hardware grid instead of integer div/mod.
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    b_ptrs = (
        b_ptr
        + offs_n[:, None].to(tl.int64) * stride_bn
        + offs_k[None, :].to(tl.int64) * stride_bk
    )
    # A row pointers advance along K exactly like the B tile pointers.
    a_ptrs = a_ptr + pid_m.to(tl.int64) * stride_am + offs_k.to(tl.int64) * stride_ak
    # FP32 reduction: INT32 vector reductions are emulated on GCU400 and cost
    # ~5x more (113 vs 34 us for the same tile). Products are <= 127*127 and the
    # running sum stays exactly representable in FP32 while K <= 1040; beyond
    # that the error is a few ULP, far below the output tolerance.
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        k_mask = offs_k < k_remaining
        a_row = tl.load(a_ptrs, mask=k_mask, other=0).to(tl.float32)
        b_tile = tl.load(
            b_ptrs,
            mask=(offs_n[:, None] < N) & k_mask[None, :],
            other=0,
        ).to(tl.float32)
        acc += tl.sum(b_tile * a_row[None, :], axis=1)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    n_mask = offs_n < N
    if A_SCALAR:
        sa = tl.load(sa_ptr).to(tl.float32)
    else:
        sa = tl.load(sa_ptr + pid_m).to(tl.float32)
    if B_SCALAR:
        sb = tl.load(sb_ptr) + tl.zeros((BLOCK_N,), dtype=tl.float32)
    else:
        sb = tl.load(sb_ptr + offs_n, mask=n_mask, other=0.0)

    scaled = acc * sa
    if HAS_BIAS:
        # Same as the tile path: avoid `scaled*sb + bias` being fused into a single-rounding FMA.
        product = tl.math.fma(scaled, sb, tl.zeros((BLOCK_N,), dtype=tl.float32))
        value = product + tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(
            tl.float32
        )
    else:
        value = scaled * sb
    out_ptrs = out_ptr + pid_m.to(tl.int64) * N + offs_n.to(tl.int64)
    tl.store(out_ptrs, value.to(out_ptr.dtype.element_ty), mask=n_mask)


def _gemv_config(n, k):
    """(BLOCK_N, BLOCK_K, num_warps) for the thin-M reduction path."""
    override = os.environ.get("FG_MM_W8A8_GEMV_TILE")
    if override:
        block_n, block_k, num_warps = (int(value) for value in override.split(","))
        return block_n, block_k, num_warps
    block_n = 128 if n >= 128 else (64 if n > 32 else 32)
    block_k = 512 if k >= 512 else (256 if k >= 256 else 128)
    return block_n, block_k, 4


def _tile_config(m, n, k):
    """Pick (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for the shape.

    GCU400 matrix path wants K >= 64 per tile and BLOCK_M >= 16; thin M shapes
    (the common decode layout) trade M tile height for N tile width so the
    wasted MMA rows stay low while the wide N still uses full tiles.
    """
    tile_override = os.environ.get("FG_MM_W8A8_TILE")
    if tile_override:
        block_m, block_n, block_k, num_warps, num_stages, group_m = (
            int(value) for value in tile_override.split(",")
        )
        return block_m, block_n, block_k, num_warps, num_stages, group_m

    # tile sweep (re-measured with batch10, 6 shapes × 3 candidates, see work/mm_w8a8_int8/probe_tile_matrix.py):
    # BLOCK_M=256/BLOCK_K=128 dominates across the board when m≥256
    # (4096³: 3.44ms vs 10.18ms@128x128x64, 256x1024x1024: 0.029 vs 0.071ms),
    # BLOCK_K=128 already pays off at k≥1024 (0.044 vs 0.066ms at k=2048, 1.7x at k=3584),
    # but for m≤64, BLOCK_M should still hug m, otherwise idle MMA rows eat up the gain
    # (64x2048x2048: 64x128x128 = 0.046ms, 32x128x64 = 0.130ms).
    if m <= 16:
        block_m = 16
        num_warps = 4
    elif m <= 64:
        block_m = 64
        num_warps = 8
    elif m <= 128:
        block_m = 128
        num_warps = 8
    else:
        block_m = 256
        num_warps = 8
    # Square tile (BLOCK_M=BLOCK_N=256): for m>128 it minimizes repeated reads of B,
    # while maximizing MMA reuse per program. Measured on GCU400 (batch10, 4096³):
    # 256x256x256 = 0.727ms / 189 TF, 256x128x256 = 1.171ms, 512x128x256 = 0.771ms;
    # On 1024x4096x4096, 0.200ms vs 0.316ms (256x128x256).
    block_n = 256 if (m > 128 and n >= 256) else 128
    # BLOCK_K tiers: a wider K tile cuts the number of K-direction iterations and
    # raises the MMA reuse of each program. Local A/B on the mm_w8a8 core cases
    # (15 cases, see the report):
    #   for k>=512 BLOCK_K=512 beats 256 across the board (4096^3: 1.192 -> 0.646ms
    #   /115 -> 212 TF, 2048^3: 0.162 -> 0.0966ms/106 -> 178 TF, 1024^3: 0.0327 ->
    #   0.0208ms), while small-K shapes regress, so the 128/64/32 tiers stay.
    if k >= 512:
        block_k = 512
    elif k >= 128:
        block_k = 128
    elif k < 64:
        block_k = 32
    else:
        block_k = 64
    num_stages = 2

    # Small square shapes (both M and N at most 512): on shapes like 384 the 256
    # square tile wastes too much on padding (384 = 256 + 128, the last block is
    # only half full). Stepping down to a 128 square tile measured 384^3 going from
    # 0.0137ms/8.3 TF to 0.0094ms/12.0 TF (+46%). Once either M or N exceeds 512
    # the reuse advantage of the 256 square tile dominates again, so keep it.
    if m <= 512 and n <= 512 and block_m > 128:
        block_m = 128
        block_n = 128
        num_warps = 4
        # A 128 square tile already fits the shared-memory/pipeline budget, and a
        # single stage saves one layer of synchronisation: measured on 384^3,
        # num_stages=1 -> 0.0094ms/12.0 TF, num_stages=2 -> 0.0134ms.
        num_stages = 1
        # Small K also benefits from a wide K tile: on 384^3, BK=512 (last block
        # half full) is 1.4x faster than BK=128 (0.0094 vs 0.0134ms) because the
        # K-direction loop count drops from 3 to 1.
        if k >= 256:
            block_k = max(block_k, 512)

    # Extremely narrow N (n < 128, mask active throughout the tile) paired with a 256-wide K tile falls into GCU's
    # silently-wrong-value zone: measurements show 256x1x2048 has 45/256 wrong elements (max absolute difference 8.75).
    # Such shapes fall back to the verified BLOCK_K=128.
    if n < 128:
        block_k = min(block_k, 128)

    # INT64 accumulation doubles accumulator register usage, and a 256-row tile hits the resource limit.
    if k > _MAX_K_WITHOUT_INT32_OVERFLOW:
        block_m = min(block_m, 128)
        block_n = min(block_n, 128)
    if n < 128:
        block_n = 64 if n > 64 else 32
    if num_warps > 4 and (block_m * block_n) < 8192:
        num_warps = 4
    return block_m, block_n, block_k, num_warps, num_stages, _GROUP_M


def _check_inputs(a, b, scale_a, scale_b, bias):
    if not all(isinstance(x, torch.Tensor) for x in (a, b, scale_a, scale_b)):
        raise TypeError("A, B and scales must be tensors")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("A and B must be prequantized INT8 codes")
    if a.device.type != flag_gems.device or any(
        x.device != a.device for x in (b, scale_a, scale_b)
    ):
        raise ValueError("inputs and scales must live on the same GCU device")
    m, k = a.shape
    n = b.shape[1]
    for scale, shapes in ((scale_a, ((m,), (m, 1))), (scale_b, ((n,), (1, n)))):
        if scale.numel() != 1 and scale.shape not in shapes:
            raise ValueError("expected scalar, per-row A or per-column B scales")
        if scale.dtype != torch.float32 or not scale.is_contiguous():
            raise ValueError("scales must be contiguous FP32 tensors")
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or bias.dtype not in _FLOATS:
            raise TypeError("bias must be an FP16/BF16/FP32 tensor")
        if bias.shape != (n,) or bias.device != a.device or not bias.is_contiguous():
            raise ValueError("bias must be contiguous [N] on the input device")
    return m, n, k


def _launch(a, b, scale_a, scale_b, bias, out, m, n, k):
    has_bias = bias is not None
    epilogue = dict(
        A_SCALAR=scale_a.numel() == 1,
        B_SCALAR=scale_b.numel() == 1,
        HAS_BIAS=has_bias,
        ACC_INT64=k > _MAX_K_WITHOUT_INT32_OVERFLOW,
    )
    bias_arg = bias if has_bias else out

    if k == 0:
        block_n = 128
        _w8a8_int8_bias_fill_kernel[(m, triton.cdiv(n, block_n))](
            bias_arg, out, m, n, HAS_BIAS=has_bias, BLOCK_N=block_n
        )
        return out

    if m <= _MAX_GEMV_M and 64 <= k <= _MAX_GEMV_K and n <= _GEMV_MAX_N:
        block_n, block_k, num_warps = _gemv_config(n, k)
        grid = (triton.cdiv(n, block_n), m)
        _w8a8_int8_gemv_kernel[grid](
            a,
            b,
            scale_a,
            scale_b,
            bias_arg,
            out,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,
            **{k2: v for k2, v in epilogue.items() if k2 != "ACC_INT64"},
        )
        return out

    block_m, block_n, block_k, num_warps, num_stages, group_m = _tile_config(m, n, k)
    # GCU400's grid.x limit is 65535, while tile counts along m/n multiply directly.
    # Large-N shapes (e.g. 8192x152064x3584 → 64×1188 = 76032) would directly
    # OutOfResources, so chunk along N and launch segment by segment: tile size, register usage, and
    # numerics are unchanged, just a few more launches.
    blocks_m = triton.cdiv(m, block_m)
    n_chunk = n
    if blocks_m * triton.cdiv(n, block_n) > _MAX_GRID_X:
        blocks_per_chunk = max(1, _MAX_GRID_X // blocks_m)
        n_chunk = min(n, blocks_per_chunk * block_n)
    n_begin = 0
    while n_begin < n:
        n_end = min(n, n_begin + n_chunk)
        width = n_end - n_begin
        if width == n:
            b_view, out_view = b, out
            sb_view = scale_b
            bias_view = bias_arg
        else:
            b_view = b.narrow(1, n_begin, width)
            out_view = out.narrow(1, n_begin, width)
            sb_view = (
                scale_b.narrow(-1, n_begin, width)
                if not epilogue["B_SCALAR"]
                else scale_b
            )
            bias_view = bias_arg.narrow(0, n_begin, width) if has_bias else bias_arg
        grid = (blocks_m * triton.cdiv(width, block_n),)
        _w8a8_int8_gemm_kernel[grid](
            a,
            b_view,
            scale_a,
            sb_view,
            bias_view,
            out_view,
            m,
            width,
            k,
            a.stride(0),
            a.stride(1),
            b_view.stride(0),
            b_view.stride(1),
            out_view.stride(0),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            EVEN_M=m % block_m == 0,
            EVEN_N=width % block_n == 0,
            EVEN_K=k % block_k == 0,
            num_warps=num_warps,
            num_stages=num_stages,
            **epilogue,
        )
        n_begin = n_end
    return out


def _prepare_inputs(a, b):
    """Materialize zero-stride (broadcast) views.

    The GCU lowering aborts on a tile axis that does not advance at all
    (``stride == 0``), so an expanded A/B is copied once instead. Plain strided
    views (sliced rows/cols) stay on the zero-copy path.
    """
    if 0 in a.stride():
        a = a.contiguous()
    if 0 in b.stride():
        b = b.contiguous()
    return a, b


def _run(a, b, scale_a, scale_b, bias, out, m, n, k):
    with torch_device_fn.device(a.device):
        a, b = _prepare_inputs(a, b)
        return _launch(a, b, scale_a, scale_b, bias, out, m, n, k)


def mm_w8a8_int8(a, b, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None):
    """Prequantized INT8 GEMM with fused dequantization.

    A is ``[M, K]`` INT8, B is ``[K, N]`` INT8 (any strides), scales are FP32
    (scalar or per-row A / per-column B), bias is an optional contiguous
    ``[N]`` vector added in FP32 before the cast to ``out_dtype``. Quantization
    is the caller's responsibility and stays outside the timed region.
    Forward inference only.
    """
    m, n, k = _check_inputs(a, b, scale_a, scale_b, bias)
    if out_dtype not in _FLOATS:
        raise TypeError("out_dtype must be FP16, BF16 or FP32")
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    logger.debug("GEMS_ENFLAMEMM_W8A8_INT8")
    return _run(a, b, scale_a, scale_b, bias, out, m, n, k)


def mm_w8a8_int8_out(a, b, scale_a, scale_b, *, out, bias=None):
    """Prequantized INT8 GEMM into a contiguous ``[M, N]`` output buffer."""
    m, n, k = _check_inputs(a, b, scale_a, scale_b, bias)
    if not isinstance(out, torch.Tensor) or out.dtype not in _FLOATS:
        raise TypeError("out must be an FP16/BF16/FP32 tensor")
    if out.shape != (m, n) or out.device != a.device or not out.is_contiguous():
        raise ValueError("out must be contiguous [M,N] on the input device")
    if out.numel() and any(
        x is not None
        and x.numel()
        and out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        for x in (a, b, scale_a, scale_b, bias)
    ):
        raise ValueError("out must not alias inputs, scales or bias")
    logger.debug("GEMS_ENFLAMEMM_W8A8_INT8_OUT")
    return _run(a, b, scale_a, scale_b, bias, out, m, n, k)


__all__ = ["mm_w8a8_int8", "mm_w8a8_int8_out"]
