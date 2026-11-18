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

"""Ascend matrix multiplication of prequantized INT8 inputs and FP32 scales."""

from __future__ import annotations

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def mm_int8_vector_loop(
    A,
    B,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK_STRIDE: tl.constexpr,
    BN_STRIDE: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    FLOAT_PARTIAL: tl.constexpr,
):
    pid = tl.program_id(0)
    columns: tl.constexpr = tl.cdiv(N, BN)
    row = pid // columns
    col = (pid % columns) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    total = tl.zeros((BN,), tl.int32)
    for begin in range(0, K, BK):
        key = begin + kk
        if K % BK == 0:
            activation = tl.load(A + row * AM + key * AK)
        else:
            activation = tl.load(A + row * AM + key * AK, key < K, other=0)
        if N % BN == 0 and K % BK == 0:
            weight = tl.load(B + col[:, None] * BN_STRIDE + key[None, :] * BK_STRIDE)
        else:
            weight = tl.load(
                B + col[:, None] * BN_STRIDE + key[None, :] * BK_STRIDE,
                (col[:, None] < N) & (key[None, :] < K),
                other=0,
            )
        if FLOAT_PARTIAL:
            partial = tl.sum(
                weight.to(tl.float32) * activation[None, :].to(tl.float32), 1
            ).to(tl.int32)
        else:
            partial = tl.sum(weight.to(tl.int32) * activation[None, :].to(tl.int32), 1)
        total += partial
    scale_a = tl.load(SA + row)
    if N % BN == 0:
        scale_b = tl.load(SB + col)
        tl.store(OUT + row * OM + col * ON, total.to(tl.float32) * scale_a * scale_b)
    else:
        scale_b = tl.load(SB + col, col < N, other=0)
        tl.store(
            OUT + row * OM + col * ON, total.to(tl.float32) * scale_a * scale_b, col < N
        )


@libentry()
@triton.jit
def mm_int8_vector_full(
    A,
    B,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BKS: tl.constexpr,
    BNS: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles: tl.constexpr = tl.cdiv(N, BN)
    row = pid // tiles
    col = pid % tiles * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    activation = tl.load(A + row * AM + kk * AK, kk < K, other=0).to(tl.float32)
    weight = tl.load(
        B + col[:, None] * BNS + kk[None, :] * BKS,
        (col[:, None] < N) & (kk[None, :] < K),
        other=0,
    ).to(tl.float32)
    product = weight * activation[None, :]
    # Every chunk's absolute integer sum is <= 2**24 and is exact in FP32.
    partial = tl.sum(product.reshape((BN, BK // 1024, 1024)), 2).to(tl.int32)
    total = tl.sum(partial, 1)
    scale_a = tl.load(SA + row)
    scale_b = tl.load(SB + col, col < N, other=0)
    tl.store(
        OUT + row * OM + col * ON, total.to(tl.float32) * scale_a * scale_b, col < N
    )


@libentry()
@triton.jit
def direct_mm_kernel(
    A,
    B,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PADDED_M: tl.constexpr,
    ASM: tl.constexpr,
    ASK: tl.constexpr,
    BSK: tl.constexpr,
    BSN: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    CORES: tl.constexpr,
    SCALED: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_m: tl.constexpr = tl.cdiv(M, BM)
    grid_n: tl.constexpr = tl.cdiv(N, BN)
    tiles: tl.constexpr = grid_m * grid_n
    quotient = tiles // CORES
    remainder = tiles % CORES
    start = tl.where(
        pid < remainder,
        pid * (quotient + 1),
        remainder * (quotient + 1) + (pid - remainder) * quotient,
    )
    count = quotient + tl.where(pid < remainder, 1, 0)
    for tile_offset in tl.range(0, count):
        tile = start + tile_offset
        row_start = (tile // grid_n * BM).to(tl.int32)
        col_start = (tile % grid_n * BN).to(tl.int32)
        a_order: tl.constexpr = (0, 1) if ASM < ASK else (1, 0)
        b_order: tl.constexpr = (0, 1) if BSK < BSN else (1, 0)
        ap = tl.make_block_ptr(
            A, (PADDED_M, K), (ASM, ASK), (row_start, 0), (BM, BK), a_order
        )
        bp = tl.make_block_ptr(B, (K, N), (BSK, BSN), (0, col_start), (BK, BN), b_order)
        acc = tl.zeros((BM, BN), tl.int32)
        for _ in tl.range(0, tl.cdiv(K, BK)):
            if PADDED_M % BM == 0 and K % BK == 0:
                av = tl.load(ap)
            else:
                av = tl.load(ap, boundary_check=(0, 1), padding_option="zero")
            if N % BN == 0 and K % BK == 0:
                bv = tl.load(bp)
            else:
                bv = tl.load(bp, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(av, bv, acc, out_dtype=tl.int32)
            ap = tl.advance(ap, (0, BK))
            bp = tl.advance(bp, (BK, 0))
        if SCALED:
            rows = row_start + tl.arange(0, BM)
            columns = col_start + tl.arange(0, BN)
            scale_a = tl.load(SA + rows, rows < M, other=0)
            scale_b = tl.load(SB + columns, columns < N, other=0)
            value = acc.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
            tl.store(
                OUT + rows[:, None] * OM + columns[None, :] * ON,
                value,
                (rows[:, None] < M) & (columns[None, :] < N),
            )
        else:
            cp = tl.make_block_ptr(
                OUT, (M, N), (OM, ON), (row_start, col_start), (BM, BN), (1, 0)
            )
            tl.store(cp, acc, boundary_check=(0, 1))


@libentry()
@triton.jit
def pad_a_kernel(
    A,
    PAD,
    M: tl.constexpr,
    K: tl.constexpr,
    PADDED_M: tl.constexpr,
    SM: tl.constexpr,
    SK: tl.constexpr,
):
    for base in range(
        ext.program_id(0) * 2048, PADDED_M * K, ext.num_programs(0) * 2048
    ):
        offsets = base + tl.arange(0, 2048)
        if SM == K and SK == 1:
            # Keep contiguous copies linear for DMA lowering.
            value = tl.load(A + offsets, offsets < M * K, other=0)
        else:
            rows = offsets // K
            columns = offsets % K
            value = tl.load(A + rows * SM + columns * SK, offsets < M * K, other=0)
        tl.store(PAD + offsets, value, offsets < PADDED_M * K)


@libentry()
@triton.jit
def finish_mm_kernel(
    PARTIAL,
    SA,
    SB,
    BIAS,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    PM: tl.constexpr,
    PN: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    PARTS: tl.constexpr,
    APPLY_SCALES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < M * N
    rows = offsets // N
    columns = offsets % N
    if PARTIAL.dtype.element_ty == tl.int32:
        # With K < 2**31, both radix-2**24 limbs convert exactly to FP32.
        high = tl.full((BLOCK,), 0, tl.int32)
        low = tl.full((BLOCK,), 0, tl.int32)
        for part in range(PARTS):
            partial = tl.load(
                PARTIAL + part.to(tl.int64) * (PM * PN) + rows * PN + columns,
                valid,
                other=0,
            )
            low += partial & 0xFFFFFF
            high += (partial >> 24) + (low >> 24)
            low = low & 0xFFFFFF
        value = high.to(tl.float32) * 16777216.0 + low.to(tl.float32)
    else:
        value = tl.full((BLOCK,), 0, tl.float32)
        for part in range(PARTS):
            value += tl.load(
                PARTIAL + part.to(tl.int64) * (PM * PN) + rows * PN + columns,
                valid,
                other=0,
            )
    if APPLY_SCALES:
        scale_a = tl.load(SA + rows, valid, other=0)
        scale_b = tl.load(SB + columns, valid, other=0)
        value = value * scale_a * scale_b
    if HAS_BIAS:
        value += tl.load(BIAS + columns, valid, other=0).to(tl.float32)
    tl.store(OUT + rows * OM + columns * ON, value, valid)


def scaled_mm_arguments(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, ...]
]:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("A and B must be prequantized INT8 tensors")
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("A and B must be prequantized INT8 tensors")
    if a.ndim < 1 or b.ndim != 2 or a.shape[-1] != b.shape[0]:
        raise ValueError("expected A[...,K] and B[K,N]")
    if a.device.type != "npu" or a.device != b.device:
        raise ValueError("A and B must be on the same NPU")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("out_dtype must be FP16, BF16 or FP32")
    k, n = b.shape
    if k >= 2**31:
        raise ValueError("K must be smaller than 2**31")
    m = math.prod(a.shape[:-1])

    def _normalize(scale, size, name):
        if not isinstance(scale, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if scale.dtype != torch.float32 or scale.device != a.device:
            raise ValueError(f"{name} must be FP32 on the input device")
        if scale.numel() not in (1, size):
            raise ValueError(f"{name} must be scalar or contain {size} values")
        flat = scale.reshape(-1)
        if flat.numel() == 1:
            return flat.expand(size).contiguous()
        return flat.contiguous()

    sa = _normalize(scale_a, m, "scale_a")
    sb = _normalize(scale_b, n, "scale_b")
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError("bias must be a tensor")
        if bias.device != a.device or bias.dtype != out_dtype or bias.numel() != n:
            raise ValueError(
                "bias must contain N values of the output dtype on the input device"
            )
        bias = bias.reshape(-1).contiguous()
    return a.reshape(m, k), sa, sb, bias, (*a.shape[:-1], n)


def launch_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
) -> None:
    m, k = a.shape
    n = b.shape[1]
    scaled = out.dtype != torch.int32
    vector = (
        scaled
        and m in (1, 2)
        and (
            (n == 64 and k == 4096)
            or (n in (1, 4) and 1024 <= k <= 16384 and k % 1024 == 0)
        )
    )
    if vector and a.stride(1) == 1 and b.stride() == (1, k) and out.is_contiguous():
        if n == 64:
            bn = 4 if m == 1 else 8
            mm_int8_vector_loop[(m * triton.cdiv(n, bn),)](
                a,
                b,
                sa,
                sb,
                out,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                *out.stride(),
                bn,
                1024,
                True,
                num_warps=1,
                multibuffer=False,
                enable_fp_fusion=False,
            )
        else:
            mm_int8_vector_full[(m * n,)](
                a,
                b,
                sa,
                sb,
                out,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                *out.stride(),
                1,
                triton.next_power_of_2(k),
                num_warps=1,
                multibuffer=False,
                enable_fp_fusion=False,
            )
        return

    # Clone also canonicalizes singleton strides that contiguous() can preserve.
    if 0 in a.stride() or a.stride(1) != 1:
        a = a.clone(memory_format=torch.contiguous_format)
    if 0 in b.stride() or 1 not in b.stride():
        b = b.clone(memory_format=torch.contiguous_format)

    if not scaled:
        bm = 16 if m <= 128 else 128
        bn = 64 if n <= 256 else 128
        bk = 256
    elif n <= 256:
        bm = 16 if m <= 128 else 128
        bn = 32 if m <= 128 else (64 if n < 256 else 256)
        bk = min(1024 if m <= 128 else 512, max(32, triton.next_power_of_2(k)))
    else:
        bm = 16 if m <= 16 else (64 if m <= 64 else 128)
        bn = 256
        bk = min(512, max(32, triton.next_power_of_2(k)))
    padded_m = m
    if scaled and n > 256 and m % bm:
        # One small A copy avoids masked-load overhead across every wide B tile.
        padded_m = triton.cdiv(m, bm) * bm
        padded = torch.empty((padded_m, k), device=a.device, dtype=a.dtype)
        pad_a_kernel[(min(40, triton.cdiv(padded_m * k, 2048)),)](
            a, padded, m, k, padded_m, *a.stride()
        )
        a = padded
    cores = min(20, triton.cdiv(m, bm) * triton.cdiv(n, bn))
    direct_mm_kernel[(cores,)](
        a,
        b,
        sa,
        sb,
        out,
        m,
        n,
        k,
        padded_m,
        *a.stride(),
        *b.stride(),
        *out.stride(),
        bm,
        bn,
        bk,
        cores,
        scaled,
        num_warps=1,
        num_stages=2,
        optimize_dynamic_offset=True,
        unit_flag=False,
        limit_auto_multi_buffer_of_local_buffer="no-l0c",
        # One workspace slot keeps persistent Cube/Vector tile reuse synchronized.
        set_workspace_multibuffer=1,
    )


def scaled_mm_execute(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    if m == 0 or n == 0:
        return out
    flat_out = out if out.ndim == 2 else out.view(m, n)
    with torch_device_fn.device(a.device):
        if k == 0:
            partials, parts, apply_scales = flat_out, 0, True
        elif k > 65536:
            # Each INT32 partial stays below 2**31 even for all -128 inputs.
            parts = triton.cdiv(k, 65536)
            partials = torch.empty((parts, m, n), device=a.device, dtype=torch.int32)
            for part in range(parts):
                start = part * 65536
                stop = min(start + 65536, k)
                launch_mm(a[:, start:stop], b[start:stop], sa, sb, partials[part])
            apply_scales = True
        elif bias is None:
            launch_mm(a, b, sa, sb, flat_out)
            return out
        else:
            # Keep FP32 through bias addition; rounding before bias changes results.
            partials = torch.empty((m, n), device=a.device, dtype=torch.float32)
            launch_mm(a, b, sa, sb, partials)
            parts, apply_scales = 1, False
        if k > 65536 and bias is not None:
            # Materialize scaled FP32 before bias, as in the short-K path.
            scaled = torch.empty((m, n), device=a.device, dtype=torch.float32)
            finish_mm_kernel[(triton.cdiv(m * n, 512),)](
                partials,
                sa,
                sb,
                None,
                scaled,
                m,
                n,
                m,
                n,
                n,
                1,
                parts,
                True,
                False,
                512,
                enable_fp_fusion=False,
            )
            partials, parts, apply_scales = scaled, 1, False
        finish_mm_kernel[(triton.cdiv(m * n, 512),)](
            partials,
            sa,
            sb,
            bias,
            flat_out,
            m,
            n,
            m,
            n,
            *flat_out.stride(),
            parts,
            apply_scales,
            bias is not None,
            512,
            enable_fp_fusion=False,
        )
    return out


def mm_w8a8_int8(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute INT8 A[...,K] @ B[K,N], FP32 scales and optional output bias."""
    logger.debug("GEMS_ASCEND MM_W8A8_INT8")
    a2d, sa, sb, bias, shape = scaled_mm_arguments(
        a, b, scale_a, scale_b, out_dtype, bias
    )
    out = torch.empty(shape, device=a.device, dtype=out_dtype)
    return scaled_mm_execute(a2d, b, sa, sb, bias, out)


def mm_w8a8_int8_out(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write prequantized INT8 matmul into caller-owned output without aliasing."""
    logger.debug("GEMS_ASCEND MM_W8A8_INT8_OUT")
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a tensor")
    a2d, sa, sb, normalized_bias, shape = scaled_mm_arguments(
        a, b, scale_a, scale_b, out.dtype, bias
    )
    if out.shape != shape or out.device != a.device:
        raise ValueError("out must have the result shape on the input device")
    if out.ndim != 2 and not out.is_contiguous():
        raise ValueError("batched output must be contiguous")
    if out.ndim == 2 and out.numel():
        rows, columns = out.shape
        stride_m, stride_n = out.stride()
        divisor = math.gcd(stride_m, stride_n)
        if (
            (stride_m == 0 and rows > 1)
            or (stride_n == 0 and columns > 1)
            or (
                divisor and rows > stride_n // divisor and columns > stride_m // divisor
            )
        ):
            raise ValueError("out elements must not overlap")
    if out.numel() and any(
        tensor is not None
        and tensor.numel()
        and out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (a, b, scale_a, scale_b, bias)
    ):
        raise ValueError("out must not alias inputs, scales or bias")
    return scaled_mm_execute(a2d, b, sa, sb, normalized_bias, out)
