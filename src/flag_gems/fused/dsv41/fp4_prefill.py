# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Portions adapted from SGLang (Copyright 2023-2026 SGLang Team, Apache-2.0).
"""Canonical packed-page casts and independent dense FP8 index scoring.

Page format and physical slots follow the explicit-slot FP4 API. No request
mapping, compressed-candidate layout or SGLang imports are accepted here.
"""

import torch
import triton
import triton.language as tl

from flag_gems.fused.dsv41.fp4_indexer import PAYLOAD_BYTES, SCALE_BYTES, _e2m1_decode

INDEX_HEAD_DIM = 128
FP8_E4M3_MAX = 448.0


@triton.jit
def _unpack_fp4_index_keys_to_fp8_kernel(
    slots_ptr,
    table_ptr,
    out_ptr,
    page_size,
    row_stride,
    HALF_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Decode block-scaled E2M1 values directly into E4M3."""
    row = tl.program_id(0)
    offs_i = tl.arange(0, HALF_D)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    page = slot // page_size
    off = slot % page_size
    row_base = page * row_stride
    pay = tl.load(table_ptr + row_base + off * PAYLOAD_BYTES + offs_i)
    scale_block = offs_i // 16
    exps = tl.load(
        table_ptr
        + row_base
        + page_size * PAYLOAD_BYTES
        + off * SCALE_BYTES
        + scale_block
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    low = tl.clamp(_e2m1_decode(pay & 0x0F) * scale, -FP8_MAX, FP8_MAX).to(
        out_ptr.dtype.element_ty
    )
    high = tl.clamp(
        _e2m1_decode((pay >> 4) & 0x0F) * scale,
        -FP8_MAX,
        FP8_MAX,
    ).to(out_ptr.dtype.element_ty)
    out = out_ptr + row * (HALF_D * 2)
    tl.store(out + 2 * offs_i, low)
    tl.store(out + 2 * offs_i + 1, high)


def unpack_fp4_index_keys_to_fp8(
    slots: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Gather FP4 index-K rows and decode them directly into E4M3."""
    if not slots.is_cuda or table.device != slots.device:
        raise ValueError("expected CUDA physical slots and packed table on one device")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    assert slots.dim() == 1
    assert table.dtype == torch.uint8 and table.dim() == 2
    slots = slots.to(torch.int64).contiguous()
    values = torch.empty(
        (slots.shape[0], INDEX_HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=slots.device,
    )
    if slots.numel() > 0:
        _unpack_fp4_index_keys_to_fp8_kernel[(slots.shape[0],)](
            slots,
            table,
            values,
            page_size,
            table.stride(0),
            HALF_D=INDEX_HEAD_DIM // 2,
            FP8_MAX=FP8_E4M3_MAX,
            num_warps=4,
        )
    return values


@triton.jit
def _quantize_bf16_index_queries_fp8_kernel(
    q_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_ob,
    stride_oh,
    H: tl.constexpr,
    D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    q = tl.load(
        q_ptr + row * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :]
    ).to(tl.float32)
    q_fp8 = tl.clamp(q, -FP8_MAX, FP8_MAX).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + row * stride_ob + offs_h[:, None] * stride_oh + offs_d[None, :],
        q_fp8,
    )


def quantize_bf16_index_queries_fp8(
    q: torch.Tensor,
) -> torch.Tensor:
    """Cast each BF16 query head to E4M3 for one K=128 tensor-core dot."""
    if not q.is_cuda:
        raise ValueError("expected CUDA query")
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    q = q.contiguous()
    rows, heads, _ = q.shape
    values = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    if rows > 0:
        _quantize_bf16_index_queries_fp8_kernel[(rows,)](
            q,
            values,
            q.stride(0),
            q.stride(1),
            values.stride(0),
            values.stride(1),
            H=heads,
            D=INDEX_HEAD_DIM,
            FP8_MAX=FP8_E4M3_MAX,
            num_warps=8,
        )
    return values


@triton.jit
def _dense_fp8_prefill_kernel(
    q_ptr,
    w_ptr,
    k_ptr,
    lens_ptr,
    out_ptr,
    L,
    OUT_L,
    stride_qb,
    stride_qh,
    stride_kl,
    stride_wb,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    lb = tl.program_id(1)
    out_cols = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_l = out_cols
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    n_vis = tl.load(lens_ptr + b)
    valid = (offs_l >= 0) & (offs_l < tl.minimum(n_vis, L)) & (out_cols < OUT_L)
    q = tl.load(q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :])
    k = tl.load(
        k_ptr + offs_l[:, None] * stride_kl + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    acc = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    s = acc.to(tl.bfloat16).to(tl.float32)
    s = tl.maximum(s, 0.0)
    w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
    s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
    logit = tl.sum(s, axis=0).to(tl.bfloat16).to(tl.float32)
    logit = tl.where(valid, logit, float("-inf"))
    tl.store(out_ptr + b * OUT_L + out_cols, logit, mask=out_cols < OUT_L)


def dense_fp8_index_logits(q, weights, keys, lens):
    if q.ndim != 3 or q.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("expected [rows, heads, 128] query")
    rows, heads, _ = q.shape
    if (
        q.dtype != torch.float8_e4m3fn
        or keys.dtype != q.dtype
        or keys.ndim != 2
        or keys.shape[1] != INDEX_HEAD_DIM
        or weights.shape != (rows, heads)
        or lens.shape != (rows,)
    ):
        raise ValueError("incompatible dense FP8 score inputs")
    if not q.is_cuda or any(t.device != q.device for t in (weights, keys, lens)):
        raise ValueError("expected tensors on one CUDA device")
    q, weights, keys = (
        q.contiguous(),
        weights.to(torch.bfloat16).contiguous(),
        keys.contiguous(),
    )
    width = keys.shape[0]
    out_width = triton.cdiv(width, 4) * 4
    if rows == 0 or out_width == 0:
        return torch.empty((rows, out_width), dtype=torch.float32, device=q.device)
    if (
        heads == 32
        and triton.__version__.split("+")[0] == "3.6.0"
        and torch.cuda.get_device_capability(q.device) == (9, 0)
        and lens.dtype in (torch.int32, torch.int64)
        and lens.is_contiguous()
    ):
        from flag_gems.fused.dsv41.fp8_prefill_score import fp8_prefill_score

        return fp8_prefill_score(q, weights, keys, lens)
    out = torch.empty((rows, out_width), dtype=torch.float32, device=q.device)
    block = (
        256 if heads == 32 and out_width >= 4096 and rows * out_width >= 262144 else 64
    )
    _dense_fp8_prefill_kernel[(rows, triton.cdiv(out_width, block))](
        q,
        weights,
        keys,
        lens.to(torch.int64).contiguous(),
        out,
        width,
        out_width,
        q.stride(0),
        q.stride(1),
        keys.stride(0),
        weights.stride(0),
        H=heads,
        D=INDEX_HEAD_DIM,
        BLOCK_L=block,
        num_warps=4,
    )
    return out
