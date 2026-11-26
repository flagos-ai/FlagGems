# SPDX-License-Identifier: Apache-2.0
"""Fused cross-layer post-attention MLP, Group-2048 SwiGLU+Quant, next-layer RMSNorm+Quant, and
QKV+RoPE+Cache for MetaX C500 Wave64."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from flag_gems.fused.metax_fused_qkv_rope_cache import (
    fused_qkv_rope_and_cache,
    fused_qkv_rope_and_cache_i8,
    fused_qkv_splitk2_rope_and_cache,
    fused_qkv_splitk2_rope_and_cache_i8,
)
from flag_gems.fused.metax_kcontig_mm import (
    _metax_nt_kcontig_part_kernel,
    try_metax_kcontig_nt_mm,
)
from flag_gems.fused.metax_rowwise_int8 import (
    project,
    project_codes,
    project_codes_g2048_k6144,
    project_codes_splitk,
    project_codes_splitk_part,
)
from flag_gems.runtime import torch_device_fn


def use_packed_i8_qkv(rows: int) -> bool:
    """Single source of truth for whether x_pack carries packed INT8 codes or BF16 activations."""
    if 8 <= rows <= 64:
        return os.environ.get("FL_C500_DECODE_I8_QKV", "1") != "0"
    return 256 <= rows <= 4096


def _pack_int8_view(x_pack: torch.Tensor) -> torch.Tensor:
    rows = x_pack.shape[0]
    return x_pack.view(torch.int8).reshape(-1)[: rows * 2048].view(rows, 2048)


@triton.jit
def _fused_splitk_add_rmsnorm_quant_k2048(
    Part,
    Res,
    WNorm,
    ResOut,
    Codes,
    Scales,
    M,
    stride_res,
    eps: tl.constexpr,
    SK: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    m = col < K
    mn = M * K
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(0, SK):
        acc += tl.load(Part + s * mn + row * K + col, mask=m, other=0.0)
    o_bf16 = acc.to(tl.bfloat16).to(tl.float32)
    r_f32 = tl.load(Res + row * stride_res + col, mask=m, other=0.0).to(tl.float32)
    s_f32 = o_bf16 + r_f32
    tl.store(ResOut + row * K + col, s_f32.to(tl.bfloat16), mask=m)

    var = tl.sum(s_f32 * s_f32, 0) * (1.0 / K)
    inv_rms = tl.rsqrt(var + eps)
    wn = tl.load(WNorm + col, mask=m, other=0.0)
    normed = ((s_f32 * inv_rms).to(tl.bfloat16) * wn).to(tl.bfloat16).to(tl.float32)

    peak = tl.max(tl.abs(normed), 0)
    scale = tl.where(peak > 0.0, peak * (1.0 / 127.0), 1.0)
    inv_scale = tl.where(peak > 0.0, 127.0 / peak, 0.0)
    q = tl.extra.cuda.libdevice.rint(normed * inv_scale).to(tl.int32).to(tl.int8)
    tl.store(Codes + row * K + col, q, mask=m)
    tl.store(Scales + row, scale)


@triton.jit
def _fused_splitk_add_rmsnorm_bf16_k2048(
    Part,
    Res,
    WNorm,
    ResOut,
    NormOut,
    M,
    stride_res,
    eps: tl.constexpr,
    SK: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    m = col < K
    mn = M * K
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(0, SK):
        acc += tl.load(Part + s * mn + row * K + col, mask=m, other=0.0)
    o_bf16 = acc.to(tl.bfloat16).to(tl.float32)
    r_f32 = tl.load(Res + row * stride_res + col, mask=m, other=0.0).to(tl.float32)
    s_f32 = o_bf16 + r_f32
    tl.store(ResOut + row * K + col, s_f32.to(tl.bfloat16), mask=m)

    var = tl.sum(s_f32 * s_f32, 0) * (1.0 / K)
    inv_rms = tl.rsqrt(var + eps)
    wn = tl.load(WNorm + col, mask=m, other=0.0)
    normed = ((s_f32 * inv_rms).to(tl.bfloat16) * wn).to(tl.bfloat16)
    tl.store(NormOut + row * K + col, normed, mask=m)


@triton.jit
def _fused_g2048_splitk3_add_rmsnorm_quant_k2048(
    Part,
    XScales,
    WScale,
    Res,
    WNorm,
    ResOut,
    Codes,
    Scales,
    M,
    stride_res,
    eps: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Cross-layer fusion: down_proj(Group-2048 SplitK-3) reduce + residual + next input RMSNorm + INT8 quant."""
    row = tl.program_id(0)
    col = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    m = col < K
    mn = M * K
    p0 = tl.load(Part + 0 * mn + row * K + col, mask=m, other=0).to(tl.float32)
    p1 = tl.load(Part + 1 * mn + row * K + col, mask=m, other=0).to(tl.float32)
    p2 = tl.load(Part + 2 * mn + row * K + col, mask=m, other=0).to(tl.float32)
    sx0 = tl.load(XScales + 0 * M + row)
    sx1 = tl.load(XScales + 1 * M + row)
    sx2 = tl.load(XScales + 2 * M + row)
    sw = tl.load(WScale + col, mask=m, other=0.0)
    d_bf16 = ((p0 * sx0 + p1 * sx1 + p2 * sx2) * sw).to(tl.bfloat16).to(tl.float32)
    r_f32 = tl.load(Res + row * stride_res + col, mask=m, other=0.0).to(tl.float32)
    s_f32 = d_bf16 + r_f32
    tl.store(ResOut + row * K + col, s_f32.to(tl.bfloat16), mask=m)

    var = tl.sum(s_f32 * s_f32, 0) * (1.0 / K)
    inv_rms = tl.rsqrt(var + eps)
    wn = tl.load(WNorm + col, mask=m, other=0.0)
    normed = ((s_f32 * inv_rms).to(tl.bfloat16) * wn).to(tl.bfloat16).to(tl.float32)

    peak = tl.max(tl.abs(normed), 0)
    scale = tl.where(peak > 0.0, peak * (1.0 / 127.0), 1.0)
    inv_scale = tl.where(peak > 0.0, 127.0 / peak, 0.0)
    q = tl.extra.cuda.libdevice.rint(normed * inv_scale).to(tl.int32).to(tl.int8)
    tl.store(Codes + row * K + col, q, mask=m)
    tl.store(Scales + row, scale)


@triton.jit
def _fused_g2048_splitk3_add_rmsnorm_bf16_k2048(
    Part,
    XScales,
    WScale,
    Res,
    WNorm,
    NormOut,
    M,
    stride_res,
    eps: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Last-layer fusion: down_proj(Group-2048 SplitK-3) reduce + residual + final model RMSNorm."""
    row = tl.program_id(0)
    col = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    m = col < K
    mn = M * K
    p0 = tl.load(Part + 0 * mn + row * K + col, mask=m, other=0).to(tl.float32)
    p1 = tl.load(Part + 1 * mn + row * K + col, mask=m, other=0).to(tl.float32)
    p2 = tl.load(Part + 2 * mn + row * K + col, mask=m, other=0).to(tl.float32)
    sx0 = tl.load(XScales + 0 * M + row)
    sx1 = tl.load(XScales + 1 * M + row)
    sx2 = tl.load(XScales + 2 * M + row)
    sw = tl.load(WScale + col, mask=m, other=0.0)
    d_bf16 = ((p0 * sx0 + p1 * sx1 + p2 * sx2) * sw).to(tl.bfloat16).to(tl.float32)
    r_f32 = tl.load(Res + row * stride_res + col, mask=m, other=0.0).to(tl.float32)
    s_f32 = d_bf16 + r_f32

    var = tl.sum(s_f32 * s_f32, 0) * (1.0 / K)
    inv_rms = tl.rsqrt(var + eps)
    wn = tl.load(WNorm + col, mask=m, other=0.0)
    normed = ((s_f32 * inv_rms).to(tl.bfloat16) * wn).to(tl.bfloat16)
    tl.store(NormOut + row * K + col, normed, mask=m)


@triton.jit
def _fused_add_rmsnorm_quant_k2048(
    X,
    Res,
    WNorm,
    ResOut,
    Codes,
    Scales,
    stride_x,
    stride_res,
    eps: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    m = col < K
    x_f32 = tl.load(X + row * stride_x + col, mask=m, other=0.0).to(tl.float32)
    r_f32 = tl.load(Res + row * stride_res + col, mask=m, other=0.0).to(tl.float32)
    s_f32 = x_f32 + r_f32
    tl.store(ResOut + row * K + col, s_f32.to(tl.bfloat16), mask=m)

    var = tl.sum(s_f32 * s_f32, 0) * (1.0 / K)
    inv_rms = tl.rsqrt(var + eps)
    wn = tl.load(WNorm + col, mask=m, other=0.0)
    normed = ((s_f32 * inv_rms).to(tl.bfloat16) * wn).to(tl.bfloat16).to(tl.float32)

    peak = tl.max(tl.abs(normed), 0)
    scale = tl.where(peak > 0.0, peak * (1.0 / 127.0), 1.0)
    inv_scale = tl.where(peak > 0.0, 127.0 / peak, 0.0)
    q = tl.extra.cuda.libdevice.rint(normed * inv_scale).to(tl.int32).to(tl.int8)
    tl.store(Codes + row * K + col, q, mask=m)
    tl.store(Scales + row, scale)


@triton.jit
def _fast_swiglu_bf16(GateUp, Out, M, K: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mk = M * K
    m = idx < mk
    row = idx // K
    col = idx % K
    base = row * (2 * K) + col
    g = tl.load(GateUp + base, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GateUp + base + K, mask=m, other=0.0)
    silu_g = (g * tl.sigmoid(g)).to(tl.bfloat16)
    tl.store(Out + idx, silu_g * u, mask=m)


@triton.jit
def _fused_swiglu_quant_g2048(GateUp, Codes, Scales, M):
    """Group-2048 fused SwiGLU + INT8 quantization: 3x SM parallelism and 1.7x lower outlier error."""
    row = tl.program_id(0)
    grp = tl.program_id(1)
    col = tl.max_contiguous(tl.multiple_of(tl.arange(0, 2048), 2048), 2048)
    base = row * 12288 + grp * 2048
    g = tl.load(GateUp + base + col).to(tl.float32)
    u = tl.load(GateUp + base + 6144 + col)
    v = ((g * tl.sigmoid(g)).to(tl.bfloat16) * u).to(tl.float32)
    peak = tl.max(tl.abs(v), 0)
    scale = tl.where(peak > 0.0, peak * (1.0 / 127.0), 1.0)
    inv_scale = tl.where(peak > 0.0, 127.0 / peak, 0.0)
    q = tl.extra.cuda.libdevice.rint(v * inv_scale).to(tl.int32).to(tl.int8)
    tl.store(Codes + row * 6144 + grp * 2048 + col, q)
    tl.store(Scales + grp * M + row, scale)


@triton.jit
def _fused_swiglu_quant_3x2048(GateUp, Codes, Scales):
    row = tl.program_id(0)
    col = tl.max_contiguous(tl.multiple_of(tl.arange(0, 2048), 2048), 2048)
    base = row * 12288

    g0 = tl.load(GateUp + base + col).to(tl.float32)
    u0 = tl.load(GateUp + base + 6144 + col)
    v0 = (g0 * tl.sigmoid(g0)).to(tl.bfloat16) * u0

    g1 = tl.load(GateUp + base + 2048 + col).to(tl.float32)
    u1 = tl.load(GateUp + base + 6144 + 2048 + col)
    v1 = (g1 * tl.sigmoid(g1)).to(tl.bfloat16) * u1

    g2 = tl.load(GateUp + base + 4096 + col).to(tl.float32)
    u2 = tl.load(GateUp + base + 6144 + 4096 + col)
    v2 = (g2 * tl.sigmoid(g2)).to(tl.bfloat16) * u2

    p0 = tl.max(tl.abs(v0.to(tl.float32)), 0)
    p1 = tl.max(tl.abs(v1.to(tl.float32)), 0)
    p2 = tl.max(tl.abs(v2.to(tl.float32)), 0)
    peak = tl.maximum(tl.maximum(p0, p1), p2)
    scale = tl.where(peak > 0.0, peak * (1.0 / 127.0), 1.0)
    inv_scale = tl.where(peak > 0.0, 127.0 / peak, 0.0)
    tl.store(Scales + row, scale)

    tl.store(
        Codes + row * 6144 + col,
        tl.extra.cuda.libdevice.rint(v0.to(tl.float32) * inv_scale)
        .to(tl.int32)
        .to(tl.int8),
    )
    tl.store(
        Codes + row * 6144 + 2048 + col,
        tl.extra.cuda.libdevice.rint(v1.to(tl.float32) * inv_scale)
        .to(tl.int32)
        .to(tl.int8),
    )
    tl.store(
        Codes + row * 6144 + 4096 + col,
        tl.extra.cuda.libdevice.rint(v2.to(tl.float32) * inv_scale)
        .to(tl.int32)
        .to(tl.int8),
    )


@triton.jit
def _fused_rmsnorm_bf16_k2048(
    X,
    WNorm,
    ResOut,
    NormOut,
    stride_x,
    eps: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    m = col < K
    x_bf16 = tl.load(X + row * stride_x + col, mask=m, other=0.0)
    tl.store(ResOut + row * K + col, x_bf16, mask=m)
    x_f32 = x_bf16.to(tl.float32)
    var = tl.sum(x_f32 * x_f32, 0) * (1.0 / K)
    inv_rms = tl.rsqrt(var + eps)
    wn = tl.load(WNorm + col, mask=m, other=0.0)
    normed = ((x_f32 * inv_rms).to(tl.bfloat16) * wn).to(tl.bfloat16)
    tl.store(NormOut + row * K + col, normed, mask=m)


@triton.jit
def _fused_add_rmsnorm_bf16_k2048(
    X,
    Res,
    WNorm,
    ResOut,
    NormOut,
    stride_x,
    stride_res,
    eps: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    m = col < K
    x_f32 = tl.load(X + row * stride_x + col, mask=m, other=0.0).to(tl.float32)
    r_f32 = tl.load(Res + row * stride_res + col, mask=m, other=0.0).to(tl.float32)
    s_f32 = x_f32 + r_f32
    tl.store(ResOut + row * K + col, s_f32.to(tl.bfloat16), mask=m)
    var = tl.sum(s_f32 * s_f32, 0) * (1.0 / K)
    inv_rms = tl.rsqrt(var + eps)
    wn = tl.load(WNorm + col, mask=m, other=0.0)
    normed = ((s_f32 * inv_rms).to(tl.bfloat16) * wn).to(tl.bfloat16)
    tl.store(NormOut + row * K + col, normed, mask=m)


@triton.jit
def _fused_rmsnorm_quant_k2048(
    X,
    WNorm,
    ResOut,
    Codes,
    Scales,
    stride_x,
    eps: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    m = col < K
    x_bf16 = tl.load(X + row * stride_x + col, mask=m, other=0.0)
    tl.store(ResOut + row * K + col, x_bf16, mask=m)
    x_f32 = x_bf16.to(tl.float32)
    var = tl.sum(x_f32 * x_f32, 0) * (1.0 / K)
    inv_rms = tl.rsqrt(var + eps)
    wn = tl.load(WNorm + col, mask=m, other=0.0)
    normed = ((x_f32 * inv_rms).to(tl.bfloat16) * wn).to(tl.bfloat16).to(tl.float32)
    peak = tl.max(tl.abs(normed), 0)
    scale = tl.where(peak > 0.0, peak * (1.0 / 127.0), 1.0)
    inv_scale = tl.where(peak > 0.0, 127.0 / peak, 0.0)
    q = tl.extra.cuda.libdevice.rint(normed * inv_scale).to(tl.int32).to(tl.int8)
    tl.store(Codes + row * K + col, q, mask=m)
    tl.store(Scales + row, scale)


def input_norm_pack_forward(
    hidden_states: torch.Tensor,
    w_inorm: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Layer-0 input RMSNorm that outputs packed INT8 (or BF16) + scales + residual for chained attention."""
    rows = hidden_states.shape[0]
    device = hidden_states.device
    x_pack = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
    s_pack = torch.empty(rows, dtype=torch.float32, device=device)
    if rows == 0:
        return x_pack, s_pack, hidden_states.clone()
    if (
        hidden_states.dtype != torch.bfloat16
        or hidden_states.stride(1) != 1
        or not 1 <= rows <= 4096
    ):
        res_out = hidden_states.clone()
        sf = hidden_states.float()
        var = sf.pow(2).mean(dim=-1, keepdim=True)
        normed = (sf * torch.rsqrt(var + eps)).to(torch.bfloat16) * w_inorm
        return normed.contiguous(), s_pack, res_out

    res_out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
    with torch_device_fn.device(device):
        if use_packed_i8_qkv(rows):
            c2048 = _pack_int8_view(x_pack)
            _fused_rmsnorm_quant_k2048[(rows,)](
                hidden_states,
                w_inorm,
                res_out,
                c2048,
                s_pack,
                hidden_states.stride(0),
                eps=eps,
                K=2048,
                BLOCK=2048,
                num_warps=4,
                num_stages=1,
            )
        else:
            _fused_rmsnorm_bf16_k2048[(rows,)](
                hidden_states,
                w_inorm,
                res_out,
                x_pack,
                hidden_states.stride(0),
                eps=eps,
                K=2048,
                BLOCK=2048,
                num_warps=4,
                num_stages=1,
            )
    return x_pack, s_pack, res_out


def qkv_pack_and_rope_cache_forward(
    x_pack: torch.Tensor,
    s_pack: torch.Tensor,
    w_qkv: torch.Tensor,
    c_qkv: torch.Tensor,
    s_qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    return_contig_kv: bool = False,
    int8_kv: bool = False,
    return_q_i8: bool = False,
):
    """Execute QKV projection from packed norm output + NeoX RoPE + paged KV cache write."""
    rows = x_pack.shape[0]
    device = x_pack.device
    if rows == 0:
        q_empty = torch.empty((0, 16, 128), dtype=torch.bfloat16, device=device)
        if return_contig_kv:
            kv_empty = torch.empty((0, 2, 128), dtype=torch.bfloat16, device=device)
            return q_empty, kv_empty, kv_empty
        if return_q_i8:
            return (
                q_empty,
                torch.empty((0, 16, 128), dtype=torch.int8, device=device),
                torch.empty((0, 16), dtype=torch.float32, device=device),
            )
        return q_empty

    if use_packed_i8_qkv(rows):
        c2048 = _pack_int8_view(x_pack)
        if 8 <= rows <= 64 and not return_contig_kv and (not int8_kv or return_q_i8):
            # Ultra-fast Decode path: Split-K=2 INT32 partials fused directly into RoPE + KV cache write!
            part_qkv = project_codes_splitk_part(c2048, c_qkv, 2)
            if int8_kv:
                return fused_qkv_splitk2_rope_and_cache_i8(
                    part_qkv,
                    s_pack,
                    s_qkv,
                    positions,
                    cos_sin_cache,
                    key_cache,
                    value_cache,
                    slot_mapping,
                )
            return fused_qkv_splitk2_rope_and_cache(
                part_qkv,
                s_pack,
                s_qkv,
                positions,
                cos_sin_cache,
                key_cache,
                value_cache,
                slot_mapping,
            )
        if 8 <= rows <= 64:
            qkv = project_codes_splitk(
                c2048, s_pack, c_qkv, s_qkv, splits=2, out_dtype=torch.bfloat16
            )
        else:
            qkv = project_codes(c2048, s_pack, c_qkv, s_qkv, out_dtype=torch.bfloat16)
    else:
        norm_out = x_pack.view(rows, 2048)
        qkv = torch.empty((rows, 2560), dtype=torch.bfloat16, device=device)
        if not try_metax_kcontig_nt_mm(norm_out, w_qkv, qkv):
            qkv = F.linear(norm_out, w_qkv)

    q, k, v = qkv.split([2048, 256, 256], dim=-1)
    if int8_kv:
        return fused_qkv_rope_and_cache_i8(
            q.view(rows, 16, 128),
            k.view(rows, 2, 128),
            v.view(rows, 2, 128),
            positions,
            cos_sin_cache,
            key_cache,
            value_cache,
            slot_mapping,
            return_contig_kv=return_contig_kv,
            return_q_i8=return_q_i8 and not return_contig_kv,
        )
    return fused_qkv_rope_and_cache(
        q.view(rows, 16, 128),
        k.view(rows, 2, 128),
        v.view(rows, 2, 128),
        positions,
        cos_sin_cache,
        key_cache,
        value_cache,
        slot_mapping,
        return_contig_kv=return_contig_kv,
    )


def post_attn_mlp_next_norm_pack_forward(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    w_o: torch.Tensor,
    c_o: torch.Tensor | None,
    s_o: torch.Tensor | None,
    w_pnorm: torch.Tensor,
    w_gu: torch.Tensor,
    c_gu: torch.Tensor,
    s_gu: torch.Tensor,
    w_down: torch.Tensor,
    c_down: torch.Tensor,
    s_down: torch.Tensor,
    eps_pnorm: float,
    w_next_inorm: torch.Tensor,
    eps_next_inorm: float,
    use_i8_o_prefill: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-layer fused op: layer i (o_proj + post_norm + gate_up + SwiGLU + down_proj) + layer i+1
    (residual + input_norm + pack)."""
    rows = attn_out.shape[0]
    device = attn_out.device
    x_pack = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
    s_pack = torch.empty(rows, dtype=torch.float32, device=device)
    if rows == 0:
        return x_pack, s_pack, residual.clone()

    if (
        not 1 <= rows <= 4096
        or attn_out.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or attn_out.stride(1) != 1
        or residual.stride(1) != 1
    ):
        mlp_out, res_mid = post_attn_mlp_forward(
            attn_out,
            residual,
            w_o,
            c_o,
            s_o,
            w_pnorm,
            w_gu,
            c_gu,
            s_gu,
            w_down,
            c_down,
            s_down,
            eps=eps_pnorm,
            use_i8_o_prefill=use_i8_o_prefill,
        )
        sf = mlp_out.float() + res_mid.float()
        res_next = sf.to(torch.bfloat16)
        var = sf.pow(2).mean(dim=-1, keepdim=True)
        normed = (sf * torch.rsqrt(var + eps_next_inorm)).to(
            torch.bfloat16
        ) * w_next_inorm
        return normed.contiguous(), s_pack, res_next

    decode_i8_down = os.environ.get("FL_C500_DECODE_I8_DOWN", "1") != "0"
    next_i8 = use_packed_i8_qkv(rows)

    with torch_device_fn.device(device):
        res_mid = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
        res_next = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)

        if rows <= 64:
            bm = 16 if rows <= 16 else (32 if rows <= 32 else 64)
            bk = 128 if 16 < rows <= 32 else 64
            waves = 4 if rows <= 16 else 8
            part_o = torch.empty((3, rows, 2048), dtype=torch.float32, device=device)
            _metax_nt_kcontig_part_kernel[(triton.cdiv(rows, bm) * 32, 3)](
                attn_out,
                w_o,
                res_mid,
                part_o,
                rows,
                2048,
                2048,
                attn_out.stride(0),
                w_o.stride(0),
                2048,
                BM=bm,
                BN=64,
                BK=bk,
                SPLIT_K=3,
                GROUP_M=1,
                EVEN_M=rows % bm == 0,
                EVEN_N=True,
                EVEN_K=True,
                num_warps=waves,
                num_stages=3,
                pipeline="basic",
            )
            if rows >= 8:
                c2048 = torch.empty((rows, 2048), dtype=torch.int8, device=device)
                s2048 = torch.empty(rows, dtype=torch.float32, device=device)
                _fused_splitk_add_rmsnorm_quant_k2048[(rows,)](
                    part_o,
                    residual,
                    w_pnorm,
                    res_mid,
                    c2048,
                    s2048,
                    rows,
                    residual.stride(0),
                    eps=eps_pnorm,
                    SK=3,
                    K=2048,
                    BLOCK=2048,
                    num_warps=4,
                    num_stages=1,
                )
                gu = project_codes(c2048, s2048, c_gu, s_gu, out_dtype=torch.bfloat16)
            else:
                norm_out = torch.empty(
                    (rows, 2048), dtype=torch.bfloat16, device=device
                )
                _fused_splitk_add_rmsnorm_bf16_k2048[(rows,)](
                    part_o,
                    residual,
                    w_pnorm,
                    res_mid,
                    norm_out,
                    rows,
                    residual.stride(0),
                    eps=eps_pnorm,
                    SK=3,
                    K=2048,
                    BLOCK=2048,
                    num_warps=4,
                    num_stages=1,
                )
                gu = torch.empty((rows, 12288), dtype=torch.bfloat16, device=device)
                if not try_metax_kcontig_nt_mm(norm_out, w_gu, gu):
                    gu = F.linear(norm_out, w_gu)

            # Ultra-fast Decode cross-layer fusion: Group-2048 SwiGLU+Quant -> down_proj(SplitK-3)
            # -> Fused Reduce+Residual+NextNorm+Quant!
            if decode_i8_down and rows >= 8 and next_i8:
                c6144 = torch.empty((rows, 6144), dtype=torch.int8, device=device)
                s6144_g = torch.empty((3, rows), dtype=torch.float32, device=device)
                _fused_swiglu_quant_g2048[(rows, 3)](
                    gu, c6144, s6144_g, rows, num_warps=4, num_stages=1
                )
                part_d = project_codes_splitk_part(c6144, c_down, 3)
                c_next = _pack_int8_view(x_pack)
                _fused_g2048_splitk3_add_rmsnorm_quant_k2048[(rows,)](
                    part_d,
                    s6144_g,
                    s_down,
                    res_mid,
                    w_next_inorm,
                    res_next,
                    c_next,
                    s_pack,
                    rows,
                    2048,
                    eps=eps_next_inorm,
                    K=2048,
                    BLOCK=2048,
                    num_warps=4,
                    num_stages=1,
                )
                return x_pack, s_pack, res_next

            if decode_i8_down and rows >= 8:
                c6144 = torch.empty((rows, 6144), dtype=torch.int8, device=device)
                s6144_g = torch.empty((3, rows), dtype=torch.float32, device=device)
                _fused_swiglu_quant_g2048[(rows, 3)](
                    gu, c6144, s6144_g, rows, num_warps=4, num_stages=1
                )
                mlp_out = project_codes_g2048_k6144(
                    c6144, s6144_g, c_down, s_down, out_dtype=torch.bfloat16
                )
            else:
                mid = torch.empty((rows, 6144), dtype=torch.bfloat16, device=device)
                _fast_swiglu_bf16[(triton.cdiv(rows * 6144, 2048),)](
                    gu,
                    mid,
                    rows,
                    K=6144,
                    BLOCK=2048,
                    num_warps=4,
                    num_stages=1,
                )
                mlp_out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
                if not try_metax_kcontig_nt_mm(mid, w_down, mlp_out):
                    mlp_out = F.linear(mid, w_down)

            if next_i8:
                c_next = _pack_int8_view(x_pack)
                _fused_add_rmsnorm_quant_k2048[(rows,)](
                    mlp_out,
                    res_mid,
                    w_next_inorm,
                    res_next,
                    c_next,
                    s_pack,
                    mlp_out.stride(0),
                    2048,
                    eps=eps_next_inorm,
                    K=2048,
                    BLOCK=2048,
                    num_warps=4,
                    num_stages=1,
                )
            else:
                _fused_add_rmsnorm_bf16_k2048[(rows,)](
                    mlp_out,
                    res_mid,
                    w_next_inorm,
                    res_next,
                    x_pack,
                    mlp_out.stride(0),
                    2048,
                    eps=eps_next_inorm,
                    K=2048,
                    BLOCK=2048,
                    num_warps=4,
                    num_stages=1,
                )
            return x_pack, s_pack, res_next

        # Medium / Large Prefill & mixed-batch path (65 <= rows <= 4096):
        if use_i8_o_prefill and rows >= 256 and c_o is not None and s_o is not None:
            o_out = project(attn_out, c_o, s_o)
        else:
            o_out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
            if not try_metax_kcontig_nt_mm(attn_out, w_o, o_out):
                o_out = F.linear(attn_out, w_o)

        c2048 = torch.empty((rows, 2048), dtype=torch.int8, device=device)
        s2048 = torch.empty(rows, dtype=torch.float32, device=device)
        _fused_add_rmsnorm_quant_k2048[(rows,)](
            o_out,
            residual,
            w_pnorm,
            res_mid,
            c2048,
            s2048,
            o_out.stride(0),
            residual.stride(0),
            eps=eps_pnorm,
            K=2048,
            BLOCK=2048,
            num_warps=4,
            num_stages=1,
        )
        gu = project_codes(c2048, s2048, c_gu, s_gu, out_dtype=torch.bfloat16)

        if rows < 512:
            mid = torch.empty((rows, 6144), dtype=torch.bfloat16, device=device)
            _fast_swiglu_bf16[(triton.cdiv(rows * 6144, 1024),)](
                gu,
                mid,
                rows,
                K=6144,
                BLOCK=1024,
                num_warps=8,
                num_stages=1,
            )
            mlp_out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
            if not try_metax_kcontig_nt_mm(mid, w_down, mlp_out):
                mlp_out = F.linear(mid, w_down)
        else:
            c6144 = torch.empty((rows, 6144), dtype=torch.int8, device=device)
            s6144 = torch.empty(rows, dtype=torch.float32, device=device)
            _fused_swiglu_quant_3x2048[(rows,)](
                gu, c6144, s6144, num_warps=4, num_stages=1
            )
            mlp_out = project_codes(
                c6144, s6144, c_down, s_down, out_dtype=torch.bfloat16
            )

        if next_i8:
            c_next = _pack_int8_view(x_pack)
            _fused_add_rmsnorm_quant_k2048[(rows,)](
                mlp_out,
                res_mid,
                w_next_inorm,
                res_next,
                c_next,
                s_pack,
                mlp_out.stride(0),
                2048,
                eps=eps_next_inorm,
                K=2048,
                BLOCK=2048,
                num_warps=4,
                num_stages=1,
            )
        else:
            _fused_add_rmsnorm_bf16_k2048[(rows,)](
                mlp_out,
                res_mid,
                w_next_inorm,
                res_next,
                x_pack,
                mlp_out.stride(0),
                2048,
                eps=eps_next_inorm,
                K=2048,
                BLOCK=2048,
                num_warps=4,
                num_stages=1,
            )
        return x_pack, s_pack, res_next


def post_attn_mlp_final_norm_forward(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    w_o: torch.Tensor,
    c_o: torch.Tensor | None,
    s_o: torch.Tensor | None,
    w_pnorm: torch.Tensor,
    w_gu: torch.Tensor,
    c_gu: torch.Tensor,
    s_gu: torch.Tensor,
    w_down: torch.Tensor,
    c_down: torch.Tensor,
    s_down: torch.Tensor,
    eps_pnorm: float,
    w_final_norm: torch.Tensor,
    eps_final_norm: float,
    use_i8_o_prefill: bool = True,
) -> torch.Tensor:
    """Final decoder layer + model.norm fused op (eliminates last-layer down_proj reduce + final RMSNorm kernel)."""
    rows = attn_out.shape[0]
    device = attn_out.device
    if rows == 0:
        return attn_out.clone()

    decode_i8_down = os.environ.get("FL_C500_DECODE_I8_DOWN", "1") != "0"
    if (
        decode_i8_down
        and 8 <= rows <= 64
        and attn_out.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and attn_out.stride(1) == 1
        and residual.stride(1) == 1
    ):
        with torch_device_fn.device(device):
            res_mid = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
            bm = 16 if rows <= 16 else (32 if rows <= 32 else 64)
            bk = 128 if 16 < rows <= 32 else 64
            waves = 4 if rows <= 16 else 8
            part_o = torch.empty((3, rows, 2048), dtype=torch.float32, device=device)
            _metax_nt_kcontig_part_kernel[(triton.cdiv(rows, bm) * 32, 3)](
                attn_out,
                w_o,
                res_mid,
                part_o,
                rows,
                2048,
                2048,
                attn_out.stride(0),
                w_o.stride(0),
                2048,
                BM=bm,
                BN=64,
                BK=bk,
                SPLIT_K=3,
                GROUP_M=1,
                EVEN_M=rows % bm == 0,
                EVEN_N=True,
                EVEN_K=True,
                num_warps=waves,
                num_stages=3,
                pipeline="basic",
            )
            c2048 = torch.empty((rows, 2048), dtype=torch.int8, device=device)
            s2048 = torch.empty(rows, dtype=torch.float32, device=device)
            _fused_splitk_add_rmsnorm_quant_k2048[(rows,)](
                part_o,
                residual,
                w_pnorm,
                res_mid,
                c2048,
                s2048,
                rows,
                residual.stride(0),
                eps=eps_pnorm,
                SK=3,
                K=2048,
                BLOCK=2048,
                num_warps=4,
                num_stages=1,
            )
            gu = project_codes(c2048, s2048, c_gu, s_gu, out_dtype=torch.bfloat16)
            c6144 = torch.empty((rows, 6144), dtype=torch.int8, device=device)
            s6144_g = torch.empty((3, rows), dtype=torch.float32, device=device)
            _fused_swiglu_quant_g2048[(rows, 3)](
                gu, c6144, s6144_g, rows, num_warps=4, num_stages=1
            )
            part_d = project_codes_splitk_part(c6144, c_down, 3)
            final_out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
            _fused_g2048_splitk3_add_rmsnorm_bf16_k2048[(rows,)](
                part_d,
                s6144_g,
                s_down,
                res_mid,
                w_final_norm,
                final_out,
                rows,
                2048,
                eps=eps_final_norm,
                K=2048,
                BLOCK=2048,
                num_warps=4,
                num_stages=1,
            )
            return final_out

    mlp_out, res_mid = post_attn_mlp_forward(
        attn_out,
        residual,
        w_o,
        c_o,
        s_o,
        w_pnorm,
        w_gu,
        c_gu,
        s_gu,
        w_down,
        c_down,
        s_down,
        eps=eps_pnorm,
        use_i8_o_prefill=use_i8_o_prefill,
    )
    if 1 <= rows <= 4096 and mlp_out.dtype == torch.bfloat16 and mlp_out.stride(1) == 1:
        res_dummy = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
        final_out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
        with torch_device_fn.device(device):
            _fused_add_rmsnorm_bf16_k2048[(rows,)](
                mlp_out,
                res_mid,
                w_final_norm,
                res_dummy,
                final_out,
                mlp_out.stride(0),
                res_mid.stride(0),
                eps=eps_final_norm,
                K=2048,
                BLOCK=2048,
                num_warps=4,
                num_stages=1,
            )
        return final_out
    sf = mlp_out.float() + res_mid.float()
    var = sf.pow(2).mean(dim=-1, keepdim=True)
    return (sf * torch.rsqrt(var + eps_final_norm)).to(torch.bfloat16) * w_final_norm


def input_norm_qkv_forward(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    w_inorm: torch.Tensor,
    w_qkv: torch.Tensor,
    c_qkv: torch.Tensor | None = None,
    s_qkv: torch.Tensor | None = None,
    eps: float = 1e-6,
    use_i8_qkv_prefill: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = hidden_states.shape[0]
    device = hidden_states.device
    if rows == 0:
        return (
            hidden_states.new_empty((0, 2560)),
            hidden_states.clone() if residual is None else residual.clone(),
        )

    if (
        not 1 <= rows <= 4096
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.stride(1) != 1
        or (
            residual is not None
            and (residual.dtype != torch.bfloat16 or residual.stride(1) != 1)
        )
    ):
        if residual is None:
            res_out = hidden_states.clone()
            sf = hidden_states.float()
        else:
            sf = hidden_states.float() + residual.float()
            res_out = sf.to(torch.bfloat16)
        var = sf.pow(2).mean(dim=-1, keepdim=True)
        normed = (sf * torch.rsqrt(var + eps)).to(torch.bfloat16) * w_inorm
        return F.linear(normed, w_qkv), res_out

    decode_i8_qkv = os.environ.get("FL_C500_DECODE_I8_QKV", "1") != "0"

    with torch_device_fn.device(device):
        res_out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
        if (
            use_i8_qkv_prefill
            and rows >= 256
            and c_qkv is not None
            and s_qkv is not None
        ):
            c2048 = torch.empty((rows, 2048), dtype=torch.int8, device=device)
            s2048 = torch.empty(rows, dtype=torch.float32, device=device)
            if residual is None:
                _fused_rmsnorm_quant_k2048[(rows,)](
                    hidden_states,
                    w_inorm,
                    res_out,
                    c2048,
                    s2048,
                    hidden_states.stride(0),
                    eps=eps,
                    K=2048,
                    BLOCK=2048,
                    num_warps=4,
                    num_stages=1,
                )
            else:
                _fused_add_rmsnorm_quant_k2048[(rows,)](
                    hidden_states,
                    residual,
                    w_inorm,
                    res_out,
                    c2048,
                    s2048,
                    hidden_states.stride(0),
                    residual.stride(0),
                    eps=eps,
                    K=2048,
                    BLOCK=2048,
                    num_warps=4,
                    num_stages=1,
                )
            qkv = project_codes(c2048, s2048, c_qkv, s_qkv, out_dtype=torch.bfloat16)
            return qkv, res_out

        if (
            decode_i8_qkv
            and 8 <= rows <= 64
            and c_qkv is not None
            and s_qkv is not None
        ):
            c2048 = torch.empty((rows, 2048), dtype=torch.int8, device=device)
            s2048 = torch.empty(rows, dtype=torch.float32, device=device)
            if residual is None:
                _fused_rmsnorm_quant_k2048[(rows,)](
                    hidden_states,
                    w_inorm,
                    res_out,
                    c2048,
                    s2048,
                    hidden_states.stride(0),
                    eps=eps,
                    K=2048,
                    BLOCK=2048,
                    num_warps=4,
                    num_stages=1,
                )
            else:
                _fused_add_rmsnorm_quant_k2048[(rows,)](
                    hidden_states,
                    residual,
                    w_inorm,
                    res_out,
                    c2048,
                    s2048,
                    hidden_states.stride(0),
                    residual.stride(0),
                    eps=eps,
                    K=2048,
                    BLOCK=2048,
                    num_warps=4,
                    num_stages=1,
                )
            qkv = project_codes_splitk(
                c2048, s2048, c_qkv, s_qkv, splits=2, out_dtype=torch.bfloat16
            )
            return qkv, res_out

        norm_out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
        if residual is None:
            _fused_rmsnorm_bf16_k2048[(rows,)](
                hidden_states,
                w_inorm,
                res_out,
                norm_out,
                hidden_states.stride(0),
                eps=eps,
                K=2048,
                BLOCK=2048,
                num_warps=4,
                num_stages=1,
            )
        else:
            _fused_add_rmsnorm_bf16_k2048[(rows,)](
                hidden_states,
                residual,
                w_inorm,
                res_out,
                norm_out,
                hidden_states.stride(0),
                residual.stride(0),
                eps=eps,
                K=2048,
                BLOCK=2048,
                num_warps=4,
                num_stages=1,
            )
        qkv = torch.empty((rows, 2560), dtype=torch.bfloat16, device=device)
        if not try_metax_kcontig_nt_mm(norm_out, w_qkv, qkv):
            qkv = F.linear(norm_out, w_qkv)
        return qkv, res_out


def post_attn_mlp_forward(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    w_o: torch.Tensor,
    c_o: torch.Tensor | None,
    s_o: torch.Tensor | None,
    w_pnorm: torch.Tensor,
    w_gu: torch.Tensor,
    c_gu: torch.Tensor,
    s_gu: torch.Tensor,
    w_down: torch.Tensor,
    c_down: torch.Tensor,
    s_down: torch.Tensor,
    eps: float = 1e-6,
    use_i8_o_prefill: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = attn_out.shape[0]
    if rows == 0:
        return attn_out.clone(), residual.clone()

    if (
        not 1 <= rows <= 4096
        or attn_out.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or attn_out.stride(1) != 1
        or residual.stride(1) != 1
    ):
        o_out = F.linear(attn_out, w_o)
        s_f32 = o_out.float() + residual.float()
        res_out = s_f32.to(torch.bfloat16)
        var = s_f32.pow(2).mean(dim=-1, keepdim=True)
        normed = (s_f32 * torch.rsqrt(var + eps)).to(torch.bfloat16) * w_pnorm
        gu = F.linear(normed, w_gu)
        mid = F.silu(gu[:, :6144]) * gu[:, 6144:]
        d_out = F.linear(mid, w_down)
        return d_out, res_out

    decode_i8_down = os.environ.get("FL_C500_DECODE_I8_DOWN", "1") != "0"

    device = attn_out.device
    with torch_device_fn.device(device):
        res_out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)

        if rows <= 64:
            bm = 16 if rows <= 16 else (32 if rows <= 32 else 64)
            bk = 128 if 16 < rows <= 32 else 64
            waves = 4 if rows <= 16 else 8
            part_o = torch.empty((3, rows, 2048), dtype=torch.float32, device=device)
            _metax_nt_kcontig_part_kernel[(triton.cdiv(rows, bm) * 32, 3)](
                attn_out,
                w_o,
                res_out,
                part_o,
                rows,
                2048,
                2048,
                attn_out.stride(0),
                w_o.stride(0),
                2048,
                BM=bm,
                BN=64,
                BK=bk,
                SPLIT_K=3,
                GROUP_M=1,
                EVEN_M=rows % bm == 0,
                EVEN_N=True,
                EVEN_K=True,
                num_warps=waves,
                num_stages=3,
                pipeline="basic",
            )
            if rows >= 8:
                c2048 = torch.empty((rows, 2048), dtype=torch.int8, device=device)
                s2048 = torch.empty(rows, dtype=torch.float32, device=device)
                _fused_splitk_add_rmsnorm_quant_k2048[(rows,)](
                    part_o,
                    residual,
                    w_pnorm,
                    res_out,
                    c2048,
                    s2048,
                    rows,
                    residual.stride(0),
                    eps=eps,
                    SK=3,
                    K=2048,
                    BLOCK=2048,
                    num_warps=4,
                    num_stages=1,
                )
                gu = project_codes(c2048, s2048, c_gu, s_gu, out_dtype=torch.bfloat16)
            else:
                norm_out = torch.empty(
                    (rows, 2048), dtype=torch.bfloat16, device=device
                )
                _fused_splitk_add_rmsnorm_bf16_k2048[(rows,)](
                    part_o,
                    residual,
                    w_pnorm,
                    res_out,
                    norm_out,
                    rows,
                    residual.stride(0),
                    eps=eps,
                    SK=3,
                    K=2048,
                    BLOCK=2048,
                    num_warps=4,
                    num_stages=1,
                )
                gu = torch.empty((rows, 12288), dtype=torch.bfloat16, device=device)
                if not try_metax_kcontig_nt_mm(norm_out, w_gu, gu):
                    gu = F.linear(norm_out, w_gu)

            if decode_i8_down and rows >= 8:
                c6144 = torch.empty((rows, 6144), dtype=torch.int8, device=device)
                s6144_g = torch.empty((3, rows), dtype=torch.float32, device=device)
                _fused_swiglu_quant_g2048[(rows, 3)](
                    gu, c6144, s6144_g, rows, num_warps=4, num_stages=1
                )
                mlp_out = project_codes_g2048_k6144(
                    c6144, s6144_g, c_down, s_down, out_dtype=torch.bfloat16
                )
                return mlp_out, res_out

            mid = torch.empty((rows, 6144), dtype=torch.bfloat16, device=device)
            _fast_swiglu_bf16[(triton.cdiv(rows * 6144, 2048),)](
                gu,
                mid,
                rows,
                K=6144,
                BLOCK=2048,
                num_warps=4,
                num_stages=1,
            )
            mlp_out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
            if not try_metax_kcontig_nt_mm(mid, w_down, mlp_out):
                mlp_out = F.linear(mid, w_down)
            return mlp_out, res_out

        if use_i8_o_prefill and rows >= 256 and c_o is not None and s_o is not None:
            o_out = project(attn_out, c_o, s_o)
        else:
            o_out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
            if not try_metax_kcontig_nt_mm(attn_out, w_o, o_out):
                o_out = F.linear(attn_out, w_o)

        c2048 = torch.empty((rows, 2048), dtype=torch.int8, device=device)
        s2048 = torch.empty(rows, dtype=torch.float32, device=device)
        _fused_add_rmsnorm_quant_k2048[(rows,)](
            o_out,
            residual,
            w_pnorm,
            res_out,
            c2048,
            s2048,
            o_out.stride(0),
            residual.stride(0),
            eps=eps,
            K=2048,
            BLOCK=2048,
            num_warps=4,
            num_stages=1,
        )
        gu = project_codes(c2048, s2048, c_gu, s_gu, out_dtype=torch.bfloat16)

        if rows < 512:
            mid = torch.empty((rows, 6144), dtype=torch.bfloat16, device=device)
            _fast_swiglu_bf16[(triton.cdiv(rows * 6144, 1024),)](
                gu,
                mid,
                rows,
                K=6144,
                BLOCK=1024,
                num_warps=8,
                num_stages=1,
            )
            mlp_out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
            if not try_metax_kcontig_nt_mm(mid, w_down, mlp_out):
                mlp_out = F.linear(mid, w_down)
            return mlp_out, res_out

        c6144 = torch.empty((rows, 6144), dtype=torch.int8, device=device)
        s6144 = torch.empty(rows, dtype=torch.float32, device=device)
        _fused_swiglu_quant_3x2048[(rows,)](gu, c6144, s6144, num_warps=4, num_stages=1)
        mlp_out = project_codes(c6144, s6144, c_down, s_down, out_dtype=torch.bfloat16)
        return mlp_out, res_out
