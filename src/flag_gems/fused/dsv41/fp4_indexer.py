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

# Portions adapted from SGLang (Copyright 2023-2026 SGLang Team).
"""Framework-independent E2M1/UE8M0 decoding and explicit-slot index scoring.

Page bytes: [PAGE*64 payload bytes | PAGE*4 UE8M0 scale bytes]. Each
payload byte encodes even/odd E2M1 nibbles; four scales cover 128 elements.
No request-to-token mapping, compressed slots or candidate layout is accepted.
"""

import torch
import triton
import triton.language as tl

INDEX_HEAD_DIM = 128
PAYLOAD_BYTES = tl.constexpr(64)
SCALE_BYTES = tl.constexpr(4)
FP4_INDEX_SCORE_ABI = 1


@triton.jit
def _e2m1_decode(code):
    # E2M1 magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6. Their exact FP32
    # encodings avoid per-element exponentiation and floating-point decoding.
    u = code.to(tl.uint32)
    mag = u & 7
    bits = tl.where(
        mag == 0, 0, tl.where(mag == 1, 0x3F000000, 0x3F000000 + (mag << 22))
    )
    v = (bits | ((u & 8) << 28)).to(tl.float32, bitcast=True)
    # Legacy decoding canonicalizes FP4 -0 to +0. An explicit exact add keeps
    # that behavior without an extra predicate/live range in the score kernel.
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, 0f00000000;",
        constraints="=f,f",
        args=[v],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def fp4_weighted_head_score(q_even, q_odd, k_low, k_high, weights):
    """BF16 dot/relu/product/head-sum rounding, with FP32 accumulator."""
    acc = tl.dot(q_even, tl.trans(k_low))
    acc += tl.dot(q_odd, tl.trans(k_high))
    score = tl.maximum(acc.to(tl.bfloat16).to(tl.float32), 0.0)
    score = (score * weights[:, None].to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    return tl.sum(score, 0).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _fp4_slot_score_kernel(
    q_ptr,
    w_ptr,
    slots_ptr,
    lens_ptr,
    table_ptr,
    out_ptr,
    L,
    page_size,
    row_stride,
    stride_qb,
    stride_qh,
    stride_wb,
    stride_out,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(0, HALF_D)
    n_vis = tl.load(lens_ptr + b)
    if lb * BLOCK_L >= tl.minimum(n_vis, L):
        tl.store(out_ptr + b * stride_out + offs_l, -float("inf"), offs_l < L)
        return
    valid = offs_l < tl.minimum(n_vis, L)
    slot = tl.load(slots_ptr + b * L + offs_l, mask=offs_l < L, other=0).to(tl.int64)
    page = slot // page_size
    off = slot % page_size
    row_base = page * row_stride
    pay = tl.load(
        table_ptr + row_base[:, None] + off[:, None] * PAYLOAD_BYTES + offs_i[None, :],
        mask=valid[:, None],
        other=0,
    )
    low = _e2m1_decode(pay & 15)
    high = _e2m1_decode(pay >> 4 & 15)
    sc_idx = offs_i // 16
    exps = tl.load(
        table_ptr
        + row_base[:, None]
        + page_size * PAYLOAD_BYTES
        + off[:, None] * SCALE_BYTES
        + sc_idx[None, :],
        mask=valid[:, None],
        other=127,
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    k_low = (low * scale).to(tl.bfloat16)
    k_high = (high * scale).to(tl.bfloat16)
    q_even = tl.load(
        q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + 2 * offs_i[None, :]
    )
    q_odd = tl.load(
        q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + 2 * offs_i[None, :] + 1
    )
    w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
    logit = fp4_weighted_head_score(q_even, q_odd, k_low, k_high, w)
    logit = tl.where(valid, logit, float("-inf"))
    tl.store(out_ptr + b * stride_out + offs_l, logit, mask=offs_l < L)


def fp4_index_logits_decode(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """q [B, H, 128] bf16, weights [B, H], slots [B, L], lens [B] int64,
    table = the layer's fp4 index-K page buffer (uint8, 2D). Returns [B, L] fp32
    logits with -inf at positions >= lens."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    B, H, _ = q.shape
    assert weights.shape == (B, H) and slots.ndim == 2 and slots.shape[0] == B
    assert lens.shape == (B,) and page_size > 0
    assert slots.dtype in (torch.int32, torch.int64)
    assert table.stride(1) == 1 and table.shape[1] == page_size * 68
    assert q.is_cuda and all(
        x.device == q.device for x in (weights, slots, lens, table)
    )
    L = slots.shape[1]
    assert table.dtype == torch.uint8 and table.dim() == 2
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    slots = slots.contiguous()
    out_storage = torch.empty(
        (B, triton.cdiv(L, 4) * 4), dtype=torch.float32, device=q.device
    )
    out = out_storage[:, :L]
    if L == 0:
        return out
    BLOCK_L = 64
    grid = (B, triton.cdiv(L, BLOCK_L))
    _fp4_slot_score_kernel[grid](
        q,
        weights,
        slots,
        lens.to(torch.int64).contiguous(),
        table,
        out,
        L,
        page_size,
        table.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        out.stride(0),
        H=H,
        HALF_D=INDEX_HEAD_DIM // 2,
        BLOCK_L=BLOCK_L,
        num_warps=4,
    )
    return out


__all__ = ["fp4_index_logits_decode"]
