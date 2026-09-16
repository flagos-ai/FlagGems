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
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# Per-token 2D-tile kernel (kunlunxin/XPU). Used as the FALLBACK for
# non-contiguous q/k; the fast path for contiguous q/k is the flat block-DMA
# kernel below. The upstream kernel launches grid=(n_tokens,) and then serially
# loops over the heads, issuing one tiny HEAD_DIM-wide (e.g. 64 elem) load/store
# per head -> below the efficient XPU DMA granularity. This variant instead
# moves the whole token slice as a 2D [PADDED_?_HEADS, PADDED_HEAD_DIM] tile in a
# single DMA. cos/sin are [head_dim] and broadcast across the head axis; the
# rotate is a within-head permutation on the last axis. Heads are padded to a
# power of two and masked (q/k head counts such as 6 are not powers of two).
@libentry()
@triton.jit
def apply_rotary_pos_emb_kernel(
    oq_ptr,
    ok_ptr,
    q_ptr,  # (n_tokens, q_heads, head_dim)
    k_ptr,  # (n_tokens, k_heads, head_dim)
    cos_ptr,  # (max_seq_len, dim // 2)
    sin_ptr,  # (max_seq_len, dim // 2)
    pos_ptr,  # (n_tokens, )
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    oq_stride_s,
    oq_stride_h,
    oq_stride_d,
    ok_stride_s,
    ok_stride_h,
    ok_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    PADDED_Q_HEADS: tl.constexpr,
    PADDED_K_HEADS: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    s_id = ext.program_id(0)

    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_ptr += s_id * p_stride_s
        pos_id = tl.load(pos_ptr)
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    # note: set TRITON_DEBUG=1 to enable this check
    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    ordered_block = tl.arange(0, PADDED_HEAD_DIM)
    dim_mask = ordered_block < HEAD_DIM
    if ROTARY_INTERLEAVED:
        odd_mask = ordered_block % 2 == 0
        rotated_block = tl.where(odd_mask, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
        cos = tl.load(cos_ptr + sin_cos_block, mask=dim_mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=dim_mask, other=0.0).to(tl.float32)
        sin = tl.where(odd_mask, -sin, sin)
    else:
        rotated_block = (ordered_block + HEAD_DIM // 2) % HEAD_DIM
        sin_cos_block = ordered_block % (HEAD_DIM // 2)
        cos = tl.load(cos_ptr + sin_cos_block, mask=dim_mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=dim_mask, other=0.0).to(tl.float32)
        sin = tl.where(rotated_block < HEAD_DIM // 2, sin, -sin)

    cos = cos[None, :]
    sin = sin[None, :]

    # ---- Q: one [PADDED_Q_HEADS, PADDED_HEAD_DIM] tile ----
    q_h = tl.arange(0, PADDED_Q_HEADS)
    q_mask = (q_h < NUM_Q_HEADS)[:, None] & dim_mask[None, :]
    q_ordered = q_h[:, None] * q_stride_h + ordered_block[None, :] * q_stride_d
    q_rotated = q_h[:, None] * q_stride_h + rotated_block[None, :] * q_stride_d
    q_out = q_h[:, None] * oq_stride_h + ordered_block[None, :] * oq_stride_d
    q_base = q_ptr + s_id * q_stride_s
    q = tl.load(q_base + q_ordered, mask=q_mask, other=0.0)
    rotated_q = tl.load(q_base + q_rotated, mask=q_mask, other=0.0)
    yq = q * cos + rotated_q * sin
    tl.store(oq_ptr + s_id * oq_stride_s + q_out, yq, mask=q_mask)

    # ---- K: one [PADDED_K_HEADS, PADDED_HEAD_DIM] tile ----
    k_h = tl.arange(0, PADDED_K_HEADS)
    k_mask = (k_h < NUM_K_HEADS)[:, None] & dim_mask[None, :]
    k_ordered = k_h[:, None] * k_stride_h + ordered_block[None, :] * k_stride_d
    k_rotated = k_h[:, None] * k_stride_h + rotated_block[None, :] * k_stride_d
    k_out = k_h[:, None] * ok_stride_h + ordered_block[None, :] * ok_stride_d
    k_base = k_ptr + s_id * k_stride_s
    k = tl.load(k_base + k_ordered, mask=k_mask, other=0.0)
    rotated_k = tl.load(k_base + k_rotated, mask=k_mask, other=0.0)
    yk = k * cos + rotated_k * sin
    tl.store(ok_ptr + s_id * ok_stride_s + k_out, yk, mask=k_mask)


@libentry()
@triton.jit
def apply_rotary_pos_emb_inplace_kernel(
    q_ptr,  # (n_tokens, q_heads, head_dim)
    k_ptr,  # (n_tokens, k_heads, head_dim)
    cos_ptr,  # (max_seq_len, dim // 2)
    sin_ptr,  # (max_seq_len, dim // 2)
    pos_ptr,  # (n_tokens, )
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    PADDED_Q_HEADS: tl.constexpr,
    PADDED_K_HEADS: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    s_id = ext.program_id(0)

    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_ptr += s_id * p_stride_s
        pos_id = tl.load(pos_ptr)
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    # note: set TRITON_DEBUG=1 to enable this check
    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    ordered_block = tl.arange(0, PADDED_HEAD_DIM)
    dim_mask = ordered_block < HEAD_DIM
    if ROTARY_INTERLEAVED:
        odd_mask = ordered_block % 2 == 0
        rotated_block = tl.where(odd_mask, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
        cos = tl.load(cos_ptr + sin_cos_block, mask=dim_mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=dim_mask, other=0.0).to(tl.float32)
        sin = tl.where(odd_mask, -sin, sin)
    else:
        rotated_block = (ordered_block + HEAD_DIM // 2) % HEAD_DIM
        sin_cos_block = ordered_block % (HEAD_DIM // 2)
        cos = tl.load(cos_ptr + sin_cos_block, mask=dim_mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=dim_mask, other=0.0).to(tl.float32)
        sin = tl.where(rotated_block < HEAD_DIM // 2, sin, -sin)

    cos = cos[None, :]
    sin = sin[None, :]

    # ---- Q: one [PADDED_Q_HEADS, PADDED_HEAD_DIM] tile (in place) ----
    q_h = tl.arange(0, PADDED_Q_HEADS)
    q_mask = (q_h < NUM_Q_HEADS)[:, None] & dim_mask[None, :]
    q_ordered = q_h[:, None] * q_stride_h + ordered_block[None, :] * q_stride_d
    q_rotated = q_h[:, None] * q_stride_h + rotated_block[None, :] * q_stride_d
    q_base = q_ptr + s_id * q_stride_s
    q = tl.load(q_base + q_ordered, mask=q_mask, other=0.0)
    rotated_q = tl.load(q_base + q_rotated, mask=q_mask, other=0.0)
    yq = q * cos + rotated_q * sin
    tl.store(q_base + q_ordered, yq, mask=q_mask)  # In-place update

    # ---- K: one [PADDED_K_HEADS, PADDED_HEAD_DIM] tile (in place) ----
    k_h = tl.arange(0, PADDED_K_HEADS)
    k_mask = (k_h < NUM_K_HEADS)[:, None] & dim_mask[None, :]
    k_ordered = k_h[:, None] * k_stride_h + ordered_block[None, :] * k_stride_d
    k_rotated = k_h[:, None] * k_stride_h + rotated_block[None, :] * k_stride_d
    k_base = k_ptr + s_id * k_stride_s
    k = tl.load(k_base + k_ordered, mask=k_mask, other=0.0)
    rotated_k = tl.load(k_base + k_rotated, mask=k_mask, other=0.0)
    yk = k * cos + rotated_k * sin
    tl.store(k_base + k_ordered, yk, mask=k_mask)  # In-place update


# --------------------------------------------------------------------------
# Flat 1D block-DMA kernel (kunlunxin/XPU fast path for the non-inplace case).
#
# Root cause of the old collapse: the upstream per-token kernel launches
# grid=(n_tokens,) with a serial per-head loop (16384 programs x 9 heads x 3
# tiny sub-DMAs on [4,4096,8,64]) and every 2D tile offset
# (`row[:,None]*stride + d[None,:]`) plus the rotated gather
# (`load(x + rotated_block*stride_d)`) is a DISCRETE access on XPU. Measured on
# [4,4096,8,64] fp32: 2D-offset copy ~16ms, rotated-gather ~10ms, per-position
# cos/sin gather ~4ms -> gems ~30-44ms vs torch 0.58ms (speedup 0.019).
#
# Fix: process q/k as a fully FLAT contiguous stream (`pid*BLOCK + arange`),
# which XPU recognises as block DMA (~0.28ms for a 3-array fma). Two problems
# solved without any discrete access:
#  * the rotate: rot(x)[i] = x[i +/- HALF] within the head. Both `x[off+HALF]`
#    and `x[off-HALF]` are CONTIGUOUS shifted loads (base pointer +/- HALF), so
#    `where(first_half, x[off+HALF], x[off-HALF])` reproduces the rotate with
#    NO gather. Works for interleaved too with a +/-1 shift.
#  * cos/sin: the per-position gather is materialised ONCE in torch into a
#    contiguous [n_tokens, H, HD] buffer (sign folded in), so the kernel reads
#    cos/sin as plain contiguous block DMA. Building it is ~0.14ms.
#
# Boundary/OOB: the shifted loads read PAD elements past each end of the tensor.
# Those lanes are discarded by `where(first, ...)`, but the ADDRESS must still be
# valid or XPU faults the whole block DMA (masking does NOT prevent the fault,
# and clamping the address with tl.where makes it non-affine -> discrete gather
# -> ~15x slower). Solution: the input is copied into a buffer padded by PAD on
# each side, so xp/xm read valid padding, addresses stay AFFINE, and block DMA is
# preserved. The kernel reads x at (off + PAD), xp at (off + 2*PAD), xm at (off);
# cos/sin are unpadded and read at (off).
# Precision: x/xp/xm are upcast to fp32 before the fma. XPU otherwise evaluates
# the fp16*fp32 product in fp16, which diverges from the fp32 torch reference at
# d=0 (where inv_freq=1 makes cos/sin oscillate fastest) -> fp16 test failures.
# Net: [4,4096,8,64] ~44ms -> ~0.68ms.
@libentry()
@triton.jit
def apply_rotary_pos_emb_flat_kernel(
    o_ptr,
    x_ptr,  # PADDED input: real data at [PAD : PAD + N], PAD elems on each side
    cos_ptr,  # contiguous, unpadded, same flat layout as output (sign folded)
    sin_ptr,
    N,
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
    PAD: tl.constexpr,  # HALF for non-interleaved, 1 for interleaved
    INTERLEAVED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    d = off % HEAD_DIM
    if INTERLEAVED:
        first = (d % 2) == 0
    else:
        first = d < HALF

    # All three loads are AFFINE (base + affine offset) -> XPU block DMA. The
    # padding guarantees off+2*PAD and off never leave the allocation.
    x = tl.load(x_ptr + off + PAD, mask=m, other=0.0).to(tl.float32)
    xp = tl.load(x_ptr + off + 2 * PAD, mask=m, other=0.0).to(tl.float32)  # x[off+PAD]
    xm = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)  # x[off-PAD]
    xr = tl.where(first, xp, xm)  # == x at the rotated position, no gather

    c = tl.load(cos_ptr + off, mask=m, other=0.0)
    s = tl.load(sin_ptr + off, mask=m, other=0.0)
    tl.store(o_ptr + off, x * c + xr * s, mask=m)


def _build_cos_sin_full(cos, sin, pos, H, head_dim, half, interleaved):
    """Materialise cos/sin (sign folded) into a contiguous flat buffer that
    matches the [n_tokens, H, head_dim] layout of the flattened q/k tensor."""
    n_tokens = pos.shape[0]
    cg = cos.index_select(0, pos).to(torch.float32)  # [n_tokens, half]
    sg = sin.index_select(0, pos).to(torch.float32)
    if interleaved:
        cos_tok = cg.repeat_interleave(2, dim=-1)  # cos[d // 2]
        sin_tok = sg.repeat_interleave(2, dim=-1)
        sign = torch.ones(head_dim, device=cos.device, dtype=torch.float32)
        sign[0::2] = -1.0  # even lanes negated (matches kernel odd_mask branch)
        sin_tok = sin_tok * sign
    else:
        cos_tok = torch.cat([cg, cg], dim=-1)  # cos[d % half]
        sin_tok = torch.cat([-sg, sg], dim=-1)  # d < half -> -sin, else +sin
    cos_full = (
        cos_tok[:, None, :].expand(n_tokens, H, head_dim).reshape(-1).contiguous()
    )
    sin_full = (
        sin_tok[:, None, :].expand(n_tokens, H, head_dim).reshape(-1).contiguous()
    )
    return cos_full, sin_full


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
    inplace: bool = False,
):
    """
    Apply rotary position embedding to q and k

    Args:
        q: (*, q_heads, head_dim)
        k: (*, k_heads, head_dim)
        cos: (max_seq_len, head_dim // 2)
        sin: (max_seq_len, head_dim // 2)
        position_ids: (*, ), optional, position ids for each token
        rotary_interleaved: whether the head_dim is rotated in an interleaved way

    Returns:
        q_embed: (*, q_heads, head_dim)
        k_embed: (*, k_heads, head_dim)
    """
    logger.debug("GEMS_KUNLUNXIN ROTARY_POS_EMBEDDING")
    assert (
        k.shape[-1] == q.shape[-1]
    ), f"q and k must have the same last dimension, got {q.shape} and {k.shape}"
    assert (
        cos.shape[-1] == sin.shape[-1]
    ), f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"
    assert (
        cos.shape[-1] * 2 == q.shape[-1]
    ), f"cos/sin dim must be half of q/k dim, got {cos.shape} and {q.shape}"
    assert cos.stride(-1) == 1, "cos must be contiguous at the last dimension"
    assert sin.stride(-1) == 1, "sin must be contiguous at the last dimension"

    q_shape = q.shape
    k_shape = k.shape

    assert (
        q.shape[:-2] == k.shape[:-2]
    ), f"q and k must have the same length, got {q.shape[:-2]} and {k.shape[:-2]}"
    if position_ids is None:
        assert (
            len(q.shape) == 4
        ), f"q must have 4 dimensions if position_ids is not provided, got {q.shape}"
        seq_len = q.shape[-3]
    else:
        assert (
            position_ids.shape == q.shape[:-2]
        ), f"position_ids must have the same length as q, got {position_ids.shape} and {q.shape[:-2]}"

        position_ids = position_ids.view(-1)
        seq_len = None

    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])

    n_tokens, q_heads, head_dim = q.shape
    k_heads = k.shape[-2]

    # The block size must be the next power of two, sometimes we need to pad it.
    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)
    # Head axis of the 2D tile must also be a power of two (masked afterwards).
    padded_q_heads = triton.next_power_of_2(q_heads)
    padded_k_heads = triton.next_power_of_2(k_heads)

    if inplace:
        grid = (n_tokens,)
        with torch_device_fn.device(q.device):
            apply_rotary_pos_emb_inplace_kernel[grid](
                q,
                k,
                cos,
                sin,
                position_ids,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                position_ids.stride(0) if position_ids is not None else 0,
                cos.stride(0),
                sin.stride(0),
                seq_len,
                q.shape[-2],
                k.shape[-2],
                head_dim,
                padded_head_dim,
                padded_q_heads,
                padded_k_heads,
                rotary_interleaved,
                MAX_POSITION_EMBEDDINGS=cos.shape[0],
                isCloseUnrollControl=True,
            )
        return q.view(q_shape), k.view(k_shape)
    # If not inplace, we need to create new tensors for q_embed and k_embed
    else:
        # Fast path: fully contiguous q/k are handled by the flat block-DMA
        # kernel (see apply_rotary_pos_emb_flat_kernel). This avoids the
        # discrete 2D/3D tile access and rotated gather that collapsed XPU
        # bandwidth. cos/sin are materialised (sign folded) into contiguous
        # buffers so the kernel does pure block DMA.
        if q.is_contiguous() and k.is_contiguous():
            half = head_dim // 2
            half_off = 1 if rotary_interleaved else half
            if position_ids is None:
                pos = torch.arange(n_tokens, device=q.device) % seq_len
            else:
                pos = position_ids.to(torch.long)

            q_embed = torch.empty_like(q)
            k_embed = torch.empty_like(k)
            BLOCK = 8192
            with torch_device_fn.device(q.device):
                cos_q, sin_q = _build_cos_sin_full(
                    cos, sin, pos, q_heads, head_dim, half, rotary_interleaved
                )
                nq = q.numel()
                # Padded copy (half_off elems on each side) so the shifted loads
                # keep AFFINE addresses in-bounds (block DMA, no fault).
                q_pad = torch.empty(nq + 2 * half_off, dtype=q.dtype, device=q.device)
                q_pad[:half_off] = 0
                q_pad[nq + half_off :] = 0
                q_pad[half_off : half_off + nq].copy_(q.reshape(-1))
                grid = (triton.cdiv(nq, BLOCK),)
                apply_rotary_pos_emb_flat_kernel[grid](
                    q_embed,
                    q_pad,
                    cos_q,
                    sin_q,
                    nq,
                    head_dim,
                    half,
                    half_off,
                    rotary_interleaved,
                    BLOCK,
                    isCloseUnrollControl=True,
                )
                if k_heads == q_heads:
                    cos_k, sin_k = cos_q, sin_q
                else:
                    cos_k, sin_k = _build_cos_sin_full(
                        cos, sin, pos, k_heads, head_dim, half, rotary_interleaved
                    )
                nk = k.numel()
                k_pad = torch.empty(nk + 2 * half_off, dtype=k.dtype, device=k.device)
                k_pad[:half_off] = 0
                k_pad[nk + half_off :] = 0
                k_pad[half_off : half_off + nk].copy_(k.reshape(-1))
                grid = (triton.cdiv(nk, BLOCK),)
                apply_rotary_pos_emb_flat_kernel[grid](
                    k_embed,
                    k_pad,
                    cos_k,
                    sin_k,
                    nk,
                    head_dim,
                    half,
                    half_off,
                    rotary_interleaved,
                    BLOCK,
                    isCloseUnrollControl=True,
                )
            return q_embed.view(q_shape), k_embed.view(k_shape)

        # Fallback (non-contiguous q/k): per-token 2D-tile kernel.
        q_embed = torch.empty_like(q)
        k_embed = torch.empty_like(k)
        grid = (n_tokens,)
        with torch_device_fn.device(q_embed.device):
            apply_rotary_pos_emb_kernel[grid](
                q_embed,
                k_embed,
                q,
                k,
                cos,
                sin,
                position_ids,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                q_embed.stride(0),
                q_embed.stride(1),
                q_embed.stride(2),
                k_embed.stride(0),
                k_embed.stride(1),
                k_embed.stride(2),
                position_ids.stride(0) if position_ids is not None else 0,
                cos.stride(0),
                sin.stride(0),
                seq_len,
                q.shape[-2],
                k.shape[-2],
                head_dim,
                padded_head_dim,
                padded_q_heads,
                padded_k_heads,
                rotary_interleaved,
                MAX_POSITION_EMBEDDINGS=cos.shape[0],
                isCloseUnrollControl=True,
            )
        q_embed = q_embed.view(q_shape)
        k_embed = k_embed.view(k_shape)
        return q_embed, k_embed
