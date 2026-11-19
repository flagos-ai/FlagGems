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

"""Tsingmicro (txda) extend / prefill attention.

Drop-in for ``sglang.srt.layers.attention.triton_ops.extend_attention.extend_attention_fwd``
(the radix-attention prefill/extend path, i.e. prefill with a shared prefix KV
cache).  It is a single "two-stage" memory-efficient kernel:

* stage 1 computes the attention over the *prefix* part of the sequence, whose
  KV lives in the paged ``k_buffer``/``v_buffer`` (page size = 1) and is looked
  up through the radix ``kv_indices`` mapping;
* stage 2 computes the causal triangle over the *extend* tokens, whose KV is the
  contiguous ``k_extend``/``v_extend``.

The two stages are fused into one online-softmax pass (``acc``/``deno``/``e_max``).

txda-specific notes
-------------------

* **Block-granular ``tl.make_block_ptr`` loads are the whole ballgame.**  The
  SGLang reference iterates *token by token* with a 2D masked
  ``tl.load(..., mask=...)`` gather for every ``BLOCK_N`` chunk; on txda that
  masked-gather pattern does not vectorize and is the dominant cost (a 16k
  prefix is ~7 s here vs ~0.12 s with block pointers).  ``_extend_attention_fwd_kernel_blockptr``
  instead loads each contiguous ``BLOCK_N`` tile of K/V as one unmasked
  ``tl.make_block_ptr`` block and applies the sequence/causal boundary with a
  single ``tl.where`` afterwards, which the txda Triton backend vectorizes.
  This is the same lever ``decode_attention.py`` uses and is ~50x on a 16k
  prefix.

* **Prefix slots are gathered, never assumed contiguous.**  The prefix KV is
  addressed through SGLang's CSR ``kv_indices`` mapping, one slot id per tile
  (the same head-dim-last load the decode kernel uses for its scattered
  suffix).  A prefix is *not* guaranteed to be an affine run: chunked prefill
  hands the second chunk the pages ``alloc_extend`` returned for the first one,
  and those are scattered whenever the pool free list is fragmented -- e.g.
  right after a long decode phase.  An affine read there silently attends to
  another request's KV with no crash, which is exactly the class of bug this
  gather prevents.  Non-power-of-two head dims, custom masks and sliding
  windows still take the general masked-gather kernel.

* **Strided inputs, no ``.contiguous()``.**  SGLang's ``forward_extend`` calls
  ``k.contiguous()``/``v.contiguous()`` before launching the kernel, which
  materialises large copies on TXDA.  This implementation reads every tensor
  through its real strides (``stride_qbs``/``stride_qh``/``stride_kbs``/...),
  so it accepts the ``torch.split`` + ``view`` strided ``q/k/v`` directly as long
  as the last dim (head dim) is contiguous.  The plugin passes them through
  without copying.

* **Larger tiles for the 3MB on-chip SPM.**  ``_get_block_sizes_for_extend_attention``
  uses a ``BLOCK_M``/``BLOCK_N`` at least as large as the NVIDIA Hopper path and
  larger for small head dims, because the TXDA SPM (3MB) is far above the ~166KB
  shared-memory threshold that constrains the CUDA tuning.
"""

import functools
import inspect
import logging
import os
import time

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)


@triton.jit
def _tanh(x):
    # Tanh is just a scaled sigmoid; avoids pulling libdevice tanh into the loop.
    return 2 * tl.sigmoid(2 * x) - 1


def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):
    """Return (BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps).

    The head-dim decomposition mirrors SGLang's so MLA-style large head dims
    (576 / 288 / 192) are split into a ``tl.dot``-friendly power-of-two block
    plus a small "DPE" tail block.
    """
    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0

    BLOCK_DV = triton.next_power_of_2(Lv)

    # TXDA exposes a 3MB on-chip SPM (vs ~166KB shared memory on NVIDIA), so we
    # can afford larger tiles than the CUDA path.  A larger BLOCK_M amortises the
    # per-program online-softmax prologue; a larger BLOCK_N lengthens the
    # tensor-core reduction rows.
    if Lq <= 128:
        BLOCK_M, BLOCK_N = (128, 128)
    elif Lq <= 256:
        BLOCK_M, BLOCK_N = (128, 64)
    else:
        BLOCK_M, BLOCK_N = (64, 64)

    num_warps = 4 if Lq <= 64 else 8

    return BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps


@triton.jit
def _extend_attention_fwd_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sink_ptr,
    window_kv_offset_ptr,
    sm_scale,
    k_scale,
    v_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    # For SWA, we should only load the mask in the sliding window
    window_kv_offset = 0
    if USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE > 0:
        window_kv_offset = tl.load(window_kv_offset_ptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    if xai_temperature_len > 0:
        offs_qidx = cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        xai_temperature_reg = tl.where(
            offs_qidx > xai_temperature_len,
            tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale,
            1.0,
        )

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # stage 1: compute scores with prefix
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            final_mask &= custom_mask
        if SLIDING_WINDOW_SIZE > 0:
            # q_id = prefix_len + cur_m, kv_id = cur_n
            window_mask = (
                cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m[:, None]
            ) <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
            final_mask &= window_mask

        SKIP_TILE = False
        if (USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK) or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_n,
                other=0,
            )

            # Load K as a (BLOCK_N, head_dim) tile -- head dim *last* -- and
            # transpose in registers for the dot.  The reverse (loading K
            # pre-transposed, so the token axis is fastest-varying) makes the
            # stride-1 axis an arbitrary gather: measured 4.1x-21.8x slower on
            # txda depending on head dim (2041 ms -> 288 ms at Hq=3/Hkv=1/D=256,
            # 4650 ms -> 1136 ms at 32/8/128, 4350 ms -> 200 ms at 16/1/576, for
            # a 8192-token prefix with a 128-token extend).  See the note in
            # decode_attention._decode_two_segment_kernel_stage1.
            offs_buf_k = (
                offs_kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(mask_n[:, None]) & (mask_d[None, :]),
                other=0.0,
            )

            qk = tl.dot(q.to(k.dtype), tl.trans(k))
            if BLOCK_DPE > 0:
                offs_kpe = (
                    offs_kv_loc[:, None] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[None, :]
                )
                kpe = tl.load(
                    K_Buffer + offs_kpe,
                    mask=mask_n[:, None],
                    other=0.0,
                )
                qk += tl.dot(qpe.to(kpe.dtype), tl.trans(kpe))
            qk *= sm_scale * k_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            offs_buf_v = (
                offs_kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v) * v_scale

            e_max = n_e_max

    # stage 2: compute the triangle part
    cur_block_m_end = (
        cur_seq_len_extend
        if not IS_CAUSAL
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    )
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            final_mask &= custom_mask
        elif IS_CAUSAL:
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_causual
        else:
            mask_non_causal = mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_non_causal

        if SLIDING_WINDOW_SIZE > 0:
            window_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) <= (
                start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE
            )
            final_mask &= window_mask

        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # Head dim last, transposed in registers for the dot -- same reason
            # as stage 1 above.
            offs_k = (
                (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Extend + offs_k, mask=(mask_n[:, None]) & (mask_d[None, :]), other=0.0
            )

            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
            if BLOCK_DPE > 0:
                offs_kpe = (
                    (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_kbs
                    + cur_kv_head * stride_kh
                    + offs_dpe[None, :]
                )
                kpe = tl.load(
                    K_Extend + offs_kpe,
                    mask=mask_n[:, None],
                    other=0.0,
                )
                qk += tl.dot(qpe, tl.trans(kpe))

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            offs_v = (
                (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
                + cur_kv_head * stride_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
            )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v)

            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O_Extend + offs_o,
        acc / deno[:, None],
        mask=mask_m[:, None] & mask_dv[None, :],
    )


@triton.jit
def _extend_attention_fwd_kernel_blockptr(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sink_ptr,
    sm_scale,
    k_scale,
    v_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    """Blockptr fast path: extend KV loaded as unmasked ``tl.make_block_ptr``
    tiles, prefix KV gathered per tile through ``kv_indices``.

    Requires a power-of-two-decomposable head dim
    (``Lq == BLOCK_DMODEL + BLOCK_DPE``), no custom mask and no sliding window.
    The prefix may live in any physical slots; it is never assumed contiguous.
    The masked-gather kernel is the general fallback.
    """
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    if xai_temperature_len > 0:
        offs_qidx = cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        xai_temperature_reg = tl.where(
            offs_qidx > xai_temperature_len,
            tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale,
            1.0,
        )

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # ---- stage 1: prefix KV via per-token slot gather ----
    # ``kv_indices`` is SGLang's CSR mapping for the already-computed prefix.
    # The slots are *not* guaranteed to be an affine run: they are the pages
    # ``alloc_extend`` handed out for the earlier chunk(s), and once the pool
    # free list is fragmented (e.g. a request split by chunked prefill after a
    # long decode phase) consecutive sequence positions map to scattered slots.
    # Reading ``kv_indices[start] + i`` there silently attends to another
    # request's KV, so gather the slot ids explicitly -- the same head-dim-last
    # vectorised load the decode kernel uses for its scattered suffix.
    if cur_seq_len_prefix > 0:
        for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            mask_tok = (start_n + offs_n) < cur_seq_len_prefix
            kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_tok,
                other=0,
            ).to(tl.int64)

            k = tl.load(
                K_Buffer
                + kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :],
                mask=mask_tok[:, None] & mask_d[None, :],
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k))
            if BLOCK_DPE > 0:
                kpe = tl.load(
                    K_Buffer
                    + kv_loc[:, None] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[None, :],
                    mask=mask_tok[:, None],
                    other=0.0,
                )
                qk += tl.dot(qpe, tl.trans(kpe))
            qk *= sm_scale * k_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)
            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            final_mask = mask_m[:, None] & mask_tok[None, :]
            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            v = tl.load(
                V_Buffer
                + kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :],
                mask=mask_tok[:, None] & mask_dv[None, :],
                other=0.0,
            )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v) * v_scale
            e_max = n_e_max

    # ---- stage 2: contiguous extend triangle via blockptr tiles ----
    cur_block_m_end = (
        cur_seq_len_extend
        if not IS_CAUSAL
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    )
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_blk_base = (cur_seq_extend_start_idx + start_n).to(tl.int64)

        k_base = K_Extend + kv_blk_base * stride_kbs + cur_kv_head * stride_kh
        k = tl.load(
            tl.make_block_ptr(
                base=k_base,
                shape=(BLOCK_N, Lq),
                strides=(stride_kbs, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_DMODEL),
                order=(0, 1),
            )
        )
        qk = tl.dot(q, tl.trans(k))
        if BLOCK_DPE > 0:
            kpe = tl.load(
                tl.make_block_ptr(
                    base=k_base,
                    shape=(BLOCK_N, Lq),
                    strides=(stride_kbs, 1),
                    offsets=(0, BLOCK_DMODEL),
                    block_shape=(BLOCK_N, BLOCK_DPE),
                    order=(0, 1),
                )
            )
            qk += tl.dot(qpe, tl.trans(kpe))
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * _tanh(qk / logit_cap)
        if xai_temperature_len > 0:
            qk *= xai_temperature_reg[:, None]

        if IS_CAUSAL:
            mask_causal = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            final_mask = (
                mask_m[:, None]
                & ((start_n + offs_n) < cur_block_m_end)[None, :]
                & mask_causal
            )
        else:
            final_mask = (
                mask_m[:, None] & ((start_n + offs_n) < cur_block_m_end)[None, :]
            )
        qk = tl.where(final_mask, qk, float("-inf"))

        row_max = tl.max(qk, 1)
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        n_e_max = tl.maximum(row_max_fixed, e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        v_base = V_Extend + kv_blk_base * stride_vbs + cur_kv_head * stride_vh
        v = tl.load(
            tl.make_block_ptr(
                base=v_base,
                shape=(BLOCK_N, Lv),
                strides=(stride_vbs, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_DV),
                order=(0, 1),
            )
        )
        # See stage 1: the unmasked tile over-reads past the end of the extend
        # tensor for the tail tile of the last sequence(s).  Those columns are
        # masked out of qk (p == 0) but their v is undefined, and 0 * NaN ==
        # NaN would turn the whole row's output into NaN.
        v = tl.where(((start_n + offs_n) < cur_block_m_end)[:, None], v, 0.0)
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)
        e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O_Extend + offs_o,
        acc / deno[:, None],
        mask=mask_m[:, None] & mask_dv[None, :],
    )


def _fmt_value(value):
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, stride={tuple(value.stride())}, "
            f"dtype={value.dtype})"
        )
    return repr(value)


def log_args(func):
    """Print each argument on one line and the wall-clock execution time in ms.

    Gated behind ``FLAG_GEMS_LOG_EXTEND_ATTENTION=1`` so the per-call stdout
    formatting/flush doesn't tax the serving hot path.
    """
    enabled = os.getenv("FLAG_GEMS_LOG_EXTEND_ATTENTION", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not enabled:
            return func(*args, **kwargs)

        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        prefix = f"[{func.__name__}]"
        for name, value in bound.arguments.items():
            print(f"{prefix} {name}: {_fmt_value(value)}", flush=True)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"{prefix} elapsed_ms={elapsed_ms:.3f}", flush=True)

    return wrapper


@log_args
def extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale,
    v_scale,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
):
    """TXDA extend attention, drop-in for SGLang's ``extend_attention_fwd``.

    ``q_extend``/``k_extend``/``v_extend``/``o_extend`` are the (possibly
    strided, head-dim-contiguous) extend tensors; ``k_buffer``/``v_buffer`` are
    the paged (prefix + extend) KV buffers with page size = 1.
    """
    logger.debug("GEMS_TSINGMICRO EXTEND_ATTENTION_FWD")
    Lq, Lv = (
        q_extend.shape[-1],
        v_extend.shape[-1],
    )

    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)
    )

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask
    HAS_SINK = sinks is not None

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))

    # The blockptr fast path needs a power-of-two-decomposable head dim and no
    # per-token custom mask / sliding window.  Prefix slots are gathered from
    # kv_indices, so they may be physically scattered.  The gather kernel is the
    # general (correct) fallback for everything else.
    USE_BLOCKPTR = (
        (Lq == BLOCK_DMODEL + BLOCK_DPE)
        and not USE_CUSTOM_MASK
        and (sliding_window_size <= 0)
    )

    logger.debug("GEMS_TSINGMICRO EXTEND ATTENTION FWD (blockptr=%s)", USE_BLOCKPTR)

    with torch_device_fn.device(device):
        if USE_BLOCKPTR:
            _extend_attention_fwd_kernel_blockptr[grid](
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                sinks,
                sm_scale,
                k_scale,
                v_scale,
                kv_group_num,
                q_extend.stride(0),
                q_extend.stride(1),
                k_extend.stride(0),
                k_extend.stride(1),
                v_extend.stride(0),
                v_extend.stride(1),
                o_extend.stride(0),
                o_extend.stride(1),
                k_buffer.stride(0),
                k_buffer.stride(1),
                v_buffer.stride(0),
                v_buffer.stride(1),
                logit_cap=logit_cap,
                xai_temperature_len=xai_temperature_len,
                Lq=Lq,
                Lv=Lv,
                BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_DPE=BLOCK_DPE,
                BLOCK_DV=BLOCK_DV,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                IS_CAUSAL=is_causal,
                HAS_SINK=HAS_SINK,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            _extend_attention_fwd_kernel[grid](
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                custom_mask,
                mask_indptr,
                sinks,
                window_kv_offsets,
                sm_scale,
                k_scale,
                v_scale,
                kv_group_num,
                q_extend.stride(0),
                q_extend.stride(1),
                k_extend.stride(0),
                k_extend.stride(1),
                v_extend.stride(0),
                v_extend.stride(1),
                o_extend.stride(0),
                o_extend.stride(1),
                k_buffer.stride(0),
                k_buffer.stride(1),
                v_buffer.stride(0),
                v_buffer.stride(1),
                SLIDING_WINDOW_SIZE=sliding_window_size,
                logit_cap=logit_cap,
                xai_temperature_len=xai_temperature_len,
                BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_DPE=BLOCK_DPE,
                BLOCK_DV=BLOCK_DV,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                Lq=Lq,
                Lv=Lv,
                USE_CUSTOM_MASK=USE_CUSTOM_MASK,
                IS_CAUSAL=is_causal,
                SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
                HAS_SINK=HAS_SINK,
                num_warps=num_warps,
                num_stages=1,
            )
