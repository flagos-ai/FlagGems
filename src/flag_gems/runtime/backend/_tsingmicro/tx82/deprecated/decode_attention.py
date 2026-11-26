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

"""Tsingmicro (txda) split-K decode attention for GQA / MLA / MHA.

Drop-in for ``vllm.attention.ops.triton_decode_attention.decode_attention_fwd``
(the vLLM V1 triton decode path, and the same algorithm SGLang runs through
``srt.layers.attention.triton_ops.decode_attention._decode_grouped_att_m_fwd``).
It implements the classic two-stage "flash decoding" scheme:

* stage 1 splits the KV sequence of every request into ``num_kv_splits`` chunks
  and computes one partial ``(acc, lse)`` per (request, head, split);
* stage 2 reduces the partials with the online-softmax recurrence and writes
  the final ``o`` / ``lse``.

txda-specific notes
-------------------

* **Page-granular ``tl.make_block_ptr`` loads are the whole ballgame.**  The
  SGLang/metax reference iterates *token by token* with a 2D masked
  ``tl.load(..., mask=...)`` for every ``BLOCK_N`` chunk; on txda's Triton
  backend that masked-load pattern does not vectorize and runs at ~0.11 ms/token
  (1.8 s for one 16k request).  Both stage-1 kernels here instead iterate *page
  by page*: each physical page is loaded as one unmasked ``tl.make_block_ptr``
  block and the query/context boundary is applied with a single ``tl.where``
  afterwards.  That, plus ``tl.dot`` for QK/PV, is ~70x faster (16k request
  ~25 ms).  The token-by-token kernels are kept as ``*_legacy`` and only used
  when the head dim / page size are not power-of-two aligned.

* **Split-K is the 16k lever, but only when the batch is short or skewed.**  A
  16k-token context on 16 SMs means a single program per (request, head) would
  serialize ~16k / PAGE_SIZE page iterations.  The caller controls
  ``num_kv_splits`` and ``None`` picks it with :func:`_choose_num_kv_splits` --
  splitting is a pessimisation for a uniform batch, which already fills the
  device with one program per request.  ``num_kv_splits`` has no correctness
  impact, only latency (and a tiny fp re-association drift from the
  online-softmax reduce).

* **No masked atomics.** txda's Triton backend lowers ``tl.atomic_add(mask=...)``
  pathologically, so this kernel deliberately performs the cross-split combine
  as a *second pass* over the (small) partial tensor instead of atomically
  accumulating into ``o``/``lse``.  The partial tensor is ``[B, H, S, Lv + 1]``
  -- negligible next to the KV cache.

* **fp8 KV is not supported** (matching the triton decode reference).  Pass
  ``k_scale``/``v_scale`` through a dequant step before calling.
"""

import functools
import inspect
import logging
import math
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


@triton.jit
def _decode_kernel_stage1_legacy(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q, mask=mask_d, other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch
                + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )
        tl.store(Att_Out + offs_mid_o, acc / e_sum, mask=mask_dv)

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )
        tl.store(Att_Out + offs_mid_o_1, e_max + tl.log(e_sum))


@triton.jit
def _decode_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    """MHA fast path: page-granular ``tl.make_block_ptr`` loads + tensor cores."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q, mask=mask_d, other=0.0)
    q2 = q[None, :]

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([1, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        page_start = split_kv_start // PAGE_SIZE
        page_end = tl.cdiv(split_kv_end, PAGE_SIZE)
        page_offsets = tl.arange(0, PAGE_SIZE)

        for page_idx in range(page_start, page_end):
            kv_page_number = tl.load(
                Req_to_tokens + cur_batch * stride_req_to_tokens_b + page_idx
            )
            kv_page_base = kv_page_number.to(tl.int64) * PAGE_SIZE

            k_base = (
                K_Buffer + kv_page_base * stride_buf_kbs + cur_kv_head * stride_buf_kh
            )
            k_block = tl.make_block_ptr(
                base=k_base,
                shape=(PAGE_SIZE, Lk),
                strides=(stride_buf_kbs, 1),
                offsets=(0, 0),
                block_shape=(PAGE_SIZE, BLOCK_DMODEL),
                order=(0, 1),
            )
            k = tl.load(k_block)
            qk = tl.dot(q2, tl.trans(k))
            qk = tl.reshape(qk, [PAGE_SIZE])
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            token_offsets = page_idx * PAGE_SIZE + page_offsets
            mask_tok = (token_offsets >= split_kv_start) & (
                token_offsets < split_kv_end
            )
            qk = tl.where(mask_tok, qk, float("-inf"))

            v_base = (
                V_Buffer + kv_page_base * stride_buf_vbs + cur_kv_head * stride_buf_vh
            )
            v_block = tl.make_block_ptr(
                base=v_base,
                shape=(PAGE_SIZE, Lv),
                strides=(stride_buf_vbs, 1),
                offsets=(0, 0),
                block_shape=(PAGE_SIZE, BLOCK_DV),
                order=(0, 1),
            )
            v = tl.load(v_block)

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.dot(p[None, :].to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )
        tl.store(
            Att_Out + offs_mid_o, tl.reshape(acc, [BLOCK_DV]) / e_sum, mask=mask_dv
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )
        tl.store(Att_Out + offs_mid_o_1, e_max + tl.log(e_sum))


@triton.jit
def _decode_grouped_kernel_stage1_legacy(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if kv_group_num > BLOCK_H:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(
            Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
        )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch
                + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            # Head dim last, transposed in registers for the dot: the stride-1
            # axis has to be the fastest-varying one or the load degrades to one
            # transaction per element (~60x).  See
            # _decode_two_segment_kernel_stage1 for the measurement.
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k.to(q.dtype)))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_loc[:, None] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[None, :]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dpe[None, :]),
                    other=0.0,
                )
                qk += tl.dot(qpe, tl.trans(kpe.to(qpe.dtype)))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )
        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


@triton.jit
def _decode_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    """txda fast path: page-granular ``tl.make_block_ptr`` loads + tensor cores.

    Requires ``Lk == BLOCK_DMODEL + BLOCK_DPE`` (a power-of-two-aligned head dim)
    and a power-of-two ``PAGE_SIZE``, so every K/V page loads as an unmasked
    vectorized block.  Tokens outside the split are masked with a single
    ``tl.where`` per page instead of a per-token masked load.
    """
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if kv_group_num > BLOCK_H:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(
            Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
        )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        page_start = split_kv_start // PAGE_SIZE
        page_end = tl.cdiv(split_kv_end, PAGE_SIZE)
        page_offsets = tl.arange(0, PAGE_SIZE)

        for page_idx in range(page_start, page_end):
            kv_page_number = tl.load(
                Req_to_tokens + cur_batch * stride_req_to_tokens_b + page_idx
            )
            kv_page_base = kv_page_number.to(tl.int64) * PAGE_SIZE

            k_base = (
                K_Buffer + kv_page_base * stride_buf_kbs + cur_kv_head * stride_buf_kh
            )
            k_block = tl.make_block_ptr(
                base=k_base,
                shape=(PAGE_SIZE, Lk),
                strides=(stride_buf_kbs, 1),
                offsets=(0, 0),
                block_shape=(PAGE_SIZE, BLOCK_DMODEL),
                order=(0, 1),
            )
            k = tl.load(k_block)
            qk = tl.dot(q, tl.trans(k))
            if BLOCK_DPE > 0:
                kpe_block = tl.make_block_ptr(
                    base=k_base,
                    shape=(PAGE_SIZE, Lk),
                    strides=(stride_buf_kbs, 1),
                    offsets=(0, BLOCK_DMODEL),
                    block_shape=(PAGE_SIZE, BLOCK_DPE),
                    order=(0, 1),
                )
                kpe = tl.load(kpe_block)
                qk = tl.dot(qpe, tl.trans(kpe), acc=qk)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            token_offsets = page_idx * PAGE_SIZE + page_offsets
            mask_tok = (token_offsets >= split_kv_start) & (
                token_offsets < split_kv_end
            )
            qk = tl.where(mask_h[:, None] & mask_tok[None, :], qk, float("-inf"))

            v_base = (
                V_Buffer + kv_page_base * stride_buf_vbs + cur_kv_head * stride_buf_vh
            )
            v_block = tl.make_block_ptr(
                base=v_base,
                shape=(PAGE_SIZE, Lv),
                strides=(stride_buf_vbs, 1),
                offsets=(0, 0),
                block_shape=(PAGE_SIZE, BLOCK_DV),
                order=(0, 1),
            )
            v = tl.load(v_block)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )
        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


@triton.jit
def _decode_kernel_stage2(
    Mid_O,
    O,
    LSE,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        # Must match _decode_two_segment_kernel_stage1 exactly, including the
        # round-up to whole BLOCK_N tiles: an inactive split is one that stage 1
        # left unwritten, and this is the test that skips it.
        kv_len_per_split = (
            tl.cdiv(tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS), BLOCK_N) * BLOCK_N
        )
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )
    tl.store(LSE + cur_batch * stride_lse_bs + cur_head, e_max + tl.log(e_sum))


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    req_to_token,
    b_seq_len,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
):
    BLOCK = 64
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]
    grid = (batch, head_num, num_kv_splits)
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    num_warps = 4 if kv_group_num == 1 else 2
    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    # The make_block_ptr fast path needs a power-of-two head dim (BLOCK_DMODEL == Lk)
    # and a power-of-two page size for unmasked vectorized page loads.
    use_blockptr = (
        (Lk == BLOCK_DMODEL) and (page_size & (page_size - 1)) == 0 and page_size > 1
    )

    constexpr = dict(
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )
    runtime = (
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        req_to_token,
        b_seq_len,
        att_out,
        req_to_token.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),  # (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),
        v_buffer.stride(-3),
        v_buffer.stride(-2),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
    )

    if use_blockptr:
        _decode_kernel_stage1[grid](*runtime, **constexpr)
    else:
        _decode_kernel_stage1_legacy[grid](*runtime, BLOCK_N=BLOCK, **constexpr)


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    req_to_token,
    b_seq_len,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    # One program handles one kv-head group (kv_group_num query heads); size the
    # head tile to the next power of two so the Q load and QK dot are not padded
    # out to a fixed 16 heads (5x waste for kv_group=3 in the TP=8 decode path).
    BLOCK_H = triton.next_power_of_2(min(kv_group_num, 16))
    grid = (batch, triton.cdiv(head_num, min(BLOCK_H, kv_group_num)), num_kv_splits)

    # The make_block_ptr fast path needs a head dim that decomposes exactly into
    # the power-of-two blocks (BLOCK_DMODEL + BLOCK_DPE) and a power-of-two page
    # size so each page is an unmasked vectorized load.  Otherwise fall back to
    # the token-by-token masked-load kernel.
    use_blockptr = (
        (Lk == BLOCK_DMODEL + BLOCK_DPE)
        and (page_size & (page_size - 1)) == 0
        and page_size > 1
    )

    constexpr = dict(
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )
    runtime = (
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        req_to_token,
        b_seq_len,
        att_out,
        req_to_token.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),
        k_buffer.stride(-2),
        v_buffer.stride(-3),
        v_buffer.stride(-2),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
    )

    if use_blockptr:
        _decode_grouped_kernel_stage1[grid](*runtime, **constexpr)
    else:
        _decode_grouped_kernel_stage1_legacy[grid](*runtime, BLOCK_N=BLOCK, **constexpr)


def _decode_softmax_reducev_fwd(
    att_out,
    q,
    o,
    lse,
    v_buffer,
    b_seq_len,
    num_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    grid = (batch, head_num)
    _decode_kernel_stage2[grid](
        att_out,
        o,
        lse,
        b_seq_len,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
):
    logger.debug("GEMS_TSINGMICRO DECODE_ATTENTION_FWD_NORMAL")
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(
        attn_logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits
    )


def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
):
    logger.debug("GEMS_TSINGMICRO DECODE_ATTENTION_FWD_GROUPED")
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(
        attn_logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits
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

    Gated behind ``FLAG_GEMS_LOG_DECODE_ATTENTION=1`` so the per-call stdout
    formatting/flush doesn't tax the serving hot path.
    """
    enabled = os.getenv("FLAG_GEMS_LOG_DECODE_ATTENTION", "0").lower() in (
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
def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size=1,
    logit_cap=0.0,
):
    """Flash-decoding decode attention for MHA / GQA / MQA / MLA.

    Args:
        q: ``[batch, q_num_heads, head_dim]``.
        k_buffer: paged key cache ``(..., PAGE_SIZE, kv_num_heads, head_dim)``.
        v_buffer: paged value cache ``(..., PAGE_SIZE, kv_num_heads, head_dim_v)``.
        o: ``[batch, q_num_heads, head_dim_v]`` output (written in place).
        lse: ``[batch, q_num_heads]`` log-sum-exp (written in place).
        req_to_token: block table ``[batch, max_blocks]`` (int32).
        b_seq_len: ``[batch]`` (int32) per-request sequence lengths.
        attn_logits: ``[batch, q_num_heads, num_kv_splits, head_dim_v + 1]``
            scratch; the trailing column holds the per-split lse.
        num_kv_splits: number of KV chunks to split each sequence into.
        sm_scale: softmax scale (typically ``1 / sqrt(head_dim)``).
        page_size: tokens per KV-cache page.
        logit_cap: optional tanh logit cap (0 disables).
    """
    logger.debug("GEMS_TSINGMICRO DECODE_ATTENTION_FWD")
    assert num_kv_splits == attn_logits.shape[2]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    with torch_device_fn.device(device):
        if kv_group_num == 1:
            decode_attention_fwd_normal(
                q,
                k_buffer,
                v_buffer,
                o,
                lse,
                req_to_token,
                b_seq_len,
                attn_logits,
                num_kv_splits,
                sm_scale,
                page_size,
                logit_cap,
            )
        else:
            decode_attention_fwd_grouped(
                q,
                k_buffer,
                v_buffer,
                o,
                lse,
                req_to_token,
                b_seq_len,
                attn_logits,
                num_kv_splits,
                sm_scale,
                page_size,
                logit_cap,
            )


@triton.jit
def _decode_two_segment_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_indptr,
    kv_indices,
    prefix_lens,
    O,
    LSE,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    Mid_O,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    logit_cap: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    """Two-segment decode attention: contiguous prefix (affine) + scattered suffix (gather).

    SGLang's chunked prefill + decode leaves the KV split into a *contiguous*
    prefix (the prompt) plus *scattered* suffix tokens (the generated tokens are
    interleaved across the batch).  This kernel processes the prefix with an
    affine tile load and the suffix with a masked gather, combining both in one
    online-softmax pass.

    Both segments load K/V as a ``(BLOCK_N, head_dim)`` tile with the **head dim
    last**.  That ordering is what makes the tile vectorize: the head dim is the
    stride-1 axis, so writing it last lets the backend emit one wide transaction
    per row.  Writing the *token* axis last instead (i.e. loading K pre-transposed
    for ``tl.dot``, as the SGLang reference does) leaves the fastest-varying axis
    as an arbitrary gather, so the backend falls back to one transaction *per
    element* -- measured at ~105x slower on txda (2229 ms vs 21 ms for the same
    3 x 9388-token decode).  K is transposed back with ``tl.trans`` inside the
    register/shared-memory path, which is free by comparison.

    ``NUM_SPLITS > 1`` splits the request's token range over ``program_id(2)``
    and stores *normalized* partials (``acc / e_sum`` plus ``e_max + log(e_sum)``)
    into ``Mid_O[batch, head, split, Lv + 1]`` for ``_decode_kernel_stage2`` to
    combine.  The split boundaries are ``tl.cdiv(seq_len, NUM_SPLITS)`` -- the
    exact rule stage 2 re-derives, so an empty split is skipped there rather than
    contributing a NaN.
    """
    cur_batch = tl.program_id(1)
    cur_head_id = tl.program_id(2)

    cur_batch_kv_start = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start

    # This program owns the half-open token range [split_start, split_end).  With
    # NUM_SPLITS == 1 that is the whole request, so the loops below collapse to
    # the unsplit kernel.
    if NUM_SPLITS > 1:
        cur_split = tl.program_id(0)
        # Round the per-split length up to a whole BLOCK_N tile.  A split whose
        # tail is only partly live still pays for a full tile, so the plain
        # cdiv wastes up to BLOCK_N-1 lanes on every split.  Rounding also
        # collapses the splits that fall entirely past the end of a short
        # request, which is what keeps a large NUM_SPLITS harmless on a batch of
        # short sequences: 128 tokens over 16 splits rounds to one 256-wide
        # split, i.e. the unsplit kernel, instead of 16 quarter-full tiles.
        # _decode_kernel_stage2 re-derives these boundaries with the same
        # expression, so the two must stay in sync.
        split_size = tl.cdiv(tl.cdiv(cur_batch_seq_len, NUM_SPLITS), BLOCK_N) * BLOCK_N
        split_start = split_size * cur_split
        split_end = tl.minimum(split_start + split_size, cur_batch_seq_len)
    else:
        cur_split = 0
        split_start = 0
        split_end = cur_batch_seq_len

    # Leave before the prologue when this split has no tokens.  That is the
    # common case whenever a long request is split inside a wide batch: every
    # short request contributes NUM_SPLITS programs but only the first has work,
    # so e.g. a 512-wide batch with one long request is ~7700 idle programs.
    # The q load and the accumulator init below cost ~35us even when no loop
    # runs, which made those idle programs cost more than the work they skipped.
    #
    # Only the split path may skip the store: stage 2 re-derives the same
    # boundaries and never reads an inactive split, whereas with NUM_SPLITS == 1
    # this program owns O and LSE and must still write them (a zero-length
    # request stores the NaN that callers already expect).
    if NUM_SPLITS > 1 and split_end <= split_start:
        return

    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    if kv_group_num > BLOCK_H:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < Lv

    prefix_len = tl.load(prefix_lens + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=mask_h[:, None] & (offs_d[None, :] < Lk), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(
            Q + off_qpe, mask=mask_h[:, None] & (offs_dpe[None, :] < Lk), other=0.0
        )

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    # ---- Segment 1: contiguous prefix via affine tiles ----
    # ``prefix_lens`` is the length of the leading run of *physically
    # contiguous* slots (the caller computes it as the longest run where
    # ``kv_indices[kv_start + i] == kv_indices[kv_start] + i``), so the whole
    # prompt is one affine range and needs no per-token index gather.
    # Clipping at ``prefix_hi`` (rather than masking on ``prefix_len``) is what
    # keeps a split from re-reading tokens another split already owns: the
    # segment-2 loop below starts at ``suffix_lo``, so the two ranges are
    # disjoint and together cover [split_start, split_end) exactly.
    prefix_hi = tl.minimum(split_end, prefix_len)
    if prefix_hi > split_start:
        prefix_base = tl.load(kv_indices + cur_batch_kv_start).to(tl.int64)
        for start_n in range(split_start, prefix_hi, BLOCK_N):
            token_offsets = start_n + tl.arange(0, BLOCK_N)
            mask_tok = token_offsets < prefix_hi
            tok = prefix_base + token_offsets

            k = tl.load(
                K_Buffer
                + tok[:, None] * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[None, :],
                mask=mask_tok[:, None] & (offs_d[None, :] < Lk),
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k.to(q.dtype)))
            if BLOCK_DPE > 0:
                kpe = tl.load(
                    K_Buffer
                    + tok[:, None] * stride_kbs
                    + cur_kv_head * stride_kh
                    + offs_dpe[None, :],
                    mask=mask_tok[:, None] & (offs_dpe[None, :] < Lk),
                    other=0.0,
                )
                qk = tl.dot(qpe, tl.trans(kpe.to(qpe.dtype)), acc=qk)
            qk *= sm_scale
            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            qk = tl.where(mask_h[:, None] & mask_tok[None, :], qk, float("-inf"))

            # Masked load, so the slots past ``prefix_len`` are never read --
            # unlike the old unmasked blockptr tile, which read whatever the
            # pool held there (possibly NaN) and relied on a ``tl.where`` to
            # keep 0 * NaN out of ``acc``.
            v = tl.load(
                V_Buffer
                + tok[:, None] * stride_vbs
                + cur_kv_head * stride_vh
                + offs_dv[None, :],
                mask=mask_tok[:, None] & (offs_dv[None, :] < Lv),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

    # ---- Segment 2: scattered suffix via masked gather ----
    suffix_lo = tl.maximum(split_start, prefix_len)
    if split_end > suffix_lo:
        for start_n in range(suffix_lo, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_tok = offs_n < split_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start + offs_n, mask=mask_tok, other=0
            ).to(tl.int64)

            k = tl.load(
                K_Buffer
                + kv_loc[:, None] * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[None, :],
                mask=mask_tok[:, None] & (offs_d[None, :] < Lk),
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k.to(q.dtype)))
            if BLOCK_DPE > 0:
                kpe = tl.load(
                    K_Buffer
                    + kv_loc[:, None] * stride_kbs
                    + cur_kv_head * stride_kh
                    + offs_dpe[None, :],
                    mask=mask_tok[:, None] & (offs_dpe[None, :] < Lk),
                    other=0.0,
                )
                qk = tl.dot(qpe, tl.trans(kpe.to(qpe.dtype)), acc=qk)
            qk *= sm_scale
            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)
            qk = tl.where(mask_h[:, None] & mask_tok[None, :], qk, float("-inf"))

            v = tl.load(
                V_Buffer
                + kv_loc[:, None] * stride_vbs
                + cur_kv_head * stride_vh
                + offs_dv[None, :],
                mask=mask_tok[:, None] & (offs_dv[None, :] < Lv),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

    if NUM_SPLITS > 1:
        # Normalized partial + its lse, in the exact layout
        # _decode_kernel_stage2 reads.  An out-of-range split leaves e_sum == 0
        # and stores NaN/-inf here; stage 2 skips it (same split arithmetic).
        offs_mid = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + cur_split * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(
            Mid_O + offs_mid,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )
        tl.store(
            Mid_O
            + cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + cur_split * stride_mid_os
            + Lv,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )
    else:
        offs_o = (
            cur_batch * stride_obs + cur_head[:, None] * stride_oh + offs_dv[None, :]
        )
        tl.store(
            O + offs_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )
        tl.store(
            LSE + cur_batch * stride_lse_bs + cur_head,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


# txda exposes 16 tiles/SMs (see the module docstring).
_TXDA_TILES = 16
# Cap on split-K.  The combine pass is one program per (request, head), so more
# splits than this stop paying for themselves once the tiles are covered.
_MAX_KV_SPLITS = _TXDA_TILES
# Below this many KV slots in total, splitting only adds the stage-2 pass: the
# whole call is already a fraction of a millisecond.
_MIN_TOKENS_FOR_SPLIT = 2048
# A request holding at least 1/_SKEW_SHARE of every KV slot in the call marks a
# *skewed* batch -- one long generation running alongside short ones.  Such a
# batch cannot be detected from ``batch`` and the total slot count alone: a
# uniform batch of 16 x 8646 holds 138336 slots, more than a batch of 16 whose
# longest request is 65536, yet the first wants no split at all and the second
# wants the maximum.  Only the longest request separates them.
_SKEW_SHARE = 10
# Prefix scan tuning; see _compute_prefix_lens.  BLOCK is the tile width the
# scan walks in and TARGET_PROGRAMS the desired grid size.
_PREFIX_BLOCK = 64
_TARGET_PROGRAMS = 256
_MAX_PREFIX_CHUNKS = 16


@triton.jit
def _prefix_len_kernel(
    kv_indptr, kv_indices, partial, N_CHUNKS: tl.constexpr, BLOCK: tl.constexpr
):
    """Leading run of physically contiguous slots, ``(N_CHUNKS, batch)`` grid.

    Request ``b`` owns ``kv_indptr[b]:kv_indptr[b + 1]``; the radix cache hands
    out physically sequential slots for the prompt, so the leading run where
    ``kv_indices[start + i] == kv_indices[start] + i`` *is* the prompt, and the
    first slot that breaks it is where the scattered generated tokens begin.

    The test is absolute (against ``kv_indices[start] + i``, not against the
    previous slot), so the run length is the smallest index that fails it -- a
    pure min-reduction, which is what lets the scan be cut into chunks at all.
    Each chunk reports ``min(first failure in chunk, seq_len)``; a chunk that
    never fails reports ``seq_len`` and survives the caller's min, so the chunks
    write *disjoint* slots and no atomics are needed.

    Chunking is required for low ``batch``: with one program per request a lone
    65536-token request runs on 1 of the 16 tiles and costs ~39 ms, versus ~3 ms
    when it is spread over 16 chunks.  The chunk count is chosen by the caller
    from ``batch`` (see :func:`_compute_prefix_lens`).

    ``chunk`` is rounded up to a whole number of ``BLOCK``s and floored at one
    ``BLOCK``.  Both matter: a tile costs its full ``BLOCK`` lanes even when only
    a few are live, so a sub-tile chunk makes a short request pay a whole tile,
    and a chunk that is not a multiple of ``BLOCK`` wastes most of its last tile.
    Empty chunks (``lo >= seq_len``) skip the loop *and* the cross-lane reduction
    -- an empty chunk that still reduced measured ~50us each, which is what made
    a 16-chunk grid cost 12.7 ms on a batch of 128-token requests.

    Two alternatives were measured and rejected:

      * ``tl.atomic_min`` to combine partial scans across programs -- it costs
        ~20us *per program* on this device regardless of contention, so a few
        hundred programs add ~10 ms on their own;
      * a flat partition over the concatenated slots, which balances work by
        token count but makes a program straddle request boundaries and so pay a
        whole tile per request it touches (``b256 x 128`` became 16 waves of
        one-tile programs, 15.6 ms vs 1.6 ms here).
    """
    chunk_id = tl.program_id(0)
    b = tl.program_id(1)

    r_lo = tl.load(kv_indptr + b)
    seq_len = tl.load(kv_indptr + b + 1) - r_lo

    chunk = tl.maximum(tl.cdiv(seq_len, N_CHUNKS * BLOCK) * BLOCK, BLOCK)
    lo = chunk_id * chunk
    if lo < seq_len:
        hi = tl.minimum(lo + chunk, seq_len)
        # int32 rather than int64: a KV pool index cannot approach 2**31 on this
        # device (that would take >= 2 TB of KV cache), and the 64-bit compare
        # costs ~15% of the scan.
        base = tl.load(kv_indices + r_lo).to(tl.int32)

        # One lane-wise running minimum for the whole chunk and a single
        # cross-lane reduction at the end; reducing every tile costs ~3x.
        lane = tl.full([BLOCK], 1 << 30, tl.int32)
        for i in range(lo, hi, BLOCK):
            offs = i + tl.arange(0, BLOCK)
            live = offs < hi
            slot = tl.load(kv_indices + r_lo + offs, mask=live, other=0).to(tl.int32)
            lane = tl.minimum(
                lane, tl.where((slot != base + offs) & live, offs, 1 << 30)
            )
        # Clamp at the request's own length: a chunk that finds no break must
        # report seq_len, not the sentinel.
        tl.store(
            partial + b * N_CHUNKS + chunk_id,
            tl.minimum(tl.min(lane, axis=0), seq_len),
        )
    else:
        tl.store(partial + b * N_CHUNKS + chunk_id, seq_len)


@triton.jit
def _prefix_merge_kernel(partial, out, N_CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
    """Min-reduce the per-chunk scans into one ``prefix_lens`` entry per request.

    This deliberately avoids ``partial.min(dim=1)``: that is the *generic*
    FlagGems ``min_dim`` on this backend (there is no ``_tsingmicro``
    specialization), which lowers ``tl.min(..., return_indices=True)`` plus an
    int64 index vector -- the pattern the backend calls out as broken in
    ``ops/argmin.py``.  A wrong merge over-estimates ``prefix_lens`` and the
    decode kernel then reads physically contiguous slots owned by another
    request: plausible but wrong logits, with no crash.  The tile is only
    ``[batch, n_chunks]`` int32, so a dedicated reduction is also cheaper than
    the generic kernel's index vector.
    """
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    vals = tl.load(
        partial + b * N_CHUNKS + offs,
        mask=offs < N_CHUNKS,
        other=(1 << 30),
    )
    tl.store(out + b, tl.min(vals, axis=0))


def _compute_prefix_lens(kv_indptr, kv_indices, batch):
    """One scan launch plus a dedicated min-merge kernel; no host sync.

    The CSR->dense formulation this replaces needed a sync for ``max_len`` and
    four full passes over a ``[batch, max_len]`` table, and measured at 7-38 ms
    across the shapes below.  This measures 0.6-10.8 ms, at most 10.8 ms where
    the old path took 38 ms.

    The chunk count trades the two failure modes against each other and is set
    by ``batch`` alone, which is the only host-known quantity that separates
    them.  Total programs is ``n_chunks * batch``:

      * low ``batch`` (a few long requests) needs many chunks, or the longest
        request is scanned serially by one program while 15 tiles idle;
      * high ``batch`` (hundreds of short requests) already fills the device with
        one program per request, and extra chunks are pure overhead -- they add
        empty programs and a chunk-combining pass.

    Aiming for ~256 total programs keeps both ends bounded, and the 16-chunk cap
    stops a batch of 1 from asking for a grid it cannot use.  Measured worst case
    over ``b1 x 131072`` .. ``b512 x 64`` is 10.8 ms (vs 38 ms before).
    """
    device = kv_indices.device
    if batch == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    n_chunks = max(1, min(_MAX_PREFIX_CHUNKS, -(-_TARGET_PROGRAMS // batch)))
    partial = torch.empty((batch, n_chunks), dtype=torch.int32, device=device)
    if kv_indices.numel() > 0:
        _prefix_len_kernel[(n_chunks, batch)](
            kv_indptr,
            kv_indices,
            partial,
            N_CHUNKS=n_chunks,
            BLOCK=_PREFIX_BLOCK,
            num_warps=1,
        )
    else:
        partial.zero_()
    if n_chunks > 1:
        out = torch.empty(batch, dtype=torch.int32, device=device)
        _prefix_merge_kernel[(batch,)](
            partial,
            out,
            N_CHUNKS=n_chunks,
            BLOCK=triton.next_power_of_2(n_chunks),
            num_warps=1,
        )
        return out
    return partial.reshape(batch)


# Cache of the longest request, keyed on (batch, total slots).  Both are exact
# functions of the per-step sequence-length vector, so every layer of one
# forward pass hits the cache and the host sync below runs once per token step
# rather than once per layer -- measured at 0.53ms, i.e. 19ms/token across 36
# layers if paid every layer.
#
# Two *different* steps can share a (batch, total slots) key -- a batch of 16 x
# 4096 collides with one holding a single 65536-token request -- and the cache
# would then serve the wrong longest-request.  That only ever costs a suboptimal
# split count (see the module docstring: ``num_kv_splits`` has no correctness
# impact), never a wrong result, which is why this is acceptable: the worst case
# is one step picking the unsplit kernel on a skewed batch, against paying 19ms
# per token for a sync that cannot be cached.
_max_seq_len_cache = (-1, -1, 0)


def _max_seq_len(kv_indptr, batch, n_kv):
    """Longest request in the call, reading the device (one host sync).

    Cached on ``(batch, n_kv)``; see ``_max_seq_len_cache``.
    """
    global _max_seq_len_cache
    cached = _max_seq_len_cache
    if cached[0] == batch and cached[1] == n_kv:
        return cached[2]
    max_len = int((kv_indptr[1 : batch + 1] - kv_indptr[:batch]).max().item())
    _max_seq_len_cache = (batch, n_kv, max_len)
    return max_len


def _choose_num_kv_splits(batch, head_blocks, kv_indptr, n_kv, block_n):
    """Split-K factor that keeps the 16 txda tiles busy on a long request.

    The kernel's only other parallelism is ``(batch, head_blocks)``, so a single
    long request runs on 3 of the 16 tiles and costs the full ``~2us * seq_len``
    while 13 tiles sit idle (measured: one 65536-token request takes 129 ms
    unsplit).  Splitting the KV range over the third grid axis uses them.

    Two independent reasons to split, and the larger one wins:

    * ``batch * head_blocks`` programs do not fill the device.  ``n_kv`` bounds
      this case well enough without a sync -- a small batch means the longest
      request is within a constant factor of the total.
    * The batch is *skewed*: one request holds ``>= 1/_SKEW_SHARE`` of all the
      slots, so the other ``batch - 1`` programs finish immediately and the
      device runs the long one alone.  This needs the longest request, and
      ``n_kv`` cannot stand in for it -- see ``_SKEW_SHARE``.

    Splitting a batch that is neither short nor skewed is actively harmful: it
    multiplies the per-program prologue and the stage-2 combine for no gain
    (measured: a uniform 16 x 8646 decode goes 17.5ms -> 34.0ms at 16 splits).

    For the skewed case the split factor is not simply "as many as possible".
    Splitting multiplies *every* request's programs by the split factor while
    only the dominant request has work to spread, so past a point the extra
    programs cost more than the depth they save (measured on a 512-wide batch
    with one long request: 74.6ms unsplit, 108.9ms at 2 splits, 334.7ms at 16).
    With ``base`` programs running ``_TXDA_TILES`` at a time, the unsplit cost is
    the dominant request's ``D`` tiles and the split cost is
    ``max(D/s, s*base/16)``, which balances at ``s = 4*sqrt(D/base)``.  That
    balance only applies once the batch is wider than the device: below
    ``base == _TXDA_TILES`` the batch cannot fill it at all, and the measured
    optimum there is the maximum (a 16-wide batch holding one 65536-token
    request wants all 16, as does one holding eight 32768-token requests).
    """
    if n_kv < _MIN_TOKENS_FOR_SPLIT:
        return 1
    base = max(1, batch * head_blocks)
    needed = (_TXDA_TILES + base - 1) // base
    if needed < _MAX_KV_SPLITS:
        max_len = _max_seq_len(kv_indptr, batch, n_kv)
        if max_len >= _MIN_TOKENS_FOR_SPLIT and max_len * _SKEW_SHARE >= n_kv:
            if base <= _TXDA_TILES:
                needed = _MAX_KV_SPLITS
            else:
                long_tiles = -(-max_len // block_n)
                skew_splits = int(math.sqrt(_TXDA_TILES * long_tiles / base) + 0.5)
                needed = max(needed, min(_MAX_KV_SPLITS, max(1, skew_splits)))
    return max(1, min(_MAX_KV_SPLITS, needed))


@log_args
def decode_attention_fwd_sglang(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    kv_indptr,
    kv_indices,
    prefix_lens,
    sm_scale,
    logit_cap=0.0,
    num_kv_splits=None,
):
    """SGLang-layout two-segment decode attention.

    ``kv_indptr`` + flattened ``kv_indices`` describe the CSR physical slots;
    ``prefix_lens`` marks, per request, how many leading slots are contiguous (the
    prompt).  The contiguous prefix runs through an affine tile path and the
    scattered suffix (generated tokens) through a masked gather.

    ``prefix_lens=None`` computes it here with :func:`_prefix_len_kernel` -- one
    kernel instead of the caller's CSR->dense materialisation plus host sync.

    ``num_kv_splits`` splits the KV range over a third grid axis (see
    :func:`_choose_num_kv_splits`); ``None`` picks it from the grid shape.
    """
    logger.debug("GEMS_TSINGMICRO DECODE_ATTENTION_FWD_SGLANG (SGLANG TWO-SEGMENT)")

    batch = q.shape[0]
    head_num = q.shape[1]
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]
    kv_group_num = head_num // k_buffer.shape[1]

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    # 256 tokens per tile is the measured sweet spot on the 3MB SPM for the
    # 256-wide head dims of this profile; the wider tile amortises the
    # per-iteration prologue/epilogue that dominated at the old BLOCK_N=32 /
    # PAGE_SIZE=16.  A 512-wide (MLA) tile is halved to keep the SPM footprint
    # reasonable.
    BLOCK_N = 256 if BLOCK_DMODEL <= 256 else 64

    # One program per (batch, head-block), and one KV head is shared by
    # ``kv_group_num`` query heads, so BLOCK_H only has to cover a single GQA
    # group.  Pinning it to 16 made every program compute a 16-row score tile of
    # which only ``kv_group_num`` rows were live -- for kv_group_num=3 that is a
    # 5x-wasted exp/max/sum over the score tile (the same fix the split-K path
    # below already carries).
    BLOCK_H = triton.next_power_of_2(min(kv_group_num, 16))

    head_blocks = triton.cdiv(head_num, min(BLOCK_H, kv_group_num))

    if prefix_lens is None:
        prefix_lens = _compute_prefix_lens(kv_indptr, kv_indices, batch)
    if num_kv_splits is None:
        num_kv_splits = _choose_num_kv_splits(
            batch, head_blocks, kv_indptr, kv_indices.numel(), BLOCK_N
        )

    if num_kv_splits > 1:
        # Mid_O[..., :Lv] holds the normalized partial, [..., Lv] its lse; the
        # row stride must stay Lv + 1 because that is where stage 2 reads it.
        mid_o = torch.empty(
            (batch, head_num, num_kv_splits, Lv + 1),
            dtype=torch.float32,
            device=q.device,
        )
        partial, stride_p_ob, stride_p_oh, stride_p_os = (
            mid_o,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
        )
        grid = (num_kv_splits, batch, head_blocks)
    else:
        partial, stride_p_ob, stride_p_oh, stride_p_os = (
            o,
            o.stride(0),
            o.stride(1),
            0,
        )
        grid = (1, batch, head_blocks)

    with torch_device_fn.device(device):
        _decode_two_segment_kernel_stage1[grid](
            q,
            k_buffer,
            v_buffer,
            sm_scale,
            kv_indptr,
            kv_indices,
            prefix_lens,
            o,
            lse,
            q.stride(0),
            q.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            o.stride(0),
            o.stride(1),
            lse.stride(0),
            partial,
            stride_p_ob,
            stride_p_oh,
            stride_p_os,
            kv_group_num=kv_group_num,
            q_head_num=head_num,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DPE=BLOCK_DPE,
            BLOCK_DV=BLOCK_DV,
            BLOCK_N=BLOCK_N,
            BLOCK_H=BLOCK_H,
            logit_cap=logit_cap,
            NUM_SPLITS=num_kv_splits,
            num_warps=4,
            num_stages=2,
            Lk=Lk,
            Lv=Lv,
        )

        if num_kv_splits > 1:
            seq_lens = kv_indptr[1 : batch + 1] - kv_indptr[:batch]
            _decode_kernel_stage2[(batch, head_num)](
                mid_o,
                o,
                lse,
                seq_lens,
                mid_o.stride(0),
                mid_o.stride(1),
                mid_o.stride(2),
                o.stride(0),
                o.stride(1),
                lse.stride(0),
                NUM_KV_SPLITS=num_kv_splits,
                BLOCK_DV=BLOCK_DV,
                BLOCK_N=BLOCK_N,
                Lv=Lv,
                num_warps=4,
                num_stages=1,
            )
