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
import math
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

# Dense dq/dkv launches use the file-local `_cudnn_attn_bwd_dq` /
# `_cudnn_attn_bwd_dkv` forks (see the comment block above them): the dense
# call sites carry no DS-placeholder tail, and importing the public
# `_flash_attn_bwd_dq_fused` / `_flash_attn_bwd_dkv` from
# ``flag_gems.ops.flash_attention_backward`` binds that module's 58/54-param
# bodies, which the c878d6faa-era DS-style call sites could not satisfy.
# The varlen kernels below are signature-identical in both modules and are
# only reachable through the varlen host branch.
from .flash_attention_backward_gcu400 import (
    _flash_attn_bwd_varlen_dkv,
    _flash_attn_bwd_varlen_dq_fused,
    _get_bias_strides_dense,
    _get_bias_strides_varlen,
    _parse_philox,
)

logger = logging.getLogger(__name__)


_CUDNN_DQ_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
]

_CUDNN_DKV_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
]


@libentry()
@triton.autotune(
    configs=_CUDNN_DQ_CONFIGS,
    key=["seqlen_q", "seqlen_k", "HEAD_DIM_QK", "HEAD_DIM_V"],
)
@triton.jit
def _cudnn_attn_bwd_dq_dual_dim(
    Q,
    K,
    V,
    dOut,
    Out,
    L,
    D,
    dQ,
    AttnBias,
    alibi_slopes,
    sm_scale,
    group_size,
    window_size_left,
    window_size_right,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_km,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vm,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lh,
    stride_db,
    stride_dh,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_bn,
    seqlen_q,
    seqlen_k,
    dim_qk,
    dim_v,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_q_idx = tl.program_id(2)
    head_k_idx = head_q_idx // group_size

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_dk = tl.arange(0, HEAD_DIM_QK)
    offs_dv = tl.arange(0, HEAD_DIM_V)
    mask_m = offs_m < seqlen_q
    mask_dk = offs_dk < dim_qk
    mask_dv = offs_dv < dim_v

    q_base = batch_idx * stride_qb + head_q_idx * stride_qh
    o_base = batch_idx * stride_ob + head_q_idx * stride_oh

    q = tl.load(
        Q + q_base + offs_m[:, None] * stride_qm + offs_dk[None, :] * stride_qd,
        mask=mask_m[:, None] & mask_dk[None, :],
        other=0.0,
    )
    do = tl.load(
        dOut + o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
        mask=mask_m[:, None] & mask_dv[None, :],
        other=0.0,
    )
    o = tl.load(
        Out + o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
        mask=mask_m[:, None] & mask_dv[None, :],
        other=0.0,
    ).to(tl.float32)
    Di = tl.sum(o * do.to(tl.float32), axis=1)

    tl.store(
        D + batch_idx * stride_db + head_q_idx * stride_dh + offs_m,
        Di,
        mask=mask_m,
    )

    Li = tl.load(
        L + batch_idx * stride_lb + head_q_idx * stride_lh + offs_m,
        mask=mask_m,
        other=0.0,
    )

    start_n_min = 0
    start_n_max = tl.cdiv(seqlen_k, BLOCK_N)
    if HAS_WINDOW:
        start_n_min = tl.maximum(
            start_n_min,
            (start_m * BLOCK_M - window_size_left) // BLOCK_N,
        )
        start_n_max = tl.minimum(
            start_n_max,
            ((start_m + 1) * BLOCK_M - 1 + window_size_right + BLOCK_N) // BLOCK_N,
        )
    if IS_CAUSAL:
        start_n_max = tl.minimum(
            start_n_max,
            ((start_m + 1) * BLOCK_M - 1 + BLOCK_N) // BLOCK_N,
        )

    dq = tl.zeros([BLOCK_M, HEAD_DIM_QK], dtype=tl.float32)

    if HAS_ALIBI:
        alibi_slope = tl.load(alibi_slopes + head_q_idx)

    k_base = batch_idx * stride_kb + head_k_idx * stride_kh
    v_base = batch_idx * stride_vb + head_k_idx * stride_vh

    for start_n in range(start_n_min, start_n_max):
        offs_n_c = start_n * BLOCK_N + offs_n
        mask_n = offs_n_c < seqlen_k

        k = tl.load(
            K + k_base + offs_n_c[:, None] * stride_km + offs_dk[None, :] * stride_kd,
            mask=mask_n[:, None] & mask_dk[None, :],
            other=0.0,
        )
        v = tl.load(
            V + v_base + offs_n_c[:, None] * stride_vm + offs_dv[None, :] * stride_vd,
            mask=mask_n[:, None] & mask_dv[None, :],
            other=0.0,
        )

        s = tl.dot(q, tl.trans(k)) * sm_scale

        if HAS_BIAS:
            bb = batch_idx * stride_bb + head_q_idx * stride_bh
            bias_block = tl.load(
                AttnBias
                + bb
                + offs_m[:, None] * stride_bm
                + offs_n_c[None, :] * stride_bn,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
            s = s + bias_block.to(tl.float32)

        dist = offs_m[:, None] - offs_n_c[None, :]
        mask_s = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            mask_s = mask_s & (dist >= 0)
        if HAS_WINDOW:
            mask_s = mask_s & (dist <= window_size_left) & (dist >= -window_size_right)
        if HAS_ALIBI:
            s = s - alibi_slope * tl.abs(dist).to(tl.float32)

        s = tl.where(mask_s, s.to(tl.float32), float("-inf"))
        p = tl.exp(s - Li[:, None])
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)

        ds = p * (dp - Di[:, None]) * sm_scale
        dq = dq + tl.dot(ds.to(q.dtype), k)

    tl.store(
        dQ + q_base + offs_m[:, None] * stride_qm + offs_dk[None, :] * stride_qd,
        dq.to(q.dtype),
        mask=mask_m[:, None] & mask_dk[None, :],
    )


@libentry()
@triton.autotune(
    configs=_CUDNN_DKV_CONFIGS,
    key=["seqlen_q", "seqlen_k", "HEAD_DIM_QK", "HEAD_DIM_V"],
)
@triton.jit
def _cudnn_attn_bwd_dkv_dual_dim(
    Q,
    K,
    V,
    dOut,
    L,
    D,
    dK,
    dV,
    AttnBias,
    alibi_slopes,
    sm_scale,
    group_size,
    window_size_left,
    window_size_right,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_km,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vm,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lh,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_bn,
    seqlen_q,
    seqlen_k,
    dim_qk,
    dim_v,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_n = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_k_idx = tl.program_id(2)

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_dk = tl.arange(0, HEAD_DIM_QK)
    offs_dv = tl.arange(0, HEAD_DIM_V)
    mask_n = offs_n < seqlen_k
    mask_dk = offs_dk < dim_qk
    mask_dv = offs_dv < dim_v

    k_base = batch_idx * stride_kb + head_k_idx * stride_kh
    v_base = batch_idx * stride_vb + head_k_idx * stride_vh

    if IS_CAUSAL:
        if start_n * BLOCK_N >= seqlen_q:
            dk_zero = tl.zeros([BLOCK_N, HEAD_DIM_QK], dtype=tl.float32)
            dv_zero = tl.zeros([BLOCK_N, HEAD_DIM_V], dtype=tl.float32)
            k_tmp = tl.load(
                K + k_base + offs_n[:, None] * stride_km + offs_dk[None, :] * stride_kd,
                mask=mask_n[:, None] & mask_dk[None, :],
                other=0.0,
            )
            v_tmp = tl.load(
                V + v_base + offs_n[:, None] * stride_vm + offs_dv[None, :] * stride_vd,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            tl.store(
                dK
                + k_base
                + offs_n[:, None] * stride_km
                + offs_dk[None, :] * stride_kd,
                dk_zero.to(k_tmp.dtype),
                mask=mask_n[:, None] & mask_dk[None, :],
            )
            tl.store(
                dV
                + v_base
                + offs_n[:, None] * stride_vm
                + offs_dv[None, :] * stride_vd,
                dv_zero.to(v_tmp.dtype),
                mask=mask_n[:, None] & mask_dv[None, :],
            )
            return

    k = tl.load(
        K + k_base + offs_n[:, None] * stride_km + offs_dk[None, :] * stride_kd,
        mask=mask_n[:, None] & mask_dk[None, :],
        other=0.0,
    )
    v = tl.load(
        V + v_base + offs_n[:, None] * stride_vm + offs_dv[None, :] * stride_vd,
        mask=mask_n[:, None] & mask_dv[None, :],
        other=0.0,
    )

    dk = tl.zeros([BLOCK_N, HEAD_DIM_QK], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM_V], dtype=tl.float32)

    num_tiles_m = tl.cdiv(seqlen_q, BLOCK_M)
    global_m_min = 0
    global_m_max = num_tiles_m

    if HAS_WINDOW:
        global_m_max = tl.minimum(
            global_m_max,
            ((start_n + 1) * BLOCK_N - 1 + window_size_left + BLOCK_M) // BLOCK_M,
        )
        global_m_min = tl.maximum(
            global_m_min,
            (start_n * BLOCK_N - window_size_right) // BLOCK_M,
        )

    if IS_CAUSAL:
        global_m_min = tl.maximum(global_m_min, (start_n * BLOCK_N) // BLOCK_M)

    valid_tiles_m = global_m_max - global_m_min
    if valid_tiles_m <= 0:
        tl.store(
            dK + k_base + offs_n[:, None] * stride_km + offs_dk[None, :] * stride_kd,
            dk.to(k.dtype),
            mask=mask_n[:, None] & mask_dk[None, :],
        )
        tl.store(
            dV + v_base + offs_n[:, None] * stride_vm + offs_dv[None, :] * stride_vd,
            dv.to(v.dtype),
            mask=mask_n[:, None] & mask_dv[None, :],
        )
        return

    total_iters = group_size * valid_tiles_m

    for linear in range(total_iters):
        gq_offset = linear // valid_tiles_m
        tile_m = global_m_min + linear % valid_tiles_m
        head_q_idx = head_k_idx * group_size + gq_offset

        q_base = batch_idx * stride_qb + head_q_idx * stride_qh
        o_base = batch_idx * stride_ob + head_q_idx * stride_oh
        ld_base = batch_idx * stride_lb + head_q_idx * stride_lh

        offs_m_c = tile_m * BLOCK_M + offs_m
        mask_m = offs_m_c < seqlen_q

        q = tl.load(
            Q + q_base + offs_m_c[:, None] * stride_qm + offs_dk[None, :] * stride_qd,
            mask=mask_m[:, None] & mask_dk[None, :],
            other=0.0,
        )
        do = tl.load(
            dOut
            + o_base
            + offs_m_c[:, None] * stride_om
            + offs_dv[None, :] * stride_od,
            mask=mask_m[:, None] & mask_dv[None, :],
            other=0.0,
        )
        Li = tl.load(L + ld_base + offs_m_c, mask=mask_m, other=0.0)
        Di = tl.load(D + ld_base + offs_m_c, mask=mask_m, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale

        if HAS_BIAS:
            bb = batch_idx * stride_bb + head_q_idx * stride_bh
            bias_block = tl.load(
                AttnBias
                + bb
                + offs_m_c[:, None] * stride_bm
                + offs_n[None, :] * stride_bn,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
            s = s + bias_block.to(tl.float32)

        dist = offs_m_c[:, None] - offs_n[None, :]
        mask_s = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            mask_s = mask_s & (dist >= 0)
        if HAS_WINDOW:
            mask_s = mask_s & (dist <= window_size_left) & (dist >= -window_size_right)
        if HAS_ALIBI:
            alibi_slope = tl.load(alibi_slopes + head_q_idx)
            s = s - alibi_slope * tl.abs(dist).to(tl.float32)

        s = tl.where(mask_s, s.to(tl.float32), float("-inf"))
        p = tl.exp(s - Li[:, None])
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)

        ds = p * (dp - Di[:, None]) * sm_scale
        dk = dk + tl.dot(tl.trans(ds.to(q.dtype)), q)
        dv = dv + tl.dot(tl.trans(p.to(q.dtype)), do)

    tl.store(
        dK + k_base + offs_n[:, None] * stride_km + offs_dk[None, :] * stride_kd,
        dk.to(k.dtype),
        mask=mask_n[:, None] & mask_dk[None, :],
    )
    tl.store(
        dV + v_base + offs_n[:, None] * stride_vm + offs_dv[None, :] * stride_vd,
        dv.to(v.dtype),
        mask=mask_n[:, None] & mask_dv[None, :],
    )


# ---------------------------------------------------------------------------
# Dense-path kernels (fork of the public flash_attention_backward bodies).
#
# The dense branch used to import `_flash_attn_bwd_dq_fused` /
# `_flash_attn_bwd_dkv` and call them with the DS-placeholder argument tail
# added by commit c878d6faa; but only `flash_attention_backward_gcu400`'s
# 64/61-param signatures accept that tail, while the module the import
# actually resolves to exposes the 58/54-param public bodies, so the launch
# died with "dynamic_func() got multiple values for argument 'HEAD_DIM'".
# The gcu400 DS variants are also measurably slower on this op's dense shapes
# (their autotune key omits HAS_BIAS and their config space lacks the
# num_stages=3 tiles), regressing the benchmark by 10-25% versus the recorded
# baseline.  Forking the public bodies here keeps this op self-contained and
# lets the tune space serve cudnn shapes specifically: the public nw=8 and
# num_stages=3 tiles are kept alongside the DS low-warp pins and the wider
# (128,128)/(64,256) tiles, and HAS_BIAS stays part of the autotune key (the
# DS key omitted it, letting bias-tuned configs leak into dense shapes).
# ---------------------------------------------------------------------------
_CUDNN_DQ_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    # GCU400 (L600) low-warp tiles (verified on the identical kernel body by
    # the flash_attention_backward gcu400 tune, -34%..-50% for the dq kernel).
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=2, num_stages=2),
    # num_warps=8 tiles from the public space: the recorded baseline tuned
    # causal D=128 shapes onto these, and they remain reachable now that
    # IS_CAUSAL is part of the autotune key.
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
]

_CUDNN_DKV_CONFIGS = [
    # Landscape retuned for the [N, M]-formulation kernel: the 2-warp
    # (128, 64) tile wins the dense non-causal shapes, (128, 64) ns=3 wins
    # causal, and the 2-warp (128, 128) tile wins the bias shapes.
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=1, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
]


@libentry()
@triton.autotune(
    configs=_CUDNN_DQ_CONFIGS,
    # IS_CAUSAL must be part of the key: with the low-warp tiles in the pool
    # the causal and non-causal optima diverge, and a non-causal-tuned winner
    # reused by the causal pass regresses 10-12% (gcu400 flash_attention_
    # backward tune finding, verified on this same kernel body).
    key=[
        "seqlen_q",
        "seqlen_k",
        "HEAD_DIM",
        "HAS_BIAS",
        "DO_BIAS_GRAD",
        "IS_CAUSAL",
    ],
)
@triton.jit
def _cudnn_attn_bwd_dq(
    Q,
    K,
    V,
    dOut,
    Out,
    L,
    D,
    dQ,
    AttnBias,
    dBias,
    alibi_slopes,
    sm_scale,
    group_size,
    window_size_left,
    window_size_right,
    dropout_p,
    drop_scale,
    seed,
    base_offset,
    drop_stride_b,
    drop_stride_h,
    drop_stride_m,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_km,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vm,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lh,
    stride_db,
    stride_dh,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_bn,
    seqlen_q,
    seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DO_BIAS_GRAD: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_q_idx = tl.program_id(2)
    head_k_idx = head_q_idx // group_size

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < seqlen_q

    q_base = batch_idx * stride_qb + head_q_idx * stride_qh
    o_base = batch_idx * stride_ob + head_q_idx * stride_oh

    q = tl.load(
        Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None],
        other=0.0,
    )
    do = tl.load(
        dOut + o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    )
    o = tl.load(
        Out + o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    Di = tl.sum(o * do.to(tl.float32), axis=1)

    tl.store(
        D + batch_idx * stride_db + head_q_idx * stride_dh + offs_m,
        Di,
        mask=mask_m,
    )

    Li = tl.load(
        L + batch_idx * stride_lb + head_q_idx * stride_lh + offs_m,
        mask=mask_m,
        other=0.0,
    )

    start_n_min = 0
    start_n_max = tl.cdiv(seqlen_k, BLOCK_N)
    if HAS_WINDOW:
        start_n_min = tl.maximum(
            start_n_min,
            (start_m * BLOCK_M - window_size_left) // BLOCK_N,
        )
        start_n_max = tl.minimum(
            start_n_max,
            ((start_m + 1) * BLOCK_M - 1 + window_size_right + BLOCK_N) // BLOCK_N,
        )
    if IS_CAUSAL:
        start_n_max = tl.minimum(
            start_n_max,
            ((start_m + 1) * BLOCK_M - 1 + BLOCK_N) // BLOCK_N,
        )

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if HAS_ALIBI:
        alibi_slope = tl.load(alibi_slopes + head_q_idx)

    k_base = batch_idx * stride_kb + head_k_idx * stride_kh
    v_base = batch_idx * stride_vb + head_k_idx * stride_vh

    for start_n in range(start_n_min, start_n_max):
        offs_n_c = start_n * BLOCK_N + offs_n
        mask_n = offs_n_c < seqlen_k

        k = tl.load(
            K + k_base + offs_n_c[:, None] * stride_km + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        v = tl.load(
            V + v_base + offs_n_c[:, None] * stride_vm + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        )

        s = tl.dot(q, tl.trans(k)) * sm_scale

        if HAS_BIAS:
            bb = batch_idx * stride_bb + head_q_idx * stride_bh
            bias_block = tl.load(
                AttnBias
                + bb
                + offs_m[:, None] * stride_bm
                + offs_n_c[None, :] * stride_bn,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
            s = s + bias_block.to(tl.float32)

        dist = offs_m[:, None] - offs_n_c[None, :]
        mask_s = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            mask_s = mask_s & (dist >= 0)
        if HAS_WINDOW:
            mask_s = mask_s & (dist <= window_size_left) & (dist >= -window_size_right)
        if HAS_ALIBI:
            s = s - alibi_slope * tl.abs(dist).to(tl.float32)

        s = tl.where(mask_s, s.to(tl.float32), float("-inf"))
        p = tl.exp(s - Li[:, None])
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)

        ds = p * (dp - Di[:, None]) * sm_scale
        dq = dq + tl.dot(ds.to(q.dtype), k)

        if DO_BIAS_GRAD:
            ds_bias = p * (dp - Di[:, None])
            bb_out = batch_idx * stride_bb + head_q_idx * stride_bh
            tl.store(
                dBias
                + bb_out
                + offs_m[:, None] * stride_bm
                + offs_n_c[None, :] * stride_bn,
                ds_bias,
                mask=mask_m[:, None] & mask_n[None, :],
            )

    tl.store(
        dQ + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        dq.to(q.dtype),
        mask=mask_m[:, None],
    )


@libentry()
@triton.autotune(
    configs=_CUDNN_DKV_CONFIGS,
    # IS_CAUSAL joins the key for the same divergence reason as the dq kernel.
    key=["seqlen_q", "seqlen_k", "HEAD_DIM", "HAS_BIAS", "IS_CAUSAL"],
)
@triton.jit
def _cudnn_attn_bwd_dkv(
    Q,
    K,
    V,
    dOut,
    L,
    D,
    dK,
    dV,
    AttnBias,
    alibi_slopes,
    sm_scale,
    group_size,
    window_size_left,
    window_size_right,
    dropout_p,
    drop_scale,
    seed,
    base_offset,
    drop_stride_b,
    drop_stride_h,
    drop_stride_m,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_km,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vm,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lh,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_bn,
    seqlen_q,
    seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_n = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_k_idx = tl.program_id(2)

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < seqlen_k

    k_base = batch_idx * stride_kb + head_k_idx * stride_kh
    v_base = batch_idx * stride_vb + head_k_idx * stride_vh

    if IS_CAUSAL:
        if start_n * BLOCK_N >= seqlen_q:
            dk_zero = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            dv_zero = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            tl.store(
                dK + k_base + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
                dk_zero.to(dK.dtype.element_ty),
                mask=mask_n[:, None],
            )
            tl.store(
                dV + v_base + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
                dv_zero.to(dV.dtype.element_ty),
                mask=mask_n[:, None],
            )
            return

    k = tl.load(
        K + k_base + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None],
        other=0.0,
    )
    v = tl.load(
        V + v_base + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
        mask=mask_n[:, None],
        other=0.0,
    )

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    num_tiles_m = tl.cdiv(seqlen_q, BLOCK_M)
    global_m_min = 0
    global_m_max = num_tiles_m

    if HAS_WINDOW:
        global_m_max = tl.minimum(
            global_m_max,
            ((start_n + 1) * BLOCK_N - 1 + window_size_left + BLOCK_M) // BLOCK_M,
        )
        global_m_min = tl.maximum(
            global_m_min,
            (start_n * BLOCK_N - window_size_right) // BLOCK_M,
        )

    if IS_CAUSAL:
        global_m_min = tl.maximum(global_m_min, (start_n * BLOCK_N) // BLOCK_M)

    valid_tiles_m = global_m_max - global_m_min
    if valid_tiles_m <= 0:
        tl.store(
            dK + k_base + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
            dk.to(k.dtype),
            mask=mask_n[:, None],
        )
        tl.store(
            dV + v_base + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
            dv.to(v.dtype),
            mask=mask_n[:, None],
        )
        return

    total_iters = group_size * valid_tiles_m

    # [N, M] formulation (GCU400): q/do are loaded as [D, BLOCK_M] tiles and
    # s/dp are computed transposed via dot(k, qT) / dot(v, doT). The two
    # per-iteration transposes then act on the small fp16 [D, BLOCK_M] tiles
    # feeding the accumulation dots instead of on the [BLOCK_M, BLOCK_N]
    # fp32 p/ds tiles, which the reference formulation transposes through DSM
    # every iteration. Measured on L600: dkv 3.148 -> 1.598 ms at
    # (2, 16, 2048, 128) fp16 non-causal (~2.0x), with larger gains on causal
    # (2.8x) and no regression on D=64 / bias shapes.
    for linear in range(total_iters):
        gq_offset = linear // valid_tiles_m
        tile_m = global_m_min + linear % valid_tiles_m
        head_q_idx = head_k_idx * group_size + gq_offset

        q_base = batch_idx * stride_qb + head_q_idx * stride_qh
        o_base = batch_idx * stride_ob + head_q_idx * stride_oh
        ld_base = batch_idx * stride_lb + head_q_idx * stride_lh

        offs_m_c = tile_m * BLOCK_M + offs_m
        mask_m = offs_m_c < seqlen_q

        qT = tl.load(
            Q + q_base + offs_d[:, None] * stride_qd + offs_m_c[None, :] * stride_qm,
            mask=mask_m[None, :],
            other=0.0,
        )
        doT = tl.load(
            dOut + o_base + offs_d[:, None] * stride_od + offs_m_c[None, :] * stride_om,
            mask=mask_m[None, :],
            other=0.0,
        )
        Li = tl.load(L + ld_base + offs_m_c, mask=mask_m, other=0.0)
        Di = tl.load(D + ld_base + offs_m_c, mask=mask_m, other=0.0)

        sT = tl.dot(k, qT) * sm_scale

        mask_sT = mask_n[:, None] & mask_m[None, :]
        distT = offs_n[:, None] - offs_m_c[None, :]
        if IS_CAUSAL:
            mask_sT = mask_sT & (distT <= 0)
        if HAS_WINDOW:
            mask_sT = (
                mask_sT & (distT >= -window_size_left) & (distT <= window_size_right)
            )
        if HAS_ALIBI:
            alibi_slope = tl.load(alibi_slopes + head_q_idx)
            sT = sT - alibi_slope * tl.abs(distT).to(tl.float32)

        if HAS_BIAS:
            bb = batch_idx * stride_bb + head_q_idx * stride_bh
            biasT = tl.load(
                AttnBias
                + bb
                + offs_n[:, None] * stride_bn
                + offs_m_c[None, :] * stride_bm,
                mask=mask_sT,
                other=0.0,
            )
            sT = sT + biasT.to(tl.float32)

        sT = tl.where(mask_sT, sT.to(tl.float32), float("-inf"))
        pT = tl.exp(sT - Li[None, :])
        dpT = tl.dot(v, doT).to(tl.float32)

        dsT = pT * (dpT - Di[None, :]) * sm_scale
        qTt = tl.trans(qT)
        doTt = tl.trans(doT)
        dk = dk + tl.dot(dsT.to(qTt.dtype), qTt)
        dv = dv + tl.dot(pT.to(qTt.dtype), doTt)

    tl.store(
        dK + k_base + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
        dk.to(k.dtype),
        mask=mask_n[:, None],
    )
    tl.store(
        dV + v_base + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
        dv.to(v.dtype),
        mask=mask_n[:, None],
    )


def _normalize_attn_bias(attn_bias, batch, num_heads, seq_len_q, seq_len_k):
    """Normalize an attention bias to a (batch, heads, seq_q, seq_k) view.

    2D (seq_q, seq_k) and 3D (batch, seq_q, seq_k) biases are unsqueezed
    to 4D, then expanded (stride-0, zero-copy) to the full shape.
    """
    if attn_bias.ndim == 2:
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
    elif attn_bias.ndim == 3:
        attn_bias = attn_bias.unsqueeze(1)
    elif attn_bias.ndim != 4:
        raise ValueError(
            f"attn_bias must be 2D, 3D, or 4D; got {attn_bias.ndim}D tensor"
        )
    return attn_bias.expand(batch, num_heads, seq_len_q, seq_len_k)


def _next_pow2(n: int) -> int:
    return triton.next_power_of_2(n)


def _flash_attn_backward_bhsd(
    dOut,
    Q,
    K,
    V,
    Out,
    L,
    cu_seq_q=None,
    cu_seq_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    is_dropout=False,
    dropout_p=0.0,
    rng_state=None,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
    alibi_slopes=None,
    softmax_scale=None,
    attn_bias=None,
    bias_requires_grad=False,
):
    """Two-kernel flash_attn_backward optimized for BHSD layout.

    Eliminates the BHSD<->BSHD permute+contiguous operations by passing
    BHSD strides directly to the Triton kernels. The BHSD layout gives
    4x better memory coalescing for Q/K/V seq-dim loads (stride=D vs
    stride=H*D in BSHD).
    """
    if is_dropout and dropout_p > 0.0 and rng_state is None:
        raise ValueError("dropout requires rng_state")

    is_varlen = (cu_seq_q is not None) and (cu_seq_k is not None)

    if is_varlen:
        Total_Q, H_q, Head_Dim = Q.shape
        _, H_k, _ = K.shape
        Batch = cu_seq_q.shape[0] - 1
        SeqLen_q = max_seqlen_q
        SeqLen_k = max_seqlen_k
    else:
        Batch, H_q, SeqLen_q, Head_Dim = Q.shape
        _, H_k, SeqLen_k, _ = K.shape

    if H_q % H_k != 0:
        raise ValueError(f"H_q ({H_q}) must be a multiple of H_k ({H_k})")
    if softmax_scale is not None:
        scale = softmax_scale
    else:
        scale = 1.0 / math.sqrt(Head_Dim)
    group_size = H_q // H_k

    use_dropout = is_dropout and (dropout_p > 0.0)
    if use_dropout:
        seed_int, base_offset_int = rng_state
        drop_scale = 1.0 / (1.0 - dropout_p)
        drop_stride_b = H_q * SeqLen_q * SeqLen_k
        drop_stride_h = SeqLen_q * SeqLen_k
        drop_stride_m = SeqLen_k
    else:
        seed_int = base_offset_int = 0
        drop_scale = 1.0
        dropout_p = 0.0
        drop_stride_b = drop_stride_h = drop_stride_m = 0

    _inf_val = SeqLen_q + SeqLen_k + 1
    wl = _inf_val if window_size_left < 0 else window_size_left
    wr = _inf_val if window_size_right < 0 else window_size_right
    has_window = (wl < SeqLen_q) or (wr < SeqLen_k)
    has_alibi = alibi_slopes is not None
    alibi_ptr = alibi_slopes if has_alibi else Q
    has_bias = attn_bias is not None
    do_bias_grad = has_bias and bias_requires_grad

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    Out = Out.contiguous()
    dOut = dOut.contiguous()
    L = L.contiguous()
    if has_bias:
        # Keep the stride-0 expanded view: the kernels index the bias through
        # its strides, so broadcast dimensions need no materialization.
        attn_bias = _normalize_attn_bias(attn_bias, Batch, H_q, SeqLen_q, SeqLen_k)

    bias_ptr = attn_bias if has_bias else Q
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    dBias: Optional[torch.Tensor] = None
    if do_bias_grad:
        dBias = torch.empty_like(attn_bias)
    dbias_ptr = dBias if do_bias_grad else Q

    if is_varlen:
        D = torch.empty(Total_Q, H_q, device=Q.device, dtype=torch.float32)
        sb_m, sb_h, sb_n = _get_bias_strides_varlen(attn_bias if has_bias else None)

        grid_dq = lambda META: (
            triton.cdiv(SeqLen_q, META["BLOCK_M"]),
            Batch,
            H_q,
        )
        _flash_attn_bwd_varlen_dq_fused[grid_dq](
            Q,
            K,
            V,
            dOut,
            Out,
            L,
            D,
            dQ,
            bias_ptr,
            dbias_ptr,
            cu_seq_q,
            cu_seq_k,
            alibi_ptr,
            scale,
            group_size,
            wl,
            wr,
            dropout_p,
            drop_scale,
            seed_int,
            base_offset_int,
            max_seqlen_q,
            max_seqlen_k,
            H_q,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            Out.stride(0),
            Out.stride(1),
            Out.stride(2),
            sb_m,
            sb_h,
            sb_n,
            HEAD_DIM=Head_Dim,
            IS_CAUSAL=is_causal,
            IS_DROPOUT=use_dropout,
            HAS_WINDOW=has_window,
            HAS_ALIBI=has_alibi,
            HAS_BIAS=has_bias,
            DO_BIAS_GRAD=do_bias_grad,
        )

        grid_dkv = lambda META: (
            triton.cdiv(SeqLen_k, META["BLOCK_N"]),
            Batch,
            H_k,
        )
        _flash_attn_bwd_varlen_dkv[grid_dkv](
            Q,
            K,
            V,
            dOut,
            L,
            D,
            dK,
            dV,
            bias_ptr,
            cu_seq_q,
            cu_seq_k,
            alibi_ptr,
            scale,
            group_size,
            wl,
            wr,
            dropout_p,
            drop_scale,
            seed_int,
            base_offset_int,
            max_seqlen_q,
            max_seqlen_k,
            H_q,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            Out.stride(0),
            Out.stride(1),
            Out.stride(2),
            sb_m,
            sb_h,
            sb_n,
            HEAD_DIM=Head_Dim,
            IS_CAUSAL=is_causal,
            IS_DROPOUT=use_dropout,
            HAS_WINDOW=has_window,
            HAS_ALIBI=has_alibi,
            HAS_BIAS=has_bias,
        )
    else:
        assert L.shape == (
            Batch,
            H_q,
            SeqLen_q,
        ), f"L.shape {L.shape} != ({Batch}, {H_q}, {SeqLen_q})"
        D = torch.empty_like(L)
        sb_b, sb_h, sb_m, sb_n = _get_bias_strides_dense(
            attn_bias if has_bias else None
        )

        grid_dq = lambda META: (
            triton.cdiv(SeqLen_q, META["BLOCK_M"]),
            Batch,
            H_q,
        )
        _cudnn_attn_bwd_dq[grid_dq](
            Q,
            K,
            V,
            dOut,
            Out,
            L,
            D,
            dQ,
            bias_ptr,
            dbias_ptr,
            alibi_ptr,
            scale,
            group_size,
            wl,
            wr,
            dropout_p,
            drop_scale,
            seed_int,
            base_offset_int,
            drop_stride_b,
            drop_stride_h,
            drop_stride_m,
            Q.stride(0),
            Q.stride(2),
            Q.stride(1),
            Q.stride(3),
            K.stride(0),
            K.stride(2),
            K.stride(1),
            K.stride(3),
            V.stride(0),
            V.stride(2),
            V.stride(1),
            V.stride(3),
            Out.stride(0),
            Out.stride(2),
            Out.stride(1),
            Out.stride(3),
            L.stride(0),
            L.stride(1),
            D.stride(0),
            D.stride(1),
            sb_b,
            sb_h,
            sb_m,
            sb_n,
            SeqLen_q,
            SeqLen_k,
            HEAD_DIM=Head_Dim,
            IS_CAUSAL=is_causal,
            IS_DROPOUT=use_dropout,
            HAS_WINDOW=has_window,
            HAS_ALIBI=has_alibi,
            HAS_BIAS=has_bias,
            DO_BIAS_GRAD=do_bias_grad,
        )

        grid_dkv = lambda META: (
            triton.cdiv(SeqLen_k, META["BLOCK_N"]),
            Batch,
            H_k,
        )
        _cudnn_attn_bwd_dkv[grid_dkv](
            Q,
            K,
            V,
            dOut,
            L,
            D,
            dK,
            dV,
            bias_ptr,
            alibi_ptr,
            scale,
            group_size,
            wl,
            wr,
            dropout_p,
            drop_scale,
            seed_int,
            base_offset_int,
            drop_stride_b,
            drop_stride_h,
            drop_stride_m,
            Q.stride(0),
            Q.stride(2),
            Q.stride(1),
            Q.stride(3),
            K.stride(0),
            K.stride(2),
            K.stride(1),
            K.stride(3),
            V.stride(0),
            V.stride(2),
            V.stride(1),
            V.stride(3),
            Out.stride(0),
            Out.stride(2),
            Out.stride(1),
            Out.stride(3),
            L.stride(0),
            L.stride(1),
            sb_b,
            sb_h,
            sb_m,
            sb_n,
            SeqLen_q,
            SeqLen_k,
            HEAD_DIM=Head_Dim,
            IS_CAUSAL=is_causal,
            IS_DROPOUT=use_dropout,
            HAS_WINDOW=has_window,
            HAS_ALIBI=has_alibi,
            HAS_BIAS=has_bias,
        )

    return dQ, dK, dV, dBias


def _flash_attn_backward_bhsd_dual_dim(
    dOut,
    Q,
    K,
    V,
    Out,
    L,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
    alibi_slopes=None,
    softmax_scale=None,
    attn_bias=None,
):
    """Dense no-dropout backward with independent q/k and value head dims.

    Mirrors the dense branch of ``_flash_attn_backward_bhsd`` but passes
    separate ``HEAD_DIM_QK`` and ``HEAD_DIM_V`` constexprs to kernels that
    use distinct offsets for the q/k dim and the value dim. Head dims that
    are not powers of two are handled by padding the constexpr to the next
    power of two and masking the tail inside the kernels.
    """
    Batch, H_q, SeqLen_q, Head_Dim_qk = Q.shape
    _, H_k, SeqLen_k, _ = K.shape
    Head_Dim_v = V.shape[-1]

    if H_q % H_k != 0:
        raise ValueError(f"H_q ({H_q}) must be a multiple of H_k ({H_k})")
    if softmax_scale is not None:
        scale = softmax_scale
    else:
        scale = 1.0 / math.sqrt(Head_Dim_qk)
    group_size = H_q // H_k

    _inf_val = SeqLen_q + SeqLen_k + 1
    wl = _inf_val if window_size_left < 0 else window_size_left
    wr = _inf_val if window_size_right < 0 else window_size_right
    has_window = (wl < SeqLen_q) or (wr < SeqLen_k)
    has_alibi = alibi_slopes is not None
    alibi_ptr = alibi_slopes if has_alibi else Q
    has_bias = attn_bias is not None

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    Out = Out.contiguous()
    dOut = dOut.contiguous()
    L = L.contiguous()
    if has_bias:
        # Keep the stride-0 expanded view: the kernels index the bias through
        # its strides, so broadcast dimensions need no materialization.
        attn_bias = _normalize_attn_bias(attn_bias, Batch, H_q, SeqLen_q, SeqLen_k)

    bias_ptr = attn_bias if has_bias else Q
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    sb_b, sb_h, sb_m, sb_n = _get_bias_strides_dense(attn_bias if has_bias else None)

    D = torch.empty_like(L)

    head_dim_qk = _next_pow2(Head_Dim_qk)
    head_dim_v = _next_pow2(Head_Dim_v)

    grid_dq = lambda META: (
        triton.cdiv(SeqLen_q, META["BLOCK_M"]),
        Batch,
        H_q,
    )
    _cudnn_attn_bwd_dq_dual_dim[grid_dq](
        Q,
        K,
        V,
        dOut,
        Out,
        L,
        D,
        dQ,
        bias_ptr,
        alibi_ptr,
        scale,
        group_size,
        wl,
        wr,
        Q.stride(0),
        Q.stride(2),
        Q.stride(1),
        Q.stride(3),
        K.stride(0),
        K.stride(2),
        K.stride(1),
        K.stride(3),
        V.stride(0),
        V.stride(2),
        V.stride(1),
        V.stride(3),
        Out.stride(0),
        Out.stride(2),
        Out.stride(1),
        Out.stride(3),
        L.stride(0),
        L.stride(1),
        D.stride(0),
        D.stride(1),
        sb_b,
        sb_h,
        sb_m,
        sb_n,
        SeqLen_q,
        SeqLen_k,
        Head_Dim_qk,
        Head_Dim_v,
        HEAD_DIM_QK=head_dim_qk,
        HEAD_DIM_V=head_dim_v,
        IS_CAUSAL=is_causal,
        HAS_WINDOW=has_window,
        HAS_ALIBI=has_alibi,
        HAS_BIAS=has_bias,
    )

    grid_dkv = lambda META: (
        triton.cdiv(SeqLen_k, META["BLOCK_N"]),
        Batch,
        H_k,
    )
    _cudnn_attn_bwd_dkv_dual_dim[grid_dkv](
        Q,
        K,
        V,
        dOut,
        L,
        D,
        dK,
        dV,
        bias_ptr,
        alibi_ptr,
        scale,
        group_size,
        wl,
        wr,
        Q.stride(0),
        Q.stride(2),
        Q.stride(1),
        Q.stride(3),
        K.stride(0),
        K.stride(2),
        K.stride(1),
        K.stride(3),
        V.stride(0),
        V.stride(2),
        V.stride(1),
        V.stride(3),
        Out.stride(0),
        Out.stride(2),
        Out.stride(1),
        Out.stride(3),
        L.stride(0),
        L.stride(1),
        sb_b,
        sb_h,
        sb_m,
        sb_n,
        SeqLen_q,
        SeqLen_k,
        Head_Dim_qk,
        Head_Dim_v,
        HEAD_DIM_QK=head_dim_qk,
        HEAD_DIM_V=head_dim_v,
        IS_CAUSAL=is_causal,
        HAS_WINDOW=has_window,
        HAS_ALIBI=has_alibi,
        HAS_BIAS=has_bias,
    )

    return dQ, dK, dV


def cudnn_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    philox_seed,
    philox_offset,
    attn_bias,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    *,
    scale=None,
):
    """Backward kernel for cuDNN attention -- BHSD-native, zero-copy path.

    Computes gradients of query, key, and value for the cuDNN attention
    forward operator. Inputs and outputs are in BHSD (batch, heads, seq,
    dim) layout, matching the cuDNN forward convention.

    This implementation passes BHSD-strided arguments directly to the
    Triton flash attention backward kernels, eliminating the expensive
    BHSD<->BSHD permute+contiguous operations present in the generic
    ``_scaled_dot_product_cudnn_attention_backward`` path. The BHSD
    layout additionally provides 4x better memory coalescing for Q/K/V
    seq-dim loads (stride=D vs stride=H*D in BSHD).
    """
    logger.debug("GEMS_ENFLAME CUDNN_ATTENTION_BACKWARD")

    lse = logsumexp.float()
    if lse.ndim == 4 and lse.shape[-1] == 1:
        lse = lse.squeeze(-1)

    if dropout_p > 0.0:
        raise NotImplementedError(
            "cudnn_attention_backward: dropout > 0 is not yet supported"
        )

    is_dropout = dropout_p > 0.0
    rng_tuple = _parse_philox(philox_seed, philox_offset) if is_dropout else None
    # Reject if either bound is given: a one-sided cum_seq would otherwise be
    # silently ignored and fall through to the dense path.
    if (cum_seq_q is not None) or (cum_seq_k is not None):
        raise NotImplementedError(
            "cudnn_attention_backward: varlen is not yet supported"
        )

    head_dim_qk = query.shape[-1]
    head_dim_v = value.shape[-1]
    if head_dim_qk != head_dim_v or triton.next_power_of_2(head_dim_qk) != head_dim_qk:
        # The shared flash backward kernels use a single HEAD_DIM constexpr
        # for q/k and value, which requires equal power-of-two head dims.
        # Route everything else to the dual-dim kernels: they cover
        # Dqk != Dv as well as equal non-power-of-two dims via padding.
        dQ, dK, dV = _flash_attn_backward_bhsd_dual_dim(
            grad_out,
            query,
            key,
            value,
            out,
            lse,
            is_causal=is_causal,
            softmax_scale=scale,
            attn_bias=attn_bias,
        )
        return dQ, dK, dV

    dQ, dK, dV, _ = _flash_attn_backward_bhsd(
        grad_out,
        query,
        key,
        value,
        out,
        lse,
        max_seqlen_q=int(max_q),
        max_seqlen_k=int(max_k),
        is_dropout=is_dropout,
        dropout_p=dropout_p,
        rng_state=rng_tuple,
        is_causal=is_causal,
        softmax_scale=scale,
        attn_bias=attn_bias,
        bias_requires_grad=False,
    )
    return dQ, dK, dV
