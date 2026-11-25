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

import contextlib
import logging
import math
import struct
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device

try:
    from triton.knobs import autotuning as _autotuning_knobs
except (ImportError, ModuleNotFoundError):
    # Triton < 3.6 does not have triton.knobs module
    _autotuning_knobs = None

logger = logging.getLogger(__name__)

_COMPENSATED_DV_PRECISION = "split"
if device.vendor_name == "ascend":
    from torch_npu.npu.utils import get_cann_version

    # CANN 9.1.1 miscompiles the FP32 dV dot. Keep IEEE accumulation on
    # toolchains where it works, particularly for causal attention.
    _COMPENSATED_DV_PRECISION = (
        "scaled_split" if get_cann_version() == "9.1.1" else "ieee"
    )


@contextlib.contextmanager
def _disable_aabs_for_small_seqlen(seqlen_q, seqlen_k):
    """Temporarily disable FlagTree AABS for very short sequences.

    AABS shrinks a tl.load-driven BLOCK to next_power_of_2(seqlen) when
    BLOCK > seqlen. For seqlen < 16 it lowers BLOCK_M/BLOCK_N below Triton's
    tl.dot K>=16 lower bound (these blocks double as a dot operand), which
    makes _flash_attn_bwd_dq_fused / _flash_attn_bwd_dkv fail to compile with
    "Input shapes should have M >= 1, N >= 1 and K >= 16" (decode: seqlen 1).
    Fall back to the plain (non-adjusted) configs for such shapes.
    """
    # The AABS knob exists only on Triton builds that carry FlagTree's
    # auto-adjust-block-size feature; on stock Triton (knobs present but no
    # such knob) there is nothing to disable.
    if _autotuning_knobs is None or not hasattr(_autotuning_knobs, "adjust_block_size"):
        yield
        return

    if min(seqlen_q, seqlen_k) >= 16:
        yield
        return
    previous = _autotuning_knobs.adjust_block_size
    _autotuning_knobs.adjust_block_size = False
    try:
        yield
    finally:
        _autotuning_knobs.adjust_block_size = previous


def _parse_rng_state(
    rng_state: torch.Tensor, offset: torch.Tensor = None
) -> Tuple[int, int]:
    if rng_state.dim() == 0 and offset is not None:
        # ROCm (AOTriton) passes philox seed and offset as two 0-dim tensors
        # instead of one packed [seed, offset] tensor.
        return int(rng_state.item()), int(offset.item())
    rng_state = rng_state.cpu()
    return int(rng_state[0].item()), int(rng_state[1].item())


def _parse_philox(
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
) -> Tuple[int, int]:
    return int(philox_seed.item()), int(philox_offset.item())


_DQ_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
]

_DKV_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
]


_SINGLE_STAGE_DENSE_CONFIG = triton.Config(
    {"BLOCK_M": 32 if device.vendor_name == "metax" else 64, "BLOCK_N": 64},
    num_warps=8,
    num_stages=1,
)


def _prune_dense_dq_configs(configs, named_args, **kwargs):
    if kwargs["S5000"] and kwargs["HEAD_DIM"] == 128:
        # Pipelined D128 dQ candidates can fault during MUSA autotuning.
        return [cfg for cfg in configs if cfg.num_stages == 1]
    return [cfg for cfg in configs if cfg != _SINGLE_STAGE_DENSE_CONFIG]


@triton.autotune(
    configs=[*_DQ_CONFIGS, _SINGLE_STAGE_DENSE_CONFIG],
    prune_configs_by={"early_config_prune": _prune_dense_dq_configs},
    # HAS_BIAS/DO_BIAS_GRAD must be part of the key: they change the kernel's
    # shared-memory footprint, and a config tuned without an attention bias
    # can exceed the device limit once recompiled with one.
    key=["seqlen_q", "seqlen_k", "HEAD_DIM", "HAS_BIAS", "DO_BIAS_GRAD", "S5000"],
)
@triton.jit
def _flash_attn_bwd_dq_fused(
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
    S5000: tl.constexpr,
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

        if IS_DROPOUT:
            rng_off = (
                base_offset
                + batch_idx * drop_stride_b
                + head_q_idx * drop_stride_h
                + offs_m[:, None] * drop_stride_m
                + offs_n_c[None, :]
            )
            rand_vals = tl.rand(seed, rng_off)
            drop_mask = rand_vals > dropout_p
            dp = tl.where(drop_mask, dp * drop_scale, 0.0)
            p = tl.where(drop_mask, p * drop_scale, 0.0)

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


def _prune_dense_dkv_configs(configs, named_args, **kwargs):
    if kwargs.get("S5000", False):
        # Multi-stage MUSA lowering can corrupt dK and dV on the S5000.
        return [cfg for cfg in configs if cfg.num_stages == 1]

    split_dv = (
        kwargs.get("DV_PRECISION") == "split"
        and device.vendor_name == "metax"
        and kwargs["HEAD_DIM"] == 128
    )
    return [cfg for cfg in configs if cfg != _SINGLE_STAGE_DENSE_CONFIG or split_dv]


@triton.autotune(
    configs=[*_DKV_CONFIGS, _SINGLE_STAGE_DENSE_CONFIG],
    prune_configs_by={"early_config_prune": _prune_dense_dkv_configs},
    # HAS_BIAS must be part of the key: it changes the kernel's shared-memory
    # footprint, and a config tuned without an attention bias can exceed the
    # device limit once recompiled with one.
    key=["seqlen_q", "seqlen_k", "HEAD_DIM", "HAS_BIAS", "S5000", "DV_PRECISION"],
)
@triton.jit
def _flash_attn_bwd_dkv(
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
    S5000: tl.constexpr,
    DV_PRECISION: tl.constexpr,
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

            k_tmp = tl.load(
                K + k_base + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None],
                other=0.0,
            )
            v_tmp = tl.load(
                V + v_base + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None],
                other=0.0,
            )
            tl.store(
                dK + k_base + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
                dk_zero.to(k_tmp.dtype),
                mask=mask_n[:, None],
            )
            tl.store(
                dV + v_base + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
                dv_zero.to(v_tmp.dtype),
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
            Q + q_base + offs_m_c[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None],
            other=0.0,
        )
        do = tl.load(
            dOut + o_base + offs_m_c[:, None] * stride_om + offs_d[None, :] * stride_od,
            mask=mask_m[:, None],
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

        if IS_DROPOUT:
            rng_off = (
                base_offset
                + batch_idx * drop_stride_b
                + head_q_idx * drop_stride_h
                + offs_m_c[:, None] * drop_stride_m
                + offs_n[None, :]
            )
            rand_vals = tl.rand(seed, rng_off)
            drop_mask = rand_vals > dropout_p
            dp = tl.where(drop_mask, dp * drop_scale, 0.0)
            p = tl.where(drop_mask, p * drop_scale, 0.0)

        ds = p * (dp - Di[:, None]) * sm_scale
        dk = dk + tl.dot(tl.trans(ds.to(q.dtype)), q)
        if DV_PRECISION == "ieee":
            dv += tl.dot(tl.trans(p), do.to(tl.float32), input_precision="ieee")
        else:
            p_hi = p.to(q.dtype)
            dv += tl.dot(tl.trans(p_hi), do)
            if DV_PRECISION != "input":
                # Scale the residual before narrowing to retain FP16 subnormals on Ascend.
                residual_scale: tl.constexpr = (
                    4096.0 if DV_PRECISION == "scaled_split" else 1.0
                )
                p_lo = ((p - p_hi.to(tl.float32)) * residual_scale).to(q.dtype)
                dv += tl.dot(tl.trans(p_lo), do) * (1.0 / residual_scale)

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


@triton.autotune(
    configs=_DQ_CONFIGS,
    # HAS_BIAS/DO_BIAS_GRAD must be part of the key: they change the kernel's
    # shared-memory footprint, and a config tuned without an attention bias
    # can exceed the device limit once recompiled with one.
    key=["max_seqlen_q", "max_seqlen_k", "HEAD_DIM", "HAS_BIAS", "DO_BIAS_GRAD"],
)
@triton.jit
def _flash_attn_bwd_varlen_dq_fused(
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
    cu_seq_q,
    cu_seq_k,
    alibi_slopes,
    sm_scale,
    group_size,
    window_size_left,
    window_size_right,
    dropout_p,
    drop_scale,
    seed,
    base_offset,
    max_seqlen_q,
    max_seqlen_k,
    H_q,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_km,
    stride_kh,
    stride_kd,
    stride_vm,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    stride_bm,
    stride_bh,
    stride_bn,
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

    start_q = tl.load(cu_seq_q + batch_idx)
    seqlen_q = tl.load(cu_seq_q + batch_idx + 1) - start_q
    start_k = tl.load(cu_seq_k + batch_idx)
    seqlen_k = tl.load(cu_seq_k + batch_idx + 1) - start_k
    if start_m * BLOCK_M >= seqlen_q:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < seqlen_q
    m_phys = start_q + offs_m

    q = tl.load(
        Q
        + m_phys[:, None] * stride_qm
        + head_q_idx * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None],
        other=0.0,
    )
    do = tl.load(
        dOut
        + m_phys[:, None] * stride_om
        + head_q_idx * stride_oh
        + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    )
    o = tl.load(
        Out
        + m_phys[:, None] * stride_om
        + head_q_idx * stride_oh
        + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    Di = tl.sum(o * do.to(tl.float32), axis=1)

    tl.store(D + m_phys * H_q + head_q_idx, Di, mask=mask_m)

    Li = tl.load(L + m_phys * H_q + head_q_idx, mask=mask_m, other=0.0)

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

    for start_n in range(start_n_min, start_n_max):
        offs_n_c = start_n * BLOCK_N + offs_n
        mask_n = offs_n_c < seqlen_k
        n_phys = start_k + offs_n_c

        k = tl.load(
            K
            + n_phys[:, None] * stride_km
            + head_k_idx * stride_kh
            + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        v = tl.load(
            V
            + n_phys[:, None] * stride_vm
            + head_k_idx * stride_vh
            + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        )

        s = tl.dot(q, tl.trans(k)) * sm_scale

        if HAS_BIAS:
            bias_block = tl.load(
                AttnBias
                + m_phys[:, None] * stride_bm
                + head_q_idx * stride_bh
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

        if IS_DROPOUT:
            rng_off = (
                base_offset
                + batch_idx * (max_seqlen_q * max_seqlen_k)
                + head_q_idx * max_seqlen_k
                + offs_m[:, None] * max_seqlen_k
                + offs_n_c[None, :]
            )
            rand_vals = tl.rand(seed, rng_off)
            drop_mask = rand_vals > dropout_p
            dp = tl.where(drop_mask, dp * drop_scale, 0.0)
            p = tl.where(drop_mask, p * drop_scale, 0.0)

        ds = p * (dp - Di[:, None]) * sm_scale
        dq = dq + tl.dot(ds.to(q.dtype), k)

        if DO_BIAS_GRAD:
            ds_bias = p * (dp - Di[:, None])
            tl.store(
                dBias
                + m_phys[:, None] * stride_bm
                + head_q_idx * stride_bh
                + offs_n_c[None, :] * stride_bn,
                ds_bias,
                mask=mask_m[:, None] & mask_n[None, :],
            )

    tl.store(
        dQ
        + m_phys[:, None] * stride_qm
        + head_q_idx * stride_qh
        + offs_d[None, :] * stride_qd,
        dq.to(q.dtype),
        mask=mask_m[:, None],
    )


@triton.autotune(
    configs=_DKV_CONFIGS,
    # HAS_BIAS must be part of the key: it changes the kernel's shared-memory
    # footprint, and a config tuned without an attention bias can exceed the
    # device limit once recompiled with one.
    key=["max_seqlen_q", "max_seqlen_k", "HEAD_DIM", "HAS_BIAS"],
)
@triton.jit
def _flash_attn_bwd_varlen_dkv(
    Q,
    K,
    V,
    dOut,
    L,
    D,
    dK,
    dV,
    AttnBias,
    cu_seq_q,
    cu_seq_k,
    alibi_slopes,
    sm_scale,
    group_size,
    window_size_left,
    window_size_right,
    dropout_p,
    drop_scale,
    seed,
    base_offset,
    max_seqlen_q,
    max_seqlen_k,
    H_q,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_km,
    stride_kh,
    stride_kd,
    stride_vm,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    stride_bm,
    stride_bh,
    stride_bn,
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

    start_q = tl.load(cu_seq_q + batch_idx)
    seqlen_q = tl.load(cu_seq_q + batch_idx + 1) - start_q
    start_k = tl.load(cu_seq_k + batch_idx)
    seqlen_k = tl.load(cu_seq_k + batch_idx + 1) - start_k
    if start_n * BLOCK_N >= seqlen_k:
        return

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < seqlen_k
    n_phys = start_k + offs_n

    if IS_CAUSAL:
        if start_n * BLOCK_N >= seqlen_q:
            dk_zero = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            dv_zero = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            k_tmp = tl.load(
                K
                + n_phys[:, None] * stride_km
                + head_k_idx * stride_kh
                + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None],
                other=0.0,
            )
            v_tmp = tl.load(
                V
                + n_phys[:, None] * stride_vm
                + head_k_idx * stride_vh
                + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None],
                other=0.0,
            )
            tl.store(
                dK
                + n_phys[:, None] * stride_km
                + head_k_idx * stride_kh
                + offs_d[None, :] * stride_kd,
                dk_zero.to(k_tmp.dtype),
                mask=mask_n[:, None],
            )
            tl.store(
                dV
                + n_phys[:, None] * stride_vm
                + head_k_idx * stride_vh
                + offs_d[None, :] * stride_vd,
                dv_zero.to(v_tmp.dtype),
                mask=mask_n[:, None],
            )
            return

    k = tl.load(
        K
        + n_phys[:, None] * stride_km
        + head_k_idx * stride_kh
        + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None],
        other=0.0,
    )
    v = tl.load(
        V
        + n_phys[:, None] * stride_vm
        + head_k_idx * stride_vh
        + offs_d[None, :] * stride_vd,
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
            dK
            + n_phys[:, None] * stride_km
            + head_k_idx * stride_kh
            + offs_d[None, :] * stride_kd,
            dk.to(k.dtype),
            mask=mask_n[:, None],
        )
        tl.store(
            dV
            + n_phys[:, None] * stride_vm
            + head_k_idx * stride_vh
            + offs_d[None, :] * stride_vd,
            dv.to(v.dtype),
            mask=mask_n[:, None],
        )
        return

    total_iters = group_size * valid_tiles_m

    for linear in range(total_iters):
        gq_offset = linear // valid_tiles_m
        tile_m = global_m_min + linear % valid_tiles_m
        head_q_idx = head_k_idx * group_size + gq_offset

        offs_m_c = tile_m * BLOCK_M + offs_m
        mask_m = offs_m_c < seqlen_q
        m_phys = start_q + offs_m_c

        q = tl.load(
            Q
            + m_phys[:, None] * stride_qm
            + head_q_idx * stride_qh
            + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None],
            other=0.0,
        )
        do = tl.load(
            dOut
            + m_phys[:, None] * stride_om
            + head_q_idx * stride_oh
            + offs_d[None, :] * stride_od,
            mask=mask_m[:, None],
            other=0.0,
        )
        Di = tl.load(D + m_phys * H_q + head_q_idx, mask=mask_m, other=0.0)
        Li = tl.load(L + m_phys * H_q + head_q_idx, mask=mask_m, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale

        if HAS_BIAS:
            bias_block = tl.load(
                AttnBias
                + m_phys[:, None] * stride_bm
                + head_q_idx * stride_bh
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

        if IS_DROPOUT:
            rng_off = (
                base_offset
                + batch_idx * (max_seqlen_q * max_seqlen_k)
                + head_q_idx * max_seqlen_k
                + offs_m_c[:, None] * max_seqlen_k
                + offs_n[None, :]
            )
            rand_vals = tl.rand(seed, rng_off)
            drop_mask = rand_vals > dropout_p
            dp = tl.where(drop_mask, dp * drop_scale, 0.0)
            p = tl.where(drop_mask, p * drop_scale, 0.0)

        ds = p * (dp - Di[:, None]) * sm_scale
        dk = dk + tl.dot(tl.trans(ds.to(q.dtype)), q)
        dv = dv + tl.dot(tl.trans(p.to(q.dtype)), do)

    tl.store(
        dK
        + n_phys[:, None] * stride_km
        + head_k_idx * stride_kh
        + offs_d[None, :] * stride_kd,
        dk.to(k.dtype),
        mask=mask_n[:, None],
    )
    tl.store(
        dV
        + n_phys[:, None] * stride_vm
        + head_k_idx * stride_vh
        + offs_d[None, :] * stride_vd,
        dv.to(v.dtype),
        mask=mask_n[:, None],
    )


def _get_bias_strides_dense(bias):
    if bias is None:
        return 0, 0, 0, 0
    return bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3)


def _get_bias_strides_varlen(bias):
    if bias is None:
        return 0, 0, 0
    return bias.stride(0), bias.stride(1), bias.stride(2)


def flash_attn_backward(
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
    compensate_dv=False,
):
    if is_dropout and dropout_p > 0.0 and rng_state is None:
        raise ValueError("开启 dropout 时，必须传入 rng_state")

    is_varlen = (cu_seq_q is not None) and (cu_seq_k is not None)

    if is_varlen:
        Total_Q, H_q, Head_Dim = Q.shape
        _, H_k, _ = K.shape
        Batch = cu_seq_q.shape[0] - 1
        SeqLen_q = max_seqlen_q
        SeqLen_k = max_seqlen_k
    else:
        Batch, SeqLen_q, H_q, Head_Dim = Q.shape
        _, SeqLen_k, H_k, _ = K.shape

    if H_q % H_k != 0:
        raise ValueError(f"H_q ({H_q}) 必须是 H_k ({H_k}) 的整数倍")
    group_size = H_q // H_k
    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(Head_Dim))

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
        attn_bias = attn_bias.contiguous()

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

        grid_dq = lambda META: (triton.cdiv(SeqLen_q, META["BLOCK_M"]), Batch, H_q)
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

        grid_dkv = lambda META: (triton.cdiv(SeqLen_k, META["BLOCK_N"]), Batch, H_k)
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

        s5000 = (
            Q.device.type == "musa"
            and Head_Dim in (64, 128)
            and Q.dtype in (torch.float16, torch.bfloat16)
            and "S5000" in torch.musa.get_device_name(Q.device)
        )

        with _disable_aabs_for_small_seqlen(SeqLen_q, SeqLen_k):
            grid_dq = lambda META: (triton.cdiv(SeqLen_q, META["BLOCK_M"]), Batch, H_q)
            _flash_attn_bwd_dq_fused[grid_dq](
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
                Q.stride(1),
                Q.stride(2),
                Q.stride(3),
                K.stride(0),
                K.stride(1),
                K.stride(2),
                K.stride(3),
                V.stride(0),
                V.stride(1),
                V.stride(2),
                V.stride(3),
                Out.stride(0),
                Out.stride(1),
                Out.stride(2),
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
                S5000=s5000,
            )

            grid_dkv = lambda META: (triton.cdiv(SeqLen_k, META["BLOCK_N"]), Batch, H_k)
            dv_precision = "input"
            if (
                compensate_dv
                and not use_dropout
                and Q.dtype in (torch.float16, torch.bfloat16)
            ):
                dv_precision = _COMPENSATED_DV_PRECISION
            _flash_attn_bwd_dkv[grid_dkv](
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
                Q.stride(1),
                Q.stride(2),
                Q.stride(3),
                K.stride(0),
                K.stride(1),
                K.stride(2),
                K.stride(3),
                V.stride(0),
                V.stride(1),
                V.stride(2),
                V.stride(3),
                Out.stride(0),
                Out.stride(1),
                Out.stride(2),
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
                DV_PRECISION=dv_precision,
                S5000=s5000,
            )

    return dQ, dK, dV, dBias


_PPU_DROPOUT_LAYOUT = tl.constexpr(device.vendor_name == "thead")


@triton.jit
def _aten_flash_dropout(
    RNG, batch, head, m, n, HQ: tl.constexpr, threshold, drop_scale
):
    # Native forward assigns Philox bytes by its MMA accumulator layout.
    # Counter coordinates are local to each sequence, independent of our tiles.
    seed = tl.load(RNG).to(tl.uint64)
    offset = tl.load(RNG + 1).to(tl.uint64)
    if _PPU_DROPOUT_LAYOUT:
        lane = 4 * (m[:, None] % 8) + n[None, :] % 4
        word = 2 * ((n[None, :] % 32) // 16) + (m[:, None] % 16) // 8
        shift = 8 * ((n[None, :] % 16) // 4)
    else:
        lane = 4 * (m[:, None] % 8) + (n[None, :] % 8) // 2
        word = (n[None, :] % 32) // 8
        shift = 8 * (2 * ((m[:, None] % 16) // 8) + n[None, :] % 2)
    offset = offset + ((batch * HQ + head) * 32 + lane).to(tl.uint64)
    c0, c1 = offset.to(tl.uint32), (offset >> 32).to(tl.uint32)
    c2 = tl.broadcast_to((m[:, None] // 16).to(tl.uint32), c0.shape)
    c3 = tl.broadcast_to((n[None, :] // 32).to(tl.uint32), c0.shape)
    k0, k1 = seed.to(tl.uint32), (seed >> 32).to(tl.uint32)
    for _ in tl.static_range(7):
        hi0 = tl.umulhi(c0, 0xD2511F53)
        hi1 = tl.umulhi(c2, 0xCD9E8D57)
        lo0 = c0 * 0xD2511F53
        lo1 = c2 * 0xCD9E8D57
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = (k0 + 0x9E3779B9).to(tl.uint32)
        k1 = (k1 + 0xBB67AE85).to(tl.uint32)
    bits = tl.where(word == 0, c0, tl.where(word == 1, c1, tl.where(word == 2, c2, c3)))
    random_byte = (bits >> shift.to(tl.uint32)) & 255
    return (random_byte <= threshold).to(tl.float32) * drop_scale


@triton.jit
def _aten_flash_probability(
    q,
    k,
    lse,
    m,
    n,
    sq,
    sk,
    scale,
    scale_log2,
    LSE_SCALE: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    TOP_LEFT_CAUSAL: tl.constexpr,
):
    scores = tl.dot(q, tl.trans(k))
    offset = 0 if TOP_LEFT_CAUSAL else sk - sq
    distance = m[:, None] + offset - n[None, :]
    valid = (m[:, None] < sq) & (n[None, :] < sk)
    if WL >= 0:
        valid &= distance <= WL
    if WR >= 0:
        valid &= distance >= -WR
    # Native forward uses +inf for empty rows; mathematical oracles may use -inf.
    valid &= (lse[:, None] != float("inf")) & (lse[:, None] != -float("inf"))
    if scale_log2 is None:
        probability = tl.exp(scores * scale - lse[:, None])
    else:
        # PyTorch FlashAttention builds with UNFUSE_FMA and rounds the
        # host-computed scale*log2(e) once before reconstructing probabilities.
        probability = tl.exp2(scores * scale_log2 - lse[:, None] * LSE_SCALE)
    return tl.where(valid, probability, 0.0)


# Some vendor dependency finders inspect dead constexpr branches. Define the
# CUDA-only helper only on CUDA so older language APIs are never traversed.
if device.vendor_name == "nvidia":

    @triton.jit
    def _aten_flash_delta(
        o, do, OUT_PTR, DO_PTR, base, rows, DIM: tl.constexpr, BM: tl.constexpr
    ):
        # Preserve native lane/page accumulation and XOR reduction order.
        # Direct loads avoid expensive gathers from the full dot operand tile.
        HD: tl.constexpr = ((DIM + 31) // 32) * 32
        STRIDE: tl.constexpr = 64 if HD % 64 == 0 else 32
        THREADS: tl.constexpr = STRIDE // 8
        lane = tl.arange(0, THREADS)
        col = lane * 8
        index = base[:, None] + col[None, :]
        mask = rows[:, None] & (col[None, :] < DIM)
        a = tl.load(OUT_PTR + index, mask, 0).to(tl.float32)
        b = tl.load(DO_PTR + index, mask, 0).to(tl.float32)
        acc = a * b
        for i in tl.static_range(1, (HD // STRIDE) * 8):
            col = (i // 8) * STRIDE + lane * 8 + i % 8
            index = base[:, None] + col[None, :]
            mask = rows[:, None] & (col[None, :] < DIM)
            a = tl.load(OUT_PTR + index, mask, 0).to(tl.float32)
            b = tl.load(DO_PTR + index, mask, 0).to(tl.float32)
            acc = acc + a * b
        for step in tl.static_range(0, 3 if THREADS == 8 else 2):
            xor_col = tl.broadcast_to(
                (lane ^ (THREADS >> (step + 1)))[None, :], (BM, THREADS)
            )
            acc = acc + tl.gather(acc, xor_col, 1)
        return tl.sum(tl.where(lane[None, :] == 0, acc, 0.0), 1)

else:

    @triton.jit
    def _aten_flash_delta(
        o, do, OUT_PTR, DO_PTR, base, rows, DIM: tl.constexpr, BM: tl.constexpr
    ):
        return tl.sum(o * do.to(tl.float32), 1)


@triton.jit
def _aten_flash_bwd_dq(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    DELTA,
    DQ,
    CQ,
    CK,
    RNG,
    threshold,
    drop_scale,
    keep_prob,
    sq_max,
    sk_max,
    total_q,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    DIM: tl.constexpr,
    BD: tl.constexpr,
    scale,
    scale_log2,
    LSE_SCALE: tl.constexpr,
    LB: tl.constexpr,
    LH: tl.constexpr,
    LT: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    TOP_LEFT_CAUSAL: tl.constexpr,
    VARLEN: tl.constexpr,
    PADDED_LSE: tl.constexpr,
    DROPOUT: tl.constexpr,
    WIDE_SEQUENCE_INDICES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    WRITE_DELTA: tl.constexpr = True,
    ASCEND_LEGACY_COMPILER: tl.constexpr = False,
    FULL_PRECISION: tl.constexpr = False,
):
    block = tl.program_id(0)
    if WIDE_SEQUENCE_INDICES:
        block = block.to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    head = tl.program_id(2).to(tl.int64)
    if VARLEN:
        q0 = tl.load(CQ + batch).to(tl.int64)
        k0 = tl.load(CK + batch).to(tl.int64)
        sq = tl.load(CQ + batch + 1).to(tl.int64) - q0
        sk = tl.load(CK + batch + 1).to(tl.int64) - k0
        lbase = head * LH + q0 * LT
        if PADDED_LSE:
            lbase = batch * LB + head * LH
    else:
        sq, sk = sq_max, sk_max
        if WIDE_SEQUENCE_INDICES:
            sq, sk = tl.cast(sq, tl.int64), tl.cast(sk, tl.int64)
        q0, k0 = batch * sq, batch * sk
        lbase = batch * LB + head * LH
    m = (block * BM + tl.arange(0, BM)).to(tl.int64)
    n = tl.arange(0, BN).to(tl.int64)
    d = tl.arange(0, BD)
    qi = ((q0 + m[:, None]) * HQ + head) * DIM + d[None, :]
    qm = (m[:, None] < sq) & (d[None, :] < DIM)
    q = tl.load(Q + qi, qm, 0)
    do = tl.load(DO + qi, qm, 0)
    o = tl.load(O + qi, qm, 0).to(tl.float32)
    delta = (
        _aten_flash_delta(o, do, O, DO, ((q0 + m) * HQ + head) * DIM, m < sq, DIM, BM)
        * keep_prob
    )
    if WRITE_DELTA:
        tl.store(DELTA + head * total_q + q0 + m, delta, m < sq)
    lse = tl.load(LSE + lbase + m * LT, m < sq, float("inf"))
    dq = tl.full((BM, BD), 0, tl.float32)
    lo = 0
    hi = sk
    offset = 0 if TOP_LEFT_CAUSAL else sk - sq
    if not ASCEND_LEGACY_COMPILER:
        if WL >= 0:
            lo = tl.maximum(0, block * BM + offset - WL)
        if WR >= 0:
            hi = tl.minimum(sk, (block + 1) * BM + offset + WR)
    for tile in range(lo // BN, tl.cdiv(tl.maximum(hi, 0), BN)):
        kn = tile * BN + n
        ki = ((k0 + kn[:, None]) * HK + head // (HQ // HK)) * DIM + d[None, :]
        km = (kn[:, None] < sk) & (d[None, :] < DIM)
        k = tl.load(K + ki, km, 0)
        v = tl.load(V + ki, km, 0)
        p = _aten_flash_probability(
            q,
            k,
            lse,
            m,
            kn,
            sq,
            sk,
            scale,
            scale_log2,
            LSE_SCALE,
            WL,
            WR,
            TOP_LEFT_CAUSAL,
        )
        dp = tl.dot(do, tl.trans(v))
        # Some NPU dot lowerings leave out-of-bounds tile lanes undefined.
        # A zero probability does not suppress a NaN in those lanes.
        dp_valid = (m[:, None] < sq) & (kn[None, :] < sk)
        if ASCEND_LEGACY_COMPILER:
            dp_valid |= dp == 0.0
        dp = tl.where(dp_valid, dp, 0.0)
        if DROPOUT:
            dp *= _aten_flash_dropout(RNG, batch, head, m, kn, HQ, threshold, 1.0)
        ds = p * (dp - delta[:, None])
        ds_valid = (m[:, None] < sq) & (kn[None, :] < sk)
        if ASCEND_LEGACY_COMPILER:
            ds_valid |= ds == 0.0
        ds = tl.where(ds_valid, ds, 0.0)
        if FULL_PRECISION:
            dq += tl.dot(ds, k.to(tl.float32), input_precision="ieee")
        else:
            dq += tl.dot(ds.to(q.dtype), k)
    tl.store(DQ + qi, tl.where(sk > 0, dq * (scale * drop_scale), 0.0), qm)


@triton.jit
def _aten_flash_bwd_dkv(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    CQ,
    CK,
    RNG,
    threshold,
    drop_scale,
    keep_prob,
    sq_max,
    sk_max,
    total_q,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    DIM: tl.constexpr,
    BD: tl.constexpr,
    scale,
    scale_log2,
    LSE_SCALE: tl.constexpr,
    LB: tl.constexpr,
    LH: tl.constexpr,
    LT: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    TOP_LEFT_CAUSAL: tl.constexpr,
    VARLEN: tl.constexpr,
    PADDED_LSE: tl.constexpr,
    DROPOUT: tl.constexpr,
    WIDE_SEQUENCE_INDICES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HEAD_OFFSET: tl.constexpr = 0,
    COMPUTE_DELTA: tl.constexpr = False,
    ASCEND_LEGACY_COMPILER: tl.constexpr = False,
    FULL_PRECISION: tl.constexpr = False,
):
    block = tl.program_id(0)
    if WIDE_SEQUENCE_INDICES:
        block = block.to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    head_k = tl.program_id(2).to(tl.int64) - HEAD_OFFSET
    if VARLEN:
        q0 = tl.load(CQ + batch).to(tl.int64)
        k0 = tl.load(CK + batch).to(tl.int64)
        sq = tl.load(CQ + batch + 1).to(tl.int64) - q0
        sk = tl.load(CK + batch + 1).to(tl.int64) - k0
        lbase = q0 * LT
        if PADDED_LSE:
            lbase = batch * LB
    else:
        sq, sk = sq_max, sk_max
        if WIDE_SEQUENCE_INDICES:
            sq, sk = tl.cast(sq, tl.int64), tl.cast(sk, tl.int64)
        q0, k0 = batch * sq, batch * sk
        lbase = batch * LB
    m = tl.arange(0, BM).to(tl.int64)
    n = (block * BN + tl.arange(0, BN)).to(tl.int64)
    d = tl.arange(0, BD)
    ki = ((k0 + n[:, None]) * HK + head_k) * DIM + d[None, :]
    km = (n[:, None] < sk) & (d[None, :] < DIM)
    k = tl.load(K + ki, km, 0)
    v = tl.load(V + ki, km, 0)
    dk = tl.full((BN, BD), 0, tl.float32)
    dv = tl.full((BN, BD), 0, tl.float32)
    lo = 0
    hi = sq
    offset = 0 if TOP_LEFT_CAUSAL else sk - sq
    if not ASCEND_LEGACY_COMPILER:
        if WR >= 0:
            lo = tl.maximum(0, block * BN - offset - WR)
        if WL >= 0:
            hi = tl.minimum(sq, (block + 1) * BN - offset + WL)
    # Native stores each query head's contribution in the input dtype before
    # reducing the GQA group. Keep that rounding boundary, with a fixed-order
    # FP32 group reduction and no expanded KV workspace or atomics.
    for group in range(HQ // HK):
        head = head_k * (HQ // HK) + group
        dk_head = tl.full((BN, BD), 0, tl.float32)
        dv_head = tl.full((BN, BD), 0, tl.float32)
        for tile in range(lo // BM, tl.cdiv(tl.maximum(hi, 0), BM)):
            qm = tile * BM + m
            qi = ((q0 + qm[:, None]) * HQ + head) * DIM + d[None, :]
            mask = (qm[:, None] < sq) & (d[None, :] < DIM)
            q = tl.load(Q + qi, mask, 0)
            do = tl.load(DO + qi, mask, 0)
            if COMPUTE_DELTA:
                o = tl.load(O + qi, mask, 0).to(tl.float32)
                delta = (
                    _aten_flash_delta(
                        o, do, O, DO, ((q0 + qm) * HQ + head) * DIM, qm < sq, DIM, BM
                    )
                    * keep_prob
                )
            else:
                delta = tl.load(DELTA + head * total_q + q0 + qm, qm < sq, 0)
            lse = tl.load(LSE + lbase + head * LH + qm * LT, qm < sq, float("inf"))
            p = _aten_flash_probability(
                q,
                k,
                lse,
                qm,
                n,
                sq,
                sk,
                scale,
                scale_log2,
                LSE_SCALE,
                WL,
                WR,
                TOP_LEFT_CAUSAL,
            )
            dp = tl.dot(do, tl.trans(v))
            dp_valid = (qm[:, None] < sq) & (n[None, :] < sk)
            if ASCEND_LEGACY_COMPILER:
                dp_valid |= dp == 0.0
            dp = tl.where(dp_valid, dp, 0.0)
            p_drop = p
            if DROPOUT:
                keep = _aten_flash_dropout(RNG, batch, head, qm, n, HQ, threshold, 1.0)
                dp *= keep
                p_drop = p * keep
            ds = p * (dp - delta[:, None])
            ds_valid = (qm[:, None] < sq) & (n[None, :] < sk)
            if ASCEND_LEGACY_COMPILER:
                ds_valid |= ds == 0.0
            ds = tl.where(ds_valid, ds, 0.0)
            if FULL_PRECISION:
                dk_head += tl.dot(
                    tl.trans(ds), q.to(tl.float32), input_precision="ieee"
                )
                dv_head += tl.dot(
                    tl.trans(p_drop), do.to(tl.float32), input_precision="ieee"
                )
            else:
                dk_head += tl.dot(tl.trans(ds.to(q.dtype)), q)
                dv_head += tl.dot(tl.trans(p_drop.to(q.dtype)), do)
        if FULL_PRECISION:
            dk += dk_head * (scale * drop_scale)
            dv += dv_head * drop_scale
        else:
            dk += (dk_head * (scale * drop_scale)).to(k.dtype).to(tl.float32)
            dv += (dv_head * drop_scale).to(v.dtype).to(tl.float32)
    tl.store(DK + ki, tl.where(sq > 0, dk, 0.0), km)
    tl.store(DV + ki, tl.where(sq > 0, dv, 0.0), km)


@triton.jit
def _aten_flash_bwd_small(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    DQ,
    DK,
    DV,
    CQ,
    CK,
    RNG,
    threshold,
    drop_scale,
    keep_prob,
    sq_max,
    sk_max,
    total_q,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    DIM: tl.constexpr,
    BD: tl.constexpr,
    scale,
    scale_log2,
    LSE_SCALE: tl.constexpr,
    LB: tl.constexpr,
    LH: tl.constexpr,
    LT: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    TOP_LEFT_CAUSAL: tl.constexpr,
    VARLEN: tl.constexpr,
    PADDED_LSE: tl.constexpr,
    DROPOUT: tl.constexpr,
    WIDE_SEQUENCE_INDICES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    FULL_PRECISION: tl.constexpr = False,
):
    block = tl.program_id(0)
    if WIDE_SEQUENCE_INDICES:
        block = block.to(tl.int64)
    # Disjoint programs own Q or KV outputs. KV programs recompute their small
    # delta tiles, removing the inter-kernel dependency and delta allocation.
    if tl.program_id(2) < HQ:
        if block * BM < sq_max:
            _aten_flash_bwd_dq(
                Q,
                K,
                V,
                O,
                DO,
                LSE,
                Q,
                DQ,
                CQ,
                CK,
                RNG,
                threshold,
                drop_scale,
                keep_prob,
                sq_max,
                sk_max,
                total_q,
                HQ,
                HK,
                DIM,
                BD,
                scale,
                scale_log2,
                LSE_SCALE,
                LB,
                LH,
                LT,
                WL,
                WR,
                TOP_LEFT_CAUSAL,
                VARLEN,
                PADDED_LSE,
                DROPOUT,
                WIDE_SEQUENCE_INDICES,
                BM,
                BN,
                WRITE_DELTA=False,
                FULL_PRECISION=FULL_PRECISION,
            )
    else:
        if block * BN < sk_max:
            _aten_flash_bwd_dkv(
                Q,
                K,
                V,
                O,
                DO,
                LSE,
                Q,
                DK,
                DV,
                CQ,
                CK,
                RNG,
                threshold,
                drop_scale,
                keep_prob,
                sq_max,
                sk_max,
                total_q,
                HQ,
                HK,
                DIM,
                BD,
                scale,
                scale_log2,
                LSE_SCALE,
                LB,
                LH,
                LT,
                WL,
                WR,
                TOP_LEFT_CAUSAL,
                VARLEN,
                PADDED_LSE,
                DROPOUT,
                WIDE_SEQUENCE_INDICES,
                BM,
                BN,
                HEAD_OFFSET=HQ,
                COMPUTE_DELTA=True,
                FULL_PRECISION=FULL_PRECISION,
            )


def _aten_flash_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    rng_state,
    unused,
    *,
    scale=None,
    window_size_left=None,
    window_size_right=None,
    _tile_override=None,
    _kernel_override=None,
    _full_precision=False,
):
    # torch-musa 2.9.1 forwards the dense ATen call directly to MuDNN's
    # BHSD SDPA entry. Normalize that producer ABI before the shared kernel.
    musa_dense = device.vendor_name == "mthreads" and query.ndim == 4
    if musa_dense:
        query, key, value, out, grad_out = (
            t.transpose(1, 2) for t in (query, key, value, out, grad_out)
        )
    # Validate metadata without reading device values or synchronizing the stream.
    varlen = query.ndim == 3
    rank = 3 if varlen else 4
    if query.ndim not in (3, 4) or any(
        t.ndim != rank for t in (key, value, out, grad_out)
    ):
        raise ValueError("flash_attention_backward expects BSHD or packed THD tensors")
    if any(
        t.dtype != query.dtype for t in (key, value, out, grad_out)
    ) or query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "flash_attention_backward requires matching float16 or bfloat16 tensors"
        )
    if any(t.device != query.device for t in (key, value, out, grad_out, logsumexp)):
        raise ValueError("flash_attention_backward tensors must be on the same device")
    if (
        query.shape != out.shape
        or query.shape != grad_out.shape
        or key.shape != value.shape
    ):
        raise ValueError("incompatible query/output/gradient or key/value shapes")
    dim, hq, hk = query.shape[-1], query.shape[-2], key.shape[-2]
    if dim != key.shape[-1] or dim <= 0 or dim > 256 or dim % 8:
        raise ValueError("head dimension must match and be a multiple of 8 in [8, 256]")
    if hq <= 0 or hk <= 0 or hq % hk:
        raise ValueError("query heads must be a positive multiple of KV heads")
    if any(t.stride(-1) != 1 for t in (query, key, value, out)):
        raise ValueError(
            "query, key, value and out must have contiguous last dimensions"
        )
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    # Match native float32 probability conversion without reading device state.
    threshold, drop_scale, keep32 = 255, 1.0, 1.0
    if dropout_p:
        p32 = struct.unpack("f", struct.pack("f", dropout_p))[0]
        if p32 >= 1.0:
            raise ValueError("dropout_p must remain below 1 after float32 conversion")
        keep32 = struct.unpack("f", struct.pack("f", 1.0 - p32))[0]
        threshold = int(keep32 * 255.0)
        drop_scale = struct.unpack("f", struct.pack("f", 1.0 / keep32))[0]
        if device.vendor_name not in ("nvidia", "thead") or torch.version.cuda is None:
            raise NotImplementedError(
                "native dropout ABI is only implemented for NVIDIA CUDA and PPU"
            )
        schema = torch.ops.aten._flash_attention_backward.default._schema
        if schema.arguments[12].name != "rng_state":
            raise NotImplementedError(
                "legacy seed/offset dropout ABI is not implemented"
            )
        rng_dtype = torch.int64 if device.vendor_name == "thead" else torch.uint64
        if (
            rng_state is None
            or rng_state.dtype != rng_dtype
            or rng_state.device != query.device
            or rng_state.shape != (2,)
            or not rng_state.is_contiguous()
        ):
            raise ValueError(
                f"FlashAttention requires a contiguous device {rng_dtype}[2] saved rng_state"
            )
    wl = -1 if window_size_left is None else window_size_left
    wr = -1 if window_size_right is None else window_size_right
    wl, wr = max(-1, wl), max(-1, wr)
    if is_causal:
        wr = 0
    if max_q < 0 or max_k < 0:
        raise ValueError("maximum sequence lengths must be nonnegative")
    # torch-musa 2.9.1 uses padded BHS; CUDA uses packed HT for varlen LSE.
    # CoreX 4.4 native returns either rank, depending on its forward path.
    # Accept its two observed formats with strict shape checks below.
    padded_lse = device.vendor_name == "mthreads" or (
        device.vendor_name == "iluvatar" and logsumexp.ndim == 3
    )
    if varlen:
        if cum_seq_q is None or cum_seq_k is None:
            raise ValueError("packed tensors require both cumulative sequence arrays")
        for cu in (cum_seq_q, cum_seq_k):
            if (
                cu.dtype != torch.int32
                or cu.device != query.device
                or cu.ndim != 1
                or not cu.is_contiguous()
                or cu.numel() < 1
            ):
                raise ValueError("cu_seqlens must be contiguous device int32 vectors")
        if cum_seq_q.shape != cum_seq_k.shape:
            raise ValueError("cumulative sequence arrays must have equal lengths")
        batch = cum_seq_q.numel() - 1
        total_q = query.shape[0]
        expected_lse = (hq, total_q)
        lb, lh, lt = (
            0,
            logsumexp.stride(0) if logsumexp.ndim == 2 else 0,
            logsumexp.stride(1) if logsumexp.ndim == 2 else 0,
        )
        if padded_lse:
            expected_lse = (batch, hq, max_q)
            lb, lh, lt = logsumexp.stride() if logsumexp.ndim == 3 else (0, 0, 0)
        # Monotone offsets, endpoints and per-sequence maxima are caller preconditions,
        # as in native FlashAttention. Checking their values would synchronize here.
        if (max_q == 0 and total_q) or (max_k == 0 and key.shape[0]):
            raise ValueError("nonempty packed inputs need nonzero maximum lengths")
    else:
        if cum_seq_q is not None or cum_seq_k is not None:
            raise ValueError(
                "dense tensors require undefined cumulative sequence arrays"
            )
        batch, sq = query.shape[:2]
        if key.shape[0] != batch:
            raise ValueError("query and key batch sizes must match")
        max_q, max_k = sq, key.shape[1]
        total_q = batch * sq
        expected_lse = (batch, hq, sq)
        lb, lh, lt = logsumexp.stride() if logsumexp.ndim == 3 else (0, 0, 0)
    if logsumexp.dtype != torch.float32 or tuple(logsumexp.shape) != expected_lse:
        raise ValueError(f"logsumexp must be float32 with shape {expected_lse}")
    # FlashAttention canonicalizes windows using the public K length bound.
    # In particular, a wide right window can unmask early rows when Sq > Sk.
    if wl >= max_k:
        wl = -1
    if wr >= max_k:
        wr = -1
    scale = dim**-0.5 if scale is None else scale
    scale_log2 = None
    lse_scale = math.log2(math.e)
    if device.vendor_name == "iluvatar":
        # CoreX 4.4's head-dimension kernels save either -ln or -log2 LSE.
        # This mapping is validated for every multiple of 8 through D=256.
        lse_scale = -math.log2(math.e) if dim in (32, 96, 160, 192, 256) else -1.0
    launch_options = {"FULL_PRECISION": True} if _full_precision else {}
    cuda_flash_arithmetic = device.vendor_name in ("nvidia", "thead")
    if cuda_flash_arithmetic or device.vendor_name == "iluvatar":
        try:
            scale32 = struct.unpack("f", struct.pack("f", scale))[0]
            scale_log2 = struct.unpack(
                "f", struct.pack("f", scale32 * math.log2(math.e))
            )[0]
        except OverflowError:
            scale_log2 = math.copysign(math.inf, scale)
    # Nonpositive scales use mathematical gradients, including an unrounded
    # GQA head reduction, on both CUDA and PPU.
    if cuda_flash_arithmetic and scale <= 0:
        launch_options["FULL_PRECISION"] = True
    if cuda_flash_arithmetic:
        launch_options["enable_fp_fusion"] = False
    q, k, v, o, do = (t.contiguous() for t in (query, key, value, out, grad_out))
    dq, dk, dv = (torch.empty_like(t) for t in (q, k, v))
    if batch == 0:
        if q.numel() or k.numel():
            raise ValueError("zero sequences require empty packed inputs")
        return (
            tuple(t.transpose(1, 2) for t in (dq, dk, dv))
            if musa_dense
            else (dq, dk, dv)
        )
    # CANN 8.5.0 / FlagTree 0.6.0+ascend3.2 showed compiler failures
    # with geometric selects and window-dependent loop bounds. Keep element
    # masks, using the validated selects and full tile range on this compiler.
    ascend_legacy_compiler = (
        device.vendor_name == "ascend" and triton.__version__ == "3.2.0"
    )
    # Below 2**29, all tile/window expressions (at most three lengths plus
    # a tile <= 128) fit int32. Pointers still widen before head/D products.
    # Large length bounds and the legacy Ascend compiler use int64 throughout.
    wide_sequence_indices = ascend_legacy_compiler or max(max_q, max_k) >= 2**29
    common = (
        rng_state if dropout_p else q,
        threshold,
        drop_scale,
        keep32,
        max_q,
        max_k,
        total_q,
        hq,
        hk,
        dim,
        max(16, triton.next_power_of_2(dim)),
        scale,
        scale_log2,
        lse_scale,
        lb,
        lh,
        lt,
        wl,
        wr,
        device.vendor_name == "mthreads" and is_causal,
        varlen,
        varlen and padded_lse,
        dropout_p > 0,
        wide_sequence_indices,
    )
    ppu_small_math = (
        device.vendor_name == "thead"
        and not varlen
        and scale <= 0
        and dim <= 128
        and max(max_q, max_k) <= 128
    )
    tile = 16 if ppu_small_math else 32
    num_warps = 4
    # Head dimensions above 128 pad to 256 lanes; a 32-row tile exceeds Ascend UB.
    if device.vendor_name == "ascend" and dim > 128:
        tile = 16
    if (
        (
            (device.vendor_name == "nvidia" and dim <= 128)
            or (device.vendor_name == "iluvatar" and not varlen and dim in (64, 128))
        )
        and hq == hk
        and min(max_q, max_k) >= 256
    ):
        tile = 64
        if (
            device.vendor_name == "iluvatar"
            and dim == 64
            and min(max_q, max_k) >= 1024
            and not is_causal
            and wl < 0
            and wr < 0
        ):
            tile = 128
            num_warps = 8
    if _tile_override is not None:
        tile = _tile_override
    # NVIDIA GQA reuses the saved delta across KV tiles; recomputing it in
    # the fused entry is slower than the second launch for these native shapes.
    if (
        ppu_small_math
        or (device.vendor_name == "nvidia" and hq == hk and max(max_q, max_k) <= 256)
        or (device.vendor_name in ("iluvatar", "hygon") and max(max_q, max_k) <= 128)
        or (
            musa_dense
            and hq == hk
            and dim == 64
            and max_q == max_k
            and max_q <= 128
            and not is_causal
            and wl < 0
            and wr < 0
        )
    ):
        if max_q or max_k:
            _aten_flash_bwd_small[
                (triton.cdiv(max(max_q, max_k), tile), batch, hq + hk)
            ](
                q,
                k,
                v,
                o,
                do,
                logsumexp,
                dq,
                dk,
                dv,
                cum_seq_q,
                cum_seq_k,
                *common,
                BM=tile,
                BN=tile,
                num_warps=num_warps,
                num_stages=1,
                **launch_options,
            )
        return (
            tuple(t.transpose(1, 2) for t in (dq, dk, dv))
            if musa_dense
            else (dq, dk, dv)
        )
    dq_kernel, dkv_kernel = (
        (_aten_flash_bwd_dq, _aten_flash_bwd_dkv)
        if _kernel_override is None
        else _kernel_override
    )
    delta = torch.empty((hq, total_q), dtype=torch.float32, device=q.device)
    # Fixed-order, disjoint writes: deterministic in strict/warn/off modes.
    if max_q:
        dq_kernel[(triton.cdiv(max_q, tile), batch, hq)](
            q,
            k,
            v,
            o,
            do,
            logsumexp,
            delta,
            dq,
            cum_seq_q,
            cum_seq_k,
            *common,
            BM=tile,
            BN=tile,
            num_warps=num_warps,
            num_stages=1,
            ASCEND_LEGACY_COMPILER=ascend_legacy_compiler,
            **launch_options,
        )
    if max_k:
        dkv_kernel[(triton.cdiv(max_k, tile), batch, hk)](
            q,
            k,
            v,
            o,
            do,
            logsumexp,
            delta,
            dk,
            dv,
            cum_seq_q,
            cum_seq_k,
            *common,
            BM=tile,
            BN=tile,
            num_warps=num_warps,
            num_stages=1,
            ASCEND_LEGACY_COMPILER=ascend_legacy_compiler,
            **launch_options,
        )
    return (
        tuple(t.transpose(1, 2) for t in (dq, dk, dv)) if musa_dense else (dq, dk, dv)
    )


def flash_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    rng_state,
    unused,
    *,
    scale=None,
    window_size_left=None,
    window_size_right=None,
):
    logger.debug("GEMS FLASH_ATTENTION_BACKWARD")
    # AOTriton has a distinct RNG ABI and uses the original Triton path.
    if device.vendor_name == "amd":
        is_dropout = dropout_p > 0.0
        rng_tuple = _parse_rng_state(rng_state, unused) if is_dropout else None
        wl = -1 if window_size_left is None else int(window_size_left)
        wr = -1 if window_size_right is None else int(window_size_right)
        dQ, dK, dV, _ = flash_attn_backward(
            grad_out,
            query,
            key,
            value,
            out,
            logsumexp,
            cu_seq_q=cum_seq_q,
            cu_seq_k=cum_seq_k,
            max_seqlen_q=int(max_q),
            max_seqlen_k=int(max_k),
            is_dropout=is_dropout,
            dropout_p=dropout_p,
            rng_state=rng_tuple,
            is_causal=is_causal,
            window_size_left=wl,
            window_size_right=wr,
            softmax_scale=scale,
        )
        return dQ, dK, dV
    return _aten_flash_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        rng_state,
        unused,
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )


def scaled_dot_product_flash_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    *,
    scale=None,
):
    logger.debug("GEMS SCALED_DOT_PRODUCT_FLASH_ATTENTION_BACKWARD")
    is_dropout = dropout_p > 0.0
    rng_tuple = _parse_philox(philox_seed, philox_offset) if is_dropout else None
    use_varlen = (cum_seq_q is not None) and (cum_seq_k is not None)
    # The opaque ATen entry and the flash backward kernels share the
    # (B, S, H, D) layout; tensors pass straight through. The log-sum-exp
    # arrives from the ATen forward in (B, H, S) already, so no transpose is
    # needed either — only a varlen caller's cumulative-sequence path differs.
    if use_varlen:
        return flash_attn_backward(
            grad_out,
            query,
            key,
            value,
            out,
            logsumexp,
            cu_seq_q=cum_seq_q,
            cu_seq_k=cum_seq_k,
            max_seqlen_q=int(max_q),
            max_seqlen_k=int(max_k),
            is_dropout=is_dropout,
            dropout_p=dropout_p,
            rng_state=rng_tuple,
            is_causal=is_causal,
            softmax_scale=scale,
        )[:3]
    dQ, dK, dV, _ = flash_attn_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        max_seqlen_q=int(max_q),
        max_seqlen_k=int(max_k),
        is_dropout=is_dropout,
        dropout_p=dropout_p,
        rng_state=rng_tuple,
        is_causal=is_causal,
        softmax_scale=scale,
    )
    return dQ, dK, dV


def scaled_dot_product_cudnn_attention_backward(
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
    bias_requires_grad=False,
):
    logger.debug("GEMS SCALED_DOT_PRODUCT_CUDNN_ATTENTION_BACKWARD")
    grad_out_bshd = grad_out.permute(0, 2, 1, 3).contiguous()
    query_bshd = query.permute(0, 2, 1, 3).contiguous()
    key_bshd = key.permute(0, 2, 1, 3).contiguous()
    value_bshd = value.permute(0, 2, 1, 3).contiguous()
    out_bshd = out.permute(0, 2, 1, 3).contiguous()
    lse = logsumexp.float()

    is_dropout = dropout_p > 0.0
    rng_tuple = _parse_philox(philox_seed, philox_offset) if is_dropout else None
    use_varlen = (cum_seq_q is not None) and (cum_seq_k is not None)

    dQ_bshd, dK_bshd, dV_bshd, dBias = flash_attn_backward(
        grad_out_bshd,
        query_bshd,
        key_bshd,
        value_bshd,
        out_bshd,
        lse,
        cu_seq_q=cum_seq_q if use_varlen else None,
        cu_seq_k=cum_seq_k if use_varlen else None,
        max_seqlen_q=int(max_q),
        max_seqlen_k=int(max_k),
        is_dropout=is_dropout,
        dropout_p=dropout_p,
        rng_state=rng_tuple,
        is_causal=is_causal,
        softmax_scale=scale,
        attn_bias=attn_bias,
        bias_requires_grad=bias_requires_grad,
        compensate_dv=device.vendor_name
        in ("mthreads", "metax", "hygon", "ascend", "thead"),
    )

    dQ = dQ_bshd.permute(0, 2, 1, 3).contiguous()
    dK = dK_bshd.permute(0, 2, 1, 3).contiguous()
    dV = dV_bshd.permute(0, 2, 1, 3).contiguous()
    return dQ, dK, dV


def efficient_attention_backward(
    grad_out_,
    query,
    key,
    value,
    bias,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    logsumexp,
    dropout_p,
    philox_seed,
    philox_offset,
    custom_mask_type,
    bias_requires_grad,
    *,
    scale=None,
    num_splits_key=None,
    window_size=None,
    shared_storage_dqdkdv=False,
):
    logger.debug("GEMS EFFICIENT_ATTENTION_BACKWARD")
    if custom_mask_type == 0:
        is_causal, wl, wr = False, -1, -1
    elif custom_mask_type == 1:
        is_causal, wl, wr = True, -1, -1
    elif custom_mask_type == 2:
        is_causal = False
        wl = 0
        wr = max_seqlen_q + max_seqlen_k + 1
    else:
        raise ValueError(f"未知 custom_mask_type: {custom_mask_type}")

    if window_size is not None and window_size >= 0:
        wl = wr = window_size

    is_dropout = dropout_p > 0.0
    rng_tuple = _parse_philox(philox_seed, philox_offset) if is_dropout else None
    use_varlen = (cu_seqlens_q is not None) and (cu_seqlens_k is not None)
    return flash_attn_backward(
        grad_out_,
        query,
        key,
        value,
        out,
        logsumexp,
        cu_seq_q=cu_seqlens_q if use_varlen else None,
        cu_seq_k=cu_seqlens_k if use_varlen else None,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        is_dropout=is_dropout,
        dropout_p=dropout_p,
        rng_state=rng_tuple,
        is_causal=is_causal,
        window_size_left=wl,
        window_size_right=wr,
        softmax_scale=scale,
        attn_bias=bias,
        bias_requires_grad=bias_requires_grad,
    )


def scaled_dot_product_efficient_attention_backward(
    grad_out_,
    query,
    key,
    value,
    attn_bias,
    out,
    logsumexp,
    philox_seed,
    philox_offset,
    dropout_p,
    grad_input_mask,
    is_causal=False,
    *,
    scale=None,
):
    logger.debug("GEMS SCALED_DOT_PRODUCT_EFFICIENT_ATTENTION_BACKWARD")
    need_dq, need_dk, need_dv, need_dbias = grad_input_mask

    grad_out_bshd = grad_out_.permute(0, 2, 1, 3).contiguous()
    query_bshd = query.permute(0, 2, 1, 3).contiguous()
    key_bshd = key.permute(0, 2, 1, 3).contiguous()
    value_bshd = value.permute(0, 2, 1, 3).contiguous()
    out_bshd = out.permute(0, 2, 1, 3).contiguous()

    is_dropout = dropout_p > 0.0
    rng_tuple = _parse_philox(philox_seed, philox_offset) if is_dropout else None
    bias_req_grad = need_dbias and (attn_bias is not None)

    dQ_bshd, dK_bshd, dV_bshd, dBias = flash_attn_backward(
        grad_out_bshd,
        query_bshd,
        key_bshd,
        value_bshd,
        out_bshd,
        logsumexp,
        cu_seq_q=None,
        cu_seq_k=None,
        max_seqlen_q=0,
        max_seqlen_k=0,
        is_dropout=is_dropout,
        dropout_p=dropout_p,
        rng_state=rng_tuple,
        is_causal=is_causal,
        softmax_scale=scale,
        attn_bias=attn_bias,
        bias_requires_grad=bias_req_grad,
        compensate_dv=need_dv
        and device.vendor_name in ("mthreads", "metax", "hygon", "ascend"),
    )

    dQ_bhsd = dQ_bshd.permute(0, 2, 1, 3).contiguous()
    dK_bhsd = dK_bshd.permute(0, 2, 1, 3).contiguous()
    dV_bhsd = dV_bshd.permute(0, 2, 1, 3).contiguous()

    if not need_dq:
        dQ_bhsd = torch.zeros_like(query)
    if not need_dk:
        dK_bhsd = torch.zeros_like(key)
    if not need_dv:
        dV_bhsd = torch.zeros_like(value)
    if not need_dbias:
        dBias = None

    return dQ_bhsd, dK_bhsd, dV_bhsd, dBias
