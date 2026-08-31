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

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)
_debug = False


@triton.jit
def u64_to_lohi(x):
    return (x >> 32).to(tl.uint32), (x & 0xFFFFFFFF).to(tl.uint32)


@triton.jit
def u64_from_lohi(lo, hi):
    return hi.to(tl.uint64) << 32 + lo.to(tl.uint64)


@triton.jit
def philox_(seed, subsequence, offset):
    kPhilox10A: tl.constexpr = 0x9E3779B9
    kPhilox10B: tl.constexpr = 0xBB67AE85
    k0, k1 = u64_to_lohi(seed.to(tl.uint64))
    c0, c1 = u64_to_lohi(offset.to(tl.uint64))
    c2, c3 = u64_to_lohi(subsequence.to(tl.uint64))

    # pragma unroll
    kPhiloxSA: tl.constexpr = 0xD2511F53
    kPhiloxSB: tl.constexpr = 0xCD9E8D57
    for _ in tl.static_range(6):
        res0 = kPhiloxSA * c0.to(tl.uint64)
        res1 = kPhiloxSB * c2.to(tl.uint64)
        res0_x, res0_y = u64_to_lohi(res0)
        res1_x, res1_y = u64_to_lohi(res1)
        c0, c1, c2, c3 = res1_y ^ c1 ^ k0, res1_x, res0_y ^ c3 ^ k1, res0_x
        k0 += kPhilox10A
        k1 += kPhilox10B

    res0 = kPhiloxSA * c0.to(tl.uint64)
    res1 = kPhiloxSB * c2.to(tl.uint64)
    res0_x, res0_y = u64_to_lohi(res0)
    res1_x, res1_y = u64_to_lohi(res1)
    c0, c1, c2, c3 = res1_y ^ c1 ^ k0, res1_x, res0_y ^ c3 ^ k1, res0_x

    return c0, c1, c2, c3


@triton.jit
def apply_dropout_mask(
    P,
    mask,
    encode_dropout_in_sign_bit: tl.constexpr,
):
    if encode_dropout_in_sign_bit:
        P = tl.where(mask, -P, P)
    else:
        P = tl.where(mask, (P * 0).to(P.dtype), P)
    return P


@triton.jit
def apply_dropout(
    P,
    row_start,
    col_start,
    n_cols,
    bid,
    hid,
    philox_seed,
    philox_offset,
    p_dropout_uint8: tl.constexpr,
    is_dropout: tl.constexpr,
    encode_dropout_in_sign_bit: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if is_dropout:
        row_start = tl.multiple_of(row_start, BLOCK_M)
        col_start = tl.multiple_of(col_start, BLOCK_N)
        row = row_start + tl.arange(0, BLOCK_M)[:, None]
        # Down scale col_idx by 4
        col = col_start // 4 + tl.arange(0, BLOCK_N // 4)[None, :]

        subsequence = row.to(tl.uint64) * n_cols + col.to(tl.uint64)

        offset = philox_offset + bid * NUM_HEADS + hid
        offset += subsequence * 0
        r0, r1, r2, r3 = philox_(philox_seed, subsequence, offset)

        r = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(BLOCK_M, BLOCK_N)

        mask = (r & 0xFF) >= p_dropout_uint8

        P = apply_dropout_mask(
            P, mask, encode_dropout_in_sign_bit=encode_dropout_in_sign_bit
        )
    return P


@triton.jit
def apply_alibi(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    is_causal: tl.constexpr,
    is_alibi: tl.constexpr,
    alibi_slope: tl.constexpr = None,
):
    if is_alibi:
        if is_causal:
            # The row independent alibi bias renders the same attention output
            # as with the standard alibi because softmax is shift invariant, i.e.,
            # softmax(A + bias + const) = softamx(A + bias). The following two
            # biases are no different if causal is true.
            # bias_1 = [
            #   -4, -3, -2,  X, X,
            #   -4, -3, -2, -1, X,
            #   -4, -3, -2, -1, 0,
            # ]
            # bias_2 = [
            #   -2, -1, 0,  X,  X,
            #   -3, -2, -1, 0,  X,
            #   -4, -3, -2, -1, 0,
            # ]
            bias = alibi_slope * (-max_seqlen_k + 1 + col_idx[None, :]).to(tl.float32)
            S += bias
        else:
            bias = -alibi_slope * tl.abs(
                col_idx[None, :] - max_seqlen_k + max_seqlen_q - row_idx[:, None]
            ).to(tl.float32)
            S += bias

    return S


@triton.jit
def apply_mask(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    window_size_left,
    window_size_right,
    is_even_mn: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
):
    need_mask = is_causal | is_local | (not is_even_mn)
    # need_mask: tl.constexpr = is_causal | is_local
    if need_mask:
        # Extra care should be taken to void one-off errors: both col_lb and col_rb are inclusive!
        col_lb = max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left)
        col_rb = min(
            max_seqlen_k - 1, row_idx + max_seqlen_k - max_seqlen_q + window_size_right
        )

        if is_causal:
            S = tl.where(col_idx[None, :] > col_rb[:, None], float("-inf"), S)

        if is_local:
            S = tl.where(
                (col_idx[None, :] > col_rb[:, None])
                | (col_idx[None, :] < col_lb[:, None]),
                float("-inf"),
                S,
            )

        if (not is_local) & (not is_causal) & (not is_even_mn):
            S = tl.where(col_idx[None, :] >= max_seqlen_k, float("-inf"), S)

    return S


@triton.jit
def softmax_rescale(
    O_acc,
    S,
    row_max,
    row_sum,
    softmax_scale_log2e: tl.constexpr,
    is_border: tl.constexpr,
    # is_init: tl.constexpr
):
    prev_max = row_max
    row_max = tl.maximum(row_max, tl.max(S, 1))

    if is_border:
        cur_max = tl.where(row_max == float("-inf"), 0, row_max)
    else:
        cur_max = row_max

    p_scale = tl.math.exp2((prev_max - cur_max) * softmax_scale_log2e)
    row_sum *= p_scale
    O_acc *= p_scale[:, None]

    max_scaled = tl.where(row_max == float("-inf"), 0, row_max * softmax_scale_log2e)
    P = tl.math.exp2(S * softmax_scale_log2e - max_scaled[:, None])
    row_sum = row_sum + tl.sum(P, 1)
    return O_acc, P, row_max, row_sum


@triton.jit
def fa3_load_dense_descales(
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_batch_stride,
    q_descale_head_stride,
    q_descale_block_stride,
    k_descale_batch_stride,
    k_descale_head_stride,
    k_descale_block_stride,
    v_descale_batch_stride,
    v_descale_head_stride,
    v_descale_block_stride,
    bid,
    hid,
    kv_hid,
    row_start,
    col_start,
):
    # W8A8 FA3 change: FA3-style block quantization stores one descale per
    # logical 128-token Q/K/V block.  These scales correct FP8 QK and FP8 PV.
    q_block = row_start // 128
    kv_block = col_start // 128
    q_descale = tl.load(
        q_descale_ptr
        + bid * q_descale_batch_stride
        + hid * q_descale_head_stride
        + q_block * q_descale_block_stride
    ).to(tl.float32)
    k_descale = tl.load(
        k_descale_ptr
        + bid * k_descale_batch_stride
        + kv_hid * k_descale_head_stride
        + kv_block * k_descale_block_stride
    ).to(tl.float32)
    v_descale = tl.load(
        v_descale_ptr
        + bid * v_descale_batch_stride
        + kv_hid * v_descale_head_stride
        + kv_block * v_descale_block_stride
    ).to(tl.float32)
    return q_descale, k_descale, v_descale


@triton.jit
def fa3_fp8_pv_dot(
    P, V, acc, v_descale, fp8_p_max: tl.constexpr, fp8_dtype: tl.constexpr
):
    p_descale = 1.0 / fp8_p_max
    P_fp8 = (P * fp8_p_max).to(fp8_dtype)
    pv = tl.dot(P_fp8, V, out_dtype=tl.float32)
    return acc + pv * (p_descale * v_descale)


@triton.jit
def apply_softcap(S, softcap, is_softcap: tl.constexpr):
    if is_softcap:
        S = tl_extra_shim.tanh(S * softcap)

    return S


def block_m_splitkv_heuristic(headdim):
    return 128 if headdim <= 128 else 64


def block_n_splitkv_heuristic(headdim):
    return 64 if headdim <= 64 else 32


def is_even_mn(M, N, BM, BN, WL, WR):
    if M % BM == 0 and N % BN == 0:
        if M % N == 0 or N % M == 0:
            if (WL == -1 or WL % BN == 0) and (WR == -1 or WR % BN == 0):
                return True
    return False


def block_m_splitkv_heuristic_spec_args(args):
    return 128 if args["d"] <= 128 else 64


def block_n_splitkv_heuristic_spec_args(args):
    return 64 if args["d"] <= 64 else 32


def is_even_mn_spec_args(args):
    if (
        args["seqlen_q"] % args["BLOCK_M"] == 0
        and args["seqlen_k"] % args["BLOCK_N"] == 0
    ):
        if (
            args["seqlen_q"] % args["seqlen_k"] == 0
            or args["seqlen_k"] % args["seqlen_q"] == 0
        ):
            if (
                args["window_size_left"] == -1
                or args["window_size_left"] % args["BLOCK_N"] == 0
            ) and (
                args["window_size_right"] == -1
                or args["window_size_right"] % args["BLOCK_N"] == 0
            ):
                return True
    return False


def fwd_configs_w8a8():
    # W8A8 change: use a dedicated config space for FP8 QK.  The original FA2
    # configs were picked for fp16/bf16 QK, while FP8 often benefits from trying
    # larger N tiles and more shape-specific choices.  num_stages is deliberately
    # left as a tunable dimension instead of forcing one pipeline depth globally.
    return [
        # D64 candidates: small shapes often prefer smaller M, while long-S
        # QK-heavy cases can benefit from larger N.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        # D128 candidates: keep the original-style narrow-N option, but also
        # test wider N because QK is FP8 and may tolerate a different balance.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, num_stages=2),
    ]


def prune_fwd_configs_w8a8(configs, nargs, **kwargs):
    d = nargs["d"]
    is_dropout = nargs["is_dropout"]
    out = []
    for cfg in configs:
        BM = cfg.kwargs["BLOCK_M"]
        BN = cfg.kwargs["BLOCK_N"]
        w = cfg.num_warps
        s = cfg.num_stages

        if is_dropout and (w != 4 or s > 3):
            continue

        # W8A8 change: keep D64 and D128 candidate sets separate to avoid trying
        # configs that are known to have poor register/occupancy balance.
        if d <= 64:
            if (BM, BN, w, s) in {
                (128, 32, 4, 2),
                (128, 128, 8, 2),
                (128, 64, 4, 2),
                (128, 64, 4, 3),
                (128, 128, 4, 2),
                (128, 128, 4, 3),
            }:
                out.append(cfg)
        else:
            if (BM, BN, w, s) in {
                (128, 32, 8, 2),
                (128, 32, 8, 3),
                (128, 64, 8, 2),
                (128, 64, 8, 3),
                (128, 128, 4, 2),
                (128, 128, 8, 2),
                (64, 128, 8, 3),
            }:
                out.append(cfg)

    return out


def flash_fwd_kernel_heur_block_k(args):
    return triton.next_power_of_2(args["d"])


@libentry()
@triton.autotune(
    configs=fwd_configs_w8a8(),
    prune_configs_by={"early_config_prune": prune_fwd_configs_w8a8},
    key=["b", "h", "seqlen_q", "seqlen_k", "d", "is_dropout"],
)
@triton.heuristics(
    values={
        "BLOCK_K": flash_fwd_kernel_heur_block_k,
        "PRE_LOAD_V": lambda args: False,
        "IS_EVEN_MN": lambda args: is_even_mn(
            args["seqlen_q"],
            args["seqlen_k"],
            args["BLOCK_M"],
            args["BLOCK_N"],
            args["window_size_left"],
            args["window_size_right"],
        ),
    }
)
@triton.jit(
    do_not_specialize=["seqlen_q", "seqlen_k", "seqlen_q_rounded", "seqlen_k_rounded"]
)
def flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    is_cu_seqlens_q,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k,
    cu_seqlens_k_ptr,
    is_seqused_k,
    seqused_k_ptr,
    # sizes
    b: tl.constexpr,
    bk: tl.constexpr,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    # scaling factors
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_batch_stride,
    q_descale_head_stride,
    q_descale_block_stride,
    k_descale_batch_stride,
    k_descale_head_stride,
    k_descale_block_stride,
    v_descale_batch_stride,
    v_descale_head_stride,
    v_descale_block_stride,
    use_fa3_fp8_scales: tl.constexpr,
    fp8_p_max: tl.constexpr,
    # dropout
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # causal and swa
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    # alibi
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # block table
    total_q: tl.constexpr,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    # kernel params
    IS_EVEN_MN: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLIT_D: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    m_block = tl.program_id(0)
    bh = tl.program_id(1)
    # For D128, split the output dimension across two D64 CTAs. D64 or
    # non-split-D uses a single CTA.
    d_split = tl.program_id(2)
    d_start = d_split * BLOCK_D
    hid = bh % h
    bid = bh // h
    num_m_blocks = tl.cdiv(seqlen_q, BLOCK_M)

    # We draw a minimum covering frame on the attention map that this CTA is assigned to process.
    # The frame edges are rounded to multiples of BLOCK_M and BLOCK_N for rows and columns respectively.

    col_min = 0
    if is_local:
        col_min = max(0, m_block * BLOCK_M + seqlen_k - seqlen_q - window_size_left)
        if not IS_EVEN_MN:
            # round left
            col_min = (col_min // BLOCK_N) * BLOCK_N

    col_max = seqlen_k
    if is_causal or is_local:
        col_max += (m_block - num_m_blocks + 1) * BLOCK_M
        if is_local:
            col_max += window_size_right
        col_max = min(seqlen_k, col_max)

    if not IS_EVEN_MN:
        # round right
        col_max = tl.cdiv(col_max, BLOCK_N) * BLOCK_N

    if (not is_causal) and (not is_local):
        if IS_EVEN_MN:
            masking_cols: tl.constexpr = 0
        else:
            masking_cols: tl.constexpr = BLOCK_N
    elif (
        is_causal | is_local
    ) and IS_EVEN_MN:  # causal implies window_size_right is zero
        masking_cols: tl.constexpr = tl.cdiv(BLOCK_M, BLOCK_N) * BLOCK_N
    else:
        # local
        masking_cols: tl.constexpr = (tl.cdiv(BLOCK_M, BLOCK_N) + 1) * BLOCK_N

    if is_dropout:
        philox_seed = tl.load(philox_args).to(tl.uint64)
        philox_offset = tl.load(philox_args + 1).to(tl.uint64)

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0.0

    q_batch_stride = tl.multiple_of(q_batch_stride, d * h)
    q_ptr += bid * q_batch_stride + hid * q_head_stride
    row_start = m_block * BLOCK_M
    row_idx = row_start + tl.arange(0, BLOCK_M)
    q_off = row_idx[:, None] * q_row_stride + tl.arange(0, BLOCK_K)[None, :]
    dmask = tl.arange(0, BLOCK_K) < d
    qmask = dmask[None, :] & (row_idx[:, None] < seqlen_q)
    if IS_EVEN_MN & d == BLOCK_K:
        Q = tl.load(q_ptr + q_off, cache_modifier=".cg")
    else:
        Q = tl.load(q_ptr + q_off, mask=qmask, cache_modifier=".cg")

    if return_softmax:
        p_ptr += (
            (bid * h + hid) * seqlen_q_rounded + m_block * BLOCK_M
        ) * seqlen_k_rounded
        p_offset = tl.arange(0, BLOCK_M)[:, None] * seqlen_k_rounded + tl.arange(
            0, BLOCK_N
        )
        p_bp0 = p_ptr + p_offset

    # Use a [BLOCK_M, BLOCK_D] PV accumulator. For D128, BLOCK_D is 64,
    # so one CTA does not hold a [BM, 128] accumulator, reducing register pressure.
    acc_ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    k_batch_stride = tl.multiple_of(k_batch_stride, d * hk)
    h_hk_ratio = h // hk
    kv_hid = hid // h_hk_ratio
    k_ptr += bid * k_batch_stride
    k_ptr += kv_hid * k_head_stride
    v_ptr += bid * k_batch_stride
    v_ptr += kv_hid * k_head_stride

    k_offset = (
        tl.arange(0, BLOCK_N)[None, :] * k_row_stride + tl.arange(0, BLOCK_K)[:, None]
    )
    # Load only the 64-wide V slice for the current D split; QK still uses
    # the full BLOCK_K of 128.
    v_d = d_start + tl.arange(0, BLOCK_D)
    v_dmask = v_d < d
    v_offset = tl.arange(0, BLOCK_N)[:, None] * k_row_stride + v_d[None, :]

    p_bk0 = k_ptr + k_offset
    p_bv0 = v_ptr + v_offset

    if is_causal | is_local | (not IS_EVEN_MN):
        # Cut short masking cols if there's not enough cols out there
        masking_cols = min(col_max - col_min, masking_cols)
        for col_shift in tl.range(0, masking_cols, step=BLOCK_N):
            col_start = col_max - col_shift - BLOCK_N
            col_start = tl.multiple_of(col_start, BLOCK_N)
            off = col_start * k_row_stride
            if IS_EVEN_MN & d == BLOCK_K:
                K = tl.load(p_bk0 + off, cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_bv0 + off, cache_modifier=".cg")
            elif d == BLOCK_K:
                col_idx = col_start + tl.arange(0, BLOCK_N)
                kvmask = col_idx < seqlen_k
                K = tl.load(p_bk0 + off, mask=kvmask[None, :], cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_bv0 + off, mask=kvmask[:, None], cache_modifier=".cg")
            else:
                col_idx = col_start + tl.arange(0, BLOCK_N)
                kvmask = col_idx < seqlen_k
                K = tl.load(
                    p_bk0 + off,
                    mask=kvmask[None, :] & dmask[:, None],
                    cache_modifier=".cg",
                )
                if PRE_LOAD_V:
                    V = tl.load(
                        p_bv0 + off,
                        mask=kvmask[:, None] & v_dmask[None, :],
                        cache_modifier=".cg",
                    )
            q_descale, k_descale, v_descale = fa3_load_dense_descales(
                q_descale_ptr,
                k_descale_ptr,
                v_descale_ptr,
                q_descale_batch_stride,
                q_descale_head_stride,
                q_descale_block_stride,
                k_descale_batch_stride,
                k_descale_head_stride,
                k_descale_block_stride,
                v_descale_batch_stride,
                v_descale_head_stride,
                v_descale_block_stride,
                bid,
                hid,
                kv_hid,
                row_start,
                col_start,
            )
            # W8A8 FA3 change: QK is FP8*FP8 tensor-core compute; the per-block
            # Q/K descales restore the original score magnitude before softmax.
            S = tl.dot(Q, K, out_dtype=tl.float32, allow_tf32=False)
            S *= q_descale * k_descale
            S = apply_softcap(S, softcap, is_softcap)
            col_idx = col_start + tl.arange(0, BLOCK_N)
            row_idx = row_start + tl.arange(0, BLOCK_M)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            # tl.store(p_bp0 + col_start, S)
            S = apply_mask(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                is_even_mn=IS_EVEN_MN,
                is_causal=is_causal,
                is_local=is_local,
            )

            acc_, P, rowmax_, rowsum_ = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=(is_causal or is_local),
            )
            # W8A8 FA3 change: keep P in fp32 until after dropout/masking, then
            # dynamically quantize it for the FP8 PV tensor-core matmul.

            if is_dropout:
                if return_softmax:
                    P_drop = P

                    P_drop = apply_dropout(
                        P_drop,
                        row_start,
                        col_start,
                        seqlen_k,
                        bid,
                        hid,
                        philox_seed,
                        philox_offset,
                        p_dropout_in_uint8_t,
                        is_dropout,
                        encode_dropout_in_sign_bit=True,
                        NUM_HEADS=h,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                    )
                    if IS_EVEN_MN:
                        # Both split-D CTAs produce the same softmax tile, so only
                        # D split 0 writes the debug P tensor.
                        tl.store(p_bp0 + col_start, P_drop, mask=d_split == 0)
                    else:
                        kvmask = col_idx < seqlen_k
                        # Both split-D CTAs produce the same softmax tile, so only
                        # D split 0 writes the debug P tensor.
                        tl.store(
                            p_bp0 + col_start,
                            P_drop,
                            mask=qmask & kvmask[None, :] & (d_split == 0),
                        )

                P = apply_dropout(
                    P,
                    row_start,
                    col_start,
                    seqlen_k,
                    bid,
                    hid,
                    philox_seed,
                    philox_offset,
                    p_dropout_in_uint8_t,
                    is_dropout,
                    encode_dropout_in_sign_bit=False,
                    NUM_HEADS=h,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )

            if not PRE_LOAD_V:
                off = col_start * k_row_stride
                if IS_EVEN_MN & d == BLOCK_K:
                    V = tl.load(p_bv0 + off, cache_modifier=".cg")
                elif d == BLOCK_K:
                    kvmask = col_idx < seqlen_k
                    V = tl.load(p_bv0 + off, mask=kvmask[:, None], cache_modifier=".cg")
                else:
                    kvmask = col_idx < seqlen_k
                    V = tl.load(
                        p_bv0 + off,
                        mask=kvmask[:, None] & v_dmask[None, :],
                        cache_modifier=".cg",
                    )
            acc_ = fa3_fp8_pv_dot(
                P, V, acc_, v_descale, fp8_p_max, v_ptr.type.element_ty
            )

    for col_start in tl.range(
        col_min, col_max - masking_cols, step=BLOCK_N, num_stages=num_stages
    ):
        col_start = tl.multiple_of(col_start, BLOCK_N)
        off = col_start * k_row_stride
        if d == BLOCK_K:
            K = tl.load(p_bk0 + off, cache_modifier=".cg")
            if PRE_LOAD_V:
                V = tl.load(p_bv0 + off, cache_modifier=".cg")
        else:
            K = tl.load(p_bk0 + off, mask=dmask[:, None], cache_modifier=".cg")
            if PRE_LOAD_V:
                V = tl.load(p_bv0 + off, mask=v_dmask[None, :], cache_modifier=".cg")

        q_descale, k_descale, v_descale = fa3_load_dense_descales(
            q_descale_ptr,
            k_descale_ptr,
            v_descale_ptr,
            q_descale_batch_stride,
            q_descale_head_stride,
            q_descale_block_stride,
            k_descale_batch_stride,
            k_descale_head_stride,
            k_descale_block_stride,
            v_descale_batch_stride,
            v_descale_head_stride,
            v_descale_block_stride,
            bid,
            hid,
            kv_hid,
            row_start,
            col_start,
        )
        # W8A8 FA3 change: use FP8*FP8 tensor-core path for QK and apply the
        # block descales before online softmax.
        S = tl.dot(Q, K, out_dtype=tl.float32)
        S *= q_descale * k_descale
        S = apply_softcap(S, softcap, is_softcap)
        col_idx = col_start + tl.arange(0, BLOCK_N)
        row_idx = row_start + tl.arange(0, BLOCK_M)
        S = apply_alibi(
            S,
            col_idx,
            row_idx,
            seqlen_q,
            seqlen_k,
            is_causal=is_causal,
            is_alibi=is_alibi,
            alibi_slope=alibi_slope,
        )
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
            is_even_mn=True,
            is_causal=False,
            is_local=is_local,
        )

        acc_, P, rowmax_, rowsum_ = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=is_local,
        )
        # W8A8 FA3 change: P is quantized at the PV call site so dropout sees
        # the high-precision softmax probabilities.

        if is_dropout:
            if return_softmax:
                P_drop = P
                P_drop = apply_dropout(
                    P_drop,
                    row_start,
                    col_start,
                    seqlen_k,
                    bid,
                    hid,
                    philox_seed,
                    philox_offset,
                    p_dropout_in_uint8_t,
                    is_dropout,
                    encode_dropout_in_sign_bit=True,
                    NUM_HEADS=h,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )
                if IS_EVEN_MN:
                    # Both split-D CTAs produce the same softmax tile, so only
                    # D split 0 writes the debug P tensor.
                    tl.store(p_bp0 + col_start, P_drop, mask=d_split == 0)
                else:
                    kvmask = col_idx < seqlen_k
                    # Both split-D CTAs produce the same softmax tile, so only
                    # D split 0 writes the debug P tensor.
                    tl.store(
                        p_bp0 + col_start,
                        P_drop,
                        mask=qmask & kvmask[None, :] & (d_split == 0),
                    )

            P = apply_dropout(
                P,
                row_start,
                col_start,
                seqlen_k,
                bid,
                hid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                is_dropout,
                encode_dropout_in_sign_bit=False,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        if not PRE_LOAD_V:
            off = col_start * k_row_stride
            if d == BLOCK_K:
                V = tl.load(p_bv0 + off, cache_modifier=".cg")
            else:
                V = tl.load(p_bv0 + off, mask=v_dmask[None, :], cache_modifier=".cg")
        acc_ = fa3_fp8_pv_dot(P, V, acc_, v_descale, fp8_p_max, v_ptr.type.element_ty)

    # LSE
    # Note, rowsum = exp(-rowmax) * exp(lse), therefore rowmax + log(rowsum) cancels
    # the effect of rowmax and outputs lse only.
    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

    if is_dropout:
        acc_ *= inv_sum[:, None] * rp_dropout
    else:
        acc_ *= inv_sum[:, None]

    out = acc_.to(o_ptr.type.element_ty)  # noqa

    # Write back output
    o_batch_stride = tl.multiple_of(o_batch_stride, d * h)
    o_ptr += bid * o_batch_stride
    o_ptr += hid * o_head_stride
    # Each split-D CTA writes only its assigned D64 output slice.
    o_cols = d_start + tl.arange(0, BLOCK_D)
    o_dmask = o_cols < d
    o_offset = row_idx[:, None] * o_row_stride + o_cols[None, :]

    if IS_EVEN_MN & (d == BLOCK_K) & (not SPLIT_D):
        tl.store(o_ptr + o_offset, out)
    else:
        tl.store(
            o_ptr + o_offset, out, mask=(row_idx[:, None] < seqlen_q) & o_dmask[None, :]
        )

    # Write back lse
    p_lse = softmax_lse_ptr + (bid * h + hid) * seqlen_q
    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)

    # Both split-D CTAs produce the same LSE, so only D split 0 writes it
    # to avoid duplicate stores to the same address.
    lse_write_mask = d_split == 0
    if IS_EVEN_MN:
        tl.store(p_lse + row_idx, lse, mask=lse_write_mask)
    else:
        tl.store(p_lse + row_idx, lse, mask=(row_idx < seqlen_q) & lse_write_mask)


@triton.jit(do_not_specialize=["seqlen_q", "seqlen_k"])
def flash_fwd_bh_parallel_kernel():
    # (TODO)
    pass


def flash_fwd_splitkv_kernel_heur_block_k(args):
    return triton.next_power_of_2(args["d"])


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": block_m_splitkv_heuristic_spec_args,
        "BLOCK_N": block_n_splitkv_heuristic_spec_args,
        "BLOCK_K": flash_fwd_splitkv_kernel_heur_block_k,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 3,
        "PRE_LOAD_V": lambda args: True,
        "IS_EVEN_MN": is_even_mn_spec_args,
    }
)
@triton.jit(
    do_not_specialize=["seqlen_q", "seqlen_k", "seqlen_q_rounded", "seqlen_k_rounded"]
)
def flash_fwd_splitkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    is_cu_seqlens_q,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k: tl.constexpr,
    cu_seqlens_k_ptr,
    is_seqused_k: tl.constexpr,
    seqused_k_ptr,
    # sizes
    b: tl.constexpr,
    bk: tl.constexpr,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    # scaling factors
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_batch_stride,
    q_descale_head_stride,
    q_descale_block_stride,
    k_descale_batch_stride,
    k_descale_head_stride,
    k_descale_block_stride,
    v_descale_batch_stride,
    v_descale_head_stride,
    v_descale_block_stride,
    use_fa3_fp8_scales: tl.constexpr,
    fp8_p_max: tl.constexpr,
    # dropout
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # causal and swa
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    # alibi
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # block table
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    # kernel params
    IS_EVEN_MN: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    blocks_per_split: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    m_block = tl.program_id(0)
    split_id = tl.program_id(1)
    bid = tl.program_id(2) // h
    hid = tl.program_id(2) % h

    split_block_min = split_id * blocks_per_split
    split_block_max = split_block_min + blocks_per_split

    n_block_max = tl.cdiv(seqlen_k, BLOCK_N)
    if is_causal:
        n_block_max = min(
            n_block_max,
            tl.cdiv(
                (m_block + 1) * BLOCK_M + seqlen_k - seqlen_q + window_size_right,
                BLOCK_N,
            ),
        )

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0

    if not is_causal:
        if IS_EVEN_MN:
            masking_block_min = n_block_max
        else:
            masking_block_min = n_block_max - 1
    elif is_causal and IS_EVEN_MN:  # causal implies window_size_right is zero
        masking_block_min = n_block_max - tl.cdiv(BLOCK_M, BLOCK_N)
    else:
        masking_block_min = n_block_max - tl.cdiv(BLOCK_M, BLOCK_N) - 1

    q_ptr += bid * q_batch_stride
    q_ptr += hid * q_head_stride
    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    q_off = row_idx[:, None] * q_row_stride + tl.arange(0, BLOCK_K)[None, :]
    p_qm = q_ptr + q_off
    dmask = tl.arange(0, BLOCK_K) < d
    qmask = dmask[None, :] & (row_idx[:, None] < seqlen_q)
    if IS_EVEN_MN & BLOCK_K == d:
        Q = tl.load(p_qm, cache_modifier=".cg")
    else:
        Q = tl.load(p_qm, mask=qmask, cache_modifier=".cg")

    h_hk_ratio = h // hk
    kv_hid = hid // h_hk_ratio
    k_ptr += bid * k_batch_stride
    k_ptr += kv_hid * k_head_stride
    v_ptr += bid * k_batch_stride
    v_ptr += kv_hid * k_head_stride

    k_offset = (
        tl.arange(0, BLOCK_N)[None, :] * k_row_stride + tl.arange(0, BLOCK_K)[:, None]
    )
    p_k0 = k_ptr + k_offset

    v_offset = (
        tl.arange(0, BLOCK_N)[:, None] * k_row_stride + tl.arange(0, BLOCK_K)[None, :]
    )
    p_v0 = v_ptr + v_offset

    acc_ = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    if split_block_max <= masking_block_min:
        # no masking needed
        for n_block in tl.range(
            split_block_min, split_block_max, num_stages=num_stages
        ):
            kv_off = n_block * BLOCK_N * k_row_stride
            if d == BLOCK_K:
                K = tl.load(p_k0 + kv_off, cache_modifier=".cg")
            else:
                K = tl.load(
                    p_k0 + kv_off, mask=dmask[:, None], cache_modifier=".cg", other=0.0
                )
            if PRE_LOAD_V:
                if d == BLOCK_K:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
                else:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :],
                        cache_modifier=".cg",
                        other=0.0,
                    )
            row_start = m_block * BLOCK_M
            col_start = n_block * BLOCK_N
            q_descale, k_descale, v_descale = fa3_load_dense_descales(
                q_descale_ptr,
                k_descale_ptr,
                v_descale_ptr,
                q_descale_batch_stride,
                q_descale_head_stride,
                q_descale_block_stride,
                k_descale_batch_stride,
                k_descale_head_stride,
                k_descale_block_stride,
                v_descale_batch_stride,
                v_descale_head_stride,
                v_descale_block_stride,
                bid,
                hid,
                kv_hid,
                row_start,
                col_start,
            )
            # W8A8 FA3 change: split-KV QK also applies per-block descales.
            S = tl.dot(Q, K, out_dtype=tl.float32)
            S *= q_descale * k_descale
            S = apply_softcap(S, softcap, is_softcap)
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            acc_, P, rowmax_, rowsum_ = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=False,
            )

            if not PRE_LOAD_V:
                if d == BLOCK_K:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
                else:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :],
                        cache_modifier=".cg",
                        other=0.0,
                    )
            # W8A8 FA3 change: split-KV PV uses an in-kernel FP8 P tile and FP8 V.
            acc_ = fa3_fp8_pv_dot(
                P, V, acc_, v_descale, fp8_p_max, v_ptr.type.element_ty
            )
    else:
        for n_block in tl.range(split_block_min, min(split_block_max, n_block_max)):
            kv_off = n_block * BLOCK_N * k_row_stride
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            if IS_EVEN_MN & d == BLOCK_K:
                K = tl.load(p_k0 + kv_off, cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
            elif d == BLOCK_K:
                kvmask = col_idx < seqlen_k
                K = tl.load(p_k0 + kv_off, mask=kvmask[None, :], cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(
                        p_v0 + kv_off, mask=kvmask[:, None], cache_modifier=".cg"
                    )
            else:
                kvmask = col_idx < seqlen_k
                K = tl.load(
                    p_k0 + kv_off,
                    mask=dmask[:, None] & kvmask[None, :],
                    cache_modifier=".cg",
                    other=0.0,
                )
                if PRE_LOAD_V:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :] & kvmask[:, None],
                        cache_modifier=".cg",
                        other=0.0,
                    )

            row_start = m_block * BLOCK_M
            col_start = n_block * BLOCK_N
            q_descale, k_descale, v_descale = fa3_load_dense_descales(
                q_descale_ptr,
                k_descale_ptr,
                v_descale_ptr,
                q_descale_batch_stride,
                q_descale_head_stride,
                q_descale_block_stride,
                k_descale_batch_stride,
                k_descale_head_stride,
                k_descale_block_stride,
                v_descale_batch_stride,
                v_descale_head_stride,
                v_descale_block_stride,
                bid,
                hid,
                kv_hid,
                row_start,
                col_start,
            )
            # W8A8 FA3 change: masked split-KV QK applies per-block descales.
            S = tl.dot(Q, K, out_dtype=tl.float32)
            S *= q_descale * k_descale
            S = apply_softcap(S, softcap, is_softcap)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            S = apply_mask(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                is_even_mn=IS_EVEN_MN,
                is_causal=is_causal,
                is_local=False,
            )

            acc_, P, rowmax_, rowsum_ = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=(is_causal or is_local),
            )

            if not PRE_LOAD_V:
                if IS_EVEN_MN & d == BLOCK_K:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
                elif d == BLOCK_K:
                    V = tl.load(
                        p_v0 + kv_off, mask=kvmask[:, None], cache_modifier=".cg"
                    )
                else:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :] & kvmask[:, None],
                        cache_modifier=".cg",
                        other=0.0,
                    )
            # W8A8 FA3 change: masked split-KV PV runs as FP8 P * FP8 V.
            acc_ = fa3_fp8_pv_dot(
                P, V, acc_, v_descale, fp8_p_max, v_ptr.type.element_ty
            )

    # LSE
    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("-inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

    # Rescale output
    acc_ *= inv_sum[:, None]

    # Write back output
    # o_splits layout = (n_splits, batch_size, num_heads, seqlen_q, head_size)
    # grid = (seq_block, split, batch * head)
    o_split_ptr = o_ptr
    # + split, batch, head offsets, seq_block offsets are already added in row_idx
    o_split_ptr += (split_id * tl.num_programs(2) + tl.program_id(2)) * seqlen_q * d
    o_split_offset = row_idx[:, None] * d + tl.arange(0, BLOCK_K)
    o_split_ptr = tl.multiple_of(o_split_ptr, d)
    p_om = o_split_ptr + o_split_offset

    if IS_EVEN_MN & BLOCK_K == d:
        tl.store(p_om, acc_, cache_modifier=".cg")
    else:
        tl.store(p_om, acc_, mask=qmask, cache_modifier=".cg")

    # Write back lse
    # lse_splits layout = (n_splits, batch_size, num_heads, seqlen_q)
    lse_split_ptr = softmax_lse_ptr
    # + split, batch, head, seq_block offsets
    lse_split_ptr += (
        split_id * tl.num_programs(2) + tl.program_id(2)
    ) * seqlen_q + m_block * BLOCK_M

    if IS_EVEN_MN:
        tl.store(lse_split_ptr + tl.arange(0, BLOCK_M), lse, cache_modifier=".cg")
    else:
        tl.store(
            lse_split_ptr + tl.arange(0, BLOCK_M),
            lse,
            mask=row_idx < seqlen_q,
            cache_modifier=".cg",
        )


@libentry()
@triton.jit
def flash_fwd_splitkv_combine_kernel(
    out_ptr,
    lse_ptr,
    out_splits_ptr,
    lse_splits_ptr,
    head_size: tl.constexpr,
    out_split_stride,
    lse_split_stride,
    out_b_stride,
    out_s_stride,
    out_h_stride,
    n_splits,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    q_total,
    MAX_N_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    lse_splits_ptr += pid * BLOCK_M
    lse_ptr += pid * BLOCK_M
    out_splits_ptr += pid * BLOCK_M * head_size
    out_ptr += pid * BLOCK_M * head_size

    # Subtracting maximum from each of the split lse's for better numerical stability
    lse_split_offset = (
        tl.arange(0, BLOCK_M)[:, None]
        + tl.arange(0, MAX_N_SPLITS)[None, :] * lse_split_stride
    )
    lse_split_mask = (pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < q_total) & (
        tl.arange(0, MAX_N_SPLITS)[None, :] < n_splits
    )
    lse_splits = tl.load(
        lse_splits_ptr + lse_split_offset, mask=lse_split_mask, other=float("-inf")
    )
    max_lse = tl.max(lse_splits, 1)

    # Sum exp(lse(i) - max_lse) over all split i to obtain Z=sumexp(QK) up to a scaled factor exp(-max_lse)
    Zi_scaled = tl.exp(lse_splits - max_lse[:, None])
    Z_scaled = tl.sum(Zi_scaled, 1)
    Zi_Z = Zi_scaled / Z_scaled[:, None]

    # Write back LSE
    lse = tl.log(Z_scaled) + max_lse
    out_mask = pid * BLOCK_M + tl.arange(0, BLOCK_M) < q_total
    tl.store(lse_ptr + tl.arange(0, BLOCK_M), lse, mask=out_mask)

    out_split_offset = (
        tl.arange(0, BLOCK_M)[:, None, None] * head_size
        + tl.arange(0, MAX_N_SPLITS)[None, :, None] * out_split_stride
        + tl.arange(0, BLOCK_K)[None, None, :]
    )
    out_split_mask = (
        (pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None, None] < q_total)
        & (tl.arange(0, MAX_N_SPLITS)[None, :, None] < n_splits)
        & (tl.arange(0, BLOCK_K)[None, None, :] < head_size)
    )
    out_splits = tl.load(
        out_splits_ptr + out_split_offset, mask=out_split_mask, other=0.0
    )
    out = tl.sum(Zi_Z[:, :, None] * out_splits, 1)
    out = out.to(out_ptr.type.element_ty)

    # Write back output
    out_offset = tl.arange(0, BLOCK_M)[:, None] * out_s_stride + tl.arange(0, BLOCK_K)
    dmask = tl.arange(0, BLOCK_K) < head_size
    tl.store(out_ptr + out_offset, out, mask=out_mask[:, None] & dmask[None, :])


@triton.jit
def virtual_to_cache(
    virtual_index,
    max_virtual_index,
    page_table_ptr,
    block_size,
    boundary_check: tl.constexpr = False,
):
    # virtual_index is the kv sequence index in the current batch element
    # page_table_ptr is already pointed at current batch element's block table entry
    # block_size is the size of each block in the page table
    virtual_page_index = virtual_index // block_size
    page_offset = virtual_index % block_size
    if boundary_check:
        page_block_index = tl.load(
            page_table_ptr + virtual_page_index,
            mask=virtual_index < max_virtual_index,
            other=0,
        ).to(tl.int32)
    else:
        page_block_index = tl.load(page_table_ptr + virtual_page_index).to(tl.int32)
    return page_block_index * block_size + page_offset


@triton.jit
def load_from_kvcache(
    virtual_index,
    max_virtual_index,
    page_table_ptr,
    k_ptr_base,
    v_ptr_base,
    block_size,
    d: tl.constexpr,
    k_row_stride,
    BLOCK_K: tl.constexpr,
    boundary_check: tl.constexpr = False,
):
    kvcache_idx = virtual_to_cache(
        virtual_index, max_virtual_index, page_table_ptr, block_size, boundary_check
    )
    k_offset = tl.arange(0, BLOCK_K)[:, None] + kvcache_idx[None, :] * k_row_stride
    v_offset = tl.arange(0, BLOCK_K)[None, :] + kvcache_idx[:, None] * k_row_stride
    if d == BLOCK_K:
        bK_mask = virtual_index[None, :] < max_virtual_index[None, :]
        bV_mask = virtual_index[:, None] < max_virtual_index[:, None]
        bK = tl.load(k_ptr_base + k_offset, mask=bK_mask, other=0.0)
        bV = tl.load(v_ptr_base + v_offset, mask=bV_mask, other=0.0)
    else:
        bK_mask = (tl.arange(0, BLOCK_K)[:, None] < d) & (
            virtual_index[None, :] < max_virtual_index[None, :]
        )
        bV_mask = (tl.arange(0, BLOCK_K)[None, :] < d) & (
            virtual_index[:, None] < max_virtual_index[:, None]
        )
        bK = tl.load(k_ptr_base + k_offset, mask=bK_mask, other=0.0)
        bV = tl.load(v_ptr_base + v_offset, mask=bV_mask, other=0.0)
    return bK, bV


@libentry()
@triton.jit(
    do_not_specialize=[
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "b",
        "bk",
        "seqlen_q",
        "seqlen_k",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
        "total_q",
    ]
)
def flash_varlen_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    is_cu_seqlens_q: tl.constexpr,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k: tl.constexpr,
    cu_seqlens_k_ptr,
    is_seqused_k: tl.constexpr,
    seqused_k_ptr,
    # sizes
    b,
    bk,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    # scaling factors
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_batch_stride,
    q_descale_head_stride,
    q_descale_block_stride,
    k_descale_batch_stride,
    k_descale_head_stride,
    k_descale_block_stride,
    v_descale_batch_stride,
    v_descale_head_stride,
    v_descale_block_stride,
    use_fa3_fp8_scales: tl.constexpr,
    fp8_p_max: tl.constexpr,
    # dropout
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # causal and swa
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    # alibi
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # block table
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    # kernel params
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLIT_D: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    m_block = tl.program_id(0)
    bid = tl.program_id(1)
    # The varlen grid has only three dimensions, so encode both the head
    # and D split in program_id(2).
    hd = tl.program_id(2)
    hid = hd % h
    d_split = hd // h
    d_start = d_split * BLOCK_D
    # num_m_blocks = tl.cdiv(seqlen_q, BLOCK_M)

    if is_cu_seqlens_q:
        q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
        q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
        q_len = q_eos - q_bos
        # Current request's start offset in the batched Q
        q_offset = q_bos * q_row_stride
        o_offset = q_bos * o_row_stride
        lse_offset = q_bos * 1
    else:
        q_len = seqlen_q
        q_offset = bid * q_batch_stride
        o_offset = bid * o_batch_stride
        lse_offset = bid * seqlen_q

    if is_cu_seqlens_k:
        k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
        k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
        k_len_cache = k_eos - k_bos
        # k_offset = k_bos * k_row_stride
    else:
        k_len_cache = seqlen_k
        # k_offset = bid * k_batch_stride

    if is_seqused_k:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    else:
        k_len = k_len_cache

    # Noop CTA
    if m_block * BLOCK_M > q_len:
        return

    # is_even_mn = (q_len % BLOCK_M == 0) and (k_len % BLOCK_N == 0)
    is_even_mn: tl.constexpr = False

    if is_local:
        n_block_min = max(
            0, (m_block * BLOCK_M + k_len - q_len - window_size_left) // BLOCK_N
        )
    else:
        n_block_min = 0

    n_block_max = tl.cdiv(k_len, BLOCK_N)
    if is_causal or is_local:
        n_block_max = min(
            n_block_max,
            tl.cdiv(
                (m_block + 1) * BLOCK_M + k_len - q_len + window_size_right, BLOCK_N
            ),
        )

    if is_dropout:
        philox_seed = tl.load(philox_args).to(tl.uint64)
        philox_offset = tl.load(philox_args + 1).to(tl.uint64)

    # Locate the page table entry for the current batch element
    if is_paged:
        page_table_ptr += bid * page_table_batch_stride
    # Calculate the starting offset of q for the current head
    q_row_offset = hid * q_head_stride
    # Calculate the starting offset of k and v for the current head
    kv_hid = hid // h_hk_ratio
    k_row_offset = kv_hid * k_head_stride
    # Shift the k, v pointers to align with the current head
    k_ptr_base = k_ptr + k_row_offset
    v_ptr_base = v_ptr + k_row_offset

    gQ = tl.make_block_ptr(
        base=q_ptr + q_offset + q_row_offset,
        shape=(q_len, d),
        strides=(q_row_stride, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    bQ = tl.load(gQ.advance([m_block * BLOCK_M, 0]), boundary_check=(0, 1))

    # Partition the varlen PV accumulator by BLOCK_D as well, so each CTA
    # holds only D64 for D128.
    acc_ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0.0

    if not is_causal and not is_local:
        n_masking_steps = 1
    elif is_even_mn:
        n_masking_steps = tl.cdiv(BLOCK_M, BLOCK_N)
    else:
        n_masking_steps = tl.cdiv(BLOCK_M, BLOCK_N) + 1

    n_masking_steps = min(n_block_max - n_block_min, n_masking_steps)

    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_block = n_block_max - 1
    for step in tl.range(0, n_masking_steps):
        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        if is_paged:
            bK, bV = load_from_kvcache(
                col_idx,
                k_len,
                page_table_ptr,
                k_ptr_base,
                v_ptr_base,
                block_size,
                d,
                k_row_stride,
                BLOCK_K=BLOCK_K,
                boundary_check=True,
            )
        else:
            start_n = n_block * BLOCK_N
            k_ptr_seq = k_ptr_base + k_bos * k_row_stride
            v_ptr_seq = v_ptr_base + k_bos * k_row_stride
            gK = tl.make_block_ptr(
                base=k_ptr_seq,
                shape=(k_len, d),
                strides=(k_row_stride, 1),
                offsets=(start_n, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(0, 1),
            )
            gV = tl.make_block_ptr(
                base=v_ptr_seq,
                shape=(k_len, d),
                strides=(k_row_stride, 1),
                # For non-paged varlen attention, load only the V slice for
                # the current D split.
                offsets=(start_n, d_start),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(0, 1),
            )
            bK = tl.load(gK, boundary_check=(0, 1))
            bK = tl.trans(bK)
            bV = tl.load(gV, boundary_check=(0, 1))
        q_descale, k_descale, v_descale = fa3_load_dense_descales(
            q_descale_ptr,
            k_descale_ptr,
            v_descale_ptr,
            q_descale_batch_stride,
            q_descale_head_stride,
            q_descale_block_stride,
            k_descale_batch_stride,
            k_descale_head_stride,
            k_descale_block_stride,
            v_descale_batch_stride,
            v_descale_head_stride,
            v_descale_block_stride,
            bid,
            hid,
            kv_hid,
            m_block * BLOCK_M,
            n_block * BLOCK_N,
        )
        # W8A8 FA3 change: varlen QK uses FP8 Q/K plus per-request block descales.
        S = tl.dot(bQ, bK, out_dtype=tl.float32)
        S *= q_descale * k_descale
        S = apply_softcap(S, softcap, is_softcap)
        S = apply_alibi(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            is_causal=is_causal,
            is_alibi=is_alibi,
            alibi_slope=alibi_slope,
        )
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            window_size_left,
            window_size_right,
            is_even_mn=is_even_mn,
            is_causal=is_causal,
            is_local=is_local,
        )

        acc_, P, rowmax_, rowsum_ = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=True,
        )
        if is_dropout:
            P = apply_dropout(
                P,
                n_block * BLOCK_N,
                m_block * BLOCK_M,
                k_len,
                bid,
                hid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                is_dropout,
                encode_dropout_in_sign_bit=False,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        # W8A8 FA3 change: varlen PV uses dynamically quantized FP8 P and FP8 V.
        acc_ = fa3_fp8_pv_dot(P, bV, acc_, v_descale, fp8_p_max, v_ptr.type.element_ty)
        n_block -= 1

    for n_block in tl.range(
        n_block_max - n_masking_steps - 1, n_block_min - 1, step=-1
    ):
        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        if is_paged:
            bK, bV = load_from_kvcache(
                col_idx,
                k_len,
                page_table_ptr,
                k_ptr_base,
                v_ptr_base,
                block_size,
                d,
                k_row_stride,
                BLOCK_K=BLOCK_K,
            )
        else:
            start_n = n_block * BLOCK_N
            k_ptr_seq = k_ptr_base + k_bos * k_row_stride
            v_ptr_seq = v_ptr_base + k_bos * k_row_stride
            gK = tl.make_block_ptr(
                base=k_ptr_seq,
                shape=(k_len, d),
                strides=(k_row_stride, 1),
                offsets=(start_n, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(0, 1),
            )
            gV = tl.make_block_ptr(
                base=v_ptr_seq,
                shape=(k_len, d),
                strides=(k_row_stride, 1),
                # For non-paged varlen attention, load only the V slice for
                # the current D split.
                offsets=(start_n, d_start),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(0, 1),
            )
            bK = tl.load(gK)
            bK = tl.trans(bK)
            bV = tl.load(gV)
        q_descale, k_descale, v_descale = fa3_load_dense_descales(
            q_descale_ptr,
            k_descale_ptr,
            v_descale_ptr,
            q_descale_batch_stride,
            q_descale_head_stride,
            q_descale_block_stride,
            k_descale_batch_stride,
            k_descale_head_stride,
            k_descale_block_stride,
            v_descale_batch_stride,
            v_descale_head_stride,
            v_descale_block_stride,
            bid,
            hid,
            kv_hid,
            m_block * BLOCK_M,
            n_block * BLOCK_N,
        )
        # W8A8 FA3 change: non-masking varlen QK uses the same block descale path.
        S = tl.dot(bQ, bK, out_dtype=tl.float32)
        S *= q_descale * k_descale
        S = apply_softcap(S, softcap, is_softcap)
        S = apply_alibi(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            is_causal=is_causal,
            is_alibi=is_alibi,
            alibi_slope=alibi_slope,
        )
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            window_size_left,
            window_size_right,
            is_even_mn=True,
            is_causal=False,
            is_local=is_local,
        )

        acc_, P, rowmax_, rowsum_ = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=is_local,
        )
        if is_dropout:
            P = apply_dropout(
                P,
                m_block * BLOCK_M,
                n_block * BLOCK_N,
                k_len,
                bid,
                hid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                is_dropout,
                encode_dropout_in_sign_bit=False,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
        # W8A8 FA3 change: non-masking varlen PV runs as FP8 P * FP8 V.
        acc_ = fa3_fp8_pv_dot(P, bV, acc_, v_descale, fp8_p_max, v_ptr.type.element_ty)

    # LSE
    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

    acc_ *= inv_sum[:, None]

    out = acc_.to(o_ptr.type.element_ty)  # noqa

    # Write back output
    o_row_offset = hid * o_head_stride

    gO = tl.make_block_ptr(
        base=o_ptr + o_offset + o_row_offset,
        shape=(q_len, d),
        strides=(o_row_stride, 1),
        # Each varlen split-D CTA writes only its current D64 output slice.
        offsets=(0, d_start),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(gO.advance([m_block * BLOCK_M, 0]), out, boundary_check=(0, 1))

    # Write back lse
    # lse shape: [h, total_q]
    softmax_lse_ptr += hid * total_q
    lse_row_offset = lse_offset + m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    # Both varlen split-D CTAs produce the same LSE, so only D split 0 writes
    # it back.
    tl.store(
        softmax_lse_ptr + lse_row_offset,
        lse,
        mask=(lse_row_offset < (lse_offset + q_len)) & (d_split == 0),
    )


def CHECK_DEVICE(x):
    assert x.device.type == flag_gems.device


# W8A8 FA3 change: Q/K/V are all FP8 inputs now, while O stays fp16/bf16 like
# FlashAttention-3 forward.  The fp32 descale tensors below carry the per-block
# quantization scales used by the QK and PV tensor-core matmuls.
_FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)
_HIGH_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def _empty_descale(device):
    # W8A8 FA3 change: use a real empty fp32 tensor instead of None so every
    # Triton kernel gets the same pointer-bearing argument list.
    return torch.empty((0,), dtype=torch.float32, device=device)


def _default_dense_descale(batch_size, num_heads, nblocks, device):
    # W8A8 FA3 change: missing descale means the caller already baked the scale
    # into the FP8 numeric values; use per-block 1.0 scales for compatibility.
    return torch.ones(
        (batch_size, num_heads, nblocks), dtype=torch.float32, device=device
    )


def _normalize_dense_descale(descale, batch_size, num_heads, nblocks, device, name):
    # W8A8 FA3 change: accept FA3-style scalar/per-head/per-block scales and
    # normalize them to [B, H, nblocks] so the kernels can index one block scale.
    if descale is None:
        return _default_dense_descale(batch_size, num_heads, nblocks, device)
    assert descale.device == device, f"{name} must be on the same device as q"
    descale = descale.to(torch.float32)
    if descale.ndim == 0:
        descale = descale.reshape(1, 1, 1).expand(batch_size, num_heads, nblocks)
    elif descale.ndim == 1:
        if descale.numel() == 1:
            descale = descale.reshape(1, 1, 1).expand(batch_size, num_heads, nblocks)
        else:
            assert descale.numel() == num_heads, f"{name} 1D scale must have H elements"
            descale = descale.reshape(1, num_heads, 1).expand(
                batch_size, num_heads, nblocks
            )
    elif descale.ndim == 2:
        assert descale.shape == (
            batch_size,
            num_heads,
        ), f"{name} 2D scale must be [B, H]"
        descale = descale[:, :, None].expand(batch_size, num_heads, nblocks)
    else:
        assert descale.shape == (
            batch_size,
            num_heads,
            nblocks,
        ), f"{name} must be [B, H, nblocks]"
    return descale.contiguous()


class fwd_params:
    __slots__ = (
        # pointers and strides
        "q_ptr",
        "k_ptr",
        "v_ptr",
        "o_ptr",
        "p_ptr",
        "softmax_lse_ptr",
        "q_row_stride",
        "k_row_stride",
        "v_row_stride",
        "q_head_stride",
        "k_head_stride",
        "v_head_stride",
        "o_row_stride",
        "o_head_stride",
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "is_cu_seqlens_q",
        "cu_seqlens_q_ptr",
        "is_cu_seqlens_k",
        "cu_seqlens_k_ptr",
        "is_seqused_k",
        "seqused_k_ptr",
        # sizes
        "b",
        "bk",
        "h",
        "hk",
        "h_hk_ratio",
        "seqlen_q",
        "seqlen_k",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
        "d",
        "d_rounded",
        # scaling factors
        "is_softcap",
        "softcap",
        "scale_softmax",
        "scale_softmax_log2",
        "q_descale_ptr",
        "k_descale_ptr",
        "v_descale_ptr",
        "q_descale_batch_stride",
        "q_descale_head_stride",
        "q_descale_block_stride",
        "k_descale_batch_stride",
        "k_descale_head_stride",
        "k_descale_block_stride",
        "v_descale_batch_stride",
        "v_descale_head_stride",
        "v_descale_block_stride",
        "use_fa3_fp8_scales",
        "fp8_p_max",
        # dropout
        "is_dropout",
        "p_dropout",
        "rp_dropout",
        "p_dropout_in_uint8_t",
        "philox_args",
        "return_softmax",
        # masking
        "is_causal",
        "is_local",
        "window_size_left",
        "window_size_right",
        "seqlenq_ngroups_swapped",
        "is_paged",
        # alibi
        "is_alibi",
        "alibi_slopes_ptr",
        "alibi_slopes_batch_stride",
        # block table
        "total_q",
        "page_table_ptr",
        "page_table_batch_stride",
        "block_size",
    )

    def __init__(
        self,
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        p_ptr,
        softmax_lse_ptr,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        q_head_stride,
        k_head_stride,
        v_head_stride,
        o_row_stride,
        o_head_stride,
        q_batch_stride,
        k_batch_stride,
        v_batch_stride,
        o_batch_stride,
        is_cu_seqlens_q,
        cu_seqlens_q_ptr,
        is_cu_seqlens_k,
        cu_seqlens_k_ptr,
        is_seqused_k,
        seqused_k_ptr,
        # sizes
        b,
        bk,
        h,
        hk,
        h_hk_ratio,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        d,
        d_rounded,
        # scaling factors
        is_softcap,
        softcap,
        scale_softmax,
        scale_softmax_log2,
        q_descale_ptr,
        k_descale_ptr,
        v_descale_ptr,
        q_descale_batch_stride,
        q_descale_head_stride,
        q_descale_block_stride,
        k_descale_batch_stride,
        k_descale_head_stride,
        k_descale_block_stride,
        v_descale_batch_stride,
        v_descale_head_stride,
        v_descale_block_stride,
        use_fa3_fp8_scales,
        fp8_p_max,
        # dropout
        is_dropout,
        p_dropout,
        rp_dropout,
        p_dropout_in_uint8_t,
        philox_args,
        return_softmax,
        # masking
        is_causal,
        is_local,
        window_size_left,
        window_size_right,
        seqlenq_ngroups_swapped,
        is_paged,
        # alibi
        is_alibi,
        alibi_slopes_ptr,
        alibi_slopes_batch_stride,
        # block table
        total_q,
        page_table_ptr,
        page_table_batch_stride,
        block_size,
    ):
        self.q_ptr = q_ptr
        self.k_ptr = k_ptr
        self.v_ptr = v_ptr
        self.o_ptr = o_ptr
        self.p_ptr = p_ptr
        self.softmax_lse_ptr = softmax_lse_ptr
        self.q_row_stride = q_row_stride
        self.k_row_stride = k_row_stride
        self.v_row_stride = v_row_stride
        self.q_head_stride = q_head_stride
        self.k_head_stride = k_head_stride
        self.v_head_stride = v_head_stride
        self.o_row_stride = o_row_stride
        self.o_head_stride = o_head_stride
        self.q_batch_stride = q_batch_stride
        self.k_batch_stride = k_batch_stride
        self.v_batch_stride = v_batch_stride
        self.o_batch_stride = o_batch_stride
        self.is_cu_seqlens_q = is_cu_seqlens_q
        self.cu_seqlens_q_ptr = cu_seqlens_q_ptr
        self.is_cu_seqlens_k = is_cu_seqlens_k
        self.cu_seqlens_k_ptr = cu_seqlens_k_ptr
        self.is_seqused_k = is_seqused_k
        self.seqused_k_ptr = seqused_k_ptr
        # sizes
        self.b = b
        self.bk = bk
        self.h = h
        self.hk = hk
        self.h_hk_ratio = h_hk_ratio
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.seqlen_q_rounded = seqlen_q_rounded
        self.seqlen_k_rounded = seqlen_k_rounded
        self.d = d
        self.d_rounded = d_rounded
        # scaling factors
        self.is_softcap = is_softcap
        self.softcap = softcap
        self.scale_softmax = scale_softmax
        self.scale_softmax_log2 = scale_softmax_log2
        self.q_descale_ptr = q_descale_ptr
        self.k_descale_ptr = k_descale_ptr
        self.v_descale_ptr = v_descale_ptr
        self.q_descale_batch_stride = q_descale_batch_stride
        self.q_descale_head_stride = q_descale_head_stride
        self.q_descale_block_stride = q_descale_block_stride
        self.k_descale_batch_stride = k_descale_batch_stride
        self.k_descale_head_stride = k_descale_head_stride
        self.k_descale_block_stride = k_descale_block_stride
        self.v_descale_batch_stride = v_descale_batch_stride
        self.v_descale_head_stride = v_descale_head_stride
        self.v_descale_block_stride = v_descale_block_stride
        self.use_fa3_fp8_scales = use_fa3_fp8_scales
        self.fp8_p_max = fp8_p_max
        # dropout
        self.is_dropout = is_dropout
        self.p_dropout = p_dropout
        self.rp_dropout = rp_dropout
        self.p_dropout_in_uint8_t = p_dropout_in_uint8_t
        self.philox_args = philox_args
        self.return_softmax = return_softmax
        # masking
        self.is_causal = is_causal
        self.is_local = is_local
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped
        self.is_paged = is_paged
        # alibi
        self.is_alibi = is_alibi
        self.alibi_slopes_ptr = alibi_slopes_ptr
        self.alibi_slopes_batch_stride = alibi_slopes_batch_stride
        # block table
        self.total_q = total_q
        self.page_table_ptr = page_table_ptr
        self.page_table_batch_stride = page_table_batch_stride
        self.block_size = block_size

    def args(self):
        return tuple(getattr(self, k) for k in self.__slots__)


def mha_varlan_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    fp8_p_max=448.0,
):
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_device = q.device
    q_dtype = q.dtype
    v_dtype = v.dtype
    # W8A8 FA3 change: varlen forward also consumes FP8 Q/K/V and uses explicit
    # descale tensors, matching the dense kernel contract.
    assert q_dtype in _FP8_DTYPES, "W8A8 FA3 FlashAttention expects q to be fp8"
    assert (
        k.dtype == q_dtype
    ), "W8A8 FA3 FlashAttention expects q and k to use the same fp8 dtype"
    assert v_dtype == q_dtype, "W8A8 FA3 FlashAttention expects v to be fp8"
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"

    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_q.is_contiguous()

    assert cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_k.is_contiguous()

    is_paged = page_table is not None
    if not is_paged:
        page_table = torch.empty((0, 0), device=q_device, dtype=torch.int32)

    # q shape: [total_q_tokens, num_heads, head_size]
    # k shape:
    #   paged_kv: [num_pages, block_size, num_heads_k, head_size]
    # batch_size, number of sentences
    total_q, num_heads, head_size = q.size()
    num_heads_k = k.size(2) if is_paged else k.size(1)
    batch_size = cu_seqlens_q.numel() - 1
    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    k_batch_size = num_pages
    # max_num_pages_per_seq = page_table.size(1)
    page_table_batch_stride = page_table.stride(0)
    k_batch_stride = k.stride(0)
    v_batch_stride = v.stride(0)

    assert k.size() == v.size()
    assert cu_seqlens_q.size() == (batch_size + 1,)
    assert cu_seqlens_k.size() == (batch_size + 1,)

    # Check output shape
    if out is not None:
        assert out.stride(-1) == 1
        # W8A8 FA3 change: output stays high precision although V is FP8.
        assert out.dtype in _HIGH_PRECISION_DTYPES
        assert out.size() == (total_q, num_heads, head_size)

    if seqused_k is not None:
        assert seqused_k.is_contiguous()
        assert seqused_k.size() == (batch_size,)

    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False

    if is_causal:
        window_size_right = 0

    # check disable swa
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_k:
        window_size_right = -1

    is_local = window_size_left >= 0

    # Optimize all single-query sequences by swapping the query-group and sequence dimensions
    seqlenq_ngroups_swapped = (
        max_seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
    )
    q_groups = num_heads // num_heads_k
    if seqlenq_ngroups_swapped:
        logger.debug("Swapping query groups and sequence dimensions")
        q = (
            q.reshape((batch_size, num_heads_k, q_groups, head_size))
            .transpose(1, 2)
            .reshape(batch_size * q_groups, num_heads_k, head_size)
        )
        max_seqlen_q = q_groups
        num_heads = num_heads_k
        cu_seqlens_q = None
        q_batch_stride = q.stride(0) * max_seqlen_q
        k_batch_stride = k.stride(0)
        v_batch_stride = v.stride(0)
        # o_batch_stride = out.stride(0) * max_seqlen_q
    else:
        q_batch_stride = 0
        k_batch_stride = 0
        v_batch_stride = 0
        o_batch_stride = 0

    total_q = q.size(0)

    assert leftpad_k is None, "leftpad_k is not supported."
    assert (
        head_size <= 256
    ), "FlashAttention forward only supports head dimension at most 256"
    assert (
        head_size % 8 == 0
    ), "head_size must be a multiple of 8, this is ensured by padding!"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"

    assert q.shape == (total_q, num_heads, head_size)
    if is_paged:
        assert k.shape == (num_pages, block_size, num_heads_k, head_size)
        assert v.shape == (num_pages, block_size, num_heads_k, head_size)
    assert k.stride() == v.stride()

    if softcap > 0.0:
        assert p_dropout == 0, "dropout is not supported if softcap is used."

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32) if head_size <= 192 else 256
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 32)

    M_LOG2E = 1.4426950408889634074
    if softcap > 0.0:
        is_softcap = True
        adjusted_scale_softmax = softcap
        adjusted_softcap = softmax_scale / softcap
        adjusted_scale_softmax_log2e = softcap * M_LOG2E
    else:
        is_softcap = False
        adjusted_softcap = 0.0
        adjusted_scale_softmax = softmax_scale
        adjusted_scale_softmax_log2e = softmax_scale * M_LOG2E

    # Set alibi params
    if alibi_slopes is not None:
        assert alibi_slopes.device == q_device
        assert alibi_slopes.dtype in (torch.float,)
        assert alibi_slopes.stride(-1) == 1
        assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
            batch_size,
            num_heads,
        )
        alibi_slopes_batch_stride = (
            alibi_slopes.stride(0) if alibi_slopes.ndim == 2 else 0
        )
        is_alibi = True
    else:
        alibi_slopes_batch_stride = 0
        is_alibi = False

    # Prepare params to kernel
    with torch_device_fn.device(q_device):
        if out is not None:
            out_ = out
            if seqlenq_ngroups_swapped:
                out = torch.empty_like(q, dtype=torch.bfloat16)
        else:
            out_ = None
            out = torch.empty_like(q, dtype=torch.bfloat16)

        if seqlenq_ngroups_swapped:
            o_batch_stride = out.stride(0) * max_seqlen_q

        lse = torch.empty((num_heads, total_q), dtype=torch.float, device=q_device)

        if p_dropout > 0:
            is_dropout = True
            increment = batch_size * num_heads * 32
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            philox_args = torch.tensor(
                [philox_seed, philox_offset], dtype=torch.int64, device=q_device
            )
        else:
            is_dropout = False
            philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)

        p_dropout = 1 - p_dropout
        p_dropout_in_uint8_t = math.floor(p_dropout * 255.0)
        rp_dropout = 1.0 / p_dropout

        if return_softmax:
            assert is_dropout, "Only supported with non-zero dropout."
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                device=q_device,
            )
        else:
            p = torch.empty((), device=q_device)

        if zero_tensors:
            out.zero_()
            lse.fill_(float("-inf"))

        q_nblocks = triton.cdiv(max_seqlen_q, 128)
        k_nblocks = triton.cdiv(max_seqlen_k, 128)
        # W8A8 FA3 change: varlen kernels use [B, H, block] descales indexed by
        # request id and local block id.  Paged KV uses the same logical blocks.
        q_descale = _normalize_dense_descale(
            q_descale, batch_size, num_heads, q_nblocks, q_device, "q_descale"
        )
        k_descale = _normalize_dense_descale(
            k_descale, batch_size, num_heads_k, k_nblocks, q_device, "k_descale"
        )
        v_descale = _normalize_dense_descale(
            v_descale, batch_size, num_heads_k, k_nblocks, q_device, "v_descale"
        )

        params = fwd_params(
            q,  # q_ptr,
            k,  # k_ptr,
            v,  # v_ptr,
            out,  # o_ptr,
            p,  # p_ptr,
            lse,  # softmax_lse_ptr,
            q.stride(-3),  # q_row_stride,
            k.stride(-3),  # k_row_stride,
            v.stride(-3),  # v_row_stride,
            q.stride(-2),  # q_head_stride,
            k.stride(-2),  # k_head_stride,
            v.stride(-2),  # v_head_stride,
            out.stride(-3),  # o_row_stride,
            out.stride(-2),  # o_head_stride,
            q_batch_stride,  # q_batch_stride,
            k_batch_stride,  # k_batch_stride,
            v_batch_stride,  # v_batch_stride,
            o_batch_stride,  # o_batch_stride,
            cu_seqlens_q is not None,  # is_cu_seqlens_q,
            cu_seqlens_q,  # cu_seqlens_q_ptr,
            seqused_k is None,  # is_cu_seqlens_k,
            cu_seqlens_k,  # cu_seqlens_k_ptr,
            seqused_k is not None,  # is_seqused_k,
            seqused_k,  # seqused_k_ptr,
            # sizes
            batch_size,  # b,
            k_batch_size,  # bk,
            num_heads,  # h,
            num_heads_k,  # hk,
            num_heads // num_heads_k,  # h_hk_ratio,
            max_seqlen_q,  # seqlen_q,
            max_seqlen_k,  # seqlen_k,
            seqlen_q_rounded,  # seqlen_q_rounded,
            seqlen_k_rounded,  # seqlen_k_rounded,
            head_size,  # d,
            head_size_rounded,  # d_rounded,
            # scaling factors
            is_softcap,
            adjusted_softcap,  # softcap,
            adjusted_scale_softmax,  # scale_softmax,
            adjusted_scale_softmax_log2e,  # scale_softmax_log2,
            q_descale,  # q_descale_ptr,
            k_descale,  # k_descale_ptr,
            v_descale,  # v_descale_ptr,
            q_descale.stride(0),  # q_descale_batch_stride,
            q_descale.stride(1),  # q_descale_head_stride,
            q_descale.stride(2),  # q_descale_block_stride,
            k_descale.stride(0),  # k_descale_batch_stride,
            k_descale.stride(1),  # k_descale_head_stride,
            k_descale.stride(2),  # k_descale_block_stride,
            v_descale.stride(0),  # v_descale_batch_stride,
            v_descale.stride(1),  # v_descale_head_stride,
            v_descale.stride(2),  # v_descale_block_stride,
            True,  # use_fa3_fp8_scales,
            fp8_p_max,  # fp8_p_max,
            # dropout
            is_dropout,
            p_dropout,
            rp_dropout,
            p_dropout_in_uint8_t,
            philox_args,
            return_softmax,
            # causal and swa
            is_causal,  # is_causal,
            is_local,  # is_local,
            window_size_left,  # window_size_left,
            window_size_right,  # window_size_right,
            seqlenq_ngroups_swapped,  # seqlenq_ngroups_swapped,
            is_paged,
            # alibi
            is_alibi,  #
            alibi_slopes,  # alibi_slopes_ptr,
            alibi_slopes_batch_stride,  # alibi_slopes_batch_stride,
            # block table params
            total_q,  # total_q,
            page_table,  # page_table_ptr,
            page_table_batch_stride,  # page_table_batch_stride,
            block_size,  # block_size,
        )

        if flag_gems.vendor_name == "iluvatar":
            params.k_ptr = k.view(k.shape[0], k.shape[1], -1)
            params.v_ptr = v.view(v.shape[0], v.shape[1], -1)
        logger.debug("kernel: flash_varlen_fwd")
        # Enable two-way split-D for non-paged D128 varlen attention. The
        # paged-cache loader continues to load the full D dimension.
        use_varlen_split_d = head_size == 128 and not is_paged
        grid = lambda args: (
            triton.cdiv(max_seqlen_q, args["BLOCK_M"]),
            batch_size,
            num_heads * (2 if use_varlen_split_d else 1),
        )
        kernel = flash_varlen_fwd_kernel[grid]
        args = tuple(getattr(params, k) for k in params.__slots__)

        # We assess which phase the requests are likely to be in and set the config accordingly.
        total_rows = total_q * num_heads
        num_sms = torch_device_fn.get_device_properties(
            flag_gems.device
        ).multi_processor_count
        avg_rows_per_sm = total_rows / num_sms
        avg_rows_per_batch = total_q / batch_size
        avg_rows_per_cta = min(avg_rows_per_batch, avg_rows_per_sm)
        # Heuristic: if avg_rows_per_sm >= 128, we are likely in prefill phase.
        # This is a rough heuristic and may not be accurate for all scenarios.
        if avg_rows_per_cta > 64:
            varlen_fwd_config_str = "mha_block_128"
        elif avg_rows_per_cta > 32:
            varlen_fwd_config_str = "mha_block_64"
        elif avg_rows_per_cta > 16:
            varlen_fwd_config_str = "mha_block_32"
        else:
            varlen_fwd_config_str = "mha_block_16"
        if flag_gems.vendor_name == "mthreads":
            varlen_fwd_config_str = "mha_block_32"

        cfg = runtime.get_heuristic_config(varlen_fwd_config_str)
        cfg_params = {
            "BLOCK_M": cfg["BLOCK_M"](args),
            "BLOCK_N": cfg["BLOCK_N"](args),
            "BLOCK_K": triton.next_power_of_2(head_size),
            # QK still uses the full BLOCK_K; PV and output use BLOCK_D=64
            # for D128 split-D.
            "BLOCK_D": 64 if use_varlen_split_d else triton.next_power_of_2(head_size),
            "SPLIT_D": use_varlen_split_d,
            "num_warps": cfg["num_warps"](args),
            "num_stages": 1 if not is_paged else cfg["num_stages"](args),
        }

        logger.debug("Running flash_varlen_fwd_kernel with config: %s", cfg_params)
        kernel(*args, **cfg_params)

        if seqlenq_ngroups_swapped:
            out = out.reshape(
                batch_size, max_seqlen_q, num_heads_k, head_size
            ).transpose(1, 2)
            if out_ is not None:
                out_.view(batch_size, num_heads_k, max_seqlen_q, head_size).copy_(out)
                out = out_
            else:
                out = out.reshape(batch_size, num_heads_k * max_seqlen_q, head_size)
            lse = lse.reshape(num_heads_k, batch_size, max_seqlen_q)
            lse = lse.reshape(num_heads_k * max_seqlen_q, batch_size)

        unused = torch.empty((), dtype=torch.int64, device=q_device)
    return out, q, k, v, lse, philox_args, unused, p


def mha_varlan_fwd_opt(
    q,
    k,
    v,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    fp8_p_max=448.0,
):
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_device = q.device
    q_dtype = q.dtype
    v_dtype = v.dtype
    # W8A8 FA3 change: optimized varlen path uses FP8 Q/K/V and per-block
    # descales just like FA3 FP8 forward.
    assert q_dtype in _FP8_DTYPES, "W8A8 FA3 FlashAttention expects q to be fp8"
    assert (
        k.dtype == q_dtype
    ), "W8A8 FA3 FlashAttention expects q and k to use the same fp8 dtype"
    assert v_dtype == q_dtype, "W8A8 FA3 FlashAttention expects v to be fp8"
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"

    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_q.is_contiguous()

    assert cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_k.is_contiguous()

    is_paged = page_table is not None
    if not is_paged:
        page_table = torch.empty((0, 0), device=q_device, dtype=torch.int32)

    # q shape: [total_q_tokens, num_heads, head_size]
    # k shape:
    #   paged_kv: [num_pages, block_size, num_heads_k, head_size]
    # batch_size, number of sentences
    total_q, num_heads, head_size = q.size()
    num_heads_k = k.size(2) if is_paged else k.size(1)
    batch_size = cu_seqlens_q.numel() - 1
    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    k_batch_size = num_pages
    # max_num_pages_per_seq = page_table.size(1)
    page_table_batch_stride = page_table.stride(0)
    k_batch_stride = k.stride(0)
    v_batch_stride = v.stride(0)

    assert k.size() == v.size()
    assert cu_seqlens_q.size() == (batch_size + 1,)
    assert cu_seqlens_k.size() == (batch_size + 1,)

    # Check output shape
    if out is not None:
        assert out.stride(-1) == 1
        # W8A8 FA3 change: optimized varlen output remains high precision.
        assert out.dtype in _HIGH_PRECISION_DTYPES
        assert out.size() == (total_q, num_heads, head_size)

    if seqused_k is not None:
        assert seqused_k.is_contiguous()
        assert seqused_k.size() == (batch_size,)

    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False

    if is_causal:
        window_size_right = 0

    # check disable swa
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_k:
        window_size_right = -1

    is_local = window_size_left >= 0

    # Optimize all single-query sequences by swapping the query-group and sequence dimensions
    seqlenq_ngroups_swapped = (
        max_seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
    )
    q_groups = num_heads // num_heads_k
    if seqlenq_ngroups_swapped:
        logger.debug("Swapping query groups and sequence dimensions")
        q = (
            q.reshape((batch_size, num_heads_k, q_groups, head_size))
            .transpose(1, 2)
            .reshape(batch_size * q_groups, num_heads_k, head_size)
        )
        max_seqlen_q = q_groups
        num_heads = num_heads_k
        cu_seqlens_q = None
        q_batch_stride = q.stride(0) * max_seqlen_q
        k_batch_stride = k.stride(0)
        v_batch_stride = v.stride(0)
        # o_batch_stride = out.stride(0) * max_seqlen_q
    else:
        q_batch_stride = 0
        k_batch_stride = 0
        v_batch_stride = 0
        o_batch_stride = 0

    total_q = q.size(0)

    assert leftpad_k is None, "leftpad_k is not supported."
    assert (
        head_size <= 256
    ), "FlashAttention forward only supports head dimension at most 256"
    assert (
        head_size % 8 == 0
    ), "head_size must be a multiple of 8, this is ensured by padding!"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"

    assert q.shape == (total_q, num_heads, head_size)
    if is_paged:
        assert k.shape == (num_pages, block_size, num_heads_k, head_size)
        assert v.shape == (num_pages, block_size, num_heads_k, head_size)
    assert k.stride() == v.stride()

    if softcap > 0.0:
        assert p_dropout == 0, "dropout is not supported if softcap is used."

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32) if head_size <= 192 else 256
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 32)

    M_LOG2E = 1.4426950408889634074
    if softcap > 0.0:
        is_softcap = True
        adjusted_scale_softmax = softcap
        adjusted_softcap = softmax_scale / softcap
        adjusted_scale_softmax_log2e = softcap * M_LOG2E
    else:
        is_softcap = False
        adjusted_softcap = 0.0
        adjusted_scale_softmax = softmax_scale
        adjusted_scale_softmax_log2e = softmax_scale * M_LOG2E

    # Set alibi params
    if alibi_slopes is not None:
        assert alibi_slopes.device == q_device
        assert alibi_slopes.dtype in (torch.float,)
        assert alibi_slopes.stride(-1) == 1
        assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
            batch_size,
            num_heads,
        )
        alibi_slopes_batch_stride = (
            alibi_slopes.stride(0) if alibi_slopes.ndim == 2 else 0
        )
        is_alibi = True
    else:
        alibi_slopes_batch_stride = 0
        is_alibi = False

    # Prepare params to kernel
    with torch_device_fn.device(q_device):
        if out is not None:
            out_ = out
            if seqlenq_ngroups_swapped:
                # W8A8 FA3 change: temporary output is high precision; V is FP8.
                out = torch.empty_like(q, dtype=torch.bfloat16)
        else:
            out_ = None
            # W8A8 FA3 change: optimized varlen creates high-precision output
            # even when the caller passes FP8 Q/K/V.
            out = torch.empty_like(q, dtype=torch.bfloat16)

        if seqlenq_ngroups_swapped:
            o_batch_stride = out.stride(0) * max_seqlen_q

        if lse is None:
            lse = torch.empty((num_heads, total_q), dtype=torch.float, device=q_device)

        if p_dropout > 0:
            is_dropout = True
            increment = batch_size * num_heads * 32
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            philox_args = torch.tensor(
                [philox_seed, philox_offset], dtype=torch.int64, device=q_device
            )
        else:
            is_dropout = False
            # philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)
            philox_args = None

        p_dropout = 1 - p_dropout
        p_dropout_in_uint8_t = math.floor(p_dropout * 255.0)
        rp_dropout = 1.0 / p_dropout

        if return_softmax:
            assert is_dropout, "Only supported with non-zero dropout."
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                device=q_device,
            )
        else:
            # p = torch.empty((), device=q_device)
            p = None
        if zero_tensors:
            out.zero_()
            lse.fill_(float("-inf"))

        q_nblocks = triton.cdiv(max_seqlen_q, 128)
        k_nblocks = triton.cdiv(max_seqlen_k, 128)
        # W8A8 FA3 change: normalize optional FA3 descales for the optimized
        # varlen path before packing the common fwd_params object.
        q_descale = _normalize_dense_descale(
            q_descale, batch_size, num_heads, q_nblocks, q_device, "q_descale"
        )
        k_descale = _normalize_dense_descale(
            k_descale, batch_size, num_heads_k, k_nblocks, q_device, "k_descale"
        )
        v_descale = _normalize_dense_descale(
            v_descale, batch_size, num_heads_k, k_nblocks, q_device, "v_descale"
        )

        params = fwd_params(
            q,  # q_ptr,
            k,  # k_ptr,
            v,  # v_ptr,
            out,  # o_ptr,
            p,  # p_ptr,
            lse,  # softmax_lse_ptr,
            q.stride(-3),  # q_row_stride,
            k.stride(-3),  # k_row_stride,
            v.stride(-3),  # v_row_stride,
            q.stride(-2),  # q_head_stride,
            k.stride(-2),  # k_head_stride,
            v.stride(-2),  # v_head_stride,
            out.stride(-3),  # o_row_stride,
            out.stride(-2),  # o_head_stride,
            q_batch_stride,  # q_batch_stride,
            k_batch_stride,  # k_batch_stride,
            v_batch_stride,  # v_batch_stride,
            o_batch_stride,  # o_batch_stride,
            cu_seqlens_q is not None,  # is_cu_seqlens_q,
            cu_seqlens_q,  # cu_seqlens_q_ptr,
            cu_seqlens_k is not None,  # is_cu_seqlens_k,
            cu_seqlens_k,  # cu_seqlens_k_ptr,
            seqused_k is not None,  # is_seqused_k,
            seqused_k,  # seqused_k_ptr,
            # sizes
            batch_size,  # b,
            k_batch_size,  # bk,
            num_heads,  # h,
            num_heads_k,  # hk,
            num_heads // num_heads_k,  # h_hk_ratio,
            max_seqlen_q,  # seqlen_q,
            max_seqlen_k,  # seqlen_k,
            seqlen_q_rounded,  # seqlen_q_rounded,
            seqlen_k_rounded,  # seqlen_k_rounded,
            head_size,  # d,
            head_size_rounded,  # d_rounded,
            # scaling factors
            is_softcap,
            adjusted_softcap,  # softcap,
            adjusted_scale_softmax,  # scale_softmax,
            adjusted_scale_softmax_log2e,  # scale_softmax_log2,
            q_descale,  # q_descale_ptr,
            k_descale,  # k_descale_ptr,
            v_descale,  # v_descale_ptr,
            q_descale.stride(0),  # q_descale_batch_stride,
            q_descale.stride(1),  # q_descale_head_stride,
            q_descale.stride(2),  # q_descale_block_stride,
            k_descale.stride(0),  # k_descale_batch_stride,
            k_descale.stride(1),  # k_descale_head_stride,
            k_descale.stride(2),  # k_descale_block_stride,
            v_descale.stride(0),  # v_descale_batch_stride,
            v_descale.stride(1),  # v_descale_head_stride,
            v_descale.stride(2),  # v_descale_block_stride,
            True,  # use_fa3_fp8_scales,
            fp8_p_max,  # fp8_p_max,
            # dropout
            is_dropout,
            p_dropout,
            rp_dropout,
            p_dropout_in_uint8_t,
            philox_args,
            return_softmax,
            # causal and swa
            is_causal,  # is_causal,
            is_local,  # is_local,
            window_size_left,  # window_size_left,
            window_size_right,  # window_size_right,
            seqlenq_ngroups_swapped,  # seqlenq_ngroups_swapped,
            is_paged,
            # alibi
            is_alibi,  #
            alibi_slopes,  # alibi_slopes_ptr,
            alibi_slopes_batch_stride,  # alibi_slopes_batch_stride,
            # block table params
            total_q,  # total_q,
            page_table,  # page_table_ptr,
            page_table_batch_stride,  # page_table_batch_stride,
            block_size,  # block_size,
        )

        if flag_gems.vendor_name == "iluvatar":
            params.k_ptr = k.view(k.shape[0], k.shape[1], -1)
            params.v_ptr = v.view(v.shape[0], v.shape[1], -1)
        logger.debug("kernel: flash_varlen_fwd")
        # Enable two-way split-D for non-paged D128 varlen attention. The
        # paged-cache loader continues to load the full D dimension.
        use_varlen_split_d = head_size == 128 and not is_paged
        grid = lambda args: (
            triton.cdiv(max_seqlen_q, args["BLOCK_M"]),
            batch_size,
            num_heads * (2 if use_varlen_split_d else 1),
        )
        kernel = flash_varlen_fwd_kernel[grid]
        args = tuple(getattr(params, k) for k in params.__slots__)

        # We assess which phase the requests are likely to be in and set the config accordingly.
        total_rows = total_q * num_heads
        num_sms = torch_device_fn.get_device_properties(
            flag_gems.device
        ).multi_processor_count
        avg_rows_per_sm = total_rows / num_sms
        avg_rows_per_batch = total_q / batch_size
        avg_rows_per_cta = min(avg_rows_per_batch, avg_rows_per_sm)
        # Heuristic: if avg_rows_per_sm >= 128, we are likely in prefill phase.
        # This is a rough heuristic and may not be accurate for all scenarios.
        if avg_rows_per_cta > 64:
            varlen_fwd_config_str = "mha_block_128"
        elif avg_rows_per_cta > 32:
            varlen_fwd_config_str = "mha_block_64"
        elif avg_rows_per_cta > 16:
            varlen_fwd_config_str = "mha_block_32"
        else:
            varlen_fwd_config_str = "mha_block_16"
        if flag_gems.vendor_name == "mthreads":
            varlen_fwd_config_str = "mha_block_32"

        cfg = runtime.get_heuristic_config(varlen_fwd_config_str)
        cfg_params = {
            "BLOCK_M": cfg["BLOCK_M"](args),
            "BLOCK_N": cfg["BLOCK_N"](args),
            "BLOCK_K": triton.next_power_of_2(head_size),
            # QK still uses the full BLOCK_K; PV and output use BLOCK_D=64
            # for D128 split-D.
            "BLOCK_D": 64 if use_varlen_split_d else triton.next_power_of_2(head_size),
            "SPLIT_D": use_varlen_split_d,
            "num_warps": cfg["num_warps"](args),
            "num_stages": 1 if not is_paged else cfg["num_stages"](args),
        }

        logger.debug("Running flash_varlen_fwd_kernel with config: %s", cfg_params)
        kernel(*args, **cfg_params)

        if seqlenq_ngroups_swapped:
            out = out.reshape(
                batch_size, max_seqlen_q, num_heads_k, head_size
            ).transpose(1, 2)
            if out_ is not None:
                out_.view(batch_size, num_heads_k, max_seqlen_q, head_size).copy_(out)
                out = out_
            else:
                out = out.reshape(batch_size, num_heads_k * max_seqlen_q, head_size)
            lse = lse.reshape(num_heads_k, batch_size, max_seqlen_q)
            lse = lse.reshape(num_heads_k * max_seqlen_q, batch_size)

        # unused = torch.empty((), dtype=torch.int64, device=q_device)
        unused = None
    return out, q, k, v, lse, philox_args, unused, p


def mha_fwd(
    q,
    k,
    v,
    out,
    alibi_slopes,
    p_dropout,
    softmax_scale,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    disable_splitkv=False,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    fp8_p_max=448.0,
):
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_dtype = q.dtype
    v_dtype = v.dtype
    q_device = q.device
    # W8A8 FA3 change: dense forward now follows FA3 FP8 mode: Q/K/V are FP8
    # tensors and the output keeps the caller's high-precision output dtype.
    assert q_dtype in _FP8_DTYPES, "W8A8 FA3 FlashAttention expects q to be fp8"
    assert (
        k.dtype == q_dtype
    ), "W8A8 FA3 FlashAttention expects q and k to use the same fp8 dtype"
    assert v_dtype == q_dtype, "W8A8 FA3 FlashAttention expects v to be fp8"
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    batch_size, seqlen_q, num_heads, head_size = q.size()
    _, seqlen_k, num_heads_k, _ = k.size()

    # Check output shape
    if out is not None:
        assert out.stride(-1) == 1
        # W8A8 FA3 change: output is explicitly high precision, matching FA3
        # forward instead of inheriting the FP8 V dtype.
        assert out.dtype in _HIGH_PRECISION_DTYPES
        assert out.size() == (batch_size, seqlen_q, num_heads, head_size)
        CHECK_DEVICE(out)

    assert (
        head_size % 8 == 0
    ), "head_size must be a multiple of 8, this is ensured by padding!"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"
    if window_size_left >= seqlen_k:
        window_size_left = -1
    if window_size_right >= seqlen_k:
        window_size_right = -1
    if seqlen_q == 1 and alibi_slopes is None:
        is_causal = False
    if is_causal:
        window_size_right = 0

    is_causal = window_size_left < 0 and window_size_right == 0
    is_local = window_size_left >= 0 and window_size_right >= 0

    seqlenq_ngroups_swapped = (
        seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
    )
    q_groups = num_heads // num_heads_k

    if seqlenq_ngroups_swapped:
        logger.debug("q_kg swapped.")
        q = q.reshape(batch_size, num_heads_k, q_groups, head_size).transpose(1, 2)
        seqlen_q = q_groups
        num_heads = num_heads_k

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32)
    seqlen_q_rounded = round_multiple(seqlen_q, 128)
    seqlen_k_rounded = round_multiple(seqlen_k, 32)

    assert (
        head_size <= 256
    ), "FlashAttention forward only supports head dimension at most 256"
    assert head_size == head_size_rounded, "head_size must be rounded to 32"

    def splits_heuristic(num_tasks, num_sms, n_blocks):
        # splits when wave efficiency is low
        n_waves = triton.cdiv(num_tasks, num_sms)
        eff = (num_tasks / num_sms) / n_waves
        if eff > 0.8 or n_waves > 1:
            return 1

        min_blocks_per_split = 2
        best_splits = min(
            triton.cdiv(n_blocks, min_blocks_per_split),
            int(math.floor(1.0 / eff)),
            num_sms,
        )

        return best_splits

    with torch_device_fn.device(q_device):
        # Set softmax params
        lse = torch.empty(
            (batch_size, num_heads, seqlen_q), dtype=torch.float, device=q_device
        )

        if out is not None:
            if seqlenq_ngroups_swapped:
                out = out.reshape(
                    batch_size, num_heads_k, q_groups, head_size
                ).transpose(1, 2)
        else:
            # W8A8 FA3 change: allocate a high-precision output even though all
            # three inputs are FP8.
            out = torch.empty_like(q, dtype=torch.bfloat16)

        # Set dropout params
        if p_dropout > 0:
            is_dropout = True
            increment = batch_size * num_heads * 32
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            philox_args = torch.tensor(
                [philox_seed, philox_offset], dtype=torch.int64, device=q_device
            )
        else:
            is_dropout = False
            philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)

        p_dropout = 1 - p_dropout
        p_dropout_in_uint8_t = math.floor(p_dropout * 255.0)
        rp_dropout = 1.0 / p_dropout

        if return_softmax:
            assert is_dropout, "Only supported with non-zero dropout."
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                device=q_device,
            )
        else:
            p = torch.empty((), device=q_device)

        q_nblocks = triton.cdiv(seqlen_q, 128)
        k_nblocks = triton.cdiv(seqlen_k, 128)
        # W8A8 FA3 change: per-block descale tensors are part of the kernel
        # contract.  Benchmarks pass real block scales; default 1.0 keeps manual
        # FP8 casts runnable while making the scale path visible.
        q_descale = _normalize_dense_descale(
            q_descale, batch_size, num_heads, q_nblocks, q_device, "q_descale"
        )
        k_descale = _normalize_dense_descale(
            k_descale, batch_size, num_heads_k, k_nblocks, q_device, "k_descale"
        )
        v_descale = _normalize_dense_descale(
            v_descale, batch_size, num_heads_k, k_nblocks, q_device, "v_descale"
        )

        M_LOG2E = 1.4426950408889634074
        if softcap > 0.0:
            is_softcap = True
            adjusted_scale_softmax = softcap
            adjusted_softcap = softmax_scale / softcap
            adjusted_scale_softmax_log2e = softcap * M_LOG2E
        else:
            is_softcap = False
            adjusted_softcap = 0.0
            adjusted_scale_softmax = softmax_scale
            adjusted_scale_softmax_log2e = softmax_scale * M_LOG2E

        # Set alibi params
        if alibi_slopes is not None:
            assert alibi_slopes.device == q_device
            assert alibi_slopes.dtype in (torch.float,)
            assert alibi_slopes.stride(-1) == 1
            assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
                batch_size,
                num_heads,
            )
            alibi_slopes_batch_stride = (
                alibi_slopes.stride(0) if alibi_slopes.ndim == 2 else 0
            )
            is_alibi = True
        else:
            alibi_slopes_batch_stride = 0
            is_alibi = False

        # ONLY EVEN_K IS SUPPORTED
        assert head_size == head_size_rounded

        # Do kernel dispatching
        def dispatch(B, H, Q, K, D, params):
            num_sms = torch_device_fn.get_device_properties(
                "cuda"
            ).multi_processor_count
            # For D128, use the split-D dense kernel and partition the output
            # dimension across two D64 CTAs to reduce register pressure in
            # flash_fwd_kernel. The split-KV path does not use split-D.
            # use_split_d = D == 128
            use_split_d = D == 128
            # For short sequences, split-KV combine and temporary-tensor overhead
            # usually outweigh the benefit, so S <= 512 uses the dense path.
            disable_splitkv1 = disable_splitkv or seqlen_q <= 512

            # Try bh parallel
            # if B * H > 0.8 * num_sms:
            #     kernel = flash_fwd_bh_parallel_kernel[(H, B)]
            #     # Yield kernel and prefilled args
            #     return kernel, default_args, None, None

            # Try splitkv
            if (
                (not use_split_d)
                and not is_dropout
                and not is_local
                and not disable_splitkv1
            ):
                BM = block_m_splitkv_heuristic(D)
                n_tasks = B * H * triton.cdiv(seqlen_q, BM)
                BN = block_n_splitkv_heuristic(D)
                n_blocks = triton.cdiv(seqlen_k, BN)
                n_splits = splits_heuristic(n_tasks, num_sms, n_blocks)

                if n_splits > 1:
                    logger.debug("kernel: flash_fwd_splitkv")
                    lse_splits = torch.empty(
                        (n_splits, B, H, Q), dtype=torch.float, device=q_device
                    )
                    out_splits = torch.empty(
                        (n_splits, B, H, Q, D), dtype=torch.float, device=q_device
                    )
                    grid = lambda args: (
                        triton.cdiv(Q, args["BLOCK_M"]),
                        n_splits,
                        B * H,
                    )
                    splitkv_kernel = flash_fwd_splitkv_kernel[grid]
                    params.o_ptr = out_splits
                    params.softmax_lse_ptr = lse_splits
                    extra_args = {"blocks_per_split": triton.cdiv(n_blocks, n_splits)}
                    kernel = splitkv_kernel(*params.args(), **extra_args)

                    if D >= 128:
                        BLOCK_M = 4
                    elif D >= 64:
                        BLOCK_M = 8
                    else:
                        BLOCK_M = 16
                    BLOCK_K = triton.next_power_of_2(D)
                    grid = lambda args: (triton.cdiv(B * H * Q, BLOCK_M),)
                    combine_kernel = flash_fwd_splitkv_combine_kernel[grid]
                    combine_args = {
                        "out_ptr": out,
                        "lse_ptr": lse,
                        "head_size": head_size,
                        "out_split_stride": out_splits.stride(0),
                        "lse_split_stride": lse_splits.stride(0),
                        "out_b_stride": out.stride(0),
                        "out_s_stride": out.stride(-3),
                        "out_h_stride": out.stride(-1),
                        "out_splits_ptr": out_splits,
                        "lse_splits_ptr": lse_splits,
                        "n_splits": n_splits,
                        "BLOCK_M": BLOCK_M,
                        "BLOCK_K": BLOCK_K,
                        "q_total": B * H * Q,
                        "MAX_N_SPLITS": triton.next_power_of_2(n_splits),
                    }
                    combine_kernel(**combine_args)
                    return kernel

            # Last option: flash_fwd
            logger.debug("kernel: flash_fwd")
            grid = lambda args: (
                triton.cdiv(Q, args["BLOCK_M"]),
                H * B,
                2 if use_split_d else 1,
            )
            kernel = flash_fwd_kernel[grid]
            # For D128 split-D, QK still uses the full BLOCK_K with head_dim=128.
            # BLOCK_D=64 applies only to PV and output stores; D64 or non-split-D
            # retains the original BLOCK_K.
            extra_args = {
                "BLOCK_D": 64 if use_split_d else triton.next_power_of_2(D),
                "SPLIT_D": use_split_d,
            }
            kernel = kernel(*params.args(), **extra_args)
            return kernel

        if _debug:
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                dtype=torch.float32,
                device=q_device,
            )
            return_softmax = True

        params = fwd_params(
            q,  # q_ptr,
            k,  # k_ptr,
            v,  # v_ptr,
            out,  # o_ptr,
            p,  # p_ptr,
            lse,  # softmax_lse_ptr,
            q.stride(-3),  # q_row_stride,
            k.stride(-3),  # k_row_stride,
            v.stride(-3),  # v_row_stride,
            q.stride(-2),  # q_head_stride,
            k.stride(-2),  # k_head_stride,
            v.stride(-2),  # v_head_stride,
            out.stride(-3),  # o_row_stride,
            out.stride(-2),  # o_head_stride,
            q.stride(0),  # q_batch_stride,
            k.stride(0),  # k_batch_stride,
            v.stride(0),  # v_batch_stride,
            out.stride(0),  # o_batch_stride,
            False,  # is_cu_seqlens_q,
            None,  # cu_seqlens_q_ptr,
            False,  # is_cu_seqlens_k,
            None,  # cu_seqlens_k_ptr,
            False,  # is_seqused_k,
            None,  # seqused_k_ptr,
            # sizes
            batch_size,  # b,
            0,  # bk,
            num_heads,  # h,
            num_heads_k,  # hk,
            num_heads // num_heads_k,  # h_hk_ratio,
            seqlen_q,  # seqlen_q,
            seqlen_k,  # seqlen_k,
            seqlen_q_rounded,  # seqlen_q_rounded,
            seqlen_k_rounded,  # seqlen_k_rounded,
            head_size,  # d,
            head_size_rounded,  # d_rounded,
            # scaling factors
            is_softcap,
            adjusted_softcap,  # softcap,
            adjusted_scale_softmax,  # scale_softmax,
            adjusted_scale_softmax_log2e,  # scale_softmax_log2,
            # W8A8 FA3 change: pass Q/K/V per-block descale metadata to every
            # compute kernel so QK and PV can both use FP8 tensor cores.
            q_descale,  # q_descale_ptr,
            k_descale,  # k_descale_ptr,
            v_descale,  # v_descale_ptr,
            q_descale.stride(0),  # q_descale_batch_stride,
            q_descale.stride(1),  # q_descale_head_stride,
            q_descale.stride(2),  # q_descale_block_stride,
            k_descale.stride(0),  # k_descale_batch_stride,
            k_descale.stride(1),  # k_descale_head_stride,
            k_descale.stride(2),  # k_descale_block_stride,
            v_descale.stride(0),  # v_descale_batch_stride,
            v_descale.stride(1),  # v_descale_head_stride,
            v_descale.stride(2),  # v_descale_block_stride,
            True,  # use_fa3_fp8_scales,
            fp8_p_max,  # fp8_p_max,
            # dropout
            is_dropout,
            p_dropout,
            rp_dropout,
            p_dropout_in_uint8_t,
            philox_args,
            return_softmax,
            # causal and swa
            is_causal,  # is_causal,
            is_local,  # is_local,
            window_size_left,  # window_size_left,
            window_size_right,  # window_size_right,
            seqlenq_ngroups_swapped,  # seqlenq_ngroups_swapped,
            False,  # is_paged,
            # alibi
            is_alibi,  #
            alibi_slopes,  # alibi_slopes_ptr,
            alibi_slopes_batch_stride,  # alibi_slopes_batch_stride,
            # block table params
            0,  # total_q,
            None,  # page_table_ptr,
            0,  # page_table_batch_stride,
            0,  # block_size,
        )

        # Move TxD to last dims for correct stride in Triton tt.load
        if flag_gems.vendor_name == "iluvatar":
            params.q_ptr = q.transpose(1, 2)
            params.k_ptr = k.transpose(1, 2)
            params.v_ptr = v.transpose(1, 2)
        kernel = dispatch(batch_size, num_heads, seqlen_q, seqlen_k, head_size, params)

        if _debug:
            print(f"{kernel.name} shared memory:", kernel.metadata.shared)
            print(f"{kernel.name} num_warps:", kernel.metadata.num_warps)
            print(f"{kernel.name} num_stages:", kernel.metadata.num_stages)
            # print(kernel.asm['ttgir'])

        if seqlenq_ngroups_swapped:
            out = out.transpose(1, 2).reshape(
                (batch_size, 1, num_heads_k * seqlen_q, head_size)
            )
            q = q.transpose(1, 2).reshape(
                (batch_size, 1, num_heads_k * seqlen_q, head_size)
            )
            lse = lse.reshape((batch_size, num_heads_k * seqlen_q, 1))

        unused = torch.empty((), dtype=torch.int64, device=q_device)

    return out, q, k, v, lse, philox_args, unused, p


def flash_attn_varlen_func_w8a8_fp8(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # Dummy FA3 arguments
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    """Compute variable-length FlashAttention-2 with block-wise FP8 Q/K/V.

    Args:
        q: Packed FP8 query tensor in ``[total_q, heads, head_dim]`` layout.
        k: Packed FP8 key tensor in ``[total_k, heads, head_dim]`` layout.
        v: Packed FP8 value tensor with the same shape as ``k``.
        max_seqlen_q: Maximum query sequence length in the batch.
        cu_seqlens_q: Cumulative query sequence lengths with shape ``[batch + 1]``.
        max_seqlen_k: Maximum key sequence length in the batch.
        cu_seqlens_k: Cumulative key sequence lengths with shape ``[batch + 1]``.
        q_descale: Query descales normalized to ``[batch, heads, q_blocks]``.
        k_descale: Key descales normalized to ``[batch, heads, kv_blocks]``.
        v_descale: Value descales normalized to ``[batch, heads, kv_blocks]``.
        softmax_scale: Score scale. Defaults to ``1 / sqrt(head_dim)``.
        causal: Whether to apply a causal mask.
        out: Optional packed FP16/BF16 output tensor.

    The public signature matches ``flash_attn_varlen_func``. Descales are
    applied per logical 128-token block. The returned tensor uses ``out.dtype``
    when supplied and BF16 otherwise. This W8A8 path currently requires Q, K,
    and V to have the same number of heads; MQA and GQA are not supported.
    """
    if fa_version != 2:
        raise RuntimeError("Only FA2 is implemented.")
    if num_splits > 0:
        raise RuntimeError("num_splits > 0 is not implemented in GEMS.")
    assert (
        cu_seqlens_k is not None or seqused_k is not None
    ), "cu_seqlens_k or seqused_k must be provided"
    assert (
        cu_seqlens_k is None or seqused_k is None
    ), "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert (
        block_table is None or seqused_k is not None
    ), "seqused_k must be provided if block_table is provided"

    num_heads_k = k.shape[2] if block_table is not None else k.shape[1]
    if q.shape[1] != num_heads_k:
        raise NotImplementedError("GQA is not supported by this W8A8 path")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])

    q, k, v = [x.contiguous() if x.stride(-1) != 1 else x for x in (q, k, v)]
    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)
    max_seqlen_q = (
        max_seqlen_q.item() if hasattr(max_seqlen_q, "item") else max_seqlen_q
    )
    max_seqlen_k = (
        max_seqlen_k.item() if hasattr(max_seqlen_k, "item") else max_seqlen_k
    )

    result = mha_varlan_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
        seqused_k,
        None,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        real_window_size[0],
        real_window_size[1],
        softcap,
        return_softmax_lse and dropout_p > 0,
        None,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        fp8_p_max=float(torch.finfo(q.dtype).max),
    )
    return (result[0], result[4]) if return_softmax_lse else result[0]
