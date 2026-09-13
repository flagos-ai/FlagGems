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

"""Direct one-pass FA3 TLE kernel family."""

import os

import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

from .common import (
    _apply_alibi_v3,
    _apply_mask_v3,
    _apply_softcap_v3,
    _load_paged_kv_tma_tile,
    _merge_attention_sink,
    _paged_blockwise_cache_indices,
    _ragged_persistent_tile_coords,
    _softmax_online_deferred,
)
from .scheduling import CommonSchedulingHeuristics, DirectSchedulingHeuristics

_prune_fa3_direct_configs = DirectSchedulingHeuristics.prune_autotune_configs
_heur_block_k = CommonSchedulingHeuristics.block_k


def _fa3_direct_configs():
    if any(
        name.startswith("FLAG_GEMS_FA3_TLE_EXPERIMENT_DIRECT_") for name in os.environ
    ):
        return DirectSchedulingHeuristics.autotune_configs()
    configs = runtime.get_tuned_config("flash_attn_varlen_fa3_direct")
    return configs or DirectSchedulingHeuristics.autotune_configs()


@libentry()
@libtuner(
    configs=_fa3_direct_configs(),
    prune_configs_by={"early_config_prune": _prune_fa3_direct_configs},
    warmup=10,
    rep=20,
    key=[
        "b",
        "h",
        "hk",
        "d",
        "block_size",
        "is_paged",
        "is_causal",
        "is_local",
        "is_alibi",
        "is_s_aux",
        "h_hk_ratio",
        "seqlen_q",
        "seqlen_k",
        "total_q",
        "DIRECT_SHAPE_BUCKET",
        "PACK_GQA",
        "BATCH_FIRST_GRID",
        "SINGLE_KV_TILE",
        "STORE_LSE",
        "RAGGED_SCHEDULER",
        "PAGED_GATHER_MODE",
        "PAGED_KV_NON_TMA",
    ],
)
@triton.heuristics(
    values={
        "BLOCK_K": _heur_block_k,
    }
)
@triton.jit(
    do_not_specialize=[
        "k_batch_stride",
        "b",
        "bk",
        "seqlen_q",
        "seqlen_k",
        "total_q",
    ]
)
def flash_varlen_fwd_v3_tle_direct_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    k_batch_stride,
    cu_seqlens_q_ptr,
    is_seqused_k: tl.constexpr,
    cu_seqlens_k_ptr,
    seqused_k_ptr,
    b,
    bk,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    d: tl.constexpr,
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    is_paged: tl.constexpr,
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    is_s_aux: tl.constexpr,
    s_aux_ptr,
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIRECT_SHAPE_BUCKET: tl.constexpr,
    PACK_GQA: tl.constexpr,
    BATCH_FIRST_GRID: tl.constexpr,
    SINGLE_KV_TILE: tl.constexpr,
    STORE_LSE: tl.constexpr,
    PAGED_KV_NON_TMA: tl.constexpr,
    RAGGED_SCHEDULER: tl.constexpr,
    HEADS_IN_L2: tl.constexpr,
    PAGED_GATHER_MODE: tl.constexpr = 2,
):
    if RAGGED_SCHEDULER:
        pack_factor: tl.constexpr = h_hk_ratio if PACK_GQA else 1
        effective_heads: tl.constexpr = hk if PACK_GQA else h
        m_block, bid, head_pid, work_valid = _ragged_persistent_tile_coords(
            tl.program_id(0),
            cu_seqlens_q_ptr,
            b,
            effective_heads,
            BLOCK_M,
            pack_factor,
            HEADS_IN_L2,
        )
    elif BATCH_FIRST_GRID:
        bid = tl.program_id(0)
        m_block = tl.program_id(1)
        head_pid = tl.program_id(2)
        work_valid = True
    else:
        m_block = tl.program_id(0)
        bid = tl.program_id(1)
        head_pid = tl.program_id(2)
        work_valid = True
    HEAD_DIM_PADDED: tl.constexpr = BLOCK_K
    PAGED_KV_NON_TMA_EFFECTIVE: tl.constexpr = PAGED_KV_NON_TMA or (
        block_size % BLOCK_N != 0
    )
    page_stride_rows = k_batch_stride // k_row_stride

    q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
    q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
    q_len = q_eos - q_bos
    q_offset = q_bos * q_row_stride
    o_offset = q_bos * o_row_stride
    lse_offset = q_bos

    if is_seqused_k:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
        k_bos = 0
    else:
        k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
        k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
        k_len = k_eos - k_bos

    packed_row = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    if PACK_GQA:
        row_idx_q = packed_row // h_hk_ratio
        query_head = head_pid * h_hk_ratio + packed_row % h_hk_ratio
        kv_head = head_pid
        effective_q_len = q_len * h_hk_ratio
        q_block_start = (m_block * BLOCK_M) // h_hk_ratio
        q_block_end = tl.cdiv((m_block + 1) * BLOCK_M, h_hk_ratio)
    else:
        row_idx_q = packed_row
        query_head = head_pid
        kv_head = head_pid // h_hk_ratio
        effective_q_len = q_len
        q_block_start = m_block * BLOCK_M
        q_block_end = (m_block + 1) * BLOCK_M

    process_q = (m_block * BLOCK_M < effective_q_len) & work_valid
    if process_q:
        if SINGLE_KV_TILE:
            n_block_min = 0
        elif is_local:
            n_block_min = tl.maximum(
                0,
                (q_block_start + k_len - q_len - window_size_left) // BLOCK_N,
            )
        else:
            n_block_min = 0

        if SINGLE_KV_TILE:
            n_block_max = 1
        else:
            n_block_max = tl.cdiv(k_len, BLOCK_N)
        if (is_causal or is_local) and not SINGLE_KV_TILE:
            n_block_max = tl.minimum(
                n_block_max,
                tl.cdiv(
                    q_block_end + k_len - q_len + window_size_right,
                    BLOCK_N,
                ),
            )

        if is_alibi:
            if PACK_GQA:
                alibi_slope = tl.load(
                    alibi_slopes_ptr + bid * alibi_slopes_batch_stride + query_head
                )[:, None]
            else:
                alibi_slope = tl.load(
                    alibi_slopes_ptr + bid * alibi_slopes_batch_stride + query_head
                )
            alibi_slope = alibi_slope / scale_softmax
        else:
            alibi_slope = 0.0

        if is_paged:
            page_table_ptr_b = page_table_ptr + bid * page_table_batch_stride
            k_base = k_ptr + kv_head * k_head_stride
            v_base = v_ptr + kv_head * v_head_stride
            if not PAGED_KV_NON_TMA_EFFECTIVE:
                paged_k_desc = tl.make_tensor_descriptor(
                    base=k_base,
                    shape=[bk, block_size, d],
                    strides=[k_batch_stride, k_row_stride, 1],
                    block_shape=[1, BLOCK_N, HEAD_DIM_PADDED],
                )
                paged_v_desc = tl.make_tensor_descriptor(
                    base=v_base,
                    shape=[bk, block_size, d],
                    strides=[k_batch_stride, v_row_stride, 1],
                    block_shape=[1, BLOCK_N, HEAD_DIM_PADDED],
                )
        else:
            k_base = k_ptr + k_bos * k_row_stride + kv_head * k_head_stride
            v_base = v_ptr + k_bos * v_row_stride + kv_head * v_head_stride

        d_idx = tl.arange(0, HEAD_DIM_PADDED)
        if PACK_GQA:
            q_ptrs = (
                q_ptr
                + q_offset
                + row_idx_q[:, None] * q_row_stride
                + query_head[:, None] * q_head_stride
                + d_idx[None, :]
            )
        else:
            q_base = q_ptr + q_offset + query_head * q_head_stride
            q_ptrs = q_base + row_idx_q[:, None] * q_row_stride + d_idx[None, :]
        q_tile = tl.load(
            q_ptrs,
            mask=(packed_row[:, None] < effective_q_len) & (d_idx[None, :] < d),
            other=0.0,
        )
        rowmax = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        rowsum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

        if SINGLE_KV_TILE:
            n_masking_steps = 1
        elif is_causal and not is_local:
            # A KV block is mask-free only when its exclusive end is no
            # greater than the earliest semantic query row's causal bound.
            # With PackGQA and an unaligned k_len-q_len offset, both trailing
            # blocks can require row-wise masking even when one packed tile
            # spans fewer than BLOCK_N semantic query rows.
            causal_col_exclusive = q_block_start + k_len - q_len + 1
            causal_full_block_max = tl.maximum(
                n_block_min, causal_col_exclusive // BLOCK_N
            )
            n_masking_steps = n_block_max - causal_full_block_max
        elif is_causal or is_local:
            n_masking_steps = tl.cdiv(
                q_block_end - q_block_start + window_size_right + 1,
                BLOCK_N,
            )
        else:
            n_masking_steps = 1
        n_masking_steps = tl.maximum(
            0, tl.minimum(n_block_max - n_block_min, n_masking_steps)
        )

        n_block_start_mask = n_block_max - 1
        for step in tl.range(0, n_masking_steps):
            n_block = n_block_start_mask - step
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            kv_mask = col_idx < k_len
            if is_paged and PAGED_KV_NON_TMA_EFFECTIVE:
                cache_idx = _paged_blockwise_cache_indices(
                    n_block * BLOCK_N,
                    tl.arange(0, BLOCK_N),
                    k_len,
                    page_table_ptr_b,
                    block_size,
                    page_stride_rows,
                    BLOCK_N,
                    PAGED_GATHER_MODE,
                    BOUNDARY_CHECK=True,
                )
                bK = tl.load(
                    k_base + cache_idx[None, :] * k_row_stride + d_idx[:, None],
                    mask=(d_idx[:, None] < d) & kv_mask[None, :],
                    other=0.0,
                )
                bV = tl.load(
                    v_base + cache_idx[:, None] * v_row_stride + d_idx[None, :],
                    mask=kv_mask[:, None] & (d_idx[None, :] < d),
                    other=0.0,
                )
            elif is_paged:
                k_tile, bV = _load_paged_kv_tma_tile(
                    paged_k_desc,
                    paged_v_desc,
                    page_table_ptr_b,
                    n_block * BLOCK_N,
                    block_size,
                    BLOCK_N,
                    HEAD_DIM_PADDED,
                )
                bK = tl.trans(k_tile)
            else:
                cache_idx = col_idx
                bK = tl.load(
                    k_base + cache_idx[None, :] * k_row_stride + d_idx[:, None],
                    mask=(d_idx[:, None] < d) & kv_mask[None, :],
                    other=0.0,
                )
                bV = tl.load(
                    v_base + cache_idx[:, None] * v_row_stride + d_idx[None, :],
                    mask=kv_mask[:, None] & (d_idx[None, :] < d),
                    other=0.0,
                )

            S = tl.dot(q_tile, bK, out_dtype=tl.float32)
            S = _apply_softcap_v3(S, softcap, is_softcap)
            S = _apply_alibi_v3(
                S,
                col_idx,
                row_idx_q,
                q_len,
                k_len,
                IS_CAUSAL=is_causal,
                IS_ALIBI=is_alibi,
                alibi_slope=alibi_slope,
            )
            S = _apply_mask_v3(
                S,
                col_idx,
                row_idx_q,
                q_len,
                k_len,
                window_size_left,
                window_size_right,
                IS_EVEN_MN=False,
                IS_CAUSAL=is_causal,
                IS_LOCAL=is_local,
            )
            alpha, P, rowmax, rowsum = _softmax_online_deferred(
                S,
                rowmax,
                rowsum,
                softmax_scale_log2e=scale_softmax_log2,
                IS_BORDER=True,
            )
            acc = acc * alpha[:, None]
            acc = tl.dot(P.to(v_ptr.dtype.element_ty), bV, acc, out_dtype=tl.float32)

        n_dense_end = n_block_max - n_masking_steps
        for n_block in tl.range(
            n_dense_end - 1,
            n_block_min - 1,
            step=-1,
            num_stages=3,
        ):
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            if is_paged and PAGED_KV_NON_TMA_EFFECTIVE:
                cache_idx = _paged_blockwise_cache_indices(
                    n_block * BLOCK_N,
                    tl.arange(0, BLOCK_N),
                    k_len,
                    page_table_ptr_b,
                    block_size,
                    page_stride_rows,
                    BLOCK_N,
                    PAGED_GATHER_MODE,
                    BOUNDARY_CHECK=False,
                )
                bK = tl.load(
                    k_base + cache_idx[None, :] * k_row_stride + d_idx[:, None],
                    mask=d_idx[:, None] < d,
                    other=0.0,
                )
                bV = tl.load(
                    v_base + cache_idx[:, None] * v_row_stride + d_idx[None, :],
                    mask=d_idx[None, :] < d,
                    other=0.0,
                )
            elif is_paged:
                k_tile, bV = _load_paged_kv_tma_tile(
                    paged_k_desc,
                    paged_v_desc,
                    page_table_ptr_b,
                    n_block * BLOCK_N,
                    block_size,
                    BLOCK_N,
                    HEAD_DIM_PADDED,
                )
                bK = tl.trans(k_tile)
            else:
                cache_idx = col_idx
                bK = tl.load(
                    k_base + cache_idx[None, :] * k_row_stride + d_idx[:, None],
                    mask=d_idx[:, None] < d,
                    other=0.0,
                )
                bV = tl.load(
                    v_base + cache_idx[:, None] * v_row_stride + d_idx[None, :],
                    mask=d_idx[None, :] < d,
                    other=0.0,
                )

            S = tl.dot(q_tile, bK, out_dtype=tl.float32)
            S = _apply_softcap_v3(S, softcap, is_softcap)
            S = _apply_alibi_v3(
                S,
                col_idx,
                row_idx_q,
                q_len,
                k_len,
                IS_CAUSAL=is_causal,
                IS_ALIBI=is_alibi,
                alibi_slope=alibi_slope,
            )
            S = _apply_mask_v3(
                S,
                col_idx,
                row_idx_q,
                q_len,
                k_len,
                window_size_left,
                window_size_right,
                IS_EVEN_MN=True,
                IS_CAUSAL=False,
                IS_LOCAL=is_local,
            )
            alpha, P, rowmax, rowsum = _softmax_online_deferred(
                S,
                rowmax,
                rowsum,
                softmax_scale_log2e=scale_softmax_log2,
                IS_BORDER=is_local,
            )
            acc = acc * alpha[:, None]
            acc = tl.dot(P.to(v_ptr.dtype.element_ty), bV, acc, out_dtype=tl.float32)

        if is_s_aux:
            sink = tl.load(s_aux_ptr + query_head)
            rowmax, rowsum = _merge_attention_sink(
                rowmax,
                rowsum,
                sink,
                softmax_scale_log2e=scale_softmax_log2,
            )
        invalid = (rowsum == 0) | (rowsum != rowsum)
        inv_sum = tl.where(invalid, 1.0, 1.0 / rowsum)
        acc = acc * inv_sum[:, None]
        if PACK_GQA:
            o_ptrs = (
                o_ptr
                + o_offset
                + row_idx_q[:, None] * o_row_stride
                + query_head[:, None] * o_head_stride
                + d_idx[None, :]
            )
        else:
            o_base = o_ptr + o_offset + query_head * o_head_stride
            o_ptrs = o_base + row_idx_q[:, None] * o_row_stride + d_idx[None, :]
        tl.store(
            o_ptrs,
            acc.to(o_ptr.dtype.element_ty),
            mask=(packed_row[:, None] < effective_q_len) & (d_idx[None, :] < d),
        )
        if STORE_LSE:
            lse = tl.where(
                invalid,
                float("-inf") if is_s_aux else float("inf"),
                rowmax * scale_softmax + tl.log(rowsum),
            )
            lse_ptr = softmax_lse_ptr + query_head * total_q + lse_offset + row_idx_q
            tl.store(lse_ptr, lse, mask=packed_row < effective_q_len)


def launch_direct(
    fwd_args,
    *,
    max_seqlen_k,
    batch_size,
    effective_max_q,
    effective_num_heads,
    pack_factor,
    is_paged,
    paged_prefill,
    pack_gqa,
    paged_gather_mode,
    paged_kv_non_tma,
    store_lse,
    total_q,
    ragged_scheduler,
    heads_in_l2,
):
    """Launch the direct family using its private grid and shape bucket."""
    plan = DirectSchedulingHeuristics.launch_plan(
        max_seqlen_k=max_seqlen_k,
        batch_size=batch_size,
        effective_max_q=effective_max_q,
        effective_num_heads=effective_num_heads,
        pack_factor=pack_factor,
        is_paged=is_paged,
        paged_prefill=paged_prefill,
        pack_gqa=pack_gqa,
    )
    resolved_heads_in_l2 = CommonSchedulingHeuristics.binary_heads_in_l2(heads_in_l2)

    def grid(meta):
        if ragged_scheduler:
            compact_m_upper = CommonSchedulingHeuristics.compact_m_upper(
                total_q=total_q,
                pack_factor=plan.pack_factor,
                batch_size=batch_size,
                block_m=meta["BLOCK_M"],
            )
            return (compact_m_upper * plan.effective_num_heads,)
        m_tiles = triton.cdiv(plan.effective_max_q, meta["BLOCK_M"])
        if plan.batch_first_grid:
            return (batch_size, m_tiles, plan.effective_num_heads)
        return (m_tiles, batch_size, plan.effective_num_heads)

    return flash_varlen_fwd_v3_tle_direct_kernel[grid](
        *fwd_args,
        DIRECT_SHAPE_BUCKET=plan.shape_bucket,
        PACK_GQA=pack_gqa,
        BATCH_FIRST_GRID=plan.batch_first_grid,
        SINGLE_KV_TILE=plan.single_kv_tile,
        STORE_LSE=store_lse,
        PAGED_KV_NON_TMA=paged_kv_non_tma,
        RAGGED_SCHEDULER=ragged_scheduler,
        HEADS_IN_L2=resolved_heads_in_l2,
        PAGED_GATHER_MODE=paged_gather_mode,
    )
