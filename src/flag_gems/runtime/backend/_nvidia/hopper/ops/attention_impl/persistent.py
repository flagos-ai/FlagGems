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

"""Persistent ping-pong kernel for the Hopper FA3 long-sequence family."""

import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

from .common import (
    _apply_alibi_v3,
    _apply_mask_v3,
    _apply_softcap_v3,
    _buf_phase_tle,
    _copy_dense_kv_tile_to_pipe,
    _copy_dense_kv_tma_tile_to_pipe,
    _copy_dense_q_tile_to_pipe,
    _copy_dense_tile_to_smem,
    _copy_packed_gqa_q_tile_to_pipe,
    _copy_packed_gqa_tile_to_smem,
    _copy_paged_kv_tile_to_pipe,
    _copy_paged_kv_tma_tile_to_pipe,
    _fence_async_shared_cta,
    _merge_attention_sink,
    _paged_blockwise_cache_indices,
    _persistent_tile_coords,
    _ragged_persistent_split_tile_coords,
    _ragged_persistent_tile_coords,
    _softmax_online_deferred,
    _split_n_block_range,
    _store_packed_gqa_tile_from_regs,
)
from .scheduling import CommonSchedulingHeuristics, PersistentSchedulingHeuristics
from .validation import tle

_prune_persistent_configs = PersistentSchedulingHeuristics.prune_autotune_configs
_heur_block_k = CommonSchedulingHeuristics.block_k


def _persistent_configs():
    if any(name.startswith("FLAG_GEMS_FA3_TLE_EXPERIMENT_") for name in os.environ):
        return PersistentSchedulingHeuristics.autotune_configs()
    configs = runtime.get_tuned_config("flash_attn_varlen_fa3_persistent")
    return configs or PersistentSchedulingHeuristics.autotune_configs()


@libentry()
@triton.jit
def _reset_scheduler_counter_kernel(counter_ptr):
    """Reset the persistent scheduler ticket on the current stream."""

    tl.store(counter_ptr, 0)


def _reset_scheduler_counter(counter):
    _reset_scheduler_counter_kernel[(1,)](counter, num_warps=1)


@triton.jit
def _flash_varlen_fwd_v3_tle_persistent_producer(
    q_ptr,
    k_ptr,
    v_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
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
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    is_paged: tl.constexpr,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    page_stride_rows,
    q_smem,
    q_writer0,
    q_writer1,
    k_writer,
    v_writer,
    q_empties,
    q_fulls,
    q_fulls_manual,
    scheduler_counter_ptr,
    scheduler_state,
    scheduler_empty,
    scheduler_full,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    BM_SPLIT: tl.constexpr,
    USE_TMA_QO: tl.constexpr,
    Q_PIPE_ASYNC: tl.constexpr,
    USE_TMA_KV: tl.constexpr,
    PAGED_KV_NON_TMA: tl.constexpr,
    PAGED_PIPE_ASYNC: tl.constexpr,
    PAGED_GATHER_MODE: tl.constexpr,
    PACK_GQA: tl.constexpr,
    RAGGED_SCHEDULER: tl.constexpr,
    HEADS_IN_L2: tl.constexpr,
    DYNAMIC_SCHEDULER: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    pack_factor: tl.constexpr = h_hk_ratio if PACK_GQA else 1
    effective_heads: tl.constexpr = hk if PACK_GQA else h
    num_pid_m = tl.cdiv(seqlen_q * pack_factor, BLOCK_M)
    total_tiles = num_pid_m * b * effective_heads

    tile_idx = prog_id
    tile_count = 0
    accum_cnt_kv = 0
    if SPLIT_KV:
        (
            m_block,
            bid,
            hid,
            split_id,
            split_count,
            total_split_tiles,
            work_valid,
        ) = _ragged_persistent_split_tile_coords(
            tile_idx,
            cu_seqlens_q_ptr,
            seqused_k_ptr,
            b,
            effective_heads,
            BLOCK_M,
            MAX_SPLITS,
            pack_factor,
            HEADS_IN_L2,
        )
    elif RAGGED_SCHEDULER:
        m_block, bid, hid, work_valid = _ragged_persistent_tile_coords(
            tile_idx,
            cu_seqlens_q_ptr,
            b,
            effective_heads,
            BLOCK_M,
            pack_factor,
            HEADS_IN_L2,
        )
    else:
        m_block, bid, hid = _persistent_tile_coords(
            tile_idx,
            num_pid_m,
            b,
            effective_heads,
            HEADS_IN_L2,
        )
        work_valid = tile_idx < total_tiles
    while work_valid:
        q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
        q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
        q_len = q_eos - q_bos
        q_offset = q_bos * q_row_stride

        if is_seqused_k:
            k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
            k_bos = 0
        else:
            k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
            k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
            k_len = k_eos - k_bos
        effective_q_len = q_len * pack_factor
        valid_q_tile = m_block * BLOCK_M < effective_q_len
        if valid_q_tile:
            q_block_start = (m_block * BLOCK_M) // pack_factor
            q_block_end = tl.cdiv((m_block + 1) * BLOCK_M, pack_factor)
            if is_local:
                n_block_min = tl.maximum(
                    0,
                    (q_block_start + k_len - q_len - window_size_left) // BLOCK_N,
                )
            else:
                n_block_min = 0

            n_block_max = tl.cdiv(k_len, BLOCK_N)
            if is_causal or is_local:
                n_block_max = tl.minimum(
                    n_block_max,
                    tl.cdiv(
                        q_block_end + k_len - q_len + window_size_right,
                        BLOCK_N,
                    ),
                )
            if SPLIT_KV:
                n_block_min, n_block_max = _split_n_block_range(
                    n_block_min,
                    n_block_max,
                    split_id,
                    split_count,
                )

            if n_block_min < n_block_max:
                if PACK_GQA:
                    kv_head = hid
                    q_base = q_ptr + q_offset
                else:
                    kv_head = hid // h_hk_ratio
                    q_base = q_ptr + q_offset + hid * q_head_stride
                if is_paged:
                    page_table_ptr_b = page_table_ptr + bid * page_table_batch_stride
                    k_base = k_ptr + kv_head * k_head_stride
                    v_base = v_ptr + kv_head * v_head_stride
                else:
                    k_base = k_ptr + k_bos * k_row_stride + kv_head * k_head_stride
                    v_base = v_ptr + k_bos * v_row_stride + kv_head * v_head_stride

                if USE_TMA_QO:
                    if PACK_GQA:
                        q_desc = tl.make_tensor_descriptor(
                            base=q_base,
                            shape=[q_len, h, d],
                            strides=[q_row_stride, q_head_stride, 1],
                            block_shape=[
                                BM_SPLIT // h_hk_ratio,
                                h_hk_ratio,
                                HEAD_DIM_PADDED,
                            ],
                        )
                    else:
                        q_desc = tl.make_tensor_descriptor(
                            base=q_base,
                            shape=[q_len, d],
                            strides=[q_row_stride, 1],
                            block_shape=[BM_SPLIT, HEAD_DIM_PADDED],
                        )
                if is_paged and (not PAGED_KV_NON_TMA):
                    k_desc = tl.make_tensor_descriptor(
                        base=k_base,
                        shape=[bk, block_size, d],
                        strides=[
                            page_stride_rows * k_row_stride,
                            k_row_stride,
                            1,
                        ],
                        block_shape=[1, BLOCK_N, HEAD_DIM_PADDED],
                    )
                    v_desc = tl.make_tensor_descriptor(
                        base=v_base,
                        shape=[bk, block_size, d],
                        strides=[
                            page_stride_rows * v_row_stride,
                            v_row_stride,
                            1,
                        ],
                        block_shape=[1, BLOCK_N, HEAD_DIM_PADDED],
                    )
                elif (not is_paged) and USE_TMA_KV:
                    k_desc = tl.make_tensor_descriptor(
                        base=k_base,
                        shape=[k_len, d],
                        strides=[k_row_stride, 1],
                        block_shape=[BLOCK_N, HEAD_DIM_PADDED],
                    )
                    v_desc = tl.make_tensor_descriptor(
                        base=v_base,
                        shape=[k_len, d],
                        strides=[v_row_stride, 1],
                        block_shape=[BLOCK_N, HEAD_DIM_PADDED],
                    )

                q_buf, q_phase_idx = _buf_phase_tle(tile_count, NUM_BUFFERS_Q)
                q0_idx = q_buf
                q1_idx = q_buf + NUM_BUFFERS_Q

                if USE_TMA_QO:
                    tle.gpu.barrier_wait(q_empties[q0_idx], phaseIdx=q_phase_idx)
                    if PACK_GQA:
                        q0_packed_smem = q_smem.slot(q0_idx).reshape(
                            [
                                BM_SPLIT // h_hk_ratio,
                                h_hk_ratio,
                                HEAD_DIM_PADDED,
                            ]
                        )
                        tle.gpu.copy(
                            q_desc,
                            q0_packed_smem,
                            [
                                BM_SPLIT // h_hk_ratio,
                                h_hk_ratio,
                                HEAD_DIM_PADDED,
                            ],
                            [
                                m_block * BLOCK_M // h_hk_ratio,
                                kv_head * h_hk_ratio,
                                0,
                            ],
                            barrier=q_fulls[q0_idx],
                        )
                    else:
                        tle.gpu.copy(
                            q_desc,
                            q_smem.slot(q0_idx),
                            [BM_SPLIT, HEAD_DIM_PADDED],
                            [m_block * BLOCK_M, 0],
                            barrier=q_fulls[q0_idx],
                        )
                elif NUM_MMA_GROUPS == 1 and Q_PIPE_ASYNC:
                    if PACK_GQA:
                        _copy_packed_gqa_q_tile_to_pipe(
                            q_writer0,
                            tile_count,
                            q_ptr,
                            q_offset,
                            q_row_stride,
                            q_head_stride,
                            kv_head,
                            m_block * BLOCK_M,
                            q_len,
                            d,
                            h_hk_ratio,
                            BM_SPLIT,
                            HEAD_DIM_PADDED,
                        )
                    else:
                        _copy_dense_q_tile_to_pipe(
                            q_writer0,
                            tile_count,
                            q_base,
                            q_row_stride,
                            m_block * BLOCK_M,
                            q_len,
                            d,
                            BM_SPLIT,
                            HEAD_DIM_PADDED,
                        )
                elif PACK_GQA:
                    tle.gpu.barrier_wait(q_empties[q0_idx], phaseIdx=q_phase_idx)
                    _copy_packed_gqa_tile_to_smem(
                        q_ptr,
                        q_offset,
                        q_row_stride,
                        q_head_stride,
                        kv_head,
                        q_smem.slot(q0_idx),
                        m_block * BLOCK_M,
                        q_len,
                        d,
                        h_hk_ratio,
                        BM_SPLIT,
                        HEAD_DIM_PADDED,
                    )
                    _fence_async_shared_cta()
                    tle.gpu.barrier_arrive(q_fulls_manual[q0_idx], phaseIdx=q_phase_idx)
                else:
                    tle.gpu.barrier_wait(q_empties[q0_idx], phaseIdx=q_phase_idx)
                    _copy_dense_tile_to_smem(
                        q_base,
                        q_row_stride,
                        q_smem.slot(q0_idx),
                        m_block * BLOCK_M,
                        q_len,
                        d,
                        BM_SPLIT,
                        HEAD_DIM_PADDED,
                    )
                    _fence_async_shared_cta()
                    tle.gpu.barrier_arrive(q_fulls_manual[q0_idx], phaseIdx=q_phase_idx)

                kv_offset = n_block_min * BLOCK_N
                if is_paged and PAGED_KV_NON_TMA:
                    if BM_SPLIT != 16:
                        _copy_paged_kv_tile_to_pipe(
                            k_writer,
                            accum_cnt_kv,
                            PAGED_PIPE_ASYNC,
                            k_base,
                            k_row_stride,
                            page_table_ptr_b,
                            kv_offset,
                            k_len,
                            d,
                            block_size,
                            page_stride_rows,
                            BLOCK_N,
                            HEAD_DIM_PADDED,
                            PAGED_GATHER_MODE,
                        )
                elif BM_SPLIT != 16:
                    if is_paged:
                        _copy_paged_kv_tma_tile_to_pipe(
                            k_writer,
                            accum_cnt_kv,
                            k_desc,
                            page_table_ptr_b,
                            kv_offset,
                            block_size,
                            BLOCK_N,
                            HEAD_DIM_PADDED,
                        )
                    elif USE_TMA_KV:
                        _copy_dense_kv_tma_tile_to_pipe(
                            k_writer,
                            accum_cnt_kv,
                            k_desc,
                            kv_offset,
                            BLOCK_N,
                            HEAD_DIM_PADDED,
                        )
                    else:
                        _copy_dense_kv_tile_to_pipe(
                            k_writer,
                            accum_cnt_kv,
                            k_base,
                            k_row_stride,
                            kv_offset,
                            k_len,
                            d,
                            BLOCK_N,
                            HEAD_DIM_PADDED,
                        )

                if NUM_MMA_GROUPS == 2:
                    if USE_TMA_QO:
                        tle.gpu.barrier_wait(q_empties[q1_idx], phaseIdx=q_phase_idx)
                        if PACK_GQA:
                            q1_packed_smem = q_smem.slot(q1_idx).reshape(
                                [
                                    BM_SPLIT // h_hk_ratio,
                                    h_hk_ratio,
                                    HEAD_DIM_PADDED,
                                ]
                            )
                            tle.gpu.copy(
                                q_desc,
                                q1_packed_smem,
                                [
                                    BM_SPLIT // h_hk_ratio,
                                    h_hk_ratio,
                                    HEAD_DIM_PADDED,
                                ],
                                [
                                    (m_block * BLOCK_M + BM_SPLIT) // h_hk_ratio,
                                    kv_head * h_hk_ratio,
                                    0,
                                ],
                                barrier=q_fulls[q1_idx],
                            )
                        else:
                            tle.gpu.copy(
                                q_desc,
                                q_smem.slot(q1_idx),
                                [BM_SPLIT, HEAD_DIM_PADDED],
                                [m_block * BLOCK_M + BM_SPLIT, 0],
                                barrier=q_fulls[q1_idx],
                            )
                    elif PACK_GQA:
                        tle.gpu.barrier_wait(q_empties[q1_idx], phaseIdx=q_phase_idx)
                        _copy_packed_gqa_tile_to_smem(
                            q_ptr,
                            q_offset,
                            q_row_stride,
                            q_head_stride,
                            kv_head,
                            q_smem.slot(q1_idx),
                            m_block * BLOCK_M + BM_SPLIT,
                            q_len,
                            d,
                            h_hk_ratio,
                            BM_SPLIT,
                            HEAD_DIM_PADDED,
                        )
                        _fence_async_shared_cta()
                        tle.gpu.barrier_arrive(
                            q_fulls_manual[q1_idx], phaseIdx=q_phase_idx
                        )
                    else:
                        tle.gpu.barrier_wait(q_empties[q1_idx], phaseIdx=q_phase_idx)
                        _copy_dense_tile_to_smem(
                            q_base,
                            q_row_stride,
                            q_smem.slot(q1_idx),
                            m_block * BLOCK_M + BM_SPLIT,
                            q_len,
                            d,
                            BM_SPLIT,
                            HEAD_DIM_PADDED,
                        )
                        _fence_async_shared_cta()
                        tle.gpu.barrier_arrive(
                            q_fulls_manual[q1_idx], phaseIdx=q_phase_idx
                        )

                if is_paged and PAGED_KV_NON_TMA:
                    _copy_paged_kv_tile_to_pipe(
                        v_writer,
                        accum_cnt_kv,
                        PAGED_PIPE_ASYNC,
                        v_base,
                        v_row_stride,
                        page_table_ptr_b,
                        kv_offset,
                        k_len,
                        d,
                        block_size,
                        page_stride_rows,
                        BLOCK_N,
                        HEAD_DIM_PADDED,
                        PAGED_GATHER_MODE,
                    )
                elif is_paged:
                    _copy_paged_kv_tma_tile_to_pipe(
                        v_writer,
                        accum_cnt_kv,
                        v_desc,
                        page_table_ptr_b,
                        kv_offset,
                        block_size,
                        BLOCK_N,
                        HEAD_DIM_PADDED,
                    )
                elif USE_TMA_KV:
                    _copy_dense_kv_tma_tile_to_pipe(
                        v_writer,
                        accum_cnt_kv,
                        v_desc,
                        kv_offset,
                        BLOCK_N,
                        HEAD_DIM_PADDED,
                    )
                else:
                    _copy_dense_kv_tile_to_pipe(
                        v_writer,
                        accum_cnt_kv,
                        v_base,
                        v_row_stride,
                        kv_offset,
                        k_len,
                        d,
                        BLOCK_N,
                        HEAD_DIM_PADDED,
                    )
                accum_cnt_kv += 1

                # Keep the paged cp.async hot loop mask-free.  A separate
                # trailing loop preserves correctness for a partial KV tile
                # without carrying its predicates through every full tile.
                if is_paged and PAGED_KV_NON_TMA:
                    full_n_block_max = tl.minimum(n_block_max, k_len // BLOCK_N)
                    for n_block in tl.range(n_block_min + 1, full_n_block_max):
                        kv_offset = n_block * BLOCK_N
                        if BM_SPLIT != 16:
                            _copy_paged_kv_tile_to_pipe(
                                k_writer,
                                accum_cnt_kv,
                                PAGED_PIPE_ASYNC,
                                k_base,
                                k_row_stride,
                                page_table_ptr_b,
                                kv_offset,
                                k_len,
                                d,
                                block_size,
                                page_stride_rows,
                                BLOCK_N,
                                HEAD_DIM_PADDED,
                                PAGED_GATHER_MODE,
                                False,
                            )
                        _copy_paged_kv_tile_to_pipe(
                            v_writer,
                            accum_cnt_kv,
                            PAGED_PIPE_ASYNC,
                            v_base,
                            v_row_stride,
                            page_table_ptr_b,
                            kv_offset,
                            k_len,
                            d,
                            block_size,
                            page_stride_rows,
                            BLOCK_N,
                            HEAD_DIM_PADDED,
                            PAGED_GATHER_MODE,
                            False,
                        )
                        accum_cnt_kv += 1

                    border_n_block_start = tl.maximum(n_block_min + 1, full_n_block_max)
                    for n_block in tl.range(border_n_block_start, n_block_max):
                        kv_offset = n_block * BLOCK_N
                        if BM_SPLIT != 16:
                            _copy_paged_kv_tile_to_pipe(
                                k_writer,
                                accum_cnt_kv,
                                PAGED_PIPE_ASYNC,
                                k_base,
                                k_row_stride,
                                page_table_ptr_b,
                                kv_offset,
                                k_len,
                                d,
                                block_size,
                                page_stride_rows,
                                BLOCK_N,
                                HEAD_DIM_PADDED,
                                PAGED_GATHER_MODE,
                                True,
                            )
                        _copy_paged_kv_tile_to_pipe(
                            v_writer,
                            accum_cnt_kv,
                            PAGED_PIPE_ASYNC,
                            v_base,
                            v_row_stride,
                            page_table_ptr_b,
                            kv_offset,
                            k_len,
                            d,
                            block_size,
                            page_stride_rows,
                            BLOCK_N,
                            HEAD_DIM_PADDED,
                            PAGED_GATHER_MODE,
                            True,
                        )
                        accum_cnt_kv += 1

                remaining_n_block_start = n_block_min + 1
                if is_paged and PAGED_KV_NON_TMA:
                    remaining_n_block_start = n_block_max
                for n_block in tl.range(remaining_n_block_start, n_block_max):
                    kv_offset = n_block * BLOCK_N

                    if is_paged and PAGED_KV_NON_TMA:
                        if BM_SPLIT != 16:
                            _copy_paged_kv_tile_to_pipe(
                                k_writer,
                                accum_cnt_kv,
                                PAGED_PIPE_ASYNC,
                                k_base,
                                k_row_stride,
                                page_table_ptr_b,
                                kv_offset,
                                k_len,
                                d,
                                block_size,
                                page_stride_rows,
                                BLOCK_N,
                                HEAD_DIM_PADDED,
                                PAGED_GATHER_MODE,
                            )
                    elif BM_SPLIT != 16:
                        if is_paged:
                            _copy_paged_kv_tma_tile_to_pipe(
                                k_writer,
                                accum_cnt_kv,
                                k_desc,
                                page_table_ptr_b,
                                kv_offset,
                                block_size,
                                BLOCK_N,
                                HEAD_DIM_PADDED,
                            )
                        elif USE_TMA_KV:
                            _copy_dense_kv_tma_tile_to_pipe(
                                k_writer,
                                accum_cnt_kv,
                                k_desc,
                                kv_offset,
                                BLOCK_N,
                                HEAD_DIM_PADDED,
                            )
                        else:
                            _copy_dense_kv_tile_to_pipe(
                                k_writer,
                                accum_cnt_kv,
                                k_base,
                                k_row_stride,
                                kv_offset,
                                k_len,
                                d,
                                BLOCK_N,
                                HEAD_DIM_PADDED,
                            )

                    if is_paged and PAGED_KV_NON_TMA:
                        _copy_paged_kv_tile_to_pipe(
                            v_writer,
                            accum_cnt_kv,
                            PAGED_PIPE_ASYNC,
                            v_base,
                            v_row_stride,
                            page_table_ptr_b,
                            kv_offset,
                            k_len,
                            d,
                            block_size,
                            page_stride_rows,
                            BLOCK_N,
                            HEAD_DIM_PADDED,
                            PAGED_GATHER_MODE,
                        )
                    elif is_paged:
                        _copy_paged_kv_tma_tile_to_pipe(
                            v_writer,
                            accum_cnt_kv,
                            v_desc,
                            page_table_ptr_b,
                            kv_offset,
                            block_size,
                            BLOCK_N,
                            HEAD_DIM_PADDED,
                        )
                    elif USE_TMA_KV:
                        _copy_dense_kv_tma_tile_to_pipe(
                            v_writer,
                            accum_cnt_kv,
                            v_desc,
                            kv_offset,
                            BLOCK_N,
                            HEAD_DIM_PADDED,
                        )
                    else:
                        _copy_dense_kv_tile_to_pipe(
                            v_writer,
                            accum_cnt_kv,
                            v_base,
                            v_row_stride,
                            kv_offset,
                            k_len,
                            d,
                            BLOCK_N,
                            HEAD_DIM_PADDED,
                        )
                    accum_cnt_kv += 1

                tile_count += 1

        if DYNAMIC_SCHEDULER:
            scheduler_lane = tl.arange(0, 128)
            claimed = tl.atomic_add(
                scheduler_counter_ptr + scheduler_lane,
                1,
                mask=scheduler_lane == 0,
                sem="relaxed",
                scope="gpu",
            )
            claimed = tl.sum(tl.where(scheduler_lane == 0, claimed, 0), axis=0)
            if SPLIT_KV:
                # The metadata semaphore persists across CUDA Graph replay.
                # Exactly one claim is made per compact work item, so reducing
                # the monotonic ticket modulo the compact work count removes a
                # separate memset launch from every attention invocation.
                claimed %= total_split_tiles
            tile_idx = claimed + num_progs
            tle.gpu.barrier_wait(scheduler_empty)
            tl.store(tle.gpu.local_ptr(scheduler_state, (0,)), tile_idx)
            tle.gpu.barrier_arrive(scheduler_full)
        else:
            tile_idx += num_progs
        if SPLIT_KV:
            (
                m_block,
                bid,
                hid,
                split_id,
                split_count,
                total_split_tiles,
                work_valid,
            ) = _ragged_persistent_split_tile_coords(
                tile_idx,
                cu_seqlens_q_ptr,
                seqused_k_ptr,
                b,
                effective_heads,
                BLOCK_M,
                MAX_SPLITS,
                pack_factor,
                HEADS_IN_L2,
            )
        elif RAGGED_SCHEDULER:
            m_block, bid, hid, work_valid = _ragged_persistent_tile_coords(
                tile_idx,
                cu_seqlens_q_ptr,
                b,
                effective_heads,
                BLOCK_M,
                pack_factor,
                HEADS_IN_L2,
            )
        else:
            m_block, bid, hid = _persistent_tile_coords(
                tile_idx,
                num_pid_m,
                b,
                effective_heads,
                HEADS_IN_L2,
            )
            work_valid = tile_idx < total_tiles


@triton.jit
def _load_paged_k_for_warp_mma(
    k_ptr,
    k_row_stride,
    k_head_stride,
    kv_head,
    page_table_ptr_b,
    n_offset,
    k_len,
    d: tl.constexpr,
    block_size: tl.constexpr,
    page_stride_rows,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_N)
    cache_idx = _paged_blockwise_cache_indices(
        n_offset,
        rows,
        k_len,
        page_table_ptr_b,
        block_size,
        page_stride_rows,
        BLOCK_N,
        2,
        BOUNDARY_CHECK=True,
    )
    d_idx = tl.arange(0, HEAD_DIM_PADDED)
    return tl.load(
        k_ptr
        + kv_head * k_head_stride
        + cache_idx[None, :] * k_row_stride
        + d_idx[:, None],
        mask=(n_offset + rows[None, :] < k_len) & (d_idx[:, None] < d),
        other=0.0,
    )


@triton.jit
def _flash_varlen_fwd_v3_tle_persistent_consumer(
    q_ptr,
    k_ptr,
    o_ptr,
    o_desc,
    softmax_lse_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    o_row_stride,
    o_head_stride,
    cu_seqlens_q_ptr,
    is_seqused_k: tl.constexpr,
    cu_seqlens_k_ptr,
    seqused_k_ptr,
    b,
    h: tl.constexpr,
    hk: tl.constexpr,
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
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    is_s_aux: tl.constexpr,
    s_aux_ptr,
    total_q,
    k_row_stride,
    k_head_stride,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    page_stride_rows,
    q_smem,
    q_reader,
    k_reader,
    v_reader,
    q_empties,
    q_fulls,
    q_fulls_manual,
    ping_to_c0,
    ping_to_c1,
    o_store_ready,
    scheduler_state,
    scheduler_empty,
    scheduler_full,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    BM_SPLIT: tl.constexpr,
    USE_TMA_QO: tl.constexpr,
    Q_PIPE_ASYNC: tl.constexpr,
    REUSE_Q_SMEM_O: tl.constexpr,
    CID: tl.constexpr,
    PACK_GQA: tl.constexpr,
    RAGGED_SCHEDULER: tl.constexpr,
    HEADS_IN_L2: tl.constexpr,
    DYNAMIC_SCHEDULER: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    INPUT_DTYPE = q_ptr.dtype.element_ty
    USE_WARP_MMA: tl.constexpr = BM_SPLIT == 16
    cid: tl.constexpr = CID
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    pack_factor: tl.constexpr = h // hk if PACK_GQA else 1
    effective_heads: tl.constexpr = hk if PACK_GQA else h
    num_pid_m = tl.cdiv(seqlen_q * pack_factor, BLOCK_M)
    total_tiles = num_pid_m * b * effective_heads

    if NUM_MMA_GROUPS == 2 and cid == 1:
        tle.gpu.barrier_arrive(ping_to_c0)
    if DYNAMIC_SCHEDULER:
        # Publish initial scheduler storage availability to the producer.  The
        # two consumer warpgroups contribute 256 arrivals; producer wait
        # contributes the remaining 128 arrivals to the 384-thread barrier.
        tle.gpu.barrier_arrive(scheduler_empty)

    tile_idx = prog_id
    tile_count = 0
    accum_cnt_kv = 0
    if SPLIT_KV:
        (
            m_block,
            bid,
            hid,
            split_id,
            split_count,
            _,
            work_valid,
        ) = _ragged_persistent_split_tile_coords(
            tile_idx,
            cu_seqlens_q_ptr,
            seqused_k_ptr,
            b,
            effective_heads,
            BLOCK_M,
            MAX_SPLITS,
            pack_factor,
            HEADS_IN_L2,
        )
    elif RAGGED_SCHEDULER:
        m_block, bid, hid, work_valid = _ragged_persistent_tile_coords(
            tile_idx,
            cu_seqlens_q_ptr,
            b,
            effective_heads,
            BLOCK_M,
            pack_factor,
            HEADS_IN_L2,
        )
    else:
        m_block, bid, hid = _persistent_tile_coords(
            tile_idx,
            num_pid_m,
            b,
            effective_heads,
            HEADS_IN_L2,
        )
        work_valid = tile_idx < total_tiles
    while work_valid:
        q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
        q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
        q_len = q_eos - q_bos
        o_offset = q_bos * o_row_stride
        lse_offset = q_bos
        if is_seqused_k:
            k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
        else:
            k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
            k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
            k_len = k_eos - k_bos
        if USE_WARP_MMA:
            page_table_ptr_b = page_table_ptr + bid * page_table_batch_stride
            kv_head = hid if PACK_GQA else hid // (h // hk)
        packed_row_start = m_block * BLOCK_M + cid * BM_SPLIT
        packed_row = packed_row_start + tl.arange(0, BM_SPLIT)
        if PACK_GQA:
            row_idx_q = packed_row // pack_factor
            query_head = hid * pack_factor + packed_row % pack_factor
        else:
            row_idx_q = packed_row
            query_head = hid

        if is_alibi:
            alibi_slope = tl.load(
                alibi_slopes_ptr + bid * alibi_slopes_batch_stride + query_head
            )
            if PACK_GQA:
                alibi_slope = alibi_slope[:, None]
            alibi_slope = alibi_slope / scale_softmax
        else:
            alibi_slope = 0.0

        effective_q_len = q_len * pack_factor
        valid_q_tile = m_block * BLOCK_M < effective_q_len
        if valid_q_tile:
            q_block_start = (m_block * BLOCK_M) // pack_factor
            q_block_end = tl.cdiv((m_block + 1) * BLOCK_M, pack_factor)
            if is_local:
                n_block_min = tl.maximum(
                    0,
                    (q_block_start + k_len - q_len - window_size_left) // BLOCK_N,
                )
            else:
                n_block_min = 0

            n_block_max = tl.cdiv(k_len, BLOCK_N)
            if is_causal or is_local:
                n_block_max = tl.minimum(
                    n_block_max,
                    tl.cdiv(
                        q_block_end + k_len - q_len + window_size_right,
                        BLOCK_N,
                    ),
                )
            if SPLIT_KV:
                n_block_min, n_block_max = _split_n_block_range(
                    n_block_min,
                    n_block_max,
                    split_id,
                    split_count,
                )

            if not PACK_GQA:
                o_base = o_ptr + o_offset + hid * o_head_stride
            q_buf, q_phase_idx = _buf_phase_tle(tile_count, NUM_BUFFERS_Q)
            q_idx = q_buf + cid * NUM_BUFFERS_Q
            if n_block_min < n_block_max:
                # A causal KV tile is mask-free only when its exclusive end
                # is no greater than the earliest query row's causal bound.
                # Using n_block_max alone is wrong for unaligned k_len-q_len
                # offsets because two trailing tiles can both be partial.
                causal_col_exclusive = (
                    packed_row_start // pack_factor + k_len - q_len + 1
                )
                causal_full_block_max = tl.maximum(
                    n_block_min, causal_col_exclusive // BLOCK_N
                )
                rowmax = tl.full([BM_SPLIT], float("-inf"), dtype=tl.float32)
                rowsum = tl.zeros([BM_SPLIT], dtype=tl.float32)
                acc = tl.zeros([BM_SPLIT, HEAD_DIM_PADDED], dtype=tl.float32)

                if USE_TMA_QO:
                    tle.gpu.barrier_wait(q_fulls[q_idx], phaseIdx=q_phase_idx)
                    q_tile = q_smem.slot(q_idx)
                elif NUM_MMA_GROUPS == 1 and Q_PIPE_ASYNC:
                    q_tile = q_reader.wait(tile_count).slot.q
                else:
                    tle.gpu.barrier_wait(q_fulls_manual[q_idx], phaseIdx=q_phase_idx)
                    q_tile = q_smem.slot(q_idx)

                if not USE_WARP_MMA:
                    k_tile = k_reader.wait(accum_cnt_kv).slot.kv

                if NUM_MMA_GROUPS == 2:
                    if cid == 0:
                        tle.gpu.barrier_wait(ping_to_c0)
                    else:
                        tle.gpu.barrier_wait(ping_to_c1)
                if USE_WARP_MMA:
                    q_regs = tl.load(tle.gpu.local_ptr(q_tile))
                    k_regs = _load_paged_k_for_warp_mma(
                        k_ptr,
                        k_row_stride,
                        k_head_stride,
                        kv_head,
                        page_table_ptr_b,
                        n_block_min * BLOCK_N,
                        k_len,
                        d,
                        block_size,
                        page_stride_rows,
                        BLOCK_N,
                        HEAD_DIM_PADDED,
                    )
                    qk = tl.dot(q_regs, k_regs, out_dtype=tl.float32)
                else:
                    qk = tle.gpu.wgmma(
                        q_tile,
                        k_tile,
                        out_dtype=tl.float32,
                        trans_b=True,
                    )
                if NUM_MMA_GROUPS == 2:
                    if cid == 0:
                        tle.gpu.barrier_arrive(ping_to_c1)
                    else:
                        tle.gpu.barrier_arrive(ping_to_c0)

                if not USE_WARP_MMA:
                    qk = tle.gpu.wgmma_wait(0, qk)
                    k_reader.release(accum_cnt_kv)

                n_block = n_block_min
                col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                qk = _apply_softcap_v3(qk, softcap, is_softcap)
                qk = _apply_alibi_v3(
                    qk,
                    col_idx,
                    row_idx_q,
                    q_len,
                    k_len,
                    IS_CAUSAL=is_causal,
                    IS_ALIBI=is_alibi,
                    alibi_slope=alibi_slope,
                )
                if is_causal and not is_local:
                    if n_block < causal_full_block_max:
                        alpha, p, rowmax, rowsum = _softmax_online_deferred(
                            qk,
                            rowmax,
                            rowsum,
                            softmax_scale_log2e=scale_softmax_log2,
                            IS_BORDER=False,
                        )
                    else:
                        qk = _apply_mask_v3(
                            qk,
                            col_idx,
                            row_idx_q,
                            q_len,
                            k_len,
                            window_size_left,
                            window_size_right,
                            IS_EVEN_MN=False,
                            IS_CAUSAL=True,
                            IS_LOCAL=False,
                        )
                        alpha, p, rowmax, rowsum = _softmax_online_deferred(
                            qk,
                            rowmax,
                            rowsum,
                            softmax_scale_log2e=scale_softmax_log2,
                            IS_BORDER=True,
                        )
                else:
                    qk = _apply_mask_v3(
                        qk,
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
                    alpha, p, rowmax, rowsum = _softmax_online_deferred(
                        qk,
                        rowmax,
                        rowsum,
                        softmax_scale_log2e=scale_softmax_log2,
                        IS_BORDER=True,
                    )
                accum_cnt_kv += 1

                if is_causal and not is_local:
                    # Full causal tiles need no elementwise mask.  Keep them
                    # in a separate lexical loop so Triton does not carry the
                    # border path's predicates and SSA values through every
                    # pipelined iteration.
                    full_loop_end = tl.minimum(n_block_max, causal_full_block_max)
                    for n_block in tl.range(n_block_min + 1, full_loop_end):
                        if not USE_WARP_MMA:
                            k_tile = k_reader.wait(accum_cnt_kv).slot.kv

                        if NUM_MMA_GROUPS == 2:
                            if cid == 0:
                                tle.gpu.barrier_wait(ping_to_c0)
                            else:
                                tle.gpu.barrier_wait(ping_to_c1)
                        if USE_WARP_MMA:
                            q_regs = tl.load(tle.gpu.local_ptr(q_tile))
                            k_regs = _load_paged_k_for_warp_mma(
                                k_ptr,
                                k_row_stride,
                                k_head_stride,
                                kv_head,
                                page_table_ptr_b,
                                n_block * BLOCK_N,
                                k_len,
                                d,
                                block_size,
                                page_stride_rows,
                                BLOCK_N,
                                HEAD_DIM_PADDED,
                            )
                            qk = tl.dot(q_regs, k_regs, out_dtype=tl.float32)
                        else:
                            qk = tle.gpu.wgmma(
                                q_tile,
                                k_tile,
                                out_dtype=tl.float32,
                                trans_b=True,
                            )
                        if NUM_MMA_GROUPS == 2:
                            if cid == 0:
                                tle.gpu.barrier_arrive(ping_to_c1)
                            else:
                                tle.gpu.barrier_arrive(ping_to_c0)

                        v_tile = v_reader.wait(accum_cnt_kv - 1).slot.kv
                        if USE_WARP_MMA:
                            v_regs = tl.load(tle.gpu.local_ptr(v_tile))
                            acc = tl.dot(
                                p.to(INPUT_DTYPE),
                                v_regs,
                                acc,
                                out_dtype=tl.float32,
                            )
                        else:
                            acc = tle.gpu.wgmma(p.to(INPUT_DTYPE), v_tile, acc)
                            qk = tle.gpu.wgmma_wait(1, qk)
                            k_reader.release(accum_cnt_kv)

                        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                        qk = _apply_softcap_v3(qk, softcap, is_softcap)
                        qk = _apply_alibi_v3(
                            qk,
                            col_idx,
                            row_idx_q,
                            q_len,
                            k_len,
                            IS_CAUSAL=True,
                            IS_ALIBI=is_alibi,
                            alibi_slope=alibi_slope,
                        )
                        alpha, p, rowmax, rowsum = _softmax_online_deferred(
                            qk,
                            rowmax,
                            rowsum,
                            softmax_scale_log2e=scale_softmax_log2,
                            IS_BORDER=False,
                        )

                        if not USE_WARP_MMA:
                            acc = tle.gpu.wgmma_wait(0, acc)
                        v_reader.release(accum_cnt_kv - 1)
                        acc = acc * alpha[:, None]
                        accum_cnt_kv += 1

                    # One or more trailing tiles can cross the earliest
                    # query's causal boundary when the varlen offset is not
                    # BLOCK_N aligned.  Mask all of those tiles.
                    border_start = tl.maximum(
                        n_block_min + 1,
                        tl.minimum(n_block_max, causal_full_block_max),
                    )
                    for n_block in tl.range(border_start, n_block_max):
                        if not USE_WARP_MMA:
                            k_tile = k_reader.wait(accum_cnt_kv).slot.kv

                        if NUM_MMA_GROUPS == 2:
                            if cid == 0:
                                tle.gpu.barrier_wait(ping_to_c0)
                            else:
                                tle.gpu.barrier_wait(ping_to_c1)
                        if USE_WARP_MMA:
                            q_regs = tl.load(tle.gpu.local_ptr(q_tile))
                            k_regs = _load_paged_k_for_warp_mma(
                                k_ptr,
                                k_row_stride,
                                k_head_stride,
                                kv_head,
                                page_table_ptr_b,
                                n_block * BLOCK_N,
                                k_len,
                                d,
                                block_size,
                                page_stride_rows,
                                BLOCK_N,
                                HEAD_DIM_PADDED,
                            )
                            qk = tl.dot(q_regs, k_regs, out_dtype=tl.float32)
                        else:
                            qk = tle.gpu.wgmma(
                                q_tile,
                                k_tile,
                                out_dtype=tl.float32,
                                trans_b=True,
                            )
                        if NUM_MMA_GROUPS == 2:
                            if cid == 0:
                                tle.gpu.barrier_arrive(ping_to_c1)
                            else:
                                tle.gpu.barrier_arrive(ping_to_c0)

                        v_tile = v_reader.wait(accum_cnt_kv - 1).slot.kv
                        if USE_WARP_MMA:
                            v_regs = tl.load(tle.gpu.local_ptr(v_tile))
                            acc = tl.dot(
                                p.to(INPUT_DTYPE),
                                v_regs,
                                acc,
                                out_dtype=tl.float32,
                            )
                        else:
                            acc = tle.gpu.wgmma(p.to(INPUT_DTYPE), v_tile, acc)
                            qk = tle.gpu.wgmma_wait(1, qk)
                            k_reader.release(accum_cnt_kv)

                        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                        qk = _apply_softcap_v3(qk, softcap, is_softcap)
                        qk = _apply_alibi_v3(
                            qk,
                            col_idx,
                            row_idx_q,
                            q_len,
                            k_len,
                            IS_CAUSAL=True,
                            IS_ALIBI=is_alibi,
                            alibi_slope=alibi_slope,
                        )
                        qk = _apply_mask_v3(
                            qk,
                            col_idx,
                            row_idx_q,
                            q_len,
                            k_len,
                            window_size_left,
                            window_size_right,
                            IS_EVEN_MN=False,
                            IS_CAUSAL=True,
                            IS_LOCAL=False,
                        )
                        alpha, p, rowmax, rowsum = _softmax_online_deferred(
                            qk,
                            rowmax,
                            rowsum,
                            softmax_scale_log2e=scale_softmax_log2,
                            IS_BORDER=True,
                        )

                        if not USE_WARP_MMA:
                            acc = tle.gpu.wgmma_wait(0, acc)
                        v_reader.release(accum_cnt_kv - 1)
                        acc = acc * alpha[:, None]
                        accum_cnt_kv += 1
                else:
                    for n_block in tl.range(n_block_min + 1, n_block_max):
                        if not USE_WARP_MMA:
                            k_tile = k_reader.wait(accum_cnt_kv).slot.kv

                        if NUM_MMA_GROUPS == 2:
                            if cid == 0:
                                tle.gpu.barrier_wait(ping_to_c0)
                            else:
                                tle.gpu.barrier_wait(ping_to_c1)
                        if USE_WARP_MMA:
                            q_regs = tl.load(tle.gpu.local_ptr(q_tile))
                            k_regs = _load_paged_k_for_warp_mma(
                                k_ptr,
                                k_row_stride,
                                k_head_stride,
                                kv_head,
                                page_table_ptr_b,
                                n_block * BLOCK_N,
                                k_len,
                                d,
                                block_size,
                                page_stride_rows,
                                BLOCK_N,
                                HEAD_DIM_PADDED,
                            )
                            qk = tl.dot(q_regs, k_regs, out_dtype=tl.float32)
                        else:
                            qk = tle.gpu.wgmma(
                                q_tile,
                                k_tile,
                                out_dtype=tl.float32,
                                trans_b=True,
                            )
                        if NUM_MMA_GROUPS == 2:
                            if cid == 0:
                                tle.gpu.barrier_arrive(ping_to_c1)
                            else:
                                tle.gpu.barrier_arrive(ping_to_c0)

                        v_tile = v_reader.wait(accum_cnt_kv - 1).slot.kv
                        if USE_WARP_MMA:
                            v_regs = tl.load(tle.gpu.local_ptr(v_tile))
                            acc = tl.dot(
                                p.to(INPUT_DTYPE),
                                v_regs,
                                acc,
                                out_dtype=tl.float32,
                            )
                        else:
                            acc = tle.gpu.wgmma(p.to(INPUT_DTYPE), v_tile, acc)
                            qk = tle.gpu.wgmma_wait(1, qk)
                            k_reader.release(accum_cnt_kv)

                        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                        qk = _apply_softcap_v3(qk, softcap, is_softcap)
                        qk = _apply_alibi_v3(
                            qk,
                            col_idx,
                            row_idx_q,
                            q_len,
                            k_len,
                            IS_CAUSAL=is_causal,
                            IS_ALIBI=is_alibi,
                            alibi_slope=alibi_slope,
                        )
                        if is_causal and not is_local:
                            if n_block + 1 < n_block_max:
                                alpha, p, rowmax, rowsum = _softmax_online_deferred(
                                    qk,
                                    rowmax,
                                    rowsum,
                                    softmax_scale_log2e=scale_softmax_log2,
                                    IS_BORDER=False,
                                )
                            else:
                                qk = _apply_mask_v3(
                                    qk,
                                    col_idx,
                                    row_idx_q,
                                    q_len,
                                    k_len,
                                    window_size_left,
                                    window_size_right,
                                    IS_EVEN_MN=False,
                                    IS_CAUSAL=True,
                                    IS_LOCAL=False,
                                )
                                alpha, p, rowmax, rowsum = _softmax_online_deferred(
                                    qk,
                                    rowmax,
                                    rowsum,
                                    softmax_scale_log2e=scale_softmax_log2,
                                    IS_BORDER=True,
                                )
                        else:
                            qk = _apply_mask_v3(
                                qk,
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
                            alpha, p, rowmax, rowsum = _softmax_online_deferred(
                                qk,
                                rowmax,
                                rowsum,
                                softmax_scale_log2e=scale_softmax_log2,
                                IS_BORDER=True,
                            )

                        if not USE_WARP_MMA:
                            acc = tle.gpu.wgmma_wait(0, acc)
                        v_reader.release(accum_cnt_kv - 1)
                        acc = acc * alpha[:, None]
                        accum_cnt_kv += 1

                v_tile = v_reader.wait(accum_cnt_kv - 1).slot.kv
                if USE_WARP_MMA:
                    v_regs = tl.load(tle.gpu.local_ptr(v_tile))
                    acc = tl.dot(
                        p.to(INPUT_DTYPE),
                        v_regs,
                        acc,
                        out_dtype=tl.float32,
                    )
                else:
                    acc = tle.gpu.wgmma(p.to(INPUT_DTYPE), v_tile, acc)
                    acc = tle.gpu.wgmma_wait(1, acc)
                # Dense output and the register-store fallback no longer need
                # Q once WGMMA has retired.  Packed TMA output reuses this Q
                # slot as its epilogue staging buffer, so keep ownership until
                # that store has completed below.
                if not (PACK_GQA and USE_TMA_QO and REUSE_Q_SMEM_O):
                    if USE_TMA_QO or NUM_MMA_GROUPS == 2 or not Q_PIPE_ASYNC:
                        tle.gpu.barrier_arrive(q_empties[q_idx], phaseIdx=q_phase_idx)
                    else:
                        q_reader.release(tile_count)

                if not USE_WARP_MMA:
                    acc = tle.gpu.wgmma_wait(0, acc)
                v_reader.release(accum_cnt_kv - 1)

                if is_s_aux:
                    if SPLIT_KV:
                        if split_id == 0:
                            sink = tl.load(s_aux_ptr + query_head)
                            rowmax, rowsum = _merge_attention_sink(
                                rowmax,
                                rowsum,
                                sink,
                                softmax_scale_log2e=scale_softmax_log2,
                            )
                    else:
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
                if SPLIT_KV or STORE_LSE:
                    lse = tl.where(
                        invalid,
                        float("-inf") if SPLIT_KV or is_s_aux else float("inf"),
                        rowmax * scale_softmax + tl.log(rowsum),
                    )
                tile_count += 1
            else:
                acc = tl.zeros([BM_SPLIT, HEAD_DIM_PADDED], dtype=tl.float32)
                if SPLIT_KV or STORE_LSE:
                    lse = tl.full(
                        [BM_SPLIT],
                        float("-inf") if SPLIT_KV else float("inf"),
                        dtype=tl.float32,
                    )
                    if is_s_aux:
                        if SPLIT_KV:
                            if split_id == 0:
                                lse = tl.load(s_aux_ptr + query_head).to(tl.float32)
                        else:
                            lse = tl.load(s_aux_ptr + query_head).to(tl.float32)

            if SPLIT_KV:
                partial_split_stride = h * total_q * d
                partial_head_stride = total_q * d
                partial_base = partial_out_ptr + split_id * partial_split_stride
                _store_packed_gqa_tile_from_regs(
                    partial_base,
                    lse_offset * d,
                    d,
                    partial_head_stride,
                    hid,
                    acc,
                    packed_row_start,
                    q_len,
                    d,
                    pack_factor,
                    BM_SPLIT,
                    HEAD_DIM_PADDED,
                )
                partial_lse_base = split_id * h * total_q + query_head * total_q
                tl.store(
                    partial_lse_ptr + partial_lse_base + lse_offset + row_idx_q,
                    lse,
                    mask=packed_row < effective_q_len,
                )

            final_q_len = 0 if SPLIT_KV else q_len

            if PACK_GQA:
                packed_row_offset = packed_row_start
                full_o_tile = packed_row_offset + BM_SPLIT <= effective_q_len
                has_kv_tile = n_block_min < n_block_max
                if SPLIT_KV:
                    pass
                elif USE_TMA_QO and REUSE_Q_SMEM_O:
                    if full_o_tile and has_kv_tile:
                        # TensorDescriptor.store(register_tensor) asks Triton
                        # for a separate BM_SPLIT x D shared staging tile.
                        # Reuse the Q slot after its final WGMMA instead,
                        # matching the CUDA FA3 epilogue and saving 32 KiB for
                        # BM_SPLIT=64, D=128.
                        packed_o = acc.to(o_ptr.dtype.element_ty)
                        tl.store(tle.gpu.local_ptr(q_smem.slot(q_idx)), packed_o)
                        # local_store is executed cooperatively by all four
                        # warps, while TMA store is issued by one elected
                        # thread. Synchronize before that thread reads SMEM.
                        tle.gpu.barrier_wait(o_store_ready[cid])
                        packed_o_smem = q_smem.slot(q_idx).reshape(
                            [BM_SPLIT // pack_factor, pack_factor, HEAD_DIM_PADDED]
                        )
                        tle.gpu.copy(
                            packed_o_smem,
                            o_desc,
                            [
                                BM_SPLIT // pack_factor,
                                pack_factor,
                                HEAD_DIM_PADDED,
                            ],
                            [
                                lse_offset + packed_row_offset // pack_factor,
                                hid * pack_factor,
                                0,
                            ],
                        )
                        # TMA completion is tracked by the elected issuing
                        # thread. Reconverge before publishing Q as reusable.
                        tle.gpu.barrier_wait(o_store_ready[cid])
                    else:
                        # Keep TensorDescriptor.store out of this compile-time
                        # specialization; otherwise its implicit 32-KiB SMEM
                        # tile defeats Q-buffer reuse even when never executed.
                        _store_packed_gqa_tile_from_regs(
                            o_ptr,
                            o_offset,
                            o_row_stride,
                            o_head_stride,
                            hid,
                            acc.to(o_ptr.dtype.element_ty),
                            packed_row_offset,
                            final_q_len,
                            d,
                            pack_factor,
                            BM_SPLIT,
                            HEAD_DIM_PADDED,
                        )
                elif USE_TMA_QO and full_o_tile:
                    o_desc.store(
                        [
                            lse_offset + packed_row_offset // pack_factor,
                            hid * pack_factor,
                            0,
                        ],
                        tl.reshape(
                            acc.to(o_ptr.dtype.element_ty),
                            [
                                BM_SPLIT // pack_factor,
                                pack_factor,
                                HEAD_DIM_PADDED,
                            ],
                        ),
                    )
                else:
                    _store_packed_gqa_tile_from_regs(
                        o_ptr,
                        o_offset,
                        o_row_stride,
                        o_head_stride,
                        hid,
                        acc.to(o_ptr.dtype.element_ty),
                        packed_row_offset,
                        final_q_len,
                        d,
                        pack_factor,
                        BM_SPLIT,
                        HEAD_DIM_PADDED,
                    )
            elif USE_TMA_QO:
                o_desc.store(
                    [
                        lse_offset + m_block * BLOCK_M + cid * BM_SPLIT,
                        hid,
                        0,
                    ],
                    tl.reshape(
                        acc.to(o_ptr.dtype.element_ty),
                        [BM_SPLIT, 1, HEAD_DIM_PADDED],
                    ),
                )
            else:
                # Store the unpacked register tile for valid Q rows and columns.
                store_rows = m_block * BLOCK_M + cid * BM_SPLIT + tl.arange(0, BM_SPLIT)
                store_cols = tl.arange(0, HEAD_DIM_PADDED)
                store_ptrs = (
                    o_base + store_rows[:, None] * o_row_stride + store_cols[None, :]
                )
                store_mask = (store_rows[:, None] < q_len) & (store_cols[None, :] < d)
                tl.store(
                    store_ptrs,
                    acc.to(o_ptr.dtype.element_ty),
                    mask=store_mask,
                )
            if STORE_LSE and not SPLIT_KV:
                lse_ptr = (
                    softmax_lse_ptr + query_head * total_q + lse_offset + row_idx_q
                )
                tl.store(lse_ptr, lse, mask=packed_row < effective_q_len)
            if PACK_GQA and USE_TMA_QO and REUSE_Q_SMEM_O and has_kv_tile:
                tle.gpu.barrier_arrive(q_empties[q_idx], phaseIdx=q_phase_idx)

        if DYNAMIC_SCHEDULER:
            tle.gpu.barrier_wait(scheduler_full)
            tile_idx = tl.load(tle.gpu.local_ptr(scheduler_state, (0,)))
            tle.gpu.barrier_arrive(scheduler_empty)
        else:
            tile_idx += num_progs
        if SPLIT_KV:
            (
                m_block,
                bid,
                hid,
                split_id,
                split_count,
                _,
                work_valid,
            ) = _ragged_persistent_split_tile_coords(
                tile_idx,
                cu_seqlens_q_ptr,
                seqused_k_ptr,
                b,
                effective_heads,
                BLOCK_M,
                MAX_SPLITS,
                pack_factor,
                HEADS_IN_L2,
            )
        elif RAGGED_SCHEDULER:
            m_block, bid, hid, work_valid = _ragged_persistent_tile_coords(
                tile_idx,
                cu_seqlens_q_ptr,
                b,
                effective_heads,
                BLOCK_M,
                pack_factor,
                HEADS_IN_L2,
            )
        else:
            m_block, bid, hid = _persistent_tile_coords(
                tile_idx,
                num_pid_m,
                b,
                effective_heads,
                HEADS_IN_L2,
            )
            work_valid = tile_idx < total_tiles


@libentry()
@libtuner(
    configs=_persistent_configs(),
    prune_configs_by={"early_config_prune": _prune_persistent_configs},
    warmup=10,
    rep=20,
    reset_to_zero=["scheduler_counter_ptr"],
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
        "PAGED_GATHER_MODE",
        "PAGED_KV_NON_TMA",
        "PACK_GQA",
        "RAGGED_SCHEDULER",
        "HEADS_IN_L2",
        "DYNAMIC_SCHEDULER",
        "SPLIT_KV",
        "MAX_SPLITS",
        "STORE_LSE",
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
def flash_varlen_fwd_v3_tle_kernel(
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
    o_desc,
    scheduler_counter_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    Q_STAGE_CAPACITY: tl.constexpr,
    USE_TMA_QO: tl.constexpr,
    Q_PIPE_ASYNC: tl.constexpr,
    REUSE_Q_SMEM_O: tl.constexpr,
    USE_TMA_KV: tl.constexpr,
    PAGED_KV_NON_TMA: tl.constexpr,
    PAGED_PIPE_ASYNC: tl.constexpr,
    PACK_GQA: tl.constexpr,
    RAGGED_SCHEDULER: tl.constexpr,
    HEADS_IN_L2: tl.constexpr,
    DYNAMIC_SCHEDULER: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    STORE_LSE: tl.constexpr,
    PAGED_GATHER_MODE: tl.constexpr = 2,
):
    BM_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    HEAD_DIM_PADDED: tl.constexpr = BLOCK_K
    Q_PIPE_ASYNC_EFFECTIVE: tl.constexpr = Q_PIPE_ASYNC and d < 256
    PAGED_KV_NON_TMA_EFFECTIVE: tl.constexpr = PAGED_KV_NON_TMA or (
        block_size % BLOCK_N != 0
    )
    INPUT_DTYPE = q_ptr.dtype.element_ty
    page_stride_rows = k_batch_stride // k_row_stride
    THREADS_IN_MMA_GROUPS: tl.constexpr = NUM_MMA_WARPS * 32

    q_smem = tle.gpu.alloc(
        [Q_STAGE_CAPACITY, BM_SPLIT, HEAD_DIM_PADDED],
        dtype=INPUT_DTYPE,
        layout=None,
        scope=tle.gpu.smem,
    )
    if USE_TMA_QO or NUM_MMA_GROUPS == 2 or not Q_PIPE_ASYNC_EFFECTIVE:
        # TMA Q and the latency-sensitive dual-consumer path keep their explicit
        # barriers. These dummy endpoints are compile-time dead in the worker
        # partitions.
        q_writer0 = q_smem
        q_writer1 = q_smem
        q_reader0 = q_smem
        q_reader1 = q_smem
    else:
        # The compact single-consumer path can overlap Q cp.async completion
        # without paying for a second pipe's lifecycle bookkeeping.
        q0_pipe = tle.pipe(
            capacity=NUM_BUFFERS_Q,
            scope="cta",
            name="fa3_q0_stage",
            q=q_smem,
        )
        q_writer0 = q0_pipe.writer()
        q_reader0 = q0_pipe.reader()
        q_writer1 = q_writer0
        q_reader1 = q_reader0
    if BM_SPLIT == 16:
        k_smem = tle.gpu.alloc(
            [NUM_BUFFERS_KV, 1, 1],
            dtype=INPUT_DTYPE,
            layout=None,
            scope=tle.gpu.smem,
        )
    else:
        k_smem = tle.gpu.alloc(
            [NUM_BUFFERS_KV, BLOCK_N, HEAD_DIM_PADDED],
            dtype=INPUT_DTYPE,
            layout=None,
            scope=tle.gpu.smem,
        )
    v_smem = tle.gpu.alloc(
        [NUM_BUFFERS_KV, BLOCK_N, HEAD_DIM_PADDED],
        dtype=INPUT_DTYPE,
        layout=None,
        scope=tle.gpu.smem,
    )

    # One transport-independent pipe owns every K/V GMEM-to-SMEM stage.  The
    # payload window selects TMA for descriptor copies and cp.async (or a
    # cooperative local-store publish fallback) for pointer-based copies.
    # Named readers ensure a stage is reusable only after both MMA warpgroups
    # have retired their WGMMA reads.
    if NUM_MMA_GROUPS == 2:
        k_pipe = tle.pipe(
            capacity=NUM_BUFFERS_KV,
            scope="cta",
            name="fa3_k_stage",
            readers=("c0", "c1"),
            kv=k_smem,
        )
        v_pipe = tle.pipe(
            capacity=NUM_BUFFERS_KV,
            scope="cta",
            name="fa3_v_stage",
            readers=("c0", "c1"),
            kv=v_smem,
        )
        k_reader0 = k_pipe.reader("c0")
        k_reader1 = k_pipe.reader("c1")
        v_reader0 = v_pipe.reader("c0")
        v_reader1 = v_pipe.reader("c1")
    else:
        k_pipe = tle.pipe(
            capacity=NUM_BUFFERS_KV,
            scope="cta",
            name="fa3_k_stage",
            kv=k_smem,
        )
        v_pipe = tle.pipe(
            capacity=NUM_BUFFERS_KV,
            scope="cta",
            name="fa3_v_stage",
            kv=v_smem,
        )
        k_reader0 = k_pipe.reader()
        v_reader0 = v_pipe.reader()

    q_empties = tle.gpu.alloc_barriers(
        num_barriers=Q_STAGE_CAPACITY,
        arrive_count=1,
        init=tle.gpu.READY,
        arrival_mode="participant",
    )
    q_fulls = tle.gpu.alloc_barriers(
        num_barriers=Q_STAGE_CAPACITY,
        arrive_count=1,
        expect_bytes=BM_SPLIT * HEAD_DIM_PADDED * 2,
    )
    q_fulls_manual = tle.gpu.alloc_barriers(
        num_barriers=Q_STAGE_CAPACITY,
        arrive_count=1,
        arrival_mode="participant",
    )
    pingpong = tle.gpu.alloc_barriers(
        num_barriers=2,
        arrive_count=THREADS_IN_MMA_GROUPS,
    )
    ping_to_c0 = pingpong[0]
    ping_to_c1 = pingpong[1]
    o_store_ready = tle.gpu.alloc_barriers(
        num_barriers=NUM_MMA_GROUPS,
        arrive_count=THREADS_IN_MMA_GROUPS // NUM_MMA_GROUPS,
    )

    if DYNAMIC_SCHEDULER:
        scheduler_state = tle.gpu.alloc(
            [1],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        scheduler_barriers = tle.gpu.alloc_barriers(
            num_barriers=2,
            arrive_count=THREADS_IN_MMA_GROUPS + 128,
        )
        scheduler_empty = scheduler_barriers[0]
        scheduler_full = scheduler_barriers[1]
    else:
        # warp_specialize cannot capture None.  These aliases are compile-time
        # dead in the static scheduler specialization.
        scheduler_state = q_smem
        scheduler_empty = q_empties[0]
        scheduler_full = q_empties[0]

    # ``warp_specialize`` lowers partition arguments to explicit IR captures
    # and cannot capture Python ``None``.  These pointers are only read behind
    # their matching constexpr flags, so use a valid, never-dereferenced
    # fallback for disabled optional inputs.
    if is_seqused_k:
        ws_seqused_k_ptr = seqused_k_ptr
    else:
        ws_seqused_k_ptr = q_ptr
    if is_paged:
        ws_page_table_ptr = page_table_ptr
    else:
        ws_page_table_ptr = q_ptr
    if is_alibi:
        ws_alibi_slopes_ptr = alibi_slopes_ptr
    else:
        ws_alibi_slopes_ptr = q_ptr
    if is_s_aux:
        ws_s_aux_ptr = s_aux_ptr
    else:
        ws_s_aux_ptr = q_ptr

    # Keep these argument tuples inline.  Assigning a mixed runtime/constexpr
    # tuple to a local first materializes its constexpr entries as scalar
    # tensors, which makes descriptor block shapes invalid in the partitions.
    if NUM_MMA_GROUPS == 2:
        tle.gpu.warp_specialize(
            [
                (
                    _flash_varlen_fwd_v3_tle_persistent_producer,
                    (
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        q_row_stride,
                        k_row_stride,
                        v_row_stride,
                        q_head_stride,
                        k_head_stride,
                        v_head_stride,
                        cu_seqlens_q_ptr,
                        is_seqused_k,
                        cu_seqlens_k_ptr,
                        ws_seqused_k_ptr,
                        b,
                        bk,
                        h,
                        hk,
                        h_hk_ratio,
                        seqlen_q,
                        seqlen_k,
                        d,
                        is_causal,
                        is_local,
                        window_size_left,
                        window_size_right,
                        is_paged,
                        ws_page_table_ptr,
                        page_table_batch_stride,
                        block_size,
                        page_stride_rows,
                        q_smem,
                        q_writer0,
                        q_writer1,
                        k_pipe.writer(),
                        v_pipe.writer(),
                        q_empties,
                        q_fulls,
                        q_fulls_manual,
                        scheduler_counter_ptr,
                        scheduler_state,
                        scheduler_empty,
                        scheduler_full,
                        BLOCK_M,
                        BLOCK_N,
                        HEAD_DIM_PADDED,
                        NUM_BUFFERS_Q,
                        NUM_MMA_GROUPS,
                        BM_SPLIT,
                        USE_TMA_QO,
                        Q_PIPE_ASYNC_EFFECTIVE,
                        USE_TMA_KV,
                        PAGED_KV_NON_TMA_EFFECTIVE,
                        PAGED_PIPE_ASYNC,
                        PAGED_GATHER_MODE,
                        PACK_GQA,
                        RAGGED_SCHEDULER,
                        HEADS_IN_L2,
                        DYNAMIC_SCHEDULER,
                        SPLIT_KV,
                        MAX_SPLITS,
                    ),
                ),
                (
                    _flash_varlen_fwd_v3_tle_persistent_consumer,
                    (
                        q_ptr,
                        k_ptr,
                        o_ptr,
                        o_desc,
                        softmax_lse_ptr,
                        partial_out_ptr,
                        partial_lse_ptr,
                        o_row_stride,
                        o_head_stride,
                        cu_seqlens_q_ptr,
                        is_seqused_k,
                        cu_seqlens_k_ptr,
                        ws_seqused_k_ptr,
                        b,
                        h,
                        hk,
                        seqlen_q,
                        seqlen_k,
                        d,
                        is_softcap,
                        softcap,
                        scale_softmax,
                        scale_softmax_log2,
                        is_causal,
                        is_local,
                        window_size_left,
                        window_size_right,
                        is_alibi,
                        ws_alibi_slopes_ptr,
                        alibi_slopes_batch_stride,
                        is_s_aux,
                        ws_s_aux_ptr,
                        total_q,
                        k_row_stride,
                        k_head_stride,
                        ws_page_table_ptr,
                        page_table_batch_stride,
                        block_size,
                        page_stride_rows,
                        q_smem,
                        q_reader0,
                        k_reader0,
                        v_reader0,
                        q_empties,
                        q_fulls,
                        q_fulls_manual,
                        ping_to_c0,
                        ping_to_c1,
                        o_store_ready,
                        scheduler_state,
                        scheduler_empty,
                        scheduler_full,
                        BLOCK_M,
                        BLOCK_N,
                        HEAD_DIM_PADDED,
                        NUM_BUFFERS_Q,
                        NUM_MMA_GROUPS,
                        BM_SPLIT,
                        USE_TMA_QO,
                        Q_PIPE_ASYNC_EFFECTIVE,
                        REUSE_Q_SMEM_O,
                        0,
                        PACK_GQA,
                        RAGGED_SCHEDULER,
                        HEADS_IN_L2,
                        DYNAMIC_SCHEDULER,
                        SPLIT_KV,
                        MAX_SPLITS,
                        STORE_LSE,
                    ),
                ),
                (
                    _flash_varlen_fwd_v3_tle_persistent_consumer,
                    (
                        q_ptr,
                        k_ptr,
                        o_ptr,
                        o_desc,
                        softmax_lse_ptr,
                        partial_out_ptr,
                        partial_lse_ptr,
                        o_row_stride,
                        o_head_stride,
                        cu_seqlens_q_ptr,
                        is_seqused_k,
                        cu_seqlens_k_ptr,
                        ws_seqused_k_ptr,
                        b,
                        h,
                        hk,
                        seqlen_q,
                        seqlen_k,
                        d,
                        is_softcap,
                        softcap,
                        scale_softmax,
                        scale_softmax_log2,
                        is_causal,
                        is_local,
                        window_size_left,
                        window_size_right,
                        is_alibi,
                        ws_alibi_slopes_ptr,
                        alibi_slopes_batch_stride,
                        is_s_aux,
                        ws_s_aux_ptr,
                        total_q,
                        k_row_stride,
                        k_head_stride,
                        ws_page_table_ptr,
                        page_table_batch_stride,
                        block_size,
                        page_stride_rows,
                        q_smem,
                        q_reader1,
                        k_reader1,
                        v_reader1,
                        q_empties,
                        q_fulls,
                        q_fulls_manual,
                        ping_to_c0,
                        ping_to_c1,
                        o_store_ready,
                        scheduler_state,
                        scheduler_empty,
                        scheduler_full,
                        BLOCK_M,
                        BLOCK_N,
                        HEAD_DIM_PADDED,
                        NUM_BUFFERS_Q,
                        NUM_MMA_GROUPS,
                        BM_SPLIT,
                        USE_TMA_QO,
                        Q_PIPE_ASYNC_EFFECTIVE,
                        REUSE_Q_SMEM_O,
                        1,
                        PACK_GQA,
                        RAGGED_SCHEDULER,
                        HEADS_IN_L2,
                        DYNAMIC_SCHEDULER,
                        SPLIT_KV,
                        MAX_SPLITS,
                        STORE_LSE,
                    ),
                ),
            ],
            [
                NUM_MMA_WARPS // NUM_MMA_GROUPS,
                NUM_MMA_WARPS // NUM_MMA_GROUPS,
            ],
            [232, 232],
        )
    else:
        tle.gpu.warp_specialize(
            [
                (
                    _flash_varlen_fwd_v3_tle_persistent_producer,
                    (
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        q_row_stride,
                        k_row_stride,
                        v_row_stride,
                        q_head_stride,
                        k_head_stride,
                        v_head_stride,
                        cu_seqlens_q_ptr,
                        is_seqused_k,
                        cu_seqlens_k_ptr,
                        ws_seqused_k_ptr,
                        b,
                        bk,
                        h,
                        hk,
                        h_hk_ratio,
                        seqlen_q,
                        seqlen_k,
                        d,
                        is_causal,
                        is_local,
                        window_size_left,
                        window_size_right,
                        is_paged,
                        ws_page_table_ptr,
                        page_table_batch_stride,
                        block_size,
                        page_stride_rows,
                        q_smem,
                        q_writer0,
                        q_writer1,
                        k_pipe.writer(),
                        v_pipe.writer(),
                        q_empties,
                        q_fulls,
                        q_fulls_manual,
                        scheduler_counter_ptr,
                        scheduler_state,
                        scheduler_empty,
                        scheduler_full,
                        BLOCK_M,
                        BLOCK_N,
                        HEAD_DIM_PADDED,
                        NUM_BUFFERS_Q,
                        NUM_MMA_GROUPS,
                        BM_SPLIT,
                        USE_TMA_QO,
                        Q_PIPE_ASYNC_EFFECTIVE,
                        USE_TMA_KV,
                        PAGED_KV_NON_TMA_EFFECTIVE,
                        PAGED_PIPE_ASYNC,
                        PAGED_GATHER_MODE,
                        PACK_GQA,
                        RAGGED_SCHEDULER,
                        HEADS_IN_L2,
                        DYNAMIC_SCHEDULER,
                        SPLIT_KV,
                        MAX_SPLITS,
                    ),
                ),
                (
                    _flash_varlen_fwd_v3_tle_persistent_consumer,
                    (
                        q_ptr,
                        k_ptr,
                        o_ptr,
                        o_desc,
                        softmax_lse_ptr,
                        partial_out_ptr,
                        partial_lse_ptr,
                        o_row_stride,
                        o_head_stride,
                        cu_seqlens_q_ptr,
                        is_seqused_k,
                        cu_seqlens_k_ptr,
                        ws_seqused_k_ptr,
                        b,
                        h,
                        hk,
                        seqlen_q,
                        seqlen_k,
                        d,
                        is_softcap,
                        softcap,
                        scale_softmax,
                        scale_softmax_log2,
                        is_causal,
                        is_local,
                        window_size_left,
                        window_size_right,
                        is_alibi,
                        ws_alibi_slopes_ptr,
                        alibi_slopes_batch_stride,
                        is_s_aux,
                        ws_s_aux_ptr,
                        total_q,
                        k_row_stride,
                        k_head_stride,
                        ws_page_table_ptr,
                        page_table_batch_stride,
                        block_size,
                        page_stride_rows,
                        q_smem,
                        q_reader0,
                        k_reader0,
                        v_reader0,
                        q_empties,
                        q_fulls,
                        q_fulls_manual,
                        ping_to_c0,
                        ping_to_c1,
                        o_store_ready,
                        scheduler_state,
                        scheduler_empty,
                        scheduler_full,
                        BLOCK_M,
                        BLOCK_N,
                        HEAD_DIM_PADDED,
                        NUM_BUFFERS_Q,
                        NUM_MMA_GROUPS,
                        BM_SPLIT,
                        USE_TMA_QO,
                        Q_PIPE_ASYNC_EFFECTIVE,
                        REUSE_Q_SMEM_O,
                        0,
                        PACK_GQA,
                        RAGGED_SCHEDULER,
                        HEADS_IN_L2,
                        DYNAMIC_SCHEDULER,
                        SPLIT_KV,
                        MAX_SPLITS,
                        STORE_LSE,
                    ),
                ),
            ],
            [NUM_MMA_WARPS // NUM_MMA_GROUPS],
            [232],
        )


def launch_persistent(
    fwd_args,
    *,
    output,
    total_q,
    head_size,
    num_sms,
    max_seqlen_k,
    batch_size,
    num_heads,
    num_heads_k,
    effective_max_q,
    effective_num_heads,
    pack_factor,
    pack_gqa,
    paged_gather_mode,
    paged_kv_non_tma,
    ragged_scheduler,
    heads_in_l2,
    dynamic_scheduler,
    store_lse,
    partial_out=None,
    partial_lse=None,
    max_splits=1,
    scheduler_counter=None,
):
    """Launch the long family using its private persistent grid."""
    gqa_ratio = num_heads // num_heads_k
    split_kv = partial_out is not None
    plan = PersistentSchedulingHeuristics.launch_plan(
        heads_in_l2=heads_in_l2,
        allow_head_swizzle=not split_kv,
        pack_gqa=pack_gqa,
        gqa_ratio=gqa_ratio,
        effective_num_heads=effective_num_heads,
        max_seqlen_k=max_seqlen_k,
        head_size=head_size,
        element_size=output.element_size(),
        dynamic_scheduler=dynamic_scheduler,
    )
    o_block_rows = plan.block_m // plan.num_mma_groups
    o_block_heads = 1
    if pack_gqa:
        o_block_rows //= pack_factor
        o_block_heads = pack_factor
    o_desc = TensorDescriptor(
        output,
        shape=[total_q, num_heads, head_size],
        strides=[output.stride(-3), output.stride(-2), 1],
        # Match the kernel's BLOCK_K heuristic.  TensorDescriptor block
        # dimensions must be powers of two, whereas FA's generic rounded
        # head dimension can be 192.
        block_shape=[
            o_block_rows,
            o_block_heads,
            CommonSchedulingHeuristics.padded_head_dim(head_size),
        ],
    )
    if scheduler_counter is None:
        scheduler_counter = torch.empty((1,), dtype=torch.int32, device=output.device)
    # Split-KV wraps the monotonically increasing ticket modulo the compact
    # work count in-kernel, which keeps CUDA Graph replay valid without a
    # separate reset launch on every attention invocation.
    if plan.dynamic_scheduler and not split_kv:
        _reset_scheduler_counter(scheduler_counter)
    if not split_kv:
        partial_out = output
        partial_lse = output
    grid = lambda meta: (
        min(
            num_sms,
            triton.cdiv(effective_max_q, meta["BLOCK_M"])
            * batch_size
            * effective_num_heads
            * max_splits,
        ),
    )
    return flash_varlen_fwd_v3_tle_kernel[grid](
        *fwd_args,
        o_desc,
        scheduler_counter,
        partial_out,
        partial_lse,
        PAGED_KV_NON_TMA=paged_kv_non_tma,
        PACK_GQA=pack_gqa,
        RAGGED_SCHEDULER=ragged_scheduler,
        HEADS_IN_L2=plan.heads_in_l2,
        DYNAMIC_SCHEDULER=plan.dynamic_scheduler,
        SPLIT_KV=split_kv,
        MAX_SPLITS=max_splits,
        STORE_LSE=store_lse,
        PAGED_GATHER_MODE=paged_gather_mode,
    )
