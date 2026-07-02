"""Flash-Decoding-style split-KV decode FA3 TLE experimental kernels.

This family keeps one program per effective query row, batch, head, and KV
split.  Each split writes partial online-softmax state and a separate combine
kernel produces the final output.  ``SPLIT_POLICY`` selects the KV ownership:

* ``0``: contiguous KV-block ranges, matching the Flash-Decoding split-KV idea.
* ``1``: round-robin KV blocks, useful as a load-balanced paged-KV variant.
"""

from .utils import *  # noqa: F401,F403
from .utils import _decode_apply_alibi, _decode_apply_mask
from .fa_hopper_splitkv import (
    flash_varlen_fwd_v3_tle_splitkv_combine_kernel as flash_varlen_fwd_v3_tle_decode_flashdecoding_combine_kernel,
)


def _fa3_decode_flashdecoding_configs():
    configs = []
    for block_n in (64, 128, 256):
        for num_warps in (4, 8):
            if block_n == 64 and num_warps == 8:
                continue
            configs.append(
                triton.Config(
                    {"BLOCK_N": block_n},
                    num_stages=3,
                    num_warps=num_warps,
                )
            )
    return configs


def _prune_fa3_decode_flashdecoding_configs(configs, nargs, **kwargs):
    head_dim = kwargs.get("d", nargs.get("d"))
    is_paged = kwargs.get("is_paged", nargs.get("is_paged"))
    split_policy = kwargs.get("SPLIT_POLICY", nargs.get("SPLIT_POLICY", 0))

    kept = []
    for cfg in configs:
        block_n = cfg.kwargs["BLOCK_N"]
        if is_paged and block_n > 128:
            continue
        if split_policy == 1 and block_n > 128:
            continue
        if head_dim >= 192 and block_n > 128:
            continue
        kept.append(cfg)
    return kept or [configs[0]]


@triton.jit
def _decode_flashdecoding_visit_block(
    n_block,
    q_row,
    q_len,
    k_len,
    d: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    is_softcap: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    is_alibi: tl.constexpr,
    is_paged: tl.constexpr,
    alibi_slope,
    q_vec,
    k_base,
    v_base,
    k_row_stride,
    v_row_stride,
    page_table_ptr_b,
    block_size: tl.constexpr,
    m_i,
    l_i,
    acc,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    PAGED_GATHER_MODE: tl.constexpr,
):
    d_idx = tl.arange(0, HEAD_DIM_PADDED)
    n_idx = tl.arange(0, BLOCK_N)
    col_idx = n_block * BLOCK_N + n_idx
    if is_paged:
        cache_idx = _paged_blockwise_cache_indices(
            n_block * BLOCK_N,
            n_idx,
            k_len,
            page_table_ptr_b,
            block_size,
            BLOCK_N,
            PAGED_GATHER_MODE,
            BOUNDARY_CHECK=True,
        )
    else:
        cache_idx = col_idx

    k_tile = tl.load(
        k_base + cache_idx[:, None] * k_row_stride + d_idx[None, :],
        mask=(col_idx[:, None] < k_len) & (d_idx[None, :] < d),
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(k_tile * q_vec[None, :], 1)
    scores = _apply_softcap_v3(scores, softcap, is_softcap)
    scores = _decode_apply_alibi(
        scores,
        col_idx,
        q_row,
        q_len,
        k_len,
        IS_CAUSAL=is_causal,
        IS_ALIBI=is_alibi,
        alibi_slope=alibi_slope,
    )
    scores = _decode_apply_mask(
        scores,
        col_idx,
        q_row,
        q_len,
        k_len,
        window_size_left,
        window_size_right,
        IS_BORDER=True,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
    )

    m_new = tl.maximum(m_i, tl.max(scores, 0))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.math.exp2((m_i - m_safe) * scale_softmax_log2)
    p_scores = tl.math.exp2(
        scores * scale_softmax_log2 - m_safe * scale_softmax_log2
    )
    l_new = l_i * alpha + tl.sum(p_scores, 0)

    v_tile = tl.load(
        v_base + cache_idx[:, None] * v_row_stride + d_idx[None, :],
        mask=(col_idx[:, None] < k_len) & (d_idx[None, :] < d),
        other=0.0,
    ).to(tl.float32)
    acc = acc * alpha + tl.sum(p_scores[:, None] * v_tile, 0)
    return m_new, l_new, acc


@libentry()
@triton.autotune(
    configs=_fa3_decode_flashdecoding_configs(),
    prune_configs_by={
        "early_config_prune": _prune_fa3_decode_flashdecoding_configs
    },
    key=[
        "d",
        "is_paged",
        "is_causal",
        "is_local",
        "is_alibi",
        "seqlen_q",
        "seqlen_k",
        "total_q",
        "NUM_SPLITS",
        "SPLIT_POLICY",
    ],
)
@triton.heuristics(
    values={
        "BLOCK_K": _heur_block_k,
    }
)
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
def flash_varlen_fwd_v3_tle_decode_flashdecoding_kernel(
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
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    partial_out_ptr,
    partial_m_ptr,
    partial_l_ptr,
    scheduler_metadata_ptr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLIT_POLICY: tl.constexpr,
    HAS_SCHEDULER_METADATA: tl.constexpr,
    PAGED_GATHER_MODE: tl.constexpr = 2,
):
    q_row = tl.program_id(0)
    bid = tl.program_id(1)
    hid_split = tl.program_id(2)
    hid = hid_split % h
    split_id = hid_split // h
    HEAD_DIM_PADDED: tl.constexpr = BLOCK_K

    if is_cu_seqlens_q:
        q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
        q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
        q_len = q_eos - q_bos
        q_offset = q_bos * q_row_stride
        lse_offset = q_bos
    else:
        q_len = seqlen_q
        q_offset = bid * q_batch_stride
        lse_offset = bid * seqlen_q

    if is_cu_seqlens_k:
        k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
        k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
        k_len_cache = k_eos - k_bos
    else:
        k_len_cache = seqlen_k
        k_bos = 0

    if is_seqused_k:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    else:
        k_len = k_len_cache

    split_count = NUM_SPLITS
    if HAS_SCHEDULER_METADATA:
        split_count = tl.load(scheduler_metadata_ptr + 1 + bid).to(tl.int32)
        split_count = tl.minimum(tl.maximum(split_count, 1), NUM_SPLITS)

    if q_row < q_len:
        d_idx = tl.arange(0, HEAD_DIM_PADDED)
        kv_head = hid // h_hk_ratio
        q_base = q_ptr + q_offset + hid * q_head_stride
        if is_paged:
            page_table_ptr_b = page_table_ptr + bid * page_table_batch_stride
            k_base = k_ptr + kv_head * k_head_stride
            v_base = v_ptr + kv_head * v_head_stride
        else:
            page_table_ptr_b = page_table_ptr
            k_base = k_ptr + k_bos * k_row_stride + kv_head * k_head_stride
            v_base = v_ptr + k_bos * v_row_stride + kv_head * v_head_stride

        if is_alibi:
            alibi_slope = tl.load(
                alibi_slopes_ptr + bid * alibi_slopes_batch_stride + hid
            )
            alibi_slope = alibi_slope / scale_softmax
        else:
            alibi_slope = 0.0

        q_vec = tl.load(
            q_base + q_row * q_row_stride + d_idx,
            mask=d_idx < d,
            other=0.0,
        ).to(tl.float32)

        n_blocks = tl.cdiv(k_len, BLOCK_N)
        m_i = tl.full((), float("-inf"), dtype=tl.float32)
        l_i = tl.full((), 0.0, dtype=tl.float32)
        acc = tl.zeros([HEAD_DIM_PADDED], dtype=tl.float32)

        if SPLIT_POLICY == 0:
            blocks_per_split = tl.cdiv(n_blocks, split_count)
            split_block_start = split_id * blocks_per_split
            split_block_end = tl.minimum(n_blocks, split_block_start + blocks_per_split)

            if split_id < split_count and split_block_start < split_block_end:
                for n_block in tl.range(
                    split_block_start,
                    split_block_end,
                    step=1,
                    num_stages=3,
                ):
                    m_i, l_i, acc = _decode_flashdecoding_visit_block(
                        n_block,
                        q_row,
                        q_len,
                        k_len,
                        d,
                        softcap,
                        scale_softmax_log2,
                        window_size_left,
                        window_size_right,
                        is_softcap,
                        is_causal,
                        is_local,
                        is_alibi,
                        is_paged,
                        alibi_slope,
                        q_vec,
                        k_base,
                        v_base,
                        k_row_stride,
                        v_row_stride,
                        page_table_ptr_b,
                        block_size,
                        m_i,
                        l_i,
                        acc,
                        BLOCK_N,
                        HEAD_DIM_PADDED,
                        PAGED_GATHER_MODE,
                    )
        else:
            if split_id < split_count and split_id < n_blocks:
                for n_block in tl.range(
                    split_id,
                    n_blocks,
                    step=NUM_SPLITS,
                    num_stages=3,
                ):
                    m_i, l_i, acc = _decode_flashdecoding_visit_block(
                        n_block,
                        q_row,
                        q_len,
                        k_len,
                        d,
                        softcap,
                        scale_softmax_log2,
                        window_size_left,
                        window_size_right,
                        is_softcap,
                        is_causal,
                        is_local,
                        is_alibi,
                        is_paged,
                        alibi_slope,
                        q_vec,
                        k_base,
                        v_base,
                        k_row_stride,
                        v_row_stride,
                        page_table_ptr_b,
                        block_size,
                        m_i,
                        l_i,
                        acc,
                        BLOCK_N,
                        HEAD_DIM_PADDED,
                        PAGED_GATHER_MODE,
                    )

        partial_row = lse_offset + q_row
        part_base = split_id * h * total_q * d + hid * total_q * d
        tl.store(
            partial_out_ptr + part_base + partial_row * d + d_idx,
            acc,
            mask=d_idx < d,
        )

        stat_base = split_id * h * total_q + hid * total_q
        tl.store(partial_m_ptr + stat_base + partial_row, m_i)
        tl.store(partial_l_ptr + stat_base + partial_row, l_i)


__all__ = [
    "flash_varlen_fwd_v3_tle_decode_flashdecoding_kernel",
    "flash_varlen_fwd_v3_tle_decode_flashdecoding_combine_kernel",
]
