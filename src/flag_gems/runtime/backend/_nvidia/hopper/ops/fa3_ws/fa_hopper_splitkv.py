"""Split-KV decode FA3 TLE kernel family."""

from .utils import *  # noqa: F401,F403
def _fa3_splitkv_configs():
    configs = []
    for block_m in (16, 32, 64):
        for block_n in (64, 128):
            configs.append(
                triton.Config(
                    {"BLOCK_M": block_m, "BLOCK_N": block_n},
                    num_stages=3,
                    num_warps=4,
                )
            )
    return configs


def _prune_fa3_splitkv_configs(configs, nargs, **kwargs):
    head_dim = kwargs.get("d", nargs.get("d"))
    is_paged = kwargs.get("is_paged", nargs.get("is_paged"))
    num_splits = kwargs.get("NUM_SPLITS", nargs.get("NUM_SPLITS", 1))

    kept = []
    for cfg in configs:
        block_m = cfg.kwargs["BLOCK_M"]
        block_n = cfg.kwargs["BLOCK_N"]
        if block_m > 32 and num_splits > 2:
            continue
        if is_paged and block_n > 64:
            continue
        if head_dim > 128 and block_n > 64:
            continue
        if head_dim > 192 and block_m > 32:
            continue
        smem_bytes = (
            (block_m + 2 * block_n) * _next_power_of_2_host(head_dim) * 2
        )
        if smem_bytes > 192 * 1024:
            continue
        kept.append(cfg)

    if kept:
        return kept
    return [configs[0]]



@libentry()
@triton.autotune(
    configs=_fa3_splitkv_configs(),
    prune_configs_by={"early_config_prune": _prune_fa3_splitkv_configs},
    key=[
        "d",
        "is_paged",
        "is_causal",
        "is_local",
        "is_alibi",
        "NUM_SPLITS",
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
def flash_varlen_fwd_v3_tle_splitkv_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PAGED_GATHER_MODE: tl.constexpr = 2,
):
    m_block = tl.program_id(0)
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

    process_q = m_block * BLOCK_M < q_len
    if process_q:
        if is_local:
            n_block_min = tl.maximum(
                0,
                (m_block * BLOCK_M + k_len - q_len - window_size_left) // BLOCK_N,
            )
        else:
            n_block_min = 0

        n_block_max = tl.cdiv(k_len, BLOCK_N)
        if is_causal or is_local:
            n_block_max = tl.minimum(
                n_block_max,
                tl.cdiv(
                    (m_block + 1) * BLOCK_M
                    + k_len
                    - q_len
                    + window_size_right,
                    BLOCK_N,
                ),
            )

        n_blocks = tl.maximum(0, n_block_max - n_block_min)
        blocks_per_split = tl.cdiv(n_blocks, NUM_SPLITS)
        split_block_start = n_block_min + split_id * blocks_per_split
        split_block_end = tl.minimum(
            n_block_max, split_block_start + blocks_per_split
        )

        if is_alibi:
            alibi_slope = tl.load(
                alibi_slopes_ptr + bid * alibi_slopes_batch_stride + hid
            )
            alibi_slope = alibi_slope / scale_softmax
        else:
            alibi_slope = 0.0

        q_base = q_ptr + q_offset + hid * q_head_stride
        q_desc = tl.make_tensor_descriptor(
            base=q_base,
            shape=[q_len, d],
            strides=[q_row_stride, 1],
            block_shape=[BLOCK_M, HEAD_DIM_PADDED],
        )

        kv_head = hid // h_hk_ratio
        if is_paged:
            page_table_ptr_b = page_table_ptr + bid * page_table_batch_stride
            k_base = k_ptr + kv_head * k_head_stride
            v_base = v_ptr + kv_head * v_head_stride
        else:
            k_base = k_ptr + k_bos * k_row_stride + kv_head * k_head_stride
            v_base = v_ptr + k_bos * v_row_stride + kv_head * v_head_stride
            k_desc = tl.make_tensor_descriptor(
                base=k_base,
                shape=[k_len_cache, d],
                strides=[k_row_stride, 1],
                block_shape=[BLOCK_N, HEAD_DIM_PADDED],
            )
            v_desc = tl.make_tensor_descriptor(
                base=v_base,
                shape=[k_len_cache, d],
                strides=[v_row_stride, 1],
                block_shape=[BLOCK_N, HEAD_DIM_PADDED],
            )

        q_tile = q_desc.load([m_block * BLOCK_M, 0])
        row_idx_q = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        rowmax = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        rowsum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

        if split_block_start < split_block_end:
            for n_block in tl.range(
                split_block_end - 1,
                split_block_start - 1,
                step=-1,
                num_stages=3,
            ):
                col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                if is_paged:
                    cache_idx = _paged_blockwise_cache_indices(
                        n_block * BLOCK_N,
                        tl.arange(0, BLOCK_N),
                        k_len,
                        page_table_ptr_b,
                        block_size,
                        BLOCK_N,
                        PAGED_GATHER_MODE,
                        BOUNDARY_CHECK=True,
                    )
                    d_idx = tl.arange(0, HEAD_DIM_PADDED)
                    d_mask = d_idx < d
                    kv_mask = col_idx < k_len
                    bK = tl.load(
                        k_base + cache_idx[None, :] * k_row_stride + d_idx[:, None],
                        mask=d_mask[:, None] & kv_mask[None, :],
                        other=0.0,
                    )
                    bV = tl.load(
                        v_base + cache_idx[:, None] * v_row_stride + d_idx[None, :],
                        mask=kv_mask[:, None] & d_mask[None, :],
                        other=0.0,
                    )
                else:
                    bK = tl.trans(k_desc.load([n_block * BLOCK_N, 0]))
                    bV = v_desc.load([n_block * BLOCK_N, 0])

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
                acc = tl.dot(
                    P.to(v_ptr.dtype.element_ty), bV, acc, out_dtype=tl.float32
                )

        partial_row = lse_offset + row_idx_q
        part_base = split_id * h * total_q * d + hid * total_q * d
        cols = tl.arange(0, HEAD_DIM_PADDED)
        part_ptrs = (
            partial_out_ptr
            + part_base
            + partial_row[:, None] * d
            + cols[None, :]
        )
        part_mask = (row_idx_q[:, None] < q_len) & (cols[None, :] < d)
        tl.store(part_ptrs, acc, mask=part_mask)

        stat_base = split_id * h * total_q + hid * total_q
        stat_ptrs = partial_m_ptr + stat_base + partial_row
        tl.store(stat_ptrs, rowmax, mask=row_idx_q < q_len)
        stat_ptrs = partial_l_ptr + stat_base + partial_row
        tl.store(stat_ptrs, rowsum, mask=row_idx_q < q_len)


@libentry()
@triton.jit(
    do_not_specialize=[
        "o_batch_stride",
        "b",
        "seqlen_q",
        "total_q",
    ]
)
def flash_varlen_fwd_v3_tle_splitkv_combine_kernel(
    o_ptr,
    softmax_lse_ptr,
    partial_out_ptr,
    partial_m_ptr,
    partial_l_ptr,
    o_row_stride,
    o_head_stride,
    o_batch_stride,
    is_cu_seqlens_q: tl.constexpr,
    cu_seqlens_q_ptr,
    b,
    h: tl.constexpr,
    seqlen_q,
    d: tl.constexpr,
    total_q,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    m_block = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)
    HEAD_DIM_PADDED: tl.constexpr = BLOCK_K

    if is_cu_seqlens_q:
        q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
        q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
        q_len = q_eos - q_bos
        o_offset = q_bos * o_row_stride
        lse_offset = q_bos
    else:
        q_len = seqlen_q
        o_offset = bid * o_batch_stride
        lse_offset = bid * seqlen_q

    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    partial_row = lse_offset + row_idx
    valid_row = row_idx < q_len
    cols = tl.arange(0, HEAD_DIM_PADDED)

    m_global = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    for split_id in tl.static_range(0, NUM_SPLITS):
        stat_base = split_id * h * total_q + hid * total_q
        m_i = tl.load(
            partial_m_ptr + stat_base + partial_row,
            mask=valid_row,
            other=float("-inf"),
        )
        m_global = tl.maximum(m_global, m_i)

    m_safe = tl.where(m_global == float("-inf"), 0.0, m_global)
    l_global = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)
    for split_id in tl.static_range(0, NUM_SPLITS):
        stat_base = split_id * h * total_q + hid * total_q
        m_i = tl.load(
            partial_m_ptr + stat_base + partial_row,
            mask=valid_row,
            other=float("-inf"),
        )
        l_i = tl.load(
            partial_l_ptr + stat_base + partial_row,
            mask=valid_row,
            other=0.0,
        )
        alpha = tl.math.exp2((m_i - m_safe) * scale_softmax_log2)
        alpha = tl.where((m_i == float("-inf")) | (l_i == 0.0), 0.0, alpha)
        part_base = split_id * h * total_q * d + hid * total_q * d
        part_ptrs = (
            partial_out_ptr
            + part_base
            + partial_row[:, None] * d
            + cols[None, :]
        )
        part = tl.load(
            part_ptrs,
            mask=valid_row[:, None] & (cols[None, :] < d),
            other=0.0,
        )
        acc += part * alpha[:, None]
        l_global += l_i * alpha

    invalid = (l_global == 0) | (l_global != l_global)
    inv_sum = tl.where(invalid, 1.0, 1.0 / l_global)
    acc = acc * inv_sum[:, None]
    lse = tl.where(
        invalid,
        float("inf"),
        m_global * scale_softmax + tl.log(l_global),
    )

    o_base = o_ptr + o_offset + hid * o_head_stride
    o_ptrs = o_base + row_idx[:, None] * o_row_stride + cols[None, :]
    tl.store(
        o_ptrs,
        acc.to(o_ptr.dtype.element_ty),
        mask=valid_row[:, None] & (cols[None, :] < d),
    )
    lse_ptr = softmax_lse_ptr + hid * total_q + lse_offset + row_idx
    tl.store(lse_ptr, lse, mask=valid_row)


__all__ = [
    "flash_varlen_fwd_v3_tle_splitkv_kernel",
    "flash_varlen_fwd_v3_tle_splitkv_combine_kernel",
]
