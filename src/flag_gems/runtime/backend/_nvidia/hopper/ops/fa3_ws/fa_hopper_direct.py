"""Direct one-pass FA3 TLE kernel family."""

from .utils import *  # noqa: F401,F403
def _fa3_direct_configs():
    configs = []
    for block_m in (16, 32, 64, 128):
        for block_n in (32, 64, 128, 256):
            stage_choices = (2, 3) if block_n <= 128 else (3,)
            warp_choices = (4,)
            if block_n >= 64 and (block_m >= 64 or block_m == 16):
                warp_choices = (4, 8)
            for num_stages in stage_choices:
                for num_warps in warp_choices:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n},
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
    return configs


def _prune_fa3_direct_configs(configs, nargs, **kwargs):
    head_dim = kwargs.get("d", nargs.get("d"))
    is_paged = kwargs.get("is_paged", nargs.get("is_paged"))
    seqlen_k = kwargs.get("seqlen_k", nargs.get("seqlen_k", 0))
    shape_bucket = kwargs.get(
        "DIRECT_SHAPE_BUCKET",
        nargs.get("DIRECT_SHAPE_BUCKET", _FA3_TLE_BUCKET_DIRECT_SMALL),
    )

    kept = []
    for cfg in configs:
        block_m = cfg.kwargs["BLOCK_M"]
        block_n = cfg.kwargs["BLOCK_N"]
        if shape_bucket in (
            _FA3_TLE_BUCKET_DIRECT_DECODE,
            _FA3_TLE_BUCKET_DIRECT_PAGED_DECODE,
        ):
            if block_m != 16 or block_n < 64:
                continue
            if head_dim >= 192 and block_n > 128:
                continue
            if is_paged and block_n > 128:
                continue
        else:
            if block_n > 128:
                continue
            if block_m < 64:
                continue
            if seqlen_k <= 128 and block_n < 64:
                continue
            if head_dim > 128 and block_n > 64:
                continue
        if head_dim > 192 and block_n > 128:
            continue
        kept.append(cfg)

    if kept:
        return kept
    return [configs[0]]



@libentry()
@triton.autotune(
    configs=_fa3_direct_configs(),
    prune_configs_by={"early_config_prune": _prune_fa3_direct_configs},
    key=[
        "d",
        "is_paged",
        "is_causal",
        "is_local",
        "is_alibi",
        "seqlen_q",
        "seqlen_k",
        "total_q",
        "DIRECT_SHAPE_BUCKET",
        "MIN_Q_LEN_TO_PROCESS",
        "MAX_Q_LEN_TO_PROCESS",
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
def flash_varlen_fwd_v3_tle_direct_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIRECT_SHAPE_BUCKET: tl.constexpr,
    MIN_Q_LEN_TO_PROCESS: tl.constexpr,
    MAX_Q_LEN_TO_PROCESS: tl.constexpr,
    PAGED_GATHER_MODE: tl.constexpr = 2,
):
    m_block = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)
    HEAD_DIM_PADDED: tl.constexpr = BLOCK_K

    if is_cu_seqlens_q:
        q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
        q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
        q_len = q_eos - q_bos
        q_offset = q_bos * q_row_stride
        o_offset = q_bos * o_row_stride
        lse_offset = q_bos
    else:
        q_len = seqlen_q
        q_offset = bid * q_batch_stride
        o_offset = bid * o_batch_stride
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

    process_q = (q_len >= MIN_Q_LEN_TO_PROCESS) & (q_len <= MAX_Q_LEN_TO_PROCESS)
    process_q = process_q & (m_block * BLOCK_M < q_len)
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

        if is_alibi:
            alibi_slope = tl.load(
                alibi_slopes_ptr + bid * alibi_slopes_batch_stride + hid
            )
            alibi_slope = alibi_slope / scale_softmax
        else:
            alibi_slope = 0.0

        q_base = q_ptr + q_offset + hid * q_head_stride
        o_base = o_ptr + o_offset + hid * o_head_stride
        kv_head = hid // h_hk_ratio
        if is_paged:
            page_table_ptr_b = page_table_ptr + bid * page_table_batch_stride
            k_base = k_ptr + kv_head * k_head_stride
            v_base = v_ptr + kv_head * v_head_stride
        else:
            k_base = k_ptr + k_bos * k_row_stride + kv_head * k_head_stride
            v_base = v_ptr + k_bos * v_row_stride + kv_head * v_head_stride

        row_idx_q = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        d_idx = tl.arange(0, HEAD_DIM_PADDED)
        q_tile = tl.load(
            q_base + row_idx_q[:, None] * q_row_stride + d_idx[None, :],
            mask=(row_idx_q[:, None] < q_len) & (d_idx[None, :] < d),
            other=0.0,
        )
        rowmax = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        rowsum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

        if is_causal or is_local:
            n_masking_steps = tl.cdiv(BLOCK_M + window_size_right + 1, BLOCK_N)
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
            if is_paged:
                cache_idx = _paged_blockwise_cache_indices(
                    n_block * BLOCK_N,
                    tl.arange(0, BLOCK_N),
                    k_len,
                    page_table_ptr_b,
                    block_size,
                    BLOCK_N,
                    PAGED_GATHER_MODE,
                    BOUNDARY_CHECK=False,
                )
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

        invalid = (rowsum == 0) | (rowsum != rowsum)
        inv_sum = tl.where(invalid, 1.0, 1.0 / rowsum)
        acc = acc * inv_sum[:, None]
        lse = tl.where(
            invalid,
            float("inf"),
            rowmax * scale_softmax + tl.log(rowsum),
        )
        tl.store(
            o_base + row_idx_q[:, None] * o_row_stride + d_idx[None, :],
            acc.to(o_ptr.dtype.element_ty),
            mask=(row_idx_q[:, None] < q_len) & (d_idx[None, :] < d),
        )
        lse_ptr = softmax_lse_ptr + hid * total_q + lse_offset + row_idx_q
        tl.store(lse_ptr, lse, mask=row_idx_q < q_len)



__all__ = ["flash_varlen_fwd_v3_tle_direct_kernel"]
