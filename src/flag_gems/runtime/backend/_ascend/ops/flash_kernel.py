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

# Ascend specialization of the shared varlen kernel. Keep fixes local to this
# backend; the shared helpers do not choose tiles or launch kernels.
import triton
import triton.language as tl

from flag_gems.ops.flash_kernel import (
    apply_alibi,
    apply_mask,
    apply_softcap,
    load_from_kvcache,
    softmax_rescale,
)
from flag_gems.utils import libentry

from .exponential import _mulhi_u32_limb


@triton.jit
def _attention_dropout(
    probabilities,
    row_idx,
    col_start,
    k_len,
    bid,
    hid,
    seed,
    offset,
    keep_threshold: tl.constexpr,
    encode_sign: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Match the shared attention RNG with 32-bit products. Ascend cannot
    # lower its vector uint32 -> uint64 casts and 64-bit multiplications.
    row = row_idx[:, None].to(tl.uint32)
    columns = col_start + tl.arange(0, BLOCK_N)[None, :]
    col = (columns // 4).to(tl.uint32)
    low = row * k_len.to(tl.uint32)
    high = _mulhi_u32_limb(row, k_len.to(tl.uint32))
    subsequence_low = low + col
    carry = ((low & col) | ((low | col) & ~subsequence_low)) >> 31
    subsequence_high = high + carry
    offset = offset + bid * NUM_HEADS + hid
    k0 = (seed >> 32).to(tl.uint32)
    k1 = seed.to(tl.uint32)
    c0 = (offset >> 32).to(tl.uint32) + tl.full((BLOCK_M, BLOCK_N), 0, tl.uint32)
    c1 = offset.to(tl.uint32) + tl.full((BLOCK_M, BLOCK_N), 0, tl.uint32)
    c2 = subsequence_high
    c3 = subsequence_low
    for _ in tl.static_range(7):
        hi0 = _mulhi_u32_limb(c0, 0xD2511F53)
        lo0 = 0xD2511F53 * c0
        hi1 = _mulhi_u32_limb(c2, 0xCD9E8D57)
        lo1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = lo1 ^ c1 ^ k0, hi1, lo0 ^ c3 ^ k1, hi0
        k0 += 0x9E3779B9
        k1 += 0xBB67AE85
    # Select each lane in two dimensions; join/reshape would leave unsupported
    # four-dimensional UB buffers after the Ascend compiler's layout passes.
    lane = columns % 4
    random = tl.where(
        lane == 0, c0, tl.where(lane == 1, c2, tl.where(lane == 2, c1, c3))
    )
    dropped = (random & 0xFF).to(tl.int32) >= keep_threshold
    if encode_sign:
        return tl.where(dropped, -probabilities, probabilities)
    return tl.where(dropped, 0, probabilities).to(probabilities.dtype)


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
    k_page_stride,
    # kernel params
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    DENSE: tl.constexpr = False,
    SPLIT_KV: tl.constexpr = False,
    blocks_per_split: tl.constexpr = 0,
):
    m_block = tl.program_id(0)
    if SPLIT_KV:
        split_id = tl.program_id(1)
        bid = tl.program_id(2) // h
        hid = tl.program_id(2) % h
    else:
        bid = tl.program_id(1)
        hid = tl.program_id(2)
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
        k_offset = k_bos * k_row_stride
        v_offset = k_bos * v_row_stride
    else:
        k_len_cache = seqlen_k
        k_offset = bid * k_batch_stride
        v_offset = bid * v_batch_stride

    if is_seqused_k:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    else:
        k_len = k_len_cache

    # Noop CTA
    if m_block * BLOCK_M >= q_len:
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

    if SPLIT_KV:
        n_block_min = max(n_block_min, split_id * blocks_per_split)
        n_block_max = min(n_block_max, (split_id + 1) * blocks_per_split)

    if is_dropout:
        philox_seed = tl.load(philox_args).to(tl.uint64)
        philox_offset = tl.load(philox_args + 1).to(tl.uint64)

    # Locate the page table entry for the current batch element
    if is_paged:
        page_table_ptr += bid * page_table_batch_stride
    # Calculate the starting offset of q for the current head
    q_row_offset = hid * q_head_stride
    # Calculate the starting offset of k and v for the current head
    k_row_offset = (hid // h_hk_ratio) * k_head_stride
    # Shift the k, v pointers to align with the current head
    k_ptr_base = k_ptr + k_row_offset
    v_ptr_base = v_ptr + (hid // h_hk_ratio) * v_head_stride

    gQ = tl.make_block_ptr(
        base=q_ptr + q_offset + q_row_offset,
        shape=(q_len, d),
        strides=(q_row_stride, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    bQ = tl.load(
        gQ.advance([m_block * BLOCK_M, 0]),
        boundary_check=(0, 1),
        padding_option="zero",
    )

    acc_ = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0.0

    # Keep the same masked, row-major load layout for every KV tile on NPU.
    # This also covers padded head dimensions and tail sequence tiles.
    n_masking_steps = max(0, n_block_max - n_block_min)

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
                k_page_stride=k_page_stride,
                boundary_check=True,
            )
        else:
            start_n = n_block * BLOCK_N
            k_ptr_seq = k_ptr_base + k_offset
            v_ptr_seq = v_ptr_base + v_offset
            # The head dimension has unit stride. Ascend requires the
            # block-pointer order to match this row-major memory layout.
            gK = tl.make_block_ptr(
                base=k_ptr_seq,
                shape=(k_len, d),
                strides=(k_row_stride, 1),
                offsets=(start_n, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            gV = tl.make_block_ptr(
                base=v_ptr_seq,
                shape=(k_len, d),
                strides=(v_row_stride, 1),
                offsets=(start_n, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            bK = tl.load(gK, boundary_check=(0, 1), padding_option="zero")
            bK = tl.trans(bK)
            bV = tl.load(gV, boundary_check=(0, 1), padding_option="zero")
        S = tl.dot(bQ, bK, out_dtype=tl.float32)
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
        P = P.to(v_ptr.type.element_ty)

        if is_dropout:
            P = _attention_dropout(
                P,
                row_idx,
                n_block * BLOCK_N,
                k_len,
                bid,
                hid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                encode_sign=return_softmax,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

            if return_softmax:
                p_offset = (
                    (bid * h + hid) * seqlen_q_rounded + row_idx[:, None]
                ) * seqlen_k_rounded + col_idx[None, :]
                tl.store(
                    p_ptr + p_offset,
                    P,
                    mask=(row_idx[:, None] < q_len) & (col_idx[None, :] < k_len),
                )
                P = tl.maximum(P, 0.0).to(v_ptr.type.element_ty)

        acc_ = tl.dot(P, bV, acc_)
        n_block -= 1

    # LSE
    lse = tl.where(
        (rowsum_ == 0) | (rowsum_ != rowsum_),
        float("-inf") if SPLIT_KV else float("inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_),
    )
    inv_sum = tl.where((rowsum_ == 0) | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

    if is_dropout:
        inv_sum *= rp_dropout
    acc_ *= inv_sum[:, None]

    out = acc_.to(o_ptr.type.element_ty)  # noqa

    if SPLIT_KV:
        # Partial buffers use [split, batch, head, query, dim].
        rows = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, BLOCK_K)
        partial_row = ((split_id * b + bid) * h + hid) * seqlen_q + rows
        tl.store(
            o_ptr + partial_row[:, None] * d + dims[None, :],
            acc_,
            mask=(rows[:, None] < q_len) & (dims[None, :] < d),
        )
        tl.store(softmax_lse_ptr + partial_row, lse, mask=rows < q_len)
    else:
        o_row_offset = hid * o_head_stride
        gO = tl.make_block_ptr(
            base=o_ptr + o_offset + o_row_offset,
            shape=(q_len, d),
            strides=(o_row_stride, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        tl.store(gO.advance([m_block * BLOCK_M, 0]), out, boundary_check=(0, 1))

        # Dense LSE is [batch, head, query]; packed LSE is [head, total_q].
        if DENSE:
            lse_offset = (bid * h + hid) * seqlen_q
        else:
            lse_offset += hid * total_q
        rows = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        tl.store(softmax_lse_ptr + lse_offset + rows, lse, mask=rows < q_len)


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
    num_heads: tl.constexpr,
    seqlen_q: tl.constexpr,
):
    pid = tl.program_id(0)
    lse_splits_ptr += pid * BLOCK_M
    lse_ptr += pid * BLOCK_M
    out_splits_ptr += pid * BLOCK_M * head_size

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
    # Fully masked rows have -inf in every split; avoid inf - inf.
    max_lse = tl.where(max_lse == float("-inf"), 0.0, max_lse)
    Zi_scaled = tl.exp(lse_splits - max_lse[:, None])
    Z_scaled = tl.sum(Zi_scaled, 1)
    Zi_Z = Zi_scaled / tl.where(Z_scaled == 0, 1.0, Z_scaled)[:, None]

    # Write back LSE
    lse = tl.where(Z_scaled == 0, float("inf"), tl.log(Z_scaled) + max_lse)
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

    # Write back output.
    # The split buffers flatten [batch, head, query], but `out` is a BSHD view
    # whose batch/head/query strides need not follow that storage order. Decode
    # each row into batch/head/query and use the real output strides, instead
    # of advancing the base by a contiguous row pitch.
    row = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    batch = row // (num_heads * seqlen_q)
    head = (row // seqlen_q) % num_heads
    query = row % seqlen_q
    out_offset = (
        batch[:, None] * out_b_stride
        + head[:, None] * out_h_stride
        + query[:, None] * out_s_stride
        + tl.arange(0, BLOCK_K)[None, :]
    )
    dmask = tl.arange(0, BLOCK_K) < head_size
    tl.store(out_ptr + out_offset, out, mask=out_mask[:, None] & dmask[None, :])
