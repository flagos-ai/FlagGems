"""Flash-Decoding style split-KV kernel for single-token (decode) attention.

Motivation (BI-V150): a decode step streams ~1 KiB of K/V per token per layer, yet
the generic paged-attention kernel reaches only ~75-90 GB/s (~10 % of HBM peak)
because each sequence is handled by a single program that walks the whole KV
range serially.  Splitting the KV range across programs raises the number of
resident CTAs and turns the walk into a bandwidth-bound stream.

Layout assumptions (vLLM):
  q            [total_q, H, D]        single token per sequence
  k / v        [num_pages, block, HK, D]
  block_table  [num_seqs, max_pages]  int32
  seqused_k    [num_seqs]             int32
  cu_seqlens_q [num_seqs + 1]         int32

The kernel writes partial (normalised) outputs plus log-sum-exp per split and a
second kernel merges them; both buffers are cached per shape so CUDA-graph
capture sees stable addresses.
"""

import torch
import triton
import triton.language as tl

LOG2E_CONST = 1.4426950408889634


@triton.jit
def _decode_splitkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    bt_ptr,
    sk_ptr,
    cu_ptr,
    scale,
    n_seqs,
    H: tl.constexpr,
    HK: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PAGE_TILES: tl.constexpr,
    k_page_stride: tl.constexpr,
    k_tok_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    bt_stride: tl.constexpr,
    q_row_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
):
    split = tl.program_id(0)
    seq = tl.program_id(1)
    hk = tl.program_id(2)

    q_row = tl.load(cu_ptr + seq).to(tl.int32)
    q_end = tl.load(cu_ptr + seq + 1).to(tl.int32)
    if q_end - q_row != 1:
        return

    k_len = tl.load(sk_ptr + seq).to(tl.int32)
    if PAGE_TILES:
        per = tl.cdiv(k_len, NUM_SPLITS * BLOCK_N) * BLOCK_N
    else:
        per = tl.cdiv(k_len, NUM_SPLITS)
    start = split * per
    end = tl.minimum(start + per, k_len)

    d = tl.max_contiguous(tl.multiple_of(tl.arange(0, D), D), D)
    g = tl.arange(0, BLOCK_M)

    q_base = q_ptr + q_row * q_row_stride + hk * G * q_head_stride
    if BLOCK_M == G:
        q = tl.load(q_base + g[:, None] * q_head_stride + d[None, :])
    else:
        g_mask = g < G
        q = tl.load(
            q_base + g[:, None] * q_head_stride + d[None, :],
            mask=g_mask[:, None],
            other=0.0,
        )

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    bt = bt_ptr + seq * bt_stride
    kv_hk_off = hk * k_head_stride
    LOG2E: tl.constexpr = 1.4426950408889634
    s_scale = scale * LOG2E

    if start < end:
        if PAGE_TILES:
            PAGES_PER_TILE: tl.constexpr = BLOCK_N // BLOCK_SIZE
            full_end = (end // BLOCK_N) * BLOCK_N
            for n0 in range(start, full_end, BLOCK_N):
                pages = n0 // BLOCK_SIZE + tl.arange(0, PAGES_PER_TILE)
                page_ids = tl.load(bt + pages)
                kv_off = (
                    page_ids[:, None] * k_page_stride
                    + tl.arange(0, BLOCK_SIZE)[None, :] * k_tok_stride
                ).reshape(BLOCK_N) + kv_hk_off
                if D == 128:
                    kv_off = tl.multiple_of(kv_off, 128)
                kk = tl.load(k_ptr + kv_off[:, None] + d[None, :])
                vv = tl.load(v_ptr + kv_off[:, None] + d[None, :])
                s = tl.dot(q, tl.trans(kk))
                m_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.exp2((m_i - m_new) * s_scale)
                p = tl.exp2((s - m_new[:, None]) * s_scale)
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(vv.dtype), vv, acc)
                m_i = m_new

            if full_end < end:
                cols = full_end + tl.arange(0, BLOCK_N)
                col_mask = cols < end
                pages = full_end // BLOCK_SIZE + tl.arange(0, PAGES_PER_TILE)
                page_ids = tl.load(bt + pages, mask=pages * BLOCK_SIZE < end, other=0)
                kv_off = (
                    page_ids[:, None] * k_page_stride
                    + tl.arange(0, BLOCK_SIZE)[None, :] * k_tok_stride
                ).reshape(BLOCK_N) + kv_hk_off
                if D == 128:
                    kv_off = tl.multiple_of(kv_off, 128)
                kk = tl.load(
                    k_ptr + kv_off[:, None] + d[None, :],
                    mask=col_mask[:, None],
                    other=0.0,
                )
                s = tl.dot(q, tl.trans(kk))
                s = tl.where(col_mask[None, :], s, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.exp2((m_i - m_new) * s_scale)
                p = tl.exp2((s - m_new[:, None]) * s_scale)
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None]
                vv = tl.load(
                    v_ptr + kv_off[:, None] + d[None, :],
                    mask=col_mask[:, None],
                    other=0.0,
                )
                acc = tl.dot(p.to(vv.dtype), vv, acc)
                m_i = m_new
        else:
            for n0 in tl.range(start, end, BLOCK_N):
                cols = n0 + tl.arange(0, BLOCK_N)
                col_mask = cols < end
                page = tl.load(bt + (cols // BLOCK_SIZE), mask=col_mask, other=0).to(
                    tl.int64
                )
                slot = cols % BLOCK_SIZE
                kv_off = page * k_page_stride + slot * k_tok_stride + kv_hk_off
                kk = tl.load(
                    k_ptr + kv_off[:, None] + d[None, :],
                    mask=col_mask[:, None],
                    other=0.0,
                )
                s = tl.dot(q, tl.trans(kk))
                s = tl.where(col_mask[None, :], s, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.exp2((m_i - m_new) * s_scale)
                p = tl.exp2((s - m_new[:, None]) * s_scale)
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None]
                vv = tl.load(
                    v_ptr + kv_off[:, None] + d[None, :],
                    mask=col_mask[:, None],
                    other=0.0,
                )
                acc = tl.dot(p.to(vv.dtype), vv, acc)
                m_i = m_new

    has = l_i > 0
    inv = tl.where(has, 1.0 / l_i, 0.0)
    o = acc * inv[:, None]
    lse = tl.where(has, m_i * scale + tl.log(tl.where(has, l_i, 1.0)), float("-inf"))

    base = ((split * n_seqs + seq) * HK + hk) * G
    if BLOCK_M == G:
        tl.store(o_ptr + (base + g[:, None]) * D + d[None, :], o)
        tl.store(lse_ptr + base + g, lse)
    else:
        g_mask = g < G
        tl.store(o_ptr + (base + g[:, None]) * D + d[None, :], o, mask=g_mask[:, None])
        tl.store(lse_ptr + base + g, lse, mask=g_mask)


@triton.jit
def _decode_merge_kernel(
    o_ptr,
    lse_ptr,
    out_ptr,
    cu_ptr,
    n_seqs,
    H: tl.constexpr,
    HK: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    o_row_stride: tl.constexpr,
    o_head_stride: tl.constexpr,
    final_lse_ptr,
    total_q: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    seq = tl.program_id(0)
    hk = tl.program_id(1)
    q_row = tl.load(cu_ptr + seq).to(tl.int32)
    q_end = tl.load(cu_ptr + seq + 1).to(tl.int32)
    if q_end - q_row != 1:
        return
    g = tl.arange(0, G)
    d = tl.arange(0, D)

    m = tl.full([G], float("-inf"), tl.float32)
    row_sum = tl.zeros([G], tl.float32)
    acc = tl.zeros([G, D], tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        base = ((s * n_seqs + seq) * HK + hk) * G
        lse = tl.load(lse_ptr + base + g)
        o = tl.load(o_ptr + (base + g[:, None]) * D + d[None, :])
        m_new = tl.maximum(m, lse)
        safe_m = tl.where(m_new == float("-inf"), 0.0, m_new)
        a = tl.exp2((m - safe_m) * 1.4426950408889634)
        p = tl.exp2((lse - safe_m) * 1.4426950408889634)
        acc = acc * a[:, None] + o * p[:, None]
        row_sum = row_sum * a + p
        m = m_new

    l_safe = tl.where(row_sum > 0, row_sum, 1.0)
    out = acc / l_safe[:, None]
    tl.store(
        out_ptr
        + q_row * o_row_stride
        + (hk * G + g[:, None]) * o_head_stride
        + d[None, :],
        out,
    )

    if STORE_LSE:
        final_lse = tl.where(row_sum > 0, m + tl.log(l_safe), float("-inf"))
        tl.store(final_lse_ptr + (hk * G + g) * total_q + q_row, final_lse)


@triton.jit
def _parallel_merge(
    O,
    LSE,
    OUT,
    CU,
    FINAL_LSE,
    batch,
    stride_row,
    stride_head,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    WIDTH: tl.constexpr,
    TOTAL: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    group = tl.program_id(2)
    row = tl.load(CU + seq).to(tl.int32)
    if tl.load(CU + seq + 1) - row != 1:
        return
    split = tl.arange(0, S)
    cols = group * WIDTH + tl.arange(0, WIDTH)
    base = (split * batch + seq) * H + head
    lse = tl.load(LSE + base)
    peak = tl.max(lse, 0)
    safe = tl.where(peak == -float("inf"), 0.0, peak)
    weights = tl.exp2((lse - safe) * 1.4426950408889634)
    total = tl.sum(weights, 0)
    partial = tl.load(O + base[:, None] * D + cols[None, :])
    result = tl.sum(weights[:, None] * partial, 0) / tl.where(total > 0, total, 1.0)
    tl.store(OUT + row * stride_row + head * stride_head + cols, result)
    if WRITE_LSE:
        if group == 0:
            tl.store(
                FINAL_LSE + head * TOTAL + row,
                tl.where(total > 0, peak + tl.log(total), -float("inf")),
            )


_RESERVED = {}


def reserve_decode_workspace(device, hk, g, d):
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, hk, g, d)
    if key not in _RESERVED:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Reserve decode workspace before graph capture")
        _RESERVED[key] = (
            torch.empty((512, hk, g, d), dtype=torch.float32, device=device),
            torch.empty((512, hk, g), dtype=torch.float32, device=device),
        )
    return _RESERVED[key]


def get_decode_workspace(device, hk, g, d):
    return _RESERVED.get((device.index, hk, g, d))


_BUFFER_CACHE = {}


def _get_buffers(n_seqs, num_splits, hk, g, d, device):
    key = (n_seqs, num_splits, hk, g, d, device.index)
    buf = _BUFFER_CACHE.get(key)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Split-KV buffers must be warmed before graph capture")
        o = torch.empty(
            (num_splits, n_seqs, hk, g, d), dtype=torch.float32, device=device
        )
        lse = torch.empty(
            (num_splits, n_seqs, hk, g), dtype=torch.float32, device=device
        )
        buf = (o, lse)
        _BUFFER_CACHE[key] = buf
    return buf


def flash_decode_splitkv(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    softmax_scale,
    num_splits=8,
    block_n=64,
    num_warps=4,
    num_stages=1,
    workspace=None,
    lse_out=None,
    page_tiles=False,
    block_m=None,
    merge_width=0,
):
    """Single-token paged attention with KV-range splits (Flash-Decoding)."""
    total_q, H, D = q.shape
    HK = k.shape[2]
    G = H // HK
    n_seqs = seqused_k.shape[0]
    if workspace is None:
        o_buf, lse_buf = _get_buffers(n_seqs, num_splits, HK, G, D, q.device)
    else:
        o_buf, lse_buf = workspace
        assert o_buf.numel() >= n_seqs * num_splits * HK * G * D
        assert lse_buf.numel() >= n_seqs * num_splits * HK * G

    bm = (
        block_m
        if block_m is not None
        else (8 if (page_tiles and G == 8 and D >= 32) else 16)
    )
    merge_warps = 2 if (G <= 8 and D <= 128) else 4

    _decode_splitkv_kernel[(num_splits, n_seqs, HK)](
        q,
        k,
        v,
        o_buf,
        lse_buf,
        block_table,
        seqused_k,
        cu_seqlens_q,
        softmax_scale,
        n_seqs,
        H=H,
        HK=HK,
        G=G,
        D=D,
        BLOCK_M=bm,
        BLOCK_N=block_n,
        BLOCK_SIZE=k.shape[1],
        NUM_SPLITS=num_splits,
        PAGE_TILES=page_tiles,
        k_page_stride=k.stride(0),
        k_tok_stride=k.stride(1),
        k_head_stride=k.stride(2),
        bt_stride=block_table.stride(0),
        q_row_stride=q.stride(0),
        q_head_stride=q.stride(1),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if merge_width:
        _parallel_merge[(n_seqs, H, D // merge_width)](
            o_buf,
            lse_buf,
            out,
            cu_seqlens_q,
            lse_out,
            n_seqs,
            out.stride(0),
            out.stride(1),
            H,
            D,
            num_splits,
            merge_width,
            total_q,
            lse_out is not None,
            num_warps=2,
        )
        return out
    _decode_merge_kernel[(n_seqs, HK)](
        o_buf,
        lse_buf,
        out,
        cu_seqlens_q,
        n_seqs,
        H=H,
        HK=HK,
        G=G,
        D=D,
        NUM_SPLITS=num_splits,
        o_row_stride=out.stride(0),
        o_head_stride=out.stride(1),
        final_lse_ptr=lse_out,
        total_q=total_q,
        STORE_LSE=lse_out is not None,
        num_warps=merge_warps,
    )
    return out
