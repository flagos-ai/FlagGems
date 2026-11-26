# SPDX-License-Identifier: Apache-2.0
"""Gather one paged history for reuse by the long-query BF16 GQA tiles."""

import threading

import torch
import triton
import triton.language as tl


@triton.jit
def _long_query(CU, LEN, META, batch, B: tl.constexpr, SECOND: tl.constexpr):
    seq = tl.arange(0, B)
    first = tl.load(CU + seq, seq < batch, 0)
    end = tl.load(CU + seq + 1, seq < batch, 0)
    longest = tl.argmax(end - first, 0)
    start = tl.load(CU + longest)
    tl.store(META, longest)
    tl.store(META + 1, start)
    tl.store(META + 2, tl.load(CU + longest + 1) - start)
    tl.store(META + 3, tl.load(LEN + longest))
    if SECOND:
        remaining = tl.where((seq < batch) & (seq != longest), end - first, -1)
        other = tl.argmax(remaining, 0)
        qlen = tl.max(remaining, 0)
        active = qlen >= 512
        tl.store(META + 4, tl.where(active, other, -1))
        tl.store(META + 5, tl.load(CU + other))
        tl.store(META + 6, tl.where(active, qlen, 0))
        tl.store(META + 7, tl.where(active, tl.load(LEN + other), 0))
    else:
        tl.store(META + 4, -1)


@triton.jit
def _pack(
    K,
    V,
    BT,
    META,
    PK,
    PV,
    page_stride: tl.constexpr,
    row_stride: tl.constexpr,
    head_stride: tl.constexpr,
    table_stride,
    capacity: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.load(META)
    length = tl.load(META + 3)
    if tile * N >= tl.cdiv(length, 128) * 128:
        return
    logical = tile * (N // PAGE) + tl.arange(0, N // PAGE)
    pages = tl.load(BT + seq * table_stride + logical, logical * PAGE < length, 0)
    slot = (
        pages[:, None].to(tl.int64) * page_stride
        + tl.arange(0, PAGE)[None, :] * row_stride
    ).reshape(N) + head * head_stride
    dim = tl.arange(0, D)
    token = tile * N + tl.arange(0, N)
    src = slot[:, None] + dim[None, :]
    k = tl.load(K + src, token[:, None] < length, 0.0)
    v = tl.load(V + src, token[:, None] < length, 0.0)
    off = head * capacity * D
    tl.store(PK + off + dim[:, None] * capacity + token[None, :], tl.trans(k))
    tl.store(PV + off + token[:, None] * D + dim[None, :], v)


@triton.jit
def _accumulate(
    q,
    K,
    V,
    t,
    pos,
    history,
    length,
    m,
    row_sum,
    acc,
    scale2,
    capacity: tl.constexpr,
    MASK: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
):
    dim = tl.arange(0, D)
    token = t * N + tl.arange(0, N)
    key = tl.load(K + dim[:, None] * capacity + token[None, :])
    value = tl.load(V + token[:, None] * D + dim[None, :])
    score = tl.dot(q, key)
    if MASK:
        valid = (token[None, :] <= history + pos[:, None]) & (token[None, :] < length)
        score = tl.where(valid, score, -float("inf"))
    peak = tl.maximum(m, tl.max(score, 1))
    safe = tl.where(peak == -float("inf"), 0.0, peak)
    adjust = tl.exp2((m - safe) * scale2)
    p = tl.exp2(score * scale2 - safe[:, None] * scale2)
    acc = tl.dot(p.to(value.dtype), value, acc * adjust[:, None])
    return peak, row_sum * adjust + tl.sum(p, 1), acc


@triton.jit
def _causal(
    Q,
    K,
    V,
    OUT,
    LSE,
    META,
    scale,
    qstride: tl.constexpr,
    hstride: tl.constexpr,
    ostride: tl.constexpr,
    ohstride: tl.constexpr,
    total_q,
    capacity: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    tile = tl.program_id(0)
    hk = tl.program_id(1)
    rows: tl.constexpr = M // 8
    base = tl.load(META + 1)
    qlen = tl.load(META + 2)
    length = tl.load(META + 3)
    if tile * rows >= qlen:
        return
    lanes = tl.arange(0, M)
    pos = tile * rows + lanes // 8
    head = hk * 8 + lanes % 8
    dims = tl.arange(0, D)
    q = tl.load(
        Q + ((base + pos) * qstride + head * hstride)[:, None] + dims[None, :],
        pos[:, None] < qlen,
        0.0,
    )
    key = K + hk * capacity * D
    value = V + hk * capacity * D
    m = tl.full((M,), -float("inf"), tl.float32)
    row_sum = tl.full((M,), 0.0, tl.float32)
    acc = tl.full((M, D), 0.0, tl.float32)
    history = length - qlen
    interior = tl.maximum(0, (history + tile * rows + 1) // N)
    end = tl.cdiv(tl.minimum(history + (tile + 1) * rows, length), N)
    scale2 = scale * 1.4426950408889634
    for t in range(interior):
        m, row_sum, acc = _accumulate(
            q,
            key,
            value,
            t,
            pos,
            history,
            length,
            m,
            row_sum,
            acc,
            scale2,
            capacity,
            False,
            N,
            D,
        )
    for t in range(interior, end):
        m, row_sum, acc = _accumulate(
            q,
            key,
            value,
            t,
            pos,
            history,
            length,
            m,
            row_sum,
            acc,
            scale2,
            capacity,
            True,
            N,
            D,
        )
    norm = tl.where(row_sum > 0.0, row_sum, 1.0)
    row = (base + pos) * ostride + head * ohstride
    tl.store(
        OUT + row[:, None] + dims[None, :], acc / norm[:, None], pos[:, None] < qlen
    )
    if WRITE_LSE:
        tl.store(
            LSE + head * total_q + base + pos,
            tl.where(row_sum > 0.0, m * scale + tl.log(norm), -float("inf")),
            pos < qlen,
        )


_reserved = {}


def _owner(device):
    return (
        device.index,
        torch.cuda.current_stream(device).cuda_stream,
        threading.get_ident(),
    )


def reserve_prefill_workspace(device, max_capacity=32768, d=128):
    key = _owner(device)
    if key not in _reserved:
        _reserved[key] = (
            torch.empty((4 * max_capacity * d,), dtype=torch.bfloat16, device=device),
            torch.empty(8, dtype=torch.int32, device=device),
        )


def dense_long_query(q, k, v, out, lse, cu, lengths, table, max_q, max_k, scale):
    capacity = triton.cdiv(max_k, 2048) * 2048
    size = 4 * capacity * 128
    owner = _owner(q.device)
    pair = _reserved.get(owner)
    if (
        pair is None
        and capacity <= 32768
        and not torch.cuda.is_current_stream_capturing()
    ):
        reserve_prefill_workspace(q.device)
        pair = _reserved[owner]
    if pair is None or pair[0].numel() < size:
        pair = (
            k.new_empty((size,)),
            torch.empty(8, device=q.device, dtype=torch.int32),
        )
    data, meta = pair
    pk, pv = data[:size].view(2, 2, capacity, 128).unbind(0)
    batch = lengths.numel()
    two = batch > 1 and max_q >= 512 and q.shape[0] - max_q >= 512
    _long_query[(1,)](
        cu, lengths, meta, batch, triton.next_power_of_2(batch), two, num_warps=1
    )
    m, warps = (128, 8) if max_q < 512 else (256, 16)
    for index in range(2 if two else 1):
        selected = meta[index * 4 :]
        _pack[(triton.cdiv(max_k, 128) * 4, 2)](
            k,
            v,
            table,
            selected,
            pk,
            pv,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            table.stride(0),
            capacity,
            32,
            128,
            16,
            num_warps=4,
        )
        _causal[(triton.cdiv(max_q, m // 8), 2)](
            q,
            pk,
            pv,
            out,
            lse,
            selected,
            scale,
            q.stride(0),
            q.stride(1),
            out.stride(0),
            out.stride(1),
            q.size(0),
            capacity,
            m,
            128,
            128,
            lse is not None,
            num_warps=warps,
            num_stages=1,
        )
    return meta


def borrow_prefill_float(device, rows, width):
    """A view valid until the next operation on this thread and device stream."""
    pair = _reserved.get(_owner(device))
    if pair is None or pair[0].numel() < 2 * rows * width:
        return None
    return pair[0].view(torch.float32)[: rows * width].view(rows, width)
