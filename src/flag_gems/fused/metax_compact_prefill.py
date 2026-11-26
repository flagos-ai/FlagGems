# SPDX-License-Identifier: Apache-2.0
"""Pack active paged K/V sequences for a contiguous FlashAttention prefill."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_kv_block(
    K,
    V,
    Table,
    CU,
    PK,
    PV,
    entries: tl.constexpr,
    skb: tl.constexpr,
    skt: tl.constexpr,
    skh: tl.constexpr,
    svb: tl.constexpr,
    svt: tl.constexpr,
    svh: tl.constexpr,
    BS: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    first = tl.load(CU + row)
    length = tl.load(CU + row + 1) - first
    base = tile * (BS // 256)
    if base < length:
        i = tl.arange(0, BS)
        t = base + i // 256
        h = (i % 256) // 128
        d = i % 128
        page = tl.load(Table + row * entries + tile // (4096 // BS))
        ko = page * skb + (t % 16) * skt + h * skh + d
        vo = page * svb + (t % 16) * svt + h * svh + d
        mask = t < length
        to = (first + t) * 256 + h * 128 + d
        tl.store(PK + to, tl.load(K + ko, mask=mask, other=0), mask=mask)
        tl.store(PV + to, tl.load(V + vo, mask=mask, other=0), mask=mask)


def compact(k, v, table, cu, maxlen, total, block=2048):
    pk = torch.empty((total, 2, 128), device=k.device, dtype=k.dtype)
    pv = torch.empty_like(pk)
    _pack_kv_block[(triton.cdiv(maxlen, block // 256), table.shape[0])](
        k,
        v,
        table,
        cu,
        pk,
        pv,
        table.stride(0),
        *k.stride()[:3],
        *v.stride()[:3],
        BS=block,
        num_warps=4 if block <= 2048 else 8,
        num_stages=1,
    )
    return pk, pv
