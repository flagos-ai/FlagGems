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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _stage(key, lanes, STRIDE: tl.constexpr, SIZE: tl.constexpr, SHIFT: tl.constexpr):
    """One network stage on a [RPP, 16] int32 lane word.

    Identical compare-exchange to the float32 form of this kernel, evaluated on
    the dense value rank: rank_i = #{x_j > x_i} counts strictly greater
    elements, so rank order is the opposite of value order (val < peer <=>
    rank(val) > rank(peer)) and equal values share a rank. peer = lane i ^
    STRIDE (single lane gather); better = strict compare of the low lane against
    the high lane, takePeer = better XOR the pair direction, which is uniform
    inside a pair.
    """
    perm = (lanes ^ STRIDE) + tl.zeros_like(key)
    pk = tl.gather(key, perm, axis=1)
    kv = key >> SHIFT
    pkv = pk >> SHIFT
    is_low = (lanes & STRIDE) == 0
    lo_k = tl.where(is_low, kv, pkv)
    hi_k = tl.where(is_low, pkv, kv)
    better = lo_k < hi_k
    take = better == ((lanes & SIZE) != 0)
    return tl.where(take, pk, key)


@triton.jit
def _topk16_kernel(
    X,
    VALS,
    IDX,
    n_rows,
    m0,
    s0,
    m1,
    s1,
    m2,
    s2,
    s_last,
    N: tl.constexpr,
    K: tl.constexpr,
    RPP: tl.constexpr,
    SHIFT: tl.constexpr,
    LARGEST: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * RPP + tl.arange(0, RPP)
    rmask = rows < n_rows

    i0r = rows % m0
    t = rows // m0
    i1r = t % m1
    t = t // m1
    i2r = t % m2
    roff = i0r * s0 + i1r * s1 + i2r * s2

    cols = tl.arange(0, N)
    x = tl.load(
        X + roff[:, None] + cols[None, :] * s_last,
        mask=rmask[:, None],
        other=0.0,
    )
    if LARGEST:
        xs = x
    else:
        xs = -x

    # dense value rank: vrank_i = #{j : x_j > x_i}; equal values share a rank,
    # so ordering and equality on vrank are ordering and equality on the value.
    # The count never exceeds N-1, so it is accumulated in the narrowest exact
    # dtype: the [RPP,N,N] count tile is the widest live tile in the program and
    # its footprint, not its op count, is what this backend charges for.
    # PyTorch orders NaNs above finite values. Resolve cutoff ties by input
    # order; equal sorted values may have any valid distinct indices.
    is_nan = x != x
    both_valid = (~is_nan[:, None, :]) & (~is_nan[:, :, None])
    better = both_valid & (xs[:, None, :] > xs[:, :, None])
    if LARGEST:
        better |= is_nan[:, None, :] & (~is_nan[:, :, None])
    else:
        better |= (~is_nan[:, None, :]) & is_nan[:, :, None]
    vrank = tl.sum(better.to(tl.int32), axis=2)
    cutoff = tl.max(tl.where(vrank < K, vrank, -1), axis=1)
    gt = vrank < cutoff[:, None]
    eq = vrank == cutoff[:, None]
    gti = gt.to(tl.int32)
    eqi = eq.to(tl.int32)
    n_gt = tl.sum(gti, axis=1)
    # One prefix count instead of two: the exclusive counts of "greater" and
    # "equal" both fit in a byte (each <= N <= 64), so they can be carried in a
    # single int32 scan with no carry between the halves.
    csum = tl.cumsum(gti * 256 + eqi, axis=1)
    prefix_gt = (csum >> 8) - gti
    prefix_eq = (csum & 255) - eqi
    slot = tl.where(gt, prefix_gt, n_gt[:, None] + prefix_eq)
    selected = gt | (eq & (slot < K))

    # one packed lane word per element: value rank in the high bits, element
    # column in the low SHIFT bits (distinct keys, the key compare masks the
    # column). The word is carried in the narrowest exact unsigned type: every
    # stage's vector block count scales with the lane byte width, so a smaller
    # lane word is cheaper, and 8 bits suffice whenever rank and column each fit
    # in four bits (K == N == 16).
    if 2 * SHIFT <= 8:
        key = ((vrank << SHIFT) | cols[None, :]).to(tl.uint8)
    else:
        key = ((vrank << SHIFT) | cols[None, :]).to(tl.int16)

    lanes = tl.arange(0, 16)[None, :]
    lanes_k = tl.arange(0, 16)[:, None]
    # [RPP, K, N] so the placement reduction runs over the innermost axis;
    # reducing a [RPP, N, K] tile over the middle axis is lowered as a serial
    # per-element loop with packed-predicate unpacking. When K == N every element
    # is a candidate, so the candidate mask is redundant.
    if K == N:
        onehot = slot[:, None, :] == lanes_k[None, :, :]
    else:
        onehot = selected[:, None, :] & (slot[:, None, :] == lanes_k[None, :, :])
    if 2 * SHIFT <= 8:
        lane_key = tl.sum(tl.where(onehot, key[:, None, :].to(tl.int32), 0), axis=2).to(
            tl.uint8
        )
    else:
        lane_key = tl.sum(tl.where(onehot, key[:, None, :].to(tl.int32), 0), axis=2).to(
            tl.int16
        )

    # 14 stages: 10-stage bitonic sort of 16 lanes + 4-stage bitonic merge.
    lane_key = _stage(lane_key, lanes, 1, 2, SHIFT)
    lane_key = _stage(lane_key, lanes, 2, 4, SHIFT)
    lane_key = _stage(lane_key, lanes, 1, 4, SHIFT)
    lane_key = _stage(lane_key, lanes, 4, 8, SHIFT)
    lane_key = _stage(lane_key, lanes, 2, 8, SHIFT)
    lane_key = _stage(lane_key, lanes, 1, 8, SHIFT)
    lane_key = _stage(lane_key, lanes, 8, 16, SHIFT)
    lane_key = _stage(lane_key, lanes, 4, 16, SHIFT)
    lane_key = _stage(lane_key, lanes, 2, 16, SHIFT)
    lane_key = _stage(lane_key, lanes, 1, 16, SHIFT)
    lane_key = _stage(lane_key, lanes, 8, 16, SHIFT)
    lane_key = _stage(lane_key, lanes, 4, 16, SHIFT)
    lane_key = _stage(lane_key, lanes, 2, 16, SHIFT)
    lane_key = _stage(lane_key, lanes, 1, 16, SHIFT)

    lane_i = (lane_key & ((1 << SHIFT) - 1)).to(tl.int32)
    # one gather recovers the exact loaded element for every output lane.
    lane_v = tl.gather(xs, lane_i, axis=1)

    out_off = rows[:, None] * K + lanes
    keep = rmask[:, None]
    if LARGEST:
        tl.store(VALS + out_off, lane_v, mask=keep)
    else:
        tl.store(VALS + out_off, -lane_v, mask=keep)
    tl.store(IDX + out_off, lane_i.to(tl.int64), mask=keep)


def topk(x, k, dim=-1, largest=True, sorted=True, *, seg_len=None):
    # Keep the Ascend backend's segmented 2D call form alongside the ATen
    # overload. The third positional argument in that call form is a segment size.
    if seg_len is not None or (x.ndim == 2 and dim > 1 and largest and sorted):
        from flag_gems.runtime.backend._ascend.ops.topk import topk as segmented_topk

        return segmented_topk(x, k, seg_len if seg_len is not None else dim)
    if (
        x.device.type != "npu"
        or x.dtype != torch.float32
        or k != 16
        or x.ndim != 3
        or dim not in (-1, 2)
        or x.shape[2] not in (16, 32, 64)
        or x.stride(-1) != 1
        or x.is_neg()
        or x.numel() >= 2**31
        or sum(s * n for s, n in zip(x.stride(), x.shape)) >= 2**31
    ):
        if (
            x.ndim == 2
            and dim in (-1, 1)
            and largest
            and sorted
            and x.dtype == torch.float32
        ):
            from flag_gems.runtime.backend._ascend.ops.topk import (
                topk as segmented_topk,
            )

            return segmented_topk(x, k)
        from flag_gems.ops.topk import topk as generic_topk

        return generic_topk(x, k, dim, largest, sorted)
    batch, m, n = x.shape
    values = torch.empty((batch, m, k), device=x.device, dtype=x.dtype)
    indices = torch.empty((batch, m, k), device=x.device, dtype=torch.int64)
    rows = batch * m
    if rows:
        rpp = 1 if rows <= 32 else 16
        with torch_device_fn.device(x.device):
            _topk16_kernel[(triton.cdiv(rows, rpp),)](
                x,
                values,
                indices,
                rows,
                m,
                x.stride(1),
                batch,
                x.stride(0),
                1,
                0,
                1,
                n,
                k,
                rpp,
                n.bit_length() - 1,
                largest,
            )
    return values, indices
