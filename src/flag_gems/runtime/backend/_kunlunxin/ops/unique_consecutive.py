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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import libentry

from .unique import _SCAN_BLOCK, _triton_inclusive_scan

logger = logging.getLogger(__name__)

_UC_BLOCK = _SCAN_BLOCK

# ---------------------------------------------------------------------------
# Fast route, for inputs where consecutive duplicates are rare.
#
# A compaction is a scatter, and on this backend an address vector read from
# memory lowers to a per-lane access: at 67M f16 the scatter below costs 191ms
# where a plain copy costs 1ms.  A per-lane *store mask* is just as expensive
# (65x), so masking the scatter does not help either.
#
# What stays at the block-DMA rate is an affine access whose base is a scalar
# and whose mask is a ONE-SIDED threshold on tl.arange.  The compaction can be
# written that way: inside a tile holding k duplicates the surviving elements
# form k+1 runs of consecutive lanes, and run s moves to `base - s + lane`,
# i.e. a scalar offset.  With `dpre` the exclusive prefix count of duplicates
# in the tile, lane p belongs to run `dpre[p]`, so run s is exactly
# `live & (dpre == s)` and its bounds are two reductions over that mask -- no
# value is carried across loop iterations (a loop-carried scalar fed by a
# reduction silently stops updating on this backend).
#
# The inverse index comes out of the same two loads: the number of surviving
# elements strictly before lane p is `p - dpre[p]`, so
#       inverse[p] = base + p - dpre[p] - duplicate[p]
#
# Cost at 67M f16 is then 20.6ms against 223.2ms for the scatter route, and
# the pipeline is 3-4 launches instead of 9.
# ---------------------------------------------------------------------------
_FAST_BLOCK = 4096
_FAST_CHUNK = 4096
_CHUNK_ONE = 1 << 30
# Above this duplicate density the per-tile run loop does more work than the
# scatter, so the slow route is actually the cheaper one.
_DENSE_SHIFT = 8


@triton.jit
def _uc_count_kernel(inp_ptr, cnt_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    ar = tl.arange(0, BLOCK)
    offs = pid * BLOCK + ar
    a = tl.load(inp_ptr + offs)
    p = tl.load(inp_ptr + offs - 1, mask=offs > 0, other=0)
    start = (a != p) | (offs == 0)
    tl.store(cnt_ptr + pid, tl.sum(start.to(tl.int32), axis=0))


@triton.jit
def _uc_scan_chunk_kernel(cnt_ptr, base_ptr, tot_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    c = tl.load(cnt_ptr + offs)
    cs = tl.cumsum(c, axis=0)
    tl.store(base_ptr + offs, cs - c)
    tl.store(tot_ptr + pid, tl.sum(c, axis=0))


@triton.jit
def _uc_scan_top_kernel(tot_ptr, cbase_ptr, BLOCK: tl.constexpr):
    ar = tl.arange(0, BLOCK)
    c = tl.load(tot_ptr + ar)
    cs = tl.cumsum(c, axis=0)
    tl.store(cbase_ptr + ar, cs - c)
    # Past the last chunk, not aliasing any of the stores above.
    tl.store(cbase_ptr + BLOCK, tl.sum(c, axis=0))


@triton.jit
def _uc_emit_kernel(
    inp_ptr,
    base_ptr,
    cbase_ptr,
    cnt_ptr,
    out_ptr,
    inv_ptr,
    start_ptr,
    N,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = tl.program_id(0)
    ar = tl.arange(0, BLOCK)
    o0 = pid * BLOCK
    offs = o0 + ar
    a = tl.load(inp_ptr + offs)
    p = tl.load(inp_ptr + offs - 1, mask=offs > 0, other=0)
    keep = (a != p) | (offs == 0)
    dup = (~keep).to(tl.int32)
    dpre = tl.cumsum(dup, axis=0) - dup
    b = tl.load(base_ptr + pid) + tl.load(cbase_ptr + pid // CHUNK)

    if return_inverse:
        tl.store(inv_ptr + offs, (b + ar - dpre - dup).to(tl.int64))

    k = tl.minimum(N - o0, BLOCK) - tl.load(cnt_ptr + pid)
    for s in range(0, k + 1):
        sel = keep & (dpre == s)
        lo = tl.min(tl.where(sel, ar, BLOCK))
        end = tl.max(tl.where(sel, ar + 1, 0))
        ln = end - lo
        v = tl.load(inp_ptr + (o0 + lo) + ar, mask=ar < ln)
        tl.store(out_ptr + (b - s + lo) + ar, v, mask=ar < ln)
        if return_counts:
            tl.store(start_ptr + (b - s + lo) + ar, o0 + lo + ar, mask=ar < ln)


def _pow2(v):
    return 1 << max(4, (v - 1).bit_length())


def _uc_fast(input, flat_input, num_tasks, return_inverse, return_counts, device):
    """Run-decomposition route.  Returns None when the input is duplicate-dense,
    in which case the per-lane scatter is the cheaper of the two."""
    g = triton.cdiv(num_tasks, _FAST_BLOCK)
    gc = triton.cdiv(g, _FAST_CHUNK)
    # Every tile these kernels reduce over has to be fully live.  A 64-lane
    # group that no live lane touches contributes garbage to `tl.sum` -- 64 per
    # dead group, measured -- even though the same vector stores correctly and
    # `tl.cumsum` over it is exact, and neither `other=`, `& live` nor
    # `tl.where` repairs it.  So: pad the input to whole tiles by replicating
    # its last element (duplicates add no starts and are never emitted), pad
    # the per-tile counts with zeros, and drop the masks.
    tail = g * _FAST_BLOCK - num_tasks
    work = flat_input
    if tail:
        work = torch.empty(g * _FAST_BLOCK, dtype=flat_input.dtype, device=device)
        work[:num_tasks] = flat_input
        work[num_tasks:] = flat_input[num_tasks - 1]

    g_pad = gc * _FAST_CHUNK if gc > 1 else _pow2(g)
    cnt_alloc = torch.zeros if g_pad > g else torch.empty
    cnt = cnt_alloc(g_pad, dtype=torch.int32, device=device)
    base = torch.empty(g_pad, dtype=torch.int32, device=device)

    with torch_device_fn.device(device.index):
        _uc_count_kernel[(g,)](work, cnt, BLOCK=_FAST_BLOCK)
        if gc > 1:
            gc_pad = _pow2(gc + 1)
            tot = torch.zeros(gc_pad, dtype=torch.int32, device=device)
            cbase = torch.empty(gc_pad + 1, dtype=torch.int32, device=device)
            _uc_scan_chunk_kernel[(gc,)](cnt, base, tot, BLOCK=_FAST_CHUNK)
            _uc_scan_top_kernel[(1,)](tot, cbase, BLOCK=gc_pad)
            chunk_base, chunk, total = cbase, _FAST_CHUNK, cbase[gc_pad]
        else:
            tot = torch.empty(1, dtype=torch.int32, device=device)
            _uc_scan_chunk_kernel[(1,)](cnt, base, tot, BLOCK=g_pad)
            chunk_base, chunk, total = base, _CHUNK_ONE, tot[0]

        n_unique = int(total.item())
        if (num_tasks - n_unique) << _DENSE_SHIFT > num_tasks:
            return None

        data_out = torch.empty(
            n_unique + _FAST_BLOCK, dtype=flat_input.dtype, device=device
        )
        inv = (
            torch.empty(g * _FAST_BLOCK, dtype=torch.int64, device=device)
            if return_inverse
            else data_out
        )
        starts = (
            torch.empty(n_unique + _FAST_BLOCK, dtype=torch.int32, device=device)
            if return_counts
            else data_out
        )
        _uc_emit_kernel[(g,)](
            work,
            base,
            chunk_base,
            cnt,
            data_out,
            inv,
            starts,
            num_tasks,
            BLOCK=_FAST_BLOCK,
            CHUNK=chunk,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )

        counts = None
        if return_counts:
            counts = torch.empty(n_unique, dtype=torch.int64, device=device)
            _uc_run_lengths_kernel[(triton.cdiv(n_unique, _UC_BLOCK),)](
                starts[:n_unique],
                counts,
                num_tasks,
                n_unique,
                BLOCK=_UC_BLOCK,
            )

    output = data_out[:n_unique]
    inverse_indices = inv[:num_tasks].view_as(input) if return_inverse else None
    return output, inverse_indices, counts


# ---------------------------------------------------------------------------
# Slow route, kept for duplicate-dense inputs: there the run loop above would
# run one masked transfer per surviving run, and with runs one element long
# that is quadratic in the tile size, so the per-lane scatter wins.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _ne_consecutive_kernel(
    data_ptr,
    ne_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    offs = pid * BLOCK + r

    live = offs < N
    # Clamp both addresses into [0, N) so no load ever leaves the tensor.
    src = tl.where(live, offs, 0)
    prev_raw = tl.where(offs > 0, offs - 1, 0)
    src_prev = tl.where(prev_raw < N, prev_raw, 0)

    a = tl.load(data_ptr + src)
    b = tl.load(data_ptr + src_prev)

    is_first = tl.where(offs == 0, 1, 0)
    diff = tl.where(a != b, 1, 0)
    has_prev = tl.where(offs > 0, 1, 0)
    ne = is_first + diff * has_prev
    ne = tl.where(live, ne, 0).to(tl.int64)
    tl.store(ne_ptr + offs, ne)


@libentry()
@triton.jit
def _unique_consecutive_finalize_kernel(
    data_ptr,
    cum_ptr,
    ne_ptr,
    out_ptr,
    start_ptr,
    n_unique,
    N,
    BLOCK: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    offs = pid * BLOCK + r
    idx = pid * BLOCK + r

    live = offs < N
    src = tl.where(live, offs, 0)
    a = tl.load(data_ptr + src)
    c = tl.load(cum_ptr + idx)
    ne = tl.load(ne_ptr + idx)

    group = (c - 1).to(tl.int32)
    scratch = n_unique + r
    is_start = ne != 0
    dst = tl.where(is_start, group, scratch)

    # data_out[group] = first element of the group (value: data dtype).
    tl.store(out_ptr + dst, a)

    if return_counts:
        tl.store(start_ptr + dst, offs)


@libentry()
@triton.jit
def _uc_inverse_kernel(
    cum_ptr,
    inv_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    live = offs < N
    c = tl.load(cum_ptr + offs)
    tl.store(inv_ptr + offs, tl.where(live, c - 1, 0))


@libentry()
@triton.jit
def _uc_run_lengths_kernel(
    start32_ptr,
    counts_ptr,
    N,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    cur = tl.load(start32_ptr + tl.minimum(i, n - 1))
    nxt = tl.load(start32_ptr + tl.minimum(i + 1, n - 1))
    cnt = tl.where(i + 1 < n, nxt - cur, N - cur)
    tl.store(counts_ptr + i, cnt.to(tl.int64), mask=i < n)


def _uc_scatter(input, flat_input, num_tasks, return_inverse, return_counts, device):
    num_ctas = triton.cdiv(num_tasks, _UC_BLOCK)
    num_pad = num_ctas * _UC_BLOCK

    # --- stage 1: group-start flag, int64, padded to whole tiles -------------
    ne = torch.empty(num_pad, dtype=torch.int64, device=device)
    with torch_device_fn.device(device.index):
        _ne_consecutive_kernel[(num_ctas,)](flat_input, ne, num_tasks, BLOCK=_UC_BLOCK)

    # --- stage 2: exclusive/inclusive prefix sum (verified toolkit) ----------
    cum = _triton_inclusive_scan(ne)
    n_unique = int(cum[num_tasks - 1].item())

    data_out = torch.empty(n_unique + _UC_BLOCK, dtype=flat_input.dtype, device=device)
    start_positions = (
        torch.empty(n_unique + _UC_BLOCK, dtype=torch.int32, device=device)
        if return_counts
        else None
    )

    # --- stage 3: scatter outputs + group start offsets ---------------------
    dummy = data_out
    with torch_device_fn.device(device.index):
        _unique_consecutive_finalize_kernel[(num_ctas,)](
            flat_input,
            cum,
            ne,
            data_out,
            start_positions if start_positions is not None else dummy,
            n_unique,
            num_tasks,
            BLOCK=_UC_BLOCK,
            return_counts=return_counts,
        )

    output = data_out[:n_unique]

    # --- stage 4: inverse indices (own kernel -> a single int64 store) ------
    inverse_indices = None
    if return_inverse:
        inverse_indices = torch.empty(num_pad, dtype=torch.int64, device=device)
        with torch_device_fn.device(device.index):
            _uc_inverse_kernel[(num_ctas,)](
                cum, inverse_indices, num_tasks, BLOCK=_UC_BLOCK
            )
        inverse_indices = inverse_indices[:num_tasks].view_as(input)

    # --- stage 5: run lengths (own kernel -> a single int64 store) ----------
    counts = None
    if return_counts:
        counts = torch.empty(n_unique, dtype=torch.int64, device=device)
        with torch_device_fn.device(device.index):
            _uc_run_lengths_kernel[(triton.cdiv(n_unique, _UC_BLOCK),)](
                start_positions[:n_unique],
                counts,
                num_tasks,
                n_unique,
                BLOCK=_UC_BLOCK,
            )

    return output, inverse_indices, counts


def unique_consecutive(
    input: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int = None,
):
    logger.debug("GEMS_KUNLUNXIN UNIQUE_CONSECUTIVE")

    if dim is not None:
        raise NotImplementedError(
            "Kunlunxin unique_consecutive currently supports only dim=None"
        )

    # Flatten input for the None dim case
    flat_input = input.ravel()
    num_tasks = flat_input.numel()
    device = flat_input.device

    if num_tasks == 0:
        # Handle empty input
        output = torch.empty(0, dtype=input.dtype, device=device)
        inverse_indices = (
            torch.empty(0, dtype=torch.int64, device=device) if return_inverse else None
        )
        counts = (
            torch.empty(0, dtype=torch.int64, device=device) if return_counts else None
        )
        return output, inverse_indices, counts

    fast = _uc_fast(
        input, flat_input, num_tasks, return_inverse, return_counts, device
    )
    if fast is not None:
        return fast

    return _uc_scatter(
        input, flat_input, num_tasks, return_inverse, return_counts, device
    )
