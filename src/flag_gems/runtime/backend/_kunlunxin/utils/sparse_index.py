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
"""Device-side construction of sparse index metadata.

The bookkeeping a COO/CSR/CSC operator needs from the host in a naive port --
per-row counts, row-pointer prefix sums, row-major permutations, duplicate
merging and compaction -- is produced here by Triton kernels instead:

* ``bisect``      -- binary search over a sorted key array: row pointers of a
                     row-ordered entry array, and per-entry column expansion
                     of a CSC column pointer array;
* ``exclusive_scan`` -- two-level prefix sum (per-block sums, one scan of the
                     block sums, then a local scan per block);
* ``compact_by_flag`` / ``gather`` -- prefix-sum driven stable scatter, so the
                     destination of an entry comes from a scan rather than an
                     atomic cursor.

Every kernel writes to locations that are distinct per lane, so no global
atomic is involved. The only host value read back is a final scalar count used
to size an allocation.
"""

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

# Elements per scan program.
_SCAN_BLOCK = 1024
# Elements per index program (bisection / gather / scatter). A 256-lane int64
# tile exceeds the lane budget of the lowering on this backend, so the tile is
# capped at 128.
_INDEX_BLOCK = 128


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------
@triton.jit(do_not_specialize=["n_keys", "n_probe"])
def _bisect_kernel(
    keys_ptr,
    probe_ptr,
    out_ptr,
    n_keys,
    n_probe,
    STEPS: tl.constexpr,
    RIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n_probe
    # Index-width sensitive scalars are widened once: mixing a narrow launch
    # argument with the lane index does not lower here.
    p = tl.load(probe_ptr + i, mask=m, other=0).to(tl.int64)
    n_keys64 = n_keys.to(tl.int64)
    lo = tl.zeros((BLOCK,), dtype=tl.int64)
    hi = tl.zeros((BLOCK,), dtype=tl.int64) + n_keys64
    for _ in range(STEPS):
        mid = (lo + hi) >> 1
        inside = m & (mid < n_keys64)
        safe_mid = tl.where(inside, mid, 0)
        v = tl.load(keys_ptr + safe_mid, mask=inside, other=0).to(tl.int64)
        if RIGHT:
            c = inside & (v <= p)
        else:
            c = inside & (v < p)
        lo = tl.where(c, mid + 1, lo)
        hi = tl.where(c, hi, mid)
    tl.store(out_ptr + i, lo, mask=m)


@triton.jit(do_not_specialize=["n_keys", "n_probe"])
def _csc_column_kernel(
    ptr_ptr,
    probe_ptr,
    out_ptr,
    n_keys,
    n_probe,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Column index of every stored entry of a CSC arrow.

    ``ptr`` is the column pointer array, so the column of entry ``p`` is
    ``searchsorted_right(ptr, p) - 1``.
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n_probe
    p = tl.load(probe_ptr + i, mask=m, other=0).to(tl.int64)
    n_keys64 = n_keys.to(tl.int64)
    lo = tl.zeros((BLOCK,), dtype=tl.int64)
    hi = tl.zeros((BLOCK,), dtype=tl.int64) + n_keys64
    for _ in range(STEPS):
        mid = (lo + hi) >> 1
        inside = m & (mid < n_keys64)
        safe_mid = tl.where(inside, mid, 0)
        v = tl.load(ptr_ptr + safe_mid, mask=inside, other=0).to(tl.int64)
        c = inside & (v <= p)
        lo = tl.where(c, mid + 1, lo)
        hi = tl.where(c, hi, mid)
    tl.store(out_ptr + i, lo - 1, mask=m)


@triton.jit(do_not_specialize=["n"])
def _block_sum_kernel(v_ptr, s_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(v_ptr + i, mask=i < n, other=0)
    tl.store(s_ptr + pid, tl.sum(v, 0))


@triton.jit(do_not_specialize=["nb"])
def _scan_blocks_kernel(s_ptr, o_ptr, nb, NB: tl.constexpr):
    i = tl.arange(0, NB)
    v = tl.load(s_ptr + i, mask=i < nb, other=0)
    c = tl.cumsum(v, 0) - v
    tl.store(o_ptr + i, c, mask=i < nb)


@triton.jit(do_not_specialize=["n"])
def _scan_exclusive_kernel(v_ptr, out_ptr, boff_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(v_ptr + i, mask=m, other=0)
    c = tl.cumsum(v, 0) - v + tl.load(boff_ptr + pid)
    tl.store(out_ptr + i, c, mask=m)


@triton.jit(do_not_specialize=["n"])
def _scan_inclusive_kernel(v_ptr, out_ptr, boff_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(v_ptr + i, mask=m, other=0)
    c = tl.cumsum(v, 0) + tl.load(boff_ptr + pid)
    tl.store(out_ptr + i, c, mask=m)


@triton.jit(do_not_specialize=["n"])
def _gather_kernel(src_ptr, idx_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    j = tl.load(idx_ptr + i, mask=m, other=0).to(tl.int32)
    tl.store(dst_ptr + i, tl.load(src_ptr + j, mask=m, other=0), mask=m)


@triton.jit(do_not_specialize=["n"])
def _compact_kernel(src_ptr, pos_ptr, flag_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    f = tl.load(flag_ptr + i, mask=m, other=0)
    p = tl.load(pos_ptr + i, mask=m, other=0).to(tl.int32)
    v = tl.load(src_ptr + i, mask=m, other=0)
    tl.store(dst_ptr + p, v, mask=m & (f != 0))


@triton.jit(do_not_specialize=["n"])
def _iota_kernel(out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out_ptr + i, i, mask=m)


@triton.jit(do_not_specialize=["n"])
def _mark_kernel(idx_ptr, flag_ptr, n, BLOCK: tl.constexpr):
    """Raise ``flag[idx[i]]``; every writer stores the same value."""
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    r = tl.load(idx_ptr + i, mask=m, other=0).to(tl.int32)
    tl.store(flag_ptr + r, 1, mask=m)


@triton.jit(do_not_specialize=["n", "scale"])
def _scale_kernel(src_ptr, dst_ptr, n, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(src_ptr + i, mask=m, other=0).to(tl.int64)
    tl.store(
        dst_ptr + i,
        (v * scale.to(tl.int64)).to(dst_ptr.dtype.element_ty),
        mask=m,
    )


@triton.jit(do_not_specialize=["n", "n_cols"])
def _key_kernel(rows_ptr, cols_ptr, key_ptr, n, n_cols, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    r = tl.load(rows_ptr + i, mask=m, other=0).to(tl.int64)
    c = tl.load(cols_ptr + i, mask=m, other=0).to(tl.int64)
    ncols = n_cols.to(tl.int64)
    tl.store(key_ptr + i, r * ncols + c, mask=m)


@triton.jit(do_not_specialize=["n", "n_rows"])
def _scatter_dense_kernel(
    rows_ptr, cols_ptr, vals_ptr, dense_ptr, n_rows, n, BLOCK: tl.constexpr
):
    """Plain scatter of unique coordinates into a dense row-major buffer."""
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    r = tl.load(rows_ptr + i, mask=m, other=0).to(tl.int64)
    c = tl.load(cols_ptr + i, mask=m, other=0).to(tl.int64)
    v = tl.load(vals_ptr + i, mask=m, other=0)
    nrows = n_rows.to(tl.int64)
    tl.store(dense_ptr + r * nrows + c, v, mask=m)


@triton.jit(do_not_specialize=["n"])
def _eq_kernel(lhs_ptr, rhs_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(lhs_ptr + i, mask=m, other=-1)
    b = tl.load(rhs_ptr + i, mask=m, other=-2)
    tl.store(out_ptr + i, (a == b).to(tl.int32), mask=m)


@triton.jit(do_not_specialize=["n"])
def _range_sum_kernel(pref_ptr, lo_ptr, hi_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    lo = tl.load(lo_ptr + i, mask=m, other=0).to(tl.int32)
    hi = tl.load(hi_ptr + i, mask=m, other=0).to(tl.int32)
    has_hi = m & (hi > 0)
    has_lo = m & (lo > 0)
    total = tl.load(pref_ptr + tl.where(has_hi, hi - 1, 0), mask=has_hi, other=0.0)
    before = tl.load(pref_ptr + tl.where(has_lo, lo - 1, 0), mask=has_lo, other=0.0)
    tl.store(out_ptr + i, total - before, mask=m)


# --------------------------------------------------------------------------
# Host wrappers
# --------------------------------------------------------------------------
def _launch(fn, grid, device, *args, **kwargs):
    with torch_device_fn.device(device):
        return fn[grid](*args, **kwargs)


def _grid(n):
    return (triton.cdiv(n, _INDEX_BLOCK),)


def iota(n, device, dtype=torch.int32):
    out = torch.empty(n, dtype=dtype, device=device)
    if n:
        _launch(_iota_kernel, _grid(n), device, out, n, BLOCK=_INDEX_BLOCK)
    return out


def bisect(keys, probes, right=False):
    """Binary search ``probes`` in the ascending int array ``keys``."""
    n_keys = int(keys.numel())
    n_probe = int(probes.numel())
    out = torch.zeros(n_probe, dtype=torch.int32, device=keys.device)
    if n_probe and n_keys:
        _launch(
            _bisect_kernel,
            _grid(n_probe),
            keys.device,
            keys,
            probes,
            out,
            n_keys,
            n_probe,
            STEPS=max(n_keys.bit_length(), 1),
            RIGHT=bool(right),
            BLOCK=_INDEX_BLOCK,
        )
    return out


def _two_level_scan(values, inclusive):
    n = int(values.numel())
    out = torch.empty_like(values)
    if n == 0:
        return out
    nb = triton.cdiv(n, _SCAN_BLOCK)
    block_sums = torch.empty(nb, dtype=values.dtype, device=values.device)
    block_offs = torch.empty(nb, dtype=values.dtype, device=values.device)
    _launch(
        _block_sum_kernel,
        (nb,),
        values.device,
        values,
        block_sums,
        n,
        BLOCK=_SCAN_BLOCK,
    )
    _launch(
        _scan_blocks_kernel,
        (1,),
        values.device,
        block_sums,
        block_offs,
        nb,
        NB=triton.next_power_of_2(nb),
    )
    _launch(
        _scan_inclusive_kernel if inclusive else _scan_exclusive_kernel,
        (nb,),
        values.device,
        values,
        out,
        block_offs,
        n,
        BLOCK=_SCAN_BLOCK,
    )
    return out


def exclusive_scan(values):
    """Exclusive prefix sum of an int32 tensor."""
    return _two_level_scan(values, inclusive=False)


def inclusive_scan(values):
    """Inclusive prefix sum (int32 or float32)."""
    return _two_level_scan(values, inclusive=True)


def gather(src, idx):
    """``src[idx]`` with an int index tensor."""
    n = int(idx.numel())
    out = torch.empty(n, dtype=src.dtype, device=src.device)
    if n:
        _launch(
            _gather_kernel, _grid(n), src.device, src, idx, out, n, BLOCK=_INDEX_BLOCK
        )
    return out


def mark_rows(rows, n_rows, flag=None):
    """Zero/one table of the rows touched by ``rows``."""
    if flag is None:
        flag = torch.zeros(n_rows, dtype=torch.int32, device=rows.device)
    n = int(rows.numel())
    if n:
        _launch(_mark_kernel, _grid(n), rows.device, rows, flag, n, BLOCK=_INDEX_BLOCK)
    return flag


def scale(values, factor):
    out = torch.empty_like(values)
    n = int(values.numel())
    if n:
        _launch(
            _scale_kernel,
            _grid(n),
            values.device,
            values,
            out,
            n,
            factor,
            BLOCK=_INDEX_BLOCK,
        )
    return out


def row_pointers_from_sorted(rows, n_rows):
    """CSR row pointers of an entry array already ordered by row."""
    probes = iota(n_rows + 1, rows.device)
    return bisect(rows, probes, right=False)


def csc_entry_columns(ptr, nnz):
    """Column index of every stored entry of a CSC arrow."""
    out = torch.empty(nnz, dtype=torch.int32, device=ptr.device)
    if nnz:
        n_keys = int(ptr.numel())
        probes = iota(nnz, ptr.device)
        _launch(
            _csc_column_kernel,
            _grid(nnz),
            ptr.device,
            ptr,
            probes,
            out,
            n_keys,
            nnz,
            STEPS=max(n_keys.bit_length(), 1),
            BLOCK=_INDEX_BLOCK,
        )
    return out


def dense_scatter(rows, cols, values, n_rows, n_cols):
    """Dense image of unique coordinates."""
    dense = torch.zeros((n_rows, n_cols), dtype=values.dtype, device=values.device)
    n = int(rows.numel())
    if n:
        _launch(
            _scatter_dense_kernel,
            _grid(n),
            values.device,
            rows,
            cols,
            values,
            dense,
            n_cols,
            n,
            BLOCK=_INDEX_BLOCK,
        )
    return dense


def equal_flags(lhs, rhs):
    out = torch.empty(lhs.numel(), dtype=torch.int32, device=lhs.device)
    n = int(lhs.numel())
    if n:
        _launch(_eq_kernel, _grid(n), lhs.device, lhs, rhs, out, n, BLOCK=_INDEX_BLOCK)
    return out


def _range_sums(inclusive, lo, hi):
    n = int(lo.numel())
    out = torch.empty(n, dtype=inclusive.dtype, device=inclusive.device)
    if n:
        _launch(
            _range_sum_kernel,
            _grid(n),
            inclusive.device,
            inclusive,
            lo,
            hi,
            out,
            n,
            BLOCK=_INDEX_BLOCK,
        )
    return out


def coalesce_triplets(rows, cols, values, n_cols):
    """Row-major unique triplets of a COO entry set; duplicates are summed."""
    from .sort import sort as _sort

    nnz = int(rows.numel())
    if nnz == 0:
        return rows, cols, values

    device = rows.device
    keys = torch.empty(nnz, dtype=torch.int64, device=device)
    _launch(
        _key_kernel,
        _grid(nnz),
        device,
        rows,
        cols,
        keys,
        nnz,
        n_cols,
        BLOCK=_INDEX_BLOCK,
    )
    sorted_keys, perm = _sort(keys)
    perm = perm.to(torch.int32)
    srows = gather(rows, perm)
    scols = gather(cols, perm)
    svals = gather(values, perm)

    lo = bisect(sorted_keys, sorted_keys, right=False)
    hi = bisect(sorted_keys, sorted_keys, right=True)
    heads = equal_flags(lo, iota(nnz, device))

    acc = inclusive_scan(svals.to(torch.float32))
    run_sum = _range_sums(acc, lo, hi).to(values.dtype)
    pos = exclusive_scan(heads)

    urows = torch.empty(nnz, dtype=rows.dtype, device=device)
    ucols = torch.empty(nnz, dtype=cols.dtype, device=device)
    uvals = torch.empty(nnz, dtype=values.dtype, device=device)
    for src, dst in ((srows, urows), (scols, ucols), (run_sum, uvals)):
        _launch(
            _compact_kernel,
            _grid(nnz),
            device,
            src,
            pos,
            heads,
            dst,
            nnz,
            BLOCK=_INDEX_BLOCK,
        )
    uniq = int(heads.sum().item())
    return urows[:uniq], ucols[:uniq], uvals[:uniq]
