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
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

ModeResult = namedtuple("mode", ["values", "indices"])

# ---------------------------------------------------------------------------
# Kunlunxin mode: per-row stable radix sort (self-contained; does NOT use the
# shared sort_stable radix, which is broken on XPU for M>1) followed by a
# linear scan over the sorted rows that reproduces ATen CPU tie semantics
# (best run with strictly-greater count; index of the last run element).
#
# XPU backend notes respected here:
#   * tl.store masks are NOT honored -> the scatter uses an unmasked
#     bijective per-lane position vector; lanes past the row end write into
#     a dedicated per-row tail slot (never read back).
#   * Signed integers need key = raw ^ signbit (NOT the bitwise complement
#     that is used for IEEE-754 floats).
#   * Radix passes must run LSB -> MSB to keep the stable prefix order.
# ---------------------------------------------------------------------------

BLOCK_N = tl.constexpr(512)
MAX_BLOCKS = tl.constexpr(1024)
N_BINS = tl.constexpr(16)


@triton.jit
def _mode_key16(x):
    if x.dtype == tl.bfloat16:
        # bf16 -> value-preserving fp32 bit pattern, keep high 16 bits
        h = x.to(tl.float32).to(tl.uint32, bitcast=True)
        sign = (h >> 31) == 1
        return (tl.where(sign, ~h, h | 0x80000000) >> 16) & 0xFFFF
    u = x.to(tl.uint16, bitcast=True)
    k = u.to(tl.uint32)
    if x.dtype.is_floating():
        sign = (k >> 15) == 1
        return tl.where(sign, ~k & 0xFFFF, k | 0x8000)
    else:
        return k ^ 0x8000


@triton.jit
def _mode_key32(x):
    if x.dtype.is_floating():
        u = x.to(tl.float32).to(tl.uint32, bitcast=True)
        sign = (u >> 31) == 1
        return tl.where(sign, ~u, u | 0x80000000)
    else:
        u = x.to(tl.uint32, bitcast=True)
        return u ^ 0x80000000


@triton.jit
def _mode_key64(x):
    if x.dtype.is_floating():
        u = x.to(tl.float64).to(tl.uint64, bitcast=True)
        hi = (u >> 32).to(tl.uint32)
        lo = (u & 0xFFFFFFFF).to(tl.uint32)
        sign = (hi >> 31) == 1
        khi = tl.where(sign, ~hi, hi | 0x80000000)
        klo = tl.where(sign, ~lo, lo)
        return (khi.to(tl.uint64) << 32) | klo.to(tl.uint64)
    else:
        u = x.to(tl.uint64, bitcast=True)
        hi = (u >> 32).to(tl.uint32)
        lo = (u & 0xFFFFFFFF).to(tl.uint32)
        return ((hi ^ 0x80000000).to(tl.uint64) << 32) | lo.to(tl.uint64)


@libentry()
@triton.jit
def _mode_radix_count(
    x_ptr,
    counts_ptr,  # (rows, 16, MAX_BLOCKS)
    NB,
    SHIFT,
    KEY_BITS: tl.constexpr,
    N: tl.constexpr,
    RS: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid // NB
    b = pid % NB
    cols = b * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    v = tl.load(x_ptr + r * RS + cols, mask=mask, other=0)
    if KEY_BITS == 16:
        k = _mode_key16(v)
    elif KEY_BITS == 64:
        k = _mode_key64(v)
    else:
        k = _mode_key32(v)
    bini = (k >> SHIFT) & (N_BINS - 1)
    off = r * 16 * MAX_BLOCKS + b
    for i in tl.static_range(0, 16):
        cnt = tl.sum(((bini == i) & mask).to(tl.int32), axis=0)
        tl.store(counts_ptr + off + i * MAX_BLOCKS, cnt)


@libentry()
@triton.jit
def _mode_radix_prefix_kernel(counts_ptr, pref_ptr, totals_ptr, NB):
    pid = tl.program_id(0)
    r = pid // 16
    bini = pid % 16
    off = tl.arange(0, MAX_BLOCKS)
    m = off < NB
    c = tl.load(
        counts_ptr + r * 16 * MAX_BLOCKS + bini * MAX_BLOCKS + off,
        mask=m,
        other=0,
    )
    ps = tl.cumsum(c, axis=0) - c
    tl.store(pref_ptr + r * 16 * MAX_BLOCKS + bini * MAX_BLOCKS + off, ps, mask=m)
    tl.store(totals_ptr + r * 16 + bini, tl.sum(c, axis=0))


@libentry()
@triton.jit
def _mode_base_kernel(totals_ptr, base_ptr):
    pid = tl.program_id(0)
    acc = 0
    for i in tl.static_range(0, 16):
        tl.store(base_ptr + pid * 16 + i, acc)
        acc = acc + tl.load(totals_ptr + pid * 16 + i)


@libentry()
@triton.jit
def _mode_radix_scatter(
    x_ptr,
    idx_ptr,
    out_x_ptr,
    out_idx_ptr,
    pref_ptr,
    base_ptr,
    NB,
    SHIFT,
    RS: tl.constexpr,
    KEY_BITS: tl.constexpr,
    N: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid // NB
    b = pid % NB
    cols = b * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    v = tl.load(x_ptr + r * RS + cols, mask=mask, other=0)
    oi = tl.load(idx_ptr + r * RS + cols, mask=mask, other=0)
    if KEY_BITS == 16:
        k = _mode_key16(v)
    elif KEY_BITS == 64:
        k = _mode_key64(v)
    else:
        k = _mode_key32(v)
    bini = (k >> SHIFT) & (N_BINS - 1)
    mypos = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for i in tl.static_range(0, 16):
        m_i = ((bini == i) & mask).to(tl.int32)
        rank = tl.cumsum(m_i, axis=0) - 1
        pref = tl.load(pref_ptr + r * 16 * MAX_BLOCKS + i * MAX_BLOCKS + b)
        base = tl.load(base_ptr + r * 16 + i)
        mypos = tl.where(m_i == 1, base + pref + rank, mypos)
    # unmasked bijective permutation store (XPU backend ignores store masks);
    # lanes >= N write into the dedicated per-row tail (never read back)
    dead = N + tl.arange(0, BLOCK_N)
    mypos = tl.where(mask, mypos, dead)
    tl.store(out_x_ptr + r * RS + mypos, v)
    tl.store(out_idx_ptr + r * RS + mypos, oi)




@libentry()
@triton.jit
def _mode_direct_kernel(
    x_ptr, out_v_ptr, out_i_ptr, M,
    N: tl.constexpr, RS: tl.constexpr, BLOCK: tl.constexpr,
):
    # O(N^2) exact counting mode for tiny rows; avoids the radix pipeline for
    # small-N shapes. Tie-break by (count desc, value asc) matching ATen CPU
    # semantics; the returned index points at an element equal to the value.
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    vals = tl.load(x_ptr + pid * RS + cols, mask=mask, other=0)
    best_cnt = 0
    # scalar-pointer load with a block mask is rejected by the Triton frontend;
    # element 0 is extracted from the already-loaded block instead
    best_val = tl.sum(tl.where(cols == 0, vals, 0), axis=0)
    best_idx = 0
    for i in tl.static_range(0, BLOCK):
        if i >= N:
            continue
        vi = tl.sum(tl.where(cols == i, vals, 0), axis=0)
        cnt = tl.sum(((vals == vi) & mask).to(tl.int32), axis=0)
        choose = (cnt > best_cnt) | ((cnt == best_cnt) & (vi < best_val))
        best_val = tl.where(choose, vi, best_val)
        best_cnt = tl.where(choose, cnt, best_cnt)
        best_idx = tl.where(choose, i, best_idx)
    tl.store(out_v_ptr + pid, best_val)
    tl.store(out_i_ptr + pid, best_idx)


@libentry()
@triton.jit(do_not_specialize=["columns"])
def _mode_sorted_rows_kernel(
    sorted_values,
    sorted_indices,
    output_values,
    output_indices,
    columns,
):
    row = tl.program_id(0)
    row_offset = row * columns
    current_value = tl.load(sorted_values + row_offset)
    current_index = tl.load(sorted_indices + row_offset)
    best_value = current_value
    best_index = current_index
    current_count = 1
    best_count = 1

    column = 1
    while column < columns:
        value = tl.load(sorted_values + row_offset + column)
        index = tl.load(sorted_indices + row_offset + column)
        same_value = value == current_value
        current_count = tl.where(same_value, current_count + 1, 1)
        current_value = tl.where(same_value, current_value, value)
        # ATen mode returns the last occurrence for the selected value.
        current_index = index
        better = current_count > best_count
        best_count = tl.where(better, current_count, best_count)
        best_value = tl.where(better, current_value, best_value)
        best_index = tl.where(better, current_index, best_index)
        column += 1

    tl.store(output_values + row, best_value)
    tl.store(output_indices + row, best_index)


def _mode_sort_rows(rows):
    M, N = rows.shape
    dtype = rows.dtype
    if dtype in (torch.float16, torch.int16):
        key_bits, npass = 16, 4
    elif dtype == torch.bfloat16:
        key_bits, npass = 32, 8
    elif dtype in (torch.float32, torch.int32):
        key_bits, npass = 32, 8
    elif dtype in (torch.float64, torch.int64):
        key_bits, npass = 64, 16
    else:
        raise NotImplementedError(f"mode: unsupported dtype {dtype}")

    BLK = int(BLOCK_N.value)
    MAXB = int(MAX_BLOCKS.value)
    NB = max(1, (N + BLK - 1) // BLK)
    RS = N + BLK

    values = torch.zeros((M, RS), dtype=dtype, device=rows.device)
    values[:, :N] = rows.contiguous()
    indices = torch.zeros((M, RS), dtype=torch.int64, device=rows.device)
    indices[:, :N] = torch.arange(
        N, dtype=torch.int64, device=rows.device
    ).reshape(1, N).expand(M, N).contiguous()
    out_v = torch.empty_like(values)
    out_i = torch.empty_like(indices)

    counts = torch.empty((M, 16, MAXB), dtype=torch.int32, device=rows.device)
    pref = torch.empty_like(counts)
    totals = torch.empty((M, 16), dtype=torch.int32, device=rows.device)
    base = torch.empty((M, 16), dtype=torch.int32, device=rows.device)

    grid = (M * NB,)
    with torch_device_fn.device(rows.device):
        for p in range(npass):
            shift = p * 4  # LSB -> MSB for stable radix order
            _mode_radix_count[grid](
                values, counts, NB, shift, KEY_BITS=key_bits, N=N, RS=RS
            )
            _mode_radix_prefix_kernel[(M * 16,)](counts, pref, totals, NB)
            _mode_base_kernel[(M,)](totals, base)
            _mode_radix_scatter[grid](
                values,
                indices,
                out_v,
                out_i,
                pref,
                base,
                NB,
                shift,
                RS=RS,
                KEY_BITS=key_bits,
                N=N,
            )
            values, out_v = out_v, values
            indices, out_i = out_i, indices
    return values[:, :N].contiguous(), indices[:, :N].contiguous()


@libentry()
@triton.jit
def _mode_fill_first(x_ptr, out_v_ptr, out_i_ptr, RS: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    v = tl.load(x_ptr + pid * RS)
    tl.store(out_v_ptr + pid, v)
    tl.store(out_i_ptr + pid, 0)


def _normalize_dim(dim, ndim):
    if ndim == 0:
        if dim in (0, -1):
            return 0
    elif -ndim <= dim < ndim:
        return dim % ndim
    raise IndexError(
        f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
    )


def _mode_impl(inp, dim, keepdim):
    if inp.ndim == 0:
        values = inp.clone()
        indices = torch.zeros((), dtype=torch.long, device=inp.device)
        return ModeResult(values=values, indices=indices)

    dim = _normalize_dim(dim, inp.ndim)
    shape = list(inp.shape)
    N = shape[dim]
    out_shape = shape[:dim] + shape[dim + 1 :]
    M = math.prod(out_shape)

    keepdim_shape = shape.copy()
    keepdim_shape[dim] = 1

    if N == 0:
        if M != 0:
            raise IndexError(
                f"mode(): Expected reduction dim {dim} to have non-zero size."
            )
        values = torch.empty(keepdim_shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(keepdim_shape, dtype=torch.long, device=inp.device)
        if not keepdim:
            values = torch.squeeze(values, dim)
            indices = torch.squeeze(indices, dim)
        return ModeResult(values=values, indices=indices)

    values = torch.empty(keepdim_shape, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(keepdim_shape, dtype=torch.long, device=inp.device)

    if M == 0:
        if not keepdim:
            values = torch.squeeze(values, dim)
            indices = torch.squeeze(indices, dim)
        return ModeResult(values=values, indices=indices)

    flat_values = values.reshape(M)
    flat_indices = indices.reshape(M)

    if dim != inp.ndim - 1:
        rows = torch.movedim(inp, dim, -1).reshape(M, N)
    else:
        rows = inp.reshape(M, N)

    BLK = int(BLOCK_N.value)
    if N == 1:
        with torch_device_fn.device(inp.device):
            _mode_fill_first[(M,)](rows, flat_values, flat_indices, RS=N, N=N)
    else:
        # N <= 32 is routed through the radix pipeline as well: the O(N^2)
        # direct kernel cannot be compiled by the XPU backend (frontend rejects
        # the scalar-pointer block-mask load, and the fallback block idiom
        # trips TritonXPUVectorize), so it is not used on this vendor.
        sorted_v, sorted_i = _mode_sort_rows(rows)
        if sorted_v.dtype == torch.bfloat16:
            # scan kernel comparisons on bf16 trip an MLIR scf.while type
            # mismatch on XPU; promote to fp32 first (exact mapping)
            sorted_v = sorted_v.to(torch.float32)
        with torch_device_fn.device(inp.device):
            _mode_sorted_rows_kernel[(M,)](
                sorted_v, sorted_i, flat_values, flat_indices, N
            )

    if not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)

    return ModeResult(values=values, indices=indices)


def mode(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN MODE")
    return _mode_impl(inp, dim, keepdim)
