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

"""Kunlunxin (XPU3) override for ``aten::combinations``.

The generic implementation unranks every row with an *unrolled binary search*
over a capped binomial evaluated by a nested ``tl.static_range``.  On this
backend ``TritonXPUUnrollControl`` never finishes on that IR: ``r >= 4`` and the
generic unranking kernel hang `make_ttxir` for tens of minutes, so the benchmark
matrix cannot even compile.  Its r == 2 kernel stores through a 2-D
``[BLOCK, 8]`` tile, which both miscompiles (wrong rows for ``n >= 300``) and
runs at 19.6 ms/call.

This override keeps the same unranking recurrence but restructures it around
what the backend likes:

* **No nested unrolled loops.**  The binomial values are materialised once by
  ``_binom_table_kernel`` into a ``(r + 1, Np + 1)`` int64 table (one flat
  product loop, no nesting) and the row kernel only *gathers* them.
* **No search loop.**  Each digit is estimated in closed form (fp32 ``sqrt`` /
  ``sqrt(sqrt(..))`` / ``exp(log(..))`` of ``k! S``, centred on ``(k-1)/2``), then
  corrected by two exact table comparisons.  The centred estimate is provably
  within one of the exact answer for ``k <= 8`` (checked exhaustively on the
  host), so two corrections always suffice.
* **1-D stores only.**  The general kernel writes one column stream per digit
  (no 2-D tile); ``r == 2`` folds a whole row into one wider word and walks one
  program per leading index, so its output is a plain block DMA.
* **Mask-free gathers.**  Tail lanes recompute the last valid row (address
  clamp), so every load/store address is in range and the values stay correct
  even where the backend drops store masks.

``with_replacement`` uses the classic bijection: unrank in the strict space of
``Np = n + r - 1`` values and subtract ``j`` from column ``j``; that preserves
the lexicographic row order (verified against ``torch.combinations`` in
``harness/perf_ir/proto_comb_unrank.py``).
"""

import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# rows per program for the general (r >= 3) kernel
_BLOCK_ROWS = 256
# copy tile for r == 1
_BLOCK_COPY = 1024
# lanes per chunk in the r == 2 packed kernel, and the max chunks per group
_BLOCK2 = 2048
_MAX_CHUNK2 = 4
# lanes for the binomial table build
_BLOCK_TAB = 1024
# digits unrolled per rows-kernel launch (more hangs TritonXPUUnrollControl)
_MAXDIG = 5

# Keep every intermediate of the exact binomial product inside int64:
# C(a, k) * a <= 2**61.
_CAP = 1 << 61


def _binom_cap(k: int) -> int:
    """Largest ``a`` with ``C(a, k) * a <= 2**61`` (overflow-free product loop)."""
    lo, hi = max(k, 1), max(k, 1) + 1
    while math.comb(hi, k) * hi <= _CAP:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if math.comb(mid, k) * mid <= _CAP:
            lo = mid
        else:
            hi = mid
    return lo


@triton.jit
def _fact(K: tl.constexpr):
    # K! as a trace-time constant (only used for the fp32 estimate).
    f = 1.0
    for i in tl.static_range(2, K + 1):
        f = f * i
    return f


@triton.jit
def _root_estimate(S, K: tl.constexpr):
    # Smallest x with C(x, K) >= S, estimated in fp32 and biased up by one.
    # Centring the K factors on (K-1)/2 keeps the estimate within one of the
    # exact answer over the whole reachable range, so two corrections suffice.
    if K == 1:
        return S
    s_f = S.to(tl.float32)
    if K == 2:
        est = (1.0 + tl.sqrt(1.0 + 8.0 * s_f)) * 0.5
    elif K == 4:
        est = tl.sqrt(tl.sqrt(24.0 * s_f)) + 1.5
    else:
        est = tl.exp(tl.log(_fact(K) * s_f) / K) + (K - 1) * 0.5
    return est.to(tl.int64) + 1


@triton.jit
def _binom_table_kernel(
    tab_ptr, Np, TSTRIDE, AMAX, CAP, R: tl.constexpr, BLOCK: tl.constexpr
):
    # Row k (1..R) of the table holds min(C(x, k), CAP) for x in [0, Np].  One
    # flat product loop: no nesting, so TritonXPUUnrollControl is never involved.
    # k comes from the grid, so the loop bound is R for every row and the factor
    # is masked out once i > k.  The row stride is a whole BLOCK: XPU3 does not
    # honour store masks (tail lanes still write), so lanes past Np must land in
    # row padding instead of the next row.
    k = tl.program_id(0) + 1  # table row 0 is unused; rows 1..R hold C(x, k)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.minimum(tl.minimum(off, Np), AMAX)
    v = x * 0 + 1
    for i in tl.static_range(1, R + 1):
        v = tl.where(i <= k, v * (x - k + i) // i, v)
    v = tl.where(off > AMAX, CAP, tl.minimum(v, CAP))
    tl.store(tab_ptr + k * TSTRIDE + off, v)


@triton.jit
def _tab_get(tab_ptr, row, x, Np, TSTRIDE):
    # min(C(x, row), CAP) from the table; x is clamped into the table range.
    return tl.load(tab_ptr + row * TSTRIDE + x)


@triton.jit
def _digit_tab_one(tab_ptr, S, hi, K: tl.constexpr, Np, TSTRIDE):
    # Cheaper variant used when the row has many digits: the centred estimate is
    # biased up by one and the table only has to confirm a possible one-step
    # shortfall.  Keeps the unrolled digit body small enough for the compiler.
    B = _root_estimate(S, K)
    B = tl.minimum(tl.maximum(B, K), hi)
    return tl.where((B < hi) & (_tab_get(tab_ptr, K, B, Np, TSTRIDE) < S), B + 1, B)


@triton.jit
def _binom_exact(a, K: tl.constexpr):
    # Exact C(a, K) with a clamped to the host-computed AMAX so the running
    # product stays inside int64.
    v = a * 0 + 1
    for i in tl.static_range(1, K + 1):
        v = v * (a - (K - i)) // i
    return v


@triton.jit
def _digit_arith(S, hi, K: tl.constexpr, AMAX):
    # Same contract as _digit_tab, but evaluates the short binomial product
    # instead of gathering it from the table.  Measured faster on XPU3 for
    # K <= 3: a gather from the int64 table costs more than 2-3 multiplies.
    if K == 1:
        return S
    B = _root_estimate(S, K)
    B = tl.minimum(tl.maximum(B, K), hi)
    bc = tl.minimum(B, AMAX)
    B = tl.where((B < hi) & (_binom_exact(bc, K) < S), B + 1, B)
    bm = B - 1
    bc = tl.minimum(bm, AMAX)
    B = tl.where(
        (bm >= K) & ((_binom_exact(bc, K) >= S) | (bm > AMAX)), B - 1, B
    )
    return tl.minimum(B, hi)


@triton.jit
def _digit_tab(tab_ptr, S, hi, K: tl.constexpr, Np, TSTRIDE):
    # Smallest B in [K, hi] with C(B, K) >= S.
    if K == 1:
        return S
    if K > 34:
        # K! no longer fits the fp32 estimate; scan the table instead.  Only
        # reachable for r > 34, where an allocatable C(n, r) forces a tiny n.
        B = S * 0 + K
        for _ in tl.static_range(64):
            nxt = B + 1
            B = tl.where((nxt <= hi) & (_tab_get(tab_ptr, K, B, Np, TSTRIDE) < S), nxt, B)
        return tl.minimum(B, hi)
    B = _root_estimate(S, K)
    B = tl.minimum(tl.maximum(B, K), hi)
    B = tl.where((B < hi) & (_tab_get(tab_ptr, K, B, Np, TSTRIDE) < S), B + 1, B)
    bm = B - 1
    B = tl.where((bm >= K) & (_tab_get(tab_ptr, K, bm, Np, TSTRIDE) >= S), B - 1, B)
    return tl.minimum(B, hi)


@triton.jit
def _comb_copy_kernel(out_ptr, in_ptr, stride, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    v = tl.load(in_ptr + off * stride, mask=m, other=0)
    tl.store(out_ptr + off, v, mask=m)


@triton.jit
def _comb2_pack_kernel(
    out_ptr,
    in_ptr,
    stride,
    Np,
    WR: tl.constexpr,
    ESIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Row m of the r == 2 output is [in[c0], in[c0 + 1 + j]] (column 1 shifted by
    # -1 for with_replacement), so one program owns the contiguous row run of a
    # single c0 and writes it as one packed word per row: a 1-D block DMA.  The
    # second column is a contiguous read of the tail of the input.
    c0 = tl.program_id(0)
    chunk = tl.program_id(1)
    rows = (Np - 1) - c0
    base = c0.to(tl.int64) * (2 * Np - c0 - 1) // 2
    j = chunk * BLOCK + tl.arange(0, BLOCK)
    m = j < rows
    # The second column is a contiguous read of the input tail; the runtime
    # scalar base means this 1-D access must stay masked (an unmasked strided
    # read is silently wrong here, and clamping the index is slower).
    i1 = c0 + 1 + j - (1 if WR else 0)
    v0 = tl.load(in_ptr + c0 * stride)
    v1 = tl.load(in_ptr + i1 * stride, mask=m, other=0)
    if ESIZE == 4:
        w = v1.to(tl.uint32, bitcast=True).to(tl.uint64) << 32 | v0.to(
            tl.uint32, bitcast=True
        ).to(tl.uint64)
    else:
        w = v1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16 | v0.to(
            tl.uint16, bitcast=True
        ).to(tl.uint32)
    tl.store(out_ptr + base + j, w, mask=m)


@triton.jit
def _comb_rows_kernel(
    out_ptr,
    in_ptr,
    tab_ptr,
    stride,
    Np,
    TSTRIDE,
    total,
    AMAX,
    R: tl.constexpr,
    WR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One output row per lane and all R digits unrolled in a single launch.
    # Every store is the 1-D "column j" stream and every input load an
    # address-clamped gather of the (tiny) input; binomials come from the table,
    # so no product loop is unrolled inside the digit loop.  The recurrence
    # carries (lo, m), m being the offset of the row inside the block of rows
    # that share the digits chosen so far.
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    rmask = rows < total
    row = tl.minimum(rows, total - 1)  # idempotent tail: recompute the last row
    m = row.to(tl.int64)
    lo = m * 0
    for j in tl.static_range(R):
        hi = Np - lo
        if R <= 3:
            # Short ranks: everything is arithmetic, so no table is needed.
            S = _binom_exact(tl.minimum(hi, AMAX), R - j) - m
            B = _digit_arith(S, hi, R - j, AMAX)
        else:
            S = _tab_get(tab_ptr, R - j, hi, Np, TSTRIDE) - m
            B = _digit_tab(tab_ptr, S, hi, R - j, Np, TSTRIDE)
        c = Np - B
        idx = c - (j if WR else 0)
        val = tl.load(in_ptr + idx * stride)
        tl.store(out_ptr + row.to(tl.int64) * R + j, val)
        if R <= 3:
            m = _binom_exact(tl.minimum(B, AMAX), R - j) - S
        else:
            m = _tab_get(tab_ptr, R - j, B, Np, TSTRIDE) - S
        lo = c + 1


@triton.jit
def _comb_row1_kernel(
    out_ptr,
    in_ptr,
    tab_ptr,
    st_lo_in,
    st_s_in,
    st_a_in,
    st_lo_out,
    st_s_out,
    st_a_out,
    stride,
    Np,
    TSTRIDE,
    total,
    J: tl.constexpr,
    R: tl.constexpr,
    FIRST: tl.constexpr,
    LAST: tl.constexpr,
    WR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One column per launch, so the kernel has no loop at all: a loop-carried
    # int64 either read from or written to global memory trips the same
    # TritonXPUUnrollControl type-mismatch failure, so ranks longer than
    # _MAXDIG advance one column per launch and hand (lo, m) forward through
    # the state buffers.
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    rmask = rows < total
    row = tl.minimum(rows, total - 1)
    if FIRST:
        m = row.to(tl.int64)
        lo = m * 0
    else:
        m = tl.load(st_a_in + row) - tl.load(st_s_in + row)
        lo = tl.load(st_lo_in + row)
    hi = Np - lo
    S = _tab_get(tab_ptr, R - J, hi, Np, TSTRIDE) - m
    B = _digit_tab(tab_ptr, S, hi, R - J, Np, TSTRIDE)
    c = Np - B
    idx = c - (J if WR else 0)
    val = tl.load(in_ptr + idx * stride)
    tl.store(out_ptr + row.to(tl.int64) * R + J, val)
    if not LAST:
        # Store the two operands of m = C(B, k) - S separately: this backend's
        # TritonXPUUnrollControl rejects an int64 store value that is a
        # subtraction of two gathers, while storing them and subtracting after
        # the reload is accepted.
        tl.store(st_lo_out + row, c + 1)
        tl.store(st_a_out + row, _tab_get(tab_ptr, R - J, B, Np, TSTRIDE))
        tl.store(st_s_out + row, S)


def combinations(self, r=2, with_replacement=False):
    logger.debug("GEMS COMBINATIONS (kunlunxin)")
    if r < 0:
        raise RuntimeError("Expect a non-negative number, but got {}".format(r))
    if self.dim() != 1:
        raise RuntimeError(
            "Expect a 1D vector, but got shape {}".format(list(self.shape))
        )
    flat = self
    n = flat.numel()
    if r == 0:
        return torch.empty(0, dtype=flat.dtype, device=flat.device)
    wr = bool(with_replacement)
    Np = n + r - 1 if wr else n
    total = math.comb(Np, r) if Np >= r else 0
    out = torch.empty((total, r), dtype=flat.dtype, device=flat.device)
    if total == 0:
        return out

    stride = flat.stride(0)
    esize = flat.element_size()

    if r == 1:
        grid = (triton.cdiv(n, _BLOCK_COPY),)
        _comb_copy_kernel[grid](out, flat, stride, n, BLOCK=_BLOCK_COPY)
        return out

    if r == 2 and esize in (2, 4) and stride == 1 and Np > 1:
        nchunk = triton.cdiv(Np - 1, _BLOCK2)
        if nchunk <= _MAX_CHUNK2:
            wdtype = torch.int64 if esize == 4 else torch.int32
            idtype = torch.int32 if esize == 4 else torch.int16
            _comb2_pack_kernel[(Np, nchunk)](
                out.view(wdtype),
                flat.view(idtype),
                stride,
                Np,
                WR=wr,
                ESIZE=esize,
                BLOCK=_BLOCK2,
            )
            return out

    # General path: materialise min(C(x, k), CAP) for k = 1..r, x = 0..Np, then
    # unrank one row per lane.  More than _MAXDIG digits in one launch makes
    # TritonXPUUnrollControl hang, so long ranks are split across launches that
    # hand (lo, S) forward through a per-row state buffer.
    amax = min(_binom_cap(k) for k in range(1, r + 1))
    cap = 4 * total + 8
    tstride = triton.cdiv(Np + 1, _BLOCK_TAB) * _BLOCK_TAB
    tdt = torch.int64
    tab = torch.empty((r + 1, tstride), dtype=tdt, device=flat.device)
    _binom_table_kernel[(r, tstride // _BLOCK_TAB)](
        tab, Np, tstride, amax, cap, R=r, BLOCK=_BLOCK_TAB
    )
    grid = (triton.cdiv(total, _BLOCK_ROWS),)
    dummy_lo = torch.empty(1, dtype=tdt, device=flat.device)
    dummy_s = torch.empty(1, dtype=tdt, device=flat.device)
    if r <= 3:
        _comb_rows_kernel[grid](
            out,
            flat,
            dummy_lo,
            stride,
            Np,
            tstride,
            total,
            min(_binom_cap(k) for k in range(1, r + 1)),
            R=r,
            WR=wr,
            BLOCK=_BLOCK_ROWS,
        )
        return out
    if r <= _MAXDIG:
        _comb_rows_kernel[grid](
            out,
            flat,
            tab,
            stride,
            Np,
            tstride,
            total,
            min(_binom_cap(k) for k in range(1, r + 1)),
            R=r,
            WR=wr,
            BLOCK=_BLOCK_ROWS,
        )
        return out
    lo_buf = dummy_lo
    s_buf = dummy_s
    a_buf = dummy_s
    for j in range(r):
        first = j == 0
        last = j + 1 == r
        if last:
            next_lo, next_s, next_a = dummy_lo, dummy_s, dummy_s
        else:
            next_lo = torch.empty(total, dtype=tdt, device=flat.device)
            next_s = torch.empty(total, dtype=tdt, device=flat.device)
            next_a = torch.empty(total, dtype=tdt, device=flat.device)
        _comb_row1_kernel[grid](
            out,
            flat,
            tab,
            lo_buf,
            s_buf,
            a_buf,
            next_lo,
            next_s,
            next_a,
            stride,
            Np,
            tstride,
            total,
            J=j,
            R=r,
            FIRST=first,
            LAST=last,
            WR=wr,
            BLOCK=_BLOCK_ROWS,
        )
        lo_buf, s_buf, a_buf = next_lo, next_s, next_a
    return out


run = combinations
