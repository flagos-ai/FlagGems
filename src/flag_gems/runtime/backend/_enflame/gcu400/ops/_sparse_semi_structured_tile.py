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

"""Semi-structured (2:4) sparse tile generation.

``aten::_sparse_semi_structured_tile(input, algorithm="", use_cutlass=True)``
prunes a dense 2-D fp16/bf16 tensor to the 2:4 semi-structured sparsity pattern
and returns the five tensors of the compressed representation:

===============  ==============================  ==========================
output           shape                           meaning
===============  ==============================  ==========================
``packed``       (rows_r, cols_r / 2)            selected values
``meta``         (rows_r, 2, cols_r / 32) i16    CUTLASS metadata
``packed_t``     (cols_r', rows_r' / 2)          transposed packed values
``meta_t``       (cols_r', 2, rows_r' / 32) i16  transposed metadata
``threads_masks``(rows_r / 8, cols_r / 16, 8) u8 per-thread 8x8 bitmasks
===============  ==============================  ==========================
where ``rows_r = ceil(rows / 32) * 32`` and ``cols_r = ceil(cols / 64) * 64``;
the transposed representation is sized for the swapped problem
(``rows_t = ceil(cols / 32) * 32``, packed width ``ceil(rows / 64) * 32``).

Selection follows the aten CUDA kernel
(``aten/src/ATen/native/sparse/cuda/SparseSemiStructuredTile.cu``): every 4x4
tile is pruned greedily by descending value - the 16 values are sorted
ascending with the CUTLASS ``StaticSort`` compare-exchange network (which
exchanges equal keys too) and scanned from the largest, keeping a value when
its tile row and tile column each still have capacity.

The main kernel handles one 32-row x 4 * TCJ-column input block per program.
The 16 values of a 4x4 tile are held in 16 separate register tensors instead of
one tensor with a reduction axis, so the compare-exchange network, the greedy
scan and the packing are pure element-wise code (no cross-lane reductions).  It
writes packed, packed_t and the per-thread 8x8 selection bitmasks, plus the
4-bit metadata quads; a second kernel applies the CUTLASS
``ColumnMajorInterleaved<2>`` reorder that produces the int16 metadata
matrices.

``algorithm`` selects the ranking: ``""`` / ``"largest_values_greedy"`` rank
by value and ``"largest_abs_values_greedy"`` ranks by magnitude.
``use_cutlass=False`` switches the metadata to the cuSPARSELt layout; only the
int16 CUTLASS layout is implemented here, matching ``use_cutlass=True`` (the
default).  ``"causal1122"`` (per-tile-row capacities ``[1, 1, 2, 2]``) is
computed by the host fallback below.

For rows with fewer than two selections the aten kernel leaves the unused
2-bit metadata field holding stale register contents, so those metadata bytes
are not bit-reproducible across runs.  This implementation emits the canonical
encoding (the unused index points at the sibling column whose packed slot is
zero), which decodes to exactly the same dense tensor.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime  # noqa: F401
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_ALGORITHMS = ("", "largest_values_greedy", "largest_abs_values_greedy", "causal1122")


def _network(n):
    """Bose-Nelson compare-exchange network emitted by cutlass StaticSort."""
    seq = []

    def pb(i, j, x, y):
        if x == 1 and y == 1:
            seq.append((i - 1, j - 1))
        elif x == 1 and y == 2:
            seq.append((i - 1, j))
            seq.append((i - 1, j - 1))
        elif x == 2 and y == 1:
            seq.append((i - 1, j - 1))
            seq.append((i, j - 1))
        else:
            left = x >> 1
            mid = (y if (x & 1) else (y + 1)) >> 1
            pb(i, j, left, mid)
            pb(i + left, j + mid, x - left, y - mid)
            pb(i + left, j, x - left, mid)

    def ps(i, m):
        if m <= 1:
            return
        left = m >> 1
        ps(i, left)
        ps(i + left, m - left)
        pb(i, i + left, left, m - left)

    ps(1, n)
    return tuple(seq)


# Reference copy of the comparator order hard-coded in `_ssst_kernel` (a jit
# function cannot loop over it); `_network(16)` documents where it comes from.
_NET16 = tl.constexpr(tuple(_network(16)))


@triton.jit
def _cx2(ka, kb):
    """Ascending compare-exchange on the value bits.

    Only the 16 value bits take part in the comparison and *equal* values are
    swapped (the reference network does `swap = ~(a < b)`), so the lane tag in
    the low 4 bits must not decide ties: min/max would order them by lane and
    diverge from aten on tiles holding duplicate values.
    """
    sw = (ka >> 4) >= (kb >> 4)
    return tl.where(sw, kb, ka), tl.where(sw, ka, kb)


@triton.jit
def _key16(v, lane: tl.constexpr, IS_ABS: tl.constexpr):
    """Order-preserving 16-bit key of the value, lane index in the low 4 bits."""
    if IS_ABS:
        v = tl.abs(v)
    xb = v.to(tl.int16, bitcast=True).to(tl.int32)
    k = tl.where(xb < 0, (~xb) & 0xFFFF, (xb & 0xFFFF) | 0x8000)
    return (k << 4) | lane


@triton.jit
def _greedy_step(k, bits16, state):
    """LargestValuesGreedy on one rank: keep if row/col counts are below 2."""
    p = k & 0xF
    srow = (p >> 2) << 2  # 4 * in-tile row
    scol = 16 + ((p & 3) << 2)  # 16 + 4 * in-tile column
    ok = (((state >> srow) & 0xF) < 2) & (((state >> scol) & 0xF) < 2)
    bits16 = bits16 | tl.where(ok, 1 << p, 0)
    state = state + tl.where(ok, (1 << srow) + (1 << scol), 0)
    return bits16, state


@triton.jit
def _pack_row(b0, b1, b2, b3, v0, v1, v2, v3):
    """Build one 4-element tile row output: quad meta + two packed values.

    b* are the per-column selection bits, v* the candidate values in
    ascending column order.
    """
    cnt = b0 + b1 + b2 + b3
    c0 = tl.where(b0, 0, tl.where(b1, 1, tl.where(b2, 2, 3)))
    c1 = tl.where(
        b0, tl.where(b1, 1, tl.where(b2, 2, 3)), tl.where(b1, tl.where(b2, 2, 3), 3)
    )
    two = cnt == 2
    one = cnt == 1
    late = one & (c0 >= 2)
    u0 = tl.where(c0 == 0, v0, tl.where(c0 == 1, v1, tl.where(c0 == 2, v2, v3)))
    c1e = tl.where(two, c1, c0)
    u1 = tl.where(c1e == 0, v0, tl.where(c1e == 1, v1, tl.where(c1e == 2, v2, v3)))
    g0 = tl.where(two | (one & (c0 < 2)), u0, 0.0)
    g1 = tl.where(two | late, u1, 0.0)
    qv = tl.where(
        two,
        c0 | (c1 << 2),
        tl.where(one, tl.where(c0 < 2, c0 | ((c0 ^ 1) << 2), (c0 ^ 1) | (c0 << 2)), 0),
    )
    return qv, g0, g1


@libentry()
@triton.jit
def _ssst_kernel(
    x_ptr,
    ld,
    rows,
    cols,
    packed_ptr,
    spm,
    packed_t_ptr,
    rows_t,
    width_t,
    qbuf_ptr,
    qb_stride,
    qbuf_t_ptr,
    qbt_stride,
    masks_ptr,
    sm0,
    sm1,
    IS_ABS: tl.constexpr,
    TRI: tl.constexpr,
    TCJ: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    row0 = pid_r * (4 * TRI)
    col0 = pid_c * (4 * TCJ)

    tr = tl.arange(0, TRI)[:, None]  # tile row inside the patch
    tc = tl.arange(0, TCJ)[None, :]  # tile column inside the patch
    rb = row0 + tr * 4
    cb = col0 + tc * 4
    mr0 = rb + 0 < rows
    mr1 = rb + 1 < rows
    mr2 = rb + 2 < rows
    mr3 = rb + 3 < rows
    mc0 = cb + 0 < cols
    mc1 = cb + 1 < cols
    mc2 = cb + 2 < cols
    mc3 = cb + 3 < cols
    ninf = float("-inf")

    # 16 lanes, lane = 4 * in-tile row + in-tile column
    s00 = tl.load(x_ptr + (rb + 0) * ld + (cb + 0), mask=mr0 & mc0, other=ninf)
    s01 = tl.load(x_ptr + (rb + 0) * ld + (cb + 1), mask=mr0 & mc1, other=ninf)
    s02 = tl.load(x_ptr + (rb + 0) * ld + (cb + 2), mask=mr0 & mc2, other=ninf)
    s03 = tl.load(x_ptr + (rb + 0) * ld + (cb + 3), mask=mr0 & mc3, other=ninf)
    s10 = tl.load(x_ptr + (rb + 1) * ld + (cb + 0), mask=mr1 & mc0, other=ninf)
    s11 = tl.load(x_ptr + (rb + 1) * ld + (cb + 1), mask=mr1 & mc1, other=ninf)
    s12 = tl.load(x_ptr + (rb + 1) * ld + (cb + 2), mask=mr1 & mc2, other=ninf)
    s13 = tl.load(x_ptr + (rb + 1) * ld + (cb + 3), mask=mr1 & mc3, other=ninf)
    s20 = tl.load(x_ptr + (rb + 2) * ld + (cb + 0), mask=mr2 & mc0, other=ninf)
    s21 = tl.load(x_ptr + (rb + 2) * ld + (cb + 1), mask=mr2 & mc1, other=ninf)
    s22 = tl.load(x_ptr + (rb + 2) * ld + (cb + 2), mask=mr2 & mc2, other=ninf)
    s23 = tl.load(x_ptr + (rb + 2) * ld + (cb + 3), mask=mr2 & mc3, other=ninf)
    s30 = tl.load(x_ptr + (rb + 3) * ld + (cb + 0), mask=mr3 & mc0, other=ninf)
    s31 = tl.load(x_ptr + (rb + 3) * ld + (cb + 1), mask=mr3 & mc1, other=ninf)
    s32 = tl.load(x_ptr + (rb + 3) * ld + (cb + 2), mask=mr3 & mc2, other=ninf)
    s33 = tl.load(x_ptr + (rb + 3) * ld + (cb + 3), mask=mr3 & mc3, other=ninf)

    v00 = s00.to(tl.float32)
    v01 = s01.to(tl.float32)
    v02 = s02.to(tl.float32)
    v03 = s03.to(tl.float32)
    v10 = s10.to(tl.float32)
    v11 = s11.to(tl.float32)
    v12 = s12.to(tl.float32)
    v13 = s13.to(tl.float32)
    v20 = s20.to(tl.float32)
    v21 = s21.to(tl.float32)
    v22 = s22.to(tl.float32)
    v23 = s23.to(tl.float32)
    v30 = s30.to(tl.float32)
    v31 = s31.to(tl.float32)
    v32 = s32.to(tl.float32)
    v33 = s33.to(tl.float32)

    k0 = _key16(s00, 0, IS_ABS)
    k1 = _key16(s01, 1, IS_ABS)
    k2 = _key16(s02, 2, IS_ABS)
    k3 = _key16(s03, 3, IS_ABS)
    k4 = _key16(s10, 4, IS_ABS)
    k5 = _key16(s11, 5, IS_ABS)
    k6 = _key16(s12, 6, IS_ABS)
    k7 = _key16(s13, 7, IS_ABS)
    k8 = _key16(s20, 8, IS_ABS)
    k9 = _key16(s21, 9, IS_ABS)
    k10 = _key16(s22, 10, IS_ABS)
    k11 = _key16(s23, 11, IS_ABS)
    k12 = _key16(s30, 12, IS_ABS)
    k13 = _key16(s31, 13, IS_ABS)
    k14 = _key16(s32, 14, IS_ABS)
    k15 = _key16(s33, 15, IS_ABS)

    # odd-even mergesort(16) generated by _network(16): ascending keys, the
    # lane index in the low bits makes the order total (no ties).
    k0, k1 = _cx2(k0, k1)
    k2, k3 = _cx2(k2, k3)
    k0, k2 = _cx2(k0, k2)
    k1, k3 = _cx2(k1, k3)
    k1, k2 = _cx2(k1, k2)
    k4, k5 = _cx2(k4, k5)
    k6, k7 = _cx2(k6, k7)
    k4, k6 = _cx2(k4, k6)
    k5, k7 = _cx2(k5, k7)
    k5, k6 = _cx2(k5, k6)
    k0, k4 = _cx2(k0, k4)
    k1, k5 = _cx2(k1, k5)
    k1, k4 = _cx2(k1, k4)
    k2, k6 = _cx2(k2, k6)
    k3, k7 = _cx2(k3, k7)
    k3, k6 = _cx2(k3, k6)
    k2, k4 = _cx2(k2, k4)
    k3, k5 = _cx2(k3, k5)
    k3, k4 = _cx2(k3, k4)
    k8, k9 = _cx2(k8, k9)
    k10, k11 = _cx2(k10, k11)
    k8, k10 = _cx2(k8, k10)
    k9, k11 = _cx2(k9, k11)
    k9, k10 = _cx2(k9, k10)
    k12, k13 = _cx2(k12, k13)
    k14, k15 = _cx2(k14, k15)
    k12, k14 = _cx2(k12, k14)
    k13, k15 = _cx2(k13, k15)
    k13, k14 = _cx2(k13, k14)
    k8, k12 = _cx2(k8, k12)
    k9, k13 = _cx2(k9, k13)
    k9, k12 = _cx2(k9, k12)
    k10, k14 = _cx2(k10, k14)
    k11, k15 = _cx2(k11, k15)
    k11, k14 = _cx2(k11, k14)
    k10, k12 = _cx2(k10, k12)
    k11, k13 = _cx2(k11, k13)
    k11, k12 = _cx2(k11, k12)
    k0, k8 = _cx2(k0, k8)
    k1, k9 = _cx2(k1, k9)
    k1, k8 = _cx2(k1, k8)
    k2, k10 = _cx2(k2, k10)
    k3, k11 = _cx2(k3, k11)
    k3, k10 = _cx2(k3, k10)
    k2, k8 = _cx2(k2, k8)
    k3, k9 = _cx2(k3, k9)
    k3, k8 = _cx2(k3, k8)
    k4, k12 = _cx2(k4, k12)
    k5, k13 = _cx2(k5, k13)
    k5, k12 = _cx2(k5, k12)
    k6, k14 = _cx2(k6, k14)
    k7, k15 = _cx2(k7, k15)
    k7, k14 = _cx2(k7, k14)
    k6, k12 = _cx2(k6, k12)
    k7, k13 = _cx2(k7, k13)
    k7, k12 = _cx2(k7, k12)
    k4, k8 = _cx2(k4, k8)
    k5, k9 = _cx2(k5, k9)
    k5, k8 = _cx2(k5, k8)
    k6, k10 = _cx2(k6, k10)
    k7, k11 = _cx2(k7, k11)
    k7, k10 = _cx2(k7, k10)
    k6, k8 = _cx2(k6, k8)
    k7, k9 = _cx2(k7, k9)
    k7, k8 = _cx2(k7, k8)

    # Greedy walk from the largest key down.  `bits16` accumulates the selected
    # lane mask, `state` the per-row / per-column counters.
    bits16 = tl.zeros((TRI, TCJ), tl.int32)
    state = tl.zeros((TRI, TCJ), tl.int32)
    bits16, state = _greedy_step(k15, bits16, state)
    bits16, state = _greedy_step(k14, bits16, state)
    bits16, state = _greedy_step(k13, bits16, state)
    bits16, state = _greedy_step(k12, bits16, state)
    bits16, state = _greedy_step(k11, bits16, state)
    bits16, state = _greedy_step(k10, bits16, state)
    bits16, state = _greedy_step(k9, bits16, state)
    bits16, state = _greedy_step(k8, bits16, state)
    bits16, state = _greedy_step(k7, bits16, state)
    bits16, state = _greedy_step(k6, bits16, state)
    bits16, state = _greedy_step(k5, bits16, state)
    bits16, state = _greedy_step(k4, bits16, state)
    bits16, state = _greedy_step(k3, bits16, state)
    bits16, state = _greedy_step(k2, bits16, state)
    bits16, state = _greedy_step(k1, bits16, state)
    bits16, state = _greedy_step(k0, bits16, state)

    ty = packed_ptr.dtype.element_ty
    tyt = packed_t_ptr.dtype.element_ty
    qrow = rb  # (TRI, 1)
    qcol = pid_c * TCJ + tc  # (1, TCJ)
    pc0 = col0 // 2 + tc * 2  # (1, TCJ), even packed slot
    mt = col0 + tc * 4 < rows_t  # transposed row guard, (1, TCJ)
    pct = pid_r * (TRI * 2) + tr * 2  # (TRI, 1), transposed packed slot
    qct = pid_r * TRI + tr  # (TRI, 1), transposed quad buffer column

    # packed / qbuf: in-tile row a -> lanes 4a..4a+3 (ascending column order)
    for a in tl.static_range(4):
        prow = qrow + a
        prow_t = col0 + tc * 4 + a  # (1, TCJ) transposed row
        qv, g0, g1 = _pack_row(
            (bits16 >> (4 * a + 0)) & 1,
            (bits16 >> (4 * a + 1)) & 1,
            (bits16 >> (4 * a + 2)) & 1,
            (bits16 >> (4 * a + 3)) & 1,
            tl.where(a == 0, v00, tl.where(a == 1, v10, tl.where(a == 2, v20, v30))),
            tl.where(a == 0, v01, tl.where(a == 1, v11, tl.where(a == 2, v21, v31))),
            tl.where(a == 0, v02, tl.where(a == 1, v12, tl.where(a == 2, v22, v32))),
            tl.where(a == 0, v03, tl.where(a == 1, v13, tl.where(a == 2, v23, v33))),
        )
        tl.store(qbuf_ptr + prow * qb_stride + qcol, qv.to(tl.int16))
        tl.store(packed_ptr + prow * spm + pc0 + 0, g0.to(ty))
        tl.store(packed_ptr + prow * spm + pc0 + 1, g1.to(ty))
        # packed_trans: same quad on the transposed tile grid (lanes a, 4+a, ...)
        qvt, u0, u1 = _pack_row(
            (bits16 >> (0 + a)) & 1,
            (bits16 >> (4 + a)) & 1,
            (bits16 >> (8 + a)) & 1,
            (bits16 >> (12 + a)) & 1,
            tl.where(a == 0, v00, tl.where(a == 1, v01, tl.where(a == 2, v02, v03))),
            tl.where(a == 0, v10, tl.where(a == 1, v11, tl.where(a == 2, v12, v13))),
            tl.where(a == 0, v20, tl.where(a == 1, v21, tl.where(a == 2, v22, v23))),
            tl.where(a == 0, v30, tl.where(a == 1, v31, tl.where(a == 2, v32, v33))),
        )
        tl.store(qbuf_t_ptr + prow_t * qbt_stride + qct, qvt.to(tl.int16), mask=mt)
        tl.store(
            packed_t_ptr + prow_t * width_t + pct + 0,
            u0.to(tyt),
            mask=mt & (pct + 0 < width_t),
        )
        tl.store(
            packed_t_ptr + prow_t * width_t + pct + 1,
            u1.to(tyt),
            mask=mt & (pct + 1 < width_t),
        )

    # threads_masks: per-tile 16-bit selection mask into the four byte slots
    lo = (bits16 & 0xFF).to(tl.uint8)
    hi = ((bits16 >> 8) & 0xFF).to(tl.uint8)
    q8 = (tr % 2) * 2 + (tc % 2)  # quadrant id of the tile
    ii = pid_r * (TRI // 2) + tr // 2
    jj = pid_c * (TCJ // 2) + tc // 2
    zero_u = tl.zeros((TRI, TCJ), tl.uint8)
    for s in tl.static_range(8):
        is_lo = s == 2 * q8
        is_hi = s == 2 * q8 + 1
        val = tl.where(is_lo, lo, tl.where(is_hi, hi, zero_u))
        offs = masks_ptr + s + ii * sm0 + jj * sm1
        tl.store(offs, val, mask=is_lo | is_hi)


@triton.jit
def _meta_reorder(qbuf_ptr, qb_stride, meta_ptr, m_total_rows, n16, ncols2):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    r = pid_r * 32 + tl.arange(0, 32)
    j = pid_c * 16 + tl.arange(0, 16)
    mr = r < m_total_rows
    mj = j < n16
    mm = mr[:, None] & mj[None, :]
    val = tl.zeros((32, 16), tl.int32)
    for q in tl.static_range(4):
        qv = tl.load(
            qbuf_ptr + r[:, None] * qb_stride + (4 * j + q)[None, :], mask=mm, other=0
        ).to(tl.int32)
        val += (qv & 0xF) << (4 * q)
    # CUTLASS ColumnMajorInterleaved<2> scatter
    dr = (r // 32) * 32 + (r % 8) * 4 + (r % 32) // 8
    dr_ = dr[:, None]
    j_ = j[None, :]
    tr = ((dr_ % 2 == 0) & (j_ % 2 == 1)).to(tl.int32)
    bl = ((dr_ % 2 == 1) & (j_ % 2 == 0)).to(tl.int32)
    dr2 = dr_ + tr - bl
    dc2 = j_ - tr + bl
    off = (dc2 // 2) * (2 * m_total_rows) + dr2 * 2 + (dc2 % 2)
    tl.store(meta_ptr + off, val.to(tl.int16), mask=mm)


def _causal1122(dense):
    """Host fallback for the causal1122 algorithm (per-tile-row capacities)."""
    rows, cols = dense.shape
    rows_r = (rows + 31) // 32 * 32
    cols_r = (cols + 63) // 64 * 64
    padded = torch.full(
        (rows_r, cols_r), float("-inf"), dtype=torch.float32, device=dense.device
    )
    padded[:rows, :cols] = dense.to(torch.float32)
    nb, ntb = rows_r // 4, cols_r // 4
    tiles = padded.view(nb, 4, ntb, 4).permute(0, 2, 1, 3).reshape(-1, 16)
    n = tiles.shape[0]
    k4 = tiles.view(n, 4, 4)
    out = torch.zeros(n, 4, 4, dtype=torch.bool, device=dense.device)
    ar = torch.arange(n, device=dense.device)
    max_row = torch.tensor([1, 1, 2, 2], device=dense.device)
    col_count = torch.zeros(n, 4, dtype=torch.long, device=dense.device)
    for row in range(4):
        vals = k4[:, row, :].clone()
        pos = torch.arange(4, device=dense.device).unsqueeze(0).repeat(n, 1)
        for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
            swap = ~(vals[:, a] < vals[:, b])
            va = vals[:, a].clone()
            pa = pos[:, a].clone()
            vals[:, a] = torch.where(swap, vals[:, b], vals[:, a])
            pos[:, a] = torch.where(swap, pos[:, b], pos[:, a])
            vals[:, b] = torch.where(swap, va, vals[:, b])
            pos[:, b] = torch.where(swap, pa, pos[:, b])
        row_count = torch.zeros(n, dtype=torch.long, device=dense.device)
        for step in range(3, -1, -1):
            which = pos[:, step]
            col = which % 4
            keep = (col_count[ar, col] < 2) & (row_count < max_row[row])
            chosen = torch.zeros(n, 4, dtype=torch.bool, device=dense.device)
            chosen[ar, which] = keep
            out[:, row, :] |= chosen
            row_count += chosen.sum(dim=1)
            col_count += chosen.long()
    sel = out.view(nb, ntb, 16)
    pruned = torch.where(sel, tiles, torch.zeros_like(tiles)).view(nb, ntb, 4, 4)
    # reuse the packing helpers through the generic path below
    return pruned.permute(0, 2, 1, 3).reshape(rows_r, cols_r)


# 8 tile rows (32 input rows) per program patch.
_TRI = 8


def _pick_tcj(cols_r):
    """Largest 4-tile-wide patch TCJ such that 4 * TCJ still divides cols_r."""
    for cand in (64, 32, 16):
        if cols_r % (4 * cand) == 0:
            return cand
    return 16


def _sparse_semi_structured_tile(input, algorithm: str = "", use_cutlass: bool = True):
    logger.debug("GEMS_ENFLAME SPARSE_SEMI_STRUCTURED_TILE")

    dense = input
    if dense.dim() != 2:
        raise RuntimeError("Can only sparsify 2d tensors")
    if dense.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            "_sparse_semi_structured_tile supports only float16 and bfloat16 "
            f"inputs, got {dense.dtype}"
        )
    if dense.stride(1) != 1:
        raise RuntimeError(
            "Can only sparsify contiguous tensors. Sparsify the transpose otherwise."
        )
    rows, cols = dense.shape
    if cols % 32 != 0:
        raise RuntimeError("Number of cols should be multiple of 32")
    if algorithm not in _ALGORITHMS:
        raise RuntimeError(f'Unknown algorithm "{algorithm}"')
    if not use_cutlass:
        raise NotImplementedError(
            "_sparse_semi_structured_tile only implements the CUTLASS metadata "
            "layout (use_cutlass=True)"
        )
    if algorithm == "causal1122":
        raise NotImplementedError(
            '_sparse_semi_structured_tile does not implement the "causal1122" '
            "algorithm yet; use the default or largest_*_greedy"
        )

    rows_r = (rows + 31) // 32 * 32
    cols_r = (cols + 63) // 64 * 64
    tcj = _pick_tcj(cols_r)
    dev = dense.device

    # The kernels below write every element of these buffers, so `empty`
    # avoids paying for seven zero-fill launches on the critical path.
    packed = torch.empty(rows_r, cols_r // 2, dtype=dense.dtype, device=dev)
    meta = torch.empty(rows_r * cols_r // 16, dtype=torch.int16, device=dev)
    rows_t = (cols + 31) // 32 * 32
    width_t = (rows + 63) // 64 * 64 // 2
    packed_t = torch.empty(rows_t, width_t, dtype=dense.dtype, device=dev)
    ncols_t = (rows + 63) // 64 * 64 // 16
    meta_t = torch.empty(rows_t * ncols_t, dtype=torch.int16, device=dev)
    masks = torch.empty(rows_r // 8, cols_r // 8, 8, dtype=torch.uint8, device=dev)
    qbuf = torch.empty(rows_r, cols_r // 4, dtype=torch.int16, device=dev)
    qbuf_t = torch.empty(rows_t, rows_r // 4, dtype=torch.int16, device=dev)

    # One program per 32-row x 4 * tcj-column patch.  A single warp keeps the 16
    # elements of every tile resident in registers and measured fastest.
    grid = (rows_r // (4 * _TRI), cols_r // (4 * tcj))
    _ssst_kernel[grid](
        dense,
        dense.stride(0),
        rows,
        cols,
        packed,
        packed.stride(0),
        packed_t,
        rows_t,
        width_t,
        qbuf,
        qbuf.stride(0),
        qbuf_t,
        qbuf_t.stride(0),
        masks,
        masks.stride(0),
        masks.stride(1),
        algorithm == "largest_abs_values_greedy",
        _TRI,
        tcj,
        num_warps=1,
    )
    _meta_reorder[(rows_r // 32, (cols_r // 16 + 15) // 16)](
        qbuf,
        qbuf.stride(0),
        meta,
        rows_r,
        cols_r // 16,
        cols_r // 32,
    )
    _meta_reorder[(rows_t // 32, (ncols_t + 15) // 16)](
        qbuf_t,
        qbuf_t.stride(0),
        meta_t,
        rows_t,
        ncols_t,
        ncols_t // 2,
    )
    # The reorder kernels scatter into the CUTLASS physical buffer; the aten
    # output view exposes it as (rows, 2, cols/32) with element (r, k, c) at
    # offset 2r + k + rows*2*c.
    meta = meta.view(cols_r // 32, rows_r, 2).permute(1, 2, 0)
    meta_t = meta_t.view(ncols_t // 2, rows_t, 2).permute(1, 2, 0)
    return (packed, meta, packed_t, meta_t, masks)


# Module-level handle keeps the Library (and its kernel) alive for the
# process; the registration itself must never break the flag_gems import.
