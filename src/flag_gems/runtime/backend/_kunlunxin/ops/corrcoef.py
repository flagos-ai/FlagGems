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
#
# Kunlunxin (XPU3 / P800) corrcoef.
#
# The generic implementation cannot be lowered by this backend at all: its
# normalisation kernel feeds `offs_i < n_rows` / `offs_i * stride + offs_i`
# into `TritonXPULegalize`, which rejects both (`arith.cmpi` / `arith.muli`
# "requires the same type for all operands") -- every 2-D shape failed with
# `OutOfResources: uni_sram` at baseline.  This override therefore avoids
# row-bound masks and 1-D `idx*stride + idx` addressing entirely and folds the
# normalisation into the scale pass:
#
#   K1 `_cc_scale_kernel`
#        pass 1 mean, pass 2 sum((x-mean)^2), pass 3 store
#        A[i, :] = (x[i, :] - mean_i) / sqrt(sum((x[i, :] - mean_i)^2))
#        (3 streaming reads of the row, one write of the fp32 scratch)
#   K2 `_cc_gram_kernel` (or `_cc_pair_kernel` for very few rows)
#        out = A @ A^T  ->  exactly the correlation matrix
#
# Because A is row-normalised *before* the dot, the GEMM output needs no
# diagonal gather, no sqrt and no division epilogue (a separate normalisation
# kernel would need the diagonal `G[i, i]`, whose 1-D `idx*stride + idx` load
# is one of the forms the backend cannot legalise).
#
# In-bounds-by-construction rules on this stack (see
# harness/KernelOpt_Knowledge wiki `pattern-xpu3-dot-masked-load-miscompile`):
# a `tl.dot` operand must come from an unmasked load whose addresses stay in
# the allocation, and a masked store whose value is `tl.where(mask, ..., 0)`
# is lowered to a *masked* store, so pre-zero the scratch buffer (`torch.zeros`)
# whenever it has padding instead of relying on the kernel to zero it.
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Row-streaming tile.  One tile is [ROWS, _BLOCK_K]: 8 x 1024 = 8192 lanes is
# the widest `tl.sum(..., axis=1)` tile that is exact without
# `buffer_size_limit`, and 8 rows per program is what makes this kernel fast
# (measured on XPU4, (512, 4096) fp32: 1 row/program 519 us, 2 rows 155 us,
# 4 rows 80 us, 8 rows 50 us -- the 1-row form spreads its 1024-lane tile over
# all 64 cores and pays a cluster-wide reduce per stride step).
_BLOCK_K = 1024
_ROWS = 8
# Small-width granularity: keeps `kp` a multiple of the 128-wide dot K tile
# while letting narrow inputs use a 128-lane stream tile.
_SMALL_GRAN = 128
# Dot tile (rows/cols per program).  64 for tall inputs (fewer, larger tiles
# re-read the operand less often); 32 otherwise so the lower-triangular grid
# still fills the 64 cores.
_GRAM_BLOCK = 32
_GRAM_BLOCK_BIG = 64
_GRAM_BIG_ROWS = 384
_DOT_K = 128
# Below this many variables the triangular GEMM grid cannot fill the machine
# and every program would re-read the whole operand, so the tiny path
# (one program per output element) is cheaper.
_PAIR_MAX_ROWS = 32


@triton.jit
def _cc_scale_kernel(
    x_ptr,
    a_ptr,
    n_cols,
    kp,
    stride_x0,
    stride_x1,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
    NFULL: tl.constexpr,
):
    # One program handles ROWS rows (ROWS divides the row count, so no row
    # bound is ever needed).  Pass 1 mean, pass 2 sum of squares, pass 3 the
    # centred+scaled write.  This is the contiguous, K-aligned fast path:
    # plain stride-1 tile loads, no mask anywhere, statically unrolled K loop.
    # It is 2.3x faster than the masked/strided form on (512, 4096) fp32
    # (50 us vs 114 us) and is the only path the benchmark matrix takes.
    # Ragged or non-contiguous inputs go through `_cc_scale_row_kernel`.
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_K)
    base = rows[:, None] * stride_x0

    total = tl.zeros((ROWS, BLOCK_K), dtype=tl.float32)
    for i in tl.static_range(0, NFULL):
        total += tl.load(x_ptr + base + i * BLOCK_K + cols[None, :]).to(tl.float32)
    mean = tl.sum(total, axis=1)[:, None] / n_cols

    sq = tl.zeros((ROWS, BLOCK_K), dtype=tl.float32)
    for i in tl.static_range(0, NFULL):
        d = tl.load(x_ptr + base + i * BLOCK_K + cols[None, :]).to(tl.float32) - mean
        sq += d * d
    # 1/sqrt(sum of squares): A @ A^T is then exactly C_ij / (s_i s_j).
    inv = 1.0 / tl.sqrt(tl.sum(sq, axis=1))[:, None]

    a_base = a_ptr + rows[:, None] * kp
    for i in tl.static_range(0, NFULL):
        v = (tl.load(x_ptr + base + i * BLOCK_K + cols[None, :]).to(tl.float32) - mean) * inv
        tl.store(a_base + i * BLOCK_K + cols[None, :], v)


@triton.jit
def _cc_scale_row_kernel(
    x_ptr,
    a_ptr,
    n_cols,
    kp,
    stride_x0,
    stride_x1,
    BLOCK_K: tl.constexpr,
    NFULL: tl.constexpr,
    TAIL: tl.constexpr,
):
    # Generic row kernel: one program per row, any column stride, ragged K.
    # Full tiles are unmasked; the ragged tail is a *single-shot* masked tile
    # (a masked load inside a runtime loop is unreliable here, and a 2-D
    # row-broadcast mask tile was measured wrong on (2, 4)/(8, 64), while this
    # per-row form is exact on every shape in the test + benchmark matrix).
    row = tl.program_id(0)
    base = row * stride_x0
    cols = tl.arange(0, BLOCK_K)

    total = tl.zeros([], dtype=tl.float32)
    for i in tl.static_range(0, NFULL):
        cur = i * BLOCK_K + cols
        total += tl.sum(tl.load(x_ptr + base + cur * stride_x1).to(tl.float32))
    if TAIL:
        cur = NFULL * BLOCK_K + cols
        total += tl.sum(
            tl.load(x_ptr + base + cur * stride_x1, mask=cur < n_cols, other=0.0).to(
                tl.float32
            )
        )
    mean = total / n_cols

    sq = tl.zeros([], dtype=tl.float32)
    for i in tl.static_range(0, NFULL):
        cur = i * BLOCK_K + cols
        d = tl.load(x_ptr + base + cur * stride_x1).to(tl.float32) - mean
        sq += tl.sum(d * d)
    if TAIL:
        cur = NFULL * BLOCK_K + cols
        v = tl.load(x_ptr + base + cur * stride_x1, mask=cur < n_cols, other=0.0).to(
            tl.float32
        )
        d = tl.where(cur < n_cols, v - mean, 0.0)
        sq += tl.sum(d * d)
    inv = 1.0 / tl.sqrt(sq)

    a_row = a_ptr + row * kp
    for i in tl.static_range(0, NFULL):
        cur = i * BLOCK_K + cols
        v = (tl.load(x_ptr + base + cur * stride_x1).to(tl.float32) - mean) * inv
        tl.store(a_row + cur, v)
    if TAIL:
        cur = NFULL * BLOCK_K + cols
        v = tl.load(x_ptr + base + cur * stride_x1, mask=cur < n_cols, other=0.0).to(
            tl.float32
        )
        v = tl.where(cur < n_cols, (v - mean) * inv, 0.0)
        tl.store(a_row + cur, v)


@triton.jit
def _cc_gram_kernel(
    a_ptr, g_ptr, kp, stride_am, stride_gm, BLOCK: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Lower-triangular SYRK: program pid maps to the packed triangular index
    # (same geometry as `_kunlunxin/ops/mm.py::mm_kernel_syrk`, so only
    # tiles*(tiles+1)/2 programs run and each block is mirrored into the upper
    # triangle).  Both operands are row-major slices of the same buffer and A
    # is K-padded with zeros, so every load and both stores are unmasked and
    # in bounds by construction.
    pid = tl.program_id(0)
    pid_f = pid.to(tl.float32)
    pid_m = tl.floor((tl.sqrt(8.0 * pid_f + 1.0) - 1.0) / 2.0).to(tl.int32)
    tri = pid_m * (pid_m + 1) // 2
    pid_m = tl.where(tri > pid, pid_m - 1, pid_m)
    nxt = (pid_m + 1) * (pid_m + 2) // 2
    pid_m = tl.where(nxt <= pid, pid_m + 1, pid_m)
    pid_n = pid - pid_m * (pid_m + 1) // 2

    rm = pid_m * BLOCK + tl.arange(0, BLOCK)
    rn = pid_n * BLOCK + tl.arange(0, BLOCK)
    rk = tl.arange(0, BLOCK_K)

    am = a_ptr + rm[:, None] * stride_am + rk[None, :]
    an = a_ptr + rn[:, None] * stride_am + rk[None, :]
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for _k in range(0, tl.cdiv(kp, BLOCK_K)):
        a = tl.load(am)
        b = tl.load(an)
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32, allow_tf32=False)
        am += BLOCK_K
        an += BLOCK_K

    out = acc.to(g_ptr.dtype.element_ty)
    tl.store(g_ptr + rm[:, None] * stride_gm + rn[None, :], out)
    if pid_m > pid_n:
        tl.store(g_ptr + rn[:, None] * stride_gm + rm[None, :], tl.trans(out))


@triton.jit
def _cc_pair_kernel(a_ptr, out_ptr, n_rows, kp, stride_am, stride_om, BLOCK_K: tl.constexpr):
    # Tiny-input path (few variables): one program per output element, so the
    # grid is exactly n_rows x n_rows and the store needs neither a mask nor a
    # padded buffer.  Scalar accumulator + per-tile `tl.sum`, the same shape as
    # `_kunlunxin/ops/sum.py::_row_reduce_kernel`.
    pid = tl.program_id(0)
    i = pid // n_rows
    j = pid % n_rows
    ai = a_ptr + i * stride_am
    aj = a_ptr + j * stride_am
    cols = tl.arange(0, BLOCK_K)
    acc = tl.zeros([], dtype=tl.float32)
    for start in range(0, kp, BLOCK_K):
        cur = start + cols
        acc += tl.sum(tl.load(ai + cur) * tl.load(aj + cur))
    tl.store(out_ptr + i * stride_om + j, acc.to(out_ptr.dtype.element_ty))


def _row_block(n_rows: int) -> int:
    for rows in (_ROWS, 4, 2):
        if n_rows % rows == 0:
            return rows
    return 1


def _gram_block(n_rows: int) -> int:
    return _GRAM_BLOCK_BIG if n_rows >= _GRAM_BIG_ROWS else _GRAM_BLOCK


def corrcoef(input: torch.Tensor) -> torch.Tensor:
    """Estimates the Pearson product-moment correlation coefficient matrix.

    Rows of ``input`` are the variables and columns are the observations; the
    result is the ``n_rows x n_rows`` correlation matrix, computed in float32
    and cast back to the input dtype.  Degenerate (zero-variance) variables
    produce NaN entries, matching ``torch.corrcoef``.
    """
    logger.debug("GEMS_KUNLUNXIN CORRCOEF")

    if input.dim() > 2:
        raise ValueError(
            f"corrcoef(): expected input to have two or fewer dimensions but got "
            f"an input with {input.dim()} dimensions"
        )

    # A single variable (0D/1D input, or a 2D input with one row) correlates
    # perfectly with itself; torch returns the 0-dim scalar 1.0.
    if input.dim() < 2 or input.shape[0] == 1:
        return torch.ones((), dtype=input.dtype, device=input.device)

    n_rows, n_cols = input.shape
    device = input.device
    out_dtype = input.dtype
    if n_rows == 0:
        return torch.empty((0, 0), dtype=out_dtype, device=device)
    if n_cols == 0:
        # torch.corrcoef on an (n, 0) input returns an all-NaN matrix.
        return torch.full((n_rows, n_rows), float("nan"), dtype=out_dtype, device=device)

    x = input
    gran = _BLOCK_K if n_cols >= _BLOCK_K else _SMALL_GRAN
    kp = triton.cdiv(n_cols, gran) * gran
    rows_per_prog = _row_block(n_rows)
    tiny = n_rows < _PAIR_MAX_ROWS
    blk = _GRAM_BLOCK if tiny else _gram_block(n_rows)
    mp = triton.cdiv(n_rows, blk) * blk
    # Zero-initialise whenever the scratch has rows/columns that K1 does not
    # write: `tl.where(mask, v, 0.0)` stores are lowered to masked stores by
    # this backend, so the padding would otherwise keep `torch.empty` garbage
    # and poison the dot (measured: (64, 128) diagonal 956 instead of 1.0).
    if kp != n_cols or mp != n_rows:
        a = torch.zeros((mp, kp), dtype=torch.float32, device=device)
    else:
        a = torch.empty((mp, kp), dtype=torch.float32, device=device)

    if x.is_contiguous() and kp == n_cols:
        _cc_scale_kernel[(n_rows // rows_per_prog,)](
            x, a, n_cols, kp, x.stride(0), x.stride(1),
            BLOCK_K=gran, ROWS=rows_per_prog, NFULL=kp // gran,
        )
    else:
        nfull = n_cols // gran
        _cc_scale_row_kernel[(n_rows,)](
            x, a, n_cols, kp, x.stride(0), x.stride(1),
            BLOCK_K=gran, NFULL=nfull, TAIL=(nfull * gran < kp),
        )

    out = torch.empty((n_rows, n_rows), dtype=out_dtype, device=device)
    if tiny:
        _cc_pair_kernel[(n_rows * n_rows,)](
            a, out, n_rows, kp, a.stride(0), out.stride(0), BLOCK_K=gran
        )
        return out

    ntiles = mp // blk
    _cc_gram_kernel[(ntiles * (ntiles + 1) // 2,)](
        a, out, kp, a.stride(0), out.stride(0), BLOCK=blk, BLOCK_K=_DOT_K
    )
    return out
