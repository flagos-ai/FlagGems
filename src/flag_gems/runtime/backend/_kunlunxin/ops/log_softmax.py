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
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# tle.gpu row-reduce path (KL3): log_softmax over the contiguous last axis.
#
# The pointer kernels below reduce the row axis with a serial-chain tl.max /
# tl.sum on XPU, which is the dominant cost on the long-row shapes. tle.gpu
# moves the tile GM -> LM with the cluster DMA and keeps the reduce core-local,
# streaming the row in YBLOCK columns, so the running max / sum-of-exp stay
# [XBLOCK] vectors instead of a full-row tile. log_softmax needs two reductions
# (max, then sum of exp) plus a subtract, so this is two kernels: an
# online-softmax reduce that writes (row max, log-sum-exp), and a pass that
# computes out = x - m - log_z. The partial-column fill is -inf (the max
# identity; exp(-inf - m) == 0 keeps the sum exact).
#
# Only fp16/fp32/bf16, contiguous [M, N] with N >= _TLE_MIN_N_FUSED (and
# M >= _TLE_MIN_M in the medium-N range) take this path; f64, N == 1, strided
# views and small rows keep the pointer kernels.
# =============================================================================
_TLE_CORE_NUM = 64
# The online-softmax reduce materialises a [XBLOCK, YBLOCK] fp32 `exp(x - m)`
# tile per step on top of the loaded tile, roughly twice the register/LM
# footprint of the plain sum row-reduce. Halve the per-core LM tile budget so
# that intermediate still fits (4 KB/core overflows for the tiles this path
# emits).
_TLE_LM_BYTES_PER_CORE = 2048
_TLE_XBLOCK = 256
_TLE_CLUSTERS = 8
_TLE_WIDE_ROW_BYTES = 32768
_TLE_NARROW_ROW_BYTES = 2048
# The fused single-kernel tle path (one launch, no GM round-trip for the row
# statistics) beats both the pointer singlepass and the two-launch reduce on
# every pow2 shape measured, so the gate is now about grid size, not row
# length: a program must cover enough rows for the 2x row read to pay off.
#   * N < 256: too short for the tle row tiling to beat the pointer path.
#   * M < 1024 in the medium-N range: grid degenerates to 1-2 programs (e.g.
#     (256, 256) -> grid=1), launch-bound; the singlepass wins there. Long
#     rows (N >= _TLE_MIN_N) keep their verified wide-row tiling regardless of
#     M, which already measures fine for (100, 65536) / (64, 65536).
_TLE_MIN_N = 8192
_TLE_MIN_M = 1024
# FG_LOGSOFTMAX_MIN_N_FUSED overrides the medium-N gate for A/B routing
# comparisons (e.g. "8192" reproduces the pre-medium-N behavior where only
# N >= 8192 rows took the fused kernel). Production default is 256.
_TLE_MIN_N_FUSED = int(os.environ.get("FG_LOGSOFTMAX_MIN_N_FUSED", "256"))

_TLE_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


def _tle_available():
    """`tle.gpu` exists only on the xpu3 (KL3) cluster pipeline."""
    if not _HAS_TLE:
        return False
    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


_TLE_AVAILABLE = _tle_available()


def _npo2(x):
    """`triton.next_power_of_2` is a few us a call, too slow for the small-shape
    host path; this is the same value without the dispatcher round trip."""
    return 1 << (x - 1).bit_length() if x > 1 else 1


_TLE_ROW_GEOM = {}
_TLE_GEOM_MISS = object()


def _tle_row_geom(M, N, itemsize):
    """Cached [M, N] row-reduce tiling: `(xblock, yblock, row_blocks)`.

    Same trade as the sum row-reduce: XBLOCK is rows per program, YBLOCK is the
    LM budget divided by it. Aim for one program per cluster, then let the row
    length pull XBLOCK down when it is long enough to need a long YBLOCK, or up
    when the reduce is small enough that launch and result write are all there
    is. XBLOCK may exceed M -- both copies clamp to the descriptor extents.
    """
    geom_key = (M, N, itemsize)
    geom = _TLE_ROW_GEOM.get(geom_key, _TLE_GEOM_MISS)
    if geom is not _TLE_GEOM_MISS:
        return geom
    xblock = min(_TLE_XBLOCK, max(128, _npo2(-(-M // _TLE_CLUSTERS))))
    row_bytes = N * itemsize
    if row_bytes >= _TLE_WIDE_ROW_BYTES:
        # Long row: buy YBLOCK. The online-softmax budget is halved (2048
        # bytes/core), so at XBLOCK == 64 YBLOCK is only 512; dropping XBLOCK
        # to 32 lets YBLOCK reach 1024 (the good width) for the long rows this
        # path serves.
        xblock = max(_TLE_CORE_NUM // 2, xblock // 4)
    elif row_bytes <= _TLE_NARROW_ROW_BYTES:
        xblock = _TLE_XBLOCK if M > _TLE_CORE_NUM else 128
    if _TLE_MIN_N_FUSED <= N < _TLE_MIN_N:
        # Medium row: empirical (xblock, yblock) per (itemsize, N) bucket.
        # Rule: the widest YBLOCK the 4KB/core LM budget allows (fewer serial
        # reduce-chain restarts), then XBLOCK for enough programs.
        # itemsize 4 (fp32): yb is capped at 1024 because [64,2048] fp32 =
        # 512KB = 8KB/core exceeds the LM budget.
        if itemsize == 2:
            if N <= 512:
                xblock, yblock = 128, 512
            elif N <= 1024:
                xblock, yblock = 32, 1024
            elif N <= 2048:
                xblock, yblock = 32, 2048
            else:  # 4096 .. 8191
                xblock, yblock = 64, 2048
        else:  # fp32
            if N <= 512:
                xblock, yblock = 128, 512
            elif N <= 1024:
                xblock, yblock = 64, 1024
            else:  # 2048 .. 8191
                xblock, yblock = 64, 1024
        while yblock > N and yblock > 1:
            yblock >>= 1
        geom = (xblock, yblock, -(-M // xblock))
        _TLE_ROW_GEOM[geom_key] = geom
        return geom

    # Wide row (N >= 8192) and narrow row (row_bytes <= 2048): nominal LM
    # budget split by XBLOCK, no medium-row doubling.
    tile_elems = _TLE_LM_BYTES_PER_CORE * _TLE_CORE_NUM // itemsize
    yblock = max(1, min(_npo2(N), tile_elems // xblock))
    while yblock > N and yblock > 1:
        yblock >>= 1
    geom = (xblock, yblock, -(-M // xblock))
    _TLE_ROW_GEOM[geom_key] = geom
    return geom


@triton.jit(
    do_not_specialize=["N"],
    do_not_specialize_on_alignment=["a_desc", "m_desc", "lz_desc"],
)
def _tle_logsoftmax_reduce_kernel(
    a_desc,
    m_desc,
    lz_desc,
    N,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    """Online-softmax row reduce: writes (row max, log-sum-exp) per row.

    The reduction axis is streamed in YBLOCK columns; the running max and
    sum-of-exp stay [XBLOCK] vectors, so each program reduces whole rows for
    its slice of the tile with no cross-core barrier. Partial tiles are not
    masked: the copy clamps to the descriptor, and the short last step is
    filled with -inf first so the pad columns cannot leak into the running max
    (their exp contributes 0 to the sum).
    """
    pid = tl.program_id(0)
    row_off = pid * XBLOCK

    a_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    m_lmem = tle.gpu.alloc([XBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.lmem)
    lz_lmem = tle.gpu.alloc([XBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.lmem)

    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
    m_ptrs = tle.gpu.local_ptr(m_lmem, (tl.arange(0, XBLOCK),))
    lz_ptrs = tle.gpu.local_ptr(lz_lmem, (tl.arange(0, XBLOCK),))

    m = tl.full([XBLOCK], value=float("-inf"), dtype=tl.float32)
    z = tl.zeros([XBLOCK], dtype=tl.float32)
    for coff in tl.range(0, N, YBLOCK):
        if NEED_ZERO:
            if coff + YBLOCK > N:
                tl.store(
                    a_ptrs,
                    tl.full([XBLOCK, YBLOCK], value=float("-inf"), dtype=IN_DTYPE),
                )
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        x = tl.load(a_ptrs).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 1))
        all_neg_inf = m_new == float("-inf")
        z = tl.where(
            all_neg_inf,
            z,
            z * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new[:, None]), 1),
        )
        m = m_new
    tl.store(m_ptrs, m)
    tl.store(lz_ptrs, tl.log(z))
    tle.gpu.copy(m_lmem, m_desc, [XBLOCK], [row_off])
    tle.gpu.copy(lz_lmem, lz_desc, [XBLOCK], [row_off])


@triton.jit
def _logsoftmax_pass_kernel(
    out_ptr,
    x_ptr,
    m_ptr,
    lz_ptr,
    N,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Elementwise out = x - m[row] - log_z[row]; reads x a second time.

    Plain pointer pass, one row per program: the reduce kernel already produced
    (row max, log-sum-exp) into m/lz, and this pass is memory-bound. The
    full-chunk path is unmasked so the load/store lowers to block-DMA; a masked
    2D/1D tile on this backend falls back to per-lane gather and is orders of
    magnitude slower, so it is kept only for the short last block.
    """
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off = pid_m * N + n_offsets
    m = tl.load(m_ptr + pid_m)
    lz = tl.load(lz_ptr + pid_m)
    if NEED_MASK:
        mask = n_offsets < N
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + off, x - m - lz, mask=mask)
    else:
        x = tl.load(x_ptr + off).to(tl.float32)
        tl.store(out_ptr + off, x - m - lz)


@triton.jit(
    do_not_specialize=["N"],
    do_not_specialize_on_alignment=["a_desc", "out_desc"],
)
def _tle_logsoftmax_fused_kernel(
    a_desc,
    out_desc,
    N,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
    ONE_READ: tl.constexpr = False,
):
    """Fused reduce + normalize for a long-row [M, N] log_softmax (N >= 8192).

    One kernel: pass 1 streams the row in YBLOCK chunks computing online
    (max, sum-exp) into per-row [XBLOCK] accumulators held in registers; pass 2
    streams again and writes out = x - m - log_z. The row is read from GM twice
    (too long to hold), but there is one launch and the per-row statistics never
    round-trip through GM -- the separate reduce kernel's LM-strip writeback of
    m/log_z costs more than a launch. Same structure as the norm skill's WIDE leg.

    ONE_READ: when YBLOCK == N the whole row fits the LM tile, so pass 2 can
    normalize the tile in place and store once: exactly 1 GM read + 1 GM write
    per program (the pass-2 re-copy, even L2-served, is pure waste). NOT
    enabled for every (dtype, tile): the bf16 [128,512] LM store of the result
    miscompiles while [32,1024] bf16 is exact, so the routing below gates
    ONE_READ to the verified (dtype, tile) pairs.
    """
    pid = tl.program_id(0)
    row_off = pid * XBLOCK

    a0 = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a0_ptrs = tle.gpu.local_ptr(a0, (row_ids, col_ids))

    if ONE_READ:
        # yb == N (single chunk): load once, reduce, normalize in place, store.
        tle.gpu.copy(a_desc, a0, [XBLOCK, YBLOCK], [row_off, 0])
        x = tl.load(a0_ptrs).to(tl.float32)
        m = tl.max(x, 1)
        z = tl.sum(tl.exp(x - m[:, None]), 1)
        o = x - m[:, None] - tl.log(z)[:, None]
        tl.store(a0_ptrs, o.to(OUT_DTYPE))
        tle.gpu.copy(a0, out_desc, [XBLOCK, YBLOCK], [row_off, 0])
        return

    full_n = N - N % YBLOCK
    m = tl.full([XBLOCK], value=float("-inf"), dtype=tl.float32)
    z = tl.zeros([XBLOCK], dtype=tl.float32)
    # ---- pass 1: online-softmax reduce ----
    for c in tl.range(0, full_n, YBLOCK):
        tle.gpu.copy(a_desc, a0, [XBLOCK, YBLOCK], [row_off, c])
        x = tl.load(a0_ptrs).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 1))
        z = z * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new[:, None]), 1)
        m = m_new
    if NEED_ZERO:
        tl.store(
            a0_ptrs, tl.full([XBLOCK, YBLOCK], value=float("-inf"), dtype=IN_DTYPE)
        )
        tle.gpu.copy(a_desc, a0, [XBLOCK, YBLOCK], [row_off, full_n])
        x = tl.load(a0_ptrs).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 1))
        z = z * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new[:, None]), 1)
        m = m_new
    log_z = tl.log(z)

    # ---- pass 2: normalize ----
    for c in tl.range(0, full_n, YBLOCK):
        tle.gpu.copy(a_desc, a0, [XBLOCK, YBLOCK], [row_off, c])
        x = tl.load(a0_ptrs).to(tl.float32)
        o = x - m[:, None] - log_z[:, None]
        tl.store(a0_ptrs, o.to(OUT_DTYPE))
        tle.gpu.copy(a0, out_desc, [XBLOCK, YBLOCK], [row_off, c])
    if NEED_ZERO:
        tl.store(
            a0_ptrs, tl.full([XBLOCK, YBLOCK], value=float("-inf"), dtype=IN_DTYPE)
        )
        tle.gpu.copy(a_desc, a0, [XBLOCK, YBLOCK], [row_off, full_n])
        x = tl.load(a0_ptrs).to(tl.float32)
        o = x - m[:, None] - log_z[:, None]
        tl.store(a0_ptrs, o.to(OUT_DTYPE))
        tle.gpu.copy(a0, out_desc, [XBLOCK, YBLOCK], [row_off, full_n])


def _tle_logsoftmax(out, inp, M, N):
    """log_softmax over a contiguous [M, N] last axis; False if not expressible."""
    if not _TLE_AVAILABLE:
        return False
    if N < _TLE_MIN_N_FUSED:
        return False
    if N < _TLE_MIN_N and M < _TLE_MIN_M:
        # Medium row, few rows: the fused kernel's grid collapses to 1-2
        # programs (e.g. (256, 256) -> xblock=256, grid=1) and its 2x row read
        # loses to the single-pass pointer path. Only long rows keep the tle
        # path at low M (their wide-row tiling still parallelises well).
        return False
    if inp.dtype not in _TLE_TL_DTYPE or out.dtype not in _TLE_TL_DTYPE:
        return False
    if not inp.is_contiguous() or not out.is_contiguous():
        return False
    a = inp.view(M, N)
    c = out.view(M, N)
    xblock, yblock, row_blocks = _tle_row_geom(M, N, inp.element_size())
    if N % yblock != 0:
        # Non-pow2 tail: the fused kernel's bf16 pass (an LM store of a bf16
        # result) is wrong on this backend for a partial last chunk, so keep
        # the verified two-kernel path here. All benchmark long-row shapes are
        # pow2 and take the fused kernel below.
        m = torch.empty((M,), dtype=torch.float32, device=inp.device)
        lz = torch.empty((M,), dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            _tle_logsoftmax_reduce_kernel[(row_blocks,)](
                TensorDescriptor.from_tensor(a, block_shape=[xblock, yblock]),
                TensorDescriptor.from_tensor(m, block_shape=[xblock]),
                TensorDescriptor.from_tensor(lz, block_shape=[xblock]),
                N,
                XBLOCK=xblock,
                YBLOCK=yblock,
                IN_DTYPE=_TLE_TL_DTYPE[inp.dtype],
                NEED_ZERO=True,
            )
            block_n = min(8192, _npo2(N))
            _logsoftmax_pass_kernel[(M, triton.cdiv(N, block_n))](
                c,
                a,
                m,
                lz,
                N,
                BLOCK_N=block_n,
                NEED_MASK=(N % block_n != 0),
            )
    else:
        # ONE_READ single-pass (whole row in LM, yb == N): verified exact and
        # faster than the two-read form on the wide-row tiles. bf16 [128,512]
        # stays two-read: its in-place bf16 LM store miscompiles there while
        # the [32,1024] bf16 tile is exact.
        one_read = (
            yblock == N
            and N in (512, 1024)
            and (
                inp.dtype in (torch.float16, torch.float32)
                or (inp.dtype == torch.bfloat16 and N == 1024)
            )
        )
        with torch_device_fn.device(inp.device):
            _tle_logsoftmax_fused_kernel[(row_blocks,)](
                TensorDescriptor.from_tensor(a, block_shape=[xblock, yblock]),
                TensorDescriptor.from_tensor(c, block_shape=[xblock, yblock]),
                N,
                XBLOCK=xblock,
                YBLOCK=yblock,
                IN_DTYPE=_TLE_TL_DTYPE[inp.dtype],
                OUT_DTYPE=_TLE_TL_DTYPE[out.dtype],
                NEED_ZERO=False,
                ONE_READ=one_read,
            )
    return True


@triton.jit
def _strided_copy_kernel(
    dst,
    src,
    N,
    D0,
    D1,
    D2,
    D3,
    D4,
    D5,
    D6,
    D7,
    SS0,
    SS1,
    SS2,
    SS3,
    SS4,
    SS5,
    SS6,
    SS7,
    DS0,
    DS1,
    DS2,
    DS3,
    DS4,
    DS5,
    DS6,
    DS7,
    BLOCK: tl.constexpr,
):
    # Elementwise same-shape strided copy: dst[i] = src[i] over the logical
    # index space (dims D0..D7 outermost-first, front-padded with 1s). No
    # native copy_/copy primitives: every lane's source/destination offset is
    # computed from the two stride vectors.
    pid = tl.program_id(axis=0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N
    idx = offs
    i7 = idx % D7
    idx = idx // D7
    i6 = idx % D6
    idx = idx // D6
    i5 = idx % D5
    idx = idx // D5
    i4 = idx % D4
    idx = idx // D4
    i3 = idx % D3
    idx = idx // D3
    i2 = idx % D2
    idx = idx // D2
    i1 = idx % D1
    i0 = idx // D1
    s_off = (
        i0 * SS0
        + i1 * SS1
        + i2 * SS2
        + i3 * SS3
        + i4 * SS4
        + i5 * SS5
        + i6 * SS6
        + i7 * SS7
    )
    d_off = (
        i0 * DS0
        + i1 * DS1
        + i2 * DS2
        + i3 * DS3
        + i4 * DS4
        + i5 * DS5
        + i6 * DS6
        + i7 * DS7
    )
    vals = tl.load(src + s_off, mask=mask)
    tl.store(dst + d_off, vals, mask=mask)


def _strided_copy(dst, src):
    # Same-shape strided copy without native copy_/copy: the gems copy_
    # override is wedged out of this path (its tle kernel is unreliable on
    # strided/transposed operands), so all offset math happens in the kernel
    # from the two stride vectors.
    assert tuple(dst.shape) == tuple(src.shape), "shape mismatch"
    n = dst.numel()
    if n:
        dims = list(src.shape)
        ss = list(src.stride())
        ds = list(dst.stride())
        pad = 8 - len(dims)
        assert pad >= 0, "strided copy supports up to 8 dims"
        dims = [1] * pad + dims
        ss = [0] * pad + ss
        ds = [0] * pad + ds
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        with torch_device_fn.device(dst.device):
            _strided_copy_kernel[grid](
                dst,
                src,
                n,
                *dims,
                *ss,
                *ds,
                BLOCK=BLOCK,
            )
    return dst


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


def _prev_pow2(x):
    x = max(1, int(x))
    return 1 << (x.bit_length() - 1)


# ------------------------  forward -------------------------------
# XPU dispatch: the old masked 2D multirow [TILE_M, N] tile and the per-row
# online inner kernel are far slower than the single-pass paths. Replaced with:
#   * N <= 4096: single-pass [TILE_M, N] tile with order-preserving uint32-key
#     integer max (the fp wide-row tl.max is a serial chain on XPU; the int-key
#     max is faster), TILE_M bucketed by N, always-true row mask compiled
#     away (NEED_MASK); a masked tail launch covers M % TILE_M != 0 rows.
#   * N > 4096: chunk split partial/combine/pass with FLAT grids and
#     pid*BN (BN constexpr) offsets -> contiguous block DMA (runtime
#     row*N + c*BN offsets collapse to discrete gathers on XPU). Full
#     8192-wide chunks are unmasked (masked column tiles miscompile); a
#     non-multiple tail is split into <= 4096-lane 1D masked pieces (masked
#     reduces are exact up to 4096 lanes). A fused combine+pass kernel
#     miscompiles on this backend -> 3-kernel structure.
# The per-row online inner kernel below is kept for the K>1 (interior-dim)
# path only (grid=(M*K,)); the old masked 2D multirow kernel is removed.
FWD_MULTIROW_MAX_N = 4096  # single-pass 2D tile family (<= 8K elems/prog)
FWD_CHUNK_BN = 8192  # big-N chunk width (tl.sum/tl.max lane-safety bound)
FWD_TAIL_PIECE = 4096  # masked 1D tail pieces kept <= 4096 lanes (exact)
# TILE_M buckets per N. Non-power-of-2 N < 64 needs TILE_M>=64 to compile
# correctly; handled in the dispatch.
FWD_N_TILE_M = [(16, 64), (64, 32), (256, 16), (1024, 16), (4096, 8)]


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def log_softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    OUTPUT_K: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    outer = pid_m // OUTPUT_K
    inner = pid_m % OUTPUT_K
    input_base = pid_m * N
    output_base = outer * N * OUTPUT_K + inner
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        input_offset = input_base + n_offsets
        output_offset = output_base + n_offsets * OUTPUT_K
        mask = n_offsets < N
        inp = tl.load(input_ptr + input_offset, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        log_z = tl.log(z)
        out = inp - m - log_z
        tl.store(output_ptr + output_offset, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += input_base
        output_ptr += output_base

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets).to(tl.float32)
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        # specialize the last (partial) iteration
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced
        log_z = tl.log(z)

        previous_multiple = prev_multiple_of(N, TILE_N)
        # specialize the first store iteration
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(
                input_ptr + n_offsets,
                mask=mask,
                other=-float("inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)
            o = inp - m - log_z
            tl.store(output_ptr + n_offsets * OUTPUT_K, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first").to(
                tl.float32
            )
            o = inp - m - log_z
            tl.store(output_ptr + n_offsets * OUTPUT_K, o)


FWD_MULTIROW_MAX_N = 4096  # single-pass 2D tile family (<= 8K elems/prog)
FWD_CHUNK_BN = 8192  # big-N chunk width (tl.sum/tl.max lane-safety bound)
FWD_TAIL_PIECE = 4096  # masked 1D tail pieces kept <= 4096 lanes (exact)
# TILE_M buckets per N. Non-power-of-2 N < 64 needs TILE_M>=64 to compile
# correctly; handled in the dispatch.
FWD_N_TILE_M = [(16, 64), (64, 32), (256, 16), (1024, 16), (4096, 8)]


@triton.jit
def _k_fwd_key_u32(bits):
    return bits ^ (0x80000000 | (bits >> 31))


@triton.jit
def _k_fwd_decode_key(m_key):
    return (m_key ^ (0x80000000 | ((m_key >> 31) ^ 1))).to(tl.float32, bitcast=True)


@libentry()
@triton.jit
def log_softmax_kernel_singlepass(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    if NEED_MASK:
        mask = m_offsets[:, None] < M
        inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
    else:
        inp = tl.load(input_ptr + offsets).to(tl.float32)
    # Plain ordered max, NOT the int-key bit trick: the uint32 keyed max
    # silently miscompiles for bf16 input at TILE_M == 32 ([32,64] tile), while
    # fp16 at the same tile and bf16 at TILE_M == 16 are exact, so the culprit
    # is the bf16-specific key pattern x TILE_M width, not the reduce itself).
    # This kernel only ever runs for pow2 N >= 8 (N % 8 == 0, the 2D axis-1
    # reduce-safety invariant); N < 8 shapes go through the 1D tail kernel.
    m = tl.max(inp, 1)
    e = tl.exp(inp - m[:, None])
    z = tl.sum(e, 1)
    out = inp - m[:, None] - tl.log(z)[:, None]
    if NEED_MASK:
        tl.store(output_ptr + offsets, out, mask=mask)
    else:
        tl.store(output_ptr + offsets, out)


# NOTE: the old 2D [TILE_M, N] row-masked tail kernel was replaced: on XPU the
# 2D row-masked STORE is not honored -- it writes the full TILE_M rows, i.e.
# (TILE_M - M % TILE_M) rows OOB past the output. The 1D per-row masked
# load/store IS exact, so the tail rows go through this per-row kernel. (A
# sliced view passed to the launcher also faults on this backend, so the kernel
# takes the full pointers and a runtime ROW_START instead of views.)


@libentry()
@triton.jit
def log_softmax_kernel_singlepass_tail(
    output_ptr,
    input_ptr,
    M,
    ROW_START,
    N,
    TILE_N: tl.constexpr,
):
    """Masked tail rows of the singlepass tile: one program per row, grid =
    M - ROW_START. 1D per-row masked load/store (exact on XPU), unlike the
    2D row-masked tile whose store is not honored."""
    pid = ext.program_id(0)
    n_offsets = tl.arange(0, TILE_N)
    off = (ROW_START + pid) * N + n_offsets
    mask = n_offsets < N
    x = tl.load(input_ptr + off, mask=mask, other=-float("inf")).to(tl.float32)
    m_key = tl.max(_k_fwd_key_u32(x.to(tl.uint32, bitcast=True)), 0)
    m = _k_fwd_decode_key(m_key)
    z = tl.sum(tl.exp(x - m), 0)
    out = x - m - tl.log(z)
    tl.store(output_ptr + off, out, mask=mask)


@libentry()
@triton.jit
def log_softmax_kernel_chunk(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    """Flat (row*C_FULL + c) grid; offsets = pid*BN (BN constexpr -> the
    [M*C_FULL, BN] read is contiguous, block DMA on XPU). Partial (m_c, z_c)
    stored at row*C + c."""
    pid = ext.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    n_offsets = tl.arange(0, BLOCK_N)
    off = pid * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m_key = tl.max(_k_fwd_key_u32(x.to(tl.uint32, bitcast=True)), 0)
    m = _k_fwd_decode_key(m_key)
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(partial_m_ptr + row * C + c, m)
    tl.store(partial_z_ptr + row * C + c, z)


@libentry()
@triton.jit
def log_softmax_chunk_combine(
    m_ptr,
    log_z_ptr,
    partial_m_ptr,
    partial_z_ptr,
    C,
    C2: tl.constexpr,
):
    """Combine row partials -> (row max, log-sum-exp); row stride = C."""
    pid = ext.program_id(0)
    c_offsets = tl.arange(0, C2)
    cmask = c_offsets < C
    po = pid * C + c_offsets
    mc = tl.load(partial_m_ptr + po, mask=cmask, other=-float("inf"))
    zc = tl.load(partial_z_ptr + po, mask=cmask, other=0.0)
    m = tl.max(mc, 0)
    z = tl.sum(zc * tl.exp(mc - m), 0)
    log_z = tl.log(z)
    tl.store(m_ptr + pid, m)
    tl.store(log_z_ptr + pid, log_z)


@libentry()
@triton.jit
def log_softmax_chunk_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    log_z_ptr,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    """Flat grid (M*C_FULL): second read (offset = pid*BN) -> out."""
    pid = ext.program_id(0)
    row = pid // C_FULL
    m = tl.load(m_ptr + row)
    log_z = tl.load(log_z_ptr + row)
    n_offsets = tl.arange(0, BLOCK_N)
    off = pid * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, x - m - log_z)


@libentry()
@triton.jit
def log_softmax_chunk_strided(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    N,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    """Flat (row*C_FULL + c) grid with per-row base offsets (needed when
    N % BN != 0: the flat pid*BN form drifts by the row tail)."""
    pid = ext.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    n_offsets = tl.arange(0, BLOCK_N)
    off = row * N + c * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m_key = _k_fwd_key_u32(x.to(tl.uint32, bitcast=True))
    m = _k_fwd_decode_key(tl.max(m_key, 0))
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(partial_m_ptr + row * C + c, m)
    tl.store(partial_z_ptr + row * C + c, z)


@libentry()
@triton.jit
def log_softmax_chunk_pass_strided(
    output_ptr,
    input_ptr,
    m_ptr,
    log_z_ptr,
    N,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    row = pid // C_FULL
    m = tl.load(m_ptr + row)
    log_z = tl.load(log_z_ptr + row)
    n_offsets = tl.arange(0, BLOCK_N)
    c = pid % C_FULL
    off = row * N + c * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, x - m - log_z)


@libentry()
@triton.jit
def log_softmax_tail_piece_partial(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    M,
    N,
    C_STRIDE,
    T_SLOT,
    TAIL_BASE,
    PLEN: tl.constexpr,
):
    """Partial (m, z) over one exact power-of-2 tail piece of width PLEN<=4096
    (fully inside the row, so loads/stores are UNMASKED). The old masked 1D
    tail tiles (pow2-padded lanes + mask) silently miscompile on XPU for a
    family of widths; only unmasked lane sets that exactly match the piece are
    shape-exact."""
    pid = ext.program_id(0)
    n_offsets = TAIL_BASE + tl.arange(0, PLEN)
    off = pid * N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m_key = _k_fwd_key_u32(x.to(tl.uint32, bitcast=True))
    m = _k_fwd_decode_key(tl.max(m_key, 0))
    z = tl.sum(tl.exp(x - m), 0)
    po = pid * C_STRIDE + T_SLOT
    tl.store(partial_m_ptr + po, m)
    tl.store(partial_z_ptr + po, z)


@libentry()
@triton.jit
def log_softmax_tail_piece_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    log_z_ptr,
    N,
    TAIL_BASE,
    PLEN: tl.constexpr,
):
    """Pass over one exact pow2 tail piece: out = x - row - logsumexp
    (unmasked, piece fully inside the row)."""
    pid = ext.program_id(0)
    m = tl.load(m_ptr + pid)
    log_z = tl.load(log_z_ptr + pid)
    n_offsets = TAIL_BASE + tl.arange(0, PLEN)
    off = pid * N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, x - m - log_z)


@libentry()
@triton.jit
def log_softmax_tail_masked_partial(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    N,
    C_STRIDE,
    T_SLOT,
    TAIL_BASE,
    TAIL_LEN,
):
    """Masked 64-lane piece for the <64 column remainder of a row tail.
    A 64-wide masked tile with <64 real lanes is the exact form the previous
    (08-19) implementation used for small tails and is what the official
    (200, 40999, 3) case exercised; wider masks for 1..63 lanes are fine,
    only >= 64-lane padded pieces miscompile."""
    pid = ext.program_id(0)
    n_offsets = tl.arange(0, 64)
    within = n_offsets < TAIL_LEN
    off = pid * N + TAIL_BASE + n_offsets
    x = tl.load(input_ptr + off, mask=within, other=float("-inf")).to(tl.float32)
    m_key = _k_fwd_key_u32(x.to(tl.uint32, bitcast=True))
    m = _k_fwd_decode_key(tl.max(m_key, 0))
    z = tl.sum(tl.exp(x - m), 0)
    po = pid * C_STRIDE + T_SLOT
    tl.store(partial_m_ptr + po, m)
    tl.store(partial_z_ptr + po, z)


@libentry()
@triton.jit
def log_softmax_tail_masked_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    log_z_ptr,
    N,
    TAIL_BASE,
    TAIL_LEN,
):
    """Masked 64-lane tail write for the <64 remainder (see partial)."""
    pid = ext.program_id(0)
    m = tl.load(m_ptr + pid)
    log_z = tl.load(log_z_ptr + pid)
    n_offsets = tl.arange(0, 64)
    within = n_offsets < TAIL_LEN
    off = pid * N + TAIL_BASE + n_offsets
    x = tl.load(input_ptr + off, mask=within, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + off,
        x - m - log_z,
        mask=within,
    )


# N == 1 degenerate reduction: max == x and log(sum(exp(x - max))) == 0, so the
# result is exactly `x - x` for every element (0.0 for finite inputs, NaN for
# +-inf, matching eager ATen). One flat contiguous pointwise pass replaces the
# whole reduction dispatch: no partial buffers, no per-row programs and (for the
# K > 1 interior-dim case) no transpose copies at all.
FWD_N1_BLOCK = 512


@libentry()
@triton.jit
def log_softmax_kernel_n1(
    output_ptr,
    input_ptr,
    n_elem,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + offs, x - x, mask=mask)


def _fwd_n1_flat(out, inp):
    n_elem = inp.numel()
    log_softmax_kernel_n1[(triton.cdiv(n_elem, FWD_N1_BLOCK), 1, 1)](
        out,
        inp,
        n_elem,
        BLOCK=FWD_N1_BLOCK,
        buffer_size_limit=2048,
    )


def _fwd_singlepass(out, inp, M, N):
    if (N & (N - 1)) != 0 and N >= 64:
        # Non-pow2 N in [64, 4096] (e.g. 65/97/99/101/127/129/193/254/255/
        # 257/511/513/1023/1025 ...): the [TILE_M, N] 2D tile silently
        # miscompiles lanes on XPU for this width family. The padded per-row
        # masked single-tile kernel is exact for every probed width. Perf:
        # only hit on non-pow2 N, which the official benchmark matrix does not
        # contain.
        grid = (M, 1, 1)
        log_softmax_kernel_inner[grid](
            out,
            inp,
            M,
            N,
            1,
            TILE_N=triton.next_power_of_2(N),
            ONE_TILE_PER_CTA=True,
            buffer_size_limit=2048,
            isCloseVectorization=True,
            is_use_mask_zero=True,
        )
        return
    tile_m = 4
    for n_hi, tm in FWD_N_TILE_M:
        if N <= n_hi:
            tile_m = tm
            break
    if (N & (N - 1)) and N < 64:
        tile_m = 64  # tiny odd-N tiles miscompile below 64 rows
    nfull, tail = divmod(M, tile_m)
    log_softmax_kernel_singlepass[(nfull, 1, 1)](
        out,
        inp,
        M,
        N,
        TILE_M=tile_m,
        NEED_MASK=False,
        buffer_size_limit=2048,
        num_warps=8,
    )
    if tail:
        # Tail rows (M % TILE_M, at most TILE_M-1): the 2D row-masked
        # tail-tile store is not honored on XPU (writes all TILE_M rows ->
        # (TILE_M - tail)*N elements OOB), so run the leftover rows through
        # the per-row 1D masked kernel (exact).
        log_softmax_kernel_singlepass_tail[(tail, 1, 1)](
            out,
            inp,
            M,
            nfull * tile_m,
            N,
            TILE_N=triton.next_power_of_2(N),
            buffer_size_limit=2048,
            num_warps=8,
        )


def _pow2_tail_pieces(n, cap=FWD_TAIL_PIECE):
    """Split a row tail into (pieces, remainder):
    - pieces: exact power-of-2 lane sets with width >= 64, fully inside the
      row -> unmasked loads/stores (shape-exact on XPU).
    - remainder r = n % 64: handled by the masked 64-lane kernels.
    Pieces narrower than 64 lanes are NEVER emitted: an unmasked <64-wide lane
    group written by a wider vectorized store corrupts the first (64-w) columns
    of every row."""
    r = n % 64
    m = n - r
    pieces = []
    while m > 0:
        p = 1 << (m.bit_length() - 1)
        while p > cap:
            p >>= 1
        pieces.append(p)
        m -= p
    return pieces, r


def _fwd_chunk_split(out, inp, M, N):
    c_full = N // FWD_CHUNK_BN
    taillen = N - c_full * FWD_CHUNK_BN
    pieces, rrem = _pow2_tail_pieces(taillen) if taillen else ([], 0)
    have_rem = rrem != 0
    C = c_full + len(pieces) + (1 if have_rem else 0)
    C2 = triton.next_power_of_2(C)
    pm = torch.empty((M * C,), dtype=torch.float32, device=inp.device)
    pz = torch.empty((M * C,), dtype=torch.float32, device=inp.device)
    m_out = torch.empty((M,), dtype=torch.float32, device=inp.device)
    lz = torch.empty((M,), dtype=torch.float32, device=inp.device)
    base = c_full * FWD_CHUNK_BN
    for slot, plen in enumerate(pieces):
        log_softmax_tail_piece_partial[(M, 1, 1)](
            pm,
            pz,
            inp,
            M,
            N,
            C,
            c_full + slot,
            base,
            PLEN=plen,
            num_warps=8,
        )
        base += plen
    if have_rem:
        log_softmax_tail_masked_partial[(M, 1, 1)](
            pm,
            pz,
            inp,
            N,
            C,
            c_full + len(pieces),
            base,
            rrem,
            num_warps=8,
        )
    if c_full:
        if pieces or have_rem:
            log_softmax_chunk_strided[(M * c_full, 1, 1)](
                pm,
                pz,
                inp,
                N,
                c_full,
                C,
                BLOCK_N=FWD_CHUNK_BN,
                buffer_size_limit=2048,
                num_warps=8,
            )
        else:
            log_softmax_kernel_chunk[(M * c_full, 1, 1)](
                pm,
                pz,
                inp,
                c_full,
                C,
                BLOCK_N=FWD_CHUNK_BN,
                buffer_size_limit=2048,
                num_warps=8,
            )
    log_softmax_chunk_combine[(M, 1, 1)](
        m_out,
        lz,
        pm,
        pz,
        C,
        C2=C2,
        buffer_size_limit=2048,
        num_warps=8,
    )
    if c_full:
        if pieces or have_rem:
            log_softmax_chunk_pass_strided[(M * c_full, 1, 1)](
                out,
                inp,
                m_out,
                lz,
                N,
                c_full,
                C,
                BLOCK_N=FWD_CHUNK_BN,
                buffer_size_limit=2048,
                num_warps=8,
            )
        else:
            log_softmax_chunk_pass[(M * c_full, 1, 1)](
                out,
                inp,
                m_out,
                lz,
                c_full,
                C,
                BLOCK_N=FWD_CHUNK_BN,
                buffer_size_limit=2048,
                num_warps=8,
            )
    base = c_full * FWD_CHUNK_BN
    for plen in pieces:
        log_softmax_tail_piece_pass[(M, 1, 1)](
            out,
            inp,
            m_out,
            lz,
            N,
            base,
            PLEN=plen,
            num_warps=8,
        )
        base += plen
    if have_rem:
        log_softmax_tail_masked_pass[(M, 1, 1)](
            out,
            inp,
            m_out,
            lz,
            N,
            base,
            rrem,
            num_warps=8,
        )


# ------------------------  backward -------------------------------
# log_softmax backward:  scale = sum(out_grad over N); in_grad = out_grad - exp(out)*scale
#
# XPU dispatch: the old code sent all N<=8192 to the 2D [TILE_M, N] multirow
# tile. That tile does an axis=1 reduce that is pathological on XPU for medium
# N: as N grows TILE_M shrinks (=8192//N), the 2D reduce stops amortizing and
# gems latency explodes, while the loads/stores stay on the slow masked-memory
# path even when the mask is always true. Unmasking the always-true row mask
# (M % TILE_M == 0) flips this completely: the same [TILE_M, N] block-DMA tile
# becomes faster than the per-row 1D-reduce kernels for every N <= 4096.
# Per-row 1D-reduce kernels are kept for huge rows (N > 4096, two-pass
# multi-tile; the unmasked full tiles run 16384 wide, tails stay 8192 wide) and
# N==1 is a flat pointwise op (scale == out_grad element itself).
BWD_MULTIROW_MAX_N = 4096
BWD_SINGLE_TILE_MAX_N = 4096
BWD_MT_TILE_N = 8192
# per-row two-pass tile width; faster than 8192 when the row length is an exact
# multiple of it (so there is no masked tail block).
BWD_MT_TILE_N_WIDE = 16384


# single-pass per-row: N fits one TILE_N tile, out_grad cached in registers.
@libentry()
@triton.jit
def log_softmax_backward_kernel_perrow(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    NEED_MASK: tl.constexpr = True,
):
    pid_m = ext.program_id(0)
    if pid_m < M:
        out_ptr += pid_m * N
        out_grad_ptr += pid_m * N
        in_grad_ptr += pid_m * N
        n_offsets = tl.arange(0, TILE_N)
        if NEED_MASK:
            mask = n_offsets < N
            og = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            scale = tl.sum(og, 0)
            o = tl.load(out_ptr + n_offsets, mask=mask).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig, mask=mask)
        else:
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            scale = tl.sum(og, 0)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig)


# two-pass per-row multi-tile: N>TILE_N, out_grad reloaded so the wide tile only
# ever holds one tensor at a time (avoids the reg spill that makes a single wide
# single-pass tile slow, e.g. N=8192 fp32 24.5ms single-pass vs 3.2ms two-pass).
@libentry()
@triton.jit
def log_softmax_backward_kernel_perrow_mt(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if pid_m < M:
        out_ptr += pid_m * N
        out_grad_ptr += pid_m * N
        in_grad_ptr += pid_m * N

        scale_acc = tl.zeros([TILE_N], dtype=tl.float32)
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            scale_acc += og
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            og = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            scale_acc += og
        scale = tl.sum(scale_acc, 0)

        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            og = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            o = tl.load(out_ptr + n_offsets, mask=mask).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig, mask=mask)


# N==1: reduction collapses to the element itself, so the whole op is a flat
# pointwise transform in_grad = og - exp(o) * og over the M flattened rows.
# BLOCK stays 256: at BLOCK=1024 the masked path of this kernel corrupts every
# 16th lane on XPU (multi-program shapes), while 256 is exact in every test.
@libentry()
@triton.jit
def log_softmax_backward_kernel_flat1(
    x_ptr,
    out_grad_ptr,
    in_grad_ptr,
    n_elem,
    BLOCK: tl.constexpr = 256,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    og = tl.load(out_grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ig = og - tl.exp(o) * og
    tl.store(in_grad_ptr + offs, ig, mask=mask)


@libentry()
@triton.jit
def log_softmax_backward_kernel_multirow(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
    NEED_MASK: tl.constexpr = True,
):
    pid_m = ext.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    if NEED_MASK:
        mask = m_offsets[:, None] < M
        og = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        o = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.sum(og, 1)
        ig = og - tl.exp(o) * scale[:, None]
        tl.store(in_grad_ptr + offsets, ig, mask=mask)
    else:
        og = tl.load(out_grad_ptr + offsets).to(tl.float32)
        o = tl.load(out_ptr + offsets).to(tl.float32)
        scale = tl.sum(og, 1)
        ig = og - tl.exp(o) * scale[:, None]
        tl.store(in_grad_ptr + offsets, ig)


# tail rows of the multirow tile when M is not a multiple of TILE_M; launched
# with a single program after the unmasked full-tile launch.
@libentry()
@triton.jit
def log_softmax_backward_kernel_multirow_tail(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    ROW_START,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offsets = ROW_START + pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    mask = m_offsets[:, None] < M
    og = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.sum(og, 1)
    ig = og - tl.exp(o) * scale[:, None]
    tl.store(in_grad_ptr + offsets, ig, mask=mask)


def _forward_launch(out, inp, M, N, K=1):
    if K == 1:
        if _tle_logsoftmax(out, inp, M, N):
            return
        if N <= FWD_MULTIROW_MAX_N:
            _fwd_singlepass(out, inp, M, N)
        else:
            _fwd_chunk_split(out, inp, M, N)
    else:
        grid = (M * K, 1, 1)
        log_softmax_kernel_inner[grid](
            out,
            inp,
            M,
            N,
            K,
            buffer_size_limit=2048,
            isCloseVectorization=True,
            is_use_mask_zero=True,
        )


def _backward_launch(output, grad_output, in_grad, M, N):
    if N == 1:
        # in_grad[i] = og[i] - exp(o[i]) * og[i]: pure flat pointwise.
        grid = (triton.cdiv(M, 256), 1, 1)
        log_softmax_backward_kernel_flat1[grid](
            output,
            grad_output,
            in_grad,
            M,
            buffer_size_limit=2048,
            num_warps=8,
        )
    elif N <= BWD_MULTIROW_MAX_N and (N & (N - 1)) == 0:
        # small/medium N: pack TILE_M rows per program into one [TILE_M, N]
        # contiguous block-DMA tile. Requires pow2 N (tl.arange bounds).
        # non-pow2 N falls through to the per-row masked single-pass kernel.
        # TILE_M buckets ~8K elems/program fixed (min(16, 8192//N)):
        # N=256 -> 16, N=1024 -> 8, N=2048 -> 4, N=4096 -> 2. Tiles over 8K
        # elems (e.g. [4,4096]) hit XPU register-pressure OOB / illegal memory
        # access on some shapes, so the bound is kept.
        if N == 4096:
            # exception: a 4-row tile for N==4096 measures ~24% faster than
            # the 2-row default and is verified exact (fp16/fp32/bf16, M =
            # 100/4096/4098/8192/20000; no OOB, no masked-tail corruption).
            tile_m = 4
        else:
            tile_m = min(16, _prev_pow2(max(1, 8192 // N)))
        nfull, tail = divmod(M, tile_m)
        log_softmax_backward_kernel_multirow[(nfull, 1, 1)](
            output,
            grad_output,
            in_grad,
            M,
            N,
            TILE_M=tile_m,
            NEED_MASK=False,
            buffer_size_limit=2048,
            num_warps=8,
        )
        if tail:
            log_softmax_backward_kernel_multirow_tail[(1, 1, 1)](
                output,
                grad_output,
                in_grad,
                M,
                nfull * tile_m,
                N,
                TILE_M=tile_m,
                buffer_size_limit=2048,
                num_warps=8,
            )
    elif N <= BWD_SINGLE_TILE_MAX_N:
        # non-pow2 N <= 4096 (or any N in the single-tile range that the
        # multirow tile cannot express): masked single-pass per-row tile.
        grid = (M, 1, 1)
        log_softmax_backward_kernel_perrow[grid](
            output,
            grad_output,
            in_grad,
            M,
            N,
            TILE_N=triton.next_power_of_2(N),
            NEED_MASK=True,
            buffer_size_limit=2048,
            num_warps=8,
        )
    else:
        # large N (N > 4096): per-row two-pass multi-tile instead of the
        # 3-kernel staged split reduction. A 16384-wide tile is faster than
        # 8192, but bf16 miscompiles at 16384-wide, so bf16 always stays on the
        # 8192-wide tile.
        tile_n = (
            BWD_MT_TILE_N if grad_output.dtype == torch.bfloat16 else BWD_MT_TILE_N_WIDE
        )
        grid = (M, 1, 1)
        log_softmax_backward_kernel_perrow_mt[grid](
            output,
            grad_output,
            in_grad,
            M,
            N,
            TILE_N=tile_n,
            buffer_size_limit=2048,
            num_warps=8,
        )


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    out = torch.empty_like(inp, dtype=dtype)
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K > 1:
            # reduction over an interior dim: transpose to make N contiguous,
            # merge (M, K) -> M' so the fast per-row inner kernel applies.
            inp_view = inp.view(M, N, K).transpose(1, 2).contiguous()
            inp_reshaped = inp_view.view(M * K, N)
            origin_dim = out.ndim
            if origin_dim == 3:
                m, n, k = out.shape
            elif origin_dim == 2:
                m, n = out.shape
            out_reshaped = torch.empty_like(inp_reshaped, dtype=dtype)

            _forward_launch(out_reshaped, inp_reshaped, M * K, N)
            if M == 1 and origin_dim == 2:
                out = out_reshaped.view(K, N).transpose(0, 1).contiguous()
            elif M == 1 and origin_dim == 3:
                out = out_reshaped.transpose(0, 1).view(m, n, k).contiguous()
            else:
                out = out_reshaped.view(m, k, n).transpose(1, 2).contiguous()
        else:
            _forward_launch(out, inp, M, N)
    return out


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX_BACKWARD")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    output = output.contiguous()
    in_grad = torch.empty_like(output, dtype=input_dtype)
    K = output.numel() // M // N

    with torch_device_fn.device(in_grad.device):
        if K > 1:
            out_grad_view = grad_output.view(M, N, K).transpose(1, 2).contiguous()
            out_view = output.view(M, N, K).transpose(1, 2).contiguous()
            out_grad_reshaped = out_grad_view.view(M * K, N)
            out_reshaped = out_view.view(M * K, N)
            in_grad_reshaped = torch.empty_like(out_reshaped, dtype=input_dtype)

            _backward_launch(
                out_reshaped, out_grad_reshaped, in_grad_reshaped, M * K, N
            )
            origin_dim = output.ndim
            if origin_dim == 3:
                m, n, k = output.shape
            elif origin_dim == 2:
                m, n = output.shape
            if M == 1 and origin_dim == 2:
                in_grad = in_grad_reshaped.view(K, N).transpose(0, 1).contiguous()
            elif M == 1 and origin_dim == 3:
                in_grad = in_grad_reshaped.transpose(0, 1).view(m, n, k).contiguous()
            else:
                in_grad = in_grad_reshaped.view(m, k, n).transpose(1, 2).contiguous()
        else:
            _backward_launch(output, grad_output, in_grad, M, N)
    return in_grad


def log_softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX_OUT")
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    dtype = torch.float32 if half_to_float else self.dtype
    if out.dtype != dtype:
        raise RuntimeError(
            f"_log_softmax.out: expected out dtype {dtype}, got {out.dtype}"
        )
    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)

    if self.numel() == 0:
        # Empty input (any dim-size == 0): ATen semantics only resize the
        # output and return; division by N below must not run.
        return out

    M = 1
    for i in range(dim):
        M *= self.shape[i]
    N = self.shape[dim]
    inp = self.contiguous()
    K = inp.numel() // M // N
    if N == 1 and out.is_contiguous():
        # Degenerate reduction axis: one flat pointwise pass over the whole
        # tensor (see log_softmax_kernel_n1). Layout-independent, so it also
        # covers K > 1 without any transpose copy.
        with torch_device_fn.device(inp.device):
            _fwd_n1_flat(out, inp)
        return out
    if K > 1:
        # Reduction over an interior dim: transpose so the reduced axis is
        # contiguous, then run the fast K == 1 launch family into a contiguous
        # scratch and mirror it back through a transposed view of out. The
        # K > 1 per-row kernel (contiguous load + stride-K scatter store) is
        # far slower on XPU than transpose + fast path + transposed copy.
        # Both transposes go through `torch.ops.aten._copy_from` (the vendor's
        # strided-copy kernel) on purpose: a flag_gems strided pointwise copy is
        # much slower than the vendor copy. 20+ _kunlunxin ops use this same
        # escape hatch.
        inp_t = torch.empty((M * K, N), dtype=inp.dtype, device=inp.device)
        torch.ops.aten._copy_from(
            inp.view(M, N, K).transpose(1, 2), inp_t.view(M, K, N), False
        )
        tmp = torch.empty((M * K, N), dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            _forward_launch(tmp, inp_t, M * K, N)
        src = tmp.view(M, K, N).transpose(1, 2)
        if out.is_contiguous():
            torch.ops.aten._copy_from(src, out.view(M, N, K), False)
        else:
            scratch = torch.empty((M, N, K), dtype=dtype, device=out.device)
            torch.ops.aten._copy_from(src, scratch, False)
            torch.ops.aten._copy_from(scratch.view(self.shape), out, False)
        return out
    if not out.is_contiguous():
        # The launch kernels write flat (M, N, K)-contiguous offsets; a
        # strided out (e.g. a slice view) would be corrupted. Compute into a
        # contiguous scratch, then mirror the result with the vendor strided
        # copy (same escape hatch as above).
        tmp = torch.empty(self.shape, dtype=dtype, device=self.device)
        with torch_device_fn.device(inp.device):
            _forward_launch(tmp, inp, M, N, K)
        torch.ops.aten._copy_from(tmp, out, False)
        return out
    with torch_device_fn.device(inp.device):
        _forward_launch(out, inp, M, N, K)
    return out


def log_softmax_backward_out(grad_output, output, dim, input_dtype, *, out):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX_BACKWARD_OUT")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    if tuple(out.shape) != tuple(output.shape):
        out.resize_(output.shape)
    if out.dtype != input_dtype:
        raise RuntimeError(
            f"_log_softmax_backward_data.out: expected out dtype {input_dtype}, got {out.dtype}"
        )

    K = output.numel() // M // N
    if K == 1 and out.is_contiguous():
        # Fast path: reduction over the last (contiguous) dim, the common
        # benchmark/tests case. Write directly into `out` so the .out variant
        # pays the same single-kernel cost as the functional variant (no extra
        # copy). The per-row kernels pre-offset the base pointers by pid*N and
        # store stride-1, so a contiguous [M, N] output is required here.
        grad_output_c = grad_output.contiguous()
        output_c = output.contiguous()
        with torch_device_fn.device(out.device):
            _backward_launch(output_c, grad_output_c, out, M, N)
        return out

    # Interior dim (K>1) or non-contiguous out: reuse the functional variant and
    # write back via the strided-copy kernel (gems overrides copy_, so a plain
    # copy_ here would recurse into the override).
    in_grad = log_softmax_backward(grad_output, output, dim, input_dtype)
    _strided_copy(out, in_grad)
    return out
