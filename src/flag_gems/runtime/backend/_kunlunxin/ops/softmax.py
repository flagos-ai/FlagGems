import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.tle_copy import tle_copy

try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# tle.gpu row-reduce path (KL3): softmax over the contiguous last axis.
#
# The pointer kernels below reduce the row axis with a serial-chain tl.max /
# tl.sum on XPU, which is the dominant cost on the long-row shapes. tle.gpu
# moves the tile GM -> LM with the cluster DMA and keeps the reduce core-local,
# streaming the row in YBLOCK columns, so the running max / sum-of-exp stay
# [XBLOCK] vectors instead of a full-row tile. softmax needs the same two
# reductions (max, then sum of exp) and differs only in the normalise pass:
# out = exp(x - m) / z. The partial-column fill is -inf (the max identity;
# exp(-inf - m) == 0 keeps the sum exact).
#
# Only fp16/fp32/bf16, contiguous [M, N] with N >= _TLE_MIN_N_FUSED (and
# M >= _TLE_MIN_M in the medium-N range, M >= _TLE_CLUSTERS everywhere) take
# this path; f64, N == 1, strided views and tiny/single-row shapes keep the
# pointer kernels.
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
_TLE_MIN_N = 8192
_TLE_MIN_M = 1024
# FG_SOFTMAX_MIN_N_FUSED overrides the medium-N gate for A/B routing. Default
# 256 mirrors log_softmax: below that the tle row tiling cannot beat the
# pointer single-pass.
_TLE_MIN_N_FUSED = int(os.environ.get("FG_SOFTMAX_MIN_N_FUSED", "256"))
# Medium rows (256 <= N < 8192) only route to tle up to this width. Above it the
# row is memory-bound and the pointer multirow single-pass (1 read + 1 exp)
# beats the tle two-read fused kernel (2 reads + 2 exps).
_TLE_MED_ROWS_MAX_N = 256

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

    Same trade as the sum/log_softmax row-reduce: XBLOCK is rows per program,
    YBLOCK is the LM budget divided by it. Aim for one program per cluster, then
    let the row length pull XBLOCK down when it is long enough to need a long
    YBLOCK, or up when the reduce is small enough that launch and result write
    are all there is. XBLOCK may exceed M -- both copies clamp to the descriptor
    extents.
    """
    geom_key = (M, N, itemsize)
    geom = _TLE_ROW_GEOM.get(geom_key, _TLE_GEOM_MISS)
    if geom is not _TLE_GEOM_MISS:
        return geom
    xblock = min(_TLE_XBLOCK, max(128, _npo2(-(-M // _TLE_CLUSTERS))))
    row_bytes = N * itemsize
    if row_bytes >= _TLE_WIDE_ROW_BYTES:
        xblock = max(_TLE_CORE_NUM // 2, xblock // 4)
    elif row_bytes <= _TLE_NARROW_ROW_BYTES:
        xblock = _TLE_XBLOCK if M > _TLE_CORE_NUM else 128
    if _TLE_MIN_N_FUSED <= N < _TLE_MIN_N:
        # Medium row: empirical (xblock, yblock) per (itemsize, N) bucket,
        # carried over from the log_softmax tle path (identical reduce phase):
        # widest YBLOCK the 4KB/core LM budget allows, then XBLOCK for enough
        # programs. itemsize 4 (fp32) caps yb at 1024 because [64,2048] fp32
        # exceeds the LM budget.
        if itemsize == 2:
            if N <= 512:
                xblock, yblock = 128, 512
            elif N <= 1024:
                xblock, yblock = 32, 1024
            elif N <= 2048:
                xblock, yblock = 32, 2048
            else:  # 4096 .. 8191
                # (32, 2048), not log_softmax's (64, 2048): softmax's normalise
                # pass adds an extra tl.exp, and (64, 2048) bf16 (= 256KB LM
                # tile) exceeds the local-memory stack budget on this arch.
                xblock, yblock = 32, 2048
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
    do_not_specialize_on_alignment=["a_desc", "m_desc", "z_desc"],
)
def _tle_softmax_reduce_kernel(
    a_desc,
    m_desc,
    z_desc,
    N,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    """Online-softmax row reduce: writes (row max, sum-of-exp) per row.

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
    z_lmem = tle.gpu.alloc([XBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.lmem)

    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
    m_ptrs = tle.gpu.local_ptr(m_lmem, (tl.arange(0, XBLOCK),))
    z_ptrs = tle.gpu.local_ptr(z_lmem, (tl.arange(0, XBLOCK),))

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
    tl.store(z_ptrs, z)
    tle.gpu.copy(m_lmem, m_desc, [XBLOCK], [row_off])
    tle.gpu.copy(z_lmem, z_desc, [XBLOCK], [row_off])


@triton.jit
def _softmax_pass_kernel(
    out_ptr,
    x_ptr,
    m_ptr,
    z_ptr,
    N,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Elementwise out = exp(x - m[row]) / z[row]; reads x a second time.

    Plain pointer pass, one row per program: the reduce kernel already produced
    (row max, sum-of-exp) into m/z, and this pass is memory-bound. The
    full-chunk path is unmasked so the load/store lowers to block-DMA; a masked
    2D/1D tile on this backend falls back to per-lane gather and is orders of
    magnitude slower, so it is kept only for the short last block.
    """
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off = pid_m * N + n_offsets
    m = tl.load(m_ptr + pid_m)
    z = tl.load(z_ptr + pid_m)
    if NEED_MASK:
        mask = n_offsets < N
        x = tl.load(x_ptr + off, mask=mask, other=float("-inf")).to(tl.float32)
        tl.store(out_ptr + off, tl.exp(x - m) / z, mask=mask)
    else:
        x = tl.load(x_ptr + off).to(tl.float32)
        tl.store(out_ptr + off, tl.exp(x - m) / z)


@triton.jit(
    do_not_specialize=["N"],
    do_not_specialize_on_alignment=["a_desc", "out_desc"],
)
def _tle_softmax_fused_kernel(
    a_desc,
    out_desc,
    N,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    """Fused reduce + normalize for a [M, N] softmax (N >= 256).

    One kernel: pass 1 streams the row in YBLOCK chunks computing online
    (max, sum-exp) into per-row [XBLOCK] accumulators held in registers; pass 2
    streams again and writes out = exp(x - m) / z. The row is read from GM
    twice (too long to hold), but there is one launch and the per-row
    statistics never round-trip through GM -- the separate reduce kernel's
    LM-strip writeback of m/z costs more than a launch (mirrors the log_softmax
    fused kernel).
    """
    pid = tl.program_id(0)
    row_off = pid * XBLOCK

    a0 = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a0_ptrs = tle.gpu.local_ptr(a0, (row_ids, col_ids))

    full_n = N - N % YBLOCK
    m = tl.full([XBLOCK], value=float("-inf"), dtype=tl.float32)
    z = tl.zeros([XBLOCK], dtype=tl.float32)
    # ---- pass 1: online-softmax reduce ----
    for c in tl.range(0, full_n, YBLOCK):
        tle.gpu.copy(a_desc, a0, [XBLOCK, YBLOCK], [row_off, c])
        x = tl.load(a0_ptrs).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 1))
        all_neg_inf = m_new == float("-inf")
        z = tl.where(
            all_neg_inf,
            z,
            z * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new[:, None]), 1),
        )
        m = m_new
    if NEED_ZERO:
        tl.store(
            a0_ptrs, tl.full([XBLOCK, YBLOCK], value=float("-inf"), dtype=IN_DTYPE)
        )
        tle.gpu.copy(a_desc, a0, [XBLOCK, YBLOCK], [row_off, full_n])
        x = tl.load(a0_ptrs).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 1))
        all_neg_inf = m_new == float("-inf")
        z = tl.where(
            all_neg_inf,
            z,
            z * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new[:, None]), 1),
        )
        m = m_new

    # ---- pass 2: normalize ----
    for c in tl.range(0, full_n, YBLOCK):
        tle.gpu.copy(a_desc, a0, [XBLOCK, YBLOCK], [row_off, c])
        x = tl.load(a0_ptrs).to(tl.float32)
        o = tl.exp(x - m[:, None]) / z[:, None]
        tl.store(a0_ptrs, o.to(OUT_DTYPE))
        tle.gpu.copy(a0, out_desc, [XBLOCK, YBLOCK], [row_off, c])
    if NEED_ZERO:
        tl.store(
            a0_ptrs, tl.full([XBLOCK, YBLOCK], value=float("-inf"), dtype=IN_DTYPE)
        )
        tle.gpu.copy(a_desc, a0, [XBLOCK, YBLOCK], [row_off, full_n])
        x = tl.load(a0_ptrs).to(tl.float32)
        o = tl.exp(x - m[:, None]) / z[:, None]
        tl.store(a0_ptrs, o.to(OUT_DTYPE))
        tle.gpu.copy(a0, out_desc, [XBLOCK, YBLOCK], [row_off, full_n])


def _tle_softmax(out, inp, M, N):
    """softmax over a contiguous [M, N] last axis; False if not expressible."""
    if not _TLE_AVAILABLE:
        return False
    if N < _TLE_MIN_N_FUSED:
        return False
    if M < _TLE_CLUSTERS:
        # Fewer rows than a cluster: the tle grid collapses to 1-2 programs and
        # loses to the parallel chunk-split path (e.g. the 1-D [268435456] flat
        # softmax goes through 32768-way chunk split, not a single program).
        return False
    if N < _TLE_MIN_N:
        # Medium rows (256 <= N < 8192). Two bounds keep the tle path off where
        # the pointer single-pass (multirow: 1 read + 1 exp) wins:
        #   * N > _TLE_MED_ROWS_MAX_N: memory-bound medium rows (N >= 512) where
        #     the fused kernel's second row read + second exp lose.
        #   * M < _TLE_MIN_M: grid collapses to 1-2 programs and its 2x row
        #     read loses to the single-pass pointer path.
        if N > _TLE_MED_ROWS_MAX_N or M < _TLE_MIN_M:
            return False
    if inp.dtype not in _TLE_TL_DTYPE or out.dtype not in _TLE_TL_DTYPE:
        return False
    if not inp.is_contiguous() or not out.is_contiguous():
        return False
    a = inp.view(M, N)
    c = out.view(M, N)
    xblock, yblock, row_blocks = _tle_row_geom(M, N, inp.element_size())
    if N % yblock != 0:
        # Non-pow2 tail: keep the two-kernel path (reduce then pass); the
        # fused kernel's partial last chunk is only exercised by pow2 rows.
        m = torch.empty((M,), dtype=torch.float32, device=inp.device)
        z = torch.empty((M,), dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            _tle_softmax_reduce_kernel[(row_blocks,)](
                TensorDescriptor.from_tensor(a, block_shape=[xblock, yblock]),
                TensorDescriptor.from_tensor(m, block_shape=[xblock]),
                TensorDescriptor.from_tensor(z, block_shape=[xblock]),
                N,
                XBLOCK=xblock,
                YBLOCK=yblock,
                IN_DTYPE=_TLE_TL_DTYPE[inp.dtype],
                NEED_ZERO=True,
            )
            block_n = min(8192, _npo2(N))
            _softmax_pass_kernel[(M, triton.cdiv(N, block_n))](
                c,
                a,
                m,
                z,
                N,
                BLOCK_N=block_n,
                NEED_MASK=(N % block_n != 0),
            )
    else:
        with torch_device_fn.device(inp.device):
            _tle_softmax_fused_kernel[(row_blocks,)](
                TensorDescriptor.from_tensor(a, block_shape=[xblock, yblock]),
                TensorDescriptor.from_tensor(c, block_shape=[xblock, yblock]),
                N,
                XBLOCK=xblock,
                YBLOCK=yblock,
                IN_DTYPE=_TLE_TL_DTYPE[inp.dtype],
                OUT_DTYPE=_TLE_TL_DTYPE[out.dtype],
                NEED_ZERO=False,
            )
    return True


@triton.jit
def next_multiple_of(a, b):
    return tl.cdiv(a, b) * b


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        input_ptr += pid_m * N
        output_ptr += pid_m * N
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
            output_ptr.dtype.element_ty
        )
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = e / z
        tl.store(output_ptr + n_offsets, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)


_SM_MR_MAX_N = 4096
_SM_MR_TILE_M = 64
_SM_MR_TILE_M_N4096 = 16


@triton.jit
def softmax_kernel_multirow(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    mo = pid * TILE_M + tl.arange(0, TILE_M)
    no = tl.arange(0, N)
    off = mo[:, None] * N + no[None, :]
    inp = tl.load(input_ptr + off).to(output_ptr.dtype.element_ty)
    m = tl.max(inp, 1)
    e = tl.exp(inp - m[:, None])
    z = tl.sum(e, 1)
    out = e / z[:, None]
    tl.store(output_ptr + off, out)


_SM_CHUNK_BN = 8192
_SM_TAIL_PIECE = 4096


def _sm_pow2_tail_pieces(n, cap=_SM_TAIL_PIECE):
    """Split a row tail into (pieces, 64-lane remainder) - see log_softmax."""
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


@libentry()
@triton.jit
def softmax_kernel_chunk(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    n_offsets = tl.arange(0, BLOCK_N)
    off = pid * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(partial_m_ptr + row * C + c, m)
    tl.store(partial_z_ptr + row * C + c, z)


_SM_COMBINE_GW = 1024


def _sm_combine_geometry(C):
    """(group width, group count, padded per-row stride) for the partial merge.

    The width is clamped to [64, 1024]: tiles of <= 32 lanes miscompile on this
    XPU, a single tile spanning all partials is untrustworthy, and 64 keeps
    every padded row a whole number of 64-element store units.  The padded
    stride stays <= max(64, 2 * C), so the partial buffers grow at most 2x.
    """
    gw = min(max(64, triton.next_power_of_2(C)), _SM_COMBINE_GW)
    ng = triton.cdiv(C, gw)
    return gw, ng, ng * gw


@libentry()
@triton.jit
def softmax_combine_pad_init(
    partial_m_ptr,
    partial_z_ptr,
    CP: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * CP + tl.arange(0, CP)
    tl.store(partial_m_ptr + off, tl.full([CP], float("-inf"), tl.float32))
    tl.store(partial_z_ptr + off, tl.zeros([CP], tl.float32))


@libentry()
@triton.jit
def softmax_chunk_combine(
    m_ptr,
    z_ptr,
    partial_m_ptr,
    partial_z_ptr,
    NG: tl.constexpr,
    GW: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, GW)
    base = pid * NG * GW
    m = float("-inf")
    for g in tl.range(NG):
        mc = tl.load(partial_m_ptr + base + g * GW + lane)
        m = tl.maximum(m, tl.max(mc, 0))
    z = 0.0
    for g in tl.range(NG):
        off = base + g * GW + lane
        mc = tl.load(partial_m_ptr + off)
        zc = tl.load(partial_z_ptr + off)
        z += tl.sum(zc * tl.exp(mc - m), 0)
    tl.store(m_ptr + pid, m)
    tl.store(z_ptr + pid, z)


@libentry()
@triton.jit
def softmax_chunk_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    z_ptr,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // C_FULL
    m = tl.load(m_ptr + row).to(tl.float32)
    z = tl.load(z_ptr + row).to(tl.float32)
    n_offsets = tl.arange(0, BLOCK_N)
    off = pid * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, tl.exp(x - m) / z)


@libentry()
@triton.jit
def softmax_kernel_chunk_strided(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    N,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    n_offsets = tl.arange(0, BLOCK_N)
    off = row * N + c * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(partial_m_ptr + row * C + c, m)
    tl.store(partial_z_ptr + row * C + c, z)


@libentry()
@triton.jit
def softmax_chunk_pass_strided(
    output_ptr,
    input_ptr,
    m_ptr,
    z_ptr,
    N,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // C_FULL
    m = tl.load(m_ptr + row).to(tl.float32)
    z = tl.load(z_ptr + row).to(tl.float32)
    n_offsets = tl.arange(0, BLOCK_N)
    c = pid % C_FULL
    off = row * N + c * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, tl.exp(x - m) / z)


@libentry()
@triton.jit
def softmax_tail_piece_partial(
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
    pid = tl.program_id(0)
    n_offsets = TAIL_BASE + tl.arange(0, PLEN)
    off = pid * N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    po = pid * C_STRIDE + T_SLOT
    tl.store(partial_m_ptr + po, m)
    tl.store(partial_z_ptr + po, z)


@libentry()
@triton.jit
def softmax_tail_piece_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    z_ptr,
    N,
    TAIL_BASE,
    PLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.load(m_ptr + pid).to(tl.float32)
    z = tl.load(z_ptr + pid).to(tl.float32)
    n_offsets = TAIL_BASE + tl.arange(0, PLEN)
    off = pid * N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, tl.exp(x - m) / z)


@libentry()
@triton.jit
def softmax_tail_masked_partial(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    N,
    C_STRIDE,
    T_SLOT,
    TAIL_BASE,
    TAIL_LEN,
):
    pid = tl.program_id(0)
    n_offsets = tl.arange(0, 64)
    within = n_offsets < TAIL_LEN
    off = pid * N + TAIL_BASE + n_offsets
    x = tl.load(input_ptr + off, mask=within, other=float("-inf")).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    po = pid * C_STRIDE + T_SLOT
    tl.store(partial_m_ptr + po, m)
    tl.store(partial_z_ptr + po, z)


@libentry()
@triton.jit
def softmax_tail_masked_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    z_ptr,
    N,
    TAIL_BASE,
    TAIL_LEN,
):
    pid = tl.program_id(0)
    m = tl.load(m_ptr + pid).to(tl.float32)
    z = tl.load(z_ptr + pid).to(tl.float32)
    n_offsets = tl.arange(0, 64)
    within = n_offsets < TAIL_LEN
    off = pid * N + TAIL_BASE + n_offsets
    x = tl.load(input_ptr + off, mask=within, other=float("-inf")).to(tl.float32)
    tl.store(output_ptr + off, tl.exp(x - m) / z, mask=within)


def _softmax_chunk_split(output, inp, M, N):
    """Chunked split forward for N > _SM_CHUNK_SPLIT_MIN (see dispatch)."""
    c_full = N // _SM_CHUNK_BN
    taillen = N - c_full * _SM_CHUNK_BN
    pieces, rrem = _sm_pow2_tail_pieces(taillen) if taillen else ([], 0)
    have_rem = rrem != 0
    C = c_full + len(pieces) + (1 if have_rem else 0)
    GW, NG, CP = _sm_combine_geometry(C)
    pm = torch.empty((M * CP,), dtype=torch.float32, device=inp.device)
    pz = torch.empty((M * CP,), dtype=torch.float32, device=inp.device)
    m_out = torch.empty((M,), dtype=torch.float32, device=inp.device)
    z_out = torch.empty((M,), dtype=torch.float32, device=inp.device)
    if CP != C:
        softmax_combine_pad_init[(M, 1, 1)](
            pm,
            pz,
            CP=CP,
            buffer_size_limit=2048,
            num_warps=8,
        )
    base = c_full * _SM_CHUNK_BN
    for slot, plen in enumerate(pieces):
        softmax_tail_piece_partial[(M, 1, 1)](
            pm,
            pz,
            inp,
            M,
            N,
            CP,
            c_full + slot,
            base,
            PLEN=plen,
            buffer_size_limit=2048,
            num_warps=8,
        )
        base += plen
    if have_rem:
        softmax_tail_masked_partial[(M, 1, 1)](
            pm,
            pz,
            inp,
            N,
            CP,
            c_full + len(pieces),
            base,
            rrem,
            buffer_size_limit=2048,
            num_warps=8,
        )
    if c_full:
        if pieces or have_rem:
            if M == 1:
                softmax_kernel_chunk[(c_full, 1, 1)](
                    pm,
                    pz,
                    inp,
                    c_full,
                    CP,
                    BLOCK_N=_SM_CHUNK_BN,
                    buffer_size_limit=2048,
                    num_warps=8,
                )
            else:
                softmax_kernel_chunk_strided[(M * c_full, 1, 1)](
                    pm,
                    pz,
                    inp,
                    N,
                    c_full,
                    CP,
                    BLOCK_N=_SM_CHUNK_BN,
                    buffer_size_limit=2048,
                    num_warps=8,
                )
        else:
            softmax_kernel_chunk[(M * c_full, 1, 1)](
                pm,
                pz,
                inp,
                c_full,
                CP,
                BLOCK_N=_SM_CHUNK_BN,
                buffer_size_limit=2048,
                num_warps=8,
            )
    softmax_chunk_combine[(M, 1, 1)](
        m_out,
        z_out,
        pm,
        pz,
        NG=NG,
        GW=GW,
        buffer_size_limit=2048,
        num_warps=8,
    )
    if c_full:
        if pieces or have_rem:
            if M == 1:
                softmax_chunk_pass[(c_full, 1, 1)](
                    output,
                    inp,
                    m_out,
                    z_out,
                    c_full,
                    C,
                    BLOCK_N=_SM_CHUNK_BN,
                    buffer_size_limit=2048,
                    num_warps=8,
                )
            else:
                softmax_chunk_pass_strided[(M * c_full, 1, 1)](
                    output,
                    inp,
                    m_out,
                    z_out,
                    N,
                    c_full,
                    C,
                    BLOCK_N=_SM_CHUNK_BN,
                    buffer_size_limit=2048,
                    num_warps=8,
                )
        else:
            softmax_chunk_pass[(M * c_full, 1, 1)](
                output,
                inp,
                m_out,
                z_out,
                c_full,
                C,
                BLOCK_N=_SM_CHUNK_BN,
                buffer_size_limit=2048,
                num_warps=8,
            )
    base = c_full * _SM_CHUNK_BN
    for plen in pieces:
        softmax_tail_piece_pass[(M, 1, 1)](
            output,
            inp,
            m_out,
            z_out,
            N,
            base,
            PLEN=plen,
            buffer_size_limit=2048,
            num_warps=8,
        )
        base += plen
    if have_rem:
        softmax_tail_masked_pass[(M, 1, 1)](
            output,
            inp,
            m_out,
            z_out,
            N,
            base,
            rrem,
            buffer_size_limit=2048,
            num_warps=8,
        )


_SM_CHUNK_SPLIT_MAX_N = 8192 * 1024


def _softmax_forward_launch(output, inp, M, N):
    """Inner launch on a contiguous [M, N] view (reduced dim innermost)."""
    if _tle_softmax(output, inp, M, N):
        return
    use_multirow = N <= _SM_MR_MAX_N and ((N & (N - 1)) == 0)
    if use_multirow:
        # Prefer a large TILE_M; shrink (by halving) until it divides M so we
        # still take the multirow path for non-power-of-two M instead of the
        # much slower per-row `softmax_kernel_inner`.
        tile_m = _SM_MR_TILE_M if N <= 2048 else _SM_MR_TILE_M_N4096
        while tile_m > 1 and M % tile_m != 0:
            tile_m >>= 1
        grid = (M // tile_m,)
        if tile_m * N > 8192:
            softmax_kernel_multirow[grid](
                output,
                inp,
                M,
                N=N,
                TILE_M=tile_m,
                num_warps=4,
                buffer_size_limit=2048,
            )
        else:
            softmax_kernel_multirow[grid](
                output, inp, M, N=N, TILE_M=tile_m, num_warps=4
            )
        return
    if N > _SM_CHUNK_SPLIT_MAX_N and M > 1:
        grid = (M, 1, 1)
        softmax_kernel_inner[grid](
            output,
            inp,
            M,
            N,
            buffer_size_limit=2048,
            is_use_mask_zero=True,
        )
        return
    if N > _SM_MR_MAX_N:
        if M * (N // _SM_CHUNK_BN) < 1024 or M == 1:
            _softmax_chunk_split(output, inp, M, N)
        else:
            grid = (M, 1, 1)
            softmax_kernel_inner[grid](
                output,
                inp,
                M,
                N,
                buffer_size_limit=2048,
                is_use_mask_zero=True,
            )
        return
    grid = (M, 1, 1)
    softmax_kernel_inner[grid](
        output,
        inp,
        M,
        N,
        buffer_size_limit=2048,
        is_use_mask_zero=True,
    )


def softmax_backward_kernel_inner_heru_tile_n(args):
    N = args["N"]
    if N <= 32768:
        return triton.next_power_of_2(N)
    return 4096


def softmax_backward_kernel_inner_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


@libentry()
@triton.heuristics(
    values={
        "TILE_N": softmax_backward_kernel_inner_heru_tile_n,
        "ONE_TILE_PER_CTA": softmax_backward_kernel_inner_heur_one_tile_per_cta,
    },
)
@triton.jit
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    out_ptr += pid_m * N
    out_grad_ptr += pid_m * N
    in_grad_ptr += pid_m * N
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        scale = tl.sum(out_tile * out_grad_tile, 0)
        in_grad_tile = out_tile * (out_grad_tile - scale)
        tl.store(in_grad_ptr + n_offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_N], dtype=tl.float32)
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            out_tile = tl.load(out_ptr + n_offsets).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            scale += out_tile * out_grad_tile
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            scale += out_tile * out_grad_tile
        scale = tl.sum(scale, 0)

        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            out_tile = tl.load(out_ptr + n_offsets).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            in_grad_tile = out_tile * (out_grad_tile - scale)
            tl.store(in_grad_ptr + n_offsets, in_grad_tile)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            in_grad_tile = out_tile * (out_grad_tile - scale)
            tl.store(in_grad_ptr + n_offsets, in_grad_tile, mask=mask)


_SB_MR_MAX_N = 4096
_SB_N_TILE_M = [(16, 64), (64, 32), (256, 16), (1024, 8), (2048, 4), (4096, 2)]
_SB_WIDE = 8192


@triton.jit
def softmax_backward_kernel_multirow(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    mo = pid * TILE_M + tl.arange(0, TILE_M)
    no = tl.arange(0, N)
    off = mo[:, None] * N + no[None, :]
    o = tl.load(out_ptr + off).to(tl.float32)
    g = tl.load(out_grad_ptr + off).to(tl.float32)
    s = tl.sum(o * g, 1)
    tl.store(in_grad_ptr + off, o * (g - s[:, None]))


@triton.jit
def softmax_backward_kernel_multirow_pad(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    W: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    mo = tl.minimum(pid * TILE_M + tl.arange(0, TILE_M), M - 1)
    no = tl.arange(0, W)
    nc = tl.minimum(no, N - 1)
    off = mo[:, None] * N + nc[None, :]
    o = tl.load(out_ptr + off).to(tl.float32)
    g = tl.load(out_grad_ptr + off).to(tl.float32)
    s = tl.sum(tl.where(no[None, :] < N, o * g, 0.0), 1)
    tl.store(in_grad_ptr + off, o * (g - s[:, None]))


@triton.jit
def softmax_backward_kernel_perrow_p2(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    W: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < M:
        out_ptr += pid * N
        out_grad_ptr += pid * N
        in_grad_ptr += pid * N
        acc = tl.zeros([W], dtype=tl.float32)
        for start_n in range(0, N, W):
            n_offsets = start_n + tl.arange(0, W)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            acc += o * og
        scale = tl.sum(acc, 0)
        for start_n in range(0, N, W):
            n_offsets = start_n + tl.arange(0, W)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            ig = o * (og - scale)
            tl.store(in_grad_ptr + n_offsets, ig)


@triton.jit
def softmax_backward_kernel_perrow_p2_tail(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    p_tail_ptr,
    scale_ptr,
    N,
    PREV,
):
    pid = tl.program_id(0)
    out_ptr += pid * N
    out_grad_ptr += pid * N
    in_grad_ptr += pid * N
    acc = tl.zeros([4096], dtype=tl.float32)
    for start_n in range(0, PREV, 4096):
        n_offsets = start_n + tl.arange(0, 4096)
        og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
        o = tl.load(out_ptr + n_offsets).to(tl.float32)
        acc += o * og
    scale = tl.sum(acc, 0) + tl.load(p_tail_ptr + pid)
    tl.store(scale_ptr + pid, scale)
    for start_n in range(0, PREV, 4096):
        n_offsets = start_n + tl.arange(0, 4096)
        og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
        o = tl.load(out_ptr + n_offsets).to(tl.float32)
        ig = o * (og - scale)
        tl.store(in_grad_ptr + n_offsets, ig)


@triton.jit
def softmax_backward_kernel_tail_partial(
    p_ptr,
    out_ptr,
    out_grad_ptr,
    N,
    PREV,
):
    pid = tl.program_id(0)
    tno = tl.arange(0, 4096)
    tmask = tno < (N - PREV)
    o = tl.load(out_ptr + pid * N + PREV + tno, mask=tmask, other=0.0).to(tl.float32)
    g = tl.load(out_grad_ptr + pid * N + PREV + tno, mask=tmask, other=0.0).to(
        tl.float32
    )
    tl.store(p_ptr + pid, tl.sum(o * g, 0))


@triton.jit
def softmax_backward_kernel_tail_pass(
    in_grad_ptr,
    scale_ptr,
    out_ptr,
    out_grad_ptr,
    N,
    PREV,
):
    pid = tl.program_id(0)
    scale = tl.load(scale_ptr + pid)
    tno = tl.arange(0, 4096)
    tmask = tno < (N - PREV)
    o = tl.load(out_ptr + pid * N + PREV + tno, mask=tmask, other=0.0).to(tl.float32)
    g = tl.load(out_grad_ptr + pid * N + PREV + tno, mask=tmask, other=0.0).to(
        tl.float32
    )
    tl.store(in_grad_ptr + pid * N + PREV + tno, o * (g - scale), mask=tmask)


def _softmax_backward_launch_k1(output, grad_output, in_grad, M, N, input_dtype):
    if N <= _SB_MR_MAX_N:
        TILE_M = 4
        for n_hi, tm in _SB_N_TILE_M:
            if N <= n_hi:
                TILE_M = tm
                break
        grid = (triton.cdiv(M, TILE_M),)
        if N == triton.next_power_of_2(N) and M % TILE_M == 0:
            softmax_backward_kernel_multirow[grid](
                output,
                grad_output,
                in_grad,
                M,
                N=N,
                TILE_M=TILE_M,
                num_warps=8,
            )
        else:
            softmax_backward_kernel_multirow_pad[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                W=triton.next_power_of_2(N),
                TILE_M=TILE_M,
                num_warps=8,
            )
    else:
        if N % _SB_WIDE == 0:
            grid = (M,)
            softmax_backward_kernel_perrow_p2[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                W=_SB_WIDE,
            )
        elif N % 4096 == 0:
            grid = (M,)
            softmax_backward_kernel_perrow_p2[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                W=4096,
            )
        else:
            prev = (N // 4096) * 4096
            p_tail = torch.empty((M,), dtype=torch.float32, device=in_grad.device)
            scale_buf = torch.empty((M,), dtype=torch.float32, device=in_grad.device)
            grid = (M,)
            softmax_backward_kernel_tail_partial[grid](
                p_tail, output, grad_output, N, prev
            )
            softmax_backward_kernel_perrow_p2_tail[grid](
                output,
                grad_output,
                in_grad,
                p_tail,
                scale_buf,
                N,
                prev,
            )
            softmax_backward_kernel_tail_pass[grid](
                in_grad, scale_buf, output, grad_output, N, prev
            )


def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX")

    if self.ndim == 0:
        assert dim in (-1, 0), "Invalid dim"
        dtype = torch.float32 if half_to_float else self.dtype
        out = torch.empty_like(self, dtype=dtype)
        with torch_device_fn.device(self.device):
            softmax_kernel_inner[(1, 1, 1)](
                out,
                self,
                1,
                1,
                buffer_size_limit=2048,
                is_use_mask_zero=True,
            )
        return out

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    if self.numel() == 0:
        out_shape = list(self.shape)
        dtype = torch.float32 if half_to_float else self.dtype
        out = torch.empty(out_shape, dtype=dtype, device=self.device)
        zero_(out)
        return out

    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    self = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    K = self.numel() // M // N

    with torch_device_fn.device(self.device):
        if K > 1:
            inp_view = self.view(M, N, K).transpose(1, 2)
            inp_reshaped = torch.empty((M * K, N), dtype=self.dtype, device=self.device)
            if not tle_copy(inp_view, inp_reshaped):
                torch.ops.aten._copy_from(inp_view, inp_reshaped, False)
            out_reshaped = torch.empty((M * K, N), dtype=dtype, device=self.device)

            _softmax_forward_launch(out_reshaped, inp_reshaped, M * K, N)

            out = out_reshaped.view(M, K, N).transpose(1, 2).reshape(self.shape)
        else:
            out = torch.empty_like(self, dtype=dtype)
            _softmax_forward_launch(out, self, M, N)
    return out


_SM_N1_BLOCK = 512


@libentry()
@triton.jit
def softmax_kernel_n1(
    output_ptr,
    input_ptr,
    n_elem,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    e = tl.exp(x - x)
    tl.store(output_ptr + offs, e / e, mask=mask)


def _softmax_n1_flat(out, inp):
    n_elem = inp.numel()
    softmax_kernel_n1[(triton.cdiv(n_elem, _SM_N1_BLOCK), 1, 1)](
        out,
        inp,
        n_elem,
        BLOCK=_SM_N1_BLOCK,
        buffer_size_limit=2048,
    )


def _native_contiguous(t):
    """Materialize `t` contiguously through the native strided copy.

    `Tensor.contiguous()` lowers to aten::contiguous -> aten::copy_, and
    `copy_` IS a gems-registered op, so inside `flag_gems.use_gems()` it turns
    into a gems strided pointwise copy, which is far slower than the native XPU
    strided copy. `aten::_copy_from` is never overridden by gems.
    """
    dst = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    if not tle_copy(t, dst):
        torch.ops.aten._copy_from(t, dst, False)
    return dst


def softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_OUT")

    if self.ndim == 0:
        assert dim in (-1, 0), "Invalid dim"
        dtype = torch.float32 if half_to_float else self.dtype
        if out.dtype != dtype:
            raise RuntimeError(
                f"_softmax.out: expected out dtype {dtype}, got {out.dtype}"
            )
        out.copy_(softmax(self, dim, half_to_float))
        return out

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    if self.numel() == 0:
        if tuple(out.shape) != tuple(self.shape):
            out.resize_(self.shape)
        zero_(out)
        return out

    dtype = torch.float32 if half_to_float else self.dtype
    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)
    if out.dtype != dtype:
        raise RuntimeError(f"_softmax.out: expected out dtype {dtype}, got {out.dtype}")

    dim = dim % self.ndim
    M = 1
    for i in range(dim):
        M *= self.shape[i]
    N = self.shape[dim]
    inp = self if self.is_contiguous() else _native_contiguous(self)
    K = inp.numel() // M // N

    if N == 1 and out.is_contiguous():
        with torch_device_fn.device(inp.device):
            _softmax_n1_flat(out, inp)
        return out

    with torch_device_fn.device(inp.device):
        if K > 1:
            inp_t = torch.empty((M * K, N), dtype=inp.dtype, device=inp.device)
            inp_view = inp.view(M, N, K).transpose(1, 2)
            if not tle_copy(inp_view, inp_t.view(M, K, N)):
                torch.ops.aten._copy_from(inp_view, inp_t.view(M, K, N), False)
            tmp = torch.empty((M * K, N), dtype=dtype, device=inp.device)
            _softmax_forward_launch(tmp, inp_t, M * K, N)
            src = tmp.view(M, K, N).transpose(1, 2)
            if out.is_contiguous():
                if not tle_copy(src, out.view(M, N, K)):
                    torch.ops.aten._copy_from(src, out.view(M, N, K), False)
            else:
                scratch = torch.empty((M, N, K), dtype=dtype, device=out.device)
                if not tle_copy(src, scratch):
                    torch.ops.aten._copy_from(src, scratch, False)
                if not tle_copy(scratch.view(self.shape), out):
                    torch.ops.aten._copy_from(scratch.view(self.shape), out, False)
        elif not out.is_contiguous():
            tmp = torch.empty(self.shape, dtype=dtype, device=self.device)
            _softmax_forward_launch(tmp, inp, M, N)
            if not tle_copy(tmp, out):
                torch.ops.aten._copy_from(tmp, out, False)
        else:
            _softmax_forward_launch(out, inp, M, N)
    return out


def softmax_backward(grad_output, output, dim, input_dtype, grad_input=None):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = (
        grad_output if grad_output.is_contiguous() else _native_contiguous(grad_output)
    )
    output = output if output.is_contiguous() else _native_contiguous(output)
    K = output.numel() // M // N
    if grad_input is not None and K == 1:
        in_grad = grad_input
    else:
        in_grad = torch.empty_like(output, dtype=input_dtype)

    with torch_device_fn.device(in_grad.device):
        if K > 1:
            out_grad_view = grad_output.view(M, N, K).transpose(1, 2)
            out_view = output.view(M, N, K).transpose(1, 2)
            out_grad_reshaped = torch.empty(
                (M * K, N), dtype=grad_output.dtype, device=grad_output.device
            )
            out_reshaped = torch.empty(
                (M * K, N), dtype=output.dtype, device=output.device
            )
            if not tle_copy(out_grad_view, out_grad_reshaped):
                torch.ops.aten._copy_from(out_grad_view, out_grad_reshaped, False)
            if not tle_copy(out_view, out_reshaped):
                torch.ops.aten._copy_from(out_view, out_reshaped, False)
            in_grad_reshaped = torch.empty(
                (M * K, N), dtype=in_grad.dtype, device=in_grad.device
            )
            _softmax_backward_launch_k1(
                out_reshaped, out_grad_reshaped, in_grad_reshaped, M * K, N, input_dtype
            )
            in_grad = in_grad_reshaped.view(M, K, N).transpose(1, 2).view(output.shape)
        else:
            _softmax_backward_launch_k1(output, grad_output, in_grad, M, N, input_dtype)
    return in_grad


def softmax_backward_out(grad_output, output, dim, input_dtype, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_VJP_OUT")
    if tuple(grad_input.shape) != tuple(output.shape):
        grad_input.resize_(output.shape)
    if grad_input.dtype != input_dtype:
        raise RuntimeError(
            f"_softmax_backward_data.out: expected out dtype {input_dtype}, "
            f"got {grad_input.dtype}"
        )
    result = softmax_backward(
        grad_output,
        output,
        dim,
        input_dtype,
        grad_input=grad_input if grad_input.is_contiguous() else None,
    )
    if result is not grad_input:
        if not tle_copy(result, grad_input):
            torch.ops.aten._copy_from(result, grad_input, False)
    return grad_input
