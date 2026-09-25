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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# --- optional tle.gpu (cluster-DMA) path for the large-N row reduce ----------
try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False


_MULTIROW_MAX_N = 4096
_CHUNK_BN = 4096

_MID_ONLINE_MAX_N = 1024
_MID_TILE_K = 1024
_MID_JCHUNK = 4096

# ---------------------------------------------------------------------------
# tle.gpu single-pass (online) row logsumexp for the large-N contiguous case.
# The stock multirow kernel loads the whole [TILE_M, N] tile then does a max
# pass + an exp/sum pass; on 4096-wide rows that leaves the GM->LM transfer on
# the critical path (measured ~0.32x f32). This path streams the row in YBLOCK
# columns through the cluster DMA (tle.gpu.copy) with a running (max, sum), and
# when YBLOCK divides N it runs num_stages=2 so the pipeline pass overlaps the
# next chunk's DMA with the current chunk's reduce. Falls back (returns False)
# whenever tle is unavailable or the shape/dtype does not fit.
_LSE_TLE = os.environ.get("TRITONXPU_LSE_TLE", "1") != "0"
_LSE_TLE_CORE_NUM = 64
_LSE_TLE_LM_BYTES_PER_CORE = 4096  # 8KB/core fails to allocate here
# logsumexp is exp-throughput-bound, so it wants many small programs (one row
# per core, grid = M/64) rather than sum's few wide ones. Measured optimum.
_LSE_TLE_XBLOCK = 64
_LSE_TLE_MIN_N = 4096  # only intercept the large-N tier for now
_LSE_TLE_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


def _lse_npo2(x):
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _tle_lse_available():
    if not _HAS_TLE or not _LSE_TLE:
        return False
    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


_LSE_TLE_AVAILABLE = _tle_lse_available()


_LSE_TLE_GEOM = {}


def _tle_lse_geom(M, N, itemsize):
    """`(xblock, yblock, num_stages, need_zero)` for a [M, N] row logsumexp.

    Unlike `sum` (memory-bound: one wide program per cluster), logsumexp is
    exp-throughput-bound, so it wants *parallelism*: a narrow XBLOCK=64 (one row
    per core, grid = M/64 spread over every cluster) with the widest YBLOCK the
    LM budget allows. Measured on 4096x4096 (XBLOCK, YBLOCK): (512,64) 0.90ms,
    (128,512) 0.28ms, (64, full) 0.27ms f32 / 0.28ms f16 -- against the stock
    kernel's 0.52/0.37ms. num_stages>=2 did not help (0.35 vs 0.36), so the loop
    runs synchronous: the win is parallelism + a long YBLOCK, not DMA prefetch.
    """
    key = (M, N, itemsize)
    cached = _LSE_TLE_GEOM.get(key)
    if cached is not None:
        return cached
    xblock = min(_LSE_TLE_XBLOCK, max(1, _lse_npo2(M)))
    # Widest YBLOCK that fits the LM budget at this dtype's native width. The
    # tile is f32 in registers but only XBLOCK/core_num == 1 row per core, so the
    # per-core f32 slice is [1, YBLOCK] and the native-width budget is the binding
    # constraint (f32 -> 1024, f16/bf16 -> 2048 on the 4 KB/core budget).
    budget_elems = _LSE_TLE_LM_BYTES_PER_CORE * _LSE_TLE_CORE_NUM // itemsize
    yblock = max(1, min(_lse_npo2(N), budget_elems // xblock))
    while yblock > N and yblock > 1:
        yblock >>= 1
    geom = (xblock, yblock, 1, (N % yblock) != 0)
    _LSE_TLE_GEOM[key] = geom
    return geom


@triton.jit(
    do_not_specialize=["N"],
    do_not_specialize_on_alignment=["a_desc", "c_desc"],
)
def _tle_lse_row_kernel(
    a_desc,
    c_desc,
    N,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """logsumexp of a [XBLOCK, YBLOCK]-tiled [M, N] slice along axis=1, online.

    Each core owns whole rows of the tile (core-tiling), so the running max `m`
    and max-shifted running sum `z` are core-local [XBLOCK] and need no barrier.
    A short last step is padded with -inf (max-neutral; exp(-inf)=0 sum-neutral)
    via a zero-fill store, which disqualifies the pipeline pass -- so NEED_ZERO
    shapes run synchronous (NUM_STAGES=1) and only N%YBLOCK==0 shapes pipeline.
    """
    pid = tl.program_id(0)
    row_off = pid * XBLOCK
    NEG_INF = float("-inf")

    a_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    c_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem)
    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
    c_ptrs = tle.gpu.local_ptr(c_lmem, (tl.arange(0, XBLOCK),))

    m = tl.full([XBLOCK], NEG_INF, tl.float32)
    z = tl.zeros([XBLOCK], tl.float32)
    for coff in tl.range(0, N, YBLOCK, num_stages=NUM_STAGES):
        if NEED_ZERO:
            if coff + YBLOCK > N:
                tl.store(a_ptrs, tl.full([XBLOCK, YBLOCK], NEG_INF, IN_DTYPE))
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        a = tl.load(a_ptrs).to(tl.float32)
        m_c = tl.max(a, axis=1)
        m_new = tl.maximum(m, m_c)
        all_neg = m_new == NEG_INF
        sc = tl.sum(tl.exp(a - m_new[:, None]), axis=1)
        z = tl.where(all_neg, z, z * tl.exp(m - m_new) + sc)
        m = m_new

    safe_m = tl.where(m == NEG_INF, 0.0, m)
    res = tl.where(
        m == NEG_INF, m, tl.where(m == float("inf"), m, safe_m + tl.log(z))
    )
    tl.store(c_ptrs, res.to(OUT_DTYPE))
    tle.gpu.copy(c_lmem, c_desc, [XBLOCK], [row_off])


def _tle_logsumexp_row(inp, out, M, N):
    """Row logsumexp `out[m] = logsumexp(inp[m, :])` on the tle.gpu path.

    Returns True on success, False to let the caller keep its own kernel.
    """
    if not _LSE_TLE_AVAILABLE or N < _LSE_TLE_MIN_N:
        return False
    if inp.dtype not in _LSE_TLE_TL_DTYPE or out.dtype not in _LSE_TLE_TL_DTYPE:
        return False
    if not inp.is_contiguous() or not out.is_contiguous():
        return False
    xblock, yblock, num_stages, need_zero = _tle_lse_geom(M, N, inp.element_size())
    a = inp if inp.ndim == 2 and inp.shape[0] == M else inp.view(M, N)
    c = out if out.ndim == 1 else out.view(M)
    grid = (triton.cdiv(M, xblock),)
    with torch_device_fn.device(inp.device):
        _tle_lse_row_kernel[grid](
            TensorDescriptor.from_tensor(a, block_shape=[xblock, yblock]),
            TensorDescriptor.from_tensor(c, block_shape=[xblock]),
            N,
            XBLOCK=xblock,
            YBLOCK=yblock,
            IN_DTYPE=_LSE_TLE_TL_DTYPE[inp.dtype],
            OUT_DTYPE=_LSE_TLE_TL_DTYPE[out.dtype],
            NEED_ZERO=need_zero,
            NUM_STAGES=num_stages,
        )
    return True


@triton.jit
def _lse_poly_combine(a, b):
    # Associative LSE combine without exp/log:
    #   log(e^a + e^b) = max(a, b) + g(|a-b|),  g(d) = log(1 + e^{-d})
    # g is a deg-4 polynomial (fit over d in [0, 6], max err ~1.9e-3); for d >= 6
    # a saturating min/max mask forces g=0 (avoids a select). -inf is the identity
    # (combine(-inf, x) = x); +inf clamps to g=0 so +inf is preserved. No overflow.
    m = tl.maximum(a, b)
    d = tl.maximum(a - b, b - a)
    dc = tl.minimum(d, 6.0)
    g = 1.10249731e-03
    g = g * dc - 2.08612699e-02
    g = g * dc + 1.51769950e-01
    g = g * dc - 5.12768776e-01
    g = g * dc + 6.94473086e-01
    keep = tl.minimum(1.0, tl.maximum(0.0, (6.0 - d) * 1.0e30))
    return m + g * keep


@libentry()
@triton.jit
def logsumexp_kernel_multirow(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
    NEED_MASK: tl.constexpr,
    USE_POLY: tl.constexpr,
):
    """Reduce the innermost dim N for many rows per program.

    USE_POLY=0: numerically-stable max-shift form (uses exp).
    USE_POLY=1: polynomial LSE reduction (no exp/log) via _lse_poly_combine as a
      custom tt.reduce, bypassing the exp throughput floor. No overflow, so no
      host-side isinf fallback is needed.
    N is constexpr; the [TILE_M, N] tile is a stride-1 contiguous block (block DMA).
    """
    pid = ext.program_id(0)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    m_mask = m_offsets < M
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    if NEED_MASK:
        inp = tl.load(
            input_ptr + offsets, mask=m_mask[:, None], other=-float("inf")
        ).to(tl.float32)
    else:
        inp = tl.load(input_ptr + offsets).to(tl.float32)
    if USE_POLY:
        res = tl.reduce(inp, 1, _lse_poly_combine)
    else:
        m = tl.max(inp, axis=1)
        safe_m = tl.where(m == float("-inf"), 0.0, m)
        z = tl.sum(tl.exp(inp - safe_m[:, None]), axis=1)
        res = tl.where(
            m == float("-inf"), m, tl.where(m == float("inf"), m, safe_m + tl.log(z))
        )
    tl.store(output_ptr + m_offsets, res, mask=m_mask)


@libentry()
@triton.jit
def logsumexp_kernel_fused2(
    output_ptr,
    input_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
    NEED_COLMASK: tl.constexpr,
):
    """Fused two-pass logsumexp for the fp32 inner-dim range (64 < N).

    Used for fp32 (and any non-16-bit dtype) where a [TILE_M, N] tile exceeds
    register capacity and the compiler spills it to local memory and re-reads
    it for each reduction (3 reads/element -> ~190GB/s). Instead, both
    reductions use the mean_dim-style persisted [BLOCK_M, BLOCK_N] fp32
    accumulator:

      - Pass 1 (max): elementwise ``tl.maximum`` accumulate over N in BLOCK_N
        chunks + a single narrow ``tl.max`` reduce over BLOCK_N. Plain float
        max here is as fast as ``tl.sum`` (~550GB/s at BLOCK_M=64/512) -- the
        uint32-key trick is a *liability* in this structure (int key ops + the
        old wide-row reduce are ~3.5x slower than plain float max), so it is
        dropped.
      - Pass 2 (exp-sum): elementwise ``z_acc += exp(a - safe_m)`` accumulate
        + a single narrow reduce over BLOCK_N.

    This reads each element from global memory twice (once per pass) instead
    of three times, and never materializes the full [BLOCK_M, N] tile. At
    BLOCK_M=64/BLOCK_N=512 this reaches ~0.32ms for [4096,4096] fp32 vs the
    old ~0.53ms (per-op split: plain float max == plain sum == 550GB/s, exp is
    the irreducible vexpf ~70Gelem/s cost).

    Single-pass online variants are all worse on this backend for fp32:
    elementwise online needs 2x exp (1.22ms), chunked-with-scalar-rescale
    needs per-chunk wide reduces (~3 Gelem/s/reduce, 0.38ms). The two-pass
    elementwise structure is the measured optimum here.
    """
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = input_ptr + pid * N
    row_mask = pid < M
    # ---- Pass 1: max (plain float elementwise accumulate + narrow reduce) ----
    m_acc = tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        a = tl.load(X + cols, mask, other=-float("inf")).to(tl.float32)
        m_acc = tl.maximum(m_acc, a)
    m = tl.max(m_acc, axis=1)[:, None]
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    # ---- Pass 2: exp-sum (elementwise accumulate + narrow reduce) ----
    z_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        a = tl.load(X + cols, mask, other=-float("inf")).to(tl.float32)
        z_acc += tl.exp(a - safe_m)
    z = tl.sum(z_acc, axis=1)[:, None]
    res = tl.where(
        m == float("-inf"),
        m,
        tl.where(m == float("inf"), m, safe_m + tl.log(z)),
    )
    tl.store(output_ptr + pid, res, row_mask)


@libentry()
@triton.jit
def logsumexp_kernel_chunked(
    output_ptr,
    input_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
    NEED_COLMASK: tl.constexpr,
):
    """Single-read online logsumexp for fp16/bf16, 64 < N <= _MULTIROW_MAX_N.

    For the 2-byte dtypes the fused two-pass kernel re-reads and re-converts
    the data (fp16/bf16 -> fp32) a second time, and that extra convert+read
    costs more than a single pass with per-chunk wide reduces. This variant
    reads each element from global memory exactly once:

      per chunk: m_c = max(a, axis=1)  (wide reduce over BLOCK_N)
                 z_c = sum(exp(a - m_new), axis=1)
                 online scalar rescale per row (z_row * exp(m_row - m_new))
      final:     out = m + log(z)

    The per-chunk wide reduce is cheap relative to the fp16/bf16 memory saved:
    measured [1024,1024] fp16 0.70 vs fused2 0.49, [4096,4096] fp16 0.44 vs
    0.37, bf16 0.70/0.45 vs 0.45/0.31. For fp32 (4-byte) the wide reduce cost
    outweighs the single-read saving, so fp32 keeps the fused two-pass kernel.
    """
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = input_ptr + pid * N
    row_mask = pid < M
    m_row = tl.full([BLOCK_M, 1], float("-inf"), tl.float32)
    z_row = tl.full([BLOCK_M, 1], 0.0, tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        a = tl.load(X + cols, mask, other=-float("inf")).to(tl.float32)
        m_c = tl.max(a, axis=1)[:, None]
        m_new = tl.maximum(m_row, m_c)
        z_c = tl.sum(tl.exp(a - m_new), axis=1)[:, None]
        all_neg = m_new == float("-inf")
        z_row = tl.where(all_neg, z_row, z_row * tl.exp(m_row - m_new) + z_c)
        m_row = m_new
    safe_m = tl.where(m_row == float("-inf"), 0.0, m_row)
    res = tl.where(
        m_row == float("-inf"),
        m_row,
        tl.where(m_row == float("inf"), m_row, safe_m + tl.log(z_row)),
    )
    tl.store(output_ptr + pid, res, row_mask)


@libentry()
@triton.jit
def logsumexp_kernel_partial(
    mrow_ptr,
    zrow_ptr,
    input_ptr,
    R,
    BN: tl.constexpr,
    TILE_R: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Per-chunk partial (max, sum-exp) for a big innermost dim.

    Input is the flattened [rows * C, BN] view of the full 4096-chunks (each
    chunk stride-1 contiguous; BN constexpr keeps block DMA). No column
    masking -- the caller routes any tail (N % BN != 0) through the per-row
    kernel instead (masked-column reductions miscompute on this backend).
    Partial (m_c, z_c) pairs are stored compactly per chunk row; the host pads
    each row to TILE_C with -inf/0 so the combine kernel reads mask-free.
    """
    pid = ext.program_id(0)
    r_offsets = pid * TILE_R + tl.arange(0, TILE_R)
    r_mask = r_offsets < R
    n_offsets = tl.arange(0, BN)
    offsets = r_offsets[:, None] * BN + n_offsets[None, :]
    if NEED_MASK:
        a = tl.load(input_ptr + offsets, mask=r_mask[:, None], other=-float("inf")).to(
            tl.float32
        )
    else:
        a = tl.load(input_ptr + offsets).to(tl.float32)
    bits = a.to(tl.uint32, bitcast=True)
    # Same select-free key construction as `logsumexp_kernel_multirow`.
    neg = (bits.to(tl.int32, bitcast=True) >> 31).to(tl.uint32, bitcast=True)
    key = bits ^ (0x80000000 | (neg & 0x7FFFFFFF))
    m_key = tl.max(key, axis=1)
    bits_m = tl.where(m_key < 0x80000000, m_key ^ 0xFFFFFFFF, m_key ^ 0x80000000)
    m = bits_m.to(tl.float32, bitcast=True)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    z = tl.sum(tl.exp(a - safe_m[:, None]), axis=1)
    tl.store(mrow_ptr + r_offsets, m, mask=r_mask)
    tl.store(zrow_ptr + r_offsets, z, mask=r_mask)


@libentry()
@triton.jit
def logsumexp_kernel_combine(
    output_ptr,
    mrow_ptr,
    zrow_ptr,
    mtail_ptr,
    ztail_ptr,
    M,
    C_FULL: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    TILE_C: tl.constexpr,
):
    """Combine the C_FULL per-chunk partials of one row plus (optionally) the
    tail partial at slot C_FULL: out = m + log(sum zc exp(mc - m))."""
    row = ext.program_id(0)
    c_offsets = tl.arange(0, TILE_C)
    mc = tl.load(mrow_ptr + row * TILE_C + c_offsets)
    zc = tl.load(zrow_ptr + row * TILE_C + c_offsets)
    if HAS_TAIL:
        m_t = tl.load(mtail_ptr + row)
        z_t = tl.load(ztail_ptr + row)
        is_tail = c_offsets == C_FULL
        mc = tl.where(is_tail, m_t, mc)
        zc = tl.where(is_tail, z_t, zc)
    m = tl.max(mc, axis=0)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    # exp(mc - safe_m) is 0 for the -inf pad chunks; zc is 0 there too.
    z = tl.sum(zc * tl.exp(mc - safe_m), axis=0)
    res = tl.where(
        m == float("-inf"), m, tl.where(m == float("inf"), m, safe_m + tl.log(z))
    )
    tl.store(output_ptr + row, res)


@libentry()
@triton.jit
def logsumexp_kernel_tail_partials(
    mrow_ptr,
    zrow_ptr,
    input_ptr,
    M,
    ROW_STRIDE,
    N,
    TILE_N: tl.constexpr,
):
    """Per-row (m, z) partials for a tail slice [M, N] strided by ROW_STRIDE.

    Tail widths are < _CHUNK_BN (<= 4096), so TILE_N is power-of-two and the
    loop body executes once; the single masked iteration is verified exact on
    this backend (unlike padded 2D-tile masked reductions). Emits compact
    per-row max m and max-shifted sum z for the combine kernel.
    """
    pid = ext.program_id(0)
    m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
    z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
    input_ptr += pid * ROW_STRIDE

    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        a = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        m_new = tl.maximum(m, a)
        all_neg_inf = m_new == float("-inf")
        z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(a - m_new))
        m = m_new

    m_r = tl.max(m, axis=0)
    z_r = tl.sum(z * tl.exp(m - m_r), axis=0)
    # all-(-inf) tails must contribute z=0 to the combine (exp(-inf - -inf)
    # would be NaN), and all-(-inf) rows are resolved by the combine's -inf
    # guard.
    tl.store(mrow_ptr + pid, m_r)
    tl.store(zrow_ptr + pid, tl.where(m_r == float("-inf"), 0.0, z_r))


def _reduce_inner_small(inp, rows, N, out):
    """Single-tile multirow kernel for N <= _MULTIROW_MAX_N (exact).

    TILE_M=32 for N > 64: the [32, N] tile is the measured sweet spot on this
    XPU (16.3us vs 18.2us/14.1us for 64/16 on [256,256] f32; ~-2% on
    [1024,1024]; ~+2% on [4096,4096]) -- the previous 64/16 split was tuned
    for the N<=64 launch-bound tier only. The N <= 64 tier keeps 16 (a single
    small tile per program; 32 would waste partial rows)."""
    is_f32 = inp.dtype == torch.float32
    if N <= 64:
        TILE_M = 16
    elif N <= 256:
        TILE_M = 32
    elif N <= 1024:
        TILE_M = 96 if is_f32 else 128
    else:
        TILE_M = 8 if is_f32 else 16
    # Non-chunked poly-LSE measured worse: the full-width custom combine still
    # blows up uni_sram compilation (even for N <= 1024), and the deg-4 polynomial
    # error (~8e-3) exceeds the test tolerance (~1e-4), so fp16 fails too.
    # Disabled for now; use the exp-stable form.
    use_poly = 0
    need_mask = 1 if rows % TILE_M else 0
    grid = (triton.cdiv(rows, TILE_M), 1, 1)
    logsumexp_kernel_multirow[grid](
        out,
        inp,
        rows,
        N=N,
        TILE_M=TILE_M,
        NEED_MASK=need_mask,
        USE_POLY=use_poly,
        num_warps=8,
        buffer_size_limit=2048,
    )


def _reduce_tail_partials(mrow, zrow, inp, rows, row_stride, tail_n):
    """Reduce a [rows, tail_n] tail-view (strided by row_stride) into compact
    (m, z) partials via the per-row online kernel."""
    TILE_N = max(1, triton.next_power_of_2(tail_n))
    grid = (rows, 1, 1)
    logsumexp_kernel_tail_partials[grid](
        mrow,
        zrow,
        inp,
        rows,
        row_stride,
        tail_n,
        TILE_N=TILE_N,
        num_warps=4,
        buffer_size_limit=2048,
    )


def _reduce_inner(inp, rows, N):
    """logsumexp over the innermost dim N of a contiguous [rows, N] tensor."""
    out = torch.empty((rows,), dtype=inp.dtype, device=inp.device)
    # Large-N: stream the row through the cluster DMA with an online (max, sum)
    # so the transfer overlaps the reduce (see _tle_logsumexp_row). Falls through
    # to the stock kernels when tle is unavailable or the shape does not fit.
    if _tle_logsumexp_row(inp, out, rows, N):
        return out
    if N <= _MULTIROW_MAX_N and (N & (N - 1)) == 0:
        _reduce_inner_small(inp, rows, N, out)
    else:
        # Chunk-split path: single data read, single exp per element. Full
        # 4096-chunks go through the tile kernel; any tail (N % 4096 != 0) is
        # reduced by the multirow kernel over a tail-slice view (masked-tail
        # reductions miscompute on this backend).
        BN = _CHUNK_BN
        C_full = N // BN
        TAIL = N - C_full * BN
        TILE_C = max(1, triton.next_power_of_2(C_full + (1 if TAIL else 0)))
        # partials compact per chunk; then per-row padded to TILE_C with
        # (-inf, 0) pad slots so the combine kernel reads mask-free.
        mrow = torch.empty((rows * C_full,), dtype=torch.float32, device=inp.device)
        zrow = torch.empty_like(mrow)
        if C_full:
            R = rows * C_full
            TILE_R = 32
            need_mask = 1 if R % TILE_R else 0
            full_view = inp[:, : C_full * BN]
            # reshape may copy only when the slice is non-contiguous (tail
            # cases with N % BN != 0); the aligned path is a null-op view.
            flat = full_view.reshape(R, BN)
            grid = (triton.cdiv(R, TILE_R), 1, 1)
            logsumexp_kernel_partial[grid](
                mrow,
                zrow,
                flat,
                R,
                BN=BN,
                TILE_R=TILE_R,
                NEED_MASK=need_mask,
                num_warps=4,
                buffer_size_limit=2048,
            )
        if C_full and TILE_C != C_full:
            mrow = mrow.view(rows, C_full)
            zrow = zrow.view(rows, C_full)
            mp = torch.full(
                (rows, TILE_C), -float("inf"), dtype=torch.float32, device=inp.device
            )
            zp = torch.zeros((rows, TILE_C), dtype=torch.float32, device=inp.device)
            mp[:, :C_full] = mrow
            zp[:, :C_full] = zrow
            mrow = mp
            zrow = zp
        elif not C_full:
            mrow = torch.full(
                (rows, TILE_C), -float("inf"), dtype=torch.float32, device=inp.device
            )
            zrow = torch.zeros((rows, TILE_C), dtype=torch.float32, device=inp.device)
        if TAIL:
            # tail slice view: [rows, TAIL] strided by N (no copy)
            tail_view = inp[:, C_full * BN : N]
            mtail = torch.empty((rows,), dtype=torch.float32, device=inp.device)
            ztail = torch.empty_like(mtail)
            _reduce_tail_partials(mtail, ztail, tail_view, rows, N, TAIL)
        else:
            # unused sentinel pointer for the HAS_TAIL=0 build
            mtail = torch.empty((1,), dtype=torch.float32, device=inp.device)
            ztail = torch.empty_like(mtail)
        logsumexp_kernel_combine[(rows, 1, 1)](
            out,
            mrow,
            zrow,
            mtail,
            ztail,
            rows,
            C_FULL=C_full,
            HAS_TAIL=1 if TAIL else 0,
            TILE_C=TILE_C,
            num_warps=4,
            buffer_size_limit=2048,
        )
    return out


def _reduce_middle(inp, dim, keepdim):
    """Reduce a non-innermost dim with the gems' own machinery.

    The reduced dim is compressed innermost (``dim_compress`` -> permute +
    contiguous, materialized by the FlagGems copy_, never the vendor engine),
    then the contiguous inner-dim kernels run as usual. Measured on
    [64,512,512] fp32 dim=1: 0.61 ms vs torch 0.78 ms (~1.26x), where the old
    native delegation reported 0.98 by construction.
    """
    N = inp.shape[dim]
    perm = dim_compress(inp, dim)
    M = perm.numel() // N
    # _reduce_inner views its input as a contiguous [rows, N] matrix; the
    # permuted tensor is contiguous, so the reshape is a free view. (Passing the
    # N-D tensor directly made the N > _MULTIROW_MAX_N chunk-split path slice
    # the wrong dim -> "shape [6000, 4096] is invalid for input of size
    # 24599400" on [200, 40999, 3].)
    out = _reduce_inner(perm.reshape(M, N), M, N)
    shape = list(perm.shape)
    shape[-1] = 1
    out = out.view(shape)
    order = [i for i in range(inp.ndim) if i != dim] + [dim]
    inverse = [0] * inp.ndim
    for pos, src in enumerate(order):
        inverse[src] = pos
    out = out.permute(inverse)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def logsumexp(inp, dim, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN LOGSUMEXP")

    if isinstance(dim, (list, tuple)):
        if len(dim) == 0:
            # Empty dim list means no reduction, just return the input.
            return inp.clone()
        if len(dim) != 1:
            # Multi-dim reduction: fold single-dim reductions (innermost
            # first so the dim indices stay valid), same as the generic
            # implementation.
            sorted_dims = sorted([d % inp.ndim for d in dim], reverse=True)
            result = inp
            for d in sorted_dims:
                result = logsumexp(result, d, keepdim=True)
            if not keepdim:
                for d in sorted(sorted_dims, reverse=True):
                    result = result.squeeze(d)
            return result
        dim = dim[0]

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    N = inp.shape[dim]
    K = 1
    for i in range(dim + 1, inp.ndim):
        K *= inp.shape[i]

    # Middle-dim reduction (K > 1) or a size-1 reduction: compress the reduced
    # dim innermost and use the contiguous inner-dim kernels (see the module
    # header note on why the native redispatch was removed).
    if K > 1 or N == 1:
        return _reduce_middle(inp, dim, keepdim)

    # K == 1: innermost-dim reduction -> fast contiguous Triton kernels.
    M = 1
    for i in range(dim):
        M *= inp.shape[i]
    inp = inp.contiguous()
    shape = list(inp.shape)
    shape[dim] = 1

    with torch_device_fn.device(inp.device):
        out = _reduce_inner(inp, M, N).view(shape)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
