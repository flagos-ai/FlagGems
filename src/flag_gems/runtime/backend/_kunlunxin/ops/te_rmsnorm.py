import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False


def _tle_available():
    """`tle.gpu` exists only on the xpu3 (KL3) cluster pipeline."""
    if not _HAS_TLE:
        return False
    import os

    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


_TLE_AVAILABLE = _tle_available()

# Latched True when this toolchain cannot compile the tle row kernel. flagtree XPU
# fails with "failed to legalize operation 'tt.make_range' that was explicitly marked
# illegal" on its 2D local_ptr range, so the fwd falls back to the stock kernels
# and every later call skips tle (the same latch ops/sum.py and ops/logsumexp.py
# use). A launch that fails to compile leaves `out` untouched, so the fallback is
# safe.
_TLE_BROKEN = False

# tle row-reduce geometry, mirroring ops/sum.py `_tle_row_geom` (see its comment
# for the full trade-offs): XBLOCK is rows per program and must reach ~64 so the
# core-tiling hands every core whole rows and the axis-1 reduce stays
# core-local; YBLOCK is the LM budget divided by XBLOCK. We halve the raw
# budget because the kernel also stages r (f32) and, for f16, keeps a YBLOCK
# weight row in the same LM footprint.
_TLE_CORE_NUM = 64
_TLE_LM_BYTES_PER_CORE = 4096
_TLE_XBLOCK = 512
_TLE_CLUSTERS = 8
_TLE_WIDE_ROW_BYTES = 32768
_TLE_NARROW_ROW_BYTES = 2048

_TLE_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


def _tle_npo2(x):
    """Cheap next-power-of-2 (triton.next_power_of_2 is ~3us per call)."""
    return 1 << (x - 1).bit_length() if x > 1 else 1


@triton.jit(
    do_not_specialize=["N"],
    do_not_specialize_on_alignment=["x_desc", "y_desc", "w_desc", "r_desc"],
)
def _tle_rmsnorm_fwd_kernel(
    x_desc,
    y_desc,
    w_desc,
    r_desc,
    N,
    eps,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    """RMSNorm forward on [M, N] rows: y[m] = x[m] / sqrt(mean(x^2)+eps) * w.

    Each program owns XBLOCK rows. tritonxpu-tle-core-tiling hands each of the
    64 cores XBLOCK/64 whole rows, so the axis-1 reduction stays core-local and
    needs no cluster barrier (the 2D-tile `tl.sum(axis=1)` path the original
    kernel uses builds one). x is streamed in YBLOCK-column chunks through LM
    via TMA; pass 1 accumulates sum(x^2), pass 2 recomputes y = x*rrms*w and
    writes it back through the same LM buffer (in place). `tle.gpu.copy`
    clamps to the descriptor extents, so partial tiles are handled by
    zero-filling LM first (NEED_ZERO) and the output copy drops the
    out-of-range rows/columns. The weight also goes through TMA: a plain GM
    tl.load cannot be mixed with local_ptr loads in a TLE kernel
    (TritonXPUTLEDtypeConvert rejects it). Only f16/f32 reach this kernel --
    bf16 is excluded in the host guard because the bf16->f32 conversion on the
    LM-loaded path returns wrong bits on this backend.
    """
    pid = tl.program_id(0)
    row_off = pid * XBLOCK

    x_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=DTYPE, layout=None, scope=tle.gpu.lmem
    )
    w_lmem = tle.gpu.alloc([YBLOCK], dtype=DTYPE, layout=None, scope=tle.gpu.lmem)
    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    x_ptrs = tle.gpu.local_ptr(x_lmem, (row_ids, col_ids))
    w_ptrs = tle.gpu.local_ptr(w_lmem, (tl.arange(0, YBLOCK),))

    # ---- pass 1: sum of squares -> rrms ----
    acc = tl.zeros([XBLOCK], dtype=tl.float32)
    for coff in tl.range(0, N, YBLOCK):
        if NEED_ZERO:
            if coff + YBLOCK > N:
                tl.store(x_ptrs, tl.zeros([XBLOCK, YBLOCK], DTYPE))
        tle.gpu.copy(x_desc, x_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        xf = tl.load(x_ptrs).to(tl.float32)
        acc += tl.sum(xf * xf, axis=1)
    rrms = 1.0 / tl.sqrt(acc / N + eps)

    # ---- rsigma: stage in LM and write back via TMA (clamps rows past M) ----
    r_lmem = tle.gpu.alloc([XBLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.lmem)
    r_ptrs = tle.gpu.local_ptr(r_lmem, (tl.arange(0, XBLOCK),))
    tl.store(r_ptrs, rrms)
    tle.gpu.copy(r_lmem, r_desc, [XBLOCK], [row_off])

    # ---- pass 2: apply y = x * rrms * w ----
    for coff in tl.range(0, N, YBLOCK):
        if NEED_ZERO:
            if coff + YBLOCK > N:
                tl.store(x_ptrs, tl.zeros([XBLOCK, YBLOCK], DTYPE))
                tl.store(w_ptrs, tl.zeros([YBLOCK], DTYPE))
        tle.gpu.copy(x_desc, x_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        tle.gpu.copy(w_desc, w_lmem, [YBLOCK], [coff])
        x = tl.load(x_ptrs)
        wv = tl.load(w_ptrs)
        y = (x.to(tl.float32) * rrms[:, None] * wv.to(tl.float32)[None, :]).to(DTYPE)
        tl.store(x_ptrs, y)
        tle.gpu.copy(x_lmem, y_desc, [XBLOCK, YBLOCK], [row_off, coff])


def _tle_rmsnorm_geom(M, N, itemsize):
    """Cached (xblock, yblock, row_blocks) for the TLE forward kernel."""
    key = (M, N, itemsize)
    geom = _TLE_RMSNORM_GEOM.get(key, _TLE_PLAN_MISS)
    if geom is not _TLE_PLAN_MISS:
        return geom
    xblock = min(_TLE_XBLOCK, max(128, _tle_npo2(-(-M // _TLE_CLUSTERS))))
    row_bytes = N * itemsize
    if row_bytes >= _TLE_WIDE_ROW_BYTES:
        xblock = max(_TLE_CORE_NUM, xblock // 4)
    elif row_bytes <= _TLE_NARROW_ROW_BYTES:
        xblock = _TLE_XBLOCK if M > _TLE_CORE_NUM else 256
    tile_elems = _TLE_LM_BYTES_PER_CORE * _TLE_CORE_NUM // itemsize
    yblock = max(1, min(_tle_npo2(N), (tile_elems // xblock) >> 1))
    while yblock > N and yblock > 1:
        yblock >>= 1
    geom = (xblock, yblock, -(-M // xblock))
    _TLE_RMSNORM_GEOM[key] = geom
    return geom


_TLE_RMSNORM_GEOM = {}
_TLE_PLAN_MISS = object()


def _tle_rmsnorm_fwd_launch(x, y, w, rsigma, M, N, eps):
    """Try the TLE row-reduction kernel on this toolchain.

    Returns True on success; False when the kernel is out of reach (bf16,
    zero_centered_gamma f32 weight, non-contiguous inputs, N < 64) or when the
    toolchain cannot compile the tle.gpu payload (flagtree XPU fails to legalize
    the 2D local_ptr `tt.make_range`), so the caller falls back to the stock
    kernels. One failed compile per process, then the `_TLE_BROKEN` latch skips
    it -- the same setup ops/sum.py and ops/logsumexp.py use. On toolchains
    that compile it, the TLE row-reduction (core-local axis-1 reduce, no
    cluster barrier, TMA staging) measures 1.3x-5.7x over the 2D-tile kernel
    and 1.07x-2.24x over the row kernel on the benchmark shapes (see
    harness/solution/te_rmsnorm_fwd/README.md).
    """
    global _TLE_BROKEN
    if not _TLE_AVAILABLE or _TLE_BROKEN or N < 64:
        return False
    if x.dtype not in _TLE_TL_DTYPE or w.dtype != x.dtype:
        return False
    if not (
        x.is_contiguous()
        and y.is_contiguous()
        and w.is_contiguous()
        and rsigma.is_contiguous()
    ):
        return False
    xb, yb, grid = _tle_rmsnorm_geom(M, N, x.element_size())
    try:
        _tle_rmsnorm_fwd_kernel[(grid,)](
            TensorDescriptor.from_tensor(x, block_shape=[xb, yb]),
            TensorDescriptor.from_tensor(y, block_shape=[xb, yb]),
            TensorDescriptor.from_tensor(w, block_shape=[yb]),
            TensorDescriptor.from_tensor(rsigma, block_shape=[xb]),
            N,
            float(eps),
            xb,
            yb,
            _TLE_TL_DTYPE[x.dtype],
            (N % yb) != 0,
        )
    except Exception as exc:  # noqa: BLE001 - any compile gap falls back
        _TLE_BROKEN = True
        logger.debug(
            "GEMS_KUNLUNXIN RMSNORM_FWD tle unavailable (%s); fallback", exc
        )
        return False
    return True


@triton.jit
def _te_rmsnorm_bwd_dx_1d_kernel(
    dx_ptr,
    dz_ptr,
    x_ptr,
    gamma_ptr,
    rsigma_ptr,
    N,
    zero_centered_gamma: tl.constexpr,
    C: tl.constexpr,
    NEED_TAIL: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * N
    rsigma = tl.load(rsigma_ptr + pid).to(tl.float32)
    acc = tl.zeros([C], dtype=tl.float32)

    for off in range(0, N, C):
        cols = off + tl.arange(0, C)
        if NEED_TAIL:
            cmask = cols < N
            x = tl.load(x_ptr + row + cols, mask=cmask, other=0.0).to(tl.float32)
            dz = tl.load(dz_ptr + row + cols, mask=cmask, other=0.0).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
            if zero_centered_gamma:
                gamma = gamma + 1.0
            acc += x * rsigma * dz * gamma
        else:
            x = tl.load(x_ptr + row + cols).to(tl.float32)
            dz = tl.load(dz_ptr + row + cols).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols).to(tl.float32)
            if zero_centered_gamma:
                gamma = gamma + 1.0
            acc += x * rsigma * dz * gamma

    c1 = tl.sum(acc) / N

    for off in range(0, N, C):
        cols = off + tl.arange(0, C)
        if NEED_TAIL:
            cmask = cols < N
            x = tl.load(x_ptr + row + cols, mask=cmask, other=0.0).to(tl.float32)
            dz = tl.load(dz_ptr + row + cols, mask=cmask, other=0.0).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
            if zero_centered_gamma:
                gamma = gamma + 1.0
            x_hat = x * rsigma
            dx = rsigma * (dz * gamma - x_hat * c1)
            tl.store(dx_ptr + row + cols, dx, mask=cmask)
        else:
            x = tl.load(x_ptr + row + cols).to(tl.float32)
            dz = tl.load(dz_ptr + row + cols).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols).to(tl.float32)
            if zero_centered_gamma:
                gamma = gamma + 1.0
            x_hat = x * rsigma
            dx = rsigma * (dz * gamma - x_hat * c1)
            tl.store(dx_ptr + row + cols, dx)


@triton.jit
def _te_rmsnorm_bwd_dgamma_kernel(
    dgamma_partial_ptr,
    dz_ptr,
    x_ptr,
    rsigma_ptr,
    M,
    N,
    BM: tl.constexpr,
    C: tl.constexpr,
):
    n0 = tl.program_id(0) * C
    mi = tl.program_id(1)
    m0 = mi * BM
    cols = tl.arange(0, C)
    cmask = (n0 + cols) < N
    acc = tl.zeros([C], dtype=tl.float32)
    for r in range(0, BM):
        m = m0 + r
        base = m * N + n0
        x = tl.load(x_ptr + base + cols, mask=cmask, other=0.0).to(tl.float32)
        dz = tl.load(dz_ptr + base + cols, mask=cmask, other=0.0).to(tl.float32)
        rsigma = tl.load(rsigma_ptr + m).to(tl.float32)
        acc += tl.where(cmask, dz * x * rsigma, 0.0)
    tl.store(dgamma_partial_ptr + mi * N + n0 + cols, acc, mask=cmask)


@triton.jit
def _te_rmsnorm_bwd_dgamma_reduce_kernel(
    dgamma_ptr,
    dgamma_partial_ptr,
    P,
    N,
    C: tl.constexpr,
):
    n0 = tl.program_id(0) * C
    cols = n0 + tl.arange(0, C)
    cmask = cols < N
    acc = tl.zeros([C], dtype=tl.float32)
    for i in range(0, P):
        acc += tl.load(dgamma_partial_ptr + i * N + cols, mask=cmask, other=0.0).to(
            tl.float32
        )
    tl.store(dgamma_ptr + cols, acc, mask=cmask)


@triton.jit
def _te_rmsnorm_bwd_dx_flat_kernel(
    dx_ptr,
    dz_ptr,
    x_ptr,
    rsigma_ptr,
    gamma_ptr,
    dgamma_partial_ptr,
    M,
    BLOCK: tl.constexpr,
    zero_centered_gamma: tl.constexpr,
):
    """N == 1 backward fast path: elementwise dx + partial dgamma."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dzv = tl.load(dz_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(rsigma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gamma_ptr).to(tl.float32)
    if zero_centered_gamma:
        g = g + 1.0
    xhat = xv * r
    dxv = r * dzv * g * (1.0 - xhat * xhat)
    tl.store(dx_ptr + offs, dxv, mask=mask)
    tl.store(dgamma_partial_ptr + pid, tl.sum(dzv * xhat))


@triton.jit
def _te_rmsnorm_bwd_dgamma_flat_reduce_kernel(
    dgamma_ptr,
    dgamma_partial_ptr,
    P,
    BLOCK: tl.constexpr,
):
    """Reduce the flat partial dgamma (N == 1) into the single dgamma[0]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < P
    s = tl.sum(tl.load(dgamma_partial_ptr + offs, mask=mask, other=0.0).to(tl.float32))
    tl.store(dgamma_ptr + pid, s.to(dgamma_ptr.dtype.element_ty))


def _dgamma_bm_size(M):
    block = min(M, _DGM_BM_MAX)
    while block > 1 and M % block != 0:
        block //= 2
    return max(1, block)


_DGM_BM_MAX = 128


def _ln_bwd_col_size(N):
    cap = min(N, 8192)
    return 1 << (cap.bit_length() - 1)


def _dx_col_size(N):
    if N <= 16:
        return 16
    return min(max(1 << (N - 1).bit_length(), 1024), 8192)


_FWD_TILE_N_MAX = 4096
_FWD_TILE_ELEMS = 65536
_FWD_ROW_BLOCK = 64 * 128


@triton.jit
def _te_rmsnorm_fwd_tile2d_kernel(
    Y,
    INV_RMS,
    X,
    W,
    eps: tl.constexpr,
    TILE_M: tl.constexpr,
    N: tl.constexpr,
):
    pid = tl.program_id(0)

    n_off = tl.arange(0, N)
    w = tl.load(W + n_off).to(tl.float32)

    m_off = pid * TILE_M + tl.arange(0, TILE_M)
    offs = m_off[:, None] * N + n_off[None, :]

    x = tl.load(X + offs).to(tl.float32)

    var = tl.sum(x * x, axis=1) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    y = (x * rrms[:, None] * w[None, :]).to(Y.dtype.element_ty)
    tl.store(Y + offs, y)
    tl.store(INV_RMS + m_off, rrms)


@triton.jit
def _te_rmsnorm_fwd_flat_kernel(
    Y,
    INV_RMS,
    X,
    W,
    M,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """N == 1 fast path: every element is its own row, so rmsnorm is elementwise.

    The per-row kernel would launch one (tiny) program per element, which is
    launch-bound for e.g. [10000, 1]; a flat BLOCK-lane kernel amortizes the
    launch cost the same way rms_norm_flat_kernel does.  Bound-masked so the
    last tile does not read/write past the end when M % BLOCK != 0.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    rrms = 1.0 / tl.sqrt(x * x + eps)
    w = tl.load(W)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + offs, y, mask=mask)
    tl.store(INV_RMS + offs, rrms, mask=mask)


@triton.jit
def _te_rmsnorm_fwd_row_kernel(
    Y,
    INV_RMS,
    X,
    W,
    N: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # N is constexpr so the unmasked (exact-multiple) loops below can use
    # tl.static_range: fully unrolling the column loop lets the scheduler hoist
    # all [BLOCK] loads ahead of the reduction and hide memory latency.
    # Measured on this XPU (do_bench warmup=1/rep=1, 2026-10-21): the unrolled
    # body is ~8-9% faster than the runtime trip-count loop on [10000, 65536]
    # (9.84ms -> 9.06ms) and [1024, 65536] (1.02ms -> 0.93ms).
    pid = tl.program_id(0)
    X += pid * N
    Y += pid * N

    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)
    if NEED_MASK:
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            sum_sq += x * x
    else:
        for off in tl.static_range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            x = tl.load(X + cols).to(tl.float32)
            sum_sq += x * x
    var = tl.sum(sum_sq, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    tl.store(INV_RMS + pid, rrms)

    if NEED_MASK:
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            y = (x * rrms * w).to(Y.dtype.element_ty)
            tl.store(Y + cols, y, mask=mask)
    else:
        for off in tl.static_range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            x = tl.load(X + cols).to(tl.float32)
            w = tl.load(W + cols).to(tl.float32)
            y = (x * rrms * w).to(Y.dtype.element_ty)
            tl.store(Y + cols, y)


def _fwd_tile_m(N, M):
    """TILE_M for the unmasked 2D tile kernel, or None if not applicable.

    Mirrors rms_norm_forward's tile selection (TILE_M=32 preference; the
    [TILE_M, N] fp32 tile must stay within _FWD_TILE_ELEMS) but refuses tiles
    wider than _FWD_TILE_N_MAX columns: the [8, 8192] tile that the rms_norm
    65536-element budget admits when N == 8192 crashes the XPU vectorizer.
    """
    if N > _FWD_TILE_N_MAX:
        return None
    if N <= 256:
        for cand in (32, 16):
            if M % cand == 0:
                return cand
        return None
    tm = 32
    while tm * N > _FWD_TILE_ELEMS:
        tm //= 2
    while tm >= 2:
        if M % tm == 0:
            return tm
        tm //= 2
    return None


def te_rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    ln_out: torch.Tensor = None,
    quantizer=None,
    otype: torch.dtype = None,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
):
    """TE-aligned RMSNorm forward (XPU backend local): y = x/sqrt(mean(x^2)+eps) * w,
    returns (y, None, rsigma).  See the module comment above for why the generic
    and rms_norm-reuse paths are not usable on this backend.
    """
    del sm_margin
    if zero_centered_gamma:
        weight = weight.to(torch.float32) + 1.0
    N = input.shape[-1]
    x = input.contiguous()
    M = x.numel() // N
    w = weight.contiguous()
    y = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=x.device)
    rsigma = torch.empty_strided((M,), (1,), dtype=torch.float32, device=x.device)

    with torch_device_fn.device(x.device):
        if N == 1:
            _te_rmsnorm_fwd_flat_kernel[(triton.cdiv(M, 4096),)](
                y, rsigma, x, w, M, eps, 4096
            )
        else:
            if not _tle_rmsnorm_fwd_launch(x, y, w, rsigma, M, N, eps):
                tile_m = _fwd_tile_m(N, M)
                if tile_m is not None:
                    _te_rmsnorm_fwd_tile2d_kernel[(M // tile_m,)](
                        y, rsigma, x, w, eps, tile_m, N
                    )
                else:
                    need_mask = (N % _FWD_ROW_BLOCK) != 0
                    _te_rmsnorm_fwd_row_kernel[(M,)](
                        y, rsigma, x, w, N, eps, _FWD_ROW_BLOCK, need_mask
                    )

    if otype is not None and y.dtype != otype:
        y = y.to(otype)
    if ln_out is not None:
        ln_out.copy_(y)
        return ln_out, None, rsigma
    return y, None, rsigma


def te_rmsnorm_bwd(
    dz: torch.Tensor,
    x: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
):
    del sm_margin
    original_shape = x.shape
    N = gamma.shape[0]
    x_2d = x.reshape(-1, N).contiguous()
    dz_2d = dz.reshape(-1, N).contiguous()
    rsigma = rsigma.contiguous()
    M = x_2d.shape[0]
    dx = torch.empty_like(x_2d)
    dgamma = torch.empty_like(gamma)

    bc = _ln_bwd_col_size(N)
    # cdx = _dx_col_size(N)

    with torch_device_fn.device(x.device):
        if N == 1:
            block = 4096
            num_p = triton.cdiv(M, block)
            dgamma_partial = torch.empty((num_p,), dtype=torch.float32, device=x.device)
            _te_rmsnorm_bwd_dx_flat_kernel[(num_p,)](
                dx,
                dz_2d,
                x_2d,
                rsigma,
                gamma,
                dgamma_partial,
                M,
                BLOCK=block,
                zero_centered_gamma=zero_centered_gamma,
                num_warps=4,
            )
            _te_rmsnorm_bwd_dgamma_flat_reduce_kernel[(1,)](
                dgamma, dgamma_partial, num_p, BLOCK=1024, num_warps=4
            )
        else:
            bm = _dgamma_bm_size(M)
            p = M // bm
            dgamma_partial = torch.empty((p, N), dtype=torch.float32, device=x.device)
            _te_rmsnorm_bwd_dx_1d_kernel[(M,)](
                dx,
                dz_2d,
                x_2d,
                gamma,
                rsigma,
                N,
                zero_centered_gamma=zero_centered_gamma,
                C=bc,
                NEED_TAIL=(N % bc != 0),
                num_warps=4,
                isCloseUnrollControl=True,
                isCloseVectorization=True,
            )
            _te_rmsnorm_bwd_dgamma_kernel[(triton.cdiv(N, bc), p)](
                dgamma_partial,
                dz_2d,
                x_2d,
                rsigma,
                M,
                N,
                BM=bm,
                C=bc,
                num_warps=4,
                isCloseUnrollControl=True,
            )
            _te_rmsnorm_bwd_dgamma_reduce_kernel[(triton.cdiv(N, bc),)](
                dgamma,
                dgamma_partial,
                p,
                N,
                C=bc,
                num_warps=4,
                isCloseUnrollControl=True,
            )

    return dx.reshape(original_shape), dgamma


def _patch_generic_wrapper():
    """Route direct calls to the generic wrapper (flag_gems.ops.te_rmsnorm
    module) to this backend override.

    Tests and benchmarks import ``te_rmsnorm_bwd`` from
    ``flag_gems.ops.te_rmsnorm`` (bypassing the top-level ``flag_gems``
    registry that SpecOpRegistrar patches), so the generic Triton kernel
    would still be hit on XPU (it cannot compile there: the 2D
    ``tl.sum(..., axis=0)`` reduction in ``rmsnorm_bwd_dgamma_kernel`` is
    rejected by the XPU legalizer with
    ``axis must not be 0 for 2D+ shapes``).  Patching the module
    attribute at import time keeps the change backend-local: the generic
    module source is untouched and other vendor backends are unaffected
    (this module is only imported for the kunlunxin backend).
    """
    try:
        import sys

        _generic_module = sys.modules.get("flag_gems.ops.te_rmsnorm")
        if _generic_module is not None:
            if hasattr(_generic_module, "te_rmsnorm_bwd"):
                _generic_module.te_rmsnorm_bwd = te_rmsnorm_bwd
            if hasattr(_generic_module, "te_rmsnorm_fwd"):
                _generic_module.te_rmsnorm_fwd = te_rmsnorm_fwd
        import flag_gems.ops as _ops

        if hasattr(_ops, "te_rmsnorm_bwd"):
            _ops.te_rmsnorm_bwd = te_rmsnorm_bwd
        if hasattr(_ops, "te_rmsnorm_fwd"):
            _ops.te_rmsnorm_fwd = te_rmsnorm_fwd
    except ImportError:
        pass


_patch_generic_wrapper()


__all__ = ["te_rmsnorm_bwd", "te_rmsnorm_fwd"]
