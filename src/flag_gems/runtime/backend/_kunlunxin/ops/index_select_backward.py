import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.index_select_backward import (
    index_select_backward as generic_index_select_backward,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

# One-hot GEMM guard.  Above these, the one-hot build (L * D elements) and the
# O(M*L*D) GEMM both blow up; the generic scatter path takes over.
_MAX_ONE_HOT_ELEMENTS = 20_000_000
_MAX_ONE_HOT_DIM = 8192

# Dedicated masked-exact GEMM for the small shapes (M, D <= 512).
#
# The general mm() path is only fast on tile-aligned shapes: ragged shapes went
# through the old _padded_mm + mm_kernel, which materialised host-side C/K
# padding (torch.zeros + _copy_from + mm + slice) -- several launches, measured
# 94~130us on (64, 64) where the vendor needs 10~17us.  The dedicated kernel
# instead emulates the addmv pattern (masks + other=, NO max_contiguous /
# multiple_of hints: those are exactly what made the mm variants lie about
# contiguity on partial tiles and fault, and any *computed* dot operand is
# rejected by the backend's mma lowering) on exact-size buffers: one launch,
# zero padding, zero copies.  Measured ~30us for the (64, 64, 72) GEMM itself.
#
# It is only used when every dim is <= 512: on the large 4096^2 shapes the
# masked loads fall back to narrow loads and lose to the aligned mm_kernel
# (1.19ms vs 0.69ms), and on the 32768x512x520 wide-short shapes it is a
# wash (0.47ms) - those stay on the padded-exact _isb_gemm_large.
_ISB_GEMM_BM = 128
_ISB_GEMM_BN = 128
_ISB_GEMM_BK = 128
_ISB_GEMM_GROUP_M = 8
_ISB_GEMM_WARPS = 4


@triton.jit
def _isb_gemm_kernel(
    X,
    Y,
    C,
    M,
    N,
    K,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # C(M, N) = X(M, K) @ Y(K, N).  Masks (with other=) handle every ragged
    # edge exactly, so X/Y/C are the exact shapes - no host padding, no
    # copy-back.  Runtime strides: X or Y may be a transpose view of the
    # one-hot (dim == 0), the kernel is layout-agnostic.
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = rm < M
    n_mask = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        rk = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = rk < K
        a = tl.load(
            X + rm[:, None] * stride_x0 + rk[None, :] * stride_x1,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            Y + rk[:, None] * stride_y0 + rn[None, :] * stride_y1,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
    tl.store(
        C + rm[:, None] * N + rn[None, :],
        acc.to(C.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _isb_gemm_small(x, y, M, K, N, device):
    """Exact-shape C(M, N) = x(M, K) @ y(K, N); single launch, no padding."""
    c = torch.empty((M, N), dtype=x.dtype, device=device)
    grid = (triton.cdiv(M, _ISB_GEMM_BM) * triton.cdiv(N, _ISB_GEMM_BN),)
    with torch_device_fn.device(device):
        _isb_gemm_kernel[grid](
            x,
            y,
            c,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=_ISB_GEMM_BM,
            BLOCK_N=_ISB_GEMM_BN,
            BLOCK_K=_ISB_GEMM_BK,
            GROUP_M=_ISB_GEMM_GROUP_M,
            num_warps=_ISB_GEMM_WARPS,
        )
    return c


def _pad_shape(sz, blk):
    return (sz + blk - 1) // blk * blk


# Dedicated aligned GEMM for the large one-hot shapes (M > 512 or D > 512).
#
# The generic mm() path is only fast on tile-aligned shapes; the ragged
# (4096, 4096, 4104) / wide-short (32768, 512, 520) shapes go through
# _padded_or_direct, which materialises host-side C/K padding and then a
# 33MB `c[:M, :N].contiguous()` copy-back -- several extra launches, measured
# 0.93ms / 0.37ms on the benchmark shapes.  This kernel instead runs on
# exact-size-padded buffers (the one-hot is built directly on the padded
# (Kp, Np) grid) with NO masks and NO contiguity hints: the former are what
# slowed the masked _isb_gemm_kernel to 1.19ms on 4096^2 (narrow-load
# fallback), the latter are exactly what the backend's mma lowering rejects
# for *computed* operands ("contiguity-hint assertion", measured
# CompilationError on every explicit-BLOCK mm variant).  One launch, no
# copy-back: the (M, N) result is a strided view of the padded output and
# _materialize disposes of it (zero-copy when already exact).
#
# Tile heuristics (locally calibrated, mm()-agnostic so mm.py can change
# freely): 256^3/w8 is the floor for M, N > 512; 128-tiles for small dims
# avoid over-padding (a 72-col output would otherwise waste 3.5x); BK=128
# for K <= 512 keeps the K-pad at 2x instead of 4x on the wide-short shapes.
_ISB_LARGE_WARPS = 8
_ISB_LARGE_GROUP_M = 8


def _isb_large_tiles(M, K, N):
    bm = 256 if M > 512 else 128
    bn = 256 if N > 512 else 128
    bk = 128 if K <= 512 else 256
    return bm, bn, bk


@triton.jit
def _isb_gemm_large_kernel(
    X,
    Y,
    C,
    M,
    N,
    K,
    M_out,
    N_out,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_c0,
    stride_c1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # C(M_out, N_out) = X(M, K) @ Y(K, N).  X/Y are on the padded exact grid
    # (Mp, Kp)x(Kp, Np) so the *loads* need no mask (the pad rows/cols are
    # zero).  The *store* targets an exact-shape contiguous C(M_out, N_out) with
    # an edge mask, so no column padding leaks into the output -- the caller can
    # reshape C to the n-D output for free (no strided de-pad copy, no aten).
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        rk = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            X + rm[:, None] * stride_x0 + rk[None, :] * stride_x1,
        )
        b = tl.load(
            Y + rk[:, None] * stride_y0 + rn[None, :] * stride_y1,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
    tl.store(
        C + rm[:, None] * stride_c0 + rn[None, :] * stride_c1,
        acc.to(C.dtype.element_ty),
        mask=(rm < M_out)[:, None] & (rn < N_out)[None, :],
    )


def _isb_large_pads(M, K, N):
    bm, bn, bk = _isb_large_tiles(M, K, N)
    return _pad_shape(M, bm), _pad_shape(K, bk), _pad_shape(N, bn)


def _isb_gemm_large(x, y, M, K, N, Mp, Kp, Np, device):
    """Padded-exact GEMM on the (Mp, Kp) x (Kp, Np) grid; single launch.

    ``x`` (M, K) / ``y`` (K, N), or either already on the padded grid (the
    one-hot is built directly on (Kp, Np) by the caller, so it is passed
    through untouched; ``x`` is padded only when ragged).  Loads run unmasked
    on the padded grid, but the store targets an **exact-shape contiguous**
    (M, N) buffer (edge-masked), so the result never carries column padding.
    The caller can therefore reshape it to the n-D output for free -- no
    strided de-pad copy (the ~26ms tall-narrow _isb_copy) and no aten.
    """
    bm, bn, bk = _isb_large_tiles(M, K, N)
    if x.shape == (Mp, Kp):
        xp = x
    else:
        xp = _pad_a(x, M, K, Mp, Kp, device)
    if y.shape == (Kp, Np):
        yp = y
    else:
        yp = _pad_a(y, K, N, Kp, Np, device)
    c = torch.empty((M, N), dtype=x.dtype, device=device)
    grid = (triton.cdiv(Mp, bm) * triton.cdiv(Np, bn),)
    with torch_device_fn.device(device):
        _isb_gemm_large_kernel[grid](
            xp,
            yp,
            c,
            Mp,
            Np,
            Kp,
            M,
            N,
            xp.stride(0),
            xp.stride(1),
            yp.stride(0),
            yp.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_M=_ISB_LARGE_GROUP_M,
            num_warps=_ISB_LARGE_WARPS,
        )
    return c


@triton.jit
def _isb_one_hot_scatter_kernel(Oh, Index, L, S0, BLOCK: tl.constexpr):
    # Dense 1D scatter: program i writes oh[i, index[i]] = 1.0 (oh pre-zeroed).
    # One launch over L elements; replaces oh.scatter_ (an aten op) with a
    # pure-triton kernel.  Duplicate index values are benign (all stores 1.0).
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < L
    idx = tl.load(Index + offs, mask=m, other=0)
    tl.store(Oh + offs * S0 + idx, 1.0, mask=m)


def _one_hot_write(oh, index, index_len, dim_size_out, device):
    """Write 1.0 at oh[i, index[i]] for i < index_len (oh is pre-zeroed).

    Pure-triton dense scatter (see _isb_one_hot_scatter_kernel); no aten.
    """
    grid = (triton.cdiv(index_len, 1024),)
    with torch_device_fn.device(device):
        _isb_one_hot_scatter_kernel[grid](oh, index, index_len, oh.stride(0), 1024)


@triton.jit
def _isb_copy_kernel(
    Src,
    Dst,
    M,
    N,
    s0,
    s1,
    d0,
    d1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Generic strided 2D copy Dst[:M, :N] <- Src[:M, :N] (masked edges).
    # Replaces torch.ops.aten._copy_from in the zero-pad / materialise paths.
    pid = ext.program_id(0)
    gn = tl.cdiv(N, BLOCK_N)
    rm = (pid // gn) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = (pid % gn) * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = (rm < M)[:, None] & (rn < N)[None, :]
    v = tl.load(Src + rm[:, None] * s0 + rn[None, :] * s1, mask=mm, other=0.0)
    tl.store(Dst + rm[:, None] * d0 + rn[None, :] * d1, v, mask=mm)


_ISB_COPY_BM = 128
_ISB_COPY_BN = 128
_ISB_COPY_WARPS = 4


def _isb_copy(src, dst, M, N, device, prefer_tle=True):
    """Copy src[:M, :N] -> dst[:M, :N].

    First offer the sub-views to ``tle_copy``: a strided 2-D copy on this
    backend is a native DMA shape (rows of a contiguous run with independent
    src/dst row strides = ``memcpy_2d_sdnn``), and the tle.dsa path runs it on
    the SDNN engine (~native), whereas a Triton element-wise/masked copy is the
    exact anti-pattern the copy-family skill warns about -- it collapses to
    ~2-3 GB/s on tall-narrow / column-padded shapes (the 26ms de-pad).  The
    Triton ``_isb_copy_kernel`` stays as the fallback for what tle cannot
    express (odd dtypes, non-unit inner stride on both sides, rank > 5).
    """
    s = src[:M, :N]
    d = dst[:M, :N]
    if prefer_tle and tle_copy(s, d):
        return
    grid = (triton.cdiv(M, _ISB_COPY_BM) * triton.cdiv(N, _ISB_COPY_BN),)
    with torch_device_fn.device(device):
        _isb_copy_kernel[grid](
            src,
            dst,
            M,
            N,
            src.stride(0),
            src.stride(1),
            dst.stride(0),
            dst.stride(1),
            BLOCK_M=_ISB_COPY_BM,
            BLOCK_N=_ISB_COPY_BN,
            num_warps=_ISB_COPY_WARPS,
        )


def _pad_a(a, M, K, Mp, Kp, device):
    """Row-major (M, K) -> zero-padded (Mp, Kp) via a masked triton copy.

    The aligned mm kernel needs exact multiples; padding only materialises a
    buffer when the shape is ragged.  The pad region stays zero (dst is
    pre-zeroed, the masked store only touches [0:M, 0:K]).  No aten.
    """
    if (Mp, Kp) == (M, K) and (a.stride(0), a.stride(1)) == (K, 1):
        return a
    ap = torch.zeros((Mp, Kp), dtype=a.dtype, device=device)
    _isb_copy(a, ap, M, K, device)
    return ap


def _materialize(res, self_sizes, dtype, device):
    """(R0, R1) GEMM result -> row-major n-D output (a free reshape).

    ``res`` is the exact-shape **contiguous** GEMM output (the large kernel now
    edge-masks its store into an (M, N) buffer, the small kernel already does),
    and it shares the flat ordering of the n-D output (the GEMM runs on the
    flattened (M, L)/(L, M) view).  So the n-D output is just ``res.reshape``
    -- a zero-copy view.  No strided de-pad copy, no aten.
    """
    if res.shape == torch.Size(self_sizes):
        return res
    return res.reshape(self_sizes)


def _isb_last(grad, self_sizes, dtype, device, index, index_len, dim_size_out):
    """dim == ndim - 1: out[..., k] = sum_i grad[..., i] * (index[i] == k)."""
    M = grad.numel() // index_len
    grad_flat = grad.reshape(M, index_len)
    Mp, Kp, Np = _isb_large_pads(M, index_len, dim_size_out)
    oh = torch.zeros((Kp, Np), dtype=dtype, device=device)
    _one_hot_write(oh, index, index_len, dim_size_out, device)
    out2d = _isb_gemm_large(
        grad_flat, oh, M, index_len, dim_size_out, Mp, Kp, Np, device
    )
    res = out2d[:M, :dim_size_out]
    if grad.ndim == 2:
        return res
    # out2d is exact-shape contiguous (M, D); (x0..xk, D) is that flat order,
    # so the n-D output is a free reshape (no strided de-pad copy, no aten).
    return _materialize(res, self_sizes, dtype, device)


def _isb_first(grad, self_sizes, dtype, device, index, index_len, dim_size_out):
    """dim == 0: out[k, ...] = sum_i grad[i, ...] * (index[i] == k)."""
    M = grad.numel() // index_len
    grad_flat = grad.reshape(index_len, M)
    Dp, Kp, Mp = _isb_large_pads(dim_size_out, index_len, M)
    oh = torch.zeros((Kp, Dp), dtype=dtype, device=device)
    _one_hot_write(oh, index, index_len, dim_size_out, device)
    # The transposed one-hot (Dp, Kp) feeds the GEMM as a runtime-strided view:
    # _isb_gemm_large_kernel is layout-agnostic and _isb_gemm_large's exact-shape
    # check matches (Mp==Dp, Kp==Kp), so no materialisation (no aten) is needed.
    oh_t = oh.t()
    out2d = _isb_gemm_large(
        oh_t, grad_flat, dim_size_out, index_len, M, Dp, Kp, Mp, device
    )
    res = out2d[:dim_size_out, :M]
    if grad.ndim == 2:
        return res
    return _materialize(res, self_sizes, dtype, device)


# Batched GEMM for the mid-dim case: out[a] (D, B) = one_hot^T (D, L) @ grad[a]
# (L, B), batched over A = prod(sizes before dim).  This replaces the old
# dim_compress + 2-D GEMM + transpose-permute-back (the permute needed either
# a ~20ms Triton transpose or torch.ops.aten._copy_from); computing in the
# natural (A, L, B) -> (A, D, B) layout writes the result straight into
# row-major order, so no permute and no aten are needed.  Masked edges (with
# other=), so grad/one-hot are the exact shapes -- no host padding.
_ISB_BMM_BD = 128
_ISB_BMM_BB = 128
_ISB_BMM_BL = 64
_ISB_BMM_GROUP_D = 8
_ISB_BMM_WARPS = 4


@triton.jit
def _isb_bmm_kernel(
    OhT,
    Grad,
    C,
    A,
    D,
    L,
    B,
    s_ot0,
    s_ot1,
    s_g0,
    s_g1,
    s_g2,
    s_c0,
    s_c1,
    s_c2,
    BLOCK_D: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_L: tl.constexpr,
    GROUP_D: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_d = tl.cdiv(D, BLOCK_D)
    grid_b = tl.cdiv(B, BLOCK_B)
    per_batch = grid_d * grid_b
    a = pid // per_batch
    pid_in = pid % per_batch
    width = GROUP_D * grid_b
    group_id = pid_in // width
    group_size = min(grid_d - group_id * GROUP_D, GROUP_D)
    pid_d = group_id * GROUP_D + (pid_in % group_size)
    pid_b = (pid_in % width) // group_size

    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    d_mask = rd < D
    b_mask = rb < B
    g_base = Grad + a * s_g0
    acc = tl.zeros((BLOCK_D, BLOCK_B), dtype=tl.float32)
    for k in range(0, tl.cdiv(L, BLOCK_L)):
        rl = k * BLOCK_L + tl.arange(0, BLOCK_L)
        l_mask = rl < L
        a_tile = tl.load(
            OhT + rd[:, None] * s_ot0 + rl[None, :] * s_ot1,
            mask=d_mask[:, None] & l_mask[None, :],
            other=0.0,
        )
        b_tile = tl.load(
            g_base + rl[:, None] * s_g1 + rb[None, :] * s_g2,
            mask=l_mask[:, None] & b_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32, allow_tf32=False)
    tl.store(
        C + a * s_c0 + rd[:, None] * s_c1 + rb[None, :] * s_c2,
        acc.to(C.dtype.element_ty),
        mask=d_mask[:, None] & b_mask[None, :],
    )


def _isb_bmm(oh_t, grad3d, A, D, L, B, dtype, device):
    """Batched C[a] (D, B) = oh_t (D, L) @ grad3d[a] (L, B); single launch."""
    out = torch.empty((A, D, B), dtype=dtype, device=device)
    grid = (A * triton.cdiv(D, _ISB_BMM_BD) * triton.cdiv(B, _ISB_BMM_BB),)
    with torch_device_fn.device(device):
        _isb_bmm_kernel[grid](
            oh_t,
            grad3d,
            out,
            A,
            D,
            L,
            B,
            oh_t.stride(0),
            oh_t.stride(1),
            grad3d.stride(0),
            grad3d.stride(1),
            grad3d.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_D=_ISB_BMM_BD,
            BLOCK_B=_ISB_BMM_BB,
            BLOCK_L=_ISB_BMM_BL,
            GROUP_D=_ISB_BMM_GROUP_D,
            num_warps=_ISB_BMM_WARPS,
        )
    return out


def _isb_mid(grad, self_sizes, dim, dtype, device, index, index_len, dim_size_out):
    """Mid-dim: batched GEMM out[a] = one_hot^T @ grad[a] in natural layout.

    grad is (A, L, B) with A=prod(sizes[:dim]), B=prod(sizes[dim+1:]); the
    output (A, D, B) is exactly self_sizes flattened, so no dim_compress and
    no transpose/permute (no aten) are needed -- the batched kernel writes the
    result straight into row-major order.
    """
    A = math.prod(self_sizes[:dim])
    B = math.prod(self_sizes[dim + 1 :])
    L = index_len
    D = dim_size_out
    grad3d = grad.reshape(A, L, B)
    oh = torch.zeros((L, D), dtype=dtype, device=device)
    _one_hot_write(oh, index, L, D, device)
    out = _isb_bmm(oh.t(), grad3d, A, D, L, B, dtype, device)
    return out.reshape(self_sizes)


def _isb_small_last(grad, self_sizes, dtype, device, index, index_len, dim_size_out):
    """dim == ndim - 1, exact-shape dedicated kernel (M, D <= 512)."""
    M = grad.numel() // index_len
    grad_flat = grad.reshape(M, index_len)
    oh = torch.zeros((index_len, dim_size_out), dtype=dtype, device=device)
    _one_hot_write(oh, index, index_len, dim_size_out, device)
    res = _isb_gemm_small(grad_flat, oh, M, index_len, dim_size_out, device)
    return _materialize(res, self_sizes, dtype, device)


def _isb_small_first(grad, self_sizes, dtype, device, index, index_len, dim_size_out):
    """dim == 0, exact-shape dedicated kernel (D, M <= 512).

    The one-hot (L, D) feeds the GEMM as the transposed operand: the kernel
    takes runtime strides (1, D), so no oh_t materialisation is needed.
    """
    M = grad.numel() // index_len
    grad_flat = grad.reshape(index_len, M)
    oh = torch.zeros((index_len, dim_size_out), dtype=dtype, device=device)
    _one_hot_write(oh, index, index_len, dim_size_out, device)
    res = _isb_gemm_small(oh.t(), grad_flat, dim_size_out, index_len, M, device)
    return _materialize(res, self_sizes, dtype, device)


def _isb_small_mid(
    grad, self_sizes, dim, dtype, device, index, index_len, dim_size_out
):
    """Mid-dim (small): same batched-GEMM path as _isb_mid (masks cover small)."""
    return _isb_mid(
        grad, self_sizes, dim, dtype, device, index, index_len, dim_size_out
    )


def index_select_backward(grad, self_sizes, dim, index):
    logger.debug("GEMS_KUNLUNXIN INDEX_SELECT_BACKWARD")

    dim = dim % grad.ndim
    index_len = index.numel()
    dim_size_out = self_sizes[dim]
    one_hot_elements = index_len * dim_size_out

    if grad.numel() == 0:
        return torch.zeros(self_sizes, dtype=grad.dtype, device=grad.device)
    if (
        index_len == 0
        or one_hot_elements > _MAX_ONE_HOT_ELEMENTS
        or index_len > _MAX_ONE_HOT_DIM
        or dim_size_out > _MAX_ONE_HOT_DIM
        or grad.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return generic_index_select_backward(grad, self_sizes, dim, index)

    index = index.to(torch.int64)
    dtype = grad.dtype
    device = grad.device

    M = grad.numel() // index_len
    # Dedicated exact-shape kernel: single launch, no padding; beats the
    # padded mm() path on every small shape (94~130us -> ~45us on 64x64).
    # Large shapes stay on _isb_gemm_large (the aligned 0.69ms floor is
    # the limit; the masked kernel loses there: 1.19ms).
    small = M <= 512 and dim_size_out <= 512

    if dim == grad.ndim - 1:
        if small:
            return _isb_small_last(
                grad, self_sizes, dtype, device, index, index_len, dim_size_out
            )
        return _isb_last(
            grad, self_sizes, dtype, device, index, index_len, dim_size_out
        )
    if dim == 0:
        if small:
            return _isb_small_first(
                grad, self_sizes, dtype, device, index, index_len, dim_size_out
            )
        return _isb_first(
            grad, self_sizes, dtype, device, index, index_len, dim_size_out
        )
    if small:
        return _isb_small_mid(
            grad, self_sizes, dim, dtype, device, index, index_len, dim_size_out
        )
    return _isb_mid(
        grad, self_sizes, dim, dtype, device, index, index_len, dim_size_out
    )
