import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def safe_softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        # Pre-offset the base pointers so the inner `ptr + n_offsets` access is a
        # scalar-base + stride-1 arange that OffsetAnalysis proves contiguous
        # (block DMA). The old inline `pid_m * N + n_offsets` addressing blocked
        # the analysis -> discrete scalar gather (~1-3 GB/s, e.g. [4096,4096] took
        # ~37ms). Pre-offsetting drops it to ~1.1ms (~35x). Same fix as softmax.py.
        input_ptr += pid_m * N
        output_ptr += pid_m * N
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        m = tl.max(inp, 0)
        # a whole row of -inf -> softmax must be 0, not nan
        all_neg_inf = m == float("-inf")
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = e / z
        out = tl.where(all_neg_inf, 0.0, out).to(output_ptr.dtype.element_ty)
        tl.store(output_ptr + n_offsets, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets).to(tl.float32)
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
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
        row_all_neg_inf = m_reduced == float("-inf")
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        # Normalize pass. Iterate ASCENDING so each `input_ptr + n_offsets` load
        # and `output_ptr + n_offsets` store is a scalar-base + stride-1 arange
        # (block DMA). The old code walked the tiles DESCENDING
        # (`previous_multiple - start_n`, with evict_first) as a cache-locality
        # trick, but on this XPU the backward walk defeats OffsetAnalysis/prefetch
        # -> discrete access (~1-3 GB/s). Ascending drops it ~35x. Same fix as
        # softmax.py.
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets).to(tl.float32)
            o = tl.exp(inp - m) / z
            o = tl.where(row_all_neg_inf, 0.0, o).to(output_ptr.dtype.element_ty)
            tl.store(output_ptr + n_offsets, o)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            o = tl.exp(inp - m) / z
            o = tl.where(row_all_neg_inf, 0.0, o).to(output_ptr.dtype.element_ty)
            tl.store(output_ptr + n_offsets, o, mask=mask)


# ------------------------  forward: XPU multirow 2D fast path (N <= 4096) --
#
# One program per [TILE_M, N] tile (TILE_M rows x N columns, lanes <= 8192).
# The whole tile is one contiguous region -> block DMA, and packing several
# rows per program removes the per-row launch overhead of the grid=(M,)
# fallback. Mirrors the validated softmax forward multirow design from
# #275 (table identical; on XPU: [1024,256] 0.87ms -> 0.02ms measured there).
#
# Correctness guards (XPU backend quirks, see HARNESS_SUMMARY):
#   * bf16 + non-power-of-2 N fails to lower (ConvertTritonXPUToLLVM) in the
#     wide-tile store -> bf16 restricted to pow2 N here; other cases fall back
#     to the per-row kernel.
#   * masked rows (M % TILE_M != 0) use the old per-row kernel so the
#     masked-load "other" semantics never appear in the multirow tile.
#   * TILE_M * N <= 8192 keeps tl.sum inside the XPU exact window (the
#     16384..32767 band miscompiles without buffer_size_limit).
#   * Compute always goes through fp32 (matches the per-row kernel and the
#     reference precision path: half -> fp32 -> target).
#   * Rows of all -inf must yield 0 (the "safe" part of safe_softmax), which
#     the per-row kernel implements via `row_all_neg_inf`; the multirow kernel
#     applies the same per-row where outside any reduction.

_SF_MR_MAX_N = 4096  # largest N handled by the 2D multirow tile
# tile_m picked so lane budget = TILE_M * N lands on a safe tl.sum window:
# <= 8192 lanes at all times, or 32768 lanes when launched with
# buffer_size_limit=2048 (validated: [4096,4096] tm=8 bsl=2048 is 2.1x faster
# than tm=2 and matches the per-row reference to dtype precision).
_SF_N_TILE_M = [
    (16, 64),
    (64, 32),
    (256, 16),
    (512, 16),
    (1024, 8),
    (2048, 4),
    (4096, 8),
]


@triton.jit
def safe_softmax_kernel_multirow(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    # Single-pass [TILE_M, N] tile (unmasked: M % TILE_M == 0 checked on host).
    pid = tl.program_id(0)
    mo = pid * TILE_M + tl.arange(0, TILE_M)
    no = tl.arange(0, N)
    off = mo[:, None] * N + no[None, :]
    inp = tl.load(input_ptr + off).to(tl.float32)
    m = tl.max(inp, 1)  # [TILE_M]
    all_neg_inf = m == float("-inf")
    e = tl.exp(inp - m[:, None])
    z = tl.sum(e, 1)  # [TILE_M]
    out = e / z[:, None]
    out = tl.where(all_neg_inf[:, None], 0.0, out)
    tl.store(output_ptr + off, out.to(output_ptr.dtype.element_ty))


_COPY_BLOCK = 1024


@triton.jit
def _strided_copy_3d(dst_ptr, src_ptr, d0, d1, d2, s0, s1, s2, BLOCK: tl.constexpr):
    """Copy a strided 3D source into a contiguous 3D destination (same shape).

    Only used as a fallback when `tle_copy` cannot express the layout (e.g. the
    tle.dsa surface is unavailable); the fast path is tle_copy's on-chip
    transpose. This is a real XPU kernel (no forbidden aten copy primitive).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = d0 * d1 * d2
    mask = offs < total
    i0 = offs // (d1 * d2)
    rem = offs % (d1 * d2)
    i1 = rem // d2
    i2 = rem % d2
    src_off = i0 * s0 + i1 * s1 + i2 * s2
    v = tl.load(src_ptr + src_off, mask=mask)
    tl.store(dst_ptr + offs, v, mask=mask)


def _copy_strided(dst, src):
    """Materialize `src` (strided 3D) into contiguous `dst` (same shape/dtype).

    Real XPU copy only: tle_copy's dsa-transpose path first, then a plain
    Triton gather copy. No forbidden aten copy primitive is used.
    """
    if tle_copy(src, dst):
        return
    d0, d1, d2 = src.shape
    s0, s1, s2 = src.stride()
    grid = (triton.cdiv(src.numel(), _COPY_BLOCK),)
    _strided_copy_3d[grid](dst, src, d0, d1, d2, s0, s1, s2, BLOCK=_COPY_BLOCK)


def _safe_softmax_launch(output, inp, M, N):
    """Inner launch on a contiguous [M, N] view (reduced dim innermost)."""
    # The 2D multirow tile uses `tl.arange(0, N)`; on this XPU non-power-of-2
    # lanes (e.g. N=65/127/129/255/513...) compile but silently miscompute
    # (same family as the log_softmax non-pow2 lane bug). Restrict the multirow
    # path to power-of-2 N for all dtypes (bf16 had this guard already); all
    # other shapes use the per-row kernel whose masked single-tile path is
    # verified correct for arbitrary N.
    use_multirow = N <= _SF_MR_MAX_N and ((N & (N - 1)) == 0)
    tile_m = 4
    for n_hi, tm in _SF_N_TILE_M:
        if N <= n_hi:
            tile_m = tm
            break
    if use_multirow and M % tile_m == 0:
        grid = (M // tile_m,)
        kw = dict(N=N, TILE_M=tile_m, num_warps=4)
        # Tiles above the 8192-lane safe window require the explicit
        # buffer_size_limit (see the table comment above).
        if tile_m * N > 8192:
            kw["buffer_size_limit"] = 2048
        safe_softmax_kernel_multirow[grid](output, inp, M, **kw)
        return
    # Fall back to the per-row kernel (grid=(M,), TILE_N heuristic).
    grid = (M, 1, 1)
    safe_softmax_kernel_inner[grid](
        output,
        inp,
        M,
        N,
        buffer_size_limit=2048,
        is_use_mask_zero=True,
    )


def _safe_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = None):
    logger.debug("GEMS_KUNLUNXIN _SAFE_SOFTMAX")
    assert x.ndim >= 1, "Input tensor must have at least 1 dimension"

    dim = dim % x.ndim
    M = 1
    N = x.shape[dim]
    for i in range(dim):
        M *= x.shape[i]

    x = x.contiguous()
    out_dtype = dtype if dtype is not None else x.dtype

    # The XPU triton kernel mishandles mixing two distinct 16-bit element types
    # (e.g. fp16 in / bf16 out) in a single launch. Use an fp32 intermediate
    # whenever the target dtype differs from the input dtype; this also matches
    # the reference precision path (half -> fp32 -> target).
    if out_dtype == x.dtype:
        kern_dtype = out_dtype
        need_cast = False
    else:
        kern_dtype = torch.float32
        need_cast = out_dtype != torch.float32

    if x.numel() == 0:
        return torch.empty_like(x, dtype=out_dtype)

    K = x.numel() // M // N  # post_dim

    with torch_device_fn.device(x.device):
        if K > 1:
            # Rearrange [M, N, K] -> [M*K, N] so the reduced dim N is innermost
            # (the only fast axis on this XPU). The transpose is materialized
            # with a real XPU copy kernel (tle_copy dsa-transpose path).
            inp_view = x.view(M, N, K).transpose(1, 2)  # [M, K, N] strided
            inp_reshaped = torch.empty((M, K, N), dtype=x.dtype, device=x.device)
            _copy_strided(inp_reshaped, inp_view)

            out_reshaped = torch.empty((M * K, N), dtype=kern_dtype, device=x.device)
            _safe_softmax_launch(out_reshaped, inp_reshaped.view(M * K, N), M * K, N)

            # [M*K, N] -> [M, N, K] (strided view) then materialize contiguous,
            # mirroring the ATen reference which returns a contiguous tensor.
            out = out_reshaped.view(M, K, N).transpose(1, 2)  # [M, N, K] strided
            final = torch.empty((M, N, K), dtype=kern_dtype, device=x.device)
            _copy_strided(final, out)
            out = final.reshape(x.shape)
        else:
            out = torch.empty_like(x, dtype=kern_dtype)
            _safe_softmax_launch(out, x, M, N)

    if need_cast:
        out = out.to(out_dtype)
    return out
