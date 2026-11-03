import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

from ..utils.tle_copy import tle_copy
from .copy import copy_ as _vendor_copy_
from .logsumexp import (
    _LSE_TLE_AVAILABLE,
    _LSE_TLE_TL_DTYPE,
    _MULTIROW_MAX_N,
    _reduce_inner,
    _tle_lse_geom,
    _tle_lse_row_kernel,
)

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:  # pragma: no cover - tle is unavailable without it
    TensorDescriptor = None

logger = logging.getLogger(__name__)

# The tle streaming row kernel is gated to N >= 4096 in `logsumexp`; measured
# on this matrix it also wins for fp32 at N == 1024 (one [XBLOCK, N] tile per
# program, no per-chunk re-reduce) while the 16-bit dtypes stay clearly faster
# on the single-tile multirow kernel there. Reuse the very same kernel and
# geometry formula, latching off on a toolchain compile failure like the
# shared helper does.
_SLSE_TLE_MIN_N = 1024  # fp32 only; 16-bit keeps the shared N >= 4096 gate
_slse_tle_broken = False


def _dma_copy(src, dst):
    if not tle_copy(src, dst):
        _vendor_copy_(dst, src)


def _tle_row_reduce(inp, out, rows, N, min_n):
    """Stream the row through the tle cluster-DMA kernel (see logsumexp)."""
    global _slse_tle_broken
    if not _LSE_TLE_AVAILABLE or _slse_tle_broken or N < min_n:
        return False
    if inp.dtype not in _LSE_TLE_TL_DTYPE or out.dtype not in _LSE_TLE_TL_DTYPE:
        return False
    if not inp.is_contiguous() or not out.is_contiguous():
        return False
    xblock, yblock, num_stages, need_zero = _tle_lse_geom(rows, N, inp.element_size())
    a = inp if inp.ndim == 2 and inp.shape[0] == rows else inp.view(rows, N)
    c = out if out.ndim == 1 else out.view(rows)
    grid = (triton.cdiv(rows, xblock),)
    try:
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
    except Exception:  # pragma: no cover - toolchain cannot compile the payload
        _slse_tle_broken = True
        return False
    return True


def _reduce_inner_known_n(inp, rows, N):
    """Extend the tle row kernel to fp32 N == 1024, else defer to `logsumexp`.

    At N == 1024 the single-tile multirow kernel is the measured bottleneck for
    fp32 (0.038 ms vs 0.026 ms native, i.e. 0.72x on the benchmark matrix),
    while the tle streaming row kernel with one [64, 1024] tile per program
    runs it in 0.032 ms. The 16-bit dtypes are clearly faster on the multirow
    tile there, so they keep the shared N >= 4096 gate. Everything else falls
    through to the shared `_reduce_inner` unchanged.
    """
    min_n = _SLSE_TLE_MIN_N if inp.dtype == torch.float32 else 4096
    if N >= min_n:
        out = torch.empty((rows,), dtype=inp.dtype, device=inp.device)
        if _tle_row_reduce(inp, out, rows, N, min_n):
            return out
    return _reduce_inner(inp, rows, N)


def _reduce_inner_any_n(inp, rows, N):
    if N <= _MULTIROW_MAX_N and (N & (N - 1)) != 0:
        P = triton.next_power_of_2(N)
        padded = torch.full(
            (rows, P), float("-inf"), dtype=inp.dtype, device=inp.device
        )
        _dma_copy(inp.reshape(rows, N), torch.as_strided(padded, (rows, N), (P, 1)))
        return _reduce_inner_known_n(padded, rows, P)
    return _reduce_inner_known_n(inp, rows, N)


def _single_dim_reduce(inp, dim, keepdim):
    n = inp.ndim
    N = inp.shape[dim]
    M = 1
    for i in range(dim):
        M *= inp.shape[i]
    K = 1
    for i in range(dim + 1, n):
        K *= inp.shape[i]

    if K == 1:
        inp = inp.contiguous()
        shape = list(inp.shape)
        shape[dim] = 1
        with torch_device_fn.device(inp.device):
            out = _reduce_inner_any_n(inp, M, N).view(shape)
        if not keepdim:
            out = out.squeeze(dim)
        return out

    perm = [i for i in range(n) if i != dim] + [dim]
    src = inp.permute(perm)
    buf = torch.empty(src.shape, dtype=inp.dtype, device=inp.device)

    _dma_copy(src, buf)

    res = _reduce_inner_any_n(buf.reshape(M * K, N), M * K, N).view(M, K)
    if keepdim:
        shape = list(inp.shape)
        shape[dim] = 1
        return res.view(shape)
    # Drop the reduced dim: `res` is the contiguous [prod(shape[:dim]),
    # prod(shape[dim+1:])] matrix, so this is a free view.
    shape = list(inp.shape)
    del shape[dim]
    return res.view(shape)


def special_logsumexp(inp, dim, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOGSUMEXP")

    if isinstance(dim, (list, tuple)):
        if len(dim) == 0:
            dim = list(range(inp.ndim))

        out = inp
        for d in dim:
            out = _single_dim_reduce(out, d % inp.ndim, True)
        if not keepdim:
            dset = {d % inp.ndim for d in dim}
            shape = [s for i, s in enumerate(out.shape) if i not in dset]
            out = out.view(shape)
        return out

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    return _single_dim_reduce(inp, dim % inp.ndim, keepdim)
