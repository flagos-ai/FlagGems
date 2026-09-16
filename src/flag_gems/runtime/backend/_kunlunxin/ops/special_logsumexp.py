import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

from ..utils.tle_copy import tle_copy
from .copy import copy_ as _vendor_copy_
from .logsumexp import _MULTIROW_MAX_N, _reduce_inner

logger = logging.getLogger(__name__)


def _dma_copy(src, dst):
    if not tle_copy(src, dst):
        _vendor_copy_(dst, src)


def _reduce_inner_any_n(inp, rows, N):
    if N <= _MULTIROW_MAX_N and (N & (N - 1)) != 0:
        P = triton.next_power_of_2(N)
        padded = torch.full(
            (rows, P), float("-inf"), dtype=inp.dtype, device=inp.device
        )
        _dma_copy(inp.reshape(rows, N), torch.as_strided(padded, (rows, N), (P, 1)))
        return _reduce_inner(padded, rows, P)
    return _reduce_inner(inp, rows, N)


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
    return res


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
