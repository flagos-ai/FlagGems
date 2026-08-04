# Kunlunxin (XPU) override of `special_log_softmax`.
#
# `special_log_softmax(self, dim, dtype=None)` is functionally identical to
# `log_softmax` over `dim` (with an optional output-dtype cast). Delegating to
# the tuned kunlunxin `log_softmax` did NOT help: log_softmax's N<=8192 path is a
# 2D [TILE_M,N] axis=1 reduce that runs at ~9 GB/s on XPU (14.5ms for [4096,4096])
# -- the reduce, not the exp, is the wall. IR baseline
# `harness/perf_ir_3/ir-special_log_softmax-dev7.log`: 0.003-0.27.
#
# Fix: dedicated per-row kernels dispatched on N (dim=-1, i.e. K==1 fast path):
#   * N <= 256                  -> multirow [TILE_M,N] 2D tile (launch amortization
#                                   beats the slow reduce when N is tiny).
#   * 256 < N <= 65536          -> single-tile constexpr-N 1D per-row: grid=(M,),
#                                   `tl.arange(0,N)` NO mask -> ONE stride-1
#                                   contiguous block DMA + axis=0 reduce (~20x over
#                                   the 2D axis=1 reduce; [4096,4096] 0.74ms).
#   * N > 65536                 -> chunked online (2-pass) per-row, TILE_N=32768
#                                   (single-tile overflows XPU SRAM for N>=131072).
# K>1 (dim not last) -> fall back to the tuned kunlunxin log_softmax. Pure numeric
# rewrite of the same log-softmax -> zero algorithm change.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from .log_softmax import log_softmax as _log_softmax_kunlunxin

logger = logging.getLogger(__name__)

_SINGLE_TILE_MAX_N = 65536
_MULTIROW_MAX_N = 256
_MULTIROW_TILE_M = 32
_CHUNK_TILE_N = 32768


@triton.jit
def _sls_multirow_kernel(o_ptr, i_ptr, M, N: tl.constexpr, TILE_M: tl.constexpr):
    pid = tl.program_id(0)
    mo = pid * TILE_M + tl.arange(0, TILE_M)
    no = tl.arange(0, N)
    off = mo[:, None] * N + no[None, :]
    mask = mo[:, None] < M
    x = tl.load(i_ptr + off, mask=mask, other=-float("inf")).to(tl.float32)
    mx = tl.max(x, 1)
    e = tl.exp(x - mx[:, None])
    z = tl.sum(e, 1)
    out = x - mx[:, None] - tl.log(z)[:, None]
    tl.store(o_ptr + off, out, mask=mask)


@triton.jit
def _sls_row1d_kernel(o_ptr, i_ptr, M, N: tl.constexpr):
    pid = tl.program_id(0)
    no = tl.arange(0, N)
    off = pid * N + no
    x = tl.load(i_ptr + off).to(tl.float32)
    mx = tl.max(x, 0)
    e = tl.exp(x - mx)
    z = tl.sum(e, 0)
    tl.store(o_ptr + off, x - mx - tl.log(z))


@triton.jit
def _sls_chunk_kernel(o_ptr, i_ptr, M, N, TILE_N: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * N
    m = tl.full([TILE_N], -float("inf"), tl.float32)
    z = tl.full([TILE_N], 0.0, tl.float32)
    for s in range(0, N, TILE_N):
        no = s + tl.arange(0, TILE_N)
        mask = no < N
        x = tl.load(i_ptr + base + no, mask=mask, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        z = tl.where(
            m_new == float("-inf"), z, z * tl.exp(m - m_new) + tl.exp(x - m_new)
        )
        m = m_new
    mr = tl.max(m, 0)
    zr = tl.sum(z * tl.exp(m - mr), 0)
    lz = tl.log(zr)
    for s in range(0, N, TILE_N):
        no = s + tl.arange(0, TILE_N)
        mask = no < N
        x = tl.load(i_ptr + base + no, mask=mask, other=0.0).to(tl.float32)
        tl.store(o_ptr + base + no, x - mr - lz, mask=mask)


def special_log_softmax(self, dim, dtype=None):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim

    inp = self.contiguous()
    if dtype is not None and dtype != inp.dtype:
        inp = inp.to(dtype)

    M = 1
    for i in range(dim):
        M *= inp.shape[i]
    N = inp.shape[dim]
    K = inp.numel() // M // N

    # dim not last -> reduce dim is strided; delegate to the tuned log_softmax
    # (handles K>1 via transpose). Cast already applied above.
    if K != 1:
        return _log_softmax_kunlunxin(inp, dim)

    out = torch.empty_like(inp)
    if N == 0 or M == 0:
        return out

    with torch_device_fn.device(inp.device):
        if N <= _MULTIROW_MAX_N:
            grid = (triton.cdiv(M, _MULTIROW_TILE_M),)
            _sls_multirow_kernel[grid](
                out, inp, M, N, TILE_M=_MULTIROW_TILE_M, num_warps=8
            )
        elif N <= _SINGLE_TILE_MAX_N:
            grid = (M,)
            _sls_row1d_kernel[grid](out, inp, M, N, num_warps=8)
        else:
            grid = (M,)
            _sls_chunk_kernel[grid](out, inp, M, N, TILE_N=_CHUNK_TILE_N, num_warps=8)

    return out
