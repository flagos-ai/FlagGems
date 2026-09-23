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

import builtins
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d
from .sum import sum_dim as _sum_dim_tle

logger = logging.getLogger(__name__)

# Dtypes the TLE sum path (moved onto tle.gpu upstream, see `sum.py`) can carry.
# Outside this set `mean_dim` keeps the original dim_compress + pointer path.
_TLE_FAST_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit
def mean_scalar_kernel(inp, out, M, BLOCK_SIZE: tl.constexpr):
    """Scalar mean over all M elements.
    On XPU (USE_XHPC): intercepted by baidu::xpu::api::mean binding.
    Triton fallback (single CTA): sequential accumulation for correctness.
    Params for binding:
      kernelParams[0] = inp, kernelParams[1] = out
      kernelConsts[2] = M,   kernelConsts[3] = BLOCK_SIZE
    """
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_SIZE):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        v = tl.load(inp + offset, mask=mask, other=0.0).to(tl.float32)
        acc += v
    result = tl.sum(acc) / M
    tl.store(out, result)


def mean(inp, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
    if M == 0:
        # torch returns NaN for the mean of an empty tensor (verified against
        # CPU torch); `get_block_size_1d(0)` is not a defined launch config on
        # this backend and the scalar kernel would divide 0 by 0.
        return torch.full([], float("nan"), dtype=dtype, device=inp.device)
    BLOCK_SIZE = get_block_size_1d(M, inp.element_size())
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_scalar_kernel[(1, 1, 1)](inp, out, M, BLOCK_SIZE, buffer_size_limit=2048)
    return out


# Persisted-accumulator tile budget. The old heuristics allowed
# BLOCK_M=next_pow2(cdiv(M,12)) (unbounded) x BLOCK_N=min(next_pow2(N),8192),
# so the persisted [BLOCK_M, BLOCK_N] accumulator became a giant 2D constexpr
# tile (IR shows tensor<1024x8192xf32> = 8.4M elements). ConvertTritonXPUToLLVM
# materializes it per element -> the 1.78GB IR dump. We keep the numerically
# correct persisted-accumulator + single final reduce (the in-loop
# tl.sum(a, axis=1) alternative miscompiles on XPU for fp16/bf16 -> wrong
# results), but bound BLOCK_M x BLOCK_N to a fixed budget so the tile can never
# explode.
#
# Tile tuning (measured on-device, dev4, 2026-09, with buffer_size_limit=2048):
#   - fp32 medium-N reductions are fastest at BLOCK_N=512, BLOCK_M=64.
#   - fp16/bf16 load half the bytes per element, so a slightly narrower
#     BLOCK_N=256 with more rows (BLOCK_M=128) wins: the fp32 accumulator
#     occupies the same SRAM, and 256 keeps the convert pipe fed without
#     bloating the persisted tile.
#   - BLOCK_M=min(next_pow2(M),64/128) (parallelize over rows) beats the old
#     cdiv(M,12) formula: it never collapses BLOCK_M on small-M shapes.
#   - For very large N (N>8192) a wide BLOCK_N (up to 2048) wins (fewer loop
#     trips, wide DMA); the budget cap then collapses BLOCK_M so the tile stays
#     bounded.
_TILE_BUDGET = 32768
_N_WIDE = 8192


def _block_n(N, dtype):
    if N > _N_WIDE:
        return builtins.min(triton.next_power_of_2(N), 2048)  # wide for large N
    if dtype == torch.float32:
        return builtins.min(triton.next_power_of_2(N), 512)
    return builtins.min(triton.next_power_of_2(N), 256)  # fp16/bf16: narrower


def _block_m(M, dtype):
    cap = 128 if dtype != torch.float32 else 64
    return builtins.min(triton.next_power_of_2(M), cap)


def heur_n_block_size(args):
    return _block_n(args["N"], args["X"].dtype)


def heur_m_block_size(args):
    block_n = _block_n(args["N"], args["X"].dtype)
    block_m = _block_m(args["M"], args["X"].dtype)
    return builtins.max(builtins.min(block_m, _TILE_BUDGET // block_n), 1)


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("mean"),
#     key=["M", "N"],
# )
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """2-D reduction: reduce N-dim for each of M rows.
    On XPU (USE_XHPC): intercepted by baidu::xpu::api::mean_dim binding.
    Params for binding:
      kernelParams[0] = X,    kernelParams[1] = Mean
      kernelParams[2] = M,    kernelParams[3] = N  (runtime scalars)
      kernelConsts[4] = BLOCK_M (constexpr), kernelConsts[5] = BLOCK_N (constexpr)
    """
    # Map the program id to the row of X it should compute.
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    # Persisted [BLOCK_M, BLOCK_N] accumulator + a SINGLE reduce after the loop.
    # Slot j accumulates cols j, j+BLOCK_N, j+2*BLOCK_N, ... (strided partials);
    # tl.sum(_mean, axis=1) then combines them. This is numerically correct for
    # any BLOCK_N. We deliberately do NOT reduce inside the loop
    # (acc += tl.sum(a, axis=1)) because that pattern miscompiles on XPU for
    # fp16/bf16 inputs (converted-tile in-loop axis=1 reduce returns garbage;
    # verified: 97% mismatch at (200,40999,3)). The tile stays bounded because
    # heur_m/heur_n cap BLOCK_M*BLOCK_N to _TILE_BUDGET, so no giant-tile IR.
    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1)[:, None] / N
    tl.store(Mean, mean, row_mask)


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN MEAN_DIM")

    if dtype is None:
        dtype = x.dtype
    if dim is None or dim == () or dim == []:
        # `dim == ()` / `dim == []` behave like `dim is None` in torch: a full
        # reduction. keepdim then returns the (1,)*ndim shape, and without it
        # the 0-d scalar that `mean` already produces.
        out = mean(x, dtype=dtype)
        if keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]

    # Empty reduction domain (e.g. a (0, 3) input reduced over dim=0): torch
    # returns a NaN-filled tensor of the reduced shape (verified against CPU
    # torch), and the `M = x.numel() // N` below would divide by zero. Guard
    # before dim_compress / any launch so zero-sized inputs never reach the
    # copy or kernel paths.
    _N_reduced = 1
    for _i in dim:
        _N_reduced *= shape[_i]
    if _N_reduced == 0:
        out_shape = [1 if _i in dim else s for _i, s in enumerate(shape)]
        if not keepdim:
            out_shape = [s for _i, s in enumerate(out_shape) if _i not in dim]
        return torch.full(out_shape, float("nan"), dtype=dtype, device=x.device)

    # --- TLE fast path (D-022) -----------------------------------------------
    # A single-axis reduce over a contiguous TLE-supported dtype does not need
    # `dim_compress`: the sum path already has a row kernel for the last axis
    # and a two-pass fold for a middle one.  `dim_compress` would permute and
    # materialise the whole tensor first — on the 1G f32 along dim=1 case that
    # is 5.69 ms of copy plus 2.50 ms of reduce, against 2.39 ms for the entire
    # aten call.  Anything outside the gate below falls through to the original
    # path, unchanged.
    if len(dim) == 1 and x.is_contiguous() and x.dtype in _TLE_FAST_DTYPES:
        N_tle = shape[dim[0]]
        if N_tle > 1 and x.numel() // N_tle > 1:
            try:
                # One launch total: `scale` is folded into the sum kernel before its
                # store, so `mean = sum * (1/N)` needs no second elementwise op. That
                # matters here: a gem-dispatched elementwise op costs a **fixed ~92µs
                # per call** on this machine regardless of element count (measured
                # 2026-09-16 on (64,64) and (4096,4096) alike), which is more than the
                # whole TLE row-reduce it would complement.
                # Accumulation stays fp32 (aten opmath) and the cast happens after the
                # scale, in the kernel's store — the same order torch.mean uses.
                out = _sum_dim_tle(
                    x, dim=dim, keepdim=keepdim, dtype=dtype, scale=1.0 / N_tle
                )
                logger.debug("GEMS_KUNLUNXIN MEAN_DIM tle fast path N=%d", N_tle)
                return out
            except Exception as exc:  # noqa: BLE001 — any gap re-uses the old path
                logger.debug(
                    "GEMS_KUNLUNXIN MEAN_DIM tle fast path unavailable (%s); "
                    "falling back to dim_compress",
                    exc,
                )

    # Compress reduced dims to the trailing dims. The permutation is
    # materialized by the gem's own copy (dim_compress -> permute+contiguous ->
    # FlagGems copy_ / TLE copy family). 2026-09-14: an earlier revision
    # redispatched the permutation to the *native* copy because the gems copy_
    # was believed to sit at a ~2.4GB/s floor (28ms for a 64MB permute); that
    # no longer reproduces on the current TLE copy family (measured 0.25ms gems
    # vs 0.11ms native for the [64,512,512] fp32 permute), and vendor
    # delegation in the measured path is banned for metric integrity.
    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N

    # Final contiguous output shape (singleton reduced dims dropped when
    # keepdim=False). Allocating the output in this shape keeps the returned
    # tensor contiguous: the squeezed view of a shape-with-singleton-dims
    # tensor (e.g. [64,1,512] -> [64,512]) is non-contiguous, and torch then
    # materializes it with contiguous()+clone()+copy_(), hitting the same slow
    # intercepted copy_. Reduction rows map 1:1 onto flat offsets of the final
    # tensor (compressed [M,N] -> flat output index m), so the kernel can write
    # directly into the contiguous buffer.
    out_shape = list(shape)
    for i in dim:
        out_shape[i] = 1
    if not keepdim:
        out_shape = [s for idx, s in enumerate(out_shape) if idx not in dim]

    # Edge case: M=1 means all dims are reduced → global mean over N elements.
    # mean_dim XPU API does not support M=1.
    if M == 1:
        scalar_out = mean(x, dtype=dtype)  # 0-d tensor
        return scalar_out.reshape(out_shape)

    # Edge case: N=1 means reducing a trivial (size-1) dimension.
    # mean of 1 element = that element; just cast (gems to_copy) and reshape.
    # mean_dim XPU API does not support N=1.
    if N == 1:
        return x.to(dtype=dtype).reshape(out_shape)

    out = torch.empty(out_shape, dtype=dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    with torch_device_fn.device(x.device):
        mean_dim_kernel[grid](x, out, M, N, buffer_size_limit=2048)
    return out
