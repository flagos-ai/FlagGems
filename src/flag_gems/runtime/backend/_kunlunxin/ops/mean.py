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

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)

# Dispatch key set used to redispatch a native (non-FlagGems) copy.
# `use_gems` intercepts aten::copy_ on the CUDA/XPU dispatch key with a
# pointwise_dynamic Triton kernel that is catastrophically slow for large
# tensors (~2.4GB/s floor: a 64MB permute copy under use_gems takes ~28ms vs
# ~0.09ms native). Redispatching to CompositeExplicitAutograd routes around that
# interception to the fast native copy. This is the same fallback mechanism
# FlagGems' own copy_ uses internally (see flag_gems/ops/copy.py).
_NATIVE_COPY_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _contiguous_native(t):
    """Contiguous copy that bypasses the (slow, intercepted) FlagGems copy_."""
    if t.is_contiguous():
        return t
    dst = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    torch.ops.aten.copy_.default.redispatch(_NATIVE_COPY_KEYSET, dst, t, False)
    return dst


def _dim_compress_native(inp, dims):
    """Replicate dim_compress (permute reduced dims to the trailing dims) but
    materialize the permutation with a native copy, never the intercepted
    FlagGems copy_. For a contiguous input whose reduced dims already form a
    trailing suffix this is a no-op (identity permute, already contiguous)."""
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return _contiguous_native(inp.permute(*order))


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
    if dim is None:
        out = mean(x, dtype=dtype)
        if not keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]

    # Compress reduced dims to the trailing dims. Uses a native (non-FlagGems)
    # copy for the permutation: under `use_gems` the intercepted FlagGems copy_
    # runs at a ~2.4GB/s floor (64MB permute = ~28ms), which is the true cause
    # of the old [64,512,512] 28-30ms pathological. Redispatch gives the native
    # ~1440GB/s transpose, after which the reduction is contiguous and fast.
    x = _dim_compress_native(x, dim)
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
    # mean of 1 element = that element; just copy with dtype conversion.
    # mean_dim XPU API does not support N=1.
    if N == 1:
        return _contiguous_native(x.to(dtype=dtype)).reshape(out_shape)

    out = torch.empty(out_shape, dtype=dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    with torch_device_fn.device(x.device):
        mean_dim_kernel[grid](x, out, M, N, buffer_size_limit=2048)
    return out
