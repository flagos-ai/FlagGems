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
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)


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
# explode. Under that fixed budget we RESHAPE the tile by N (the all_dim
# lesson): large-N reductions want a wide/short tile (few loop trips, wide DMA)
# while small/medium-N want a tall tile (more rows in flight). The wide path
# raised fp16 [1024,65536]/[1024,1M] but a blanket-wide tile starved medium-M
# shapes like [4096,4096] (BLOCK_M collapsed to 16), so we switch on N.
_TILE_BUDGET = 32768
_N_WIDE = 8192


def _block_n(N):
    if N > _N_WIDE:
        return builtins.min(triton.next_power_of_2(N), 2048)  # wide for large N
    return builtins.min(triton.next_power_of_2(N), 512)  # tall-friendly otherwise


def heur_n_block_size(args):
    return _block_n(args["N"])


def heur_m_block_size(args):
    block_n = _block_n(args["N"])
    block_m = triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num
    return builtins.min(block_m, builtins.max(_TILE_BUDGET // block_n, 1))


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

    # Fast path: reduce a SINGLE non-last dim of a contiguous fp16/bf16 tensor.
    # Reducing the middle dim of a [M0, N, M1] view is exactly
    #   bmm(ones[M0, 1, N], x[M0, N, M1]) / N  ->  [M0, M1].
    # This reads x in its native (contiguous) layout through the matmul unit, so
    # it AVOIDS the dim_compress .contiguous() transpose copy that dominates the
    # 3-D middle-axis case (the "transpose wall": ~447ms of the 452ms for
    # [100,65536,100]). Under use_gems, torch.bmm re-dispatches to the fast gems
    # matmul kernel. gems fp32 bmm is broken on XPU (wrong results for non-pow2
    # M1), so this path is fp16/bf16 only; fp32 falls through to the reduce
    # kernel. We sum with ones=1.0 and divide afterwards (bmm accumulates in
    # fp32; dividing after keeps the accumulator well-scaled).
    #
    # The gems bmm kernel can fail to compile (uni_sram OOM / SDNN combine
    # failure) for extreme matmul shapes -- a huge output free dim (large M1) or
    # a tiny free dim paired with a large odd contraction dim (e.g. M1=3,
    # K=40999). We therefore try the bmm path and, on ANY compile/runtime error,
    # fall through to the numerically-correct reduce kernel below. This keeps
    # the speedup for the shapes bmm handles while guaranteeing correctness.
    if (
        len(dim) == 1
        and dim[0] != x.ndim - 1
        and dtype == x.dtype
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.is_contiguous()
        and shape[dim[0]] > 1
    ):
        d = dim[0]
        N = shape[d]
        M0 = 1
        for s in shape[:d]:
            M0 *= s
        M1 = 1
        for s in shape[d + 1 :]:
            M1 *= s
        try:
            x3 = x.reshape(M0, N, M1)
            ones = torch.ones((M0, 1, N), dtype=x.dtype, device=x.device)
            out = (torch.bmm(ones, x3).reshape(M0, M1) / N).to(dtype)
            out_shape = list(shape)
            out_shape[d] = 1
            out = out.reshape(out_shape)
            if not keepdim:
                out = out.squeeze(d)
            return out
        except Exception:
            logger.debug("GEMS_KUNLUNXIN MEAN_DIM bmm fast path unavailable, fallback")

    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N

    # Edge case: M=1 means all dims are reduced → global mean over N elements.
    # mean_dim XPU API does not support M=1.
    if M == 1:
        scalar_out = mean(x, dtype=dtype)  # 0-d tensor
        out = scalar_out.reshape(shape)
        if not keepdim:
            out = out.squeeze(dim)
        return out

    # Edge case: N=1 means reducing a trivial (size-1) dimension.
    # mean of 1 element = that element; just copy with dtype conversion.
    # mean_dim XPU API does not support N=1.
    if N == 1:
        out = x.to(dtype=dtype).reshape(shape)
        if not keepdim:
            out = out.squeeze(dim)
        return out

    out = torch.empty(shape, dtype=dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    with torch_device_fn.device(x.device):
        mean_dim_kernel[grid](x, out, M, N, buffer_size_limit=2048)
    if not keepdim:
        out = out.squeeze(dim)
    return out
