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

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def sum_kernel_1(
    inp,
    mid,
    M,
    INP_OFFSET,
    MID_OFFSET,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = ext.program_id(0)
    # Reduce inp[INP_OFFSET : INP_OFFSET + M], program pid covers its pid-th
    # BLOCK_SIZE chunk; partial sum goes to mid[MID_OFFSET + pid]. The offsets
    # let the hybrid path isolate the tail into a single masked program without
    # Python-level view ops (FlagGems overrides narrow and friends).
    offset = INP_OFFSET + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset

    # NEED_MASK=False (chunk fully in-bounds) avoids the XPU slow masked-memory
    # path (HARNESS 1.4); measured 2x+ on the contiguous split reduction.
    if NEED_MASK:
        mask = offset < INP_OFFSET + M
        inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(cdtype)
    else:
        inp_val = tl.load(inp_ptrs).to(cdtype)
    sum_val = tl.sum(inp_val)
    mid_ptr = mid + MID_OFFSET + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def sum_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    if tl.constexpr(mid.dtype.element_ty == tl.float16) or tl.constexpr(
        mid.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = mid.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


# Row-reduce tile bounds. We accumulate elementwise into a persisted
# [BLOCK_M, BLOCK_N] tile and reduce ONCE after the loop (reduce-OUTSIDE). This is
# exact for ALL N: the reduce-INSIDE variant (tl.sum per iteration) is faster in
# theory but MISCOMPILES on this XPU whenever several full blocks are followed by a
# partially-masked tail (verified in isolation), so we do not use it.
#
# BLOCK_N is capped at 8192 and BLOCK_M is fixed at 128: this bounds the live tile at
# [128, 8192] (~1M elts, compiles cold in ~3.4s, no IR explosion). The old code
# scaled BLOCK_M as next_pow2(cdiv(M, 12)) up to 131072, producing tensor<131072x8192>
# tiles and multi-GB IR dumps. A wide BLOCK_N (up to 8192) is a big win for the
# small-M / huge-N regime (e.g. [1024, 1048576]: 0.21 -> 0.52 speedup vs BLOCK_N=512)
# and never hurts the other shapes. See harness/solution/sum_perf_fix.md sweeps.
_BLOCK_M = 128
_BLOCK_N_MAX = 8192
# For small M + huge N, BLOCK_M=128 leaves only a handful of row-programs (e.g.
# M=1024 -> grid=8) which under-fills the 12 clusters; a smaller BLOCK_M exposes more
# row-parallelism (M=1024, N=1048576: 0.52 -> 0.71). It is catastrophic for large M
# (grid over-subscription), so it is gated on M being small.
_SMALL_M = 4096
_HUGE_N = 32768
_SMALL_BLOCK_M = 8

# Full-tensor split reduction (sum / sum_out) first-stage block. For fp32 a
# 131072-element unmasked block (streamed via buffer_size_limit=2048) is
# exactness-verified on XPU (all-ones per-program == 131072, +-1 alternation,
# randn vs fp64 reference) and ~25-35% faster than the get_block_size_1d byte cap
# (32768 elts) on large divisible inputs. Taken ONLY when dtype is fp32, M is an
# exact multiple of 131072 and M is large enough to keep >=128 programs; every
# other case keeps the get_block_size_1d block. See
# harness/solution/performance/sum_out_perf_fix.md (probe3/probe4).
_FP32_SPLIT_BLK = 131072
_FP32_SPLIT_MIN_M = 16 * 1024 * 1024

# Small full-tensor reductions run in ONE program directly into `out`: the
# two-stage path pays two triton launches + a mid allocation (~8us floor) while
# a single unmasked program over a pow2 M <= 8192 is the documented-safe tl.sum
# regime (HARNESS 1.5). Non-pow2 / larger M keeps the two-stage split path.
_SINGLE_KERNEL_MAX_M = 8192
# Hybrid tail split (unmasked prefix + 1 masked tail program) only pays off once
# enough full blocks exist to amortize the extra launch (~3us, ~0.3us/program).
_HYBRID_MIN_FULL = 16


def _full_sum_block_size(inp_dtype, M, element_size):
    block_size = get_block_size_1d(M, element_size)
    if (
        inp_dtype is torch.float32
        and M >= _FP32_SPLIT_MIN_M
        and M % _FP32_SPLIT_BLK == 0
    ):
        block_size = _FP32_SPLIT_BLK
    return block_size


@libentry()
@triton.jit
def sum_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Reduce-OUTSIDE: elementwise-accumulate into a persisted [BLOCK_M, BLOCK_N] tile,
    # reduce once after the loop. Exact for all N (see comment above).
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < N)
        a = tl.load(inp + cols, mask, other=0).to(cdtype)
        _sum += a
    tl.store(out, tl.sum(_sum, axis=1)[:, None], row_mask)


def _launch_sum_dim(inp, out, M, N):
    if M == 1:
        # Degenerate: the whole tensor reduces to a single element. The row-parallel
        # kernel would launch grid=1 and serialize the entire N-element reduction in
        # one program (e.g. N=2**28 -> ~1.4s). Route to the two-stage split reduction
        # (parallel over N), the same machinery the full-tensor sum() uses.
        block_size = get_block_size_1d(N, inp.element_size())
        mid_size = triton.cdiv(N, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        mid = torch.empty((mid_size,), dtype=out.dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            sum_kernel_1[(mid_size, 1, 1)](
                inp,
                mid,
                N,
                0,
                0,
                block_size,
                M % block_size != 0,
                buffer_size_limit=2048,
            )
            if mid_size == 1:
                out.copy_(mid.reshape(out.shape))
            else:
                sum_kernel_2[(1, 1, 1)](
                    mid, out, mid_size, block_mid, buffer_size_limit=2048
                )
        return

    block_n = min(triton.next_power_of_2(N), _BLOCK_N_MAX)
    if M <= _SMALL_M and N >= _HUGE_N:
        block_m = _SMALL_BLOCK_M
    else:
        block_m = _BLOCK_M
    grid = (triton.cdiv(M, block_m),)
    with torch_device_fn.device(inp.device):
        sum_kernel[grid](inp, out, M, N, block_m, block_n, buffer_size_limit=2048)


def _full_sum_launch(inp, out, M, dtype):
    # Shared device path for full-tensor reductions (sum / sum_out). `out` is a
    # 0-dim tensor receiving the scalar result.
    with torch_device_fn.device(inp.device):
        if 0 < M <= _SINGLE_KERNEL_MAX_M and M & (M - 1) == 0:
            sum_kernel_1[(1, 1, 1)](
                inp,
                out,
                M,
                0,
                0,
                M,
                False,
                buffer_size_limit=2048,
            )
            return

        block_size = _full_sum_block_size(dtype, M, inp.element_size())
        full = M // block_size
        tail = M - full * block_size
        if tail == 0:
            mid_size = full
            mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
            sum_kernel_1[(mid_size, 1, 1)](
                inp,
                mid,
                M,
                0,
                0,
                block_size,
                False,
                buffer_size_limit=2048,
            )
        elif full < _HYBRID_MIN_FULL:
            # Small tail: a single masked kernel handles everything (cheap, and
            # the extra launch of the hybrid path would not pay off).
            mid_size = full + 1
            mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
            sum_kernel_1[(mid_size, 1, 1)](
                inp,
                mid,
                M,
                0,
                0,
                block_size,
                True,
                buffer_size_limit=2048,
            )
        else:
            # Large tail: run the divisible prefix UNMASKED (fast DMA path) and
            # isolate the tail into ONE masked program. Masked loads take a slow
            # path on XPU (HARNESS 1.4); previously every program paid it.
            mid_size = full + 1
            mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
            sum_kernel_1[(full, 1, 1)](
                inp,
                mid,
                full * block_size,
                0,
                0,
                block_size,
                False,
                buffer_size_limit=2048,
            )
            sum_kernel_1[(1, 1, 1)](
                inp,
                mid,
                tail,
                full * block_size,
                full,
                block_size,
                True,
                buffer_size_limit=2048,
            )
        if mid_size == 1:
            out.copy_(mid.reshape(out.shape))
            return
        sum_kernel_2[(1, 1, 1)](
            mid,
            out,
            mid_size,
            triton.next_power_of_2(mid_size),
            buffer_size_limit=2048,
        )


def sum(inp, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN SUM")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64
    out = torch.empty([], dtype=dtype, device=inp.device)
    _full_sum_launch(inp, out, M, dtype)
    return out


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS_KUNLUNXIN SUM_OUT")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64
    _full_sum_launch(inp, out, M, dtype)
    return out


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN SUM_DIM")
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if dim is None or dim == []:
            out_shape = [1] * len(out_shape) if keepdim else []
        else:
            dims = dim if isinstance(dim, (list, tuple)) else [dim]
            if keepdim:
                for d in dims:
                    out_shape[d % inp.ndim] = 1
            else:
                for d in sorted(dims, key=lambda x: x % inp.ndim, reverse=True):
                    out_shape.pop(d % inp.ndim)
        out = torch.empty(out_shape, dtype=dtype, device=inp.device)
        zero_(out)
        return out

    if dim == []:
        if not keepdim:
            return sum(inp, dtype=dtype)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum(inp, dtype=dtype), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=dtype, device=inp.device)

    _launch_sum_dim(inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS_KUNLUNXIN SUM_DIM_OUT")
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

    if inp.numel() == 0:
        dims = (
            dim
            if isinstance(dim, (list, tuple))
            else ([dim] if dim is not None else [])
        )
        if keepdim:
            for d in dims:
                pass  # out shape already correct from caller
        zero_(out)
        return out

    if dim == []:
        if not keepdim:
            return sum_out(inp, dtype=dtype, out=out)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum_out(inp, dtype=dtype, out=out), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out.resize_(shape)
    _launch_sum_dim(inp, out, M, N)
    if not keepdim:
        # Compute squeezed shape and resize in-place
        out_shape = [s for i, s in enumerate(shape) if i not in dim]
        out.resize_(out_shape)
    return out
