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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)

# GCU400 (gcu400 family): a program keeps at most an accumulator tile plus one
# loaded tile live in DSM.  Staying within this budget keeps two programs
# resident per SP.  Measured per-program fixed cost (~68ns/CTA) makes few large
# programs far cheaper than many small ones for this streaming reduction.
AMIN_DSM_BUDGET = 384 * 1024


@libentry()
@triton.jit
def amin_kernel_1(
    inp,
    mid,
    M,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Phase 1 of the full reduction: each program folds a CHUNK-sized span
    # into a running [BLOCK] minimum (elementwise only, one tree at the end)
    # and stores one scalar partial into mid.
    pid = tle.program_id(0)
    base = pid * CHUNK
    offs = tl.arange(0, BLOCK)
    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    acc = tl.full([BLOCK], value=max_value, dtype=acc_type)
    for off in range(0, CHUNK, BLOCK):
        ptrs = inp + base + off + offs
        if EVEN:
            inp_val = tl.load(ptrs)
        else:
            mask = base + off + offs < M
            inp_val = tl.load(ptrs, mask=mask, other=max_value)
        acc = tl.minimum(acc, inp_val)
    amin_val = tl.min(acc)
    tl.store(mid + pid, amin_val)


@libentry()
@triton.jit
def amin_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    amin_val = tl.min(mid_val)
    tl.store(out, amin_val)


@libentry()
@triton.jit
def amin_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    # Row reduction over a (M, N) contiguous view: rows spaced by N, row
    # elements contiguous.  EVEN_M/EVEN_N are compile-time specializations
    # that drop all masks when the grid divides M and N exactly.  The outer
    # group loop bounds the launch grid: huge-M/small-N inputs would need
    # more than 65535 programs (grid.x hardware limit), so the host launches
    # at most 65535 programs and each one strides over its share of groups.
    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)
    pid = tle.program_id(0)
    nprog = tl.num_programs(0)
    num_groups = tl.cdiv(M, BLOCK_M)
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype

    for g in range(pid, num_groups, nprog):
        rows = g * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_ptrs = inp + rows * N
        out_ptrs = out + rows
        row_mask = rows < M
        _all = tl.full([BLOCK_M, BLOCK_N], value=max_value, dtype=acc_type)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            if EVEN_M and EVEN_N:
                a = tl.load(row_ptrs + cols)
            else:
                col_mask = cols < N
                mask = row_mask and col_mask
                a = tl.load(row_ptrs + cols, mask, other=max_value)
            _all = tl.minimum(_all, a)
        all = tl.min(_all, axis=1)[:, None]
        if EVEN_M:
            tl.store(out_ptrs, all)
        else:
            tl.store(out_ptrs, all, row_mask)


@libentry()
@triton.jit
def amin_kernel_strided(
    inp,
    out,
    K,
    J,
    num_jt,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_J: tl.constexpr,
):
    # Reduction of a middle dim on a contiguous tensor, viewed as (I, K, J):
    # the reduced span K is strided by J while each J span stays contiguous.
    # Program (pid_i, pid_j) reduces all K rows of one (i, j-tile) with no
    # copy of the input (dim_compress would transpose the whole tensor).
    # The grid is flattened to 1D (grid.y is capped at 255 on gcu400).
    pid = tle.program_id(0)
    pid_i = pid // num_jt
    pid_j = pid % num_jt
    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)
    base = inp + pid_i * (K * J)
    cols = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)[None, :]
    col_mask = cols < J

    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    _all = tl.full([BLOCK_K, BLOCK_J], value=max_value, dtype=acc_type)
    for off in range(0, K, BLOCK_K):
        rows = off + tl.arange(0, BLOCK_K)[:, None]
        ptrs = base + rows * J + cols
        if EVEN_K and EVEN_J:
            a = tl.load(ptrs)
        else:
            mask = rows < K and col_mask
            a = tl.load(ptrs, mask, other=max_value)
        _all = tl.minimum(_all, a)
    all = tl.min(_all, axis=0)
    out_ptrs = out + pid_i * J + cols
    if EVEN_J:
        tl.store(out_ptrs, all)
    else:
        tl.store(out_ptrs, all, mask=col_mask)


def _amin_full(inp, keepdim):
    # Full reduction over all elements: two-phase (partials + combine).
    # CHUNK targets ~32 programs (per-program fixed cost dominates when the
    # grid is large); BLOCK picks a 16KB tile per dtype.  Measured on gcu400:
    # 1M fp32 15.4us -> 5.15us, 1M fp16 -> 4.06us.
    M = inp.numel()
    dtype = inp.dtype
    block = max(4096, 16384 // inp.element_size())
    span = triton.next_power_of_2((M + 31) // 32)
    chunk = min(max(span, block), 262144)
    mid_size = triton.cdiv(M, chunk)
    block_mid = triton.next_power_of_2(mid_size)
    even = M % chunk == 0
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    if not keepdim:
        out = torch.empty([], dtype=dtype, device=inp.device)
    else:
        shape = [1] * inp.dim()
        out = torch.empty(shape, dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        amin_kernel_1[(mid_size, 1)](
            inp,
            mid,
            M,
            CHUNK=chunk,
            BLOCK=block,
            EVEN=even,
            num_warps=2,
        )
        amin_kernel_2[(1, 1)](mid, out, mid_size, block_mid)
    return out


def _amin_rows_launch(inp, out, M, N):
    # Tile shape per dtype, tuned on gcu400 for (M, N) = (4096, 4096):
    # fp16 128x512/nw2 = 22.2us, fp32 32x1024/nw2 = 38.3us,
    # bf16 32x1024/nw2 = 31.7us (acc is fp32 for bf16, so its tile is wider
    # in bytes).  All keep (acc + load) DSM at ~256KB or below.
    esize = inp.element_size()
    if esize == 4:
        block_n = min(triton.next_power_of_2(N), 1024)
        lane_target = 32768
    elif inp.dtype == torch.bfloat16:
        block_n = min(triton.next_power_of_2(N), 1024)
        lane_target = 32768
    else:
        block_n = min(triton.next_power_of_2(N), 512)
        lane_target = 65536
    block_m = min(triton.next_power_of_2(M), lane_target // block_n, 128)
    while triton.cdiv(M, block_m) < 24 and block_m > 8:
        block_m //= 2
    even_m = M % block_m == 0
    even_n = N % block_n == 0
    grid = (min(triton.cdiv(M, block_m), 65535),)
    with torch_device_fn.device(inp.device):
        amin_kernel[grid](
            inp,
            out,
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            EVEN_M=even_m,
            EVEN_N=even_n,
            num_warps=2,
        )


def _amin_strided_launch(inp, out, outer_size, K, J):
    esize = inp.element_size()
    # BJ=512 / num_warps=2 / BLOCK_K=64 measured best on gcu400: BJ=1024
    # tiles run ~29% slower at equal data, and nw=4 adds ~21% on this kernel.
    block_j = min(triton.next_power_of_2(J), 512)
    block_k = 64
    while block_k > 8 and block_k * block_j * esize * 2 > AMIN_DSM_BUDGET:
        block_k //= 2
    even_k = K % block_k == 0
    even_j = J % block_j == 0
    num_jt = triton.cdiv(J, block_j)
    grid = (outer_size * num_jt,)
    with torch_device_fn.device(inp.device):
        amin_kernel_strided[grid](
            inp,
            out,
            K,
            J,
            num_jt,
            BLOCK_K=block_k,
            BLOCK_J=block_j,
            EVEN_K=even_k,
            EVEN_J=even_j,
            num_warps=2,
        )


def amin(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ENFLAME_GCU400 AMIN")
    assert inp.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), "amin only supports float dtypes"
    if isinstance(dim, int):
        dim = [dim]
    ndim = inp.ndim
    if dim is not None and len(dim) > 0:
        dim = [d % ndim for d in dim]
    if dim is None or len(dim) == 0 or len(set(dim)) == ndim:
        return _amin_full(inp, keepdim)

    shape = list(inp.shape)
    reduced = set(dim)
    out_shape = []
    for i in range(ndim):
        if i in reduced:
            out_shape.append(1)
        else:
            out_shape.append(shape[i])
    dtype = inp.dtype

    # Contiguous inputs: reduce in place with no dim_compress copy.
    if inp.is_contiguous() and inp.numel() > 0:
        sorted_dims = sorted(reduced)
        lo, hi = sorted_dims[0], sorted_dims[-1]
        if sorted_dims == list(range(lo, hi + 1)):
            if lo == ndim - len(sorted_dims):
                # Reduced dims are trailing: plain 2D (M, N) row reduction.
                N = 1
                for d in reduced:
                    N *= shape[d]
                M = inp.numel() // N
                out = torch.empty(out_shape, dtype=dtype, device=inp.device)
                _amin_rows_launch(inp, out, M, N)
                if not keepdim:
                    out = out.squeeze(dim=dim)
                return out
            J = 1
            for i in range(hi + 1, ndim):
                J *= shape[i]
            if J >= 64:
                # Middle-dim reduction: strided (I, K, J) tiles, no copy.
                K = 1
                for d in reduced:
                    K *= shape[d]
                outer_size = inp.numel() // (K * J)
                block_j = min(triton.next_power_of_2(J), 512)
                num_jt = triton.cdiv(J, block_j)
                if 24 <= outer_size * num_jt <= 65535:
                    out = torch.empty(out_shape, dtype=dtype, device=inp.device)
                    _amin_strided_launch(inp, out, outer_size, K, J)
                    if not keepdim:
                        out = out.squeeze(dim=dim)
                    return out

    # Generic fallback: compress the reduced dims to the end, then row-reduce.
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
    M = inp.numel() // N
    out = torch.empty(out_shape, dtype=dtype, device=inp.device)
    _amin_rows_launch(inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def amin_(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ENFLAME_GCU400 AMIN_")
    assert inp.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), "amin_ only supports float dtypes"
    if isinstance(dim, int):
        dim = [dim]
    ndim = inp.ndim
    if dim is not None and len(dim) > 0:
        dim = [d % ndim for d in dim]
    if dim is None or len(dim) == 0 or len(set(dim)) == ndim:
        # Same tuned two-phase kernels as amin, then broadcast the scalar
        # back into inp (0-dim or all-ones source broadcasts under copy_).
        out = _amin_full(inp, keepdim)
        inp.copy_(out)
        return inp
    else:
        result = amin(inp, dim=dim, keepdim=True)
        inp.copy_(result)
        return inp
