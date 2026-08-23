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
import math
from functools import reduce as _reduce

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def reduce_mul(a, b):
    return a * b


@libentry()
@triton.jit
def prod_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
    mid_value = tl.reduce(inp_val, axis=0, combine_fn=reduce_mul)
    mid_ptr = mid + pid
    tl.store(mid_ptr, mid_value.to(inp_val.dtype))


@libentry()
@triton.jit
def prod_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=1.0).to(tl.float32)
    prod_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_mul)
    tl.store(out, prod_val)


def prod(inp, *, dtype=None):
    logger.debug("GEMS PROD")
    if dtype is None:
        dtype = inp.dtype

    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        prod_kernel_mid[(mid_size, 1, 1)](inp, mid, M, block_size)
        prod_kernel_result[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def prod_dim_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        inp = tl.load(input_ptr + inp_offset, mask=mask, other=1.0).to(tl.float32)
        out = tl.reduce(inp, axis=0, combine_fn=reduce_mul, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)
    else:
        acc = tl.full([TILE_N, TILE_K], value=1.0, dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=1.0).to(tl.float32)
            acc *= inp
        out = tl.reduce(acc, axis=0, combine_fn=reduce_mul, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def prod_dim_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        inp_offset = pid_m * N + n_offsets
        mask = n_offsets < N
        inp = tl.load(input_ptr + inp_offset, mask=mask, other=1.0).to(tl.float32)
        out = tl.reduce(inp, axis=0, combine_fn=reduce_mul)
        tl.store(output_ptr + pid_m, out)
    else:
        acc = tl.full([TILE_N], value=1.0, dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_offsets = pid_m * N + n_offsets
            mask = n_offsets < N
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=1.0).to(tl.float32)
            acc *= inp
        out = tl.reduce(acc, axis=0, combine_fn=reduce_mul)
        tl.store(output_ptr + pid_m, out)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def prod_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = ext.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.full((BLOCK_M, BLOCK_N), value=1.0, dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]

        # set mask
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
        acc *= inp_vals
    result_index = tl.reduce(acc, axis=1, combine_fn=reduce_mul)

    offset_index = m_offset
    out_ptrs = out + offset_index
    mask1 = m_offset < M
    tl.store(out_ptrs, result_index, mask=mask1)


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS PROD DIM")

    if not (-inp.ndim <= dim < inp.ndim):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}])"
        )
    if dtype is None:
        dtype = inp.dtype

    shape = list(inp.shape)
    d = dim % inp.ndim
    N = inp.shape[d]
    M = _reduce(lambda x, y: x * y, shape[:d], 1)
    K = _reduce(lambda x, y: x * y, shape[d + 1 :], 1)
    shape[d] = 1
    out = torch.empty(shape, dtype=dtype, device=inp.device)

    if M == 0 or K == 0:
        # A spectator dimension is empty: the output is a valid empty tensor.
        if not keepdim:
            out = torch.squeeze(out, d)
        return out
    if N == 0:
        # The product over an empty dimension is the identity, 1 (unlike max/min
        # which have no identity). Fill the output rather than launching.
        out.fill_(1)
        if not keepdim:
            out = torch.squeeze(out, d)
        return out

    inp = inp.contiguous()
    with torch_device_fn.device(inp.device):
        if K == 1:
            grid = (M, 1, 1)
            prod_dim_kernel_inner[grid](out, inp, M, N)
        else:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            prod_dim_kernel_non_inner[grid](out, inp, M, N, K)
    if not keepdim:
        out = torch.squeeze(out, d)
    return out
