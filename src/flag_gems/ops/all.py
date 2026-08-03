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
from functools import reduce

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# torch.all: Tests if all elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if all elements in input evaluate to non-zero value
# In triton function, test if all elements in input evaluate to non-zero value is ok.


@triton.jit
def reduce_all(a, b):
    return a and b


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def all_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=1.0)
        _all = _all and (a != 0)
    all = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    tl.store(out, all[:, None], row_mask)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def all_dim_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # Boolean AND over the reduced dim is the elementwise minimum of 0/1 values,
    # whose identity is 1. This mirrors amax's structure with maximum -> minimum.
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        inp = (tl.load(input_ptr + inp_offset, mask=mask, other=1) != 0).to(tl.int32)
        out = tl.min(inp, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out != 0, mask=k_offsets < K)
    else:
        acc = tl.full([TILE_N, TILE_K], value=1, dtype=tl.int32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = (tl.load(input_ptr + inp_offsets, mask=mask, other=1) != 0).to(
                tl.int32
            )
            acc = tl.minimum(acc, inp)
        out = tl.min(acc, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out != 0, mask=k_offsets < K)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def all_dim_kernel_inner(
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
        inp = (tl.load(input_ptr + inp_offset, mask=mask, other=1) != 0).to(tl.int32)
        out = tl.min(inp, axis=0)
        tl.store(output_ptr + pid_m, out != 0)
    else:
        acc = tl.full([TILE_N], value=1, dtype=tl.int32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_offsets = pid_m * N + n_offsets
            mask = n_offsets < N
            inp = (tl.load(input_ptr + inp_offsets, mask=mask, other=1) != 0).to(
                tl.int32
            )
            acc = tl.minimum(acc, inp)
        out = tl.min(acc, axis=0)
        tl.store(output_ptr + pid_m, out != 0)


@libentry()
@triton.jit
def all_kernel_1(
    inp,
    mid,
    n_elements,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0)
    all_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_all)
    mid_ptr = mid + pid
    tl.store(mid_ptr, all_val)


@libentry()
@triton.jit
def all_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=1).to(tl.int1)
    all_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_all)
    tl.store(out, all_val)


def all(inp):
    logger.debug("GEMS ALL")
    n_elements = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    with torch_device_fn.device(inp.device):
        all_kernel_1[(mid_size, 1)](inp, mid, n_elements, mid_size, block_size)
        all_kernel_2[(1, 1)](mid, out, mid_size, block_mid)

    return out


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS ALL DIM")
    shape = list(inp.shape)
    if dim is None:
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim

        # Single-dim reduction: split into (M, N, K) and reduce over N with a
        # strided kernel, avoiding the dim_compress copy. K == 1 means the
        # reduced dim is innermost (contiguous); K > 1 means it is a middle or
        # outer dim, where the copy used to dominate the runtime.
        N = shape[dim]
        M = reduce(lambda x, y: x * y, shape[:dim], 1)
        K = reduce(lambda x, y: x * y, shape[dim + 1 :], 1)
        shape[dim] = 1
        out = torch.empty(shape, dtype=torch.bool, device=inp.device)
        if M * N * K == 0:
            # An empty reduction dim reduces to the AND identity (True), and an
            # empty spectator dim gives an empty output. Both match torch, and
            # torch.empty is already filled for the empty-output case.
            if N == 0 and M * K > 0:
                out.fill_(True)
            if not keepdim:
                out = out.squeeze(dim=dim)
            return out
        inp = inp.contiguous()
        with torch_device_fn.device(inp.device):
            if K == 1:
                grid = (M, 1, 1)
                all_dim_kernel_inner[grid](out, inp, M, N)
            else:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                all_dim_kernel_non_inner[grid](out, inp, M, N, K)
        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS ALL DIMS")

    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    # Cast to bool to avoid float16/bfloat16 comparison issues in kernel
    if inp.dtype != torch.bool:
        inp = inp.bool()

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        all_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
