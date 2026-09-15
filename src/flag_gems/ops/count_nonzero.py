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
from functools import reduce

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def count_nonzero_kernel_1(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int64)
    nonzero_count = tl.sum(is_nonzero, axis=0)
    tl.atomic_add(out_ptr, nonzero_count)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("count_nonzero"), key=["numel"])
@triton.jit
def count_nonzero_kernel(x_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr):
    pid_x = ext.program_id(0)

    nonzero_count = tl.full((), value=0, dtype=out_ptr.dtype.element_ty)
    for start_n in range(0, N, BLOCK_SIZE):
        cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        offset = pid_x * N + cols_offsets
        mask = (offset < numel) & (cols_offsets < N)
        x = tl.load(x_ptr + offset, mask=mask, other=0)
        is_nonzero = (x != 0).to(tl.int64)
        nonzero_count += tl.sum(is_nonzero)

    tl.store(out_ptr + pid_x, nonzero_count)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("count_nonzero"), key=["numel"])
@triton.jit
def count_nonzero_combin_kernel_1(x_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr):
    pid_x = ext.program_id(0)
    nonzero_count = tl.full((), value=0, dtype=out_ptr.dtype.element_ty)
    for start_n in range(0, N, BLOCK_SIZE):
        cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        offset = pid_x * N + cols_offsets
        mask = (offset < numel) & (cols_offsets < N)
        x = tl.load(x_ptr + offset, mask=mask, other=0)
        nonzero_count += tl.sum(x)
    tl.store(out_ptr + pid_x, nonzero_count)


@libentry()
@triton.jit
def count_nonzero_combin_kernel(
    x_ptr, combin_ptr, N, combin_N, numel, BLOCK_SIZE: tl.constexpr
):
    pid_x = ext.program_id(0)
    pid_y = ext.program_id(1)
    cols_offsets = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset = pid_x * N + cols_offsets
    mask = (offset < numel) & (cols_offsets < N)
    x = tl.load(x_ptr + offset, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int64)
    nonzero_count = tl.sum(is_nonzero)
    tl.store(combin_ptr + pid_x * combin_N + pid_y, nonzero_count)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def count_nonzero_dim_kernel_non_inner(
    out_ptr,
    x_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # Split the input into (M, N, K) and count nonzeros over N with a strided
    # read, avoiding the dim_compress copy. Masked lanes load 0 and contribute
    # nothing to the count.
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k_idx = pid_k * TILE_K + tl.arange(0, TILE_K)
    k_offsets = k_idx[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        offsets = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        count = tl.sum((x != 0).to(tl.int64), axis=0)
    else:
        acc = tl.zeros([TILE_N, TILE_K], dtype=tl.int64)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            x = tl.load(x_ptr + offsets, mask=mask, other=0)
            acc += (x != 0).to(tl.int64)
        count = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid_m * K + k_idx, count, mask=k_idx < K)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def count_nonzero_dim_kernel_inner(
    out_ptr,
    x_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(x_ptr + pid_m * N + n_offsets, mask=mask, other=0)
        count = tl.sum((x != 0).to(tl.int64), axis=0)
    else:
        acc = tl.zeros([TILE_N], dtype=tl.int64)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            x = tl.load(x_ptr + pid_m * N + n_offsets, mask=mask, other=0)
            acc += (x != 0).to(tl.int64)
        count = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid_m, count)


def count_nonzero(x, dim=None):
    logger.debug("GEMS COUNT NONZERO")

    if x.is_sparse:
        x = x.to_dense()

    if dim is not None:
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        dim = dim % x.ndim
        # Single-dim reduction: split into (M, N, K) and count over N with a
        # strided kernel, avoiding the dim_compress copy. K == 1 means the
        # reduced dim is innermost (contiguous); K > 1 means it is a middle or
        # outer dim, where the copy used to dominate the runtime.
        shape = list(x.shape)
        N = shape[dim]
        M = reduce(lambda a, b: a * b, shape[:dim], 1)
        K = reduce(lambda a, b: a * b, shape[dim + 1 :], 1)
        out_shape = shape[:dim] + shape[dim + 1 :]
        out = torch.zeros(out_shape, dtype=torch.int64, device=x.device)
        if M == 0 or K == 0 or N == 0:
            # An empty reduction dim counts to 0 (already zero-filled); an empty
            # spectator dim gives an empty output. Both match torch.
            return out
        x = x.contiguous()
        with torch_device_fn.device(x.device):
            if K == 1:
                grid = (M, 1, 1)
                count_nonzero_dim_kernel_inner[grid](out, x, M, N)
            else:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                count_nonzero_dim_kernel_non_inner[grid](out, x, M, N, K)
        return out
    else:
        x = x.contiguous().flatten()
        numel = x.numel()

        out = torch.zeros(1, dtype=torch.int64, device=x.device)

        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

        count_nonzero_kernel_1[grid](x, out, numel, BLOCK_SIZE=BLOCK_SIZE)

        return out[0]
