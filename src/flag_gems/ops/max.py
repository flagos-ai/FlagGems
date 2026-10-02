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
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)


@libentry()
@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = ext.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    min_value = get_dtype_min(dtype)
    result_value = tl.full([BLOCK_M], value=min_value, dtype=acc_type)
    result_index = tl.zeros([BLOCK_M], dtype=tl.int64)
    for i in range(0, N, BLOCK_N):
        n_offset = i + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        # set mask
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        max_value, max_index = tl.max(inp_vals, axis=1, return_indices=True)
        update_mask = max_value > result_value
        result_value = tl.where(update_mask, max_value, result_value)
        result_index = tl.where(update_mask, i + max_index, result_index)
    mask1 = m_offset < M
    offset_index = m_offset
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def max_dim_kernel_non_inner(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # Split the input into (M, N, K) and reduce over N with a strided read,
    # avoiding the dim_compress copy. Mirrors argmax_kernel_non_inner but also
    # stores the max value. Ties resolve to the lowest index, matching torch.
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k_offset = pid_k * TILE_K + tl.arange(0, TILE_K)

    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    min_value = get_dtype_min(cdtype)

    if ONE_TILE_PER_CTA:
        n_offset = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offset[:, None] * K + k_offset
        mask = k_offset < K and n_offset[:, None] < N
        inp_vals = tl.load(inp + offset, mask=mask, other=min_value)
        local_max, local_argmax = tl.max(
            inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
        )
        offset_index = pid_m * K + k_offset
        mask1 = k_offset < K
        tl.store(out_value + offset_index, local_max, mask=mask1)
        tl.store(out_index + offset_index, local_argmax, mask=mask1)
    else:
        max_values = tl.full([TILE_K], dtype=cdtype, value=min_value)
        argmax_values = tl.full([TILE_K], dtype=tl.int64, value=0)
        for start_n in range(0, N, TILE_N):
            n_offset = start_n + tl.arange(0, TILE_N)
            offset = pid_m * N * K + n_offset[:, None] * K + k_offset
            mask = k_offset < K and n_offset[:, None] < N
            inp_vals = tl.load(inp + offset, mask=mask, other=min_value)
            local_max, local_argmax = tl.max(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(update, start_n + local_argmax, argmax_values)
        offset_index = pid_m * K + k_offset
        mask1 = k_offset < K
        tl.store(out_value + offset_index, max_values, mask=mask1)
        tl.store(out_index + offset_index, argmax_values, mask=mask1)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def max_dim_kernel_inner(
    inp,
    out_value,
    out_index,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)

    if ONE_TILE_PER_CTA:
        n_offset = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offset
        mask = n_offset < N
        inp_vals = tl.load(inp + offset, mask=mask, other=min_value)
        local_max, local_argmax = tl.max(
            inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
        )
        tl.store(out_value + pid_m, local_max)
        tl.store(out_index + pid_m, local_argmax)
    else:
        max_values = min_value
        argmax_values = 0
        loop_time = N // TILE_N
        remainder = N % TILE_N
        for start_n in range(0, loop_time):
            n_offset = start_n * TILE_N + tl.arange(0, TILE_N)
            offset = pid_m * N + n_offset
            inp_vals = tl.load(inp + offset)
            local_max, local_argmax = tl.max(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(
                update, start_n * TILE_N + local_argmax, argmax_values
            )
        if remainder:
            n_offset = loop_time * TILE_N + tl.arange(0, TILE_N)
            offset = pid_m * N + n_offset
            mask = n_offset < N
            inp_vals = tl.load(inp + offset, mask=mask, other=min_value)
            local_max, local_argmax = tl.max(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(
                update, loop_time * TILE_N + local_argmax, argmax_values
            )
        tl.store(out_value + pid_m, max_values)
        tl.store(out_index + pid_m, argmax_values)


def max(inp):
    logger.debug("GEMS MAX")
    inp = inp.contiguous()
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        max_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        max_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def max_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS MAX DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim

    # Single-dim reduction: split into (M, N, K) and reduce over N with a
    # strided kernel, avoiding the dim_compress copy. K == 1 means the reduced
    # dim is innermost (contiguous); K > 1 means it is a middle or outer dim,
    # where the copy used to dominate the runtime.
    N = shape[dim]
    if N == 0:
        # Reducing over an empty dimension has no maximum; torch raises here.
        raise IndexError("max(): Expected reduction dim to have non-zero size.")
    M = math.prod(shape[:dim])
    K = math.prod(shape[dim + 1 :])
    shape[dim] = 1

    out_value = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    inp = inp.contiguous()
    with torch_device_fn.device(inp.device):
        if K == 1:
            grid = lambda meta: (M, 1, 1)
            max_dim_kernel_inner[grid](inp, out_value, out_index, M, N)
        else:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            max_dim_kernel_non_inner[grid](inp, out_value, out_index, M, N, K)
    Max_out = namedtuple("max", ["values", "indices"])
    out = Max_out(values=out_value, indices=out_index)
    return out
