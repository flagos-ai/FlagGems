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
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def min_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    max_value = get_dtype_max(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=max_value)
    min_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, min_val)


@libentry()
@triton.jit
def min_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    min_val = tl.min(mid_val)
    tl.store(out, min_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def min_kernel(
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
    # you just cannot create a function that return a tl.dtype in triton lang
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_value = get_dtype_max(dtype)
    min_values = tl.full([BLOCK_M], dtype=acc_type, value=max_value)
    argmin_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value)
        local_min, local_argmin = tl.min(inp_vals, 1, return_indices=True)
        # if return indices is not supported, call a tl.argmax in addition
        # local_argmin = tl.argmin(inp_vals, 1)
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(update, start_n + local_argmin, argmin_values)

    offset_index = m_offset
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_value_ptrs, min_values, mask=mask1)
    tl.store(out_index_ptrs, argmin_values, mask=mask1)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def min_dim_kernel_non_inner(
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
    # avoiding the dim_compress copy. Ties resolve to the lowest index,
    # matching torch.
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k_offset = pid_k * TILE_K + tl.arange(0, TILE_K)

    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    max_value = get_dtype_max(cdtype)

    if ONE_TILE_PER_CTA:
        n_offset = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offset[:, None] * K + k_offset
        mask = k_offset < K and n_offset[:, None] < N
        inp_vals = tl.load(inp + offset, mask=mask, other=max_value)
        local_min, local_argmin = tl.min(
            inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
        )
        offset_index = pid_m * K + k_offset
        mask1 = k_offset < K
        tl.store(out_value + offset_index, local_min, mask=mask1)
        tl.store(out_index + offset_index, local_argmin, mask=mask1)
    else:
        min_values = tl.full([TILE_K], dtype=cdtype, value=max_value)
        argmin_values = tl.full([TILE_K], dtype=tl.int64, value=0)
        for start_n in range(0, N, TILE_N):
            n_offset = start_n + tl.arange(0, TILE_N)
            offset = pid_m * N * K + n_offset[:, None] * K + k_offset
            mask = k_offset < K and n_offset[:, None] < N
            inp_vals = tl.load(inp + offset, mask=mask, other=max_value)
            local_min, local_argmin = tl.min(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            update = local_min < min_values
            min_values = tl.where(update, local_min, min_values)
            argmin_values = tl.where(update, start_n + local_argmin, argmin_values)
        offset_index = pid_m * K + k_offset
        mask1 = k_offset < K
        tl.store(out_value + offset_index, min_values, mask=mask1)
        tl.store(out_index + offset_index, argmin_values, mask=mask1)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def min_dim_kernel_inner(
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
    max_value = get_dtype_max(dtype)

    if ONE_TILE_PER_CTA:
        n_offset = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offset
        mask = n_offset < N
        inp_vals = tl.load(inp + offset, mask=mask, other=max_value)
        local_min, local_argmin = tl.min(
            inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
        )
        tl.store(out_value + pid_m, local_min)
        tl.store(out_index + pid_m, local_argmin)
    else:
        min_values = max_value
        argmin_values = 0
        loop_time = N // TILE_N
        remainder = N % TILE_N
        for start_n in range(0, loop_time):
            n_offset = start_n * TILE_N + tl.arange(0, TILE_N)
            offset = pid_m * N + n_offset
            inp_vals = tl.load(inp + offset)
            local_min, local_argmin = tl.min(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            update = local_min < min_values
            min_values = tl.where(update, local_min, min_values)
            argmin_values = tl.where(
                update, start_n * TILE_N + local_argmin, argmin_values
            )
        if remainder:
            n_offset = loop_time * TILE_N + tl.arange(0, TILE_N)
            offset = pid_m * N + n_offset
            mask = n_offset < N
            inp_vals = tl.load(inp + offset, mask=mask, other=max_value)
            local_min, local_argmin = tl.min(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            update = local_min < min_values
            min_values = tl.where(update, local_min, min_values)
            argmin_values = tl.where(
                update, loop_time * TILE_N + local_argmin, argmin_values
            )
        tl.store(out_value + pid_m, min_values)
        tl.store(out_index + pid_m, argmin_values)


def min(inp):
    logger.debug("GEMS MIN")
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        min_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        min_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def min_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS MIN DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim

    # Single-dim reduction: split into (M, N, K) and reduce over N with a
    # strided kernel, avoiding the dim_compress copy. K == 1 means the reduced
    # dim is innermost (contiguous); K > 1 means it is a middle or outer dim,
    # where the copy used to dominate the runtime.
    N = shape[dim]
    if N == 0:
        # Reducing over an empty dimension has no minimum; torch raises here.
        raise IndexError("min(): Expected reduction dim to have non-zero size.")
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
            min_dim_kernel_inner[grid](inp, out_value, out_index, M, N)
        else:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            min_dim_kernel_non_inner[grid](inp, out_value, out_index, M, N, K)
    Min_out = namedtuple("min", ["values", "indices"])
    out = Min_out(values=out_value, indices=out_index)
    return out
