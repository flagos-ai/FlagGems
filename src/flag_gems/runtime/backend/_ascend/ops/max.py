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

import importlib
import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)
MAX_GRID_DIM = 65535
MaxOut = namedtuple("max", ["values", "indices"])
_generic_max = importlib.import_module("flag_gems.ops.max")


@libentry()
@triton.jit
def max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0).to(tl.int64)
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
@triton.autotune(
    configs=runtime.get_tuned_config("max"),
    key=[
        "M",
        "N",
    ],
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0).to(tl.int64)
    pid_k = ext.program_id(1).to(tl.int64)
    num_pid_m = tl.num_programs(0).to(tl.int64)
    num_pid_k = tl.num_programs(1).to(tl.int64)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    min_value = get_dtype_min(dtype)

    for m_start in range(pid_m * BLOCK_M, M, num_pid_m * BLOCK_M):
        m_offset = m_start + tl.arange(0, BLOCK_M)
        mask1 = m_offset < M
        for k_offset in range(pid_k, K, num_pid_k):
            result_value = tl.full([BLOCK_M], value=min_value, dtype=acc_type)
            result_index = tl.zeros([BLOCK_M], dtype=tl.int64)
            for i in range(0, N, BLOCK_N):
                n_offset = i + tl.arange(0, BLOCK_N)
                offset = m_offset[:, None] * N * K + n_offset[None, :] * K + k_offset
                mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
                inp_ptrs = inp + offset
                inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
                if dtype is tl.int64:
                    inp_vals = tl.where(mask, inp_vals, min_value)
                max_value, max_index = tl.max(
                    inp_vals,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                update_mask = max_value > result_value
                result_value = tl.where(update_mask, max_value, result_value)
                result_index = tl.where(update_mask, i + max_index, result_index)
            offset_index = m_offset * K + k_offset
            out_value_ptrs = out_value + offset_index
            out_index_ptrs = out_index + offset_index

            tl.store(out_value_ptrs, result_value, mask=mask1)
            tl.store(out_index_ptrs, result_index, mask=mask1)


def _max_impl(inp):
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


def _max_dim_impl(inp, dim=None, keepdim=False):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    if M == 0 or K == 0:
        return MaxOut(values=out_value, indices=out_index)

    def grid(meta):
        axis0 = triton.cdiv(M, meta["BLOCK_M"])
        axis0 = axis0 if axis0 <= MAX_GRID_DIM else MAX_GRID_DIM
        axis0 = axis0 if axis0 >= 1 else 1
        max_axis1 = MAX_GRID_DIM // axis0
        axis1 = K if K <= max_axis1 else max_axis1
        axis1 = axis1 if axis1 >= 1 else 1
        return (
            axis0,
            axis1,
        )

    with torch_device_fn.device(inp.device):
        max_kernel[grid](inp, out_value, out_index, M, N, K)
    return MaxOut(values=out_value, indices=out_index)


class MaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        inp = inp.contiguous()
        out = _max_impl(inp)
        ctx.save_for_backward(inp, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None
        inp, result = ctx.saved_tensors
        return _generic_max.evenly_distribute_backward(grad_output, inp, result)


class MaxDimFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dim, keepdim):
        inp = inp.contiguous()
        values, indices = _max_dim_impl(inp, dim, keepdim)
        ctx.save_for_backward(inp, values, indices)
        ctx.dim = dim % inp.ndim
        ctx.mark_non_differentiable(indices)
        return values, indices

    @staticmethod
    def backward(ctx, grad_values, grad_indices):
        if not ctx.needs_input_grad[0]:
            return None, None, None
        inp, values, indices = ctx.saved_tensors
        return (
            _generic_max.max_backward(grad_values, inp, values, indices, ctx.dim),
            None,
            None,
        )


def max(inp):
    logger.debug("GEMS_ASCEND MAX")
    if torch.is_grad_enabled() and inp.requires_grad and inp.is_floating_point():
        return MaxFunction.apply(inp)
    return _max_impl(inp)


def max_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ASCEND MAX_DIM")
    if torch.is_grad_enabled() and inp.requires_grad and inp.is_floating_point():
        values, indices = MaxDimFunction.apply(inp, dim, keepdim)
        return MaxOut(values=values, indices=indices)
    return _max_dim_impl(inp, dim=dim, keepdim=keepdim)
