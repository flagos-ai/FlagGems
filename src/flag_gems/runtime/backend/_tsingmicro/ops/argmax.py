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

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.limits import get_dtype_min
from flag_gems.utils.shape_utils import can_use_int32_index

logger = logging.getLogger(__name__)


def cfggen_reduce_op():
    return runtime.get_tuned_config("argmax_kernel_1")


@libentry()
@triton.jit
def argmax_kernel_once(
    inp,
    out,
    M: tl.constexpr,
):
    offset = tl.arange(0, M)
    inp_val = tl.load(inp + offset)
    index_val = tl.argmax(inp_val, axis=0)
    tl.store(out, index_val.to(tl.int64))


@libentry()
# @libtuner(
#     configs=cfggen_reduce_op(),
#     key=["M"],
#     strategy=["log"],
# )
@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    num_jobs = tl.num_programs(axis=0)

    size_per_job = (M + num_jobs - 1) // num_jobs
    start_idx = pid * size_per_job
    end_idx = min(start_idx + size_per_job, M)

    min_value = get_dtype_min(inp.type.element_ty)
    max_tmp = min_value
    index_tmp = 0
    if INT64_INDEX:
        index_tmp = index_tmp.to(tl.int64)
    for off in range(start_idx, end_idx, BLOCK_SIZE):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < end_idx
        inp_val = tl.load(inp + offset, mask=mask, other=min_value)
        max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
        if max_val > max_tmp:
            max_tmp = max_val.to(tl.float32)
            index_tmp = max_index + off
    mid_value_ptr = mid_value + pid
    max_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, max_tmp)
    tl.store(max_index_ptr, index_tmp)


@libentry()
@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    min_value = get_dtype_min(mid_value.type.element_ty)
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mid_val = tl.load(mid_ptrs, mask=(offset < mid_size), other=min_value)
    index_val = tl.argmax(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("argmax"),
    key=[
        "M",
        "N",
    ],
    strategy=["log", "log"],
)
@triton.jit
def argmax_kernel(
    inp,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    # set offset
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if INT64_INDEX:
        pid_m = pid_m.to(tl.int64)
        pid_k = pid_k.to(tl.int64)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    min_value = get_dtype_min(dtype)
    max_values = tl.full([BLOCK_M], dtype=acc_type, value=min_value)
    argmax_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        local_max, local_argmax = tl.max(
            inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
        )
        # if return indices is not supported, call a tl.argmax in addition
        # local_argmax = tl.argmax(inp_vals, 1)
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, start_n + local_argmax, argmax_values)

    offset_index = m_offset * K + pid_k
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_index_ptrs, argmax_values, mask=mask1)


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_TSINGMICRO ARGMAX")
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype

        use_int64_index = not can_use_int32_index(inp)

        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        if M <= 65536:
            with torch_device_fn.device(inp.device):
                argmax_kernel_once[(1, 1, 1)](inp, out, M)
        else:
            block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
            mid_size = triton.cdiv(M, block_size)
            block_mid = triton.next_power_of_2(mid_size)
            mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
            mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
            with torch_device_fn.device(inp.device):
                argmax_kernel_1[(mid_size,)](
                    inp,
                    mid_value,
                    mid_index,
                    M,
                    block_size,
                    INT64_INDEX=use_int64_index,
                )
                argmax_kernel_2[(1, 1, 1)](
                    mid_value, mid_index, out, mid_size, block_mid
                )
        return out
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        shape = inp.shape
        dim = dim % inp.ndim
        if inp.numel() == 0:
            out_shape = list(shape)
            if keepdim:
                out_shape[dim] = 1
            else:
                del out_shape[dim]
            return torch.zeros(out_shape, dtype=torch.int64, device=inp.device)
        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N

        inp = inp.contiguous()
        use_int64_index = not can_use_int32_index(inp)

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        with torch_device_fn.device(inp.device):
            argmax_kernel[grid](inp, out_index, M, N, K, INT64_INDEX=use_int64_index)

        return out_index
