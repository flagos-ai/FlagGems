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

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def _unscale_check_kernel(
    x_ptr,
    found_inf_ptr,
    inv_scale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    program_stride = tl.num_programs(0) * BLOCK_SIZE
    scale = tl.load(inv_scale_ptr)
    if IS_FP16:
        scale = scale.to(tl.float16)
    found = 0

    while block_start < n_elements:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = x * scale
        tl.store(x_ptr + offsets, y, mask=mask)
        max_abs = tl.max(tl.abs(x), axis=0)
        if IS_FP16:
            finite_limit = tl.full((), 65504.0, tl.float16)
            non_finite = ~(max_abs <= finite_limit)
        elif IS_FP32:
            finite_limit = tl.full((), 3.4028234663852886e38, tl.float32)
            non_finite = ~(max_abs <= finite_limit)
        else:
            non_finite = (max_abs != max_abs) | (max_abs == float("inf"))
        found |= non_finite.to(tl.int32)
        block_start += program_stride

    tl.store(found_inf_ptr, 1.0, mask=found != 0)


@triton.jit
def _unscale_check_two_kernel(
    x0_ptr,
    x1_ptr,
    found_inf_ptr,
    inv_scale_ptr,
    n0,
    n1,
    BLOCK_SIZE: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    program_id = tl.program_id(0)
    program_stride = tl.num_programs(0) * BLOCK_SIZE
    scale = tl.load(inv_scale_ptr)
    if IS_FP16:
        scale = scale.to(tl.float16)
    found = 0

    block_start = program_id * BLOCK_SIZE
    while block_start < n0:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n0
        x = tl.load(x0_ptr + offsets, mask=mask, other=0.0)
        tl.store(x0_ptr + offsets, x * scale, mask=mask)
        max_abs = tl.max(tl.abs(x), axis=0)
        if IS_FP16:
            finite_limit = tl.full((), 65504.0, tl.float16)
            non_finite = ~(max_abs <= finite_limit)
        elif IS_FP32:
            finite_limit = tl.full((), 3.4028234663852886e38, tl.float32)
            non_finite = ~(max_abs <= finite_limit)
        else:
            non_finite = (max_abs != max_abs) | (max_abs == float("inf"))
        found |= non_finite.to(tl.int32)
        block_start += program_stride

    block_start = program_id * BLOCK_SIZE
    while block_start < n1:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n1
        x = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
        tl.store(x1_ptr + offsets, x * scale, mask=mask)
        max_abs = tl.max(tl.abs(x), axis=0)
        if IS_FP16:
            finite_limit = tl.full((), 65504.0, tl.float16)
            non_finite = ~(max_abs <= finite_limit)
        elif IS_FP32:
            finite_limit = tl.full((), 3.4028234663852886e38, tl.float32)
            non_finite = ~(max_abs <= finite_limit)
        else:
            non_finite = (max_abs != max_abs) | (max_abs == float("inf"))
        found |= non_finite.to(tl.int32)
        block_start += program_stride

    tl.store(found_inf_ptr, 1.0, mask=found != 0)


def _amp_foreach_non_finite_check_and_unscale_(tensors, found_inf, inv_scale):
    if len(tensors) == 2 and tensors[0].dtype == tensors[1].dtype:
        tensor0, tensor1 = tensors
        n0 = tensor0.numel()
        n1 = tensor1.numel()
        max_elements = max(n0, n1)
        if max_elements == 0:
            return
        if max_elements <= 8192:
            block_size = 4096 if tensor0.dtype == torch.float16 else 256
            grid = (triton.cdiv(max_elements, block_size),)
        else:
            num_vectorcore = driver.active.utils.get_device_properties(
                torch.npu.current_device()
            )["num_vectorcore"]
            block_size = (
                16384
                if tensor0.element_size() == 2 and tensor1.element_size() == 2
                else 8192
            )
            grid = (min(triton.cdiv(max_elements, block_size), num_vectorcore),)
        _unscale_check_two_kernel[grid](
            tensor0,
            tensor1,
            found_inf,
            inv_scale,
            n0,
            n1,
            BLOCK_SIZE=block_size,
            IS_FP16=tensor0.dtype == torch.float16,
            IS_FP32=tensor0.dtype == torch.float32,
        )
        return

    num_vectorcore = None
    for tensor in tensors:
        n_elements = tensor.numel()
        if n_elements == 0:
            continue
        if n_elements <= 8192:
            block_size = 256
            grid = (triton.cdiv(n_elements, block_size),)
        else:
            if num_vectorcore is None:
                num_vectorcore = driver.active.utils.get_device_properties(
                    torch.npu.current_device()
                )["num_vectorcore"]
            block_size = 16384 if tensor.element_size() == 2 else 8192
            grid = (min(triton.cdiv(n_elements, block_size), num_vectorcore),)
        _unscale_check_kernel[grid](
            tensor,
            found_inf,
            inv_scale,
            n_elements,
            BLOCK_SIZE=block_size,
            IS_FP16=tensor.dtype == torch.float16,
            IS_FP32=tensor.dtype == torch.float32,
        )
