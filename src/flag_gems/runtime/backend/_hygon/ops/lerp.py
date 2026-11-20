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
from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(__name__)


@triton.jit
def _lerp_tensor_opmath(input, end, weight):
    is_double: tl.constexpr = (
        input.type.scalar.is_fp64()
        or end.type.scalar.is_fp64()
        or weight.type.scalar.is_fp64()
    )
    compute_dtype: tl.constexpr = tl.float64 if is_double else tl.float32
    input = input.to(compute_dtype)
    end = end.to(compute_dtype)
    weight = weight.to(compute_dtype)
    delta = end - input
    # Keep the multiply separate from the add/subtract, as in PyTorch.
    # Contracting the tail into FMA changes signed zeros when weight == 1.
    multiply: tl.constexpr = (
        "v_mul_f64 $0, $1, $2" if is_double else "v_mul_f32 $0, $1, $2"
    )
    head_product = tl.inline_asm_elementwise(
        multiply,
        constraints="=v,v,v",
        args=[weight, delta],
        dtype=delta.dtype,
        is_pure=True,
        pack=1,
    )
    tail_product = tl.inline_asm_elementwise(
        multiply,
        constraints="=v,v,v",
        args=[delta, 1.0 - weight],
        dtype=delta.dtype,
        is_pure=True,
        pack=1,
    )
    return tl.where(tl.abs(weight) < 0.5, input + head_product, end - tail_product)


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def lerp_tensor_kernel(input, end, weight):
    return _lerp_tensor_opmath(input, end, weight)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit(do_not_specialize=["weight"])
def lerp_scalar_kernel_head(input, end, weight):
    return input + weight * (end - input)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit(do_not_specialize=["weight"])
def lerp_scalar_kernel_tail(input, end, weight):
    return end - (end - input) * (1 - weight)


@libentry()
@triton.jit
def _lerp_tensor_contiguous_kernel(
    input_ptr,
    end_ptr,
    weight_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if N > 2147483647:
        offsets = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
    if N % BLOCK == 0:
        start = tl.load(input_ptr + offsets).to(tl.float32)
        end = tl.load(end_ptr + offsets).to(tl.float32)
        weight = tl.load(weight_ptr + offsets).to(tl.float32)
    else:
        mask = offsets < N
        start = tl.load(input_ptr + offsets, mask, other=0).to(tl.float32)
        end = tl.load(end_ptr + offsets, mask, other=0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask, other=0).to(tl.float32)

    # Match PyTorch's opmath precision and cancellation-aware branch.
    result = _lerp_tensor_opmath(start, end, weight)
    if N % BLOCK == 0:
        tl.store(out_ptr + offsets, result)
    else:
        tl.store(out_ptr + offsets, result, offsets < N)


def lerp_tensor(input, end, weight):
    logger.debug("GEMS_HYGON LERP_TENSOR")
    dtype = input.dtype
    if (
        dtype in (torch.float16, torch.bfloat16, torch.float32)
        and dtype == end.dtype == weight.dtype
        and input.shape == end.shape == weight.shape
        and input.is_contiguous()
        and end.is_contiguous()
        and weight.is_contiguous()
    ):
        out = torch.empty_like(input)
        count = input.numel()
        if count:
            if count <= 8192:
                block, warps = 128, 2
            else:
                block, warps = 2048 // input.element_size(), 4
                if count >= 1 << 28:
                    block *= 2
                    warps = 8
            with torch_device_fn.device(input.device):
                _lerp_tensor_contiguous_kernel[(triton.cdiv(count, block),)](
                    input, end, weight, out, count, BLOCK=block, num_warps=warps
                )
        return out
    return lerp_tensor_kernel(input, end, weight)


def lerp_tensor_(input, end, weight):
    logger.debug("GEMS_HYGON LERP_INPLACE_TENSOR")
    return lerp_tensor_kernel(input, end, weight, out0=input)


def lerp_scalar(input, end, weight):
    logger.debug("GEMS_HYGON LERP_TENSOR")
    if weight < 0.5:
        out = lerp_scalar_kernel_head(input, end, weight)
    else:
        out = lerp_scalar_kernel_tail(input, end, weight)
    return out


def lerp_scalar_(input, end, weight):
    logger.debug("GEMS_HYGON LERP_INPLACE_TENSOR")
    if weight < 0.5:
        return lerp_scalar_kernel_head(input, end, weight, out0=input)
    else:
        return lerp_scalar_kernel_tail(input, end, weight, out0=input)
