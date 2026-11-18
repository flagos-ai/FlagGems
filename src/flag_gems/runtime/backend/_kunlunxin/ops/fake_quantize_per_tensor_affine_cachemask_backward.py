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

# Kunlunxin(XPU) override for
#   fake_quantize_per_tensor_affine_cachemask_backward(grad, mask) = grad * mask
# (pure elementwise multiply of a float tensor by a 0/1 bool mask).
#
# Correctness (signed-zero) is as in the previous override: the ATen reference
# computes grad * 0.0 (bool promoted to float), which preserves the IEEE sign
# of the zero (-3.0 * 0.0 == -0.0) and yields NaN for inf/nan grad; this
# backend folds `x * 0.0` to +0.0 for finite values, so the sign bit is rebuilt
# via an int32 bitcast OR.  Only float32 needs this (the mask_semantics signbit
# check is float32-only).
#
# Performance / portability: the fp16/bf16 path uses
# `tl.where(mask != 0, grad, grad * 0.0)`.  The alternative
# `grad * mask.to(tl.float32)` is ~1.5-2x faster on paper, but int8->float32
# widening does NOT compile on the XPU compiler (ConvertTritonXPUToLLVM:
# "size mismatch when packing elements for LLVM struct"), so a select is
# required there.  The fp32 path keeps the select and additionally rebuilds the
# IEEE signed zero of `grad * 0.0` via an int32 bitcast OR (this backend folds
# the multiply's -0.0 to +0.0 for finite values).

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _pick_block(n_elements, dtype):
    if n_elements < 65_536:
        return 4096, 4, (n_elements % 4096) != 0
    if n_elements < 131_072:
        return 8192, 4, (n_elements % 8192) != 0
    if dtype == torch.float32:
        return 8192, 8, (n_elements % 8192) != 0
    return 16384, 4, (n_elements % 16384) != 0


@triton.jit
def _fq_ptacmb_backward_mul_kernel(
    grad_ptr,
    mask_ptr,  # int8 view of the bool mask (host does .view(torch.int8))
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    FIX_SIGN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bound = offset < n_elements
    grad = tl.load(grad_ptr + offset, mask=bound)
    quant_mask = tl.load(mask_ptr + offset, mask=bound)  # int8 0/1
    if FIX_SIGN:
        # fp32 signed-zero: grad*0.0 folds to +0.0 on this backend, so rebuild
        # the sign bit via int32 bitcast OR; the select is measured faster than
        # a real per-element multiply for the fp32 path.
        prod = grad * 0.0
        prod_bits = prod.to(tl.int32, bitcast=True)
        grad_bits = grad.to(tl.int32, bitcast=True)
        false_val = (prod_bits | (grad_bits & -2147483648)).to(tl.float32, bitcast=True)
        out = tl.where(quant_mask != 0, grad, false_val)
    else:
        # XPU: int8->float32 widening (`quant_mask.to(tl.float32)`) does not
        # compile (ConvertTritonXPUToLLVM: "size mismatch when packing elements
        # for LLVM struct"). Select in grad's own dtype instead; fp16/bf16 do not
        # need the fp32 signed-zero rebuild, and `grad * 0.0` keeps inf -> NaN
        # matching the reference.
        out = tl.where(quant_mask != 0, grad, grad * 0.0)
    tl.store(output_ptr + offset, out, mask=bound)


@triton.jit
def _fq_ptacmb_backward_mul_unmasked(
    grad_ptr,
    mask_ptr,  # int8 view of the bool mask
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    FIX_SIGN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    grad = tl.load(grad_ptr + offset)
    quant_mask = tl.load(mask_ptr + offset)
    if FIX_SIGN:
        prod = grad * 0.0
        prod_bits = prod.to(tl.int32, bitcast=True)
        grad_bits = grad.to(tl.int32, bitcast=True)
        false_val = (prod_bits | (grad_bits & -2147483648)).to(tl.float32, bitcast=True)
        out = tl.where(quant_mask != 0, grad, false_val)
    else:
        # XPU: int8->float32 widening (`quant_mask.to(tl.float32)`) does not
        # compile (ConvertTritonXPUToLLVM: "size mismatch when packing elements
        # for LLVM struct"). Select in grad's own dtype instead; fp16/bf16 do not
        # need the fp32 signed-zero rebuild, and `grad * 0.0` keeps inf -> NaN
        # matching the reference.
        out = tl.where(quant_mask != 0, grad, grad * 0.0)
    tl.store(output_ptr + offset, out)


def fake_quantize_per_tensor_affine_cachemask_backward(grad, mask):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_BACKWARD")
    grad = grad.contiguous()
    mask = mask.contiguous()
    output = torch.empty_like(grad)
    n_elements = grad.numel()
    if n_elements == 0:
        return output
    block_size, num_warps, masked = _pick_block(n_elements, grad.dtype)
    grid = (triton.cdiv(n_elements, block_size),)
    mask_view = mask.view(torch.int8)
    fix_sign = grad.dtype == torch.float32
    with torch_device_fn.device(grad.device):
        if masked:
            _fq_ptacmb_backward_mul_kernel[grid](
                grad,
                mask_view,
                output,
                n_elements,
                BLOCK_SIZE=block_size,
                FIX_SIGN=fix_sign,
                num_warps=num_warps,
            )
        else:
            _fq_ptacmb_backward_mul_unmasked[grid](
                grad,
                mask_view,
                output,
                BLOCK_SIZE=block_size,
                FIX_SIGN=fix_sign,
                num_warps=num_warps,
            )
    return output
