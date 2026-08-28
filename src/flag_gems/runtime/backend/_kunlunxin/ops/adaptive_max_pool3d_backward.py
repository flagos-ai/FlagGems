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

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_DEVICE_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit
def adaptive_max_pool3d_backward_scalar_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    IS_BFLOAT16: tl.constexpr,
    IS_FLOAT32: tl.constexpr,
):
    input_offset = tl.program_id(0)
    in_spatial = in_d * in_h * in_w
    out_spatial = out_d * out_h * out_w
    nc_idx = input_offset // in_spatial
    input_spatial_idx = input_offset % in_spatial
    d_in = input_spatial_idx // (in_h * in_w)
    input_remainder = input_spatial_idx % (in_h * in_w)
    h_in = input_remainder // in_w
    w_in = input_remainder % in_w

    d_out_start = d_in * out_d // in_d
    h_out_start = h_in * out_h // in_h
    w_out_start = w_in * out_w // in_w
    d_out_end = ((d_in + 1) * out_d + in_d - 1) // in_d
    h_out_end = ((h_in + 1) * out_h + in_h - 1) // in_h
    w_out_end = ((w_in + 1) * out_w + in_w - 1) // in_w
    grad_dtype = grad_output_ptr.type.element_ty
    grad_acc = 0.0

    for d_out_idx in tl.range(d_out_start, d_out_end):
        for h_out_idx in tl.range(h_out_start, h_out_end):
            for w_out_idx in tl.range(w_out_start, w_out_end):
                output_spatial_idx = (
                    d_out_idx * out_h * out_w + h_out_idx * out_w + w_out_idx
                )
                flat_output_offset = nc_idx * out_spatial + output_spatial_idx
                output_index = tl.load(indices_ptr + flat_output_offset)
                output_grad = tl.load(grad_output_ptr + flat_output_offset)
                contribution = tl.where(
                    output_index == input_spatial_idx, output_grad, 0.0
                )
                grad_acc += contribution
                if IS_BFLOAT16:
                    bits = grad_acc.to(tl.uint32, bitcast=True)
                    bits += 0x7FFF + ((bits >> 16) & 1)
                    bits &= 0xFFFF0000
                    grad_acc = bits.to(tl.float32, bitcast=True)
                elif not IS_FLOAT32:
                    grad_acc = grad_acc.to(grad_dtype).to(tl.float32)

    tl.store(grad_input_ptr + input_offset, grad_acc)


@libentry()
@triton.jit
def adaptive_max_pool3d_backward_window_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    num_inputs,
    MAX_CANDIDATE_D: tl.constexpr,
    MAX_CANDIDATE_H: tl.constexpr,
    MAX_CANDIDATE_W: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
    IS_FLOAT32: tl.constexpr,
    BLOCK_INPUT: tl.constexpr,
):
    input_offset = tl.program_id(0) * BLOCK_INPUT + tl.arange(0, BLOCK_INPUT)
    input_valid = input_offset < num_inputs
    in_spatial = in_d * in_h * in_w
    out_spatial = out_d * out_h * out_w
    nc_idx = input_offset // in_spatial
    input_spatial_idx = input_offset % in_spatial
    d_in = input_spatial_idx // (in_h * in_w)
    input_remainder = input_spatial_idx % (in_h * in_w)
    h_in = input_remainder // in_w
    w_in = input_remainder % in_w

    d_out_start = d_in * out_d // in_d
    h_out_start = h_in * out_h // in_h
    w_out_start = w_in * out_w // in_w
    d_out_end = ((d_in + 1) * out_d + in_d - 1) // in_d
    h_out_end = ((h_in + 1) * out_h + in_h - 1) // in_h
    w_out_end = ((w_in + 1) * out_w + in_w - 1) // in_w
    grad_dtype = grad_output_ptr.type.element_ty
    grad_acc = tl.zeros((BLOCK_INPUT,), tl.float32)
    match_count = tl.zeros((BLOCK_INPUT,), tl.float32)

    for d_delta in tl.static_range(0, MAX_CANDIDATE_D):
        d_out_idx = d_out_start + d_delta
        for h_delta in tl.static_range(0, MAX_CANDIDATE_H):
            h_out_idx = h_out_start + h_delta
            for w_delta in tl.static_range(0, MAX_CANDIDATE_W):
                w_out_idx = w_out_start + w_delta
                output_spatial_idx = (
                    d_out_idx * out_h * out_w + h_out_idx * out_w + w_out_idx
                )
                flat_output_offset = nc_idx * out_spatial + output_spatial_idx
                candidate_valid = (
                    input_valid
                    & (d_out_idx < d_out_end)
                    & (h_out_idx < h_out_end)
                    & (w_out_idx < w_out_end)
                )
                output_index = tl.load(
                    indices_ptr + flat_output_offset,
                    mask=candidate_valid,
                    other=-1,
                )
                output_grad = tl.load(
                    grad_output_ptr + flat_output_offset,
                    mask=candidate_valid,
                    other=0.0,
                )
                is_contribution = candidate_valid & (output_index == input_spatial_idx)
                contribution = tl.where(is_contribution, output_grad, 0.0)
                match_count += is_contribution.to(tl.float32)
                grad_acc += contribution
                if IS_BFLOAT16:
                    bits = grad_acc.to(tl.uint32, bitcast=True)
                    bits += 0x7FFF + ((bits >> 16) & 1)
                    bits &= 0xFFFF0000
                    grad_acc = bits.to(tl.float32, bitcast=True)
                elif not IS_FLOAT32:
                    grad_acc = grad_acc.to(grad_dtype).to(tl.float32)

    grad_acc = tl.where(match_count > 0.0, grad_acc, 0.0)
    tl.store(grad_input_ptr + input_offset, grad_acc, mask=input_valid)


def _normalized_shape(input):
    if input.ndim == 5:
        return input.shape
    if input.ndim == 4:
        c, d, h, w = input.shape
        return 1, c, d, h, w
    raise RuntimeError(f"expected 4D or 5D input, got {input.ndim}D")


def adaptive_max_pool3d_backward(grad_output, input, indices):
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_MAX_POOL3D_BACKWARD")

    n, c, in_d, in_h, in_w = _normalized_shape(input)
    if grad_output.ndim != input.ndim:
        raise RuntimeError("grad_output and input must have the same rank")
    if tuple(grad_output.shape[:-3]) != tuple(input.shape[:-3]):
        raise RuntimeError("grad_output and input batch/channel shapes must match")
    if tuple(indices.shape) != tuple(grad_output.shape):
        raise RuntimeError("indices and grad_output shapes must match")
    if indices.dtype != torch.int64:
        raise RuntimeError("adaptive_max_pool3d_backward indices must be int64")
    if grad_output.dtype != input.dtype:
        raise RuntimeError("grad_output and input must have the same dtype")
    if grad_output.device != input.device or indices.device != input.device:
        raise RuntimeError("grad_output, input, and indices must share a device")
    if input.dtype not in _DEVICE_DTYPES:
        cpu_grad_input = torch.ops.aten.adaptive_max_pool3d_backward(
            grad_output.cpu(), input.cpu(), indices.cpu()
        )
        return cpu_grad_input.to(input.device)

    out_d, out_h, out_w = grad_output.shape[-3:]
    grad_input = torch.empty(input.shape, device=input.device, dtype=grad_output.dtype)
    if grad_input.numel() == 0:
        return grad_input

    grad_output = grad_output.contiguous()
    indices = indices.contiguous()
    num_inputs = n * c * in_d * in_h * in_w
    max_candidate_d = (out_d + in_d - math.gcd(out_d, in_d) + in_d - 1) // in_d
    max_candidate_h = (out_h + in_h - math.gcd(out_h, in_h) + in_h - 1) // in_h
    max_candidate_w = (out_w + in_w - math.gcd(out_w, in_w) + in_w - 1) // in_w
    candidate_volume = max_candidate_d * max_candidate_h * max_candidate_w
    if candidate_volume <= 64:
        block_input = 256
        adaptive_max_pool3d_backward_window_kernel[
            (triton.cdiv(num_inputs, block_input),)
        ](
            grad_output,
            indices,
            grad_input,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            num_inputs,
            MAX_CANDIDATE_D=max_candidate_d,
            MAX_CANDIDATE_H=max_candidate_h,
            MAX_CANDIDATE_W=max_candidate_w,
            IS_BFLOAT16=grad_output.dtype == torch.bfloat16,
            IS_FLOAT32=grad_output.dtype == torch.float32,
            BLOCK_INPUT=block_input,
            num_warps=1,
            num_stages=1,
        )
    else:
        adaptive_max_pool3d_backward_scalar_kernel[(num_inputs,)](
            grad_output,
            indices,
            grad_input,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            IS_BFLOAT16=grad_output.dtype == torch.bfloat16,
            IS_FLOAT32=grad_output.dtype == torch.float32,
            num_warps=1,
            num_stages=1,
        )
    return grad_input


# Importing the vendor backend also replaces the broken native backward used by
# ordinary autograd outside use_gems(). Keep the Library alive for the process.
_ATEN_LIB = torch.library.Library("aten", "IMPL")
try:
    _ATEN_LIB.impl(
        "adaptive_max_pool3d_backward",
        adaptive_max_pool3d_backward,
        "CUDA",
        allow_override=True,
    )
except TypeError:
    _ATEN_LIB.impl("adaptive_max_pool3d_backward", adaptive_max_pool3d_backward, "CUDA")
