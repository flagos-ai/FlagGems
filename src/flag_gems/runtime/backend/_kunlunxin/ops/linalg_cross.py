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

from flag_gems.ops.linalg_cross import (
    _linalg_cross_impl as _generic_linalg_cross_impl,
    _resolve_view,
    _validate_inputs,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# The generic complex kernels of flag_gems.ops.linalg_cross do not compile on
# the Kunlunxin XPU backend: the 2D-tile + tl.split contiguous kernel fails
# with "out of resource: uni_sram" for every BLOCK_SIZE, and the strided /
# lastdim-broadcast / dim1-3d variants crash the XPU compiler (LLVM APInt
# assertion or "Failed to tune buffer size") around BLOCK_SIZE=128. Only the
# plain 1D interleaved kernel compiles for every tested (BLOCK, warps) pair.
# This override therefore keeps the generic, already-passing real-dtype
# implementation, and routes complex64 through its own materialized
# last-dimension-last layout with the 1D kernel below (probe: BLOCK 8..256,
# warps 1..8 all compile).
_BLOCK_SIZE = 256
_NUM_WARPS = 4


@libentry()
@triton.jit
def _linalg_cross_complex_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_vectors,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute cross products over interleaved real/imaginary (6-float) vectors."""
    vector_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = vector_offsets < num_vectors
    offsets = vector_offsets * 6

    input_0_real = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    input_0_imag = tl.load(input_ptr + offsets + 1, mask=mask, other=0.0)
    input_1_real = tl.load(input_ptr + offsets + 2, mask=mask, other=0.0)
    input_1_imag = tl.load(input_ptr + offsets + 3, mask=mask, other=0.0)
    input_2_real = tl.load(input_ptr + offsets + 4, mask=mask, other=0.0)
    input_2_imag = tl.load(input_ptr + offsets + 5, mask=mask, other=0.0)

    other_0_real = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    other_0_imag = tl.load(other_ptr + offsets + 1, mask=mask, other=0.0)
    other_1_real = tl.load(other_ptr + offsets + 2, mask=mask, other=0.0)
    other_1_imag = tl.load(other_ptr + offsets + 3, mask=mask, other=0.0)
    other_2_real = tl.load(other_ptr + offsets + 4, mask=mask, other=0.0)
    other_2_imag = tl.load(other_ptr + offsets + 5, mask=mask, other=0.0)

    product_12_real = input_1_real * other_2_real - input_1_imag * other_2_imag
    product_12_imag = input_1_real * other_2_imag + input_1_imag * other_2_real
    product_21_real = input_2_real * other_1_real - input_2_imag * other_1_imag
    product_21_imag = input_2_real * other_1_imag + input_2_imag * other_1_real

    product_20_real = input_2_real * other_0_real - input_2_imag * other_0_imag
    product_20_imag = input_2_real * other_0_imag + input_2_imag * other_0_real
    product_02_real = input_0_real * other_2_real - input_0_imag * other_2_imag
    product_02_imag = input_0_real * other_2_imag + input_0_imag * other_2_real

    product_01_real = input_0_real * other_1_real - input_0_imag * other_1_imag
    product_01_imag = input_0_real * other_1_imag + input_0_imag * other_1_real
    product_10_real = input_1_real * other_0_real - input_1_imag * other_0_imag
    product_10_imag = input_1_real * other_0_imag + input_1_imag * other_0_real

    tl.store(output_ptr + offsets, product_12_real - product_21_real, mask=mask)
    tl.store(output_ptr + offsets + 1, product_12_imag - product_21_imag, mask=mask)
    tl.store(output_ptr + offsets + 2, product_20_real - product_02_real, mask=mask)
    tl.store(output_ptr + offsets + 3, product_20_imag - product_02_imag, mask=mask)
    tl.store(output_ptr + offsets + 4, product_01_real - product_10_real, mask=mask)
    tl.store(output_ptr + offsets + 5, product_01_imag - product_10_imag, mask=mask)


def _linalg_cross_complex_xpu(input, other, dim, output=None):
    dim, output_shape = _validate_inputs(input, other, dim)
    input = _resolve_view(input)
    other = _resolve_view(other)

    # Canonicalize to a contiguous layout with the cross dimension last, which
    # is the only layout the XPU compiler can lower for complex cross products.
    input_moved = input.movedim(dim, -1)
    other_moved = other.movedim(dim, -1)
    input_moved, other_moved = torch.broadcast_tensors(input_moved, other_moved)
    input_contig = input_moved.contiguous()
    other_contig = other_moved.contiguous()
    result_moved = torch.empty(
        input_contig.shape, dtype=input_contig.dtype, device=input_contig.device
    )

    num_vectors = result_moved.numel() // 3
    if num_vectors > 0:
        grid = (triton.cdiv(num_vectors, _BLOCK_SIZE),)
        with torch_device_fn.device(result_moved.device):
            _linalg_cross_complex_kernel[grid](
                torch.view_as_real(input_contig),
                torch.view_as_real(other_contig),
                torch.view_as_real(result_moved),
                num_vectors,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=_NUM_WARPS,
            )

    result = result_moved.movedim(-1, dim)
    if output is None:
        return result
    # Write through the user's (possibly strided) out tensor with the native
    # strided-copy engine; flag_gems never overrides _copy_from.
    torch.ops.aten._copy_from(result, output, False)
    return output


def linalg_cross(input, other, *, dim=-1):
    """Kunlunxin XPU implementation of ``torch.linalg.cross``."""
    logger.debug("GEMS LINALG_CROSS (kunlunxin)")
    if input.is_complex():
        return _linalg_cross_complex_xpu(input, other, dim)
    # The generic real kernels compile and pass on XPU; keep them untouched.
    return _generic_linalg_cross_impl(input, other, dim)


def linalg_cross_out(input, other, *, dim=-1, out):
    logger.debug("GEMS LINALG_CROSS_OUT (kunlunxin)")
    if torch._C._is_alias_of(out, input) or torch._C._is_alias_of(out, other):
        raise RuntimeError(
            "linalg_cross: out must not share memory with either input tensor"
        )
    dim, output_shape = _validate_inputs(input, other, dim)
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"linalg_cross: expected out dtype {input.dtype}, but got {out.dtype}"
        )
    if out.device != input.device:
        raise RuntimeError("linalg_cross: out must be on the same device as input")
    if out.shape != output_shape:
        out.resize_(output_shape)
    if input.is_complex():
        return _linalg_cross_complex_xpu(input, other, dim, output=out)
    return _generic_linalg_cross_impl(input, other, dim, output=out)