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

"""Kunlunxin (XPU) specialization of ``aten::fliplr``.

``torch.fliplr`` flips a tensor along dimension 1 (left/right).  On the XPU
stack ``torch.fliplr`` materializes a *contiguous* result (it is a flip + copy,
not a view), so the generic ``torch.empty_like(self)`` with the default
``preserve_format`` produces the wrong strides for transposed / non-contiguous
inputs: the generic kernel values are correct, but the returned tensor keeps the
input's non-contiguous strides and ``result.stride() != torch.fliplr(ref).stride()``
(the CPU reference ``.to("cpu")`` also makes the reference contiguous).

The generic implementation is also extremely slow on this backend for the
contiguous case: it performs a per-element reversed gather along the width axis,
and a stride ``-1`` gather has no vectorised path on TritonXPU (D-017), so it
runs ~10x slower than the vendor flip.

The contiguous input is laid out as ``(outer, width, inner)`` with
``outer = shape[0]``, ``width = shape[1]``, ``inner = prod(shape[2:])``.
``fliplr`` only reverses the ``width`` blocks of ``inner`` *contiguous* elements,
so when ``inner`` is a multiple of a vector-friendly tile the reversal can be
done by block-DMA copies with the block index reversed once per program, keeping
both load and store fully affine.  For ``inner == 1`` (plain 2-D) the reversal is
along the innermost axis and must fall back to the reversed gather, which is the
structural wall on this backend (no vectorised negative-stride load).
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def fliplr_contiguous_kernel(
    input,
    output,
    n_elements,
    width,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)
    column = (offsets // inner_size) % width
    input_offsets = offsets + (width - 1 - 2 * column) * inner_size
    values = tl.load(input + input_offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


@libentry()
@triton.jit
def fliplr_block_kernel(
    input,
    output,
    width,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program copies one contiguous BLOCK_SIZE chunk that lies entirely
    # inside a single (row, column) block of ``inner_size`` elements.  The column
    # block index is reversed once per program, so both load and store stay
    # affine (block DMA) and only the block *order* is reversed.
    pid = tl.program_id(0)
    chunks_per_unit = inner_size // BLOCK_SIZE
    unit = pid // chunks_per_unit
    chunk = pid % chunks_per_unit
    j = unit % width
    row = unit // width
    src_unit = row * width + (width - 1 - j)
    offsets = unit * inner_size + chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    src_offsets = src_unit * inner_size + chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input + src_offsets)
    tl.store(output + offsets, values)


@libentry()
@triton.jit
def fliplr_strided_kernel(
    input,
    output,
    n_elements,
    SHAPE: tl.constexpr,
    INPUT_STRIDES: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    linear = offsets.to(tl.int64)
    input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for dim in tl.static_range(NDIM - 1, -1, -1):
        index = linear % SHAPE[dim]
        linear //= SHAPE[dim]
        input_index = SHAPE[dim] - 1 - index if dim == 1 else index
        input_offsets += input_index * INPUT_STRIDES[dim]
    values = tl.load(input + input_offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


# Candidate block sizes for the block-reverse path, largest first.  Only large
# blocks (>= 1024 lanes) are kept: the block-reverse only pays off when each
# program moves a wide contiguous block (block DMA).  Smaller tiles (512/256/
# 128/64) produce many more programs with a worse tile width and end up slower
# than the plain reversed gather (measured on the (64, 512, 512) inner==512
# case: block 512 is ~1.2x slower than the gather for fp16/bf16), so inner
# sizes below 1024 fall back to the gather kernel instead.
_BLOCK_SIZES = (4096, 2048, 1024)


def _pick_block_size(inner_size: int):
    for block_size in _BLOCK_SIZES:
        if inner_size % block_size == 0:
            return block_size
    return None


def fliplr(self: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN FLIPLR")

    if self.ndim < 2:
        raise RuntimeError("Input must be >= 2-d.")

    # torch.fliplr (and the CPU reference via ``.to("cpu")``) yields a contiguous
    # result, so allocate a contiguous output regardless of input layout.
    output = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    n_elements = self.numel()
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(self.device):
        if self.is_contiguous():
            width = self.shape[1]
            inner_size = math.prod(self.shape[2:])
            block_size = _pick_block_size(inner_size)
            if block_size is not None and inner_size > 1:
                grid = (n_elements // block_size,)
                fliplr_block_kernel[grid](
                    self,
                    output,
                    width,
                    inner_size,
                    BLOCK_SIZE=block_size,
                )
            else:
                fliplr_contiguous_kernel[grid](
                    self,
                    output,
                    n_elements,
                    width,
                    inner_size,
                    BLOCK_SIZE=1024,
                )
        else:
            fliplr_strided_kernel[grid](
                self,
                output,
                n_elements,
                SHAPE=tuple(self.shape),
                INPUT_STRIDES=tuple(self.stride()),
                NDIM=self.ndim,
                BLOCK_SIZE=1024,
            )

    return output
