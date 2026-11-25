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

"""Kunlunxin/XPU override of ``flipud``.

Two things differ from the generic implementation:

* native ``torch.flipud`` on XPU returns a contiguous tensor, while the
  generic impl allocates with ``torch.empty_like`` and preserves the input
  strides -- this override allocates contiguously;
* the generic contiguous kernel indexes flat offsets in ``int64`` and computes
  the row reversal per element, which is slow on XPU; this override copies
  whole rows (and a separate flat kernel for 1-D) with ``int32`` addressing.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.flipud import flipud_strided_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def flipud_rows_kernel(
    input, output, height, row_size, TAIL: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    src = (height - 1 - row) * row_size + col
    dst = row * row_size + col
    if TAIL:
        mask = col < row_size
        tl.store(output + dst, tl.load(input + src, mask=mask), mask=mask)
    else:
        tl.store(output + dst, tl.load(input + src))


@triton.jit
def flipud_flat_kernel(input, output, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(output + off, tl.load(input + (n - 1 - off), mask=mask), mask=mask)


def _flipud_impl(self: torch.Tensor) -> torch.Tensor:
    if self.ndim < 1:
        raise RuntimeError("Input must be >= 1-d.")

    if self.is_complex():
        # view_as_real/view_as_complex are stride-only reinterpretations (no
        # device compute); the complex case is skipped on XPU anyway.
        real = torch.view_as_real(self)  # allow-torch-compute
        return torch.view_as_complex(_flipud_impl(real))  # allow-torch-compute

    output = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    n_elements = self.numel()
    if n_elements == 0:
        return output

    with torch_device_fn.device(self.device):
        if self.is_contiguous():
            row_size = math.prod(self.shape[1:])
            if row_size == 1:
                grid = (triton.cdiv(n_elements, 1024),)
                flipud_flat_kernel[grid](self, output, n_elements, BLOCK=1024)
            else:
                block = 4096
                grid = (self.shape[0], triton.cdiv(row_size, block))
                flipud_rows_kernel[grid](
                    self,
                    output,
                    self.shape[0],
                    row_size,
                    TAIL=row_size % block != 0,
                    BLOCK=block,
                )
        else:
            grid = (triton.cdiv(n_elements, 1024),)
            flipud_strided_kernel[grid](
                self,
                output,
                n_elements,
                SHAPE=tuple(self.shape),
                INPUT_STRIDES=tuple(self.stride()),
                OUTPUT_STRIDES=tuple(output.stride()),
                NDIM=self.ndim,
                BLOCK_SIZE=1024,
            )

    return output


class Flipud(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return _flipud_impl(input)

    @staticmethod
    def backward(ctx, grad_output):
        return _flipud_impl(grad_output)


def flipud(self: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN FLIPUD")

    if torch.is_grad_enabled() and self.requires_grad:
        return Flipud.apply(self)
    return _flipud_impl(self)
