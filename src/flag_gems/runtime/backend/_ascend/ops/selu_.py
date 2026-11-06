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
from triton.runtime import driver

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@triton.jit
def _selu_inplace_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_start in tl.range(
        program_id * BLOCK_SIZE,
        n_elements,
        num_programs * BLOCK_SIZE,
    ):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        x_f32 = x.to(tl.float32)
        alpha: tl.constexpr = 1.6732632423543772
        scale: tl.constexpr = 1.0507009873554805
        y_f32 = scale * tl.where(
            x_f32 > 0.0,
            x_f32,
            alpha * (tl.exp(x_f32) - 1.0),
        )
        y = y_f32.to(x.dtype)
        tl.store(x_ptr + offsets, y, mask=mask)


def selu_(self):
    logger.debug("GEMS SELU_")
    # ATen dispatches selu_ to elu_, which only implements floating types;
    # integral/bool inputs must raise rather than being computed in fp32 and
    # silently written back in the original dtype.
    if self.dtype not in _FLOAT_DTYPES:
        raise RuntimeError(f"\"selu_\" not implemented for '{self.dtype}'")

    # The kernel indexes storage linearly, so a non-dense view (e.g. x[::2])
    # would update the wrong elements.
    if not self.is_contiguous():
        raise ValueError(
            "selu_ Triton kernel currently supports only contiguous tensors."
        )

    n_elements = self.numel()
    # An empty tensor is a valid no-op for ATen, but a zero-sized grid aborts
    # the process on Ascend (coreDim must not be 0).
    if n_elements == 0:
        return self

    properties = driver.active.utils.get_device_properties(torch.npu.current_device())
    if n_elements <= 4096:
        block_size = 1024
        grid = (min(triton.cdiv(n_elements, block_size), properties["num_vectorcore"]),)
        _selu_inplace_kernel[grid](self, n_elements, BLOCK_SIZE=block_size)
    else:
        block_size = 8192
        grid = (min(triton.cdiv(n_elements, block_size), properties["num_vectorcore"]),)
        _selu_inplace_kernel[grid](
            self, n_elements, BLOCK_SIZE=block_size, num_stages=1
        )
    return self
