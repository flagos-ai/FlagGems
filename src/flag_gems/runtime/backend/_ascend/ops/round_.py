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


@triton.jit
def _round_kernel(
    x_ptr,
    n_elements,
    SCALE: tl.constexpr,
    GUARD_LARGE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    scaled = x * SCALE
    if GUARD_LARGE:
        bias = tl.where(tl.abs(scaled) < 4194304.0, 12582912.0, 0.0)
    else:
        bias = 12582912.0
    rounded = (scaled + bias) - bias
    tl.store(x_ptr + offsets, rounded / SCALE, mask=mask)


def round_(input, *, decimals=0):
    n_elements = input.numel()
    scale = float(10.0**decimals)
    unguarded = (input.dtype == torch.float16 and decimals <= 1) or (
        input.dtype == torch.bfloat16 and decimals == 0
    )
    guard_large = not unguarded
    block_size = (
        4096
        if n_elements <= 4096
        else 8192 if guard_large or n_elements <= 8192 else 16384
    )
    grid = (triton.cdiv(n_elements, block_size),)
    _round_kernel[grid](
        input,
        n_elements,
        scale,
        guard_large,
        block_size,
    )
    return input
