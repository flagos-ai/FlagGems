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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

NUM_SIPS = 24
BLOCK = 8192


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def hardsigmoid_kernel(x_ptr, out_ptr, N_total, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        xf = x.to(tl.float32)
        y = xf * (1.0 / 6.0) + 0.5
        y = tl.minimum(tl.maximum(y, 0.0), 1.0)
        tl.store(out_ptr + off, y.to(x.dtype), mask=mask)


def _grid(n_elements):
    return min(triton.cdiv(n_elements, BLOCK), NUM_SIPS * 2)


def hardsigmoid(x: torch.Tensor):
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out
    with torch_device_fn.device(x.device):
        hardsigmoid_kernel[(_grid(n_elements),)](
            x, out, n_elements, BLOCK_SIZE=BLOCK, num_warps=4
        )
    return out


def hardsigmoid_out(x: torch.Tensor, out: torch.Tensor):
    assert x.numel() == out.numel()
    n_elements = x.numel()
    if n_elements == 0:
        return out
    with torch_device_fn.device(x.device):
        hardsigmoid_kernel[(_grid(n_elements),)](
            x, out, n_elements, BLOCK_SIZE=BLOCK, num_warps=4
        )
    return out
