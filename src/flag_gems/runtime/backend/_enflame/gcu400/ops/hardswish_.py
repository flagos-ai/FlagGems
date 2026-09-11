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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24
BLOCK = 8192


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def hardswish_kernel_(x_ptr, N_total, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        tmp = tl.maximum(x + 3.0, 0.0)
        tmp = tl.minimum(tmp, 6.0)
        y = x * (tmp / 6.0)
        tl.store(x_ptr + off, y, mask=mask)


def hardswish_(*args, **kwargs):
    logger.debug("GEMS_ENFLAME HARDSWISH_")
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))

    if x is None:
        raise ValueError("hardswish_: expected a Tensor as the first argument")
    if not isinstance(x, torch.Tensor):
        raise TypeError("hardswish_: expected a Tensor")
    if not x.is_floating_point():
        raise TypeError("hardswish_: expected a floating point tensor")

    orig = x
    x_work = x if x.is_contiguous() else x.contiguous()
    n_elements = x_work.numel()
    if n_elements == 0:
        return orig

    grid = min(triton.cdiv(n_elements, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(x_work.device):
        hardswish_kernel_[(grid,)](x_work, n_elements, BLOCK_SIZE=BLOCK, num_warps=4)

    if x_work.data_ptr() != orig.data_ptr():
        orig.copy_(x_work)

    return orig
