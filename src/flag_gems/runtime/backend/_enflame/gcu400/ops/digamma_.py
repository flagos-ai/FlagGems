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


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def digamma_kernel_(x_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    pi = 3.1415926535897932384626433832795028841971

    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)

        reflect_mask = x < 0.5
        xr = tl.where(reflect_mask, 1.0 - x, x)

        s = tl.zeros_like(x)
        y = xr
        for _ in range(8):
            m = y < 8.0
            s = s - tl.where(m, 1.0 / y, 0.0)
            y = tl.where(m, y + 1.0, y)

        r = 1.0 / y
        r2 = r * r
        t2 = r2
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        series = (
            (-0.5 * r)
            + (-1.0 / 12.0) * t2
            + (1.0 / 120.0) * t4
            + (-1.0 / 252.0) * t6
            + (1.0 / 240.0) * t8
        )
        psi_y = tl.log(y) + s + series

        cot_term = tl.cos(pi * x) / tl.sin(pi * x)
        result = tl.where(reflect_mask, psi_y - pi * cot_term, psi_y)
        tl.store(x_ptr + off, result, mask=mask)


def digamma_(*args, **kwargs):
    logger.debug("GEMS_ENFLAME DIGAMMA_")
    x = args[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError("digamma_ expects a torch.Tensor as the first argument")

    BLOCK = 8192
    grid_size_fn = lambda n: min((n + BLOCK - 1) // BLOCK, NUM_SIPS * 2)

    if not x.is_contiguous():
        y = x.contiguous()
        N_total = y.numel()
        if N_total == 0:
            return x
        with torch_device_fn.device(y.device):
            digamma_kernel_[(grid_size_fn(N_total),)](
                y, N_total, BLOCK=BLOCK, num_warps=4
            )
        x.copy_(y)
        return x

    N_total = x.numel()
    if N_total == 0:
        return x
    with torch_device_fn.device(x.device):
        digamma_kernel_[(grid_size_fn(N_total),)](x, N_total, BLOCK=BLOCK, num_warps=4)
    return x
