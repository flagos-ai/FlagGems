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
def relu6_kernel(x_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        y = tl.maximum(x, 0.0)
        y = tl.minimum(y, 6.0)
        tl.store(out_ptr + off, y, mask=mask)


def relu6(*args, **kwargs):
    logger.debug("GEMS_ENFLAME RELU6")
    x = (
        args[0]
        if len(args) > 0
        else kwargs.get("input", kwargs.get("self", kwargs.get("x")))
    )
    if x is None:
        raise TypeError(
            "relu6 expects a tensor as the first positional argument or keyword 'input'/'self'/'x'."
        )

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    N = out.numel()
    if N == 0:
        return out

    BLOCK = 8192
    grid = min((N + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(x_contig.device):
        relu6_kernel[(grid,)](x_contig, out, N, BLOCK=BLOCK, num_warps=4)
    return out
