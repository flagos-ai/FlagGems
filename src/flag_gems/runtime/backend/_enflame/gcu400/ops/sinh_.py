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
def sinh_kernel_(x_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        y = 0.5 * (tl.exp(x) - tl.exp(-x))
        tl.store(x_ptr + off, y, mask=mask)


def sinh_(*args, **kwargs):
    logger.debug("GEMS_ENFLAME SINH_")
    x = args[0] if args else kwargs.get("self", kwargs.get("input", None))
    if x is None:
        raise TypeError("sinh_ expected a Tensor as the first argument")
    if not isinstance(x, torch.Tensor):
        raise TypeError("sinh_ expected a torch.Tensor")
    if x.numel() == 0:
        return x
    if not x.is_contiguous():
        raise RuntimeError(
            "sinh_ Triton kernel currently supports only contiguous tensors"
        )
    supported_dtypes = (torch.float16, torch.float32, torch.bfloat16)
    if x.dtype not in supported_dtypes:
        raise RuntimeError(
            f"sinh_ Triton kernel supports dtypes {supported_dtypes}, but got {x.dtype}"
        )

    N_total = x.numel()
    BLOCK = 8192
    grid_size = min((N_total + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(x.device):
        sinh_kernel_[(grid_size,)](x, N_total, BLOCK=BLOCK, num_warps=4)
    return x
