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
def sgn_kernel_(x_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        pos = x > 0
        neg = x < 0
        res = pos.to(x.dtype) - neg.to(x.dtype)
        is_nan = x != x
        res = tl.where(is_nan, x, res)
        tl.store(x_ptr + off, res, mask=mask)


def sgn_(*args, **kwargs):
    logger.debug("GEMS_ENFLAME SGN_")
    x = None
    if len(args) == 1 and isinstance(args[0], torch.Tensor):
        x = args[0]
    elif "input" in kwargs and isinstance(kwargs["input"], torch.Tensor):
        x = kwargs["input"]
    elif "self" in kwargs and isinstance(kwargs["self"], torch.Tensor):
        x = kwargs["self"]

    if x is None:
        raise TypeError("sgn_ expects a single Tensor argument")

    unsupported = (not x.is_contiguous()) or x.is_complex()
    supported_dtypes = {
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if unsupported or x.dtype not in supported_dtypes:
        return torch.ops.aten.sgn_(x)

    N_total = x.numel()
    if N_total == 0:
        return x

    BLOCK = 8192
    grid_size = min((N_total + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(x.device):
        sgn_kernel_[(grid_size,)](x, N_total, BLOCK=BLOCK, num_warps=4)
    return x
