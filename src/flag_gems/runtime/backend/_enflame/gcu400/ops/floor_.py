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
def floor_kernel_(
    x_ptr,
    N_total,
    BLOCK: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        out = x
        if IS_FP32:
            out = tl.floor(x.to(tl.float32))
        elif IS_FP16:
            out = tl.cast(tl.floor(tl.cast(x, tl.float32)), tl.float16)
        elif IS_BF16:
            out = tl.cast(tl.floor(tl.cast(x, tl.float32)), tl.bfloat16)
        tl.store(x_ptr + off, out, mask=mask)


def floor_(input):
    logger.debug("GEMS_ENFLAME FLOOR_")
    x = input
    if not isinstance(x, torch.Tensor):
        raise TypeError("floor_ expects a torch.Tensor.")
    if x.is_complex():
        raise TypeError("floor_ is not supported for complex tensors.")
    if not x.is_contiguous():
        raise ValueError(
            "floor_ Triton kernel currently supports only contiguous tensors."
        )

    N = x.numel()
    if N == 0:
        return x

    dtype = x.dtype
    IS_FP32 = dtype == torch.float32
    IS_FP16 = dtype == torch.float16
    IS_BF16 = dtype == torch.bfloat16

    BLOCK = 8192
    grid = min((N + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(x.device):
        floor_kernel_[(grid,)](
            x,
            N,
            BLOCK=BLOCK,
            IS_FP32=IS_FP32,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
            num_warps=4,
        )
    return x
