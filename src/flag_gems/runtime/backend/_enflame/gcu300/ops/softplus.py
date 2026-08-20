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
@triton.jit(do_not_specialize=["N"])
def softplus_beta1_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=3):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        out = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        tl.store(out_ptr + off, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N", "beta_val", "threshold_val"])
def softplus_kernel(
    x_ptr,
    out_ptr,
    N,
    beta_val,
    threshold_val,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=3):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        z = x * beta_val
        soft_z = tl.where(z > threshold_val, z, tl.log(1.0 + tl.exp(z)))
        out = soft_z / beta_val
        tl.store(out_ptr + off, out, mask=mask)


def _choose_block(N, is_fp32):
    if N <= 1024:
        return 1024
    if N <= 32768:
        return triton.next_power_of_2(N)
    return 65536 if is_fp32 else 131072


def softplus(self, beta=1.0, threshold=20.0):
    logger.debug("GEMS_ENFLAME SOFTPLUS GCU300")
    inp = self if self.is_contiguous() else self.contiguous()
    N = inp.numel()
    out = torch.empty_like(inp)

    is_fp32 = inp.dtype == torch.float32
    BLOCK = _choose_block(N, is_fp32)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS)

    with torch_device_fn.device(inp.device):
        if beta == 1.0 and threshold == 20.0:
            softplus_beta1_kernel[(grid_size,)](
                inp,
                out,
                N,
                BLOCK=BLOCK,
                num_warps=4,
            )
        else:
            softplus_kernel[(grid_size,)](
                inp,
                out,
                N,
                float(beta),
                float(threshold),
                BLOCK=BLOCK,
                num_warps=4,
            )
    return out
