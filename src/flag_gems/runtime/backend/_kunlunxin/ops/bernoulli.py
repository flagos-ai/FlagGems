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
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

UNROLL = 4


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def bernoulli_kernel(
    out_ptr,
    x_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    counters = philox_offset + offsets // 4
    c0 = (counters & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((counters >> 32) & 0xFFFFFFFF).to(tl.uint32)
    zero = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, zero, zero)
    lane = offsets % 4
    random = tl.where(
        lane == 0,
        r0,
        tl.where(lane == 1, r1, tl.where(lane == 2, r2, r3)),
    )
    random = uint_to_uniform_float(random)
    probabilities = tl.load(x_ptr + offsets, mask=offsets < N, other=0.0)
    output = tl.where(random < probabilities, 1.0, 0.0)
    tl.store(
        out_ptr + offsets,
        output,
        mask=offsets < N,
        eviction_policy="evict_first",
    )


def _launch_config(N):
    return 512, 4


def bernoulli(self, *, generator=None):
    logger.debug("GEMS_KUNLUNXIN BERNOULLI")
    assert self.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"bernoulli only supports floating point dtypes, got {self.dtype}"
    device = self.device
    self = self.contiguous()
    out = torch.empty_like(self)
    N = self.numel()
    BLOCK, num_warps = _launch_config(N)
    grid = (triton.cdiv(N, BLOCK),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        bernoulli_kernel[grid](
            out,
            self,
            N,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return out
