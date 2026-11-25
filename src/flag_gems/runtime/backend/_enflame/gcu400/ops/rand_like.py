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


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def _rand_like_local_kernel(
    out_ptr, N, philox_seed, philox_offset, BLOCK: tl.constexpr
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    zero = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, zero, zero)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    tl.store(
        out_ptr + off_0,
        uint_to_uniform_float(r0),
        mask=off_0 < N,
        eviction_policy="evict_first",
    )
    tl.store(
        out_ptr + off_0 + BLOCK,
        uint_to_uniform_float(r1),
        mask=off_0 + BLOCK < N,
        eviction_policy="evict_first",
    )
    tl.store(
        out_ptr + off_0 + 2 * BLOCK,
        uint_to_uniform_float(r2),
        mask=off_0 + 2 * BLOCK < N,
        eviction_policy="evict_first",
    )
    tl.store(
        out_ptr + off_0 + 3 * BLOCK,
        uint_to_uniform_float(r3),
        mask=off_0 + 3 * BLOCK < N,
        eviction_policy="evict_first",
    )


logger = logging.getLogger(__name__)
UNROLL = 4


def rand_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_ENFLAME_GCU400 RAND_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
    # hence we cannot obtain the per thread offset as in Pytorch.
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(x.device):
        _rand_like_local_kernel[(triton.cdiv(N, 16384 * UNROLL),)](
            out, N, philox_seed, philox_offset, BLOCK=16384, num_warps=4
        )
    return out
