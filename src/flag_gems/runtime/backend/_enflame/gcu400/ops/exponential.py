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

"""gcu400-specific out-of-place ``exponential``.

The generic implementation launches ``cdiv(N, BLOCK * 8)`` programs with an
autotuned BLOCK of at most 2048.  For the benchmark shape ``(1024**3,)`` that
exceeds the GCU400 grid.x limit of 65535 (measured: ``Required: 2097152``),
so the op fails with ``OutOfResources`` before doing any work.

The gcu400 inplace ``exponential_`` already solves this by picking a much
larger BLOCK (16384 / 32768 for big inputs).  This out-of-place variant reuses
that exact kernel and BLOCK policy, writing to a fresh tensor instead.  Small
inputs keep the same BLOCK rule so distribution/KS behaviour is unchanged.
"""

import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

from .exponential_ import fused_exponential_kernel_f32

logger = logging.getLogger(__name__)


def exponential(x, lambd: float = 1.0, *, generator=None):
    logger.debug("GEMS_ENFLAME EXPONENTIAL")

    dtype = x.dtype
    device = x.device
    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    N = x.numel()
    inv_lambd = 1.0 / lambd
    eps_minus = -0.5 * torch.finfo(dtype).eps
    out = torch.empty_like(x)

    UNROLL = 4
    if N >= 67108864:
        BLOCK = 32768
    elif N >= 65536:
        BLOCK = 16384
    else:
        BLOCK = min(triton.next_power_of_2(N), 1024)

    grid_size = triton.cdiv(N, BLOCK * UNROLL)
    # Every program draws BLOCK philox counters, including the lanes its masked
    # tail stores drop, so a call consumes exactly grid_size * BLOCK counters.
    # `N // UNROLL` under-reserves whenever BLOCK > 1 (e.g. small N), and the
    # next call then replays part of this call's random stream (Issue #7162).
    increment = grid_size * BLOCK
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    # Write directly to the target dtype (store implicitly does f32 -> fp16/bf16), removing the original three-step
    # f32 intermediate buffer + .to(dtype) + copy_ path for large fp16/bf16 shapes; numerically equivalent, and saves
    # two kernels plus one full memory pass.
    out_buf = out

    with torch_device_fn.device(device):
        fused_exponential_kernel_f32[(grid_size,)](
            out_buf,
            N,
            inv_lambd,
            eps_minus,
            philox_seed,
            philox_offset,
            BLOCK,
            num_warps=1,
        )

    return out
