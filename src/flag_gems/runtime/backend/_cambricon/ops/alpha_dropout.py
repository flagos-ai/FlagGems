# Copyright 2026, The FlagOS Contributors.
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
import math

import torch
import torch_mlu  # noqa: F401
import triton
import triton.language as tl
from triton.language.extra.mlu.libdevice import philox as _philox

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)

UNROLL = 4
_ALPHA = 1.6732632423543772848170429916717
_SCALE = 1.0507009873554804934193349852946
_ALPHA_PRIME = -_ALPHA * _SCALE


def _alpha_dropout_affine(p: float):
    if p == 0.0:
        return 1.0, 0.0
    a = 1.0 / math.sqrt((1.0 - p) * (1.0 + p * _ALPHA_PRIME * _ALPHA_PRIME))
    b = -a * p * _ALPHA_PRIME
    return a, b


@libtuner(
    configs=[
        triton.Config(kwargs={"BLOCK": 1024}, num_stages=3, num_warps=1),
        triton.Config(kwargs={"BLOCK": 4096}, num_stages=3, num_warps=1),
        triton.Config(kwargs={"BLOCK": 16384}, num_stages=3, num_warps=1),
        triton.Config(kwargs={"BLOCK": 32768}, num_stages=3, num_warps=1),
    ],
    key=["N"],
)
@libentry()
@triton.jit(do_not_specialize=["p", "a", "b", "philox_seed", "philox_offset"])
def alpha_dropout_forward_kernel(
    X,
    Y,
    N,
    p,
    a,
    b,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 4
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    i4_start = pid * BLOCK
    block_start = pid * UNROLL * BLOCK
    step = num_jobs * UNROLL * BLOCK

    sl = (philox_seed & 0xFFFFFFFF).to(tl.uint32)
    sh = ((philox_seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c0_base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    # alpha_prime is the value replacing dropped elements before affine transform
    alpha_prime = -1.7580993408473766

    for block_offset in range(block_start, N, step):
        r = _philox(BLOCK, sl, sh, c0_base + i4_start, c1, 0, 0, 10)
        r0 = uint_to_uniform_float(r[:, 0])
        r1 = uint_to_uniform_float(r[:, 1])
        r2 = uint_to_uniform_float(r[:, 2])
        r3 = uint_to_uniform_float(r[:, 3])

        mask0 = r0 > p
        mask1 = r1 > p
        mask2 = r2 > p
        mask3 = r3 > p

        off_0 = block_offset + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK

        x0 = tl.load(
            X + off_0, mask=off_0 < N, other=0.0, eviction_policy="evict_first"
        )
        x1 = tl.load(
            X + off_1, mask=off_1 < N, other=0.0, eviction_policy="evict_first"
        )
        x2 = tl.load(
            X + off_2, mask=off_2 < N, other=0.0, eviction_policy="evict_first"
        )
        x3 = tl.load(
            X + off_3, mask=off_3 < N, other=0.0, eviction_policy="evict_first"
        )

        y0 = tl.where(mask0, a * x0 + b, a * alpha_prime + b)
        y1 = tl.where(mask1, a * x1 + b, a * alpha_prime + b)
        y2 = tl.where(mask2, a * x2 + b, a * alpha_prime + b)
        y3 = tl.where(mask3, a * x3 + b, a * alpha_prime + b)

        tl.store(Y + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
        tl.store(Y + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
        tl.store(Y + off_2, y2, mask=off_2 < N, eviction_policy="evict_first")
        tl.store(Y + off_3, y3, mask=off_3 < N, eviction_policy="evict_first")
        i4_start += num_jobs * BLOCK


def alpha_dropout(input, p=0.5, train=True):
    logger.debug("GEMS_CAMBRICON ALPHA_DROPOUT")
    if not train or p == 0:
        return input.clone()
    if p == 1:
        return torch.zeros_like(input)

    assert 0.0 < p < 1.0, "p must be in (0, 1)"

    device = input.device
    input = input.contiguous()
    out = torch.empty_like(input)
    N = input.numel()
    grid_fn = lambda meta: (
        min(triton.cdiv(N, meta["BLOCK"] * UNROLL), TOTAL_CORE_NUM),
    )
    increment = triton.cdiv(N, UNROLL)
    a, b = _alpha_dropout_affine(p)

    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        alpha_dropout_forward_kernel[grid_fn](
            input,
            out,
            N,
            p,
            a,
            b,
            philox_seed,
            philox_offset,
        )
    return out
