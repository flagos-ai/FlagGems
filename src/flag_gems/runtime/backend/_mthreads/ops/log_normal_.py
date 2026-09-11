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

from flag_gems import runtime
from flag_gems.ops.log_normal_ import log_normal_ as default_log_normal_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
UNROLL = 4

try:
    pair_uniform_to_normal = tl.pair_uniform_to_normal
except AttributeError:

    @triton.jit
    def pair_uniform_to_normal(u1, u2):
        """Box-Muller transform"""
        u1 = tl.maximum(1.0e-7, u1)
        th = 6.283185307179586 * u2
        r = tl.sqrt(-2.0 * tl.log(u1))
        return r * tl.cos(th), r * tl.sin(th)


@triton.heuristics(runtime.get_heuristic_config("randn"))
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "mean", "std"])
def log_normal_kernel(
    out_ptr,
    N,
    mean,
    std,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Fused single-kernel log-normal sampler: Philox RNG -> uniform float ->
    # Box-Muller normal -> exp(n*std + mean). The generic implementation writes
    # the normal samples to a temporary fp32 buffer and runs a second
    # elementwise kernel to apply the exp() transform into the output tensor.
    # Fusing everything into one kernel drops the temporary allocation and the
    # second launch, which is the win on large shapes.
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.uint32)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    n0, n1 = pair_uniform_to_normal(r0, r1)
    n2, n3 = pair_uniform_to_normal(r2, r3)

    y0 = tl.exp(n0 * std + mean)
    y1 = tl.exp(n1 * std + mean)
    y2 = tl.exp(n2 * std + mean)
    y3 = tl.exp(n3 * std + mean)

    off_0 = (tl.program_id(0) * BLOCK * 4).to(tl.int64) + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    tl.store(out_ptr + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, y2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, y3, mask=off_3 < N, eviction_policy="evict_first")


def _use_triton_kernel(self) -> bool:
    if not isinstance(self, torch.Tensor):
        return False
    if self.device.type != "musa" or self.dtype not in _SUPPORTED_DTYPES:
        return False
    if not self.is_contiguous() or self.numel() == 0:
        return False
    return True


def log_normal_(self, mean=1.0, std=2.0, *, generator=None):
    logger.debug("GEMS_MTHREADS LOG_NORMAL_")
    if not _use_triton_kernel(self):
        return default_log_normal_(self, mean=mean, std=std, generator=generator)

    N = volume(self.shape)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(self.device):
        log_normal_kernel[grid_fn](
            self, N, float(mean), float(std), philox_seed, philox_offset
        )
    return self
