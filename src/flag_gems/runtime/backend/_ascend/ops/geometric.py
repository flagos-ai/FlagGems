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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.ops.exponential import _philox4x32_10
from flag_gems.runtime.backend._ascend.utils import CORE_NUM

logger = logging.getLogger(__name__)


@triton.jit
def _store_sample(out, index, value, N, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    offset = index
    if len(SHAPE) > 0:
        offset = tl.full(index.shape, 0, tl.int64)
        remaining = index
        for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
            offset += (remaining % SHAPE[dim]) * STRIDES[dim]
            remaining = remaining // SHAPE[dim]
    tl.store(out + offset, value, index < N)


@triton.jit
def _sample(r, scale):
    # The midpoint conversion excludes both endpoints, so log never sees zero.
    bits = (r.to(tl.int32, bitcast=True) >> 9) & 0x7FFFFF
    u = (bits.to(tl.float32) + 0.5) * 1.1920928955078125e-7
    return tl.floor(tl.log(u) * scale) + 1.0


@triton.jit(do_not_specialize=["seed", "offset", "scale"])
def _geometric_kernel(
    out,
    N,
    seed,
    offset,
    scale,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for tile in range(tl.program_id(0), tl.cdiv(N, 4 * BLOCK), tl.num_programs(0)):
        counter = offset.to(tl.int64) + (tile * BLOCK).to(tl.int64)
        c0 = counter.to(tl.uint32) + lane.to(tl.uint32)
        carry = (c0 ^ 0x80000000).to(tl.int32, bitcast=True) < (
            lane.to(tl.uint32) ^ 0x80000000
        ).to(tl.int32, bitcast=True)
        c1 = (counter >> 32).to(tl.uint32) + carry.to(tl.uint32)
        zero = tl.full((BLOCK,), 0, tl.uint32)
        r0, r1, r2, r3 = _philox4x32_10(seed, c0, c1, zero, zero)
        y0 = _sample(r0, scale)
        y1 = _sample(r1, scale)
        y2 = _sample(r2, scale)
        y3 = _sample(r3, scale)
        start = (tile * (4 * BLOCK)).to(tl.int64) + lane
        _store_sample(out, start, y0, N, SHAPE, STRIDES)
        _store_sample(out, start + BLOCK, y1, N, SHAPE, STRIDES)
        _store_sample(out, start + 2 * BLOCK, y2, N, SHAPE, STRIDES)
        _store_sample(out, start + 3 * BLOCK, y3, N, SHAPE, STRIDES)


def _fill(self, p, generator):
    if not 0.0 < p < 1.0:
        raise RuntimeError("geometric_ expects p to be in (0, 1)")
    if self.dtype == torch.bool or self.is_complex():
        raise RuntimeError("geometric_ is not implemented for this dtype")
    n = self.numel()
    if n == 0:
        return self
    if any(size > 1 and stride == 0 for size, stride in zip(self.shape, self.stride())):
        raise RuntimeError("unsupported operation: tensor has internal overlap")
    dense = self.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(self)
    shape = () if dense else tuple(self.shape)
    strides = () if dense else tuple(self.stride())
    scale = 1.0 / math.log1p(-p)
    block = 256
    tiles = triton.cdiv(n, 4 * block)
    with torch_device_fn.device(self.device):
        if generator is None:
            generator = torch_device_fn.default_generators[
                torch_device_fn.current_device()
            ]
        if generator.device != self.device:
            raise RuntimeError("Expected a generator on the same device as self")
        seed = generator.initial_seed()
        offset = generator.get_offset()
        generator.set_offset(offset + tiles * block)
        grid = (min(tiles, CORE_NUM),)
        _geometric_kernel[grid](
            self,
            n,
            seed,
            offset,
            scale,
            shape,
            strides,
            BLOCK=block,
            num_warps=4,
        )
    return self


def geometric_(self, p=0.5, *, generator=None):
    logger.debug("GEMS_ASCEND GEOMETRIC_")
    return _fill(self, p, generator)


def geometric(self, p=0.5, *, generator=None):
    logger.debug("GEMS_ASCEND GEOMETRIC")
    return _fill(torch.empty_like(self), p, generator)
