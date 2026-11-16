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
def _sample(r, scale, HALF: tl.constexpr):
    # The midpoint conversion excludes both endpoints, so log never sees zero.
    u = ((r >> 9).to(tl.float32) + 0.5) * 1.1920928955078125e-7
    if HALF:
        # For p=1/2, inverse-CDF sampling is exactly the binary exponent.
        return (127 - (u.to(tl.int32, bitcast=True) >> 23)).to(tl.float32)
    return tl.ceil(tl.log(u) * scale)


@triton.jit(do_not_specialize=["seed", "offset", "scale"])
def _geometric_kernel(
    out,
    N,
    seed,
    offset,
    scale,
    HALF: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    tile = tl.program_id(0)
    counter = offset.to(tl.uint64) + (tile * BLOCK).to(tl.uint64)
    c0 = counter.to(tl.uint32) + lane.to(tl.uint32)
    c1 = (counter >> 32).to(tl.uint32) + (c0 < lane.to(tl.uint32)).to(tl.uint32)
    zero = tl.full((BLOCK,), 0, tl.uint32)
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, zero, zero)
    y0 = _sample(r0, scale, HALF)
    y1 = _sample(r1, scale, HALF)
    y2 = _sample(r2, scale, HALF)
    y3 = _sample(r3, scale, HALF)
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
    dense = self.is_contiguous()
    if not dense:
        if any(
            size > 1 and stride == 0 for size, stride in zip(self.shape, self.stride())
        ):
            raise RuntimeError("unsupported operation: tensor has internal overlap")
        dense = torch.ops.aten.is_non_overlapping_and_dense(self)
    shape = () if dense else tuple(self.shape)
    strides = () if dense else tuple(self.stride())
    scale = 1.0 / math.log1p(-p)
    # Spread small tensors across SMs instead of using one oversized program.
    block = 64 if n <= 4096 else 1024
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
        grid = (tiles,)
        _geometric_kernel[grid](
            self,
            n,
            seed,
            offset,
            scale,
            p == 0.5,
            shape,
            strides,
            BLOCK=block,
            num_warps=4,
        )
    return self


def geometric_(self, p=0.5, *, generator=None):
    logger.debug("GEMS_MTHREADS GEOMETRIC_")
    return _fill(self, p, generator)


def geometric(self, p=0.5, *, generator=None):
    logger.debug("GEMS_MTHREADS GEOMETRIC")
    return _fill(torch.empty_like(self), p, generator)
