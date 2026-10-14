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
import operator

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _pad_can_use_int32(shape, strides, out_shape, before, after, numel, block):
    limit = (1 << 31) - 1
    # Include masked lanes in the last tile: their index arithmetic executes
    # even though they never load or store. Tensor numel alone also misses the
    # address span of transposed, sliced and expanded inputs.
    if triton.cdiv(numel, block) * block - 1 > limit:
        return False
    source_bound = 0
    for size, stride, out, left, right in zip(shape, strides, out_shape, before, after):
        # Bound coord-left, reflection's 2*size-2-idx, and circular's
        # idx-start / idx+extent before their selects or modulo operations.
        # This deliberately also bounds invalid constant-padding addresses.
        index_bound = out - 1 + abs(left) + 2 * size + abs(right)
        if index_bound > limit or abs(stride) > limit:
            return False
        source_bound += index_bound * abs(stride)
        if source_bound > limit:
            return False
    return True


@libentry()
@triton.jit
def _pad_kernel(
    X,
    Y,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    OUT_SHAPE: tl.constexpr,
    BEFORE: tl.constexpr,
    AFTER: tl.constexpr,
    MODE: tl.constexpr,
    VALUE: tl.constexpr,
    USE_INT32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Grid-stride iteration keeps even large rank-5 vision tensors below the
    # Ascend coreDim limit; strides also support transposed and expanded input.
    if USE_INT32:
        index_dtype: tl.constexpr = tl.int32
    else:
        index_dtype: tl.constexpr = tl.int64
    for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        offsets = tile.to(index_dtype) * BLOCK + tl.arange(0, BLOCK)
        # Coordinate division must remain exact for large flattened offsets.
        # The range guard still permits int32 source-address arithmetic.
        remaining = offsets.to(tl.int64)
        source = tl.full((BLOCK,), 0, index_dtype)
        valid = offsets < N
        for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
            coord = (remaining % OUT_SHAPE[dim]).to(index_dtype)
            remaining = remaining // OUT_SHAPE[dim]
            idx = coord - BEFORE[dim]
            if MODE == "constant":
                valid = valid & (idx >= 0) & (idx < SHAPE[dim])
            elif MODE == "replicate":
                idx = tl.minimum(tl.maximum(idx, 0), SHAPE[dim] - 1)
            elif MODE == "reflect":
                idx = tl.where(idx < 0, -idx, idx)
                idx = tl.where(idx >= SHAPE[dim], 2 * SHAPE[dim] - 2 - idx, idx)
            elif MODE == "circular":
                # Native circular padding crops negative edges before wrapping.
                start = max(-BEFORE[dim], 0)
                extent = SHAPE[dim] - start - max(-AFTER[dim], 0)
                idx = (idx - start) % extent
                idx = tl.where(idx < 0, idx + extent, idx) + start
            source += idx * STRIDES[dim]
        if MODE == "constant":
            # Ascend may form gather addresses for masked lanes. Keep those
            # pointers inside the nonempty input instead of below its storage.
            source = tl.where(valid, source, 0)
            data = tl.load(X + source, mask=valid, other=VALUE)
        else:
            data = tl.load(X + source, mask=valid, other=0)
        tl.store(Y + offsets, data, mask=offsets < N)


@libentry()
@triton.jit
def _pad_fill_kernel(Y, N: tl.constexpr, VALUE: tl.constexpr, BLOCK: tl.constexpr):
    # A masked load from an empty input may still fault on Ascend. This kernel
    # has no input pointer and emits only stores for the all-padding case.
    for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        offsets = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        data = tl.full((BLOCK,), VALUE, Y.dtype.element_ty)
        tl.store(Y + offsets, data, mask=offsets < N)


def _pad_impl(self, pad, mode="constant", value=None):
    logger.debug("GEMS ASCEND PAD")
    padding = tuple(operator.index(p) for p in pad)
    if len(padding) % 2 or len(padding) > 2 * self.ndim:
        raise RuntimeError("Padding length must be even and at most twice input rank")
    if mode not in ("constant", "reflect", "replicate", "circular"):
        raise NotImplementedError(f"Unrecognised padding mode {mode}")
    if mode != "constant" and value not in (None, 0):
        raise RuntimeError(f"Padding mode {mode} doesn't take a value argument")
    pairs = len(padding) // 2
    if mode != "constant" and (
        pairs not in (1, 2, 3) or self.ndim not in (pairs + 1, pairs + 2)
    ):
        raise NotImplementedError(
            "Non-constant padding supports 1 to 3 trailing spatial dimensions"
        )
    before = [0] * self.ndim
    after = [0] * self.ndim
    for i in range(pairs):
        before[self.ndim - 1 - i], after[self.ndim - 1 - i] = padding[2 * i : 2 * i + 2]
    shape = tuple(self.shape)
    out_shape = tuple(s + b + a for s, b, a in zip(shape, before, after))
    for i, (s, b, a, out) in enumerate(zip(shape, before, after, out_shape)):
        if out < 0 or (mode == "constant" and (max(-b, 0) > s or max(-a, 0) > s)):
            raise RuntimeError("Negative padding crops beyond the input size")
        if mode != "constant" and i >= self.ndim - pairs:
            if s <= 0:
                raise RuntimeError(
                    "Non-constant padding requires nonempty spatial dimensions"
                )
            if mode == "reflect" and (b >= s or a >= s):
                raise RuntimeError(
                    "Reflection padding must be smaller than the input dimension"
                )
            if mode == "circular" and (b > s or a > s):
                raise RuntimeError("Circular padding cannot wrap more than once")
            if mode == "circular" and s - max(-b, 0) - max(-a, 0) <= 0 and out > 0:
                raise RuntimeError(
                    "Circular padding requires a nonempty cropped dimension"
                )
    if mode != "constant":
        channel_dim = self.ndim - pairs - 1
        if shape[channel_dim] == 0:
            raise RuntimeError(
                "Non-constant padding requires a nonempty channel dimension"
            )
        if mode in ("reflect", "replicate") and max(out_shape[-pairs:]) <= 0:
            raise RuntimeError(
                "Padding produces an output spatial size that is too small"
            )
    output = torch.empty(out_shape, dtype=self.dtype, device=self.device)
    numel = math.prod(out_shape)
    if numel:
        with torch_device_fn.device(self.device):
            grid = (min(triton.cdiv(numel, 1024), 65535),)
            if self.numel() == 0:
                _pad_fill_kernel[grid](
                    output, numel, 0 if value is None else value, BLOCK=1024
                )
            else:
                strides = tuple(self.stride())
                _pad_kernel[grid](
                    self,
                    output,
                    numel,
                    shape,
                    strides,
                    out_shape,
                    tuple(before),
                    tuple(after),
                    mode,
                    0 if value is None else value,
                    USE_INT32=_pad_can_use_int32(
                        shape, strides, out_shape, before, after, numel, 1024
                    ),
                    BLOCK=1024,
                )
    return output


def pad(self, pad, mode="constant", value=None):
    return _pad_impl(self, pad, mode, value)


def constant_pad_nd(self, pad, value=0):
    return _pad_impl(self, pad, mode="constant", value=value)
