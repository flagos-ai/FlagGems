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

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fft_ifftshift_scalar_copy(X, Y):
    tl.store(Y, tl.load(X))


@libentry()
@triton.jit
def _fft_ifftshift_permute(
    X,
    Y,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    ADDS: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE_OFFSETS: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE_OFFSETS:
        pid = pid.to(tl.int64)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    if N % BLOCK:
        task = tl.minimum(off, N - 1)
    else:
        task = off
    if WIDE_OFFSETS:
        task = task.to(tl.int64)
    src = tl.full((BLOCK,), 0, tl.int64 if WIDE_OFFSETS else tl.int32)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        idx = task % SHAPE[d]
        task = task // SHAPE[d]
        if ADDS[d]:
            idx = idx + ADDS[d]
            idx = tl.where(idx >= SHAPE[d], idx - SHAPE[d], idx)
        src = src + idx * STRIDES[d]
    value = tl.load(X + src)
    if N % BLOCK:
        tl.store(Y + off, value, off < N)
    else:
        tl.store(Y + off, value)


def fft_ifftshift(self, dim=None):
    logger.debug("GEMS_KUNLUNXIN FFT_IFFTSHIFT")
    shape = tuple(self.shape)
    ndim = len(shape)
    dims = range(ndim) if dim is None else (dim,) if isinstance(dim, int) else dim
    adds = [0] * ndim
    for d in dims:
        if d < -ndim or d >= ndim:
            raise IndexError(
                f"Dimension out of range (got {d} for tensor of rank {ndim})"
            )
        d %= ndim
        if shape[d]:
            adds[d] = (adds[d] + shape[d] // 2) % shape[d]

    out = torch.empty(shape, dtype=self.dtype, device=self.device)
    n = self.numel()
    if n == 0:
        return out
    strides = tuple(self.stride())
    element_size = self.element_size()
    # Move storage units unchanged, including complex and NaN payloads.
    storage_type = {1: tl.int8, 2: tl.int16, 4: tl.int32, 8: tl.int64}.get(
        element_size, tl.int64
    )
    if element_size == 16:
        shape = (*shape, 2)
        strides = (*tuple(s * 2 for s in strides), 1)
        adds.append(0)
        n *= 2
    src = triton.reinterpret(self, storage_type)
    dst = triton.reinterpret(out, storage_type)
    if n == 1:
        _fft_ifftshift_scalar_copy[(1,)](src, dst)
    else:
        # Large tiles can be mis-vectorized across short wrap intervals by
        # this backend. Bound them by the contiguous run, not by total size.
        suffix = 1
        shortest_run = n
        for size, add in zip(reversed(shape), reversed(adds)):
            if add:
                shortest_run = min(shortest_run, min(add, size - add) * suffix)
            suffix *= size
        tile_limit = 16384
        if shortest_run < 256:
            tile_limit = min(tile_limit, 1 << ((32 * shortest_run).bit_length() - 1))
        if not self.is_contiguous():
            tile_limit = min(tile_limit, 4096)
        block = min(triton.next_power_of_2(n), tile_limit)
        span = sum((size - 1) * abs(stride) for size, stride in zip(shape, strides))
        _fft_ifftshift_permute[((n + block - 1) // block,)](
            src,
            dst,
            n,
            shape,
            strides,
            tuple(adds),
            block,
            max(n, span) > 2147483647,
            num_warps=4,
        )
    return out
