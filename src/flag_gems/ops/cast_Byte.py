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

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@triton.jit
def _cast_byte_value(x):
    # PyTorch narrows floating inputs through int64, preserving wraparound.
    if x.dtype.is_floating():
        if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
            x = x.to(tl.float32)
        x = x.to(tl.int64)
    return x.to(tl.uint8)


@libentry()
@triton.jit
def _cast_byte_strided(
    X, Y, N, SHAPE: tl.constexpr, STRIDES: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remaining = offsets
    src = tl.full((BLOCK,), 0, tl.int64)
    for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
        src += (remaining % SHAPE[dim]) * STRIDES[dim]
        remaining //= SHAPE[dim]
    x = tl.load(X + src, offsets < N, other=0)
    tl.store(Y + offsets, _cast_byte_value(x), offsets < N)


@libentry()
@triton.jit
def _cast_byte_kernel(X, Y, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + offsets, _cast_byte_value(x), offsets < N)


def cast_Byte(self, non_blocking=False):
    logger.debug("GEMS CAST_BYTE")
    if self.dtype == torch.uint8:
        return self
    if (
        self.device.type == "cpu"
        or self.layout != torch.strided
        or self.is_complex()
        or self.is_quantized
        or self.is_neg()
        or self.dtype
        not in (
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )
    ):
        return torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET, self, dtype=torch.uint8, non_blocking=non_blocking
        )
    out = torch.empty_like(self, dtype=torch.uint8)
    n = self.numel()
    if n == 0:
        return out
    with torch_device_fn.device(self.device):
        grid = (triton.cdiv(n, 1024),)
        if self.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(self):
            _cast_byte_kernel[grid](self, out, n, BLOCK=1024)
        else:
            order = sorted(
                range(self.ndim), key=lambda dim: out.stride(dim), reverse=True
            )
            shape = tuple(self.shape[dim] for dim in order)
            strides = tuple(self.stride(dim) for dim in order)
            _cast_byte_strided[grid](self, out, n, shape, strides, BLOCK=1024)
    return out
