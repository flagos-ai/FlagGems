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

from .bitwise_and import bitwise_and_tensor

# Use the generic op's logger name so the functional test's
# ``caplog.at_level("DEBUG", logger="flag_gems.ops.and_tensor")`` assertion on
# the "GEMS AND" message still fires from this backend override.
logger = logging.getLogger("flag_gems.ops.and_tensor")

# Below this element count the tuned pointwise_dynamic path is dominated by its
# Python-side rank/stride analysis + task build (host overhead), so a flat 1-D
# launch that skips that machinery is faster. Above it, the XPU DMA-tuned
# pointwise kernel wins. Crossover measured on P800 (see solution doc).
_SMALL_NUMEL = 65536


@triton.jit
def _and_tensor_flat_kernel(x_ptr, y_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < numel
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(x_ptr.dtype.element_ty)
    y = tl.load(y_ptr + offset, mask=mask, other=0).to(y_ptr.dtype.element_ty)
    tl.store(out_ptr + offset, x & y, mask=mask)


def _use_flat_path(x, y):
    return (
        isinstance(x, torch.Tensor)
        and isinstance(y, torch.Tensor)
        and x.dtype == y.dtype
        and x.is_contiguous()
        and y.is_contiguous()
        and x.shape == y.shape
        and x.device == y.device
        and 0 < x.numel() <= _SMALL_NUMEL
    )


def and_tensor(self, other):
    """``__and__.Tensor`` for Kunlunxin.

    ``__and__.Tensor`` and ``bitwise_and.Tensor`` are the same ATen operation's
    two entry points. Large tensors use the XPU DMA-tuned ``bitwise_and_tensor``
    kernel (the generic fallback's fixed ``BLOCK=2048`` flat launch is
    launch-bound and ~10-40x slower there). Small same-shape contiguous pairs
    take a single flat launch that skips the ``pointwise_dynamic`` host-side
    preparation, which dominates latency at that size.
    """
    logger.debug("GEMS_KUNLUNXIN AND_TENSOR")
    if _use_flat_path(self, other):
        numel = self.numel()
        out = torch.empty_like(self)
        BLOCK = 2048
        grid = (triton.cdiv(numel, BLOCK),)
        _and_tensor_flat_kernel[grid](self, other, out, numel, BLOCK=BLOCK, num_warps=4)
        return out
    return bitwise_and_tensor(self, other)
