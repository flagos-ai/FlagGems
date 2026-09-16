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

logger = logging.getLogger(__name__)


# hardswish(x) = x * relu6(x + 3) / 6 = x * min(max(x + 3, 0), 6) / 6
# Boundaries: x <= -3 -> 0; x >= 3 -> x.  Compute in float32 so fp16/bf16
# inputs match PyTorch's accumulation precision.
#
# BLOCK_SIZE = 8192: XPU 0.143ms vs 3.88ms at BLOCK=1024 for [4096,4096] fp16
# (isolation sweep: 256/512/1024/2048/4096/8192 -> 7.63/5.10/3.88/0.48/0.25/0.14),
# fp32 0.94ms -> 0.16ms; small shapes plateau at ~5us, no case regresses.
@triton.jit
def hardswish_kernel_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    xf = x.to(tl.float32)
    inner = tl.minimum(tl.maximum(xf + 3.0, 0.0), 6.0)
    y = xf * inner * (1.0 / 6.0)
    y = y.to(x.dtype)

    tl.store(x_ptr + offsets, y, mask=mask)


def hardswish_(*args, **kwargs):
    logger.debug("GEMS_KUNLUNXIN HARDSWISH_")
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))

    if x is None:
        raise ValueError("hardswish_: expected a Tensor as the first argument")
    if not isinstance(x, torch.Tensor):
        raise TypeError("hardswish_: expected a Tensor")
    if not x.is_floating_point():
        raise TypeError("hardswish_: expected a floating point tensor")

    orig = x
    x_work = x if x.is_contiguous() else x.contiguous()

    n_elements = x_work.numel()
    if n_elements == 0:
        return orig

    BLOCK_SIZE = 8192
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(x_work.device):
        hardswish_kernel_[grid](x_work, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if x_work.data_ptr() != orig.data_ptr():
        orig.copy_(x_work)

    return orig
