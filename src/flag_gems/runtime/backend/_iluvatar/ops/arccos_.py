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

"""In-place arccos (torch.arccos_ equivalent) as a Triton pointwise kernel.

run(x) mutates x in place and returns it, aliasing the reference semantics of
Tensor.arccos_/acos_.

Numerics: Triton's libdevice acos only accepts fp32 on this target, so fp16/bf16
inputs are upcast to fp32, computed, and rounded back on store.  The bf16
path performs both conversions manually: bf16->fp32 via (u16<<16) bitcast
(exact), and fp32->bf16 with round-to-nearest-even and explicit NaN handling.

Tuning (measured on Iluvatar BI-V150): the load->acos->store pipeline is the
bottleneck, not the math.  fp32 peaks at ~605 GB/s with 1 element/thread at
1024 threads/block and a .cg streaming store; fp16/bf16 peak at ~540-545
GB/s with 4 elements/thread.  The maskless fast path (n % BLOCK_SIZE == 0)
is used for fp32 and bf16 (measured +0.7% fp32, +2.6% bf16) while fp16 keeps
the mask (maskless fp16 measured ~3% slower).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(f'flag_gems.runtime._iluvatar.ops.{__name__.split(".")[-1]}')


@triton.jit
def _arccos_inplace_f32_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.extra.libdevice.acos(x)
    tl.store(x_ptr + offsets, y, cache_modifier=".cg")


@triton.jit
def _arccos_inplace_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.extra.libdevice.acos(x.to(tl.float32))
    tl.store(x_ptr + offsets, y, mask=mask)


@triton.jit
def _arccos_inplace_bf16_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)  # bf16
    u = x.to(tl.uint16, bitcast=True)
    f = (u.to(tl.uint32) << 16).to(tl.float32, bitcast=True)  # bf16 -> fp32 (exact)
    y = tl.extra.libdevice.acos(f)
    bits = y.to(tl.uint32, bitcast=True)
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    yb = tl.where(is_nan, (bits >> 16) | 0x40, rounded >> 16)
    tl.store(x_ptr + offsets, yb.to(tl.uint16).to(tl.bfloat16, bitcast=True), mask=mask)


@triton.jit
def _arccos_inplace_bf16_kernel_nomask(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)  # bf16
    u = x.to(tl.uint16, bitcast=True)
    f = (u.to(tl.uint32) << 16).to(tl.float32, bitcast=True)  # bf16 -> fp32 (exact)
    y = tl.extra.libdevice.acos(f)
    bits = y.to(tl.uint32, bitcast=True)
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    yb = tl.where(is_nan, (bits >> 16) | 0x40, rounded >> 16)
    tl.store(x_ptr + offsets, yb.to(tl.uint16).to(tl.bfloat16, bitcast=True))


def arccos_(x):
    logger.debug("GEMS_ILUVATAR ARCCOS_")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("arccos_ only supports float16, bfloat16, and float32")
    if not x.is_contiguous():
        raise ValueError(
            "the Iluvatar arccos_ implementation requires contiguous input"
        )
    xf = x.view(-1)
    n = xf.numel()
    if n == 0:
        return x
    if x.dtype == torch.float32:
        BLOCK_SIZE, NUM_WARPS = 1024, 16
        if n % BLOCK_SIZE == 0:
            _arccos_inplace_f32_kernel[(n // BLOCK_SIZE,)](
                xf, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS
            )
        else:
            _arccos_inplace_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
                xf, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS
            )
    elif x.dtype == torch.bfloat16:
        BLOCK_SIZE, NUM_WARPS = 1024, 4
        if n % BLOCK_SIZE == 0:
            _arccos_inplace_bf16_kernel_nomask[(n // BLOCK_SIZE,)](
                xf, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS
            )
        else:
            _arccos_inplace_bf16_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
                xf, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS
            )
    else:
        BLOCK_SIZE, NUM_WARPS = 512, 2
        _arccos_inplace_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
            xf, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS
        )
    return x
