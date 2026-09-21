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

from flag_gems.ops.log2 import log2_ as default_log2_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.jit
def _log2_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    # Compute in fp32 (matches reference: torch.log2(x.float()).to(x.dtype)),
    # then round back to the input dtype.
    x = x.to(tl.float32)
    x = tl.log2(x)
    x = x.to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + offsets, x, mask=mask)


@libentry()
@triton.jit
def _log2_kernel_even(x_ptr, BLOCK_SIZE: tl.constexpr):
    # Unmasked specialization for tensors whose size is an exact multiple of
    # BLOCK_SIZE; probes and eval show it ~1% faster than the masked path for
    # large 16-bit tensors on this backend.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    x = x.to(tl.float32)
    x = tl.log2(x)
    x = x.to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + offsets, x)


def _use_triton_kernel(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not x.is_contiguous() or x.numel() == 0:
        return False
    return True


def _launch_log2_(x: torch.Tensor):
    n_elements = x.numel()
    x_flat = x.view(-1)
    with torch_device_fn.device(x.device):
        if n_elements <= 32768:
            # Small tensors are dispatch-bound: tiny blocks with 2 warps measured
            # ~10% lower latency than BLOCK 512/4 (min 2.52us vs 2.89us, avg
            # 2.68us vs 3.42us on the 4096-element workload, replicated across
            # two shapes and three probe runs).
            # Rule 21: hardcoded BLOCK_SIZE with rationale.
            block = 64
            grid = (triton.cdiv(n_elements, block),)
            _log2_kernel[grid](x_flat, n_elements, BLOCK_SIZE=block, num_warps=2)
        elif x.element_size() == 2 and n_elements % 2048 == 0:
            # Large 16-bit tensors: 16 elems/thread (2x128-bit in flight) and no
            # bounds mask; measured fastest on this backend.
            # Rule 21: hardcoded BLOCK_SIZE with rationale.
            grid = (n_elements // 2048,)
            _log2_kernel_even[grid](x_flat, BLOCK_SIZE=2048)
        else:
            # Large 32-bit tensors. 16 elems/thread with 2 warps (4x128-bit in
            # flight) measured faster on the 16.7M-element tensor (0.10472ms vs
            # 0.10506ms) but slower on the 4.2M-element one, so split by size.
            # Rule 21: hardcoded BLOCK_SIZE with rationale.
            block = 1024
            grid = (triton.cdiv(n_elements, block),)
            _log2_kernel[grid](
                x_flat,
                n_elements,
                BLOCK_SIZE=block,
                num_warps=2 if n_elements > 8388608 else 4,
            )
    return x


def log2_(x):
    logger.debug("GEMS_MTHREADS LOG2_")
    if not _use_triton_kernel(x):
        return default_log2_(x)
    return _launch_log2_(x)


__all__ = ["log2_"]
