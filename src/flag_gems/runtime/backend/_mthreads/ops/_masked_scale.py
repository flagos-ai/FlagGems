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

from flag_gems.ops._masked_scale import _masked_scale as default__masked_scale
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.jit
def _masked_scale_kernel(
    input_ptr,
    mask_ptr,
    out_ptr,
    numel,
    scale,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN:
        x = tl.load(input_ptr + offs)
        maskv = tl.load(mask_ptr + offs)
        res = tl.where(maskv != 0, x * scale, 0.0)
        tl.store(out_ptr + offs, res)
    else:
        valid = offs < numel
        x = tl.load(input_ptr + offs, mask=valid, other=0.0)
        maskv = tl.load(mask_ptr + offs, mask=valid, other=0)
        res = tl.where(maskv != 0, x * scale, 0.0)
        tl.store(out_ptr + offs, res, mask=valid)


def _use_triton_kernel(input, mask) -> bool:
    if not isinstance(input, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return False
    if input.device.type != "musa" or input.dtype not in _SUPPORTED_DTYPES:
        return False
    if not input.is_contiguous() or not mask.is_contiguous():
        return False
    if input.numel() == 0 or input.shape != mask.shape:
        return False
    return True


def _masked_scale(input, mask, scale):
    logger.debug("GEMS_MTHREADS _MASKED_SCALE")
    if not _use_triton_kernel(input, mask):
        return default__masked_scale(input, mask, scale)
    out = torch.empty_like(input)
    numel = input.numel()
    if isinstance(scale, torch.Tensor):
        scale = scale.item()
    BLOCK_SIZE = 1024
    even = (numel % BLOCK_SIZE) == 0
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    with torch_device_fn.device(input.device):
        _masked_scale_kernel[grid](
            input,
            mask,
            out,
            numel,
            scale,
            BLOCK_SIZE=BLOCK_SIZE,
            EVEN=even,
            num_warps=4,
        )
    return out


__all__ = ["_masked_scale"]
