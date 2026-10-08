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

from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _pick_block(n_elements, dtype):
    if n_elements >= 4_194_304:
        if dtype == torch.bfloat16:
            return 131072, 32, True
        if dtype == torch.float32:
            return 65536, 16, True
        return 16384, 8, False
    if n_elements >= 1_048_576:
        if dtype == torch.bfloat16:
            return 32768, 8, True
        return 16384, 8, False
    if n_elements >= 262_144:
        if dtype == torch.bfloat16:
            return 16384, 8, True
        if dtype == torch.float32:
            return 8192, 4, False
        return 4096, 4, False
    if n_elements >= 65_536:
        if dtype in (torch.float32, torch.bfloat16):
            return 4096, 4, dtype == torch.bfloat16
        return 8192, 4, False
    return 2048, 4, False


@triton.jit
def _expit_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offset, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _expit_kernel_unmasked(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offset, y.to(out_ptr.dtype.element_ty))


def _launch(x, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, vec = _pick_block(n_elements, x.dtype)
    masked = n_elements % block_size != 0
    launch_kwargs = dict(
        num_warps=num_warps,
        unroll_num=8 if vec else 16,
        buffer_size_limit=4096 if vec else 8192,
    )
    if vec:
        launch_kwargs["isCloseVectorization"] = True
    else:
        launch_kwargs["isCloseMemoryAsync"] = False
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        _expit_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            **launch_kwargs,
        )
    else:
        grid = (n_elements // block_size,)
        _expit_kernel_unmasked[grid](
            x,
            out,
            BLOCK_SIZE=block_size,
            **launch_kwargs,
        )


def special_expit(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_EXPIT")
    x = A.contiguous()
    out = torch.empty_like(x)
    _launch(x, out)
    return out
