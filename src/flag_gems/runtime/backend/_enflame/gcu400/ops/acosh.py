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

"""GCU400 专属 acosh：contiguous 浮点输入走 1D flat kernel 快路径。"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

FLAT_BLOCK = 1024
FLAT_SIPS = 48
FLAT_WARPS = 4


@libentry()
@triton.jit(do_not_specialize=["N"])
def _acosh_flat_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    for block in tl.range(tl.program_id(0), tl.cdiv(N, BLOCK_SIZE), tl.num_programs(0)):
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(X + offsets, mask=mask, other=1.0).to(tl.float32)
        y = tl.log(x + tl.sqrt(x * x - 1.0))
        tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def acosh_func(x):
    # acosh(x) = log(x + sqrt(x*x - 1))
    # Domain: [1, inf), returns NaN for x < 1 (except x=1 returns 0)
    x_f32 = x.to(tl.float32)
    sqrt_term = tl.sqrt(x_f32 * x_f32 - 1.0)
    result = tl.log(x_f32 + sqrt_term)
    return result.to(x.dtype)


def _flat_ready(A):
    return (
        A.is_contiguous()
        and A.numel() > 0
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
    )


def _flat_launch(A, out):
    n = A.numel()
    grid = (min(triton.cdiv(n, FLAT_BLOCK), FLAT_SIPS),)
    with torch_device_fn.device(A.device):
        _acosh_flat_kernel[grid](
            A.view(-1),
            out.view(-1),
            n,
            BLOCK_SIZE=FLAT_BLOCK,
            num_warps=FLAT_WARPS,
        )


def acosh(A):
    logger.debug("GEMS_ENFLAME ACOSH")
    if _flat_ready(A):
        out = torch.empty(A.shape, dtype=A.dtype, device=A.device)
        _flat_launch(A, out)
        return out
    return acosh_func(A)


def acosh_(A):
    logger.debug("GEMS_ENFLAME ACOSH_")
    if _flat_ready(A):
        _flat_launch(A, A)
        return A
    acosh_func(A, out0=A)
    return A
