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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _square_native(x):
    return x * x


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _square_bf16(x):
    xf = x.to(tl.float32)
    return (xf * xf).to(x.dtype)


@triton.jit
def _square_small(a_ptr, n, CAST: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    p = pid * BLOCK + tl.arange(0, BLOCK)
    mask = p < n
    x = tl.load(a_ptr + p, mask=mask)
    if CAST == 2:
        x = x.to(tl.float32)
    y = x * x
    if CAST == 2:
        y = y.to(tl.bfloat16)
    tl.store(a_ptr + p, y, mask=mask)


_SMALL_MAX = 8192
_SMALL_MAX_BF16 = 12288
_SMALL_BLOCK = 2048


def square_(A):
    logger.debug("GEMS_KUNLUNXIN SQUARE_")
    with torch_device_fn.device(A.device):
        n = A.numel()
        if n == 0:
            return A
        bf16 = A.dtype is torch.bfloat16
        if not A.is_contiguous():
            fn = _square_bf16 if bf16 else _square_native
            return fn(A, out0=A)
        limit = _SMALL_MAX_BF16 if bf16 else _SMALL_MAX
        if n <= limit:
            BLOCK = (
                _SMALL_BLOCK if bf16 else min(_SMALL_BLOCK, triton.next_power_of_2(n))
            )
            _square_small[(triton.cdiv(n, BLOCK),)](
                A, n, CAST=2 if bf16 else 0, BLOCK=BLOCK, num_warps=4
            )
            return A
        fn = _square_bf16 if bf16 else _square_native
        return fn(A, out0=A)
