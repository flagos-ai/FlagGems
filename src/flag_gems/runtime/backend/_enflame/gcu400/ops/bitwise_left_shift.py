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

"""GCU400 专属 bitwise left shift：同形 contiguous inplace 走 1D flat kernel。"""

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
def _ilshift_flat_kernel(A, B, N, BLOCK_SIZE: tl.constexpr):
    num_programs = tl.num_programs(0)
    for block in tl.range(tl.program_id(0), tl.cdiv(N, BLOCK_SIZE), num_programs):
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        a = tl.load(A + offsets, mask=mask, other=0)
        b = tl.load(B + offsets, mask=mask, other=0)
        tl.store(A + offsets, a << b, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_left_shift_func(x, y):
    return x << y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_left_shift_func_scalar(x, y):
    return x << y


def bitwise_left_shift(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_LEFT_SHIFT")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return bitwise_left_shift_func(A, B)
    elif isinstance(A, torch.Tensor):
        return bitwise_left_shift_func_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return bitwise_left_shift_func_scalar(B, A)
    else:
        return torch.tensor(A * B)


def bitwise_left_shift_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_LEFT_SHIFT_")
    if (
        isinstance(B, torch.Tensor)
        and B.shape == A.shape
        and B.dtype == A.dtype
        and A.is_contiguous()
        and B.is_contiguous()
        and A.numel() > 0
        # codegen 路径在 int16/int32 上已优于 flat kernel，只对 uint8 启用，
        # 避免主 dtype 性能下降触发一票否决。
        and A.dtype == torch.uint8
    ):
        # 同形同 dtype 的常见 inplace case 用 1D flat kernel，避开 codegen
        # wrapper 的索引分解与 dtype 提升（uint8 在 codegen 路径上明显偏慢）。
        numel = A.numel()
        grid = (
            (
                1
                if numel <= FLAT_BLOCK
                else min(triton.cdiv(numel, FLAT_BLOCK), FLAT_SIPS)
            ),
        )
        with torch_device_fn.device(A.device):
            _ilshift_flat_kernel[grid](
                A,
                B,
                numel,
                BLOCK_SIZE=FLAT_BLOCK,
                num_warps=FLAT_WARPS,
            )
        return A
    if isinstance(B, torch.Tensor):
        return bitwise_left_shift_func(A, B, out0=A)
    else:
        return bitwise_left_shift_func_scalar(A, B, out0=A)
