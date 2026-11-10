# Copyright 2026, The FlagOS Contributors.
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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def _l1_loss(input, target):
    return tl.abs(input.to(tl.float32) - target.to(tl.float32))


@libentry()
@triton.jit
def _l1_loss_partial_sum_kernel(
    inp, target, mid, n_elements, reduction: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # NOTE: do not use tl.load(..., mask=mask, other=0.0) here. The kunlunxin
    # backend's masked load returns wrong values for the tail block, which
    # silently corrupts the reduction. Clamp the address and zero with
    # tl.where instead, which is exact.
    safe = tl.minimum(offsets, n_elements - 1)
    inp_val = tl.load(inp + safe).to(tl.float32)
    target_val = tl.load(target + safe).to(tl.float32)
    diff = tl.abs(inp_val - target_val)
    diff = tl.where(mask, diff, 0.0)
    acc = tl.sum(diff, axis=0)
    if reduction == 1:
        acc = acc / n_elements
    tl.store(mid + pid, acc)


@libentry()
@triton.jit
def _l1_loss_sum_kernel(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offsets = tl.arange(0, BLOCK_MID)
    mask = offsets < mid_size
    safe = tl.minimum(offsets, mid_size - 1)
    vals = tl.load(mid + safe).to(tl.float32)
    acc = tl.sum(tl.where(mask, vals, 0.0), axis=0)
    tl.store(out, acc)


_MAX_MID = 32768


def _normalize_reduction(reduction):
    # Accept both string and enum/int forms: 0=none, 1=mean, 2=sum
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        if r == "mean":
            return 1
        if r == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _check_input(input, target):
    if input.device != target.device:
        raise AssertionError("l1_loss: input and target must be on the same device.")
    input, target = torch.broadcast_tensors(input, target)
    return input.contiguous(), target.contiguous()


def l1_loss(input: torch.Tensor, target: torch.Tensor, reduction=1):
    logger.debug("GEMS_KUNLUNXIN L1_LOSS")
    reduction = _normalize_reduction(reduction)
    input, target = _check_input(input, target)
    n_elements = input.numel()

    if reduction == 0:
        return _l1_loss(input, target)

    if n_elements == 0:
        # Follow PyTorch behavior: sum -> 0, mean -> NaN
        if reduction == 2:
            return torch.zeros((), device=input.device, dtype=input.dtype)
        return torch.full((), float("nan"), device=input.device, dtype=input.dtype)

    block_size = get_block_size_1d(n_elements, input.element_size() * 2)
    mid_size = triton.cdiv(n_elements, block_size)
    if mid_size > _MAX_MID:
        block_size = triton.next_power_of_2(triton.cdiv(n_elements, _MAX_MID))
        mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), device=input.device, dtype=torch.float32)
    out = torch.empty((), device=input.device, dtype=input.dtype)

    with torch_device_fn.device(input.device):
        _l1_loss_partial_sum_kernel[(mid_size, 1, 1)](
            input,
            target,
            mid,
            n_elements,
            reduction=reduction,
            BLOCK_SIZE=block_size,
            buffer_size_limit=2048,
        )
        _l1_loss_sum_kernel[(1, 1, 1)](
            mid, out, mid_size, BLOCK_MID=block_mid, buffer_size_limit=2048
        )
    return out
