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

"""GCU400 专属 BCEWithLogits：默认路径用融合 1D flat kernel，其余走公共 Triton kernel。"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_PROGRAMS = 48
BLOCK_SIZE = 8192


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def _bce_logits_flat_kernel(
    input_ptr,
    target_ptr,
    output_ptr,
    num_elements,
    BLOCK: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(program_id, tl.cdiv(num_elements, BLOCK), num_programs):
        element_offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        valid_mask = element_offsets < num_elements
        logits = tl.load(input_ptr + element_offsets, mask=valid_mask, other=0).to(
            tl.float32
        )
        targets = tl.load(target_ptr + element_offsets, mask=valid_mask, other=0).to(
            tl.float32
        )
        # 数值稳定形式：max(x,0) - x*y + log(1 + exp(-|x|))
        loss = (
            tl.maximum(logits, 0.0)
            - logits * targets
            + tl.log(1.0 + tl.exp(-tl.abs(logits)))
        )
        tl.store(output_ptr + element_offsets, loss, mask=valid_mask)


def _can_use_fused(input_tensor, target, weight, pos_weight):
    return (
        weight is None
        and pos_weight is None
        and input_tensor.is_contiguous()
        and target.is_contiguous()
        and input_tensor.shape == target.shape
    )


def _launch_fused(input_tensor, target, output_tensor):
    num_elements = input_tensor.numel()
    if num_elements == 0:
        return output_tensor
    grid_size = min(triton.cdiv(num_elements, BLOCK_SIZE), NUM_PROGRAMS)
    with torch_device_fn.device(input_tensor.device):
        _bce_logits_flat_kernel[(grid_size,)](
            input_tensor,
            target,
            output_tensor,
            num_elements,
            BLOCK=BLOCK_SIZE,
            num_warps=4,
        )
    return output_tensor


def _common_elementwise(self, target, weight, pos_weight, out0=None):
    """保留公共实现的 weight / pos_weight / 广播语义，全部仍是 Triton kernel。"""
    from flag_gems.ops.binary_cross_entropy_with_logits import (
        binary_cross_entropy_with_logits_kernel,
        binary_cross_entropy_with_logits_pos_weight_kernel,
        binary_cross_entropy_with_logits_weight_kernel,
        binary_cross_entropy_with_logits_weight_pos_weight_kernel,
    )

    if weight is not None and pos_weight is not None:
        kernel = binary_cross_entropy_with_logits_weight_pos_weight_kernel
        args = (self, target, weight, pos_weight)
    elif weight is not None:
        kernel = binary_cross_entropy_with_logits_weight_kernel
        args = (self, target, weight)
    elif pos_weight is not None:
        kernel = binary_cross_entropy_with_logits_pos_weight_kernel
        args = (self, target, pos_weight)
    else:
        kernel = binary_cross_entropy_with_logits_kernel
        args = (self, target)
    if out0 is None:
        return kernel(*args)
    return kernel(*args, out0=out0)


def _apply_reduction(output_tensor, reduction):
    # reduction: 0=NONE, 1=MEAN, 2=SUM；与公共实现口径一致。
    if reduction == 1:
        return output_tensor.mean()
    if reduction == 2:
        return output_tensor.sum()
    return output_tensor


def binary_cross_entropy_with_logits(
    self, target, weight=None, pos_weight=None, reduction=1
):
    logger.debug("GEMS_ENFLAME BINARY_CROSS_ENTROPY_WITH_LOGITS")
    if _can_use_fused(self, target, weight, pos_weight):
        out = torch.empty_like(self)
        _launch_fused(self, target, out)
    else:
        out = _common_elementwise(self, target, weight, pos_weight)
    return _apply_reduction(out, reduction)


def binary_cross_entropy_with_logits_(
    self, target, weight=None, pos_weight=None, reduction=1
):
    logger.debug("GEMS_ENFLAME BINARY_CROSS_ENTROPY_WITH_LOGITS_")
    if _can_use_fused(self, target, weight, pos_weight):
        _launch_fused(self, target, self)
        out = self
    else:
        out = _common_elementwise(self, target, weight, pos_weight, out0=self)
    return _apply_reduction(out, reduction)
