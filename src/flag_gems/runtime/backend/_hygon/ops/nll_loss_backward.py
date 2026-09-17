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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    HAS_WEIGHT: tl.constexpr,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 1024,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=ignore_index).to(tl.int32)
    ignore_mask = mask_n & (tgt != ignore_index)

    if HAS_WEIGHT:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)
    else:
        wgt_tgt = ignore_mask.to(tl.float32)

    if reduction == 0:
        out_grad = tl.load(out_grad_ptr + offsets_n, mask=mask_n, other=0).to(
            tl.float32
        )
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
        inv_w = tl.where(total_w != 0.0, 1.0 / total_w, 0.0)
    else:
        inv_w = 1.0

    inp_grad = tl.where(ignore_mask, -out_grad * wgt_tgt * inv_w, 0.0)
    inp_grad_ptrs = inp_grad_ptr + offsets_n * C + tgt
    tl.store(inp_grad_ptrs, inp_grad, mask=mask_n)


def heur_block_n(N):
    # Reduce kernel-launch grid size by processing more rows per program.
    if N <= 1024:
        return triton.next_power_of_2(max(N, 1))
    return 2048


def nll_loss_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    logger.debug("GEMS NLL Loss BWD (hygon)")
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    has_weight = weight is not None
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.zeros_like(self).contiguous()

    block_n = heur_block_n(N)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    with torch_device_fn.device(self.device):
        nll_loss_backward_kernel[grid](
            grad_output,
            target,
            weight,
            grad_input,
            ignore_index,
            total_weight,
            N,
            C,
            HAS_WEIGHT=has_weight,
            reduction=reduction,
            BLOCK_N=block_n,
        )

    return grad_input
