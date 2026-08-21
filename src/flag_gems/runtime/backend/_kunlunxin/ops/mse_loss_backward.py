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
from flag_gems.utils import triton_lang_extension as ext

from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from _kunlunxin.utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# mse_loss_backward is a memory-bound elementwise op (3 reads + 1 write).
# The old flat masked kernel (BLOCK=8192, always-masked) leaves ~10x on the
# table for large tensors: XPU's masked memory path runs far below the
# unmasked block-DMA bandwidth for fp16/bf16. Two proven machinery pieces from
# prior kunlunxin elementwise fixes are combined here:
#   - pointwise_dynamic codegen tuned with buffer_size_limit=4096 + unroll 8
#     (KernelGen codegenConfig sweep: big-tensor winner for 3-read ops; the
#     addcmul-style config with unroll 16 is up to 30% worse on 1024x65536 /
#     10000x65536 for fp16).
#   - per-call NEED_MASK dispatch so fully-divisible shapes skip the masked
#     memory path entirely.
_SMALL_NUMEL = 8192
_SMALL_MAX_BLOCK = 8192
_SMALL_WARPS = 4

_tuned_pd_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, True, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=_tuned_pd_config,
)
@triton.jit
def mse_loss_backward_pd(grad_output, self, target, scale):
    return (
        (self.to(tl.float32) - target.to(tl.float32))
        * grad_output.to(tl.float32)
        * scale
    )


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _mse_loss_backward_small_kernel(
    grad_output,
    self,
    target,
    output,
    n_elements,
    SCALE,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < n_elements
        self_value = tl.load(self + offsets, mask=mask, other=0.0).to(tl.float32)
        target_value = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_value = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
        result = grad_value * SCALE * (self_value - target_value)
        tl.store(output + offsets, result.to(output.dtype.element_ty), mask=mask)
    else:
        self_value = tl.load(self + offsets).to(tl.float32)
        target_value = tl.load(target + offsets).to(tl.float32)
        grad_value = tl.load(grad_output + offsets).to(tl.float32)
        result = grad_value * SCALE * (self_value - target_value)
        tl.store(output + offsets, result.to(output.dtype.element_ty))


def mse_loss_backward(grad_output, self, target, reduction=1):
    logger.debug("GEMS_KUNLUNXIN MSE_LOSS_BACKWARD")
    self_contiguous = self.contiguous()
    target_contiguous = target.contiguous()
    grad_contiguous = grad_output.contiguous()
    output = torch.empty_like(self_contiguous)
    n_elements = self_contiguous.numel()
    if n_elements == 0:
        return output

    # Reduction.MEAN(=1) divides by numel; none(=0)/sum(=2) do not.
    scale = 2.0 / n_elements if reduction == 1 else 2.0

    with torch_device_fn.device(self.device):
        if n_elements <= _SMALL_NUMEL:
            block_size = min(_SMALL_MAX_BLOCK, triton.next_power_of_2(n_elements))
            need_mask = (n_elements % block_size) != 0
            if need_mask and block_size == 1024:
                # XPU masked fp32 tile with BLOCK=1024 drops lane 0; use 512.
                block_size = 512
                need_mask = (n_elements % block_size) != 0
            _mse_loss_backward_small_kernel[
                (triton.cdiv(n_elements, block_size),)
            ](
                grad_contiguous,
                self_contiguous,
                target_contiguous,
                output,
                n_elements,
                SCALE=scale,
                BLOCK_SIZE=block_size,
                NEED_MASK=need_mask,
                num_warps=_SMALL_WARPS,
            )
        else:
            mse_loss_backward_pd(
                grad_contiguous,
                self_contiguous,
                target_contiguous,
                scale,
                out0=output,
            )
    return output