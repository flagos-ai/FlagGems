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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    y = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return y


# silu_backward uses a dedicated bounded-tile kernel on XPU. The previous
# pointwise_dynamic implementation (tile = next_pow2(numel/12), up to 2M-wide
# per CTA) combined with `div_rn` (IEEE round-to-nearest division, ~2.9x slower
# than plain `/` on XPU) and `isCloseVectorization/unroll_num` left large shapes
# at ~0.51 gems speedup. This custom kernel uses a bounded BLOCK (<= 65536,
# more CTAs for the compute-heavy exp), plain division (within torch tolerance),
# and `buffer_size_limit=4096` with vectorization open: measured [4096,4096]
# fp16 speedup 0.51 -> ~1.02, fp32 0.51 -> ~0.92, bf16 0.63 -> ~0.79.
_SILU_BW_MAX_BLOCK = 65536


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def silu_backward_kernel_xpu(
    x_ptr, dy_ptr, out_ptr, n_elements, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    mask = tid < n_elements
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + tid, mask=mask).to(tl.float32)
    sigma = 1.0 / (1.0 + tl.exp(-x))
    dx = dy * sigma * (1.0 + x * (1.0 - sigma))
    tl.store(out_ptr + tid, dx.to(x_ptr.type.element_ty), mask=mask)


def silu(self):
    logger.debug("GEMS_KUNLUNXIN SILU")
    output = silu_forward(self)
    return output


def silu_backward(grad_output, self):
    logger.debug("GEMS_KUNLUNXIN SILU_BACKWARD")
    x = self if self.is_contiguous() else self.contiguous()
    dy = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    grad_input = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements > 0:
        block = min(triton.next_power_of_2(n_elements), _SILU_BW_MAX_BLOCK)
        grid = (triton.cdiv(n_elements, block), 1, 1)
        silu_backward_kernel_xpu[grid](
            x,
            dy,
            grad_input,
            n_elements,
            BLOCK=block,
            num_warps=16,
            buffer_size_limit=4096,
        )
    if grad_input.shape != self.shape or grad_input.stride() != self.stride():
        grad_input = grad_input.reshape(self.shape).as_strided(
            self.size(), self.stride()
        )
    return grad_input


def silu_(A):
    logger.debug("GEMS_KUNLUNXIN SILU_")
    out = silu_forward(A, out0=A)
    return out
