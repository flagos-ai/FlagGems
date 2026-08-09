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

from flag_gems.utils import tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
exp2 = tl_extra_shim.exp2


config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def sigmoid_forward(x):
    # log2e: tl.constexpr = math.log2(math.e)
    # triton 3.0.0 disallow calling non-jitted function inside jitted function, even if it is in
    # the rhs of an assignment to a constexpr, so we use numeric literal instead to work around this.
    # log2e: tl.constexpr = 1.4426950408889634
    return 1 / (1 + tl.exp(-x.to(tl.float32)))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_backward_kernel(dy, y):
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    return dy_f32 * (1.0 - y_f32) * y_f32


# sigmoid_backward fast path (contiguous fp16/fp32/bf16, small/medium
# tensors): a flat 1D kernel that skips the pointwise_dynamic wrapper
# machinery. With the gradients computed through torch.autograd.grad (as the
# official benchmark does), the wrapper's Python-side dispatch overhead shows
# up directly in the measured latency at small shapes (~27us/call vs ~6us for
# a flat launch). Measurement on XPU 2 (12-shape official matrix): flat wins
# up to and including 1M elements; at >= 4M elements the pointwise codegen
# kernel sustains higher bandwidth (flat b16384 16.7M fp16 124us vs pointwise
# 75us), so large tensors keep the original kernel. Masked <2048-padded tiles
# (same finding as ceil). Math identical to the pointwise kernel
# (dy * y * (1-y), computed in fp32, downcast at store).
_FAST_MAX_NUMEL = 1 << 20  # flat path ceiling (1M elements)
_FAST_BLOCK = 16384
_FAST_WARPS = 8
_TINY_BLOCK = 2048
_TINY_WARPS = 4


@triton.jit
def sigmoid_backward_fast_kernel(dy_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(y_ptr + offs).to(tl.float32)
    dy = tl.load(dy_ptr + offs).to(tl.float32)
    tl.store(out_ptr + offs, (dy * (1.0 - y) * y).to(out_ptr.dtype.element_ty))


@triton.jit
def sigmoid_backward_masked_kernel(
    dy_ptr, y_ptr, out_ptr, numel, BLOCK: tl.constexpr
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask).to(tl.float32)
    tl.store(
        out_ptr + offs,
        (dy * (1.0 - y) * y).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _sigmoid_backward_fast(grad_output, output):
    numel = output.numel()
    # Allocate via empty_strided (unregistered by gems) to dodge the
    # registered-empty dispatch tax inside use_gems contexts.
    out = torch.empty_strided(
        output.shape, output.stride(), dtype=output.dtype, device=output.device
    )
    if numel == 0:
        return out
    if numel < _TINY_BLOCK:
        sigmoid_backward_masked_kernel[(1,)](
            grad_output,
            output,
            out,
            numel,
            BLOCK=_TINY_BLOCK,
            num_warps=_TINY_WARPS,
        )
        return out
    block = min(_FAST_BLOCK, triton.next_power_of_2(numel))
    if numel % block == 0:
        sigmoid_backward_fast_kernel[(numel // block,)](
            grad_output, output, out, BLOCK=block, num_warps=_FAST_WARPS
        )
    else:
        sigmoid_backward_masked_kernel[(triton.cdiv(numel, block),)](
            grad_output, output, out, numel, BLOCK=block, num_warps=_FAST_WARPS
        )
    return out


def sigmoid(self):
    logger.debug("GEMS_KUNLUNXIN SIGMOID")
    output = sigmoid_forward(self)
    return output


def sigmoid_backward(grad_output, output):
    logger.debug("GEMS_KUNLUNXIN SIGMOID_BACKWARD")
    if (
        output.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and output.is_contiguous()
        and grad_output.is_contiguous()
        and output.dim() > 0
        and output.numel() <= _FAST_MAX_NUMEL
    ):
        return _sigmoid_backward_fast(grad_output, output)
    grad_input = sigmoid_backward_kernel(grad_output, output)
    return grad_input


def sigmoid_(A):
    logger.debug("GEMS_KUNLUNXIN SIGMOID_")
    out = sigmoid_forward(A, out0=A)
    return out
