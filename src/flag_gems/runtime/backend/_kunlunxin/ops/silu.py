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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
div_rn = tl_extra_shim.div_rn

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


# silu_backward_kernel was config-less: on XPU a bare pointwise_dynamic
# recompiles per shape (tile<512>) and never unrolls -> large shapes stall at
# ~0.32 gems speedup. Reuse silu_forward's tuned config_ (vec CLOSE + unroll8):
# a swept comparison showed all unroll8 variants land at ~0.55ms for
# [4096,4096] fp16 (vs 0.80ms config-less, ~1.45x) with bit-identical output;
# vec OPEN spiked to 28.9ms on fp32 [1024,65536] so keep isCloseVectorization.
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def silu_backward_kernel(x, dy):
    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    return dx


# silu_backward tiny fast path (contiguous fp16/fp32/bf16, numel <= 2048):
# a flat 1D masked/unmasked kernel that skips the pointwise_dynamic wrapper.
# Measurement on XPU 5 (2026-08-19, official 12-shape matrix, do_bench A/B):
# at numel <= 2048 the pointwise codegen (kunlunAutoGrid=False) pays a fixed
# ~8us wrapper/grid overhead per call (e.g. [1024,1] fp16 15.5us vs 7.1us flat);
# at numel > 2048 the tuned pointwise config_ is strictly faster than every
# flat/NEED_MASK tier (B2048..B32768 x w4..16) and every CodeGenConfig variant
# (unroll 8/16/32 x buffer 4096/8192/16384 x tile 256/512/1024 x autogrid),
# so only the tiny window uses the flat kernel. Math bit-identical to
# silu_backward_kernel (fp32 staging, div_rn, downcast at store).
_TINY_MAX_NUMEL = 2048
_TINY_BLOCK = 2048
_TINY_WARPS = 4


@triton.jit
def silu_backward_tiny_kernel(g_ptr, x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr, NEED_MASK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        dy = tl.load(g_ptr + offs, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offs)
        dy = tl.load(g_ptr + offs)
    x_fp32 = x.to(tl.float32)
    dy_fp32 = dy.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    if NEED_MASK:
        tl.store(out_ptr + offs, dx.to(x.dtype), mask=mask)
    else:
        tl.store(out_ptr + offs, dx.to(x.dtype))


def _silu_backward_tiny(grad_output, self):
    numel = grad_output.numel()
    out = torch.empty_like(self)
    if numel == 0:
        return out
    if numel == _TINY_BLOCK:
        silu_backward_tiny_kernel[(1,)](
            grad_output, self, out, numel,
            BLOCK=_TINY_BLOCK, NEED_MASK=False, num_warps=_TINY_WARPS,
        )
    else:
        silu_backward_tiny_kernel[(1,)](
            grad_output, self, out, numel,
            BLOCK=_TINY_BLOCK, NEED_MASK=True, num_warps=_TINY_WARPS,
        )
    return out


def silu(self):
    logger.debug("GEMS_KUNLUNXIN SILU")
    output = silu_forward(self)
    return output


def silu_backward(grad_output, self):
    logger.debug("GEMS_KUNLUNXIN SILU_BACKWARD")
    if (
        grad_output.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and grad_output.is_contiguous()
        and self.is_contiguous()
        and grad_output.numel() <= _TINY_MAX_NUMEL
    ):
        return _silu_backward_tiny(grad_output, self)
    grad_input = silu_backward_kernel(self, grad_output)
    return grad_input


def silu_(A):
    logger.debug("GEMS_KUNLUNXIN SILU_")
    out = silu_forward(A, out0=A)
    return out
