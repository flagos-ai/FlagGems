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
import struct

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=4,  # PROBE-CANDIDATE unroll4 (baseline unroll8); revert if not strictly better
)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def threshold_kernel(self, threshold, value):
    if self.dtype == tl.float16:
        big = tl.full((), 1.0e30, dtype=self.dtype)
        d = (self - threshold) * big
        m = tl.minimum(1.0, tl.maximum(0.0, d))
        return self * m + value * (1.0 - m)
    # f32 fma form v + (x - v)*m: one extra rounding on the m==1 path
    # (|err| <= 6e-8 in f32, far inside RESOLUTION), but measurably faster
    # than the two-term form on fp32/bf16.
    d = (self - threshold) * 1.0e30
    m = tl.minimum(1.0, tl.maximum(0.0, d))
    return value + (self - value) * m


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def threshold_backward_kernel(grad_output, self, threshold):
    return grad_output * (self > threshold)


_THRESHOLD_BWD_BLOCK = 16384
_THRESHOLD_BWD_BLOCK_SMALL = 8192
_THRESHOLD_BWD_WARPS = 1


@triton.jit
def _threshold_backward_bits_kernel(
    grad,
    self,
    out,
    n_elements,
    threshold_bits,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        m = offs < n_elements
        x = tl.load(grad + offs, mask=m)
        y = tl.load(self + offs, mask=m)
        yb = y.to(tl.float32).to(tl.uint32, bitcast=True)
        keep = (yb > threshold_bits) & (yb < 0x7F800000)
        tl.store(out + offs, x * keep.to(x.dtype), mask=m)
    else:
        x = tl.load(grad + offs)
        y = tl.load(self + offs)
        yb = y.to(tl.float32).to(tl.uint32, bitcast=True)
        keep = (yb > threshold_bits) & (yb < 0x7F800000)
        tl.store(out + offs, x * keep.to(x.dtype))


def threshold(self, threshold, value):
    logger.debug("GEMS_KUNLUNXIN THRESHOLD")
    output = threshold_kernel(self, threshold, value)
    return output


def threshold_(self, threshold, value):
    logger.debug("GEMS_KUNLUNXIN THRESHOLD_")
    threshold_kernel(self, threshold, value, out0=self)
    return self


def threshold_backward(grad_output, self, threshold):
    logger.debug("GEMS_KUNLUNXIN THRESHOLD_BACKWARD")
    use_bits = (
        grad_output.is_contiguous()
        and self.is_contiguous()
        and grad_output.dtype == self.dtype
        and grad_output.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and grad_output.numel() >= 8192
    )
    if use_bits:
        tbits = struct.unpack("I", struct.pack("f", float(threshold)))[0]
        if tbits < 0x80000000:  # non-negative fp32 threshold keeps the bit test exact
            n = grad_output.numel()
            out = torch.empty_like(grad_output)
            if n == 0:
                return out
            block = (
                _THRESHOLD_BWD_BLOCK
                if n >= _THRESHOLD_BWD_BLOCK
                else _THRESHOLD_BWD_BLOCK_SMALL
            )
            need_mask = (n % block) != 0
            grid = (triton.cdiv(n, block),)
            _threshold_backward_bits_kernel[grid](
                grad_output,
                self,
                out,
                n,
                tbits,
                BLOCK=block,
                NEED_MASK=need_mask,
                num_warps=_THRESHOLD_BWD_WARPS,
            )
            return out
    grad_input = threshold_backward_kernel(grad_output, self, threshold)
    return grad_input
