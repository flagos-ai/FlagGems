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

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# lerp_tensor is a memory-bound elementwise op (reads 3 tensors, writes 1). The
# previous override computed the numerically-stable TWO-branch form
#   where(|w|<0.5, i + w*(e-i), e - (e-i)*(1-w))
# in fp32. On XPU that `tl.where` + second branch is compiled into a heavy path
# that pins gems latency at ~0.34ms for ALL dtypes (fp16/fp32/bf16 identical) ->
# ~4x slower than torch on 4096^2 and NOT memory-bound. The two-branch is purely
# a torch precision trick for weight near 1; both branches are algebraically
# `input + weight*(end-input)` (the definition of lerp), so the single head form
# is kept. Vec-open config (buffer 4096 + unroll8, same as the addcmul sibling)
# is tied-or-better everywhere.
#
# The head form is also left in the operand dtype rather than computed via an
# explicit fp32 upcast. The upcast cost fp16 a measurable amount for nothing:
# dropping it takes 4096^2 fp16 from 0.100 to 0.089ms (0.75 -> 0.85 of torch)
# and changes neither fp32 nor bf16, and tests/test_lerp.py still passes for
# all three dtypes including its out-of-range weight=5.0 cases. Narrow-dtype
# loads are converted at the load/store boundary by the XPU compiler anyway, so
# the arithmetic still happens in fp32 internally.
#
# isCloseMemoryAsync=False (i.e. async gm2lm/lm2gm enabled) is the single
# largest remaining win. With the default synchronous copies the three input
# streams are fetched back-to-back and the tile cannot overlap DMA with the
# multiply-add, which left fp16/bf16 at roughly half of torch while moving half
# the bytes of the fp32 case -- a sign the tile was latency- rather than
# bandwidth-bound. Overlapping the copies on 4096^2: fp16 0.158->0.100ms
# (0.47->0.75 of torch), fp32 0.182->0.139 (0.79->1.04), bf16 0.256->0.193
# (0.43->0.57). Mean speedup over the core shape/dtype grid 0.63->0.80.
#
# bf16 stays the outlier because its vextf/vtruncf converts, not memory, are
# the limit: the same bytes reinterpreted as fp16 run at 1.54 TB/s versus
# 0.87 TB/s as bf16, while a single-stream copy reaches 3.68 TB/s. torch is
# itself only at 1.23 TB/s for bf16 against 1.86 for fp16, so both sides pay
# it. Packing bf16 pairs into int32 and converting by shifts was tried and is
# worse (0.35 of torch); isCloseDtypeConvert fails to compile here.
#
# A larger tile would help further (fp32 reaches 0.98 of torch at 32768+
# elements) but tile_size is capped at 32768 by lane_limit in
# pointwise_dynamic, and 32768 exactly is deliberately bounced down to 16384
# there because it wedges the P800 with an illegal memory access (#5748).
# Both are shared by every pointwise op, so that is left alone.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
    isCloseMemoryAsync=False,
)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def lerp_tensor_kernel(input, end, weight):
    return input + weight * (end - input)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit(do_not_specialize=["weight"])
def lerp_scalar_kernel_head(input, end, weight):
    input32 = input.to(tl.float32)
    end32 = end.to(tl.float32)
    weight32 = weight.to(tl.float32)
    return (input32 + weight32 * (end32 - input32)).to(input.dtype)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit(do_not_specialize=["weight"])
def lerp_scalar_kernel_tail(input, end, weight):
    input32 = input.to(tl.float32)
    end32 = end.to(tl.float32)
    weight32 = weight.to(tl.float32)
    return (end32 - (end32 - input32) * (1 - weight32)).to(input.dtype)


def lerp_tensor(input, end, weight):
    logger.debug("GEMS_KUNLUNXIN LERP_TENSOR")
    out = lerp_tensor_kernel(input, end, weight)
    return out


def lerp_tensor_(input, end, weight):
    logger.debug("GEMS_KUNLUNXIN LERP_TENSOR_")
    return lerp_tensor_kernel(input, end, weight, out0=input)


def lerp_scalar(input, end, weight):
    logger.debug("GEMS_KUNLUNXIN LERP_SCALAR")
    if weight < 0.5:
        out = lerp_scalar_kernel_head(input, end, weight)
    else:
        out = lerp_scalar_kernel_tail(input, end, weight)
    return out


def lerp_scalar_(input, end, weight):
    logger.debug("GEMS_KUNLUNXIN LERP_SCALAR_")
    if weight < 0.5:
        return lerp_scalar_kernel_head(input, end, weight, out0=input)
    else:
        return lerp_scalar_kernel_tail(input, end, weight, out0=input)
