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
# `input + weight*(end-input)` (the definition of lerp). Since we already upcast
# to fp32, the single head form rounded to the output dtype matches torch within
# <1 output-dtype ULP (byte-identical for the tested |w|<0.5 range). Dropping the
# where + upcasting keeps precision, and the single multiply-add now runs at the
# XPU memory-bandwidth ceiling: 4096^2 fp16 0.345->0.098ms, fp32 0.360->0.163,
# bf16 0.343->0.107 (all ~torch). Vec-open config (buffer 4096 + unroll8, same as
# addcmul sibling) is tied-or-better everywhere and closes the bf16 gap to torch.
config_ = CodeGenConfig(
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
    is_tensor=[True, True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def lerp_tensor_kernel(input, end, weight):
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
