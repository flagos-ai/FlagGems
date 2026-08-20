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
from triton.language.extra.xpu.libdevice import rint as _rint

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
    unroll_num=8,
)


# rint(fp32) implements round-half-to-even, matching torch.round semantics.
# XPU libdevice rint only supports fp32, so always cast to fp32 for computation.
# The scale trick handles non-zero decimals: round(x, d) = rint(x * 10^d) / 10^d.
@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def round_func(x, scale):
    x_fp32 = x.to(tl.float32)
    return (_rint(x_fp32 * scale) / scale).to(x.dtype)


# decimals==0 fast path: a single-tensor kernel (no scalar arg, no mul/div).
# The scalar `scale` argument in round_func makes it a mixed tensor+scalar
# kernel whose per-element runtime multiply/divide collapses the store
# bandwidth to ~250 GB/s (half of the native single-tensor path). Since
# decimals==0 is the common case (and the only one the benchmark exercises),
# route it through this scalar-free kernel that matches the floor/ceil/trunc
# family speed (~500 GB/s).
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def round_func_0(x):
    return _rint(x.to(tl.float32)).to(x.dtype)


def _scale(decimals):
    return 10.0**decimals


def round(input, decimals=0):
    logger.debug("GEMS_KUNLUNXIN ROUND")
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")
    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        return input.clone()
    if input.numel() == 0:
        return torch.empty_like(input)
    if not input.is_contiguous():
        input = input.contiguous()
    if decimals == 0:
        return round_func_0(input)
    return round_func(input, _scale(decimals))


def round_out(input, *, decimals=0, out=None):
    logger.debug("GEMS_KUNLUNXIN ROUND_OUT")
    if out is None:
        return round(input, decimals=decimals)
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")
    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        out.copy_(input)
        return out
    if input.numel() == 0:
        return out
    if not input.is_contiguous():
        input = input.contiguous()
    if decimals == 0:
        round_func_0(input, out0=out)
    else:
        round_func(input, _scale(decimals), out0=out)
    return out


def round_(input, *, decimals=0):
    logger.debug("GEMS_KUNLUNXIN ROUND_")
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")
    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        return input
    if input.numel() == 0:
        return input
    if not input.is_contiguous():
        raise ValueError(
            "round Triton kernel currently supports only contiguous tensors."
        )
    if decimals == 0:
        round_func_0(input, out0=input)
    else:
        round_func(input, _scale(decimals), out0=input)
    return input
