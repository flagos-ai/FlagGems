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

from flag_gems.runtime import torch_device_fn

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
# round-half-to-even via magic-number RN  arithmetic:
#   r = (x + C) - C, C = 1.5 * 2^23
# The fused add+sub rounds x to the nearest integer with ties-to-even,
# identical to torch.round for |x| < 2^22, using only fast scalar ops.
# This replaces the previous libdevice rint path, whose extern call
# collapses store bandwidth to ~60-250 GB/s on XPU (measured 1.1-1.6ms
# on 16M-element fp16/fp32 vs ~70us for this arithmetic-only kernel).
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def round_func_0(x):
    x_fp32 = x.to(tl.float32)
    return ((x_fp32 + 12582912.0) - 12582912.0).to(x.dtype)


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


# torch.round(decimals=...) is not a FlagGems op, so ATen's composite
# implementation scales the input by 10**decimals in the input dtype. On XPU
# that overflows for fp16 (10**5 is already inf) and the aten::round underneath
# saturates large magnitudes, corrupting the result (Issue #8305). Implement
# the aten::round.decimals overload directly, scaling in fp32.
_ROUND_BLOCK = 1024


@triton.jit
def _round_decimals_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    DECIMALS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    scale = 10.0**DECIMALS
    s = x * scale
    # Round half to even in fp32. For |s| >= 2**23 every representable fp32
    # value is already integral; keeping floor() away from that range also
    # avoids the XPU lowering saturating it at int32/int64.
    r = tl.floor(s)
    frac = s - r
    is_odd = tl.abs(r - 2.0 * tl.floor(r * 0.5)) > 0.5
    rounded = tl.where((frac > 0.5) | ((frac == 0.5) & is_odd), r + 1.0, r)
    y = tl.where(tl.abs(s) >= 8388608.0, s, rounded) / scale
    tl.store(out_ptr + offsets, y.to(out_ptr.type.element_ty), mask=mask)


def round_decimals(input, decimals=0):
    logger.debug("GEMS_KUNLUNXIN ROUND_DECIMALS")
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
    if decimals >= 39:
        # 10**decimals overflows fp32, so ATen's scaled round returns NaN.
        return torch.full_like(input, float("nan"))
    out = torch.empty_like(input)
    n = input.numel()
    with torch_device_fn.device(input.device):
        _round_decimals_kernel[(triton.cdiv(n, _ROUND_BLOCK),)](
            input, out, n, decimals, _ROUND_BLOCK
        )
    return out
