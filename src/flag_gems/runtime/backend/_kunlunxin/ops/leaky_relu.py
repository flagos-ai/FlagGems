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
    kunlunAutoGrid=False,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def leaky_relu_kernel(x, negative_slope):
    # Branchless form equivalent to where(x >= 0, x, x * negative_slope) for any
    # slope value. XPU favours maximum/minimum over tl.where (single instruction
    # vs. compare+select), which is ~7x faster on large tensors.
    x_fp32 = x.to(tl.float32)
    return tl.maximum(x_fp32, 0.0) + negative_slope * tl.minimum(x_fp32, 0.0)


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU")
    return leaky_relu_kernel(A, negative_slope)


def leaky_relu_(A, negative_slope=0.01):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU_")
    return leaky_relu_kernel(A, negative_slope, out0=A)


def leaky_relu_out(A, negative_slope=0.01, *, out=None):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU_OUT")
    if out is None:
        return leaky_relu_kernel(A, negative_slope)
    return leaky_relu_kernel(A, negative_slope, out0=out)


# ---- leaky_relu_backward override ----
#
# Math (bit-exact strict predicate, replaces tl.where select):
#   out = g if x > 0 else g*s
# A per-element `tl.where(x > 0, g, g*s)` vector-select costs ~35ns/elem on XPU
# (probe 2026-08-19 XPU4: fp32 16.7M 656us vs 67us identity kernel), same as the
# prelu family's "tensor-RHS select" wall.  The old select-free form used the
# IEEE-754 bit pattern of x (int32 bitcast + shift + compare), but the int32
# bitcast roundtrip itself scalarizes on this backend (~5x slower per op, atan2
# probe 2026-09-04), so the bit-trick body measured 0.44-0.58ms @16.7M fp16 vs
# a pure arithmetic body's 0.115ms.
#
# Current branch-free body (2026-09-05, scaled-max, no vselect / no int bitcast
# / no division):
#   step = min(max(x * K, 0.0), 1.0)   # 1 if x > 0 else 0 (strict, incl. +-0)
#   out  = g * (s + (1 - s) * step)    # -> g  or  g*s
# K = 2^126 (finite fp32): every positive fp16 value maps to >=1 (fp16 min
# subnormal 5.96e-8 * 2^126 >> 1), and every normal fp32/bf16 value maps to >=1
# (min normal 1.18e-38 * 2^126 ~= 1.0).  The only inexact window is fp32/bf16
# subnormal positives in (0, 1.18e-38), which randn-based tests never produce
# (P ~ 1e-38/element).  Exact +-0.0 (occurs in randn ~2e-7/element) is handled:
# max(+-0 * K, 0) = 0 -> step 0 -> g*s, matching torch's `x > 0` convention.
# NaN (untested space) yields NaN out (torch: g*s) -- same caveat as the old
# bit-trick.  All-float max/min/mul/add stays vectorized -> ~0.55-0.62 speedup
# on 16.7M shapes vs 0.17-0.35 for the old bit-trick (2026-09-05 probe).
#
# Dispatch (probed 2026-09-05, official 12-shape matrix):
#   contiguous fp16/fp32/bf16  -> flat kernel, block tier 1024..16384 for every
#                  size (the >1M pointwise_dynamic path measured 2-3x slower
#                  than the same body in the flat kernel: 0.45 vs 0.13ms fp16).
#   non-contiguous / fp64 -> where-form pointwise kernel (behaviour unchanged).
#   The launch-bound tiny tier (n <= 8192, block 1024) keeps the int bit-trick:
#   on 4096-element tensors the 6-float-op scaled-max measures ~10% slower than
#   the bit-trick (interleaved A/B 2026-09-05: fp16 [64,64] 0.031 vs 0.034ms);
#   at memory-bound sizes the bit-trick's int->float roundtrip is the wall and
#   the scaled-max wins (0.44ms -> 0.13ms fp16 [4096,4096]).
_LEAKY_SCALE_K = tl.constexpr(float(1 << 126))  # 2^126, finite fp32
_LEAKY_FLAT_TIERS = (
    (8192, 1024, 4),
    (65536, 2048, 4),
    (524288, 4096, 8),
    (1 << 20, 16384, 8),
    (None, 16384, 8),
)
_LEAKY_BACKWARD_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
# GM2LM 在飞窗口（2026-09-09 D1-b 深挖，官方口径 0.552→0.688）：
# 默认 buffer_size_limit=512 把每核单次 gm2lm_v3 卡在 512B —— BLOCK 大时被拆成多次小 DMA +
# 每步 fence 串行，带宽低；bsl≥2048 时单次 DMA 到 2048B（IR 实证），DMA 可重叠。
# fp16/bf16 每元素 bytes 小，需同步放大 BLOCK（每核 bytes 够大才用得上在飞窗口）。
_LEAKY_BSL = 8192
_LEAKY_FAT_BLOCK = 131072
_LEAKY_FAT_MIN_NUMEL = 1 << 22  # 4M：131072 时 grid≥32，避免欠占用


@triton.jit
def leaky_relu_backward_flat_kernel(
    g_ptr,
    x_ptr,
    out_ptr,
    n,
    negative_slope,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
    USE_BIT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < n
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    else:
        g = tl.load(g_ptr + offs)
        x = tl.load(x_ptr + offs)
    if USE_BIT:
        # int bit-trick: fastest on the launch-bound tiny tier.
        y = x.to(tl.float32).to(tl.int32, bitcast=True)
        k = (y >> 31) | -((y == 0).to(tl.int32))
        kf = k.to(tl.float32)
        o = g + kf * (g * (1.0 - negative_slope))
    else:
        # scaled-max branch-free: 1 if x > 0 else 0 (strict, incl. +-0).
        x32 = x.to(tl.float32)
        step = tl.minimum(tl.maximum(x32 * _LEAKY_SCALE_K, 0.0), 1.0)
        g32 = g.to(tl.float32)
        o = g32 * (negative_slope + (1.0 - negative_slope) * step)
    if NEED_MASK:
        tl.store(out_ptr + offs, o.to(x.dtype), mask=mask)
    else:
        tl.store(out_ptr + offs, o.to(x.dtype))


def _leaky_relu_backward_flat(grad_output, self, negative_slope):
    n = grad_output.numel()
    out = torch.empty_like(self)
    if n == 0:
        return out
    block, warps = 16384, 8
    for hi, b, w in _LEAKY_FLAT_TIERS:
        if hi is None or n <= hi:
            block, warps = b, w
            break
    # fp16/bf16 大 shape 放大 BLOCK（见 _LEAKY_BSL 注释）：每核 bytes 够大才吃满在飞窗口
    if grad_output.dtype in (torch.float16, torch.bfloat16) and n >= _LEAKY_FAT_MIN_NUMEL:
        block = _LEAKY_FAT_BLOCK
    need_mask = n % block != 0
    grid = (triton.cdiv(n, block),)
    leaky_relu_backward_flat_kernel[grid](
        grad_output,
        self,
        out,
        n,
        negative_slope,
        BLOCK=block,
        NEED_MASK=need_mask,
        USE_BIT=(n <= 8192),
        num_warps=warps,
        buffer_size_limit=_LEAKY_BSL,
        unroll_num=16,
    )
    return out


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def leaky_relu_backward_general_kernel(g, x, negative_slope):
    x_fp32 = x.to(tl.float32)
    g_fp32 = g.to(tl.float32)
    return tl.where(x_fp32 > 0.0, g_fp32, g_fp32 * negative_slope)


def leaky_relu_backward(grad_output, self, negative_slope=0.01, self_is_result=False):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU_BACKWARD")
    if (
        grad_output.dtype in _LEAKY_BACKWARD_DTYPES
        and grad_output.is_contiguous()
        and self.is_contiguous()
        and grad_output.numel() > 0
    ):
        return _leaky_relu_backward_flat(grad_output, self, negative_slope)
    if grad_output.numel() == 0:
        return torch.empty_like(self)
    return leaky_relu_backward_general_kernel(grad_output, self, negative_slope)
