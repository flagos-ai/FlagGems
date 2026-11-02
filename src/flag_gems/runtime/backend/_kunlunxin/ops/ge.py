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
import os

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
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def ge_func(x, y):
    return x.to(tl.float32) >= y


# ---------------------------------------------------------------------------
# Flat fast path for same-dtype contiguous tensors.
#
# Measured problem with the generic pointwise_dynamic route: its wrapper caps
# tile_size with `lane_limit = 32 * 1024` (see _kunlunxin/utils/pointwise_dynamic.py),
# so a 16.7M-element tensor is split into 512+ tiles -> ~1032 CTAs.  The P800
# schedules pointwise work over 12 clusters, so those CTAs execute in ~86 waves
# and the kernel lands at ~0.61x of torch.ge (fp16) even though the work is
# purely bandwidth bound.
#
# A flat kernel that takes one monolithic tile per CTA needs only
# numel/131072 == 128 CTAs for the same tensor and reaches ~0.99x.  The
# sweep over (tile, num_warps, buffer_size_limit, unroll_num) put the optimum at
# TILE = 131072 with num_warps=16 for large shapes; below TILE the measured best
# is a single CTA that hugs numel (tile = next_pow2(numel)), which is what the
# small-shape branch below produces.
#
# bf16 note: XPU3 has no native bf16 compare, so the widening chain bounds the
# op at ~0.78x regardless of tiling.  Integer-key and bitcast-to-fp32 compare
# substitutes were both tried and blow the uni_sram budget
# (`OutOfResources: out of resource: uni_sram`), so bf16 accepts the native
# compare.  The fp16/fp32 paths are near parity.
_GE_FLAT_TILE = 131072
_GE_FLAT_WARPS_LARGE = 16


@triton.jit
def ge_flat_kernel(out_ptr, x_ptr, y_ptr, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * TILE + tl.arange(0, TILE)
    mask = off < numel
    x = tl.load(x_ptr + off, mask=mask)
    y = tl.load(y_ptr + off, mask=mask)
    tl.store(out_ptr + off, x.to(tl.float32) >= y, mask=mask)


def _ge_flat(A, B):
    numel = A.numel()
    out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
    if numel == 0:
        return out
    tile = min(_GE_FLAT_TILE, max(128, triton.next_power_of_2(numel)))
    grid = (triton.cdiv(numel, tile),)
    ge_flat_kernel[grid](
        out,
        A.reshape(-1),
        B.reshape(-1),
        numel,
        TILE=tile,
        num_warps=_GE_FLAT_WARPS_LARGE if tile >= 65536 else 4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    return out


def ge(A, B):
    logger.debug("GEMS_KUNLUNXIN GE")
    if (
        A.dtype == B.dtype
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and A.is_contiguous()
        and B.is_contiguous()
        and A.shape == B.shape
        and A.device == B.device
    ):
        # Env must be set before the (first) launch so the fusion pass sees it
        # at compile time -- same set/del pattern as the generic path below.
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
        os.environ["TRITONXPU_FP16_FAST"] = "1"
        try:
            return _ge_flat(A, B)
        finally:
            del os.environ["TRITONXPU_COMPARE_FUSION"]
            del os.environ["TRITONXPU_FP16_FAST"]
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = ge_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def ge_func_scalar(x, y):
    return x.to(tl.float32) >= y


def ge_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN GE_SCALAR")
    res = ge_func_scalar(A, B)
    return res


# greater_equal_ is the in-place alias of ge_ (out = (x >= y) written back into
# x). It was NOT overridden by kunlunxin, so it fell to the generic
# ops/greater_equal.py which calls the generic ops/ge.py::ge_func -- a bare
# `@pointwise_dynamic` with NO CodeGenConfig -> discrete / launch-bound slow
# path. Baseline (IR ir-greater_equal_-dev0): large shapes ~0.005-0.011
# ([64,64,65536] gems ~5.2s), total avg gems speedup ~0.0797.
#
# Fix: a dedicated tuned pointwise_dynamic that writes in place (out0=A).
# CRITICAL: this in-place variant must NOT reuse ge's config_ -- ge's config has
# `isCloseMemoryAsync=False` (async memory copy ON), and with in-place aliasing
# (input tensor == output tensor) the async double-buffered copy path deadlocks
# the device ("noc idle timeout" hang, confirmed on device 6). The out-of-place
# ge is fine because its output is a fresh bool tensor (no aliasing). So use a
# config with the DEFAULT isCloseMemoryAsync (True = async closed), mirroring the
# proven in-place bool op logical_and_. Body returns tl.where(...,1,0) (int 0/1)
# which stores cleanly into A's original fp16/bf16/fp32 dtype.
config_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=config_inplace_)
@triton.jit
def greater_equal_func_(x, y):
    return tl.where(x.to(tl.float32) >= y.to(tl.float32), 1, 0)


def greater_equal_(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER_EQUAL_")
    if A.device != B.device:
        B = B.to(A.device)
    greater_equal_func_(A, B, out0=A)
    return A
