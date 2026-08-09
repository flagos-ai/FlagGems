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
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
_isnan = tl_extra_shim.isnan

_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=_config)
@triton.jit
def isnan_func(x):
    # Convert to float32 for consistent NaN detection across dtypes
    return _isnan(x.to(tl.float32))


# ---------------------------------------------------------------------------
# isnan fast path (contiguous fp16/fp32/bf16, numel >= per-dtype gate).
#
# Why: the alternate generic path (pointwise_dynamic 1d-tile codegen) stores
# the vendor is-nan `i32` result through the ALWAYS_BOOL conversion, an
# elementwise i32 -> bool per-lane store that on XPU costs ~2.2ns/elem on top
# of the (free) isnan intrinsic. Writing the 0/1 result as int8 -- bit-
# identical to a bool tensor -- through a flat unmasked tile instead cuts
# 11-17% on the mid/large benchmark shapes (probe 2026-08-13, XPU 6:
# fp32 [1024,65536] 432.3us -> ~361us, fp16 [4096,4096] 97.4us -> 91.1us,
# bf16 [1024,65536] 365.2us -> 324.6us), at the price of a ~6us launch floor
# on small tensors (kept on the generic path below the per-dtype gate).
#
# The returned tensor is `uint8` storage viewed as bool -- one byte per
# element with 0/1 payloads, i.e. bit-identical to what the generic path (and
# the native torch kernel) produce, so no value conversion ever happens
# (uint8 -> bool `view` is a pure metadata reinterpretation).
_FAST_TILE = 131072
_FAST_MIN_NUMEL_F16 = 1 << 20  # fp16/bf16: flat tile beats generic from 1M
_FAST_MIN_NUMEL_F32 = 1 << 22  # fp32: 4M (1M fp32 flat tile benchmarked slower)
_FAST_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@triton.jit
def isnan_fast_kernel(out_ptr, x_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    r = _isnan(x).to(tl.int8)
    tl.store(out_ptr + tid, r)


@triton.jit
def isnan_fast_masked_kernel(out_ptr, x_ptr, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    m = tid < numel
    x = tl.load(x_ptr + tid, mask=m).to(tl.float32)
    r = _isnan(x).to(tl.int8)
    tl.store(out_ptr + tid, r, mask=m)


def _isnan_fast(A):
    numel = A.numel()
    # NOTE: plain `torch.empty(shape, dtype=uint8)` is pathologically slow
    # under flag_gems.use_gems() scope on this vendor (~20-90ms/call), while
    # torch.empty_like is the fast patched path used by the pointwise codegen;
    # A is contiguous here so empty_like keeps the exact shape/layout.
    out8 = torch.empty_like(A, dtype=torch.uint8)
    if numel % _FAST_TILE == 0:
        isnan_fast_kernel[(numel // _FAST_TILE,)](
            out8,
            A,
            TILE=_FAST_TILE,
            num_warps=4,
            buffer_size_limit=8192,
            unroll_num=16,
            isCloseMemoryAsync=False,
        )
    else:
        isnan_fast_masked_kernel[(math.ceil(numel / _FAST_TILE),)](
            out8,
            A,
            numel,
            TILE=_FAST_TILE,
            num_warps=4,
            buffer_size_limit=8192,
            unroll_num=16,
            isCloseMemoryAsync=False,
        )
    return out8.view(torch.bool)


def isnan(A):
    logger.debug("GEMS_KUNLUNXIN ISNAN")
    if (
        A.is_floating_point()
        and A.is_contiguous()
        and A.dtype in _FAST_FLOAT_DTYPES
        and A.numel()
        >= (_FAST_MIN_NUMEL_F32 if A.dtype == torch.float32 else _FAST_MIN_NUMEL_F16)
    ):
        return _isnan_fast(A)
    return isnan_func(A)