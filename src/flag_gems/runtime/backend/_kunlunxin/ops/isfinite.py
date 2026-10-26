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

from flag_gems.utils import tl_extra_shim

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

try:
    _isfinited = tl_extra_shim.isfinited
    _finitef = tl_extra_shim.finitef
except Exception:
    pass

# Flat 1-D pointwise kernel (isinf/isneginf recipe).  `|x| < +inf` lowers to a
# plain cmpf -> extui -> store chain, which measures faster than (a) the
# libdevice _finitef extern (llvm.call per lane + i32->i8 trunc; 126us vs 68us
# on 16.8M fp32 -- the 68us figure was measured with
# TRITONXPU_COMPARE_FUSION=1) and (b) arithmetic/bitwise variants
# (`(x-x)==0`, `(bits&MASK)!=MASK`, `x*0==0`) whose extra op blocks
# vectorized compare-store.  The compare fusion is intentionally NOT enabled:
# tritonxpu-normalize's [cmpf, extui] fusion builds triton_xpu.cmpf with a
# ui32 result, which the older triton shipped in the CI container rejects in
# the op verifier ("'triton_xpu.cmpf' op result #0 ... but got
# 'tensor<...xui32...>'"), failing every isfinite test there; the plain path is
# accepted by both versions.  The previous override also tried a bitwise
# uint32 path; the full-tile `tl.full` mask materialization made fp32
# [1024, 256] take ~2.7ms, so it was dropped.  Masked loads/stores also block
# vectorized store (the pointwise_dynamic auto-grid path generates an
# always-true mask), hence the flat unmasked fast kernel.
_FUSION_TILE_SMALL = 1024
_FUSION_TILE_MID = 4096
_FUSION_TILE_LARGE = 8192
_FUSION_TILE_HUGE_FP16 = 131072
_FUSION_TILE_HUGE_FP32 = 65536
_FUSION_SMALL_NUMEL = 2048
_FUSION_MID_NUMEL = 16384
_FUSION_LARGE_NUMEL = 262144
_FUSION_BUFFER_SIZE_LIMIT = 8192
_FUSION_UNROLL_NUM = 16


def _pick_fast_tile(numel: int, dtype: torch.dtype) -> tuple:
    if numel <= _FUSION_SMALL_NUMEL:
        return _FUSION_TILE_SMALL, 2
    if numel <= _FUSION_MID_NUMEL:
        return _FUSION_TILE_MID, 4
    if numel <= _FUSION_LARGE_NUMEL:
        return _FUSION_TILE_LARGE, 8
    if dtype == torch.float16:
        return _FUSION_TILE_HUGE_FP16, 8
    return _FUSION_TILE_HUGE_FP32, 8


@triton.jit
def _isfinite_fast_kernel(x_ptr, o_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + off)
    # isfinite(x) == |x| < +inf: exactly False for +/-inf and NaN, True for
    # every finite value (incl. +/-0 and subnormals).  Single FP compare keeps
    # TRITONXPU_COMPARE_FUSION eligible (no AND/OR/arithmetic in between).
    tl.store(o_ptr + off, (tl.abs(x.to(tl.float32)) < float("inf")).to(tl.int8))


@triton.jit
def _isfinite_fast_masked_kernel(x_ptr, o_ptr, n_elements, TILE: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * TILE + tl.arange(0, TILE)
    mask = off < n_elements
    x = tl.load(x_ptr + off, mask=mask, other=0.0)
    tl.store(
        o_ptr + off,
        (tl.abs(x.to(tl.float32)) < float("inf")).to(tl.int8),
        mask=mask,
    )


# Fallback for non-contiguous / fp64 / otherwise unsupported inputs.  float64
# -> float32 cast would overflow large-but-finite values to inf, so keep the
# dedicated fp64 intrinsic; every other float dtype is exact after the upcast
# (isfinite only inspects the exponent).
_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=16,
)


@pointwise_dynamic(
    is_tensor=[True], promotion_methods=[(0, "ALWAYS_BOOL")], config=_config
)
@triton.jit
def isfinite_func(x):
    return _isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32))


def isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ISFINITE")
    if not A.is_floating_point():
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
    if A.dtype in (torch.float16, torch.float32, torch.bfloat16) and A.is_contiguous():
        numel = A.numel()
        if numel == 0:
            return torch.empty_like(A, dtype=torch.bool)
        tile, num_warps = _pick_fast_tile(numel, A.dtype)
        out = torch.empty_like(A, dtype=torch.bool)
        if numel % tile == 0:
            _isfinite_fast_kernel[(numel // tile,)](
                A,
                out,
                TILE=tile,
                num_warps=num_warps,
                buffer_size_limit=_FUSION_BUFFER_SIZE_LIMIT,
                unroll_num=_FUSION_UNROLL_NUM,
                isCloseMemoryAsync=False,
            )
        else:
            _isfinite_fast_masked_kernel[(triton.cdiv(numel, tile),)](
                A,
                out,
                numel,
                TILE=tile,
                num_warps=num_warps,
                buffer_size_limit=_FUSION_BUFFER_SIZE_LIMIT,
                unroll_num=_FUSION_UNROLL_NUM,
                isCloseMemoryAsync=False,
            )
        return out
    return isfinite_func(A)
