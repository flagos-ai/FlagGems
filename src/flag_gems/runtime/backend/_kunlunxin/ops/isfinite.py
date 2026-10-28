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
# vectorized compare-store.  Masked loads/stores also block vectorized store
# (the pointwise_dynamic auto-grid path generates an always-true mask), hence
# the flat unmasked fast kernel.
#
# TRITONXPU_COMPARE_FUSION fuses the compare into a 32-bit triton_xpu.cmpf
# result, which the old flagtree triton shipped in the CI container (flagtree
# 0.7.0+xpu3.6, an older fork pinned by the CI) rejects in the op verifier
# ("'triton_xpu.cmpf' op result #0 ... but got 'tensor<...xui32...>'"),
# failing every isfinite test there; the plain path is accepted by both
# versions.  Detect it once here and set the fusion env (see isfinite below)
# only on the local triton build that legalizes it, so the flagtree path stays
# byte-identical to the plain compare.
_IS_FLAGTREE = os.path.exists(
    os.path.join(os.path.dirname(triton.__file__), "FLAGTREE_BACKEND")
)
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
    # Both half dtypes get the 131072-lane tile (isneginf _pick_tile style):
    # bf16 must be widened to f32 for the compare anyway (no bf16 ALU on
    # XPU3) and the wider tile measured faster there (61.7us vs 67.5us on
    # 16.7M elements).
    if dtype in (torch.float16, torch.bfloat16):
        return _FUSION_TILE_HUGE_FP16, 8
    return _FUSION_TILE_HUGE_FP32, 8


@triton.jit
def _isfinite_fast_kernel(x_ptr, o_ptr, TILE: tl.constexpr, NATIVE_F16: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + off)
    if NATIVE_F16:
        # isfinite(x) == |x| < +inf compared in f16: the 0-d tl.full constant
        # (unlike a Python float literal, which the front-end promotes to f32)
        # keeps the compare in f16, so the backend emits the 32-lane vvltfp16
        # instead of the 16-lane f32 compare -- 61.9us -> 40.9us on 16.7M f16.
        # Exact for every value: +-inf/NaN -> False (unordered compare),
        # +-0/subnormal/finite -> True.
        r = (tl.abs(x) < tl.full((), float("inf"), x.dtype)).to(tl.int8)
    else:
        # isfinite(x) == |x| < +inf: exactly False for +/-inf and NaN, True for
        # every finite value (incl. +/-0 and subnormals).  Single FP compare
        # keeps TRITONXPU_COMPARE_FUSION eligible (no AND/OR/arithmetic in
        # between).
        r = (tl.abs(x.to(tl.float32)) < float("inf")).to(tl.int8)
    tl.store(o_ptr + off, r)


@triton.jit
def _isfinite_fast_masked_kernel(
    x_ptr, o_ptr, n_elements, TILE: tl.constexpr, NATIVE_F16: tl.constexpr
):
    pid = tl.program_id(0)
    off = pid * TILE + tl.arange(0, TILE)
    mask = off < n_elements
    x = tl.load(x_ptr + off, mask=mask, other=0.0)
    if NATIVE_F16:
        r = (tl.abs(x) < tl.full((), float("inf"), x.dtype)).to(tl.int8)
    else:
        r = (tl.abs(x.to(tl.float32)) < float("inf")).to(tl.int8)
    tl.store(o_ptr + off, r, mask=mask)


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
        # f16 keeps the compare in f16 (32-lane vvltfp16) only on the local
        # triton build; bf16 must be widened to f32 (no bf16 ALU on XPU3) and
        # f32 already compares in f32, so both keep the widened body.  On
        # flagtree the NATIVE_F16 branch is constexpr-elided to the plain body.
        native_f16 = (not _IS_FLAGTREE) and A.dtype == torch.float16
        if not _IS_FLAGTREE:
            # Read at kernel JIT-compile time, so set before the launch and
            # restore afterwards; TRITONXPU_FP16_FAST must be explicitly "0"
            # (its getBoolEnv default) to survive a global "1" -- with it on,
            # the .to(f32) widening trips a TritonXPUDtypeConvert compile
            # failure (same constraint as isneginf).
            os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
            os.environ["TRITONXPU_FP16_FAST"] = "0"
        try:
            if numel % tile == 0:
                _isfinite_fast_kernel[(numel // tile,)](
                    A,
                    out,
                    TILE=tile,
                    NATIVE_F16=native_f16,
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
                    NATIVE_F16=native_f16,
                    num_warps=num_warps,
                    buffer_size_limit=_FUSION_BUFFER_SIZE_LIMIT,
                    unroll_num=_FUSION_UNROLL_NUM,
                    isCloseMemoryAsync=False,
                )
        finally:
            if not _IS_FLAGTREE:
                os.environ.pop("TRITONXPU_COMPARE_FUSION", None)
                os.environ.pop("TRITONXPU_FP16_FAST", None)
        return out
    return isfinite_func(A)
