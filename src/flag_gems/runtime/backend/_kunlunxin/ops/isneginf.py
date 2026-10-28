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

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# A direct comparison with -inf has the same floating-point semantics as
# isinf(x) & (x < 0), but avoids the libdevice isinf extern call.
_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=_config)
@triton.jit
def isneginf_func(x):
    return x.to(tl.float32) == -float("inf")


# ---------------------------------------------------------------------------
# Fast path (fp16/bf16/fp32, contiguous): one fused compare+store kernel per
# launch. The always-true mask of the generic codegen path was the dominant
# cost on the big shapes (masked-memory path ~2x + fp32-widened compare), so
# the fast kernels are flat 1D tiles with a constexpr-split mask: exact
# multiples launch with no mask at all, anything else gets a real tail mask.
# With TRITONXPU_COMPARE_FUSION=1 the backend fuses CmpFOp(i1)+ExtUI(i8)+Store
# into one vendor compare-store intrinsic (3 B/elem, one pass). Measured on
# (4096,4096): 373us -> 67us (fp32), 389us -> 59us (fp16/bf16).
# TRITONXPU_FP16_FAST must stay off: with it on, the half-precision direct
# compare trips a TritonXPUDtypeConvert compile failure (same constraint as
# not_equal's scalar native path).
#
# The flagtree compiler shipped in the CI image (flagtree 0.7.0+xpu3.6, an
# older triton fork pinned in backends.yaml) rejects both TRITONXPU_COMPARE_FUSION
# and the tuned launch opts used below: COMPARE_FUSION fuses the compare into a
# 32-bit triton_xpu.cmpf result that TritonXPUNormalize refuses with uni_sram
# (OutOfResources) on fp16/bf16/fp32 direct compares -- the same failure le.py
# documents -- and the masked kernel launched with _FAST_LAUNCH_OPTS raises a
# kernel exception / NOC idle timeout at runtime. The newer triton build
# legalizes both, so keep the fast config there and degrade on flagtree.
_IS_FLAGTREE = os.path.exists(
    os.path.join(os.path.dirname(triton.__file__), "FLAGTREE_BACKEND")
)
_FAST_TILE_SMALL = 1024
_FAST_SMALL_MAX = 2048
_FAST_TILE_MID = 4096
_FAST_MID_MAX = 16384
_FAST_TILE_MED = 8192
_FAST_MED_MAX = 262144
# The transfer bandwidth scales with the per-program contiguous run, but the
# grid must stay wide enough to keep all channels busy; 64K (f32) / 128K (half)
# measured fastest across 4M-67M elements.
_FAST_TILE_F32 = 65536
_FAST_TILE_HALF = 131072
_FAST_LAUNCH_OPTS = dict(
    buffer_size_limit=8192,
    unroll_num=16,
    isCloseMemoryAsync=False,
)


@triton.jit
def isneginf_fast_kernel(x_ptr, out_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    tl.store(out_ptr + tid, (x == -float("inf")).to(tl.int8))


@triton.jit
def isneginf_fast_masked_kernel(x_ptr, out_ptr, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    # Clamp the load address (masked loads still issue memory accesses on XPU;
    # every address must be legal) and zero the OOB lanes' result instead of
    # masking the load, so tiny tensors (numel < TILE) never read past their
    # allocation. The store keeps its guard mask.
    tid_c = tl.minimum(tid, numel - 1)
    x = tl.load(x_ptr + tid_c)
    r = (x == -float("inf")) & (tid < numel)
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _pick_tile(numel, dtype):
    if numel <= _FAST_SMALL_MAX:
        return _FAST_TILE_SMALL, 2
    if numel <= _FAST_MID_MAX:
        return _FAST_TILE_MID, 4
    if numel <= _FAST_MED_MAX:
        return _FAST_TILE_MED, 8
    if dtype == torch.float32:
        return _FAST_TILE_F32, 8
    return _FAST_TILE_HALF, 8


def _isneginf_fast(A, out, numel):
    tile, warps = _pick_tile(numel, A.dtype)
    if not _IS_FLAGTREE:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "0"
    # The tuned launch opts crash the flagtree compiler at runtime (kernel
    # exception / NOC idle timeout, e.g. the fp16 (16,7,57,32,29) masked case),
    # while the same kernel with a plain launch runs and is correct there.
    # Keep the tuned opts on the newer triton build that legalizes them.
    launch_opts = {} if _IS_FLAGTREE else _FAST_LAUNCH_OPTS
    try:
        if numel % tile == 0:
            isneginf_fast_kernel[(numel // tile,)](
                A, out, TILE=tile, num_warps=warps, **launch_opts
            )
        else:
            isneginf_fast_masked_kernel[(triton.cdiv(numel, tile),)](
                A, out, numel, TILE=tile, num_warps=warps, **launch_opts
            )
    finally:
        os.environ.pop("TRITONXPU_COMPARE_FUSION", None)
        os.environ.pop("TRITONXPU_FP16_FAST", None)
    return out


def isneginf(A):
    logger.debug("GEMS_KUNLUNXIN ISNEGINF")
    numel = A.numel()
    if (
        A.is_contiguous()
        and A.dtype
        in (
            torch.float16,
            torch.float32,
            torch.bfloat16,
        )
        and numel > 0
    ):
        out = torch.empty_like(A, dtype=torch.bool)
        return _isneginf_fast(A, out, numel)
    return isneginf_func(A)


def isneginf_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ISNEGINF_OUT")
    numel = A.numel()
    if (
        out is not None
        and A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and numel > 0
        and out.dtype == torch.bool
    ):
        return _isneginf_fast(A, out, numel)
    if out is None:
        return isneginf(A)
    isneginf_func(A, out0=out)
    return out
