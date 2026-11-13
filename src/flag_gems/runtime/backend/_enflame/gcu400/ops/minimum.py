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

from flag_gems.runtime import device
from flag_gems.utils.libentry import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name

# Flat fast path for large same-shape contiguous float tensors: skips the
# block-pointer + grid-stride codegen path, which measured ~20-25% slower at
# 16M-element shapes and ~5-10% slower at 1G elements on GCU400/L600.
# Configs bench-validated (do_bench kernel median): fp16/bf16 (65536, w2) at
# all sizes; fp32 (32768, w2) only in [2^23, 2^27) - beyond that the codegen
# path is faster. Small shapes stay on the codegen path (already >= 1.3x
# native), as do broadcast/strided/mixed-dtype/non-float cases.
_FLAT_MIN_NUMEL = 1 << 23
_FLAT_FP32_MAX_NUMEL = 1 << 27
_INT32_MAX = 2**31 - 1
_FLAT_GRID_CAP = 48
_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _minimum_flat_kernel(
    X, Y, OUT, N, BLOCK: tl.constexpr, CAST32: tl.constexpr, EVEN: tl.constexpr
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_tiles = tl.cdiv(N, BLOCK)
    for tile in range(pid, num_tiles, num_ctas):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(X + offs)
            y = tl.load(Y + offs)
            if CAST32:
                x = x.to(tl.float32)
                y = y.to(tl.float32)
            tl.store(OUT + offs, tl.minimum(x, y).to(OUT.type.element_ty))
        else:
            mask = offs < N
            x = tl.load(X + offs, mask=mask)
            y = tl.load(Y + offs, mask=mask)
            if CAST32:
                x = x.to(tl.float32)
                y = y.to(tl.float32)
            tl.store(OUT + offs, tl.minimum(x, y).to(OUT.type.element_ty), mask=mask)


def _minimum_flat(X, Y):
    numel = X.numel()
    out = torch.empty_like(X)
    if X.dtype == torch.float32:
        block, num_warps = 32768, 2
    else:
        block, num_warps = 65536, 2
    grid = (min(triton.cdiv(numel, block), _FLAT_GRID_CAP),)
    _minimum_flat_kernel[grid](
        X,
        Y,
        out,
        numel,
        BLOCK=block,
        CAST32=(X.dtype != torch.float32),
        EVEN=(numel % block == 0),
        num_warps=num_warps,
    )
    return out


def _use_flat_path(X, Y):
    dtype = X.dtype
    if dtype not in _FLAT_DTYPES or dtype != Y.dtype:
        return False
    numel = X.numel()
    if numel < _FLAT_MIN_NUMEL or numel > _INT32_MAX:
        return False
    if dtype == torch.float32 and numel >= _FLAT_FP32_MAX_NUMEL:
        return False
    return X.shape == Y.shape and X.is_contiguous() and Y.is_contiguous()


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 0, "DEFAULT")])
@triton.jit
def minimum_kernel(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)
    return tl.minimum(X, Y)


def minimum(X, Y):
    logger.debug("GEMS_ENFLAME MINIMUM")
    assert X.device.type == device and Y.device.type == device
    if _use_flat_path(X, Y):
        return _minimum_flat(X, Y)
    return minimum_kernel(X, Y)
