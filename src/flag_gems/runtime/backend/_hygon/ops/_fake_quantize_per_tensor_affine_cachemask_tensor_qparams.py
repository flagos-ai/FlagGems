# Copyright (c) 2024 FlagGems. All rights reserved.
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

from flag_gems.ops._fake_quantize_per_tensor_affine_cachemask_tensor_qparams import (
    _fake_quantize_per_tensor_affine_cachemask_tensor_qparams as _generic,
)
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)
_KERNELS = {}


@libentry()
@triton.jit
def _quantize_flat(
    X,
    Scale,
    Zero,
    Enabled,
    Out,
    Mask,
    N: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if N % BLOCK == 0:
        raw = tl.load(X + offsets)
    else:
        raw = tl.load(X + offsets, offsets < N, 0)
    scale = tl.load(Scale).to(tl.float32)
    zero = tl.load(Zero).to(tl.float32)
    enabled = tl.load(Enabled) != 0
    inv = tl.fdiv(1.0, scale, ieee_rounding=True)
    rounded = tl_extra_shim.rint(raw.to(tl.float32) * inv) + zero
    inside = (rounded >= QMIN) & (rounded <= QMAX)
    clipped = tl.minimum(tl.maximum(rounded, QMIN), QMAX)
    clipped = tl.where(rounded != rounded, QMIN, clipped)
    quantized = ((clipped - zero) * scale).to(raw.dtype)
    result = tl.where(enabled, quantized, raw)
    valid = inside | ~enabled
    if N % BLOCK == 0:
        tl.store(Out + offsets, result)
        tl.store(Mask + offsets, valid)
    else:
        tl.store(Out + offsets, result, offsets < N)
        tl.store(Mask + offsets, valid, offsets < N)


def _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
    self, scale, zero_point, fake_quant_enabled, quant_min, quant_max
):
    logger.debug("GEMS_HYGON FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_TENSOR_QPARAMS")
    if not self.is_contiguous() or any(
        t.device != self.device or t.numel() != 1
        for t in (scale, zero_point, fake_quant_enabled)
    ):
        return _generic(
            self, scale, zero_point, fake_quant_enabled, quant_min, quant_max
        )
    out = torch.empty_like(self)
    mask = torch.empty_like(self, dtype=torch.bool)
    n = self.numel()
    if n:
        block = 2048 if self.element_size() <= 2 else 1024
        args = (
            self,
            scale,
            zero_point,
            fake_quant_enabled,
            out,
            mask,
            n,
            quant_min,
            quant_max,
            block,
        )
        key = (
            self.get_device(),
            tuple(t.dtype for t in args[:6]),
            n,
            quant_min,
            quant_max,
            block,
            tuple(t.data_ptr() % 16 == 0 for t in args[:4]),
        )
        grid = (triton.cdiv(n, block), 1, 1)
        kernel = _KERNELS.get(key)
        if kernel is None:
            compiled = _quantize_flat[grid](*args, num_warps=4, enable_fp_fusion=False)
            kernel = compiled[0] if isinstance(compiled, tuple) else compiled
            if kernel is not None:
                if len(_KERNELS) >= 128:
                    _KERNELS.pop(next(iter(_KERNELS)))
                _KERNELS[key] = kernel
        else:
            kernel[grid](*args, stream=torch.cuda.current_stream().cuda_stream)
    return out, mask
