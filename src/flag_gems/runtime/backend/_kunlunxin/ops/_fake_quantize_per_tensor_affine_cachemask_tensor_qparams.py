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

"""Kunlunxin (XPU) implementation of
``_fake_quantize_per_tensor_affine_cachemask_tensor_qparams``.

A Triton raw XPU3 payload uses SIMD quantization and packs canonical boolean
bytes directly, avoiding the backend's scalarized computed-byte stores.
ATen multiplies by the rounded fp32 reciprocal of scale and rounds to nearest
even before adding zero_point. FP fusion is disabled at both compilation stages.
Disabled quantization copies the original bytes, preserving NaNs and signed zero.
"""

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops._fake_quantize_per_tensor_affine_cachemask_tensor_qparams import (
    _fake_quantize_per_tensor_affine_cachemask_tensor_qparams as _generic_fq,
)
from flag_gems.ops.contiguous import contiguous
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# Hand-written tle.raw payloads (xpu3), precompiled once to device ELF objects
# under ../payload/obj and linked at the ELF stage via object=. This keeps the
# test path independent of the XTDK clang (which the CI runtime image does not
# ship). object= itself is only available on newer flagtree builds; on older
# toolchains the decorator raises and we fall back to the generic pure-Triton
# kernel. The three entry symbols are gems_quantize_{f32,f16,bf16} (built with
# -DFAKE_QUANT_DTYPE={0,1,2}).
_PAY_OBJ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "payload", "obj"
)


try:
    import triton.experimental.tle as tle

    @tle.raw.dialect(
        "xpu3",
        object=os.path.join(_PAY_OBJ, "gems_quantize_f32.o"),
        arch=3,
    )
    def gems_quantize_f32(
        x, y, mask, n, scale, inv, zero, enabled, qmin, qmax, pid, np, dtype
    ): ...

    @tle.raw.dialect(
        "xpu3",
        object=os.path.join(_PAY_OBJ, "gems_quantize_f16.o"),
        arch=3,
    )
    def gems_quantize_f16(
        x, y, mask, n, scale, inv, zero, enabled, qmin, qmax, pid, np, dtype
    ): ...

    @tle.raw.dialect(
        "xpu3",
        object=os.path.join(_PAY_OBJ, "gems_quantize_bf16.o"),
        arch=3,
    )
    def gems_quantize_bf16(
        x, y, mask, n, scale, inv, zero, enabled, qmin, qmax, pid, np, dtype
    ): ...

    @libentry()
    @triton.jit
    def _simd_kernel(
        X,
        S,
        Z,
        E,
        Y,
        M,
        N,
        QMIN: tl.constexpr,
        QMAX: tl.constexpr,
        NP: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        scale = tl.load(S).to(tl.float32)
        inv = tl.div_rn(1.0, scale)
        zero = tl.load(Z).to(tl.float32)
        enabled = (tl.load(E) != 0).to(tl.int32)
        args = (
            X,
            Y,
            M,
            tl.cast(N, tl.int32),
            scale,
            inv,
            zero,
            enabled,
            tl.full((), QMIN, tl.int32),
            tl.full((), QMAX, tl.int32),
            tl.program_id(0),
            tl.full((), NP, tl.int32),
            tl.full((), DTYPE, tl.int32),
        )
        if DTYPE == 0:
            tle.raw.call(gems_quantize_f32, args)
        elif DTYPE == 1:
            tle.raw.call(gems_quantize_f16, args)
        else:
            tle.raw.call(gems_quantize_bf16, args)

    _HAS_TLE_RAW = True
except Exception:  # pragma: no cover - object= channel not available
    _HAS_TLE_RAW = False


def _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
    self, scale, zero_point, fake_quant_enabled, quant_min, quant_max
):
    logger.debug(
        "GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_TENSOR_QPARAMS"
    )
    if not _HAS_TLE_RAW:
        return _generic_fq(
            self, scale, zero_point, fake_quant_enabled, quant_min, quant_max
        )
    if not self.is_contiguous():
        self = contiguous(self)
    out = torch.empty_like(self)
    n_elements = self.numel()
    mask = torch.empty_like(self, dtype=torch.bool)

    if n_elements == 0:
        return out, mask

    programs = 1 if n_elements <= 65536 else 12
    dtype = (
        0 if self.dtype == torch.float32 else (1 if self.dtype == torch.float16 else 2)
    )
    with torch_device_fn.device(self.device):
        _simd_kernel[(programs, 1, 1)](
            self,
            scale,
            zero_point,
            fake_quant_enabled,
            out,
            mask,
            n_elements,
            QMIN=quant_min,
            QMAX=quant_max,
            NP=programs,
            DTYPE=dtype,
            num_warps=32,
            enable_fp_fusion=False,
        )
    return out, mask
