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


def _allow_as_jit_global(handle):
    """Make a ``tle.raw.dialect`` handle visible to the Triton AST visitor.

    The installed compiler only lets a kernel read a module-level global if
    ``getattr(val, "__module__", "").startswith("triton.experimental.tle")``.
    The handle returned by ``@tle.raw.dialect`` is a ``RawJITFunction`` whose
    class reports ``__module__ == "runtime"`` (the extension module), so the
    allowlist check fails with ``NameError: Cannot access global variable``.
    Tagging the instance with the package name it actually comes from restores
    the intended lookup; nothing else about the handle changes.
    """
    handle.__module__ = "triton.experimental.tle.raw"
    return handle


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

    @_allow_as_jit_global
    @tle.raw.dialect(
        "xpu3",
        object=os.path.join(_PAY_OBJ, "gems_quantize_f32.o"),
        arch=3,
    )
    def gems_quantize_f32(
        x, y, mask, n, scale, inv, zero, enabled, qmin, qmax, pid, np, dtype
    ): ...

    @_allow_as_jit_global
    @tle.raw.dialect(
        "xpu3",
        object=os.path.join(_PAY_OBJ, "gems_quantize_f16.o"),
        arch=3,
    )
    def gems_quantize_f16(
        x, y, mask, n, scale, inv, zero, enabled, qmin, qmax, pid, np, dtype
    ): ...

    @_allow_as_jit_global
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


def _programs(n_elements):
    """Number of clusters to launch (measured on XPU3, fp32, ``do_bench``).

    The payload is one work item per cluster (64 cores) and its element rate
    saturates at 12 clusters, but the per-cluster share has to be at least a
    few 10^4 elements for the split to pay off (a cluster launch costs a few
    microseconds): n=10K -> 1 (14.5 us) beats 12 (15.6 us); n=262K -> 4
    (16.9 us) beats 12 (19.0 us); n=1M -> 8 (28.2 us) beats 12 (29.3 us);
    n>=2.5M -> 12 is best (40.1 us at 2.56M, 180 us at 16.7M).
    """
    if n_elements <= 65536:
        return 1
    if n_elements <= 131072:
        return 2
    if n_elements <= 262144:
        return 4
    if n_elements <= 1048576:
        return 8
    return 12


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

    programs = _programs(n_elements)
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
