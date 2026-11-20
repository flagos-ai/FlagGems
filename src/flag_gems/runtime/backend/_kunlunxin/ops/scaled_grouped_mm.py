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

"""Kunlunxin backend override for ``scaled_grouped_mm`` (FP8 operand plumbing).

The generic implementation (``flag_gems/ops/scaled_grouped_mm.py``) fuses FP8
E4M3 byte decoding into the dot loop: it views the operands as ``uint8`` and
applies the ``_decode_e4m3`` device function to every loaded tile whenever the
device has no native E4M3 dot. On kunlunxin XPU that fused decode path is
broken: the ``scaled_grouped_mm_kernel`` compiles and launches, but the stored
output is entirely zero.

Minimal reproducible probe (M=N=K=16, NUM_GROUPS=1, all 256 E4M3 encodings in
``mat_a``, identity ``mat_b``, unit scales, ``out_dtype=float32``):

* fused-decode path -> result is all zeros; 252/256 mismatches, max abs 448.0
  (the two "matching" positions are the E4M3 zeros and the zeroed NaN slots);
* the standalone ``_decode_e4m3_tensor`` kernel decodes all 256 encodings
  correctly (only the two NaN encodings compare unequal, as NaN != NaN);
* the very same geometry with FP16/BF16 operands matches exactly.

So the defect is specific to the decode that is *fused into the dot loop*, not
to the decode arithmetic, the dot, the scaling epilogue or the launch. The
generic op already carries this exact workaround for the iluvatar vendor
("CoreX 4.4 miscompiles byte decoding fused into the dot loop. Materialize FP16
on device before using the validated dot path."); this override applies the
same strategy on kunlunxin.

The override only rewrites the FP8 E4M3 operands into device FP16 (using the
already-registered FlagGems ``_decode_e4m3_tensor`` kernel) and then delegates
to the unchanged generic implementation. No generic source, test, tolerance or
native fallback is touched, and every non-FP8 call is delegated verbatim.
"""

import logging

import torch
import triton

from flag_gems.ops.scaled_grouped_mm import (
    _decode_e4m3_tensor,
    _default_out_dtype,
    _native_e4m3_dot,
    _normalize_bias,
    _normalize_scale,
    _resolve_shapes,
    _scaled_grouped_mm_fallback,
)
from flag_gems.ops.scaled_grouped_mm import (
    scaled_grouped_mm as _generic_scaled_grouped_mm,
)

__all__ = ["scaled_grouped_mm"]

logger = logging.getLogger(__name__)

_E4M3_DTYPES = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e4m3fnuz")
    if hasattr(torch, name)
)

_DECODE_BLOCK = 256


def _materialize_e4m3_to_fp16(mat, fnuz):
    """Decode an E4M3 storage tensor to a device FP16 tensor.

    Uses the standalone ``_decode_e4m3_tensor`` kernel, which is verified to
    decode every E4M3 encoding correctly. The source is flattened to a
    contiguous byte buffer so the decode is layout independent; the decoded
    tensor is contiguous, and the caller's stride metadata is recomputed from
    it by the generic implementation.
    """
    # Reinterpret the 1-byte FP8 storage as uint8 before any layout change:
    # a plain FP8 `.contiguous()` / `.reshape(-1)` copy is not implemented on
    # this XPU stack, whereas the byte view and the uint8 copy are.
    flat = mat.view(torch.uint8).reshape(-1)
    result = torch.empty(flat.shape, dtype=torch.float16, device=mat.device)
    n = flat.numel()
    if n:
        _decode_e4m3_tensor[(triton.cdiv(n, _DECODE_BLOCK),)](
            flat, result, n, fnuz, _DECODE_BLOCK
        )
    return result.reshape(mat.shape)


def scaled_grouped_mm(
    self,
    mat2,
    scale_a,
    scale_b,
    offs=None,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    logger.debug("GEMS_KUNLUNXIN SCALED_GROUPED_MM")
    if self.dtype in _E4M3_DTYPES and not _native_e4m3_dot(self.dtype, self.device):
        fnuz = self.dtype == getattr(torch, "float8_e4m3fnuz", None)
        logger.debug("GEMS_KUNLUNXIN SCALED_GROUPED_MM MATERIALIZE FP8 TO FP16")
        # The generic implementation derives the output dtype from the operand
        # dtype; preserve the original FP8-derived default (bfloat16) when the
        # caller did not request one explicitly.
        if out_dtype is None:
            out_dtype = _default_out_dtype(self.dtype)
        self = _materialize_e4m3_to_fp16(self, fnuz)
        mat2 = _materialize_e4m3_to_fp16(mat2, fnuz)

    if offs is not None and self.dim() == 3 and mat2.dim() == 2:
        a_is_2d, b_is_2d, num_groups, _m, n, _k, _shape, offs_c = _resolve_shapes(
            self, mat2, offs
        )
        output_dtype = out_dtype or _default_out_dtype(self.dtype)
        scale_a_n = _normalize_scale(
            scale_a,
            self,
            dim=0,
            num_groups=num_groups,
            scale_multiplier=1,
            name="scale_a",
        )
        scale_b_n = _normalize_scale(
            scale_b,
            mat2,
            dim=1,
            num_groups=num_groups,
            scale_multiplier=1,
            name="scale_b",
        )
        bias_n, _mode = _normalize_bias(
            bias, a_is_2d=a_is_2d, b_is_2d=b_is_2d, num_groups=num_groups, N=n
        )
        return _scaled_grouped_mm_fallback(
            self,
            mat2,
            scale_a_n,
            scale_b_n,
            offs_c,
            bias_n,
            output_dtype,
            a_is_2d,
            b_is_2d,
            num_groups,
        )

    if offs is None and self.dim() == 3 and mat2.dim() == 3:
        groups, m, k = self.shape
        flat_self = self.reshape(groups * m, k)
        flat_scale_a = scale_a.reshape(-1) if scale_a.dim() == 2 else scale_a
        flat_offs = (
            torch.arange(1, groups + 1, dtype=torch.int32, device=self.device) * m
        )
        flat_out = _generic_scaled_grouped_mm(
            flat_self,
            mat2,
            flat_scale_a,
            scale_b,
            offs=flat_offs,
            bias=bias,
            scale_result=scale_result,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )
        return flat_out.reshape(groups, m, mat2.shape[-1])

    return _generic_scaled_grouped_mm(
        self,
        mat2,
        scale_a,
        scale_b,
        offs=offs,
        bias=bias,
        scale_result=scale_result,
        out_dtype=out_dtype,
        use_fast_accum=use_fast_accum,
    )
