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

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl

from flag_gems.ops.contiguous import contiguous
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


_SIMD_SOURCE = r"""
#include "xpu/kernel/xtdk.h"
#include "xpu/kernel/xtdk_simd.h"

// In-place staging keeps the per-core LM footprint below 8 KB.
template<int D> __device__ void quantize(
    __global_ptr__ const char* x, __global_ptr__ char* y,
    __global_ptr__ unsigned char* mask, int n, float scale, float inv,
    float zero, int enabled, int qmin, int qmax, int pid, int np) {
    const int tile = D == 0 ? 1280 : 2048;
    __simd__ char data[tile * (D == 0 ? 4 : 2)];
    __simd__ unsigned char masks[tile];
    const int bytes = D == 0 ? 4 : 2;
    const float32x16_t lo = (float32x16_t)(float(qmin));
    const float32x16_t hi = (float32x16_t)(float(qmax));
    const float32x16_t one = (float32x16_t)(1.0f);
    const float32x16_t nil = (float32x16_t)(0.0f);
    for (int base = (pid * core_num() + core_id()) * tile;
         base < n; base += np * core_num() * tile) {
        int count = n - base < tile ? n - base : tile;
        GM2LM(x + base * bytes, data, count * bytes);
        char* work = data;
        for (int j = 0; j < count; j += 64) {
            float32x16_t v[4];
            if (D == 0) {
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    v[k] = vload_lm_float32x16(work + (j + k * 16) * 4);
            } else if (D == 1) {
                float16x32_t a = vload_lm_float16x32(data + j * 2);
                float16x32_t b = vload_lm_float16x32(data + (j + 32) * 2);
                v[0] = vfp162float_l(a); v[1] = vfp162float_h(a);
                v[2] = vfp162float_l(b); v[3] = vfp162float_h(b);
            } else {
                // Packed bf16: even and odd elements stay in separate vectors.
                // XPU shifts take (shift_amount, value), unlike C operators.
                uint32x16_t a = vload_lm_uint32x16(data + j * 2);
                uint32x16_t b = vload_lm_uint32x16(data + (j + 32) * 2);
                v[0] = (float32x16_t)vvsll_uint32x16((uint32x16_t)16u, a);
                v[1] = (float32x16_t)vvand_uint32x16(a, (uint32x16_t)0xffff0000u);
                v[2] = (float32x16_t)vvsll_uint32x16((uint32x16_t)16u, b);
                v[3] = (float32x16_t)vvand_uint32x16(b, (uint32x16_t)0xffff0000u);
            }
            float32x16_t m[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                m[k] = one;
                if (enabled) {
                    float32x16_t q = vvadd_float32x16(vfix2float(vfloat2fix(
                        vvmul_float32x16(v[k], (float32x16_t)inv))), (float32x16_t)zero);
                    float32x16_t clipped = vvmin_float32x16(vvmax_float32x16(q, lo), hi);
                    int finite_or_inf = vveq_float32x16(v[k], v[k]);
                    m[k] = vvsub_float32x16_mz(one, nil,
                        vveq_float32x16(q, clipped) & finite_or_inf);
                    q = vvsub_float32x16_mh(clipped, nil, lo, finite_or_inf);
                    v[k] = vvmul_float32x16(vvsub_float32x16(q, (float32x16_t)zero),
                                           (float32x16_t)scale);
                }
            }
            if (D == 2) {
                float32x16_t a = vvmerge_l_float32x16(m[0], m[1]);
                float32x16_t b = vvmerge_h_float32x16(m[0], m[1]);
                float32x16_t c = vvmerge_l_float32x16(m[2], m[3]);
                float32x16_t d = vvmerge_h_float32x16(m[2], m[3]);
                m[0] = a; m[1] = b; m[2] = c; m[3] = d;
            }
            uint8x64_t packed = (uint8x64_t)(0);
            packed = vfloat2ufix8_ll(packed, m[0]);
            packed = vfloat2ufix8_lh(packed, m[1]);
            packed = vfloat2ufix8_hl(packed, m[2]);
            packed = vfloat2ufix8_hh(packed, m[3]);
            vstore_lm_int8x64_mz(masks + j, (int8x64_t)packed);
            if (enabled) {
                if (D == 0) {
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        vstore_lm_float32x16(work + (j + k * 16) * 4, v[k]);
                } else if (D == 1) {
                    vstore_lm_float16x32(data + j * 2, vfloat2fp16_lh(v[0], v[1]));
                    vstore_lm_float16x32(data + (j + 32) * 2, vfloat2fp16_lh(v[2], v[3]));
                } else {
                    uint32x16_t h[4];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        uint32x16_t bits = (uint32x16_t)v[k];
                        uint32x16_t odd = vvand_uint32x16(
                            vvsrl_uint32x16((uint32x16_t)16u, bits), (uint32x16_t)1u);
                        h[k] = vvadd_uint32x16(bits,
                            vvadd_uint32x16((uint32x16_t)32767u, odd));
                    }
                    uint32x16_t a = vvor_uint32x16(
                        vvsrl_uint32x16((uint32x16_t)16u, h[0]),
                        vvand_uint32x16(h[1], (uint32x16_t)0xffff0000u));
                    uint32x16_t b = vvor_uint32x16(
                        vvsrl_uint32x16((uint32x16_t)16u, h[2]),
                        vvand_uint32x16(h[3], (uint32x16_t)0xffff0000u));
                    vstore_lm_uint32x16(data + j * 2, a);
                    vstore_lm_uint32x16(data + (j + 32) * 2, b);
                }
            }
        }
        mfence_lm();
        LM2GM_ASYNC(data, y + base * bytes, count * bytes);
        LM2GM_ASYNC(masks, mask + base, count);
        mfence_lm_gm();
    }
}
extern "C" __device__ void gems_quantize(
    __global_ptr__ const char* x, __global_ptr__ char* y,
    __global_ptr__ unsigned char* mask, int n, float scale, float inv,
    float zero, int enabled, int qmin, int qmax, int pid, int np, int dtype) {
    quantize<FAKE_QUANT_DTYPE>(x,y,mask,n,scale,inv,zero,enabled,qmin,qmax,pid,np);
}
"""


@tle.raw.dialect(
    "xpu3",
    source=_SIMD_SOURCE.replace("gems_quantize", "gems_quantize_f32"),
    arch=3,
    flags=("-ffp-contract=off", "-DFAKE_QUANT_DTYPE=0"),
)
def gems_quantize_f32(
    x, y, mask, n, scale, inv, zero, enabled, qmin, qmax, pid, np, dtype
): ...


@tle.raw.dialect(
    "xpu3",
    source=_SIMD_SOURCE.replace("gems_quantize", "gems_quantize_f16"),
    arch=3,
    flags=("-ffp-contract=off", "-DFAKE_QUANT_DTYPE=1"),
)
def gems_quantize_f16(
    x, y, mask, n, scale, inv, zero, enabled, qmin, qmax, pid, np, dtype
): ...


@tle.raw.dialect(
    "xpu3",
    source=_SIMD_SOURCE.replace("gems_quantize", "gems_quantize_bf16"),
    arch=3,
    flags=("-ffp-contract=off", "-DFAKE_QUANT_DTYPE=2"),
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


def _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
    self, scale, zero_point, fake_quant_enabled, quant_min, quant_max
):
    logger.debug(
        "GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_TENSOR_QPARAMS"
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
