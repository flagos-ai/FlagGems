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

# flake8: noqa: E501,F841

"""FP8 dense MLA decode for per-token-scaled CKV caches on Hopper GPUs.

The partial kernel uses two four-warp compute groups, explicit TMA completion
barriers, and explicit WGMMA issue/wait control. It implements FP8 content QK,
BF16 RoPE QK, online softmax, FP8 probability quantization, and FP8 PV
accumulation with adaptive Split-K scheduling.

The public API provides metadata planning, prepared descriptors and workspace,
CUDA Graph-compatible replay, and direct execution.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

try:  # pragma: no cover - CPU fallback
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

try:
    import triton.experimental.tle.language as tle  # noqa: F401

    HAS_TLE = True
except Exception:
    tle = None
    HAS_TLE = False

D_QK = 576  # Q/K head dim (content 512 + rope 64)
D_CKV = 512  # content / V head dim
D_ROPE = 64  # rope tail dim
PAGE_SIZE = 64  # paged KV cache page size (= BK)
FP8_MAX = 448.0  # E4M3 dynamic range upper bound
FP8_DTYPE = torch.float8_e4m3fn
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
P_AMAX_FLOOR = 1e-26

TLE_FP8_BK = 64  # KV tokens per iteration (= PAGE_SIZE)
TLE_FP8_BH = 64  # heads per iteration
TLE_FP8_DPH = 256  # output 512 dim split into left/right halves of 256
K_CONTENT_TILE_HOST = 128
K_CONTENT_TILE = (
    tl.constexpr(K_CONTENT_TILE_HOST) if HAS_TRITON else K_CONTENT_TILE_HOST
)

DEFAULT_PAGES_PER_SPLIT = 2
COMBINE_BLOCK_SPLITS = 8
COMBINE_BLOCK_D = 128
CUDA_COARSE_COMBINE_BLOCK_SPLITS = 32
CUDA_COARSE_COMBINE_BLOCK_ROWS = 8
CUDA_COARSE_COMBINE_MIN_BATCH = 4
LSE_FINALIZE_BLOCK = 256

# Adaptive H800 execution-grain policy.
ADAPTIVE_MODEL_MIN_PAGES = 69
ADAPTIVE_MIN_FIXED_PAGES = 4
ADAPTIVE_MAX_FIXED_PAGES = 32
ADAPTIVE_TAIL_WAVE_MAX_FIXED_PAGES = 34
ADAPTIVE_CTA_PENALTY = 0.5
MAX_SEQUENCE_LENGTH = 33280
CUDA_REF_FIXED_OVERHEAD_PAGES = 5

NUM_SLOTS = (
    tl.constexpr(2) if HAS_TRITON else 2 if HAS_TRITON else 2 if HAS_TRITON else 2
)
_PACK32_F32_BASE_CONSTRAINTS = ",".join(["=r"] * 32 + ["f"] * 32 + ["r"] * 32)


if HAS_TLE:

    @triton.jit
    def _publish_p_fp8_sw64_cuda_stmatrix(s_p, p):
        """CUDA save_rPb_to_sP: swap uint16 packs 1/2, then two x4 STSM."""
        raw = p.to(tl.uint8, bitcast=True).to(tl.uint32)
        base = tle.gpu.local_ptr(s_p, (0, 0))
        base_u32 = tl.inline_asm_elementwise(
            asm="mov.u32 $0, $1;",
            constraints="=r,r",
            args=[base],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .b32 tid, warp_off, row_off, common, tmp, phys0, phys1, addr0, addr1;\n"
                ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3, lane, src0, src1, selector, lo, hi;\n"
                ".reg .pred take_hi;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 warp_off, tid, 96;\n"
                "shl.b32 warp_off, warp_off, 5;\n"
                "and.b32 row_off, tid, 15;\n"
                "shl.b32 row_off, row_off, 6;\n"
                "or.b32 common, warp_off, row_off;\n"
                "and.b32 tmp, tid, 16;\n"
                "or.b32 common, common, tmp;\n"
                "and.b32 lane, tid, 31;\n"
                "and.b32 src0, lane, 28;\n"
                "and.b32 tmp, lane, 1;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "add.u32 src0, src0, tmp;\n"
                "add.u32 src1, src0, 1;\n"
                "and.b32 selector, lane, 2;\n"
                "setp.ne.u32 take_hi, selector, 0;\n"
                "selp.u32 selector, 0x7632, 0x5410, take_hi;\n"
                "and.b32 a0, $32, 255;\n"
                "and.b32 tmp, $33, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 a0, a0, tmp;\n"
                "and.b32 tmp, $36, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 a0, a0, tmp;\n"
                "and.b32 tmp, $37, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 a0, a0, tmp;\n"
                "shfl.sync.idx.b32 lo, a0, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, a0, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 a0, lo, hi, selector;\n"
                "and.b32 a1, $34, 255;\n"
                "and.b32 tmp, $35, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 a1, a1, tmp;\n"
                "and.b32 tmp, $38, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 a1, a1, tmp;\n"
                "and.b32 tmp, $39, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 a1, a1, tmp;\n"
                "shfl.sync.idx.b32 lo, a1, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, a1, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 a1, lo, hi, selector;\n"
                "and.b32 a2, $40, 255;\n"
                "and.b32 tmp, $41, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 a2, a2, tmp;\n"
                "and.b32 tmp, $44, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 a2, a2, tmp;\n"
                "and.b32 tmp, $45, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 a2, a2, tmp;\n"
                "shfl.sync.idx.b32 lo, a2, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, a2, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 a2, lo, hi, selector;\n"
                "and.b32 a3, $42, 255;\n"
                "and.b32 tmp, $43, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 a3, a3, tmp;\n"
                "and.b32 tmp, $46, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 a3, a3, tmp;\n"
                "and.b32 tmp, $47, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 a3, a3, tmp;\n"
                "shfl.sync.idx.b32 lo, a3, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, a3, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 a3, lo, hi, selector;\n"
                "and.b32 b0, $48, 255;\n"
                "and.b32 tmp, $49, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 b0, b0, tmp;\n"
                "and.b32 tmp, $52, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 b0, b0, tmp;\n"
                "and.b32 tmp, $53, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 b0, b0, tmp;\n"
                "shfl.sync.idx.b32 lo, b0, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, b0, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 b0, lo, hi, selector;\n"
                "and.b32 b1, $50, 255;\n"
                "and.b32 tmp, $51, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 b1, b1, tmp;\n"
                "and.b32 tmp, $54, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 b1, b1, tmp;\n"
                "and.b32 tmp, $55, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 b1, b1, tmp;\n"
                "shfl.sync.idx.b32 lo, b1, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, b1, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 b1, lo, hi, selector;\n"
                "and.b32 b2, $56, 255;\n"
                "and.b32 tmp, $57, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 b2, b2, tmp;\n"
                "and.b32 tmp, $60, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 b2, b2, tmp;\n"
                "and.b32 tmp, $61, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 b2, b2, tmp;\n"
                "shfl.sync.idx.b32 lo, b2, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, b2, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 b2, lo, hi, selector;\n"
                "and.b32 b3, $58, 255;\n"
                "and.b32 tmp, $59, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 b3, b3, tmp;\n"
                "and.b32 tmp, $62, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 b3, b3, tmp;\n"
                "and.b32 tmp, $63, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 b3, b3, tmp;\n"
                "shfl.sync.idx.b32 lo, b3, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, b3, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 b3, lo, hi, selector;\n"
                "shr.u32 tmp, common, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 phys0, common, tmp;\n"
                "add.u32 common, common, 32;\n"
                "shr.u32 tmp, common, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 phys1, common, tmp;\n"
                "add.u32 addr0, $64, phys0;\n"
                "add.u32 addr1, $64, phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr0], {a0, a1, a2, a3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr1], {b0, b1, b2, b3};\n"
                "fence.proxy.async.shared::cta;\n"
                "mov.u32 $0, $32;\n"
                "}"
            ),
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
            args=[raw, base_u32],
            dtype=tl.uint32,
            is_pure=False,
            pack=32,
        )

    @triton.jit
    def _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p, p):
        """CUDA-native P publication; V repack carries the matching K permutation."""
        base = tle.gpu.local_ptr(s_p, (0, 0))
        base_u32 = tl.inline_asm_elementwise(
            asm="mov.u32 $0, $1;",
            constraints="=r,r",
            args=[base],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .b16 h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15;\n"
                ".reg .b32 tid, warp_off, row_off, common, tmp, phys0, phys1, addr0, addr1;\n"
                ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 warp_off, tid, 96;\n"
                "shl.b32 warp_off, warp_off, 5;\n"
                "and.b32 row_off, tid, 15;\n"
                "shl.b32 row_off, row_off, 6;\n"
                "or.b32 common, warp_off, row_off;\n"
                "and.b32 tmp, tid, 16;\n"
                "or.b32 common, common, tmp;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h0, $33, $32;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h1, $35, $34;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h2, $37, $36;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h3, $39, $38;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h4, $41, $40;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h5, $43, $42;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h6, $45, $44;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h7, $47, $46;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h8, $49, $48;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h9, $51, $50;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h10, $53, $52;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h11, $55, $54;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h12, $57, $56;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h13, $59, $58;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h14, $61, $60;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h15, $63, $62;\n"
                "mov.b32 a0, {h0, h2};\n"
                "mov.b32 a1, {h1, h3};\n"
                "mov.b32 a2, {h4, h6};\n"
                "mov.b32 a3, {h5, h7};\n"
                "mov.b32 b0, {h8, h10};\n"
                "mov.b32 b1, {h9, h11};\n"
                "mov.b32 b2, {h12, h14};\n"
                "mov.b32 b3, {h13, h15};\n"
                "shr.u32 tmp, common, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 phys0, common, tmp;\n"
                "add.u32 common, common, 32;\n"
                "shr.u32 tmp, common, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 phys1, common, tmp;\n"
                "add.u32 addr0, $64, phys0;\n"
                "add.u32 addr1, $64, phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr0], {a0, a1, a2, a3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr1], {b0, b1, b2, b3};\n"
                "fence.proxy.async.shared::cta;\n"
                "mov.u32 $0, $64;\n"
                "}"
            ),
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
            args=[p, base_u32],
            dtype=tl.uint32,
            is_pure=False,
            pack=32,
        )

    @triton.jit
    def _load_qrope_rs_fragment(s_qr, k_tile: tl.constexpr):
        # Map the physical register/lane/warp ownership bits of a 1024-wide
        # arange into the WGMMA K16 register-A layout using views only.
        physical = tl.arange(0, 1024).to(tl.bfloat16)
        ownership_bits = tl.reshape(physical, (2, 2, 2, 2, 2, 2, 2, 2, 2, 2))
        logical_bits = tl.permute(ownership_bits, (0, 1, 8, 2, 3, 4, 7, 5, 6, 9))
        carrier = tl.reshape(logical_bits, (64, 16))
        tile = carrier.to(tl.int32) * 0 + k_tile

        # Keep the shared operand visible to allocation/liveness while using
        # inline PTX only for the permitted pointer move and LDSM path.
        base = tle.gpu.local_ptr(s_qr, (0, 0))
        base_u32 = tl.inline_asm_elementwise(
            asm="mov.u32 $0, $1;",
            constraints="=r,r",
            args=[base],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .b32 raw<4>;\n"
                ".reg .b32 lane, tid, x, y, tmp, off, smem;\n"
                ".reg .b32 lane16, group, src0, src1;\n"
                ".reg .b32 v00, v01, v20, v21, v02, v03, v22, v23;\n"
                ".reg .pred upper;\n"
                "mov.u32 lane, %laneid;\n"
                "mov.u32 tid, %tid.x;\n"
                "mov.u32 smem, $16;\n"
                "shl.b32 off, tid, 6;\n"
                "and.b32 off, off, 8064;\n"
                "shl.b32 x, tid, 2;\n"
                "and.b32 x, x, 56;\n"
                "shl.b32 y, tid, 3;\n"
                "and.b32 y, y, 8;\n"
                "shl.b32 tmp, $8, 4;\n"
                "or.b32 y, y, tmp;\n"
                "xor.b32 x, x, y;\n"
                "shl.b32 x, x, 1;\n"
                "add.u32 off, off, x;\n"
                "add.u32 off, smem, off;\n"
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                "{raw0, raw1, raw2, raw3}, [off];\n"
                "and.b32 lane16, lane, 15;\n"
                "and.b32 src0, lane16, 3;\n"
                "and.b32 group, lane16, 12;\n"
                "shl.b32 group, group, 1;\n"
                "add.u32 src0, src0, group;\n"
                "add.u32 src1, src0, 4;\n"
                "setp.ge.u32 upper, lane, 16;\n"
                "shfl.sync.idx.b32 v00, raw0, src0, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v01, raw1, src0, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v20, raw2, src0, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v21, raw3, src0, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v02, raw0, src1, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v03, raw1, src1, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v22, raw2, src1, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v23, raw3, src1, 31, 0xffffffff;\n"
                "selp.b32 $0, v01, v00, upper;\n"
                "selp.b32 $1, v21, v20, upper;\n"
                "selp.b32 $2, v03, v02, upper;\n"
                "selp.b32 $3, v23, v22, upper;\n"
                "}"
            ),
            constraints=(
                "=r,=r,=r,=r," "r,r,r,r," "r,r,r,r,r,r,r,r," "r,r,r,r,r,r,r,r"
            ),
            args=[carrier, tile, base_u32],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=8,
        )

    @triton.jit
    def _zero_invalid_fp8_rows_sw128(s_src, valid_tokens):
        """Zero invalid rows of one SW128 64x128 FP8 tile with 16B stores."""
        carrier = tl.arange(0, 128).to(tl.uint32)
        src_base = tle.gpu.local_ptr(s_src, (0, 0))
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .pred invalid;\n"
                ".reg .b32 tid, row, col, logical, swz, addr, z;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 tid, tid, 127;\n"
                "shr.u32 row, tid, 1;\n"
                "and.b32 col, tid, 1;\n"
                "shl.b32 col, col, 6;\n"
                "setp.ge.u32 invalid, row, $3;\n"
                "mov.u32 z, 0;\n"
                "shl.b32 logical, row, 7;\n"
                "add.u32 logical, logical, col;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 addr, logical, swz;\n"
                "add.u32 addr, $2, addr;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 logical, logical, 16;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 addr, logical, swz;\n"
                "add.u32 addr, $2, addr;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 logical, logical, 16;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 addr, logical, swz;\n"
                "add.u32 addr, $2, addr;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 logical, logical, 16;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 addr, logical, swz;\n"
                "add.u32 addr, $2, addr;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "mov.u32 $0, $1;\n"
                "}"
            ),
            constraints="=r,r,r,r",
            args=[carrier, src_base, valid_tokens],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _zero_invalid_fp8_rows_sw128_x4(
        s_src0,
        s_src1,
        s_src2,
        s_src3,
        valid_tokens,
    ):
        """Zero the same invalid rows in four SW128 64x128 FP8 tiles.

        The four content tiles share row validity and SW128 addressing.  Keep
        the four 16B stores per tile, but compute the predicate and swizzled
        byte offset only once.
        """
        carrier = tl.arange(0, 128).to(tl.uint32)
        src0_base = tle.gpu.local_ptr(s_src0, (0, 0))
        src1_base = tle.gpu.local_ptr(s_src1, (0, 0))
        src2_base = tle.gpu.local_ptr(s_src2, (0, 0))
        src3_base = tle.gpu.local_ptr(s_src3, (0, 0))
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .pred invalid;\n"
                ".reg .b32 tid, lane, warp, row, col, logical, swz, off, addr, z;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 tid, tid, 127;\n"
                "and.b32 lane, tid, 31;\n"
                "shr.u32 warp, tid, 5;\n"
                "mov.u32 row, lane;\n"
                "shl.b32 col, warp, 4;\n"
                "setp.ge.u32 invalid, row, $6;\n"
                "mov.u32 z, 0;\n"
                "shl.b32 logical, row, 7;\n"
                "add.u32 logical, logical, col;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 off, logical, swz;\n"
                "add.u32 addr, $2, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $3, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $4, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $5, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 logical, logical, 64;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 off, logical, swz;\n"
                "add.u32 addr, $2, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $3, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $4, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $5, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 row, row, 32;\n"
                "setp.ge.u32 invalid, row, $6;\n"
                "shl.b32 logical, row, 7;\n"
                "add.u32 logical, logical, col;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 off, logical, swz;\n"
                "add.u32 addr, $2, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $3, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $4, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $5, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 logical, logical, 64;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 off, logical, swz;\n"
                "add.u32 addr, $2, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $3, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $4, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $5, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "mov.u32 $0, $1;\n"
                "}"
            ),
            constraints="=r,r,r,r,r,r,r",
            args=[
                carrier,
                src0_base,
                src1_base,
                src2_base,
                src3_base,
                valid_tokens,
            ],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _cuda_vtranspose_fp8_64x128_plain(
        s_src,
        s_dst,
        dst_row: tl.constexpr,
    ):
        """CUDA-authority SW128 -> SW64 FP8 transpose for one 64x128 tile."""
        carrier = tl.arange(0, 128).to(tl.uint32)
        src_base = tle.gpu.local_ptr(s_src, (0, 0))
        dst_base = tle.gpu.local_ptr(s_dst, (dst_row, 0))
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .b32 tid, lane, warp, src_row, tmp, tmp2;\n"
                ".reg .b32 src_log, src_phys, src_addr0, src_addr1;\n"
                ".reg .b32 dst_row_r, dst_col, dst_log0, dst_log1;\n"
                ".reg .b32 dst_phys0, dst_phys1, dst_addr0, dst_addr1;\n"
                ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
                ".reg .b32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 tid, tid, 127;\n"
                "and.b32 lane, tid, 31;\n"
                "shr.u32 warp, tid, 5;\n"
                # CUDA's LDSM/STSM register order presents source-row bits as
                # [b1,b3,b2,b0] to TLE's logical SW64 view.  Apply the inverse
                # [b3,b1,b2,b0] mapping at the load boundary so PV observes
                # the same logical transpose as the tensor path.
                "and.b32 src_row, lane, 17;\n"
                "and.b32 tmp, lane, 8;\n"
                "shr.u32 tmp, tmp, 2;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 2;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 4;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "shl.b32 src_log, src_row, 7;\n"
                "shl.b32 tmp, warp, 4;\n"
                "add.u32 src_log, src_log, tmp;\n"
                "shr.u32 tmp, src_log, 7;\n"
                "and.b32 tmp, tmp, 7;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 src_phys, src_log, tmp;\n"
                "add.u32 src_addr0, $2, src_phys;\n"
                "add.u32 src_addr1, src_addr0, 4096;\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{a0, a1, a2, a3}, [src_addr0];\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{b0, b1, b2, b3}, [src_addr1];\n"
                "prmt.b32 c0, a0, a1, 0x6420;\n"
                "prmt.b32 c1, a0, a1, 0x7531;\n"
                "prmt.b32 c2, a2, a3, 0x6420;\n"
                "prmt.b32 c3, a2, a3, 0x7531;\n"
                "prmt.b32 d0, b0, b1, 0x6420;\n"
                "prmt.b32 d1, b0, b1, 0x7531;\n"
                "prmt.b32 d2, b2, b3, 0x6420;\n"
                "prmt.b32 d3, b2, b3, 0x7531;\n"
                "and.b32 dst_row_r, lane, 7;\n"
                "shl.b32 dst_row_r, dst_row_r, 1;\n"
                "shr.u32 tmp, lane, 3;\n"
                "and.b32 tmp, tmp, 1;\n"
                "add.u32 dst_row_r, dst_row_r, tmp;\n"
                "shl.b32 tmp, warp, 4;\n"
                "add.u32 dst_row_r, dst_row_r, tmp;\n"
                "shr.u32 dst_col, lane, 4;\n"
                "and.b32 dst_col, dst_col, 1;\n"
                "shl.b32 dst_col, dst_col, 4;\n"
                "shl.b32 dst_log0, dst_row_r, 6;\n"
                "add.u32 dst_log0, dst_log0, dst_col;\n"
                "add.u32 dst_log1, dst_log0, 32;\n"
                "shr.u32 tmp, dst_log0, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 dst_phys0, dst_log0, tmp;\n"
                "shr.u32 tmp2, dst_log1, 7;\n"
                "and.b32 tmp2, tmp2, 3;\n"
                "shl.b32 tmp2, tmp2, 4;\n"
                "xor.b32 dst_phys1, dst_log1, tmp2;\n"
                "add.u32 dst_addr0, $3, dst_phys0;\n"
                "add.u32 dst_addr1, $3, dst_phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr0], {c0, c1, c2, c3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr1], {d0, d1, d2, d3};\n"
                "add.u32 src_log, src_log, 64;\n"
                "shr.u32 tmp, src_log, 7;\n"
                "and.b32 tmp, tmp, 7;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 src_phys, src_log, tmp;\n"
                "add.u32 src_addr0, $2, src_phys;\n"
                "add.u32 src_addr1, src_addr0, 4096;\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{a0, a1, a2, a3}, [src_addr0];\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{b0, b1, b2, b3}, [src_addr1];\n"
                "prmt.b32 c0, a0, a1, 0x6420;\n"
                "prmt.b32 c1, a0, a1, 0x7531;\n"
                "prmt.b32 c2, a2, a3, 0x6420;\n"
                "prmt.b32 c3, a2, a3, 0x7531;\n"
                "prmt.b32 d0, b0, b1, 0x6420;\n"
                "prmt.b32 d1, b0, b1, 0x7531;\n"
                "prmt.b32 d2, b2, b3, 0x6420;\n"
                "prmt.b32 d3, b2, b3, 0x7531;\n"
                "add.u32 dst_log0, dst_log0, 4096;\n"
                "add.u32 dst_log1, dst_log1, 4096;\n"
                "shr.u32 tmp, dst_log0, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 dst_phys0, dst_log0, tmp;\n"
                "shr.u32 tmp2, dst_log1, 7;\n"
                "and.b32 tmp2, tmp2, 3;\n"
                "shl.b32 tmp2, tmp2, 4;\n"
                "xor.b32 dst_phys1, dst_log1, tmp2;\n"
                "add.u32 dst_addr0, $3, dst_phys0;\n"
                "add.u32 dst_addr1, $3, dst_phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr0], {c0, c1, c2, c3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr1], {d0, d1, d2, d3};\n"
                "mov.u32 $0, $1;\n"
                "}"
            ),
            constraints="=r,r,r,r",
            args=[carrier, src_base, dst_base],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _cuda_vtranspose_fp8_64x128_kperm(
        s_src,
        s_dst,
        dst_row: tl.constexpr,
    ):
        """CUDA-authority SW128 -> SW64 FP8 transpose for one 64x128 tile."""
        carrier = tl.arange(0, 128).to(tl.uint32)
        src_base = tle.gpu.local_ptr(s_src, (0, 0))
        dst_base = tle.gpu.local_ptr(s_dst, (dst_row, 0))
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .b32 tid, lane, warp, src_row, tmp, tmp2;\n"
                ".reg .b32 src_log, src_phys, src_addr0, src_addr1;\n"
                ".reg .b32 dst_row_r, dst_col, dst_log0, dst_log1;\n"
                ".reg .b32 dst_phys0, dst_phys1, dst_addr0, dst_addr1;\n"
                ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
                ".reg .b32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 tid, tid, 127;\n"
                "and.b32 lane, tid, 31;\n"
                "shr.u32 warp, tid, 5;\n"
                # CUDA's LDSM/STSM register order presents source-row bits as
                # [b1,b3,b2,b0] to TLE's logical SW64 view.  Apply the inverse
                # [b3,b1,b2,b0] mapping at the load boundary so PV observes
                # the same logical transpose as the tensor path.
                "and.b32 src_row, lane, 17;\n"
                "and.b32 tmp, lane, 8;\n"
                "shr.u32 tmp, tmp, 2;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 2;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 4;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                # Direct CUDA STSM presents P to TLE as dest <- source pi,
                # pi=[0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15] per K16.
                # Load V from pi(dest) as well, preserving the dot product
                # while removing the publication-side cross-lane shuffle.
                "mov.u32 tmp2, src_row;\n"
                "and.b32 src_row, tmp2, 17;\n"
                "and.b32 tmp, tmp2, 4;\n"
                "shr.u32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, tmp2, 8;\n"
                "shr.u32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, tmp2, 2;\n"
                "shl.b32 tmp, tmp, 2;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "shl.b32 src_log, src_row, 7;\n"
                "shl.b32 tmp, warp, 4;\n"
                "add.u32 src_log, src_log, tmp;\n"
                "shr.u32 tmp, src_log, 7;\n"
                "and.b32 tmp, tmp, 7;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 src_phys, src_log, tmp;\n"
                "add.u32 src_addr0, $2, src_phys;\n"
                "add.u32 src_addr1, src_addr0, 4096;\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{a0, a1, a2, a3}, [src_addr0];\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{b0, b1, b2, b3}, [src_addr1];\n"
                "prmt.b32 c0, a0, a1, 0x6420;\n"
                "prmt.b32 c1, a0, a1, 0x7531;\n"
                "prmt.b32 c2, a2, a3, 0x6420;\n"
                "prmt.b32 c3, a2, a3, 0x7531;\n"
                "prmt.b32 d0, b0, b1, 0x6420;\n"
                "prmt.b32 d1, b0, b1, 0x7531;\n"
                "prmt.b32 d2, b2, b3, 0x6420;\n"
                "prmt.b32 d3, b2, b3, 0x7531;\n"
                "and.b32 dst_row_r, lane, 7;\n"
                "shl.b32 dst_row_r, dst_row_r, 1;\n"
                "shr.u32 tmp, lane, 3;\n"
                "and.b32 tmp, tmp, 1;\n"
                "add.u32 dst_row_r, dst_row_r, tmp;\n"
                "shl.b32 tmp, warp, 4;\n"
                "add.u32 dst_row_r, dst_row_r, tmp;\n"
                "shr.u32 dst_col, lane, 4;\n"
                "and.b32 dst_col, dst_col, 1;\n"
                "shl.b32 dst_col, dst_col, 4;\n"
                "shl.b32 dst_log0, dst_row_r, 6;\n"
                "add.u32 dst_log0, dst_log0, dst_col;\n"
                "add.u32 dst_log1, dst_log0, 32;\n"
                "shr.u32 tmp, dst_log0, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 dst_phys0, dst_log0, tmp;\n"
                "shr.u32 tmp2, dst_log1, 7;\n"
                "and.b32 tmp2, tmp2, 3;\n"
                "shl.b32 tmp2, tmp2, 4;\n"
                "xor.b32 dst_phys1, dst_log1, tmp2;\n"
                "add.u32 dst_addr0, $3, dst_phys0;\n"
                "add.u32 dst_addr1, $3, dst_phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr0], {c0, c1, c2, c3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr1], {d0, d1, d2, d3};\n"
                "add.u32 src_log, src_log, 64;\n"
                "shr.u32 tmp, src_log, 7;\n"
                "and.b32 tmp, tmp, 7;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 src_phys, src_log, tmp;\n"
                "add.u32 src_addr0, $2, src_phys;\n"
                "add.u32 src_addr1, src_addr0, 4096;\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{a0, a1, a2, a3}, [src_addr0];\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{b0, b1, b2, b3}, [src_addr1];\n"
                "prmt.b32 c0, a0, a1, 0x6420;\n"
                "prmt.b32 c1, a0, a1, 0x7531;\n"
                "prmt.b32 c2, a2, a3, 0x6420;\n"
                "prmt.b32 c3, a2, a3, 0x7531;\n"
                "prmt.b32 d0, b0, b1, 0x6420;\n"
                "prmt.b32 d1, b0, b1, 0x7531;\n"
                "prmt.b32 d2, b2, b3, 0x6420;\n"
                "prmt.b32 d3, b2, b3, 0x7531;\n"
                "add.u32 dst_log0, dst_log0, 4096;\n"
                "add.u32 dst_log1, dst_log1, 4096;\n"
                "shr.u32 tmp, dst_log0, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 dst_phys0, dst_log0, tmp;\n"
                "shr.u32 tmp2, dst_log1, 7;\n"
                "and.b32 tmp2, tmp2, 3;\n"
                "shl.b32 tmp2, tmp2, 4;\n"
                "xor.b32 dst_phys1, dst_log1, tmp2;\n"
                "add.u32 dst_addr0, $3, dst_phys0;\n"
                "add.u32 dst_addr1, $3, dst_phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr0], {c0, c1, c2, c3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr1], {d0, d1, d2, d3};\n"
                "mov.u32 $0, $1;\n"
                "}"
            ),
            constraints="=r,r,r,r",
            args=[carrier, src_base, dst_base],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _cuda_vtranspose_fp8_64x128(
        s_src,
        s_dst,
        dst_row: tl.constexpr,
        permute_k: tl.constexpr,
    ):
        if permute_k:
            return _cuda_vtranspose_fp8_64x128_kperm(s_src, s_dst, dst_row)
        return _cuda_vtranspose_fp8_64x128_plain(s_src, s_dst, dst_row)


def _set_triton_descriptor_allocator(device: torch.device) -> None:
    """Install the allocator required by FlagTree TLE shared descriptors."""
    assert triton is not None

    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=device)

    try:
        triton.set_allocator(alloc_fn)
    except AttributeError:
        pass


def _per_token_scale(content_abs_amax: torch.Tensor, safe: bool) -> torch.Tensor:
    scale = content_abs_amax / FP8_MAX
    if safe:
        scale = torch.where(content_abs_amax == 0, torch.ones_like(scale), scale)
    return scale


def quantize_q_ckv_per_token(
    q: torch.Tensor,
    head_dim_v: int = D_CKV,
    safe: bool = True,
):
    assert q.shape[-1] == head_dim_v + D_ROPE
    q_nope = q[..., :head_dim_v]
    q_rope = q[..., head_dim_v:]
    amax = q_nope.float().abs().amax(dim=-1, keepdim=True)
    scale = _per_token_scale(amax, safe)
    q_nope_fp8 = (q_nope.float() / scale).to(FP8_DTYPE)
    q_rope_aligned = (q_rope.float() / scale).to(q.dtype)
    return q_nope_fp8, q_rope_aligned, scale.float()


def quantize_k_ckv_per_token(
    blocked_k: torch.Tensor,
    head_dim_v: int = D_CKV,
    safe: bool = True,
):
    assert blocked_k.shape[-1] == head_dim_v + D_ROPE
    k_lora = blocked_k[..., :head_dim_v]
    k_rope = blocked_k[..., head_dim_v:]
    amax = k_lora.float().abs().amax(dim=-1, keepdim=True)
    scale = _per_token_scale(amax, safe)
    k_lora_fp8 = (k_lora.float() / scale).to(FP8_DTYPE)
    k_rope_aligned = (k_rope.float() / scale).to(blocked_k.dtype)
    return k_lora_fp8, k_rope_aligned, scale.float()


class FlashMLAFp8SplitKSchedMeta:
    """Reusable Split-K scheduling metadata."""

    def __init__(self) -> None:
        self.have_initialized = False
        self.config = None
        self.tile_scheduler_metadata = None
        self.num_splits = None
        self.split_batch = None
        self.split_page_begin = None
        self.split_page_end = None
        self.split_num_pages = None
        self.max_splits = 1
        self.total_split_capacity = 0
        self.max_pages_per_split = 0
        self.lifetime_safe_one_pair = True
        self.cache_seqlens_data_ptr = 0
        self.cache_seqlens_version = -1
        self.num_splits_data_ptr = 0
        self.num_splits_version = -1
        self.adaptive_fixed_pages = None
        self.adaptive_fixed_pairs = None
        self.adaptive_selection = ()
        self.capacity_splits = ()
        self.padded_pages = 0


FlashMLAFp8SchedMeta = FlashMLAFp8SplitKSchedMeta


def _tensor_version(tensor: torch.Tensor) -> int:
    try:
        return int(tensor._version)
    except RuntimeError:
        return -1


def _host_lengths(value, label: str, *, batch_size: Optional[int] = None):
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        lengths = tuple(int(item) for item in value.detach().cpu().tolist())
    else:
        lengths = tuple(int(item) for item in value)
    if batch_size is not None and len(lengths) != batch_size:
        raise ValueError(f"{label} must have {batch_size} entries")
    if any(length < 0 or length > MAX_SEQUENCE_LENGTH for length in lengths):
        raise ValueError(f"{label} entries must be in [0, {MAX_SEQUENCE_LENGTH}]")
    return lengths


def _host_certificate_lengths(
    value,
    label: str,
    *,
    batch_size: Optional[int] = None,
):
    """Parse the host-only vectors used by the prepared decode contract."""
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be host integers, not a Tensor")
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be an iterable of host integers")
    try:
        iterator = iter(value)
    except TypeError:
        lengths = (int(value),)
    else:
        lengths = tuple(int(item) for item in iterator)
    if not lengths:
        raise ValueError(f"{label} must not be empty")
    if batch_size is not None and len(lengths) != batch_size:
        raise ValueError(f"{label} must have {batch_size} entries")
    if any(length <= 0 or length > MAX_SEQUENCE_LENGTH for length in lengths):
        raise ValueError(f"{label} entries must be in [1, {MAX_SEQUENCE_LENGTH}]")
    return lengths


def _length_page_state(lengths, pages_per_split: int):
    pages = tuple(math.ceil(int(length) / PAGE_SIZE) for length in lengths)
    splits = tuple(math.ceil(page_count / pages_per_split) for page_count in pages)
    return pages, splits


def _adaptive_fixed_pages(max_pages: int) -> int:
    safety_splits = math.ceil(max_pages / 16)
    required_pages = math.ceil(max_pages / safety_splits)
    return min(16, max(2, 2 * math.ceil(required_pages / 2)))


def _wave_grain_selection(
    max_cache_seqlens: tuple[int, ...],
    h_q: int,
    sm_count: int,
):
    capacity_pages = tuple(
        math.ceil(int(length) / PAGE_SIZE) for length in max_cache_seqlens
    )
    max_pages = max(capacity_pages, default=0)
    if max_pages < ADAPTIVE_MODEL_MIN_PAGES:
        selected = _adaptive_fixed_pages(max_pages)
        return selected, (
            {
                "pages": selected,
                "policy": "adaptive_short_sequence",
                "max_pages": max_pages,
            },
        )

    if h_q <= 0 or h_q % TLE_FP8_BH:
        raise ValueError("HQ must be a positive multiple of 64")
    if sm_count <= 0:
        raise ValueError("SM count must be positive")
    rh = h_q // TLE_FP8_BH

    # CUDA authority (`get_mla_metadata.cu`) assigns each SM partition a
    # payload that includes five fixed-overhead page blocks.  Preserve this
    # implementation's fixed even-pair routing, but derive its grain from the
    # same payload model and round the usable page count up to a whole pair.
    num_sm_parts = max(1, sm_count // rh)
    total_num_blocks = sum(
        pages + CUDA_REF_FIXED_OVERHEAD_PAGES for pages in capacity_pages
    )
    payload_blocks = max(
        math.ceil(total_num_blocks / num_sm_parts) + CUDA_REF_FIXED_OVERHEAD_PAGES,
        2 * CUDA_REF_FIXED_OVERHEAD_PAGES,
    )
    usable_pages = payload_blocks - CUDA_REF_FIXED_OVERHEAD_PAGES
    selected_pages = min(
        ADAPTIVE_MAX_FIXED_PAGES,
        max(
            ADAPTIVE_MIN_FIXED_PAGES,
            2 * math.ceil(usable_pages / 2),
        ),
    )

    # A uniform per-row grain can leave only a handful of CTAs in a second
    # wave.  Keep the fixed even-pair contract, but allow the smallest larger
    # even grain when it collapses that sparse tail back into one H800 wave.
    # This is deliberately capped at 34 pages: it changes B8/L33280 from
    # 8 * ceil(520 / 32) = 136 CTAs to 8 * ceil(520 / 34) = 128 CTAs while
    # leaving the other formal routing points unchanged.
    initial_selected_pages = selected_pages
    initial_counts = tuple(
        max(1, math.ceil(pages / selected_pages)) for pages in capacity_pages
    )
    initial_total_ctas = sum(initial_counts) * rh
    tail_wave_eliminated = False
    if sm_count < initial_total_ctas <= 2 * sm_count:
        for candidate_pages in range(
            selected_pages + 2,
            ADAPTIVE_TAIL_WAVE_MAX_FIXED_PAGES + 1,
            2,
        ):
            candidate_counts = tuple(
                max(1, math.ceil(pages / candidate_pages)) for pages in capacity_pages
            )
            if sum(candidate_counts) * rh <= sm_count:
                selected_pages = candidate_pages
                tail_wave_eliminated = True
                break

    records = []
    for fixed_pages in range(
        ADAPTIVE_MIN_FIXED_PAGES,
        ADAPTIVE_MAX_FIXED_PAGES + 1,
        2,
    ):
        counts = tuple(
            max(1, math.ceil(pages / fixed_pages)) for pages in capacity_pages
        )
        total_splits = sum(counts)
        total_ctas = total_splits * rh
        waves = math.ceil(total_ctas / sm_count)
        fixed_pairs = fixed_pages // 2
        score = waves * (fixed_pairs + 1) + ADAPTIVE_CTA_PENALTY * total_ctas / sm_count
        records.append(
            {
                "pages": fixed_pages,
                "pairs": fixed_pairs,
                "total_splits": total_splits,
                "total_ctas": total_ctas,
                "waves": waves,
                "score": score,
                "policy": "h800_wave_cost",
            }
        )
    selected_counts = tuple(
        max(1, math.ceil(pages / selected_pages)) for pages in capacity_pages
    )
    selection = {
        "pages": selected_pages,
        "pairs": selected_pages // 2,
        "total_splits": sum(selected_counts),
        "total_ctas": sum(selected_counts) * rh,
        "num_sm_parts": num_sm_parts,
        "fixed_overhead_pages": CUDA_REF_FIXED_OVERHEAD_PAGES,
        "payload_blocks": payload_blocks,
        "usable_pages_before_pair_rounding": usable_pages,
        "policy": (
            "cuda_tail_wave_elimination_even_pair"
            if tail_wave_eliminated
            else "cuda_fixed_overhead_even_pair_payload"
        ),
        "tail_wave_eliminated": tail_wave_eliminated,
        "initial_selected_pages": initial_selected_pages,
        "initial_total_ctas": initial_total_ctas,
    }
    return int(selected_pages), (selection, *records)


def _adaptive_schedule(max_cache_seqlens, pages_per_split: int):
    capacity_pages = tuple(
        math.ceil(int(length) / PAGE_SIZE) for length in max_cache_seqlens
    )
    counts = tuple(
        max(1, math.ceil(pages / pages_per_split)) for pages in capacity_pages
    )
    prefix = [0]
    split_batch = []
    split_page_begin = []
    split_page_end = []
    split_num_pages = []
    for batch_index, (pages, count) in enumerate(zip(capacity_pages, counts)):
        prefix.append(prefix[-1] + count)
        for split_index in range(count):
            if pages_per_split == 16 and pages == 520 and count == 33:
                # Pair-aligned balancing: 29x16 + 4x14 = 520 pages.  Spread
                # the four 14-page splits through the row so no 8-page tail
                # remains, while every split retains an even page count.
                short_before = (split_index * 4) // count
                short_through = ((split_index + 1) * 4) // count
                page_begin = split_index * 16 - 2 * short_before
                num_pages = 14 if short_through != short_before else 16
                page_end = page_begin + num_pages
            elif pages_per_split == 34 and pages == 520 and count == 16:
                page_begin = (pages * split_index) // count
                page_end = (pages * (split_index + 1)) // count
                num_pages = page_end - page_begin
            else:
                page_begin = split_index * pages_per_split
                num_pages = max(0, min(pages_per_split, pages - page_begin))
            split_batch.append(batch_index)
            split_page_begin.append(page_begin)
            split_page_end.append(page_begin + num_pages)
            split_num_pages.append(num_pages)
    padded_pages = max(
        1,
        max(
            (count * pages_per_split for count in counts),
            default=1,
        ),
    )
    return (
        tuple(prefix),
        tuple(split_batch),
        tuple(split_page_begin),
        tuple(split_page_end),
        tuple(split_num_pages),
        counts,
        padded_pages,
    )


def _build_adaptive_execution_meta(
    max_cache_seqlens,
    h_q: int,
    device: torch.device,
    short_pages_per_split: int,
):
    capacity_pages = tuple(
        math.ceil(int(length) / PAGE_SIZE) for length in max_cache_seqlens
    )
    max_pages = max(capacity_pages, default=0)
    if max_pages <= 2:
        fixed_pages = int(short_pages_per_split)
        selection = (
            {
                "pages": fixed_pages,
                "policy": "direct_two_page",
                "max_pages": max_pages,
            },
        )
    elif 3 <= max_pages <= 8:
        # Short-K route: expose one physical-page CTA at a time instead of
        # serializing the complete 3-8 page row in a single CTA.  This is
        # host scheduling only; the strict-2WG kernel and pair pipeline are
        # unchanged.
        fixed_pages = 1
        selection = (
            {
                "pages": fixed_pages,
                "pairs": 1,
                "policy": "shortk_pagegrain_3_to_8_pages_v1",
                "max_pages": max_pages,
            },
        )
    elif (
        max_pages == 10
        and len(capacity_pages) >= 32
        and all(pages == 10 for pages in capacity_pages)
    ):
        # At high batch, five two-page split CTAs per row over-subscribe the
        # short ten-page workload and require a combine kernel. Use one
        # direct-output CTA per (batch, 64-head group) and only finalize LSE.
        # Keep this eligibility exact for heterogeneous rows and adjacent lengths.
        fixed_pages = max_pages
        selection = (
            {
                "pages": fixed_pages,
                "pairs": math.ceil(fixed_pages / 2),
                "policy": "b32plus_l640_direct_single",
                "max_pages": max_pages,
            },
        )
    elif (
        max_pages == 10
        and h_q == 128
        and len(capacity_pages) == 16
        and all(pages == 10 for pages in capacity_pages)
    ):
        # For B16/L640, reduce the partial grid from 160 CTAs (five
        # two-page splits per row) to 96 CTAs (three four-page-capacity
        # splits per row and two head groups).
        fixed_pages = 4
        selection = (
            {
                "pages": fixed_pages,
                "pairs": fixed_pages // 2,
                "policy": "b16_l640_four_page_grain",
                "max_pages": max_pages,
            },
        )
    elif 9 <= max_pages <= 10:
        # A ten-page direct-single CTA is not the best short-sequence route.  Use the finest
        # legal two-page pair grain to expose five split CTAs, mirroring the
        # CUDA reference's short-workload parallel split behavior.  Keep the
        # policy deliberately narrow until adjacent page ranges are measured.
        fixed_pages = 2
        selection = (
            {
                "pages": fixed_pages,
                "pairs": 1,
                "policy": "cuda_short_parallel_pair_9_to_10_pages",
                "max_pages": max_pages,
            },
        )
    elif (
        max_pages == 128
        and h_q == 64
        and len(capacity_pages) >= 64
        and all(pages == 128 for pages in capacity_pages)
    ):
        # Choose an even per-row grain that targets one H800
        # partial-CTA wave for a regular high-batch 8192-token workload.
        # This changes only host scheduling metadata; the partial kernel,
        # TMA/WGMMA/barrier structure, math, and route contracts are reused.
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        target_splits_per_row = max(1, sm_count // len(capacity_pages))
        fixed_pages = min(
            max_pages,
            2 * math.ceil(math.ceil(max_pages / target_splits_per_row) / 2),
        )
        selection = (
            {
                "pages": fixed_pages,
                "pairs": fixed_pages // 2,
                "policy": "b64plus_l8192_onewave_even_grain",
                "max_pages": max_pages,
            },
        )
    elif (
        max_pages == 520
        and h_q == 64
        and len(capacity_pages) >= 16
        and all(pages == 520 for pages in capacity_pages)
    ):
        # Choose the minimum even grain that caps the regular high-batch
        # L33280 workload at one H800 partial-CTA wave.
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        target_splits_per_row = max(1, sm_count // len(capacity_pages))
        fixed_pages = min(
            max_pages,
            2 * math.ceil(math.ceil(max_pages / target_splits_per_row) / 2),
        )
        selection = (
            {
                "pages": fixed_pages,
                "pairs": fixed_pages // 2,
                "policy": "b16plus_l33280_onewave_even_grain",
                "max_pages": max_pages,
            },
        )
    else:
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        fixed_pages, selection = _wave_grain_selection(
            tuple(max_cache_seqlens), h_q, sm_count
        )
    (
        prefix,
        split_batch,
        split_page_begin,
        split_page_end,
        split_num_pages,
        counts,
        padded_pages,
    ) = _adaptive_schedule(max_cache_seqlens, fixed_pages)

    meta = FlashMLAFp8SplitKSchedMeta()
    meta.have_initialized = True
    meta.num_splits = torch.tensor(prefix, dtype=torch.int32, device=device)
    meta.split_batch = torch.tensor(split_batch, dtype=torch.int32, device=device)
    meta.split_page_begin = torch.tensor(
        split_page_begin, dtype=torch.int32, device=device
    )
    meta.split_page_end = torch.tensor(split_page_end, dtype=torch.int32, device=device)
    meta.split_num_pages = torch.tensor(
        split_num_pages, dtype=torch.int32, device=device
    )
    meta.max_splits = max(counts, default=1)
    meta.total_split_capacity = len(split_batch)
    meta.max_pages_per_split = max(split_num_pages, default=0)
    meta.lifetime_safe_one_pair = meta.max_pages_per_split <= 2
    meta.num_splits_data_ptr = int(meta.num_splits.data_ptr())
    meta.num_splits_version = _tensor_version(meta.num_splits)
    meta.adaptive_fixed_pages = fixed_pages
    meta.adaptive_fixed_pairs = math.ceil(fixed_pages / 2)
    meta.adaptive_selection = selection
    meta.capacity_splits = counts
    meta.padded_pages = padded_pages
    return meta


def _pad_block_table(block_table: torch.Tensor, padded_pages: int):
    if int(block_table.shape[1]) >= padded_pages:
        return block_table
    padded = torch.zeros(
        (int(block_table.shape[0]), padded_pages),
        dtype=block_table.dtype,
        device=block_table.device,
    )
    padded[:, : int(block_table.shape[1])].copy_(block_table)
    return padded


def _fixed_split_counts(
    cache_seqlens: torch.Tensor,
    pages_per_split: int,
    max_splits: Optional[int] = None,
) -> torch.Tensor:
    if pages_per_split <= 0:
        raise ValueError("pages_per_split must be positive")
    pages = torch.div(
        cache_seqlens.to(torch.int64) + PAGE_SIZE - 1,
        PAGE_SIZE,
        rounding_mode="floor",
    )
    counts = torch.div(
        pages + pages_per_split - 1,
        pages_per_split,
        rounding_mode="floor",
    ).clamp_min(1)
    if max_splits is not None:
        if max_splits <= 0:
            raise ValueError("max_splits must be positive")
        counts = counts.clamp_max(max_splits)
    return counts.to(torch.int32)


def _prefix_from_counts(counts: torch.Tensor) -> torch.Tensor:
    prefix = torch.empty((counts.numel() + 1,), dtype=torch.int32, device=counts.device)
    prefix[0] = 0
    prefix[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
    return prefix


def get_mla_ckv_fp8_metadata(
    cache_seqlens: Optional[torch.Tensor] = None,
    num_q_heads_per_k_head: Optional[int] = None,
    num_k_heads: int = 1,
    *,
    pages_per_split: int = DEFAULT_PAGES_PER_SPLIT,
    max_splits: Optional[int] = None,
) -> Tuple[FlashMLAFp8SplitKSchedMeta, Optional[torch.Tensor]]:
    meta = FlashMLAFp8SplitKSchedMeta()
    if cache_seqlens is None:
        return meta, None
    if cache_seqlens.ndim != 1 or cache_seqlens.dtype != torch.int32:
        raise AssertionError("cache_seqlens must be a 1-D int32 tensor")

    counts = _fixed_split_counts(cache_seqlens, pages_per_split, max_splits)
    prefix = _prefix_from_counts(counts)
    actual_max = int(counts.max().item()) if counts.numel() else 1
    h_q = int(num_q_heads_per_k_head or TLE_FP8_BH) * int(num_k_heads)
    meta.have_initialized = True
    meta.max_splits = actual_max
    meta.total_split_capacity = int(cache_seqlens.numel()) * actual_max
    meta.lifetime_safe_one_pair = pages_per_split <= 2 and max_splits is None
    meta.num_splits = prefix
    meta.cache_seqlens_data_ptr = int(cache_seqlens.data_ptr())
    meta.cache_seqlens_version = _tensor_version(cache_seqlens)
    meta.num_splits_data_ptr = int(prefix.data_ptr())
    meta.num_splits_version = _tensor_version(prefix)
    (
        meta.split_batch,
        meta.split_page_begin,
        meta.split_page_end,
        meta.split_num_pages,
    ) = _build_compact_split_plan(cache_seqlens, prefix)
    meta.total_split_capacity = int(meta.split_batch.numel())
    meta.max_pages_per_split = (
        int(meta.split_num_pages.max().item()) if meta.total_split_capacity else 0
    )
    meta.lifetime_safe_one_pair = meta.max_pages_per_split <= 2
    return meta, prefix


# Compatibility alias for internal validation callers.
get_mla_metadata = get_mla_ckv_fp8_metadata


def _split_page_bounds(num_pages: int, split_idx: int, split_count: int):
    if split_count <= 0 or not 0 <= split_idx < split_count:
        raise ValueError("invalid split index/count")
    return (
        (num_pages * split_idx) // split_count,
        (num_pages * (split_idx + 1)) // split_count,
    )


def _build_compact_split_plan(cache_seqlens, num_splits):
    seqlens_cpu = cache_seqlens.detach().cpu()
    prefix_cpu = num_splits.detach().cpu()
    split_batch = []
    split_page_begin = []
    split_page_end = []
    split_num_pages = []
    for batch_idx in range(cache_seqlens.numel()):
        cache_len = int(seqlens_cpu[batch_idx].item())
        num_pages = (cache_len + PAGE_SIZE - 1) // PAGE_SIZE
        begin = int(prefix_cpu[batch_idx].item())
        end = int(prefix_cpu[batch_idx + 1].item())
        split_count = end - begin
        if split_count <= 0:
            raise AssertionError("every request must own at least one split")
        for local_split in range(split_count):
            page_begin, page_end = _split_page_bounds(
                num_pages, local_split, split_count
            )
            split_batch.append(batch_idx)
            split_page_begin.append(page_begin)
            split_page_end.append(page_end)
            split_num_pages.append(page_end - page_begin)

    device = cache_seqlens.device
    return (
        torch.tensor(split_batch, dtype=torch.int32, device=device),
        torch.tensor(split_page_begin, dtype=torch.int32, device=device),
        torch.tensor(split_page_end, dtype=torch.int32, device=device),
        torch.tensor(split_num_pages, dtype=torch.int32, device=device),
    )


if HAS_TLE:  # pragma: no cover - H800 + FlagTree TLE only
    _TLE_LOG2E = tl.constexpr(LOG2E)
    _TLE_LN2 = tl.constexpr(LN2)
    _TLE_FP8_MAX = tl.constexpr(FP8_MAX)
    _TLE_P_AMAX_FLOOR = tl.constexpr(P_AMAX_FLOOR)
    _TLE_NEG_INF = tl.constexpr(float("-inf"))
    _TLE_POS_INF = tl.constexpr(float("inf"))

    @triton.jit
    def _fp8_mla_wg0(
        q_desc,
        qr_desc,
        qs_desc,
        out_desc,
        k_desc,
        block_table,
        stride_bt_pg,
        row0,
        num_pages,
        q_ckv_full,
        q_rope_full,
        q_scale_full,
        k_content_full,
        k_rope_full,
        k_scale_full,
        state0_ready,
        state1_ready,
        p0_ready,
        p1_ready,
        v0_ready,
        v1_ready,
        slot0_empty,
        slot1_empty,
        tail0_zero_ready,
        s_q,
        s_qr,
        s_kc_a0,
        s_kc_a1,
        s_kc_a2,
        s_kc_a3,
        s_kc_b0,
        s_kc_b1,
        s_kc_b2,
        s_kc_b3,
        s_kr_a,
        s_vt0_a,
        s_vt1_a,
        s_vt0_b,
        s_p_a,
        s_p_b,
        s_beta_a,
        s_beta_b,
        s_state0_m,
        s_state0_s,
        s_state0_l,
        s_state0_valid,
        s_state1_m,
        s_state1_s,
        s_state1_l,
        s_state1_valid,
        split_cache_seqlen,
        out_ptr,
        lse2_ptr,
        stride_po_h,
        stride_pl_h,
        h_base,
        softmax_scale,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        DP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        PAGE_GRAIN_TAIL_ZERO: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        ENABLE_PDL: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
        DIRECT_LSE: tl.constexpr,
    ):
        """WG0: Q owner, even-page math, and the left output half."""
        # The three CUDA-aligned Q payloads are one-shot TMA transactions.  The
        # scale temporarily occupies state1_m; WG1 cannot overwrite that field
        # until state0_ready, after both workers have consumed Q scale.
        s_state1_m_row = s_state1_m.slot(0)
        s_beta_a_row = s_beta_a.slot(0)
        s_beta_b_row = s_beta_b.slot(0)
        state_idx = tl.arange(0, BH)
        tle.gpu.copy(q_desc, s_q, [BH, CKV], [row0, 0], barrier=q_ckv_full)
        tle.gpu.copy(qr_desc, s_qr, [BH, ROPE], [row0, 0], barrier=q_rope_full)
        tle.gpu.copy(
            qs_desc,
            s_state1_m,
            [1, BH],
            [row0 // HQ, h_base],
            barrier=q_scale_full,
        )
        tle.gpu.barrier_wait(q_ckv_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_rope_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_scale_full, phaseIdx=0)

        offs_t = tl.arange(0, BK)
        offs_h = h_base + tl.arange(0, BH)
        mask_h = offs_h < HQ
        qs = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), volatile=True)

        acc_left = tl.zeros((BH, DP), dtype=tl.float32)
        state_m = tl.full((BH,), float("-inf"), tl.float32)
        state_s = tl.full((BH,), 1.0, tl.float32)
        state_l = tl.zeros((BH,), dtype=tl.float32)
        state_valid = tl.zeros((BH,), dtype=tl.int32) != 0

        q_rows_d128 = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, K_CONTENT_TILE))
        q_c0_cols = tl.broadcast_to(
            tl.arange(0, K_CONTENT_TILE)[None, :], (BH, K_CONTENT_TILE)
        )
        q_c1_cols = tl.broadcast_to(
            (K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c2_cols = tl.broadcast_to(
            (2 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c3_cols = tl.broadcast_to(
            (3 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c0 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c0_cols)))
        q_c1 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c1_cols)))
        q_c2 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c2_cols)))
        q_c3 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c3_cols)))
        k_a_c0 = s_kc_a0
        k_a_c1 = s_kc_a1
        k_a_c2 = s_kc_a2
        k_a_c3 = s_kc_a3
        k_b_c0 = s_kc_b0
        k_b_c1 = s_kc_b1
        prow = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, BK))
        pcol = tl.broadcast_to(tl.arange(0, BK)[None, :], (BH, BK))
        kv_rows_d128 = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DP // 2))
        kv_c0_cols = tl.broadcast_to(tl.arange(0, DP // 2)[None, :], (BK, DP // 2))
        kv_c1_cols = tl.broadcast_to(
            (DP // 2 + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        kv_c2_cols = tl.broadcast_to(
            (DP + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        kv_c3_cols = tl.broadcast_to(
            (DP + DP // 2 + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        vt_c0_rows = tl.broadcast_to(tl.arange(0, DP // 2)[:, None], (DP // 2, BK))
        vt_c1_rows = tl.broadcast_to(
            (DP // 2 + tl.arange(0, DP // 2))[:, None], (DP // 2, BK)
        )
        vt_cols_d128 = tl.broadcast_to(tl.arange(0, BK)[None, :], (DP // 2, BK))

        num_pairs = (num_pages + 1) // 2
        # Fixed writer ownership applies to cold prime and steady state: WG0
        # issues content tiles 0/1 for both physical slots.
        if num_pages > 0:
            first_phys = tl.load(block_table)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_a_c0,
                [BK, K_CONTENT_TILE],
                [first_base, 0],
                barrier=k_content_full[0],
            )
            tle.gpu.copy(
                k_desc,
                k_a_c1,
                [BK, K_CONTENT_TILE],
                [first_base, K_CONTENT_TILE],
                barrier=k_content_full[1],
            )
        if num_pages > 1:
            first_phys = tl.load(block_table + stride_bt_pg)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_b_c0,
                [BK, K_CONTENT_TILE],
                [first_base, 0],
                barrier=k_content_full[4],
            )
            tle.gpu.copy(
                k_desc,
                k_b_c1,
                [BK, K_CONTENT_TILE],
                [first_base, K_CONTENT_TILE],
                barrier=k_content_full[5],
            )

        # Cold prime: page 0 QK, scale, and V are steady-loop live-ins. Rope
        # accumulates after content tile 3, matching the CUDA rP0 sequence.
        qk = tl.zeros((BH, BK), dtype=tl.float32)
        ks = tl.zeros((BK,), dtype=tl.float32)
        if num_pages > 0:
            tle.gpu.barrier_wait(k_content_full[0], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c0, k_a_c0, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[1], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c1, k_a_c1, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[2], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c2, k_a_c2, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[3], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c3, k_a_c3, qk, trans_b=True)
            tle.gpu.barrier_wait(k_rope_full[0], phaseIdx=0)
            qk = tle.gpu.wgmma(s_qr, s_kr_a, qk, trans_b=True)
            qk = tle.gpu.wgmma_wait(0, qk)

            tle.gpu.barrier_wait(k_scale_full[0], phaseIdx=0)
            prime_valid = offs_t < split_cache_seqlen
            ks_raw = tl.load(tle.gpu.local_ptr(s_beta_a_row, (offs_t,)))
            ks = tl.where(prime_valid, ks_raw, 0.0)

        steady_pairs = tl.maximum(num_pairs - 1, 0)
        for pair in tl.range(steady_pairs, disable_licm=True):
            page = pair * 2
            if FULL_TAIL:
                valid = tl.full((BK,), True, tl.int1)
            else:
                valid = page * PAGE_SIZE + offs_t < split_cache_seqlen
            valid_row = valid[None, :]
            score = qk * qs[:, None] * ks[None, :] * softmax_scale
            score_safe = score if FULL_TAIL else tl.where(valid_row, score, 0.0)
            x = score_safe * _TLE_LOG2E
            page_m = tl.max(
                x if FULL_TAIL else tl.where(valid_row, x, _TLE_NEG_INF), axis=1
            )
            old_m = tl.where(state_valid, state_m, _TLE_NEG_INF)
            old_s = tl.where(state_valid, state_s, 1.0)
            old_l = tl.where(state_valid, state_l, 0.0)
            m_new = tl.maximum(old_m, page_m)
            m_safe = tl.where(m_new == _TLE_NEG_INF, 0.0, m_new)
            e = (
                tl.exp2(x - m_safe[:, None])
                if FULL_TAIL
                else tl.where(valid_row, tl.exp2(x - m_safe[:, None]), 0.0)
            )
            f = e * ks[None, :]
            amax = tl.max(tl.abs(f), axis=1)
            s_new = tl.where(
                amax == 0.0,
                1.0,
                tl.maximum(amax, _TLE_P_AMAX_FLOOR) / _TLE_FP8_MAX,
            )
            page_valid = True if FULL_TAIL else page * PAGE_SIZE < split_cache_seqlen
            if USE_HOTLOOP_RECIP:
                inv_s_new = 1.0 / s_new
                p_scaled = f * inv_s_new[:, None]
            else:
                p_scaled = f / s_new[:, None]
            p_new = tl.clamp(p_scaled, -_TLE_FP8_MAX, _TLE_FP8_MAX)
            p0 = (
                p_new
                if FULL_TAIL
                else tl.where(page_valid, p_new, tl.zeros_like(p_new))
            )
            if FULL_TAIL:
                _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p_a, p0)
            else:
                p0_store = p_new.to(tl.float8e4nv)
                p0_store = tl.where(page_valid, p0_store, tl.zeros_like(p0_store))
                tl.store(tle.gpu.local_ptr(s_p_a, (prow, pcol)), p0_store)
            old_m_finite = tl.where(state_valid, old_m, 0.0)
            alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
            if USE_HOTLOOP_RECIP:
                beta = alpha * old_s * inv_s_new
                l_new = old_l * beta + tl.sum(e, axis=1) * inv_s_new
            else:
                beta = alpha * old_s / s_new
                l_new = old_l * beta + tl.sum(e, axis=1) / s_new
            state_m = tl.where(page_valid, m_new, old_m)
            state_s = tl.where(page_valid, s_new, old_s)
            state_l = tl.where(page_valid, l_new, old_l)
            beta = tl.where(page_valid, beta, 1.0)
            state_valid = state_valid | page_valid

            tl.store(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)), beta)
            tl.store(tle.gpu.local_ptr(s_state0_m, (state_idx,)), state_m)
            tl.store(tle.gpu.local_ptr(s_state0_s, (state_idx,)), state_s)
            tl.store(tle.gpu.local_ptr(s_state0_l, (state_idx,)), state_l)
            tl.store(
                tle.gpu.local_ptr(s_state0_valid, (state_idx,)),
                state_valid.to(tl.int32),
            )

            # CUDA publishes the completed online-softmax state at its last
            # shared write. Do not serialize WG1 softmax behind the unrelated
            # V repack that follows in WG0.
            if not MERGE_STATE_V:
                tle.gpu.barrier_arrive(state0_ready, phaseIdx=pair)

            # This loop excludes the final pair, so its even page is always a
            # complete logical page.  Match CUDA's compile-time steady-state
            # specialization and keep the masked tensor fallback in the
            # epilogue only.
            _cuda_vtranspose_fp8_64x128(s_kc_a0, s_vt0_a, 0, FULL_TAIL)
            _cuda_vtranspose_fp8_64x128(s_kc_a1, s_vt0_a, DP // 2, FULL_TAIL)
            _cuda_vtranspose_fp8_64x128(s_kc_a2, s_vt1_a, 0, FULL_TAIL)
            _cuda_vtranspose_fp8_64x128(s_kc_a3, s_vt1_a, DP // 2, FULL_TAIL)

            tle.gpu.barrier_arrive(v0_ready, phaseIdx=pair)

            # CUDA local-P wait point: finish current local PV, then launch
            # slot-A generation pair+1 content0/1 for p+2.
            acc_left *= beta[:, None]
            acc_left = tle.gpu.wgmma(s_p_a, s_vt0_a, acc_left, trans_b=True)
            acc_left = tle.gpu.wgmma_wait(0, acc_left)

            next_even_page = page + 2
            next_generation = pair + 1
            next_qk = tl.zeros((BH, BK), dtype=tl.float32)
            next_ks = tl.zeros((BK,), dtype=tl.float32)
            if True:
                next_even_phys = tl.load(block_table + next_even_page * stride_bt_pg)
                next_even_base = (next_even_phys * BK).to(tl.int32)
                tle.gpu.copy(
                    k_desc,
                    k_a_c0,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 0],
                    barrier=k_content_full[0],
                )
                tle.gpu.copy(
                    k_desc,
                    k_a_c1,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, K_CONTENT_TILE],
                    barrier=k_content_full[1],
                )

            odd_page = page + 1
            if True:
                tle.gpu.barrier_wait(v1_ready, phaseIdx=pair)
                beta1 = tl.load(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)))
                acc_left *= beta1[:, None]
                acc_left = tle.gpu.wgmma(s_p_b, s_vt0_b, acc_left, trans_b=True)

                # Keep the async rP0 chain inside one real-p+2 branch:
                # TLE permits loop-carried accumulators but not an async value
                # yielded through an intermediate scf.if.
                if True:
                    # CUDA QK phase-0. Two younger QK groups allow wait2 to
                    # retire only the oldest remote-P group.
                    tle.gpu.barrier_wait(k_content_full[0], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c0, k_a_c0, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[1], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c1, k_a_c1, next_qk, trans_b=True)
                    phase0_waited_qk = tle.gpu.wgmma_wait(2, next_qk)
                    tle.gpu.barrier_arrive(slot1_empty, phaseIdx=pair)

                    # CUDA wait2 point starts p+3 content0/1 before p+2
                    # phase-2.
                    next_odd_page = odd_page + 2
                    if next_odd_page < num_pages:
                        next_odd_phys = tl.load(
                            block_table + next_odd_page * stride_bt_pg
                        )
                        next_odd_base = (next_odd_phys * BK).to(tl.int32)
                        tle.gpu.copy(
                            k_desc,
                            k_b_c0,
                            [BK, K_CONTENT_TILE],
                            [next_odd_base, 0],
                            barrier=k_content_full[4],
                        )
                        tle.gpu.copy(
                            k_desc,
                            k_b_c1,
                            [BK, K_CONTENT_TILE],
                            [next_odd_base, K_CONTENT_TILE],
                            barrier=k_content_full[5],
                        )

                    # CUDA QK phase-2 completes p+2 in the current pair.
                    tle.gpu.barrier_wait(k_content_full[2], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c2, k_a_c2, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[3], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c3, k_a_c3, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_rope_full[0], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(s_qr, s_kr_a, next_qk, trans_b=True)
                    next_qk = tle.gpu.wgmma_wait(0, next_qk)
                    # The wait is global in hardware, but TLE also requires
                    # the remote-P SSA value itself to pass through a wait.
                    acc_left = tle.gpu.wgmma_wait(0, acc_left)

                    tle.gpu.barrier_wait(k_scale_full[0], phaseIdx=next_generation)
                    next_valid = (
                        next_even_page * PAGE_SIZE + offs_t < split_cache_seqlen
                    )
                    next_ks_raw = tl.load(tle.gpu.local_ptr(s_beta_a_row, (offs_t,)))
                    next_ks = tl.where(next_valid, next_ks_raw, 0.0)

                else:
                    # Tail pair: no younger QK groups exist to retain.
                    acc_left = tle.gpu.wgmma_wait(0, acc_left)
                    tle.gpu.barrier_arrive(slot1_empty, phaseIdx=pair)

                if not MERGE_STATE_V:
                    tle.gpu.barrier_wait(state1_ready, phaseIdx=pair)
                state_m = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)))
                state_s = tl.load(tle.gpu.local_ptr(s_state1_s, (state_idx,)))
                state_l = tl.load(tle.gpu.local_ptr(s_state1_l, (state_idx,)))
                state_valid = (
                    tl.load(tle.gpu.local_ptr(s_state1_valid, (state_idx,))) != 0
                )

            # WG1 publishes this only after its remote P0/V0 wait0.
            tle.gpu.barrier_wait(slot0_empty, phaseIdx=pair)
            qk = next_qk
            ks = next_ks

        # CUDA-style epilogue: the final pair never creates a younger QK
        # accumulator, so every PV dependency is retired inside this tail.
        if num_pairs > 0:
            pair = steady_pairs
            page = pair * 2
            valid = page * PAGE_SIZE + offs_t < split_cache_seqlen
            valid_row = valid[None, :]
            tail_ks_raw = tl.load(tle.gpu.local_ptr(s_beta_a_row, (offs_t,)))
            tail_ks = tail_ks_raw if FULL_TAIL else tl.where(valid, tail_ks_raw, 0.0)
            score = qk * qs[:, None] * tail_ks[None, :] * softmax_scale
            score_safe = score if FULL_TAIL else tl.where(valid_row, score, 0.0)
            x = score_safe * _TLE_LOG2E
            page_m = tl.max(
                x if FULL_TAIL else tl.where(valid_row, x, _TLE_NEG_INF), axis=1
            )
            old_m = tl.where(state_valid, state_m, _TLE_NEG_INF)
            old_s = tl.where(state_valid, state_s, 1.0)
            old_l = tl.where(state_valid, state_l, 0.0)
            m_new = tl.maximum(old_m, page_m)
            m_safe = tl.where(m_new == _TLE_NEG_INF, 0.0, m_new)
            e = (
                tl.exp2(x - m_safe[:, None])
                if FULL_TAIL
                else tl.where(valid_row, tl.exp2(x - m_safe[:, None]), 0.0)
            )
            f = e * tail_ks[None, :]
            amax = tl.max(tl.abs(f), axis=1)
            s_new = tl.where(
                amax == 0.0,
                1.0,
                tl.maximum(amax, _TLE_P_AMAX_FLOOR) / _TLE_FP8_MAX,
            )
            page_valid = True if FULL_TAIL else page * PAGE_SIZE < split_cache_seqlen
            inv_s_new = 1.0 / s_new
            p_new = tl.clamp(f * inv_s_new[:, None], -_TLE_FP8_MAX, _TLE_FP8_MAX)
            p0 = (
                p_new
                if FULL_TAIL
                else tl.where(page_valid, p_new, tl.zeros_like(p_new))
            )
            if FULL_TAIL:
                _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p_a, p0)
            else:
                p0_store = p_new.to(tl.float8e4nv)
                p0_store = tl.where(page_valid, p0_store, tl.zeros_like(p0_store))
                tl.store(tle.gpu.local_ptr(s_p_a, (prow, pcol)), p0_store)
            old_m_finite = tl.where(state_valid, old_m, 0.0)
            alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
            beta = alpha * old_s * inv_s_new
            l_new = old_l * beta + tl.sum(e, axis=1) * inv_s_new
            state_m = tl.where(page_valid, m_new, old_m)
            state_s = tl.where(page_valid, s_new, old_s)
            state_l = tl.where(page_valid, l_new, old_l)
            beta = tl.where(page_valid, beta, 1.0)
            state_valid = state_valid | page_valid

            tl.store(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)), beta)
            tl.store(tle.gpu.local_ptr(s_state0_m, (state_idx,)), state_m)
            tl.store(tle.gpu.local_ptr(s_state0_s, (state_idx,)), state_s)
            tl.store(tle.gpu.local_ptr(s_state0_l, (state_idx,)), state_l)
            tl.store(
                tle.gpu.local_ptr(s_state0_valid, (state_idx,)),
                state_valid.to(tl.int32),
            )

            # Tail generation follows the same last-write publication rule.
            if not MERGE_STATE_V:
                tle.gpu.barrier_arrive(state0_ready, phaseIdx=pair)

            # Invalid probability columns are already exact FP8 zero after
            # the masked softmax above, so their V values cannot contribute
            # to PV.  Reuse the CUDA-aligned vectorized transpose for a
            # partial physical page instead of materializing a masked tensor
            # transpose in registers.
            if PAGE_GRAIN_TAIL_ZERO:
                if not FULL_TAIL:
                    valid_tokens = tl.minimum(split_cache_seqlen, BK)
                    if valid_tokens < BK:
                        _zero_invalid_fp8_rows_sw128_x4(
                            s_kc_a0,
                            s_kc_a1,
                            s_kc_a2,
                            s_kc_a3,
                            valid_tokens,
                        )
                        tle.gpu.barrier_arrive(tail0_zero_ready, phaseIdx=pair)
                        tle.gpu.barrier_wait(tail0_zero_ready, phaseIdx=pair)
                _cuda_vtranspose_fp8_64x128(s_kc_a0, s_vt0_a, 0, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_a1, s_vt0_a, DP // 2, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_a2, s_vt1_a, 0, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_a3, s_vt1_a, DP // 2, FULL_TAIL)
            elif FULL_TAIL or (page + 1) * PAGE_SIZE <= split_cache_seqlen:
                _cuda_vtranspose_fp8_64x128(s_kc_a0, s_vt0_a, 0, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_a1, s_vt0_a, DP // 2, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_a2, s_vt1_a, 0, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_a3, s_vt1_a, DP // 2, FULL_TAIL)
            else:
                kc_tile = tl.load(
                    tle.gpu.local_ptr(s_kc_a0, (kv_rows_d128, kv_c0_cols))
                )
                kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                tl.store(
                    tle.gpu.local_ptr(s_vt0_a, (vt_c0_rows, vt_cols_d128)),
                    tl.trans(kc_tile),
                )
                kc_tile = tl.load(
                    tle.gpu.local_ptr(s_kc_a1, (kv_rows_d128, kv_c0_cols))
                )
                kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                tl.store(
                    tle.gpu.local_ptr(s_vt0_a, (vt_c1_rows, vt_cols_d128)),
                    tl.trans(kc_tile),
                )
                kc_tile = tl.load(
                    tle.gpu.local_ptr(s_kc_a2, (kv_rows_d128, kv_c0_cols))
                )
                kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                tl.store(
                    tle.gpu.local_ptr(s_vt1_a, (vt_c0_rows, vt_cols_d128)),
                    tl.trans(kc_tile),
                )
                kc_tile = tl.load(
                    tle.gpu.local_ptr(s_kc_a3, (kv_rows_d128, kv_c0_cols))
                )
                kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                tl.store(
                    tle.gpu.local_ptr(s_vt1_a, (vt_c1_rows, vt_cols_d128)),
                    tl.trans(kc_tile),
                )

            tle.gpu.barrier_arrive(v0_ready, phaseIdx=pair)

            acc_left *= beta[:, None]
            acc_left = tle.gpu.wgmma(s_p_a, s_vt0_a, acc_left, trans_b=True)
            acc_left = tle.gpu.wgmma_wait(0, acc_left)

            odd_page = page + 1
            if odd_page < num_pages:
                tle.gpu.barrier_wait(v1_ready, phaseIdx=pair)
                beta1 = tl.load(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)))
                acc_left *= beta1[:, None]
                acc_left = tle.gpu.wgmma(s_p_b, s_vt0_b, acc_left, trans_b=True)
                acc_left = tle.gpu.wgmma_wait(0, acc_left)
                tle.gpu.barrier_arrive(slot1_empty, phaseIdx=pair)

                if not MERGE_STATE_V:
                    tle.gpu.barrier_wait(state1_ready, phaseIdx=pair)
                state_m = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)))
                state_s = tl.load(tle.gpu.local_ptr(s_state1_s, (state_idx,)))
                state_l = tl.load(tle.gpu.local_ptr(s_state1_l, (state_idx,)))
                state_valid = (
                    tl.load(tle.gpu.local_ptr(s_state1_valid, (state_idx,))) != 0
                )

            tle.gpu.barrier_wait(slot0_empty, phaseIdx=pair)

        # CUDA-aligned programmatic dependency trigger.  Only the B>=4
        # coarse-combine specialization receives ENABLE_PDL=True.
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()

        offs_d = tl.arange(0, DP)
        l_div = tl.where(state_l > 0.0, state_l, 1.0)
        inv_l_div = 1.0 / l_div
        out_left = tl.where(state_valid[:, None], acc_left * inv_l_div[:, None], 0.0)
        if USE_TMA_OUTPUT:
            # The final K-rope read has retired before the loop exits, so its
            # existing 64x64 BF16 buffer can stage four output chunks without
            # adding shared memory. Each copy is a complete TMA S2G group; the
            # TLE store scheduler inserts the reuse-safe commit/wait sequence.
            out_left_lo, out_left_hi = tl.split(
                tl.permute(tl.reshape(out_left, (BH, 2, DP // 2)), (0, 2, 1))
            )
            out_left_0, out_left_1 = tl.split(
                tl.permute(tl.reshape(out_left_lo, (BH, 2, ROPE)), (0, 2, 1))
            )
            out_left_2, out_left_3 = tl.split(
                tl.permute(tl.reshape(out_left_hi, (BH, 2, ROPE)), (0, 2, 1))
            )
            tile_rows = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, ROPE))
            tile_cols = tl.broadcast_to(tl.arange(0, ROPE)[None, :], (BH, ROPE))
            tl.store(
                tle.gpu.local_ptr(s_kr_a, (tile_rows, tile_cols)),
                out_left_0.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_a, out_desc, [BH, ROPE], [row0, 0])
            tl.store(
                tle.gpu.local_ptr(s_kr_a, (tile_rows, tile_cols)),
                out_left_1.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_a, out_desc, [BH, ROPE], [row0, ROPE])
            tl.store(
                tle.gpu.local_ptr(s_kr_a, (tile_rows, tile_cols)),
                out_left_2.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_a, out_desc, [BH, ROPE], [row0, 2 * ROPE])
            tl.store(
                tle.gpu.local_ptr(s_kr_a, (tile_rows, tile_cols)),
                out_left_3.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_a, out_desc, [BH, ROPE], [row0, 3 * ROPE])
        else:
            tl.store(
                out_ptr + offs_h[:, None] * stride_po_h + offs_d[None, :],
                out_left,
                mask=mask_h[:, None],
            )
        lse_arg = state_l * state_s
        lse_ok = state_valid & (lse_arg > 0.0)
        lse2_value = tl.where(
            lse_ok,
            state_m + tl.log(tl.where(lse_arg > 0.0, lse_arg, 1.0)) * _TLE_LOG2E,
            _TLE_NEG_INF,
        )
        tl.store(
            lse2_ptr + offs_h * stride_pl_h,
            lse2_value * _TLE_LN2 if DIRECT_LSE else lse2_value,
            mask=mask_h,
        )

    @triton.jit
    def _fp8_mla_wg1(
        k_desc,
        kr_desc,
        ks_desc,
        out_desc,
        block_table,
        stride_bt_pg,
        row0,
        num_pages,
        q_ckv_full,
        q_rope_full,
        q_scale_full,
        k_content_full,
        k_rope_full,
        k_scale_full,
        state0_ready,
        state1_ready,
        p0_ready,
        p1_ready,
        v0_ready,
        v1_ready,
        slot0_empty,
        slot1_empty,
        tail1_zero_ready,
        s_q,
        s_qr,
        s_kc_a0,
        s_kc_a1,
        s_kc_a2,
        s_kc_a3,
        s_kr_a,
        s_kc_b0,
        s_kc_b1,
        s_kc_b2,
        s_kc_b3,
        s_kr_b,
        s_vt0_b,
        s_vt1_b,
        s_vt1_a,
        s_p_a,
        s_p_b,
        s_beta_a,
        s_beta_b,
        s_state0_m,
        s_state0_s,
        s_state0_l,
        s_state0_valid,
        s_state1_m,
        s_state1_s,
        s_state1_l,
        s_state1_valid,
        split_cache_seqlen,
        out_ptr,
        stride_po_h,
        h_base,
        softmax_scale,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        DP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        PAGE_GRAIN_TAIL_ZERO: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
    ):
        """WG1: odd-page math and the right output half."""
        s_state1_m_row = s_state1_m.slot(0)
        s_beta_a_row = s_beta_a.slot(0)
        s_beta_b_row = s_beta_b.slot(0)
        tle.gpu.barrier_wait(q_ckv_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_rope_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_scale_full, phaseIdx=0)

        offs_t = tl.arange(0, BK)
        offs_h = h_base + tl.arange(0, BH)
        mask_h = offs_h < HQ
        state_idx = tl.arange(0, BH)
        qs = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), volatile=True)

        acc_right = tl.zeros((BH, DP), dtype=tl.float32)
        state_m = tl.full((BH,), float("-inf"), tl.float32)
        state_s = tl.full((BH,), 1.0, tl.float32)
        state_l = tl.zeros((BH,), dtype=tl.float32)
        state_valid = tl.zeros((BH,), dtype=tl.int32) != 0

        q_rows_d128 = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, K_CONTENT_TILE))
        q_c0_cols = tl.broadcast_to(
            tl.arange(0, K_CONTENT_TILE)[None, :], (BH, K_CONTENT_TILE)
        )
        q_c1_cols = tl.broadcast_to(
            (K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c2_cols = tl.broadcast_to(
            (2 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c3_cols = tl.broadcast_to(
            (3 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c0 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c0_cols)))
        q_c1 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c1_cols)))
        q_c2 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c2_cols)))
        q_c3 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c3_cols)))
        k_a_c2 = s_kc_a2
        k_a_c3 = s_kc_a3
        k_b_c0 = s_kc_b0
        k_b_c1 = s_kc_b1
        k_b_c2 = s_kc_b2
        k_b_c3 = s_kc_b3
        prow = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, BK))
        pcol = tl.broadcast_to(tl.arange(0, BK)[None, :], (BH, BK))
        kv_rows_d128 = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DP // 2))
        kv_c0_cols = tl.broadcast_to(tl.arange(0, DP // 2)[None, :], (BK, DP // 2))
        kv_c1_cols = tl.broadcast_to(
            (DP // 2 + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        kv_c2_cols = tl.broadcast_to(
            (DP + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        kv_c3_cols = tl.broadcast_to(
            (DP + DP // 2 + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        vt_c0_rows = tl.broadcast_to(tl.arange(0, DP // 2)[:, None], (DP // 2, BK))
        vt_c1_rows = tl.broadcast_to(
            (DP // 2 + tl.arange(0, DP // 2))[:, None], (DP // 2, BK)
        )
        vt_cols_d128 = tl.broadcast_to(tl.arange(0, BK)[None, :], (DP // 2, BK))

        num_pairs = (num_pages + 1) // 2
        # WG1 completes generation zero for both slots. The writer groups use
        # disjoint slices and independent completion barriers.
        if num_pages > 0:
            first_phys = tl.load(block_table)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_a_c2,
                [BK, K_CONTENT_TILE],
                [first_base, 2 * K_CONTENT_TILE],
                barrier=k_content_full[2],
            )
            tle.gpu.copy(
                k_desc,
                k_a_c3,
                [BK, K_CONTENT_TILE],
                [first_base, 3 * K_CONTENT_TILE],
                barrier=k_content_full[3],
            )
            tle.gpu.copy(
                kr_desc,
                s_kr_a,
                [BK, ROPE],
                [first_base, 0],
                barrier=k_rope_full[0],
            )
            tle.gpu.copy(
                ks_desc,
                s_beta_a,
                [1, BK],
                [first_phys, 0],
                barrier=k_scale_full[0],
            )
        if num_pages > 1:
            first_phys = tl.load(block_table + stride_bt_pg)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_b_c2,
                [BK, K_CONTENT_TILE],
                [first_base, 2 * K_CONTENT_TILE],
                barrier=k_content_full[6],
            )
            tle.gpu.copy(
                k_desc,
                k_b_c3,
                [BK, K_CONTENT_TILE],
                [first_base, 3 * K_CONTENT_TILE],
                barrier=k_content_full[7],
            )
            tle.gpu.copy(
                kr_desc,
                s_kr_b,
                [BK, ROPE],
                [first_base, 0],
                barrier=k_rope_full[1],
            )
            tle.gpu.copy(
                ks_desc,
                s_beta_b,
                [1, BK],
                [first_phys, 0],
                barrier=k_scale_full[1],
            )

        # Cold prime: page 1 QK, scale, and V become loop live-ins. No page-1
        # QK is repeated in pair zero.
        qk = tl.zeros((BH, BK), dtype=tl.float32)
        ks = tl.zeros((BK,), dtype=tl.float32)
        if num_pages > 1:
            tle.gpu.barrier_wait(k_content_full[4], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c0, k_b_c0, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[5], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c1, k_b_c1, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[6], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c2, k_b_c2, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[7], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c3, k_b_c3, qk, trans_b=True)
            tle.gpu.barrier_wait(k_rope_full[1], phaseIdx=0)
            qk = tle.gpu.wgmma(s_qr, s_kr_b, qk, trans_b=True)
            qk = tle.gpu.wgmma_wait(0, qk)

            tle.gpu.barrier_wait(k_scale_full[1], phaseIdx=0)
            prime_valid = PAGE_SIZE + offs_t < split_cache_seqlen
            ks_raw = tl.load(tle.gpu.local_ptr(s_beta_b_row, (offs_t,)))
            ks = tl.where(prime_valid, ks_raw, 0.0)

        full_pairs = tl.maximum(num_pages // 2 - 1, 0)
        for pair in tl.range(full_pairs, disable_licm=True):
            even_page = pair * 2
            odd_page = even_page + 1

            if MERGE_STATE_V:
                # The odd-page V repack reads only
                # this WG's already-waited K content (k_content_full[4..7]
                # retired by the QK chain that produced the resident qk) and
                # its prior-generation readers retired through slot1_empty
                # (WG0 PV, waited last iteration) and this WG's own
                # wgmma_wait.  It does not depend on WG0's incoming state, P,
                # or V, so issue it before the merged completion wait and
                # remove it from the wait->v1_ready critical path.  The
                # v1_ready arrive below still follows every one of these
                # shared writes in program order.
                _cuda_vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, FULL_TAIL)

                # The merged completion is intentionally later than the old
                # state-only publication.  Hide part of that wait with the
                # page-local score work, which depends only on the resident
                # QK accumulator and scales, not on WG0's incoming state.
                if FULL_TAIL:
                    valid = tl.full((BK,), True, tl.int1)
                else:
                    valid = odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                valid_row = valid[None, :]
                score = qk * qs[:, None] * ks[None, :] * softmax_scale
                score_safe = score if FULL_TAIL else tl.where(valid_row, score, 0.0)
                x = score_safe * _TLE_LOG2E
                page_m = tl.max(
                    x if FULL_TAIL else tl.where(valid_row, x, _TLE_NEG_INF),
                    axis=1,
                )
                tle.gpu.barrier_wait(v0_ready, phaseIdx=pair)
            else:
                tle.gpu.barrier_wait(state0_ready, phaseIdx=pair)
            state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
            state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
            state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
            state_valid = tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0

            beta1 = tl.full((BH,), 1.0, tl.float32)
            if True:
                # Preserve the same schedule for every non-merged specialization.  MERGE_STATE_V is constexpr, so only one
                # copy of this page-local chain survives lowering.
                if not MERGE_STATE_V:
                    if FULL_TAIL:
                        valid = tl.full((BK,), True, tl.int1)
                    else:
                        valid = odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                    valid_row = valid[None, :]
                    score = qk * qs[:, None] * ks[None, :] * softmax_scale
                    score_safe = score if FULL_TAIL else tl.where(valid_row, score, 0.0)
                    x = score_safe * _TLE_LOG2E
                    page_m = tl.max(
                        x if FULL_TAIL else tl.where(valid_row, x, _TLE_NEG_INF),
                        axis=1,
                    )
                old_m = tl.where(state_valid, state_m, _TLE_NEG_INF)
                old_s = tl.where(state_valid, state_s, 1.0)
                old_l = tl.where(state_valid, state_l, 0.0)
                m_new = tl.maximum(old_m, page_m)
                m_safe = tl.where(m_new == _TLE_NEG_INF, 0.0, m_new)
                e = (
                    tl.exp2(x - m_safe[:, None])
                    if FULL_TAIL
                    else tl.where(valid_row, tl.exp2(x - m_safe[:, None]), 0.0)
                )
                f = e * ks[None, :]
                amax = tl.max(tl.abs(f), axis=1)
                s_new = tl.where(
                    amax == 0.0,
                    1.0,
                    tl.maximum(amax, _TLE_P_AMAX_FLOOR) / _TLE_FP8_MAX,
                )
                page_valid = (
                    True if FULL_TAIL else odd_page * PAGE_SIZE < split_cache_seqlen
                )
                if USE_HOTLOOP_RECIP:
                    inv_s_new = 1.0 / s_new
                    p_scaled = f * inv_s_new[:, None]
                else:
                    p_scaled = f / s_new[:, None]
                p_new = tl.clamp(p_scaled, -_TLE_FP8_MAX, _TLE_FP8_MAX)
                p1 = (
                    p_new
                    if FULL_TAIL
                    else tl.where(page_valid, p_new, tl.zeros_like(p_new))
                )
                if FULL_TAIL:
                    _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p_b, p1)
                else:
                    p1_store = p_new.to(tl.float8e4nv)
                    p1_store = tl.where(page_valid, p1_store, tl.zeros_like(p1_store))
                    tl.store(tle.gpu.local_ptr(s_p_b, (prow, pcol)), p1_store)
                old_m_finite = tl.where(state_valid, old_m, 0.0)
                alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
                if USE_HOTLOOP_RECIP:
                    beta1 = alpha * old_s * inv_s_new
                    l_new = old_l * beta1 + tl.sum(e, axis=1) * inv_s_new
                else:
                    beta1 = alpha * old_s / s_new
                    l_new = old_l * beta1 + tl.sum(e, axis=1) / s_new
                state_m = tl.where(page_valid, m_new, old_m)
                state_s = tl.where(page_valid, s_new, old_s)
                state_l = tl.where(page_valid, l_new, old_l)
                beta1 = tl.where(page_valid, beta1, 1.0)
                state_valid = state_valid | page_valid

                tl.store(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)), beta1)
                tl.store(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), state_m)
                tl.store(tle.gpu.local_ptr(s_state1_s, (state_idx,)), state_s)
                tl.store(tle.gpu.local_ptr(s_state1_l, (state_idx,)), state_l)
                tl.store(
                    tle.gpu.local_ptr(s_state1_valid, (state_idx,)),
                    state_valid.to(tl.int32),
                )

                # Publish WG1 state before V repack/PV/next-QK, matching the
                # CUDA scale/state hand-off rather than delaying the consumer
                # behind unrelated work.
                if not MERGE_STATE_V:
                    tle.gpu.barrier_arrive(state1_ready, phaseIdx=pair)

                # full_pairs excludes the residual/tail pair.  The steady odd
                # page is therefore complete and can use CUDA's single
                # LDSM/PRMT/STSM path without a runtime fallback branch.
                # The merged-state specialization moves this repack before its
                # completion wait; all other specializations keep it here.
                if not MERGE_STATE_V:
                    _cuda_vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, FULL_TAIL)

                tle.gpu.barrier_arrive(v1_ready, phaseIdx=pair)

            # CUDA remote-P wait point for the current even page.
            if not MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready, phaseIdx=pair)
            beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
            acc_right *= beta0[:, None]
            acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
            acc_right = tle.gpu.wgmma_wait(0, acc_right)

            # After remote-P wait0, issue p+2 content2/3/rope/scale.
            next_even_page = even_page + 2
            if True:
                next_even_phys = tl.load(block_table + next_even_page * stride_bt_pg)
                next_even_base = (next_even_phys * BK).to(tl.int32)
                tle.gpu.copy(
                    k_desc,
                    k_a_c2,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 2 * K_CONTENT_TILE],
                    barrier=k_content_full[2],
                )
                tle.gpu.copy(
                    k_desc,
                    k_a_c3,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 3 * K_CONTENT_TILE],
                    barrier=k_content_full[3],
                )
                tle.gpu.copy(
                    kr_desc,
                    s_kr_a,
                    [BK, ROPE],
                    [next_even_base, 0],
                    barrier=k_rope_full[0],
                )
                tle.gpu.copy(
                    ks_desc,
                    s_beta_a,
                    [1, BK],
                    [next_even_phys, 0],
                    barrier=k_scale_full[0],
                )
            tle.gpu.barrier_arrive(slot0_empty, phaseIdx=pair)

            next_qk = tl.zeros((BH, BK), dtype=tl.float32)
            next_ks = tl.zeros((BK,), dtype=tl.float32)
            if True:
                # CUDA local-P PV and wait0 precede p+3 upper transactions.
                acc_right *= beta1[:, None]
                acc_right = tle.gpu.wgmma(s_p_b, s_vt1_b, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)

                next_odd_page = odd_page + 2
                next_generation = pair + 1
                if True:
                    next_odd_phys = tl.load(block_table + next_odd_page * stride_bt_pg)
                    next_odd_base = (next_odd_phys * BK).to(tl.int32)
                    tle.gpu.copy(
                        k_desc,
                        k_b_c2,
                        [BK, K_CONTENT_TILE],
                        [next_odd_base, 2 * K_CONTENT_TILE],
                        barrier=k_content_full[6],
                    )
                    tle.gpu.copy(
                        k_desc,
                        k_b_c3,
                        [BK, K_CONTENT_TILE],
                        [next_odd_base, 3 * K_CONTENT_TILE],
                        barrier=k_content_full[7],
                    )
                    tle.gpu.copy(
                        kr_desc,
                        s_kr_b,
                        [BK, ROPE],
                        [next_odd_base, 0],
                        barrier=k_rope_full[1],
                    )

                    # CUDA QK phase-1 completes p+3 in this pair.
                    tle.gpu.barrier_wait(k_content_full[4], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c0, k_b_c0, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[5], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c1, k_b_c1, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[6], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c2, k_b_c2, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[7], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c3, k_b_c3, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_rope_full[1], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(s_qr, s_kr_b, next_qk, trans_b=True)
                    next_qk = tle.gpu.wgmma_wait(0, next_qk)

                tle.gpu.barrier_wait(slot1_empty, phaseIdx=pair)

                if True:
                    # Keep the scale copy after slot release to preserve its storage lifetime.
                    next_scale_phys = tl.load(
                        block_table + next_odd_page * stride_bt_pg
                    )
                    tle.gpu.copy(
                        ks_desc,
                        s_beta_b,
                        [1, BK],
                        [next_scale_phys, 0],
                        barrier=k_scale_full[1],
                    )
                    tle.gpu.barrier_wait(k_scale_full[1], phaseIdx=next_generation)
                    next_valid = next_odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                    next_ks_raw = tl.load(tle.gpu.local_ptr(s_beta_b_row, (offs_t,)))
                    next_ks = tl.where(next_valid, next_ks_raw, 0.0)

            qk = next_qk
            ks = next_ks

        # CUDA-style WG1 epilogue. The first residual pair is either the last
        # full pair or the 3-page transition; an odd transition has one final
        # even-only pair after it.
        if num_pages > 0:
            pair = full_pairs
            even_page = pair * 2
            odd_page = even_page + 1

            if MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready, phaseIdx=pair)
            else:
                tle.gpu.barrier_wait(state0_ready, phaseIdx=pair)
            state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
            state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
            state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
            state_valid = tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0

            beta1 = tl.full((BH,), 1.0, tl.float32)
            if odd_page < num_pages:
                valid = odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                valid_row = valid[None, :]
                tail_ks_raw = tl.load(tle.gpu.local_ptr(s_beta_b_row, (offs_t,)))
                tail_ks = (
                    tail_ks_raw if FULL_TAIL else tl.where(valid, tail_ks_raw, 0.0)
                )
                score = qk * qs[:, None] * tail_ks[None, :] * softmax_scale
                score_safe = score if FULL_TAIL else tl.where(valid_row, score, 0.0)
                x = score_safe * _TLE_LOG2E
                page_m = tl.max(
                    x if FULL_TAIL else tl.where(valid_row, x, _TLE_NEG_INF), axis=1
                )
                old_m = tl.where(state_valid, state_m, _TLE_NEG_INF)
                old_s = tl.where(state_valid, state_s, 1.0)
                old_l = tl.where(state_valid, state_l, 0.0)
                m_new = tl.maximum(old_m, page_m)
                m_safe = tl.where(m_new == _TLE_NEG_INF, 0.0, m_new)
                e = (
                    tl.exp2(x - m_safe[:, None])
                    if FULL_TAIL
                    else tl.where(valid_row, tl.exp2(x - m_safe[:, None]), 0.0)
                )
                f = e * tail_ks[None, :]
                amax = tl.max(tl.abs(f), axis=1)
                s_new = tl.where(
                    amax == 0.0,
                    1.0,
                    tl.maximum(amax, _TLE_P_AMAX_FLOOR) / _TLE_FP8_MAX,
                )
                page_valid = (
                    True if FULL_TAIL else odd_page * PAGE_SIZE < split_cache_seqlen
                )
                inv_s_new = 1.0 / s_new
                p_new = tl.clamp(f * inv_s_new[:, None], -_TLE_FP8_MAX, _TLE_FP8_MAX)
                p1 = (
                    p_new
                    if FULL_TAIL
                    else tl.where(page_valid, p_new, tl.zeros_like(p_new))
                )
                if FULL_TAIL:
                    _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p_b, p1)
                else:
                    p1_store = p_new.to(tl.float8e4nv)
                    p1_store = tl.where(page_valid, p1_store, tl.zeros_like(p1_store))
                    tl.store(tle.gpu.local_ptr(s_p_b, (prow, pcol)), p1_store)
                old_m_finite = tl.where(state_valid, old_m, 0.0)
                alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
                beta1 = alpha * old_s * inv_s_new
                l_new = old_l * beta1 + tl.sum(e, axis=1) * inv_s_new
                state_m = tl.where(page_valid, m_new, old_m)
                state_s = tl.where(page_valid, s_new, old_s)
                state_l = tl.where(page_valid, l_new, old_l)
                beta1 = tl.where(page_valid, beta1, 1.0)
                state_valid = state_valid | page_valid

                tl.store(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)), beta1)
                tl.store(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), state_m)
                tl.store(tle.gpu.local_ptr(s_state1_s, (state_idx,)), state_s)
                tl.store(tle.gpu.local_ptr(s_state1_l, (state_idx,)), state_l)
                tl.store(
                    tle.gpu.local_ptr(s_state1_valid, (state_idx,)),
                    state_valid.to(tl.int32),
                )

                # Tail generation follows the same last-write publication rule.
                if not MERGE_STATE_V:
                    tle.gpu.barrier_arrive(state1_ready, phaseIdx=pair)

                # As on the even-page owner, invalid P columns are exact zero,
                # so a masked V transpose is unnecessary for PV correctness.
                if PAGE_GRAIN_TAIL_ZERO:
                    if not FULL_TAIL:
                        valid_tokens = tl.minimum(
                            split_cache_seqlen - odd_page * PAGE_SIZE,
                            BK,
                        )
                        valid_tokens = tl.maximum(valid_tokens, 0)
                        if valid_tokens < BK:
                            _zero_invalid_fp8_rows_sw128_x4(
                                s_kc_b0,
                                s_kc_b1,
                                s_kc_b2,
                                s_kc_b3,
                                valid_tokens,
                            )
                            tle.gpu.barrier_arrive(tail1_zero_ready, phaseIdx=pair)
                            tle.gpu.barrier_wait(tail1_zero_ready, phaseIdx=pair)
                    _cuda_vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, FULL_TAIL)
                elif FULL_TAIL or (odd_page + 1) * PAGE_SIZE <= split_cache_seqlen:
                    _cuda_vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, FULL_TAIL)
                else:
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b0, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt0_b, (vt_c0_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b1, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt0_b, (vt_c1_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b2, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt1_b, (vt_c0_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b3, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt1_b, (vt_c1_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                tle.gpu.barrier_arrive(v1_ready, phaseIdx=pair)

            if not MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready, phaseIdx=pair)
            beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
            acc_right *= beta0[:, None]
            acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
            acc_right = tle.gpu.wgmma_wait(0, acc_right)

            next_even_page = even_page + 2
            if next_even_page < num_pages:
                next_even_phys = tl.load(block_table + next_even_page * stride_bt_pg)
                next_even_base = (next_even_phys * BK).to(tl.int32)
                tle.gpu.copy(
                    k_desc,
                    k_a_c2,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 2 * K_CONTENT_TILE],
                    barrier=k_content_full[2],
                )
                tle.gpu.copy(
                    k_desc,
                    k_a_c3,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 3 * K_CONTENT_TILE],
                    barrier=k_content_full[3],
                )
                tle.gpu.copy(
                    kr_desc,
                    s_kr_a,
                    [BK, ROPE],
                    [next_even_base, 0],
                    barrier=k_rope_full[0],
                )
                tle.gpu.copy(
                    ks_desc,
                    s_beta_a,
                    [1, BK],
                    [next_even_phys, 0],
                    barrier=k_scale_full[0],
                )
            tle.gpu.barrier_arrive(slot0_empty, phaseIdx=pair)

            if odd_page < num_pages:
                acc_right *= beta1[:, None]
                acc_right = tle.gpu.wgmma(s_p_b, s_vt1_b, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)
                tle.gpu.barrier_wait(slot1_empty, phaseIdx=pair)

            if next_even_page < num_pages:
                final_pair = pair + 1
                if MERGE_STATE_V:
                    tle.gpu.barrier_wait(v0_ready, phaseIdx=final_pair)
                else:
                    tle.gpu.barrier_wait(state0_ready, phaseIdx=final_pair)
                state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
                state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
                state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
                state_valid = (
                    tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0
                )
                if not MERGE_STATE_V:
                    tle.gpu.barrier_wait(v0_ready, phaseIdx=final_pair)
                beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
                acc_right *= beta0[:, None]
                acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)
                tle.gpu.barrier_arrive(slot0_empty, phaseIdx=final_pair)

        offs_d = tl.arange(0, DP)
        l_div = tl.where(state_l > 0.0, state_l, 1.0)
        inv_l_div = 1.0 / l_div
        out_right = tl.where(state_valid[:, None], acc_right * inv_l_div[:, None], 0.0)
        if USE_TMA_OUTPUT:
            out_right_lo, out_right_hi = tl.split(
                tl.permute(tl.reshape(out_right, (BH, 2, DP // 2)), (0, 2, 1))
            )
            out_right_0, out_right_1 = tl.split(
                tl.permute(tl.reshape(out_right_lo, (BH, 2, ROPE)), (0, 2, 1))
            )
            out_right_2, out_right_3 = tl.split(
                tl.permute(tl.reshape(out_right_hi, (BH, 2, ROPE)), (0, 2, 1))
            )
            tile_rows = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, ROPE))
            tile_cols = tl.broadcast_to(tl.arange(0, ROPE)[None, :], (BH, ROPE))
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_0.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP])
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_1.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + ROPE])
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_2.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + 2 * ROPE])
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_3.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + 3 * ROPE])
        else:
            tl.store(
                out_ptr + offs_h[:, None] * stride_po_h + DP + offs_d[None, :],
                out_right,
                mask=mask_h[:, None],
            )

    @triton.jit
    def _fp8_dense_mla_splitk_partial(
        qc_ptr,
        qr_ptr,
        qs_ptr,
        kc_ptr,
        kr_ptr,
        ks_ptr,
        block_table,
        cache_seqlens,
        split_batch_ptr,
        split_page_begin_ptr,
        split_num_pages_ptr,
        partial_out_ptr,
        partial_lse2_ptr,
        q_desc,
        qr_desc,
        qs_desc,
        out_desc,
        k_desc,
        kr_desc,
        ks_desc,
        stride_qc_b: tl.constexpr,
        stride_qc_h: tl.constexpr,
        stride_qr_b: tl.constexpr,
        stride_qr_h: tl.constexpr,
        stride_qs_b: tl.constexpr,
        stride_qs_h: tl.constexpr,
        stride_kc_blk: tl.constexpr,
        stride_kc_pg: tl.constexpr,
        stride_kr_blk: tl.constexpr,
        stride_kr_pg: tl.constexpr,
        stride_ks_blk: tl.constexpr,
        stride_ks_pg: tl.constexpr,
        stride_bt_b: tl.constexpr,
        stride_bt_pg: tl.constexpr,
        stride_seqlen: tl.constexpr,
        stride_split_batch: tl.constexpr,
        stride_split_begin: tl.constexpr,
        stride_split_num_pages: tl.constexpr,
        stride_po_split: tl.constexpr,
        stride_po_h: tl.constexpr,
        stride_pl_split: tl.constexpr,
        stride_pl_h: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_CKV_BYTES: tl.constexpr,
        Q_ROPE_BYTES: tl.constexpr,
        Q_SCALE_BYTES: tl.constexpr,
        K_CONTENT_TILE_BYTES: tl.constexpr,
        K_ROPE_BYTES: tl.constexpr,
        K_SCALE_BYTES: tl.constexpr,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        RH: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        DP: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        PAGE_GRAIN_TAIL_ZERO: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
        FIXED_NUM_PAGES: tl.constexpr,
        DIRECT_LSE: tl.constexpr,
    ):
        """One strict-2WG CTA per (split, head block)."""
        pid = tl.program_id(0)
        global_split = pid // RH
        h_base = (pid % RH) * BH
        global_split64 = global_split.to(tl.int64)
        batch_idx = tl.load(split_batch_ptr + global_split64 * stride_split_batch)
        batch_idx64 = batch_idx.to(tl.int64)
        page_begin = tl.load(split_page_begin_ptr + global_split64 * stride_split_begin)
        split_num_pages_runtime = tl.load(
            split_num_pages_ptr + global_split64 * stride_split_num_pages
        )
        split_num_pages = (
            FIXED_NUM_PAGES if FIXED_NUM_PAGES > 0 else split_num_pages_runtime
        )
        page_end = page_begin + split_num_pages
        full_cache_seqlen = tl.load(cache_seqlens + batch_idx64 * stride_seqlen)
        token_begin = page_begin * PAGE_SIZE
        token_end = tl.minimum(page_end * PAGE_SIZE, full_cache_seqlen)
        split_cache_seqlen = tl.maximum(token_end - token_begin, 0)

        block_table_ptr = (
            block_table
            + batch_idx64 * stride_bt_b
            + page_begin.to(tl.int64) * stride_bt_pg
        )
        out_split_ptr = partial_out_ptr + global_split64 * stride_po_split
        lse2_split_ptr = partial_lse2_ptr + global_split64 * stride_pl_split

        s_q = tle.gpu.alloc([BH, CKV], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_qr = tle.gpu.alloc([BH, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)

        s_kc_a0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_a = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_kc_b0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_b = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_p_a = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_p_b = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_beta_a = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_beta_b = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)

        s_state0_m = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)
        s_state1_m = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)

        # One TMA copy == one completion-barrier generation.
        q_ckv_full = tle.gpu.alloc_barrier(expect_bytes=Q_CKV_BYTES)
        q_rope_full = tle.gpu.alloc_barrier(expect_bytes=Q_ROPE_BYTES)
        q_scale_full = tle.gpu.alloc_barrier(expect_bytes=Q_SCALE_BYTES)
        k_content_full = tle.gpu.alloc_barriers(8, expect_bytes=K_CONTENT_TILE_BYTES)
        k_rope_full = tle.gpu.alloc_barriers(2, expect_bytes=K_ROPE_BYTES)
        k_scale_full = tle.gpu.alloc_barriers(2, expect_bytes=K_SCALE_BYTES)

        # Cross-WG control uses all eight logical phaseful mbarrier slots in
        # one power-of-two eight-slot allocation.  Each
        # slot has one unique writer WG, so one elected arrival completes each
        # pair generation; the opposing WG waits on the same pair phase.  This
        # keeps the storage live across warp_specialize and avoids aliasing it
        # with the capture mailbox used to enter the worker partition.
        control_barriers = tle.gpu.alloc_barriers(num_barriers=8, arrive_count=1)
        state0_ready = control_barriers[0]
        state1_ready = control_barriers[1]
        # P is stored before V repack.  Publishing v*_ready after the repack
        # therefore certifies both P and V visibility to the remote PV owner.
        p0_ready = control_barriers[2]
        p1_ready = control_barriers[3]
        v0_ready = control_barriers[2]
        v1_ready = control_barriers[3]
        slot0_empty = control_barriers[4]
        slot1_empty = control_barriers[5]

        # CUDA fill_oob_V publishes shared zeros before the LDSM transpose.
        # Reuse one previously idle control mbarrier per compute warp-group;
        # each has one fixed elected writer and a private pair generation.
        tail0_zero_ready = control_barriers[6]
        tail1_zero_ready = control_barriers[7]

        row0 = (batch_idx * HQ + h_base).to(tl.int32)

        tle.gpu.warp_specialize(
            [
                (
                    _fp8_mla_wg0,
                    (
                        q_desc,
                        qr_desc,
                        qs_desc,
                        out_desc,
                        k_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        tail0_zero_ready,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_a,
                        s_vt0_a,
                        s_vt1_a,
                        s_vt0_b,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        out_split_ptr,
                        lse2_split_ptr,
                        stride_po_h,
                        stride_pl_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        PAGE_GRAIN_TAIL_ZERO,
                        MERGE_STATE_V,
                        False,
                        USE_TMA_OUTPUT,
                        DIRECT_LSE,
                    ),
                ),
                (
                    _fp8_mla_wg1,
                    (
                        k_desc,
                        kr_desc,
                        ks_desc,
                        out_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        tail1_zero_ready,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kr_a,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_b,
                        s_vt0_b,
                        s_vt1_b,
                        s_vt1_a,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        out_split_ptr,
                        stride_po_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        PAGE_GRAIN_TAIL_ZERO,
                        MERGE_STATE_V,
                        USE_TMA_OUTPUT,
                    ),
                ),
            ],
            [4],
            [255],
        )

    @triton.jit
    def _fp8_dense_mla_splitk_partial_pdl(
        qc_ptr,
        qr_ptr,
        qs_ptr,
        kc_ptr,
        kr_ptr,
        ks_ptr,
        block_table,
        cache_seqlens,
        split_batch_ptr,
        split_page_begin_ptr,
        split_num_pages_ptr,
        partial_out_ptr,
        partial_lse2_ptr,
        q_desc,
        qr_desc,
        qs_desc,
        out_desc,
        k_desc,
        kr_desc,
        ks_desc,
        stride_qc_b: tl.constexpr,
        stride_qc_h: tl.constexpr,
        stride_qr_b: tl.constexpr,
        stride_qr_h: tl.constexpr,
        stride_qs_b: tl.constexpr,
        stride_qs_h: tl.constexpr,
        stride_kc_blk: tl.constexpr,
        stride_kc_pg: tl.constexpr,
        stride_kr_blk: tl.constexpr,
        stride_kr_pg: tl.constexpr,
        stride_ks_blk: tl.constexpr,
        stride_ks_pg: tl.constexpr,
        stride_bt_b: tl.constexpr,
        stride_bt_pg: tl.constexpr,
        stride_seqlen: tl.constexpr,
        stride_split_batch: tl.constexpr,
        stride_split_begin: tl.constexpr,
        stride_split_num_pages: tl.constexpr,
        stride_po_split: tl.constexpr,
        stride_po_h: tl.constexpr,
        stride_pl_split: tl.constexpr,
        stride_pl_h: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_CKV_BYTES: tl.constexpr,
        Q_ROPE_BYTES: tl.constexpr,
        Q_SCALE_BYTES: tl.constexpr,
        K_CONTENT_TILE_BYTES: tl.constexpr,
        K_ROPE_BYTES: tl.constexpr,
        K_SCALE_BYTES: tl.constexpr,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        RH: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        DP: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        PAGE_GRAIN_TAIL_ZERO: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
        FIXED_NUM_PAGES: tl.constexpr,
        DIRECT_LSE: tl.constexpr,
    ):
        """One strict-2WG CTA per (split, head block)."""
        pid = tl.program_id(0)
        global_split = pid // RH
        h_base = (pid % RH) * BH
        global_split64 = global_split.to(tl.int64)
        batch_idx = tl.load(split_batch_ptr + global_split64 * stride_split_batch)
        batch_idx64 = batch_idx.to(tl.int64)
        page_begin = tl.load(split_page_begin_ptr + global_split64 * stride_split_begin)
        split_num_pages_runtime = tl.load(
            split_num_pages_ptr + global_split64 * stride_split_num_pages
        )
        split_num_pages = (
            FIXED_NUM_PAGES if FIXED_NUM_PAGES > 0 else split_num_pages_runtime
        )
        page_end = page_begin + split_num_pages
        full_cache_seqlen = tl.load(cache_seqlens + batch_idx64 * stride_seqlen)
        token_begin = page_begin * PAGE_SIZE
        token_end = tl.minimum(page_end * PAGE_SIZE, full_cache_seqlen)
        split_cache_seqlen = tl.maximum(token_end - token_begin, 0)

        block_table_ptr = (
            block_table
            + batch_idx64 * stride_bt_b
            + page_begin.to(tl.int64) * stride_bt_pg
        )
        out_split_ptr = partial_out_ptr + global_split64 * stride_po_split
        lse2_split_ptr = partial_lse2_ptr + global_split64 * stride_pl_split

        s_q = tle.gpu.alloc([BH, CKV], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_qr = tle.gpu.alloc([BH, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)

        s_kc_a0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_a = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_kc_b0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_b = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_p_a = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_p_b = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_beta_a = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_beta_b = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)

        s_state0_m = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)
        s_state1_m = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)

        # One TMA copy == one completion-barrier generation.
        q_ckv_full = tle.gpu.alloc_barrier(expect_bytes=Q_CKV_BYTES)
        q_rope_full = tle.gpu.alloc_barrier(expect_bytes=Q_ROPE_BYTES)
        q_scale_full = tle.gpu.alloc_barrier(expect_bytes=Q_SCALE_BYTES)
        k_content_full = tle.gpu.alloc_barriers(8, expect_bytes=K_CONTENT_TILE_BYTES)
        k_rope_full = tle.gpu.alloc_barriers(2, expect_bytes=K_ROPE_BYTES)
        k_scale_full = tle.gpu.alloc_barriers(2, expect_bytes=K_SCALE_BYTES)

        # Cross-WG control uses all eight logical phaseful mbarrier slots in
        # one power-of-two eight-slot allocation.  Each
        # slot has one unique writer WG, so one elected arrival completes each
        # pair generation; the opposing WG waits on the same pair phase.  This
        # keeps the storage live across warp_specialize and avoids aliasing it
        # with the capture mailbox used to enter the worker partition.
        control_barriers = tle.gpu.alloc_barriers(num_barriers=8, arrive_count=1)
        state0_ready = control_barriers[0]
        state1_ready = control_barriers[1]
        p0_ready = control_barriers[2]
        p1_ready = control_barriers[3]
        v0_ready = control_barriers[2]
        v1_ready = control_barriers[3]
        slot0_empty = control_barriers[4]
        slot1_empty = control_barriers[5]

        tail0_zero_ready = control_barriers[6]
        tail1_zero_ready = control_barriers[7]

        row0 = (batch_idx * HQ + h_base).to(tl.int32)

        tle.gpu.warp_specialize(
            [
                (
                    _fp8_mla_wg0,
                    (
                        q_desc,
                        qr_desc,
                        qs_desc,
                        out_desc,
                        k_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        tail0_zero_ready,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_a,
                        s_vt0_a,
                        s_vt1_a,
                        s_vt0_b,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        out_split_ptr,
                        lse2_split_ptr,
                        stride_po_h,
                        stride_pl_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        PAGE_GRAIN_TAIL_ZERO,
                        MERGE_STATE_V,
                        True,
                        USE_TMA_OUTPUT,
                        DIRECT_LSE,
                    ),
                ),
                (
                    _fp8_mla_wg1,
                    (
                        k_desc,
                        kr_desc,
                        ks_desc,
                        out_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        tail1_zero_ready,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kr_a,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_b,
                        s_vt0_b,
                        s_vt1_b,
                        s_vt1_a,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        out_split_ptr,
                        stride_po_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        PAGE_GRAIN_TAIL_ZERO,
                        MERGE_STATE_V,
                        USE_TMA_OUTPUT,
                    ),
                ),
            ],
            [4],
            [255],
        )

    @triton.jit
    def _fp8_mla_wg1_pretranspose(
        k_desc,
        kr_desc,
        ks_desc,
        out_desc,
        block_table,
        stride_bt_pg,
        row0,
        num_pages,
        q_ckv_full,
        q_rope_full,
        q_scale_full,
        k_content_full,
        k_rope_full,
        k_scale_full,
        state0_ready,
        state1_ready,
        p0_ready,
        p1_ready,
        v0_ready,
        v1_ready,
        slot0_empty,
        slot1_empty,
        s_q,
        s_qr,
        s_kc_a0,
        s_kc_a1,
        s_kc_a2,
        s_kc_a3,
        s_kr_a,
        s_kc_b0,
        s_kc_b1,
        s_kc_b2,
        s_kc_b3,
        s_kr_b,
        s_vt0_b,
        s_vt1_b,
        s_vt1_a,
        s_p_a,
        s_p_b,
        s_beta_a,
        s_beta_b,
        s_state0_m,
        s_state0_s,
        s_state0_l,
        s_state0_valid,
        s_state1_m,
        s_state1_s,
        s_state1_l,
        s_state1_valid,
        split_cache_seqlen,
        out_ptr,
        stride_po_h,
        h_base,
        softmax_scale,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        DP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
    ):
        """WG1: odd-page math and the right output half."""
        s_state1_m_row = s_state1_m.slot(0)
        s_beta_a_row = s_beta_a.slot(0)
        s_beta_b_row = s_beta_b.slot(0)
        tle.gpu.barrier_wait(q_ckv_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_rope_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_scale_full, phaseIdx=0)

        offs_t = tl.arange(0, BK)
        offs_h = h_base + tl.arange(0, BH)
        mask_h = offs_h < HQ
        state_idx = tl.arange(0, BH)
        qs = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), volatile=True)

        acc_right = tl.zeros((BH, DP), dtype=tl.float32)
        state_m = tl.full((BH,), float("-inf"), tl.float32)
        state_s = tl.full((BH,), 1.0, tl.float32)
        state_l = tl.zeros((BH,), dtype=tl.float32)
        state_valid = tl.zeros((BH,), dtype=tl.int32) != 0

        q_rows_d128 = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, K_CONTENT_TILE))
        q_c0_cols = tl.broadcast_to(
            tl.arange(0, K_CONTENT_TILE)[None, :], (BH, K_CONTENT_TILE)
        )
        q_c1_cols = tl.broadcast_to(
            (K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c2_cols = tl.broadcast_to(
            (2 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c3_cols = tl.broadcast_to(
            (3 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c0 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c0_cols)))
        q_c1 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c1_cols)))
        q_c2 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c2_cols)))
        q_c3 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c3_cols)))
        k_a_c2 = s_kc_a2
        k_a_c3 = s_kc_a3
        k_b_c0 = s_kc_b0
        k_b_c1 = s_kc_b1
        k_b_c2 = s_kc_b2
        k_b_c3 = s_kc_b3
        prow = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, BK))
        pcol = tl.broadcast_to(tl.arange(0, BK)[None, :], (BH, BK))
        kv_rows_d128 = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DP // 2))
        kv_c0_cols = tl.broadcast_to(tl.arange(0, DP // 2)[None, :], (BK, DP // 2))
        kv_c1_cols = tl.broadcast_to(
            (DP // 2 + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        kv_c2_cols = tl.broadcast_to(
            (DP + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        kv_c3_cols = tl.broadcast_to(
            (DP + DP // 2 + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        vt_c0_rows = tl.broadcast_to(tl.arange(0, DP // 2)[:, None], (DP // 2, BK))
        vt_c1_rows = tl.broadcast_to(
            (DP // 2 + tl.arange(0, DP // 2))[:, None], (DP // 2, BK)
        )
        vt_cols_d128 = tl.broadcast_to(tl.arange(0, BK)[None, :], (DP // 2, BK))

        num_pairs = (num_pages + 1) // 2
        # WG1 completes generation zero for both slots. The writer groups use
        # disjoint slices and independent completion barriers.
        if num_pages > 0:
            first_phys = tl.load(block_table)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_a_c2,
                [BK, K_CONTENT_TILE],
                [first_base, 2 * K_CONTENT_TILE],
                barrier=k_content_full[2],
            )
            tle.gpu.copy(
                k_desc,
                k_a_c3,
                [BK, K_CONTENT_TILE],
                [first_base, 3 * K_CONTENT_TILE],
                barrier=k_content_full[3],
            )
            tle.gpu.copy(
                kr_desc,
                s_kr_a,
                [BK, ROPE],
                [first_base, 0],
                barrier=k_rope_full[0],
            )
            tle.gpu.copy(
                ks_desc,
                s_beta_a,
                [1, BK],
                [first_phys, 0],
                barrier=k_scale_full[0],
            )
        if num_pages > 1:
            first_phys = tl.load(block_table + stride_bt_pg)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_b_c2,
                [BK, K_CONTENT_TILE],
                [first_base, 2 * K_CONTENT_TILE],
                barrier=k_content_full[6],
            )
            tle.gpu.copy(
                k_desc,
                k_b_c3,
                [BK, K_CONTENT_TILE],
                [first_base, 3 * K_CONTENT_TILE],
                barrier=k_content_full[7],
            )
            tle.gpu.copy(
                kr_desc,
                s_kr_b,
                [BK, ROPE],
                [first_base, 0],
                barrier=k_rope_full[1],
            )
            tle.gpu.copy(
                ks_desc,
                s_beta_b,
                [1, BK],
                [first_phys, 0],
                barrier=k_scale_full[1],
            )

        # Cold prime: page 1 QK, scale, and V become loop live-ins. No page-1
        # QK is repeated in pair zero.
        qk = tl.zeros((BH, BK), dtype=tl.float32)
        ks = tl.zeros((BK,), dtype=tl.float32)
        if num_pages > 1:
            tle.gpu.barrier_wait(k_content_full[4], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c0, k_b_c0, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[5], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c1, k_b_c1, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[6], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c2, k_b_c2, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[7], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c3, k_b_c3, qk, trans_b=True)
            tle.gpu.barrier_wait(k_rope_full[1], phaseIdx=0)
            qk = tle.gpu.wgmma(s_qr, s_kr_b, qk, trans_b=True)
            qk = tle.gpu.wgmma_wait(0, qk)

            tle.gpu.barrier_wait(k_scale_full[1], phaseIdx=0)
            prime_valid = PAGE_SIZE + offs_t < split_cache_seqlen
            ks_raw = tl.load(tle.gpu.local_ptr(s_beta_b_row, (offs_t,)))
            ks = tl.where(prime_valid, ks_raw, 0.0)

        full_pairs = tl.maximum(num_pages // 2 - 1, 0)
        for pair in tl.range(full_pairs, disable_licm=True):
            even_page = pair * 2
            odd_page = even_page + 1

            # V1 is independent of WG0's state payload.  Execute useful
            # transpose work while WG0 completes state0; keep publication after
            # P1 so the v1_ready payload/happens-before edge is unchanged.
            _cuda_vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, FULL_TAIL)
            _cuda_vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, FULL_TAIL)
            _cuda_vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, FULL_TAIL)
            _cuda_vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, FULL_TAIL)

            if MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready, phaseIdx=pair)
            else:
                tle.gpu.barrier_wait(state0_ready, phaseIdx=pair)
            state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
            state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
            state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
            state_valid = tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0

            beta1 = tl.full((BH,), 1.0, tl.float32)
            if True:
                if FULL_TAIL:
                    valid = tl.full((BK,), True, tl.int1)
                else:
                    valid = odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                valid_row = valid[None, :]
                score = qk * qs[:, None] * ks[None, :] * softmax_scale
                score_safe = score if FULL_TAIL else tl.where(valid_row, score, 0.0)
                x = score_safe * _TLE_LOG2E
                page_m = tl.max(
                    x if FULL_TAIL else tl.where(valid_row, x, _TLE_NEG_INF), axis=1
                )
                old_m = tl.where(state_valid, state_m, _TLE_NEG_INF)
                old_s = tl.where(state_valid, state_s, 1.0)
                old_l = tl.where(state_valid, state_l, 0.0)
                m_new = tl.maximum(old_m, page_m)
                m_safe = tl.where(m_new == _TLE_NEG_INF, 0.0, m_new)
                e = (
                    tl.exp2(x - m_safe[:, None])
                    if FULL_TAIL
                    else tl.where(valid_row, tl.exp2(x - m_safe[:, None]), 0.0)
                )
                f = e * ks[None, :]
                amax = tl.max(tl.abs(f), axis=1)
                s_new = tl.where(
                    amax == 0.0,
                    1.0,
                    tl.maximum(amax, _TLE_P_AMAX_FLOOR) / _TLE_FP8_MAX,
                )
                page_valid = (
                    True if FULL_TAIL else odd_page * PAGE_SIZE < split_cache_seqlen
                )
                if USE_HOTLOOP_RECIP:
                    inv_s_new = 1.0 / s_new
                    p_scaled = f * inv_s_new[:, None]
                else:
                    p_scaled = f / s_new[:, None]
                p_new = tl.clamp(p_scaled, -_TLE_FP8_MAX, _TLE_FP8_MAX)
                p1 = (
                    p_new
                    if FULL_TAIL
                    else tl.where(page_valid, p_new, tl.zeros_like(p_new))
                )
                if FULL_TAIL:
                    _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p_b, p1)
                else:
                    p1_store = p_new.to(tl.float8e4nv)
                    p1_store = tl.where(page_valid, p1_store, tl.zeros_like(p1_store))
                    tl.store(tle.gpu.local_ptr(s_p_b, (prow, pcol)), p1_store)
                old_m_finite = tl.where(state_valid, old_m, 0.0)
                alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
                if USE_HOTLOOP_RECIP:
                    beta1 = alpha * old_s * inv_s_new
                    l_new = old_l * beta1 + tl.sum(e, axis=1) * inv_s_new
                else:
                    beta1 = alpha * old_s / s_new
                    l_new = old_l * beta1 + tl.sum(e, axis=1) / s_new
                state_m = tl.where(page_valid, m_new, old_m)
                state_s = tl.where(page_valid, s_new, old_s)
                state_l = tl.where(page_valid, l_new, old_l)
                beta1 = tl.where(page_valid, beta1, 1.0)
                state_valid = state_valid | page_valid

                tl.store(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)), beta1)
                tl.store(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), state_m)
                tl.store(tle.gpu.local_ptr(s_state1_s, (state_idx,)), state_s)
                tl.store(tle.gpu.local_ptr(s_state1_l, (state_idx,)), state_l)
                tl.store(
                    tle.gpu.local_ptr(s_state1_valid, (state_idx,)),
                    state_valid.to(tl.int32),
                )

                # Publish WG1 state before V repack/PV/next-QK, matching the
                # CUDA scale/state hand-off rather than delaying the consumer
                # behind unrelated work.
                if not MERGE_STATE_V:
                    tle.gpu.barrier_arrive(state1_ready, phaseIdx=pair)

                tle.gpu.barrier_arrive(v1_ready, phaseIdx=pair)

            # CUDA remote-P wait point for the current even page.
            if not MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready, phaseIdx=pair)
            beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
            acc_right *= beta0[:, None]
            acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
            acc_right = tle.gpu.wgmma_wait(0, acc_right)

            # After remote-P wait0, issue p+2 content2/3/rope/scale.
            next_even_page = even_page + 2
            if True:
                next_even_phys = tl.load(block_table + next_even_page * stride_bt_pg)
                next_even_base = (next_even_phys * BK).to(tl.int32)
                tle.gpu.copy(
                    k_desc,
                    k_a_c2,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 2 * K_CONTENT_TILE],
                    barrier=k_content_full[2],
                )
                tle.gpu.copy(
                    k_desc,
                    k_a_c3,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 3 * K_CONTENT_TILE],
                    barrier=k_content_full[3],
                )
                tle.gpu.copy(
                    kr_desc,
                    s_kr_a,
                    [BK, ROPE],
                    [next_even_base, 0],
                    barrier=k_rope_full[0],
                )
                tle.gpu.copy(
                    ks_desc,
                    s_beta_a,
                    [1, BK],
                    [next_even_phys, 0],
                    barrier=k_scale_full[0],
                )
            tle.gpu.barrier_arrive(slot0_empty, phaseIdx=pair)

            next_qk = tl.zeros((BH, BK), dtype=tl.float32)
            next_ks = tl.zeros((BK,), dtype=tl.float32)
            if True:
                # CUDA local-P PV and wait0 precede p+3 upper transactions.
                acc_right *= beta1[:, None]
                acc_right = tle.gpu.wgmma(s_p_b, s_vt1_b, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)

                next_odd_page = odd_page + 2
                next_generation = pair + 1
                if True:
                    next_odd_phys = tl.load(block_table + next_odd_page * stride_bt_pg)
                    next_odd_base = (next_odd_phys * BK).to(tl.int32)
                    tle.gpu.copy(
                        k_desc,
                        k_b_c2,
                        [BK, K_CONTENT_TILE],
                        [next_odd_base, 2 * K_CONTENT_TILE],
                        barrier=k_content_full[6],
                    )
                    tle.gpu.copy(
                        k_desc,
                        k_b_c3,
                        [BK, K_CONTENT_TILE],
                        [next_odd_base, 3 * K_CONTENT_TILE],
                        barrier=k_content_full[7],
                    )
                    tle.gpu.copy(
                        kr_desc,
                        s_kr_b,
                        [BK, ROPE],
                        [next_odd_base, 0],
                        barrier=k_rope_full[1],
                    )

                    # CUDA QK phase-1 completes p+3 in this pair.
                    tle.gpu.barrier_wait(k_content_full[4], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c0, k_b_c0, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[5], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c1, k_b_c1, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[6], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c2, k_b_c2, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[7], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c3, k_b_c3, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_rope_full[1], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(s_qr, s_kr_b, next_qk, trans_b=True)
                    next_qk = tle.gpu.wgmma_wait(0, next_qk)

                tle.gpu.barrier_wait(slot1_empty, phaseIdx=pair)

                if True:
                    # Keep the scale copy after slot release to preserve its storage lifetime.
                    next_scale_phys = tl.load(
                        block_table + next_odd_page * stride_bt_pg
                    )
                    tle.gpu.copy(
                        ks_desc,
                        s_beta_b,
                        [1, BK],
                        [next_scale_phys, 0],
                        barrier=k_scale_full[1],
                    )
                    tle.gpu.barrier_wait(k_scale_full[1], phaseIdx=next_generation)
                    next_valid = next_odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                    next_ks_raw = tl.load(tle.gpu.local_ptr(s_beta_b_row, (offs_t,)))
                    next_ks = tl.where(next_valid, next_ks_raw, 0.0)

            qk = next_qk
            ks = next_ks

        # CUDA-style WG1 epilogue. The first residual pair is either the last
        # full pair or the 3-page transition; an odd transition has one final
        # even-only pair after it.
        if num_pages > 0:
            pair = full_pairs
            even_page = pair * 2
            odd_page = even_page + 1

            if MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready, phaseIdx=pair)
            else:
                tle.gpu.barrier_wait(state0_ready, phaseIdx=pair)
            state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
            state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
            state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
            state_valid = tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0

            beta1 = tl.full((BH,), 1.0, tl.float32)
            if odd_page < num_pages:
                valid = odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                valid_row = valid[None, :]
                tail_ks_raw = tl.load(tle.gpu.local_ptr(s_beta_b_row, (offs_t,)))
                tail_ks = (
                    tail_ks_raw if FULL_TAIL else tl.where(valid, tail_ks_raw, 0.0)
                )
                score = qk * qs[:, None] * tail_ks[None, :] * softmax_scale
                score_safe = score if FULL_TAIL else tl.where(valid_row, score, 0.0)
                x = score_safe * _TLE_LOG2E
                page_m = tl.max(
                    x if FULL_TAIL else tl.where(valid_row, x, _TLE_NEG_INF), axis=1
                )
                old_m = tl.where(state_valid, state_m, _TLE_NEG_INF)
                old_s = tl.where(state_valid, state_s, 1.0)
                old_l = tl.where(state_valid, state_l, 0.0)
                m_new = tl.maximum(old_m, page_m)
                m_safe = tl.where(m_new == _TLE_NEG_INF, 0.0, m_new)
                e = (
                    tl.exp2(x - m_safe[:, None])
                    if FULL_TAIL
                    else tl.where(valid_row, tl.exp2(x - m_safe[:, None]), 0.0)
                )
                f = e * tail_ks[None, :]
                amax = tl.max(tl.abs(f), axis=1)
                s_new = tl.where(
                    amax == 0.0,
                    1.0,
                    tl.maximum(amax, _TLE_P_AMAX_FLOOR) / _TLE_FP8_MAX,
                )
                page_valid = (
                    True if FULL_TAIL else odd_page * PAGE_SIZE < split_cache_seqlen
                )
                inv_s_new = 1.0 / s_new
                p_new = tl.clamp(f * inv_s_new[:, None], -_TLE_FP8_MAX, _TLE_FP8_MAX)
                p1 = (
                    p_new
                    if FULL_TAIL
                    else tl.where(page_valid, p_new, tl.zeros_like(p_new))
                )
                if FULL_TAIL:
                    _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p_b, p1)
                else:
                    p1_store = p_new.to(tl.float8e4nv)
                    p1_store = tl.where(page_valid, p1_store, tl.zeros_like(p1_store))
                    tl.store(tle.gpu.local_ptr(s_p_b, (prow, pcol)), p1_store)
                old_m_finite = tl.where(state_valid, old_m, 0.0)
                alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
                beta1 = alpha * old_s * inv_s_new
                l_new = old_l * beta1 + tl.sum(e, axis=1) * inv_s_new
                state_m = tl.where(page_valid, m_new, old_m)
                state_s = tl.where(page_valid, s_new, old_s)
                state_l = tl.where(page_valid, l_new, old_l)
                beta1 = tl.where(page_valid, beta1, 1.0)
                state_valid = state_valid | page_valid

                tl.store(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)), beta1)
                tl.store(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), state_m)
                tl.store(tle.gpu.local_ptr(s_state1_s, (state_idx,)), state_s)
                tl.store(tle.gpu.local_ptr(s_state1_l, (state_idx,)), state_l)
                tl.store(
                    tle.gpu.local_ptr(s_state1_valid, (state_idx,)),
                    state_valid.to(tl.int32),
                )

                # Tail generation follows the same last-write publication rule.
                if not MERGE_STATE_V:
                    tle.gpu.barrier_arrive(state1_ready, phaseIdx=pair)

                if FULL_TAIL or (odd_page + 1) * PAGE_SIZE <= split_cache_seqlen:
                    _cuda_vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, FULL_TAIL)
                else:
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b0, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt0_b, (vt_c0_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b1, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt0_b, (vt_c1_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b2, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt1_b, (vt_c0_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b3, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt1_b, (vt_c1_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                tle.gpu.barrier_arrive(v1_ready, phaseIdx=pair)

            if not MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready, phaseIdx=pair)
            beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
            acc_right *= beta0[:, None]
            acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
            acc_right = tle.gpu.wgmma_wait(0, acc_right)

            next_even_page = even_page + 2
            if next_even_page < num_pages:
                next_even_phys = tl.load(block_table + next_even_page * stride_bt_pg)
                next_even_base = (next_even_phys * BK).to(tl.int32)
                tle.gpu.copy(
                    k_desc,
                    k_a_c2,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 2 * K_CONTENT_TILE],
                    barrier=k_content_full[2],
                )
                tle.gpu.copy(
                    k_desc,
                    k_a_c3,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 3 * K_CONTENT_TILE],
                    barrier=k_content_full[3],
                )
                tle.gpu.copy(
                    kr_desc,
                    s_kr_a,
                    [BK, ROPE],
                    [next_even_base, 0],
                    barrier=k_rope_full[0],
                )
                tle.gpu.copy(
                    ks_desc,
                    s_beta_a,
                    [1, BK],
                    [next_even_phys, 0],
                    barrier=k_scale_full[0],
                )
            tle.gpu.barrier_arrive(slot0_empty, phaseIdx=pair)

            if odd_page < num_pages:
                acc_right *= beta1[:, None]
                acc_right = tle.gpu.wgmma(s_p_b, s_vt1_b, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)
                tle.gpu.barrier_wait(slot1_empty, phaseIdx=pair)

            if next_even_page < num_pages:
                final_pair = pair + 1
                if MERGE_STATE_V:
                    tle.gpu.barrier_wait(v0_ready, phaseIdx=final_pair)
                else:
                    tle.gpu.barrier_wait(state0_ready, phaseIdx=final_pair)
                state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
                state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
                state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
                state_valid = (
                    tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0
                )
                if not MERGE_STATE_V:
                    tle.gpu.barrier_wait(v0_ready, phaseIdx=final_pair)
                beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
                acc_right *= beta0[:, None]
                acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)
                tle.gpu.barrier_arrive(slot0_empty, phaseIdx=final_pair)

        offs_d = tl.arange(0, DP)
        l_div = tl.where(state_l > 0.0, state_l, 1.0)
        inv_l_div = 1.0 / l_div
        out_right = tl.where(state_valid[:, None], acc_right * inv_l_div[:, None], 0.0)
        if USE_TMA_OUTPUT:
            out_right_lo, out_right_hi = tl.split(
                tl.permute(tl.reshape(out_right, (BH, 2, DP // 2)), (0, 2, 1))
            )
            out_right_0, out_right_1 = tl.split(
                tl.permute(tl.reshape(out_right_lo, (BH, 2, ROPE)), (0, 2, 1))
            )
            out_right_2, out_right_3 = tl.split(
                tl.permute(tl.reshape(out_right_hi, (BH, 2, ROPE)), (0, 2, 1))
            )
            tile_rows = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, ROPE))
            tile_cols = tl.broadcast_to(tl.arange(0, ROPE)[None, :], (BH, ROPE))
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_0.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP])
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_1.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + ROPE])
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_2.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + 2 * ROPE])
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_3.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + 3 * ROPE])
        else:
            tl.store(
                out_ptr + offs_h[:, None] * stride_po_h + DP + offs_d[None, :],
                out_right,
                mask=mask_h[:, None],
            )

    @triton.jit
    def _fp8_dense_mla_splitk_partial_pretranspose(
        qc_ptr,
        qr_ptr,
        qs_ptr,
        kc_ptr,
        kr_ptr,
        ks_ptr,
        block_table,
        cache_seqlens,
        split_batch_ptr,
        split_page_begin_ptr,
        split_num_pages_ptr,
        partial_out_ptr,
        partial_lse2_ptr,
        q_desc,
        qr_desc,
        qs_desc,
        out_desc,
        k_desc,
        kr_desc,
        ks_desc,
        stride_qc_b: tl.constexpr,
        stride_qc_h: tl.constexpr,
        stride_qr_b: tl.constexpr,
        stride_qr_h: tl.constexpr,
        stride_qs_b: tl.constexpr,
        stride_qs_h: tl.constexpr,
        stride_kc_blk: tl.constexpr,
        stride_kc_pg: tl.constexpr,
        stride_kr_blk: tl.constexpr,
        stride_kr_pg: tl.constexpr,
        stride_ks_blk: tl.constexpr,
        stride_ks_pg: tl.constexpr,
        stride_bt_b: tl.constexpr,
        stride_bt_pg: tl.constexpr,
        stride_seqlen: tl.constexpr,
        stride_split_batch: tl.constexpr,
        stride_split_begin: tl.constexpr,
        stride_split_num_pages: tl.constexpr,
        stride_po_split: tl.constexpr,
        stride_po_h: tl.constexpr,
        stride_pl_split: tl.constexpr,
        stride_pl_h: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_CKV_BYTES: tl.constexpr,
        Q_ROPE_BYTES: tl.constexpr,
        Q_SCALE_BYTES: tl.constexpr,
        K_CONTENT_TILE_BYTES: tl.constexpr,
        K_ROPE_BYTES: tl.constexpr,
        K_SCALE_BYTES: tl.constexpr,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        RH: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        DP: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        PAGE_GRAIN_TAIL_ZERO: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
        FIXED_NUM_PAGES: tl.constexpr,
        DIRECT_LSE: tl.constexpr,
    ):
        """One strict-2WG CTA per (split, head block)."""
        pid = tl.program_id(0)
        global_split = pid // RH
        h_base = (pid % RH) * BH
        global_split64 = global_split.to(tl.int64)
        batch_idx = tl.load(split_batch_ptr + global_split64 * stride_split_batch)
        batch_idx64 = batch_idx.to(tl.int64)
        page_begin = tl.load(split_page_begin_ptr + global_split64 * stride_split_begin)
        split_num_pages_runtime = tl.load(
            split_num_pages_ptr + global_split64 * stride_split_num_pages
        )
        split_num_pages = (
            FIXED_NUM_PAGES if FIXED_NUM_PAGES > 0 else split_num_pages_runtime
        )
        page_end = page_begin + split_num_pages
        full_cache_seqlen = tl.load(cache_seqlens + batch_idx64 * stride_seqlen)
        token_begin = page_begin * PAGE_SIZE
        token_end = tl.minimum(page_end * PAGE_SIZE, full_cache_seqlen)
        split_cache_seqlen = tl.maximum(token_end - token_begin, 0)

        block_table_ptr = (
            block_table
            + batch_idx64 * stride_bt_b
            + page_begin.to(tl.int64) * stride_bt_pg
        )
        out_split_ptr = partial_out_ptr + global_split64 * stride_po_split
        lse2_split_ptr = partial_lse2_ptr + global_split64 * stride_pl_split

        s_q = tle.gpu.alloc([BH, CKV], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_qr = tle.gpu.alloc([BH, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)

        s_kc_a0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_a = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_kc_b0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_b = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_p_a = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_p_b = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_beta_a = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_beta_b = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)

        s_state0_m = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)
        s_state1_m = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)

        # One TMA copy == one completion-barrier generation.
        q_ckv_full = tle.gpu.alloc_barrier(expect_bytes=Q_CKV_BYTES)
        q_rope_full = tle.gpu.alloc_barrier(expect_bytes=Q_ROPE_BYTES)
        q_scale_full = tle.gpu.alloc_barrier(expect_bytes=Q_SCALE_BYTES)
        k_content_full = tle.gpu.alloc_barriers(8, expect_bytes=K_CONTENT_TILE_BYTES)
        k_rope_full = tle.gpu.alloc_barriers(2, expect_bytes=K_ROPE_BYTES)
        k_scale_full = tle.gpu.alloc_barriers(2, expect_bytes=K_SCALE_BYTES)

        # Cross-WG control uses all eight logical phaseful mbarrier slots in
        # one power-of-two eight-slot allocation.  Each
        # slot has one unique writer WG, so one elected arrival completes each
        # pair generation; the opposing WG waits on the same pair phase.  This
        # keeps the storage live across warp_specialize and avoids aliasing it
        # with the capture mailbox used to enter the worker partition.
        control_barriers = tle.gpu.alloc_barriers(num_barriers=8, arrive_count=1)
        state0_ready = control_barriers[0]
        state1_ready = control_barriers[1]
        # P is stored before V repack.  Publishing v*_ready after the repack
        # therefore certifies both P and V visibility to the remote PV owner.
        p0_ready = control_barriers[2]
        p1_ready = control_barriers[3]
        v0_ready = control_barriers[2]
        v1_ready = control_barriers[3]
        slot0_empty = control_barriers[4]
        slot1_empty = control_barriers[5]

        tail0_zero_ready = control_barriers[6]
        tail1_zero_ready = control_barriers[7]

        row0 = (batch_idx * HQ + h_base).to(tl.int32)

        tle.gpu.warp_specialize(
            [
                (
                    _fp8_mla_wg0,
                    (
                        q_desc,
                        qr_desc,
                        qs_desc,
                        out_desc,
                        k_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        tail0_zero_ready,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_a,
                        s_vt0_a,
                        s_vt1_a,
                        s_vt0_b,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        out_split_ptr,
                        lse2_split_ptr,
                        stride_po_h,
                        stride_pl_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        PAGE_GRAIN_TAIL_ZERO,
                        MERGE_STATE_V,
                        False,
                        USE_TMA_OUTPUT,
                        DIRECT_LSE,
                    ),
                ),
                (
                    _fp8_mla_wg1_pretranspose,
                    (
                        k_desc,
                        kr_desc,
                        ks_desc,
                        out_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kr_a,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_b,
                        s_vt0_b,
                        s_vt1_b,
                        s_vt1_a,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        out_split_ptr,
                        stride_po_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        MERGE_STATE_V,
                        USE_TMA_OUTPUT,
                    ),
                ),
            ],
            [4],
            [255],
        )

    @triton.jit
    def _triton_fp8_splitk_combine_kernel(
        partial_out_ptr,
        partial_lse2_ptr,
        num_splits_ptr,
        out_ptr,
        lse_ptr,
        stride_po_split,
        stride_po_h,
        stride_pl_split,
        stride_pl_h,
        stride_ns,
        stride_out_b,
        stride_out_h,
        stride_lse_b,
        stride_lse_h,
        HQ: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        d_block = tl.program_id(1)
        batch_idx = row // HQ
        head_idx = row % HQ
        split_begin = tl.load(num_splits_ptr + batch_idx * stride_ns)
        split_end = tl.load(num_splits_ptr + (batch_idx + 1) * stride_ns)
        split_count = split_end - split_begin

        split_lanes = tl.arange(0, BLOCK_SPLITS)
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < DV

        max_lse2 = _TLE_NEG_INF
        for split_base in tl.range(0, split_count, BLOCK_SPLITS):
            local_split = split_base + split_lanes
            mask_split = local_split < split_count
            global_split = split_begin + local_split
            local_lse2 = tl.load(
                partial_lse2_ptr
                + global_split * stride_pl_split
                + head_idx * stride_pl_h,
                mask=mask_split,
                other=_TLE_NEG_INF,
            )
            finite_lse = (
                mask_split & (local_lse2 > _TLE_NEG_INF) & (local_lse2 < _TLE_POS_INF)
            )
            local_lse2 = tl.where(finite_lse, local_lse2, _TLE_NEG_INF)
            max_lse2 = tl.maximum(max_lse2, tl.max(local_lse2, axis=0))

        finite_max = max_lse2 != _TLE_NEG_INF
        safe_max = tl.where(finite_max, max_lse2, 0.0)
        denom = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for split_base in tl.range(0, split_count, BLOCK_SPLITS):
            local_split = split_base + split_lanes
            mask_split = local_split < split_count
            global_split = split_begin + local_split
            local_lse2 = tl.load(
                partial_lse2_ptr
                + global_split * stride_pl_split
                + head_idx * stride_pl_h,
                mask=mask_split,
                other=_TLE_NEG_INF,
            )
            finite_lse = (
                mask_split & (local_lse2 > _TLE_NEG_INF) & (local_lse2 < _TLE_POS_INF)
            )
            local_lse2 = tl.where(finite_lse, local_lse2, _TLE_NEG_INF)
            weights = tl.where(
                finite_lse,
                tl.exp2(local_lse2 - safe_max),
                0.0,
            )
            partial = tl.load(
                partial_out_ptr
                + global_split[:, None] * stride_po_split
                + head_idx * stride_po_h
                + offs_d[None, :],
                mask=mask_split[:, None] & mask_d[None, :],
                other=0.0,
            )
            partial = tl.where(finite_lse[:, None], partial, 0.0)
            denom += tl.sum(weights, axis=0)
            acc += tl.sum(partial * weights[:, None], axis=0)

        valid = finite_max & (denom > 0.0)
        safe_denom = tl.where(valid, denom, 1.0)
        result = tl.where(valid, acc / safe_denom, 0.0)
        tl.store(
            out_ptr + batch_idx * stride_out_b + head_idx * stride_out_h + offs_d,
            result,
            mask=mask_d,
        )

        global_lse = tl.where(
            valid,
            (safe_max + tl.log(safe_denom) * _TLE_LOG2E) * _TLE_LN2,
            _TLE_NEG_INF,
        )
        tl.store(
            lse_ptr + batch_idx * stride_lse_b + head_idx * stride_lse_h,
            global_lse,
            mask=d_block == 0,
        )

    @triton.jit
    def _triton_fp8_cuda_coarse_combine_kernel_impl(
        partial_out_ptr,
        partial_lse2_ptr,
        num_splits_ptr,
        out_ptr,
        lse_ptr,
        stride_po_split,
        stride_po_h,
        stride_pl_split,
        stride_pl_h,
        stride_ns,
        stride_out_b,
        stride_out_h,
        stride_lse_b,
        stride_lse_h,
        HQ: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        ENABLE_PDL: tl.constexpr,
    ):
        """CUDA-aligned combine: one warp owns one row, eight rows per CTA."""
        # A PDL consumer may become resident before the partial grid retires.
        # No workspace or split metadata read is legal before this wait.
        if ENABLE_PDL:
            tl.extra.cuda.gdc_wait()

        batch_idx = tl.program_id(0)
        row_block = tl.program_id(1)
        heads = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        mask_h = heads < HQ

        split_begin = tl.load(num_splits_ptr + batch_idx * stride_ns)
        split_end = tl.load(num_splits_ptr + (batch_idx + 1) * stride_ns)
        split_count = split_end - split_begin

        split_lanes = tl.arange(0, BLOCK_SPLITS)
        max_lse2 = tl.full((BLOCK_ROWS,), _TLE_NEG_INF, tl.float32)
        for split_base in tl.range(0, split_count, BLOCK_SPLITS):
            local_splits = split_base + split_lanes
            mask_split = local_splits < split_count
            global_splits = split_begin + local_splits
            local_lse2 = tl.load(
                partial_lse2_ptr
                + global_splits[None, :] * stride_pl_split
                + heads[:, None] * stride_pl_h,
                mask=mask_h[:, None] & mask_split[None, :],
                other=_TLE_NEG_INF,
            )
            finite_lse = (
                mask_h[:, None]
                & mask_split[None, :]
                & (local_lse2 > _TLE_NEG_INF)
                & (local_lse2 < _TLE_POS_INF)
            )
            local_lse2 = tl.where(finite_lse, local_lse2, _TLE_NEG_INF)
            max_lse2 = tl.maximum(max_lse2, tl.max(local_lse2, axis=1))

        finite_max = mask_h & (max_lse2 != _TLE_NEG_INF)
        safe_max = tl.where(finite_max, max_lse2, 0.0)
        denom = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
        offs_d = tl.arange(0, DV)
        acc = tl.zeros((BLOCK_ROWS, DV), dtype=tl.float32)

        # Match CUDA split-order accumulation. One warp owns one logical row
        # and each lane owns DV/32 output columns.
        for local_split in tl.range(0, split_count):
            global_split = split_begin + local_split
            local_lse2 = tl.load(
                partial_lse2_ptr + global_split * stride_pl_split + heads * stride_pl_h,
                mask=mask_h,
                other=_TLE_NEG_INF,
            )
            finite_lse = (
                mask_h & (local_lse2 > _TLE_NEG_INF) & (local_lse2 < _TLE_POS_INF)
            )
            weights = tl.where(
                finite_lse,
                tl.exp2(local_lse2 - safe_max),
                0.0,
            )
            partial = tl.load(
                partial_out_ptr
                + global_split * stride_po_split
                + heads[:, None] * stride_po_h
                + offs_d[None, :],
                mask=mask_h[:, None],
                other=0.0,
            )
            denom += weights
            acc += partial * weights[:, None]

        valid = finite_max & (denom > 0.0)
        safe_denom = tl.where(valid, denom, 1.0)
        result = tl.where(valid[:, None], acc / safe_denom[:, None], 0.0)
        tl.store(
            out_ptr
            + batch_idx * stride_out_b
            + heads[:, None] * stride_out_h
            + offs_d[None, :],
            result,
            mask=mask_h[:, None],
        )

        global_lse = tl.where(
            valid,
            (safe_max + tl.log(safe_denom) * _TLE_LOG2E) * _TLE_LN2,
            _TLE_NEG_INF,
        )
        tl.store(
            lse_ptr + batch_idx * stride_lse_b + heads * stride_lse_h,
            global_lse,
            mask=mask_h,
        )

    @triton.jit
    def _triton_fp8_cuda_coarse_combine_kernel(
        partial_out_ptr,
        partial_lse2_ptr,
        num_splits_ptr,
        out_ptr,
        lse_ptr,
        stride_po_split,
        stride_po_h,
        stride_pl_split,
        stride_pl_h,
        stride_ns,
        stride_out_b,
        stride_out_h,
        stride_lse_b,
        stride_lse_h,
        HQ: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Run the non-PDL coarse combine path."""
        _triton_fp8_cuda_coarse_combine_kernel_impl(
            partial_out_ptr,
            partial_lse2_ptr,
            num_splits_ptr,
            out_ptr,
            lse_ptr,
            stride_po_split,
            stride_po_h,
            stride_pl_split,
            stride_pl_h,
            stride_ns,
            stride_out_b,
            stride_out_h,
            stride_lse_b,
            stride_lse_h,
            HQ,
            DV,
            BLOCK_SPLITS,
            BLOCK_ROWS,
            False,
        )

    @triton.jit
    def _triton_fp8_cuda_coarse_combine_pdl_kernel(
        partial_out_ptr,
        partial_lse2_ptr,
        num_splits_ptr,
        out_ptr,
        lse_ptr,
        stride_po_split,
        stride_po_h,
        stride_pl_split,
        stride_pl_h,
        stride_ns,
        stride_out_b,
        stride_out_h,
        stride_lse_b,
        stride_lse_h,
        HQ: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """PDL consumer entrypoint; the wait is inlined before all reads."""
        _triton_fp8_cuda_coarse_combine_kernel_impl(
            partial_out_ptr,
            partial_lse2_ptr,
            num_splits_ptr,
            out_ptr,
            lse_ptr,
            stride_po_split,
            stride_po_h,
            stride_pl_split,
            stride_pl_h,
            stride_ns,
            stride_out_b,
            stride_out_h,
            stride_lse_b,
            stride_lse_h,
            HQ,
            DV,
            BLOCK_SPLITS,
            BLOCK_ROWS,
            True,
        )

    @triton.jit
    def _triton_fp8_single_split_lse_finalize_kernel(
        partial_lse2_ptr,
        lse_ptr,
        stride_pl_split: tl.constexpr,
        stride_pl_h: tl.constexpr,
        stride_lse_b: tl.constexpr,
        stride_lse_h: tl.constexpr,
        HQ: tl.constexpr,
        TOTAL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Convert one-split log2 LSE to the public natural-log convention."""
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL
        batch_idx = offsets // HQ
        head_idx = offsets - batch_idx * HQ
        lse2 = tl.load(
            partial_lse2_ptr
            + batch_idx.to(tl.int64) * stride_pl_split
            + head_idx.to(tl.int64) * stride_pl_h,
            mask=mask,
            other=0.0,
        )
        tl.store(
            lse_ptr
            + batch_idx.to(tl.int64) * stride_lse_b
            + head_idx.to(tl.int64) * stride_lse_h,
            lse2 * _TLE_LN2,
            mask=mask,
        )


def _make_tma_descriptor(tensor: torch.Tensor, block_shape):
    from triton.tools.tensor_descriptor import TensorDescriptor

    return TensorDescriptor.from_tensor(tensor, block_shape=block_shape)


def _prepare_compiled_runner(
    jit_function,
    args,
    grid,
    *,
    num_warps: int = 4,
    launch_pdl: bool = False,
):
    """Bind one already-specialized Triton kernel to its direct launcher."""
    from triton.runtime.driver import driver

    if jit_function.pre_run_hooks:
        raise RuntimeError(
            "prepared compiled launch does not support JIT pre-run hooks"
        )
    device = driver.active.get_current_device()
    _, _, _, _, binder = jit_function.device_caches[device]
    bound_args, _, _ = binder(
        *args,
        num_warps=num_warps,
        num_stages=1,
        launch_pdl=launch_pdl,
    )
    kernel = jit_function.warmup(
        *args,
        grid=grid,
        num_warps=num_warps,
        num_stages=1,
        launch_pdl=launch_pdl,
    )
    if hasattr(kernel, "result"):
        kernel = kernel.result()
    grid3 = tuple(grid) + (1,) * (3 - len(grid))
    return kernel[grid3], tuple(bound_args.values())


def _launch_partial(
    q_nope,
    q_rope,
    q_scale,
    k_cache_lora,
    k_cache_rope,
    k_scale,
    block_table,
    cache_seqlens,
    split_batch,
    split_page_begin,
    split_num_pages,
    target_out,
    partial_lse2,
    q_desc,
    qr_desc,
    qs_desc,
    k_desc,
    kr_desc,
    ks_desc,
    h_q: int,
    softmax_scale: float,
    direct_output: bool = False,
):
    if not HAS_TLE:
        raise RuntimeError("Split-K FP8 MLA launcher called without TLE support")

    rh = h_q // TLE_FP8_BH
    partial_grid = (int(split_batch.numel()) * rh,)

    _fp8_dense_mla_splitk_partial[partial_grid](
        q_nope,
        q_rope,
        q_scale,
        k_cache_lora,
        k_cache_rope,
        k_scale,
        block_table,
        cache_seqlens,
        split_batch,
        split_page_begin,
        split_num_pages,
        target_out,
        partial_lse2,
        q_desc,
        qr_desc,
        qs_desc,
        k_desc,
        kr_desc,
        ks_desc,
        q_nope.stride(0),
        q_nope.stride(2),
        q_rope.stride(0),
        q_rope.stride(2),
        q_scale.stride(0),
        q_scale.stride(2),
        k_cache_lora.stride(0),
        k_cache_lora.stride(1),
        k_cache_rope.stride(0),
        k_cache_rope.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        cache_seqlens.stride(0),
        split_batch.stride(0),
        split_page_begin.stride(0),
        split_num_pages.stride(0),
        target_out.stride(0),
        target_out.stride(2 if direct_output else 1),
        partial_lse2.stride(0),
        partial_lse2.stride(1),
        softmax_scale,
        64 * 512,
        64 * 64 * 2,
        64 * 4,
        64 * K_CONTENT_TILE_HOST,
        64 * 64 * 2,
        64 * 4,
        D_CKV,
        D_ROPE,
        TLE_FP8_BK,
        TLE_FP8_BH,
        h_q,
        rh,
        PAGE_SIZE,
        TLE_FP8_DPH,
        True,
        False,
        num_warps=4,
        num_stages=1,
    )


def _launch_combine(
    partial_out,
    partial_lse2,
    num_splits,
    out,
    lse,
    h_q: int,
):
    batch_size = int(out.shape[0])
    common_args = (
        partial_out,
        partial_lse2,
        num_splits,
        out,
        lse,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_lse2.stride(0),
        partial_lse2.stride(1),
        num_splits.stride(0),
        out.stride(0),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        h_q,
        D_CKV,
    )
    if batch_size >= CUDA_COARSE_COMBINE_MIN_BATCH:
        combine_grid = (
            batch_size,
            math.ceil(h_q / CUDA_COARSE_COMBINE_BLOCK_ROWS),
        )
        _triton_fp8_cuda_coarse_combine_kernel[combine_grid](
            *common_args,
            CUDA_COARSE_COMBINE_BLOCK_SPLITS,
            CUDA_COARSE_COMBINE_BLOCK_ROWS,
            num_warps=8,
            num_stages=1,
        )
        return

    # Use the fine-grained combine path for B=1 and B=2.
    combine_grid = (
        batch_size * h_q,
        math.ceil(D_CKV / COMBINE_BLOCK_D),
    )
    _triton_fp8_splitk_combine_kernel[combine_grid](
        *common_args,
        COMBINE_BLOCK_SPLITS,
        COMBINE_BLOCK_D,
        num_warps=4,
        num_stages=1,
    )


def _launch_single_split_lse_finalize(
    partial_lse2,
    lse,
    h_q: int,
):
    total = int(lse.shape[0]) * h_q
    grid = (triton.cdiv(total, LSE_FINALIZE_BLOCK),)
    _triton_fp8_single_split_lse_finalize_kernel[grid](
        partial_lse2,
        lse,
        partial_lse2.stride(0),
        partial_lse2.stride(1),
        lse.stride(0),
        lse.stride(1),
        h_q,
        total,
        LSE_FINALIZE_BLOCK,
        num_warps=4,
        num_stages=1,
    )


def prepare_flash_mla_ckv_fp8_per_token(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata=None,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    pages_per_split: int = DEFAULT_PAGES_PER_SPLIT,
    max_splits: Optional[int] = None,
    *,
    initial_cache_seqlens,
    max_cache_seqlens,
):
    """Build an adaptive split plan and a CUDA Graph-compatible replay handle."""
    batch_size = int(q_nope.shape[0])
    if batch_size <= 0 or int(cache_seqlens.numel()) != batch_size:
        raise ValueError("batch dimensions do not match")
    if int(q_nope.shape[1]) != 1:
        raise ValueError("SQ=1 decode only")
    h_q = int(q_nope.shape[2])
    if h_q <= 0 or h_q % TLE_FP8_BH:
        raise ValueError("HQ must be a positive multiple of 64")
    initial = _host_certificate_lengths(
        initial_cache_seqlens,
        "initial_cache_seqlens",
        batch_size=batch_size,
    )
    capacity = _host_certificate_lengths(
        max_cache_seqlens,
        "max_cache_seqlens",
        batch_size=batch_size,
    )
    if any(current > maximum for current, maximum in zip(initial, capacity)):
        raise ValueError("initial_cache_seqlens cannot exceed max_cache_seqlens")
    current = _host_lengths(cache_seqlens, "cache_seqlens", batch_size=batch_size)
    if current != initial:
        raise ValueError(
            "cache_seqlens storage must match initial_cache_seqlens at prepare"
        )
    required_pages = max(math.ceil(length / PAGE_SIZE) for length in capacity)
    if block_table.ndim != 2 or int(block_table.shape[0]) != batch_size:
        raise ValueError("block_table must be a two-dimensional batch table")
    if int(block_table.shape[1]) < required_pages:
        raise ValueError(
            "block_table does not cover the prepared max_cache_seqlens capacity"
        )
    if num_splits is None:
        if tile_scheduler_metadata is not None:
            num_splits = tile_scheduler_metadata.num_splits
        else:
            _, num_splits = get_mla_ckv_fp8_metadata(
                cache_seqlens,
                int(q_nope.shape[2]),
                1,
                pages_per_split=pages_per_split,
                max_splits=max_splits,
            )
    if num_splits is None:
        raise ValueError("no split plan")

    # Preserve caller-visible capacity metadata while selecting a compact
    # 4..32-page execution grain.
    meta = _build_adaptive_execution_meta(
        capacity,
        h_q,
        q_nope.device,
        pages_per_split,
    )
    if max_splits is not None and int(max_splits) < max(meta.capacity_splits):
        raise ValueError("max_splits cannot be below an adaptive row capacity")
    execution_block_table = _pad_block_table(block_table, meta.padded_pages)
    total_splits = int(meta.split_batch.numel())
    scale = float(softmax_scale if softmax_scale is not None else D_QK**-0.5)

    direct_single_output = bool(meta.capacity_splits) and all(
        count == 1 for count in meta.capacity_splits
    )
    if direct_single_output:
        partial_out = torch.empty((0,), dtype=torch.float32, device=q_nope.device)
    else:
        partial_out = torch.empty(
            (total_splits, h_q, D_CKV),
            dtype=torch.float32,
            device=q_nope.device,
        )
    partial_lse2 = torch.empty(
        (total_splits, h_q), dtype=torch.float32, device=q_nope.device
    )
    out = torch.empty(
        (batch_size, 1, h_q, head_dim_v), dtype=q_rope.dtype, device=q_nope.device
    )
    lse = torch.empty((batch_size, h_q, 1), dtype=torch.float32, device=q_nope.device)

    handle = _FlashMLAFp8PreparedHandle(
        q_nope,
        q_rope,
        q_scale,
        k_cache_lora,
        k_cache_rope,
        k_scale,
        block_table,
        execution_block_table,
        cache_seqlens,
        meta,
        partial_out,
        partial_lse2,
        out,
        lse,
        h_q,
        scale,
        head_dim_v,
        initial,
        capacity,
    )
    first_result = handle()
    return handle, first_result


prepare_flash_mla_ckv_fp8_per_token_decode = prepare_flash_mla_ckv_fp8_per_token


class _FlashMLAFp8PreparedHandle:
    """Callable prepared decode handle with stable descriptors and workspace."""

    __slots__ = (
        "q_nope",
        "q_rope",
        "q_scale",
        "k_cache_lora",
        "k_cache_rope",
        "k_scale",
        "block_table",
        "_execution_block_table",
        "cache_seqlens",
        "_meta",
        "_partial_out",
        "_partial_lse2",
        "_out",
        "_lse",
        "_h_q",
        "_scale",
        "_head_dim_v",
        "_initial_cache_seqlens",
        "_max_cache_seqlens",
        "_cache_seqlens_host",
        "_cache_version",
        "_num_pages",
        "_logical_active_splits",
        "_direct_single_output",
        "_in_use",
        "_q_desc",
        "_qr_desc",
        "_qs_desc",
        "_out_desc",
        "_k_desc",
        "_kr_desc",
        "_ks_desc",
        "_launch_pack_key",
        "_partial_compiled_runner",
        "_partial_compiled_args",
        "_aux_compiled_runner",
        "_aux_compiled_args",
        "_launch_pack_reuses",
        "_cuda_graph_key",
        "_cuda_graph",
        "_cuda_graph_capture_stream",
        "_cuda_graph_eligible",
    )

    def __init__(
        self,
        q_nope,
        q_rope,
        q_scale,
        k_cache_lora,
        k_cache_rope,
        k_scale,
        block_table,
        execution_block_table,
        cache_seqlens,
        meta,
        partial_out,
        partial_lse2,
        out,
        lse,
        h_q,
        scale,
        head_dim_v,
        initial_cache_seqlens,
        max_cache_seqlens,
    ) -> None:
        self.q_nope = q_nope
        self.q_rope = q_rope
        self.q_scale = q_scale
        self.k_cache_lora = k_cache_lora
        self.k_cache_rope = k_cache_rope
        self.k_scale = k_scale
        self.block_table = block_table
        self._execution_block_table = execution_block_table
        self.cache_seqlens = cache_seqlens
        self._meta = meta
        self._partial_out = partial_out
        self._partial_lse2 = partial_lse2
        self._out = out
        self._lse = lse
        self._h_q = h_q
        self._scale = scale
        self._head_dim_v = head_dim_v
        self._direct_single_output = bool(meta.capacity_splits) and all(
            count == 1 for count in meta.capacity_splits
        )
        # Prepared replay keeps the bound Q/K tensors and their storage
        # addresses stable.  Match CUDA's launch-parameter lifetime by
        # materializing the six immutable TMA descriptors once instead of
        # rebuilding them on every decode step.
        _set_triton_descriptor_allocator(q_nope.device)
        self._q_desc = _make_tma_descriptor(
            q_nope.reshape(-1, D_CKV), [TLE_FP8_BH, D_CKV]
        )
        self._qr_desc = _make_tma_descriptor(
            q_rope.reshape(-1, D_ROPE), [TLE_FP8_BH, D_ROPE]
        )
        self._qs_desc = _make_tma_descriptor(q_scale.reshape(-1, h_q), [1, TLE_FP8_BH])
        # Rebound in _partial_launch_args when caller-provided output storage
        # changes; non-direct routes never consume this placeholder.
        self._out_desc = self._q_desc
        self._k_desc = _make_tma_descriptor(
            k_cache_lora.reshape(-1, D_CKV),
            [TLE_FP8_BK, K_CONTENT_TILE_HOST],
        )
        self._kr_desc = _make_tma_descriptor(
            k_cache_rope.reshape(-1, D_ROPE), [TLE_FP8_BK, D_ROPE]
        )
        self._ks_desc = _make_tma_descriptor(
            k_scale.reshape(-1, TLE_FP8_BK), [1, TLE_FP8_BK]
        )
        self._initial_cache_seqlens = tuple(initial_cache_seqlens)
        self._max_cache_seqlens = tuple(max_cache_seqlens)
        self._cache_seqlens_host = tuple(initial_cache_seqlens)
        self._cache_version = _tensor_version(cache_seqlens)
        self._num_pages, self._logical_active_splits = _length_page_state(
            self._cache_seqlens_host,
            int(meta.adaptive_fixed_pages),
        )
        if self._direct_single_output:
            if int(meta.split_batch.numel()) != len(self._max_cache_seqlens):
                raise AssertionError(
                    "direct-single schedule must contain one split per batch row"
                )
            if self._partial_out.numel() != 0:
                raise AssertionError(
                    "direct-single schedule must not allocate an FP32 output workspace"
                )
        self._in_use = False
        self._launch_pack_key = None
        self._partial_compiled_runner = None
        self._partial_compiled_args = None
        self._aux_compiled_runner = None
        self._aux_compiled_args = None
        self._launch_pack_reuses = 0
        self._cuda_graph_key = None
        self._cuda_graph = None
        self._cuda_graph_capture_stream = None
        self._cuda_graph_eligible = False

    def _programmatic_dependency_capacity(self):
        batch_size = int(self._out.shape[0])
        partial_ctas = int(self._meta.split_batch.numel()) * (self._h_q // TLE_FP8_BH)
        consumer_ctas = batch_size * math.ceil(
            self._h_q / CUDA_COARSE_COMBINE_BLOCK_ROWS
        )
        sm_count = int(
            torch.cuda.get_device_properties(self._out.device).multi_processor_count
        )
        return partial_ctas, consumer_ctas, sm_count

    def _use_programmatic_dependent_launch(self) -> bool:
        partial_ctas, consumer_ctas, sm_count = self._programmatic_dependency_capacity()
        # Keep one full consumer grid of scheduling headroom beyond the
        # producer and consumer fit.
        return (
            not self._direct_single_output
            and int(self._out.shape[0]) >= CUDA_COARSE_COMBINE_MIN_BATCH
            and partial_ctas + 2 * consumer_ctas <= sm_count + 1
        )

    def _use_full_tail_specialization(self) -> bool:
        # Every scheduled capacity page must be real and complete.  A handle
        # whose current lengths have not reached prepared capacity keeps the
        # masked kernel even when the current token count is 64-aligned.
        return (
            self._cache_seqlens_host == self._max_cache_seqlens
            and all(length % PAGE_SIZE == 0 for length in self._cache_seqlens_host)
            and tuple(self._logical_active_splits) == tuple(self._meta.capacity_splits)
        )

    def _use_merged_state_v_completion(self) -> bool:
        return (
            self._h_q == 64
            # Admit B8 full-tail shapes to the split-structure-agnostic
            # merged-completion schedule.
            and int(self._out.shape[0]) >= 8
            # L8192 also satisfies the full-tail certificate required below.
            and all(length in (33280, 8192) for length in self._max_cache_seqlens)
            and self._use_full_tail_specialization()
        )

    def _use_pretranspose_v1(self) -> bool:
        return (
            self._h_q == 128
            and int(self._out.shape[0]) in (16, 32)
            and all(length == 640 for length in self._max_cache_seqlens)
        )

    def _use_fixed_ten_page_v1(self) -> bool:
        return (
            self._h_q == 128
            and int(self._out.shape[0]) in (32, 64, 128)
            and all(length == 640 for length in self._max_cache_seqlens)
            and self._use_full_tail_specialization()
        )

    def _use_fixed_two_page_v1(self) -> bool:
        # The direct one-pair family has one CTA per batch row.  With every
        # certified length in (64, 128], each CTA has exactly two real pages,
        # although page one may be partial.  Constant-fold only num_pages;
        # retain the masked math and worker schedule unchanged.
        return (
            self._direct_single_output
            and int(self._meta.max_pages_per_split) == 2
            and self._cache_seqlens_host == self._max_cache_seqlens
            and all(
                PAGE_SIZE < length <= 2 * PAGE_SIZE
                for length in self._cache_seqlens_host
            )
        )

    def _use_direct_lse_v2(self) -> bool:
        # A direct-single route has exactly one partial CTA for each output
        # row/head block, so its WG0 LSE store has no cross-split reduction.
        # Write natural-log LSE directly and remove the separate conversion
        # kernel for the short one-pair route.
        return (
            self._direct_single_output and int(self._meta.max_pages_per_split) <= 2
        ) or self._use_fixed_ten_page_v1()

    def _partial_launch_args(self, target_out, target_lse):
        h_q = self._h_q
        rh = h_q // TLE_FP8_BH
        if self._direct_single_output:
            self._out_desc = _make_tma_descriptor(
                target_out.reshape(-1, D_CKV), [TLE_FP8_BH, D_ROPE]
            )
        return (
            self.q_nope,
            self.q_rope,
            self.q_scale,
            self.k_cache_lora,
            self.k_cache_rope,
            self.k_scale,
            self._execution_block_table,
            self.cache_seqlens,
            self._meta.split_batch,
            self._meta.split_page_begin,
            self._meta.split_num_pages,
            target_out,
            target_lse,
            self._q_desc,
            self._qr_desc,
            self._qs_desc,
            self._out_desc,
            self._k_desc,
            self._kr_desc,
            self._ks_desc,
            self.q_nope.stride(0),
            self.q_nope.stride(2),
            self.q_rope.stride(0),
            self.q_rope.stride(2),
            self.q_scale.stride(0),
            self.q_scale.stride(2),
            self.k_cache_lora.stride(0),
            self.k_cache_lora.stride(1),
            self.k_cache_rope.stride(0),
            self.k_cache_rope.stride(1),
            self.k_scale.stride(0),
            self.k_scale.stride(1),
            self._execution_block_table.stride(0),
            self._execution_block_table.stride(1),
            self.cache_seqlens.stride(0),
            self._meta.split_batch.stride(0),
            self._meta.split_page_begin.stride(0),
            self._meta.split_num_pages.stride(0),
            target_out.stride(0),
            target_out.stride(2 if self._direct_single_output else 1),
            target_lse.stride(0),
            target_lse.stride(1),
            self._scale,
            64 * 512,
            64 * 64 * 2,
            64 * 4,
            64 * K_CONTENT_TILE_HOST,
            64 * 64 * 2,
            64 * 4,
            D_CKV,
            D_ROPE,
            TLE_FP8_BK,
            TLE_FP8_BH,
            h_q,
            rh,
            PAGE_SIZE,
            TLE_FP8_DPH,
            int(self._meta.adaptive_fixed_pairs) >= 2,
            self._use_full_tail_specialization(),
            int(self._meta.max_pages_per_split) <= 2,
            self._use_merged_state_v_completion(),
            self._direct_single_output,
            (
                10
                if self._use_fixed_ten_page_v1()
                else (2 if self._use_fixed_two_page_v1() else 0)
            ),
            self._use_direct_lse_v2(),
        )

    def _aux_launch_spec(self, out, lse):
        if self._direct_single_output:
            total = int(lse.shape[0]) * self._h_q
            return (
                _triton_fp8_single_split_lse_finalize_kernel,
                (
                    self._partial_lse2,
                    lse,
                    self._partial_lse2.stride(0),
                    self._partial_lse2.stride(1),
                    lse.stride(0),
                    lse.stride(1),
                    self._h_q,
                    total,
                    LSE_FINALIZE_BLOCK,
                ),
                (triton.cdiv(total, LSE_FINALIZE_BLOCK),),
            )

        common_args = (
            self._partial_out,
            self._partial_lse2,
            self._meta.num_splits,
            out,
            lse,
            self._partial_out.stride(0),
            self._partial_out.stride(1),
            self._partial_lse2.stride(0),
            self._partial_lse2.stride(1),
            self._meta.num_splits.stride(0),
            out.stride(0),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            self._h_q,
            D_CKV,
        )
        batch_size = int(out.shape[0])
        if batch_size >= CUDA_COARSE_COMBINE_MIN_BATCH:
            coarse_jit = (
                _triton_fp8_cuda_coarse_combine_pdl_kernel
                if self._use_programmatic_dependent_launch()
                else _triton_fp8_cuda_coarse_combine_kernel
            )
            return (
                coarse_jit,
                common_args
                + (
                    CUDA_COARSE_COMBINE_BLOCK_SPLITS,
                    CUDA_COARSE_COMBINE_BLOCK_ROWS,
                ),
                (
                    batch_size,
                    math.ceil(self._h_q / CUDA_COARSE_COMBINE_BLOCK_ROWS),
                ),
            )
        return (
            _triton_fp8_splitk_combine_kernel,
            common_args + (COMBINE_BLOCK_SPLITS, COMBINE_BLOCK_D),
            (
                batch_size * self._h_q,
                math.ceil(D_CKV / COMBINE_BLOCK_D),
            ),
        )

    def _ensure_compiled_launch_pack(self, out, lse) -> None:
        # Exact pointer identity preserves every alignment specialization that
        # the public caller-provided output contract previously admitted. Keep
        # only the most recent pack so workloads that rotate output buffers do
        # not grow an unbounded launcher cache.
        key = (
            int(out.data_ptr()),
            int(lse.data_ptr()),
            self._use_full_tail_specialization(),
            self._use_merged_state_v_completion(),
            self._use_pretranspose_v1(),
            self._use_fixed_ten_page_v1(),
            self._use_fixed_two_page_v1(),
        )
        if key == self._launch_pack_key:
            self._launch_pack_reuses += 1
            return

        target_out = out if self._direct_single_output else self._partial_out
        direct_lse = self._use_direct_lse_v2()
        target_lse = lse if direct_lse else self._partial_lse2
        use_pdl = self._use_programmatic_dependent_launch()
        partial_grid = (
            int(self._meta.split_batch.numel()) * (self._h_q // TLE_FP8_BH),
        )
        partial_jit = (
            _fp8_dense_mla_splitk_partial_pdl
            if use_pdl
            else (
                _fp8_dense_mla_splitk_partial_pretranspose
                if self._use_pretranspose_v1()
                else _fp8_dense_mla_splitk_partial
            )
        )
        partial_runner, partial_args = _prepare_compiled_runner(
            partial_jit,
            self._partial_launch_args(target_out, target_lse),
            partial_grid,
            launch_pdl=False,
        )
        if direct_lse:
            aux_runner, aux_bound_args = None, None
        else:
            aux_jit, aux_args, aux_grid = self._aux_launch_spec(out, lse)
            aux_runner, aux_bound_args = _prepare_compiled_runner(
                aux_jit,
                aux_args,
                aux_grid,
                num_warps=(
                    8
                    if aux_jit
                    in (
                        _triton_fp8_cuda_coarse_combine_kernel,
                        _triton_fp8_cuda_coarse_combine_pdl_kernel,
                    )
                    else 4
                ),
                launch_pdl=use_pdl,
            )
        self._partial_compiled_runner = partial_runner
        self._partial_compiled_args = partial_args
        self._aux_compiled_runner = aux_runner
        self._aux_compiled_args = aux_bound_args
        self._launch_pack_key = key
        self._launch_pack_reuses = 0
        self._cuda_graph_key = None
        self._cuda_graph = None
        self._cuda_graph_capture_stream = None
        self._cuda_graph_eligible = not use_pdl

    def _ensure_cuda_graph_replay(self):
        """Capture the stable two-kernel prepared replay after one pointer hit."""
        key = self._launch_pack_key
        if key is None or not self._cuda_graph_eligible or self._launch_pack_reuses < 1:
            return None
        if self._cuda_graph_key == key and self._cuda_graph is not None:
            return self._cuda_graph

        current_stream = torch.cuda.current_stream(self._out.device)
        capture_stream = torch.cuda.Stream(device=self._out.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            self._partial_compiled_runner(*self._partial_compiled_args)
            if self._aux_compiled_runner is not None:
                self._aux_compiled_runner(*self._aux_compiled_args)
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            self._partial_compiled_runner(*self._partial_compiled_args)
            if self._aux_compiled_runner is not None:
                self._aux_compiled_runner(*self._aux_compiled_args)
        current_stream.wait_stream(capture_stream)
        self._cuda_graph_key = key
        self._cuda_graph = graph
        self._cuda_graph_capture_stream = capture_stream
        return graph

    def _claim(self) -> None:
        if self._in_use:
            raise RuntimeError("prepared handle is already in use")
        self._in_use = True

    def _validate_length_bounds(self, values) -> None:
        for batch_index, (value, previous, capacity) in enumerate(
            zip(values, self._cache_seqlens_host, self._max_cache_seqlens)
        ):
            if value < previous:
                raise RuntimeError(f"cache_seqlens[{batch_index}] must be monotonic")
            if value > capacity:
                raise RuntimeError(
                    f"cache_seqlens[{batch_index}] {value} exceeds prepared "
                    f"capacity {capacity}"
                )

    def _apply_length_certificate(
        self,
        cache_seqlens,
        *,
        require_version_change: bool,
    ) -> None:
        values = _host_certificate_lengths(
            cache_seqlens,
            "cache_seqlens",
            batch_size=len(self._cache_seqlens_host),
        )
        self._validate_length_bounds(values)

        version = _tensor_version(self.cache_seqlens)
        changed = values != self._cache_seqlens_host
        if changed and require_version_change and version == self._cache_version:
            raise RuntimeError(
                "cache_seqlens storage did not receive an observable in-place "
                "PyTorch update before the new host certificate"
            )
        if not changed and version != self._cache_version:
            raise RuntimeError(
                "cache_seqlens changed without a new host length certificate"
            )

        pages, logical_active_splits = _length_page_state(
            values,
            int(self._meta.adaptive_fixed_pages),
        )
        for batch_index, (logical, capacity) in enumerate(
            zip(logical_active_splits, self._meta.capacity_splits)
        ):
            if logical > capacity:
                raise RuntimeError(
                    f"logical split count for batch {batch_index} exceeds capacity"
                )

        self._cache_seqlens_host = values
        self._num_pages = pages
        self._logical_active_splits = logical_active_splits
        self._cache_version = version

    def set_cache_seqlens_(self, cache_seqlens) -> None:
        """Own a monotonic in-place length update on the bound CUDA stream."""
        values = _host_certificate_lengths(
            cache_seqlens,
            "cache_seqlens",
            batch_size=len(self._cache_seqlens_host),
        )
        self._validate_length_bounds(values)
        self._claim()
        try:
            if values != self._cache_seqlens_host:
                update = torch.tensor(
                    values,
                    dtype=self.cache_seqlens.dtype,
                    device=self.cache_seqlens.device,
                )
                self.cache_seqlens.copy_(update)
            self._apply_length_certificate(
                values,
                require_version_change=False,
            )
        finally:
            self._in_use = False

    def _validate_output(self, tensor: torch.Tensor, *, lse: bool) -> None:
        template = self._lse if lse else self._out
        label = "lse" if lse else "out"
        if tuple(tensor.shape) != tuple(template.shape):
            raise RuntimeError(
                f"{label} shape must be {tuple(template.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != template.dtype or tensor.device != template.device:
            raise RuntimeError(f"{label} dtype/device mismatch")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{label} must be contiguous")

    def launch(self, *, cache_seqlens=None, out=None, lse=None):
        """Submit one prepared decode step using a host length certificate."""
        self._claim()
        try:
            if cache_seqlens is None:
                if _tensor_version(self.cache_seqlens) != self._cache_version:
                    raise RuntimeError(
                        "cache_seqlens changed without a host length certificate"
                    )
            else:
                self._apply_length_certificate(
                    cache_seqlens,
                    require_version_change=True,
                )

            if (out is None) != (lse is None):
                raise RuntimeError(
                    "out and lse must either both be supplied or both omitted"
                )
            if out is None:
                out = torch.empty_like(self._out)
                lse = torch.empty_like(self._lse)
            else:
                self._validate_output(out, lse=False)
                self._validate_output(lse, lse=True)

            self._ensure_compiled_launch_pack(out, lse)
            graph = self._ensure_cuda_graph_replay()
            if graph is None:
                self._partial_compiled_runner(*self._partial_compiled_args)
                if self._aux_compiled_runner is not None:
                    self._aux_compiled_runner(*self._aux_compiled_args)
            else:
                graph.replay()
            return out, lse
        finally:
            self._in_use = False

    __call__ = launch

    def debug_state(self) -> dict:
        return {
            "batch_parallel": True,
            "programmatic_dependent_launch": (
                self._use_programmatic_dependent_launch()
            ),
            "pdl_scope": (
                "consumer_headroom_coarse_combine"
                if self._use_programmatic_dependent_launch()
                else None
            ),
            "pdl_partial_ctas": (self._programmatic_dependency_capacity()[0]),
            "pdl_consumer_ctas": (self._programmatic_dependency_capacity()[1]),
            "pdl_sm_count": (self._programmatic_dependency_capacity()[2]),
            "pdl_capacity_safe": (
                sum(self._programmatic_dependency_capacity()[:2])
                <= self._programmatic_dependency_capacity()[2]
            ),
            "pdl_consumer_headroom_safe": (
                self._programmatic_dependency_capacity()[0]
                + 2 * self._programmatic_dependency_capacity()[1]
                <= self._programmatic_dependency_capacity()[2]
            ),
            "per_batch_loop": False,
            "batch_launch_count": 1,
            "partial": True,
            "combine": not self._direct_single_output,
            "cuda_coarse_combine": (
                not self._direct_single_output
                and int(self._out.shape[0]) >= CUDA_COARSE_COMBINE_MIN_BATCH
            ),
            "combine_policy": (
                None
                if self._direct_single_output
                else (
                    "cuda_coarse_8_rows"
                    if int(self._out.shape[0]) >= CUDA_COARSE_COMBINE_MIN_BATCH
                    else "fine_splitk"
                )
            ),
            "combine_block_rows": (
                CUDA_COARSE_COMBINE_BLOCK_ROWS
                if (
                    not self._direct_single_output
                    and int(self._out.shape[0]) >= CUDA_COARSE_COMBINE_MIN_BATCH
                )
                else None
            ),
            "combine_min_batch": CUDA_COARSE_COMBINE_MIN_BATCH,
            "direct_single_output": self._direct_single_output,
            "direct_output_dtype": (
                str(self._out.dtype) if self._direct_single_output else None
            ),
            "lse_only_finalize": self._direct_single_output,
            "full_dv_finalize": not self._direct_single_output,
            "compact_workspace_bytes": int(
                self._partial_out.numel() * self._partial_out.element_size()
                + self._partial_lse2.numel() * self._partial_lse2.element_size()
            ),
            "max_splits": int(self._meta.max_splits),
            "max_pages_per_split": int(self._meta.max_pages_per_split),
            "total_splits": int(self._meta.split_batch.numel()),
            "adaptive_fixed_pages": int(self._meta.adaptive_fixed_pages),
            "adaptive_fixed_pairs": int(self._meta.adaptive_fixed_pairs),
            "adaptive_selection": list(self._meta.adaptive_selection),
            "capacity_splits": list(self._meta.capacity_splits),
            "initial_cache_seqlens": list(self._initial_cache_seqlens),
            "cache_seqlens": list(self._cache_seqlens_host),
            "num_pages": list(self._num_pages),
            "logical_active_splits": list(self._logical_active_splits),
            "full_tail_specialization": self._use_full_tail_specialization(),
            "fixed_two_page_specialization": self._use_fixed_two_page_v1(),
            "merged_state_v_completion": self._use_merged_state_v_completion(),
            "masked_splits": int(
                sum(self._meta.capacity_splits) - sum(self._logical_active_splits)
            ),
            "max_cache_seqlens": list(self._max_cache_seqlens),
            "immutable_capacity_schedule": True,
            "fresh_output_storage": "two_empty_like",
            "padded_block_table": self._execution_block_table is not self.block_table,
            "schedule": "all_batch_csr_h800_wave_cost_fixed_pairs_tail34",
            "explicit_pipeline": True,
        }


def flash_mla_ckv_fp8_per_token(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata=None,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    pages_per_split: int = DEFAULT_PAGES_PER_SPLIT,
    max_splits: Optional[int] = None,
    out=None,
    lse=None,
):
    """Public one-shot op: metadata + prepare + run."""
    batch_size = int(q_nope.shape[0])
    h_q = int(q_nope.shape[2])
    if h_q <= 0 or h_q % TLE_FP8_BH:
        raise ValueError("HQ must be a positive multiple of 64")
    if num_splits is None:
        _, num_splits = get_mla_ckv_fp8_metadata(
            cache_seqlens,
            h_q,
            1,
            pages_per_split=pages_per_split,
            max_splits=max_splits,
        )
    capacity = _host_lengths(
        cache_seqlens,
        "cache_seqlens",
        batch_size=batch_size,
    )
    if any(length <= 0 or length > MAX_SEQUENCE_LENGTH for length in capacity):
        raise ValueError(f"cache_seqlens entries must be in [1, {MAX_SEQUENCE_LENGTH}]")
    required_pages = max(math.ceil(length / PAGE_SIZE) for length in capacity)
    if block_table.ndim != 2 or int(block_table.shape[0]) != batch_size:
        raise ValueError("block_table must be a two-dimensional batch table")
    if int(block_table.shape[1]) < required_pages:
        raise ValueError("block_table does not cover cache_seqlens")
    meta = _build_adaptive_execution_meta(
        capacity,
        h_q,
        q_nope.device,
        pages_per_split,
    )
    if max_splits is not None and int(max_splits) < max(meta.capacity_splits):
        raise ValueError("max_splits cannot be below an adaptive row capacity")
    execution_block_table = _pad_block_table(block_table, meta.padded_pages)
    if softmax_scale is None:
        softmax_scale = float(D_QK**-0.5)
    if out is None:
        out = torch.empty(
            (batch_size, 1, int(q_nope.shape[2]), head_dim_v),
            dtype=q_rope.dtype,
            device=q_nope.device,
        )
    if lse is None:
        lse = torch.empty(
            (batch_size, int(q_nope.shape[2]), 1),
            dtype=torch.float32,
            device=q_nope.device,
        )
    handle = _FlashMLAFp8PreparedHandle(
        q_nope,
        q_rope,
        q_scale,
        k_cache_lora,
        k_cache_rope,
        k_scale,
        block_table,
        execution_block_table,
        cache_seqlens,
        meta,
        (
            torch.empty((0,), dtype=torch.float32, device=q_nope.device)
            if bool(meta.capacity_splits)
            and all(count == 1 for count in meta.capacity_splits)
            else torch.empty(
                (int(meta.split_batch.numel()), h_q, D_CKV),
                dtype=torch.float32,
                device=q_nope.device,
            )
        ),
        torch.empty(
            (int(meta.split_batch.numel()), h_q),
            dtype=torch.float32,
            device=q_nope.device,
        ),
        out,
        lse,
        h_q,
        float(softmax_scale),
        head_dim_v,
        capacity,
        capacity,
    )
    return handle(out=out, lse=lse)
