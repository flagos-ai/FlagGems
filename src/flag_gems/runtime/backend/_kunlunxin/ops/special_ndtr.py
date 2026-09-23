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

# Kunlunxin( XPU ) ndtr(x) = 0.5 * erfc(-x / sqrt(2)) = 0.5 * (1 - erf(-x/sqrt(2))).
#
# The generic flag_gems/ops/special_ndtr.py routes ndtr through
# tl.math.erf -> extern `_ZN3xpu4erfEf` (XPU software libdevice-style
# implementation), which measures ~73x slower than torch-native on 16.7M+
# element tiles (baseline Gems Speedup ~= 0.0137x on (4096, 4096) fp32,
# 41.2ms vs 0.565ms).  This override reuses the exact odd polynomial
# u*P(u^2) already validated in _kunlunxin/ops/erf.py (LSQ fit on
# t in [0, 9], deg-12 in t, fp32 Horner max abs err 3.7e-5 on [0, 3.0] and
# hard saturation at |u| > 3) with u = -x/sqrt(2):
#   x >  3*sqrt(2) : erf(-x/sqrt2) saturates to -1 => ndtr = 1.0
#   x < -3*sqrt(2) : erf(-x/sqrt2) saturates to +1 => ndtr = 0.0
#   else           : 0.5 * (1 + (x/sqrt2) * P(x^2/2))
# Dense-grid check vs torch.special.ndtr on [-20, 20]: max |diff| 1.79e-5,
# 0.18x of the fp32 test tolerance (atol 1e-4 + rtol 1.3e-6*|ref|); fp16/bf16
# compute in fp32 and round once at store, matching the reference path.
# There is no transcendental at all (no exp), only FMA/dp2a-friendly Horner,
# matching the design that made the erfc kernel ~2.9x faster than torch-native
# on the generic path (this op ~73x on the baseline extern-erf path).
# NaN/Inf semantics: comparisons are false for NaN so the polynomial
# propagates NaN; +/-Inf saturate to 1.0 / 0.0 exactly like torch.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Same CodeGenConfig as the sibling erf(-family) kernels, tuned on XPU.
CUT_BOUND = tl.constexpr(4.242640687119285)  # 3 * sqrt(2)
UNROLL_NUM = 16
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    # Same tile buckets as _kunlunxin/ops/erfc.py (identical pointwise
    # load/store pattern and shape matrix, sweep-measured on XPU, see
    # harness/solution/performance/erfc_perf.md).  num_warps measures as a
    # no-op on this backend, so buckets are chosen on tile width alone.
    if n_elements >= 16_777_216 and n_elements % 65536 == 0:
        return 65536, 16, False
    if n_elements >= 1_048_576 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 65_536 and n_elements % 8192 == 0:
        return 8192, 4, False
    if n_elements <= 65_536:
        return 2048, 4, True
    return 8192, 4, True


@triton.jit
def ndtr_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    # ndtr(x) = 0.5 * (1 - erf(u)) with u = -x/sqrt(2);
    # erf(u) = u*P(u^2) => t = u^2 = x^2/2, v = u*P = -(x/sqrt(2))*P
    t = x * x * 0.5
    p = 2.0958534e-12
    p = p * t + -1.4200718e-10
    p = p * t + 4.475963e-09
    p = p * t + -8.813736e-08
    p = p * t + 1.2336866e-06
    p = p * t + -1.3277817e-05
    p = p * t + 0.00011584865
    p = p * t + -0.00084552215
    p = p * t + 0.005211773
    p = p * t + -0.0268563
    p = p * t + 0.112833545
    p = p * t + -0.37612554
    p = p * t + 1.1283791
    v = (x * p) * -0.7071067811865476
    r = tl.where(x < -CUT_BOUND, 1.0, v)  # u > 3 -> erf = 1 -> ndtr = 0
    r = tl.where(x > CUT_BOUND, -1.0, r)  # u < -3 -> erf = -1 -> ndtr = 1
    y = 0.5 * (1.0 - r)
    tl.store(out_ptr + offset, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def ndtr_kernel_unmasked(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    t = x * x * 0.5
    p = 2.0958534e-12
    p = p * t + -1.4200718e-10
    p = p * t + 4.475963e-09
    p = p * t + -8.813736e-08
    p = p * t + 1.2336866e-06
    p = p * t + -1.3277817e-05
    p = p * t + 0.00011584865
    p = p * t + -0.00084552215
    p = p * t + 0.005211773
    p = p * t + -0.0268563
    p = p * t + 0.112833545
    p = p * t + -0.37612554
    p = p * t + 1.1283791
    v = (x * p) * -0.7071067811865476
    r = tl.where(x < -CUT_BOUND, 1.0, v)
    r = tl.where(x > CUT_BOUND, -1.0, r)
    y = 0.5 * (1.0 - r)
    tl.store(out_ptr + offset, y.to(out_ptr.dtype.element_ty))


def _launch(x, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        ndtr_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        ndtr_kernel_unmasked[grid](
            x,
            out,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def special_ndtr(A):
    # Route torch.special.ndtr (aten::special_ndtr) to the polynomial fast
    # path instead of the generic tl.math.erf extern (XPU software impl).
    logger.debug("GEMS_KUNLUNXIN SPECIAL_NDTR")
    x = A.contiguous()
    out = torch.empty_like(x)
    _launch(x, out)
    return out
