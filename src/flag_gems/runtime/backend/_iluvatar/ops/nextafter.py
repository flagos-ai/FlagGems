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


import torch
import triton
import triton.language as tl


@triton.jit
def _nextafter_pointwise(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
    IS32: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)

    # ---- nextafter via IEEE-754 bit manipulation (musl semantics) ----
    if IS32:
        ix = x.to(tl.int32, bitcast=True)
        iy = y.to(tl.int32, bitcast=True)
    else:
        ix = x.to(tl.int16, bitcast=True).to(tl.int32)
        iy = y.to(tl.int16, bitcast=True).to(tl.int32)

    ABS_MASK = 2147483647  # 0x7FFFFFFF
    NEG_ONE_BITS = -2147483647  # 0x80000001 (sign bit | 1)

    ax = ix & ABS_MASK
    ay = iy & ABS_MASK
    x_is_zero = ax == 0
    y_is_zero = ay == 0
    sign_diff = (ix < 0) != (iy < 0)
    dec = (ax > ay) | sign_diff
    zero_bits = tl.where(iy < 0, NEG_ONE_BITS, 1)
    new_bits = tl.where(x_is_zero, zero_bits, tl.where(dec, ix - 1, ix + 1))
    new_bits = tl.where(x_is_zero & y_is_zero, iy, new_bits)
    new_bits = tl.where(ix == iy, iy, new_bits)

    # NaN propagation (x + y is NaN iff x or y is NaN)
    isnan = (x != x) | (y != y)
    if IS32:
        nan_bits = (x + y).to(tl.int32, bitcast=True)
    else:
        nan_bits = (x + y).to(tl.float32).to(tl.int32, bitcast=True)
    new_bits = tl.where(isnan, nan_bits, new_bits)

    if IS32:
        out = new_bits.to(tl.float32, bitcast=True)
    elif IS_BF16:
        out = new_bits.to(tl.int16).to(tl.bfloat16, bitcast=True)
    else:
        out = new_bits.to(tl.int16).to(tl.float16, bitcast=True)
    tl.store(out_ptr + offs, out, mask=mask, cache_modifier=".cg")


_BLOCK = 512
_NUM_WARPS = 8

# Per-dtype block geometry chosen from a BLOCK x num_warps sweep on the target:
# BI-V150 reaches peak streaming bandwidth with ~64-bit vector accesses per
# thread and high thread counts per block (f32: 2 el/thread x 256 thr,
# f16: 4 el/thread x 128 thr, bf16: 4 el/thread x 512 thr).
_CONFIG = {
    torch.float32: (512, 8),
    torch.float16: (512, 4),
    torch.bfloat16: (2048, 16),
}

# Tiny tensors run best as a single block: BLOCK=1024/warps=4 keeps the whole
# f16 64x64 workload in one launch (measured 1.48x vs 1.38x with 8 blocks).
_SMALL_F16 = (1024, 4)
_SMALL_THRESHOLD = 8192


def run(input, other):
    output = torch.empty_like(input)
    numel = input.numel()
    dt = input.dtype
    if dt == torch.float32:
        is32, is_bf16 = True, False
    elif dt == torch.float16:
        is32, is_bf16 = False, False
    else:  # bfloat16
        is32, is_bf16 = False, True
    block, num_warps = _CONFIG.get(dt, (_BLOCK, _NUM_WARPS))
    if dt == torch.float16 and numel <= _SMALL_THRESHOLD:
        block, num_warps = _SMALL_F16
    grid = (triton.cdiv(numel, block),)
    _nextafter_pointwise[grid](
        input,
        other,
        output,
        numel,
        BLOCK=block,
        IS32=is32,
        IS_BF16=is_bf16,
        num_warps=num_warps,
    )
    return output
