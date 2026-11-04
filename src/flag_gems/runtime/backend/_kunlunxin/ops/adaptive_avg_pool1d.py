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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

FLAT_BLOCK = 1024


@triton.jit
def _load_f32_1d(ptr, mask, BF16: tl.constexpr):
    # bf16 strided/gather loads on XPU3 go through a vmerge float de-interleave
    # that mis-maps lanes; loading the raw int16 bits and reconstructing the
    # fp32 value via (bits << 16) avoids the buggy path and is bit-exact.
    if BF16:
        raw = tl.load(ptr, mask=mask, other=0)
        return ((raw.to(tl.int32) & 0xFFFF) << 16).to(tl.float32, bitcast=True)
    return tl.load(ptr, mask=mask, other=0.0).to(tl.float32)


@triton.jit
def _store_f32_1d(ptr, val, mask, BF16: tl.constexpr):
    # Mixing an int16-view load with a bf16-typed store in one kernel trips the
    # XPU3 vectorizer, so the bf16 result is rounded to bf16 bits
    # (round-to-nearest-even) and stored through an int16 view too.
    if BF16:
        rb = val.to(tl.int32, bitcast=True)
        rounded = (rb + 0x7FFF + ((rb >> 16) & 1)) >> 16
        tl.store(ptr, rounded.to(tl.int16), mask=mask)
    else:
        tl.store(ptr, val, mask=mask)


@libentry()
@triton.jit
def _adaptive_avg_pool1d_flat_kernel(
    input,
    output,
    total,
    L: tl.constexpr,
    OW: tl.constexpr,
    KW: tl.constexpr,
    AREA: tl.constexpr,
    BLOCK: tl.constexpr,
    BF16: tl.constexpr,
):
    # Exact-division path for small windows / few rows. One output element per
    # lane, KW contiguous loads.
    pid = ext.program_id(0)
    q = pid * BLOCK + tl.arange(0, BLOCK)
    mask = q < total
    ow = q % OW
    nc = q // OW
    base = nc * L + ow * KW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kw in tl.static_range(KW):
        acc += _load_f32_1d(input + base + kw, mask, BF16)
    _store_f32_1d(output + q, acc / AREA, mask, BF16)


@libentry()
@triton.jit
def _adaptive_avg_pool1d_general_kernel(
    input,
    output,
    total,
    L: tl.constexpr,
    OW: tl.constexpr,
    MAX_KW: tl.constexpr,
    BLOCK: tl.constexpr,
    BF16: tl.constexpr,
):
    # Non-exact (true adaptive) path. One output element per lane; each lane
    # computes its own adaptive window bounds and sums with a static unroll.
    pid = ext.program_id(0)
    # ext.program_id returns int64 on XPU; cast to int32 so the window/index
    # arithmetic below stays int32 (mixing int64 with int32 literals corrupts it).
    q = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int32)
    mask = q < total
    ow = q % OW
    nc = q // OW
    iw_start = (ow * L) // OW
    iw_end = ((ow + 1) * L + OW - 1) // OW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kw in tl.static_range(MAX_KW):
        iw = iw_start + kw
        active = mask & (iw < iw_end)
        safe_iw = tl.minimum(iw, iw_end - 1)
        v = _load_f32_1d(input + nc * L + safe_iw, mask, BF16)
        acc += tl.where(active, v, 0.0)
    area = (iw_end - iw_start).to(tl.float32)
    _store_f32_1d(output + q, acc / area, mask, BF16)


def adaptive_avg_pool1d(input: torch.Tensor, output_size):
    """Adaptive average pooling over the innermost 1D axis.

    The generic implementation delegates to ``flag_gems.ops.adaptive_avg_pool2d``
    (the generic 2D kernel), which on XPU compiles to a 2D gather load that
    mis-maps lanes for non-divisible windows (and mis-maps bf16 gather loads in
    general), producing wrong values. This override computes the 1D reduction
    directly with a flat per-lane kernel (exact windows) and a flat adaptive
    kernel (non-exact windows), both with the bf16 int16-bit reconstruction that
    avoids the XPU3 strided/gather de-interleave bug.
    """
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_AVG_POOL1D")

    assert input.ndim == 3, f"adaptive_avg_pool1d expects 3D input, got {input.ndim}D"

    if isinstance(output_size, int):
        out_size = output_size
    else:
        out_size = output_size[0]

    input_contiguous = input.contiguous()
    n, c, lin = input_contiguous.shape
    output = torch.empty((n, c, out_size), dtype=input.dtype, device=input.device)
    if output.numel() == 0 or lin == 0:
        return output

    is_bf16 = input_contiguous.dtype == torch.bfloat16
    flat_input = input_contiguous.view(torch.int16) if is_bf16 else input_contiguous
    flat_output = output.view(torch.int16) if is_bf16 else output

    with torch_device_fn.device(input.device):
        if lin % out_size == 0:
            kw = lin // out_size
            _adaptive_avg_pool1d_flat_kernel[
                (triton.cdiv(output.numel(), FLAT_BLOCK),)
            ](
                flat_input,
                flat_output,
                output.numel(),
                L=lin,
                OW=out_size,
                KW=kw,
                AREA=kw,
                BLOCK=FLAT_BLOCK,
                BF16=is_bf16,
            )
        else:
            max_kw = triton.cdiv(lin, out_size) + 1
            _adaptive_avg_pool1d_general_kernel[
                (triton.cdiv(output.numel(), FLAT_BLOCK),)
            ](
                flat_input,
                flat_output,
                output.numel(),
                L=lin,
                OW=out_size,
                MAX_KW=max_kw,
                BLOCK=FLAT_BLOCK,
                BF16=is_bf16,
            )
    return output
