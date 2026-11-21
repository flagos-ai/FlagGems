# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SM90 group32 E4M3FN quantization with numeric FP32 UE8M0 scales.

Native-v2 arithmetic with fresh outputs and no CUDA queries at import.
Scales are numeric FP32 powers of two, not packed exponent bytes.
"""

import torch
import triton
import triton.language as tl

DENSE_GROUP32_UE8M0_ABI = 1
SUPPORTED_K = frozenset((288, 1024, 1280, 5120, 6144))


@triton.jit
def _quant32_ue8_kernel(X, Q, S, GROUPS, BG: tl.constexpr):
    group = tl.program_id(0) * BG + tl.arange(0, BG)
    lane = tl.arange(0, 32)
    value = tl.load(
        X + group[:, None] * 32 + lane[None, :], group[:, None] < GROUPS, 0.0
    ).to(tl.float32)
    # Native fmaxf ignores NaN; its initial per-thread maximum is 1e-10f.
    absolute = tl.where(value == value, tl.abs(value), 0.0)
    amax = tl.maximum(tl.max(absolute, axis=1), 1e-10)
    # C++ constexpr float MAX_8BIT_INV = 1.0f / 448.0f.  Do not replace
    # multiply-by-rounded-reciprocal with a division or approximate log2.
    scaled = amax * (1.0 / 448.0)
    bits = scaled.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 255).to(tl.int32) - 127
    exponent += ((bits & 0x7FFFFF) != 0).to(tl.int32)
    dequant_bits = (exponent + 127).to(tl.uint32) << 23
    multiplier_bits = (127 - exponent).to(tl.uint32) << 23
    dequant = dequant_bits.to(tl.float32, bitcast=True)
    multiplier = multiplier_bits.to(tl.float32, bitcast=True)
    quant_value = value * multiplier[:, None]
    # Native fminf(fmaxf(v,-448),448) maps NaN to -448.  Explicit where
    # avoids depending on a compiler-specific NaN propagation default.
    quant_value = tl.where(quant_value == quant_value, quant_value, -448.0)
    quant_value = tl.minimum(tl.maximum(quant_value, -448.0), 448.0)
    quantized = quant_value.to(tl.float8e4nv)
    tl.store(
        Q + group[:, None] * 32 + lane[None, :], quantized, group[:, None] < GROUPS
    )
    tl.store(S + group, dequant, group < GROUPS)


def supports(x):
    return (
        isinstance(x, torch.Tensor)
        and x.ndim == 2
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and 0 < x.shape[0] <= 32768
        and x.shape[1] in SUPPORTED_K
        and x.stride() == (x.shape[1], 1)
        and x.data_ptr() % 16 == 0
        and torch.cuda.get_device_capability(x.device) == (9, 0)
    )


def launch(x):
    """Fresh output allocations; caller must already enforce supports()."""
    q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (x.shape[0], x.shape[1] // 32), device=x.device, dtype=torch.float32
    )
    groups = x.numel() // 32
    _quant32_ue8_kernel[(triton.cdiv(groups, 64),)](
        x, q, scales, groups, 64, num_warps=4, num_stages=1
    )
    return q, scales


def quantize_group32_ue8m0(x):
    if not supports(x):
        raise ValueError("expected a contiguous supported SM90 BF16 matrix")
    return launch(x)


__all__ = ["DENSE_GROUP32_UE8M0_ABI", "supports", "launch", "quantize_group32_ue8m0"]
