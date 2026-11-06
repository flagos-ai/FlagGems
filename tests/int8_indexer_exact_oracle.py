# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact integer/rational oracle, independent of Torch/Triton floating math.

No production imports. Inputs are IEEE bit patterns; all rounding decisions
use integer quotient/remainder with round-to-nearest, ties-to-even.
"""

import struct
from functools import lru_cache

FORMATS = {"float16": (5, 10), "bfloat16": (8, 7), "float32": (8, 23)}
FLOOR_F32_BITS = 0x2EDBE6FF  # FP32(1e-10), applied to amax BEFORE /127.


def ratio(bits, fmt="float32"):
    exponent_bits, mantissa_bits = FORMATS[fmt]
    sign = -1 if bits >> (exponent_bits + mantissa_bits) else 1
    exponent = (bits >> mantissa_bits) & ((1 << exponent_bits) - 1)
    mantissa = bits & ((1 << mantissa_bits) - 1)
    assert exponent != (1 << exponent_bits) - 1, "finite inputs only"
    bias = (1 << (exponent_bits - 1)) - 1
    if exponent:
        mantissa |= 1 << mantissa_bits
        power = exponent - bias - mantissa_bits
    else:
        power = 1 - bias - mantissa_bits
    return (
        (sign * (mantissa << power), 1)
        if power >= 0
        else (sign * mantissa, 1 << -power)
    )


def rn_integer(numerator, denominator):
    assert denominator > 0
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    twice = remainder * 2
    quotient += twice > denominator or (twice == denominator and quotient & 1)
    return sign * quotient


def _floor_log2_ratio(numerator, denominator):
    exponent = numerator.bit_length() - denominator.bit_length()
    below = (
        numerator < denominator << exponent
        if exponent >= 0
        else numerator << -exponent < denominator
    )
    return exponent - int(below)


def _scaled_rn(numerator, denominator, shift):
    if shift >= 0:
        numerator <<= shift
    else:
        denominator <<= -shift
    return rn_integer(numerator, denominator)


def round_binary_bits(numerator, denominator, fmt="float32"):
    """RN-even exact rational to IEEE binary16/bfloat16/binary32 bits."""
    exponent_bits, mantissa_bits = FORMATS[fmt]
    sign = int(numerator < 0) << (exponent_bits + mantissa_bits)
    numerator = abs(numerator)
    if not numerator:
        return sign
    assert denominator > 0
    bias = (1 << (exponent_bits - 1)) - 1
    exponent = _floor_log2_ratio(numerator, denominator)
    if exponent < 1 - bias:
        significand = _scaled_rn(numerator, denominator, mantissa_bits + bias - 1)
        return sign | significand  # includes rounding into minimum normal
    significand = _scaled_rn(numerator, denominator, mantissa_bits - exponent)
    if significand == 1 << (mantissa_bits + 1):
        significand >>= 1
        exponent += 1
    if exponent > bias:
        return sign | (((1 << exponent_bits) - 1) << mantissa_bits)
    return (
        sign
        | ((exponent + bias) << mantissa_bits)
        | (significand - (1 << mantissa_bits))
    )


def div127_rn_bits(amax_bits):
    numerator, denominator = ratio(amax_bits)
    return round_binary_bits(numerator, denominator * 127)


def ceil_pow2_bits(raw_bits):
    exponent = (raw_bits >> 23) & 255
    mantissa = raw_bits & 0x7FFFFF
    assert 0 < exponent < 254 and raw_bits >> 31 == 0
    return (exponent + int(mantissa != 0)) << 23


def expected_row(input_bits, fmt="float32"):
    f32 = [round_binary_bits(*ratio(bits, fmt)) for bits in input_bits]
    maximum = max((bits & 0x7FFFFFFF for bits in f32), default=0)
    clamped = max(maximum, FLOOR_F32_BITS)
    raw_bits = div127_rn_bits(clamped)
    scale_bits = ceil_pow2_bits(raw_bits)
    scale_numerator, scale_denominator = ratio(scale_bits)
    quantized = []
    for bits in input_bits:
        numerator, denominator = ratio(bits, fmt)
        numerator *= scale_denominator
        denominator *= scale_numerator
        # nearest-even and saturation agree with clamping the quotient first
        quantized.append(max(-127, min(127, rn_integer(numerator, denominator))))
    return quantized, scale_bits, raw_bits


def float32_value(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


@lru_cache(maxsize=1)
def fixture_cases():
    cases = []
    # Every relevant scale exponent, plus values below the amax floor.
    powers = sorted(set(range(-40, 122)) | {-149, -126, -80, -60, -42, -41})
    for fmt, (exponent_bits, mantissa_bits) in FORMATS.items():
        limit = ((1 << exponent_bits) - 1) << mantissa_bits
        sign_mask = 1 << (exponent_bits + mantissa_bits)
        seeds = [
            ("zero", 0),
            ("minsub", 1),
            ("minnormal", 1 << mantissa_bits),
            ("maxfinite", limit - 1),
        ]
        for power in powers:
            numerator, denominator = (
                (127 << power, 1) if power >= 0 else (127, 1 << -power)
            )
            center = round_binary_bits(numerator, denominator, fmt)
            if not 0 < center < limit:
                continue
            seeds += [
                (f"threshold_{power}_{delta:+d}", center + delta)
                for delta in (-1, 0, 1)
                if 0 <= center + delta < limit
            ]
        seen = set()
        for label, peak in seeds:
            if peak in seen:
                continue
            seen.add(peak)
            for negative_first in (False, True):
                row = [
                    peak | (sign_mask if negative_first else 0),
                    peak | (0 if negative_first else sign_mask),
                ] + [0] * 126
                _, scale, _ = expected_row(row, fmt)
                scale_n, scale_d = ratio(scale)
                # ±0.5/1.5/2.5/126.5 test sign and both tie parities. Typed
                # conversion is itself exact RN-even integer arithmetic.
                for offset, twice in enumerate((1, -1, 3, -3, 5, -5, 253, -253), 2):
                    value = round_binary_bits(scale_n * twice, scale_d * 2, fmt)
                    # Preserve the intended row maximum and finite typed data.
                    if (value & (sign_mask - 1)) < limit and (
                        value & (sign_mask - 1)
                    ) <= peak:
                        row[offset] = value
                q, scale, raw = expected_row(row, fmt)
                cases.append(
                    {
                        "id": f"{fmt}:{label}:negative_first={negative_first}",
                        "dtype": fmt,
                        "input_bits": row,
                        "expected_q": q,
                        "expected_scale_bits": scale,
                        "expected_raw_bits": raw,
                    }
                )
    return cases


@lru_cache(maxsize=3)
def adversarial_tensors(dtype_name):
    """Typed CPU inputs and expected bits without candidate arithmetic."""
    import torch

    cases = [case for case in fixture_cases() if case["dtype"] == dtype_name]
    storage_dtype = torch.uint32 if dtype_name == "float32" else torch.uint16
    inputs = torch.tensor(
        [case["input_bits"] for case in cases], dtype=storage_dtype
    ).view(getattr(torch, dtype_name))
    quantized = torch.tensor([case["expected_q"] for case in cases], dtype=torch.int8)
    scales = (
        torch.tensor(
            [case["expected_scale_bits"] for case in cases], dtype=torch.uint32
        )
        .view(torch.float32)
        .reshape(-1, 1)
    )
    return inputs, quantized, scales
