# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest

from .int8_indexer_exact_oracle import (
    FLOOR_F32_BITS,
    FORMATS,
    ceil_pow2_bits,
    div127_rn_bits,
    expected_row,
    fixture_cases,
    ratio,
    rn_integer,
    round_binary_bits,
)


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_round_even_midpoints_across_exponents(fmt):
    eb, mb = FORMATS[fmt]
    bias = (1 << (eb - 1)) - 1
    for exponent in (1 - bias, -10, -1, 0, 1, bias - 1):
        if not 1 - bias <= exponent <= bias:
            continue
        for mantissa in (0, 1, 2, (1 << mb) - 2):
            lower = ((exponent + bias) << mb) | mantissa
            upper = lower + 1
            ln, ld = ratio(lower, fmt)
            un, ud = ratio(upper, fmt)
            midpoint_n, midpoint_d = ln * ud + un * ld, 2 * ld * ud
            assert round_binary_bits(midpoint_n, midpoint_d, fmt) == (
                lower if lower % 2 == 0 else upper
            )
            assert round_binary_bits(-midpoint_n, midpoint_d, fmt) == (
                (lower if lower % 2 == 0 else upper) | (1 << (eb + mb))
            )
    # Zero/min-subnormal tie rounds to even zero.
    n, d = ratio(1, fmt)
    assert round_binary_bits(n, d * 2, fmt) == 0


def test_integer_signed_halfway_parities():
    assert [rn_integer(n, 2) for n in (-253, -5, -3, -1, 1, 3, 5, 253)] == [
        -126,
        -2,
        -2,
        0,
        0,
        2,
        2,
        126,
    ]


def test_floor_order_zero_and_full_finite_range():
    assert expected_row([0] * 128)[1] == (127 - 40) << 23
    low = div127_rn_bits(FLOOR_F32_BITS)
    high = div127_rn_bits(0x7F7FFFFF)
    assert (low >> 23) & 255 == 86
    assert (high >> 23) & 255 == 248
    assert ceil_pow2_bits(low) >> 23 == 87
    assert ceil_pow2_bits(high) >> 23 == 249


def test_bitceil_against_exact_rational_inequalities():
    for exponent in range(86, 249):
        for mantissa in (0, 1, 2, 0x3FFFFF, 0x7FFFFF):
            raw = exponent << 23 | mantissa
            scale = ceil_pow2_bits(raw)
            rn, rd = ratio(raw)
            sn, sd = ratio(scale)
            assert rn * sd <= sn * rd
            if raw != scale:
                assert rn * sd * 2 > sn * rd
            else:
                assert mantissa == 0


def test_division_rn_not_rounded_reciprocal_multiply():
    # 127*2**e has an exact quotient. One FP32 step above it has a /127
    # quotient just above the RN half-ULP boundary, so precise division matters.
    reciprocal = round_binary_bits(1, 127)
    reciprocal_n, reciprocal_d = ratio(reciprocal)
    failures = 0
    for exponent in range(-40, 122):
        n, d = (127 << exponent, 1) if exponent >= 0 else (127, 1 << -exponent)
        center = round_binary_bits(n, d)
        assert div127_rn_bits(center) == ((exponent + 127) << 23)
        x = center + 1
        xn, xd = ratio(x)
        precise = div127_rn_bits(x)
        approximate = round_binary_bits(xn * reciprocal_n, xd * reciprocal_d)
        assert precise == ((exponent + 127) << 23) + 1
        if ceil_pow2_bits(precise) != ceil_pow2_bits(approximate):
            failures += 1
    assert failures == 162


def test_adversarial_fixture_range_and_signs():
    cases = fixture_cases()
    assert {c["dtype"] for c in cases} == set(FORMATS)
    assert len(cases) > 1000
    for case in cases:
        q, s, raw = expected_row(case["input_bits"], case["dtype"])
        assert q == case["expected_q"] and s == case["expected_scale_bits"]
        assert all(-127 <= value <= 127 for value in q)
        assert q[0] == -q[1]
        assert 86 <= ((raw >> 23) & 255) <= 248


@pytest.mark.parametrize("dtype_name", list(FORMATS))
def test_exact_oracle_agrees_with_unchanged_math_reference(dtype_name):
    import torch

    from .int8_indexer_exact_oracle import adversarial_tensors
    from .int8_indexer_reference import quantize_reference

    inputs, expected_q, expected_scales = adversarial_tensors(dtype_name)
    actual_q, actual_scales = quantize_reference(inputs)
    torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
    assert torch.equal(
        actual_scales.view(torch.int32), expected_scales.view(torch.int32)
    )
    reconstructed = expected_q.double() * expected_scales.double()
    error = (reconstructed - inputs.double()).abs()
    assert torch.all(error <= expected_scales.double() * 0.5)
