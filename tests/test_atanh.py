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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.atanh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.SMALL_INT_DTYPES)
def test_atanh(shape, dtype):
    # Generate input in (-1, 1) range for valid atanh values. Integer inputs
    # only admit 0 inside that range, so the {-1, 0, 1} window is used instead
    # to also cover the +-inf and NaN edges that ATen produces outside it;
    # atanh promotes integer inputs to float32, matching ATen.
    if dtype in utils.SMALL_INT_DTYPES:
        inp = utils.randint_in_range(shape, dtype, low=-1, high=2).to(flag_gems.device)
    else:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            -0.99, 0.99
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.atanh(ref_inp)
    res_out = flag_gems.atanh(inp)

    utils.gems_assert_close(
        res_out, ref_out, utils.promoted_float_dtype(dtype, "atanh"), equal_nan=True
    )


# ---------------------------------------------------------------------------
# Domain-matrix additions: value-range coverage ([-1,1], [0,1], [-1,0] and the
# out-of-domain edges beyond both bounds) crossed with shapes and dtypes.
# ---------------------------------------------------------------------------

ATANH_VALUE_RANGES = ["full_domain", "positive", "negative", "above_max", "below_min"]


def _atanh_range_input(shape, dtype, range_name):
    """Build inputs for the atanh domain matrix: valid domain (-1, 1) plus the
    out-of-domain bands beyond both endpoints (ATen yields NaN there)."""
    if range_name == "full_domain":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            -1.0, 1.0
        )
    if range_name == "positive":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            0.0, 1.0
        )
    if range_name == "negative":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            -1.0, 0.0
        )
    if range_name == "above_max":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            1.5, 4.0
        )
    if range_name == "below_min":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            -4.0, -1.5
        )
    raise ValueError(f"Unknown range_name: {range_name}")


@pytest.mark.atanh
@pytest.mark.parametrize("range_name", ATANH_VALUE_RANGES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atanh_range(shape, dtype, range_name):
    inp = _atanh_range_input(shape, dtype, range_name)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.atanh(ref_inp)
    res_out = flag_gems.atanh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.atanh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.SMALL_INT_DTYPES)
def test_atanh_integer(shape, dtype):
    # Integer inputs are promoted to float32; values outside {-1, 0, 1} are
    # out-of-domain for atanh and produce NaN on both sides.
    inp = utils.randint_in_range(shape, dtype, low=-2, high=3).to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.atanh(ref_inp)
    res_out = flag_gems.atanh(inp)

    utils.gems_assert_close(
        res_out, ref_out, utils.promoted_float_dtype(dtype, "atanh"), equal_nan=True
    )


@pytest.mark.atanh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_atanh_all_integer(shape, dtype):
    # Wider integer dtypes (int16/32/64) promote to float32 the same way int8
    # and uint8 do; values outside {-1, 0, 1} are out-of-domain.
    inp = utils.randint_in_range(shape, dtype, low=-2, high=3).to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.atanh(ref_inp)
    res_out = flag_gems.atanh(inp)

    utils.gems_assert_close(
        res_out, ref_out, utils.promoted_float_dtype(dtype, "atanh"), equal_nan=True
    )


@pytest.mark.atanh
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atanh_special_values(dtype):
    # Param coverage type 5: zero (with sign preserved), nan and the infinities.
    values = torch.tensor(
        [0.0, -0.0, float("nan"), float("inf"), -float("inf")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_values = utils.to_reference(values)

    ref_out = torch.atanh(ref_values)
    res_out = flag_gems.atanh(values)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.atanh
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atanh_boundary_values(dtype):
    # Exact endpoints and near-endpoint values: atanh(+-1) = +-inf, atanh(0) = 0,
    # and |x| > 1 must be NaN.
    values = torch.tensor(
        [-1.0, -0.999, -0.5, 0.0, 0.5, 0.999, 1.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_values = utils.to_reference(values)

    ref_out = torch.atanh(ref_values)
    res_out = flag_gems.atanh(values)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.atanh
@pytest.mark.parametrize("shape", utils.POINTWISE_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atanh_noncontiguous(shape, dtype):
    # Param coverage type 5: non-contiguous inputs must produce the same
    # values as their contiguous copies. Shapes with any 1-sized dim give
    # a contiguous result ( stride-1 squeeze), and 0-dim has no stride at
    # all, so both are excluded from the shape list.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noncontig = inp.repeat(2, *[1] * (inp.ndim - 1))[::2]
    assert not noncontig.is_contiguous()
    ref_inp = utils.to_reference(noncontig.contiguous(), True)

    ref_out = torch.atanh(ref_inp)
    res_out = flag_gems.atanh(noncontig)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.atanh
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atanh_zero_size(dtype):
    # Param coverage type 5: 0-size tensors must pass through unchanged.
    for shape in [(0,), (0, 3), (2, 0, 4)]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)

        ref_out = torch.atanh(ref_inp)
        res_out = flag_gems.atanh(inp)

        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
        assert res_out.numel() == 0


@pytest.mark.atanh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.SMALL_INT_DTYPES + utils.BOOL_TYPES)
def test_atanh__integer_rejects(shape, dtype):
    # ATen raises for in-place atanh on integer/bool inputs; FlagGems mirrors
    # that instead of silently truncating.
    inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.atanh_(inp)
