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


@pytest.mark.asin
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.SMALL_INT_DTYPES)
def test_asin(shape, dtype):
    # asin promotes integer inputs to float32, matching ATen. Integers only hit
    # {-1, 0, 1} of the valid domain, so -2/2 are kept to cover the out-of-domain
    # NaN edges as well.
    if dtype in utils.SMALL_INT_DTYPES:
        res_inp = utils.randint_in_range(shape, dtype, low=-2, high=3).to(
            flag_gems.device
        )
    else:
        res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.asin(ref_inp)
    res_out = flag_gems.asin(res_inp)
    out_dtype = utils.promoted_float_dtype(dtype, "asin")
    ref_out = ref_out.to(out_dtype)

    utils.gems_assert_close(res_out, ref_out, out_dtype, equal_nan=True)


@pytest.mark.asin_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.asin_(ref_inp)
    res_out = flag_gems.asin_(inp)

    ref_out = ref_out.to(res_out.dtype)
    ref_inp = ref_inp.to(inp.dtype)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(inp, ref_inp, dtype, equal_nan=True)


# ---------------------------------------------------------------------------
# Domain-matrix additions: value-range coverage ([-1,1], [0,1], [-1,0] and the
# out-of-domain edges beyond both bounds) crossed with shapes and dtypes.
# ---------------------------------------------------------------------------

ASIN_VALUE_RANGES = ["full_domain", "positive", "negative", "above_max", "below_min"]


def _asin_range_input(shape, dtype, range_name):
    """Build inputs for the asin domain matrix: valid domain [-1, 1] plus the
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


@pytest.mark.asin
@pytest.mark.parametrize("range_name", ASIN_VALUE_RANGES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin_range(shape, dtype, range_name):
    inp = _asin_range_input(shape, dtype, range_name)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.asin(ref_inp)
    res_out = flag_gems.asin(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.asin
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.SMALL_INT_DTYPES)
def test_asin_integer(shape, dtype):
    # Integer inputs are promoted to float32; values outside {-1, 0, 1} are
    # out-of-domain for asin and produce NaN on both sides.
    inp = utils.randint_in_range(shape, dtype, low=-2, high=3).to(flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.asin(ref_inp)
    res_out = flag_gems.asin(inp)

    utils.gems_assert_close(
        res_out, ref_out, utils.promoted_float_dtype(dtype, "asin"), equal_nan=True
    )


@pytest.mark.asin
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin_boundary_values(dtype):
    # Exact endpoints and near-endpoint values: asin(+-1) = +-pi/2, asin(0) = 0,
    # and |x| > 1 must be NaN.
    values = torch.tensor(
        [-1.0, -0.999, -0.5, 0.0, 0.5, 0.999, 1.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_values = utils.to_reference(values, True)

    ref_out = torch.asin(ref_values)
    res_out = flag_gems.asin(values)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.asin
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_asin_all_integer(shape, dtype):
    # Wider integer dtypes (int16/32/64) promote to float32 the same way int8
    # and uint8 do; the value window stays inside the valid domain plus its
    # immediately out-of-domain neighbours.
    inp = utils.randint_in_range(shape, dtype, low=-2, high=3).to(flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.asin(ref_inp)
    res_out = flag_gems.asin(inp)

    utils.gems_assert_close(
        res_out, ref_out, utils.promoted_float_dtype(dtype, "asin"), equal_nan=True
    )


@pytest.mark.asin
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin_special_values(dtype):
    # Param coverage type 5: signed zero, nan and the infinities. asin(+0) is
    # +0, asin(-0) is -0, and nan/inf map to nan.
    values = torch.tensor(
        [0.0, -0.0, float("nan"), float("inf"), -float("inf")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_values = utils.to_reference(values, True)

    ref_out = torch.asin(ref_values)
    res_out = flag_gems.asin(values)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    # Only the zero lanes carry a meaningful sign bit; NaN sign bits are
    # platform-defined, so they are not compared.
    assert torch.equal(
        torch.signbit(res_out[:2].cpu()), torch.signbit(ref_out[:2].cpu())
    )


@pytest.mark.asin
@pytest.mark.parametrize("shape", utils.POINTWISE_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin_noncontiguous(shape, dtype):
    # Param coverage type 5: non-contiguous inputs must produce the same
    # values as their contiguous copies. Shapes with any 1-sized dim give
    # a contiguous result ( stride-1 squeeze), and 0-dim has no stride at
    # all, so both are excluded from the shape list.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noncontig = inp.repeat(2, *[1] * (inp.ndim - 1))[::2]
    assert not noncontig.is_contiguous()
    ref_inp = utils.to_reference(noncontig.contiguous(), True)

    ref_out = torch.asin(ref_inp)
    res_out = flag_gems.asin(noncontig)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.asin
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin_zero_size(dtype):
    # Param coverage type 5: 0-size tensors must pass through unchanged.
    for shape in [(0,), (0, 3), (2, 0, 4)]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)

        ref_out = torch.asin(ref_inp)
        res_out = flag_gems.asin(inp)

        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
        assert res_out.numel() == 0


@pytest.mark.asin_
@pytest.mark.parametrize("range_name", ASIN_VALUE_RANGES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin__range(shape, dtype, range_name):
    inp = _asin_range_input(shape, dtype, range_name)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.asin_(ref_inp)
    res_out = flag_gems.asin_(inp)

    ref_out = ref_out.to(res_out.dtype)
    ref_inp = ref_inp.to(inp.dtype)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(inp, ref_inp, dtype, equal_nan=True)


@pytest.mark.asin_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin__inplace_identity(shape, dtype):
    # In-place contract: the returned tensor is the input object itself.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.asin_(inp)
    assert res_out is inp


@pytest.mark.asin_
@pytest.mark.parametrize("shape", utils.POINTWISE_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin__noncontiguous(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noncontig = inp.repeat(2, *[1] * (inp.ndim - 1))[::2]
    assert not noncontig.is_contiguous()
    ref_inp = utils.to_reference(noncontig.contiguous().clone(), True)

    ref_out = torch.asin_(ref_inp)
    res_out = flag_gems.asin_(noncontig)

    ref_out = ref_out.to(res_out.dtype)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.asin_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_asin__zero_size(dtype):
    for shape in [(0,), (0, 3)]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        res_out = flag_gems.asin_(inp)
        assert res_out.numel() == 0
        assert res_out.dtype == dtype
