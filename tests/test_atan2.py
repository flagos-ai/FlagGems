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


@pytest.mark.atan2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.SMALL_INT_DTYPES)
def test_atan2(shape, dtype):
    # Integer inputs are promoted to float32 by atan2, matching ATen; they are
    # built on CPU since GPU randn has no integer implementation.
    if dtype in utils.SMALL_INT_DTYPES:
        x = utils.randint_in_range(shape, dtype).to(flag_gems.device)
        y = utils.randint_in_range(shape, dtype).to(flag_gems.device)
    else:
        x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.atan2(ref_x, ref_y)

    res_out = flag_gems.atan2(x, y)

    utils.gems_assert_close(
        res_out, ref_out, utils.promoted_float_dtype(dtype, "atan2")
    )


@pytest.mark.atan2_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atan2_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.atan2.out(ref_x, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.atan2_out(x, y, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.atan2_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.SMALL_INT_DTYPES + utils.BOOL_TYPES)
def test_atan2_out_integer_rejects(shape, dtype):
    # ATen raises for an integer/bool out buffer; FlagGems mirrors that instead
    # of silently truncating the float result.
    x = utils.make_reduction_input(shape, dtype)
    y = utils.make_reduction_input(shape, dtype)
    out = utils.make_reduction_input(shape, dtype)
    with pytest.raises(RuntimeError):
        flag_gems.atan2_out(x, y, out=out)


# ---------------------------------------------------------------------------
# Domain-matrix additions: value-range coverage ([0,1], [-1,0], full domain and
# the quadrant mix) crossed with shapes and dtypes.
# ---------------------------------------------------------------------------

ATAN2_VALUE_RANGES = [
    "full_domain",
    "positive",
    "negative",
    "large_values",
    "small_values",
]


def _atan2_range_input(shape, dtype, range_name):
    """Build inputs for the atan2 quadrant matrix. atan2 is defined on the
    whole plane, so the out-of-domain bands of asin/atanh are replaced by
    large/small magnitude bands."""
    if range_name == "full_domain":
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if range_name == "positive":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            0.0, 1.0
        )
    if range_name == "negative":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            -1.0, 0.0
        )
    if range_name == "large_values":
        # fp16 is out of bounds above 65504, so the band is capped there;
        # bf16/fp32 still span [1e3, 1e6].
        hi = 65000.0 if dtype == torch.float16 else 1e6
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            1e3, hi
        )
    if range_name == "small_values":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            -1e-3, 1e-3
        )
    raise ValueError(f"Unknown range_name: {range_name}")


@pytest.mark.atan2
@pytest.mark.parametrize("range_name", ATAN2_VALUE_RANGES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atan2_range(shape, dtype, range_name):
    x = _atan2_range_input(shape, dtype, range_name)
    y = _atan2_range_input(shape, dtype, range_name)
    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)

    ref_out = torch.atan2(ref_x, ref_y)
    res_out = flag_gems.atan2(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.atan2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.SMALL_INT_DTYPES)
def test_atan2_integer(shape, dtype):
    # Integer inputs are promoted to float32; boundary values {-2, -1, 0, 1, 2}
    # cover all four quadrants plus the axes.
    x = utils.randint_in_range(shape, dtype, low=-2, high=3).to(flag_gems.device)
    y = utils.randint_in_range(shape, dtype, low=-2, high=3).to(flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.atan2(ref_x, ref_y)

    res_out = flag_gems.atan2(x, y)

    utils.gems_assert_close(
        res_out, ref_out, utils.promoted_float_dtype(dtype, "atan2")
    )


@pytest.mark.atan2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_atan2_all_integer(shape, dtype):
    # Wider integer dtypes (int16/32/64) promote to float32 the same way int8
    # and uint8 do; the window covers all four quadrants.
    x = utils.randint_in_range(shape, dtype, low=-2, high=3).to(flag_gems.device)
    y = utils.randint_in_range(shape, dtype, low=-2, high=3).to(flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.atan2(ref_x, ref_y)

    res_out = flag_gems.atan2(x, y)

    utils.gems_assert_close(
        res_out, ref_out, utils.promoted_float_dtype(dtype, "atan2")
    )


@pytest.mark.atan2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atan2_special_values(dtype):
    # Param coverage type 5: signed zeros, nan and infinities on both operands.
    x = torch.tensor(
        [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1.0, -1.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    y = torch.tensor(
        [0.0, -0.0, 1.0, 1.0, 1.0, float("inf"), -float("inf")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.atan2(ref_x, ref_y)

    res_out = flag_gems.atan2(x, y)

    ref_nan = torch.isnan(ref_out)
    res_nan = torch.isnan(res_out)
    assert torch.equal(res_nan.cpu(), ref_nan.cpu())
    utils.gems_assert_close(res_out[~res_nan], ref_out[~ref_nan], dtype, equal_nan=True)


@pytest.mark.atan2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atan2_boundary_values(dtype):
    # Axes and diagonals: atan2(0, 0) = 0, atan2(+, 0) = pi/2, atan2(-, 0) =
    # -pi/2, atan2(0, +) = 0, atan2(0, -) = pi.
    values = torch.tensor(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_values = utils.to_reference(values, True)

    ref_out = torch.atan2(ref_values, ref_values)
    res_out = flag_gems.atan2(values, values)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.atan2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atan2_noncontiguous(dtype):
    # Param coverage type 5: non-contiguous inputs must produce the same
    # values as their contiguous copies.
    inp = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    noncontig = inp.repeat(2, 1)[::2]
    assert not noncontig.is_contiguous()
    ref_inp = utils.to_reference(noncontig.contiguous(), True)

    ref_out = torch.atan2(ref_inp, ref_inp)
    res_out = flag_gems.atan2(noncontig, noncontig)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.atan2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atan2_zero_size(dtype):
    # Param coverage type 5: 0-size tensors must pass through unchanged.
    for shape in [(0,), (0, 3), (2, 0, 4)]:
        x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

        ref_x = utils.to_reference(x, True)
        ref_y = utils.to_reference(y, True)
        ref_out = torch.atan2(ref_x, ref_y)
        res_out = flag_gems.atan2(x, y)

        utils.gems_assert_close(res_out, ref_out, dtype)
        assert res_out.numel() == 0


@pytest.mark.atan2_out
@pytest.mark.parametrize("range_name", ATAN2_VALUE_RANGES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atan2_out_range(shape, dtype, range_name):
    x = _atan2_range_input(shape, dtype, range_name)
    y = _atan2_range_input(shape, dtype, range_name)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.atan2.out(ref_x, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.atan2_out(x, y, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.atan2_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atan2_out_noncontiguous(dtype):
    inp = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    noncontig = inp.repeat(2, 1)[::2]
    assert not noncontig.is_contiguous()
    ref_inp = utils.to_reference(noncontig.contiguous(), True)

    ref_out_buf = torch.empty(ref_inp.shape, dtype=ref_inp.dtype, device=ref_inp.device)
    ref_out = torch.ops.aten.atan2.out(ref_inp, ref_inp, out=ref_out_buf)

    res_out_buf = torch.empty(noncontig.shape, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.atan2_out(noncontig, noncontig, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.atan2_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atan2_out_zero_size(dtype):
    for shape in [(0,), (0, 3), (2, 0, 4)]:
        x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

        ref_x = utils.to_reference(x, True)
        ref_y = utils.to_reference(y, True)
        ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
        ref_out = torch.ops.aten.atan2.out(ref_x, ref_y, out=ref_out_buf)

        res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        res_out = flag_gems.atan2_out(x, y, out=res_out_buf)

        utils.gems_assert_close(res_out, ref_out, dtype)
        assert res_out.numel() == 0
