import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.arctan2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.SMALL_INT_DTYPES)
def test_arctan2(shape, dtype):
    # Integer inputs are promoted to float32 by arctan2, matching ATen.
    if dtype in utils.SMALL_INT_DTYPES:
        x = utils.randint_in_range(shape, dtype).to(flag_gems.device)
        y = utils.randint_in_range(shape, dtype).to(flag_gems.device)
    else:
        x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.arctan2(ref_x, ref_y)

    res_out = flag_gems.arctan2(x, y)

    utils.gems_assert_close(
        res_out, ref_out, utils.promoted_float_dtype(dtype, "arctan2")
    )


@pytest.mark.arctan2_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2_(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = ref_x.arctan2_(ref_y)

    x1 = x.clone()
    res_out = flag_gems.arctan2_(x1, y)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(x1, ref_x, dtype)
    assert res_out is x1


@pytest.mark.arctan2_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.SMALL_INT_DTYPES + utils.BOOL_TYPES)
def test_arctan2__integer_rejects(shape, dtype):
    # ATen raises for in-place arctan2 on integer/bool inputs; FlagGems mirrors
    # that instead of silently truncating.
    x = utils.make_reduction_input(shape, dtype)
    y = utils.make_reduction_input(shape, dtype)
    with pytest.raises(RuntimeError):
        flag_gems.arctan2_(x, y)


@pytest.mark.arctan2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2_special_values(dtype):
    x = torch.tensor(
        [
            0.0,
            float("inf"),
            -float("inf"),
            1.0,
            1.0,
            float("nan"),
            1.0,
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    y = torch.tensor(
        [
            0.0,
            1.0,
            1.0,
            float("inf"),
            -float("inf"),
            1.0,
            float("nan"),
        ],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.arctan2(ref_x, ref_y)

    res_out = flag_gems.arctan2(x, y)

    # NaN cannot be compared by the default gems_assert_close,
    # because equal_nan=False by default.
    ref_nan_mask = torch.isnan(ref_out)
    res_nan_mask = torch.isnan(res_out)

    assert torch.equal(res_nan_mask.cpu(), ref_nan_mask.cpu())

    utils.gems_assert_close(
        res_out[~res_nan_mask],
        ref_out[~ref_nan_mask],
        dtype,
    )


@pytest.mark.arctan2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2_broadcast(dtype):
    x = torch.randn((4, 1), dtype=dtype, device=flag_gems.device)
    y = torch.randn((1, 4), dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.arctan2(ref_x, ref_y)

    res_out = flag_gems.arctan2(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Domain-matrix additions: value-range coverage ([0,1], [-1,0], full domain and
# the quadrant mix) crossed with shapes and dtypes.
# ---------------------------------------------------------------------------

ARCTAN2_VALUE_RANGES = [
    "full_domain",
    "positive",
    "negative",
    "large_values",
    "small_values",
]


def _arctan2_range_input(shape, dtype, range_name):
    """Build inputs for the arctan2 quadrant matrix. arctan2 is defined on the
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


@pytest.mark.arctan2
@pytest.mark.parametrize("range_name", ARCTAN2_VALUE_RANGES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2_range(shape, dtype, range_name):
    x = _arctan2_range_input(shape, dtype, range_name)
    y = _arctan2_range_input(shape, dtype, range_name)
    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)

    ref_out = torch.arctan2(ref_x, ref_y)
    res_out = flag_gems.arctan2(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.arctan2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.SMALL_INT_DTYPES)
def test_arctan2_integer(shape, dtype):
    # Integer inputs are promoted to float32; boundary values
    # {-2, -1, 0, 1, 2} cover all four quadrants plus the axes.
    x = utils.randint_in_range(shape, dtype, low=-2, high=3).to(flag_gems.device)
    y = utils.randint_in_range(shape, dtype, low=-2, high=3).to(flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.arctan2(ref_x, ref_y)

    res_out = flag_gems.arctan2(x, y)

    utils.gems_assert_close(
        res_out, ref_out, utils.promoted_float_dtype(dtype, "arctan2")
    )


@pytest.mark.arctan2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_arctan2_all_integer(shape, dtype):
    # Wider integer dtypes (int16/32/64) promote to float32 the same way int8
    # and uint8 do; the window covers all four quadrants.
    x = utils.randint_in_range(shape, dtype, low=-2, high=3).to(flag_gems.device)
    y = utils.randint_in_range(shape, dtype, low=-2, high=3).to(flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.arctan2(ref_x, ref_y)

    res_out = flag_gems.arctan2(x, y)

    utils.gems_assert_close(
        res_out, ref_out, utils.promoted_float_dtype(dtype, "arctan2")
    )


@pytest.mark.arctan2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2_boundary_values(dtype):
    # Axes and diagonals: arctan2(0, 0) = 0, arctan2(+, 0) = pi/2,
    # arctan2(-, 0) = -pi/2, arctan2(0, +) = 0, arctan2(0, -) = pi.
    values = torch.tensor(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_values = utils.to_reference(values, True)

    ref_out = torch.arctan2(ref_values, ref_values)
    res_out = flag_gems.arctan2(values, values)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.arctan2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2_noncontiguous(dtype):
    # Param coverage type 5: non-contiguous inputs must produce the same
    # values as their contiguous copies.
    inp = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    noncontig = inp.repeat(2, 1)[::2]
    assert not noncontig.is_contiguous()
    ref_inp = utils.to_reference(noncontig.contiguous(), True)

    ref_out = torch.arctan2(ref_inp, ref_inp)
    res_out = flag_gems.arctan2(noncontig, noncontig)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.arctan2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2_zero_size(dtype):
    # Param coverage type 5: 0-size tensors must pass through unchanged.
    for shape in [(0,), (0, 3), (2, 0, 4)]:
        x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

        ref_x = utils.to_reference(x, True)
        ref_y = utils.to_reference(y, True)
        ref_out = torch.arctan2(ref_x, ref_y)
        res_out = flag_gems.arctan2(x, y)

        utils.gems_assert_close(res_out, ref_out, dtype)
        assert res_out.numel() == 0


@pytest.mark.arctan2_
@pytest.mark.parametrize("range_name", ARCTAN2_VALUE_RANGES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2__range(shape, dtype, range_name):
    x = _arctan2_range_input(shape, dtype, range_name)
    y = _arctan2_range_input(shape, dtype, range_name)

    ref_x = utils.to_reference(x.clone(), True)
    ref_y = utils.to_reference(y, True)
    ref_out = ref_x.arctan2_(ref_y)

    x1 = x.clone()
    res_out = flag_gems.arctan2_(x1, y)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(x1, ref_x, dtype)
    assert res_out is x1


@pytest.mark.arctan2_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2__noncontiguous(dtype):
    inp = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    noncontig = inp.repeat(2, 1)[::2]
    assert not noncontig.is_contiguous()
    ref_inp = utils.to_reference(noncontig.contiguous().clone(), True)

    ref_out = ref_inp.arctan2_(ref_inp)
    res_out = flag_gems.arctan2_(noncontig, noncontig)

    utils.gems_assert_close(res_out, ref_out, dtype)
