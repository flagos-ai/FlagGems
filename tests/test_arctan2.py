import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.arctan2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.arctan2(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.arctan2(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


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
    with flag_gems.use_gems():
        res_out = x1.arctan2_(y)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(x1, ref_x, dtype)
    assert res_out is x1


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

    with flag_gems.use_gems():
        res_out = torch.arctan2(x, y)

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

    with flag_gems.use_gems():
        res_out = torch.arctan2(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)
