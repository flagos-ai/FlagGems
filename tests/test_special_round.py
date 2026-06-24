import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_round
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_round(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.round(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.round(inp)

    utils.gems_assert_equal(res_out, ref_out)


# torch.special.round_ does not exist as a native PyTorch function.
# Test in-place behavior via torch.special.round(..., out=inp) which
# dispatches to special_round_out with out==inp.
@pytest.mark.special_round_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_round_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.special.round(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.round(inp, out=inp)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp, ref_out)


@pytest.mark.special_round_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_round_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.round(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.round(inp, out=out)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(out, ref_out)


# Round-half-to-even midpoint tests: verify that ties round to the nearest
# even integer (banker's rounding), matching torch.round behavior.
@pytest.mark.special_round
def test_special_round_midpoints():
    # Test values: inputs and expected outputs for round-half-to-even
    test_cases = [
        (0.0, 0.0),
        (-0.0, 0.0),
        (0.5, 0.0),  # tie to even (0)
        (1.5, 2.0),  # tie to even (2)
        (2.5, 2.0),  # tie to even (2)
        (3.5, 4.0),  # tie to even (4)
        (-0.5, 0.0),  # negative tie to even (0)
        (-1.5, -2.0),  # negative tie to even (-2)
        (-2.5, -2.0),  # negative tie to even (-2)
        (-3.5, -4.0),  # negative tie to even (-4)
        (1.0, 1.0),  # integer
        (2.0, 2.0),  # integer
        (-1.0, -1.0),  # negative integer
        (-2.0, -2.0),  # negative integer
        (0.3, 0.0),
        (0.7, 1.0),
        (-0.3, 0.0),
        (-0.7, -1.0),
    ]

    inp_values = [v for v, _ in test_cases]
    expected_values = [v for _, v in test_cases]

    inp = torch.tensor(inp_values, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.round(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.round(inp)

    utils.gems_assert_equal(res_out, ref_out)

    # Also verify the expected values match
    expected_tensor = torch.tensor(
        expected_values, dtype=torch.float32, device=flag_gems.device
    )
    utils.gems_assert_equal(res_out, utils.to_reference(expected_tensor))
