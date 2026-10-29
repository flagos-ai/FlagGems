import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _special_round_storage_overlaps(left, right):
    if left.device != right.device:
        return False
    left_storage = left.untyped_storage()
    right_storage = right.untyped_storage()
    if left_storage._cdata == right_storage._cdata:
        return True
    left_bytes = left_storage.nbytes()
    right_bytes = right_storage.nbytes()
    if left_bytes == 0 or right_bytes == 0:
        return False
    left_start = left_storage.data_ptr()
    right_start = right_storage.data_ptr()
    return (
        left_start < right_start + right_bytes and right_start < left_start + left_bytes
    )


def _special_round_snapshot(tensor):
    metadata = (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tensor.stride(),
        tensor.storage_offset(),
        tensor.untyped_storage()._cdata,
        tensor.data_ptr(),
        tensor.requires_grad,
        tensor.is_conj(),
        tensor.is_neg(),
        tensor.untyped_storage().nbytes(),
    )
    logical = torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
        tensor.untyped_storage().cpu(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tensor.stride(),
    )
    return metadata, logical.contiguous().reshape(-1).view(torch.uint8).clone()


def _special_round_assert_unchanged(tensor, snapshot, *, values=True):
    metadata, payload = _special_round_snapshot(tensor)
    assert metadata == snapshot[0]
    if values:
        assert torch.equal(payload, snapshot[1])


def _call_special_round_checked(inp, **kwargs):
    original = _special_round_snapshot(inp)
    try:
        result = flag_gems.special_round(inp, **kwargs)
    except BaseException:
        _special_round_assert_unchanged(inp, original)
        raise
    _special_round_assert_unchanged(inp, original)
    assert isinstance(result, torch.Tensor)
    assert result.device == original[0][0] and result.dtype == original[0][1]
    assert tuple(result.shape) == original[0][2]
    assert result is not inp
    assert not _special_round_storage_overlaps(
        result, inp
    ), "candidate output aliases input storage"
    return result


@pytest.mark.special_round
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_round(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.round(ref_inp)
    res_out = _call_special_round_checked(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="enflame does not support fp64",
)
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
    res_out = _call_special_round_checked(inp)

    utils.gems_assert_equal(res_out, ref_out)

    # Also verify the expected values match
    expected_tensor = torch.tensor(
        expected_values, dtype=torch.float32, device=flag_gems.device
    )
    utils.gems_assert_equal(res_out, utils.to_reference(expected_tensor))


@pytest.mark.special_round
@pytest.mark.parametrize("decimals", [0, 1, 2, 3, -1, -2])
def test_special_round_decimals(decimals):
    inp = torch.tensor(
        [1.2345, 2.5678, -3.4567, 4.0, 0.0, -0.0, 0.5, -0.5],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.round(ref_inp, decimals=decimals)
    res_out = _call_special_round_checked(inp, decimals=decimals)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="enflame does not support fp64",
)
@pytest.mark.special_round_out
@pytest.mark.parametrize("decimals", [0, 1, 2, 3, -1, -2])
def test_special_round_out_decimals(decimals):
    inp = torch.tensor(
        [1.2345, 2.5678, -3.4567, 4.0, 0.0, -0.0, 0.5, -0.5],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.round(ref_inp, decimals=decimals)
    with flag_gems.use_gems():
        res_out = torch.special.round(inp, decimals=decimals, out=out)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(out, ref_out)
