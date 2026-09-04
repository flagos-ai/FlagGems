import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _assert_nested_close(res, ref, dtype):
    assert res.is_nested
    assert ref.is_nested

    res_unbind = torch.unbind(res)
    ref_unbind = torch.unbind(ref)

    assert len(res_unbind) == len(ref_unbind)
    for res_t, ref_t in zip(res_unbind, ref_unbind):
        assert res_t.shape == ref_t.shape
        utils.gems_assert_close(res_t, ref_t, dtype)


@pytest.mark.nested_view_from_jagged_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_view_from_jagged_copy_2d(dtype):
    total = 64
    inner_dim = 8
    values = torch.randn(total, inner_dim, dtype=dtype, device=flag_gems.device)
    offsets = torch.tensor(
        [0, 5, 17, 33, 64], dtype=torch.int64, device=flag_gems.device
    )
    dummy = torch.randn(
        offsets.numel() - 1, inner_dim, dtype=dtype, device=flag_gems.device
    )

    ref_values = utils.to_reference(values)
    ref_offsets = utils.to_reference(offsets)

    ref_out = torch.nested.nested_tensor_from_jagged(
        ref_values.clone(), offsets=ref_offsets, jagged_dim=1
    )

    res_out = flag_gems._nested_view_from_jagged_copy(values, offsets, dummy)

    _assert_nested_close(res_out, ref_out, dtype)


@pytest.mark.nested_view_from_jagged_copy
def test_nested_view_from_jagged_copy_1d():
    dtype = torch.float32
    values = torch.randn(12, dtype=dtype, device=flag_gems.device)
    offsets = torch.tensor([0, 3, 5, 12], dtype=torch.int64, device=flag_gems.device)
    dummy = torch.randn(offsets.numel() - 1, dtype=dtype, device=flag_gems.device)

    ref_values = utils.to_reference(values)
    ref_offsets = utils.to_reference(offsets)

    ref_out = torch.nested.nested_tensor_from_jagged(
        ref_values.clone(), offsets=ref_offsets, jagged_dim=1
    )

    res_out = flag_gems._nested_view_from_jagged_copy(values, offsets, dummy)

    _assert_nested_close(res_out, ref_out, dtype)


@pytest.mark.nested_view_from_jagged_copy
def test_nested_view_from_jagged_copy_lengths():
    dtype = torch.float32
    values = torch.arange(12, dtype=dtype, device=flag_gems.device).reshape(6, 2)
    offsets = torch.tensor([0, 2, 3, 6], dtype=torch.int64, device=flag_gems.device)
    lengths = torch.tensor([1, 1, 2], dtype=torch.int64, device=flag_gems.device)
    dummy = torch.randn(3, 2, dtype=dtype, device=flag_gems.device)

    ref_values = utils.to_reference(values)
    ref_offsets = utils.to_reference(offsets)
    ref_lengths = utils.to_reference(lengths)

    ref_out = torch.nested.nested_tensor_from_jagged(
        ref_values.clone(), offsets=ref_offsets, lengths=ref_lengths, jagged_dim=1
    )

    res_out = flag_gems._nested_view_from_jagged_copy(
        values, offsets, dummy, lengths, 1
    )

    _assert_nested_close(res_out, ref_out, dtype)


@pytest.mark.nested_view_from_jagged_copy
def test_nested_view_from_jagged_copy_min_seqlen():
    dtype = torch.float32
    values = torch.randn(8, 3, dtype=dtype, device=flag_gems.device)
    offsets = torch.tensor([0, 2, 5, 8], dtype=torch.int64, device=flag_gems.device)
    dummy = torch.randn(3, 3, dtype=dtype, device=flag_gems.device)
    # Torch encodes min_seqlen / max_seqlen as torch.zeros(val, 0) tensors.
    min_seqlen = torch.zeros(1, 0, device=flag_gems.device)
    max_seqlen = torch.zeros(3, 0, device=flag_gems.device)

    ref_values = utils.to_reference(values)
    ref_offsets = utils.to_reference(offsets)

    ref_out = torch.nested.nested_tensor_from_jagged(
        ref_values.clone(),
        offsets=ref_offsets,
        jagged_dim=1,
        min_seqlen=1,
        max_seqlen=3,
    )

    res_out = flag_gems._nested_view_from_jagged_copy(
        values, offsets, dummy, None, 1, min_seqlen, max_seqlen
    )

    _assert_nested_close(res_out, ref_out, dtype)
