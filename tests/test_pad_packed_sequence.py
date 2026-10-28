import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _make_batch_sizes(lengths):
    # lengths: per-batch sequence lengths, sorted descending.
    # batch_sizes[t] = number of sequences still active at time step t.
    max_len = max(lengths)
    lengths_t = torch.tensor(lengths, dtype=torch.int64)
    steps = torch.arange(max_len).unsqueeze(1)
    batch_sizes = (lengths_t.unsqueeze(0) > steps).sum(dim=1).to(torch.int64)
    return batch_sizes


@pytest.mark.pad_packed_sequence
@pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
@pytest.mark.parametrize("max_length", [8, 16, 32, 128])
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pad_packed_sequence(batch_size, max_length, batch_first, dtype):
    # Descending-sorted sequence lengths, as required by the packed layout.
    np.random.seed(42)
    lengths = np.random.randint(1, max_length + 1, size=batch_size).tolist()
    lengths = sorted(lengths, reverse=True)

    batch_sizes = _make_batch_sizes(lengths)
    total = int(batch_sizes.sum().item())
    data = torch.randn(total, 16, dtype=dtype, device=flag_gems.device)

    ref_data = utils.to_reference(data)

    ref_out, ref_lengths = torch.ops.aten._pad_packed_sequence(
        ref_data, batch_sizes, batch_first, 0.0, max_length
    )

    res_out, res_lengths = flag_gems._pad_packed_sequence(
        data, batch_sizes, batch_first, 0.0, max_length
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    # `lengths` is returned on CPU by aten for both paths.
    assert torch.equal(res_lengths, ref_lengths)


@pytest.mark.pad_packed_sequence
@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("max_length", [16, 32])
@pytest.mark.parametrize("padding_value", [0.0, -1.0, 1.5])
def test_pad_packed_sequence_padding(batch_size, max_length, padding_value):
    np.random.seed(42)
    lengths = np.random.randint(1, max_length + 1, size=batch_size).tolist()
    lengths = sorted(lengths, reverse=True)

    batch_sizes = _make_batch_sizes(lengths)
    total = int(batch_sizes.sum().item())
    data = torch.randn(total, 16, dtype=torch.float32, device=flag_gems.device)

    ref_data = utils.to_reference(data)

    ref_out, ref_lengths = torch.ops.aten._pad_packed_sequence(
        ref_data, batch_sizes, False, padding_value, max_length
    )

    res_out, res_lengths = flag_gems._pad_packed_sequence(
        data, batch_sizes, False, padding_value, max_length
    )

    utils.gems_assert_close(res_out, ref_out, torch.float32)
    # `lengths` is returned on CPU by aten for both paths.
    assert torch.equal(res_lengths, ref_lengths)
