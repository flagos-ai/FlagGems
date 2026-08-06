import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.pack_padded_sequence
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize(
    "time_steps, batch_size, feat_size",
    # (T, B, feature-size) covering short/long sequences and small/large hidden dims
    [(6, 4, 8), (10, 8, 16), (16, 12, 32), (7, 5, 1)],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pack_padded_sequence(time_steps, batch_size, feat_size, batch_first, dtype):
    # lengths must be a 1-D CPU int64 tensor sorted in descending order.
    lengths = torch.linspace(time_steps, 1, steps=batch_size, dtype=torch.int64).clamp(
        min=1
    )
    lengths, _ = torch.sort(lengths, descending=True)

    if batch_first:
        shape = (batch_size, time_steps, feat_size)
    else:
        shape = (time_steps, batch_size, feat_size)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_data, ref_batch_sizes = torch.ops.aten._pack_padded_sequence(
        ref_inp, lengths, batch_first
    )
    with flag_gems.use_gems():
        res_data, res_batch_sizes = torch.ops.aten._pack_padded_sequence(
            inp, lengths, batch_first
        )

    utils.gems_assert_close(res_data, ref_data, dtype)
    utils.gems_assert_equal(res_batch_sizes, ref_batch_sizes)
