# Copyright 2026, The FlagOS Contributors.
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
from flag_gems.ops._pack_padded_sequence import _pack_padded_sequence

from . import accuracy_utils as utils

# (time_steps, batch_size, feature-size) covering short/long sequences, a
# feature size below and above one kernel block, and small/large hidden dims.
PACK_PADDED_SEQUENCE_SHAPES = [
    (6, 4, 8),
    (10, 8, 16),
    (16, 12, 32),
    (7, 5, 1),
    (33, 7, 1025),
    (64, 32, 256),
]


def _make_lengths(time_steps, batch_size):
    # aten requires a 1-D CPU int64 tensor of lengths sorted in descending
    # order, with every length within [1, time_steps].
    lengths = torch.linspace(time_steps, 1, steps=batch_size, dtype=torch.int64).clamp(
        min=1
    )
    lengths, _ = torch.sort(lengths, descending=True)
    return lengths


@pytest.mark.pack_padded_sequence
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize(
    "time_steps, batch_size, feat_size", PACK_PADDED_SEQUENCE_SHAPES
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pack_padded_sequence(time_steps, batch_size, feat_size, batch_first, dtype):
    lengths = _make_lengths(time_steps, batch_size)

    if batch_first:
        shape = (batch_size, time_steps, feat_size)
    else:
        shape = (time_steps, batch_size, feat_size)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_data, ref_batch_sizes = torch.ops.aten._pack_padded_sequence(
        ref_inp, lengths, batch_first
    )
    res_data, res_batch_sizes = _pack_padded_sequence(inp, lengths, batch_first)

    assert res_data.shape == ref_data.shape
    assert res_batch_sizes.shape == ref_batch_sizes.shape
    utils.gems_assert_close(res_data, ref_data, dtype)
    utils.gems_assert_equal(res_batch_sizes, ref_batch_sizes)


@pytest.mark.pack_padded_sequence
@pytest.mark.parametrize("batch_first", [False, True])
def test_pack_padded_sequence_ragged_lengths(batch_first):
    # Ragged, mostly-unequal lengths (a different batch_sizes shape than the
    # evenly spaced case) so the row -> (t, b) mapping is exercised properly.
    time_steps, batch_size, feat_size = 9, 6, 4
    lengths = torch.tensor([9, 7, 7, 3, 1, 1], dtype=torch.int64)

    shape = (
        (batch_size, time_steps, feat_size)
        if batch_first
        else (time_steps, batch_size, feat_size)
    )
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_data, ref_batch_sizes = torch.ops.aten._pack_padded_sequence(
        ref_inp, lengths, batch_first
    )
    res_data, res_batch_sizes = _pack_padded_sequence(inp, lengths, batch_first)

    assert res_data.shape == ref_data.shape
    utils.gems_assert_close(res_data, ref_data, torch.float32)
    utils.gems_assert_equal(res_batch_sizes, ref_batch_sizes)
