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

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.jagged_to_padded_dense_forward
@pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
@pytest.mark.parametrize("max_length", [8, 16, 32, 128])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_jagged_to_padded_dense_forward(batch_size, max_length, dtype):
    # Create variable-length sequences
    # Generate random sequence lengths
    np.random.seed(42)
    seq_lengths = np.random.randint(1, max_length + 1, size=batch_size).tolist()

    # Create offsets tensor (cumulative)
    offsets = [0] + list(np.cumsum(seq_lengths).astype(int).tolist())
    offsets = torch.tensor(offsets, device=flag_gems.device, dtype=torch.int64)

    # Create values tensor (concatenated sequences)
    total_length = sum(seq_lengths)
    values = torch.randn(total_length, dtype=dtype, device=flag_gems.device)

    # Reference implementation
    ref_values = utils.to_reference(values)
    ref_offsets = utils.to_reference(offsets)

    ref_out = torch.ops.aten._jagged_to_padded_dense_forward(
        ref_values, [ref_offsets], [max_length], 0.0
    )

    # GEMS implementation
    with flag_gems.use_gems():
        res_out = torch.ops.aten._jagged_to_padded_dense_forward(
            values, [offsets], [max_length], 0.0
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.jagged_to_padded_dense_forward
@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("max_length", [16, 32])
@pytest.mark.parametrize("padding_value", [0.0, -1.0, 1.5])
def test_jagged_to_padded_dense_forward_padding(batch_size, max_length, padding_value):
    # Test with different padding values
    np.random.seed(42)
    seq_lengths = np.random.randint(1, max_length + 1, size=batch_size).tolist()

    offsets = [0] + list(np.cumsum(seq_lengths).astype(int).tolist())
    offsets = torch.tensor(offsets, device=flag_gems.device, dtype=torch.int64)

    total_length = sum(seq_lengths)
    values = torch.randn(total_length, dtype=torch.float32, device=flag_gems.device)

    ref_values = utils.to_reference(values)
    ref_offsets = utils.to_reference(offsets)

    ref_out = torch.ops.aten._jagged_to_padded_dense_forward(
        ref_values, [ref_offsets], [max_length], padding_value
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten._jagged_to_padded_dense_forward(
            values, [offsets], [max_length], padding_value
        )

    utils.gems_assert_close(res_out, ref_out, torch.float32)
