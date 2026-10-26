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

SHAPES = [(6, 4, 8), (10, 8, 16), (33, 7, 1025), (64, 32, 256)]


def _lengths(time_steps, batch_size):
    values = torch.linspace(time_steps, 1, steps=batch_size, dtype=torch.int64).clamp(
        min=1
    )
    return torch.sort(values, descending=True).values


@pytest.mark.pack_padded_sequence
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("time_steps,batch_size,features", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pack_padded_sequence(time_steps, batch_size, features, batch_first, dtype):
    lengths = _lengths(time_steps, batch_size)
    shape = (
        (batch_size, time_steps, features)
        if batch_first
        else (time_steps, batch_size, features)
    )
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Ascend's native primitive returns unpacked T*N rows for nonuniform
    # lengths. CPU ATen preserves the exact compact data/host-metadata ABI.
    ref_inp = (
        inp.cpu().double()
        if flag_gems.vendor_name == "ascend"
        else utils.to_reference(inp, True)
    )
    ref_data, ref_batch_sizes = torch.ops.aten._pack_padded_sequence(
        ref_inp, lengths, batch_first
    )
    data, batch_sizes = flag_gems.pack_padded_sequence(inp, lengths, batch_first)
    assert data.device == inp.device
    assert data.dtype == inp.dtype
    assert data.shape == ref_data.shape
    assert batch_sizes.device.type == "cpu"
    assert batch_sizes.dtype == torch.int64
    assert batch_sizes.shape == ref_batch_sizes.shape
    assert data is not inp
    if data.numel():
        assert data.data_ptr() != inp.data_ptr()
    if flag_gems.vendor_name == "ascend":
        # Target metadata and alias assertions above precede CPU observation.
        utils.gems_assert_close(data.cpu(), ref_data.cpu(), dtype)
    else:
        utils.gems_assert_close(data, ref_data, dtype)
    utils.gems_assert_equal(batch_sizes, ref_batch_sizes)
