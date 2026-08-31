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


def _make_nested(batch_size, max_length, trailing, dtype):
    # Force component 0 to the full row count so the inferred padded shape is
    # deterministic across the reference and GEMS paths; the rest are ragged.
    np.random.seed(42)
    row_counts = np.random.randint(1, max_length + 1, size=batch_size)
    row_counts[0] = max_length
    comps = [
        torch.randn([int(r)] + list(trailing), dtype=dtype, device=flag_gems.device)
        for r in row_counts
    ]
    return torch.nested.nested_tensor(comps, device=flag_gems.device)


@pytest.mark.nested_to_padded_tensor
@pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
@pytest.mark.parametrize("max_length", [8, 16, 32, 128])
# trailing=(32,) keeps one row within a single kernel BLOCK_SIZE (1024)
@pytest.mark.parametrize("trailing", [(16,), (32,), (64,)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_to_padded_tensor(batch_size, max_length, trailing, dtype):
    nt = _make_nested(batch_size, max_length, trailing, dtype)

    ref_out = utils.to_reference(
        torch.ops.aten.nested_to_padded_tensor(nt, 0.0, None), True
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.nested_to_padded_tensor(nt, 0.0, None)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.nested_to_padded_tensor
@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("max_length", [16, 32])
@pytest.mark.parametrize("padding_value", [0.0, -1.0, 1.5])
def test_nested_to_padded_tensor_padding(batch_size, max_length, padding_value):
    # trailing=(32,) keeps one row within a single kernel BLOCK_SIZE (1024)
    nt = _make_nested(batch_size, max_length, (32,), torch.float32)

    ref_out = utils.to_reference(
        torch.ops.aten.nested_to_padded_tensor(nt, padding_value, None), True
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.nested_to_padded_tensor(nt, padding_value, None)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.nested_to_padded_tensor
def test_nested_to_padded_tensor_dispatch():
    # nested_to_padded_tensor is a CompositeImplicitAutograd op; confirm the
    # FlagGems implementation is actually registered under use_gems() so the
    # accuracy tests exercise the GEMS path rather than silently falling back.
    with flag_gems.use_gems():
        assert "nested_to_padded_tensor" in flag_gems.all_registered_ops()


@pytest.mark.nested_to_padded_tensor
@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("max_length", [16, 32])
def test_nested_to_padded_tensor_output_size(batch_size, max_length):
    # trailing=(32,) keeps one row within a single kernel BLOCK_SIZE (1024)
    nt = _make_nested(batch_size, max_length, (32,), torch.float32)

    # Request an explicit padded shape larger than the inferred one.
    output_size = [batch_size, max_length + 4, 32]

    ref_out = utils.to_reference(
        torch.ops.aten.nested_to_padded_tensor(nt, 0.0, output_size), True
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.nested_to_padded_tensor(nt, 0.0, output_size)

    utils.gems_assert_close(res_out, ref_out, torch.float32)
