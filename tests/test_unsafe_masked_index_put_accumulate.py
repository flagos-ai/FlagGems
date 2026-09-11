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


def _flat_to_per_dim_indices(flat_indices, inp_shape):
    """Convert flat linear indices to per-dimension index tensors for the aten op."""
    strides = []
    prod = 1
    for s in reversed(inp_shape):
        strides.append(prod)
        prod *= s
    strides = tuple(reversed(strides))
    result = []
    for i in range(len(inp_shape)):
        result.append((flat_indices // strides[i]) % inp_shape[i])
    return tuple(result)


@pytest.mark.unsafe_masked_index_put_accumulate
@pytest.mark.parametrize("shape", utils._UNSAFE_MASKED_INDEX_PUT_SHAPES)
@pytest.mark.parametrize(
    "dtype", [d for d in utils.FLOAT_DTYPES if d != torch.bfloat16]
)
def test_unsafe_masked_index_put_accumulate(shape, dtype):
    inp_shape, mask_shape, indices_shape, values_shape = shape
    assert (
        mask_shape == indices_shape == values_shape
    ), "mask, indices, and values must have same shape"

    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randint(0, 2, mask_shape, dtype=torch.int32, device=flag_gems.device)
    flat_indices = torch.randint(
        0, max(inp.numel(), 1), indices_shape, device=flag_gems.device
    )
    values = torch.randn(values_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone())
    ref_mask = utils.to_reference(mask.clone())
    ref_flat_indices = utils.to_reference(flat_indices.clone())
    ref_values = utils.to_reference(values.clone())

    # Convert flat indices to per-dimension indices for the aten op
    ref_idx_tuple = _flat_to_per_dim_indices(ref_flat_indices.clone(), inp_shape)
    idx_tuple = _flat_to_per_dim_indices(flat_indices, inp_shape)

    op = torch._unsafe_masked_index_put_accumulate
    ref_out = op(ref_inp, ref_mask.clone(), ref_idx_tuple, ref_values)
    with flag_gems.use_gems():
        res_out = op(inp.clone(), mask, idx_tuple, values)

    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 5e-3
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)
