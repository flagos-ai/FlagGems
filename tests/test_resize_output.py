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


def _reference_resize_output(inp, size, device):
    """Reference implementation of _resize_output using PyTorch operations."""
    new_numel = 1
    for s in size:
        new_numel *= s
    old_numel = inp.numel()

    # Create output
    out = torch.empty(size, dtype=inp.dtype, device=device)

    # Copy data
    copy_numel = min(old_numel, new_numel)
    if copy_numel > 0:
        inp_flat = inp.reshape(-1)[:copy_numel]
        out_flat = out.reshape(-1)[:copy_numel]
        out_flat.copy_(inp_flat)

    return out


@pytest.mark.resize_output
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_resize_output(shape, dtype):
    # Create input tensor with known values for easier debugging
    inp = torch.arange(np.prod(shape), dtype=dtype, device=flag_gems.device)
    inp = inp.reshape(shape)

    # Target size: same total elements but different shape when possible
    total_elements = inp.numel()

    # Test resizing to same size but different shape
    if total_elements == 8:
        target_size = [2, 4]
    elif total_elements == 4:
        target_size = [2, 2]
    elif total_elements == 2:
        target_size = [2]
    else:
        target_size = [total_elements]

    device = inp.device

    # Use reference implementation
    ref_inp = utils.to_reference(inp)
    ref_device = torch.device("cpu") if utils.TO_CPU else device
    ref_out = _reference_resize_output(ref_inp, target_size, ref_device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._resize_output(inp, target_size, device)

    # Check shape matches
    assert (
        res_out.shape == ref_out.shape
    ), f"Shape mismatch: {res_out.shape} vs {ref_out.shape}"
    # Check data matches for overlapping elements
    utils.gems_assert_close(res_out, ref_out, dtype)
