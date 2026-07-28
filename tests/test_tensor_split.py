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
from . import conftest as cfg

# Define shapes for tensor_split tests
if cfg.QUICK_MODE:
    TENSOR_SPLIT_SHAPES = [
        (64,),
        (64, 128),
    ]
else:
    TENSOR_SPLIT_SHAPES = [
        (64,),
        (64, 128),
        (8, 16, 32),
        (4, 8, 16, 32),
    ]


@pytest.mark.tensor_split
@pytest.mark.parametrize("shape", TENSOR_SPLIT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensor_split_by_int(shape, dtype):
    """Test tensor_split with integer sections."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Test splitting into 3 sections
    sections = 3
    ref_out = torch.tensor_split(ref_inp, sections, dim=0)

    with flag_gems.use_gems():
        res_out = torch.tensor_split(inp, sections, dim=0)

    # Compare number of outputs
    assert len(res_out) == len(ref_out)

    # Compare each split
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)


@pytest.mark.tensor_split
@pytest.mark.parametrize("shape", TENSOR_SPLIT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensor_split_by_list(shape, dtype):
    """Test tensor_split with list of indices."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Test splitting at specific indices
    indices = [shape[0] // 3, shape[0] * 2 // 3]
    ref_out = torch.tensor_split(ref_inp, indices, dim=0)

    with flag_gems.use_gems():
        res_out = torch.tensor_split(inp, indices, dim=0)

    # Compare number of outputs
    assert len(res_out) == len(ref_out)

    # Compare each split
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)


@pytest.mark.tensor_split
@pytest.mark.parametrize("shape", TENSOR_SPLIT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensor_split_dim(shape, dtype):
    """Test tensor_split with different dim values."""
    if len(shape) < 2:
        pytest.skip("Need at least 2 dimensions for dim test")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Test splitting along dim=1
    sections = 2
    ref_out = torch.tensor_split(ref_inp, sections, dim=1)

    with flag_gems.use_gems():
        res_out = torch.tensor_split(inp, sections, dim=1)

    # Compare number of outputs
    assert len(res_out) == len(ref_out)

    # Compare each split
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)


@pytest.mark.tensor_split
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensor_split_uneven(dtype):
    """Test tensor_split with uneven splits."""
    # Specific shape for uneven split test
    shape = (10, 20)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Split 10 into 3 sections: [4, 3, 3]
    sections = 3
    ref_out = torch.tensor_split(ref_inp, sections, dim=0)

    with flag_gems.use_gems():
        res_out = torch.tensor_split(inp, sections, dim=0)

    # Compare number of outputs
    assert len(res_out) == len(ref_out)

    # Compare each split
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)
