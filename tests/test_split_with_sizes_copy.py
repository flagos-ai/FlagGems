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

# Shapes covering 1D to 4D tensors with various dimension sizes
if cfg.QUICK_MODE:
    SPLIT_WITH_SIZES_COPY_SHAPES = [
        (10,),
        (16, 32),
    ]
else:
    SPLIT_WITH_SIZES_COPY_SHAPES = [
        (10,),
        (10, 4),
        (10, 4, 8),
        (10, 4, 8, 16),
        (16, 32),
        (8, 64, 128),
        (1, 8192),
        (32, 50257),
    ]


@pytest.mark.split_with_sizes_copy
@pytest.mark.parametrize("shape", SPLIT_WITH_SIZES_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_split_with_sizes_copy(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Define split sizes that sum to the first dimension size
    dim_size = shape[0]
    split_sizes = [dim_size // 4, dim_size // 4, dim_size - 2 * (dim_size // 4)]
    dim = 0

    ref_out = torch.split_with_sizes_copy(ref_inp, split_sizes, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.split_with_sizes_copy(inp, split_sizes, dim=dim)

    assert len(res_out) == len(ref_out), "Number of splits mismatch"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(res, ref, dtype)


@pytest.mark.split_with_sizes_copy
@pytest.mark.parametrize("shape", [(10, 4, 8), (20, 32, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0, 1, 2])
def test_split_with_sizes_copy_different_dims(shape, dtype, dim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Generate split sizes that sum to the dimension size
    dim_size = shape[dim]
    split_sizes = [dim_size // 4, dim_size // 4, dim_size - 2 * (dim_size // 4)]

    ref_out = torch.split_with_sizes_copy(ref_inp, split_sizes, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.split_with_sizes_copy(inp, split_sizes, dim=dim)

    assert len(res_out) == len(ref_out), "Number of splits mismatch"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(res, ref, dtype)


@pytest.mark.split_with_sizes_copy
@pytest.mark.parametrize("shape", [(10, 4), (8, 16, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_split_with_sizes_copy_edge_cases(shape, dtype):
    # Test with zero-size split
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    split_sizes = [2, 0, shape[0] - 2]
    dim = 0

    ref_out = torch.split_with_sizes_copy(ref_inp, split_sizes, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.split_with_sizes_copy(inp, split_sizes, dim=dim)

    assert len(res_out) == len(ref_out), "Number of splits mismatch"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(res, ref, dtype)
