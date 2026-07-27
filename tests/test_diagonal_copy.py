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

DIAGONAL_COPY_OFFSETS = [0, 1, -1, 2, -2]
# Cover common dimension pairs for 2D, 3D, and 4D tensors
DIAGONAL_COPY_DIMS = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 3),
]


# Various tensor ranks (3D/4D), square/non-square dims, and edge cases (1-sized dims)
DIAGONAL_COPY_SHAPES = [
    (3, 4, 5),
    (5, 4, 3),
    (2, 3, 4, 5),
    (1, 2, 3),
    (3, 3, 3),
]


@pytest.mark.diagonal_copy
@pytest.mark.parametrize("shape", DIAGONAL_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("offset", DIAGONAL_COPY_OFFSETS)
@pytest.mark.parametrize("dims", DIAGONAL_COPY_DIMS)
def test_diagonal_copy(shape, dtype, offset, dims):
    dim1, dim2 = dims
    # Validate dims are within bounds
    if dim1 >= len(shape) or dim2 >= len(shape):
        pytest.skip("dim out of bounds for shape")
    if dim1 == dim2:
        pytest.skip("dim1 and dim2 must be different")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.diagonal_copy(ref_inp, offset, dim1, dim2)
    with flag_gems.use_gems():
        res_out = torch.diagonal_copy(inp, offset, dim1, dim2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diagonal_copy
@pytest.mark.parametrize("shape", DIAGONAL_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
@pytest.mark.parametrize("offset", [0, 1, -1])
@pytest.mark.parametrize("dims", [(0, 1), (1, 2), (0, 2)])
def test_diagonal_copy_int(shape, dtype, offset, dims):
    dim1, dim2 = dims
    # Validate dims are within bounds
    if dim1 >= len(shape) or dim2 >= len(shape):
        pytest.skip("dim out of bounds for shape")
    if dim1 == dim2:
        pytest.skip("dim1 and dim2 must be different")

    inp = torch.randint(-1000, 1000, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.diagonal_copy(ref_inp, offset, dim1, dim2)
    with flag_gems.use_gems():
        res_out = torch.diagonal_copy(inp, offset, dim1, dim2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagonal_copy
def test_diagonal_copy_empty():
    # Test edge case: empty tensor
    shape = (0, 3, 4)
    dtype = torch.float32
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.diagonal_copy(ref_inp, 0, 1, 2)
    with flag_gems.use_gems():
        res_out = torch.diagonal_copy(inp, 0, 1, 2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagonal_copy
def test_diagonal_copy_single_element():
    # Test edge case: single element
    shape = (1, 1, 1)
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.diagonal_copy(ref_inp, 0, 1, 2)
    with flag_gems.use_gems():
        res_out = torch.diagonal_copy(inp, 0, 1, 2)

    utils.gems_assert_close(res_out, ref_out, dtype)
