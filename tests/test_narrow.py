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

# Shape configurations for narrow testing: 3D, 2D, and 1D tensors
NARROW_SHAPES = [(16, 32, 64), (32, 64), (64,)]


@pytest.mark.narrow
@pytest.mark.parametrize("shape", NARROW_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_narrow(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Test different dims, start positions and lengths
    for dim in range(inp.ndim):
        dim_size = inp.size(dim)
        for start in [0, dim_size // 4, dim_size // 2]:
            for length in [1, dim_size // 4, dim_size // 2]:
                if start + length <= dim_size:
                    ref_out = torch.narrow(ref_inp, dim, start, length)
                    with flag_gems.use_gems():
                        res_out = torch.narrow(inp, dim, start, length)
                    utils.gems_assert_equal(res_out, ref_out)
                    # narrow is a view op: output must share storage with input.
                    assert (
                        res_out.untyped_storage().data_ptr()
                        == inp.untyped_storage().data_ptr()
                    )


@pytest.mark.narrow
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_narrow_negative_start(dtype):
    # Test negative start index
    shape = (8, 16, 32)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.narrow(ref_inp, 1, -8, 4)
    with flag_gems.use_gems():
        res_out = torch.narrow(inp, 1, -8, 4)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.narrow
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_narrow_negative_dim(dtype):
    # Test negative dim
    shape = (8, 16, 32)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.narrow(ref_inp, -1, 4, 16)
    with flag_gems.use_gems():
        res_out = torch.narrow(inp, -1, 4, 16)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.narrow
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_narrow_tensor_start(dtype):
    # Test start given as a 0-dim tensor
    shape = (8, 16, 32)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    start = torch.tensor(2, device=flag_gems.device)
    ref_out = torch.narrow(ref_inp, 2, start, 8)
    with flag_gems.use_gems():
        res_out = torch.narrow(inp, 2, start, 8)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.narrow
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_narrow_full_slice(dtype):
    # Test when length equals the full dimension
    shape = (4, 8, 16)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.narrow(ref_inp, 0, 0, 4)
    with flag_gems.use_gems():
        res_out = torch.narrow(inp, 0, 0, 4)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.narrow
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_narrow_1d(dtype):
    # Test 1D tensor
    inp = torch.randn(64, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.narrow(ref_inp, 0, 10, 20)
    with flag_gems.use_gems():
        res_out = torch.narrow(inp, 0, 10, 20)
    utils.gems_assert_equal(res_out, ref_out)
