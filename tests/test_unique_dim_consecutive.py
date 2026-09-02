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


@pytest.mark.unique_dim_consecutive
@pytest.mark.parametrize("shape", [(10, 5), (20, 10), (100, 50), (8, 256)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_unique_dim_consecutive_basic(shape, dtype, dim):
    # Generate input with some consecutive duplicates
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inv, ref_counts = torch.unique_consecutive(
        ref_inp, dim=dim, return_inverse=True, return_counts=True
    )
    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(
        inp, dim=dim, return_inverse=True, return_counts=True
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_inv, ref_inv)
    utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique_dim_consecutive
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unique_dim_consecutive_all_same(dtype):
    # Test with all identical consecutive elements
    inp = torch.ones((10, 5), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inv, ref_counts = torch.unique_consecutive(
        ref_inp, dim=0, return_inverse=True, return_counts=True
    )
    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(
        inp, dim=0, return_inverse=True, return_counts=True
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_inv, ref_inv)
    utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique_dim_consecutive
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unique_dim_consecutive_all_unique(dtype):
    # Test with all unique elements
    inp = torch.arange(20, dtype=dtype, device=flag_gems.device).view(10, 2)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inv, ref_counts = torch.unique_consecutive(
        ref_inp, dim=0, return_inverse=True, return_counts=True
    )
    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(
        inp, dim=0, return_inverse=True, return_counts=True
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_inv, ref_inv)
    utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique_dim_consecutive
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unique_dim_consecutive_no_inverse_no_counts(dtype):
    # Test without return_inverse and return_counts
    inp = torch.randn((20, 10), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inv, ref_counts = torch.unique_consecutive(
        ref_inp, dim=0, return_inverse=True, return_counts=True
    )
    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(
        inp, dim=0, return_inverse=False, return_counts=False
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    # When return_inverse=False and return_counts=False, they should be empty tensors
    assert res_inv.numel() == 0
    assert res_counts.numel() == 0


@pytest.mark.unique_dim_consecutive
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_unique_dim_consecutive_int(dtype):
    # Test with integer types
    inp = torch.randint(0, 10, (20, 5), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inv, ref_counts = torch.unique_consecutive(
        ref_inp, dim=0, return_inverse=True, return_counts=True
    )
    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(
        inp, dim=0, return_inverse=True, return_counts=True
    )

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inv, ref_inv)
    utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique_dim_consecutive
def test_unique_dim_consecutive_large_row():
    # Test with large row size
    dtype = torch.float32
    inp = torch.randn((10, 2048), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inv, ref_counts = torch.unique_consecutive(
        ref_inp, dim=0, return_inverse=True, return_counts=True
    )
    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(
        inp, dim=0, return_inverse=True, return_counts=True
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_inv, ref_inv)
    utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique_dim_consecutive
def test_unique_dim_consecutive_3d():
    # Test with 3D tensor
    dtype = torch.float32
    inp = torch.randn((5, 10, 8), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in [0, 1, 2]:
        ref_out, ref_inv, ref_counts = torch.unique_consecutive(
            ref_inp, dim=dim, return_inverse=True, return_counts=True
        )
        res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(
            inp, dim=dim, return_inverse=True, return_counts=True
        )

        utils.gems_assert_close(res_out, ref_out, dtype)
        utils.gems_assert_equal(res_inv, ref_inv)
        utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique_dim_consecutive_out
@pytest.mark.parametrize("shape", [(10, 5), (20, 10), (8, 256)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_unique_dim_consecutive_out(shape, dtype, dim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out, ref_inv, ref_counts = torch.unique_consecutive(
        ref_inp, dim=dim, return_inverse=True, return_counts=True
    )

    out0 = torch.empty(0, dtype=dtype, device=flag_gems.device)
    out1 = torch.empty(0, dtype=torch.int64, device=flag_gems.device)
    out2 = torch.empty(0, dtype=torch.int64, device=flag_gems.device)
    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive_out(
        inp,
        dim,
        return_inverse=True,
        return_counts=True,
        out0=out0,
        out1=out1,
        out2=out2,
    )

    assert res_out is out0
    assert res_inv is out1
    assert res_counts is out2
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_inv, ref_inv)
    utils.gems_assert_equal(res_counts, ref_counts)
