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


def _reference_bucketize(inp, boundaries, **kwargs):
    ref_inp = utils.to_reference(inp, True)
    ref_boundaries = utils.to_reference(boundaries)
    return torch.bucketize(ref_inp, ref_boundaries, **kwargs)


@pytest.mark.bucketize
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bucketize(shape, dtype):
    boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0], device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = _reference_bucketize(inp, boundaries)

    with flag_gems.use_gems():
        res_out = torch.bucketize(inp, boundaries)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bucketize
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bucketize_right(shape, dtype):
    boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0], device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = _reference_bucketize(inp, boundaries, right=True)

    with flag_gems.use_gems():
        res_out = torch.bucketize(inp, boundaries, right=True)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bucketize
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bucketize_int32(shape, dtype):
    boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0], device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = _reference_bucketize(inp, boundaries, out_int32=True)

    with flag_gems.use_gems():
        res_out = torch.bucketize(inp, boundaries, out_int32=True)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bucketize
@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize(
    ("boundary_values", "boundary_dtype"),
    [
        pytest.param([1, 3, 5, 7, 9], torch.int64, id="integer"),
        pytest.param([], torch.float32, id="empty"),
        pytest.param([5.0], torch.float32, id="single"),
        pytest.param([1.0, 3.0], torch.float32, id="two"),
        pytest.param([-5.0, -3.0, 0.0, 3.0], torch.float32, id="negative"),
        pytest.param([float(i) for i in range(0, 64, 2)], torch.float32, id="many"),
    ],
)
def test_bucketize_boundary_cases(right, boundary_values, boundary_dtype):
    boundaries = torch.tensor(
        boundary_values, dtype=boundary_dtype, device=flag_gems.device
    )
    inp = torch.tensor(
        [-6.0, -5.0, -3.0, 0.0, 2.0, 5.0, 8.0, 65.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )

    ref_out = _reference_bucketize(inp, boundaries, right=right)

    with flag_gems.use_gems():
        res_out = torch.bucketize(inp, boundaries, right=right)

    utils.gems_assert_equal(res_out, ref_out)
