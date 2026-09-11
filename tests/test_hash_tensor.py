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


@pytest.mark.hash_tensor
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [None, 0, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_hash_tensor(shape, dtype, dim, keepdim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # hash_tensor is bit-exact: never upcast the reference, or the hashed
    # bit patterns (and therefore the result) would differ.
    ref_inp = utils.to_reference(inp)

    dim_arg = [] if dim is None else [dim]

    ref_out = torch.ops.aten.hash_tensor(ref_inp, dim_arg, keepdim=keepdim, mode=0)
    res_out = flag_gems.hash_tensor(inp, dim_arg, keepdim=keepdim, mode=0)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.hash_tensor
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + [torch.bool])
@pytest.mark.parametrize("dim", [None, -1])
def test_hash_tensor_int_dtypes(shape, dtype, dim):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
    ref_inp = utils.to_reference(inp)

    dim_arg = [] if dim is None else [dim]

    ref_out = torch.ops.aten.hash_tensor(ref_inp, dim_arg, keepdim=False, mode=0)
    res_out = flag_gems.hash_tensor(inp, dim_arg, keepdim=False, mode=0)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.hash_tensor
@pytest.mark.parametrize(
    "shape,dim",
    [
        # Multi-axis reductions: trailing-contiguous, leading, and a
        # non-collapsible mix that forces the wrapper's materializing path.
        ((64, 128, 256), [1, 2]),
        ((16, 32, 64), [0, 1]),
        ((8, 16, 32, 64), [0, 2]),
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
def test_hash_tensor_multidim(shape, dim, keepdim):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.hash_tensor(ref_inp, dim, keepdim=keepdim, mode=0)
    res_out = flag_gems.hash_tensor(inp, dim, keepdim=keepdim, mode=0)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.hash_tensor
def test_hash_tensor_scalar():
    inp = torch.tensor(42, dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.hash_tensor(ref_inp, [], keepdim=False, mode=0)
    res_out = flag_gems.hash_tensor(inp, [], keepdim=False, mode=0)

    utils.gems_assert_equal(res_out, ref_out)
