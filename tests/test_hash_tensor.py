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


def _assert_hash_result(res_out, ref_out, device):
    assert res_out.device == device
    assert res_out.dtype == ref_out.dtype == torch.uint64
    assert res_out.shape == ref_out.shape
    # The signed view compares the same 64 bits on runtimes without uint64 eq.
    if res_out.device != ref_out.device:
        res_out = res_out.to(ref_out.device)
    utils.gems_assert_equal(res_out.view(torch.int64), ref_out.view(torch.int64))


@pytest.mark.hash_tensor
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [None, 0, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_hash_tensor(shape, dtype, dim, keepdim):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Hashing is bit-exact. Upcasting would change the hashed bit patterns.
    ref_inp = utils.to_reference(inp, False)
    dim_arg = [] if dim is None else [dim]

    ref_out = torch.ops.aten.hash_tensor(ref_inp, dim_arg, keepdim=keepdim, mode=0)
    res_out = flag_gems.hash_tensor(inp, dim_arg, keepdim=keepdim, mode=0)

    _assert_hash_result(res_out, ref_out, inp.device)


@pytest.mark.hash_tensor
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + [torch.uint8, torch.bool])
@pytest.mark.parametrize("dim", [None, -1])
def test_hash_tensor_integer_and_bool(shape, dtype, dim):
    torch.manual_seed(0)
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, device=flag_gems.device).to(dtype)
    elif dtype == torch.uint8:
        inp = torch.randint(0, 255, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
    ref_inp = utils.to_reference(inp, False)
    dim_arg = [] if dim is None else [dim]

    ref_out = torch.ops.aten.hash_tensor(ref_inp, dim_arg, keepdim=False, mode=0)
    res_out = flag_gems.hash_tensor(inp, dim_arg, keepdim=False, mode=0)

    _assert_hash_result(res_out, ref_out, inp.device)


@pytest.mark.hash_tensor
@pytest.mark.parametrize(
    "shape, dim",
    [
        ((64, 128, 256), [1, 2]),
        ((16, 32, 64), [0, 1]),
        ((8, 16, 32, 64), [0, 2]),
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
def test_hash_tensor_multiple_dims(shape, dim, keepdim):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.ops.aten.hash_tensor(ref_inp, dim, keepdim=keepdim, mode=0)
    res_out = flag_gems.hash_tensor(inp, dim, keepdim=keepdim, mode=0)

    _assert_hash_result(res_out, ref_out, inp.device)


@pytest.mark.hash_tensor
@pytest.mark.parametrize("dim", [[0], [1], [0, 2]])
def test_hash_tensor_noncontiguous(dim):
    torch.manual_seed(0)
    inp = torch.randn((4, 5, 6), device=flag_gems.device).transpose(0, 2)
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.ops.aten.hash_tensor(ref_inp, dim, keepdim=True, mode=0)
    res_out = flag_gems.hash_tensor(inp, dim, keepdim=True, mode=0)

    _assert_hash_result(res_out, ref_out, inp.device)


@pytest.mark.hash_tensor
def test_hash_tensor_scalar():
    inp = torch.tensor(42, dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.ops.aten.hash_tensor(ref_inp, [], keepdim=False, mode=0)
    res_out = flag_gems.hash_tensor(inp, [], keepdim=False, mode=0)

    _assert_hash_result(res_out, ref_out, inp.device)


@pytest.mark.hash_tensor
def test_hash_tensor_scalar_and_no_grad():
    inp = torch.tensor(42.0, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.ops.aten.hash_tensor(ref_inp, [], keepdim=False, mode=0)
    res_out = flag_gems.hash_tensor(inp, [], keepdim=False, mode=0)

    _assert_hash_result(res_out, ref_out, inp.device)
    assert res_out.shape == torch.Size([])
    assert not res_out.requires_grad
