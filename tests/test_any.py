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

HYGON_ONLY = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon any contract regression"
)

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIMS_LIST = [1]
    KIND_KEEPDIM_DIMS_SHAPE = [("normal", True, 1, utils.REDUCTION_SHAPES[0])]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    KIND_KEEPDIM_DIMS_SHAPE = list(
        zip(
            ["normal", "allTrue"] * 2,
            [True, False] * 2,
            DIMS_LIST,
            utils.REDUCTION_SHAPES + [(7, 4, 11, 1)],
        )
    )


@pytest.mark.any
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_any(shape, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.any(ref_inp)
    res_out = flag_gems.any(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.any
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon empty any regression"
)
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8])
def test_any_hygon_empty(dtype):
    inp_cpu = torch.empty((0,), dtype=dtype, device="cpu")
    inp = inp_cpu.to(flag_gems.device)
    actual = flag_gems.any(inp)
    assert actual.device == inp.device
    torch.testing.assert_close(actual.cpu(), torch.any(inp_cpu), rtol=0, atol=0)


@pytest.mark.any_dim
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon empty any regression"
)
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8])
@pytest.mark.parametrize(
    "shape,dim,keepdim",
    [
        pytest.param((2, 0), 1, False, id="empty-reduction"),
        pytest.param((0, 3), -2, True, id="keepdim"),
        pytest.param((0, 3), -1, False, id="empty-output"),
        pytest.param((0,), 0, False, id="scalar-output"),
    ],
)
def test_any_dim_hygon_empty(shape, dim, keepdim, dtype):
    inp_cpu = torch.empty(shape, dtype=dtype, device="cpu")
    inp = inp_cpu.to(flag_gems.device)
    actual = flag_gems.any_dim(inp, dim=dim, keepdim=keepdim)
    expected = torch.any(inp_cpu, dim=dim, keepdim=keepdim)
    assert actual.device == inp.device
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.any
@HYGON_ONLY
@pytest.mark.parametrize("layout", ["stepped", "broadcast"])
def test_any_hygon_strided(layout):
    base = torch.tensor([9, 0, 8, 0, 7, 0, 6, 0], dtype=torch.int32)

    def as_view(x):
        if layout == "stepped":
            return x[1::2]
        return x[1:2].expand(4)

    inp = as_view(base.to(flag_gems.device))
    expected = torch.any(as_view(base))
    torch.testing.assert_close(flag_gems.any(inp).cpu(), expected, rtol=0, atol=0)


@pytest.mark.any_dim
@HYGON_ONLY
@pytest.mark.parametrize("value,dtype", [(False, torch.bool), (7, torch.uint8)])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_any_dim_hygon_scalar(value, dtype, dim, keepdim):
    cpu = torch.tensor(value, dtype=dtype)
    inp = cpu.to(flag_gems.device)
    expected = torch.any(cpu, dim=dim, keepdim=keepdim)
    actual = flag_gems.any_dim(inp, dim=dim, keepdim=keepdim)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.any_dim
@HYGON_ONLY
@pytest.mark.parametrize("shape,dim", [((2, 0), 2), ((0, 2), -3), ((), 1), ((), -2)])
def test_any_dim_hygon_invalid_dim(shape, dim):
    cpu = torch.zeros(shape, dtype=torch.bool)
    inp = cpu.to(flag_gems.device)
    with pytest.raises(IndexError) as expected:
        torch.any(cpu, dim=dim)
    with pytest.raises(IndexError) as actual:
        flag_gems.any_dim(inp, dim=dim)
    assert str(actual.value) == str(expected.value)


@pytest.mark.any_dims
@HYGON_ONLY
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "shape,dims", [((2, 0, 3), (0, 1)), ((2, 0, 3), (0, 2)), ((0, 0, 3), (0, -1))]
)
def test_any_dims_hygon_empty(shape, dims, keepdim, dtype):
    cpu = torch.empty(shape, dtype=dtype)
    inp = cpu.to(flag_gems.device)
    expected = torch.any(cpu, dim=dims, keepdim=keepdim)
    actual = flag_gems.any_dims(inp, dim=dims, keepdim=keepdim)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.any_dims
@HYGON_ONLY
@pytest.mark.parametrize("dims", [(), (1, 0)])
def test_any_dims_hygon_uint8(dims):
    cpu = torch.tensor([[0, 7], [9, 0]], dtype=torch.uint8)
    inp = cpu.to(flag_gems.device)
    expected = torch.any(cpu, dim=dims)
    actual = flag_gems.any_dims(inp, dim=dims)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.any_dims
@pytest.mark.parametrize("kind, keepdim, dim, shape", KIND_KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
def test_any_dims(shape, dim, keepdim, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.any(ref_inp, dim=dim, keepdim=keepdim)
    res_out = flag_gems.any_dims(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.any_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("kind", ["normal", "allFalse", "allTrue"])
def test_any_dim(shape, dtype, keepdim, dim, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    elif kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.any(ref_inp, dim=dim, keepdim=keepdim)
    res_out = flag_gems.any_dim(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)
