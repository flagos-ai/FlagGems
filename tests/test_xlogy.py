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


@pytest.mark.xlogy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # keep ``other`` positive so ``log`` stays finite for a clean comparison
    y = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5.0 + 0.01

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.xlogy(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.xlogy(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.xlogy_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5.0 + 0.01

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.xlogy.OutTensor(ref_x, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.xlogy.OutTensor(x, y, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.xlogy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_special_values(dtype):
    # Exercise the PyTorch precedence: NaN(other) -> NaN; x == 0 -> 0; else x*log(y)
    x = torch.tensor([0.0, 0.0, 2.0, 3.0, 0.0], dtype=dtype, device=flag_gems.device)
    y = torch.tensor(
        [5.0, 0.0, 4.0, float("nan"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.xlogy(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.xlogy(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.xlogy_tensor_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_tensor_scalar(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scalar = 3.5

    ref_x = utils.to_reference(x, True)
    ref_out = torch.xlogy(ref_x, scalar)

    with flag_gems.use_gems():
        res_out = torch.xlogy(x, scalar)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.xlogy_tensor_scalar_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_tensor_scalar_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scalar = 3.5

    ref_x = utils.to_reference(x, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.xlogy.OutScalar_Other(ref_x, scalar, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.xlogy.OutScalar_Other(x, scalar, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.xlogy_scalar_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_scalar_tensor(shape, dtype):
    scalar = 2.0
    y = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5.0 + 0.01

    ref_y = utils.to_reference(y, True)
    ref_out = torch.xlogy(scalar, ref_y)

    with flag_gems.use_gems():
        res_out = torch.xlogy(scalar, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.xlogy_scalar_tensor_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_scalar_tensor_out(shape, dtype):
    scalar = 2.0
    y = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5.0 + 0.01

    ref_y = utils.to_reference(y, True)
    ref_out_buf = torch.empty(shape, dtype=ref_y.dtype, device=ref_y.device)
    ref_out = torch.ops.aten.xlogy.OutScalar_Self(scalar, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.xlogy.OutScalar_Self(scalar, y, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)
