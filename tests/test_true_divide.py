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

TRUE_DIV_SCALAR_TENSOR_DTYPES = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "kunlunxin"
            and not cfg.TO_CPU
            and dtype in (torch.float16, torch.bfloat16),
            reason=(
                "Kunlunxin PyTorch baseline does not implement a Python "
                "scalar divided by a low-precision tensor"
            ),
        ),
        id=str(dtype),
    )
    for dtype in utils.FLOAT_DTYPES
]


@pytest.mark.div_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.true_divide(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.true_divide(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_inp1.true_divide_(ref_inp2)
    with flag_gems.use_gems():
        inp1.true_divide_(inp2)

    utils.gems_assert_close(inp1, ref_inp1, dtype, equal_nan=True)


@pytest.mark.div_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_out(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.empty_like(ref_inp1)
    res_out = torch.empty_like(inp1)

    torch.true_divide(ref_inp1, ref_inp2, out=ref_out)
    with flag_gems.use_gems():
        torch.true_divide(inp1, inp2, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1, False)

    ref_out = torch.true_divide(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.true_divide(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_scalar_(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1.clone(), False)

    ref_inp1.true_divide_(inp2)
    with flag_gems.use_gems():
        inp1.true_divide_(inp2)

    utils.gems_assert_close(inp1, ref_inp1, dtype, equal_nan=True)


@pytest.mark.div_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", TRUE_DIV_SCALAR_TENSOR_DTYPES)
def test_true_divide_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.true_divide(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.true_divide(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
