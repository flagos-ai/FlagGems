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

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.true_divide__tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["transpose", "slice", "scalar"])
def test_true_divide_inplace_strides(dtype, layout):
    op = importlib.import_module("flag_gems.ops.true_divide_")

    op._PLAN.clear()
    op._LAUNCH_STATE.clear()
    # Keep caches warm across different divisor layouts, including strides of 1.
    for rhs in (
        "dense",
        "strided",
        "broadcast",
        "zero_dim",
        "singleton",
        "python",
        "self",
    ):
        base = torch.arange(1, 97, dtype=dtype, device=flag_gems.device).reshape(8, 12)
        ref = base.cpu().clone()

        def view(t):
            if layout == "transpose":
                return t.t()
            if layout == "slice":
                return t[1::2, 1::3]
            return t[1, 1]

        x, expected = view(base), view(ref)
        if rhs == "self":
            y = x
        elif rhs == "python":
            y = 2.0
        elif rhs in ("zero_dim", "singleton"):
            y = torch.full(
                () if rhs == "zero_dim" else (1,) * x.ndim,
                2.0,
                dtype=dtype,
                device=x.device,
            )
        elif rhs == "strided":
            y = torch.arange(1, 2 * x.numel() + 1, dtype=dtype, device=x.device)[
                ::2
            ].reshape(x.shape)
        elif rhs == "broadcast" and x.ndim:
            y = torch.arange(1, x.shape[-1] + 1, dtype=dtype, device=x.device)
        else:
            y = torch.full(x.shape, 2.0, dtype=dtype, device=x.device)
        expected.div_(y.cpu() if isinstance(y, torch.Tensor) else y)
        pointer, stride, offset = x.data_ptr(), x.stride(), x.storage_offset()
        assert op.true_divide_tensor_(x, y) is x
        assert (x.data_ptr(), x.stride(), x.storage_offset()) == (
            pointer,
            stride,
            offset,
        )
        # Compare the whole allocation, not just the view: holes must be untouched.
        torch.testing.assert_close(base.cpu(), ref)


@pytest.mark.true_divide_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.true_divide(ref_inp1, ref_inp2)
    res_out = flag_gems.true_divide_tensor(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.true_divide__tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide__tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_inp1.true_divide_(ref_inp2)
    res_out = flag_gems.true_divide__tensor(inp1, inp2)
    assert res_out is inp1

    utils.gems_assert_close(inp1, ref_inp1, dtype, equal_nan=True)


@pytest.mark.true_divide_out
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
    returned = flag_gems.true_divide_out(inp1, inp2, out=res_out)
    assert returned is res_out

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.true_divide_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1, False)

    ref_out = torch.true_divide(ref_inp1, inp2)
    res_out = flag_gems.true_divide_scalar(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.true_divide__scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide__scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1.clone(), False)

    ref_inp1.true_divide_(inp2)
    res_out = flag_gems.true_divide__scalar(inp1, inp2)
    assert res_out is inp1

    utils.gems_assert_close(inp1, ref_inp1, dtype, equal_nan=True)


@pytest.mark.true_divide_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_scalar_numerator(shape, scalar, dtype):
    inp1 = torch.tensor(scalar, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.true_divide(ref_inp1, ref_inp2)
    res_out = flag_gems.true_divide_tensor(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.true_divide_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_dispatch(shape, dtype, caplog):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.ops.aten.true_divide.Tensor(ref_inp1, ref_inp2)
    with caplog.at_level(
        "DEBUG", logger=utils.gems_log_logger(flag_gems.true_divide_tensor)
    ):
        res_out = flag_gems.true_divide_tensor(inp1, inp2)

    assert (
        f"{utils.gems_log_prefix(flag_gems.true_divide_tensor)} TRUE_DIVIDE"
        in caplog.text
    )
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.true_divide__tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide__tensor_dispatch(shape, dtype, caplog):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    torch.ops.aten.true_divide_.Tensor(ref_inp1, ref_inp2)
    with caplog.at_level(
        "DEBUG", logger=utils.gems_log_logger(flag_gems.true_divide__tensor)
    ):
        res_out = flag_gems.true_divide__tensor(inp1, inp2)

    assert (
        f"{utils.gems_log_prefix(flag_gems.true_divide__tensor)} TRUE_DIVIDE_"
        in caplog.text
    )
    assert res_out is inp1
    utils.gems_assert_close(inp1, ref_inp1, dtype, equal_nan=True)
