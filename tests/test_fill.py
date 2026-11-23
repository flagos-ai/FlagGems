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


@pytest.mark.fill_tensor
@pytest.mark.parametrize("value", [0, 1, 9])
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_fill_tensor(value, shape, dtype):
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = utils.to_reference(x, False)

    value_tensor = torch.tensor(value, device=flag_gems.device, dtype=dtype)
    ref_value_tensor = utils.to_reference(value_tensor, False)
    ref_out_tensor = torch.fill(ref_x, ref_value_tensor)
    res_out_tensor = flag_gems.fill_tensor(x, value_tensor)

    utils.gems_assert_equal(res_out_tensor, ref_out_tensor)


@pytest.mark.fill_scalar
@pytest.mark.parametrize("value", [0, 1, 9])
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_fill_scalar(value, shape, dtype):
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = utils.to_reference(x, False)

    ref_out = torch.fill(ref_x, value)
    res_out = flag_gems.fill_scalar(x, value)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fill_tensor_out
@pytest.mark.parametrize("value", [0, 1, 9])
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_fill_tensor_out(value, shape, dtype):
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = utils.to_reference(x, False)

    value_tensor = torch.tensor(value, device=flag_gems.device, dtype=dtype)
    ref_value_tensor = utils.to_reference(value_tensor, False)
    out_tensor = torch.empty_like(x)
    ref_out_tensor = torch.empty_like(ref_x)

    ref_result_tensor = torch.ops.aten.fill.Tensor_out(
        ref_x, ref_value_tensor, out=ref_out_tensor
    )
    res_result_tensor = flag_gems.fill_tensor_out(x, value_tensor, out=out_tensor)

    utils.gems_assert_equal(res_result_tensor, ref_result_tensor)
    assert (
        res_result_tensor is out_tensor
    ), "fill.Tensor_out should return the out tensor"


@pytest.mark.fill_scalar_out
@pytest.mark.parametrize("value", [0, 1, 9])
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_fill_scalar_out(value, shape, dtype):
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = utils.to_reference(x, False)
    out = torch.empty_like(x)
    ref_out = torch.empty_like(ref_x)

    ref_result = torch.ops.aten.fill.Scalar_out(ref_x, value, out=ref_out)
    res_result = flag_gems.fill_scalar_out(x, value, out=out)

    utils.gems_assert_equal(res_result, ref_result)
    assert res_result is out, "fill.Scalar_out should return the out tensor"


# fill_.Scalar
@pytest.mark.fill_scalar_
@pytest.mark.parametrize("value", [0, 1, 9])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_fill_scalar_(value, shape, dtype):
    # Test fill_.Scalar
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = utils.to_reference(x.clone(), False)

    ref_x.fill_(value)
    flag_gems.fill_scalar_(x, value)


@pytest.mark.fill_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shape", [(0,), (3, 4)])
@pytest.mark.parametrize("tensor_value", [False, True])
def test_fill_backward(dtype, shape, tensor_value):
    inp = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    grad = torch.ones_like(inp)
    ref = utils.to_reference(inp.detach(), False).requires_grad_()
    actual = inp.detach().requires_grad_()
    ref_grad = utils.to_reference(grad, False)
    if tensor_value:
        value = torch.tensor(
            2.0, device=flag_gems.device, dtype=dtype, requires_grad=True
        )
        ref_value = utils.to_reference(value.detach(), False).requires_grad_()
        expected = torch.autograd.grad(
            torch.fill(ref, ref_value), (ref, ref_value), ref_grad
        )
        result = torch.autograd.grad(
            flag_gems.fill_tensor(actual, value), (actual, value), grad
        )
        utils.gems_assert_close(result[1], expected[1], dtype)
    else:
        expected = torch.autograd.grad(torch.fill(ref, 2.0), ref, ref_grad)
        result = torch.autograd.grad(flag_gems.fill_scalar(actual, 2.0), actual, grad)
    utils.gems_assert_equal(result[0], expected[0])


@pytest.mark.fill_backward
@pytest.mark.parametrize("value_device", ["cpu", "gpu"])
def test_fill_tensor_value_gradient_mixed_dtype(value_device):
    inplace = False
    inp = torch.randn((3, 4), device=flag_gems.device, dtype=torch.float16)
    grad = torch.ones_like(inp)
    ref = utils.to_reference(inp.detach(), False).requires_grad_()
    actual = inp.detach().requires_grad_()
    ref_grad = utils.to_reference(grad, False)
    value = torch.tensor(
        2.0,
        device="cpu" if value_device == "cpu" else flag_gems.device,
        dtype=torch.float32,
        requires_grad=True,
    )
    ref_value = utils.to_reference(value.detach(), False).requires_grad_()
    native_input = ref.clone() if inplace else ref
    gems_input = actual.clone() if inplace else actual
    native_result = (
        native_input.fill_(ref_value)
        if inplace
        else torch.fill(native_input, ref_value)
    )
    gems_result = (
        flag_gems.fill_tensor_(gems_input, value)
        if inplace
        else flag_gems.fill_tensor(gems_input, value)
    )
    expected = torch.autograd.grad(native_result, (ref, ref_value), ref_grad)
    result = torch.autograd.grad(gems_result, (actual, value), grad)
    utils.gems_assert_equal(result[0], expected[0])
    utils.gems_assert_equal(result[1], expected[1])


@pytest.mark.fill_backward
@pytest.mark.parametrize("value_device", ["cpu", "gpu"])
def test_fill_tensor_double_backward(value_device):
    inplace = False
    x = torch.randn((3, 4), device=flag_gems.device, requires_grad=True)
    value = torch.tensor(
        2.0,
        device="cpu" if value_device == "cpu" else flag_gems.device,
        requires_grad=True,
    )
    dout = torch.randn_like(x, requires_grad=True)

    def gradient(op, inp, scalar, upstream):
        result = op(inp.clone(), scalar) if inplace else op(inp, scalar)
        first = torch.autograd.grad(result, scalar, upstream, create_graph=True)[0]
        return torch.autograd.grad(first, upstream)[0]

    native = lambda inp, scalar: (
        inp.fill_(scalar) if inplace else torch.fill(inp, scalar)
    )
    gems = flag_gems.fill_tensor_ if inplace else flag_gems.fill_tensor
    expected = gradient(
        native,
        utils.to_reference(x.detach(), False).requires_grad_(),
        utils.to_reference(value.detach(), False).requires_grad_(),
        utils.to_reference(dout.detach(), False).requires_grad_(),
    )
    actual = gradient(gems, x, value, dout)
    utils.gems_assert_equal(actual, expected)


FILL_SLICE_CASES = [
    # (shape, slice)
    ((4, 128), (slice(None), slice(64, None))),
    ((2, 1, 1, 512), (slice(None), slice(None), slice(None), slice(358, None))),
    ((8, 32, 64), (slice(None), slice(16, None))),
]


@pytest.mark.fill_scalar_
@pytest.mark.parametrize("shape, slc", FILL_SLICE_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.BOOL_TYPES)
@pytest.mark.parametrize(
    "value", [0, 1, True, float("-inf")], ids=["zero", "one", "true", "neginf"]
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_fill_scalar_sliced_view(shape, slc, dtype, value):
    if dtype == torch.bool and value == float("-inf"):
        # bool value cannot be -inf
        return

    x = torch.randn(shape, device=flag_gems.device).to(dtype)
    ref_x = utils.to_reference(x, False)

    ref_x[slc] = value
    flag_gems.fill_scalar_(x[slc], value)

    utils.gems_assert_equal(x, ref_x)


# fill_.Tensor
@pytest.mark.fill_tensor_
@pytest.mark.parametrize("value", [0, 1, 9])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_fill_(value, shape, dtype):
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = utils.to_reference(x.clone(), False)
    value_tensor = torch.tensor(value, device=flag_gems.device, dtype=dtype)

    if flag_gems.vendor_name == "mthreads":
        ref_x.fill_(value_tensor.cpu())
    else:
        ref_value_tensor = utils.to_reference(value_tensor)
        ref_x.fill_(ref_value_tensor)

    flag_gems.fill_tensor_(x, value_tensor)

    utils.gems_assert_equal(x, ref_x)


@pytest.mark.fill_tensor_
@pytest.mark.parametrize("shape, slc", FILL_SLICE_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.BOOL_TYPES)
@pytest.mark.parametrize("value", [0, 1, True], ids=["zero", "one", "true"])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_fill_sliced_view_tensor(shape, slc, dtype, value):
    x = torch.randn(shape, device=flag_gems.device).to(dtype)
    ref_x = utils.to_reference(x, False)

    value_tensor = torch.tensor(value, device=flag_gems.device, dtype=dtype)
    ref_value_tensor = utils.to_reference(value_tensor, False)
    ref_x[slc] = ref_value_tensor
    flag_gems.fill_tensor_(x[slc], value_tensor)

    utils.gems_assert_equal(x, ref_x)
