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

_EQUAL_SHAPES = [*utils.POINTWISE_SHAPES, (2, 0, 3)]
_BROADCAST_CASES = [
    pytest.param((7, 13), (1, 13), id="leading-dimension"),
    pytest.param((2, 3, 5), (1, 3, 1), id="multiple-dimensions"),
    pytest.param((2, 3, 5), (), id="scalar-tensor"),
    pytest.param((2, 0, 3), (1, 0, 1), id="zero-size"),
]


def _nonzero_input(shape, dtype):
    tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return tensor.sign().masked_fill_(tensor == 0, 1) * (tensor.abs() + 0.5)


@pytest.mark.divide
@pytest.mark.parametrize("shape", _EQUAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_divide_equal_shape(shape, dtype, caplog, request):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = _nonzero_input(shape, dtype)
    inp_before = inp.clone()
    other_before = other.clone()
    ref_inp = utils.to_reference(inp, True)
    ref_other = utils.to_reference(other, True)

    ref_out = torch.ops.aten.divide.Tensor(ref_inp, ref_other)
    with caplog.at_level("DEBUG", logger="flag_gems.ops.divide"):
        res_out = flag_gems.divide(inp, other)

    registry = getattr(request.config, "_override_registry", None)
    if registry is None or registry.get_override("divide") is None:
        assert f"{utils.gems_log_prefix(flag_gems.divide)} DIVIDE" in caplog.text
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_equal(inp, inp_before)
    utils.gems_assert_equal(other, other_before)


@pytest.mark.divide
@pytest.mark.parametrize("self_shape,other_shape", _BROADCAST_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_divide_broadcast(self_shape, other_shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(self_shape, dtype=dtype, device=flag_gems.device)
    other = _nonzero_input(other_shape, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_other = utils.to_reference(other, True)

    ref_out = torch.ops.aten.divide.Tensor(ref_inp, ref_other)
    res_out = flag_gems.divide(inp, other)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.divide
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_divide_noncontiguous(dtype):
    torch.manual_seed(0)
    inp = torch.randn((4, 7, 9), dtype=dtype, device=flag_gems.device).transpose(0, 2)
    other = _nonzero_input((4, 7, 9), dtype).transpose(0, 2)
    assert not inp.is_contiguous()
    assert not other.is_contiguous()
    ref_inp = utils.to_reference(inp, True)
    ref_other = utils.to_reference(other, True)

    ref_out = torch.ops.aten.divide.Tensor(ref_inp, ref_other)
    res_out = flag_gems.divide(inp, other)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.divide
def test_divide_special_values():
    inp = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan")],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    other = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), 2.0, 2.0, 1.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)
    ref_other = utils.to_reference(other, True)

    ref_out = torch.ops.aten.divide.Tensor(ref_inp, ref_other)
    res_out = flag_gems.divide(inp, other)

    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.divide
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_divide_backward(dtype):
    torch.manual_seed(0)
    inp = torch.randn((4, 7), dtype=dtype, device=flag_gems.device, requires_grad=True)
    other = _nonzero_input((4, 7), dtype).requires_grad_(True)
    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    ref_other = utils.to_reference(other.detach(), True).requires_grad_(True)

    ref_out = torch.ops.aten.divide.Tensor(ref_inp, ref_other)
    res_out = flag_gems.divide(inp, other)

    grad_output = torch.randn_like(res_out)
    ref_grad_output = utils.to_reference(grad_output, True)
    ref_grads = torch.autograd.grad(ref_out, (ref_inp, ref_other), ref_grad_output)
    res_grads = torch.autograd.grad(res_out, (inp, other), grad_output)

    utils.gems_assert_close(res_grads[0], ref_grads[0], dtype)
    utils.gems_assert_close(res_grads[1], ref_grads[1], dtype)


@pytest.mark.divide
def test_divide_rejects_nonbroadcastable_shape():
    inp = torch.ones((2, 3), device=flag_gems.device)
    other = torch.ones((4,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.divide(inp, other)
