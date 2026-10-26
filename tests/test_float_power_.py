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


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


def _reference_inplace(inp, exponent):
    result = torch.float_power(inp, exponent)
    inp.copy_(result)
    return inp


def _assert_inplace(result, inp, pointer, reference, dtype):
    assert result is inp
    assert result.data_ptr() == pointer
    assert result.shape == reference.shape
    assert result.dtype == dtype
    utils.gems_assert_close(result, reference, dtype, equal_nan=True)


def _assert_tensor_unchanged(tensor, snapshot, pointer):
    assert tensor.data_ptr() == pointer
    assert tensor.device == snapshot.device
    utils.gems_assert_equal(tensor, utils.to_reference(snapshot, False))


@pytest.mark.float_power_tensor_tensor_
@pytest.mark.float_power_
@pytest.mark.parametrize("shape", [*utils.POINTWISE_SHAPES, (2, 0, 3)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_float_power_tensor_equal_shape_(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(0.25, 2.0)
    exponent = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
        -2.0, 3.0
    )
    exponent_before = exponent.clone()
    exponent_pointer = exponent.data_ptr()
    pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_exponent = utils.to_reference(exponent, True)
    ref_out = _reference_inplace(ref_inp, ref_exponent)

    result = flag_gems.float_power_tensor_tensor_(inp, exponent)

    _assert_inplace(result, inp, pointer, ref_out, dtype)
    _assert_tensor_unchanged(exponent, exponent_before, exponent_pointer)


@pytest.mark.float_power_tensor_tensor_
@pytest.mark.float_power_
@pytest.mark.parametrize(
    "self_shape,exponent_shape",
    [((2, 3, 5), (1, 3, 1)), ((7, 13), (1, 13)), ((2, 0, 3), (1, 0, 1))],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_float_power_tensor_broadcast_(self_shape, exponent_shape, dtype):
    inp = torch.empty(self_shape, dtype=dtype, device=flag_gems.device).uniform_(0.25, 2.0)
    exponent = torch.empty(
        exponent_shape, dtype=dtype, device=flag_gems.device
    ).uniform_(-2.0, 3.0)
    exponent_before = exponent.clone()
    exponent_pointer = exponent.data_ptr()
    pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_exponent = utils.to_reference(exponent, True)
    ref_out = _reference_inplace(ref_inp, ref_exponent)

    result = flag_gems.float_power_tensor_tensor_(inp, exponent)

    _assert_inplace(result, inp, pointer, ref_out, dtype)
    _assert_tensor_unchanged(exponent, exponent_before, exponent_pointer)


@pytest.mark.float_power_tensor_scalar_
@pytest.mark.float_power_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("exponent", [-2.0, 0.5, 2.0, 3.0])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_float_power_scalar_(shape, exponent, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(0.25, 2.0)
    pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_out = _reference_inplace(ref_inp, exponent)

    result = flag_gems.float_power_tensor_scalar_(inp, exponent)

    _assert_inplace(result, inp, pointer, ref_out, dtype)


@pytest.mark.float_power_tensor_tensor_
@pytest.mark.float_power_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_float_power_noncontiguous_(dtype):
    base = torch.empty((4, 7, 9), dtype=dtype, device=flag_gems.device).uniform_(0.25, 2.0)
    inp = base.transpose(0, 2)
    exponent = torch.full(inp.shape, 2.0, dtype=dtype, device=flag_gems.device)
    exponent_before = exponent.clone()
    exponent_pointer = exponent.data_ptr()
    pointer = inp.data_ptr()
    stride = inp.stride()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_exponent = utils.to_reference(exponent, True)
    ref_out = _reference_inplace(ref_inp, ref_exponent)

    result = flag_gems.float_power_tensor_tensor_(inp, exponent)

    assert result.stride() == stride
    _assert_inplace(result, inp, pointer, ref_out, dtype)
    _assert_tensor_unchanged(exponent, exponent_before, exponent_pointer)


@pytest.mark.float_power_tensor_tensor_
@pytest.mark.float_power_
def test_float_power_special_values_():
    inp = torch.tensor([0.0, 1.0, float("inf"), float("nan")], dtype=torch.float32, device=flag_gems.device)
    exponent = torch.tensor(
        [2.0, -1.0, 0.0, 1.0], dtype=torch.float32, device=flag_gems.device
    )
    exponent_before = exponent.clone()
    exponent_pointer = exponent.data_ptr()
    pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_exponent = utils.to_reference(exponent, True)
    ref_out = _reference_inplace(ref_inp, ref_exponent)

    result = flag_gems.float_power_tensor_tensor_(inp, exponent)

    _assert_inplace(result, inp, pointer, ref_out, torch.float32)
    _assert_tensor_unchanged(exponent, exponent_before, exponent_pointer)


@pytest.mark.float_power_tensor_tensor_
@pytest.mark.float_power_
def test_float_power_backward_():
    leaf = torch.empty((4, 7), dtype=torch.float64, device=flag_gems.device).uniform_(0.5, 2.0).requires_grad_(True)
    exponent = (
        torch.empty((4, 7), dtype=torch.float64, device=flag_gems.device)
        .uniform_(0.5, 2.0)
        .requires_grad_(True)
    )
    exponent_before = exponent.detach().clone()
    exponent_pointer = exponent.data_ptr()
    inp = leaf + 0
    ref_leaf = utils.to_reference(leaf.detach().clone(), False).requires_grad_(True)
    ref_exponent = utils.to_reference(exponent.detach().clone(), False).requires_grad_(True)
    ref_inp = ref_leaf + 0
    ref_out = ref_inp.float_power_(ref_exponent)

    result = flag_gems.float_power_tensor_tensor_(inp, exponent)

    _assert_tensor_unchanged(
        exponent.detach(), exponent_before, exponent_pointer
    )
    upstream = torch.randn_like(result)
    ref_grads = torch.autograd.grad(ref_out, (ref_leaf, ref_exponent), utils.to_reference(upstream, False))
    result_grads = torch.autograd.grad(result, (leaf, exponent), upstream)

    assert result is inp
    utils.gems_assert_close(result_grads[0], ref_grads[0], torch.float64)
    utils.gems_assert_close(result_grads[1], ref_grads[1], torch.float64)


@pytest.mark.float_power_tensor_tensor_
@pytest.mark.float_power_
def test_float_power_rejects_nonbroadcastable_shape_():
    inp = torch.ones((2, 3), dtype=torch.float64, device=flag_gems.device)
    exponent = torch.ones((4,), dtype=torch.float64, device=flag_gems.device)
    exponent_before = exponent.clone()
    exponent_pointer = exponent.data_ptr()

    with pytest.raises(RuntimeError):
        flag_gems.float_power_tensor_tensor_(inp, exponent)

    _assert_tensor_unchanged(exponent, exponent_before, exponent_pointer)
