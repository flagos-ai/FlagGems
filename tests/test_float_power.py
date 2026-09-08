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

DTYPES = utils.FLOAT_DTYPES + [torch.int32]
TENSOR_TENSOR_SHAPES = [
    ((), ()),
    ((0,), (1,)),
    ((7,), (7,)),
    ((2, 3), (3,)),
    ((4, 1, 5), (1, 3, 1)),
]
POINTWISE_SHAPES = [(), (0,), (7,), (2, 3), (4, 3, 5)]


def _make_input(shape, dtype, *, positive=False):
    if dtype.is_floating_point:
        tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        return tensor.abs().add_(0.25) if positive else tensor
    return torch.randint(
        1 if positive else -3, 5, shape, dtype=dtype, device=flag_gems.device
    )


def _assert_result(result, reference):
    assert result.dtype == torch.float64
    utils.gems_assert_close(
        result, reference, torch.float64, equal_nan=True, atol=1e-12
    )


@pytest.mark.float_power_tensor_tensor
@pytest.mark.parametrize("shapes", TENSOR_TENSOR_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_float_power_tensor_tensor(shapes, dtype):
    base = _make_input(shapes[0], dtype, positive=True)
    exponent = _make_input(shapes[1], dtype)
    ref = torch.ops.aten.float_power.Tensor_Tensor(
        utils.to_reference(base), utils.to_reference(exponent)
    )

    with flag_gems.use_gems():
        result = torch.ops.aten.float_power.Tensor_Tensor(base, exponent)

    _assert_result(result, ref)


@pytest.mark.float_power_tensor_tensor
def test_float_power_tensor_tensor_noncontiguous_and_special_values():
    base = torch.tensor(
        [[0.0, -0.0, 1.0, -1.0], [2.0, float("inf"), float("nan"), 4.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    ).T
    exponent = torch.tensor(
        [[-1.0, 3.0, 0.5, 2.0], [0.0, -2.0, 1.0, float("inf")]],
        dtype=torch.float32,
        device=flag_gems.device,
    ).T
    ref = torch.ops.aten.float_power.Tensor_Tensor(
        utils.to_reference(base), utils.to_reference(exponent)
    )

    with flag_gems.use_gems():
        result = torch.ops.aten.float_power.Tensor_Tensor(base, exponent)

    _assert_result(result, ref)


@pytest.mark.float_power_tensor_scalar
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("exponent", [2.0, -0.5])
def test_float_power_tensor_scalar(shape, dtype, exponent):
    base = _make_input(shape, dtype, positive=True)
    ref = torch.ops.aten.float_power.Tensor_Scalar(utils.to_reference(base), exponent)

    with flag_gems.use_gems():
        result = torch.ops.aten.float_power.Tensor_Scalar(base, exponent)

    _assert_result(result, ref)


@pytest.mark.float_power_scalar_tensor
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("base", [2.0, -2.0])
def test_float_power_scalar_tensor(shape, dtype, base):
    exponent = _make_input(shape, dtype)
    ref = torch.ops.aten.float_power.Scalar(base, utils.to_reference(exponent))

    with flag_gems.use_gems():
        result = torch.ops.aten.float_power.Scalar(base, exponent)

    _assert_result(result, ref)


@pytest.mark.float_power_tensor_tensor_out
@pytest.mark.parametrize("shapes", TENSOR_TENSOR_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_float_power_tensor_tensor_out(shapes, dtype):
    base = _make_input(shapes[0], dtype, positive=True)
    exponent = _make_input(shapes[1], dtype)
    result_shape = torch.broadcast_shapes(*shapes)
    ref_out = torch.empty(
        result_shape, dtype=torch.float64, device=utils.to_reference(base).device
    )
    ref = torch.ops.aten.float_power.Tensor_Tensor_out(
        utils.to_reference(base), utils.to_reference(exponent), out=ref_out
    )
    out = torch.empty(result_shape, dtype=torch.float64, device=flag_gems.device)

    with flag_gems.use_gems():
        result = torch.ops.aten.float_power.Tensor_Tensor_out(base, exponent, out=out)

    assert result is out
    _assert_result(result, ref)


@pytest.mark.float_power_tensor_scalar_out
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_float_power_tensor_scalar_out(shape, dtype):
    base = _make_input(shape, dtype, positive=True)
    ref_base = utils.to_reference(base)
    ref_out = torch.empty(shape, dtype=torch.float64, device=ref_base.device)
    ref = torch.ops.aten.float_power.Tensor_Scalar_out(ref_base, -0.5, out=ref_out)
    out = torch.empty(0, dtype=torch.float64, device=flag_gems.device)

    with flag_gems.use_gems():
        result = torch.ops.aten.float_power.Tensor_Scalar_out(base, -0.5, out=out)

    assert result is out
    assert result.shape == base.shape
    _assert_result(result, ref)


@pytest.mark.float_power_scalar_tensor_out
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_float_power_scalar_tensor_out(shape, dtype):
    exponent = _make_input(shape, dtype)
    ref_exponent = utils.to_reference(exponent)
    ref_out = torch.empty(shape, dtype=torch.float64, device=ref_exponent.device)
    ref = torch.ops.aten.float_power.Scalar_out(2.0, ref_exponent, out=ref_out)
    out = torch.empty(shape, dtype=torch.float64, device=flag_gems.device)

    with flag_gems.use_gems():
        result = torch.ops.aten.float_power.Scalar_out(2.0, exponent, out=out)

    assert result is out
    _assert_result(result, ref)


@pytest.mark.float_power_tensor_tensor_out
def test_float_power_tensor_tensor_out_noncontiguous():
    base = torch.rand((3, 4), device=flag_gems.device)
    exponent = torch.rand((3, 4), device=flag_gems.device)
    out = torch.empty((4, 3), dtype=torch.float64, device=flag_gems.device).T
    ref = torch.float_power(utils.to_reference(base), utils.to_reference(exponent))

    with flag_gems.use_gems():
        result = torch.ops.aten.float_power.Tensor_Tensor_out(base, exponent, out=out)

    assert result is out
    assert not result.is_contiguous()
    _assert_result(result, ref)


@pytest.mark.float_power_tensor_scalar_out
def test_float_power_out_rejects_non_double_output():
    base = torch.rand(8, device=flag_gems.device)
    out = torch.empty_like(base)

    with (
        flag_gems.use_gems(),
        pytest.raises(RuntimeError, match="requires dtype Double"),
    ):
        torch.ops.aten.float_power.Tensor_Scalar_out(base, 2.0, out=out)
