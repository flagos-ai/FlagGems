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

BACKWARD_DTYPES = [torch.float32]
if utils.bf16_is_supported:
    BACKWARD_DTYPES.append(torch.bfloat16)
QUANT_PARAMS = [
    (-128, 127, 0.0, 1.0),
    (0, 255, 128.0, 1.0),
    (-127, 127, 0.0, 0.5),
    (-5, 5, 2.0, 0.7),
]
_ATEN_OP = torch.ops.aten._fake_quantize_learnable_per_tensor_affine_backward


def _seed():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def _reference(tensor):
    return utils.to_reference(tensor, True).to(tensor.dtype)


def _reference_backward(inputs, quant_min, quant_max, grad_factor):
    reference_inputs = tuple(_reference(tensor) for tensor in inputs)
    try:
        return _ATEN_OP(*reference_inputs, quant_min, quant_max, grad_factor)
    except (RuntimeError, ValueError) as error:
        if (
            reference_inputs[1].dtype == torch.bfloat16
            and (
                (
                    isinstance(error, RuntimeError)
                    and "Expected X.scalar_type() == ScalarType::Float to be true, but got false."
                    in str(error)
                )
                or (
                    flag_gems.vendor_name == "kunlunxin"
                    and reference_inputs[1].device.type == "cuda"
                    and isinstance(error, ValueError)
                    and "[NOT IMPLEMENTED]: error code= 4(at " in str(error)
                    and "generated/CUDANativeFunctions.cpp:409)" in str(error)
                )
            )
        ):
            pytest.skip("This native fake-quant backward reference rejects bfloat16")
        raise


@pytest.mark.fake_quantize_learnable_per_tensor_affine_backward
@pytest.mark.underscore_fake_quantize_learnable_per_tensor_affine_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("quant_min, quant_max, zero_point, grad_factor", QUANT_PARAMS)
def test_fake_quantize_learnable_per_tensor_affine_backward(
    shape, dtype, quant_min, quant_max, zero_point, grad_factor
):
    _seed()
    self_t = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    scale = torch.tensor([0.1], dtype=dtype, device=flag_gems.device)
    zero_point_t = torch.tensor([zero_point], dtype=dtype, device=flag_gems.device)
    inputs = grad, self_t, scale, zero_point_t
    snapshots = tuple(tensor.clone() for tensor in inputs)
    ref_out = _reference_backward(inputs, quant_min, quant_max, grad_factor)
    result = flag_gems._fake_quantize_learnable_per_tensor_affine_backward(
        *inputs, quant_min, quant_max, grad_factor
    )

    reduce_size = max(1, self_t.numel())
    utils.gems_assert_close(result[0], ref_out[0], torch.float32)
    utils.gems_assert_close(
        result[1], ref_out[1], torch.float32, reduce_dim=reduce_size
    )
    utils.gems_assert_close(
        result[2], ref_out[2], torch.float32, reduce_dim=reduce_size
    )
    for value, snapshot in zip(inputs, snapshots):
        assert value.device == snapshot.device
        utils.gems_assert_equal(value, utils.to_reference(snapshot))
    assert result[0].shape == self_t.shape
    assert result[1].shape == scale.shape
    assert result[2].shape == zero_point_t.shape
    assert all(value.dtype == torch.float32 for value in result)


@pytest.mark.fake_quantize_learnable_per_tensor_affine_backward
@pytest.mark.underscore_fake_quantize_learnable_per_tensor_affine_backward
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_fake_quantize_learnable_per_tensor_affine_backward_clamped(dtype):
    _seed()
    self_t = torch.randn((8, 16), dtype=dtype, device=flag_gems.device) * 100.0
    grad = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    scale = torch.tensor([0.01], dtype=dtype, device=flag_gems.device)
    zero_point = torch.tensor([0.0], dtype=dtype, device=flag_gems.device)
    inputs = grad, self_t, scale, zero_point
    ref_out = _reference_backward(inputs, -3, 3, 1.0)
    result = flag_gems._fake_quantize_learnable_per_tensor_affine_backward(
        *inputs, -3, 3, 1.0
    )

    utils.gems_assert_close(result[0], ref_out[0], torch.float32)
    utils.gems_assert_close(result[1], ref_out[1], torch.float32, reduce_dim=128)
    utils.gems_assert_close(result[2], ref_out[2], torch.float32, reduce_dim=128)


@pytest.mark.fake_quantize_learnable_per_tensor_affine_backward
@pytest.mark.underscore_fake_quantize_learnable_per_tensor_affine_backward
def test_fake_quantize_learnable_per_tensor_affine_backward_empty():
    if flag_gems.vendor_name == "kunlunxin" and not utils.TO_CPU:
        pytest.skip("Kunlunxin native fake-quant backward rejects empty input")
    self_t = torch.empty((2, 0, 3), device=flag_gems.device)
    grad = torch.empty_like(self_t)
    scale = torch.tensor([0.1], device=flag_gems.device)
    zero_point = torch.tensor([0.0], device=flag_gems.device)
    reference_inputs = tuple(
        _reference(tensor) for tensor in (grad, self_t, scale, zero_point)
    )
    reference = _ATEN_OP(*reference_inputs, -128, 127, 1.0)
    if (
        (
            (
                reference[1].data_ptr() == reference_inputs[2].data_ptr()
                and reference[2].data_ptr() == reference_inputs[3].data_ptr()
            )
            or (
                flag_gems.vendor_name in {"ascend", "enflame"}
                and not utils.TO_CPU
                and torch.equal(reference[1], reference_inputs[2])
                and torch.equal(reference[2], reference_inputs[3])
            )
        )
        and torch.count_nonzero(reference[1]).item() != 0
    ):
        pytest.skip("Native empty backward returns scale instead of a zero gradient")
    result = flag_gems._fake_quantize_learnable_per_tensor_affine_backward(
        grad, self_t, scale, zero_point, -128, 127, 1.0
    )

    assert result[0].shape == self_t.shape
    assert result[1].shape == scale.shape
    assert result[2].shape == zero_point.shape
    utils.gems_assert_equal(result[1], torch.zeros_like(scale))
    utils.gems_assert_equal(result[2], torch.zeros_like(zero_point))


@pytest.mark.fake_quantize_learnable_per_tensor_affine_backward
@pytest.mark.underscore_fake_quantize_learnable_per_tensor_affine_backward
def test_fake_quantize_learnable_per_tensor_affine_backward_rejects_integer_input():
    self_t = torch.ones((4, 8), dtype=torch.int32, device=flag_gems.device)
    grad = torch.ones((4, 8), device=flag_gems.device)
    scale = torch.tensor([0.1], device=flag_gems.device)
    zero_point = torch.tensor([0.0], device=flag_gems.device)
    with pytest.raises((ValueError, RuntimeError)):
        flag_gems._fake_quantize_learnable_per_tensor_affine_backward(
            grad, self_t, scale, zero_point, -128, 127, 1.0
        )
