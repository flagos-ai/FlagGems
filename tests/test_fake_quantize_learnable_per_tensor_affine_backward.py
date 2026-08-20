# Copyright 2026 FlagOS Contributors.
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

# The ATen reference rejects float16 gradients ("Expected dY_.scalar_type() ==
# Float"), so we only exercise the float32 and bfloat16 paths here. Both the
# reference and the Gems kernel return float32 gradients.
BACKWARD_DTYPES = [torch.float32]
if utils.bf16_is_supported:
    BACKWARD_DTYPES.append(torch.bfloat16)

# Symmetric (zero_point == 0) and asymmetric (non-zero zero_point) parameter
# sets spanning common quantization ranges.
QUANT_PARAMS = [
    # (quant_min, quant_max, zero_point, grad_factor)
    (-128, 127, 0.0, 1.0),
    (0, 255, 128.0, 1.0),
    (-127, 127, 0.0, 0.5),
    (-5, 5, 2.0, 0.7),
]


@pytest.mark.fake_quantize_learnable_per_tensor_affine_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("quant_min, quant_max, zero_point, grad_factor", QUANT_PARAMS)
def test_fake_quantize_learnable_per_tensor_affine_backward(
    shape, dtype, quant_min, quant_max, zero_point, grad_factor
):
    # `grad` (the upstream gradient) must be float32-compatible: the ATen
    # reference rejects float16, so we use float32 grads even for bfloat16
    # inputs to stay within the supported contract.
    self_t = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_t = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    scale_t = torch.tensor([0.1], dtype=dtype, device=flag_gems.device)
    zero_point_t = torch.tensor([zero_point], dtype=dtype, device=flag_gems.device)

    ref_self = utils.to_reference(self_t)
    ref_grad = utils.to_reference(grad_t)
    ref_scale = utils.to_reference(scale_t)
    ref_zero_point = utils.to_reference(zero_point_t)

    ref_out = torch.ops.aten._fake_quantize_learnable_per_tensor_affine_backward(
        ref_grad, ref_self, ref_scale, ref_zero_point, quant_min, quant_max, grad_factor
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._fake_quantize_learnable_per_tensor_affine_backward(
            grad_t,
            self_t,
            scale_t,
            zero_point_t,
            quant_min,
            quant_max,
            grad_factor,
        )

    # The op returns a tuple (grad_self, grad_scale, grad_zero_point), all float32.
    # grad_self is element-wise (STE mask * grad), compared at default tolerance.
    # grad_scale / grad_zero_point are full reductions over the input tensor:
    # float32 reduction order (tree-reduce in Triton vs. sequential on the CPU
    # reference) differs in the last bits, so scale the atol by the reduction
    # length (matching the convention used by the sum/mean accuracy tests).
    utils.gems_assert_close(res_out[0], ref_out[0], torch.float32)
    reduce_dim = max(1, self_t.numel())
    utils.gems_assert_close(
        res_out[1], ref_out[1], torch.float32, reduce_dim=reduce_dim
    )
    utils.gems_assert_close(
        res_out[2], ref_out[2], torch.float32, reduce_dim=reduce_dim
    )


@pytest.mark.fake_quantize_learnable_per_tensor_affine_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_fake_quantize_learnable_per_tensor_affine_backward_clamped(shape, dtype):
    """Exercise the clamp boundary: with a tiny scale many inputs fall outside
    the quantization range, so the STE mask zeros the corresponding gradients
    and grad_zero_point becomes non-trivial."""
    self_t = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 100.0
    grad_t = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    scale_t = torch.tensor([0.01], dtype=dtype, device=flag_gems.device)
    zero_point_t = torch.tensor([0.0], dtype=dtype, device=flag_gems.device)
    quant_min, quant_max, grad_factor = -3, 3, 1.0

    ref_self = utils.to_reference(self_t)
    ref_grad = utils.to_reference(grad_t)
    ref_scale = utils.to_reference(scale_t)
    ref_zero_point = utils.to_reference(zero_point_t)

    ref_out = torch.ops.aten._fake_quantize_learnable_per_tensor_affine_backward(
        ref_grad, ref_self, ref_scale, ref_zero_point, quant_min, quant_max, grad_factor
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._fake_quantize_learnable_per_tensor_affine_backward(
            grad_t,
            self_t,
            scale_t,
            zero_point_t,
            quant_min,
            quant_max,
            grad_factor,
        )

    # grad_self is element-wise; grad_scale / grad_zero_point are full
    # reductions (scale the atol by the reduction length, matching the sum/mean
    # accuracy-test convention for float32 reduction-order sensitivity).
    utils.gems_assert_close(res_out[0], ref_out[0], torch.float32)
    reduce_dim = max(1, self_t.numel())
    utils.gems_assert_close(
        res_out[1], ref_out[1], torch.float32, reduce_dim=reduce_dim
    )
    utils.gems_assert_close(
        res_out[2], ref_out[2], torch.float32, reduce_dim=reduce_dim
    )
