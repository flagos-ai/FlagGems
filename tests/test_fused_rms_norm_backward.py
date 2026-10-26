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

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = [(2, 16), (4, 64), (8, 128)]


def _reference_backward(grad_out, inp, normalized_shape, rstd, weight, output_mask):
    ndim_batch = inp.ndim - len(normalized_shape)
    reduce_dims = tuple(range(ndim_batch, inp.ndim))
    upcast = inp.to(torch.float32)
    inv_rms = rstd.to(torch.float32).reshape(
        inp.shape[:ndim_batch] + (1,) * len(normalized_shape)
    )
    normalized = upcast * inv_rms
    numel = int(np.prod(normalized_shape))
    grad_input = None
    if output_mask[0]:
        grad = grad_out.to(torch.float32)
        if weight is not None:
            grad = grad * weight.to(torch.float32)
        row_sum = (normalized * grad).sum(dim=reduce_dims, keepdim=True)
        grad_input = ((grad - normalized / numel * row_sum) * inv_rms).to(inp.dtype)
    grad_weight = None
    if output_mask[1] and weight is not None:
        batch_dims = tuple(range(ndim_batch))
        grad_weight = (grad_out.to(torch.float32) * normalized).sum(dim=batch_dims)
        grad_weight = grad_weight.reshape(normalized_shape).to(inp.dtype)
    return grad_input, grad_weight


@pytest.mark.fused_rms_norm_backward
@pytest.mark.underscore_fused_rms_norm_backward
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("output_mask", [(True, True), (True, False), (False, True)])
def test_fused_rms_norm_backward(shape, dtype, output_mask):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 0.1
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 0.01
    weight = torch.randn((shape[-1],), dtype=dtype, device=flag_gems.device) * 0.1
    rstd = torch.rsqrt(inp.to(torch.float32).pow(2).mean(dim=-1) + 1e-5)
    ref = _reference_backward(
        utils.to_reference(grad, True), utils.to_reference(inp, True),
        [shape[-1]], utils.to_reference(rstd, False),
        utils.to_reference(weight, True), output_mask,
    )
    result = flag_gems._fused_rms_norm_backward(
        grad, inp, [shape[-1]], rstd, weight, list(output_mask)
    )
    for enabled, actual, expected in zip(output_mask, result, ref):
        if enabled:
            assert actual.device == inp.device
            assert actual.dtype == dtype
            assert actual.shape == expected.shape
            utils.gems_assert_close(actual, expected, dtype, reduce_dim=shape[-1])
        else:
            assert actual is None


@pytest.mark.fused_rms_norm_backward
@pytest.mark.underscore_fused_rms_norm_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fused_rms_norm_backward_without_weight(dtype):
    torch.manual_seed(0)
    shape = (4, 32)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    rstd = torch.rsqrt(inp.to(torch.float32).pow(2).mean(dim=-1) + 1e-5)
    ref_input, _ = _reference_backward(
        utils.to_reference(grad, True), utils.to_reference(inp, True), [32],
        utils.to_reference(rstd, False), None, (True, False),
    )
    grad_input, grad_weight = flag_gems._fused_rms_norm_backward(
        grad, inp, [32], rstd, None, [True, False]
    )
    assert grad_weight is None
    assert grad_input.device == inp.device
    assert grad_input.dtype == dtype
    assert grad_input.shape == inp.shape
    utils.gems_assert_close(grad_input, ref_input, dtype, reduce_dim=32)
