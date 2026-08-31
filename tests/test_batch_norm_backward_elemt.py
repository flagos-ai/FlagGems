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

# Shapes representative of common batch-norm workloads (2D up to 5D).
# The feature dimension is always shape[1].
SHAPES = [
    (16, 3),
    (32, 32, 32),
    (8, 32, 224, 224),
    (2050, 16, 32, 32),
    (8, 16, 3, 224, 224),
]


def _ref_batch_norm_backward_elemt(
    grad_out, inp, mean, invstd, weight, sum_dy, sum_dy_xmu, count
):
    """Pure-tensor reference for batch_norm_backward_elemt."""
    ndim = grad_out.ndim
    shape = [1] * ndim
    shape[1] = -1
    mean = mean.reshape(shape)
    invstd = invstd.reshape(shape)
    sum_dy = sum_dy.reshape(shape)
    sum_dy_xmu = sum_dy_xmu.reshape(shape)
    if weight is not None:
        weight = weight.reshape(shape)

    count_val = count.to(grad_out.dtype).item()
    xmu = inp - mean
    grad_input = invstd * (
        grad_out - (sum_dy + xmu * (invstd * invstd) * sum_dy_xmu) / count_val
    )
    if weight is not None:
        grad_input = weight * grad_input
    return grad_input


@pytest.mark.batch_norm_backward_elemt
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("has_weight", [True, False])
def test_batch_norm_backward_elemt(shape, dtype, has_weight):
    C = shape[1]
    spatial = 1
    for s in shape[2:]:
        spatial *= s
    N = shape[0]
    count_val = N * spatial

    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mean = torch.randn(C, dtype=dtype, device=flag_gems.device)
    invstd = torch.randn(C, dtype=dtype, device=flag_gems.device).abs() + 0.1
    weight = (
        torch.randn(C, dtype=dtype, device=flag_gems.device) if has_weight else None
    )
    sum_dy = torch.randn(C, dtype=dtype, device=flag_gems.device)
    sum_dy_xmu = torch.randn(C, dtype=dtype, device=flag_gems.device)
    count = torch.tensor([count_val], dtype=torch.int32, device=flag_gems.device)

    # Compute reference using manual formula with upcast precision.
    # batch_norm_backward_elemt has no CPU backend, so we stay on GPU
    # and use to_reference only for dtype upcast (not device move).
    ref_grad_output = utils.to_reference(grad_output, True)
    ref_inp = utils.to_reference(inp, True)
    ref_mean = utils.to_reference(mean, True)
    ref_invstd = utils.to_reference(invstd, True)
    ref_weight = utils.to_reference(weight, True) if weight is not None else None
    ref_sum_dy = utils.to_reference(sum_dy, True)
    ref_sum_dy_xmu = utils.to_reference(sum_dy_xmu, True)
    ref_count = utils.to_reference(count, False)

    ref_out = _ref_batch_norm_backward_elemt(
        ref_grad_output,
        ref_inp,
        ref_mean,
        ref_invstd,
        ref_weight,
        ref_sum_dy,
        ref_sum_dy_xmu,
        ref_count,
    )

    with flag_gems.use_gems():
        res_out = torch.batch_norm_backward_elemt(
            grad_output,
            inp,
            mean,
            invstd,
            weight,
            sum_dy,
            sum_dy_xmu,
            count,
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
