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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import TO_CPU

# Shapes representative of common batch-norm workloads.
# The feature dimension is always shape[1].
SHAPES = [
    (16, 3, 32),
    (32, 32, 32),
    (8, 32, 224, 224),
    (32, 64, 56, 56),
    (64, 128, 28, 28),
    (16, 256, 14, 14),
]


@pytest.mark.batch_norm_backward_reduce
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("input_g", [True, False])
@pytest.mark.parametrize("weight_g", [True, False])
@pytest.mark.parametrize("bias_g", [True, False])
def test_accuracy_batch_norm_backward_reduce(shape, dtype, input_g, weight_g, bias_g):
    # batch_norm_backward_reduce has no CPU backend in PyTorch, skip under --ref=cpu
    if TO_CPU:
        pytest.skip(
            "batch_norm_backward_reduce has no CPU backend; skip under --ref=cpu"
        )

    C = shape[1]

    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # mean and invstd must always be float32, even when input is float16
    mean = torch.randn(C, dtype=torch.float32, device=flag_gems.device)
    invstd = torch.randn(C, dtype=torch.float32, device=flag_gems.device).abs() + 0.1
    weight = torch.randn(C, dtype=dtype, device=flag_gems.device)

    # Reference: PyTorch native CUDA kernel with same-precision inputs.
    # We do NOT upcast because this op has no CPU backend and upcasting would
    # compare a high-precision reference against a low-precision kernel result,
    # where the precision gap of the *inputs* (not the implementation) dominates.
    ref_out = torch.batch_norm_backward_reduce(
        grad_output,
        inp,
        mean,
        invstd,
        weight,
        input_g,
        weight_g,
        bias_g,
    )

    with flag_gems.use_gems():
        res_out = torch.batch_norm_backward_reduce(
            grad_output,
            inp,
            mean,
            invstd,
            weight,
            input_g,
            weight_g,
            bias_g,
        )

    reduce_dim = math.prod(shape) // C

    # Tolerance is bounded by input dtype precision since different reduce
    # orderings in Triton vs PyTorch CUDA kernel cause implementation-level
    # differences proportional to input precision, not output precision.
    RTOL = {torch.float16: 1e-3, torch.bfloat16: 0.016, torch.float32: 1.3e-6}
    rtol = RTOL[dtype]
    atol = 1e-4 * reduce_dim

    for i, (ref_val, res_val) in enumerate(zip(ref_out, res_out)):
        if ref_val is None:
            assert res_val is None, f"Output {i}: expected None, got {res_val}"
        else:
            assert (
                ref_val.dtype == res_val.dtype
            ), f"Output {i}: dtype mismatch ref={ref_val.dtype} res={res_val.dtype}"
            torch.testing.assert_close(res_val, ref_val, atol=atol, rtol=rtol)
