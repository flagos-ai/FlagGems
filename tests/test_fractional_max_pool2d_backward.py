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


CASES = [
    ((4, 3, 32, 32), 2, (16, 16)),
    ((8, 16, 28, 28), 2, (14, 14)),
    ((2, 4, 20, 20), 2, (10, 10)),
    ((1, 64, 56, 56), 2, (28, 28)),
    ((2, 8, 16, 16), 2, (8, 8)),
    ((2, 8, 16, 20), 2, (8, 10)),
]


@pytest.mark.fractional_max_pool2d_backward
@pytest.mark.parametrize("shape,kernel_size,output_size", CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fractional_max_pool2d_backward(shape, kernel_size, output_size, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)
    random_samples = torch.rand(
        shape[0], shape[1], 2, dtype=torch.float32, device=flag_gems.device
    )
    ref_random_samples = random_samples.to(device=ref_inp.device, dtype=ref_inp.dtype)
    ref_out, ref_indices = torch.nn.functional.fractional_max_pool2d(
        ref_inp,
        kernel_size=kernel_size,
        output_size=output_size,
        return_indices=True,
        _random_samples=ref_random_samples,
    )
    # Backward consumes trusted native metadata; it must not require a separate
    # unoverridden generated forward kernel to run first.
    actual_indices = ref_indices.to(device=inp.device)
    grad_output = torch.randn(ref_out.shape, dtype=dtype, device=inp.device)
    ref_grad_output = utils.to_reference(grad_output, True)
    (expected,) = torch.autograd.grad(ref_out, ref_inp, ref_grad_output)
    actual = flag_gems.fractional_max_pool2d_backward(
        grad_output,
        inp,
        kernel_size=kernel_size,
        output_size=output_size,
        indices=actual_indices,
    )

    assert actual.device == inp.device
    assert actual.shape == inp.shape
    assert actual.dtype == inp.dtype
    utils.gems_assert_close(actual, expected, dtype)
