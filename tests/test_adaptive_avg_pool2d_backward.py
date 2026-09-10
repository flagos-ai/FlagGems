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


@pytest.mark.adaptive_avg_pool2d_backward
# Covers batched, global-pooling, unbatched, and non-square adaptive windows.
@pytest.mark.parametrize(
    "shape,output_size",
    [
        ((2, 3, 7, 9), (3, 4)),
        ((1, 2, 8, 8), (1, 1)),
        ((3, 5, 6), (2, 4)),
        ((1, 2, 65, 71), (2, 3)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool2d_backward(shape, output_size, dtype):
    inp = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    grad_output = torch.randn(
        (*shape[:-2], *output_size), device=flag_gems.device, dtype=dtype
    )
    ref_inp = utils.to_reference(inp, True)
    ref_grad_output = utils.to_reference(grad_output, True)
    expected = torch.ops.aten._adaptive_avg_pool2d_backward(ref_grad_output, ref_inp)

    with flag_gems.use_gems():
        actual = torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, inp)

    utils.gems_assert_close(
        actual, expected, dtype, reduce_dim=output_size[0] * output_size[1]
    )
