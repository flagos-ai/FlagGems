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

TRACE_BACKWARD_SHAPES = [
    (1, 1),
    (3, 3),
    (4, 3),
    (3, 4),
    (128, 128),
    (200, 100),
    (100, 200),
    (1024, 1024),
    (1, 1000),
    (1000, 1),
    (0, 10),
    (10, 0),
    (0, 0),
    (1025, 1025),
]


@pytest.mark.trace_backward
@pytest.mark.parametrize("shape", TRACE_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trace_backward(shape, dtype):
    res_grad = torch.randn((), dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(res_grad)

    ref_out = torch.ops.aten.trace_backward(ref_grad, list(shape))
    res_out = flag_gems.trace_backward(res_grad, list(shape))

    utils.gems_assert_equal(res_out, ref_out)
