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

# Cover 3D input, small/large 4D inputs for representative im2col testing
IM2COL_SHAPES = [(3, 8, 8), (1, 3, 16, 16), (16, 64, 64), (32, 128, 128)]
IM2COL_CONFIGS = [
    ((3, 3), (1, 1), (1, 1), (1, 1)),
    ((3, 3), (1, 1), (0, 0), (2, 2)),
    ((5, 4), (2, 2), (2, 1), (1, 2)),
    ((1, 1), (1, 1), (0, 0), (1, 1)),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.im2col
@pytest.mark.parametrize("shape", IM2COL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("kernel_size, dilation, padding, stride", IM2COL_CONFIGS)
def test_im2col(shape, dtype, kernel_size, dilation, padding, stride):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    ref_out = torch.ops.aten.im2col(ref_x, kernel_size, dilation, padding, stride)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.im2col(x, kernel_size, dilation, padding, stride)

    utils.gems_assert_close(act_out, ref_out, dtype=dtype)
