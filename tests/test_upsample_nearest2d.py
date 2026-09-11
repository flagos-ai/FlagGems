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

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)

if cfg.QUICK_MODE:
    UPSAMPLE_NEAREST2D_SCALES = [(2, 2)]
else:
    UPSAMPLE_NEAREST2D_SCALES = [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("scale", UPSAMPLE_NEAREST2D_SCALES)
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_upsample_nearest2d(dtype, shape, scale):
    if flag_gems.vendor_name == "sunrise" and shape[2] * shape[3] >= 1023 * 1025:
        pytest.skip("Issue #3836: Skip for big shape, '--ref cpu' too slow.")
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)
