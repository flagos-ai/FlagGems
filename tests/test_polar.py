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
from .accuracy_utils import _SKIP_FP64


# Issue #2840
@pytest.mark.polar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", [torch.float32, pytest.param(torch.float64, marks=_SKIP_FP64)]
)
def test_polar(shape, dtype):
    if flag_gems.vendor_name == "cambricon" and dtype == torch.float64:
        pytest.skip("Issue #5253: Not supported")
    if dtype == torch.float64 and not flag_gems.runtime.device.support_fp64:
        pytest.skip("float64 is not supported on this backend")
    abs = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5
    angle = (torch.rand(shape, dtype=dtype, device=flag_gems.device) - 0.5) * (
        8 * math.pi
    )
    ref_abs = utils.to_reference(abs)
    ref_angle = utils.to_reference(angle)
    ref_out = torch.polar(ref_abs, ref_angle)
    res_out = flag_gems.polar(abs, angle)

    utils.gems_assert_close(res_out.real, ref_out.real, dtype)
    utils.gems_assert_close(res_out.imag, ref_out.imag, dtype)
