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


@pytest.mark.randn
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_randn(shape, dtype):
    if flag_gems.vendor_name in ["cambricon", "iluvatar"]:
        torch.manual_seed(42)

    with flag_gems.use_gems():
        res_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = utils.to_reference(res_out).float()
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)

    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01
