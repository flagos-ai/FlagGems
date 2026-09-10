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
from . import conftest as cfg

if cfg.QUICK_MODE:
    ARANGE_ENDS = [100]
    ARANGE_DTYPES = [torch.float32]
else:
    ARANGE_ENDS = [10, 100, 1000, 5.0]
    ARANGE_DTYPES = [torch.float32, torch.float16, torch.int64]


@pytest.mark.arange
@pytest.mark.parametrize("end", ARANGE_ENDS)
@pytest.mark.parametrize("dtype", ARANGE_DTYPES)
def test_arange(end, dtype):
    with flag_gems.use_gems():
        res_out = torch.arange(end, dtype=dtype, device=flag_gems.device)
    ref_out = torch.arange(end, dtype=dtype, device="cpu")

    utils.gems_assert_equal(res_out.cpu(), ref_out)
