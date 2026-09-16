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

device = flag_gems.device


@pytest.mark.randperm
@pytest.mark.parametrize("n", [123, 12345, 123456])
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_randperm(n, dtype):
    if n > torch.iinfo(torch.int16).max and dtype == torch.int16:
        return

    # Skip int16 for Moore Threads backend due to runtime crash
    if flag_gems.vendor_name == "mthreads" and dtype == torch.int16:
        pytest.skip("Issue #2845: Moore Threads int16 randperm causes runtime crash")

    ref_out = torch.randperm(n, dtype=dtype, device="cpu" if cfg.TO_CPU else device)
    with flag_gems.use_gems():
        res_out = torch.randperm(n, dtype=dtype, device=flag_gems.device)

    sorted_ref, _ = torch.sort(ref_out)
    sorted_res, _ = torch.sort(res_out)
    utils.gems_assert_equal(sorted_res, sorted_ref)
