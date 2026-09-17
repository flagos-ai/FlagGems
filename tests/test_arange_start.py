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


@pytest.mark.arange_start
@pytest.mark.arange_start_step
@pytest.mark.parametrize("start", utils.ARANGE_START)
@pytest.mark.parametrize("step", [1] if cfg.QUICK_MODE else [1, 2, 5])
@pytest.mark.parametrize("end", [128] if cfg.QUICK_MODE else [128, 256, 1024])
@pytest.mark.parametrize(
    "dtype",
    (
        [torch.float32]
        if cfg.QUICK_MODE
        else utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + [None]
    ),
)
@pytest.mark.parametrize(
    "device", [flag_gems.device] if cfg.QUICK_MODE else [flag_gems.device, None]
)
# Since triton only target to GPU, pin_memory only used in CPU tensors.
@pytest.mark.parametrize("pin_memory", [False] if cfg.QUICK_MODE else [False, None])
def test_arange(start, step, end, dtype, device, pin_memory):
    ref_out = torch.arange(
        start,
        end,
        step,
        dtype=dtype,
        device="cpu" if cfg.TO_CPU else device,
        pin_memory=pin_memory,
    )

    with flag_gems.use_gems():
        res_out = torch.arange(
            start, end, step, dtype=dtype, device=device, pin_memory=pin_memory
        )

    utils.gems_assert_equal(res_out, ref_out)
