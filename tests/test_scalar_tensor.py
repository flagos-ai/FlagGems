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


@pytest.mark.scalar_tensor
@pytest.mark.parametrize(
    "dtype", utils.BOOL_TYPES + utils.INT_DTYPES + utils.FLOAT_DTYPES
)
@pytest.mark.parametrize("fill_value", [0.01, 2, 0, -1, True, False])
def test_scalar_tensor(dtype, fill_value):
    ref_out = torch.scalar_tensor(
        fill_value, dtype=dtype, device="cpu" if cfg.TO_CPU else device
    )

    with flag_gems.use_gems():
        res_out = torch.scalar_tensor(fill_value, dtype=dtype, device=flag_gems.device)

    utils.gems_assert_equal(res_out, ref_out)
