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
    ARGSORT_BATCH_SIZES = [4]
    ARGSORT_HIDDEN_SIZES = [256, 2048]
else:
    ARGSORT_BATCH_SIZES = [4, 8]
    ARGSORT_HIDDEN_SIZES = [1, 256, 2048, 9333, 65536, 32768, 128 * 1024, 256 * 1024]


@pytest.mark.argsort
@pytest.mark.parametrize("batch_size", ARGSORT_BATCH_SIZES)
@pytest.mark.parametrize("hiddensize", ARGSORT_HIDDEN_SIZES)
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize("dim", [0, -1])
def test_accuracy_argsort(batch_size, hiddensize, descending, dtype, dim):
    if dtype in utils.BOOL_TYPES:
        y = torch.randint(
            0, 2, (batch_size, hiddensize), dtype=dtype, device=flag_gems.device
        )
    elif dtype in utils.ALL_INT_DTYPES:
        min_v, max_v = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        y = torch.randint(
            min_v, max_v, (batch_size, hiddensize), dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    else:
        y = torch.randn((batch_size, hiddensize), dtype=dtype, device=flag_gems.device)

    ref_y = utils.to_reference(y)
    ref_index = torch.argsort(ref_y, dim=dim, stable=True, descending=descending)

    with flag_gems.use_gems():
        res_index = torch.argsort(y, dim=dim, stable=True, descending=descending)

    utils.gems_assert_equal(res_index, ref_index)
