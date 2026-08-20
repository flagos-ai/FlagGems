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

import logging

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

ADAPTIVE_AVGPOOL1D_CONFIGS = [
    # (input_shape, output_size)
    ((4, 3, 32), 1),
    ((4, 3, 32), 2),
    ((4, 3, 32), 8),
    ((4, 3, 32), 16),
    ((2, 16, 56), 7),
    ((8, 16, 28), 14),
    ((4, 8, 60), 15),
    ((1, 64, 224), 7),
    ((2, 4, 10), 5),
    ((4, 2, 50), 25),
    ((2, 3, 10), 3),
    ((2, 3, 5), 3),
    ((1, 4, 224), 13),
    ((4, 3, 17), 5),
]


@pytest.mark.adaptive_avg_pool1d
@pytest.mark.parametrize("shape, output_size", ADAPTIVE_AVGPOOL1D_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_avg_pool1d_forward(shape, output_size, dtype, caplog):
    caplog.set_level(logging.DEBUG)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.adaptive_avg_pool1d(ref_inp, [output_size])

    with flag_gems.use_gems():
        res_out = torch.ops.aten.adaptive_avg_pool1d(inp, [output_size])

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert "GEMS ADAPTIVE_AVG_POOL1D" in caplog.text
