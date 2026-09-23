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


@pytest.mark.randint
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_randint(shape, dtype):
    high = 100
    with flag_gems.use_gems():
        res_out = torch.randint(
            high=high, size=shape, dtype=dtype, device=flag_gems.device
        )
    assert res_out.shape == shape
    assert res_out.dtype == dtype
    assert (res_out >= 0).all()
    assert (res_out < high).all()
