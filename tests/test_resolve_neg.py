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

random.seed(time.time() // 100)


@pytest.mark.resolve_neg
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_accuracy_resolve_neg(shape, dtype):
    if flag_gems.vendor_name == "ascend":
        x = torch.randn(size=shape, dtype=dtype).to(device=flag_gems.device)
    else:
        x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)

    y = x.conj()
    z = y.imag
    assert z.is_neg()

    with flag_gems.use_gems():
        out = z.resolve_neg()
    assert not out.is_neg()
