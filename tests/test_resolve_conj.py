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


@pytest.mark.resolve_conj
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_resolve_conj(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cpu")
    y = x.conj()

    assert y.is_conj()

    with flag_gems.use_gems():
        res_y = y.to(device=flag_gems.device)
        z = res_y.resolve_conj()

    assert not z.is_conj()
