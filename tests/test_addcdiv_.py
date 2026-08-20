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

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.addcdiv_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addcdiv_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(res_inp, True)
    ref_t1 = utils.to_reference(t1, True)
    ref_t2 = utils.to_reference(t2, True)

    v = float(np.float32(random.random()))

    ref_out = ref_inp.addcdiv_(ref_t1, ref_t2, value=v)
    with flag_gems.use_gems():
        res_out = res_inp.addcdiv_(t1, t2, value=v)

    utils.gems_assert_close(res_out, ref_out, dtype)
