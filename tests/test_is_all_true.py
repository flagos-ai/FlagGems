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
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.is_all_true
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("kind", ["allTrue", "allFalse", "mixed"])
def test_is_all_true(shape, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=torch.bool, device=flag_gems.device)
    elif kind == "allFalse":
        inp = torch.zeros(shape, dtype=torch.bool, device=flag_gems.device)
    else:
        # Mixed: random boolean values
        inp = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch._is_all_true(ref_inp)
    with flag_gems.use_gems():
        res_out = torch._is_all_true(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.is_all_true
@pytest.mark.parametrize("shape", [(0,), (0, 5), (3, 0, 4)])
def test_accuracy_is_all_true_empty(shape):
    # Empty tensors should return True (vacuous truth)
    inp = torch.empty(shape, dtype=torch.bool, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch._is_all_true(ref_inp)
    with flag_gems.use_gems():
        res_out = torch._is_all_true(inp)

    utils.gems_assert_equal(res_out, ref_out)
