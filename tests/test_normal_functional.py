# Copyright 2026, The FlagOS Contributors.
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

device = flag_gems.device


@pytest.mark.normal_functional
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_normal_functional(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    if flag_gems.vendor_name in ["metax", "iluvatar", "kunlunxin"]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    loc = 3.0
    scale = 10.0
    inp = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    original_ptr = inp.data_ptr()

    res_out = flag_gems.normal_functional(inp, loc, scale)

    # Out-of-place: a fresh tensor is returned and the input is not mutated.
    assert res_out.data_ptr() != original_ptr
    assert res_out.shape == inp.shape
    assert res_out.dtype == inp.dtype

    # Distribution op: validate statistically that the samples follow N(loc, scale).
    ref_out = utils.to_reference(res_out)
    out_float = ref_out.to(torch.float32)
    mean_res = torch.mean(out_float)
    std_res = torch.std(out_float)
    expected_mean = torch.tensor(loc, device=mean_res.device)
    expected_std = torch.tensor(scale, device=std_res.device)

    # Loose tolerance to account for sampling variance across the tested shapes.
    utils.gems_assert_close(mean_res, expected_mean, torch.float32, atol=0.2)
    utils.gems_assert_close(std_res, expected_std, torch.float32, atol=0.2)
