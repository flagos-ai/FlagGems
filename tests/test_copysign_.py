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


@pytest.mark.copysign_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_(shape, dtype):
    # Test copysign_: in-place modification of first tensor
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input.clone(), True)
    ref_other = utils.to_reference(other, True)
    ref_out = ref_input.copysign_(ref_other)

    with flag_gems.use_gems():
        res_out = input.copysign_(other)

    assert res_out.data_ptr() == input.data_ptr()
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(input, ref_input, dtype)
