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


@pytest.mark.heaviside_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("values_kind", ["same", "scalar", "row", "col"])
@pytest.mark.parametrize("zero_fraction", [0.0, 0.2])
def test_heaviside_(shape, dtype, values_kind, zero_fraction):
    self_input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if zero_fraction > 0.0:
        zmask = torch.rand(shape, device=flag_gems.device) < zero_fraction
        self_input = self_input.masked_fill(zmask, 0.0)

    if values_kind == "same":
        values = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif values_kind == "scalar":
        values = torch.randn((), dtype=dtype, device=flag_gems.device)
    elif values_kind == "row":
        values = torch.randn((1, shape[1]), dtype=dtype, device=flag_gems.device)
    else:  # col
        values = torch.randn((shape[0], 1), dtype=dtype, device=flag_gems.device)

    ref_self = utils.to_reference(self_input, True)
    ref_values = utils.to_reference(values, True)
    ref_out = torch.ops.aten.heaviside_(ref_self, ref_values)

    act_self = self_input.clone()
    with flag_gems.use_gems():
        res_out = torch.ops.aten.heaviside_(act_self, values)

    # In-place: result and mutated self both match the reference.
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(act_self, ref_out, dtype)
