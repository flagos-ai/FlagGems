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

"""Public addmm_out: fixed IEEE arithmetic and pre-execution selection guards."""

import flag_gems
import pytest
import torch

from .test_addmm import AddmmVectorBiasBenchmark, _input_fn_vector_bias_out


class HopperAddmmBenchmark(AddmmVectorBiasBenchmark):
    def set_more_shapes(self):
        return [(1, 414, 5376, 32), (1, 8598, 96, 5376), (1, 75938, 32, 5376)]


@pytest.mark.addmm
def test_addmm_hopper_fixed_launch():
    if flag_gems.vendor_name != "nvidia" or torch.cuda.get_device_capability() != (
        9,
        0,
    ):
        pytest.skip("Fixed launch configurations target SM90")
    bench = HopperAddmmBenchmark(
        op_name="addmm_out",
        input_fn=_input_fn_vector_bias_out,
        torch_op=torch.addmm,
        gems_op=flag_gems.addmm_out,
        dtypes=[torch.float32],
    )
    bench.run()
