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

from . import base, consts


def normal_functional_input_fn(shape, dtype, device):
    self = torch.empty(shape, dtype=dtype, device=device)
    loc = 3.0
    scale = 10.0
    yield self, loc, scale


@pytest.mark.normal_functional
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_normal_functional():
    bench = base.GenericBenchmark(
        input_fn=normal_functional_input_fn,
        op_name="normal_functional",
        torch_op=torch.ops.aten.normal_functional,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
