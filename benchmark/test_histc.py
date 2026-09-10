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

from . import base


def _input_fn(shape, dtype, device):
    inp = torch.rand(shape, dtype=dtype, device=device) * 10
    yield inp, {"bins": 100, "min": 0, "max": 10}


@pytest.mark.histc
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_histc():
    bench = base.GenericBenchmark2DOnly(
        input_fn=_input_fn,
        op_name="histc",
        torch_op=torch.histc,
        dtypes=[torch.float32],
    )
    bench.run()
