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

import math

import pytest
import torch

import flag_gems

from . import base


def _input_fn(shape, dtype, device):
    base = 1.05
    # calculate the max limit according to dtype
    limit = math.log2(torch.finfo(dtype).max - 1) / math.log2(base)
    end = int(limit)
    yield {
        "start": 0,
        "end": end,
        "steps": math.prod(shape),  # steps influence speed up a lot
        "base": base,
        "dtype": dtype,
        "device": device,
    },


@pytest.mark.logspace
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_logspace():
    bench = base.GenericBenchmark(
        op_name="logspace", input_fn=_input_fn, torch_op=torch.logspace
    )
    bench.run()
