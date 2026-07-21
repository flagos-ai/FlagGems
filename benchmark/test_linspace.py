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
import random

import pytest
import torch

import flag_gems

from . import base


def _input_fn(shape, dtype, device):
    limit = torch.finfo(dtype).max - 1
    num = int(min(limit, math.prod(shape)))
    yield {
        "start": 0,
        "end": num,
        "steps": random.randint(1, num),
        "dtype": dtype,
        "device": device,
    },


@pytest.mark.linspace
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_linspace():
    bench = base.GenericBenchmark(
        op_name="linspace", input_fn=_input_fn, torch_op=torch.linspace
    )
    bench.run()
