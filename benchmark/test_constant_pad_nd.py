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

import pytest
import torch

import flag_gems

from . import base, consts


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, device=device, dtype=dtype)
    rank = inp.ndim
    pad = [random.randint(0, 10) for _ in range(rank * 2)]
    value = 1.5
    yield inp, {"pad": pad, "value": value}


@pytest.mark.constant_pad_nd
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_constant_pad_nd():
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="constant_pad_nd",
        torch_op=torch.constant_pad_nd,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
