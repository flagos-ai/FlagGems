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

from . import base, consts


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield {"input": inp, "high": 10},


@pytest.mark.randint_like
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_randint_like():
    bench = base.GenericBenchmark(
        op_name="randint_like",
        input_fn=_input_fn,
        torch_op=torch.randint_like,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
