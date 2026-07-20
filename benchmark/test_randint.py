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


def randint_input_fn(shape, dtype, device):
    high = 100
    yield high, shape


class RandintBenchmark(base.GenericBenchmarkExcluse1D):
    # Override set_more_shapes to provide custom shapes for randint
    def set_more_shapes(self):
        return [(1024, 1), (1024, 512), (16, 128 * 1024), (8, 256 * 1024)]


@pytest.mark.randint
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_randint():
    bench = RandintBenchmark(
        input_fn=randint_input_fn,
        op_name="randint",
        torch_op=torch.randint,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
