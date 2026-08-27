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

from . import base, consts, utils


class ModeBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return [(1024, 1), (1024, 512), (16, 128 * 1024), (8, 256 * 1024)]


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": -1},


@pytest.mark.mode
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_perf_mode():
    bench = ModeBenchmark(
        input_fn=_input_fn,
        op_name="mode",
        torch_op=torch.mode,
        dtypes=consts.INT_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.run()
