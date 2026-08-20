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


class GluBenchmark(base.UnaryPointwiseBenchmark):
    # Glu test requires even numbers
    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(1, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(1, 15, 4)]
        return special_shapes_2d + sp_shapes_3d


@pytest.mark.glu
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_glu():
    bench = GluBenchmark(
        op_name="glu",
        torch_op=torch.nn.functional.glu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.glu_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_glu_backward():
    bench = GluBenchmark(
        op_name="glu_backward",
        torch_op=torch.nn.functional.glu,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
