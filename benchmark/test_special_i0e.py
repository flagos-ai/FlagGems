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


@pytest.mark.special_i0e
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_special_i0e():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_i0e",
        torch_op=torch.ops.aten.special_i0e,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_i0e_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_special_i0e_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_i0e_out",
        torch_op=torch.ops.aten.special_i0e,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
