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


@pytest.mark.hardswish_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_hardswish_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="hardswish_",
        torch_op=torch.ops.aten.hardswish_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.hardswish
def test_hardswish():
    bench = base.UnaryPointwiseBenchmark(
        op_name="hardswish",
        torch_op=torch.nn.functional.hardswish,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.hardswish_out
def test_hardswish_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="hardswish_out",
        torch_op=torch.ops.aten.hardswish.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
