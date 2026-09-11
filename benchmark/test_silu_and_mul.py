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


@pytest.mark.silu_and_mul
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_silu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.silu(x), y)

    bench = base.GenericBenchmark(
        input_fn=utils.binary_input_fn,
        op_name="silu_and_mul",
        gems_op=flag_gems.silu_and_mul,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.silu_and_mul_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_silu_and_mul_out():
    def gems_op(x, y):
        out = torch.empty_like(x)
        return flag_gems.silu_and_mul_out(x, y, out)

    def torch_op(x, y):
        out = torch.empty_like(x)
        return torch.mul(torch.nn.functional.silu(x), y, out=out)

    bench = base.GenericBenchmark(
        input_fn=utils.binary_input_fn,
        op_name="silu_and_mul_out",
        gems_op=gems_op,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
