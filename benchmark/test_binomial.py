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
    count = torch.randint(1, 100, shape, device=device).to(dtype)
    prob = torch.rand(shape, dtype=dtype, device=device)
    yield (count, prob)


@pytest.mark.binomial
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_binomial():
    bench = base.GenericBenchmark2DOnly(
        op_name="binomial",
        torch_op=torch.binomial,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.binomial_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_binomial_out():
    def out_op(count, prob):
        out = torch.empty_like(count)
        return torch.ops.aten.binomial.out(count, prob, out=out)

    bench = base.GenericBenchmark2DOnly(
        op_name="binomial_out",
        torch_op=out_op,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
