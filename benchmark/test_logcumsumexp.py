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

from . import base, consts


def logcumsumexp_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    yield inp, 1


@pytest.mark.logcumsumexp
def test_logcumsumexp():
    bench = base.GenericBenchmark2DOnly(
        op_name="logcumsumexp",
        input_fn=logcumsumexp_input_fn,
        torch_op=torch.logcumsumexp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def logcumsumexp_out_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    out = torch.empty_like(inp)
    yield inp, 1, {"out": out}


@pytest.mark.logcumsumexp_out
def test_logcumsumexp_out():
    bench = base.GenericBenchmark2DOnly(
        op_name="logcumsumexp_out",
        input_fn=logcumsumexp_out_input_fn,
        torch_op=torch.ops.aten.logcumsumexp.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
