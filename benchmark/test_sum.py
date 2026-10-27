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

from . import base, consts, utils


@pytest.mark.sum
def test_sum():
    bench = base.UnaryReductionBenchmark(
        op_name="sum", torch_op=torch.sum, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


def sum_dim_out_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    dim = 1 if len(shape) > 1 else 0
    out = torch.sum(inp, dim=dim)
    yield inp, dim, {"out": out}


@pytest.mark.sum_dim_out
def test_sum_dim_out():
    bench = base.GenericBenchmark(
        op_name="sum_dim_out",
        torch_op=torch.sum,
        input_fn=sum_dim_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def sum_out_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    out = torch.empty([], dtype=dtype, device=device)
    yield inp, {"out": out}


@pytest.mark.sum_out
def test_sum_out():
    bench = base.GenericBenchmark(
        op_name="sum_out",
        torch_op=torch.ops.aten.sum.out,
        input_fn=sum_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
